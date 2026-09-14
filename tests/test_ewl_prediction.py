"""
tests/test_ewl_prediction.py — guard the pre-registered hardware prediction.

EWL_HARDWARE_PREREGISTRATION.md commits a threshold, a protocol and an error
budget before any quantum hardware is used. These tests assert the pieces that
must not drift afterwards: the threshold matches the closed form, the deviating
strategy really is classical Defect and really is binding at the threshold, and
the error budget still carries its systematic term.

The systematic term matters. Budgeting shot noise alone would set a criterion a
real device fails for reasons unrelated to whether the EWL threshold is right:
depolarizing noise biases the measured crossing downward by an amount comparable
to the 3-sigma shot tolerance.
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quantum_games.ewl_prediction import (  # noqa: E402
    binding_deviation_at_zero_entanglement, payoff_and_sd)
from quantum_games.ewl_equilibrium import Q_STRAT, strategy  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..")
GAMMA_C = float(np.arccos(np.sqrt(3.0 / 5.0)))


@pytest.fixture(scope="module")
def prediction():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "p.json")
        subprocess.run([sys.executable, "-m", "quantum_games.ewl_prediction",
                        "--two-qubit-error", "8e-3", "--out", out],
                       cwd=REPO, check=True, capture_output=True)
        with open(out) as f:
            return json.load(f)


def test_threshold_matches_closed_form(prediction):
    assert prediction["prediction"]["gamma_c"] == pytest.approx(GAMMA_C, abs=1e-9)
    assert prediction["prediction"]["gamma_c"] == pytest.approx(0.684719203, abs=1e-9)


def test_deviation_is_classical_defect():
    _, theta, alpha = binding_deviation_at_zero_entanglement()
    assert theta == pytest.approx(np.pi, abs=1e-9)
    assert alpha == pytest.approx(0.0, abs=1e-9)


def test_deviation_is_binding_exactly_at_the_threshold():
    """Defect must pay above 3 below the threshold, and exactly 3 at it."""
    d = strategy(np.pi, 0.0)
    assert payoff_and_sd(d, Q_STRAT, GAMMA_C - 0.05)[0] > 3.0
    assert payoff_and_sd(d, Q_STRAT, GAMMA_C)[0] == pytest.approx(3.0, abs=1e-9)
    assert payoff_and_sd(d, Q_STRAT, GAMMA_C + 0.05)[0] < 3.0


def test_QQ_is_deterministic_at_the_threshold():
    """(Q,Q) always yields CC, so on a noiseless device it contributes no shot noise."""
    mean, sd = payoff_and_sd(Q_STRAT, Q_STRAT, GAMMA_C)
    assert mean == pytest.approx(3.0, abs=1e-9)
    assert sd == pytest.approx(0.0, abs=1e-12)


def test_selecting_deviation_at_max_entanglement_would_be_degenerate():
    """Why the strategy is chosen at gamma=0: at pi/2 the best reply to Q is Q,
    so the deviation gain would be identically zero and the sweep would find
    no crossing at all."""
    assert payoff_and_sd(Q_STRAT, Q_STRAT, np.pi / 2)[0] == pytest.approx(3.0, abs=1e-9)
    assert payoff_and_sd(strategy(np.pi, 0.0), Q_STRAT, np.pi / 2)[0] < 3.0


def test_error_budget_includes_a_systematic_term(prediction):
    pc = prediction["pass_criterion"]
    shot = pc["components"]["three_sigma_shot_rad"]
    syst = pc["components"]["systematic_budget_rad"]
    assert syst > 0, "shot noise alone is not the whole error budget"
    assert pc["tolerance_rad"] == pytest.approx(shot + syst, abs=1e-9)
    assert syst == pytest.approx(0.002 + 8e-3, abs=1e-9)


def test_no_bias_direction_is_predicted(prediction):
    """Amendment 1 withdrew the one-directional bias claim. Asserting a direction
    again would re-justify an asymmetric pass rule that the evidence does not
    support: depolarizing noise cannot move the crossing in either direction."""
    direction = prediction["expected_device_bias"]["direction"].lower()
    assert "no direction is predicted" in direction


def test_declares_no_measurement_performed(prediction):
    assert prediction["declared"]["measurement_performed"] is False


# --- amendment 1: the withdrawn bias table, and what replaced it ---------------


def test_the_withdrawn_bias_literals_are_not_treated_as_measurements(prediction):
    """0.003986 / 0.007384 / 0.019214 could not be reconstructed. They may appear
    only as a withdrawn record, never as a live input to the budget."""
    bias = prediction["expected_device_bias"]
    assert bias["superseded_table"]["status"] == "withdrawn -- unreproducible"
    assert "simulated_bias_by_channel" in bias
    assert "measured_in_simulation" not in bias


def test_depolarizing_noise_cannot_move_the_crossing():
    """The registration's original mechanism, checked rather than asserted: a
    depolarizing channel rescales the gain by (1-q) and cannot move its root."""
    from quantum_games.ewl_device_bias import discretization_bias, measured_crossing
    grid = discretization_bias()
    for error in (3.0e-3, 8.0e-3, 2.0e-2):
        crossing = measured_crossing(error, "depolarizing")
        assert crossing - GAMMA_C == pytest.approx(grid, abs=1e-9)


def test_simulated_bias_stays_below_the_systematic_allowance(prediction):
    allowance = prediction["pass_criterion"]["components"]["systematic_budget_rad"]
    assert prediction["expected_device_bias"]["largest_simulated_bias_rad"] < allowance


def test_pass_criterion_is_symmetric_and_exhaustive(prediction):
    pc = prediction["pass_criterion"]
    assert pc["symmetric"] is True
    assert pc["no_sign_change_is"] == "FAIL"
    rule = pc["decision_rule"]
    assert "if and only if" in rule
    assert "no separate 'materially above' clause" in rule, \
        "the vague clause must stay explicitly retired, not silently reappear"
    assert "falsified_if" not in pc, "the old ambiguous key must not return"
    assert pc["systematic_is_an_assumption"] is True


def test_tolerance_was_not_loosened_by_the_amendment(prediction):
    """Amendment 1 relabelled the systematic term; it must not have widened it."""
    pc = prediction["pass_criterion"]
    assert pc["tolerance_rad"] == pytest.approx(0.026573, abs=1e-6)
    assert pc["components"]["systematic_budget_rad"] == pytest.approx(0.010, abs=1e-9)


def test_run_protocol_commitments_are_documented():
    """Device choice, binding run and discard policy were undisclosed degrees of
    freedom; they must stay pinned in the registration."""
    text = open(os.path.join(REPO, "EWL_HARDWARE_PREREGISTRATION.md"),
                encoding="utf-8").read()
    for phrase in ("the first eligible run is binding",
                   "lowest published median two-qubit gate error",
                   "calibration snapshot",
                   "permitted discards",
                   "exactly one"):
        assert phrase in text, f"run protocol lost its commitment: {phrase}"


def test_scope_denies_testing_new_physics():
    text = open(os.path.join(REPO, "EWL_HARDWARE_PREREGISTRATION.md"),
                encoding="utf-8").read()
    assert "not a\ntest of quantum cognition" in text or \
           "not a test of quantum cognition" in text.replace("\n", " ")
    assert "weak (non-strict) Nash equilibrium" in text
