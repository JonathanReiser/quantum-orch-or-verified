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


def test_predicted_bias_direction_is_downward(prediction):
    assert "below" in prediction["expected_device_bias"]["direction"].lower()


def test_declares_no_measurement_performed(prediction):
    assert prediction["declared"]["measurement_performed"] is False
