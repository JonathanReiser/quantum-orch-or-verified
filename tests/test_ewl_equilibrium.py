"""
tests/test_ewl_equilibrium.py — pin the EWL results to the published literature.

These assert landmarks with known analytic answers, so the module cannot drift
into agreeing with itself. Every value here is checkable against a paper.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "quantum_games"))

from ewl_equilibrium import (  # noqa: E402
    C_STRAT, D_STRAT, Q_STRAT, build_grid, classical_correlated_equilibrium,
    critical_gamma, deviation_check, entangling_gate, payoff_matrices, pure_nash,
    strategy,
)

MAX_ENT = np.pi / 2


def _payoff(ua, ub, gamma):
    pa, pb = payoff_matrices([ua], [ub], gamma)
    return float(pa[0, 0]), float(pb[0, 0])


def test_gamma_zero_reproduces_the_classical_game():
    """With no entanglement the quantised game must be the ordinary PD."""
    assert _payoff(C_STRAT, C_STRAT, 0.0) == pytest.approx((3.0, 3.0))
    assert _payoff(C_STRAT, D_STRAT, 0.0) == pytest.approx((0.0, 5.0))
    assert _payoff(D_STRAT, C_STRAT, 0.0) == pytest.approx((5.0, 0.0))
    assert _payoff(D_STRAT, D_STRAT, 0.0) == pytest.approx((1.0, 1.0))


def test_entangling_gate_is_unitary_and_identity_at_zero():
    assert np.allclose(entangling_gate(0.0), np.eye(4))
    for gamma in (0.3, 0.9, MAX_ENT):
        j = entangling_gate(gamma)
        assert np.allclose(j @ j.conj().T, np.eye(4))


def test_strategies_are_su2():
    for params in [(0.0, 0.0, 0.0), (np.pi, 0.0, 0.0), (0.0, np.pi / 2, 0.0),
                   (1.1, 2.2, 3.3)]:
        u = strategy(*params)
        assert np.allclose(u @ u.conj().T, np.eye(2))
        assert np.isclose(np.linalg.det(u), 1.0)


def test_ewl_quantum_strategy_beats_defection_when_entangled():
    """EWL: at maximal entanglement Q defeats D, which is why (D,D) collapses."""
    assert _payoff(Q_STRAT, D_STRAT, MAX_ENT) == pytest.approx((5.0, 0.0))
    assert _payoff(D_STRAT, Q_STRAT, MAX_ENT) == pytest.approx((0.0, 5.0))
    assert _payoff(Q_STRAT, Q_STRAT, MAX_ENT) == pytest.approx((3.0, 3.0))


def test_QQ_is_nash_in_restricted_space_only_above_threshold():
    """EWL's headline result, and its entanglement threshold."""
    below = deviation_check(Q_STRAT, Q_STRAT, 0.3, "restricted", n_samples=20_000)
    above = deviation_check(Q_STRAT, Q_STRAT, MAX_ENT, "restricted", n_samples=20_000)
    assert not below["is_nash"]
    assert above["is_nash"]
    assert above["base_payoff"] == [3.0, 3.0]


def test_critical_gamma_matches_its_closed_form():
    """cos^2(gamma_c) = (R-S)/(T-S) = 3/5, verified against a dense sweep."""
    cg = critical_gamma(verify=True, n_theta=201, n_alpha=101)
    assert cg["analytic_rad"] == pytest.approx(np.arccos(np.sqrt(0.6)), abs=1e-9)
    assert cg["numeric_rad"] == pytest.approx(cg["analytic_rad"], abs=1e-4)


def test_benjamin_hayden_no_pure_nash_in_full_su2():
    """The standard objection: widening the strategy space kills the equilibrium."""
    result = deviation_check(Q_STRAT, Q_STRAT, MAX_ENT, "full", n_samples=50_000)
    assert not result["is_nash"]
    assert max(result["best_deviation_gain_A"],
               result["best_deviation_gain_B"]) > 0.5

    mats, _ = build_grid("full", 9, 9)
    pa, pb = payoff_matrices(mats, mats, MAX_ENT)
    assert int(pure_nash(pa, pb).sum()) == 0


def test_classical_defection_is_nash_without_entanglement():
    mats, _ = build_grid("restricted", 21, 11)
    pa, pb = payoff_matrices(mats, mats, 0.0)
    mask = pure_nash(pa, pb)
    assert mask.sum() > 0
    assert float(((pa + pb) / 2)[mask].max()) == pytest.approx(1.0)


def test_correlated_equilibrium_cannot_reach_cooperation():
    """Shared randomness provably cannot do what entanglement does here:
    D strictly dominates C, so every correlated equilibrium sits on (D,D)."""
    ce = classical_correlated_equilibrium()
    assert ce["payoff_A"] == pytest.approx(1.0, abs=1e-6)
    assert ce["payoff_B"] == pytest.approx(1.0, abs=1e-6)
    assert ce["distribution"]["DD"] == pytest.approx(1.0, abs=1e-6)
