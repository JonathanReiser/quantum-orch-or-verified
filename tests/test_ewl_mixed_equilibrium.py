"""
tests/test_ewl_mixed_equilibrium.py — pin the full-SU(2) mixed equilibrium.

The central claims are exact, so these assert to near machine precision rather
than to a tolerance chosen to make them pass.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "quantum_games"))

from ewl_equilibrium import C_STRAT, D_STRAT, Q_STRAT, strategy  # noqa: E402
from ewl_mixed_equilibrium import (  # noqa: E402
    exploitability, haar_equilibrium_value, haar_payoff_closed, haar_payoff_exact,
)

MAX_ENT = np.pi / 2


def _haar(n, seed=0):
    """Haar-uniform SU(2) via random unit quaternions."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    a, b, c, d = q.T
    u = np.empty((n, 2, 2), dtype=complex)
    u[:, 0, 0] = a + 1j * b
    u[:, 0, 1] = c + 1j * d
    u[:, 1, 0] = -c + 1j * d
    u[:, 1, 1] = a - 1j * b
    return u


def test_closed_form_matches_the_twirl():
    """The closed form is derived, so it should agree to machine precision."""
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(400):
        theta = rng.uniform(0, np.pi)
        u = strategy(theta, rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi))
        gamma = rng.uniform(0, MAX_ENT)
        worst = max(worst, abs(haar_payoff_exact(u, gamma)
                               - haar_payoff_closed(theta, gamma)))
    assert worst < 1e-12


def test_payoff_against_haar_is_phase_independent():
    """Only theta matters against a Haar opponent; both phases drop out."""
    base = haar_payoff_exact(strategy(1.0, 0.0, 0.0), 0.8)
    for alpha, beta in [(0.7, 0.0), (0.0, 2.1), (1.3, 4.4), (5.9, 0.2)]:
        assert haar_payoff_exact(strategy(1.0, alpha, beta), 0.8) == pytest.approx(
            base, abs=1e-12)


def test_every_strategy_is_indifferent_at_maximal_entanglement():
    """This is what makes Haar an equilibrium: total indifference."""
    vals = [haar_payoff_exact(u, MAX_ENT) for u in _haar(300, seed=5)]
    vals += [haar_payoff_exact(u, MAX_ENT) for u in (C_STRAT, D_STRAT, Q_STRAT)]
    assert max(vals) - min(vals) < 1e-12
    assert np.mean(vals) == pytest.approx(2.25, abs=1e-12)


def test_haar_is_an_equilibrium_only_at_maximal_entanglement():
    assert exploitability(MAX_ENT) < 1e-12
    for gamma in (0.0, 0.5, 1.0, 1.4, MAX_ENT - 0.05):
        assert exploitability(gamma) > 1e-6


def test_defection_is_the_best_response_below_maximal_entanglement():
    """Below pi/2 the exploiter plays D, which is why Haar is not an equilibrium."""
    for gamma in (0.0, 0.6, 1.2):
        d = haar_payoff_exact(D_STRAT, gamma)
        c = haar_payoff_exact(C_STRAT, gamma)
        assert d > c
        assert d - c == pytest.approx(exploitability(gamma), abs=1e-12)


def test_equilibrium_value_sits_between_the_classical_and_ewl_values():
    """The headline ordering: 1.0 < 2.25 < 3.0."""
    value = haar_equilibrium_value()
    assert value == pytest.approx(2.25, abs=1e-12)
    assert 1.0 < value < 3.0
    assert (value - 1.0) / (3.0 - 1.0) == pytest.approx(0.625, abs=1e-12)


def test_classical_limit_reproduces_the_uniform_opponent_payoffs():
    """At gamma=0 a Haar opponent defects with probability 1/2, so D pays
    (T+P)/2 = 3 and C pays (R+S)/2 = 1.5."""
    assert haar_payoff_exact(D_STRAT, 0.0) == pytest.approx(3.0, abs=1e-12)
    assert haar_payoff_exact(C_STRAT, 0.0) == pytest.approx(1.5, abs=1e-12)
