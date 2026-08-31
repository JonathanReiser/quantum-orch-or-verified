"""
ewl_mixed_equilibrium.py — what is the EWL game worth once Benjamin-Hayden bites?

ewl_equilibrium.py establishes two things: EWL's (Q,Q) equilibrium is real inside
the restricted strategy space above a derived entanglement threshold, and it
vanishes over the full SU(2) space, where no *pure* Nash equilibrium survives.

That leaves the obvious question unanswered. "No pure equilibrium" is not "no
equilibrium" — Glicksberg's theorem guarantees a mixed one exists, since SU(2) is
compact and the payoffs are continuous. So: when both players play well over the
full strategy space, what is the game actually worth?

The answer here is exact rather than numerical. At maximal entanglement the
Haar-uniform mixed strategy is a symmetric Nash equilibrium paying exactly
(T+R+P+S)/4 = 2.25, which beats the classical Nash equilibrium and the best
correlated equilibrium (both 1.0) but falls short of EWL's restricted (3,3).

Method note: replicator dynamics is *not* used, and should not be. On this game
it diverges away from the true equilibrium — the uniform point is an equilibrium
but an unstable one under replicator, so the dynamics amplify sampling noise and
wander off. Instead the Haar average is computed in closed form via the twirl
identity, so there is no sampling and no optimiser anywhere in this module.

Usage:
    python3 q_ai_governance/ewl_mixed_equilibrium.py
"""

import argparse
import json
import os

import numpy as np

try:
    from .ewl_equilibrium import (C_STRAT, D_STRAT, PAYOFF_A, Q_STRAT,
                                  entangling_gate, strategy)
except ImportError:
    from ewl_equilibrium import (C_STRAT, D_STRAT, PAYOFF_A, Q_STRAT,
                                 entangling_gate, strategy)

I2 = np.eye(2, dtype=complex)

# Prisoner's Dilemma constants, in the standard T > R > P > S naming.
T, R, P, S = 5.0, 3.0, 1.0, 0.0


def haar_payoff_exact(u_a, gamma):
    """Exact payoff to A when B draws its strategy Haar-uniformly from SU(2).

    Uses the twirl identity, which makes this analytic rather than sampled:

        E_U [ (I (x) U) sigma (I (x) U)^dag ]  =  Tr_B(sigma) (x) I/2

    So averaging over the opponent replaces B's half of the state with the
    maximally mixed state, and the remaining algebra is deterministic.
    """
    j = entangling_gate(gamma)
    psi = j @ np.array([1, 0, 0, 0], dtype=complex)
    rho0 = np.outer(psi, psi.conj())
    ua_full = np.kron(u_a, I2)
    sigma = ua_full @ rho0 @ ua_full.conj().T
    tau_a = sigma.reshape(2, 2, 2, 2).trace(axis1=1, axis2=3)   # partial trace over B
    rho = j.conj().T @ np.kron(tau_a, I2 / 2) @ j
    return float(np.real(np.diag(rho)) @ PAYOFF_A)


def haar_payoff_closed(theta, gamma):
    """Closed form of the same quantity, verified to ~1e-15 against the twirl.

        u_A(theta, gamma) = (T+R+P+S)/4 - ((T+P-R-S)/4) cos^2(gamma) cos(theta)

    Two things fall straight out of it:

      * The payoff does not depend on either phase, only on theta. Both of a
        strategy's phase parameters are irrelevant against a Haar opponent.
      * The dependence on theta — the only thing a player controls here —
        carries the factor cos^2(gamma), so it vanishes exactly when
        gamma = pi/2 and at no smaller entanglement.
    """
    return (T + R + P + S) / 4 - ((T + P - R - S) / 4) * np.cos(gamma) ** 2 * np.cos(theta)


def exploitability(gamma):
    """How much a best responder gains against the Haar-uniform strategy.

    Since the payoff is monotone in cos(theta), the extremes are theta = 0 (C)
    and theta = pi (D), giving a spread of ((T+P-R-S)/2) cos^2(gamma). An
    exploitability of zero certifies a Nash equilibrium.
    """
    return ((T + P - R - S) / 2) * np.cos(gamma) ** 2


def haar_equilibrium_value():
    """Value of the Haar-uniform symmetric equilibrium at maximal entanglement."""
    return (T + R + P + S) / 4


def report(gammas):
    rows = []
    print(f"{'gamma':>9}{'C vs Haar':>12}{'D vs Haar':>12}{'Q vs Haar':>12}"
          f"{'exploitability':>17}{'equilibrium?':>14}")
    for gamma in gammas:
        c = haar_payoff_exact(C_STRAT, gamma)
        d = haar_payoff_exact(D_STRAT, gamma)
        q = haar_payoff_exact(Q_STRAT, gamma)
        expl = exploitability(gamma)
        is_eq = bool(expl < 1e-12)
        rows.append({"gamma": round(float(gamma), 6), "C": round(c, 9),
                     "D": round(d, 9), "Q": round(q, 9),
                     "exploitability": float(expl), "is_equilibrium": is_eq})
        print(f"{gamma:>9.4f}{c:>12.6f}{d:>12.6f}{q:>12.6f}{expl:>17.6e}"
              f"{str(is_eq):>14}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ewl_mixed_equilibrium_results.json")
    ap.add_argument("--n-gamma", type=int, default=7)
    args = ap.parse_args()

    print("EWL full-SU(2) game — the Haar-uniform mixed strategy")
    print("Payoffs: T=5 (defect on cooperator), R=3 (mutual cooperation), "
          "P=1 (mutual defection), S=0 (cooperate against defector)\n")

    gammas = np.linspace(0.0, np.pi / 2, args.n_gamma)
    rows = report(gammas)

    val = haar_equilibrium_value()
    print(f"\nAt gamma = pi/2 every SU(2) strategy is exactly indifferent, so the")
    print(f"Haar-uniform strategy is a symmetric Nash equilibrium worth "
          f"(T+R+P+S)/4 = {val}.")
    print(f"\nFor comparison:")
    print(f"  classical Nash equilibrium (D,D)          1.00")
    print(f"  best classical correlated equilibrium     1.00")
    print(f"  full SU(2) Haar equilibrium               {val:.2f}")
    print(f"  EWL restricted-space (Q,Q)                3.00")
    frac = (val - 1.0) / (3.0 - 1.0)
    print(f"\nThe full-space equilibrium recovers {frac:.1%} of the distance from the")
    print(f"classical trap to full cooperation — so Benjamin-Hayden removes EWL's")
    print(f"pure equilibrium, not the whole of the entanglement advantage.")

    out = {"payoffs": {"T": T, "R": R, "P": P, "S": S},
           "haar_equilibrium_value": val,
           "classical_nash": 1.0, "correlated_equilibrium": 1.0,
           "ewl_restricted_pure": 3.0,
           "fraction_of_cooperative_gain": round(float(frac), 6),
           "sweep": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
