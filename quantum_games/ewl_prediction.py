#!/usr/bin/env python3
"""
ewl_prediction.py -- the EWL hardware prediction, emitted BEFORE any hardware run.

This exists to make one distinction enforceable: a prediction is a number
committed before the measurement; a description is a number written after it.
Every claim in CORRECTIONS.md failed on exactly that boundary.

So this file computes, from the payoff matrix alone and with no measurement of
any kind, everything the hardware run will be judged against:

  * the threshold gamma_c where (Q,Q) becomes a WEAK (non-strict) Nash
    equilibrium -- at and above gamma_c the best deviation ties with Q at 3.0
    rather than being strictly beaten
  * which deviating strategy the run must use, and why that one
  * the payoffs and shot-noise spreads expected at the threshold
  * the tolerance, derived from the design rather than chosen
  * what result would falsify the prediction

Nothing here samples, and nothing here reads a result file. Output is a pure
function of the payoff matrix and the declared shot budget.

Usage: python3 -m quantum_games.ewl_prediction --out data/ewl_prediction.json
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

from quantum_games.ewl_equilibrium import (
    strategy, entangling_gate, payoff_matrices, PAYOFF_A, Q_STRAT)
from quantum_games.ewl_device_bias import bias_table, discretization_bias

T, R, P, S = 5.0, 3.0, 1.0, 0.0      # temptation, reward, punishment, sucker
PLANNED_SHOTS = 8192
PLANNED_POINTS = 41


def outcome_probs(u_a, u_b, gamma):
    jg = entangling_gate(gamma)
    w = (jg @ np.array([1, 0, 0, 0], dtype=complex)).reshape(2, 2)
    return np.abs(jg.conj().T @ (u_a @ w @ u_b.T).reshape(4)) ** 2


def payoff_and_sd(u_a, u_b, gamma):
    p = outcome_probs(u_a, u_b, gamma)
    m = float(p @ PAYOFF_A)
    return m, float(np.sqrt(max(p @ PAYOFF_A ** 2 - m ** 2, 0.0)))


def binding_deviation_at_zero_entanglement(n=181):
    """Chosen at gamma = 0, far from the threshold, so the strategy cannot be
    picked to suit the answer. At gamma = pi/2 the best reply to Q is Q itself,
    which would make the deviation gain identically zero."""
    best = (-np.inf, None, None)
    for t in np.linspace(0.0, np.pi, n):
        for a in np.linspace(0.0, np.pi / 2, n):
            v = float(payoff_matrices([strategy(t, a)], [Q_STRAT], 0.0)[0][0, 0])
            if v > best[0]:
                best = (v, float(t), float(a))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shots", type=int, default=PLANNED_SHOTS)
    ap.add_argument("--two-qubit-error", type=float, default=8.0e-3,
                    help="the target backend's reported 2-qubit gate error; IBM "
                         "publishes this per device. Enters the systematic budget.")
    a = ap.parse_args()

    gamma_c = float(np.arccos(np.sqrt((R - S) / (T - S))))
    dev_payoff, dev_theta, dev_alpha = binding_deviation_at_zero_entanglement()
    u_dev = strategy(dev_theta, dev_alpha)

    qq_m, qq_sd = payoff_and_sd(Q_STRAT, Q_STRAT, gamma_c)
    dv_m, dv_sd = payoff_and_sd(u_dev, Q_STRAT, gamma_c)

    h = 1e-5
    def gain(g):
        return (payoff_and_sd(u_dev, Q_STRAT, g)[0] - payoff_and_sd(Q_STRAT, Q_STRAT, g)[0])
    slope = (gain(gamma_c + h) - gain(gamma_c - h)) / (2 * h)

    se = np.sqrt(qq_sd ** 2 + dv_sd ** 2) / np.sqrt(a.shots)
    sigma = float(se / abs(slope))

    # The systematic allowance is an ASSUMPTION, not a measurement. Amendment 1
    # withdrew the simulated table it was originally fitted to. It is retained at
    # its registered value so the criterion is not loosened, and the simulation
    # below shows it remains a conservative upper bound on every channel tried.
    systematic = 0.002 + 1.0 * a.two_qubit_error
    tolerance = 3 * sigma + systematic

    bias_channels = bias_table()
    grid_bias = discretization_bias()
    worst_bias = max(abs(entry["bias_rad"])
                     for channel in bias_channels.values()
                     for entry in channel.values())

    out = {
        "prediction": {
            "gamma_c": round(gamma_c, 9),
            "derivation": "cos^2(gamma_c) = (R-S)/(T-S) = 3/5",
            "payoffs": {"T": T, "R": R, "P": P, "S": S},
            "source_doc": "EWL_EQUILIBRIUM.md",
        },
        "protocol_fixed_in_advance": {
            "deviation_strategy": {
                "theta": round(dev_theta, 9), "alpha": round(dev_alpha, 9),
                "is_classical_defect": bool(abs(dev_theta - np.pi) < 1e-9
                                            and abs(dev_alpha) < 1e-9),
                "payoff_at_gamma_0": round(dev_payoff, 9),
                "selected_at_gamma": 0.0,
                "why": "at gamma=pi/2 the best reply to Q is Q, so an argmax taken "
                       "there returns Q and the deviation gain is identically zero",
            },
            "gamma_range": [0.0, round(float(np.pi / 2), 9)],
            "gamma_points": PLANNED_POINTS,
            "shots_per_circuit": a.shots,
            "circuits_per_gamma": 2,
            "estimator": "last sign change of (payoff(dev,Q) - payoff(Q,Q)), "
                         "linearly interpolated between adjacent grid points",
        },
        "expected_at_threshold": {
            "payoff_QQ": round(qq_m, 9), "sd_QQ": round(qq_sd, 9),
            "payoff_devQ": round(dv_m, 9), "sd_devQ": round(dv_sd, 9),
            "d_gain_d_gamma": round(float(slope), 6),
            "note": "(Q,Q) is deterministic at every gamma -- the outcome is always CC, "
                    "so on a noiseless device all shot noise comes from the (dev,Q) arm.",
        },
        "expected_device_bias": {
            "direction": "no direction is predicted. See amendment 1: the original "
                         "one-directional claim rested on a mechanism that does not "
                         "hold, so the pass criterion is symmetric.",
            "mechanism": "a depolarizing channel maps BOTH arms by the same affine "
                         "rule, p -> (1-q)p + q*uniform, so the gain becomes "
                         "(1-q)*gain_ideal. A positive rescaling cannot move a zero "
                         "crossing. The earlier claim that the falling (dev,Q) curve "
                         "is dragged down faster than the flat (Q,Q) curve is wrong: "
                         "both are pulled toward 2.25 in the same proportion.",
            "simulated_bias_by_channel": bias_channels,
            "discretization_bias_rad": round(grid_bias, 9),
            "largest_simulated_bias_rad": round(worst_bias, 9),
            "simulation_note": "computed by quantum_games.ewl_device_bias, a "
                               "density-matrix simulation of the same circuit, sweep "
                               "and estimator. No sampling, so no shot-noise floor.",
            "superseded_table": {
                "values_rad": [0.003986, 0.007384, 0.019214],
                "status": "withdrawn -- unreproducible",
                "note": "attributed to simulation at 200k shots per point, but no code "
                        "produced them and no channel tried here reproduces them; they "
                        "are 5x to 100x larger than any simulated bias.",
            },
        },
        "pass_criterion": {
            "primary": "|measured - predicted| <= 3*sigma_shot + systematic_budget",
            "tolerance_rad": round(tolerance, 6),
            "components": {"three_sigma_shot_rad": round(3 * sigma, 6),
                           "systematic_budget_rad": round(systematic, 6)},
            "two_qubit_error_assumed": a.two_qubit_error,
            "rationale": "shot noise alone is NOT the whole error budget: a real device "
                         "biases the crossing downward by an amount comparable to the "
                         "3-sigma shot tolerance. Pre-registering shot noise only would "
                         "have set a criterion the hardware fails for reasons unrelated "
                         "to whether the EWL threshold is correct. Both terms are fixed "
                         "here, in advance, and neither may be widened after the run.",
            "design_sigma_rad": round(sigma, 6),
            "symmetric": True,
            "decision_rule": "PASS if and only if |measured - 0.684719203| <= "
                             "0.026573 rad. FAIL otherwise, in either direction, "
                             "including when no sign change is found in the swept "
                             "range. There is no separate 'materially above' clause: "
                             "amendment 1 removed the one-directional bias claim that "
                             "would have justified an asymmetric rule.",
            "no_sign_change_is": "FAIL",
            "systematic_is_an_assumption": True,
            "systematic_vs_largest_simulated_bias": round(
                systematic / worst_bias, 2),
        },
        "declared": {
            "measurement_performed": False,
            "note": "This file must be committed before any hardware run. Its value is "
                    "its timestamp: pinned after the fact it proves nothing.",
        },
    }
    Path(a.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"predicted gamma_c        : {gamma_c:.9f}")
    print(f"deviation strategy       : theta={dev_theta:.6f} alpha={dev_alpha:.6f} "
          f"(classical Defect: {out['protocol_fixed_in_advance']['deviation_strategy']['is_classical_defect']})")
    print(f"design sigma @ {a.shots} shots: {sigma:.6f} rad")
    print(f"systematic budget        : {systematic:.6f} rad "
          f"(2q error {a.two_qubit_error:.1e})")
    print(f"PASS if |measured - {gamma_c:.6f}| <= {tolerance:.6f} rad")


if __name__ == "__main__":
    main()
