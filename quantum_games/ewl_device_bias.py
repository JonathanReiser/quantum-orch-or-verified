#!/usr/bin/env python3
"""
ewl_device_bias.py -- how far a noisy device moves the measured EWL crossing.

Written to replace three literals. EWL_HARDWARE_PREREGISTRATION.md originally
carried a table of device biases (0.003986 / 0.007384 / 0.019214 rad) attributed
to "simulation at 200k shots per point", but no code in this repository produced
them and they could not be reconstructed. See the amendment in that document.

This module computes the quantity honestly, from a density-matrix simulation of
the same circuit the hardware run will execute, with the same 41-point sweep and
the same crossing estimator.

The headline result is negative and worth stating up front: **a depolarizing
channel cannot move this crossing at all.** The estimator locates the root of

    gain(gamma) = payoff(dev, Q) - payoff(Q, Q)

and a depolarizing channel maps every outcome distribution to

    p_noisy = (1 - q) * p_ideal + q * uniform

so each payoff becomes ``(1 - q) * payoff_ideal + q * 2.25`` -- the same affine
map for both arms, because both circuits carry the same gate structure. The
difference is therefore ``(1 - q) * gain_ideal``, a positive rescaling, and a
positive rescaling cannot move a zero crossing. The registered document's stated
mechanism ("the falling curve is dragged down faster than the flat one") does not
hold: both are dragged toward 2.25 in the same proportion.

Channels that are *not* a common affine map on both arms -- amplitude damping,
readout error -- do move the crossing, but by far less than the registered table
claimed. Those are computed here too so the claim is checkable rather than
asserted.

Usage: python3 -m quantum_games.ewl_device_bias --out data/ewl_device_bias.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from quantum_games.ewl_equilibrium import (
    PAYOFF_A, Q_STRAT, entangling_gate, strategy)

TWO_QUBIT_ERRORS = (3.0e-3, 8.0e-3, 2.0e-2)
SWEEP_POINTS = 41
UNIFORM = np.eye(4) / 4.0


def _depolarize(rho, q):
    return (1.0 - q) * rho + q * np.trace(rho) * UNIFORM


def outcome_probs(u_a, u_b, gamma, two_qubit_error, channel):
    """Density-matrix simulation of the EWL circuit under one noise channel.

    Both two-qubit gates (J and J-dagger) carry the depolarizing error. The
    one-qubit strategy gates are treated as ideal, which is the usual hierarchy
    on superconducting hardware and is the assumption the preregistration makes
    by budgeting from the two-qubit error alone.
    """
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = 1.0
    gate = entangling_gate(gamma)

    rho = gate @ rho @ gate.conj().T
    rho = _depolarize(rho, two_qubit_error)

    unitary = np.kron(u_a, u_b)
    rho = unitary @ rho @ unitary.conj().T

    if channel == "amplitude_damping":
        k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - two_qubit_error)]], dtype=complex)
        k1 = np.array([[0.0, np.sqrt(two_qubit_error)], [0.0, 0.0]], dtype=complex)
        damped = np.zeros_like(rho)
        for left in (k0, k1):
            for right in (k0, k1):
                kraus = np.kron(left, right)
                damped += kraus @ rho @ kraus.conj().T
        rho = damped

    inverse = gate.conj().T
    rho = inverse @ rho @ inverse.conj().T
    rho = _depolarize(rho, two_qubit_error)

    probabilities = np.real(np.diag(rho))
    if channel == "readout":
        flip = np.array([[1.0 - two_qubit_error, two_qubit_error],
                         [two_qubit_error, 1.0 - two_qubit_error]])
        probabilities = np.kron(flip, flip) @ probabilities
    return probabilities


def measured_crossing(two_qubit_error, channel="depolarizing", points=SWEEP_POINTS):
    """The registered estimator: last sign change, linearly interpolated."""
    deviation = strategy(np.pi, 0.0)
    gammas = np.linspace(0.0, np.pi / 2.0, points)
    gain = np.array([
        outcome_probs(deviation, Q_STRAT, g, two_qubit_error, channel) @ PAYOFF_A
        - outcome_probs(Q_STRAT, Q_STRAT, g, two_qubit_error, channel) @ PAYOFF_A
        for g in gammas
    ])
    changes = np.nonzero(np.sign(gain[:-1]) != np.sign(gain[1:]))[0]
    if changes.size == 0:
        return None
    index = int(changes[-1])
    span = gammas[index + 1] - gammas[index]
    return float(gammas[index] + span * gain[index] / (gain[index] - gain[index + 1]))


def discretization_bias(points=SWEEP_POINTS):
    """Bias of the estimator with no noise at all -- the grid's own contribution."""
    gamma_c = float(np.arccos(np.sqrt(3.0 / 5.0)))
    return float(measured_crossing(0.0, "depolarizing", points) - gamma_c)


def bias_table(points=SWEEP_POINTS):
    gamma_c = float(np.arccos(np.sqrt(3.0 / 5.0)))
    table = {}
    for channel in ("depolarizing", "amplitude_damping", "readout"):
        entries = {}
        for error in TWO_QUBIT_ERRORS:
            crossing = measured_crossing(error, channel, points)
            entries[f"{error:g}"] = {
                "crossing_rad": round(crossing, 9),
                "bias_rad": round(crossing - gamma_c, 9),
            }
        table[channel] = entries
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--points", type=int, default=SWEEP_POINTS)
    args = parser.parse_args()

    gamma_c = float(np.arccos(np.sqrt(3.0 / 5.0)))
    grid = discretization_bias(args.points)
    table = bias_table(args.points)
    depolarizing = [abs(v["bias_rad"]) for v in table["depolarizing"].values()]
    worst = max(abs(v["bias_rad"])
                for channel in table.values() for v in channel.values())

    out = {
        "gamma_c": round(gamma_c, 9),
        "sweep_points": args.points,
        "estimator": "last sign change of payoff(dev,Q) - payoff(Q,Q), "
                     "linearly interpolated between adjacent grid points",
        "method": "density-matrix simulation; no sampling, so these are exact "
                  "channel biases with no shot-noise floor to subtract",
        "discretization_bias_rad": round(grid, 9),
        "bias_by_channel": table,
        "finding": {
            "depolarizing_cannot_move_the_crossing": bool(
                max(depolarizing) - abs(grid) < 1e-9),
            "why": "a depolarizing channel maps both arms by the same affine rule, "
                   "p -> (1-q)p + q*uniform, so the gain becomes (1-q)*gain_ideal. "
                   "A positive rescaling cannot move a zero crossing. The residual "
                   "equals the discretization bias of the 41-point grid.",
            "largest_bias_any_channel_rad": round(worst, 9),
            "supersedes": "the unreproducible table 0.003986 / 0.007384 / 0.019214 rad",
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"gamma_c                         : {gamma_c:.9f}")
    print(f"discretization bias (41 points) : {grid:+.9f} rad")
    for channel, entries in table.items():
        for error, values in entries.items():
            print(f"  {channel:18s} 2q={error:>7s} -> bias {values['bias_rad']:+.9f} rad")
    print(f"largest bias, any channel       : {worst:.9f} rad")


if __name__ == "__main__":
    main()
