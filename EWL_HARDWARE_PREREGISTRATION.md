# EWL hardware run — prediction, registered before measurement

This document and `data/ewl_prediction.json` are committed **before** any
quantum hardware is used. That ordering is the whole point: pinned afterwards,
none of it would mean anything.

Regenerate: `python3 -m quantum_games.ewl_prediction --out data/ewl_prediction.json`

## The prediction

`EWL_EQUILIBRIUM.md` derives, from the payoff matrix alone, that the quantum
profile (Q,Q) becomes a Nash equilibrium of the quantised Prisoner's Dilemma
above

```
cos^2(gamma_c) = (R-S)/(T-S) = 3/5   ->   gamma_c = 0.684719203 rad
```

The hardware run measures where a defector's payoff against a quantum
cooperator falls through 3.0. **Predicted crossing: 0.684719203 rad.**

## Protocol, fixed in advance

| item | value |
|---|---|
| deviating strategy | theta = pi, alpha = 0 (classical Defect) |
| chosen at | gamma = 0, before the sweep |
| gamma range / points | 0 to pi/2, 41 points |
| shots per circuit | 8192 |
| circuits per gamma | 2 |
| estimator | last sign change of payoff(dev,Q) − payoff(Q,Q), linearly interpolated |

The deviating strategy is selected at **gamma = 0** deliberately. At maximal
entanglement the best reply to Q is Q itself, so an argmax taken there returns
Q and the deviation gain is identically zero. Selecting in the classical limit
asks the well-posed question instead: take the strategy that beats Q with no
entanglement, then raise entanglement until it stops winning.

## Error budget

Shot noise is **not** the whole budget. A real device biases this measurement,
and pre-registering shot noise alone would set a criterion the hardware fails
for reasons unrelated to whether the threshold is right.

```
3 * sigma_shot        = 0.016573 rad     (8192 shots, from the analytic payoff spreads)
systematic budget     = 0.010000 rad     (0.002 + 1.0 * two_qubit_error, at 8e-3)
tolerance             = 0.026573 rad
```

The systematic term was measured in simulation at 200k shots per point, so the
shot-noise floor (~0.0011 rad) sits well below the residual:

| 2-qubit error | measured bias |
|---|---|
| 3e-3 | 0.003986 |
| 8e-3 | 0.007384 |
| 2e-2 | 0.019214 |

**The bias is one-directional.** Depolarizing noise pulls both payoff curves
toward the uniform-mixture value 2.25, and the falling (dev,Q) curve is dragged
down faster than the flat (Q,Q) curve, moving the crossing left. So the measured
crossing should land at or below gamma_c, never materially above it.

## Pass criterion

**PASS if |measured − 0.684719203| <= 0.026573 rad.**

Falsified if the crossing lies outside that tolerance, lies materially *above*
gamma_c (the bias direction is predicted, so a high result is not a near miss —
it is a wrong result), or if no sign change is found in the swept range.

Neither term of the tolerance may be widened after the run. If the device is
worse than the assumed 8e-3 two-qubit error, the correct move is to re-register
with that device's published error rate before running — not to re-derive the
budget afterwards.

## Scope

This tests a 1999 game-theory result and a threshold derived from it. It says
nothing about Orch-OR, consciousness, DAO governance, or human cooperation: the
"players" are two qubits under one encoding of one game. It is also conditional
on the restricted 2-parameter strategy set — Benjamin & Hayden (2001) show no
pure equilibrium survives over full SU(2), which `EWL_EQUILIBRIUM.md`
reproduces.
