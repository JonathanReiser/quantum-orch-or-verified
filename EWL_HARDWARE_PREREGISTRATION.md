# EWL hardware run — prediction, registered before measurement

This document and `data/ewl_prediction.json` are committed **before** any
quantum hardware is used. That ordering is the whole point: pinned afterwards,
none of it would mean anything.

Regenerate: `python3 -m quantum_games.ewl_prediction --out data/ewl_prediction.json`

## The prediction

`EWL_EQUILIBRIUM.md` derives, from the payoff matrix alone, that the quantum
profile (Q,Q) becomes a **weak (non-strict) Nash equilibrium** of the quantised
Prisoner's Dilemma above

```
cos^2(gamma_c) = (R-S)/(T-S) = 3/5   ->   gamma_c = 0.684719203 rad
```

The equilibrium is weak, not strict: at and above `gamma_c` the best available
deviation *ties* with Q at exactly 3.0 rather than losing to it. Nothing strictly
beats Q, and nothing is strictly beaten by it.

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

```
3 * sigma_shot        = 0.016573 rad     (8192 shots, from the analytic payoff spreads)
systematic allowance  = 0.010000 rad     (0.002 + 1.0 * two_qubit_error, at 8e-3)
tolerance             = 0.026573 rad
```

**The systematic allowance is an assumption, not a measurement.** See amendment 1.
It is retained at its registered value so the criterion is not loosened.

`quantum_games.ewl_device_bias` computes three **illustrative** channels. They
are not a model of the target device and **not an upper bound on its error**:

| channel | bias at 3e-3 | bias at 8e-3 | bias at 2e-2 |
|---|---:|---:|---:|
| depolarizing | -0.000076 | -0.000076 | -0.000076 |
| amplitude damping | -0.000145 | -0.000260 | -0.000535 |
| readout | -0.000688 | -0.001708 | -0.004155 |

The largest simulated bias across these three channels is **0.004155 rad**. That
number bounds nothing. What these simulations establish is one negative result
and nothing more: **the mechanism the original registration invoked does not
produce a shift.** The depolarizing row is flat because it equals the 41-point
grid's own discretization bias — a depolarizing channel cannot move this crossing
at all (amendment 1).

What they omit is most of what a real device does: coherent and calibration
errors, crosstalk, qubit-dependent and pair-dependent error rates, drift between
calibration and execution, leakage, correlated readout error, and every gate the
transpiler introduces when it maps this circuit onto physical qubits and a native
basis. Each channel is applied uniformly to an idealised two-qubit circuit with
ideal one-qubit gates. A real device can therefore bias the crossing by **more**
than 0.004155 rad, and in either direction.

So the 0.010 rad allowance is not justified by these simulations. It is a
declared assumption whose only support is that it was registered before the run.

## Pass criterion

**PASS if and only if `|measured - 0.684719203| <= 0.026573` rad.**

The rule is **symmetric and exhaustive**. Every other outcome is a FAIL:

- `|measured - 0.684719203| > 0.026573` rad, in either direction;
- no sign change found anywhere in the swept range.

There is no "materially above" clause. The earlier text carried one, justified by
a predicted one-directional bias; amendment 1 withdrew that prediction, so no
asymmetry is justified and none is applied.

Neither term of the tolerance may be widened after the run. If the device is
worse than the assumed 8e-3 two-qubit error, the correct move is to re-register
with that device's published error rate before running — not to re-derive the
budget afterwards.

## Run protocol, fixed in advance

These were unspecified in the original registration and are fixed here, before
any hardware use.

| item | commitment |
|---|---|
| device selection | the IBM Quantum device with the **lowest published median two-qubit gate error** among those available to the account at submission time, subject to >= 2 qubits and support for the required basis gates. Ties broken by lowest median readout error, then alphabetically by device name. |
| eligibility gate | the device's published median two-qubit error must be <= 8e-3, the registered assumption. If no available device qualifies, the run does not proceed under this registration; it is re-registered against the qualifying device's published rate. |
| binding run | **the first eligible run is binding.** Its result is the result. |
| number of runs | exactly one. |
| calibration snapshot | the backend properties are fetched and committed **before** submission, and the reported median two-qubit error is taken from that snapshot. |
| permitted discards | only for a documented *infrastructure* failure that produces no usable counts: job error, cancellation, timeout, or a returned shot count differing from 2 x 8192. A discarded job is recorded with its job id and the reason. A run that returns counts is never discarded, whatever it shows. |
| forbidden | re-running after seeing a result, selecting among completed runs, changing the device after submission, or post-hoc recalibration of the budget. |

### Execution stack, frozen

A result is only reproducible if the circuit that ran can be rebuilt. These are
fixed now; a change to any of them is a different experiment and requires
re-registration.

| item | commitment |
|---|---|
| qiskit version | the exact `qiskit` and `qiskit-ibm-runtime` versions are recorded in the calibration snapshot before submission and pinned in `requirements.txt`. Transpilation is performed with those versions. |
| optimization level | `optimization_level=1`. Chosen because higher levels may re-synthesise the J / J-dagger pair, and the prediction is about *this* circuit's crossing, not an algebraically equivalent rewrite. |
| transpiler seed | `seed_transpiler=20260914`, fixed. Without it layout and routing are non-deterministic and the executed circuit is not reproducible. |
| physical qubit layout | `initial_layout` is fixed **before** submission to the connected pair with the lowest two-qubit gate error in the calibration snapshot, recorded as explicit physical qubit indices. Ties broken by lower sum of single-qubit errors, then by lower qubit index. |
| selected pair error | the two-qubit gate error **of that chosen pair**, from that snapshot, is recorded and is the number compared against the 8e-3 eligibility gate. The device-level median is used only for ranking devices, never as the eligibility figure. |
| dynamical decoupling / error suppression | none. No error mitigation, no measurement-error mitigation, no post-selection. Raw counts only. |

The distinction in the last two rows matters: a device can advertise a good
median while the pair actually used is worse. The gate that carries this
experiment is the one on the selected pair, so that is the error the registration
is conditioned on.

## Scope

**This tests a hardware implementation against an analytic result. It is not a
test of quantum cognition, Orch-OR, or any new physics, and it cannot discover
anything about nature.** `gamma_c = arccos(sqrt(3/5))` follows by algebra from
the payoff matrix and the restricted strategy set; it is not empirically at risk.
If the measurement disagrees, the conclusion is that the device, the transpiled
circuit, or the protocol is at fault — never that the threshold is wrong. What a
PASS buys is evidence that this apparatus reproduces a known number under a
budget fixed in advance; what a FAIL buys is a reason to debug the apparatus.

This tests a 1999 game-theory result and a threshold derived from it. It says
nothing about Orch-OR, consciousness, DAO governance, or human cooperation: the
"players" are two qubits under one encoding of one game. It is also conditional
on the restricted 2-parameter strategy set — Benjamin & Hayden (2001) show no
pure equilibrium survives over full SU(2), which `EWL_EQUILIBRIUM.md`
reproduces.


---

# Amendment 1 — 2026-09-14, before any hardware run

## The defect

The original registration justified its systematic allowance with a table of
simulated device biases:

| 2-qubit error | claimed bias |
|---|---|
| 3e-3 | 0.003986 |
| 8e-3 | 0.007384 |
| 2e-2 | 0.019214 |

attributed to "simulation at 200k shots per point". **No code in this repository
produced those numbers, and they could not be reconstructed.** They are withdrawn.

Worse than unreproducible, the stated mechanism is wrong. The registration said
depolarizing noise "pulls both payoff curves toward the uniform-mixture value
2.25, and the falling (dev,Q) curve is dragged down faster than the flat (Q,Q)
curve, moving the crossing left." A depolarizing channel maps every outcome
distribution by the same affine rule,

```
p_noisy = (1 - q) * p_ideal + q * uniform
```

so each payoff becomes `(1 - q) * payoff_ideal + q * 2.25`. Both arms carry the
same gate structure and therefore the same `q`, so the difference is
`(1 - q) * gain_ideal` — a positive rescaling, which **cannot move a zero
crossing**. The predicted leftward drag does not exist.

Simulating it confirms this: under depolarizing noise the measured crossing is
identical at 3e-3, 8e-3 and 2e-2, and equals the 41-point grid's own
discretization bias of -0.000076 rad. Channels that are not a common affine map
on both arms do move the crossing, but by far less than the withdrawn table:
amplitude damping by at most 0.000535 rad, readout error by at most 0.004155 rad.

## The correction

1. `quantum_games/ewl_device_bias.py` computes all of the above from the same
   circuit, sweep and estimator the hardware run will use. It is registered in
   the ledger and regenerated in CI, so the numbers are checkable rather than
   asserted.
2. The systematic allowance is **relabelled as an assumption**. Its value is
   unchanged at 0.010 rad, so the criterion is not loosened. The simulation shows
   these simulations do not justify it: they are three illustrative channels on
   an idealised circuit, not a device model and not a bound.
3. The pass criterion is now **symmetric and exhaustive**, and the vague
   "materially above" clause is removed. Its only justification was the
   one-directional bias claim withdrawn above.
4. Device selection, run count, binding-run rule, calibration snapshot and
   permitted discards are fixed in the run protocol table.
5. The scope section states plainly that this tests an implementation against an
   analytic result.
6. The equilibrium is described as weak (non-strict).

`gamma_c` is unchanged at 0.684719203 rad and the tolerance is unchanged at
0.026573 rad. Nothing here makes the test easier to pass.

## What remains an assumption

The 0.010 rad allowance is the least-supported number in this document, and the
simulations do not repair that. They rule out one mechanism; they do not
characterise the device. A real backend can bias this crossing by more than any
figure in the table above, through channels not simulated here — coherent error,
crosstalk, drift, transpiled gate depth — so 0.010 rad is neither demonstrably
conservative nor demonstrably adequate. It is retained rather than tightened only because
lowering a registered tolerance is still a change to a registered criterion, and
the conservative reading is that a pre-run change should not be able to cut
either way at the author's discretion. A future registration should derive the
allowance from the device's own measured channel rather than assume it.
