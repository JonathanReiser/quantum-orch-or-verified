# Entanglement and the quantised Prisoner's Dilemma

```bash
python3 -m quantum_games.ewl_equilibrium
python3 -m quantum_games.ewl_mixed_equilibrium
```

## The setup

The Eisert–Wilkens–Lewenstein quantisation (*Phys. Rev. Lett.* **83**, 3077,
1999). Two players share an entangled qubit pair, each applies a local
unitary, the pair is disentangled, and the result is measured. Payoffs:
(C,C)=(3,3), (C,D)=(0,5), (D,C)=(5,0), (D,D)=(1,1). An entanglement parameter
γ tunes the game: γ=0 is the ordinary classical dilemma, γ=π/2 is maximal
entanglement.

Two classical baselines matter here, not one. The Nash equilibrium (D,D) pays
(1,1). So does the best **correlated** equilibrium — defection strictly
dominates, so every correlated equilibrium places all its mass on mutual
defection (solved by linear programming, not asserted). That answers the
standard objection that entanglement is just shared randomness: correlated
randomness provably cannot do better than (1,1) here.

## Result 1 — a real equilibrium above a derived threshold

[`ewl_equilibrium.py`](ewl_equilibrium.py), restricted two-parameter strategy
space, exhaustive search over 741,321 profiles: above a threshold, (Q,Q) with
Q = diag(i, −i) becomes the unique pure Nash equilibrium, paying **(3,3)** —
beating both classical baselines outright.

The threshold is derived, not fitted. Against Q the best deviation is D,
paying (T−S)cos²γ + S, which meets the cooperative payoff at

> cos²(γ_c) = (R−S)/(T−S) = 3/5,  γ_c = arccos(√(3/5)) ≈ 0.684719 rad ≈ 39.23°

confirmed against a dense numerical sweep to |error| ≈ 2×10⁻¹⁰.

## Result 2 — and it depends on an unjustified restriction

Widen the strategy space to full SU(2) (4,826,809 profiles): no pure-strategy
Nash equilibrium survives above γ ≈ 0.63, and (Q,Q) is never one. This is the
standard Benjamin & Hayden (*Phys. Rev. Lett.* **87**, 069801) objection,
reproduced rather than asserted.

## Result 3 — but a different equilibrium survives

"No pure equilibrium" is not "no equilibrium" — Glicksberg's theorem
guarantees a mixed one, since SU(2) is compact and payoffs are continuous.
[`ewl_mixed_equilibrium.py`](ewl_mixed_equilibrium.py) computes it exactly at
maximal entanglement, via the twirl identity
E_U[(I⊗U)σ(I⊗U)†] = Tr_B(σ)⊗I/2:

> u_A(θ, γ) = (T+R+P+S)/4 − [(T+P−R−S)/4] · cos²γ · cos θ

matching the exact channel to ~10⁻¹⁵ over random strategies and γ. The payoff
depends on neither phase, and its θ-dependence carries cos²γ — vanishing
exactly at γ=π/2 and nowhere below. Every strategy is then indifferent, and
Haar-uniform play is a symmetric equilibrium:

| equilibrium | payoff |
|---|---|
| classical Nash (D,D) | 1.00 |
| best correlated equilibrium | 1.00 |
| **full SU(2) Haar equilibrium** | **2.25** |
| restricted-space (Q,Q) | 3.00 |

**62.5%** of the distance from the classical trap to full cooperation
survives widening the strategy space.

## Three limits, stated rather than buried

1. **Not cooperation.** The equilibrium outcome distribution is uniform
   across all four cells — 25% mutual cooperation, 50% one-sided
   exploitation. 2.25 is the flat average of the payoff matrix. Players
   randomise; they don't coordinate.
2. **A knife-edge.** Haar-uniform play is an equilibrium only at γ=π/2
   exactly. The regime 0.63 < γ < π/2 is open.
3. **Uniqueness not established.** One equilibrium is certified exactly; no
   claim is made that it's the only one, or that 2.25 is *the* value of the
   game.

None of this transfers to real-world collective decision-making. It's a
result about a specific two-player game, not a claim about DAOs, markets, or
consensus mechanisms.

## Method notes

* Payoffs are computed analytically from the statevector. No sampling, no
  fitted optimiser, exactly reproducible.
* Equilibria are found by exhaustive grid search, never an optimiser. Grid
  coarseness biases *towards* finding spurious equilibria, so a null result
  ("no equilibrium found") is the conservative direction; positive claims are
  separately verified against a dense random deviation search.
* Replicator dynamics was tried for Result 3 and rejected — the equilibrium
  point is unstable under replicator dynamics, so the dynamics wander away
  from it rather than converge. The closed form has no such failure mode.
