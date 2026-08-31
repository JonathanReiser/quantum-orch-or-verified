# quantum-orch-or-verified

Two small, independently reproducible results, extracted from a larger project
whose headline claims did not hold up.

**Origin.** This repo is the verified subset of
[quantum-orch-or](https://github.com/JonathanReiser/quantum-orch-or), a project
that published seven empirical claims — "835,000 Snapshot DAO votes," an
"86.7% error reduction," "R² = 0.98," entanglement "doubling" public-good
approval — none of which were produced by the code that cited them. The full
accounting is in that repo's
[CORRECTIONS.md](https://github.com/JonathanReiser/quantum-orch-or/blob/main/CORRECTIONS.md).
Nothing here depends on any of the retracted claims, and none of the retracted
code is included. Everything in this repo is checked by CI on every commit —
see [Verification](#verification) below.

## What's here

### 1. DAO proposal approval is close to unpredictable ([`dao_governance/`](dao_governance/))

905 closed, cleanly-binary Snapshot proposals across Uniswap, Arbitrum,
Optimism, Gitcoin, and Aave — 6,242,940 vote records, fetched reproducibly from
the Snapshot GraphQL API. On a hindsight-free temporal split, the lowest error
comes from ignoring the proposal and predicting the historical median
(10.44pp MAE); a ridge model on pre-vote features does *worse* (11.20pp), and
every R² is within noise of zero.

A follow-up asks a better-posed question — will a proposal be *contested*
(final YES share in [5%, 95%])? That's genuinely predictable (AUC 0.660, 95%
CI [0.555, 0.763]) — but the signal turns out to be entirely which DAO it is,
not the proposal. Conditioning on venue, the same model collapses to a median
AUC of 0.416 — below chance. Some DAOs argue, others rubber-stamp; the
individual proposal adds nothing. Details: [`dao_governance/README.md`](dao_governance/README.md).

### 2. Entanglement buys a real equilibrium — smaller and narrower than claimed ([`quantum_games/`](quantum_games/))

A reproduction of Eisert–Wilkens–Lewenstein (1999) and the Benjamin–Hayden
(1999) rebuttal, in the quantised Prisoner's Dilemma. In a restricted strategy
space, entanglement above a *derived* threshold (cos²γ_c = 3/5,
γ_c ≈ 0.6847 rad) creates a cooperative equilibrium beating both classical
Nash and the best classical correlated equilibrium. Widen the strategy space
to full SU(2) and that particular equilibrium disappears — but a different one
survives: at maximal entanglement, an exact closed-form result shows every
strategy becomes indifferent, and the resulting equilibrium is worth 62.5% of
the gap between the classical trap and full cooperation. It is not
cooperation, though — the outcome distribution is uniform across all four
game cells. Details: [`quantum_games/README.md`](quantum_games/README.md).

## Verification

[`tools/ledger/`](tools/ledger/) contains a mechanism, not just documentation:
`check_ledger.py` re-runs every benchmark and equilibrium search in this repo
and byte-compares the output to what's committed, then checks that every
number quoted above still appears verbatim in the file that's supposed to
contain it. It runs in CI on every push. A number that stops tracing to the
command that produces it fails the build — that's the direct response to how
the parent project's claims went wrong.

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
python3 tools/ledger/check_ledger.py
```

## Reproducing from scratch

```bash
python3 -m dao_governance.fetch_snapshot_dataset --out data/snapshot_dao_dataset.json
python3 -m dao_governance.benchmark_snapshot_real
python3 -m dao_governance.benchmark_contestedness
python3 -m quantum_games.ewl_equilibrium
python3 -m quantum_games.ewl_mixed_equilibrium
```

Only `dao_governance.fetch_snapshot_dataset` touches the network; everything
else runs on the committed data and is exactly reproducible (no sampling, no
fitted optimiser — see each module's docstring for why that was a deliberate
choice, not an oversight).

## Dependencies

`numpy`, `scipy`, `pytest`. Nothing else — no quantum computing framework is
used or needed for either result; the game-theory work is closed-form linear
algebra over 2×2 complex matrices, not a simulated circuit.

## License

MIT. See [LICENSE](LICENSE).
