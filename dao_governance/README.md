# DAO proposal approval is close to unpredictable

```bash
python3 -m dao_governance.benchmark_snapshot_real
python3 -m dao_governance.benchmark_contestedness
```

## The dataset

[`fetch_snapshot_dataset.py`](fetch_snapshot_dataset.py) pulls every closed
proposal from five DAOs (Uniswap, Arbitrum, Optimism, Gitcoin, Aave) via the
Snapshot GraphQL hub, keeping only those with a settled tally and an
unambiguous binary ballot:

| | |
|---|---|
| Closed proposals retrieved | 1,864 |
| Kept (settled tally, unambiguous binary ballot) | **905** |
| Vote records across kept proposals | **6,242,940** |
| Date range | 2020-09-11 to 2026-08-20 |

DAO votes pass overwhelmingly: the median proposal carries **99.75% YES**, and
74% clear 90%.

## Predicting the YES share fails

[`benchmark_snapshot_real.py`](benchmark_snapshot_real.py) evaluates on a
temporal split — fit on the earlier 633 proposals, test on the later 272 — so
nothing is predicted with hindsight.

| model | MAE (pp) | RMSE (pp) | R² |
|---|---|---|---|
| constant (train mean) | 21.66 | 26.55 | −0.109 |
| **constant (train median)** | **10.44** | 26.73 | −0.125 |
| per-DAO historical mean | 17.26 | 24.55 | 0.052 |
| ridge on pre-vote features | 11.20 | 25.04 | 0.013 |

The lowest error comes from ignoring the proposal entirely and predicting the
historical median. A ridge model given DAO identity, proposal length, ballot
structure, voting-window length, quorum, and the DAO's own prior approval
history does *worse* than that constant, and every R² sits within noise of
zero. From information available before a vote closes, YES share is close to
unpredictable beyond "it will probably pass."

## Predicting whether it's contested works — but not for the reason it looks like

Rubber-stamp votes dominate the dataset, so most of the variance above is
simply absent. A better-posed question: will a proposal be **contested**
(final YES share in [5%, 95%])? Base rate 26.5%, and it varies sharply by
venue (Optimism 47.7%, Arbitrum 40.1%, Uniswap 17.9%, Gitcoin 12.2%, Aave
10.0%).

[`benchmark_contestedness.py`](benchmark_contestedness.py), same temporal
split, AUC as the headline metric (the base rate shifts 30.0% → 18.4% across
the split, so accuracy would flatter every model):

| model | AUC | 95% CI | P(AUC ≤ 0.5) |
|---|---|---|---|
| base rate constant | 0.500 | — | — |
| per-DAO base rate | 0.628 | [0.527, 0.719] | 0.0095 |
| **all features** | **0.660** | [0.555, 0.763] | 0.0010 |
| content only | 0.651 | [0.547, 0.755] | 0.0010 |

That's real, above-chance signal. But a content-only model that never sees
which DAO the proposal belongs to still reaches 0.651 — which looks like
proposals themselves carry signal. Refitting the same model **inside each
DAO** shows it doesn't:

| DAO | content-only AUC |
|---|---|
| Uniswap | 0.516 |
| Arbitrum | 0.416 |
| Optimism | 0.541 |
| Gitcoin | 0.188 |
| Aave | 0.137 |

**Median within-DAO AUC: 0.416** — below chance. Simpson's paradox: the
strongest pooled feature was `duration_days`, but three DAOs use a fixed
voting window, so duration fingerprints the venue rather than describing the
proposal.

**The honest claim is about organisations, not proposals.** Some DAOs argue
and others rubber-stamp, stably enough to forecast; which proposal is in
front of them doesn't measurably change that.

## Method notes

* Every feature is computable *before* the vote closes; nothing here uses
  post-hoc voter counts or realised turnout.
* A DAO's own history is folded in as a strictly past-only running mean —
  row *i* never sees its own outcome or a later one.
* R² is reported as computed, including negative values. No clamping.
* AUC's confidence intervals are 2,000-sample bootstraps.
