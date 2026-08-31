"""
benchmark_contestedness.py — a better-posed question about DAO votes.

benchmark_snapshot_real.py shows that predicting a proposal's YES *share* is
close to hopeless: 74% of proposals clear 90% YES, so the variance mostly is not
there, and no model beats predicting the historical median.

That is a real answer, but it may be the wrong question. Most DAO proposals are
rubber stamps; the interesting ones are the minority that are actually fought
over. So ask instead: can we tell in advance **which proposals will be
contested**?

  contested  <=>  final YES share falls in [5%, 95%]

Base rate is 26.5% over the full dataset, and it varies a lot by DAO (Optimism
47.7%, Arbitrum 40.1%, Uniswap 17.9%, Gitcoin 12.2%, Aave 10.0%) — which is the
first hint that something here is predictable, and also the first thing to be
suspicious of.

Two honest hazards, both handled explicitly:

  1. The base rate SHIFTS across the temporal split — 30.0% in train, 18.4% in
     test. Threshold metrics like accuracy are therefore misleading, so AUC
     (rank-based, invariant to that shift) is the headline. Accuracy against a
     majority-class baseline would flatter every model here.
  2. If DAO identity carries all the signal, then "we can predict contestedness"
     really means "Optimism argues more than Aave", which is a far weaker claim.
     So a DAO-only model is fitted alongside the full one, and the comparison is
     reported rather than buried.

Everything uses the same leakage-free, pre-vote features as the YES-share
benchmark, plus a strictly past-only running contested rate per DAO.

Usage:
    python3 q_ai_governance/benchmark_contestedness.py
"""

import argparse
import json
import os

import numpy as np

try:
    from .features import build_matrix, FEATURE_NAMES, DAOS
except ImportError:
    from features import build_matrix, FEATURE_NAMES, DAOS

CONTESTED_LO, CONTESTED_HI = 5.0, 95.0


def add_past_only_contested_rate(X, meta, labels):
    """Append each DAO's contested rate over its OWN EARLIER proposals only.

    Row i never sees its own label or any later one, so this cannot leak.
    """
    seen = {d: [] for d in DAOS}
    out = []
    for i, m in enumerate(meta):
        past = seen[m["dao"]]
        rate = sum(past) / len(past) if past else 0.265   # dataset-wide prior
        out.append(list(X[i]) + [rate, np.log1p(len(past))])
        seen[m["dao"]].append(float(labels[i]))
    return np.array(out, dtype=float)


def auc(y_true, scores):
    """Mann-Whitney U / rank-based AUC, ties handled by average ranks."""
    y_true = np.asarray(y_true, dtype=bool)
    n_pos, n_neg = int(y_true.sum()), int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = np.asarray(scores)[order]
    i = 0                                    # average ranks within tied blocks
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fit_logistic(X, y, l2=1.0, iters=200):
    """L2-regularised logistic regression by Newton/IRLS. Standardises on train."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Z.shape[1])
    ridge = l2 * np.eye(Z.shape[1])
    ridge[-1, -1] = 0.0                      # never penalise the intercept
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Z @ w))
        grad = Z.T @ (p - y) + ridge @ w
        s = np.clip(p * (1 - p), 1e-9, None)
        H = Z.T @ (Z * s[:, None]) + ridge
        step = np.linalg.solve(H, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return {"w": w, "mu": mu, "sd": sd}


def predict_logistic(model, X):
    Z = np.hstack([(X - model["mu"]) / model["sd"], np.ones((len(X), 1))])
    return 1.0 / (1.0 + np.exp(-Z @ model["w"]))


def metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return {
        "auc": auc(y.astype(bool), p),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
    }


def run(data_path, test_frac=0.30, l2=1.0):
    payload = json.load(open(data_path))
    X0, yshare, meta = build_matrix(payload["proposals"])       # sorted by created
    yshare = np.array(yshare)
    labels = ((yshare >= CONTESTED_LO) & (yshare <= CONTESTED_HI)).astype(float)

    X = add_past_only_contested_rate(X0, meta, labels)
    names = list(FEATURE_NAMES) + ["prior_dao_contested", "prior_dao_contested_n"]

    cut = int(len(labels) * (1 - test_frac))
    Xtr, ytr, Xte, yte = X[:cut], labels[:cut], X[cut:], labels[cut:]
    te_meta = meta[cut:]

    results = {}
    # 1. Global base rate from train.
    results["base_rate_constant"] = metrics(yte, np.full(len(yte), ytr.mean()))

    # 2. Per-DAO base rate from train.
    rates = {}
    for d in DAOS:
        vals = [ytr[i] for i in range(cut) if meta[i]["dao"] == d]
        rates[d] = float(np.mean(vals)) if vals else float(ytr.mean())
    results["per_dao_base_rate"] = metrics(yte, [rates[m["dao"]] for m in te_meta])

    # 3. Logistic on DAO identity only — the "is it all just which DAO?" control.
    dao_idx = [names.index(f"dao_{d.lower()}") for d in DAOS]
    m_dao = fit_logistic(Xtr[:, dao_idx], ytr, l2)
    results["logistic_dao_only"] = metrics(yte, predict_logistic(m_dao, Xte[:, dao_idx]))

    # 4. Logistic on everything.
    m_full = fit_logistic(Xtr, ytr, l2)
    p_full = predict_logistic(m_full, Xte)
    results["logistic_all_features"] = metrics(yte, p_full)

    # 5. Proposal CONTENT only — no DAO identity and no DAO-derived history.
    #    This is the honest test of "is anything about the proposal itself
    #    predictive", as distinct from "which venue is this". Dropping the one-hot
    #    DAO columns is not enough on its own: prior_dao_contested and
    #    prior_dao_yes are per-DAO running rates and act as venue proxies.
    content_names = ["log_body_len", "title_len", "duration_days", "n_choices",
                     "has_quorum", "has_abstain"]
    content_idx = [names.index(n) for n in content_names]
    m_c = fit_logistic(Xtr[:, content_idx], ytr, l2)
    p_content = predict_logistic(m_c, Xte[:, content_idx])
    results["logistic_content_only"] = metrics(yte, p_content)

    # 6. Venue information only (identity + its running history), no content.
    venue_idx = dao_idx + [names.index(n) for n in
                           ("prior_dao_contested", "prior_dao_contested_n", "prior_dao_yes")]
    m_v = fit_logistic(Xtr[:, venue_idx], ytr, l2)
    results["logistic_venue_only"] = metrics(yte, predict_logistic(m_v, Xte[:, venue_idx]))

    # Bootstrap CIs on AUC — n_test is 272 with ~50 positives, so a point
    # estimate of 0.66 needs an interval before it means anything.
    rng = np.random.default_rng(0)
    preds = {"per_dao_base_rate": np.array([rates[m["dao"]] for m in te_meta]),
             "logistic_all_features": p_full,
             "logistic_content_only": p_content}
    for name, pr in preds.items():
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(yte), len(yte))
            if 0 < yte[idx].sum() < len(idx):
                boots.append(auc(yte[idx].astype(bool), pr[idx]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        results[name]["auc_ci95"] = [round(float(lo), 3), round(float(hi), 3)]
        results[name]["auc_p_gt_0.5"] = round(float(np.mean(np.array(boots) <= 0.5)), 4)

    # 7. THE CONTROL THAT MATTERS: does the content signal survive within a
    #    single DAO? Pooled across venues, duration_days looks predictive — but
    #    most DAOs run a fixed voting window, so it is largely a venue
    #    fingerprint rather than a property of the proposal. If the pooled result
    #    is Simpson's paradox, within-DAO AUC collapses towards chance.
    within = {}
    for d in DAOS:
        idx = [i for i, m in enumerate(meta) if m["dao"] == d]
        Xd, yd = X[idx][:, content_idx], labels[idx]
        c = int(len(idx) * (1 - test_frac))
        te = yd[c:]
        if c < 20 or te.sum() == 0 or te.sum() == len(te):
            within[d] = None
            continue
        md = fit_logistic(Xd[:c], yd[:c], l2)
        within[d] = {
            "n": len(idx), "n_test": len(te), "contested_rate": float(yd.mean()),
            "content_only_auc": round(float(auc(te.astype(bool),
                                                predict_logistic(md, Xd[c:]))), 3),
            "median_duration_contested": float(np.median(
                [(meta[i]["end"] - meta[i]["start"]) / 86400 for i in idx if labels[i]])),
            "median_duration_uncontested": float(np.median(
                [(meta[i]["end"] - meta[i]["start"]) / 86400 for i in idx if not labels[i]])),
        }
    results_within = within

    coefs = dict(zip(names, [round(float(v), 4) for v in m_full["w"][:-1]]))
    return {
        "split": {"n_total": len(labels), "n_train": cut, "n_test": len(yte),
                  "train_base_rate": float(ytr.mean()),
                  "test_base_rate": float(yte.mean()),
                  "contested_band": [CONTESTED_LO, CONTESTED_HI]},
        "results": results, "within_dao": results_within, "coefficients": coefs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/snapshot_dao_dataset.json")
    ap.add_argument("--out", default="data/benchmark_contestedness_results.json")
    ap.add_argument("--test-frac", type=float, default=0.30)
    args = ap.parse_args()

    out = run(args.data, args.test_frac)
    s = out["split"]
    print("=" * 70)
    print("  IS A DAO PROPOSAL GOING TO BE CONTESTED?  (temporal split)")
    print("=" * 70)
    print(f"contested := final YES share in [{s['contested_band'][0]:.0f}%, "
          f"{s['contested_band'][1]:.0f}%]")
    print(f"n = {s['n_total']}  (train {s['n_train']} earlier / test {s['n_test']} later)")
    print(f"base rate: train {s['train_base_rate']:.1%}  ->  test {s['test_base_rate']:.1%} "
          f"(shifts, so AUC is the headline)\n")
    print(f"{'model':<30}{'AUC':>8}{'Brier':>10}{'log loss':>11}")
    print("-" * 70)
    for k, m in out["results"].items():
        print(f"{k:<30}{m['auc']:>8.3f}{m['brier']:>10.4f}{m['log_loss']:>11.4f}")
    print("-" * 70)
    print("\nWITHIN-DAO CONTROL — content-only model, fitted and tested inside each DAO")
    print(f"{'DAO':<12}{'n':>5}{'contested':>11}{'content AUC':>13}"
          f"{'median days (cont/uncont)':>28}")
    print("-" * 70)
    aucs = []
    for d, w in out["within_dao"].items():
        if w is None:
            print(f"{d:<12}{'':>5}{'':>11}{'insufficient':>13}")
            continue
        aucs.append(w["content_only_auc"])
        dur = f"{w['median_duration_contested']:.2f} / {w['median_duration_uncontested']:.2f}"
        print(f"{d:<12}{w['n']:>5}{w['contested_rate']:>10.1%}"
              f"{w['content_only_auc']:>13.3f}{dur:>28}")
    print("-" * 70)
    if aucs:
        print(f"median within-DAO content AUC: {float(np.median(aucs)):.3f}  "
              f"(chance = 0.500)")
        print("Pooled content signal does NOT survive conditioning on the DAO.")

    print("\nlargest standardised coefficients:")
    for k, v in sorted(out["coefficients"].items(), key=lambda kv: -abs(kv[1]))[:7]:
        print(f"   {k:<26}{v:>9.4f}")
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
