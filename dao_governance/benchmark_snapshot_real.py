"""
benchmark_snapshot_real.py — honest benchmark on the real Snapshot dataset.

Replaces benchmark_real_dao_data.py, which scored an untrained agent against
five hand-written proposals whose "public_good_score" and "roi_score" inputs
were assigned by the author after the outcomes were known, and which clamped
its reported R^2 to a 0.98 ceiling.

Here:
  * the data is every closed, cleanly-binary proposal from the five DAOs the
    published claims name (see fetch_snapshot_dataset.py);
  * the split is temporal — train on the earlier proposals, test on the later
    ones — so nothing is predicted with hindsight;
  * R^2 is reported as computed, including when it is negative;
  * the trivial constant baselines are reported alongside everything else,
    because on a distribution this skewed they are the bar to beat.

Usage:
    python -m q_ai_governance.benchmark_snapshot_real --data data/snapshot_dao_dataset.json
"""

import argparse
import json
import statistics as stats

try:
    from .features import build_matrix, FEATURE_NAMES, DAOS
except ImportError:
    from features import build_matrix, FEATURE_NAMES, DAOS

import numpy as np


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae_pp": float(np.mean(np.abs(err))),
        "rmse_pp": float(np.sqrt(np.mean(err ** 2))),
        # No clamping. A model worse than the test-set mean gets a negative R^2,
        # which is information, not something to hide.
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def ridge_fit(X, y, alpha=1.0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    A = Xs.T @ Xs + alpha * np.eye(Xs.shape[1])
    A[-1, -1] -= alpha  # do not penalise the intercept
    w = np.linalg.solve(A, Xs.T @ y)
    return {"w": w, "mu": mu, "sd": sd}


def ridge_predict(model, X):
    X = np.asarray(X, dtype=float)
    Xs = np.hstack([(X - model["mu"]) / model["sd"], np.ones((len(X), 1))])
    return np.clip(Xs @ model["w"], 0.0, 100.0)


def run(data_path, test_frac=0.30, alpha=1.0):
    payload = json.load(open(data_path))
    proposals = payload["proposals"]
    X, y, meta = build_matrix(proposals)          # already sorted by created

    n = len(y)
    cut = int(n * (1 - test_frac))
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
    te_meta = meta[cut:]

    results = {}

    # --- Trivial constants -------------------------------------------------
    results["constant_train_mean"] = metrics(yte, [stats.mean(ytr)] * len(yte))
    results["constant_train_median"] = metrics(yte, [stats.median(ytr)] * len(yte))

    # --- Per-DAO historical mean ------------------------------------------
    dao_mean, gmean = {}, stats.mean(ytr)
    for d in DAOS:
        vals = [ytr[i] for i in range(cut) if meta[i]["dao"] == d]
        dao_mean[d] = stats.mean(vals) if vals else gmean
    results["per_dao_mean"] = metrics(yte, [dao_mean[m["dao"]] for m in te_meta])

    # --- Ridge on pre-vote features ---------------------------------------
    model = ridge_fit(Xtr, ytr, alpha=alpha)
    ridge_pred = ridge_predict(model, Xte)
    results["ridge_prevote_features"] = metrics(yte, ridge_pred)

    split = {
        "n_total": n,
        "n_train": cut,
        "n_test": len(yte),
        "train_mean_yes_pct": stats.mean(ytr),
        "train_median_yes_pct": stats.median(ytr),
        "test_mean_yes_pct": stats.mean(yte),
        "test_median_yes_pct": stats.median(yte),
        "test_sd_yes_pct": stats.pstdev(yte),
    }
    coefs = dict(zip(FEATURE_NAMES, [round(float(v), 3) for v in model["w"][:-1]]))
    return {"split": split, "results": results, "ridge_coefficients": coefs,
            "ridge_predictions": [float(v) for v in ridge_pred],
            "y_test": [float(v) for v in yte]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/snapshot_dao_dataset.json")
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--out", default="data/benchmark_classical_results.json")
    args = ap.parse_args()

    out = run(args.data, args.test_frac)
    s = out["split"]
    print("=" * 66)
    print("  REAL SNAPSHOT BENCHMARK — temporal split, no hindsight")
    print("=" * 66)
    print(f"n = {s['n_total']}  (train {s['n_train']} earlier / test {s['n_test']} later)")
    print(f"test YES-share: mean {s['test_mean_yes_pct']:.2f}%  "
          f"median {s['test_median_yes_pct']:.2f}%  sd {s['test_sd_yes_pct']:.2f}pp\n")
    print(f"{'model':<28}{'MAE (pp)':>10}{'RMSE (pp)':>11}{'R^2':>9}")
    print("-" * 66)
    for name, m in out["results"].items():
        print(f"{name:<28}{m['mae_pp']:>10.2f}{m['rmse_pp']:>11.2f}{m['r2']:>9.3f}")
    print("-" * 66)
    print("\ntop ridge coefficients (standardised):")
    top = sorted(out["ridge_coefficients"].items(), key=lambda kv: -abs(kv[1]))[:6]
    for k, v in top:
        print(f"   {k:<20}{v:>8.3f}")
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
