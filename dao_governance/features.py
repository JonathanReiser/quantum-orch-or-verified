"""
features.py — pre-vote features for the real Snapshot dataset.

Every feature here must be knowable BEFORE a proposal's tally is settled.
Post-hoc quantities (final voter counts, realised voting power) are excluded
on purpose: including them would turn a forecast into a description and make
any reported error meaningless.
"""

import math

DAOS = ["Uniswap", "Arbitrum", "Optimism", "Gitcoin", "Aave"]

FEATURE_NAMES = (
    [f"dao_{d.lower()}" for d in DAOS]
    + ["log_body_len", "title_len", "duration_days", "n_choices",
       "has_quorum", "has_abstain", "prior_dao_yes", "prior_dao_n"]
)


def build_matrix(proposals):
    """Return (X, y) with a strictly past-only DAO prior.

    `prior_dao_yes` is an expanding mean over that DAO's *earlier* proposals
    only, so row i never sees its own outcome or any later one.
    """
    rows = sorted(proposals, key=lambda p: p["created"])
    seen = {d: [] for d in DAOS}
    X, y, meta = [], [], []

    for p in rows:
        dao = p["dao"]
        past = seen[dao]
        prior = sum(past) / len(past) if past else 84.0  # neutral until a DAO has history
        dur_days = max(0.0, (p["end"] - p["start"]) / 86400.0)

        feats = [1.0 if dao == d else 0.0 for d in DAOS] + [
            math.log1p(p["body_len"]),
            len(p["title"]) / 100.0,
            dur_days,
            float(len(p["choices"])),
            1.0 if p["quorum"] else 0.0,
            1.0 if len(p["choices"]) > 2 else 0.0,
            prior / 100.0,
            math.log1p(len(past)),
        ]
        X.append(feats)
        y.append(p["yes_pct"])
        meta.append(p)
        seen[dao].append(p["yes_pct"])

    return X, y, meta
