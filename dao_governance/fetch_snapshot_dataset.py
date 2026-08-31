"""
fetch_snapshot_dataset.py — build a real Snapshot DAO voting dataset.

Replaces the five hardcoded proposals in benchmark_real_dao_data.py with every
closed, tallied proposal from the five DAOs the project's published claims name
(Uniswap, Arbitrum, Optimism, Gitcoin, Aave).

For each proposal we keep the final on-chain-anchored tally: the choice labels,
their scores, total voting power, and the number of distinct voters. A proposal
is kept only if a YES-like and a NO-like choice can be identified unambiguously,
because the whole benchmark predicts a YES share.

Usage:
    python -m q_ai_governance.fetch_snapshot_dataset --out data/snapshot_dao_dataset.json
"""

import argparse
import json
import os
import re
import time

try:
    from .snapshot_api import gql
except ImportError:  # running as a plain script
    from snapshot_api import gql

SPACES = {
    "uniswapgovernance.eth": "Uniswap",
    "arbitrumfoundation.eth": "Arbitrum",
    "opcollective.eth": "Optimism",
    "gitcoindao.eth": "Gitcoin",
    "aavedao.eth": "Aave",
}

PAGE = 1000

PROPOSAL_QUERY = """
query($space: String!, $first: Int!, $skip: Int!) {
  proposals(
    first: $first, skip: $skip,
    where: {space: $space, state: "closed"},
    orderBy: "created", orderDirection: desc
  ) {
    id title body space { id } type choices scores scores_total scores_state
    votes quorum created start end
  }
}
"""

# Choice-label matching. Deliberately strict: anything we cannot classify with
# confidence is dropped rather than guessed at, since a mislabelled YES share
# silently corrupts every downstream error metric.
YES_PAT = re.compile(r"^\s*(yes|for|approve[d]?|in favou?r|accept|aye|support|agree)\b", re.I)
NO_PAT = re.compile(r"^\s*(no\b|against|reject|nay|disagree|do not|don'?t)", re.I)
ABSTAIN_PAT = re.compile(r"^\s*(abstain|neutral|no opinion)", re.I)


def classify(choices):
    """Return (yes_idx, no_idx) or None when the ballot is not a clean binary."""
    yes = [i for i, c in enumerate(choices) if YES_PAT.match(str(c))]
    no = [i for i, c in enumerate(choices) if NO_PAT.match(str(c))]
    other = [
        i for i, c in enumerate(choices)
        if i not in yes and i not in no and not ABSTAIN_PAT.match(str(c))
    ]
    if len(yes) != 1 or len(no) != 1 or other:
        return None
    return yes[0], no[0]


def fetch_space(space, sleep=0.3):
    """Page through every closed proposal in one space."""
    rows, skip = [], 0
    while True:
        data = gql(PROPOSAL_QUERY, {"space": space, "first": PAGE, "skip": skip})
        batch = data["proposals"]
        rows.extend(batch)
        if len(batch) < PAGE:
            break
        skip += PAGE
        time.sleep(sleep)
    return rows


def build(spaces=None):
    spaces = spaces or SPACES
    kept, dropped = [], {"not_binary": 0, "no_tally": 0, "not_final": 0, "no_votes": 0}
    raw_total = 0

    for space, label in spaces.items():
        proposals = fetch_space(space)
        raw_total += len(proposals)
        print(f"  {label:<10} {space:<26} {len(proposals):>5} closed proposals")

        for p in proposals:
            # scores_state == 'final' means Snapshot has settled the tally.
            if p.get("scores_state") != "final":
                dropped["not_final"] += 1
                continue
            if not p.get("scores") or not p.get("scores_total"):
                dropped["no_tally"] += 1
                continue
            if not p.get("votes"):
                dropped["no_votes"] += 1
                continue
            idx = classify(p["choices"])
            if idx is None:
                dropped["not_binary"] += 1
                continue

            yes_i, no_i = idx
            scores = p["scores"]
            yes_vp, no_vp = float(scores[yes_i]), float(scores[no_i])
            decisive = yes_vp + no_vp
            if decisive <= 0:
                dropped["no_tally"] += 1
                continue

            kept.append({
                "dao": label,
                "space": space,
                "proposal_id": p["id"],
                "title": p["title"],
                "choices": p["choices"],
                "scores": [float(s) for s in scores],
                "yes_choice": p["choices"][yes_i],
                "no_choice": p["choices"][no_i],
                "yes_vp": yes_vp,
                "no_vp": no_vp,
                "scores_total": float(p["scores_total"]),
                # YES share of decisive (non-abstain) voting power — the target.
                "yes_pct": 100.0 * yes_vp / decisive,
                "voter_count": int(p["votes"]),
                "quorum": float(p.get("quorum") or 0.0),
                "created": p["created"],
                "start": p["start"],
                "end": p["end"],
                "body_len": len(p.get("body") or ""),
            })

    return kept, dropped, raw_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/snapshot_dao_dataset.json")
    args = ap.parse_args()

    print("Fetching closed proposals from the Snapshot hub...")
    kept, dropped, raw_total = build()

    total_voters = sum(r["voter_count"] for r in kept)
    payload = {
        "source": "https://hub.snapshot.org/graphql",
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spaces": SPACES,
        "closed_proposals_seen": raw_total,
        "proposals_kept": len(kept),
        "dropped": dropped,
        "total_votes_cast": total_voters,
        "proposals": kept,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nClosed proposals seen : {raw_total}")
    print(f"Kept (clean binary)   : {len(kept)}")
    print(f"Dropped               : {dropped}")
    print(f"Total votes cast      : {total_voters:,}")
    print(f"Written to            : {args.out}")


if __name__ == "__main__":
    main()
