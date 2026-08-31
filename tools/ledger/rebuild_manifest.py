#!/usr/bin/env python3
"""
rebuild_manifest.py — (re)generate tools/ledger/manifest.json.

This is the ONLY way manifest.json's sha256 fields should change. Editing them
by hand to make a failing check_ledger.py pass defeats the point of the
ledger; this script exists so there is never a reason to.

    python3 tools/ledger/rebuild_manifest.py
    git diff tools/ledger/manifest.json

Only the hash(es) you meant to change should move.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "manifest.json"

SCHEMA_NOTE = (
    "Every number quoted in this repository's README files must trace to a "
    "command that regenerates it. check_ledger.py enforces two things: (1) each "
    "entry's command, run fresh, byte-matches the committed output file, and "
    "(2) the doc_claims listed for every entry actually appear verbatim in the "
    "named documentation files."
)

ENTRIES = [
    {
        "id": "snapshot-classical-benchmark",
        "description": "Classical baselines (constant/ridge) predicting DAO proposal YES share.",
        "reproducible": True,
        "command": ["python3", "-m", "dao_governance.benchmark_snapshot_real",
                    "--data", "data/snapshot_dao_dataset.json", "--out", "{tmp}"],
        "committed_output": "data/benchmark_classical_results.json",
        "doc_claims": [
            {"file": "README.md", "must_contain": ["(10.44pp MAE)"]},
            {"file": "dao_governance/README.md", "must_contain": [
                "| **constant (train median)** | **10.44** | 26.73 | −0.125 |",
                "| ridge on pre-vote features | 11.20 | 25.04 | 0.013 |"]},
        ],
    },
    {
        "id": "contestedness-benchmark",
        "description": "Contestedness classification, including the within-DAO Simpson's-paradox control.",
        "reproducible": True,
        "command": ["python3", "-m", "dao_governance.benchmark_contestedness",
                    "--data", "data/snapshot_dao_dataset.json", "--out", "{tmp}"],
        "committed_output": "data/benchmark_contestedness_results.json",
        "doc_claims": [
            {"file": "README.md", "must_contain": ["AUC 0.660, 95%"]},
            {"file": "dao_governance/README.md", "must_contain": [
                "| **all features** | **0.660** | [0.555, 0.763] | 0.0010 |",
                "**Median within-DAO AUC: 0.416**"]},
        ],
    },
    {
        "id": "ewl-equilibrium",
        "description": "EWL restricted/full-SU(2) equilibrium search and the derived entanglement threshold.",
        "reproducible": True,
        "command": ["python3", "-m", "quantum_games.ewl_equilibrium", "--out", "{tmp}"],
        "committed_output": "data/ewl_equilibrium_results.json",
        "doc_claims": [
            {"file": "quantum_games/README.md", "must_contain": [
                "γ_c = arccos(√(3/5)) ≈ 0.684719 rad ≈ 39.23°"]},
            {"file": "README.md", "must_contain": ["γ_c ≈ 0.6847 rad"]},
        ],
    },
    {
        "id": "ewl-mixed-equilibrium",
        "description": "Closed-form Haar-uniform equilibrium value on the full SU(2) game.",
        "reproducible": True,
        "command": ["python3", "-m", "quantum_games.ewl_mixed_equilibrium", "--out", "{tmp}"],
        "committed_output": "data/ewl_mixed_equilibrium_results.json",
        "doc_claims": [
            {"file": "quantum_games/README.md", "must_contain": [
                "| **full SU(2) Haar equilibrium** | **2.25** |",
                "**62.5%**"]},
            {"file": "README.md", "must_contain": ["worth 62.5% of"]},
        ],
    },
    {
        "id": "snapshot-dataset",
        "description": "The real Snapshot DAO dataset itself.",
        "reproducible": False,
        "note": ("Fetched live from the Snapshot GraphQL hub, which requires network "
                 "access and returns whatever the hub currently holds — not expected "
                 "to be byte-identical across runs. This entry pins the hash of the "
                 "committed snapshot only, so a hand-edit of the file is still "
                 "detectable."),
        "committed_output": "data/snapshot_dao_dataset.json",
        "doc_claims": [
            {"file": "README.md", "must_contain": [
                "6,242,940 vote records"]},
            {"file": "dao_governance/README.md", "must_contain": [
                "| Kept (settled tally, unambiguous binary ballot) | **905** |",
                "| Vote records across kept proposals | **6,242,940** |"]},
        ],
    },
]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    entries = []
    for spec in ENTRIES:
        committed = ROOT / spec["committed_output"]
        if not committed.exists():
            raise SystemExit(f"missing committed output for {spec['id']}: {committed}")
        entry = {
            "id": spec["id"],
            "description": spec["description"],
            "reproducible": spec["reproducible"],
            "committed_output": spec["committed_output"],
            "sha256": sha256_file(committed),
            "doc_claims": spec.get("doc_claims", []),
        }
        if spec["reproducible"]:
            entry["command"] = spec["command"]
        else:
            entry["note"] = spec["note"]
        entries.append(entry)

    manifest = {"$schema_note": SCHEMA_NOTE, "entries": entries}
    OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT} with {len(entries)} entries")


if __name__ == "__main__":
    main()
