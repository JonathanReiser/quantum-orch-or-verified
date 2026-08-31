#!/usr/bin/env python3
"""
check_ledger.py — verify every published number in this repository still
traces to a command that produces it.

This exists because of what CORRECTIONS.md documents: seven headline claims
that were never regenerable from anything, including an R^2 clamped to a
ceiling and a "trained" model whose fit accepted zero moves. A results ledger
cannot prevent someone from writing a number without running anything — but it
can make that failure loud instead of silent, by refusing to pass CI when a
documented claim and the artifact behind it disagree.

Two checks per manifest entry:

  1. REPRODUCIBILITY. For entries marked reproducible, the listed command is
     run fresh and its output is byte-compared to the committed file. A
     mismatch means either the code changed behaviour without the committed
     output being updated, or the committed output was hand-edited. For
     entries marked NOT reproducible (network-dependent fetches, or fits this
     project already showed are noise-dominated — see CORRECTIONS.md section
     7), only the committed file's hash is checked against the pinned value,
     which proves it was not silently altered without saying so is a weaker
     guarantee than reproducibility, on purpose.

  2. DOCUMENTATION. Every doc_claims string is checked for literal presence in
     its named file. This catches the other half of the original failure mode:
     a table typed by hand that quietly drifted from the benchmark it claimed
     to cite (see UNISWAP_GOVERNANCE_PROPOSAL.md's history in CORRECTIONS.md).

Usage:
    python3 tools/ledger/check_ledger.py
    python3 tools/ledger/check_ledger.py --entry ewl-equilibrium
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).resolve().parent / "manifest.json"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_reproducibility(entry):
    committed = ROOT / entry["committed_output"]
    if not committed.exists():
        return False, f"committed output missing: {entry['committed_output']}"

    pinned = entry["sha256"]
    committed_hash = sha256_file(committed)
    if committed_hash != pinned:
        return False, (
            f"committed file no longer matches the manifest's pinned hash.\n"
            f"      pinned:    {pinned}\n"
            f"      on disk:   {committed_hash}\n"
            f"      The file was edited without updating the ledger. Regenerate the\n"
            f"      manifest entry rather than editing this hash by hand."
        )

    if not entry.get("reproducible", False):
        return True, "pinned-only (not regenerated in CI) — see 'note'"

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "out.json"
        cmd = [str(part).format(tmp=str(out_path)) for part in entry["command"]]
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            return False, (
                f"command failed (exit {result.returncode}): {' '.join(cmd)}\n"
                f"      stderr (last 400 chars): ...{result.stderr[-400:]}"
            )
        if not out_path.exists():
            return False, f"command produced no output at expected path: {out_path}"

        fresh_hash = sha256_file(out_path)
        if fresh_hash != pinned:
            fresh = json.loads(out_path.read_text())
            committed_data = json.loads(committed.read_text())
            diff_keys = _top_level_diff(fresh, committed_data)
            return False, (
                f"regenerated output does NOT match the committed file.\n"
                f"      command:   {' '.join(cmd)}\n"
                f"      expected:  {pinned}\n"
                f"      got:       {fresh_hash}\n"
                f"      differing top-level keys: {diff_keys or '(structure differs)'}\n"
                f"      Either the code's behaviour changed and the committed file is\n"
                f"      stale, or the committed file was edited by hand. If the change\n"
                f"      is intentional, regenerate {entry['committed_output']} and update\n"
                f"      this manifest entry's sha256 -- do not just widen a tolerance."
            )
    return True, "regenerated output byte-matches the committed file"


def _top_level_diff(a, b):
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return []
    keys = set(a) | set(b)
    return sorted(k for k in keys if a.get(k) != b.get(k))


def check_doc_claims(entry):
    problems = []
    for claim in entry.get("doc_claims", []):
        doc_path = ROOT / claim["file"]
        if not doc_path.exists():
            problems.append(f"{claim['file']}: file does not exist")
            continue
        text = doc_path.read_text(encoding="utf-8")
        for needle in claim["must_contain"]:
            if needle not in text:
                problems.append(f"{claim['file']}: missing literal text: {needle!r}")
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", help="check only this entry id")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    entries = manifest["entries"]
    if args.entry:
        entries = [e for e in entries if e["id"] == args.entry]
        if not entries:
            print(f"no such entry: {args.entry}", file=sys.stderr)
            return 2

    failures = 0
    print(f"Results ledger — {len(entries)} entr{'y' if len(entries)==1 else 'ies'}\n")
    for entry in entries:
        ok, message = check_reproducibility(entry)
        status = "OK  " if ok else "FAIL"
        print(f"[{status}] {entry['id']}")
        print(f"       {message}")
        if not ok:
            failures += 1

        doc_problems = check_doc_claims(entry)
        if doc_problems:
            failures += 1
            print(f"[FAIL] {entry['id']} (documentation)")
            for p in doc_problems:
                print(f"       {p}")
        elif entry.get("doc_claims"):
            print(f"[OK  ] {entry['id']} (documentation) — "
                  f"{sum(len(c['must_contain']) for c in entry['doc_claims'])} claim(s) verified verbatim")
        print()

    if failures:
        print(f"{failures} check(s) failed.")
        return 1
    print("All entries reproducible (or honestly pinned) and all documented claims verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
