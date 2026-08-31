# Results ledger

`manifest.json` lists every reproducible artifact behind this repository's
published numbers. `check_ledger.py` verifies two things for each entry:

1. **Reproducibility.** For entries marked `reproducible`, the listed command
   is re-run. If its output is byte-identical to the committed file, that's
   the strongest pass. If not, the two are compared structurally with a
   floating-point tolerance (`rtol=1e-6`) before failing outright — computing
   the same deterministic operations on a different BLAS backend (Apple
   Accelerate on a dev machine vs. OpenBLAS on Linux CI, say) can legitimately
   change the last few bits of a float without the result being wrong, and a
   byte-hash can't tell the difference between that and a real regression.
   A genuine mismatch (wrong value, missing key, changed structure) still
   fails outside that tolerance. The committed file's own hash is still
   checked exactly against the manifest's pin — that's a different question
   (has anyone hand-edited this file?) and doesn't need the same tolerance.
2. **Documentation.** Every `doc_claims` string is checked for literal
   presence in the named file. A published number that has drifted from the
   artifact it cites fails the build too.

One entry, `snapshot-dataset`, is marked `reproducible: false` — it's a live
network fetch from the Snapshot GraphQL hub, not expected to be
byte-identical run to run. It's still hash-pinned, so a hand-edit of the
committed copy is caught even though the fetch itself isn't re-run in CI.

## Running it

```bash
python3 tools/ledger/check_ledger.py                          # everything
python3 tools/ledger/check_ledger.py --entry ewl-equilibrium   # one entry
```

Runs in CI on every push and PR (`.github/workflows/tests.yml`, job
`results-ledger`).

## Adding or updating an entry

```bash
python3 tools/ledger/rebuild_manifest.py
git diff tools/ledger/manifest.json   # confirm only the expected hash moved
```

Never hand-edit a `sha256` field to make a failing check pass. If the check is
failing, either the change was unintentional and should be reverted, or it
was intentional and the manifest should be regenerated — never patched
directly.
