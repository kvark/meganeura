# Development experiment evidence

Read each cohort's README and small `summary.json` first. Raw records and
dependency locks are retained as deterministic gzip archives (`gzip -n -9`),
not multi-megabyte pretty-printed JSON diffs. Compression preserves the exact
original bytes, including float spellings, rejected attempts and source hashes.
These experiments are separate from the frozen paper evidence.

From this directory, verify the archives without extracting them:

```sh
sha256sum --check SHA256SUMS
```

`RAW-SHA256SUMS` records the decompressed hashes. Older per-cohort checksum
files also refer to original, uncompressed filenames. For example:

```sh
gzip -dc split-k-sequence-2026-09-06/run-01.json.gz | sha256sum
gzip -dc split-k-sequence-2026-09-06/run-01.json.gz | jq '.cases[].report'
```

To recreate a lockfile next to its archive, `gzip -dk` preserves the archive;
use that extracted file as the root `Cargo.lock` when rebuilding the stated
measured revision. Replay tests read compressed records directly using the
existing `flate2` dependency, with all previous numerical/timing checks intact.

Runners may still emit readable JSON locally. Raw JSON and extracted locks in
this directory are ignored; only small summaries stay as JSON in Git. Compress
retained raw output and refresh both manifests before committing. CI verifies
archive hashes and rejects tracked uncompressed records/locks.

The development PR was squashed onto `230cab0` during cleanup. That does not
make the squashed source the measured source: original revisions and published
evidence tags remain the provenance of their records. The old branch was also
saved in a verified local Git bundle before rewriting. No timings, oracle gates,
or measurement tags were relabeled by compression.
