# Popcorn

**Custom kernel path teardown — one hot path, three implementations, honest before/after.**

The hot path: **numeric-aware text normalization** — the step every inference pipeline runs on every prompt before tokenization (strip noise, fold numbers/versions/emails into placeholder tokens). On a busy gateway this runs thousands of times per second.

Part of the [TechGuardCoders portfolio](https://github.com/orgs/TechGuardCoders/repositories): Cost Peep → Batcher → Spinal → Ball Knowledge → Bastion → Furnace → Truffle → **Popcorn** → …

## Result (live, this machine, 400-text mixed corpus × 30 rounds)

| Implementation | mean | p50 | p95 | Speedup |
|---|---|---|---|---|
| Python (regex chain) | 5.409 ms | 5.102 ms | 7.298 ms | 1.00× |
| **C (hand-rolled CPython extension)** | **0.382 ms** | **0.370 ms** | **0.529 ms** | **14.16×** |

**Correctness gate: 400/400 byte-identical** — timings only count after the fast implementation proves it produces exactly the baseline's output. A fast wrong answer is a failure.

## The gate caught three real bugs

This is the honest part. The first C version was fast and WRONG in three separate ways, each found only by the byte-identical gate:

1. **Branch-order bug** — digits entered the alphanumeric-run branch before the digit branch; `42` was never folded. Regex pipelines have an ORDER (EMAIL → VERSION → NUM); the C port must branch in that same priority, not in whatever order is natural to write.
2. **Marker collision** — the C version marks folded spans with the letters `E`/`V`/`N`… which are also ordinary text bytes. Every `Error` in the corpus became `<EMAIL> rror`. Fix: markers moved to non-printable bytes (`\x01`–`\x03`) that pass-1 filtering guarantees can't appear in text.
3. **Semantic mismatch on digit runs** — Python's `(?<![a-zA-Z])\d+(\.\d+)?` folds every digit in a run *except one preceded by a letter* (so `x42` → `x<NUM>`: the regex engine retries at the second digit). Naive "whole run or nothing" logic silently diverged. Byte-exactness forced replicating the regex engine's retry semantics.

**The lesson is the project:** frameworks (regex in C already!) are fast *enough* until they aren't, and hand-rolling the last 14× requires inheriting the framework's *semantics*, not just its goal.

## Run it

```bash
uv venv --python 3.11 .venv
uv pip install pytest numpy
# build the extension (needs gcc — MSYS2 on Windows)
./scripts/build_c.py          # or the gcc one-liner in CI
python popcorn.py             # gate + benchmark
python -m pytest tests/ -q
```

## Tests

6 offline tests: baseline semantics (number/version/email folding, whitespace/noise), corpus determinism, and gate exactness (byte-identical or fail).

## What's next

This feeds **Benchwarmer (#15)**: the teardown report includes "when frameworks aren't enough" — measured, with the gate-caught bugs as the honesty exhibit.
