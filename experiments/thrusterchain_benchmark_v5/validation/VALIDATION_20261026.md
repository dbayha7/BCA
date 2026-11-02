# V5 bounded validation — 2026-10-26

## Source and scope

The validated v5 code/test/contract manifest has source SHA-256
`f5f68c60565f471321f303a6f765cb93948e5d231bc649f81c2d636e3c9efcee`
over 71 files, recorded in `../CURRENT_SOURCE_MANIFEST.json`. Its digest was
recomputed after validation and matched. All 52 files in the original v4 source
manifest matched their recorded hashes; this is a source identity check, not a
validation of every historical result or untracked asset.

## Executed checks

- `validate_v5.py`: 223 tests ran, 63 explicitly skipped; the other 160 passed.
  The skips are six v4 historical-evidence classes that need unavailable saved
  results and their original v4 loader. The unfiltered v4 suite remains
  unverified; skipping is not a pass of those historical assertions.
- Combined host-integration suite: 21 tests passed on WSL Ubuntu-22.04, CPU,
  Python 3.10.12, JAX 0.6.2, Flax 0.10.7, Optax 0.2.8, NumPy 2.2.6. These
  tests use the repository-native TD3+BC, IQL, CQL and ReBRAC hosts and actual
  v5 task batches. They cover numerical losses/targets, termination and timeout
  semantics, optimizer counters, seed mapping, checkpoint resume and frozen
  evaluation. This does not validate the pinned unified UniFloral implementation
  or establish algorithm performance, GPU portability or research-scale speed.
- A fresh ten-episode finite-task smoke dataset was generated and validated.
  Its content digest is
  `75a7a6126a3fd75a8d7e61bb6ce5ac78b1c95e441e534f5a0b3ce56c8b706670`.
  The predeclared affine-SGD protocol smoke in `SMOKE_PLAN.md` passed. The 8+3
  resumed and 11-step uninterrupted checkpoints were byte-identical, and the
  dataset remained unchanged. Strict and zero-online modes recorded zero
  adaptation steps; the adaptation mode recorded 48 interactions and 24 updates
  across four families. Each of the three modes recorded eight complete frozen
  evaluation episodes, totaling 800 environment steps per mode, with unchanged
  frozen policy identity. See `bounded_smoke_v5/run/completion.json` and the
  individual mode receipts. These are plumbing checks, not learned neural
  baseline comparisons.

## Remaining limits

The legacy artifact tests must be run against their original source and exact
historical assets. Unified UniFloral has not been integrated: the inspected
runtimes lack a jointly compatible dependency stack, and its native sampling
and two-critic-steps-per-actor cadence require an explicit runner design. ReBRAC explicitly
refuses bootstrapped rows without a recorded successor action, including the
current continuing-mode dataset boundary. Fixed learner configurations and
eager update throughput need separate research-protocol and runtime evaluation.
The dataset's random behavior recorder and deterministic start-law shifts do not
cover every offline RL setting. Seed separation cannot establish that final
outcomes or task families were unobserved during development. No full experiment,
sweep, or scientific performance comparison was run in this validation.
