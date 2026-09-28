# OOD desktop-restart recovery plan

**Goal:** carry out David's explicit restart instruction while retaining every
completed action outcome and the interrupted attempt.

**Architecture:** add a separately tested recovery entry point. Keep the original
worker, simulator, actor operations, checkpoints, state/candidate/key banks and
all numerical gates byte-identical. Use a new exclusive output directory and
the original cumulative resource ledger and shared local lock. No subagents.

**Tech stack:** existing Python/JAX/MuJoCo, immutable JSON and SQLite evidence.

- [x] Verify new boot, old process absence and free shared lock. Preserve unknown
  original worker/supervisor exits as unknown; do not fabricate a process exit.
- [x] Hash-pin the old attempt and validate its durable panel/slot prefix. Initial
  inspection finds 143 panels plus eight completed actions in panel 143: 1,438
  completed actions. Slot 8 has 236 reserved outcome calls and one repeat call,
  with no completed-action receipt. Confirm against immutable archive/ledger.
- [x] Reuse all complete panels and complete actions, independent of their scores.
  Re-execute only that one incomplete action from its original full snapshot and
  precommitted action, checking each surviving transition against the old trace.
  Keep the old partial trace excluded from primary estimates and visible as
  interruption overhead. Then execute only unstarted actions/panels.
- [x] Keep original ledger entries/caps unchanged. Charge the re-executed prefix
  (at most 236 outcome calls plus one repeat) and one new constructor to the
  existing engineering allowance, never refund old reservations. New remaining
  science consumes the unused original outcome/repeat allowances. The total new
  physical ceiling is 924,183 transitions / 3,696,732 physics steps if the
  inspected prefix is accepted. No training, new seeds or coverage experiment.
- [x] Test complete-action reuse, interrupted-action prefix checking, exact
  remaining arithmetic, refusal of altered evidence, failed attempts, duplicate
  dispatch and incorrect budget routing. Compare the recovery panel driver with
  the untouched original driver on the exact toy simulator before real dispatch.
- [x] Save a separate acceptance binding, then launch one CPU worker with original
  eager operations and local lock. Save fresh real worker/supervisor exits and
  boot identity. Stop on any new failure; no automatic retry loop.
- [x] Verify successful startup and increasing saved progress, then update the
  existing monitor and readable status. Preserve 143 original panel receipts and
  all action receipts. Completion and scientific acceptance remain separate.

The local CQL validation failure preceded the reboot and is not a desktop
interruption. This recovery does not restart that failed training queue or the
old 810-group campaign. The cluster is independent; do not submit a duplicate.
