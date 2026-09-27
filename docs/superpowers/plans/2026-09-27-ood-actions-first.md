# OOD action execution implementation plan

> Execute locally without subagents, as David requested. Real OOD testing is already
> authorized. This is an execution supplement to the frozen September 24 protocol,
> not a new recipe or tuning study. Preserve every failed attempt.

**Goal:** move the next work from performance reporting to measured consequences of
OOD candidate actions, beginning with compatible accepted 1M checkpoint pairs.

**Architecture:** reuse the existing read-only adapters, simulator, oracle and
analysis. Add one small execution layer in `experiments/ood/` with exclusive stage
outputs and an append-only resource ledger. Bind the existing training audits
without rerunning closed science. Collection, candidate preparation and outcomes
are separate frozen stages so outcomes cannot influence action selection.

**Tech Stack:** existing Python/JAX/Flax, MuJoCo/D4RL, NumPy and Matplotlib.

## Priority and preserved contracts

Performance logs are supporting context. Do not wait for the 280-run training grid
before testing an accepted pair. Use queue order/readiness rather than performance
selection, retain missing cells and all five seeds. Standard no-IW first, later IW
selection on independently declared development evidence. IQL must retain its
shared-Q/V pair when its adapters are added. Unifloral remains an exact, separately
labeled baseline family with native width N/A.

Do not edit any of the 108 frozen training source/configuration files, including
the root README, existing scientific design and `configs/ood.yaml`. Training
checkpoints at 1M retain 1M critic/500k actor counts for TD3+BC and ReBRAC. Keep the
old 810-group campaign, native inventory and failed behavioral recoveries intact.

The existing runner has no finish-current hold switch. SIGTERM propagates to its
active worker. Do not implement a queue hold by killing that worker, editing frozen
files, or launching a second controller. Resolve any requested hold through a
separately documented execution-only mechanism after checking process ownership.

## 1. Bind the first accepted pair

Read `docs/validation/ood-actions-first-readiness.json`; it names exact checkpoints
and accepted audit hashes, but correctly leaves collection readiness false.

- [x] Reuse accepted host/BCA audits; bind the first TD3+BC Hopper pair and its
  final checkpoint identities without reopening weights or replaying training.
- [x] Use `experiments/ood/standard_receipt.py` on explicit run/attempt paths to
  bind the actual worker exits. Read the established source/data acceptance rather
  than repeat closed audits. Preserve separate training-acceptance and
  trained-adapter-acceptance flags.
- [x] Create a new exclusive `runs/ood/<attempt>/` with the frozen full declaration
  and a per-pair execution binding. No directory-search-based checkpoint selection.
- [x] Save direct original-host forward/target/width references, then restored
  adapter outputs, devices and error arrays before the unchanged absolute 1e-6
  action gate. Failure stops the attempt. No backend sweep or tolerance change.
- [x] Compare actual prepared arrays, reserved IDs and normalization against the
  accepted evidence. Freeze all execution source/runtime/checkpoint identities.

## 2. Implement a bounded execution layer

Create `experiments/ood/collect.py` and `experiments/ood/test_collect.py`; reuse
`adapters.py`, `simulator.py`, `protocol.py` and `oracle.py`. Keep learner files
unchanged. Add functions for declaration/preparation, state collection, candidate
precommit and outcomes; each stage requires accepted predecessor hashes.

- [x] First add refusal tests for an unaccepted pair, stale checkpoint/source hash,
  changed candidate bank, reused attempt, incomplete state and changed RNG keys.
  Run `python -m unittest experiments.ood.test_collect`; the new tests must fail
  for the missing implementation before adding it.
- [ ] Implement exclusive outputs and a persistent ledger charged before simulator
  calls, including constructor steps. Count environment and physics steps
  separately; distinguish engineering, state collection and scientific outcomes.
- [ ] Implement the declared 64 paired-reset episodes per collector and captures
  at 0/100. Store every required restore field, wrapper/time-limit/RNG state and
  missing capture. Shared reset states retain dependence and verified reuse.
- [ ] Implement the training-complement support bank and its episode split exactly
  as declared. Require episode IDs for the threshold; raw distance alone does not
  authorize thresholded OOD labels when IDs are unavailable.
- [ ] Generate all ten candidate slots per state from saved seeds. Save proposed
  and applied coordinates, clipping, duplicate aliases, BCA width/dose/radii and
  support/constant/random scores. Freeze every per-step continuation key before
  outcomes. Neither width nor observed harm may guide candidate selection.
- [ ] Execute each unique applied action from the same restored state under both
  frozen continuations, maximum 250 total transitions. Save actual applied vectors,
  dtype and simulator controls; independently reconstruct raw reward at absolute
  1e-7. Stop at original natural termination/time limit, with no learned tail.
- [x] Test the full pipeline with the exact deterministic oracle, including
  unfamiliar helpful actions, familiar harmful actions, ties, missing captures,
  early termination, duplicate aliases and interrupted attempts. Compare every
  saved result with hand-enumerated expectations. This is not a BCA result.

## 3. Gate the real checkpoint/simulator connection

September 27, 22:23 UTC: the first-pair CPU engineering connection passed at nine
real transitions (one constructor plus eight explicit), 36 physics steps, actual
worker/supervisor exits 0. Full reset/mid-state repeats, controls and raw-reward
gates pass; cumulative history is bound in the SQLite resource ledger. See
`docs/validation/ood-real-state-connection.json`. Reuse this closed gate.
Production collection, native-bound/candidate/key/lock artifacts still require
implementation and acceptance. No scientific acceptance follows from this gate.
Fourteen new tests plus prior suites: 73 pass. Local training is now stopped on
CQL's evaluation-schema validation failure; preserve it and do not silently retry.

- [ ] Declare a separate engineering attempt using disjoint engineering seeds and
  the existing maximum 10,000-transition-per-cell budget; debit all previous/new
  engineering steps according to their declarations. Test only what the newly
  connected execution path requires. Keep engineering data out of scientific
  estimates. Do not repeat already closed standalone smoke tests routinely.
- [ ] Check complete state round-trips and repeated first transitions against the
  actual restore function. Save diagnostic arrays before gates and record actual
  process exits. Preserve every failure; no automatic retry or synthetic repair.
- [ ] Run `python -m unittest experiments.ood.test_collect
  experiments.ood.test_protocol experiments.ood.test_standard_receipt
  experiments.ood.test_analysis` after execution-layer changes. Run the real
  adapter tests only inside the declared engineering budget when changes justify
  them. Confirm all 108 training source hashes still match the frozen manifest.
- [ ] Declare a per-pair behavioral execution subset with exact sources, banks,
  seeds, maximum counts, ledger and lock. Keep fresh residual coverage separately
  pending. The original protocol's global ready flag must not be changed to true
  to bypass unavailable ReBRAC coverage. If interface separation is needed, use a
  versioned execution binding and refusal tests, preserving the old declaration.

## 4. Collect and report action consequences

- [ ] Dispatch once only after the trained-adapter, state-schema, arithmetic and
  pre-outcome bindings pass. Honor the shared local GPU lock and total two-GPU
  cap; do not compete with an active training worker or spawn duplicate controllers.
- [ ] Check actual exit and every expected/missing/failed row. Accept no silent
  replacement and no best seed. Reuse completed outcomes by verified identity.
- [ ] Feed accepted outcomes to `analyze.py` and `report.py`. First report novelty
  versus width, actual harm versus width, within-state width versus support AUROC,
  and paired first-action/continuation effects. Include support/harm quadrants,
  duplicate/tie counts, valid states, missing strata and both continuation policies.
- [ ] Show first-seed results as exploratory; only attach five-seed uncertainty
  after all declared pairs are accepted. Keep pooled AUROC secondary and residual
  coverage distinct. Link existing performance plots rather than recreate them.
- [ ] Verify plotted arrays, inspect every exported scientific figure, then commit
  and push only owned code/readouts, excluding weights. Preserve all failures,
  original frozen settings and main thesis PDF. Expand hosts/datasets only after
  their own checkpoint and simulator contracts pass.

## Completion criteria

September 27 milestones: the first accepted TD3 Hopper pair passes the 64-row
CPU saved-action/target/width gate and the separate nine-transition real full-state
engineering connection. Seventy-three regression tests pass. The persistent ledger
includes prior and new engineering/constructor reservations. Production state
collection, native-bound artifacts, candidate/support/JAX-key banks, efficient
transition storage, lock binding and outcome execution remain incomplete. The
original runtime-import exit1 and separate library-path correction/exit0 remain
preserved in `docs/OOD_ACTION_COLLECTION_STATUS.md`. No scientific action outcomes
were collected. The local CQL queue is stopped with actual exit1; do not retry it
or edit frozen source. The cluster lane continues independently.

This task is complete only when the declared action outcomes and calculations are
verified and readable, with complete/missing/failed status and actual exits. A
training completion, accepted scale fit, positive policy score or this plan is
not evidence that BCA ranks OOD action harm correctly.
