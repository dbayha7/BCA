# Cluster OOD collection is running

Snapshot: September 29, 2026, 06:57 UTC. This is execution and engineering
evidence, not a completed OOD result. Job 27067675 runs the accepted ReBRAC
Hopper seed 202609171 host/standard-BCA pair on an Orion CPU allocation. The
existing cluster GPU training job 27045057 and local TD3+BC OOD worker continue.

Checkpoint/query job 27067658 completed with actual exit 0. Both actor errors
were exactly zero; target arithmetic and frozen BCA width/dose checks passed.
The saved calibrator normalization was restored from the trained checkpoint,
not recomputed on CPU. This preserves the accepted GPU preparation hashes.

The real simulator connection and an independent saved-array engineering
review passed before state collection: exact actor/first-transition/full-state
parity and constructor reward error 7.746e-11, below the unchanged 1e-7 limit.
At the captured status, collection had 521 explicit transitions. The actual
worker command and start identity match, all staged inputs are unchanged, and
the job's final exit is still pending. These are dated counters, not live ones.

The running staged `cluster_run.py` is the byte-identical copy of this package's
`cluster_run_v2.py`; `cluster_run.py` here preserves the failed original attempt.
The only worker change is the filesystem-specific lock import. Existing
candidate, collection, outcome and simulator modules are reused without edits;
their exact source hashes and paths are in the dispatch pins. No weights,
scientific NPZ arrays, live SQLite journals or credential files are published.

## What this experiment will answer

- Starting from the same saved state, compare the consequences of increasingly
  unfamiliar actions under the host and BCA continuation policies.
- Check whether larger BCA widths identify actions that actually cause more
  harm, and whether BCA recovers better after the same unfamiliar first action.
- Separate useful warning signals, better action choice and better later
  recovery. Good performance on one does not establish the others. A higher
  whole-policy training score alone does not answer these action-level questions.

The frozen design keeps both Bayesian and conformal components, standard BCA
without fitting IW, original coverage, 1M checkpoints, 64 paired resets per
collector, capture steps 100/300, and 250-step outcomes. All candidate actions,
scores and continuation keys are committed before outcomes. Missing captures
and candidate quotas remain visible. Independent outcome verification and
multiple training seeds are still required before a replicated benefit claim.

See [resource ownership](RESOURCE_PARTITION.md) for disjoint local/cluster
budgets and the explicit limitation that remote ancestry uses a pinned local
attestation. See [preserved failure and correction](CLUSTER_EXECUTION_CORRECTION.md)
for the failed startup and successful cross-process lock check. No outcome was
replayed in that correction. The twenty-pair tranche is not fully queued.
