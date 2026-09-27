# Action-level OOD execution: real state restoration checked

## September 27, 22:23 UTC: real CPU simulator connection

The first accepted TD3+BC Hopper pair passes a bounded **real full-state engineering check**. For both frozen policies, reset and mid-episode states restore exactly; repeated first transitions reproduce every saved record and subsequent state exactly. Paired resets reproduce the complete state, including wrapper, time-limit and RNG fields. Incoming/applied actions retain float32; simulator controls equal applied actions. The largest independent raw-reward error was **9.6161e-11**, below the unchanged absolute **1e-7** limit.

Worker and supervisor actual exits are **0**. The attempt used **9 environment transitions / 36 physics steps**: one observed constructor step and eight explicit steps, plus four actor-query batches. This CPU engineering is excluded from scientific estimates. The closed 64-row trained query gate was reused. No model update, GPU worker, scientific state bank, action-harm outcome or coverage result was added. All 108 frozen training files still match.

- [Validation and independent saved-record check](validation/ood-real-state-connection.json)
- [Full states, controls, repeats and actual exits](../outputs/ood/td3_bc/hopper/s202609171/real-connection-v1/)

The cumulative resource ledger reserves calls durably before construction or stepping. It preserves all six prior adapter attempts, including failures: **72 prior physical transitions**, charged once globally. The 36 shared Hopper checks consume allowance in both supported Hopper host cells, and likewise for Walker. Shared allowance charges are not extra physical work. After this attempt, global reservations are **81 environment / 324 physics**, and the TD3 Hopper cell has **45 / 180**. Uncertain calls are never refunded. The original declarations and actual exits were checked for accounting; old scientific results were not reopened.

SQLite transactions serialize reservations and enforce per-cell/per-pair/global ceilings. Unique tokens prevent repeated calls; failed simulator instances cannot continue. Full-chain audits run on opening and stage completion. Each reservation uses indexed totals and head checks, avoiding a full history scan per step. The database lives on local WSL storage. Fourteen new mock/resource tests plus the prior 59 pass: **73 tests, actual exit 0**. The initial missing-module failure and a Windows test-fixture cleanup failure remain recorded; the latter was fixed by explicitly closing test SQLite connections. Neither involved a real simulator.

The CPU CUDA-discovery warning and Gym/package warnings remain. The active device was CPU; no backend switch or retry occurred.

Production remains gated: 64 paired episodes per collector and captures at 0/100, an explicit live native-action-bound artifact, support/candidate banks, JAX step keys, efficient transition storage, and execution/lock binding remain unfinished. Saved applied vectors also match the declared unit-bound D4RL transform; production must explicitly save the live bounds. The bounded engineering facade refuses scientific scopes or more than eight explicit steps. The original toy collector still cannot execute real science. All five training seeds remain required; ReBRAC fresh residual coverage remains unavailable.

## Training status discovered at 22:24 UTC

The local CQL Hopper host failed at **21:56 UTC**, after its final 1M evaluation, when the common validator requested `normalized_score` but CQL stores `score` (and `return`, rather than `raw_return`). Worker and controller actual exits are **1**. `result.json` and learner `exit.json` are absent. All three checkpoint files and the original journal remain, but the run is **not accepted**. See the [preserved failure receipt](validation/standard-cql-host-validation-failure.json). No frozen file edit, controller restart or scientific retry was made.

The separate cluster lane had matching live identities, with ReBRAC Walker host at 645k. Three earlier new cluster closures still await independent audits. OOD preparation remains the priority. Any local recovery requires separately checked handling of the schema failure while preserving the original attempt.

## Earlier closed CPU checkpoint-query connection


The accepted standard TD3+BC Hopper host/BCA pair, seed 202609171, passes the
**CPU trained-checkpoint query gate**. Both restored actors exactly reproduce
saved direct reference actions on 64 fixed observations per method (maximum
absolute error 0, unchanged limit 1e-6). Native targets agree exactly, including
independent target arithmetic. BCA width and dose arrays also agree exactly;
host width remains N/A.

This earlier gate validates the measurement tool, **not OOD action harm**. It added
no simulator step, model update or GPU worker. The later nine-step engineering
connection above is separate. Scientific state collection and action outcomes
have not started.

- [Validation receipt](validation/ood-trained-td3-hopper-connection.json)
- [Saved arrays, runtime and actual exits](../outputs/ood/td3_bc/hopper/s202609171/trained-connection-v2/)
- [Original training identities](validation/ood-actions-first-readiness.json)

## Accepted trained-checkpoint connection

The binding joins the closed training audits to separate actual worker exit
receipts and exact final checkpoint hashes. All 108 frozen source files match.
Actual prepared training/heldout/reference arrays, row IDs and normalization
match the accepted hashes. Closed training journals and performance readouts
were not audited again.

Engineering seeds come from the full OOD declaration, disjoint from scientific
streams. Sixty-four heldout positions and their data-derived raw observations,
transition inputs and target key are saved before model queries. Direct outputs
precede adapter loading; restored outputs, errors and devices precede gates.
Twelve direct/restored query batches ran on `TFRT_CPU_0`, using unchanged 1M
critic/500k actor checkpoints. No parameter update or radius refit occurs.
This is CPU-to-CPU restoration, not comparison with historical GPU-exported
actions or proof about simulator states.

## Preserved failure and execution correction

`td3-hopper-s202609171-connection-v1` exited 1 during runtime import: the loader
could not find `libglewosmesa.so`. No model query or simulator construction was
reached. The libraries already existed in the installed MuJoCo bin directory.
A separate loader check verified that exporting its `LD_LIBRARY_PATH` before
Python startup resolves all dependencies; the helper's in-process assignment
alone was too late for this invocation.

The documented correction used new attempt
`td3-hopper-s202609171-connection-v2`, with identical connection source, CPU
backend, libraries, checkpoints and gates. It exited 0. Original failure logs,
source and exit 1 remain. There is no automatic retry loop or scientific outcome
retry. The successful attempt retains CUDA-plugin discovery's
`CUDA_ERROR_NO_DEVICE` warning while GPUs were hidden, plus Gym/package
deprecation messages. CPU devices were verified; no backend switch or replay
to suppress warnings occurred.

## Collector engineering

`collect.py` adds lossless array/RNG artifacts, exclusive stages, predecessor
hash checks and a persistent hash-chained reservation ledger. Reservations are
saved before callbacks; interrupted calls are not refunded. Environment and
physics counts are separate. Started/failed stages cannot restart.

Its current executable pipeline is **exact-oracle only**. Arbitrary factories
and real simulator construction are refused. Production resource/restore
acceptance is unfinished and cannot be bypassed with a caller's ready flag.

Twelve collector tests and the existing protocol/receipt/analysis suites pass:
**59 tests, actual exit 0**. The initial missing-module failure remains. Tests
cover unaccepted/stale checkpoint, source, process and data bindings; changed
artifacts/streams; incomplete restore fields; duplicate attempts; interruption;
and resource ceilings.

The saved toy fixture contains 8 intended captures: 6 present, 2 missing,
12 completed state/continuation panels and 4 missing panels. Its 8 collection,
120 repeat-check and 276 outcome transitions are toy dynamics, with zero MuJoCo
steps. Returns match direct enumeration, including termination and time limits.
Familiar action -1 loses 9.1 units relative to action 0; unfamiliar +1 gains 2.9.
These are toy facts, not BCA observations. Duplicate slots keep aliases and one
ranking vote; constant scores give 0.5 AUROC only when both harm classes exist.

The support preparation uses an 80/20 training-episode split, a saved reference
bank, one sampled row per validation episode and the declared 95th-percentile
distance threshold. Missing episode IDs leave thresholded labels unavailable.
Real data-to-support-bank binding remains pending.

## Remaining boundaries

1. Reuse the accepted real full-state/repeat/reward engineering check above.
   Bind production native bounds, transition storage, budgets and locks while
   retaining cumulative engineering/constructor accounting and unchanged gates.
2. Implement and accept production collection of 64 paired episodes per
   collector with captures at 0/100. The toy's 2 episodes and 0/2 captures are
   not that scientific bank. Preserve missing captures and reset dependence.
3. Freeze ten real candidate slots, clipping/aliases, BCA and support scores,
   and actual JAX continuation keys. Explicitly toy-only keys cannot substitute.
4. Bind resource accounting and GPU locks before the 250-transition outcome
   stage. Report within-state and pooled AUROC separately. First-seed evidence
   is exploratory; all five declared seeds remain required.

Fresh coverage remains separate; ReBRAC's missing recorded-next-action target
is unavailable. Global readiness stays false. Frozen science, old campaigns,
native inventory and thesis PDF remain unchanged. At 21:53 UTC both training
lanes had valid live identities. Three new cluster closures await independent
audits; they are not new verified results here. No training queue was changed.
