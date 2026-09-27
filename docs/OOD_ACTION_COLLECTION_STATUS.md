# Action-level OOD execution: first trained pair connected

The accepted standard TD3+BC Hopper host/BCA pair, seed 202609171, passes the
**CPU trained-checkpoint query gate**. Both restored actors exactly reproduce
saved direct reference actions on 64 fixed observations per method (maximum
absolute error 0, unchanged limit 1e-6). Native targets agree exactly, including
independent target arithmetic. BCA width and dose arrays also agree exactly;
host width remains N/A.

This validates the measurement tool, **not OOD action harm**. Real state
collection and action outcomes have not started. No MuJoCo environment/physics
step, optimizer update or GPU worker was added.

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

1. Connect the real simulator with full restore/wrapper/RNG/time-limit state,
   actual transformed controls/dtypes, exact repeated transitions and raw-reward
   reconstruction under absolute 1e-7. Debit prior/new engineering use and
   constructor calls to the shared cumulative ceiling before execution.
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
