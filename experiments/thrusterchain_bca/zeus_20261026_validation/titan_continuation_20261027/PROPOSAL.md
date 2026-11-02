# Proposed Titan continuation after the stopped Zeus retry

Status: input handoff and proposal only. No continuation scheduler or GPU run is
enabled. The user canceled attempt 2 after seed 0 completed and requested that
experiment execution move to Titan. The original supervisor retains
`resume_supported: false`; its sources and reservation are unchanged.

## Preserved outcome

Attempt 2 ended with `stopped_after_incomplete_job`. The supervisor received
SIGINT (signal 2), and its shutdown handler killed its owned worker, producing
exit -9. The timeout flag is false. The user confirmed the cancellation; no OOM
or scientific failure is inferred.

The two completed seed-0 warmups each performed 10,000 critic, 5,000 actor and
10,000 auxiliary updates with one accepted calibration. Their report, supervisor,
actor and full-state seals match. The third warmup, seed 1 restricted, has a
last journal at 2026-10-27 04:46:46.886428 UTC recording another 320 critic,
160 actor and 320 auxiliary updates. An earlier console excerpt recorded 260/130/260.
The later journal is the stronger lower bound; exact final partial counts remain
unknown. No policy branches or evaluation ran, so there is no BCA utility result.

The complete planned warmup order was seed 0 broad, seed 0 restricted,
seed 1 restricted, then seed 1 broad. Seed 1 broad was intended fourth; its absence
from the terminal job list means it had not started.

The immutable copy-preserving archive is
`../archived_attempts/support_shift_retry2_interrupted_20261027T042116Z`.
It contains 75 snapshot files totaling 23,575,975 bytes, including two full states
of 10,350,811 bytes each. Original files and reservations remain in place on Zeus.
The archive manifest SHA-256 is
`8c5c11c41085e4201392c747c8e06f21df3c06fdcdc9b28a3cf84013c628fbf2`.

## Exact reuse that the format supports

Reuse only the two sealed completed warmups. Each exports the full TrainingState:
native and target parameters and optimizer states, auxiliary optimizer and scale,
calibration posterior, native and auxiliary PRNG state, cached minibatch, phase,
update counters and calibration schedule. Condition normalization is sealed in
the manifests, and untrained actors survive for the learning-adequacy controls.
There are 227 declared state leaves per full checkpoint.

| Cell | Full-state SHA-256 | Frozen distance coefficient |
| --- | --- | ---: |
| seed0_broad_count_matched | `4ee9e7b372130d1ec84bac5e9464563808e3dae0d26878ee59680c2c214b9e67` | 0.07585831620304931 |
| seed0_restricted | `57ce1bc5d2c59a32719dc0817137dda0d66ce3544317289ea43243a0916e0bbf` | 0.0670392257429505 |

Existing branch code already reconstructs the pinned native template, requires
identical initial parameter hashes and normalization, restores the full state,
and demands a byte-exact serialization round trip before applying the unchanged
branch RNG fold-ins. It verifies the matching reference and actor anchor hashes.
This supports a new explicit continuation design; it does not authorize bypassing
the original no-resume guard.

## Proposed remaining work

Use a new, separately versioned continuation manifest and scheduler on Titan.
Import the two seed-0 warmups as read-only, hash-verified inherited inputs, with
their prior counts recorded separately from newly executed work. Never import the
partial seed-1 state as a trained warmup or an evaluation policy.

1. Run fresh seed-1 restricted and seed-1 broad warmups using the original seeds,
   data plan, architecture and budgets. Keep seed 0's fixed coefficients; compute
   only the two originally planned seed-1 coefficients from their logged training
   probes. No rematching from evaluation outcomes.
2. Validate paired initializations across both conditions for each seed, and
   distinct initialization across seeds, using all four completed warmup records.
3. Run all twelve originally planned branches (native UniFloral, original direct-Q
   BCA at kappa 3, and the training-gradient-matched distance control in each cell).
   Preserve midpoint calibration, branch RNG/data pairing and original ordering.
4. Perform the original final-only evaluation: twelve final policies plus four
   condition-specific untrained references, on the same 32 held-out reset seeds.
   Apply the unchanged learning gate, eight BCA-versus-control comparisons, and
   separate action-gap exposure interpretation. No extra evaluation or tuning.

| New work | Maximum |
| --- | ---: |
| Warmups / branches | 2 / 12 |
| Critic updates | 44,000 |
| Actor updates | 22,000 |
| Auxiliary updates | 44,000 |
| Calibrations / posterior draws | 14 / 14,000 |
| Evaluation episodes / simulator transitions | 512 / 51,200 |

The retained scientific training plan is 20,000 inherited plus 44,000 new critic
and auxiliary updates. Physical consumption is larger: attempt 2 plus completion
would total at least 64,320 critic/auxiliary and 32,160 actor updates, including
discarded partial work. Attempt 1 is additional and separately lower-bounded.
Do not describe this as an unchanged aggregate 64,000-update physical cap.

Proposed new-stage ceiling: **7,594 seconds (126 minutes 34 seconds)**, with original
per-worker caps (1,800 seconds per warmup, 300 per branch, 600 for evaluation),
each also bounded by the remaining stage budget. This derives from
floor(9,000 - 1,405.109554826), using the original supervisor's reported elapsed
time. It is a proposed cumulative active-execution rule that excludes the migration
pause, a change from the old continuous supervisor deadline, and requires review.
The reservation-to-terminal-file UTC interval was 1,533.79 seconds; the discrepancy
from reported monotonic elapsed is unexplained. Neither clock supports a claim of
exact end-to-end wall duration across the migration.

## Required Titan checks and limitations

First pull this handoff and run `verify_handoff.py` with any available Python 3.10+.
It performs file/metadata checks only. The historical Titan runtime was
`/home/dbayha/bca-gpu/bin/python`; its current existence, packages and GPU have not
been inspected in this task. No installation or environment change is authorized
by this proposal.

After continuation scope review, a zero-update Titan readiness step must verify
the pinned package/device configuration, exact template and normalization identity,
and complete state serialization round trip for both inherited warmups. Include
that startup work in the new stage budget. If any required identity check fails,
stop for review. Do not replace normalization, loosen equality checks or silently
substitute CORL code, checkpoints, packages or methods. A fresh Titan attempt is
the fallback if exact reuse cannot be established; it is not automatic.

All new branches and evaluation would run on Titan, but seed-0 warmups ran on Zeus
and seed-1 warmups would run on Titan. Paired arms within each cell still share an
identical base; cross-seed differences cannot be labeled seed-only variation.
Byte-exact restored state does not imply bit-identical future GPU arithmetic.
Logged-probe gradient matching does not guarantee equal penalty strength along
every learned-policy trajectory. Action filtering also changes the state mix, as
already disclosed in the original protocol.

No head sweep, failed one-sided revision, confirmation reuse, additional tuning,
or broader training is included. No execution command is supplied until the
continuation scope is reviewed and implemented.
