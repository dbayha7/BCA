# Seed 2 continuation: preparation only

Preparation authorized November 1, 2026 at 2:02:17 p.m. EST by message
`Sentinel_c8c6ed08ff6c81919c0e199d5e5c2402`. Expanded physical work is not yet
approved. No launch approval file is created and no reservation is made.

This is a thin scheduling adapter around the existing runner and immutable
loader. It preserves all original sources, approvals, attempts and checkpoints.
The original config remains byte-identical: native UniFloral versus original
BCA kappa 3, three paired seeds, the same dataset, splits, initialization,
normalization, fitting/calibration schedule, evaluation and exploratory screen.
The separate kappa-63 native_screen is not an input to this study.

## Jobs and reuse

Reuse the six completed seed 0/1 jobs in place, authenticated through their
reports, job/config/source bindings and 20 actor/full-state checkpoint files.
Schedule only seed 2's warmup (10k critic updates), native continuation (10k)
and BCA continuation (10k), followed by the same three evaluation jobs.
Evaluation references old seed 0/1 directories and new seed 2 directories.
Six final policies each have 20k cumulative updates. Continuations add 10k.

Seed 2's old warmup has no checkpoint; restarting it is repeated physical work,
not a checkpoint resume. Its last progress records 5k critic/5k auxiliary/2.5k
actor updates and no calibration. The unknown interruption tail stays counted.
No completed job is retrained. No old reservation is reset or copied.

## Proposed work and approval needed

| Work | Critic | Actor | Auxiliary | Calibrations |
|---|---:|---:|---:|---:|
| Six completed jobs | 60,000 | 30,000 | 60,000 | 6 |
| Interrupted warmup, including unknown tail | 5,000–10,000 | 2,500–5,000 | 5,000–10,000 | 0–1 |
| New seed-2 jobs | 30,000 | 15,000 | 30,000 | 3 |
| Cumulative physical bounds | 95,000–100,000 | 47,500–50,000 | 95,000–100,000 | 9–10 |

Approval is needed for cumulative caps **100,000 critic, 50,000 actor,
100,000 auxiliary updates and 10 calibrations**. These exceed the original
90k/45k/90k/nine caps solely by the conservative maximum lost warmup.
The first failed loader attempt retains its raw unknown counters and separate
source-path zero-update assessment; its receipt is not rewritten.

Evaluation remains 768 return episodes and 170,400 new simulator steps. The
previous terminal receipt conservatively bounds the interrupted worker's
simulator use at 0–128,000; preserve that bound, giving up to 298,400 cumulative
steps, still below the unchanged original 1,706,400 cap. Reused producer-side
dataset validation is never counted as newly executed simulation.

## Time

Known prior active time: 3,154.914181852 seconds. Previous charged time including
wrapper allowance: 3,228.505985861 seconds. Thus 7,571.494014139 seconds remains
from the original 10,800-second allowance.

Propose a **7,500-second process cap (2h 5m)** and reserve the remaining
71.494014139 seconds for additional unmeasured wrapper overhead. Total charged
prior time becomes 3,300 seconds; 3,300 + 7,500 = 10,800. Allowances are not
measured overhead or a proven bound on it. The three-hour allowance is not reset.

The reused runner's alarm/owned-child deadlines will interrupt work at 7,480
seconds, leaving shutdown/reporting time. A future reviewed foreground command
will use GNU timeout at 7,495 seconds plus at most five seconds for forced
termination. No training command is issued until expanded-work approval is
bound to this manifest. `--execute` refuses missing, old or mismatched approval.

## SIGHUP and launch procedure

The old receipt records `KeyboardInterrupt: Recovery supervisor signal 1`,
`interrupted=true`, `timed_out=false`, worker exit -9. The supervisor received
SIGHUP and then killed its owned worker during cleanup. It does not identify
the sender. No OOM, numerical failure or deadline expiration is established.

Use an ordinary foreground PowerShell/Windows Terminal session kept open until
the terminal receipt appears; avoid closing the terminal or restarting WSL
during execution. This reduces a common hangup risk but does not prove or cure
the cause of the previous signal. No system/session settings are changed and
no new terminal infrastructure is required. `nohup` alone is insufficient
because the runner installs its own SIGHUP handler. The existing read-only
overlap check runs in the user's launch and refuses overlapping experiment
workers. The separate native_screen attempt 3 has a terminal receipt at
2:05:10 p.m. EST on November 1: inconclusive at G1, with no training.
This is separate evidence, not an outcome of this comparison. The overlap
check remains required because a terminal receipt is not a global process inventory.

## Evidence and limits

The one-ULP WSL reset-replay failure, historical producer dependency-attestation
limit and failed RS1 criterion remain disclosed in the existing protocol.
This is an exploratory restricted-data comparison, not an action-only or
Bayesian-uniqueness claim. No evaluation outcomes have been inspected or used
to choose this continuation.

Checks are hashes, saved-file analysis, static imports and mocked dispatch only.
They do not load model state into JAX or execute environment/model workloads.
Outputs, if later authorized, go to this directory's exclusive `run/`.
`RESULT.json` and `RECOVERY_RESULT.json` preserve new-attempt accounting;
`CONTINUATION_RESULT.json` adds cumulative bounds without erasing old receipts.

## Preparation verification

Ten tests passed on the Windows Python runtime: authentication of the six
completed reports and 20 checkpoints, exact seed-2 scheduling and evaluation
paths, cumulative work/time accounting, preservation of unknown tails,
original configuration/deadline adaptation, rejection of missing/old approval,
mocked full dispatch and duplicate-reservation rejection, construction of the
actual worker adapter chain without configuring models, and corrupt-pin
rejection before importing the prior worker. Both CLI help interfaces worked.
An independent source reviewer found no blocking schedule, pairing or
pre-import verification defect; its requested worker-chain checks were added.
These are not CUDA or production runtime tests.

Preparation snapshot: branch `codex/thrusterchain-bca-handoff-20261026`,
HEAD `45fc6cb338df85a76103e982658deb20f668b708`. Concurrent files were
preserved. The new manifest authenticates the prior pinned source closure,
interrupted evidence, completed checkpoints and this adapter, not every
unrelated working-tree file.
