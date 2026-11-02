# Loader-only recovery v1

Authorized November 1, 2026 at 12:39:22 p.m. EST (user message
`Sentinel_fb77d34980fc81919f581633ce116fce`). The original package, validator,
manifest, approval, failed attempt and reset diagnostic remain unchanged.
This sibling directory adds a dataset consumer and thin adapters around the
original launcher/worker. It does not copy or revise the learner/evaluator.

## Exact change

The producer generated and validated all 1,280 episodes, including exact
reset/recorder lineage and 128,000 next-state/reward rows, before computing
the canonical digest and saving. The new loader authenticates that successful
collection path, all 44 producer source pins, the recorded producer evidence,
the exact archive SHA-256 and a freshly recomputed canonical content SHA-256.
It checks metadata/schema/shapes/dtypes/finite values and constructs the same
dataset class with unchanged, bytes-backed immutable arrays. No reset or
dynamics are invoked by this consumer. Every worker report records reuse.

The original validator is not patched. The exact WSL reset check remains a
recorded failure: episode 16 differed by one ULP in wave_1, although the
float32 observation matched. No numerical tolerance has been introduced.
The producer evidence is a sealed successful collection call chain, not a
standalone validation receipt, and its manifest does not fully attest the
historical runtime dependency closure. Reuse does not certify bit-exact WSL
reset/dynamics regeneration. The new GPU adapter path remains unexecuted.

## Science preserved

The original config is read without modification: stall3/CUTOFF, native
UniFloral versus original direct-Q BCA at kappa 3, three paired initialization
seeds, 10k shared native warmups and 10k continuations. Nine training jobs,
six policies, 90,000 critic, 45,000 actor and 90,000 auxiliary updates, nine
calibrations, 768 return episodes and the same own-policy Q-pi probes.
The original dataset, splits, normalization, checkpoint/RNG restoration,
reference policies, thresholds and exploratory interpretation are unchanged.
RS1 remains failed (121/128 versus 122); no action-only or Bayesian-uniqueness
claim is added.

## Accounting and time limit

Known earlier active execution: 19.40567558 + 7.002520411 = 26.408195991 seconds.
The original receipt's unknown update bounds are preserved. A separate
source-path assessment records zero prior learned updates/calibrations:
the authenticated traceback failed in data loading before bind/state
initialization/training/calibration. This is not a replacement terminal counter.

Charge 100 seconds against the original 10,800-second allowance: the measured
26.408195991 seconds plus a conservative 73.591804009-second allowance for
unmeasured old/new WSL and timeout-wrapper overhead. The allowance is not an
observation or a proven bound; exact external wrapper elapsed time remains
unknown. No new three-hour allowance is granted.

The recovery process has a 10,700-second maximum. At 10,680 seconds a Python
alarm interrupts execution, including analysis/checks; the reused supervisor
kills only its owned worker process group on interruption. Child dispatch is
also bounded by the absolute recovery deadline. The manual GNU timeout fires
at 10,695 seconds, with at most five seconds before forced termination.
Existing per-job limits remain 1,800 seconds for training and 600 seconds for
evaluation. Partial receipts are evidence, never permission for an automatic
retry. No reservation is reset.

Expected new simulation is 170,400 steps: 76,800 return steps + 86,400 Q-pi
reference steps + 7,200 anchor-prefix steps. The original 1,706,400 cap remains
a conservative upper bound. Its 1,536,000 dataset validation replay component
is historical validation reused, not newly performed work. Reports use actual
counter values. Prior observed simulator steps were zero (the diagnostic
performed one reset).

## Manual launch

```powershell
wsl -d Ubuntu -- timeout --signal=INT --kill-after=5s 10695s /home/david/bca-operator-audit-20261026/bin/python -B -u /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/minimal_comparison_recovery_v1/launch.py --execute --approval /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/minimal_comparison_recovery_v1/LAUNCH_APPROVAL.json
```

The existing Linux overlap check runs before an exclusive new `run/`
reservation. Source/provenance checks run before dispatch and after work.
Jobs write `run/<job>/progress.json`, `report.json`, and supervisor receipts.
`run/RESULT.json` is the original runner's terminal result for this recovery;
`run/RECOVERY_RESULT.json` adds lineage and cumulative accounting. A hard kill
may leave incomplete receipts; do not restart that reservation.

Preparation checks use saved arrays, fixture classes and mocked dispatch only.
They do not import real models, instantiate/reset an environment, launch a
worker process, train, calibrate or evaluate. This is a reviewed recovery
package, not a claim that CUDA execution or scientific efficacy has passed.
