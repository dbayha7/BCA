# Path A recovery on Titan

Status: implemented and CPU-tested, **not sealed, not approved and not executed**. `RECOVERY_MANIFEST.json` and `RECOVERY_APPROVAL.json` are absent. The copied Zeus artifacts are not yet on Titan. Nothing in the original seal, the smoke amendment, Zeus's smoke or the interrupted attempt is edited. The scientific design is unchanged.

## What happened and what is reused

On Zeus, the smoke amendment's runtime smoke passed (`smoke_attempt_1`). The production attempt `run_amendment_1` completed and accepted `honeypot_s0_warmup`, then was interrupted during `honeypot_s0_native`. That attempt is never reopened or retried.

- **Reused:** the `honeypot_s0_warmup` job folder, byte for byte. Its checkpoint chain (report → checkpoint receipt → `full_state.msgpack` → state digest) is authenticated at `--prepare` and again before reuse. Before the first job it is copied into `run_recovery_1/` and hashed on both sides, so the sealed `train.run_job` and `worker.evaluate_job` run unchanged with every job folder as a sibling.
- **Discarded and re-run:** the interrupted `honeypot_s0_native`. Branches save no mid-branch checkpoint (`resume_supported: False`), so it restarts from the warmup fork. Its persisted per-job counts are the lower bound. The full job cap (10,000 critic / 5,000 actor / 10,000 auxiliary / 1 calibration) is its upper bound unless a failed-job report records exact counts. It is never counted as zero.
- **Remaining work:** the 17 unchanged training jobs (`honeypot_s0_native` … `bottleneck_s2_bca`, in the sealed order), then the frozen evaluation (12 policies, 1,536 return episodes, own-policy and common-controller references).

## Why Titan needs its own checks

Titan's `/home/dbayha/bca-gpu` environment matches all 11 package versions that the frozen runtime pins. That is not enough on its own:

1. **Smoke proof.** Zeus's smoke validated Zeus's runtime. Titan runs the same sealed smoke itself (`smoke_launch.run_smoke`, same 14/8/14 updates, 3 calibrations, 3,320-transition cap, 900 s). It is accepted by the same `smoke_launch.smoke_passed` rule.
2. **Controller numerics.** `two_scenario_v2/task.py:212` solves a Riccati recursion with `np.linalg.solve`. Commit `4535d9b7` found that Titan's default OpenBLAS kernels reproduce a comparable recursion one last bit off Zeus's, and that `OPENBLAS_CORETYPE=Haswell` reproduces Zeus exactly. Every recovery process requires `OPENBLAS_CORETYPE=Haswell`. Two Zeus-equivalence gates also verify the result:
   - **Controller actions, 0 simulator transitions:** all 598 saved fixed-controller actions in `analysis/two_scenario_v2_repair_v1/danger_attempt_1/transitions.jsonl` (Zeus, t29 → goal, 15 families) must be recomputed to identical float32 bytes. This runs before any learner is initialized.
   - **Probe banks, 0 extra transitions:** after the smoke, all 398 controller- and plant-only bank files (plan, own-policy anchor, and per bank: starts, 16 anchor traces, 48 common edges, coverage) must be byte-identical to Zeus's `smoke_attempt_1/physical/`. Learner-dependent files are excluded.
3. **Warmup restore, 0 updates.** Zeus's warmup must restore in Titan's runtime with the exact saved state digest and fork into identical native/BCA states. That takes one model initialization and no optimizer step.
4. **Same device.** Every production worker requires the GPU device kind and packages recorded by the passed Titan smoke.

A failed gate stops the attempt and is recorded. Nothing is replaced, retried or loosened.

## Work requested for approval

| Component | Maximum |
|---|---:|
| Titan smoke: critic / actor / auxiliary updates, calibrations | 14 / 8 / 14 / 3 |
| Titan smoke simulator transitions | 3,320 |
| Titan smoke process time | 900 s |
| Production: 17 training jobs, critic / actor / auxiliary updates, calibrations | 170,000 / 85,000 / 170,000 / 17 |
| Production evaluation transitions | 237,600 |
| Production process time | 28,800 s |

**Optimizer updates, cumulative.** The prior completed work is Zeus smoke 14 / 8 / 14 / 3, plus warmup 10,000 / 5,000 / 10,000 / 1, plus the interrupted job (≤ 10,000 / 5,000 / 10,000 / 1). With this recovery the cumulative maximum is **190,028 critic, 95,016 actor and 190,028 auxiliary updates, and 25 calibrations**. That exceeds the earlier 180,014-critic approval by 10,014: the discarded interrupted job plus the second smoke. `--prepare` writes the exact bounds from the copied receipts.

**Simulator transitions.** 189,655 prior + Zeus smoke (≤ 3,320; exact from its report) + Titan smoke (≤ 3,320) + evaluation (≤ 237,600) = **at most 433,895 of the unchanged 498,400 ceiling**.

**Time.** The original four-hour allowance (1,950 + 60 + 900 + 11,490 s) is fully reserved, and this recovery does not reclaim any of it. It requests a new, separate **900 + 28,800 s** on Titan. The basis is Titan's only comparable measurement: `titan_20261026/payload/unifloral_checkpoint_v1/train_20261023T183248_552877Z` ran 20,000 critic, 10,000 actor and 20,000 auxiliary updates in 2,339 s, about 1,170 s per Path A job. 17 jobs come to about 19,900 s, plus up to 3,600 s reserved for evaluation. The ceiling is not a promise of completion. Zeus's warmup elapsed time is shown by `--check`; the ceiling should be reconsidered before `--prepare` if it disagrees.

**Job-boundary stop.** Before each training job the supervisor requires `1.25 × slowest completed job + 3,600 s` to remain, and before evaluation it requires 3,600 s. If that is not available, it stops between jobs (`stopped_at_job_boundary`). Every accepted job keeps its authenticated checkpoint, and continuing needs a new approval. This changes only dispatch, never the science, and it avoids losing a job to the deadline.

## Limitations

- **Seeds come from two machines.** Seed 0's Honeypot warmup comes from Zeus's GPU; all branches and the other five warmups come from Titan's RTX 5070 Ti. Each native/BCA pair still shares one warmup state and runs on one device, so pairing holds. Absolute results are not what Zeus would have produced. Report this.
- **Exploratory scope is unchanged.** It remains a three-seed exploratory study on inspected development banks. Smoke and equivalence checks are technical. They establish no exposure, over- or underestimation, or BCA effect.
- **No Path B.** No Path B execution or budget.

## Exact sequence

From PowerShell in `C:\Users\David Bayha\BCA` (Windows Python; no WSL, no learner):

```powershell
py -3.12 -B experiments\thrusterchain_bca\two_scenario_path_a\recovery_launch.py --check     # verifies the copied Zeus artifacts, prints bounds
py -3.12 -B experiments\thrusterchain_bca\two_scenario_path_a\recovery_launch.py --prepare   # seals RECOVERY_MANIFEST.json once
```

After review, **only you** run the approval (it prints the approval text and writes `RECOVERY_APPROVAL.json`):

```powershell
py -3.12 -B experiments\thrusterchain_bca\two_scenario_path_a\recovery_launch.py --approve
```

Then the Titan smoke, in a foreground terminal:

```powershell
wsl -d Ubuntu -- bash -lc "cd '/mnt/c/Users/David Bayha/BCA' && OPENBLAS_CORETYPE=Haswell timeout --signal=TERM --kill-after=5s 895s /home/dbayha/bca-gpu/bin/python -B -u experiments/thrusterchain_bca/two_scenario_path_a/recovery_launch.py --smoke"
```

Inspect `titan_smoke_1/RESULT.json`. Only if it reports `passed_runtime_smoke`:

```powershell
wsl -d Ubuntu -- bash -lc "cd '/mnt/c/Users/David Bayha/BCA' && OPENBLAS_CORETYPE=Haswell timeout --signal=TERM --kill-after=5s 28795s /home/dbayha/bca-gpu/bin/python -B -u experiments/thrusterchain_bca/two_scenario_path_a/recovery_launch.py --execute"
```

`--execute` refuses to reserve `run_recovery_1/`, copy the warmup or start a child unless the passed Titan smoke, both seals, all pinned Zeus artifacts and the approval verify. The final results are `run_recovery_1/RESULT.json` and `run_recovery_1/evaluation/report.json`. Do not run this study on Zeus as well. Titan's r5 screen and this recovery should not share the GPU.
