# Required end-to-end runtime smoke before Path A

This is the current launch plan. It supersedes the old direct-launch instructions in README.md without editing the original sealed package, dataset, protocol, checkpoints or receipts. The original manifest remains `2f10a640684f6ea7040df9fe0388544207e1c4562a547a2ff24a04d31bf1a366`. No old `LAUNCH_APPROVAL.json` is created or accepted. A separate amendment approval is required and is currently absent. No runtime smoke or scientific run has started.

## Exact execution order

1. Authenticate the original and additive source seals, the new approval, and the unchanged full dataset/geometry. In the target runtime, reconstruct and compare the exact prepared Honeypot and Bottleneck tuple, calibration-row and normalization metadata.
2. Initialize one **disposable Bottleneck** learner with the original initialization seed and fixed BC. Use the production update kernels with an explicitly short disposable schedule: four critic/auxiliary steps and genuine calibration at step 4. This does not impersonate a 10k checkpoint.
3. Fork matched native and BCA numerical paths using the production RNG-fold equations and unchanged actor objectives. Each path executes steps 5, 6 and 7, including genuine calibration at 6. Serialize the complete odd-phase state at 7, including its held minibatch, optimizers, targets, RNG and calibration reference. Execute actor-bearing step 8 from both the original and restored state and require exact complete-state equality. No tolerances are loosened. Invalid calibration is a failed smoke; no reference is fabricated.
4. Using the exact final evaluator helpers, physically construct **all six frozen probe banks**: three seeds × two conditions, 16 anchors each. Execute all 48 predefined common first-action edges per bank. Each bank must contain at least one retained and one omitted edge, with all cases retained. No resampling, replacement or threshold adjustment is allowed.
5. With both disposable actors, exercise the production raw-Q, width, adjusted-score and own-policy rollout callbacks. From the saved nominal t29 danger state, replay the fatal first action and require exact next state/reward and immediate termination at time 30. Separately force the controller first action and then follow each disposable actor to its actual terminal/horizon; also perform one free rollout per actor from A. Poor returns or large Q errors cannot fail this technical check. Shape, finite-value, exact-replay, termination, identity and accounting errors can.
6. Persist full smoke checkpoints, traces, bank counts and receipts. Only a completed supervisor result, exact work accounting, both restored actor steps, all six coverage checks, unchanged sources and a hashed raw-artifact manifest can authorize the scientific dispatcher.
7. **Only then**, in a separately reserved attempt, initialize all scientific learners afresh and execute the unchanged 18-job comparison and frozen evaluation. Disposable smoke models/calibration references are never reused. The successful smoke proof hashes are recorded in the scientific attempt and each job and checked again before dispatch and final acceptance.

The numerical smoke uses Bottleneck fitting/calibration rows; it does not run a separate Honeypot optimization smoke or validate learning efficacy after 20k updates. Honeypot data binding is checked exactly, both conditions' physical probe banks are exercised, and both numerical actor paths are tested. The same runtime kernels serve both scientific data conditions. Source-only checks or exit code 0 cannot substitute for this complete smoke PASS.

## Work requested for approval

| Component | Maximum |
|---|---:|
| Disposable critic / actor / auxiliary updates | **14 / 8 / 14** |
| Genuine disposable calibrations | **3** |
| Fixed prefixes and one-step parity | 2,688 physical transitions |
| Common first-action coverage edges | 288 |
| Two fatal and two safe-first own-policy references | 144 |
| Two free-policy rollouts | 200 |
| **Smoke physical maximum** | **3,320** |
| Smoke process reservation | **900 seconds** |

Update arithmetic: warmup 4 critic / 2 actor / 4 auxiliary / 1 calibration; each of two branches contributes 5 critic / 3 actor / 5 auxiliary / 1 calibration, including the duplicated actor-bearing step 8. Calibration at step 6 is not duplicated. These are real disposable learning operations, not zero work or a claim of “inference only.”

Scientific work remains 180,000 critic / 90,000 actor / 180,000 auxiliary updates and 18 calibrations, through 18 jobs and 12 final policies. With smoke, the proposed cumulative learning caps are **180,014 critic, 90,008 actor, 180,014 auxiliary and 21 calibrations**. These extra optimizer/calibration operations require explicit approval.

Physical work becomes at most **189,655 prior + 3,320 smoke + 237,600 scientific evaluation = 430,575**, within the original 498,400 ceiling. Unused capacity is not permission for additional queries or retries. Scientific evaluation remains 1,536 return episodes, plus the existing own-policy and common-controller panels.

Time accounting: **1,950 seconds previously charged + 60 seconds reserved for amendment preparation + 900 seconds reserved for smoke + 11,490 seconds reserved for scientific execution = 14,400 seconds**. The prior known measured Python time remains 30.126427546 seconds, separate from conservative charges. A short smoke does not automatically enlarge the scientific allowance. The scientific phase has at most **3h 11m 30s**, including source checks, initialization, compilation, checkpoint I/O and evaluation. This is a ceiling, not a promise of completion.

## Geometry and claims remain fixed

The plant has five position/velocity pairs plus current and wave state. The task uses a radius-.1 goal at the nominal t75 configuration and a radius-.1 danger region at the five-opposite-action excursion endpoint. Step reward is `-1-.01||a||²`, plus 100 on goal entry or minus 100 on danger entry. Danger crossings take precedence; goal/danger/horizon100 truly terminate. State-space coordinates are not literal physical metres, and this is an explicit absorbing danger task rather than claimed chaotic physics.

Honeypot probes the verified local danger approach at t29 after four prescribed opposite actions. This is not the separately proposed one-action-from-B geometry of Path B. Bottleneck probes the nominal B approach at t25 and uses the closed projection slab `[.2,.8]`. The final fitting input retains zero complete successful routes, 716 genuine fragment pairs and B/C support from 93/167 parents. Original transitions and real fragments are preserved. BCA remains the original actor `Q1-3U`; BC stays unchanged. No signed correction, new world, sweep or optional arm is added.

The smoke verifies actual first-edge coverage of the fixed banks; it does not establish free learned-policy exposure, overestimation, underestimation or BCA benefit. The bank construction is now explicitly inspected for technical support coverage, so it is not untouched confirmation. Final paired returns and each actor's own-policy Q errors remain the scientific outcomes, with retained/omitted counts and harms reported. A failed coverage or runtime smoke remains a recorded failure and blocks training.

## Manual sequence after explicit amendment approval

The assistant has not executed WSL, CUDA, learning, calibration or the simulator. Its existing WSL denial remains in force. After review and a manifest-bound `AMENDMENT_APPROVAL.json`, run the smoke once from PowerShell:

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/c/Users/David/dev/BCA && timeout --signal=TERM --kill-after=5s 895s /home/david/bca-operator-audit-20261026/bin/python -B -u experiments/thrusterchain_bca/two_scenario_path_a/smoke_launch.py --smoke'
```

Inspect `smoke_attempt_1/RESULT.json`. Only if it reports `passed_runtime_smoke`, the separate conditional command is:

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/c/Users/David/dev/BCA && timeout --signal=TERM --kill-after=5s 11485s /home/david/bca-operator-audit-20261026/bin/python -B -u experiments/thrusterchain_bca/two_scenario_path_a/smoke_launch.py --execute'
```

The command itself rejects missing, failed, incomplete, altered or differently sealed smoke evidence before reserving scientific output or creating a learner. Smoke and scientific attempts are exclusive and have no automatic retry. Do not use the older direct launcher or run the same study on another machine. Keep the foreground terminal open. Source-only preflight is `smoke_launch.py --check`; it does not run the GPU smoke.

Current status: implementation and cheap checks prepared; actual target-runtime PASS is still pending. Path B remains a separate, unapproved additional execution budget.
