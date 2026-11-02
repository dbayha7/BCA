# Path A: final geometry, data and comparison review

Prepared for review only. No training, calibration fit, GPU workload or new simulator call was executed during this preparation. `LAUNCH_APPROVAL.json` is absent. Path B is separate and has no additional execution budget approved.

## What is established

The unchanged v2 goal/danger task has fixed physics and rewards: step cost `-1 - .01 ||a||^2`, a +100 goal-entry bonus, a -100 danger-entry penalty, discount .98 and horizon 100. Goal, danger and horizon are true terminal events. The hazard is an explicit absorbing task region in benchmark state space; it is not a claim of physically chaotic motion. All references preserve the complete 13-state vector and actual time; observations contain remaining time.

The same-state diagnostic finished November 1, 2026 at 11:12:54 p.m. EST. Independent review verified all 15 case files and all 613 physical transition records. Every fatal first step exactly reproduced its saved source transition. All 15 alternative controller continuations reached the goal: 13 at time 69 and two at time 68. Their safe-minus-fatal discounted returns ranged from 118.6586888357 to 120.0428924937; nominal advantage was 118.6594424410. This establishes a useful local choice under the fixed controller, not that a learned critic will be optimistic or that BCA will succeed.

Honeypot retains the original exclusion of every source/next/chord edge touching danger. Bottleneck now excludes every edge whose projected interval intersects the fixed slab `[.2,.8]` along B to C. This cuts all observed successful fitting routes while retaining real fragments; there is no fabricated join, next action or terminal flag at a cut. The separation proof and development-informed amendment remain in `analysis/two_scenario_v2_repair_v1/PROPOSAL.md`. This is unsupported-connector generalization, not guaranteed stitching or a claim that the connector is a unique physical route.

## Exact consumed data

`binding.py` authenticates the unchanged v2 archive, content, geometry, masks and reviewed evidence. `DATA_RECEIPT.json` records the final tuples and selected calibration IDs. These actual learner inputs were independently checked:

| Item | Honeypot | Bottleneck |
|---|---:|---:|
| Fitting rows | 55,649 | 53,669 |
| Fitting parents | 768 | 768 |
| Complete successful fitting trajectories retained | 716 | 0 |
| Genuine separated goal-fragment pairs | Not the Honeypot requirement | 716 |
| Retained B25 / C30 support parents | Original accepted approach checks | 93 / 167 |
| Calibration rows / distinct parents | 128 / 128 | 128 / 128 |

Calibration selects one retained tuple from each of the first 128 eligible calibration parents with the original fixed PCG64 seed. Fitting, auxiliary fitting and normalization use only condition-retained fitting rows. Calibration and reserved parent IDs never enter those streams. At target runtime, complete metadata must match this prepared receipt before model initialization; differing NumPy selection is an implementation failure, not permission to substitute rows. Original next observations, times and terminal flags remain byte-equivalent at native float32 precision. No `next_action` is consumed.

This bank and its construction evidence have already been inspected as development. New evaluation seeds are separate from collection, split, calibration and learner streams, but this small study is still exploratory. Region calibration does not guarantee error coverage inside either omitted region or under selected actions. A global separating cut can alter the state/time mixture; there is no action-only or missing-label causal claim without further controls.

## Frozen comparison and outcomes

Two conditions, three independent initialization seeds per condition, native UniFloral versus original BCA. Each condition/seed receives a fresh 10k native warmup, then the same full state forks into native and BCA continuations for 10k additional updates. Final policies have 20k cumulative critic updates. Full-state checkpoints include actor, twin critics, targets, optimizers, batch/phase, native and auxiliary RNG, residual-scale model and calibration reference. No historical model initializes this task.

The actor loss remains `-lambda * mean(Q1) + BC`, with detached `lambda = 2.5 / (mean(abs(Q1)) + 1e-7)`. BCA adds `lambda * 3 * mean(U)` with live action gradients and frozen reference parameters. BC coefficient stays 1. Native critic targets remain smoothed target-policy twin-min. Original uncertainty fitting and Bayesian/conformal calibration are reused. All arms receive matched auxiliary work; this is not an efficiency comparison. Calibration occurs at warmup 10k and once per continuation at cumulative 15k. No coefficient/head sweep, distance arm or method revision is added.

| Question | Fixed readout |
|---|---|
| Does BCA improve the deployed policy? | 128 paired return starts per final policy; report all three BCA-minus-native seed effects per condition. Five return units is the fixed practical reference, not a superiority guarantee. |
| Does the critic overestimate or underestimate unsupported choices? | Exact forced-first-action return followed by that same deployed actor; raw Q1 signed errors, magnitudes, MAE and over/under/near-zero counts with tolerance 1.0. |
| Does the downward score help or harm choices? | Separately report Q1 and Q1-3U errors, over-to-under crossings, worsened underestimation, common-action ranking regret and beneficial/harmful switches. |
| Do the policies actually encounter missing support? | Actual returned policy edges labeled by the same danger/slab construction mask; goal/danger/horizon outcomes and denominators retained. |

Final action banks are now explicit: controller, opposite controller and zero, plus each actor's own action as a fourth diagnostic slot. The fourth slot is excluded from shared-action rankings. Honeypot probes use t29 after controller actions until t25 and opposite actions at 25-28; Bottleneck probes use the nominal t25 approach. These fill the original protocol's unfinished bank definition using validated task mechanics, independently of BCA outcomes. Each bank has 16 fresh starts per condition/seed. A separate fixed-controller continuation panel assesses common action values and is never substituted for own-policy Q error. Controller references preserve full precision rather than reconstructing state from float32 observations.

Each prefix is followed by exactly one replay of its last step for parity. An early-terminal/invalid prefix is recorded as incomplete evaluation; no replacement anchor or selected subset is used. All negative, neutral and positive model results remain in the complete report. Three training seeds are the independent units; episodes and anchors are nested samples, not additional training replications. No Bayesian-uniqueness conclusion is supported by this two-arm comparison. A downward penalty cannot directly raise a fixed underestimated Q.

## Work and remaining allowance

| Work | Maximum |
|---|---:|
| Training jobs / final policies | 18 / 12 |
| Additional critic / actor / auxiliary updates | 180,000 / 90,000 / 180,000 |
| Calibration calls | 18 |
| Return episodes | 1,536 |
| Own-policy / shared-controller forced-action references | 768 / 288 |
| Reserved evaluation transitions | 237,600 |
| Tight evaluation transition bound from final recipes | 233,376 |

Tight phases: 153,600 free-policy steps + 56,064 own-policy reference steps + 21,024 controller-reference steps + 2,688 prefix/parity steps. The unused 4,224 reserve does not authorize extra probes. No collection or dynamics replay is added. Real runtime initialization, exact checkpoint serialization/restoration and inference compilation consume time but no optimizer updates or simulator calls. The first two actual training updates receive an exact-state serialization check within their existing count; no duplicated optimizer steps are run for a smoke test.

Prior completed physical work is 189,655 transitions. With the unchanged evaluation ceiling, the cumulative planned maximum is **427,255 of 498,400**, leaving 71,145 unallocated setup transitions. Known measured prior Python time is **30.126427546 seconds**; this is not the budget debit. Prior conservative charges remain **1,890 seconds**. This final CPU preparation reserves another **60 seconds**, leaving a **12,450-second (3h 27m 30s) process ceiling**; 1,890 + 60 + 12,450 = the original four hours. No allowance is reset or reclaimed from shorter observations.

The supervisor has a 12,435-second internal deadline. The manual command adds an external 12,445-second timeout and five-second kill grace. All jobs share this deadline; it is not per job. Runtime is a ceiling, not a promise that training plus evaluation will finish. A timeout, signal, changed source, failed calibration, nonfinite state, failed restore or incomplete report stops this one attempt. A hard-killed job has saved lower bounds and its declared upper bounds; it is never counted as zero and is not automatically restarted. Only this launcher's child process group is stopped on its own deadline.

## Checks and remaining limitation

**41 CPU tests passed** on Windows Python 3.10.11 / NumPy 2.2.6. They cover actual immutable saved data and fake runtime/plant behavior, including complete negative-outcome evaluation, exact fork/restore interfaces, source/postcheck failures, corrupted checkpoints, calibration isolation, full-state time/termination, counts and persistence failures. Both real CLI `--help` paths passed. The original sealed data loader emits an unclosed-reader ResourceWarning; its bytes and data checks still pass, and that historical file was preserved.

Independent code review found no remaining must-fix blocker. **The new adapter has not been executed on the target GPU.** Its worker requires the existing frozen CUDA/JAX runtime and matches prepared metadata before initialization, then verifies exact complete-state restoration at initialization, after its first two counted updates and at final checkpoints. Final policies are authenticated and restored again for evaluation. These guards do not constitute a prior real CUDA smoke pass. There is no extra synthetic fit or simulator smoke hidden in the budget.

The source-only preflight command can be run without training:

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/c/Users/David/dev/BCA && /home/david/bca-operator-audit-20261026/bin/python -B experiments/thrusterchain_bca/two_scenario_path_a/launch.py --check'
```

Only after final review and a new manifest-bound `LAUNCH_APPROVAL.json`, the conditional one-shot command is:

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/c/Users/David/dev/BCA && timeout --signal=TERM --kill-after=5s 12445s /home/david/bca-operator-audit-20261026/bin/python -B -u experiments/thrusterchain_bca/two_scenario_path_a/launch.py --execute'
```

Keep the foreground terminal open. The dispatcher creates `run_1/ATTEMPT.json`; each job has `worker.log`, `progress.jsonl`, full checkpoint and report, plus worker and supervisor receipts. Final results are `run_1/RESULT.json` and `run_1/evaluation/report.json`. Existing reservations block a duplicate. This is a local guard, not a distributed lock. Do not launch the same study on two machines. No older approval is accepted, no approval was created during preparation, and assistant-side WSL denial remains in force.
