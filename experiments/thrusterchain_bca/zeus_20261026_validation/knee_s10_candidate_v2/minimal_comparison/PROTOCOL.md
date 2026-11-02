# Minimal native versus BCA comparison — prepared for review

Question: after a shared native 10,000-update warmup on one fixed restricted-action offline dataset, does enabling original BCA improve subsequent policy return compared with continuing native UniFloral TD3+BC?

This is a development experiment. The candidate and earlier diagnostics have been inspected. It makes no claim of action-only distribution shift, withholding-specific rescue, Bayesian uniqueness, or independent confirmation. Historical G2 PASS and RS1 FAIL remain unchanged: 121/128 supported starts versus 122 required. No threshold is waived. This comparison requires separate launch approval.

## Fixed methods and data

- Pinned Titan UniFloral source and original BCA bridge; no CORL host, architecture change, auxiliary target revision, coefficient sweep or new simulator physics.
- One stall3 dataset from candidate v2 formal G2 attempt 1. Archive SHA256 `125131d77f54600497ce4a5d95abff885c1cefbf8ccbafd2d74fd4d52da4bdb2`; canonical content SHA256 `c2a6d61665a539ad8740d191424f90aeea9973566f3c6325226d928161fee533`. The launch manifest binds exact sources, dataset and configuration.
- CUTOFF retains executed float32 commands with abs(a0) <= float32(0.8), at all loads: 47,978 fitting rows from 768 fitting episodes. This is a constructed action-support gap; retention also changes the mixture of states. No supported-state guarantee is inferred from the mask.
- Both methods use identical CUTOFF replay, auxiliary-fitting rows and normalization. Normalization uses all retained fitting rows, not full-data or RANDOM rows. No privileged Qpi labels enter training.
- Existing calibration selection: one non-withheld row from each of the first 128 tuning episodes by group ID; outcome-independent uniform replacement within that episode when its first draw is withheld. No selection/final-calibration episodes are bound to a learner. Repeated calibration on these tuning episodes does not confer a guarantee in the withheld region or after adaptive action selection.
- The reused dataset loader validates the entire 1,280-episode archive, including protected-role transitions, against recorded dynamics. Those integrity reads are counted and are not used to fit, calibrate, select or tune either learner. This is not a claim of zero protected-transition reads or a fresh untouched-dataset confirmation.
- Three independent initialization/auxiliary seed pairs, derived by the fixed named SHA256 rule in launch.py. Native and BCA continuations share the complete 10k fork, branch replay and target-noise streams. Protected confirmation seeds are not repurposed. Config records the literal seeds and audit.
- Native actor objective is unchanged. BCA uses the original direct actor score Q1 - 3 U, with live action gradients through U and the ordinary BC term. Raw-Q normalization, twin-min smoothed critic targets, architecture and optimizers remain native. Calibration is at 10k and 15k; matched auxiliary work continues for native as in the existing runtime. This is not BCA training from step zero and not an efficiency comparison.

## Exact work and measurement plan

Three 10k warmups plus six 10k continuations produce six final 20k policies through nine training jobs: 90,000 critic updates, 45,000 actor updates, 90,000 auxiliary updates, nine calibration refreshes. No distance or BC-only training arm is included. Existing warmup distance-matching arithmetic is retained solely because the authenticated restore receipt expects it; no distance policy is trained.

Final evaluation only, separately paired within each training seed:

- 128 fresh common reset starts per native/BCA pair, current in [0.9, 1.1], original reset law otherwise, 100 steps, gamma 0.98. Total 768 return episodes / 76,800 steps. Actual deployed actions are used, with no knee projection. Save returns, complete actions and exposure by load band.
- 48 common anchors: 16 at each actual time 0, 25, 50. Reuse the v2 retained-support prefix rule and its two-world parity check; prefix work totals 7,200 steps. These are forced diagnostics, distinct from freely visited states. Fresh named streams are fixed before training; all anchors are reported.
- Three fixed complete-action candidates: a0 at edge_knee, s10 and edge_700, with a1..a3 zero. Add the policy's own complete four-action command as a fourth probe, reported separately from the common candidate ranking. For each probe, restore the exact float64 state and time, execute the first action, then follow that same policy without knee clipping through step 99. This is Qpi, not Q-star or the smoothed TD target. Total 86,400 continuation steps; no continuation-policy projection.
- Raw Q1 and Q1-3U are compared against identical own-policy Qpi labels within each model. Across methods, own-policy Qpi targets differ; return is the primary directly comparable outcome. Do not describe lower penalized values alone as better critics.

Primary: every seed's paired BCA-minus-native mean discounted return and episode-difference distribution. Report all three seed effects and their mean/range; episodes are nested in training seeds, not independent training replications. A descriptive promising screen requires a positive difference in each seed and mean gain >= 2 return units. Any miss means only no consistent practical benefit at this fixed dose/budget, not that BCA cannot work. No retuning or automatic rerun follows.

Secondary: actual withheld-action exposure, signed error and MAE, over/under/near-zero counts using tolerance 0.15, raw versus adjusted absolute-error improvement/worsening, and candidate-bank regret/beneficial/harmful ranking changes. Numerical ties within 1e-10 choose the earliest declared candidate. Report all cases, including actions for which a lower score hurts. Width calibration is not treated as true-Q coverage. No absence of exposure is repaired by selecting another region.

## Physical budget and failure handling

- 12 dataset loads (9 training + 3 evaluation), each including 128,000 validation-replay dynamics steps: **1,536,000**. This is validation of existing data, not collection; it still consumes simulator work and is charged.
- Returns 76,800 + Qpi 86,400 + prefix parity 7,200: **170,400** new evaluation steps.
- Total maximum: **1,706,400 actual simulator transitions**, counted by the existing SimulatorCounter. Training-job cap 128,000; evaluation-job cap 184,800. Unexpected additional simulator work fails closed rather than silently expanding the budget. Runtime preflight has zero simulator allowance.
- Proposed global cap **10,800 seconds**, including startup and evaluations; each training job <= 1,800 seconds, each evaluation <= 600 seconds, runtime preflight <= 120 seconds. The existing owned-process supervisor enforces child limits bounded by remaining global time, with shutdown margin. Final receipt writing can occur after a child is stopped; no new computation is dispatched beyond the deadline.
- Cost basis: the old dose-screen receipt recorded 5,649.433 seconds for twelve 10k BCA branches. Nine jobs at that rate is 70.6 minutes. New warmup/native timing and compilation are unmeasured; three hours is a proposed ceiling, not a duration promise.
- One exclusive run directory; no overwrite, retry, resume, checkpoint selection or continuation is automatic. Stop on any missing/failed job, non-finite result, bad calibration, restore mismatch, pairing mismatch, integrity-check failure or cap. Preserve partial outputs. A lost worker report makes counts unknown with its declared cap as the upper bound, never zero.

## Preparation and execution status

The small entrypoint is necessary because the old Stage 1 binds a different dataset and the dose wrapper fixes kappa 63. The concurrent native_screen targets the larger r4 program and is not imported or modified. Existing training, checkpoint, physics and supervisor implementations are imported rather than copied. Only the dataset/seed binding, selected job list and narrow final-evaluation panel are new.

Preparation runs pure unit/configuration/static checks only. No real fitting, calibration, simulator rollout or CUDA job is part of preparation. Existing successful runtime/restore evidence supports reuse; it does not validate these new bindings on CUDA. Runtime identity and exact restore checks remain mandatory in the user-run path. An unverified target-runtime integration must be disclosed, not called fully validated.

No launch approval file is included. After review and explicit approval bound to MANIFEST.json, the manual command is:

```powershell
wsl -d Ubuntu -- /home/david/bca-operator-audit-20261026/bin/python -B -u /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/minimal_comparison/launch.py --execute --approval /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/minimal_comparison/LAUNCH_APPROVAL.json
```

This command is for later review, not an instruction to start now. The task's assistant-side WSL restriction remains. Run records would be under `run/`: ATTEMPT.json, per-job progress/worker logs/receipts, and RESULT.json.
