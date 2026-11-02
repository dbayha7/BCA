# Frozen UniFloral usefulness diagnostic: prepared, not executed

No eligible trained UniFloral ThrusterChain checkpoint was found in the bounded inventory. The two available local UniFloral snapshots come from tiny synthetic validation fixtures. The paused million-step pilot and older runs use CORL. None is substituted here. The inventory covers the named project and validation folders, not every file or unopened archive on Titan.

This folder prepares an evaluator and a proposed inference-export contract. It contains no trained weights, trainer, checkpoint exporter, generated evaluation data or usefulness result. Scientific runtime execution has not been performed. Syntax and source-manifest checks are recorded separately in `PREPARATION_RECEIPT.json`. The existing scientific sources, validation reports and paused pilot are preserved.

## Question and exact value target

Does a frozen TD-residual-width penalty identify optimistic Q1 errors and improve first-action selection within a fixed candidate bank? For saved full state x at time t and candidate a, the evaluator computes

`Q^pi_t(x,a) = r(x,a) + sum_{j=1}^{99-t} 0.98^j r(x_j, pi(o_j))`.

The candidate is executed first; every later action comes from the same frozen current UniFloral actor. There is no target-policy smoothing, target actor substitution or terminal bootstrap. The environment reward uses the pre-transition state and commanded action. States retain all 13 physical float64 values plus explicit time; actor observations contain the float32 physical state and remaining-time coordinate. Actions are deployed as float32 commands in [-1,1]. No observation-only reconstruction or reset-at-zero substitution is used.

The authenticated ThrusterChain dynamics and frozen actor are deterministic. Each candidate therefore receives one deterministic continuation; conditional Monte Carlo sampling error is zero. Floating-point error is not certified. The predeclared numerical tie band makes unresolved near ties inconclusive. A stochastic policy or simulator is outside this protocol and must not be silently assigned the same zero-error claim.

The native critic was trained with a lagged, smoothed TD target; its comparison here is explicitly with the unsmoothed current actor's finite-horizon return. That is an intentional evaluation estimand, not a claim those two targets coincide during training.

## Fixed development scope

- 64 independent recorded-family episodes, one query per episode at times 0, 25, 50 or 75; frozen-actor prefixes.
- Up to nine feasible candidates per query: the frozen actor, zero command and seven seeded uniform actions. Stable deduplication follows float32 deployment. All selectors use exactly the same bank.
- At most 576 continuations and 38,504 expected transition rows: 2,400 prefixes, at most 36,000 candidate continuations and 104 evaluator consistency-check rows. Hard cap: 60,000 transitions and a 600-second child-process timeout.
- Zero actor, critic or auxiliary optimizer updates. No reference refresh, fitting or dataset generation. Reserved confirmation starts are not generated.
- The fixed kappa grid is 0, 0.03, 0.1, 0.3, 1, 3 and 10. Every row is reported; the program selects no best coefficient, checkpoint or method.

`protocol.json` fixes seeds, task settings and budgets before labels exist. Runtime inference uses the actual pinned UniFloral actor/critics and actual BCA frozen-width implementation. The learner and calibrator receive no simulator object or labels. The evaluator writes and hashes every score and action selection before computing candidate-return labels. The cooperative inference guard and unchanged-state digests provide implementation checks, not a security-isolation guarantee.

## Comparators and interpretation

The selectors are actor action, raw Q1, twin-min Q, Q1 minus kappa times U, raw disagreement, and strength-matched disagreement/actor-distance controls. Disagreement means `abs(Q1-Q2)`. Actor distance means the mean squared coordinate distance from the frozen actor action; it measures geometry, not support density.

All strength matching uses a frozen bank of 256 observed fitting-only state/action rows, disjoint from calibration groups and without simulator labels. Matched control penalties multiply disagreement or actor distance by `RMS(U)/RMS(control)`. Equal numerical kappa alone does not imply fair magnitude across different units. Both raw and matched disagreement are retained. A zero control RMS is unavailable rather than patched with an epsilon. A valid finite zero-U signal is reported as degenerate, distinct from invalid calibration.

Reports contain raw Q1/Q2, U, applied penalties, penalty-to-Q ratios, first-action gains, candidate-bank regret, and pooled and within-state error/optimism associations. Pooled relationships may be confounded by state difficulty. Candidate-bank regret is not global action-space regret. The 2,000 bootstrap draws resample independent episode groups within each fixed query-time stratum, retaining 16 episodes at each of four times, and provide pointwise 95% intervals conditional on one frozen checkpoint. They are exploratory, not simultaneous confidence guarantees over 31 selectors.

U is a twin-min one-step TD-residual proxy. It is not calibrated Q1-error coverage. Unready, nonfinite or unusable ready calibration blocks evaluation; it is never relabeled as successful zero-penalty BCA. Legacy scientific source remains unchanged. Frozen first-action gains cannot establish that training the actor with the penalty improves a learned policy.

Fitting, calibration, development coefficient selection and untouched confirmation remain separate. This stage only reports the fixed development grid. Any later coefficient choice uses development data, followed by a newly frozen confirmation protocol; no tuning against confirmation labels is permitted.

## Inputs, output and commands

Read `CHECKPOINT_CONTRACT.md` for the proposed numeric inference export and independently reviewed training evidence required. Hashes establish byte identity and receipt consistency, not the truth of training declarations. There is currently no compatible exporter or eligible checkpoint, so a meaningful trained-policy run is blocked.

The default command is a stdlib-only provenance preflight. It does not import JAX, run a model or simulate transitions. With the currently absent checkpoint it is expected to return `blocked_or_failed` and name that exact missing input. There is no need to rerun it to reconfirm the inventory.

```powershell
wsl -d Ubuntu -- /home/dbayha/bca-gpu/bin/python -B -u "/mnt/c/Users/David Bayha/Documents/Codex/2026-10-22/task/unifloral_frozen_usefulness_v1/run_diagnostic.py" --preflight
```

Only after a separately approved checkpoint-generation stage and independent receipt review, the future evaluation command is:

```powershell
wsl -d Ubuntu -- /home/dbayha/bca-gpu/bin/python -B -u "/mnt/c/Users/David Bayha/Documents/Codex/2026-10-22/task/unifloral_frozen_usefulness_v1/run_diagnostic.py" --checkpoint-manifest "/absolute/Linux/path/to/reviewed/export/manifest.json" --allow-outside-upstream-constraints --execute
```

The supervisor launches a fresh child with `JAX_PLATFORMS=cuda`, `JAX_DEFAULT_MATMUL_PRECISION=highest`, `XLA_FLAGS=--xla_gpu_triton_gemm_any=True`, unbuffered logs and preallocation disabled. Conflicting existing XLA flags are rejected. The validated provisional JAX/NumPy environment exceeds native upstream bounds; the explicit acknowledgement does not turn it into an upstream-conforming runtime. No package installation is performed.

Each invocation creates a new result directory. `report.json` records preflight/child status and exit information; `diagnostic.json` records runtime, input identities, accounting and scientific analysis; `prelabel_scores_and_selections.json` seals decisions; `queries_and_actions.npz` preserves exact states/time/actions; `conditional_values.json` preserves deterministic labels and transition counts. Failed simulator evaluations retain partial accounting where available. Originals are never overwritten.

`CHECKPOINT_GENERATION_PROPOSAL.md` scopes the missing next stage. It is a proposal, not authorization or a launch command. WSL execution remains unavailable to this agent; the commands above are for the user after the appropriate input exists.
