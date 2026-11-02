# Frozen development comparison of penalty strength

This is the next usefulness test after the passing 256-row gradient audit. It keeps the same trained UniFloral checkpoint, BCA reference and native normalization. It tests first-action selection on new development starts; it does not train an actor or approve a new normalization method.

```powershell
wsl -d Ubuntu -- /home/dbayha/bca-gpu/bin/python -B -u "/mnt/c/Users/David Bayha/Documents/Codex/2026-10-22/task/unifloral_frozen_scale_comparison_v1/run_diagnostic.py" --execute --allow-outside-upstream-constraints
```

**Bounded work:** 64 fresh development episodes, one query per episode, 16 at each time 0/25/50/75; at most nine candidates per query and 576 deterministic continuations. The enforced transition cap is 60,000 and worker timeout is 600 seconds. No optimizer updates, calibration fitting/refresh or confirmation starts are allowed. Successful gradient checks do not predict rollout timing.

Every selector uses the same candidate bank: frozen actor, zero action and seven seeded uniform feasible actions. Each candidate is executed once, followed by the same frozen actor until the remaining horizon ends. Discount .98, total horizon 100, exact full simulator state/time, pre-transition reward and no terminal bootstrap are preserved from the validated evaluator. All scores and selections are saved before continuation returns.

There are 31 selector entries: raw Q1, frozen actor, twin-min Q, and four families across kappa {0, .03, .1, .3, 1, 3, 10}:

| Family | Score |
|---|---|
| Raw BCA | Q1 - kappa * U |
| Contrast BCA | Q1 - kappa * 31.372587203979492 * U |
| Contrast disagreement | Q1 - kappa * 4.926129267651156 * abs(Q1-Q2) |
| Contrast actor distance | Q1 - kappa * 19.29619572569095 * mean((a-frozen_actor(s))^2) |

All factors are frozen from the saved fitting audit. Contrast matching matches candidate-value differences, not action/parameter gradients. At this checkpoint the BCA factor simply rescales kappa. Actor distance is a geometric comparator, not behavioral support density or a trained stronger-BC method. Four kappa-zero entries duplicate raw Q; these are not 31 independent alternatives.

The primary outcome is paired frozen-continuation value gain over raw-Q selection. Candidate-bank regret, actor comparisons and paired BCA/control comparisons are secondary. All coefficient results are retained. Confidence intervals are pointwise 95% bootstrap intervals, using common resamples of independent episodes within each time stratum. They are not corrected for multiple comparisons and cannot justify a best-method significance claim.

New development episode seeds are 2026102700 through 2026102763. Previous development seeds 2026102400 through 2026102463 are excluded; confirmation seeds 2026102500 through 2026102563 remain untouched. Candidate seed is 2026102807. Bootstrap seed 2026102343 is retained to reuse the validated analysis unchanged; it resamples the new episodes, not old returns. Training/calibration seed disjointness is checked against the export.

The inference and analysis sources are reused unchanged. The local simulation copy changes only the two pinned query/candidate seed literals, recorded in SIMULATION_DIFF.json. Legacy RMS summaries may be retained as unused descriptive output; no new method uses their factors. Production scientific sources and all previous results are preserved.

No coefficient or normalizer is chosen automatically. A later development-based choice must be locked before a separate confirmation protocol. Frozen first-action reranking does not establish improved policy training, global optimality, cross-environment robustness or Q1-error coverage. GPU execution remains manual under the existing WSL tool-access restriction.
