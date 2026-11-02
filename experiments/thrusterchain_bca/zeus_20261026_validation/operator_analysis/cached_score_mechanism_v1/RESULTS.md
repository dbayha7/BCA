# Cached target/action-scoring mechanism test

**The fixed comparison passed its exploratory stopping screen in both development batches.** Selecting by the mean native target improved mean saved continuation value over Q1 by **0.2588780890** in development 1 and **0.3268168644** in development 2. The corresponding numerical allowances were 0.0000595154 and 0.0000507262. This is a development mechanism result, not statistical superiority, learned-policy benefit or OOD validation.

The comparison was executed once at the user's authorization, using saved arrays on the native CPU environment. It incurred **zero new evaluator transitions, rollouts, model forwards, GPU calls, fits, optimizer/calibration updates, candidate actions or confirmation reads**. Original UniFloral source, checkpoint, normalization and BCA remained unchanged. No proposed pilot was executed; full training remains paused.

## The Q penalty is still the intended mechanism

Original BCA's candidate score is `Q1 - kappa * U`. Its actor loss is:

```text
c = stop_gradient(alpha / (mean(abs(Q1)) + 1e-7))
actor_loss = -c * mean(Q1) + mean(BC) + c * kappa * mean(U)
```

The coefficient c is computed from unpenalized Q1. The U term retains its action gradient. This lowers the actor's Q score; it does not multiply the BC term, rewrite stored critic predictions, or subtract U from the native critic training target. Both critic heads still fit their native smoothed twin-target-min target. The transferred checkpoint was trained with kappa zero.

This particular diagnostic compared four **control scores**, with no new nonzero kappa applied: Q1, current twin-min, the fixed actor action and the four-draw mean native target. The last is a privileged mechanism control because it requires evaluator-derived reward and next-state information. It is not a silent replacement of BCA and cannot be deployed as the learned policy score.

## Frozen procedure and complete result

The [protocol](protocol.json) and [freeze receipt](FREEZE.json) were written before the new selected-action outcomes. They fix the existing 128 states by nine candidates, four recorded target draws, both batches, all query times and all 15 strata. Audited `q_current` and `native_y` were converted from float32 to float64; the latter was averaged in its original draw order. There was no recomputation of networks or targets.

For each score, candidates within `1e-6 + 1e-6 * abs(max_score)` of the maximum were tied and the first original index was selected. No score recentering, candidate reordering, V-based tie breaking or ambiguous-state exclusion was allowed. The actor control was candidate 0, checked against the saved actor action. All selection records were [sealed](SELECTIONS_SEALED.json) before decoding saved V for this comparison. Earlier development diagnostics were already known, so this sealing does not create a fresh blind experiment.

All means below use equal state weights. Values are discounted-return units; higher, including less negative, is better. Regret is the best of the nine saved candidate values minus the selected value.

| Fixed selector | Mean selected V | Mean gain vs Q1 | Mean regret | Switches vs Q1 | Helped / harmed |
|---|---:|---:|---:|---:|---:|
| Q1 | -54.027583 | 0 | 1.109267 | 0 | 0 / 0 |
| Current twin-min | -53.997925 | +0.029658 | 1.079609 | 25 | 13 / 12 |
| Frozen actor | -54.185204 | -0.157621 | 1.266888 | 54 | 21 / 33 |
| Mean native target | -53.734736 | +0.292847 | 0.816420 | 59 | 38 / 21 |

The target control improved 38 states and harmed 21, leaving 69 unchanged. Harmful switches were **16.40625% of all states**, or **35.59322% of switches**. Their mean loss was **0.5683467690**, and the largest loss was **1.8903505334**; the largest gain was **9.1293940811**. No switches were numerically inconclusive and there were no score ties. A positive mean is therefore not uniform improvement or a safety guarantee.

Twin-min alone gave +0.0622307004 in batch 1 and -0.0029154946 in batch 2. It was a fixed descriptive control, not promoted as a winning repair. The actor control is the original actor action, not a tuned actor-distance penalty.

All prespecified target-control strata are shown below. The [CSV](all_prespecified_strata.csv) and [report](report.json) preserve all 60 stratum/method summaries, including the other controls. The [per-state outcomes](STATE_OUTCOMES.json) preserve all candidate values and selected indices.

| Stratum | States | Gain vs Q1 | Gain vs actor | Regret | Switches | Helped | Harmed |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 128 | +0.292847 | +0.450468 | 0.816420 | 59 | 38 | 21 |
| development_1 | 64 | +0.258878 | +0.491806 | 0.852722 | 29 | 19 | 10 |
| development_2 | 64 | +0.326817 | +0.409131 | 0.780117 | 30 | 19 | 11 |
| time0 | 32 | +0.155021 | +0.344284 | 0.516780 | 11 | 7 | 4 |
| time25 | 32 | +0.393790 | +0.441396 | 1.064855 | 18 | 14 | 4 |
| time50 | 32 | +0.156618 | +0.076367 | 0.985582 | 12 | 8 | 4 |
| time75 | 32 | +0.465962 | +0.939827 | 0.698461 | 18 | 9 | 9 |
| development_1/time0 | 16 | +0.218929 | +0.434750 | 0.579600 | 3 | 3 | 0 |
| development_1/time25 | 16 | +0.433562 | +0.388812 | 1.287929 | 10 | 7 | 3 |
| development_1/time50 | 16 | +0.168238 | -0.105391 | 0.942296 | 8 | 5 | 3 |
| development_1/time75 | 16 | +0.214783 | +1.249055 | 0.601065 | 8 | 4 | 4 |
| development_2/time0 | 16 | +0.091112 | +0.253819 | 0.453959 | 8 | 4 | 4 |
| development_2/time25 | 16 | +0.354017 | +0.493979 | 0.841782 | 8 | 7 | 1 |
| development_2/time50 | 16 | +0.144998 | +0.258125 | 1.028869 | 4 | 3 | 1 |
| development_2/time75 | 16 | +0.717141 | +0.630600 | 0.795857 | 10 | 5 | 5 |

All eight batch/time cells have positive mean gain versus Q1, but development 1 / time 50 loses 0.105391 on average to the actor control. Small cells and reused development data do not support a universal or confirmatory claim.

## What this changes, and what it does not

The earlier operator audit found weak within-state radius tracking of absolute residuals, `corr(U, abs(R)) = 0.068565`, and nearly zero association of absolute residual with signed Q1 error, `corr(abs(R), E) = 0.008895` (0.082144 with positive E). Those findings remain unchanged. The new test answers a narrower decision question: the fixed native-target score selects higher-value actions on average in both saved batches. A useful control score does not prove that U predicts the needed correction or that penalizing U will learn a better policy.

Algebraically the target score is `mean(y) = Q1 - (H + mean(R))`. That signed correction can lower **or raise** a candidate's score relative to Q1. Its benefit cannot be attributed solely to a nonnegative penalty, target smoothing, head choice or any one component. The four-draw mean is a finite noise average, not the exact expected target. Saved V is a deterministic continuation; there is no conditional return-sampling Monte Carlo noise, but numerical, population and training-seed uncertainties remain.

No fresh confirmation, independent training seed, policy optimization or OOD test was performed. The original locked BCA conclusion and full-training hold remain unchanged. This result supports one focused deployable-candidate proposal; it does not require every offline predictor to succeed before a separately authorized bounded policy pilot.

## Historical Q-penalty proposal — superseded, never executed

**Scope update:** the user subsequently required preserving BCA's uncertainty modeling and meaningful Bayesian/conformal calibration. The generic supervised correction proposal below is withdrawn. The separately authorized [lower-only feasibility check](../cached_lower_only_v1/RESULTS.md) passed, and its report contains the current BCA-consistent target/calibration proposal. This historical section records the earlier idea only; it is not the recommended next action.

Propose a separate **penalty for positive one-step Q excess**, leaving original BCA intact. This label describes excess above the native target, not true harmful optimism relative to V:

```text
training target z(s,a) = max(Q1(s,a) - mean_native_y(s,a), 0)
deployable P_psi(s,a) >= 0 takes only state and action
candidate/actor score = Q1(s,a) - P_psi(s,a)
```

Use fixed coefficient one because z is already in Q units. Preserve the native critic target, original normalization and BC term. If later integrated into an actor, freeze the predictor's parameters while retaining the live action gradient through P. This is a new penalty target, not a calibration fix or a renamed original radius. At deployment it needs no reward, next-state evaluation or continuation V.

The smallest proposed pilot is two batch-swapped fits: fit P on the 576 cached state/action/target rows of one development batch and evaluate its action choices on the other 64 states, then swap. Keep each state's nine candidates together. V is used only for evaluation, never as a fitting label. Freeze one architecture, optimizer/update cap and seed before fitting; reuse the existing auxiliary implementation where appropriate and run no hyperparameter or penalty-strength search. Judge direct saved-value gain, regret and harmful switches with the same two-batch stopping screen, not whether every correlation is positive. If either held-out batch fails the direct-utility screen, stop this proposed penalty branch.

Even with perfect prediction, `Q1 - max(Q1 - mean(y), 0) = min(Q1, mean(y))`, not the unrestricted mean-target score. The passed control may rely on upward corrections as well as downward penalties, so the clipped nonnegative candidate is **not validated by this result**. Fitting could also fail to reproduce useful action differences. The proposal remains development-only and requires separate authorization and a fixed fitting budget; no fitting, candidate implementation or pilot execution has begun.

## Verification and reproduction

Seven synthetic selection/tolerance fixtures passed before execution. Independent saved-output verification passed **9,790 numeric comparisons**, five array descriptor hashes, all 512 method/state records and all 60 stratum/method summaries, including CSV, regret, harmful switches and both stopping screens. Input/output hashes and seal chronology matched. The original execution script was not rerun for this review. See [INDEPENDENT_REVIEW.json](INDEPENDENT_REVIEW.json).

Protocol SHA-256: `287c0b0dd050fa87d20ec17a36004f976d3434b7a6f617278e0b228a911f116c`.

Execution script SHA-256: `8e0d61ccb2b06ac7357b8ed8db39e27d5f3d9273a49f0085a172e0d4ee0c411a`.

Result SHA-256: `f6005b3428575182dd21be17a2dce154b253704271df67539473ed5c51d48dde`.

To verify the saved result from the repository root:

```powershell
.\.venv\Scripts\python.exe -B experiments/thrusterchain_bca/zeus_20261026_validation/operator_analysis/cached_score_mechanism_v1/verify_cached_comparison.py
```

Do not rerun `run_cached_comparison.py --execute` or remove its permanent attempt evidence. No existing source or model was edited and nothing was committed or pushed. Assistant WSL/process-inspection denials were respected; this task needed only permitted local files and CPU arithmetic. There is no uncertain execution outcome or remaining blocker to this completed comparison.
