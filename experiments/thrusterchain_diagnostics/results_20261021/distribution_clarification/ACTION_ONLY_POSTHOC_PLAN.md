# Controlled action-only diagnostic: post-hoc plan, not executed

Freeze the available Zeus **full-retention BCA replicate 01, 100k checkpoint** and a bank of source states. Its choice is availability-based, not selected for its outcomes. Change only the conditional law of the *current queried action*. This isolates action shift from state/time/context shift and from a changed critic or label target. This plan was written after seeing the frozen experiment results; it is not a prospective preregistration and does not alter the frozen experiment.

## Fixed contract

Record checkpoint/hash, training seed, observation normalizer, current critic, actor, target actor/critic, frozen scale parameters, frozen residual unit, horizon/mask convention, target-noise parameters and random streams. Use one checkpoint consistently, preferably the available final 100k checkpoint. Its stored scale/reference may originate at the 95k refresh: record that provenance, but recompute **both** compared calibration score banks using the same fixed 100k predictor and fixed 100k target kernel. Do not compare old 95k scores against new 100k scores and call the difference action shift. No fitting or optimizer update occurs.

For every state/time/regime/load row x and queried action a, define the score once:

`S(x,a,xi) = abs(Y_fixed(x,a,xi) - Q_fixed(x,a)) / (max(eta_frozen(x,a),1e-6)*unit_frozen)`.

Use the actual native critic reduction and `native_target` semantics. Its next-action target noise, including its clipping, remains a fixed conditional **label-generating kernel** shared by all arms. A clipped noise sampler is valid here: no continuous density or importance ratio is assigned to that label noise. Ratio weighting applies only to the current queried action a. Distinguish Bellman-target residuals from error against true policy Q; they are different estimands.

Select 64 physical states from 64 source episodes by episode ID and a fixed time schedule without inspecting scores, widths or oracle errors, preserving the four original context strata. Freeze their state/time/regime/load values and define the finite reference law mu explicitly, for example uniform over these 64 rows. Reusing already inspected/reserved state episodes makes this a post-hoc conditional diagnostic, not a fresh holdout performance claim. Do not assert that fixed one-row-per-state strata are IID from a common state law.

Draw a **fresh calibration bank of 128 state IDs independently from this same finite mu**, then draw each current action directly from the actual b(.|x), and draw its label noise independently. Use the exact same bank for unweighted and IW methods. Independently draw the evaluation state IDs from mu and actions from each declared q_lambda, sharing that evaluation bank between the two methods. The banks therefore differ in their independent draws, not their physical state support, state law, predictor or target kernel. Reuse common uniforms to couple action-law contrasts when convenient. Existing equal-episode retained records are not relabeled as fresh draws from b. The whole operation uses a diagnostic copy; no frozen artifact is overwritten.

## Exact categorical action laws

Actual source: `experiments/thrusterchain_diagnostics/training_data.py:21` and `:61–83`. There are 5 atoms per coordinate, `[-1,-KNEE,0,KNEE,1]`, so the joint grid A has 625 vectors. At state x the nearest controller atom has coordinate mass .84 and each other atom .04. Thus

`b(a|x) = product_k p_k(a_k|x)`, with minimum grid mass `.04^4 = 0.00000256`.

Use the actual `behavior_parameters` helper and verify the 625 masses sum to one and match recorded `action_probability` for logged actions. This is a probability-mass calculation, not a continuous density.

Let h(x) be the nearest grid vector to the **fixed learned actor** output, with the documented lower-atom tie rule. Define

`q_lambda(a|x) = (1-lambda)*b(a|x) + lambda*1[a=h(x)]`, for `lambda = 0, .5, 1`.

This is a bounded stochastic categorical actor-centered mixture, with a deterministic **grid** actor endpoint at lambda=1. It keeps positive mass on every source atom for lambda<1; the endpoint remains absolutely continuous with respect to b because b assigns positive mass to its chosen atom. Exact oracle weights are `w_lambda(x,a)=q_lambda(a|x)/b(a|x)`. Their bound is finite, `w <= 1-lambda + lambda/.04^4`, but can be large; bounded does not imply useful finite-sample ESS. At lambda=1, count positive-weight calibration draws: an empty positive-weight bank must trigger refusal, not epsilon repair or fabricated scores. The lambda=0 unit-weight reduction is mandatory. The grid projection is a specified diagnostic target, not the deployed continuous actor.

For the deterministic actor target `delta_actor(x)`, check exact grid membership under the recorded numeric representation. Off-grid actor outputs have zero source probability and must be reported as unsupported. Report the fraction of the fixed state bank that is off-grid, rather than assuming it is 100%. An actor output exactly on the grid is supported: its target weight is `1/b(actor(x)|x)` on that atom and zero elsewhere. Lack of that atom in a finite calibration sample is a sample-coverage problem, not structural zero support. If a guaranteed support-negative fixture is needed, explicitly choose an off-grid action as a synthetic negative control and label it separately from the actor. Do not silently round an off-grid actor and claim the original actor is supported.

Start with full retention. If rare/zero retention is later examined, the *retained conditional source* is `b_r(a|x)=b(a|x)*r(x,a)/sum_A b(a|x)*r(x,a)`. This formula defines a **new direct conditional sampler on the fixed mu**; it does not describe the actual frozen equal-episode retained calibration bank by fiat. Both the normalizer and structural zeros must be enumerated. Reusing b without retention correction is wrong; changing retained state frequencies would also break the action-only comparison. Such an extension is a separate declared block.

## Minimal paired computation and advisor stage map

| Stage | Calculation and invariant | Required evidence |
|---|---|---|
| 1: freeze / N1–N4 | Hash model, target, scale, unit, states, episode/time IDs and label-noise streams. All arms share them. | Freeze manifest; explicit final-model versus refresh-age provenance. |
| 2: behavior / I1 | Enumerate 625 atoms and exact b. | Unit sum, positive support and logged-mass agreement. |
| 3: target / I2–I3 | Construct q_lambda, exact q/b; detect the deterministic actor's off-grid mass. | Target unit sum, lambda=0 weights=1, exact support table. No density estimate or clipping atoms. |
| 4: calibration / N5 | Use the same freshly sampled calibration rows and scores, changing only weights and query mass. Couple posterior exponential streams. | Native unweighted versus exact-IW thresholds, empirical/posterior components, positive-weight count, ESS and infinity/refusal rate. |
| 5: query mass / I6 | Prefer existing per-query IW lookup with `t=w(x,a)`; retain common bank/query scale. | Query mass, grid-rounding gap, refusal and threshold for each queried row. If scalar population-query mass is tested, label it separately: `t=sum_x rho(x)*sum_A q(a|x)^2/b(a|x)`. It is not interchangeable with the per-query rule. |
| 6: evaluation / N6 | Evaluate independent probability-sampled state/action/label draws under the same finite mu and each declared q, shared across weighted/unweighted methods. Optionally enumerate the 625 actions on the fixed state bank as a finite-sum audit. | Tail-risk/coverage estimates under each q, half-width and conditional action errors, with draw counts and Monte Carlo uncertainty. A handpicked six-action grid is not a probability sample from b or q. |
| 7: penalty / N7–N8 | Apply the same finite dose formula and the same native neutral dose=1 refusal policy to both weighting arms. | Paired dose/width changes with support and refusal shown separately. Any max-caution fallback is a separate sensitivity arm, never confounded with IW. |
| 8: utility / N8 | Compare error alignment only for the named target. Use true-Q action disadvantage only when a separately validated fixed-policy oracle is available. | Bellman residual versus true-Q error clearly separated; grid disadvantage is relative to the stated grid, not a continuous-action optimum. No policy-improvement claim from a changed multiplier alone. |
| 9: inference / N9 | Preserve episode clusters and original context stratification in summaries. | Per-seed paired results, cluster counts, uncertainty method and untested claims. |

For each lambda, include **both** unweighted and exact-IW calibration evaluated under that same q_lambda. This yields two distinct contrasts: holding q fixed isolates reweighting; holding the calibration method fixed and changing lambda measures the effect of action shift. Use identical scored tables and common random numbers within each contrast. Neither changes the state visitation distribution. Do not use an actor trajectory bank for just one arm.

Enumerating actions gives exact finite sums over the declared action grid for a fixed realized label-noise table. It does not integrate the native Gaussian label-noise kernel exactly. Share a small, recorded noise-replicate set across arms, report Monte Carlo error separately, and add replicates only if that error obscures the paired contrast. Never advertise an exact coverage probability from a finite noise sample. Recompute calibration thresholds locally; do not overwrite trained references or frozen-run records.

## Statistical assumptions and limits

The weighting identity requires the same state/time/context law and the same conditional label kernel under source and target, plus target absolute continuity with respect to source on the queried actions. The exact categorical ratios satisfy the action-law part; they do not certify the native WBCP construction's frequentist coverage or compensate for model/label drift.

Calibration and final evaluation use independent *new draw IDs* from the same fixed finite mu; the physical state bank is intentionally shared. Conditional on that bank and all frozen networks, independent state-ID, action and label-noise draws satisfy the common-source-law sampling premise; a deterministic balanced one-per-state design does not automatically do so. This is distinct from a claim about a fresh population of source episodes. Any choice based on residuals, errors, widths or prior evaluation belongs to a separate adaptation set. Lock lambda values and reporting rules before executing this newly planned diagnostic, while retaining its post-hoc status. No per-state certificate is claimed from 128 calibration draws.

For the primary fixed-bank estimand, report repeated independent calibration/evaluation draw realizations or conditional Monte Carlo errors; do not call a stratified fixed-bank description a population confidence interval. If extending the estimand to new source episodes, use separate adaptation/calibration/evaluation episode sets and justify their common stratum/time law first. Treat source episodes as clusters, with context-stratified resampling to match the balanced design. Multiple actions, times or label-noise draws from one episode are not additional independent episodes. Pair the same episode resamples across all laws/methods and recompute thresholds inside each replicate when estimating the full procedure. Such a bootstrap is approximate and may be unstable at infinity/refusal boundaries, so always report raw counts and paired changes too. A conditional-on-model episode interval does not cover training-seed variation. One Zeus full-retention seed supports no across-seed generalization claim.

No policy-occupancy shift is tested: the fixed bank intentionally retains its source-state distribution. A rollout under the actor would change states and requires a separate occupancy-aware diagnostic. Existing full-versus-query comparisons can change state/time sampling, conditional action law, and predictor/target age simultaneously; they cannot identify the action-only effect this plan isolates.

Status: plan only. No training, source edits, calibration reruns, or new performance results were produced for this addition.
