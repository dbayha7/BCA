# Source and evidence review: decision signals versus residual width

Read-only review of the completed knee/stall confirmation, performed November 2, 2026 EST. This review concerns existing saved evidence only. It does not change an algorithm, train a model, evaluate a model, simulate, calibrate, or alter Path A.

## Main source finding

The actual uncertainty objective is not a signed-error detector, a ranking-error detector, or a pointwise absolute-residual regression. It fits **a weighted minibatch average of symmetric soft coverage**, regularized toward smaller local widths. Its center is the current twin-min critic and its target is a smoothed-target twin-min one-step TD target. The deployed actor instead uses Q1 minus kappa times the calibrated width. This is a concrete mismatch between the fitted quantity and the desired decision quantity. It is not, by itself, proof that this mismatch caused the observed policy gains or costs.

For one saved state and withheld candidate versus its logged alternative, the relevant quantities are:

- True advantage `g = Qpi(withheld) - Qpi(logged)`.
- Predicted margin `m = Q1(withheld) - Q1(logged)`.
- Relative ranking error `e = m - g = error(withheld) - error(logged)`.
- Penalty contrast `p = kappa * (U(withheld) - U(logged))`.
- Adjusted margin `m - p`.

A common additive critic bias cancels from `m` and `e`. Harmful preference can exist when both absolute errors are negative, and a large positive shared bias can coexist with correct ranking. A downward width can remove a wrong positive margin, miss it, or reverse a useful positive margin. These are distinct from reducing mean signed Q error.

## Actual implemented quantities and caller settings

Paths below are relative to `experiments/thrusterchain_bca/`.

| Component | Exact source and relevant lines | Verified behavior |
|---|---|---|
| Actor objective | `titan_20261026/payload/unifloral_td3bc_validation_v1/candidate.py:72` through 123 | Native `-lambda*mean(Q1) + mean(BC)` plus `lambda*kappa*mean(U)`; lambda is stopped `2.5/(mean(abs(Q1))+1e-7)`. Reference parameters are stopped, action gradients remain live. BC definition and coefficient are unchanged. |
| Critic target/loss | Same `candidate.py:129` through 155 | Native target action smoothing, target twin-min, termination mask and gamma; squared error of both heads against the same target. No BCA penalty is added to this critic loss or target. |
| Uncertainty TD target | `.../unifloral_bca_pipeline_validation_v1/uncertainty_source/historical/td3_bc_bca.py:146` | `y = reward + (1-done)*gamma*min(target critic(next_obs, clipped target actor + clipped Gaussian noise))`; target is stopped. Gamma .98, policy noise .2, clipping .5. |
| Scale fitting | Same `td3_bc_bca.py:159` through 231 | Current twin-min center, stopped normalized exponential minibatch weights. Loss is `(sum(w*soft_coverage(y/unit,Qmin/unit,eta))-0.9)^2 + .005*sum(w*eta^2)`. EMA unit is `.99*old + .01*max(std(y-Qmin),1e-6)`. Learning rate .001. |
| Scale inputs/positivity | `.../uncertainty_source/calibration/network.py:8` through 41 | Normalized state plus action; two 64-unit ReLU layers; `softplus(log_eta)+.01`. State normalization epsilon .001. Soft coverage is product of two sigmoids with steepness 20. |
| Refresh/calibration | `.../historical/td3_bc_bca.py:234`; `.../calibration/reference.py:153` through 197 | Freeze scale network and unit; held-out scores are `abs(y-Qmin)/(max(eta,1e-6)*unit)`. No Qpi oracle values enter. |
| Bayesian threshold | `.../calibration/reference.py:18`; `.../calibration/wbcp.py:57` through 117 | Uniform calibration weights plus test atom; alpha .1, credibility .95, 1,000 exponential-weight posterior draws. Threshold is max(empirical selection, posterior credible selection). These parameters do not promise conditional support-gap Qpi coverage. |
| Width consumed | `.../historical/direct_extension.py:45` through 66 | `U = frozen_threshold * max(eta,1e-6) * frozen_unit`, with stopped reference parameters and live action derivative. Numerical usability does not certify support or statistical validity at a query. |
| Exact component import | `.../unifloral_bca_pipeline_validation_v1/bca_bridge.py:24`, 59, 219, 265 | Uses selected original AST definitions and authenticated calibration modules. It does not import or execute the historical BC-modulating host update. Calibrator is constructed state-dependent; caller forwards native discount/noise settings. |
| Runtime contract | `zeus_20261026_validation/central_quartile_training_v1_evaluation_rerun1/runtime.py:137` through 146 | Gamma .98, batch 256; explicitly asserts `.001,20,.005,.99,.1,.95,1000` uncertainty settings. |
| Saved probe measurement | `.../knee_s10_candidate_v2/overestimation_v1/oe_eval.py:31`, 54, 69, 81 | 192 retained fitting rows with source IDs; each logged action plus same-sign magnitudes .85/.9/1 with other coordinates unchanged. Forced first action followed by each model's own unclipped policy, real state/time and finite horizon. |

The scalar calibrated threshold and residual unit multiply all action scales at a checkpoint. Their positive product does not change within-checkpoint rank/AUROC ordering. It can change whether `kappa*deltaU` overwhelms a predicted Q margin. No saved-data ranking result can establish a uniquely Bayesian advantage.

## What information is and is not available

**Available before choosing an action:** observation including remaining time, candidate action coordinates, Q1/Q2, fitted U, known cutoff mask/distance, and quantities derived from these such as margins, delta U, and Q1-minus-twin-min head excess. These can be studied without oracle leakage. The actual actor does not enumerate the four diagnostic bank candidates; bank argmax is a retrospective readout, not deployed control.

**Observed after a logged transition:** reward, successor, termination flag and the corresponding one-step TD residual after evaluating the relevant target networks. The historical fitting/calibration machinery uses those transitions. A current decision cannot observe its unexecuted alternatives' actual reward/successor or residual. No alternative-action transition was fabricated here.

**Evaluator-only oracle:** each candidate's saved own-policy Qpi, true advantage, regret, useful/harmful label and ranking error. These identify what happened in the saved banks but cannot serve as decision-time inputs or training labels in this analysis.

The 24 evaluation reports persist source row IDs, times, complete actions, Q1, Q2, U and Qpi. They do **not** persist candidate-level TD targets/residuals, eta, residual unit, calibration radius or candidate-action gradients. Consequently the current arrays cannot identify whether the local network, the frozen scalar, or particular training residuals produced a given bad margin. Full-state checkpoints were not decoded or loaded into a model for this review; their raw file hashes were checked for the limited feasibility question below. No missing model values were regenerated.

Training reports retain final scalar auxiliary loss and scalar objective terms. Separate 15k/20k `gradient_audit_*.json` files retain Q/BC/actual-penalty action gradients on 768 training-only probes and parameter-gradient norms/cosines, with source row IDs. They are actor-action gradients on a different probe bank, not the 192 four-action banks with Qpi labels. They do not directly supply the missing oracle gradient or candidate-local residual-to-harm relationship. The saved `prospective_bca3` gradient field is shadow information; the actual penalty fields and `actual_kappa` identify the applied dose.

## Numerical near-tie sensitivity

Independently recomputed from saved own-probe Qpi and the matched rollout starts:

| Model | Start index | Own-probe Qpi | Rollout return | Absolute difference | Relative difference |
|---|---:|---:|---:|---:|---:|
| BCA 21, stall, seed 0 | 1 | -312.31137759091143 | -312.3117253811694 | 0.00034779025799025476 | 1.1135997457853514e-6 |
| Native, stall, seed 2 | 29 | -352.8606675450834 | -352.86001021772285 | 0.000657327360556792 | 1.8628559245101355e-6 |

Both exceed the original relative 1e-6 I1 rule. The rule and INCONCLUSIVE result stay unchanged. The maximum observed absolute discrepancy, epsilon = 0.000657327360556792, and twice it, 0.001314654721113584, are defensible **descriptive sensitivity levels**, not established error bounds on every candidate Qpi. These differences concern rollout-versus-forced-action t0 measurements and small actor batching differences; no new execution establishes their cause. A tolerance band can show whether a conclusion depends on tiny gaps without retroactively repairing I1 or declaring robust truth outside the band.

## Source authentication

All ten sources below match exact hashes in the recorded completed attempt-3 `bca_branch_s0_k189_nostall/job.json` integrity map. That job record's SHA256 is `08f60ca590bac1999c22e80a592d5271c150073945d761f1b35f5f5cc62f92b3`. Hash checking was read-only; scientific modules were not imported. This authenticates the cited source snapshot and caller bindings, not universal closure of every historical dependency.

| Source suffix | SHA256 |
|---|---|
| `unifloral_td3bc_validation_v1/candidate.py` | `e0af0aed323f0b8d33dc56d9308f862b4c7a231508d747213308022ef8801ce6` |
| `unifloral_bca_pipeline_validation_v1/bca_bridge.py` | `9137487fad717dab4fb83445cb28798fe1a68ba501000049ba15a655227c0268` |
| `uncertainty_source/historical/td3_bc_bca.py` | `3f8452df67b1ac51c78b67c36a6f6269c849aa9746f5e15df771d553fce0536e` |
| `uncertainty_source/historical/direct_extension.py` | `6af068817b3e3ca8b3c183d8a203ad88c44d3f18ab19046ba03e0f8894f1f462` |
| `uncertainty_source/calibration/reference.py` | `f26273784c458f058c9ebd9b4ebe1e3e7e20c1fc05abfc3297aead724e3b6b1b` |
| `uncertainty_source/calibration/network.py` | `cfde14313cc13cfe96225234c81a8546121ff50f3daf789717ac14c7720f354c` |
| `uncertainty_source/calibration/wbcp.py` | `fdd36b42a1d2c440cfa5e4df4b59c421c8fca477f8b3ffc5e7d9f095e889c023` |
| `central_quartile_training_v1_evaluation_rerun1/runtime.py` | `e17a33d00847e3f989fe1ed01c64644a194f007f295805b05a1b95198bac1c2c` |
| `knee_s10_candidate_v2/confirmation_v1/cf_worker.py` | `24ae1db86a4949577c9df5abc4a00e9f3246c63be0f1ecb8f9c992b965554340` |
| `knee_s10_candidate_v2/overestimation_v1/oe_eval.py` | `8c2a2b0ee6217417ca32d4a1750cf0189cfd12df25591116ade144e4cdf738cd` |

## Interpretation constraints for the new calculations

Hold complete state, action pair and continuation policy fixed when interpreting a margin. Within-world bank inputs match across models, but own-policy continuations differ across trained models. Cross-world matching row IDs do not match physical states. Do not pool reused states or three action alternatives as independent training replications. Show each of three seeds, each strength and sparse mixed-label counts. World separability is not action selectivity within a world. Saved support labels describe the constructed a0 gap, not full joint support.

The added head diagnostic is limited and justified: `h=Q1-min(Q1,Q2)` and its withheld-minus-logged contrast quantify exactly the center mismatch present in source. Comparing Q1 and twin-min relative margin errors is not a new policy arm or an endorsement of replacing Q1. It cannot establish the TD residual that trained U, because targets and pointwise residuals were not saved for these candidates.

Skills actually read and applied: `.agents/skills/bca-code-and-calibration-audit/SKILL.md`, `.agents/skills/offline-rl-evidence-and-visualization/SKILL.md`, and `.agents/skills/offline-rl-experiment-design/SKILL.md`. Applied their requirements for actual operator tracing, role/estimand separation, source authentication, action-level penalty contrasts, independent units and unchanged historical verdicts.

No algorithm change is justified solely by source mismatch. The smallest prospective test should be selected only after the saved-margin results: hold an existing critic and complete probe inputs fixed, compare the specifically implicated signal or mapping at matched strength, use independent evaluation evidence, and leave BC unchanged. Any new measurements or training require separate approval.

## Independent numerical verification of the final implementation

Reviewed `analyze.py`, its fixtures and canonical `output/run2` results. A separate NumPy calculation loaded all 24 original report arrays directly, without calling the builder's metric functions. It verified **1,584 comparisons**: exact/epsilon/2-epsilon class counts and wrong-preference/correction counts, Q1/twin-min/adjusted relative MAE, four pooled AUROCs, four within-bank AUROCs and their eligible-bank counts. AUROC was independently computed by explicit positive-negative pair comparisons with half credit for ties, rather than the implementation's rank-sum formula. Relative MAE matched exactly; all AUCs agreed within 1e-13. Every recorded original input and final output hash matched.

Confirmed native totals across three seeds (1,728 candidate-versus-logged pairs per world, nested within 192 reused states):

| World | Harmful / useful pairs | Harmful pairs raw Q prefers | Useful pairs raw Q misses | Wrong harmful preferences without positive absolute withheld-Q error |
|---|---:|---:|---:|---:|
| Stall | 1,212 / 516 | 510 | 353 | 117 / 510 |
| No-stall | 1,119 / 609 | 75 | 109 | 38 / 75 |

The discrepancy from earlier 197/34 and 28/171 counts is intentional: those earlier counts considered only the **one raw-Q argmax action in each four-action bank**. This follow-up counts all three withheld-versus-logged pairs. Do not interchange denominators or interpret all 510 false pair preferences as 510 deployed decisions.

Native Q1 relative MAE means are approximately 2.12546 stall and .33645 no-stall; twin-min gives 2.12064 and .34421. The head mismatch therefore does not, in these saved native banks, explain away the relative-value error. Native delta-U pooled harm AUCs are .47356/.48884/.48469 in stall and .81329/.81290/.75092 in no-stall. This is stronger no-stall discrimination than **absolute U** showed in the preceding analysis because subtracting the same-bank logged width changes cross-state comparisons. Within a bank this subtraction is a constant, so U and delta U have exactly the same within-bank ordering. That distinction prevents an apparent contradiction or a claim of reliable fine-grained no-stall choice from its very few mixed-label banks.

The implementation keeps exact historical labels, reports the two numerical sensitivity levels separately, uses declared signal orientation, exposes sparse/one-class bank counts, and changes no original criterion. No remaining numerical blocker was found in this focused review.

## Limited feasibility of a proposed residual measurement

All six native branch receipts contain both 15k and 20k full-state checkpoints. All twelve files exist, are 10,350,815 bytes each, and match their recorded SHA256 hashes. Stall branches are reused from confirmation attempt 2; no-stall branches are in attempt 3. All six final receipts record `last_refresh_critic=15000`. Existing archives hold the exact source full states, times, rewards and logged successors.

This supports proposing a later **measurement**, not a new training matrix: at most two worlds times 192 banks times three withheld actions = 1,152 distinct new one-step successors; logged successors can be reused. Six native models times 192 banks times four actions = 4,608 model-row target evaluations per checkpoint level. These are prospective counts, not consumed work or an approved runtime allowance. No new simulator/model calls occurred.

Because final U uses the 15k frozen reference, a 20k TD-residual diagnostic alone would measure the current critic's residual, not the residual operator contemporaneous with the frozen U. The 15k checkpoints make the latter closer in time, but a newly declared smoothing key still creates a new residual measurement rather than reconstructing the precise historical minibatch or refresh noise unless RNG lineage is separately established. Measuring both levels doubles model-row work, not the distinct physical successors. Neither measurement automatically becomes independent confirmation, since these candidate banks have already been inspected. No algorithm revision is selected or implemented by this feasibility finding.
