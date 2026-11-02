# Central-quartile BCA training and evaluation

This development study found that the BCA-adjusted score improved the prespecified Q-accuracy comparison in all three seeds. It did **not** pass the prespecified policy-return screen or either rescue screen. In the strong-current hole condition, the BCA-trained policy outperformed the native policy in all three seeds, but it did not outperform the distance-penalty control in every seed. These results describe one fixed dataset and removal mask; they are not a general efficacy or confirmation claim.

The [figure guide](FIGURE_GUIDE.md) links all plots and explains their units and scope. The underlying [evaluation report](../../experiments/thrusterchain_bca/zeus_20261026_validation/central_quartile_training_v1_evaluation_rerun1/run/evaluation/report.json), [terminal receipt](../../experiments/thrusterchain_bca/zeus_20261026_validation/central_quartile_training_v1_evaluation_rerun1/RUN_RESULT.json), and [frozen protocol](../../experiments/thrusterchain_bca/zeus_20261026_validation/central_quartile_training_v1_evaluation_rerun1/PROTOCOL.md) define the primary evidence and decision rules.

## Design and completed training

For each of three independent training seeds and three data conditions, a fresh model received 10,000 critic updates, followed by a complete-state fork into native, BCA and distance-penalty branches, each receiving another 10,000 critic updates. All nine warmups and 27 final branches completed, and 36 scheduled calibrations were recorded as accepted. The distance coefficient was matched at the fork to BCA's logged-action penalty-gradient RMS; this is a strength control, not a Bayesian-specific ablation.

The full-data replay had 76,800 rows. The hole and random controls each retained 67,108; the hole withheld original logged tuples satisfying `|a[0]| <= 0.25` and `obs[10] + obs[11] >= 0.4`. Random removal matched counts within episode, ten-step time bin and high/low load, but not every state, reward or successor distribution. This is conditional label withholding, not a claim that the behavior policy never visited those states. Calibrations used a common retained-support set and were reused adaptively.

Evaluation used 32 shared starts per training seed and context, the fixed final 20k checkpoint, and discount 0.98. Strong current changes the observed initial-current distribution from `[0.1, 0.7]` to `[0.7, 1.1]`; simulator dynamics remain unchanged. Returns are dimensionless, discounted, and higher is better. The same starts are paired across conditions and methods, but three training seeds—not the episode, action or checkpoint counts—are the independent replications.

## Policy return

Equal-seed final mean discounted returns are:

| Context | Data condition | Native | BCA | Distance control |
|---|---|---:|---:|---:|
| Ordinary | Full | -77.90 | -75.59 | -77.70 |
| Ordinary | Hole | -82.92 | -82.15 | -82.65 |
| Ordinary | Random | -79.81 | -78.76 | -79.58 |
| Strong current | Full | -145.66 | -142.33 | -144.01 |
| Strong current | Hole | -182.61 | -178.26 | -180.75 |
| Strong current | Random | -166.75 | -163.49 | -165.68 |

The **primary ordinary-hole policy screen failed**. BCA minus native was `+2.21, +0.10, -0.002` for seeds 0–2, while BCA minus distance was `+1.43, -0.67, +0.72`. The frozen rule required BCA minus native of at least +1 and BCA minus distance at least zero in every seed. The equally weighted BCA-minus-native mean was +0.77; the precomputed seed/start bootstrap interval is a coarse descriptive interval, not a significance claim.

In the strong-current hole condition, BCA minus native was `+7.11, +2.37, +3.58`, while BCA minus distance was `+2.36, -0.57, +5.67`. Strong-current results are descriptive shifted-context comparisons, not the prespecified ordinary-current primary screen.

The **policy-rescue screen failed**. Native hole minus random in ordinary current was `+4.99, -4.72, -9.60`: withholding did not damage native return in every seed. The BCA-versus-native hole-minus-random interaction was `-0.43, +1.35, -1.76`. Mixed effects do not establish equivalence or a consistent rescue of missing-label damage.

The full-data native policy exceeded its untrained ordinary reference by at least one unit for all seeds, so the learning-adequacy screen passed. Learning was not monotonic: the equal-seed strong-current hole return was -138.03 at the common 10k fork, then -182.61 for native and -178.26 for BCA at 20k. This is an observed change across fixed checkpoints, not an identified causal mechanism; selecting a checkpoint after seeing these results is not the frozen final comparison.

## Q accuracy and score adjustment

Q accuracy is measured against each checkpoint's own finite-horizon `Q^pi`: take the candidate action, then follow that checkpoint's deterministic actor to the end of the episode. Signed error is score minus this reference; a positive value is overestimation. The primary panel uses fixed first actions `-0.25, 0, +0.25`, with other coordinates zero, at high-load anchors. It has 21 ordinary or 32 strong-current anchor families per seed and three correlated action rows per anchor.

The **ordinary-hole accuracy screen passed**: native Q1 MAE minus BCA-adjusted `Q1 - 3U` MAE was `+7.94, +11.38, +7.80` for seeds 0–2, exceeding the required +1 in every seed. Under strong current the corresponding differences were `+5.85, +11.74, +10.27`. These cross-model comparisons use different continuation policies and hence different `Q^pi` references.

The following within-critic comparisons hold the BCA-trained critic, action, state and reference fixed:

| Context | Seed | Raw Q1 MAE | Adjusted MAE | Improved / worsened errors |
|---|---:|---:|---:|---:|
| Ordinary hole | 0 | 25.98 | 19.23 | 45 / 18 of 63 |
| Ordinary hole | 1 | 33.95 | 22.53 | 51 / 12 of 63 |
| Ordinary hole | 2 | 33.44 | 25.70 | 48 / 15 of 63 |
| Strong-current hole | 0 | 25.11 | 21.17 | 57 / 39 of 96 |
| Strong-current hole | 1 | 28.86 | 16.87 | 78 / 18 of 96 |
| Strong-current hole | 2 | 41.40 | 32.70 | 72 / 24 of 96 |

Improved/worsened means absolute error decreased/increased by more than 0.1; action rows are correlated and do not create more seed replications. The correction also introduced underestimation: ordinary-hole over-to-under crossings were 9, 5 and 16. It was not universally beneficial: for strong-current full-data seed 0, the BCA-trained critic's high-load fixed-band MAE increased from 16.29 to 20.80 after adjustment. The [same-critic](figures_q/02_same_critic_accuracy.png), [harm](figures_q/03_adjustment_harm.png), [visited-state](figures_q/04_visited_accuracy.png), and [common-reference](figures_q/07_common_reference.png) figures show distinct estimands.

The **accuracy-rescue screen failed**. Ordinary native hole-minus-random MAE was `-1.98, +3.53, +5.03`, and the BCA-specific hole-minus-random accuracy interaction was `-0.87, +2.82, +0.70`. The correction improves the studied accuracy panel but does not show consistent repair of damage caused by the hole.

## Interpretation and provenance

The native and distance baselines use matched shadow auxiliary and calibration work; this is not a comparison of ordinary native compute cost. Retained-support calibration and adaptive reuse do not establish fresh conformal coverage in the withheld region, at selected actor actions, under strong current, or for true Q. Free-state mask labels do not establish local state density or joint-action support. The six-action forced-bank diagnostics concern one forced action followed by the policy; their oracle regret is limited to that bank and is not policy return or a global optimum.

One BCA branch was interrupted and later resumed from its 16k checkpoint, with exact restore checks across 227 saved arrays. All planned branch checkpoints completed. The recorded physical training counts are lower bounds: at least 360,600 critic, 180,300 actor and 360,600 auxiliary updates, including at least 600, 300 and 600 repeated updates respectively; unlogged in-flight work remains unknown.

The first evaluation stopped on a progress-file permission error after 143 of 144 contexts. The separately approved evaluation-only rerun completed with no training or calibration updates, reproduced both banks and all 143 previous contexts byte-for-byte, and used only its own 144 contexts for the decision. The original calendar deadline was waived; there were two continuations and two evaluation attempts, both continuations approved before evaluation outcomes were inspected. The [terminal receipt](../../experiments/thrusterchain_bca/zeus_20261026_validation/central_quartile_training_v1_evaluation_rerun1/RUN_RESULT.json) records these qualifications.
