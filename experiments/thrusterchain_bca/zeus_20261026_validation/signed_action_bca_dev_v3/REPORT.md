# Expanded action-bank and signed-scale diagnostic

**Outcome: the bounded run completed; the prespecified guarded-hybrid screen failed in both contexts and both state strata.** All checkpoint/context/stratum cells had enough informative states to be evaluated. In strong current on the natural stream, the guarded hybrid qualified for two of the three original-BCA checkpoints but none of the three native checkpoints. The full gate required at least two qualifying seeds in each group.

This is a frozen-action experiment after earlier exploratory development. It did not change any policy, critic, target or auxiliary model, and it cannot establish trained-policy performance or deployment safety.

## Inputs and comparisons

The run restored six authenticated 20,000-update hole-condition checkpoints, three each from the native and original-BCA branches, together with the previously fitted signed, Q1 and twin-min auxiliary models. All were frozen. Each model was recalibrated on 64 new behavior episodes; a disjoint 64 episodes supplied the descriptive one-step TD check. The six checkpoints share these episode sets. One retained non-hole tuple was selected per episode. The original protected transition records were not read.

For each context the run generated 64 new state families, equally distributed over times 0, 25, 50 and 75. Sixteen were selected in advance as the natural stream, and 16 more were selected from the remaining families using a frozen raw-Q1 ambiguity rule without simulator Qπ. The natural stream is sampled from the prespecified **random-prefix state distribution**, not from a saved actor's deployment visitation distribution. The ambiguity stream is conditional on the raw-Q1 selection rule and is not a natural-state average. Both streams use the same selected states across checkpoints.

Each checkpoint/state used 15 candidates: the saved actor's action; eight ±0.2 single-coordinate actor perturbations with other coordinates unchanged; a deterministic nearest retained logged action; and the five original fixed vectors. Duplicate actions caused by clipping were retained in the budget and excluded only where the distinct-action margin or informative actor-versus-perturbation comparison required this. Every Qπ reference is that forced first action followed by the same checkpoint's frozen deterministic actor, discount 0.98, finite horizon 100, with no terminal bootstrap. Bank regret refers only to the best of those 15 candidates.

The principal candidate was symmetric Q1 BCA plus the guarded revised-compatibility penalty. The guarded term uses a training-distance heuristic; it is not a certificate of state/action support. Controls included raw Q1, symmetric Q1 BCA, signed BCA, signed penalties matched separately to symmetric mean magnitude and action-gradient RMS, compatibility only, an unguarded hybrid, and a symmetric-width control matched to hybrid gradient RMS. Multiplying the signed penalty makes a score control; it does not retain the original calibration coverage claim. Matching used the original retained training probes and occurred before all Qπ evaluations. Compatibility neighbors/statistics were fold-excluding; the auxiliary models themselves were fit on the full retained training data, so their predictions at these probes are in-sample.

## Prespecified decision screen

For each checkpoint/context/stratum, eligibility required at least eight families with an informative actor-versus-perturbation reference gap above 0.1 and at least one raw-Q1 bank mistake above 0.1. All cells were eligible: all 16 state families were informative and raw-Q1 mistake counts ranged from 5 to 14. A qualifying guarded-hybrid cell additionally required mean bank regret at least 0.05 below both raw Q1 and the strength-matched symmetric width, no higher than symmetric Q1 BCA, and no more harmful than improved switches at tolerance ±0.1. A context/stratum passed only if at least two seeds qualified in **each** checkpoint group.

| Context | State stream | Eligible native / BCA seeds | Qualifying native / BCA seeds | Screen |
|---|---|---|---|---|
| Ordinary | Natural | 3 / 3 | 0 / 0 | Failed |
| Ordinary | Q1-ambiguity selected | 3 / 3 | 0 / 0 | Failed |
| Strong current | Natural | 3 / 3 | 0 / 2 | Failed |
| Strong current | Q1-ambiguity selected | 3 / 3 | 0 / 0 | Failed |

Counts are out of three seeds per group. The previous six-action gate was inconclusive from too few mistakes; the expanded bank produced enough informative opportunities for this different, prespecified screen. Its failure is a mechanism-screen result, not evidence that a newly trained policy would necessarily fail.

## Regret and choices on the natural stream

Values are mean discounted-return regret over 16 state families, by training seed; lower is better. They are relative to the best candidate in each checkpoint's own 15-action bank, not Q-star. Candidate actions and shared states are dependent, and the three training seeds are the replication units.

| Context / saved checkpoint | Score | Seed 0 | Seed 1 | Seed 2 | Equal-seed mean |
|---|---|---:|---:|---:|---:|
| Ordinary / native | Raw Q1 | 0.364 | 0.430 | 0.569 | 0.454 |
| Ordinary / native | Symmetric Q1 + guarded compatibility | 0.428 | 0.449 | 0.569 | 0.482 |
| Ordinary / BCA | Raw Q1 | 0.400 | 0.534 | 0.605 | 0.513 |
| Ordinary / BCA | Symmetric Q1 + guarded compatibility | 0.458 | 0.524 | 0.535 | 0.505 |
| Strong / native | Raw Q1 | 1.537 | 1.701 | 1.115 | 1.451 |
| Strong / native | Symmetric Q1 + guarded compatibility | 1.765 | 1.813 | 1.242 | 1.607 |
| Strong / BCA | Raw Q1 | 1.776 | 1.656 | 1.358 | 1.596 |
| Strong / BCA | Symmetric Q1 + guarded compatibility | 1.562 | 1.602 | 1.292 | 1.485 |

On the strong-current BCA checkpoints, the guarded hybrid improved regret versus raw Q1 in all three seeds, with improved versus harmed choice counts **3/1, 2/1, 3/1**. Seed 1 did not qualify because its regret was higher than symmetric Q1 BCA and tied the matched-width control. For the native checkpoints, hybrid regret was higher than raw Q1 for all three strong-current seeds; their improved/harmed counts were **2/3, 0/3, 3/3**. The native and original-BCA checkpoint groups have different saved actors and therefore different own-policy Qπ references; these comparisons do not isolate a causal training effect.

Actor-versus-coordinate-perturbation rankings tell a related but distinct story. In strong-current BCA natural states, raw-Q1 pairwise accuracy was **78.4%, 73.7%, 81.5%** across seeds, while the guarded hybrid's was **74.8%, 74.6%, 77.8%**. These percentages count correlated candidate pairs within the same 16 families, not independent trials. See [regret](plots_v1/01_regret.pdf), [pairwise accuracy](plots_v1/02_pairwise_accuracy.pdf), and [choice changes](plots_v1/03_choice_changes.pdf).

## Signed scale and calibration

The signed-bound multipliers selected on training probes were **2.571–4.181** for matching mean symmetric penalty magnitude and **1.722–2.025** for matching action-gradient RMS. One multiplier cannot generally match both. For the strong-current BCA natural states, all-action Q MAE was:

| Score | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Raw Q1 | 26.293 | 33.276 | 44.374 |
| Symmetric Q1 BCA | 24.954 | 24.767 | 34.438 |
| Unscaled signed BCA | 25.318 | 30.574 | 42.020 |
| Signed, matched mean magnitude | 28.150 | 23.776 | 36.392 |
| Signed, matched gradient RMS | 25.909 | 28.215 | 39.826 |

Matching the mean magnitude lowered the equal-seed mean error compared with unscaled signed BCA, but worsened seed 0; it was not a uniform gain, and matching one scalar does not isolate the signed architecture. Q accuracy and decision regret also did not move together consistently. On ordinary natural states, symmetric Q1 BCA had larger all-action MAE than raw Q1 in every seed of both checkpoint groups.

On the 64 fresh behavior check rows, the signed residual predictor had lower one-step TD MAE than the zero predictor in **three of six** checkpoints. The one-sided Bayesian TD threshold covered **59–63 of 64** rows. These descriptive shared-episode checks apply to the sampled behavior-row distribution and one-step target; they do not establish true-Q, selected-action or shifted-state coverage. The guarded compatibility term was active for **15/16 ordinary natural**, **14/16 ordinary selected**, and **7/16** in each strong-current stream. Distance threshold passage is a heuristic, not a support label. See [calibration and matching](plots_v1/04_calibration_and_matching.pdf).

## Work and evidence

The run completed the exact cap of **377,704 simulator transitions**: 4,800 random-prefix steps, 104 parity steps, 12,800 fresh behavior steps, and 360,000 forced continuations. It produced **5,760 own-policy Qπ references**, performed **18 calibration refreshes**, and made **zero actor, critic, actor-target, critic-target, or auxiliary optimizer updates**. Sixteen natural and 16 Q1-selected states per context are paired across the checkpoints.

The frozen [protocol](PROTOCOL.md), [source seal](SOURCE_SEAL.json), [terminal result](diagnostic_run_01/RESULT.json), [fresh behavior archive](diagnostic_run_01/fresh_behavior_episodes.npz), both `*_pool_and_selection.npz` archives, and [figure guide](plots_v1/FIGURE_GUIDE.md) record the source identities, seeds, selection, metrics and outputs. The terminal result SHA-256 is `3b607c1c810d75477ccc927f23000d2649d5f7585ad65838cc07745e67698387`, also recorded in [PLOT_RECEIPT.json](plots_v1/PLOT_RECEIPT.json); the guide and complete [per-method metrics CSV](plots_v1/all_metrics.csv) are linked to that source. Earlier studies and their outputs remain intact.
