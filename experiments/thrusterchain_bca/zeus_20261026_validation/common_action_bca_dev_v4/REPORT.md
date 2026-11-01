# Common-action, common-continuation diagnostic

**Outcome: completed; the prespecified own-visited action-selection screen was inconclusive in ordinary and strong current.** Each seed/arm had fewer than the required four non-tied changed actions when guarded compatibility was added to symmetric Q1 BCA. This is a frozen-action mechanism diagnostic, not evidence of a trained-policy return or safety improvement.

## What was held common

Three paired training seeds supplied six authenticated, final 20k hole-condition checkpoints, native and original BCA. For a given seed, context and state, both critics scored the same 15 actions: the two saved actors' actions, eight ±0.2 single-coordinate perturbations around their midpoint, and five historical fixed vectors. Clipped duplicates were retained. Each action was evaluated under **both** saved actor continuations. Thus four critic/continuation comparisons use the same physical states and candidate actions. Only the native/native and BCA/BCA diagonals use each critic's own actor continuation; off-diagonal cells are deliberately cross-policy references and are not own-policy critic-accuracy measurements.

For each context, 16 fresh random-prefix states were shared across all seeds. For each seed and context, eight additional states were reached by each frozen actor from distinct fresh starts. Both critics and continuations were then evaluated on all 32 pooled states. The random-prefix, native-visited, and BCA-visited sources remain separate; actor-visited prefixes are not full-episode policy returns.

The existing v3 signed and symmetric model parameters, bandwidth and coefficients were retained. The v3 reference thresholds were reconstructed from its already-consumed 64 calibration rows and recorded keys, and verified against the stored threshold values; 18 refreshes are deterministic reconstructions, not fresh calibration or fitting. The compatibility reference was deterministically reconstructed using the old retained training data and checked against v3's fingerprint, bandwidth and radius. No Qπ value was used to select or tune candidates, scores or coefficients; all common banks and scores were saved before any Qπ reference was generated. The 95th-percentile state-distance gate remains a heuristic, not a support certificate.

## Prespecified own-visited decision screen

The screen compared symmetric Q1 BCA with symmetric Q1 BCA plus guarded compatibility, on each checkpoint's own actor-visited states and own actor continuation. A cell needed at least four non-tied changed states, then a mean Qπ improvement of at least 0.05 with harmful switches not exceeding improvements. A context required two qualifying seeds in each checkpoint group. There were only eight states per cell.

| Context | Native eligible / qualifying seeds | BCA eligible / qualifying seeds | Outcome |
|---|---:|---:|---|
| Ordinary | 0 / 0 of 3 | 0 / 0 of 3 | Inconclusive |
| Strong current | 0 / 0 of 3 | 0 / 0 of 3 | Inconclusive |

For strong current, mean selected Qπ gain from adding guarded compatibility was **−0.061 / +0.095 / +0.105** for native seeds and **0.000 / +0.081 / +0.174** for BCA seeds. Native improved/harmed counts were **0/1, 1/0, 2/0**; BCA counts were **0/0, 2/0, 1/0**. These sparse conditional results do not satisfy the opportunity rule and must not be interpreted as a successful screen.

## Paired common-action findings

The entries below show mean selected Qπ gain from symmetric Q1 BCA to the guarded hybrid, by seed. Within each row, the two critics use the same state source, candidate bank, and specified continuation. Positive values mean the hybrid's selected first action had the higher specified continuation value. Native/BCA in the final two columns labels the **critic**, not the continuation.

| Strong-current state source | Continuation | Native critic, seeds 0 / 1 / 2 | BCA critic, seeds 0 / 1 / 2 |
|---|---|---|---|
| Random prefix | Native | 0.000 / +0.079 / +0.003 | 0.000 / +0.079 / +0.017 |
| Random prefix | BCA | 0.000 / +0.080 / −0.004 | 0.000 / +0.080 / +0.013 |
| Native visited | Native | −0.061 / +0.095 / +0.105 | −0.061 / +0.113 / −0.068 |
| BCA visited | BCA | +0.141 / +0.095 / +0.004 | 0.000 / +0.081 / +0.174 |

On the 48 strong-current actor-visited seed/source/state comparisons, native and BCA **raw Q1** selected the same candidate slot in all 48. The symmetric scores selected different slots in 11/48 and the guarded-hybrid scores in 9/48. This shows that the adjusted score choices can differ even with the state and action bank held common; the counts do not identify the source of critic differences or a causal effect of training.

For the symmetric-to-guarded switches actually observed, there were no cases across the two contexts and three state sources in which the same switch had a gain above +0.1 under one continuation and below −0.1 under the other. Some near-zero signs differed, and individual switch gains did vary between continuations. This small, conditional sample supports neither universal continuation invariance nor a precise estimate of its effect. See [common-action gains](plots_v1/01_common_action_gain.pdf), [continuation sensitivity](plots_v1/03_continuation_sensitivity.pdf), and the full [paired comparison CSV](plots_v1/paired_comparisons.csv).

The gate was on for **6/16** strong-current random-prefix states, **6/8, 8/8, 7/8** native-visited states, and **7/8** BCA-visited states for each seed. Since the same query/reference determines the gate, those counts are common across both critics on a given state. The gate-off part of the guarded score is exactly the symmetric score, so that stratum cannot identify an added guarded-compatibility effect. See [guard activation](plots_v1/05_guard_activation.pdf) and [choice changes](plots_v1/04_choice_changes.pdf).

## Accounting and evidence

The run completed **364,904 simulator transitions**: 1,200 random-prefix, 3,600 frozen-actor-prefix, 104 simulator-parity, and 360,000 forced-continuation transitions. It recorded **5,760** candidate/continuation references, **129** resets, 18 replayed reference refreshes, and **zero actor, critic, target, or auxiliary optimizer updates**. No policy-training study or full-episode reranker evaluation ran. Three paired training seeds and one dataset remain the independent trained-replication scope; candidate actions and shared states are dependent.

The [protocol](PROTOCOL.md), [source seal](SOURCE_SEAL.json), [preflight receipt](PREFLIGHT_REAL.json), [terminal result](diagnostic_run_01/RESULT.json), six saved common-bank plans, and [figure guide](plots_v1/FIGURE_GUIDE.md) record the exact inputs, seed audit, state origins, scores, own- and cross-policy Qπ references and numerical summaries. The [plot receipt](plots_v1/PLOT_RECEIPT.json) links every plotted file to the exact result digest. Earlier studies remain untouched.
