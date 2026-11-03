# Independent review of the saved-data analysis

Reviewed November 2, 2026, EST. Scope: the completed knee/stall confirmation and the new pure-array analysis in this directory. This is not a review or execution of Path A. Original experiment files, decisions and receipts were not edited.

**Verdict: the current saved-data calculations and three diagnostic figures are suitable for exploratory interpretation, with the limits below. No remaining numerical or integration blocker was found in the reviewed analysis.** This does not convert the original experiment's INCONCLUSIVE classification into a pass or establish a uniquely Bayesian mechanism.

## What was independently checked

- Read the dataset selectors, bindings, normalization, evaluator and BCA scale/calibration source actually used by the completed study. Read both new analysis implementations, their fixtures and the small `run_analysis.py` wrapper. The wrapper dispatches only the two new saved-array scripts; no experiment module, model, simulator or optimizer is imported or run.
- Independently loaded the two original archives with NumPy and reproduced the main retained-row, field-identity and preceding-withheld-action-history counts. Checked all 112 consulted dataset/source/receipt input hashes. The reconstruction agrees with all 30 effective completed training receipts and their five stream identities.
- Independently recomputed 216 uncertainty/control comparisons directly from the 24 original evaluation reports: four pooled AUROCs, four conditional AUROC means/eligible counts and the range-matched distance-control regret for each policy. Pooled AUROCs matched exactly; conditional means and regret agreed within 1e-14. No model calls were involved.
- Independently exercised positive AUROC orientation, reversed orientation, score ties, one-class handling, eligible-bank accounting, invariance of the distance control to a constant U offset, and zero-range U/distance behavior on explicit arrays. Read the builders' 9 dataset and 13 uncertainty fixtures. Those fixture suites are separate from the empirical recomputations.
- Inspected all three final PNGs. Corrected versions show explicit world labels, seed markers, the prespecified kappa 63 annotation, undefined AUROCs and the eligible-bank denominator in every cell. Event labels no longer cover low-AUROC points. No fake confidence interval or independent-384-seed claim is present.
- Checked the completed integrated wrapper receipt and all 15 recorded output hashes. Both analyses exited 0. Its dataset findings match `dataset_final` except the execution timestamp; diagnostic numerical JSON and all three PNGs are byte-identical to the reviewed standalone outputs.

## Findings that can be stated

**The worlds do not give the learner identical retained data.** Both select the same 47,978 of 76,800 fitting rows, with identical requested/executed actions, masks, initial states, episode identities and terminal flags. But 46,532 retained source observations/states, rewards and successors differ. Every one of those rows follows an earlier removed command beyond the knee; no differing retained state lacks that history. The analogous calibration count is 122 of 128 rows. This saved-array association is consistent with the different dynamics leaving consequences in subsequent retained states. It is not evidence that excluded transition tuples or evaluation labels leaked into training. No counterfactual dynamics replay was performed.

The normalization fixture has 32,927 differing joint rows out of 34,085. Actual means and standard deviations differ in 10 coordinates across worlds. Applying the saved float32 normalization yields different normalized observations at all 47,978 retained rows. Arms and seeds within a world share preprocessing. The inherited phrase “identical all conditions” must therefore not be interpreted as identical across worlds. Equal episode/time IDs are not equal physical states across these worlds, so an information-impossibility claim is unsupported.

**U distinguishes the constructed action gap more consistently than it identifies harmful value errors within a state.** For the 18 BCA policies, support AUROC pooled over states is approximately 0.517–0.552, while within-bank withheld-versus-logged concordance is approximately 0.691–0.943. These are different estimands: state-dependent width levels can dominate pooled comparisons. Neither is a full joint state-action density or support test. Cutoff distance has perfect separation of this label by construction; this is a sanity control, not a learned OOD detector result.

For lower-valued withheld actions, stall-world mixed-label evidence spans 59–69 eligible banks per BCA policy. No-stall results use only 0–6 of 192 banks; a conditional AUROC of 1 in such a cell is not broad evidence of reliable discrimination. One no-stall cell has no eligible banks and is correctly undefined. Positive raw-error discrimination also has far fewer than 192 eligible banks. The plot displays these denominators instead of hiding them behind aggregate averages.

## Definitions and interpretation limits

1. **Signed error is raw Q1 minus that model's own unclipped-policy Q-pi.** Lower-valued than the logged candidate is a separate relative-value event. Their conjunction is called harmful optimism in the analysis, but does not require that the actor or raw-Q argmax actually selects the action, and is not proof that optimism caused a policy mistake. Exact zero, positive and negative are descriptive signs here; this adds no fitted practical tolerance and changes no original pass criterion.
2. Candidate banks are identical across models within a world: 192 states, four complete actions each. Own-policy continuation changes with the model, and own-visited probes additionally change state visitation. Cross-model error changes therefore cannot be attributed solely to improved prediction at a fixed reference policy. Three training seeds are the independent training units; the 128 starts and 192 banks are nested/shared observations.
3. **The distance control is a penalty-shape diagnostic.** It uses `kappa * range_bank(U) * (distance - min_bank(distance)) / range_bank(distance)`, with zero penalty for zero distance range. It uses no Q-pi or evaluation-return labels, and is invariant to constant U offsets. Nevertheless it borrows the learned U amplitude and adapts per bank. It is not an independent fixed-strength baseline, a deployed actor or a newly trained control. Its favorable or unfavorable candidate regret does not establish policy-return performance or isolate Bayesian contribution. The cutoff-only control has just the logged action available in these banks.
4. The code's local scale is fitted with symmetric soft residual coverage plus a width regularizer, not a simple absolute-error regression. Its reference uses the implemented twin-min/smoothed-target one-step residual operator. U multiplies local scale by positive calibration and residual-scale factors. A shared positive scalar cancels in within-checkpoint rank/AUROC comparisons; these diagnostics primarily test local-scale ordering, not calibration's causal contribution or a conditional OOD guarantee.
5. First-index score tie handling is explicit. True-value improvements, harms, changed equal-value choices and unchanged choices are separate. The current actual data have no changed equal-value cases. Candidate regret is restricted to the four saved candidates, not optimal-control regret.
6. This retrospective analysis adds neither new learner arms nor a new experimental pass rule. The formal INCONCLUSIVE result and two Q-pi identity misses remain. The observed kappa 189 stall-world return gains in all three seeds remain useful evidence, together with its no-stall losses; neither observation is erased by the formal label or upgraded to general superiority.

## Issues resolved during review

- The initial uncertainty script lacked the wrapper's `--output-dir` interface. It now accepts an exclusive new output directory. The actual integrated run completed successfully; earlier analysis outputs were preserved.
- Wording suggesting removed labels remained was corrected to the consequences of earlier removed transitions remaining in later states.
- Figure labels that obscured low-AUROC values and missing world headings were corrected. All final rendered figures were inspected again.
- Range matching now explicitly describes borrowing U amplitude; the harmful-optimism label explicitly does not imply actual selection. These are interpretation clarifications, not metric changes.

## Reviewed identities and actual execution

| Artifact | SHA256 |
|---|---|
| Original completed `CF_RESULT.json` | `b136b39c670e7224c22fe1b8f38062a22d30f608d22c9b4253c406194b53d197` |
| `dataset_comparison.py` | `350758abddf2600f10fef50d6378b3450a1e783c8b79da680c10b7e04851016e` |
| `uncertainty_diagnostics.py` | `42fb91e08f96f80310cd3167c20b10fe77ef8b17660e0662aa56e8c2753cc75d` |
| `run_analysis.py` | `1e811dd0af82d2d0c8f70c66e64d253b521bc7d81927211834050ca06fc9b08c` |
| `output/verified_run/RUN_RECEIPT.json` | `8705604e8547c0ec836412336aebfca2281a2803a38ac9ac3523568786004a86` |
| Integrated `dataset/dataset_summary.json` | `7c4af8227fae989978e85b00c0d67b2a317cc8f566be8c58e533cb982a40ba19` |
| Integrated `diagnostics/uncertainty_metrics.json` | `1c450084e1043dc508d2a200d35ec81b692aecc66e5b96a35f2748d15c07f0c9` |

Integrated CPU analysis ran from 10:02:01 p.m. to 10:02:08 p.m. EST on November 2, 2026, with 6.875 seconds recorded. No model calls, simulator calls, optimizer updates or calibrations occurred. Review computations likewise used only saved arrays and explicit synthetic arrays. PDF exports were produced by the same plotting code; independent visual inspection was of the rendered PNGs, not a separate PDF renderer.

Skills actually read and applied by this reviewer: `.agents/skills/offline-rl-experiment-design/SKILL.md`, `.agents/skills/bca-code-and-calibration-audit/SKILL.md`, and `.agents/skills/offline-rl-evidence-and-visualization/SKILL.md`. Applied checks covered original-successor provenance, role separation, actual actor/error semantics, paired units, fixed diagnostic orientation, calibration limitations and honest figure interpretation. No additional scientific experiment or gate is required by this review.
