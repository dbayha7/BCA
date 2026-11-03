# Independent figure and data QA

Reviewed November 2, 2026, 6:14 p.m. EST. Verdict: **ready for saved-data figure delivery**, with the scientific limits below. This is a visualization review, not a new experiment or a revision of the study verdict.

## Evidence and arithmetic

- Independently read all 24 final evaluation reports in `experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/confirmation_v1/cf_run/attempt_3`.
- Verified the final `CF_RESULT.json` SHA256: `b136b39c670e7224c22fe1b8f38062a22d30f608d22c9b4253c406194b53d197`.
- Verified all 29 source-file hashes and sizes and all four CSV-extract hashes recorded in `SOURCE_MANIFEST.json`.
- Recomputed 13 metrics for each of 24 policies directly from the original reports: mean return, paired return difference, episode wins, exposure, raw and adjusted signed error and MAE, raw and adjusted candidate regret, helpful and harmful switches, and unchanged choices. All **312 comparisons** matched `policy_metrics.json` exactly; maximum absolute difference was **0**.
- Checked 128 return starts, 48 own-action probes, and 192 four-action banks per policy. All relevant values were finite. Within each world, the complete bank state IDs and all four action vectors matched across seeds and methods. No raw-score, adjusted-score, or true-value ties occurred in these banks.
- Checked the reported kappa-189 stall gains of +11.245, +121.479, and +9.760 and no-stall losses of -41.421, -98.321, and -20.525. The no-stall policies won on 0/128 starts in each seed. The large stall seed-1 gain contributes approximately 85% of the summed improvement; all three stall seed means nevertheless improve.
- Checked candidate helpful/harmful counts across 192 banks times three models: stall 78/32, 135/39, 161/60 and no-stall 20/80, 33/121, 38/157 at kappa 21/63/189 respectively.

## Implementation and visual review

Inspected `reproduce_figures.py`, the extracted tables, metrics, and `FIGURE_GUIDE.md`. The reproduction path uses saved data, NumPy and matplotlib; it does not import a learner or simulator. Source extraction requires the final receipt hash, and reproduction validates the CSV hashes. No model, simulator, calibration, GPU or WSL operation was run by this review.

Viewed every final PNG individually after revisions. Initial overlapping legends, figure-3 label collisions, missing figure-2 world/seed labels, and an overly long figure-5 heading were corrected. The final renders have readable titles, legends, axes, seed symbols, units, reference lines and captions, with no observed clipping or obscured data labels. The consistent blue/orange palette is supplemented by labels and seed shapes. All three seeds and all strengths remain visible; no confidence interval or significance claim is manufactured from nested evaluations.

PNG metadata reports approximately 300 dpi. Figures 1-3 are 3450 by 2130 pixels; figures 4-5 are 3450 by 2820 pixels. The PDF export code saves the same matplotlib figure objects with vector output and TrueType font embedding. The PDFs were not independently rasterized in this review because the available Python environment lacked a PDF renderer; no software was installed. Final PNGs, rather than a separate PDF rendering, received visual inspection.

Reviewed script SHA256: `fbe6600a0bc6c3a50e9d100bd34b88f70ef0f7c5a909b3a71addb51e319cc3f3`.

Reviewed guide SHA256: `b70cf05ca7fbb38e178bfc6bb072a10da427bcaf129f216636a6186b3d38007d`.

Reviewed validation record SHA256: `9c7f9b03a9952e5523c2fb07b4c1748e6ac24ca9a44d5a0492f888a764c7e9b3`.

## Interpretation limits preserved

- The frozen overall result remains **INCONCLUSIVE**; C1 remains **NOT CONFIRMED**, and C2 remains **OVER-CORRECTION**. The two relative Q-pi identity misses are not erased, excused by a new tolerance, or used to dismiss descriptive return observations.
- Kappa 63 is distinguished from the 21/189 dose profile. Kappa is penalty strength, not a head count.
- Three training seeds are the independent training replications. The 128 common starts are nested evaluations. The 576 bank evaluations per dose/world reuse 192 states across three models and are not 576 independent states.
- Own-action Q errors use each policy's visited states and its own continuation. Differences between trained policies do not isolate matched-input critic accuracy. Raw Q and adjusted actor score are shown separately; subtraction of kappa U is not a two-sided correction or a full-return Q guarantee.
- Candidate-bank switches are retrospective score diagnostics, not deployed actor decisions or proof of the causal return mechanism. The differing regret-axis scales are explicitly disclosed.
- These figures do not establish uniquely Bayesian benefit, withholding-specific rescue, a purely action-only shift, or a missing-connector result. Path A is a separate study.

## Skills used and coverage boundary

The reviewer read `.agents/skills/offline-rl-experiment-design/SKILL.md` and `.agents/skills/bca-code-and-calibration-audit/SKILL.md`. Applied checks covered pairing, nested evaluation units, metric signs and denominators, raw-versus-adjusted score semantics, own-policy reference interpretation, provenance, and preservation of failed or inconclusive criteria.

This review covers the source-to-figure arithmetic, reproduction implementation, final PNG presentation, and captions. It does not revalidate training execution, recompute Q-pi with the simulator, resolve the two identity misses, or certify the entire repository. Original experiment artifacts were read only.
