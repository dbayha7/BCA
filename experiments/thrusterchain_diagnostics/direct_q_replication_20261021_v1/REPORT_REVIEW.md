# Independent report review — 2026-10-21

**Verdict: no material scientific or reporting issue found in the reviewed report.** This review used saved files only. It did not rerun models, oracle continuations, training, bootstrap generation, or browser rendering. Independent arithmetic reproduction is documented separately; this review checked the report's interpretation, presentation and source mapping.

## Checked claims and limits

- The primary alignment is the equal-state mean of within-state **Pearson(width, signed Q1−Qpi error)**, distinct from the earlier positive-part Spearman endpoint. The equivalent all-pairs difference formula is correctly described as an identity, not independent evidence.
- Qpi is the deterministic numerical finite-horizon return after one candidate command followed by the unchanged original actor. The report explicitly distinguishes it from Q-star, a target-network label and a controller that repeatedly chooses corrected-score commands.
- The report treats the negative observed mean alignment and near-zero ranking change as this bank's result. It makes no significance claim, universal impossibility claim, all-IWBCA conclusion, automatic training gate or new-policy efficacy claim. Negative interval endpoints for some contrasts are not relabeled hypothesis-test results.
- Unsupported commands, finite native widths, native refusal and invalid-state handling remain distinct. The report gives 192 unsupported candidate commands and does not infer joint state/calibration coverage from command-grid membership or finite width.
- The nominal 90% historical one-step setting is not reported as achieved Q coverage. The 95k historical reference versus current 100k predictor/actual continuation distinction is retained.
- Whole-state bootstrap intervals are described as conditional/descriptive, using one saved plan with endpoint masks and denominators. The 2–3 states per context/time cell and one trained model are explicit limitations. Actions/pairs are not treated as independent trained replicates.
- The width, distance, constant-offset, shuffled-width and original-actor controls are identified correctly. The single fixed shuffle is not claimed to be a randomization test. Action-ranking concordance and selected-action value are kept as different endpoints, including their differing observed directions.
- Completion counts are stated as 384 candidate continuations and 21,696 plant transitions, with no new training or additional validation transitions. Historical replay validation is reused, not recounted as a fresh run.

## Saved-file checks

- Checked the labels and interpretation of the result tables against `protocol.json` and `results/paired_results.json`.
- All **33 displayed numeric metric rows** match saved means, 95% interval endpoints, defined/planned state counts, per-draw denominator ranges and undefined-draw counts at the displayed precision.
- All **13 embedded source panels / 222 numbered lines** match their referenced implementation lines exactly after HTML unescaping.
- Removing the fresh-replication section and its navigation addition from V5 restores the reviewed V4 HTML **byte for byte**. Existing sections are preserved, rather than silently rewritten or pooled with fresh results.
- Static report/source checks do not establish rendered layout, navigation behaviour or runtime JavaScript correctness. V5 explicitly records that browser QA was not performed; no workaround was attempted.

One nonblocking source-link refinement was sent to the author: pseudocode stage 07 describes selected-value and actor contrasts as well as concordance, while its direct link points only to `concordance()`. Those additional outputs are implemented in `state_metrics()`/`analyze()` and are already present in the expanded source panels. No formula or result correction is needed.

## Reviewed byte identities

| File | SHA256 |
|---|---|
| `protocol.json` | `7583491e3bce5ff757f1eb0bfcdfd20a064ffe041771c442f2e51f8c06dcbb41` |
| `results/paired_results.json` | `6e815ac9fe1e86b08619a45812635fe9ea677abbd4c6cb77508617c398bd157b` |
| `report_assets/RESULTS.md` | `84c905b1233382a8e0a505ee0bd9701b59d9c10da163111a5a03e6b247d75132` |
| `report_assets/section.html` | `0c2fd47f09cdcf9f00d384ad8e1cb8d611b91c770e8ba4a607ad1498a5e2dcc9` |
| `presentation/ThrusterChain_diagnostics_results_v5_fresh_replication.html` | `4d5d72b3d2b057f40012b17af4c792a640911c458483bab8379ad8226fe3cb5f` |

These are raw-file byte hashes, not hashes of newline-normalized text. Any later rebuild should bind its own updated artifact hashes. No core source or result files were changed by this reviewer.
