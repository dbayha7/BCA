# Independent post-run review

2026-10-21. **PASS: no material discrepancy in the completed run's saved arithmetic, denominators, bootstrap summaries or frozen identities.** This is a descriptive replication for one trained model; it does not establish uncertainty calibration, repeated-intervention policy benefit or a generally superior method.

The independent checker imports NumPy/SciPy, not `replication.analyze` or any model/simulator implementation. It verifies the saved data using independently written correlation, pair-ranking, choice and aggregation arithmetic. No inference, new labels, rollouts or training occurred in this review.

## Verified

- All seven result-file hashes, six prepared-artifact hashes, three production source/protocol hashes, eight prior-file hashes, and preflight/test/freeze/execution links match. All eight parameter identities agree before preparation, before execution and after execution.
- All 32 states are valid. All 384 continuations completed with 21,696 transitions, zero nonfinite returns and zero failure events. No new replay-fixture transitions or training updates. Prepared arrays and all five score arrays are unchanged in the raw results.
- Recomputed all 96 state correlations, 160 state concordances/selected choices, 384 candidate records and all 37 aggregate metrics. Actor return is a separate control. Individual and paired correlation masks all have 32 eligible states in this realized bank; invalid/undefined counts are zero.
- Regenerated the frozen 10,000-by-32 bootstrap plan from stream 7 and verified every original context/time cell count. All 370,000 saved bootstrap means, their denominators and all percentile interval endpoints match. Every denominator is 32; no bootstrap replicate is undefined.
- Six true near-ties leave 2,106 comparable candidate pairs. Decisions retain indexed candidates, use first exact argmax, and distinguish changed indices from actual command changes. Constant offset exactly preserves the raw choices and ranking results.

## Measured results and limits

| Endpoint | Mean | Descriptive 95% percentile interval |
|---|---:|---:|
| Within-state width versus signed-Q-error Pearson | -0.111290 | [-0.231189, 0.002985] |
| Paired correlation: width minus distance | -0.125563 | [-0.242948, -0.002111] |
| Paired correlation: width minus shuffled width | -0.132611 | [-0.280195, 0.014854] |
| Live-width concordance change versus raw Q | 0.000524 | [-0.005653, 0.006214] |
| Live-width selected-Qpi gain versus raw Q | 0.044886 | [-0.367432, 0.421841] |
| Selected Qpi: live width minus distance | -0.283109 | [-0.652308, -0.009318] |
| Selected Qpi: live width minus shuffled width | -0.325345 | [-0.820429, 0.113812] |

Live width changes five actual commands: four improve, one harms and 27 remain unchanged. The harm is state 24 (monotone-high, time 0): common-grid candidate 0 changes to unsupported candidate 7, reducing Qpi by **6.006465**. The four improvements are states 1, 2, 6 and 18. There are no live-width actor restorations. Distance improves four choices with no harmed choice; shuffled width improves four and harms two. These observed counts do not imply a general safety or superiority guarantee.

The fresh-bank width/error association is not positive: 12 state correlations are positive and 20 negative. The selected-return point estimate alone therefore does not demonstrate useful alignment. Report the controls and harms together; do not convert any individual interval into a significance gate or declare a winner. Intervals describe sampled-state variation conditional on this single model and the sparse 2-3-state strata. They are not rollout-noise intervals or independent-training-seed evidence. The historical 95k one-step residual width still does not certify 100k Q1 or the full-return Qpi target.

## Reproduction and bindings

Run `independent_postrun_check.py --receipt <new-receipt-path>` with the existing NumPy/SciPy environment. The checker refuses to overwrite a receipt. The delivered receipt records the checks, all statewise selection effects and key artifact hashes.

| Artifact | SHA256 |
|---|---|
| `results/raw_results.npz` | `f31718e6c1261cdd784d6d009cc7494499467f5881f4ddf4931257c7667920bb` |
| `results/paired_results.json` | `6e815ac9fe1e86b08619a45812635fe9ea677abbd4c6cb77508617c398bd157b` |
| `results/bootstrap_statistics.npz` | `9011ec1d2fab4f1b6ff87586624ba23792e31d606d4d36c440af90729e1ca744` |
| `independent_postrun_check.py` | `43208431b0cf006e497692da6bb71014d39ff7fef11ebd98570fa4831bfa13eb` |
| `INDEPENDENT_POSTRUN_RECEIPT.json` | `6d8a9f72891e6dbb8916bf548693e0504f957e45fd1fed1791959e00b25f9a40` |

No independent model or rollout reproduction, browser validation or future presentation review is implied by this check.
