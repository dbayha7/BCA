# Independent saved-record alignment review

2026-10-21. **No material arithmetic or scientific reporting findings in the authoritative final analysis.** Read-only review of `analysis.py`, `PROTOCOL.md`, `results_final/`, `FINDINGS.md`, `FRESH_BANK_PLAN.md` and preserved source definitions. I independently recomputed saved-array arithmetic without importing the analysis, simulator or model. No new inference, labels, rollouts, training or historical-file changes.

Verified all 10 input hashes, all 10 numeric output hashes, source/protocol identities and all six delivery-receipt files. Independently checked 192 within-state correlations, 480 associations, 2,112 pair records, 224 choice records, grouped choice/head metrics and variance partitions.

- All 22 undefined positive-part correlations have exactly zero positive errors across their 12 actions. Width, signed-error and absolute-error variation remains nonconstant at every state. The defined 10-state subset is not generalized to all states.
- Signed, absolute and positive errors remain distinct. Mean within-state width/signed-error Spearman is 0.041302; absolute-error Spearman is -0.063811. Pooled associations and state-centered associations differ; neither identifies causal confounding.
- Both orientations make the pair correlation invariant to arbitrary pair order (`analysis.py:149-193`). There are 2,112 pairs, six true near-ties and 2,106 comparable pairs. Exactly 18 repairs and 18 breaks leave pair-weighted concordance unchanged at 0.661918328585. Equal-state concordance changes from 0.661868762748 to 0.661876274767 because eligible-pair counts differ. The 4,224 oriented entries are dependent arithmetic entries, not independent observations.
- The sole full-bank width switch is state 6, restoring the original actor and improving Qpi by 4.922153. Both candidate errors are negative: this corrects relative overvaluation, not detected positive absolute overestimation. The original actor is an explicit unrestricted baseline; its 32 actions are unsupported by the categorical behavior law and are not inserted into restricted common/rare comparisons.
- Width and eta ranks match at all 32 states, with no observed floor hits or constant widths. The shared positive factor R times unit cannot improve their ranks by scalar rescaling. It can change score strength and selection. Saved values do not identify activation saturation, derivatives, epistemic uncertainty or causal effects of lag/smoothing. The arithmetic twin-head comparison does not reconstruct the historical calibration target.

The final prose accurately limits the evidence to one fitted model and an already inspected candidate bank. The proposed independent 32-state replication has useful fixed distance, actor, constant-offset and permuted-width controls; no execution is authorized by it. Before any launch, its numerical advancement thresholds, interval rule, multiplicity handling, seeds and fixture budget still require an explicit freeze. It must allow an inconclusive outcome without adaptive expansion. The final plan states these limitations.

Authoritative identities (SHA256):

| Artifact | SHA256 |
|---|---|
| Saved v2 raw NPZ | `623ec4c5f8bf5f9b7ec9277a4d8b98d82973f162f8847c79292a792370014e35` |
| `analysis.py` | `e484ceb743bd81d8b597875ccd68229c58218d220df45744459ae80a66d1a99d` |
| `PROTOCOL.md` | `104b24e02a12a311bf02c1ba9d03c2f36d3c953d14f5f29c9cb6a6028c1e9cdc` |
| `results_final/summary.json` | `a28ac10f13dcaec3427d12c3b404cde5d4ed9a962d6448b12e2c263e2abaffb2` |
| `results_final/manifest.json` | `7c58559cee2d7a56d69ed4c284b8f776f7e9915b41589a25a7f485d7f3fcfa6e` |
| `FINDINGS.md` | `07f7c4b4c92ce1cdc09e8cdb2ccb197e505679ad61d28b85ac52485ec2e80b4f` |
| `FRESH_BANK_PLAN.md` | `20da3407e15c9b83d9bd130efe139122fe59f4e33897cfcb6d057325e2fd9409` |

The preliminary `results/` directory is not the authoritative result: its exact older source snapshot is unavailable, as disclosed in the delivery receipt. This does not affect the verified final source/output linkage. No fresh-bank execution or independent rollout reproduction was part of this review.
