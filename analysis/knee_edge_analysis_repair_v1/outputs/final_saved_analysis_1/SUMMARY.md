# Completed exploratory knee-edge analysis

The original attempt completed at October 31, 2026, 1:33:10 a.m. EST. All 12 warmups, 12 native continuations, three BC-only runs and three final evaluations completed with no failed supervisor receipts. The original exploratory label is INCONCLUSIVE; its protocol label remains null. This separately recorded analysis preserves that outcome and cannot authorize Stage 2.

The added written V0(f) RANDOM check is INCONCLUSIVE in every seed. The resolvability floor passes, but the one-sided residual upper bound fails. The original V0(b), V0(d) and V0(e) failures also remain. These are validity failures, not a demonstrated failure or success of BCA.

## Missing RANDOM check

V0(f) uses R-valid anchors, common validity plus all RANDOM item-5 conditions, among 48 gated E2 anchors at times 0, 25 and 50. C-valid counts are descriptive. The floor is median rho-hat >= 0.30; the upper bound is median KC <= 0.3 * (median rho-hat + 0.15). Negative KC is permitted.

| Seed | Common / R-valid / C-valid | Median rho-hat | RANDOM median KC | Upper bound | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | 48 / 48 / 46 | 1.287105952 | 1.352309819 | 0.431131786 | INCONCLUSIVE |
| 1 | 48 / 43 / 47 | 1.296911061 | 1.264296489 | 0.434073318 | INCONCLUSIVE |
| 2 | 48 / 43 / 46 | 1.636362924 | 1.624846793 | 0.535908877 | INCONCLUSIVE |

KC measures overvaluation of an action relative to the edge action, not absolute Q bias. The labelled RANDOM control itself exceeds the positive-error allowance. This does not isolate missing-label effects from more general critic or reference-policy problems.

## Reference-policy effect

The exact row-level identity is KC_clip = KC_own + D, with D = delta Qpi_unclipped - delta Qpi_clip. It also holds for paired CUTOFF-minus-RANDOM rows. The table uses means over all 48 common anchors, so the additive identity is meaningful. The primary clipped reference is unchanged.

| Seed | Mean paired KC_clip, s10 | Mean paired KC_own, s10 | Mean paired D, s10 |
| --- | ---: | ---: | ---: |
| 0 | +0.206921381 | -0.453472457 | +0.660393838 |
| 1 | +0.047315536 | +1.186661352 | -1.139345817 |
| 2 | +0.008626867 | +0.099371285 | -0.090744418 |

The continuation-policy contribution does not cancel across arms. The protocol statement that it cancels in the contrast is therefore contradicted by these saved values. At s10, the paired median changes from +0.130027 to -0.211799 in seed 0, from +0.009817 to +0.295549 in seed 1, and from -0.056284 to -0.080358 in seed 2 when changing from clipped to own-policy references. These are separate medians; medians must not be added. Each arm still has its own continuation policy, so the own-policy comparison is not a common-reference-policy experiment. Neither reference equals the native smoothed TD operator.

## Native policy returns

20k checkpoint, stall3-trained policies evaluated in stall3. Larger (less negative) return is better. E1 has 64 shared episode starts per policy; E2 has 128. CUTOFF-minus-RANDOM is paired by that fixed bank.

| Seed | E1 CUTOFF | E1 RANDOM | E1 difference | E2 CUTOFF | E2 RANDOM | E2 difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | -81.115910 | -78.205620 | -2.910289 | -380.582003 | -368.501638 | -12.080365 |
| 1 | -78.767993 | -82.763510 | +3.995517 | -381.280555 | -386.655799 | +5.375244 |
| 2 | -81.039697 | -80.880177 | -0.159520 | -378.876576 | -378.617164 | -0.259412 |

The final return differences are not directionally consistent across seeds. At 15k, E2 CUTOFF-minus-RANDOM was -6.775664, -2.878722 and -4.145355; at 20k it is -12.080365, +5.375244 and -0.259412. Checkpoints are reported, not selected after looking at outcomes.

Clipping only a0 to the knee improves E2 returns by 296.679541, 296.110155 and 292.820424 for CUTOFF, and 285.288437, 302.349780 and 294.876676 for RANDOM. This is a physical clipping diagnostic, not a deployable learned BCA penalty. Its benefit is large in both arms, and the extra clipping gain for CUTOFF has mixed signs. The frozen K1a onset contrast is +0.075507, +0.006022 and -0.026672, each below its +0.5 requirement; because the validity gates fail, this must not be given the protocol's withholding-mechanism failure interpretation.

## Limits and next decision

The run remains exploratory after the earlier A3 INCONCLUSIVE result. No BCA policy was trained or compared here. The original native policies have shadow auxiliary fitting/calibration, but no active BCA actor penalty. These results do not establish BCA detection, action correction or policy improvement. They also do not prove that withholding can never matter.

Three training seeds and shared evaluation banks limit inference. The per-seed episode standard errors in SAVED_REPORT_SUMMARY.json describe variation across that bank for a fixed pair of policies, not training-seed or dataset uncertainty. Independent anchor IDs and full transitive producer authentication remain unavailable; shared-anchor construction, source/job hashes, receipt checks and immutable input hashes support the paired analysis. No source drift was observed.

The recorded stage caps report 13,615.872746 training seconds and 1,528.643773 evaluation seconds, with 6,270,800 charged evaluator transitions. These are recorded values, not a fresh certification of physical execution totals: the known batched-counter defect remains unresolved. The isolated analysis added zero model updates, calibration updates and simulator transitions.

Do not advance this attempt as a validated BCA efficacy benchmark or loosen its thresholds. Preserve the failed control checks and the corrected reference decomposition. Any further study needs a separately reviewed native/reference-validity repair; this analysis supplies no new execution authorization.

## Verification and provenance

The unchanged isolated package previously passed 45 fixture tests. All empirical inputs passed its completion, schema, identity, source/lineage and pre/post-read hash checks. An independent saved-array review reproduced every decomposition row exactly (maximum identity residual 7.11e-15) and predictor quantities within 1.71e-13, with identical V0(f) masks and results. The original reports and their hashes are recorded in RESULT.json and SAVED_REPORT_SUMMARY.json. Nothing in the live experiment was changed.

- Original terminal report SHA-256: `38e307160c0656d0a2989e4a32d98f91d49451c92e422f47b8bab66ac39c8550`
- Original decision SHA-256: `db15e8eb2562ad15281bca2013270a71e028bfcc91b89ccb086985f6ef8f84f7`
- Isolated result SHA-256: `2f3386652ec528b77fb55511dee914aebec3a3f63a95916b7308c47f8128ae8c`
- Analysis contract SHA-256: `82f47ba4f58d363978414d3744bea2c8bbcad59013a5710598b0ed3d248c6937`
