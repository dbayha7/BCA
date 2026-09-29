# BCA metrics beyond return

The earlier advisor deck omitted already-available evidence. This revision separates calibration execution, calibration validity, warning quality and action quality. It reuses closed exports; it adds no training, simulator outcomes or checkpoint queries.

## Completed TD3+BC Hopper pilot, seed 202609171

Primary continuation is the frozen BCA policy. Harm means a loss greater than 1 raw reward unit relative to the host first action at the same state over up to 250 total transitions. These are the original pilot results, not newly completed revised OOD pairs.

| Metric | Result | Interpretation |
| --- | --- | --- |
| Within-state harm AUROC, equal-stratum mean | Width 0.529; support distance 0.566; fixed random 0.513; constant 0.500 | Weak overall separation by width in this pilot |
| Within-state average precision, same aggregation | Width 0.510; distance 0.542; random 0.432; constant 0.275 | Some enrichment among high warnings; distance remains higher |
| Eligible ranking panels | 111 / 256 | Other panels have one harm class; their AUROC/AP are N/A |
| Pooled AUROC, separate quantity | Width 0.432; distance 0.539 | Across-state mixing changes the question; not a substitute for within-state results |
| Harm rate by action support | Near: 231/2071 = 11.15%; distant: 45/233 = 19.31% | Distant actions are more often harmful descriptively, but novelty is not harm |
| Harm/support mismatch | 231/276 harmful actions are near; 188/233 distant actions are not harmful | Distance alone cannot identify every bad action |
| Tested-action regret, BCA continuation | Host first action 6.333; BCA first action 6.453 | No mean regret reduction; lower is better |
| Tested-action regret, host continuation | Host first action 8.242; BCA first action 8.490 | Same direction under the other fixed continuation |
| Saved width min / median / max | 1.352 / 1.747 / 2.479 | Width varies; none of the panels ties all alternative widths |
| Fresh empirical residual coverage | Unverified | Nominal 90% is a target, not observed coverage |

Average precision uses only eligible panels and equal stratum weights. Its constant-score baseline is therefore **0.275**, not the pooled 11.98% harm frequency. AP enrichment and weak AUROC can coexist. Neither establishes a repeated five-seed effect.

Regret is exploratory and means the difference from the best of ten tested actions within the same state and continuation. It is not regret against the unknown optimal policy. The archived per-panel data also contain complete-tie risk-retention curves; these have not been converted into a new aggregate endpoint or tuned rejection rule here.

The original pilot's accepted readout has limitations: shared reset blocks and candidate/continuation dependence, interrupted attempts retained, and missing final following full-state contents for 5,120 completed last steps plus one excluded partial output. Those states were not invented. Recorded rewards, controls, return entries and hashes were checked under the original restricted acceptance. See the original [pilot report](</C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/README.md>) in the BCA repository for the full provenance; its exact path also appears in the deck's notes and metric evidence.

## Reviewed standard-BCA Walker training, seed 202609171

| Diagnostic | TD3+BC | ReBRAC |
| --- | --- | --- |
| Accepted scale fits, independently decoded checkpoint counter | 1,000,000 | 1,000,000 |
| Bayesian + conformal refreshes | 198 | 198 |
| Refreshes with conformal below Bayesian | 198/198 | 198/198 |
| Last 1,000-update block actor BC multiplier | 1.33136 | 1.37745 |
| Last block scale-fitting objective | 0.01093 | 0.01000 |
| Last block Q, host → BCA | 207.10 → 191.92 | 232.63 → 216.31 |
| Last block critic loss, host → BCA | 55.42 → 61.31 | 20.30 → 24.75 |

TD3 Q is Q1 at proposed actor actions; ReBRAC Q is the twin minimum at recorded actions. They are not equivalent optimism measures. Actor skip placeholders are excluded. Lower Q is not proof of more accurate Q, and these losses are not true-Q error. Fits and multipliers establish that BCA operated, not that its uncertainty is calibrated or useful. Neither Bayesian nor conformal is removed.

## CQL and remaining gaps

- The deck plots CQL Walker Q1, Bellman loss, conservative objective and consumed penalty multiplier from 1,000 sparse last-row snapshots. These are preliminary, not complete per-update logs; independent checkpoint review remains pending.
- Numerical historical CQL radius values are absent from the extracted refresh records. No history was fabricated. Successful refresh records and sparse fit acceptance do not substitute for independent counters.
- This is the standard equal-fitting-weight comparison. It is not a new importance-weighting treatment, so no claim about importance-weight ESS or weighted-conformal validity is made from it.
- IQL and the remaining host/dataset combinations are not represented as having complete non-return metric readouts here. Training completion and shared-Q/V actor results do not establish those diagnostic analyses or OOD closure.
- The revised OOD snapshot still has five active pairs and zero completed pairs. Fresh coverage requires a separately accepted response/target and sampling/independence contract; the behavioral panel alone cannot supply that claim.

The next analysis should report warning quality, support-stratified harm, local decision quality and calibration validity alongside performance. It should not attribute a failure to excessive coverage or assume importance weighting repairs it.
