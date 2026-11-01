# Independent saved-report cross-check

E1 is the `knee_mix15` collection mixture, not a purely ordinary-current condition. E2 is the stress bank. The final policy tables should be described by their bank names, without claiming an isolated current-only effect.

The original per-seed validity failures remain:

| Criterion | Seed 0 | Seed 1 | Seed 2 | Frozen requirement |
| --- | ---: | ---: | ---: | --- |
| V0(b), RANDOM onset KC | 0.639756 | 0.615962 | 0.756602 | Absolute median at most 0.280442 / 0.281762 / 0.347645 respectively |
| V0(b), RANDOM deep-entry rate | 67.6172% | 69.1641% | 63.7344% | At most 10% |
| V0(b), RANDOM predictor-valid | 48/48 | 43/48 | 43/48 | At least 90% |
| V0(d), positive-side sign check | 4/48 | 4/48 | 3/48 | At least 90% |
| V0(d), K1a reach clause (i) | -0.064201 | -0.126335 | -0.129300 | At least +0.5 |
| V0(e), largest other-thruster knee-crossing count | 107, RANDOM | 65, CUTOFF | 79, RANDOM | At most 64/12,800 steps |

CUTOFF item-5 validity and the other K1 reach clauses pass. Some individual criteria passing does not override the original failed validity checks. The newly calculated V0(f) also fails in every seed.

Logical update accounting from the completed receipts is 240,000 critic updates, 120,000 native actor updates, 240,000 auxiliary updates and 24 calibrations across 12 warmups plus 12 continuations. Branch receipts are cumulative 20k counters: summing them with their warmup totals would double-count the first 10k. Three BC-only jobs add 30,000 BC updates. All 32 worker reports, including preflight, R0 and three evaluations, completed.

Reported stage time charges are 13,615.873 training seconds and 1,528.644 evaluation seconds. The reported evaluator count is 6,270,800; summing all worker simulator counters gives 6,305,360. Neither is certified as an exact physical total. The previously identified counter at `knee_edge_ood_v2/units.py:279` counts only `states.shape[0]`, while `dataset_knee.py:190` validates a `(1280, 100, 13)` batch: 128,000 transition checks are recorded as 1,280 for each such load. This analysis does not rewrite those receipts or silently correct aggregate physical spend without a full accounting audit.

The independent review reproduced episode means and contrasts from saved episode vectors, as well as the reported validity clauses. This is read-only arithmetic on existing reports. It added no scientific model updates, simulator calls or calibration work. Three training seeds remain the independent training replication; shared test episodes are not additional training replications.
