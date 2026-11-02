# Replacement design gate: zero-thrust slab on the first thruster (design record)

Written 28 October 2026, before this gate reads any data. It replaces the stall-knee band, shelved in
`../conditional_ood_design_v3/SHELVED_V2_BAND_20261028.md`, as candidate A of
`../conditional_ood_design_v3/STAGE0_OPTIONS_MEMO.md`.

**Authorization.** The user wrote "Make the replacement". These numbers were set by an agent under that instruction,
each by the rule stated below. The user's manual launch is the approval to run them. Any change before launch means
editing this record and the contract before the freeze. Any change after launch needs a new, labeled version.

This is a no-training development screen. It reuses the repaired evaluator of attempt 2
(`../conditional_ood_gate_v1_retry_20261028/`). It does not test BCA, train any model or establish a missing-label
effect.

## The region

Withhold the logged training rows where the first thruster's command `a0` is within `h` of zero, at states where the
observed water load `g` (current plus first wave coordinate) is at least 0.4. The same commands stay labeled at lower
load, and all other commands stay labeled at high load.

## Every number, its rule, and what was known when it was set

| Number | Value | Rule | Status |
|---|---|---|---|
| Band center | 0.0 | Mechanics: zero thrust is where the first thruster switches from opposing the water to assisting it. The hole is symmetric about the recorder's mean logged command, so the behavior-cloning target at in-region states is unchanged. | New |
| Half-width `h` | 0.1 | The smallest of {0.05, 0.10, 0.15, 0.20} at which the options memo's pre-recorded value model predicts a gap of at least the 1-unit practical size between the band center and a flank center `2h` away. The memo states the gap as about `10 × h` return units. Constraint: `3h < k`, so no alternative band reaches the stall knee. | New |
| Load threshold | 0.4 | Unchanged from the shelved design. Only what attempt 2 showed broken is changed. Load-depth strata are reported. | Unchanged |
| Alternative bands | `[−3h, −h)` and `(h, 3h]` | Same construction as before, around the new center. | Same rule |
| Candidates | actor, −0.2, `nextafter(float32(−0.1), 0)` (about −0.099999994), 0.0, `nextafter(float32(0.1), 0)` (about +0.099999994), +0.2 | The same six-slot construction as before: actor, left flank center, lower interior edge, center, upper interior edge, right flank center. The other three commands are held at the actor's own. | Same rule |
| Support rule | Each anchor's 64 nearest training rows in the same ten-step bin and load side, with no fixed radius. **Locality:** the anchor passes if its distance to the 64th neighbor is at most the 0.95 quantile of the same statistic over training rows at the same time and side. Each reference row's own episode is excluded from its pool. **Counts:** as before. | Replaces radius 0.5, which attempt 2 showed cannot pass in 13 standardized coordinates. The threshold comes from training-state geometry only. It is computed inside the gate from training rows, never from anchors or outcomes. | New |
| Support count criteria | 32 rows from 8 episodes, before and after removal; 2 band rows from 2 episodes; 2 retained rows from 2 episodes in each adjacent band; 80% of anchors per load side; at least 16 anchors per side | Unchanged | Unchanged |
| Neighbor count | 64 | Unchanged | Unchanged |
| Value gate | Center-to-flank gap of at least 1 return unit at 25% of high-load anchors | Unchanged | Unchanged |
| Exposure gate | At least 100 high-load free steps, entry on at least 1% of them, in at least 2 episodes | Unchanged | Unchanged |
| Coverage, balance and calibration criteria | As before | Unchanged | Unchanged |
| Anchors | The last 64 of the 128 remaining development tuning-validation groups, sorted by group ID, at times 0, 25, 50 and 75 | Attempt 2 used the first 64. These were never anchors. | New episodes |
| Free episodes | The first 16 of the new anchors' start seeds | Same rule | Same rule |
| Random-removal seed | 2026102901 | The first unused value in the 20261029xx range. A repository search found none in use. | New |
| Calibration replacement seed | 2026102902 | Same search | New |
| Calibration original rows | Seed 2026102362 | The identity of the original calibration rows | Unchanged |
| Budget | 900 seconds; 28,104 simulator transitions; zero updates | Unchanged. The support reference adds CPU distance arithmetic only. | Unchanged |

**What was known when these were set:**
- the saved actor's 128 development-bank commands and the memo's tallies of them;
- the memo's fitted value model;
- attempt 2's full report: support distances, value gaps around −0.8, and free-actor entry counts for the old band.

None of these values was set from a count of how many anchors or free steps would fall in this region, or how wide
the 64-neighbor distances are. The half-width rule uses the memo's model prediction, not attempt 2's measured gaps,
which were around a different center. The 0.95 quantile is a declared convention. It makes the locality check
certify that an anchor is as well surrounded as typical training states. It is not a severe test of state support.

## Predictions, before any data

These are analytic expectations from the recorder and start law, not measurements.

| Quantity | Expected |
|---|---|
| Hole rows in the 76,800-row training pool | about 3,840 (5%) |
| Training episodes with at least one hole row | about 663 of 768 |
| Hole rows per ten-step bin | about 384 |
| Hole rows by load depth [0.4, 0.5), [0.5, 0.6), at least 0.6 | about 1,120, 960 and 1,760 |
| Retained band rows at low load | about 3,840 |
| Original calibration rows inside the hole | about 6.4 of 128 |
| Rows in each of the band and its two neighbors, per 64 neighbors | 6.4 expected; at least 2 with probability about 0.99 |
| Anchors passing locality | about 95% by construction, if anchors are typical recorded states |

**Value and exposure** are model expectations, not measurements:
- **Value gap.** The memo's model gives about 1.0 unit between 0 and −0.2, with a 10th to 90th percentile of 0.36 to
  1.56. So a pass on the 25% hurdle is likely but not certain.
- **Exposure.** The saved actor's first command was within 0.1 of zero at 9 of 59 high-load bank states. So free entry
  well above 1% is likely, but its own trajectories have not been measured.

## Kill rule and reading

- **Pass.** The design passes only if every integrity, coverage, balance, support, value and exposure criterion passes.
  A pass permits proposing a separately reviewed training specification. It is not BCA efficacy.
- **Fail.** Any failure shelves this exact region. There is no change to the band, the threshold, the quantile or the
  neighbor count, and no automatic rerun. A failure is written up in the standing format: The idea / What we tested
  (and the kill rule) / What happened / Why.
- **Order of components.** The three components finish once, under the same cap, even if geometry fails.

## What changed in code, relative to attempt 2

- **`geometry.py`:** the region constants and the band center, a new support rule (`support_report` and the reference
  quantile), and the new seeds.
- **`reference.py`:** the candidate bank and the free-rollout hole test now use the geometry constants.
- **`analysis.py`:** the local pass adds the locality condition, and the load-depth edges come from geometry.
- **`worker.py`:** the anchors are the last 64 validation groups.
- **`launch.py`:** v1's launcher, with no retry lineage.
- **Tests:** updated for the new region and the support rule, and the retry's lineage test is removed.
