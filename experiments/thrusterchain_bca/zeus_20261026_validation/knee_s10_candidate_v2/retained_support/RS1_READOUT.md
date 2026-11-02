# RS1, candidate knee_s10_v2: state support against the retained rows, readout

Run 2026-11-01 on the formal G2 attempt-1 datasets (stall3 `c2a6d616...`, no-stall `e3549869...`, loaded by digest),
under `RS1_RULE.md` as written before the run. Record: `rs1_result.json` (code sha256 recorded there; 8 unit tests
pass). This readout reads the record against the rule. The one section marked post hoc holds numbers the record does
not; everything else is the record.

## Headline

**RETAINED SUPPORT FAILS**, on the start share in both worlds: 121 of 128 starts (0.9453) inside the primary t = 0
threshold, against a bar of 0.95, which needs 122. The anchor shares hold with room: 93 of 96 (stall3) and 91 of 96
(no-stall) against 87. By the rule as written, the r4 native package does not run on this candidate, the finding goes
to the ledger, and the definition decides what a retained-support miss means. RS1 changes no G2 label.

## Every number

The retained pool is 47,978 of 76,800 fitting rows in each world (0.6247; retention 0.5076 at high load and 0.7978 at
low load, against the recorder's 0.506 and 0.800). Deep-load rows: 12,600 of which 6,294 retained.

| Check | stall3 | no-stall | bar | result |
|---|---|---|---|---|
| Primary starts inside the t = 0 threshold | 121 of 128 (0.9453) | 121 of 128 (0.9453) | >= 0.95 (122) | **miss** |
| Primary anchors inside their own time's threshold | 93 of 96 (0.9688) | 91 of 96 (0.9479) | >= 0.90 (87) | pass |
| Primary anchors by time, t = 0 / 25 / 50 | 29, 32, 32 of 32 | 29, 32, 30 of 32 | | reported |
| Primary groups (rows, episodes, threshold) at t = 0 | 61, 61, 4.2099 | 61, 61, 4.1125 | floor 32 rows | above the floor |
| at t = 25 | 50, 50, 4.2304 | 50, 50, 4.0855 | | |
| at t = 50 | 63, 63, 3.5191 | 63, 63, 3.3676 | | |
| Instrument check: sealed all-rows D3 recomputed | starts 126 of 128, anchors 96 of 96 | starts 127 of 128, anchors 93 of 96 | equal to the formal report | equal |

Reported only, never deciding:

| Reading | stall3 | no-stall |
|---|---|---|
| Secondary (window +-2) starts inside | 126 of 128 | 127 of 128 |
| Secondary anchors inside | 96 of 96 | 93 of 96 |
| Secondary anchors by time, t = 0 / 25 / 50 | 32, 32, 32 | 32, 32, 29 |
| Secondary groups (rows, episodes, threshold) t = 0 / 25 / 50 | 168, 99, 4.7091; 298, 123, 4.3306; 314, 123, 3.5191 | 168, 99, 4.6197; 298, 123, 4.0855; 314, 123, 3.2716 |
| Sealed all-rows thresholds (126 rows each) t = 0 / 25 / 50 | 4.4843, 4.0096, 3.2543 | 4.4062, 3.6850, 3.1103 |
| Primary threshold over sealed threshold, t = 0 / 25 / 50 | 0.939, 1.055, 1.081 | 0.933, 1.109, 1.083 |
| Anchor distance to the retained pool over distance to all rows (median, min, max) | 1.068, 1.029, 1.115 | 1.065, 1.028, 1.104 |

Budget: 4,800 exact simulator steps of 20,000 (the 96 anchor prefixes in both worlds); 15.3 s of 600, of which 9.5 s
loading and validating the two datasets.

## What the numbers say

- **The secondary reproduces the sealed D3 exactly.** With about five times the rows per group (168 to 314 rows,
  99 to 123 episodes), the retained-pool check gives 126 and 127 of 128 starts and 96 and 93 of 96 anchors, the same
  counts the sealed all-rows D3 gave. That is the agreement the rule expected by construction: within a load band,
  retention is independent of the state, so the retained rows carry the same state distribution.
- **Half the density raised the distances by a steady factor.** Every anchor's distance to the retained pool is 1.03
  to 1.12 times its distance to all rows (median 1.07 in both worlds), and the secondary and the t = 25 and t = 50
  primary thresholds rose by 1.05 to 1.11, the same order.
- **The primary t = 0 threshold went the other way.** It is 0.94 of the sealed threshold (4.21 against 4.48 in
  stall3, 4.11 against 4.41 in no-stall), while the start distances rose like everything else. The primary group at
  t = 0 holds 61 rows, so its 0.99 "higher" quantile is the group's single largest leave-one-episode-out distance, as
  the rule said it would be for groups under 101 rows. That one draw sets the bar for all 128 starts, and here it
  drew low relative to the sealed second-highest-of-126.
- **The starts are reset states, identical in both worlds**, so both worlds miss by the same count. The anchor
  results differ between worlds only where the pools differ (t >= 1).

## Post hoc, not part of the record

Read from a scratchpad script after the run, to size the miss; it adds nothing to the label.

- The seven outside starts lie 1.04 to 1.14 times the threshold in stall3 (distances 4.39 to 4.79 against 4.21) and
  1.05 to 1.13 in no-stall (4.31 to 4.63 against 4.11). The largest inside distance is 4.17 and 4.04.
- The t = 0 group's three largest leave-one-episode-out distances are 4.21, 4.01, 3.95 (stall3) and 4.11, 3.91, 3.85
  (no-stall). At the second-largest, 115 starts would be inside in either world.

## What this leads to

By `RS1_RULE.md`: the r4 native package does not run on this candidate under the rule as written; the finding goes to
the ledger; what a retained-support miss means for the candidate is the definition's decision, not RS1's. The
observations above are for that decision, not a relabeling: the miss is one start of 128 at the primary t = 0 group,
where the threshold is a 61-row maximum; the scored states (the anchors) hold with room in both worlds; the
reported-only secondary, built from the same retained rows with wider time groups, agrees with the sealed D3 to the
count. Nothing here was rerun, and the primary's definition was fixed before any number existed.
