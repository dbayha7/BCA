# BCA-specific credit and power, from the records (20 October 2026)

Analysis of recorded numbers only. Nothing was trained or run. Every number below was recomputed from the files named
in section 9 with the scripts in this folder (`extract.py`, `analyze1.py` to `analyze5.py`; their outputs
`analyze1.out` to `analyze5.out`, `extracted.json`, `switches.json`, `net_diffs.json`, `power.json`).

**Labels.** Seeds 0-4 in every setting are exploratory (seen by every earlier study). Seeds 11 and 14 are description
only: they are the two of five fresh seeds on which BCA's certificate succeeded, a selected subset, and carry no
claim. The near-80 version 8 confirmatory claim 4 failed by its pre-set rule (seeds 10, 12, 13 stopped). No number
here is a pre-registered result for BCA.

**Words.** `obca_k1` = BCA's switched ramp (the primary). Points: 0 = the controller that made the log, 100 = the
world's stall-aware reference. NET across worlds = (arm minus no caution, stalling world) minus (no caution minus arm,
no-stall world), strong current, per seed (step 2's `net`, `restructure/step2/step2lib.py:396-404`). "Dose" = the
ramp's k times its height h (the critic's value units).

---

## 1. In short

1. **On the per-seed points, BCA's switched ramp is never beaten by a switched rival in the main setting, and the whole
   margin is the height, not the switch.** Every switched rival in the pipeline makes BCA's on/off call in (almost)
   every seed-world (section 3): the mean-difference switch makes exactly BCA's call in **40 of 40** recorded
   seed-worlds, the split-conformal switch in 37 of 40, the rank test in 36 of 40. The rivals differ from BCA in the
   stalling-world height alone, and within each seed the points rise with the dose (section 5): the five main-setting
   seeds 0-4 are strictly monotone over meangap (0.49-0.68) < fixed 1.0 < conformal (1.10-1.73) < BCA (1.61-6.02) <
   twice BCA. That is step 2's own recorded attribution ("a larger dose did better ... nothing is attributed to the
   Bayesian part", `restructure/step2_results/results.md`, A1-A3), and seeds 11 and 14 repeat it.
2. **The margin over the split-conformal switch is small and unstable where the heights are close, large where they
   are not.** Main seeds 0-4: BCA minus conformal NET +6.3, +3.7, +6.5, +7.9, +3.5 (mean 5.6, sd 1.9). Near-80 seeds
   0-4: -9.3, +0.6, +9.5, +3.9, +1.1 (mean 1.1, sd 6.8; conformal is ahead on seed 0, where BCA's height 5.94 is
   4.4 times conformal's). Seeds 11 and 14: +27.6 and +25.4, because conformal's height there (0.892, 0.881) is the
   smallest of any trained stalling seed-world and BCA's (2.29, 4.60) is 2.6 and 5.2 times larger (section 4).
3. **A rule with no BCA part in it does as well or better on the recorded arms.** The mean-difference switch (on when
   the held-out full-throttle group's mean critic error exceeds the rest's by 0.10) gating the hand-set range limit,
   built from the recorded arms (range limit where the switch is on, no caution where it is off), nets 107.4 on main
   seeds 0-4 against BCA's 102.9, 103.0 against 100.6 on near-80 seeds 0-4, and 108.6 / 110.0 against 106.3 / 109.5 on
   seeds 11 / 14 (section 6). BCA is behind it on 8 of 12 seed-settings; the reverse one-sided test ("it beats BCA")
   has 48% power at 5 seeds. This construct was never trained as one arm; both halves are recorded arms.
4. **Where BCA could not act it was its own credibility rule, not the data, that stopped it.** Seeds 10, 12, 13: BCA
   on with an infinite height (19, 22, 18 held-out full-throttle rows; finite-sample share 0.879, 0.903, 0.850 below
   0.95). The split-conformal and mean-difference switches had finite heights on the same seeds (conformal 2.71, 1.29,
   1.52; mean gap 0.71, 0.59, 0.58) and would have trained (section 4.3). What their points would have been is unknown
   (the phases stopped before any arm trained).
5. **Power (section 7).** "BCA beats the split-conformal switch by m", one-sided paired t at alpha 0.05, 80% power:
   with the main seeds 0-4 spread (mean 5.6, sd 1.9): m = 0 needs 3 seeds, m = 3 needs 5, m = 5 needs 65, m = 10 is
   unreachable (mean below m). With the near-80 spread (mean 1.1, sd 6.8): m = 0 needs 225 seeds, m >= 3 unreachable.
   With all seven main-setting seeds (mean 11.6, sd 10.3; 11 and 14 are a selected subset): 7, 11, 17, 275. Against
   the best non-BCA switch the test is the same test (the split-conformal switch is the best non-BCA switch in every
   setting; the others trail by 38-55 points). Against the mean-difference-gated range limit, "BCA beats it" is
   unreachable at every m (BCA's mean is negative); "it beats BCA by 0" needs 10 seeds on the main 0-4 spread.

---

## 2. Per-seed points, strong current (the pre-registered situation), both worlds

Columns: seeds in order; `|` separates the stalling world (left) from the no-stall world (right). Source: near-80
version 8 `files/eval/<setting>/<world>/seed<k>/values.json` (`evaluations.<arm>.strong_current.points`). The
main-setting seeds 0-4 values in that folder equal step 2's `restructure/step2_results/results.json` `points` to the
last digit (largest difference 0; `analyze1.out` line 2), so the earlier restructure step-2 records and the near-80
main setting on seeds 0-4 are the same numbers (step 2's saved actors re-evaluated, `near80/PREREGISTRATION.md` 5.4).

### 2.1 Main setting, seeds 0-4 (= restructure step 2; exploratory)

| arm | stalling world s0..s4 (mean) | no-stall world s0..s4 (mean) |
|---|---|---|
| none (no caution) | -13.0, -21.5, -13.0, -13.4, -10.7 (-14.3) | 96.1, 94.1, 95.0, 95.1, 94.4 (94.9) |
| obca_k1 (BCA's switched ramp) | 94.2, 94.2, 88.1, 87.4, 79.4 (88.6) | = none (94.9) |
| obca_k2 (twice BCA's height) | 98.7, 95.9, 89.1, 93.7, 84.2 (92.3) | = none |
| obca_forced (always-on ramp) | = obca_k1 (88.6) | 63.2, 70.6, 72.8, 74.5, 67.4 (69.7) |
| fixed_gated (BCA's switch, height 1.0) | 72.3, 79.9, 79.1, 72.6, 71.2 (75.0) | = none |
| conformal_k1 (split-conformal switch) | 87.8, 90.5, 81.6, 79.5, 75.9 (83.0) | = none |
| meangap_k1 (rank-test switch) | 45.7, 56.1, 44.5, 32.4, 36.9 (43.1) | 96.1, 94.1, 94.2, 95.1, 94.4 (94.8) |
| perfect_band (reference) | 77.6, 82.4, 83.5, 82.1, 72.7 (79.7) | 94.8, 93.7, 93.7, 93.8, 92.1 (93.6) |
| range_hinge (hand-set range limit) | 93.6, 93.0, 93.7, 95.7, 89.8 (93.1) | 61.5, 61.3, 70.2, 65.3, 61.8 (64.0) |
| standin (matched range limit) | 91.5, 86.3, 88.1, 61.6, 64.6 (78.4) | 62.8, 72.1, 75.6, 79.2, 70.7 (72.1) |
| even_pull (matched) | -13.0, -17.2, -10.4, -15.7, -4.3 (-12.1) | 85.1, 85.0, 88.0, 86.3, 86.2 (86.1) |
| full_none (full-log controller) | 82.2, 79.0, 89.8, 83.9, 88.2 (84.6) | - |
| clone | 51.5, 61.8, 56.5, 47.6, 54.4 (54.4) | -48.1, -34.8, -32.2, 6.9, -13.9 (-24.4) |

(`analyze1.out`; push_less, random_1, random_2 there too.)

### 2.2 Near-80 setting, seeds 0-4 (exploratory)

| arm | stalling world s0..s4 (mean) | no-stall world s0..s4 (mean) |
|---|---|---|
| none | -11.4, -23.2, -10.8, -15.1, -5.7 (-13.2) | 94.8, 93.0, 93.0, 93.9, 91.1 (93.2) |
| obca_k1 | 81.2, 91.6, 83.8, 88.2, 92.1 (87.4) | = none (93.2) |
| obca_k2 | 80.7, 89.5, 84.6, 82.5, 90.2 (85.5) | = none |
| obca_forced | = obca_k1 (87.4) | 29.0, 49.6, 32.8, 38.1, 58.6 (41.6) |
| fixed_gated | 81.6, 85.3, 74.5, 84.8, 87.2 (82.7) | = none |
| conformal_k1 | 90.5, 90.9, 74.4, 88.8, 91.1 (87.1) | 94.8, 93.0, 93.0, **89.4**, 91.1 (92.3) |
| meangap_k1 | 56.8, 52.2, 42.7, 46.4, 50.4 (49.7) | 94.8, 93.0, 90.4, 93.9, 90.4 (92.5) |
| meandiff_k1 (mean-difference switch) | = meangap_k1 (49.7) | = none (93.2) |
| knee_ramp (BCA's switch and height, ramp from 80%) | 89.8, 82.6, 88.0, 91.8, 86.2 (87.7) | = none |
| perfect_band | 83.5, 75.6, 88.1, 88.9, 88.3 (84.9) | 94.5, 94.0, 92.8, 92.2, 91.1 (92.9) |
| range_hinge | 89.5, 91.4, 90.5, 85.8, 91.7 (89.8) | 30.9, 57.5, 43.8, 31.1, 60.7 (44.8) |
| standin | 88.1, 92.2, 89.8, 88.8, 93.6 (90.5) | 17.4, 60.3, 43.8, 48.5, 65.5 (47.1) |
| knee_limit (range limit at 80%) | 80.7, 68.0, 75.6, 88.6, 80.9 (78.8) | 78.4, 82.8, 77.7, 78.4, 81.5 (79.8) |
| full_none | 85.9, 64.9, 88.1, 85.9, 86.6 (82.3) | - |
| version6_bca | -11.8, -22.3, -10.5, -14.7, -5.8 (-13.0) | 95.1, 92.9, 93.0, 93.7, 90.8 (93.1) |

In the near-80 setting meangap_k1 and meandiff_k1 have the same stalling-world actor in every seed (same switch, same
height = the mean gap; `files/policy/near80/stall/seed*/lock.json` `arm_table`).

### 2.3 Main setting, seeds 11 and 14 (description only; a selected subset of the five fresh seeds)

| arm | stalling world s11, s14 | no-stall world s11, s14 |
|---|---|---|
| none | -17.2, -13.5 | 90.4, 95.2 |
| obca_k1 | **89.1, 96.0** | = none |
| obca_k2 | 87.9, 97.7 | = none |
| obca_forced | = obca_k1 | 60.7, 67.1 |
| fixed_gated | 68.4, 74.7 | = none |
| conformal_k1 | **61.5, 70.6** | = none |
| meangap_k1 | 26.4, 48.9 | 90.4, 94.0 |
| perfect_band | 81.4, 84.7 | 88.5, 94.5 |
| range_hinge | 91.4, 96.4 | 45.8, 64.5 |
| standin | 76.3, 92.4 | 60.9, 67.7 |
| even_pull | -19.1, -9.0 | 79.1, 84.9 |
| full_none | 80.7, 84.8 | - |

### 2.4 The other three situations (stalling world, BCA minus rival, mean over seeds [sd]; `analyze3.out`)

| rival | main 0-4: recorded / big waves / big kicks | near-80 0-4 | seeds 11, 14 |
|---|---|---|---|
| conformal_k1 | 0.4 [0.2] / 0.9 [0.5] / 3.0 [1.1] | 0.1 / 0.5 / 4.5 | 0.8 / 2.3 / 8.3 |
| fixed_gated | 0.6 / 1.8 / 5.2 | 0.2 / 0.6 / 7.0 | 0.8 / 1.9 / 7.3 |
| meangap_k1 | 1.4 / 3.9 / 9.7 | 1.0 / 2.2 / 13.1 | 1.9 / 4.9 / 12.2 |
| obca_k2 | -0.4 / -0.8 / -2.1 | -0.4 / -0.4 / -1.9 | -0.3 / -0.6 / -1.9 |
| range_hinge | 0.2 / -0.3 / -2.5 | -0.3 / -0.1 / -0.4 | 0.1 / -0.4 / -1.7 |
| none | 8.0 / 21.2 / 19.8 | 6.2 / 16.5 / 22.8 | 8.5 / 21.2 / 21.8 |

The ordering of the strong-current situation holds in the other three at a smaller scale (big kicks largest). In
the no-stall world every switched arm scores exactly no caution's points wherever its switch is off; the always-on
arms lose 0.4-2.5 (recorded), 2.8-8.7 (waves), 2.9-8.2 (kicks) there.

---

## 3. Every switch's decision and height, all 40 recorded seed-worlds

Source: `files/learners/<setting>/<world>/seed<k>/switch.json` (`entry.certificate`, `entry.groups`,
`entry.rank_test`) for near-80 v8 (main 10-14, near80 0-4, random_rows 0-4) and `restructure/step1/results/results.json`
`per_seed_world` for main 0-4; rules from `restructure/step2/step2lib.py:315-365` (`arm_table`) and
`near80/code/policy_phase.py:121-123` (meandiff: mean gap > 0.10). Full table: `analyze2.out`.

| switch | its statistic | agrees with BCA's call | right call (on where it stalls, off where it does not) | disagreements |
|---|---|---|---|---|
| BCA's certificate (q05 of the posterior gap > 0.10) | q05: stalling 0.56-1.26; no-stall -0.047-0.053 | - | 40 of 40 | none, but 3 refusals (on, height infinite): main/stall 10, 12, 13 |
| mean-difference (mean gap > 0.10) | stalling 0.466-0.729; no-stall -0.031-0.066 | **40 of 40** | 40 of 40 | none |
| split conformal (90% quantile gap > 0.10) | stalling 0.881-2.708; no-stall -0.033-0.194 | 37 of 40 | 37 of 40 | wrongly on: main/no_stall 10 (0.194), 13 (0.112), near80/no_stall 3 (0.103) |
| rank test (p < 0.05) | - | 36 of 40 | 36 of 40 | wrongly on: main/no_stall 2 (step 1), 14; near80/no_stall 2, 4 |

The margin 0.10 sits inside a wide empty band for the mean gap (no-stall maximum 0.066, stalling minimum 0.466) and
for BCA's q05 (0.053 to 0.559). The two statistics make the same call because the full-throttle group's mean error
is 0.47-0.73 above the rest's in every stalling world and within 0.07 of it in every no-stall world; BCA's posterior
adds nothing to the decision on these 40 seed-worlds. The one thing BCA's posterior does differently is to refuse at
small group sizes: the certificate needs at least 95% finite posterior samples, which needs about 29 rows
(`near80_results_check/REPORT.md`); seeds 10, 12, 13 had 19, 22, 18.

Heights where the switch is on (stalling worlds), per seed:

| setting / seed | BCA h | conformal h | mean gap (meangap, meandiff) | fixed | BCA / conformal |
|---|---|---|---|---|---|
| main 0, 1, 2, 3, 4 | 6.019, 2.420, 2.065, 1.605, 1.796 | 1.733, 1.446, 1.096, 1.199, 1.170 | 0.657, 0.681, 0.533, 0.492, 0.539 | 1.0 | 3.47, 1.67, 1.88, 1.34, 1.54 |
| near80 0-4 | 5.943, 2.558, 3.145, 1.915, 1.411 | 1.359, 1.626, 1.015, 1.344, 1.200 | 0.623, 0.630, 0.561, 0.506, 0.523 | 1.0 | 4.37, 1.57, 3.10, 1.42, 1.18 |
| main 11, 14 | 2.289, 4.595 | 0.892, 0.881 | 0.466, 0.615 | 1.0 | 2.57, 5.22 |
| main 10, 12, 13 (stopped) | inf, inf, inf | 2.708, 1.290, 1.520 | 0.706, 0.589, 0.583 | 1.0 | - |

---

## 4. Per rival: paired difference from BCA's switched ramp (NET across worlds, strong current)

Per-seed BCA minus rival; mean, sd over seeds; and what makes the difference. Source: `analyze3.out`,
`net_diffs.json`. The matched rivals' step-2 NETs equal `restructure/step2_results/results.json` `net_strong_current`
(e.g. conformal_k1 97.33, fixed_gated 89.30, meangap_k1 57.24, obca_forced 77.66, range_hinge 76.49, standin 69.87).

| rival | main 0-4: per seed; mean (sd) | near-80 0-4: per seed; mean (sd) | seeds 11, 14 | decision or height? |
|---|---|---|---|---|
| conformal_k1 | +6.3, +3.7, +6.5, +7.9, +3.5; **5.6 (1.9)** | -9.3, +0.6, +9.5, +3.9, +1.1; **1.1 (6.8)** | +27.6, +25.4 | **height** in 11 of 12 seed-settings (same decision as BCA); near-80 seed 3 also a wrong "on" in the no-stall world (h 0.103, cost 4.4 of that seed's +3.9) |
| fixed_gated | +21.9, +14.3, +8.9, +14.8, +8.3; 13.6 (5.5) | -0.4, +6.2, +9.3, +3.4, +4.9; 4.7 (3.6) | +20.7, +21.3 | **height only** (BCA's own switch) |
| meangap_k1 | +48.4, +38.1, +44.4, +55.0, +42.6; 45.7 (6.4) | +24.4, +39.4, +43.7, +41.8, +42.4; 38.4 (7.9) | +62.7, +48.2 | **height** (mean gap 0.47-0.73); wrong "on" in 4 no-stall seed-worlds costs 0.7-2.6 each |
| meandiff_k1 (near-80 only) | - | +24.4, +39.4, +41.1, +41.8, +41.7; 37.7 (7.5) | - | **height only** (identical decision to BCA in all 10 near-80 seed-worlds) |
| obca_k2 | -4.6, -1.7, -1.0, -6.3, -4.7; -3.7 (2.2) | +0.5, +2.1, -0.8, +5.7, +1.9; 1.9 (2.4) | +1.2, -1.7 | **height only** (doubled): BCA's height lies on the rising part of the dose curve in the main setting, at or past its top in near-80 |
| obca_forced | +32.9, +23.6, +22.2, +20.6, +27.0; 25.3 (4.9) | +65.9, +43.4, +60.2, +55.8, +32.5; 51.5 (13.5) | +29.7, +28.1 | **decision only**: the stalling-world actor is BCA's own (difference 0.0 in every seed); the whole gap is the no-stall world |
| knee_ramp (near-80) | - | -8.6, +9.0, -4.2, -3.6, +5.9; -0.3 (7.4) | - | neither: BCA's switch and height, ramp started at 80% instead of 70% (a physics reference) |
| perfect_band (reference) | +17.9, +12.2, +6.0, +6.6, +9.0; 10.3 (4.9) | -2.0, +14.9, -4.0, +1.0, +3.9; 2.8 (7.4) | +9.6, +11.9 | different penalty (knows the physics); not a switch comparison |
| range_hinge (hand-set range limit) | +35.2, +34.1, +19.2, +21.5, +22.2; 26.4 (7.6) | +55.6, +35.6, +42.6, +65.2, +30.8; 46.0 (14.3) | +42.3, +30.2 | **decision** (always on: loses 24.8-63.9 in the no-stall world); in the stalling world it is level with or ahead of BCA (BCA minus it -4.5 (5.2), -2.4 (4.8), -1.4) |
| standin (matched range limit) | +35.9, +29.9, +19.4, +41.7, +38.5; 33.1 (8.8) | +70.5, +32.1, +43.3, +44.8, +24.1; 43.0 (17.6) | +42.3, +31.1 | decision (always on) plus a weaker stalling-world limit on main seeds 3-4 (BCA ahead by 25.8, 14.8 there) |
| knee_limit (near-80) | - | +16.9, +33.7, +23.5, +15.1, +20.9; 22.0 (7.3) | - | decision (always on; loses 9.7-16.4 no-stall) plus a weaker stalling limit |
| even_pull, push_less, random_1, random_2 | 105-110 | 105-114 | 110-120 | matched controls: no caution's level |
| full_none (stalling world only, points) | +4.0 (9.8) | +5.1 (12.8) | +9.8 | - |

Reading: among the arms that share BCA's pipeline, every one with BCA's decision and a smaller height scores lower,
and the two with a larger or equal height (obca_k2, knee_ramp) score the same within noise. No comparison in the
records separates a *calibrated* height from a *larger* height: no rival's height reaches BCA's in any stalling
seed-world (step 2's `attribution.calibration_credit.no_rival_height_reaches_bca: true`).

---

## 5. Mechanism: why the split-conformal switch scored 61.5 / 70.6 on seeds 11 / 14 against BCA's 89.1 / 96.0

Source: `files/learners/main/stall/seed11/switch.json` and `seed14/switch.json`; `files/policy/main/stall/seed11/lock.json`
and `seed14/lock.json` (`arm_table.arms`); `files/eval/main/stall/seed1{1,4}/values.json` (`points`,
`unrecorded_throttle_share`, `loss_from_unrecorded_throttle`, `capped_points`; the last two are defined in
`near80/code/evaluate_phase.py:6-11`: `capped_points` is the same actor capped at 80% by the recorder, and the loss
is capped minus points).

**Decision.** On both seeds both switches were on: conformal gap 0.892 > 0.10 (seed 11), 0.881 (seed 14); BCA on with
q05 0.760 and 0.675. Same decision, so the decision explains nothing.

**Height.** Conformal's height is the 90% quantile gap, 0.892 and 0.881: the two smallest conformal heights of any
trained stalling seed-world (seeds 0-4: 1.10-1.73 main, 1.02-1.63 near-80). BCA's height is R_tail minus R_rest from
the weighted Bayesian quantile (credibility 0.95): 2.289 (seed 11, 53 rows: R_tail 2.262, R_rest -0.027) and 4.595
(seed 14, 30 rows). Ratios 2.57 and 5.22.

**What the small height did to the actor (seed 11, stalling world, strong current).** Share of steps past the
recorded range and the loss from them:

| arm | dose | points | past-80 share | loss from unrecorded throttle | capped at 80% |
|---|---|---|---|---|---|
| none | - | -17.2 | 0.747 | 116.9 | 99.8 |
| meangap_k1 | 0.466 | 26.4 | 0.546 | 72.2 | 98.6 |
| conformal_k1 | 0.892 | **61.5** | 0.492 | **37.7** | 99.2 |
| fixed_gated | 1.000 | 68.4 | 0.504 | 30.6 | 99.0 |
| obca_k1 | 2.289 | **89.1** | 0.396 | **10.0** | 99.1 |
| obca_k2 | 4.578 | 87.9 | 0.454 | 11.5 | 99.4 |
| range_hinge | - | 91.4 | 0.329 | 7.4 | 98.8 |

Seed 14: conformal dose 0.881, points 70.6, loss 32.1; fixed 1.0: 74.7; BCA 4.595: 96.0, loss 5.5; k2: 97.7.
Every arm capped at 80% scores 98.6-102.7, so the whole spread between arms is how often and how far the actor goes
past 80%: conformal's ramp was too low to keep it out (loss 37.7 / 32.1), BCA's was high enough (10.0 / 5.5). On seed
11 conformal sits exactly where its dose puts it on the ladder (below fixed 1.0, which scored 68.4): the dose ladder
is monotone on seed 14 and on all five main seeds 0-4, and the within-seed line through all 65 ramp arms is about
17.3 points per unit of ln(dose) (R^2 0.63, `analyze4.out`); it predicts a BCA-conformal gap of 16.3 on seed 11 and
28.6 on seed 14 against 27.6 and 25.4 observed.

**Why the heights differ so much at these sizes.** Both statistics read the same 53 (30) held-out scores. Conformal's
is the empirical 90% quantile difference. BCA's is the 95%-credible weighted quantile, which with 30-53 rows and
alpha 0.1 lands near the top one or two scores of the full-throttle group, far above the 90% point; the fewer the
rows, the larger the gap between the two (seed 14, 30 rows: 4.60 against 0.88; seed 11, 53 rows: 2.29 against 0.89).
So conformal's score on seeds 11/14 is not a new failure mode: it is the same dose curve seen on seeds 0-4, read at a
lower dose because these two seeds' 90% gaps were smaller (0.88-0.89 against 1.10-1.73), while BCA's height grew as
the group shrank. (The near-80 setting shows the other end of the same curve: on seed 0 BCA's 5.94 was past the top
and scored 81.2 against conformal's 90.5 at 1.36.)

**Seeds 10, 12, 13 (stopped before training).** Conformal's switch was on with finite heights 2.708, 1.290, 1.520
(the first larger than any seeds-0-4 conformal height); the mean-difference switch on with 0.706, 0.589, 0.583;
the rank test on (p 5.6e-6, 9.5e-10, 5.7e-7). Only BCA's arm was undefined (on, height inf). `step2lib.arm_table`
builds all six ramp arms together and refuses the whole table on one undefined arm (`step2lib.py:355-357`;
`files/stops/policy.main.stall.s1{0,2,3}.json`), so no rival trained either. In the no-stall world conformal was also
wrongly on for seeds 10 (0.194) and 13 (0.112); at near-80 seed 3 a wrong "on" of 0.103 cost 4.4 points, so those
two would have cost conformal a few points had they trained. No points exist for any arm on these three seeds.

---

## 6. The rule with no BCA part: the mean-difference switch gating a range limit (built from recorded arms)

Construct (never trained as one arm; both halves are recorded arms on the same seeds): where the mean-difference
switch is on (every stalling seed-world) take the range limit's recorded actor; where it is off (every no-stall
seed-world) take no caution. Its NET is therefore the range limit's stalling gain. Source: `analyze3.out`,
`analyze5.out`.

| gated arm | main 0-4: NET per seed (mean) vs BCA | near-80 0-4 | seeds 11, 14 |
|---|---|---|---|
| hand-set range limit (range_hinge) | 106.6, 114.4, 106.6, 109.0, 100.5 (**107.4**) vs 102.9; BCA minus it +0.6, +1.2, -5.6, -8.3, -10.4 (mean -4.5, sd 5.2) | 100.9, 114.6, 101.3, 100.9, 97.5 (**103.0**) vs 100.6; -8.3, +0.1, -6.7, +2.5, +0.4 (-2.4, 4.8) | **108.6, 110.0** vs 106.3, 109.5; -2.3, -0.4 |
| matched range limit (standin) | 104.5, 107.8, 101.1, 74.9, 75.3 (92.7); BCA ahead 10.2 (10.4) | 99.5, 115.4, 100.6, 103.9, 99.3 (103.8); BCA behind 3.1 (3.1) | 93.5, 105.9; BCA ahead 8.2 |
| range limit at 80% (knee_limit, near-80) | - | 92.2, 91.2, 86.4, 103.7, 86.6 (92.0); BCA ahead 8.6 (9.7) | - |

The lost analysis's figures (107.4 vs 102.9; 108.6 / 110.0 vs 106.3 / 109.5; the switch making BCA's call in every
recorded seed-world) are confirmed; the agreement count is 40 of 40 seed-worlds, not 20 (it includes seeds 10-14 and
the random-rows setting). The hand-set range limit's stalling-world actor is ahead of BCA's on 8 of 12 seed-settings
and never more than 2.3 behind on seeds 11/14. In the other three situations the gated range limit and BCA are
within 0.4 (recorded), 0.4 (waves), 2.5 (kicks) of each other in the stalling world (section 2.4), and identical in
the no-stall world (both are no caution there).

What this construct does not show: (i) it was never trained as one arm, so no interaction with training is recorded
(the two halves are separate recorded actors, each the pre-registered arm on that seed-world); (ii) the range limit's
coefficient was hand-set and the band test showed it loses 17.5-19.0 points when the good commands below the stall
are hidden and the data stop at 70% (near-80 `PREREGISTRATION.md` risks; the recorded near-80 figures are 89.8 against
BCA's 87.4 stalling, so on seeds 0-4 it did not lose that); (iii) it says nothing about settings with a shift that is
not a hard edge in the recorded commands.

---

## 7. Power

Test: paired one-sided one-sample t on the per-seed differences d_i = NET(BCA) - NET(rival), H1: mean > m, alpha
0.05, noncentral t (`scipy.special.nctdtr`; `scipy.stats` does not load on this machine because an Application
Control policy blocks one of its DLLs, so the noncentral t was taken from `scipy.special`, checked at one point). n
is the smallest number of seeds with power >= 0.80 under the stated mean and sd. Full output: `analyze5.out`,
`power.json`. Sign and exact Wilcoxon one-sided p-values are given for the recorded sets.

### 7.1 BCA beats the split-conformal switch (the best non-BCA switch in every setting)

| spread taken from | mean, sd (n) | m = 0 | m = 3 | m = 5 | m = 10 | power at 5 seeds (m = 0, 3, 5, 10) |
|---|---|---|---|---|---|---|
| main seeds 0-4 | 5.60, 1.91 (5) | 3 | 5 | 65 | unreachable | 1.00, 0.80, 0.15, 0.00 |
| near-80 seeds 0-4 | 1.14, 6.81 (5) | 225 | unreachable | unreachable | unreachable | 0.09, 0.02, 0.00, 0.00 |
| main 11, 14 (selected) | 26.51, 1.55 (2) | 2 | 2 | 2 | 2 | 1.00 x4 |
| main 0-4 + 11, 14 (7) | 11.57, 10.34 (7) | 7 | 11 | 17 | 275 | 0.66, 0.46, 0.32, 0.09 |
| all 12 seed-settings (0-4 counted twice) | 7.23, 10.20 (12) | 14 | 38 | 135 | unreachable | 0.37, 0.19, 0.11, 0.02 |

Recorded sign tests: main 0-4, 5 of 5 positive (p 0.031); near-80, 4 of 5 (p 0.19); 7 main seeds, 7 of 7 (p 0.008);
12 seed-settings, 11 of 12 (p 0.003). The stalling-world-only version gives the same numbers in the main setting
(no-stall losses are 0 for both arms) and mean 0.25, sd 6.7 in near-80.

The honest planning spread is not the main 0-4 one: the near-80 setting and seeds 11/14 show that the BCA-conformal
difference moves with the ratio of the two heights (section 5), which varies 1.2-5.2 across seeds. For a planning
mean of 5.6 and sd 10 (the pooled sd), m = 0 needs 22 seeds, m = 3 needs 95; for mean 11.6 and sd 10: 7, 10, 16,
250. "Best non-BCA switch": the split-conformal switch is the best non-BCA switch by NET in every recorded set
(97.3 against meangap 57.2 on main 0-4; 99.5 against meangap 62.3 / meandiff 62.9 on near-80; 78.7 / 84.1 against
43.6 / 61.4 on seeds 11 / 14), so that test is the same test. Against fixed_gated (BCA's switch, height 1.0): main
0-4 mean 13.6, sd 5.5: 3, 4, 5, 16 seeds; near-80 mean 4.7, sd 3.6: 6, 30, unreachable, unreachable.

### 7.2 BCA against the mean-difference-gated range limit (no BCA part)

| direction | spread from | mean, sd | m = 0 | m = 3 | m = 5 | m = 10 |
|---|---|---|---|---|---|---|
| BCA beats it | main 0-4 | -4.49, 5.21 | unreachable at every m (BCA's mean is below 0) | | | |
| BCA beats it | near-80 0-4 | -2.40, 4.75 | unreachable | | | |
| BCA beats it | main 7 | -3.60, 4.55 | unreachable | | | |
| it beats BCA | main 0-4 | 4.49, 5.21 | 10 | 80 | unreachable | unreachable |
| it beats BCA | near-80 0-4 | 2.40, 4.75 | 26 | unreachable | | |
| it beats BCA | main 7 | 3.60, 4.55 | 12 | 375 | unreachable | |

Against the gated *matched* range limit BCA is ahead on the main seeds (mean 10.2, sd 10.4: 8, 15, 26, >400 seeds)
and behind on near-80 (-3.1, sd 3.1; "it beats BCA by 0" needs 8 seeds).

### 7.3 What the refusal does to any fresh-seed test of BCA

Three of five fresh stalling-world seeds had fewer than 29 held-out full-throttle rows (19, 22, 18) and BCA's arm was
undefined. A paired test of BCA against anything needs BCA defined on every seed; under the current grouping each
fresh seed certifies with probability about 0.75-0.8 (the gate's own 30 stalling-world logs: 7 of 30 below 29 rows,
`near80_results_check/REPORT.md`). The seed counts above are counts of *certified* seeds; the number of seeds to run
is larger by that factor unless the grouping is changed before the test (pooling every held-out row outside the
removal ball, as proposed in the lost analysis, was not re-verified here).

---

## 8. What the records support, in one list

- Supported (exploratory, seeds 0-4; description, seeds 11/14): BCA's switched ramp beats every *matched world-blind*
  rival by 20-50 NET points, because it is off in the no-stall world; BCA's switch is right in 40 of 40 seed-worlds.
- Supported: that credit is not BCA-specific. The mean-difference switch makes the identical call in 40 of 40, the
  conformal switch in 37 of 40 (its three errors cost 0-4.4 points each), and the switched rivals' lower scores are
  their lower heights, within a monotone dose ladder.
- Supported: BCA's height is larger than every rival's in every stalling seed-world; twice BCA's height does as well
  (main: +3.7 better; near-80: 1.9 worse; seeds 11/14: +1.2, -1.7); so the height's credit is "large enough", not
  "calibrated".
- Supported: a non-BCA rule (mean-difference switch + hand-set range limit) matches or beats BCA's NET on 8 of 12
  seed-settings, built from recorded arms.
- Not supported by anything in the records: any pre-registered pass for BCA; any measurable return credit to the
  Bayesian step, to eta or to the credibility level beyond "a larger dose"; and any statement about seeds 10, 12, 13
  beyond the refusal.
- Not checked here: eta's share of the band width (the lost analysis's 0.2%), the importance weights' coverage
  credit, and the pooled-grouping certification probabilities. Those need the step-1 scores and wbcp code, not the
  points.

---

## 9. Sources

All paths under `B:\outputs\shift\beyond-the-data\` (branch `claude/main-bca-connection-x4o46s` at 52914e4) unless
stated. N8 = `near80/results-20261020T115246Z/files/`.

- Points, all settings and seeds: `N8/eval/<setting>/<world>/seed<k>/values.json` (`arm_ids`, `evaluations.<arm>.<situation>.{points, points_se, unrecorded_throttle_share, loss_from_unrecorded_throttle, capped_points}`); 34 files (main 0-4, 11, 14; near80 0-4; random_rows 0-4; both worlds).
- The same main 0-4 numbers: `restructure/step2_results/results.json` (`points`, `net_strong_current`, `attribution`, `dose_table_strong_current`), `results.md`; equality checked to 0 difference.
- Switch records (certificate, groups, rank test): `N8/learners/<setting>/<world>/seed<k>/switch.json` (30 files); main 0-4: `restructure/step1/results/results.json` `per_seed_world`.
- Arm tables actually trained (switch, h, k per arm): `N8/policy/<setting>/<world>/seed<k>/lock.json` `arm_table` (24 files); main 0-4 from step 1 via step 2's rule (`restructure/step2/PRESPEC.md` arms table, lines 98-125, matches).
- Rules: `restructure/step2/step2lib.py:315-365` (`arm_table`), `:396-404` (`net`), `:434-456` (`go_no_go`); `near80/code/policy_phase.py:96-150` (meandiff_k1, knee_ramp, the stop on an infinite height), `near80/code/near80_constants.py:100-129` (KNEE32, ARM_RULE, MEANDIFF_MARGIN, arm lists); `near80/code/evaluate_phase.py:6-11, 50` (capped points, loss from unrecorded throttle).
- Stops: `N8/stops/policy.main.stall.s1{0,2,3}.json`; `N8/status.json` (no-stall and evaluate phases of seeds 10, 12, 13 skipped).
- Looks: `N8/looks/exploratory.md`, `confirmatory.md`, `.json` (`reported.nets`, `reported.switches`, `marks.claim4`).
- Report and checks: `near80_results_check/REPORT.md` (the 29-row explanation, the seeds 11/14 table), `checks.json`.
- Pre-registration texts: `near80/PREREGISTRATION.md` (arms 5.2-5.4, departures 15.5/15.10, risk 9 "at least 9 rows" at line 1204), `restructure/step2/PRESPEC.md`.
- Scripts and outputs in this folder: `extract.py` -> `extracted.json`; `analyze1.py` -> `analyze1.out` (per-seed points, every arm, situation, setting); `analyze2.py` -> `analyze2.out`, `switches.json` (40 switch records and agreement counts); `analyze3.py` -> `analyze3.out`, `net_diffs.json` (paired differences, gated constructs); `analyze4.py` -> `analyze4.out` (dose ladders, past-80 shares, the dose line); `analyze5.py` -> `analyze5.out`, `power.json` (power).
