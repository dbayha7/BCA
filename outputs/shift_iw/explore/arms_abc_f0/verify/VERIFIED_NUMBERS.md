# Verified numbers: arms A, C and F0 (EXPLORATORY)

**EXPLORATORY.** Seeds 0-4 are seen seeds. Nothing here is confirmatory; a confirmatory claim needs fresh seeds 15-19, its own pre-registration and a new go from the owner.

An independent recomputation of every mark of arms A, C and F0 in `routes2/arms/RESULTS.md`, from the per-seed raw scores, following the frozen file's sections 3.2, 4.2-4.3, 6.2-6.3 and 7.4. It does not import or call `score.py` or any project code: `verify_marks.py` reads the raw records with plain `json`, and does the statistics in pure Python (`math.fsum`, `statistics.stdev`), not numpy.

- Frozen file: `PASSMARKS_arms_ABC_and_F_control.md`, SHA-256 `a073be97003938c112e9ba77581a311e812fe5851f7e64211a11fb4bf47d9ecf` (matches its sidecar).
- `KILLS.json` SHA-256 `96db647e112998fddd0fe85a6dbad61c955e797478372d03cf833d576ad5e61f`; `results.json` SHA-256 `7c9a1d3004b9a70eee60a12c01eb39059d669b18fa0bd9b20f0a8bad5f911e04` (matches its sidecar); `RESULTS.md` SHA-256 `ade7a08ec9454dfe56e002bb9d2ab3a6e839aeb4480693cc43186e0f4c0ee5cf`.
- Scripts (this folder): `verify_marks.py` (marks and information rows, writes `verified.json`), `le_none_check.py` (the le_none question, run with the sandbox's interpreter for flax's msgpack reader, writes `le_none_check.json`), `make_md.py` (this file).

## Verdict

- **Rows recomputed:** 95 (26 arm A marks, 33 arm C rows, 2 F0 marks, 34 information rows).
- **Mismatches beyond 1e-9:** 0. The largest difference between a recomputed number (each d_k, the mean, the lower bound) and `results.json` is 3.6e-15. Every 2-decimal rendering in `RESULTS.md` (mean, seeds positive, lower bound, each d_k, Met) equals the recomputed value rendered the same way.
- **Raw-record problems:** 0. Every evaluation record verifies against its `.sha256`, has status finished, carries the frozen file's digest and label, names the current `KILLS.json`, and was scored from the arm record that is there now; its `quality` and `seed` fields equal its path (pairing by seed). Every F0 record verifies, points at A2's record of the same quality and seed with that record's current SHA-256, has KF3's two clone digests equal, and has A2's row-stream digest. Every A2 `evaluate_seed<k>.json` names its own quality and seed and is newer than A2's record. A2's seed-0 evaluation equals the result F rerun's `scores_seed0.json` on all three banks (medium-expert: True, expert: True).
- **Arm A:** P1 True, P2 True, P3 True, P4 False; passing qualities medium-replay, medium; GO False. Same as `RESULTS.md`, including the outcome line and the rivals it names.
- **Arm C:** B1 True (passing random, medium-replay, medium), B5 True; B2-B4 not scored (the guard is void); GO False. Same as `RESULTS.md`.
- **F0:** credit on neither quality; wording "Result F's gain comes from refusing credit past the data, not from BCA's band.". Same as `RESULTS.md`.
- **Arm B:** no B return entered any mark (section 'Arm B' below).
- **le_none:** arm A's `none` actor is **not** result F's `le_none` actor (section 'Is arm A's none result F's no-extra-caution actor?'). A2 − none and F0 − none were therefore not computed.

## What was checked, and how

- **Per seed** k: d_k = (arm's points) − (comparator's points), same quality, same seed, same bank. Points are the records' `points` (0 = random-quality data, 100 = stall-aware controller), each over the bank's 1,000 starts; a number counts only when its `finite` flag is true, it is finite and `starts` is 1,000.
- **Mean** = fsum(d_0..d_4) / 5. **Seeds positive** = count of d_k > 0; a tie (d_k = 0) would count as not positive. No d_k in any row is exactly 0 (the smallest |d_k| is 0.0007, B2 random).
- **Thresholds are inclusive** and compared at full precision (mean ≥ bar; positives ≥ 4 where named).
- **Lower bound** = mean − 2.132 × SD / √5. **Which SD:** the frozen file says "SD" with "(t, 4 degrees of freedom)". A t interval with n − 1 = 4 degrees of freedom is built on the sample SD (divisor n − 1), so the file implies the sample SD. The code (`score.summary`) uses `numpy.std(ddof=1)`, the sample SD. They agree. `verified.json` also holds the bound with the population SD (divisor n) for contrast; it is not what the file implies and is not used. The lower bound decides nothing.
- **Best plain margin** (P4): on each passing quality, the one of `plain_1s`, `plain_2s`, `plain_4s` with the highest 5-seed mean points on that quality's strong test, chosen after the fact. **Best plain test** (B4): the one of the five plain tests with the highest 5-seed mean of rho 0.5 + test on that quality's strong test. No ties for best occurred.
- **P2 and B5** on every bank (none, mild, strong) and all five qualities: 15 rows each.
- **Missing numbers:** none. `KILLS.json` lists seeds 0-4 as usable for every quality of A and C and both F0 qualities as decided; every number every mark needs is present and finite.

## Marks

Each d_k is in points, arm minus comparator, strong test unless a bank is named after a slash. 'Mine' is the recomputed value; 'RESULTS.md' is the cell as printed there (2 decimals). The last column is the largest absolute difference between the recomputed d_k, mean and lower bound and the full-precision values in `results.json`.

### Arm A (sections 4.2-4.3)

| Mark | Bar | d_0, d_1, d_2, d_3, d_4 (recomputed) | Mean: mine / RESULTS.md | Seeds positive: mine / RESULTS.md | Lower bound: mine / RESULTS.md | Met: mine / RESULTS.md | max abs dev vs results.json |
|---|---|---|---|---|---|---|---|
| P1 A1 - none, medium-replay | >= +2.0, 4 of 5 | +13.9603, +11.1782, +9.5320, +11.2822, +10.3927 | 11.27 / 11.27 | 5 / 5 | 9.69 / 9.69 | yes / yes | 0.0e+00 |
| P1 A1 - none, medium | >= +2.0, 4 of 5 | +7.0802, +11.7610, +9.9138, +10.4064, +11.1594 | 10.06 / 10.06 | 5 / 5 | 8.34 / 8.34 | yes / yes | 1.8e-15 |
| P1 A1 - none, medium-expert | >= +2.0, 4 of 5 | +1.7064, +0.7190, +0.7193, +0.3338, +1.1701 | 0.93 / 0.93 | 5 / 5 | 0.43 / 0.43 | no / no | 0.0e+00 |
| P2 A1 - none, random/none | >= -1.0 | +13.7354, +13.5144, +11.8393, +18.5688, +17.4017 | 15.01 / 15.01 | 5 / 5 | 12.30 / 12.30 | yes / yes | 0.0e+00 |
| P2 A1 - none, random/mild | >= -1.0 | +16.3654, +15.6770, +14.9108, +22.8648, +17.7354 | 17.51 / 17.51 | 5 / 5 | 14.49 / 14.49 | yes / yes | 0.0e+00 |
| P2 A1 - none, random/strong | >= -1.0 | +17.8885, +16.6022, +16.9959, +25.2224, +17.4787 | 18.84 / 18.84 | 5 / 5 | 15.40 / 15.40 | yes / yes | 0.0e+00 |
| P2 A1 - none, medium-replay/none | >= -1.0 | +9.9715, +10.2067, +6.9665, +8.6568, +7.7422 | 8.71 / 8.71 | 5 / 5 | 7.38 / 7.38 | yes / yes | 0.0e+00 |
| P2 A1 - none, medium-replay/mild | >= -1.0 | +12.3268, +11.3309, +8.6009, +10.5178, +9.5553 | 10.47 / 10.47 | 5 / 5 | 9.07 / 9.07 | yes / yes | 0.0e+00 |
| P2 A1 - none, medium-replay/strong | >= -1.0 | +13.9603, +11.1782, +9.5320, +11.2822, +10.3927 | 11.27 / 11.27 | 5 / 5 | 9.69 / 9.69 | yes / yes | 0.0e+00 |
| P2 A1 - none, medium/none | >= -1.0 | +6.5566, +9.3617, +8.0816, +8.5695, +9.1069 | 8.34 / 8.34 | 5 / 5 | 7.28 / 7.28 | yes / yes | 0.0e+00 |
| P2 A1 - none, medium/mild | >= -1.0 | +7.1723, +11.0721, +9.3965, +10.0334, +10.6338 | 9.66 / 9.66 | 5 / 5 | 8.20 / 8.20 | yes / yes | 0.0e+00 |
| P2 A1 - none, medium/strong | >= -1.0 | +7.0802, +11.7610, +9.9138, +10.4064, +11.1594 | 10.06 / 10.06 | 5 / 5 | 8.34 / 8.34 | yes / yes | 1.8e-15 |
| P2 A1 - none, medium-expert/none | >= -1.0 | +0.9269, +0.5639, +0.4117, +0.3028, +0.5748 | 0.56 / 0.56 | 5 / 5 | 0.33 / 0.33 | yes / yes | 0.0e+00 |
| P2 A1 - none, medium-expert/mild | >= -1.0 | +1.3228, +0.6430, +0.5357, +0.3037, +0.8413 | 0.73 / 0.73 | 5 / 5 | 0.36 / 0.36 | yes / yes | 0.0e+00 |
| P2 A1 - none, medium-expert/strong | >= -1.0 | +1.7064, +0.7190, +0.7193, +0.3338, +1.1701 | 0.93 / 0.93 | 5 / 5 | 0.43 / 0.43 | yes / yes | 0.0e+00 |
| P2 A1 - none, expert/none | >= -1.0 | +0.6162, +0.3063, +0.4599, +0.6874, +0.0874 | 0.43 / 0.43 | 5 / 5 | 0.20 / 0.20 | yes / yes | 0.0e+00 |
| P2 A1 - none, expert/mild | >= -1.0 | +0.7511, +0.2008, +0.6116, +0.9393, -0.1251 | 0.48 / 0.48 | 4 / 4 | 0.06 / 0.06 | yes / yes | 0.0e+00 |
| P2 A1 - none, expert/strong | >= -1.0 | +0.7582, +0.0493, +0.7880, +1.0648, -0.4063 | 0.45 / 0.45 | 4 / 4 | -0.13 / -0.13 | yes / yes | 1.1e-16 |
| P3 A1 - untargeted, medium-replay | >= +1.5 | +14.6039, +12.7993, +12.8488, +9.6601, +16.3453 | 13.25 / 13.25 | 5 / 5 | 10.88 / 10.88 | yes / yes | 0.0e+00 |
| P3 A1 - untargeted, medium | >= +1.5 | +7.9457, +7.0903, +10.6260, +12.0829, +10.6457 | 9.68 / 9.68 | 5 / 5 | 7.69 / 7.69 | yes / yes | 0.0e+00 |
| P4 A1 - margin0, medium-replay | >= +1.0 | +1.0249, -2.4678, -6.1477, -4.1690, -0.6045 | -2.47 / -2.47 | 1 / 1 | -5.17 / -5.17 | no / no | 0.0e+00 |
| P4 A1 - plain_1s, medium-replay | >= +1.0 | -2.2799, -3.3429, -6.6032, -2.8755, -0.4248 | -3.11 / -3.11 | 0 / 0 | -5.25 / -5.25 | no / no | 0.0e+00 |
| P4 A1 - matched_share, medium-replay | >= +1.0 | -0.1504, -0.5671, -0.3612, -0.3516, +0.5180 | -0.18 / -0.18 | 1 / 1 | -0.58 / -0.58 | no / no | 0.0e+00 |
| P4 A1 - margin0, medium | >= +1.0 | +2.7071, +1.2720, -0.8571, +2.1976, +0.6893 | 1.20 / 1.20 | 4 / 4 | -0.13 / -0.13 | yes / yes | 2.2e-16 |
| P4 A1 - plain_2s, medium | >= +1.0 | +0.0183, -0.2575, -1.2350, -0.4035, -0.2365 | -0.42 / -0.42 | 1 / 1 | -0.88 / -0.88 | no / no | 0.0e+00 |
| P4 A1 - matched_share, medium | >= +1.0 | -0.2263, -0.7600, -0.4531, -0.3079, +0.4676 | -0.26 / -0.26 | 1 / 1 | -0.69 / -0.69 | no / no | 0.0e+00 |

P4's best plain margin, by 5-seed mean strong-test points of the rival:

- medium-replay: plain_1s 92.2393, plain_2s 90.6882, plain_4s 85.0282 → best **plain_1s** (RESULTS.md names plain_1s). A1 minus every plain margin: plain_1s -3.11, plain_2s -1.55, plain_4s +4.11.
- medium: plain_1s 93.5428, plain_2s 93.8135, plain_4s 89.6790 → best **plain_2s** (RESULTS.md names plain_2s). A1 minus every plain margin: plain_1s -0.15, plain_2s -0.42, plain_4s +3.71.

P1 passing qualities: medium-replay, medium (medium-expert's mean 0.9297 is below +2.0 although 5 of 5 seeds are positive). P3 and P4 are judged on the passing qualities only, as section 4.3 says. P4 fails on both passing qualities whatever plain margin is taken as best: A1 − margin0 and A1 − matched_share already fail on medium-replay, and A1 − matched_share fails on medium.

Outcome line, reconstructed from the frozen wording (P1, P2, P3 met, P4 not), equals RESULTS.md's: "Filtering helps; the band adds nothing over a plain margin. Tied or won: margin0 on medium-replay, plain_1s on medium-replay, matched_share on medium-replay, plain_2s on medium, matched_share on medium."

### Arm C (sections 6.2-6.3)

| Mark | Bar | d_0, d_1, d_2, d_3, d_4 (recomputed) | Mean: mine / RESULTS.md | Seeds positive: mine / RESULTS.md | Lower bound: mine / RESULTS.md | Met: mine / RESULTS.md | max abs dev vs results.json |
|---|---|---|---|---|---|---|---|
| B1 (rho 0.5 + BCA) - rho 1, random | >= +3.0, 4 of 5 | +14.0007, +16.5761, +13.5062, +21.2995, +12.0528 | 15.49 / 15.49 | 5 / 5 | 12.02 / 12.02 | yes / yes | 0.0e+00 |
| B1 (rho 0.5 + BCA) - rho 1, medium-replay | >= +3.0, 4 of 5 | +10.1319, +8.9087, +8.5787, +9.8535, +7.0474 | 8.90 / 8.90 | 5 / 5 | 7.74 / 7.74 | yes / yes | 0.0e+00 |
| B1 (rho 0.5 + BCA) - rho 1, medium | >= +3.0, 4 of 5 | +6.3705, +9.2055, +8.8699, +9.2572, +8.6872 | 8.48 / 8.48 | 5 / 5 | 7.33 / 7.33 | yes / yes | 0.0e+00 |
| B5 (rho 0.5 + BCA) - rho 1, random/none | >= -1.0 | +11.6746, +14.0900, +11.2870, +16.6155, +10.6337 | 12.86 / 12.86 | 5 / 5 | 10.50 / 10.50 | yes / yes | 0.0e+00 |
| B5 (rho 0.5 + BCA) - rho 1, random/mild | >= -1.0 | +13.0462, +15.6506, +12.4746, +19.4773, +11.2668 | 14.38 / 14.38 | 5 / 5 | 11.27 / 11.27 | yes / yes | 0.0e+00 |
| B5 (rho 0.5 + BCA) - rho 1, random/strong | >= -1.0 | +14.0007, +16.5761, +13.5062, +21.2995, +12.0528 | 15.49 / 15.49 | 5 / 5 | 12.02 / 12.02 | yes / yes | 0.0e+00 |
| B5 (rho 0.5 + BCA) - rho 1, medium-replay/none | >= -1.0 | +7.8742, +8.6686, +6.4689, +7.6528, +6.1324 | 7.36 / 7.36 | 5 / 5 | 6.36 / 6.36 | yes / yes | 0.0e+00 |
| B5 (rho 0.5 + BCA) - rho 1, medium-replay/mild | >= -1.0 | +9.1397, +9.1735, +7.5169, +8.7035, +6.6859 | 8.24 / 8.24 | 5 / 5 | 7.20 / 7.20 | yes / yes | 0.0e+00 |
| B5 (rho 0.5 + BCA) - rho 1, medium-replay/strong | >= -1.0 | +10.1319, +8.9087, +8.5787, +9.8535, +7.0474 | 8.90 / 8.90 | 5 / 5 | 7.74 / 7.74 | yes / yes | 0.0e+00 |
| B5 (rho 0.5 + BCA) - rho 1, medium/none | >= -1.0 | +5.3976, +7.0477, +6.8391, +6.2090, +6.7912 | 6.46 / 6.46 | 5 / 5 | 5.82 / 5.82 | yes / yes | 0.0e+00 |
| B5 (rho 0.5 + BCA) - rho 1, medium/mild | >= -1.0 | +6.0512, +8.5410, +8.0087, +7.9661, +7.8485 | 7.68 / 7.68 | 5 / 5 | 6.78 / 6.78 | yes / yes | 0.0e+00 |
| B5 (rho 0.5 + BCA) - rho 1, medium/strong | >= -1.0 | +6.3705, +9.2055, +8.8699, +9.2572, +8.6872 | 8.48 / 8.48 | 5 / 5 | 7.33 / 7.33 | yes / yes | 0.0e+00 |
| B5 (rho 0.5 + BCA) - rho 1, medium-expert/none | >= -1.0 | +2.3177, +2.1712, +1.8282, +2.5195, +2.7646 | 2.32 / 2.32 | 5 / 5 | 1.98 / 1.98 | yes / yes | 0.0e+00 |
| B5 (rho 0.5 + BCA) - rho 1, medium-expert/mild | >= -1.0 | +3.6717, +3.4635, +2.7849, +4.2028, +4.6257 | 3.75 / 3.75 | 5 / 5 | 3.08 / 3.08 | yes / yes | 0.0e+00 |
| B5 (rho 0.5 + BCA) - rho 1, medium-expert/strong | >= -1.0 | +5.1250, +5.0230, +3.9812, +6.1155, +6.7451 | 5.40 / 5.40 | 5 / 5 | 4.38 / 4.38 | yes / yes | 0.0e+00 |
| B5 (rho 0.5 + BCA) - rho 1, expert/none | >= -1.0 | +3.7968, +3.4601, +3.8160, +3.7345, +3.8217 | 3.73 / 3.73 | 5 / 5 | 3.58 / 3.58 | yes / yes | 0.0e+00 |
| B5 (rho 0.5 + BCA) - rho 1, expert/mild | >= -1.0 | +6.5392, +6.0031, +6.6039, +6.4706, +6.7287 | 6.47 / 6.47 | 5 / 5 | 6.20 / 6.20 | yes / yes | 0.0e+00 |
| B5 (rho 0.5 + BCA) - rho 1, expert/strong | >= -1.0 | +9.4150, +8.7233, +9.6044, +9.3075, +9.8373 | 9.38 / 9.38 | 5 / 5 | 8.98 / 8.98 | yes / yes | 0.0e+00 |
| B2 (rho 0.5 + BCA) - rho 0.5, random | >= +1.5 (not scored) | +0.0264, +0.1628, +0.1497, -0.0007, +0.0757 | 0.08 / 0.08 | 4 / 4 | 0.01 / 0.01 | no / no | 0.0e+00 |
| B2 (rho 0.5 + BCA) - rho 0.5, medium-replay | >= +1.5 (not scored) | +0.0199, +0.0467, +0.5350, +0.3019, +0.1342 | 0.21 / 0.21 | 5 / 5 | 0.00 / 0.00 | no / no | 0.0e+00 |
| B2 (rho 0.5 + BCA) - rho 0.5, medium | >= +1.5 (not scored) | +0.1826, +0.3006, +0.4734, +1.3439, +0.5749 | 0.58 / 0.58 | 5 / 5 | 0.14 / 0.14 | no / no | 0.0e+00 |
| B3 BCA - range_limit, random | >= +1.5 (not scored) | +0.0264, +0.1628, +0.1497, -0.0007, +0.0757 | 0.08 / 0.08 | 4 / 4 | 0.01 / 0.01 | no / no | 0.0e+00 |
| B3 BCA - shuffled, random | >= +1.5 (not scored) | +0.0504, +0.0026, +0.1331, +0.0367, +0.0821 | 0.06 / 0.06 | 5 / 5 | 0.01 / 0.01 | no / no | 6.9e-18 |
| B3 BCA - matched_global, random | >= +1.0 (not scored) | +0.0634, +0.3629, +0.3705, +0.0190, -0.0911 | 0.14 / 0.14 | 4 / 4 | -0.06 / -0.06 | no / no | 0.0e+00 |
| B3 BCA - range_limit, medium-replay | >= +1.5 (not scored) | +0.0199, +0.0467, +0.5350, +0.3019, +0.1342 | 0.21 / 0.21 | 5 / 5 | 0.00 / 0.00 | no / no | 0.0e+00 |
| B3 BCA - shuffled, medium-replay | >= +1.5 (not scored) | +0.0151, +0.0408, +0.2635, +0.3452, +0.1195 | 0.16 / 0.16 | 5 / 5 | 0.02 / 0.02 | no / no | 0.0e+00 |
| B3 BCA - matched_global, medium-replay | >= +1.0 (not scored) | -0.0217, -0.1649, +0.2722, -0.1676, -0.1397 | -0.04 / -0.04 | 1 / 1 | -0.22 / -0.22 | no / no | 2.8e-17 |
| B3 BCA - range_limit, medium | >= +1.5 (not scored) | +0.1825, +0.2994, +0.4734, +1.3260, +0.5743 | 0.57 / 0.57 | 5 / 5 | 0.14 / 0.14 | no / no | 5.6e-17 |
| B3 BCA - shuffled, medium | >= +1.5 (not scored) | +0.1827, +0.2001, +0.4405, +0.8326, +0.5752 | 0.45 / 0.45 | 5 / 5 | 0.19 / 0.19 | no / no | 0.0e+00 |
| B3 BCA - matched_global, medium | >= +1.0 (not scored) | -0.0466, -0.2191, -0.0447, -0.1896, -0.1172 | -0.12 / -0.12 | 0 / 0 | -0.20 / -0.20 | no / no | 0.0e+00 |
| B4 BCA - sign, random | >= +1.0 (not scored) | +0.0112, -0.0415, -0.0200, -0.0318, -0.0172 | -0.02 / -0.02 | 1 / 1 | -0.04 / -0.04 | no / no | 0.0e+00 |
| B4 BCA - sign, medium-replay | >= +1.0 (not scored) | +0.0021, -0.0010, -0.0068, -0.0038, -0.0252 | -0.01 / -0.01 | 1 / 1 | -0.02 / -0.02 | no / no | 0.0e+00 |
| B4 BCA - wttest, medium | >= +1.0 (not scored) | +0.0122, +0.0525, +0.3041, -0.9161, +0.0032 | -0.11 / -0.11 | 4 / 4 | -0.55 / -0.55 | no / no | 0.0e+00 |

B1 passing qualities: random, medium-replay, medium. B5 holds on all 15 quality-bank pairs.

**B2, B3, B4 are not scored.** Section 6.4 voids the guard when B's K1 fires, or when C's own map has a thruster-1 ceiling of 0.85 or higher with at least 30 tries in rung (0.80, 0.85]. Both hold: B's K1 fired at the seed-0 screen on medium-replay, medium, medium-expert, and C's own map meets the condition on 18 seed-quality pairs (read here from the C records' rung counts; the list equals KILLS.json's `guard_void.by_own_map`). The rows above give their values for information; RESULTS.md prints them with Met 'no' and its summary line gives None. Had they been scored, every one of them would fail its bar by a wide margin (largest B2 mean +0.58 against +1.5; largest B4 mean −0.01 against +1.0), so nothing turns on it.

B4's best plain test, by 5-seed mean strong-test points of rho 0.5 + test:

- random: wmedian 80.9294, wmean 80.5795, sign 80.9811, wttest 80.6020, wbernstein 76.6803 → best **sign** (RESULTS.md names sign).
- medium-replay: wmedian 86.6547, wmean 86.6868, sign 86.7760, wttest 86.7027, wbernstein 81.3614 → best **sign** (RESULTS.md names sign).
- medium: wmedian 91.6574, wmean 91.7400, sign 91.7475, wttest 91.9134, wbernstein 88.3179 → best **wttest** (RESULTS.md names wttest).

### F0 (section 7.4)

| Mark | Bar | d_0, d_1, d_2, d_3, d_4 (recomputed) | Mean: mine / RESULTS.md | Seeds positive: mine / RESULTS.md | Lower bound: mine / RESULTS.md | Met: mine / RESULTS.md | max abs dev vs results.json |
|---|---|---|---|---|---|---|---|
| A2 - F0, medium-expert (strong) | >= +1.0, 4 of 5 | +0.0429, +0.1310, +0.0550, +0.3254, +0.1564 | 0.14 / 0.14 | 5 / 5 | 0.03 / 0.03 | no / no | 1.4e-17 |
| A2 - F0, expert (strong) | >= +1.0, 4 of 5 | +0.2794, -0.0138, +0.0600, -0.0344, +0.1136 | 0.08 / 0.08 | 3 / 3 | -0.04 / -0.04 | no / no | 0.0e+00 |

Credit needs mean ≥ +1.0 and 4 of 5 seeds positive: met on neither quality. The 'lowered return' line needs mean ≤ −1.0 with 4 of 5 negative: on neither. The outcome is the frozen 'credit on neither' wording, as RESULTS.md prints it.

## Information rows (decide nothing)

### Arm A: even_x1.33 − none and A1c − none (strong test)

| Row | d_0, d_1, d_2, d_3, d_4 (recomputed) | Mean: mine / RESULTS.md | Seeds positive: mine / RESULTS.md | Lower bound (mine; not shown in RESULTS.md) | max abs dev vs results.json |
|---|---|---|---|---|---|
| even_x1.33-none, random | -7.5947, -8.4305, -6.5111, -8.9455, -6.4866 | -7.59 / -7.59 | 0 / 0 | -8.65 | 0.0e+00 |
| A1c-none, random | +18.6451, +17.1783, +18.4423, +27.5423, +17.7341 | 19.91 / 19.91 | 5 / 5 | 15.80 | 0.0e+00 |
| even_x1.33-none, medium-replay | -4.9181, -4.9738, -4.6018, -4.4285, -4.2746 | -4.64 / -4.64 | 0 / 0 | -4.93 | 0.0e+00 |
| A1c-none, medium-replay | +14.6107, +11.8693, +10.0125, +11.2126, +10.9754 | 11.74 / 11.74 | 5 / 5 | 10.08 | 1.8e-15 |
| even_x1.33-none, medium | -4.0317, -4.0489, -5.3988, -4.2200, -4.5138 | -4.44 / -4.44 | 0 / 0 | -4.98 | 0.0e+00 |
| A1c-none, medium | +7.3180, +12.3231, +10.2053, +9.9078, +11.7404 | 10.30 / 10.30 | 5 / 5 | 8.44 | 0.0e+00 |
| even_x1.33-none, medium-expert | +0.4582, +0.5293, +0.4226, +0.6123, +0.4974 | 0.50 / 0.50 | 5 / 5 | 0.43 | 0.0e+00 |
| A1c-none, medium-expert | +1.6555, +0.9440, +0.6464, +0.6586, +1.4407 | 1.07 / 1.07 | 5 / 5 | 0.63 | 0.0e+00 |
| even_x1.33-none, expert | +0.1298, +0.1931, +0.3653, +0.2638, +0.2590 | 0.24 / 0.24 | 5 / 5 | 0.16 | 0.0e+00 |
| A1c-none, expert | +0.4921, +0.0200, +0.8007, +0.9218, -0.2050 | 0.41 / 0.41 | 4 / 4 | -0.06 | 5.6e-17 |

### Arm C: against rho 1 (strong test)

| Row | d_0, d_1, d_2, d_3, d_4 (recomputed) | Mean: mine / RESULTS.md | Seeds positive: mine / RESULTS.md | Lower bound (mine; not shown in RESULTS.md) | max abs dev vs results.json |
|---|---|---|---|---|---|
| rho0.5-rho1, random | +13.9743, +16.4133, +13.3566, +21.3003, +11.9771 | 15.40 / 15.40 | 5 / 5 | 11.91 | 0.0e+00 |
| rho0.5+knee_clip-rho1, random | +14.1373, +16.8478, +13.5279, +21.4131, +12.2114 | 15.63 / 15.63 | 5 / 5 | 12.15 | 3.6e-15 |
| rho1+bca_built_for_rho1-rho1, random | -0.0104, -0.3137, +0.0624, -0.1087, -0.1203 | -0.10 / -0.10 | 1 / 1 | -0.23 | 0.0e+00 |
| rho0.5+bca-rho1, random (= B5 strong) | +14.0007, +16.5761, +13.5062, +21.2995, +12.0528 | 15.49 / 15.49 | 5 / 5 | 12.02 | 0.0e+00 |
| rho0.5-rho1, medium-replay | +10.1120, +8.8621, +8.0437, +9.5516, +6.9131 | 8.70 / 8.70 | 5 / 5 | 7.49 | 0.0e+00 |
| rho0.5+knee_clip-rho1, medium-replay | +10.3149, +9.1581, +8.5882, +10.0846, +7.2689 | 9.08 / 9.08 | 5 / 5 | 7.91 | 0.0e+00 |
| rho1+bca_built_for_rho1-rho1, medium-replay | -0.0277, -0.0053, -1.1477, -0.0383, +0.0812 | -0.23 / -0.23 | 1 / 1 | -0.72 | 0.0e+00 |
| rho0.5+bca-rho1, medium-replay (= B5 strong) | +10.1319, +8.9087, +8.5787, +9.8535, +7.0474 | 8.90 / 8.90 | 5 / 5 | 7.74 | 0.0e+00 |
| rho0.5-rho1, medium | +6.1879, +8.9049, +8.3966, +7.9133, +8.1123 | 7.90 / 7.90 | 5 / 5 | 6.92 | 0.0e+00 |
| rho0.5+knee_clip-rho1, medium | +7.1602, +9.8000, +9.4145, +11.1504, +9.0190 | 9.31 / 9.31 | 5 / 5 | 7.93 | 0.0e+00 |
| rho1+bca_built_for_rho1-rho1, medium | +0.0269, +0.1569, +0.1302, +0.3008, -0.0449 | 0.11 / 0.11 | 4 / 4 | -0.01 | 2.8e-17 |
| rho0.5+bca-rho1, medium (= B5 strong) | +6.3705, +9.2055, +8.8699, +9.2572, +8.6872 | 8.48 / 8.48 | 5 / 5 | 7.33 | 0.0e+00 |
| rho0.5-rho1, medium-expert | -1.2193, -0.9959, -0.7469, -0.5720, -0.5025 | -0.81 / -0.81 | 0 / 0 | -1.09 | 0.0e+00 |
| rho0.5+knee_clip-rho1, medium-expert | +6.6337, +6.3249, +7.2028, +7.6384, +8.3722 | 7.23 / 7.23 | 5 / 5 | 6.46 | 0.0e+00 |
| rho1+bca_built_for_rho1-rho1, medium-expert | +4.4158, +4.3633, +3.6194, +5.5925, +6.1025 | 4.82 / 4.82 | 5 / 5 | 3.86 | 0.0e+00 |
| rho0.5+bca-rho1, medium-expert (= B5 strong) | +5.1250, +5.0230, +3.9812, +6.1155, +6.7451 | 5.40 / 5.40 | 5 / 5 | 4.38 | 0.0e+00 |
| rho0.5-rho1, expert | -0.7017, -0.5836, -0.6052, -0.1553, -0.2282 | -0.45 / -0.45 | 0 / 0 | -0.69 | 0.0e+00 |
| rho0.5+knee_clip-rho1, expert | +9.4252, +8.7312, +9.6117, +9.3153, +9.8443 | 9.39 / 9.39 | 5 / 5 | 8.99 | 0.0e+00 |
| rho1+bca_built_for_rho1-rho1, expert | +9.0120, +8.3388, +9.2075, +8.8769, +9.4885 | 8.98 / 8.98 | 5 / 5 | 8.58 | 0.0e+00 |
| rho0.5+bca-rho1, expert (= B5 strong) | +9.4150, +8.7233, +9.6044, +9.3075, +9.8373 | 9.38 / 9.38 | 5 / 5 | 8.98 | 0.0e+00 |

`rho0.5+bca-rho1` on the strong test is B5's strong row; RESULTS.md prints it only there, so its RESULTS.md cells above are taken from that row.

### F0: A2 − F0 on the mild and none banks

| Row | d_0, d_1, d_2, d_3, d_4 (recomputed) | Mean: mine / RESULTS.md | Seeds positive: mine / RESULTS.md | Lower bound (mine; not shown in RESULTS.md) | max abs dev vs results.json |
|---|---|---|---|---|---|
| A2 - F0, medium-expert (mild) | +0.0693, +0.1206, +0.0618, +0.2248, +0.1275 | 0.12 / 0.12 | 5 / 5 | 0.06 | 2.8e-17 |
| A2 - F0, medium-expert (none) | +0.0696, +0.1172, +0.0489, +0.1523, +0.0843 | 0.09 / 0.09 | 5 / 5 | 0.06 | 0.0e+00 |
| A2 - F0, expert (mild) | +0.1776, +0.0019, +0.0410, -0.0196, +0.0876 | 0.06 / 0.06 | 4 / 4 | -0.02 | 1.4e-17 |
| A2 - F0, expert (none) | +0.1003, +0.0069, +0.0300, -0.0058, +0.0555 | 0.04 / 0.04 | 4 / 4 | -0.00 | 0.0e+00 |

## Arm B: no B return entered any mark

- `KILLS.json`: B `go_on` False, stopped by ['K1'] (BCA's thruster-1 ceiling 0.85 or higher with at least 30 tries in rung (0.80, 0.85] at the seed-0 screen on medium-replay, medium, medium-expert).
- No B evaluation exists: `/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/iw_rerun/repo/runs/shift_iw_v5_explore/arms_abc_f0/eval/B` exists: False.
- `results.json` read no B record (inputs naming an `eval/B/` or `arms_abc_f0/B/` path: 0). Every per-seed value of G1, G2 (15 rows) and G3 (3 rows) is null (non-null: 0); G4's best plain test is None and its total None; met {'G1': False, 'G2': False, 'G3': False, 'G4': False}; GO False.
- No C mark reads B's map: the policies in every recomputed C mark are `rho0.5`, `rho0.5+bca`, `rho0.5+matched_global`, `rho0.5+range_limit`, `rho0.5+shuffled`, `rho0.5+sign`, `rho0.5+wttest`, `rho1` (B's host map used: False). The C information row 'rho 0.5 + B_host_map' (section 6.3's 'rho 0.5 with B's host-built map') has a seed-0 value only, from the screen's map; it enters no mark.

## Is arm A's none result F's no-extra-caution actor (le_none)?

**No.** They share every input, but they come from different training programs and are not bit-identical.

| | medium-expert | expert |
|---|---|---|
| Clone SHA-256 (arm A start / le_none start) | `f87093cad7a4…` / `f87093cad7a4…` (same) | `73e03006c0f6…` / `73e03006c0f6…` (same) |
| Row stream SHA-256 (arm A / le_none) | `c837711bb996…` / `c837711bb996…` (same) | `062ebb7ab80d…` / `062ebb7ab80d…` (same) |
| Steps | arm A: 5 chunks x 4,000 (record chunk_metrics: 5 chunks); le_none: 20000 in chunks of 4000 | arm A: 5 chunks x 4,000 (record chunk_metrics: 5 chunks); le_none: 20000 in chunks of 4000 |
| Alpha | 2.5 both (settings.alpha); first-chunk lambda 0.061467056 vs 0.061466953 | 2.5 both (settings.alpha); first-chunk lambda 0.130807169 vs 0.130807278 |
| Penalty | arm A: none (Arm('none', 'even', 1.0)); le_none: k = 0 × the oracle width | arm A: none (Arm('none', 'even', 1.0)); le_none: k = 0 × the oracle width |
| Training program | arm A: learners.ActorOnly.program, float32 step, 12 arms vmapped; le_none: lower_edge.Runner 'oracle', float64 step under jax enable_x64 (Adam's bias correction in float64), 3 arms vmapped (le_none, le_oracle, le_oracle_x1.5) | arm A: learners.ActorOnly.program, float32 step, 12 arms vmapped; le_none: lower_edge.Runner 'oracle', float64 step under jax enable_x64 (Adam's bias correction in float64), 3 arms vmapped (le_none, le_oracle, le_oracle_x1.5) |
| First chunk's mean actor loss (arm A / le_none) | 2.679859527 / 2.679855862 | 2.563326477 / 2.563329110 |
| Final actors bit-identical? | False; largest parameter difference 0.0525 | False; largest parameter difference 0.0707 |
| Arm A none vs the v5 room check's 'none' arm | bit-identical: True | bit-identical: True |
| le_none record's own check_b (vs room 'none') | 0.0525 | 0.0707 |
| Seed-0 strong test, points: arm A none / le_none | 93.223518 / 93.230056 (difference -0.0065) | 90.758108 / 90.747764 (difference +0.0103) |
| Seed-0 none / mild banks, points: arm A none / le_none | 96.3781 / 96.3707; 94.8704 / 94.8669 | 95.8835 / 95.8854; 93.3857 / 93.3811 |

Why they differ, exactly: le_none is the k = 0 lane of `lower_edge.room_check`'s float64-step program (traced under `jax.experimental.enable_x64`, so Adam's bias correction is computed in float64 and rounded, vmapped with the two oracle lanes). Arm A's `none` is `learners.ActorOnly.program`'s float32 step (vmapped with the other 11 arms), and it is bit-identical to the version-5 room check's `none` arm. `lower_edge.py`'s own docstring says the k = 0 arm 'can differ from learners.ActorOnly's none arm by rounding', and the room_lower_edge record measures that difference (check_b: 0.0525 on medium-expert, 0.0707 on expert, the same numbers found here between the two saved actors). The rounding difference shows from the first chunk and grows into different actors whose seed-0 strong-test scores differ by 0.0065 and 0.0103 points.

Same inputs, but not the same code path and not bit-identical, so A2 − none and F0 − none were **not** computed. (Arm A's none actors for seeds 1-4 exist; result F has no le_none for seeds 1-4 in this sandbox.)

### Section 7.4's seed-0 information item (absent from RESULTS.md)

The frozen file lists, as information only: 'on seed 0, F0 and A2 each minus the result F rerun's no extra caution actor (le_none)'. RESULTS.md and results.json do not report it. From the raw numbers (strong test, points; A2 and F0 from this run's records, le_none from `resultF/results/scores_seed0.json`):

| Quality | A2 | F0 | le_none | A2 − le_none | F0 − le_none |
|---|---|---|---|---|---|
| medium-expert | 94.6854 | 94.6424 | 93.2301 | +1.4553 | +1.4124 |
| expert | 97.0352 | 96.7558 | 90.7478 | +6.2874 | +6.0080 |

One seed; information only; decides nothing.

## Raw numbers and their source files

Every raw number is a record's `points` for one policy on one bank. Sources:

- Arm A: `/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/iw_rerun/repo/runs/shift_iw_v5_explore/arms_abc_f0/eval/A/<quality>/seed<k>/record.json` → `evaluations.<policy>.<bank>.points`.
- Arm C: `/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/iw_rerun/repo/runs/shift_iw_v5_explore/arms_abc_f0/eval/C/<quality>/seed<k>/record.json` → `evaluations.<policy>.<bank>.points`.
- F0: `/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/iw_rerun/repo/runs/shift_iw_v5_explore/arms_abc_f0/F0/<quality>/F0_no_band/seed<k>/record.json` → `evaluation.<bank>.points`.
- A2: `/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/iw_rerun/repo/runs/shift_iw_v5_explore/channel/train_f1/<quality>/evaluate_seed<k>.json` → `evaluations.A2_cmd_strong.<bank>.points` (seeds 1-4 built in this run; seed 0 is the result F rerun's and equals `resultF/results/scores_seed0.json`).

The per-row source path of every raw number (arm and comparator, each seed) is in `verified.json` (`table[].mine.source_arm` and `source_comparator`). Below, the raw points per policy, bank and seed, with the SHA-256 of each source file.

### Source files

| Arm | Quality | Seed | File SHA-256 |
|---|---|---|---|
| A eval | random | 0 | `22f8ed852af81b7bd9972e75978752c9b4a9a7e2166e57befbe6c2628e6158d6` |
| A eval | random | 1 | `0d9f966270c102791b037b19d160ff70cc46f3a533730dd3d8ddee7d07e4639a` |
| A eval | random | 2 | `79fa8a800c6591cdeef851f92d1fde6c4ce63fbc677720fc833e4496cfe0b0a8` |
| A eval | random | 3 | `992c8502e895f17d50ef4b1ce794bb8f5aa80135dad11ff312c9111cd881f049` |
| A eval | random | 4 | `57c20151af7bdae37b9bb599c87da91ae27393b4d7f719744e9c5263805565a4` |
| A eval | medium-replay | 0 | `bdf1266d98fd56bdd336e37d89d5a21401f96d8857c0e078644d12a86d6924ec` |
| A eval | medium-replay | 1 | `c1e7864b105b4d95a9ca3d92ce0808453f93a7937b39944e24c9d0795348642e` |
| A eval | medium-replay | 2 | `9d7daf7158c04f444dd08887ec46b008acc038140c593c4a7e7e3656f89a409f` |
| A eval | medium-replay | 3 | `3a34439d0bbca9eb3c767b0834749654e205d2231322db54e41d6f4149eafa00` |
| A eval | medium-replay | 4 | `ec92b300126a35b3934627eb70594aa6eb7457a559118e43aec5e27bec39bd51` |
| A eval | medium | 0 | `50368d60a96503af58b0ea3ae7eed599c13688536d4b745befdbb4e6054342f3` |
| A eval | medium | 1 | `75bf05671d82d2108837c877ea48519803a077745209225b8036bf72f3ba5f9c` |
| A eval | medium | 2 | `4f7ffd07232743a33497a9d38e6ed879729c2d93d21e7aec15c189e1ab2a648d` |
| A eval | medium | 3 | `0aefbe2e65f7778a0eeafc4e001486772c8612cf1eec6bccf5236d59eddf94cd` |
| A eval | medium | 4 | `a331d902be2dcf04b89a583cb57bc1df6878b38c00672c3825ba27c8e9d6fae5` |
| A eval | medium-expert | 0 | `f70b83052b4e3f1a00f8836c3c38be2e76e991adb680e59e6da58acb5c48b781` |
| A eval | medium-expert | 1 | `515ad0f9ca191a6ff7d050e1f6fd36fdf47e145519c1cee018c104edd30fc3ae` |
| A eval | medium-expert | 2 | `425b54274d6691218ce885014919f0feb35a782bc82da4a4e729a935a2060fe1` |
| A eval | medium-expert | 3 | `cb990725f8e64ceb7cb489f31f6af930de01b688a13a3c2af61e3415d2e37e6d` |
| A eval | medium-expert | 4 | `d5d109fd8becbf859f3440a7e54d2947eb5da6e5081543e8aaa898e0d93b49fe` |
| A eval | expert | 0 | `afcd16024fdeece41dbbe09676bb93e7e9d27fa0adbee364b062b78f31a9cd1d` |
| A eval | expert | 1 | `34830b95da6f87ca44fc9fd6b377de40981b0dd10cbc9f06b779f9fe95998920` |
| A eval | expert | 2 | `4b6a5e1bb914cc47e126845c10a54336f9516151845a2f7e03ddc8941239a43f` |
| A eval | expert | 3 | `0e7e6982b112256ef202f77bf70e63dcda1ab403a16c5b4c5702c0f69866953e` |
| A eval | expert | 4 | `7b9cf7a12df37bb3ca3b6881451cef7ada8d15bba3358ad65621ad04145a941a` |
| C eval | random | 0 | `eaf87c34b7997ef0bb7b94c60dc2274d141cc41bf74f582e779832183deb1c93` |
| C eval | random | 1 | `1634c48b5b327131764f4e18748b87ba0de64dee26253c75d5dadcc074cf03a0` |
| C eval | random | 2 | `a35f96a1210cde37bb508cfa98a8088effb128a7c2ac248b01d7856df35df704` |
| C eval | random | 3 | `0a62f77bdf2354faa2af42ce2881bf2282dec2b138bd8a1098fa6a6da7c076ab` |
| C eval | random | 4 | `a0394d59c2d95862a7d32bfe70f07fe0bf73fb605ffa4730d84cfed6f517cb68` |
| C eval | medium-replay | 0 | `26255fbd71b45584c6300254765a0188c50cc2f51a7fafd2a6daeefc18dda492` |
| C eval | medium-replay | 1 | `dddd95501197f715704bf2b968aa836482c5c25220bd262ebacb7ff498932612` |
| C eval | medium-replay | 2 | `062cd3c0213bdaf39e1a04d9d5316e67220c7247a48fdaac75068cd6f7e7e154` |
| C eval | medium-replay | 3 | `2ec4d19b187cb79b4289f9d9bb31d658a1c11b9adc0bc122a442ddf126987f75` |
| C eval | medium-replay | 4 | `b542a6dd0b956efd55ffc667137ed0f5e00493c9ea7377eedda2ad1d1abed3af` |
| C eval | medium | 0 | `7415806908ff94bbf12c7dd6a801a5d178af3fabc681e781b184351ff2f2b490` |
| C eval | medium | 1 | `21094f39a821ec8e0452aa0878759d72c552723da664bfb8444b6d48938d4ac1` |
| C eval | medium | 2 | `dea59078a60aa77736940ff7142fae6cc6cbd34b21a1573de680827bac81e5a2` |
| C eval | medium | 3 | `35801fc06380c0fd1f7ef3040c6f8e3ce5d95064a899311668ce59500579a2f6` |
| C eval | medium | 4 | `9bce0203ff6baa33f86166217b0bbf9b026000ce0ef5718daf40c1b89e26f6c8` |
| C eval | medium-expert | 0 | `591b552063204af4875f0fc0803c489e5ab0bc2757185e0eed11ed64c2772b96` |
| C eval | medium-expert | 1 | `166c941f00653497ef336f8fc4ccfa85bf57fa707af3271b1bce849d11b13cb4` |
| C eval | medium-expert | 2 | `b6fdcfa7a9d7deb9efa56e1aff98e8aee9046887ad59fb80426e335e3831e833` |
| C eval | medium-expert | 3 | `b3d453bd9a900e4104a58279fc4ff276e820620d42fdd2600e595f3e0afcea06` |
| C eval | medium-expert | 4 | `9a7b649e6091e0521576e4c5354f7eeed01f7673b982f94d2803981f31ae17da` |
| C eval | expert | 0 | `a74c9742ef9c40d2c57d374fec9621de6b54cef307060c91e2efbe9f445ecc0e` |
| C eval | expert | 1 | `51f55e856156efdf11f123c37ddb9c555da5349febd8b248481fc20b8562e024` |
| C eval | expert | 2 | `a24eb2474de00f54d8a277dc055b9f4269a42af4624be57e25a8f76ec6d0492d` |
| C eval | expert | 3 | `9ab7087f664fdc3cbc93d6b51a173790ba0b1e3e24a0e210977e5002615dfe5c` |
| C eval | expert | 4 | `cb92c4586cf4deb45e896cbf03053a9149946c6dd1ddf9ce727ef2fad9c223c6` |
| F0 record | medium-expert | 0 | `1b25fd781910199bc9c0fc7f9947dddacd48606891e80d2ca52c261210fdddba` |
| A2 evaluate | medium-expert | 0 | `e1a51d58a3cb2ba18f76713ff642e7d6f46bd582505b075be5af71e5c0362167` |
| F0 record | medium-expert | 1 | `4e947b7670e68a24f0716cb551565f2c574f3e402e387b5670a4d806e5245185` |
| A2 evaluate | medium-expert | 1 | `eeeda5ee57e69632e34e6da291af6bbb1dc7f2d24befa3135f423b89bf83dca4` |
| F0 record | medium-expert | 2 | `94eb72319932bdaa2b9d5b6296793a5df34c083d25309a58789afa4e0d0db3ee` |
| A2 evaluate | medium-expert | 2 | `bb1ea4cef6f9703594d85cee5ba9d6a6a64565577d5817b3a6b4526a4fe05409` |
| F0 record | medium-expert | 3 | `e5474c7c1160abd745fcf480ef6fdf502aec08f24764fe2c9424d5d1dc9d7968` |
| A2 evaluate | medium-expert | 3 | `67a2a6f5221c2be4e7f1b5e558fd5a49f79003560a56b984da85a06eaaeff786` |
| F0 record | medium-expert | 4 | `20c78eb42e75832ba9e2bb6a0f130475567d17413996ff3ae4b1e6c6440b28bd` |
| A2 evaluate | medium-expert | 4 | `b07d1ccd915235a13fcbd2a9826c514ba84353103454657ad02cb6747072bf10` |
| F0 record | expert | 0 | `247f1a2c9b730abc7e3bc370cf9683c9434a66529e0106c6efeed75fe5f59d9b` |
| A2 evaluate | expert | 0 | `b4d8a5481a9523f1a4c98c35e899797de7f5e9cfca921d4bd149dccdf4afd3c8` |
| F0 record | expert | 1 | `1a265ea0776ba6ad1a5522f145e420cede909a804d83de4defd87b6210a51b65` |
| A2 evaluate | expert | 1 | `a3df1721a1c17beaedcf84ea2ee8510ba2970cbd469edb1de0afe65207d73263` |
| F0 record | expert | 2 | `2b04bea74d15cd0af2b1ca540c755709421dce9f62cd0ad8cc7aa80c562d847e` |
| A2 evaluate | expert | 2 | `7999c0f51091124437e417864cb602e38207e9861b70f798ae053f120bb7b9d5` |
| F0 record | expert | 3 | `9bffefe960a6d0e47b8d99be182c574973fe2aa9375c7c45750916bb76d1d7fd` |
| A2 evaluate | expert | 3 | `810b1c6294262f5218c1d00fb87f09a2ff4d2571abf396a4f6d17efa1ca04032` |
| F0 record | expert | 4 | `f9c0d2cc714bf5992043e58f5e45c5935b22fca2dc319ba37620d0bf53f0d1cd` |
| A2 evaluate | expert | 4 | `bb227f10a5351b48a8cc86396cf08f85a55a1f58a6ce49acc8404e5afdb1d72f` |

### Raw points (6 decimals here; full precision in the source files)

| Arm | Quality | Policy | Bank | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---|---|---|---|---|---|---|---|
| A | random | A1 | none | 85.488217 | 83.872368 | 83.381755 | 82.371657 | 84.629420 |
| A | random | A1 | mild | 85.649188 | 84.866358 | 84.073318 | 83.107720 | 83.346481 |
| A | random | A1 | strong | 86.064905 | 85.472980 | 84.509786 | 83.806903 | 81.704101 |
| A | random | none | none | 71.752827 | 70.357983 | 71.542434 | 63.802858 | 67.227720 |
| A | random | none | mild | 69.283820 | 69.189383 | 69.162565 | 60.242919 | 65.611039 |
| A | random | none | strong | 68.176421 | 68.870781 | 67.513912 | 58.584479 | 64.225414 |
| A | random | untargeted | strong | 69.429697 | 69.500133 | 65.780166 | 61.212496 | 67.862604 |
| A | random | margin0 | strong | 90.318907 | 91.587533 | 92.361675 | 88.774051 | 86.463764 |
| A | random | plain_1s | strong | 89.844646 | 91.064405 | 89.152271 | 89.151121 | 86.677685 |
| A | random | plain_2s | strong | 89.026892 | 84.937378 | 86.619949 | 84.858274 | 83.703472 |
| A | random | plain_4s | strong | 81.809340 | 78.212268 | 77.578323 | 79.966126 | 70.993437 |
| A | random | matched_share | strong | 86.534212 | 85.848550 | 84.151114 | 85.117581 | 81.941406 |
| A | random | even_x1.33 | strong | 60.581729 | 60.440261 | 61.002778 | 49.638998 | 57.738801 |
| A | random | A1c | strong | 86.821540 | 86.049104 | 85.956260 | 86.126795 | 81.959513 |
| A | medium-replay | A1 | none | 90.739571 | 90.440356 | 87.610912 | 89.042189 | 91.977810 |
| A | medium-replay | A1 | mild | 91.049764 | 89.974256 | 86.870386 | 88.319967 | 91.620287 |
| A | medium-replay | A1 | strong | 91.354325 | 89.129900 | 86.497682 | 87.401011 | 91.287226 |
| A | medium-replay | none | none | 80.768067 | 80.233679 | 80.644420 | 80.385413 | 84.235624 |
| A | medium-replay | none | mild | 78.722922 | 78.643358 | 78.269495 | 77.802201 | 82.065016 |
| A | medium-replay | none | strong | 77.394063 | 77.951702 | 76.965721 | 76.118834 | 80.894526 |
| A | medium-replay | untargeted | strong | 76.750458 | 76.330631 | 73.648899 | 77.740908 | 74.941942 |
| A | medium-replay | margin0 | strong | 90.329403 | 91.597725 | 92.645405 | 91.570025 | 91.891762 |
| A | medium-replay | plain_1s | strong | 93.634268 | 92.472837 | 93.100886 | 90.276502 | 91.711979 |
| A | medium-replay | plain_2s | strong | 92.014121 | 90.037438 | 89.560180 | 91.212119 | 90.617341 |
| A | medium-replay | plain_4s | strong | 86.693087 | 83.728995 | 83.843899 | 84.455241 | 86.419942 |
| A | medium-replay | matched_share | strong | 91.504687 | 89.696955 | 86.858886 | 87.752649 | 90.769238 |
| A | medium-replay | even_x1.33 | strong | 72.475917 | 72.977891 | 72.363923 | 71.690377 | 76.619963 |
| A | medium-replay | A1c | strong | 92.004784 | 89.820966 | 86.978247 | 87.331459 | 91.869957 |
| A | medium | A1 | none | 94.993125 | 94.794579 | 94.582029 | 93.937420 | 94.583587 |
| A | medium | A1 | mild | 94.780443 | 94.306016 | 94.339706 | 93.223774 | 93.882744 |
| A | medium | A1 | strong | 94.324020 | 93.630562 | 93.803545 | 92.259813 | 92.935395 |
| A | medium | none | none | 88.436546 | 85.432900 | 86.500414 | 85.367960 | 85.476637 |
| A | medium | none | mild | 87.608171 | 83.233878 | 84.943172 | 83.190335 | 83.248951 |
| A | medium | none | strong | 87.243859 | 81.869554 | 83.889721 | 81.853457 | 81.776031 |
| A | medium | untargeted | strong | 86.378343 | 86.540296 | 83.177561 | 80.176917 | 82.289731 |
| A | medium | margin0 | strong | 91.616940 | 92.358514 | 94.660602 | 90.062233 | 92.246072 |
| A | medium | plain_1s | strong | 94.110302 | 93.016554 | 94.930966 | 92.678064 | 92.978027 |
| A | medium | plain_2s | strong | 94.305722 | 93.888048 | 95.038573 | 92.663313 | 93.171858 |
| A | medium | plain_4s | strong | 91.498311 | 88.779048 | 91.709092 | 87.569298 | 88.839135 |
| A | medium | matched_share | strong | 94.550358 | 94.390582 | 94.256658 | 92.567733 | 92.467796 |
| A | medium | even_x1.33 | strong | 83.212165 | 77.820675 | 78.490957 | 77.633458 | 77.262202 |
| A | medium | A1c | strong | 94.561896 | 94.192643 | 94.094976 | 91.761298 | 93.516391 |
| A | medium-expert | A1 | none | 97.304927 | 96.929255 | 96.679973 | 96.470719 | 96.693553 |
| A | medium-expert | A1 | mild | 96.193161 | 95.552521 | 95.100320 | 94.646380 | 94.796983 |
| A | medium-expert | A1 | strong | 94.929891 | 93.897380 | 93.292032 | 92.562222 | 92.657162 |
| A | medium-expert | none | none | 96.378064 | 96.365404 | 96.268244 | 96.167945 | 96.118769 |
| A | medium-expert | none | mild | 94.870356 | 94.909537 | 94.564585 | 94.342689 | 93.955685 |
| A | medium-expert | none | strong | 93.223518 | 93.178366 | 92.572715 | 92.228453 | 91.487033 |
| A | medium-expert | untargeted | strong | 93.274292 | 93.034152 | 92.304312 | 92.206019 | 91.654315 |
| A | medium-expert | margin0 | strong | 95.527782 | 92.801037 | 92.467496 | 93.628110 | 91.953946 |
| A | medium-expert | plain_1s | strong | 91.441926 | 94.074917 | 93.077308 | 92.901480 | 93.296847 |
| A | medium-expert | plain_2s | strong | 94.643169 | 94.195597 | 93.300027 | 92.458588 | 92.784568 |
| A | medium-expert | plain_4s | strong | 94.331411 | 93.288922 | 92.712101 | 91.605545 | 91.526828 |
| A | medium-expert | matched_share | strong | 94.757912 | 93.751678 | 93.234842 | 92.182311 | 92.579809 |
| A | medium-expert | even_x1.33 | strong | 93.681707 | 93.707698 | 92.995295 | 92.840721 | 91.984398 |
| A | medium-expert | A1c | strong | 94.879011 | 94.122331 | 93.219083 | 92.887022 | 92.927699 |
| A | expert | A1 | none | 96.499714 | 96.553089 | 96.369576 | 96.662698 | 96.016350 |
| A | expert | A1 | mild | 94.136771 | 94.175125 | 93.929200 | 94.376832 | 93.117379 |
| A | expert | A1 | strong | 91.516307 | 91.578194 | 91.305740 | 91.878816 | 89.945414 |
| A | expert | none | none | 95.883516 | 96.246746 | 95.909632 | 95.975315 | 95.928935 |
| A | expert | none | mild | 93.385671 | 93.974318 | 93.317616 | 93.437515 | 93.242470 |
| A | expert | none | strong | 90.758108 | 91.528873 | 90.517692 | 90.814039 | 90.351744 |
| A | expert | untargeted | strong | 91.200241 | 92.051058 | 91.129838 | 91.623819 | 90.487618 |
| A | expert | margin0 | strong | 95.212515 | 94.589971 | 95.142610 | 94.988027 | 93.345275 |
| A | expert | plain_1s | strong | 91.811545 | 92.233428 | 91.746545 | 91.902899 | 90.160516 |
| A | expert | plain_2s | strong | 91.227594 | 91.653936 | 91.342069 | 91.397641 | 90.318420 |
| A | expert | plain_4s | strong | 90.773613 | 91.446674 | 90.541147 | 90.995335 | 90.371735 |
| A | expert | matched_share | strong | 91.434747 | 91.951721 | 91.226893 | 91.761914 | 90.437780 |
| A | expert | even_x1.33 | strong | 90.887869 | 91.721951 | 90.882997 | 91.077880 | 90.610727 |
| A | expert | A1c | strong | 91.250244 | 91.548834 | 91.318361 | 91.735810 | 90.146772 |
| C | random | rho0.5+bca | none | 83.427410 | 84.447971 | 82.829407 | 80.418389 | 77.861445 |
| C | random | rho0.5+bca | mild | 82.329992 | 84.839999 | 81.637132 | 79.720201 | 76.877860 |
| C | random | rho0.5+bca | strong | 82.177094 | 85.446899 | 81.020136 | 79.884027 | 76.278206 |
| C | random | rho1 | none | 71.752827 | 70.357983 | 71.542434 | 63.802858 | 67.227720 |
| C | random | rho1 | mild | 69.283820 | 69.189383 | 69.162565 | 60.242919 | 65.611039 |
| C | random | rho1 | strong | 68.176421 | 68.870781 | 67.513912 | 58.584479 | 64.225414 |
| C | random | rho0.5 | strong | 82.150672 | 85.284095 | 80.870466 | 79.884751 | 76.202549 |
| C | random | rho0.5+range_limit | strong | 82.150672 | 85.284095 | 80.870466 | 79.884751 | 76.202549 |
| C | random | rho0.5+shuffled | strong | 82.126708 | 85.444325 | 80.887034 | 79.847347 | 76.196104 |
| C | random | rho0.5+matched_global | strong | 82.113717 | 85.084011 | 80.649666 | 79.865048 | 76.369281 |
| C | random | rho0.5+wmedian | strong | 82.163716 | 85.301118 | 81.029306 | 79.859572 | 76.293211 |
| C | random | rho0.5+wmean | strong | 82.157934 | 85.296466 | 79.265864 | 79.885925 | 76.291392 |
| C | random | rho0.5+sign | strong | 82.165919 | 85.488434 | 81.040092 | 79.915866 | 76.295423 |
| C | random | rho0.5+wttest | strong | 82.194707 | 85.700187 | 79.271573 | 79.859130 | 75.984633 |
| C | random | rho0.5+wbernstein | strong | 78.359180 | 79.614043 | 76.598169 | 75.949434 | 72.880734 |
| C | random | rho0.5+knee_clip | strong | 82.313744 | 85.718558 | 81.041839 | 79.997588 | 76.436816 |
| C | random | rho1+bca_built_for_rho1 | strong | 68.166034 | 68.557062 | 67.576334 | 58.475741 | 64.105151 |
| C | medium-replay | rho0.5+bca | none | 88.642263 | 88.902302 | 87.113320 | 88.038177 | 90.368023 |
| C | medium-replay | rho0.5+bca | mild | 87.862651 | 87.816868 | 85.786434 | 86.505714 | 88.750873 |
| C | medium-replay | rho0.5+bca | strong | 87.525920 | 86.860449 | 85.544449 | 85.972338 | 87.941883 |
| C | medium-replay | rho1 | none | 80.768067 | 80.233679 | 80.644420 | 80.385413 | 84.235624 |
| C | medium-replay | rho1 | mild | 78.722922 | 78.643358 | 78.269495 | 77.802201 | 82.065016 |
| C | medium-replay | rho1 | strong | 77.394063 | 77.951702 | 76.965721 | 76.118834 | 80.894526 |
| C | medium-replay | rho0.5 | strong | 87.506041 | 86.813794 | 85.009470 | 85.670441 | 87.807652 |
| C | medium-replay | rho0.5+range_limit | strong | 87.506041 | 86.813794 | 85.009470 | 85.670441 | 87.807652 |
| C | medium-replay | rho0.5+shuffled | strong | 87.510796 | 86.819611 | 85.280914 | 85.627125 | 87.822374 |
| C | medium-replay | rho0.5+matched_global | strong | 87.547615 | 87.025340 | 85.272226 | 86.139947 | 88.081559 |
| C | medium-replay | rho0.5+wmedian | strong | 87.507122 | 86.824289 | 85.288772 | 85.698821 | 87.954488 |
| C | medium-replay | rho0.5+wmean | strong | 87.522963 | 86.861773 | 85.289393 | 85.805406 | 87.954488 |
| C | medium-replay | rho0.5+sign | strong | 87.523815 | 86.861437 | 85.551268 | 85.976181 | 87.967116 |
| C | medium-replay | rho0.5+wttest | strong | 87.524966 | 86.860449 | 85.275995 | 85.976450 | 87.875828 |
| C | medium-replay | rho0.5+wbernstein | strong | 82.185829 | 81.926817 | 80.070796 | 80.569619 | 82.054150 |
| C | medium-replay | rho0.5+knee_clip | strong | 87.708941 | 87.109845 | 85.553886 | 86.203405 | 88.163419 |
| C | medium-replay | rho1+bca_built_for_rho1 | strong | 77.366357 | 77.946376 | 75.818044 | 76.080580 | 80.975744 |
| C | medium | rho0.5+bca | none | 93.834190 | 92.480614 | 93.339553 | 91.576918 | 92.267824 |
| C | medium | rho0.5+bca | mild | 93.659362 | 91.774924 | 92.951824 | 91.156436 | 91.097439 |
| C | medium | rho0.5+bca | strong | 93.614406 | 91.075020 | 92.759646 | 91.110640 | 90.463220 |
| C | medium | rho1 | none | 88.436546 | 85.432900 | 86.500414 | 85.367960 | 85.476637 |
| C | medium | rho1 | mild | 87.608171 | 83.233878 | 84.943172 | 83.190335 | 83.248951 |
| C | medium | rho1 | strong | 87.243859 | 81.869554 | 83.889721 | 81.853457 | 81.776031 |
| C | medium | rho0.5 | strong | 93.431760 | 90.774467 | 92.286278 | 89.766779 | 89.888288 |
| C | medium | rho0.5+range_limit | strong | 93.431878 | 90.775629 | 92.286280 | 89.784647 | 89.888904 |
| C | medium | rho0.5+shuffled | strong | 93.431697 | 90.874925 | 92.319112 | 90.278010 | 89.888035 |
| C | medium | rho0.5+matched_global | strong | 93.660995 | 91.294084 | 92.804389 | 91.300222 | 90.580453 |
| C | medium | rho0.5+wmedian | strong | 93.615883 | 91.077225 | 92.308274 | 91.107096 | 90.178302 |
| C | medium | rho0.5+wmean | strong | 93.615883 | 91.078528 | 92.440239 | 91.107096 | 90.458124 |
| C | medium | rho0.5+sign | strong | 93.621844 | 91.086270 | 92.458838 | 91.106298 | 90.464196 |
| C | medium | rho0.5+wttest | strong | 93.602213 | 91.022506 | 92.455497 | 92.026764 | 90.460059 |
| C | medium | rho0.5+wbernstein | strong | 94.291068 | 84.312276 | 93.391435 | 84.841502 | 84.753152 |
| C | medium | rho0.5+knee_clip | strong | 94.404056 | 91.669535 | 93.304197 | 93.003904 | 90.795055 |
| C | medium | rho1+bca_built_for_rho1 | strong | 87.270756 | 82.026406 | 84.019876 | 82.154208 | 81.731099 |
| C | medium-expert | rho0.5+bca | none | 98.695790 | 98.536607 | 98.096421 | 98.687446 | 98.883394 |
| C | medium-expert | rho0.5+bca | mild | 98.542018 | 98.373033 | 97.349489 | 98.545440 | 98.581374 |
| C | medium-expert | rho0.5+bca | strong | 98.348491 | 98.201371 | 96.553921 | 98.343978 | 98.232129 |
| C | medium-expert | rho1 | none | 96.378064 | 96.365404 | 96.268244 | 96.167945 | 96.118769 |
| C | medium-expert | rho1 | mild | 94.870356 | 94.909537 | 94.564585 | 94.342689 | 93.955685 |
| C | medium-expert | rho1 | strong | 93.223518 | 93.178366 | 92.572715 | 92.228453 | 91.487033 |
| C | medium-expert | rho0.5 | strong | 92.004211 | 92.182454 | 91.825841 | 91.656455 | 90.984498 |
| C | medium-expert | rho0.5+range_limit | strong | 92.539876 | 92.747327 | 92.380972 | 92.179736 | 91.612324 |
| C | medium-expert | rho0.5+shuffled | strong | 92.058494 | 92.207466 | 91.848473 | 91.672499 | 91.013338 |
| C | medium-expert | rho0.5+matched_global | strong | 99.125553 | 98.541294 | 97.151601 | 99.200111 | 98.298647 |
| C | medium-expert | rho0.5+wmedian | strong | 96.467215 | 96.470070 | 96.523300 | 96.443638 | 96.138668 |
| C | medium-expert | rho0.5+wmean | strong | 96.461170 | 96.470070 | 96.519721 | 98.341278 | 96.120339 |
| C | medium-expert | rho0.5+sign | strong | 96.461170 | 96.491400 | 96.553921 | 98.358858 | 96.157757 |
| C | medium-expert | rho0.5+wttest | strong | 98.343948 | 96.494204 | 98.348779 | 98.359502 | 98.228116 |
| C | medium-expert | rho0.5+wbernstein | strong | 99.613565 | 99.267379 | 99.673174 | 89.700575 | 99.592005 |
| C | medium-expert | rho0.5+knee_clip | strong | 99.857172 | 99.503304 | 99.775543 | 99.866858 | 99.859261 |
| C | medium-expert | rho1+bca_built_for_rho1 | strong | 97.639343 | 97.541673 | 96.192091 | 97.820909 | 97.589579 |
| C | expert | rho0.5+bca | none | 99.680353 | 99.706824 | 99.725608 | 99.709790 | 99.750643 |
| C | expert | rho0.5+bca | mild | 99.924920 | 99.977398 | 99.921545 | 99.908108 | 99.971153 |
| C | expert | rho0.5+bca | strong | 100.173127 | 100.252185 | 100.122116 | 100.121574 | 100.189015 |
| C | expert | rho1 | none | 95.883516 | 96.246746 | 95.909632 | 95.975315 | 95.928935 |
| C | expert | rho1 | mild | 93.385671 | 93.974318 | 93.317616 | 93.437515 | 93.242470 |
| C | expert | rho1 | strong | 90.758108 | 91.528873 | 90.517692 | 90.814039 | 90.351744 |
| C | expert | rho0.5 | strong | 90.056393 | 90.945309 | 89.912537 | 90.658701 | 90.123506 |
| C | expert | rho0.5+range_limit | strong | 99.676718 | 99.743302 | 99.638301 | 99.678136 | 99.612684 |
| C | expert | rho0.5+shuffled | strong | 100.167138 | 100.247771 | 100.122116 | 100.117996 | 100.191062 |
| C | expert | rho0.5+matched_global | strong | 99.883186 | 100.069942 | 99.979659 | 99.945636 | 100.117428 |
| C | expert | rho0.5+wmedian | strong | 100.181546 | 100.254698 | 100.127797 | 100.125825 | 100.192485 |
| C | expert | rho0.5+wmean | strong | 100.181546 | 100.246691 | 100.127797 | 100.125825 | 100.192485 |
| C | expert | rho0.5+sign | strong | 100.173127 | 100.252185 | 100.117799 | 100.121574 | 100.191062 |
| C | expert | rho0.5+wttest | strong | 100.177847 | 100.238742 | 100.122116 | 100.121574 | 100.186093 |
| C | expert | rho0.5+wbernstein | strong | 100.088539 | 100.149905 | 100.058653 | 100.037603 | 100.102218 |
| C | expert | rho0.5+knee_clip | strong | 100.183285 | 100.260099 | 100.129348 | 100.129329 | 100.196041 |
| C | expert | rho1+bca_built_for_rho1 | strong | 99.770079 | 99.867683 | 99.725236 | 99.690973 | 99.840202 |
| F | medium-expert | A2 | none | 96.732780 | 96.719913 | 96.775120 | 96.793379 | 96.882013 |
| F | medium-expert | A2 | mild | 95.708866 | 95.815999 | 95.619537 | 95.576532 | 95.509037 |
| F | medium-expert | A2 | strong | 94.685372 | 94.722089 | 94.253332 | 94.043237 | 93.981373 |
| F | medium-expert | F0 | none | 96.663193 | 96.602695 | 96.726258 | 96.641103 | 96.797738 |
| F | medium-expert | F0 | mild | 95.639602 | 95.695349 | 95.557769 | 95.351727 | 95.381490 |
| F | medium-expert | F0 | strong | 94.642435 | 94.591125 | 94.198346 | 93.717791 | 93.824930 |
| F | expert | A2 | none | 97.943883 | 98.138832 | 97.950784 | 97.976790 | 97.617819 |
| F | expert | A2 | mild | 97.513282 | 97.628388 | 97.220357 | 97.290567 | 96.571655 |
| F | expert | A2 | strong | 97.035176 | 97.070655 | 96.376317 | 96.445569 | 95.341672 |
| F | expert | F0 | none | 97.843605 | 98.131935 | 97.920798 | 97.982622 | 97.562358 |
| F | expert | F0 | mild | 97.335668 | 97.626473 | 97.179393 | 97.310191 | 96.484067 |
| F | expert | F0 | strong | 96.755775 | 97.084483 | 96.316280 | 96.479963 | 95.228081 |

## Notes that change no mark

- RESULTS.md prints B2, B3 and B4 with Met 'no'. They are not scored (guard void), which its summary line shows as None. Reading 'no' as 'not met' would be harmless here, since each would fail its bar anyway.
- Arm C's outcome line in RESULTS.md ("Halving the pull, guarded by BCA's ceiling, met B1 on random, medium-replay, medium; B2-B4 are not scored ...") is not one of section 6.5's fixed wordings; section 6.5 has none for a void guard. The numbers say the B1 gain is the release itself: rho 0.5 alone − rho 1 is +15.40, +8.70 and +7.90 on those qualities, and the BCA ceiling adds +0.08, +0.21 and +0.58 over rho 0.5 alone. Section 6.1 says the release is TD3+BC's alpha and is never credited to BCA.
- Section 7.4's information items 'F0's no-credit share per chunk beside A2's' and the seed-0 'F0 and A2 each minus le_none' are not reported in RESULTS.md or results.json. The second is computed above. The F0 records hold A2's per-chunk share (`a2_no_credit_share_per_chunk`).
- The C information row 'rho 0.5 + B_host_map' shows a seed-0 value only (B's maps exist only for seed 0) and carries no 'stopped by K1' tag. It is C's information item (section 6.3), not a B return, and enters no mark.
