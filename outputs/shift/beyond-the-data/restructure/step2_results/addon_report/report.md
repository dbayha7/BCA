# Step 2 add-on: regret, paired standard errors, cost split and lower tail

EXPLORATORY reporting add-on to step 2 of the restructured BCA (the policy pilot). It decides nothing: NET, G1-G3, A1-A3, version 6's marks and step 2's verdict are step 2's own and are not changed. Written and committed before step 2's official results existed.

Mode: official. Seeds [0, 1, 2, 3, 4]. report.json SHA-256 fd19f0c857d5c95f498c136a34ea7ffa170d8b566f4268787a985b038b7d0bba. Written 2026-10-14T23:41:57-0400.

**This report decides nothing.** Step 2's verdict, copied from its results.json and unchanged: **GO** (G1 yes, G2 yes, G3 yes).

Every check in section 6 passed.

## Read this first

1. 100 points is the stall-aware reference, not the best known controller. In the stalling world it is the stall-aware controller (the textbook linear rule cut off at 80% throttle); in the no-stall world it is the rising-thrust controller (the same rule, allowed full throttle). Step 2's frozen PRESPEC.md calls it "the best known controller"; read that as "the stall-aware reference". The best possible scores are 100.8 / 109.0 / 102.6 / 106.7 points in the stalling world and 100.6 / 104.2 / 101.0 / 103.9 in the no-stall world (recorded conditions / strong current / big waves / big kicks); section 1 gives the exact values this report used.
2. All points are measured on four fixed banks of 1,000 starts, one bank per situation. Every arm, seed and world is scored on the same starts. The numbers describe these starts; the start-to-start standard error in section 4 says how much a mean could move with another bank of 1,000 starts.
3. Penalties are in the critic's discounted value units (one unit of predicted discounted reward), while scores are undiscounted 100-step points. A penalty's size (the height h, or the dose k*h) therefore does not translate into points.
4. BCA treats the 5 calibration rows it takes from each episode as independent. The measured design effect (how much the likeness of rows from one episode inflates the uncertainty; 1 means none) is 1.0-1.4. Step 1's certificate survives thinning to one row per episode in all 10 combinations of seed and world; this was measured with the gate's critic as a stand-in, because step 1's own critics are not on this machine.
5. Read the attribution ladder support-first. The range hinge and the matched stand-in are the support rule alone: they look only at where the recorded data are, never at an outcome. obca_forced is BCA's switch test: it keeps BCA's stalling-world height in both worlds, so in the stalling world it is identical to obca_k1 (same switch, height and k; section 6 checks the trained actors), and NET(obca_k1) - NET(obca_forced) comes only from the no-stall world, that is, from BCA switching itself off there. fixed_gated is the fixed height, meangap_k1 the uncalibrated error scale and conformal_k1 plain split conformal. In the stalling world the best possible plan never goes past 80% throttle, so refusing unrecorded throttle already has the right shape there: the stalling half of NET can be won without any calibration, and the no-stall half is what tests whether BCA knows when not to refuse.
6. Regret changes no ranking, no difference and no verdict: on a given bank the best possible score is one number for every arm, so a difference between two arms is the same in regret as in points. In particular NET(x) = total regret of none - total regret of x (both worlds, strong current).

## The four situations

- Recorded conditions: the conditions of the training log, inside its range.
- Strong current: currents 0.70-1.10, against the training log's 0.10-0.70. It lies entirely outside the training range.
- Big waves: wave sizes 0.40-0.70, against the training log's 0.10-0.40. It lies entirely outside the training range.
- Big kicks: almost every start (98.6-99.1%) lies beyond every recorded float state, and the floats settle back within about 10-30 steps. It is a short excursion beyond the data, not a heavier tail.

## Words used here

- **Points**: the score of an actor on one bank: 0 for the recording controller (the controller that made the training log), 100 for the stall-aware reference (note 1), from exact 100-step returns.
- **Return**: the score of one 100-step episode: the sum of its rewards. Each step's reward is minus (the sum of squares of the 10 float numbers, the five floats' positions and speeds, plus 0.3 times the sum of squares of the four throttles). Closer to 0 is better.
- **Best possible**: the score of the exact best list of 100 commands from each start (the optimal reference, outputs/shift/oracle). No controller can beat it on any start.
- **Regret**: best possible points minus the arm's points, on the same bank. 0 would be perfect; smaller is better.
- **Total regret across both worlds**: regret in the stalling world plus regret in the no-stall world, strong current.
- **Paired difference**: the difference between two arms measured on the same starts, which cancels most of the luck of the starts.
- **Standard error (SE)**: how much a mean would typically move if the experiment were repeated with other random draws. Section 4 gives three kinds (SE starts, SE seeds, SE).
- **Tracking cost**: per episode, the sum over the 100 steps of the squared distance of the floats from rest (the sum of squares of the 10 float numbers): the part of the cost from the floats being away from rest.
- **Energy cost**: per episode, 0.3 times the sum over the 100 steps of the squared throttles. Tracking cost plus energy cost equals minus the return.
- **Late peak**: per episode, the largest distance of the floats from rest (the square root of the sum of squares of the 10 float numbers) over the states after the 10th step (steps 10-99), once the start's own push has had time to die down; then the mean over the bank. Over all steps the largest distance is usually the start itself, which no controller changes.
- **10th percentile**: of per-start return: the return that 10% of the 1,000 starts fall below (linear interpolation, numpy's default).
- **Worst-10% mean**: the mean return of the worst 100 of the 1,000 starts.
- **NET**: step 2's score of an arm x: [points of x minus points of none, stalling world] minus [points of none minus points of x, no-stall world], strong current, mean over seeds.
- **G1-G3, A1-A3, M1-M4**: step 2's go/no-go conditions, its attribution comparisons and version 6's marks restated (step 2's PRESPEC.md). This report adds standard errors beside the differences they read; it does not re-decide them.
- **none, clone, full_none**: the actor trained against the frozen critic with no caution; the actor that copies the recorded commands; the no-caution actor of a critic trained on the full log (stalling world only, a reference).
- **Arms**: obca_k1 (BCA's outcome ramp, the primary arm), obca_k2 (twice its dose), obca_forced (always on), fixed_gated, conformal_k1, meangap_k1, the matched controls even_pull, push_less, random_1 and random_2, the matched stand-in (range hinge), the room's range_hinge and the perfect band (a reference that knows the true error, not a method).

## 1. The best possible points

From the exact optimal reference (outputs/shift/oracle/results.json, bank means; nothing of it was recomputed here). 100 points is the stall-aware reference, not the best possible.

| world | recorded | strong current | big waves | big kicks |
|---|---|---|---|---|
| stalling world | 100.82 | 108.97 | 102.62 | 106.69 |
| no-stall world | 100.58 | 104.16 | 101.03 | 103.88 |

Read from /home/user/bca/outputs/shift/oracle/results.json (SHA-256 816d582e7588188f..., modified 2026-10-14T22:08:00-0400).

## 2. Regret beside points, strong current

Each cell is points / regret, mean over seeds [0, 1, 2, 3, 4]. The points are step 2's own (values.json); the regret is the best possible points (section 1) minus them. Smaller regret is better. "Both worlds" is the total regret (stalling plus no-stall). NET(x) = total regret of none - total regret of x.

| arm | stalling world | no-stall world | both worlds |
|---|---|---|---|
| 100-point reference | 100.0 / 9.0 | 100.0 / 4.2 | 13.1 |
| recording controller | 0.0 / 109.0 | 0.0 / 104.2 | 213.1 |
| none | -14.3 / 123.3 | 94.9 / 9.2 | 132.5 |
| full_none | 84.6 / 24.3 | - | - |
| clone | 54.4 / 54.6 | -24.4 / 128.6 | 183.2 |
| obca_k1 | 88.6 / 20.3 | 94.9 / 9.2 | 29.5 |
| obca_k2 | 92.3 / 16.7 | 94.9 / 9.2 | 25.9 |
| obca_forced | 88.6 / 20.3 | 69.7 / 34.5 | 54.8 |
| fixed_gated | 75.0 / 34.0 | 94.9 / 9.2 | 43.2 |
| conformal_k1 | 83.0 / 25.9 | 94.9 / 9.2 | 35.1 |
| meangap_k1 | 43.1 / 65.9 | 94.8 / 9.4 | 75.2 |
| standin | 78.4 / 30.5 | 72.1 / 32.1 | 62.6 |
| even_pull | -12.1 / 121.1 | 86.1 / 18.0 | 139.1 |
| push_less | -9.7 / 118.7 | 88.7 / 15.5 | 134.2 |
| random_1 | -19.1 / 128.0 | 93.4 / 10.7 | 138.8 |
| random_2 | -15.1 / 124.1 | 93.4 / 10.8 | 134.9 |
| range_hinge | 93.1 / 15.8 | 64.0 / 40.1 | 56.0 |
| perfect_band | 79.7 / 29.3 | 93.6 / 10.6 | 39.9 |

Per-seed values of every arm, situation and world are in report.json (points, regret).

## 3. Regret beside points, the other situations

**Stalling world** (points / regret, mean over seeds)

| arm | recorded | big waves | big kicks |
|---|---|---|---|
| 100-point reference | 100.0 / 0.8 | 100.0 / 2.6 | 100.0 / 6.7 |
| none | 85.5 / 15.3 | 70.0 / 32.6 | 52.5 / 54.2 |
| full_none | 91.3 / 9.5 | 86.6 / 16.0 | 64.3 / 42.4 |
| clone | 56.3 / 44.5 | 43.6 / 59.0 | 2.0 / 104.7 |
| obca_k1 | 93.5 / 7.3 | 91.3 / 11.3 | 72.4 / 34.3 |
| obca_k2 | 94.0 / 6.9 | 92.1 / 10.6 | 74.5 / 32.2 |
| obca_forced | 93.5 / 7.3 | 91.3 / 11.3 | 72.4 / 34.3 |
| fixed_gated | 92.9 / 7.9 | 89.5 / 13.1 | 67.2 / 39.5 |
| conformal_k1 | 93.2 / 7.7 | 90.4 / 12.2 | 69.3 / 37.4 |
| meangap_k1 | 92.1 / 8.7 | 87.3 / 15.3 | 62.6 / 44.0 |
| standin | 92.9 / 7.9 | 90.0 / 12.6 | 70.3 / 36.4 |
| even_pull | 83.2 / 17.6 | 66.6 / 36.0 | 46.7 / 60.0 |
| push_less | 82.6 / 18.2 | 69.8 / 32.8 | 55.2 / 51.5 |
| random_1 | 82.4 / 18.4 | 67.9 / 34.7 | 49.9 / 56.8 |
| random_2 | 83.3 / 17.5 | 67.3 / 35.3 | 51.2 / 55.5 |
| range_hinge | 93.3 / 7.5 | 91.6 / 11.1 | 74.9 / 31.8 |
| perfect_band | 93.4 / 7.4 | 89.7 / 13.0 | 69.4 / 37.3 |

**No-stall world** (points / regret, mean over seeds)

| arm | recorded | big waves | big kicks |
|---|---|---|---|
| 100-point reference | 100.0 / 0.6 | 100.0 / 1.0 | 100.0 / 3.9 |
| none | 93.5 / 7.1 | 88.1 / 12.9 | 91.2 / 12.6 |
| clone | 48.1 / 52.5 | 11.3 / 89.8 | 9.2 / 94.7 |
| obca_k1 | 93.5 / 7.1 | 88.1 / 12.9 | 91.2 / 12.6 |
| obca_k2 | 93.5 / 7.1 | 88.1 / 12.9 | 91.2 / 12.6 |
| obca_forced | 92.9 / 7.7 | 85.3 / 15.7 | 88.4 / 15.5 |
| fixed_gated | 93.5 / 7.1 | 88.1 / 12.9 | 91.2 / 12.6 |
| conformal_k1 | 93.5 / 7.1 | 88.1 / 12.9 | 91.2 / 12.6 |
| meangap_k1 | 93.4 / 7.1 | 88.1 / 12.9 | 91.2 / 12.7 |
| standin | 92.5 / 8.1 | 84.8 / 16.3 | 86.2 / 17.6 |
| even_pull | 87.6 / 12.9 | 77.4 / 23.6 | 81.1 / 22.8 |
| push_less | 86.9 / 13.7 | 81.1 / 19.9 | 86.7 / 17.2 |
| random_1 | 90.9 / 9.7 | 85.4 / 15.7 | 89.8 / 14.1 |
| random_2 | 91.7 / 8.9 | 85.6 / 15.4 | 90.3 / 13.6 |
| range_hinge | 92.4 / 8.1 | 83.9 / 17.1 | 85.3 / 18.6 |
| perfect_band | 93.1 / 7.5 | 87.5 / 13.6 | 90.7 / 13.2 |

## 4. Paired standard errors of step 2's pooled differences

How they are computed. One extra exact rollout of every scored locked actor on the full banks keeps each start's return. For one difference (for example obca_k1 minus none, or a NET), each start i of the bank gives a paired difference e(s, i) in seed s: the arms' points on that same start, combined the way step 2 combines the bank means (for a NET: [x - none, stalling world] + [x - none, no-stall world], on the same start, since both worlds use the same bank). The value is the mean over seeds and starts, which equals step 2's number (checked in section 6).

- **SE starts** = sd over starts of (the mean over seeds of e(s, i)) / sqrt(1,000): how much the value would move with another bank of 1,000 starts. The same bank serves every seed, so the start luck is shared by the seeds and is counted once.
- **SE seeds** = sd over seeds of (the mean over starts of e(s, i)) / sqrt(number of seeds): the plain seed-to-seed standard error, how much the value would move with another set of seeds on this bank.
- **SE** = sqrt(SE starts^2 + SE seeds^2): both kinds of luck together (the usual formula for a seeds-by-starts design). It is slightly conservative: it counts the seed-by-start interaction once more than an exact two-way analysis, which report.json also gives (se_two_way).
- Every sd divides by n - 1. With one seed, SE seeds and SE are not defined ("-").

**NET, strong current**

| difference | value | SE | SE starts | SE seeds |
|---|---|---|---|---|
| NET(clone) | -50.7 | 8.73 | 0.54 | 8.71 |
| NET(obca_k1) | +102.9 | 4.49 | 1.58 | 4.20 |
| NET(obca_k2) | +106.6 | 4.23 | 1.69 | 3.88 |
| NET(obca_forced) | +77.7 | 4.81 | 1.04 | 4.69 |
| NET(fixed_gated) | +89.3 | 3.65 | 1.22 | 3.44 |
| NET(conformal_k1) | +97.3 | 4.54 | 1.43 | 4.31 |
| NET(meangap_k1) | +57.2 | 5.70 | 0.64 | 5.66 |
| NET(standin) | +69.9 | 6.56 | 0.87 | 6.50 |
| NET(even_pull) | -6.6 | 1.90 | 0.35 | 1.86 |
| NET(push_less) | -1.7 | 2.97 | 0.17 | 2.96 |
| NET(random_1) | -6.3 | 2.93 | 0.09 | 2.93 |
| NET(random_2) | -2.4 | 1.95 | 0.07 | 1.95 |
| NET(range_hinge) | +76.5 | 3.00 | 1.10 | 2.79 |
| NET(perfect_band) | +92.6 | 3.91 | 1.35 | 3.67 |

**G1: NET(obca_k1) against each world-blind rival**

| difference | value | SE | SE starts | SE seeds |
|---|---|---|---|---|
| NET(obca_k1) - NET(standin) | +33.1 | 4.02 | 0.81 | 3.94 |
| NET(obca_k1) - NET(even_pull) | +109.6 | 5.47 | 1.87 | 5.14 |
| NET(obca_k1) - NET(random_1) | +109.2 | 4.74 | 1.63 | 4.46 |
| NET(obca_k1) - NET(random_2) | +105.3 | 3.71 | 1.58 | 3.36 |
| NET(obca_k1) - NET(obca_forced) | +25.3 | 2.26 | 0.59 | 2.18 |

**G2 (and M3's none parts): stalling world, obca_k1 - none**

| difference | value | SE | SE starts | SE seeds |
|---|---|---|---|---|
| recorded conditions: obca_k1 - none | +8.0 | 0.70 | 0.59 | 0.39 |
| big waves: obca_k1 - none | +21.2 | 1.28 | 0.93 | 0.88 |
| big kicks: obca_k1 - none | +19.8 | 1.44 | 0.77 | 1.22 |

**G3 (= M1): stalling world, strong current**

| difference | value | SE | SE starts | SE seeds |
|---|---|---|---|---|
| obca_k1 - even_pull (matched) | +100.8 | 5.09 | 1.76 | 4.78 |

**A1-A3 and the dose readout**

| difference | value | SE | SE starts | SE seeds |
|---|---|---|---|---|
| A1: NET(conformal_k1) - NET(obca_k1) | -5.6 | 0.87 | 0.19 | 0.85 |
| A2: NET(fixed_gated) - NET(obca_k1) | -13.6 | 2.50 | 0.46 | 2.46 |
| A3: NET(meangap_k1) - NET(obca_k1) | -45.7 | 3.12 | 1.28 | 2.85 |
| dose: NET(obca_k2) - NET(obca_k1) | +3.7 | 1.00 | 0.12 | 1.00 |

**M2: stalling world, strong current**

| difference | value | SE | SE starts | SE seeds |
|---|---|---|---|---|
| obca_k1 - random_1 (matched) | +107.7 | 4.91 | 1.64 | 4.62 |
| obca_k1 - random_2 (matched) | +103.8 | 4.01 | 1.59 | 3.68 |

**M3's clone part: stalling world, recorded conditions**

| difference | value | SE | SE starts | SE seeds |
|---|---|---|---|---|
| obca_k1 - clone | +37.2 | 1.96 | 0.57 | 1.87 |

**M4: stalling world, strong current**

| difference | value | SE | SE starts | SE seeds |
|---|---|---|---|---|
| obca_k1 - standin (matched) | +10.2 | 4.66 | 0.34 | 4.64 |

**NET parts, strong current: stalling gain over none**

| difference | value | SE | SE starts | SE seeds |
|---|---|---|---|---|
| clone - none, stalling world | +68.6 | 4.17 | 1.49 | 3.89 |
| obca_k1 - none, stalling world | +102.9 | 4.49 | 1.58 | 4.20 |
| obca_k2 - none, stalling world | +106.6 | 4.23 | 1.69 | 3.88 |
| obca_forced - none, stalling world | +102.9 | 4.49 | 1.58 | 4.20 |
| fixed_gated - none, stalling world | +89.3 | 3.65 | 1.22 | 3.44 |
| conformal_k1 - none, stalling world | +97.3 | 4.54 | 1.43 | 4.31 |
| meangap_k1 - none, stalling world | +57.4 | 5.70 | 0.64 | 5.66 |
| standin - none, stalling world | +92.7 | 7.38 | 1.30 | 7.26 |
| even_pull - none, stalling world | +2.2 | 1.56 | 0.26 | 1.53 |
| push_less - none, stalling world | +4.6 | 3.10 | 0.18 | 3.10 |
| random_1 - none, stalling world | -4.8 | 3.36 | 0.10 | 3.36 |
| random_2 - none, stalling world | -0.8 | 1.61 | 0.07 | 1.61 |
| range_hinge - none, stalling world | +107.4 | 2.85 | 1.74 | 2.25 |
| perfect_band - none, stalling world | +93.9 | 3.65 | 1.36 | 3.39 |

**NET parts, strong current: no-stall loss against none**

| difference | value | SE | SE starts | SE seeds |
|---|---|---|---|---|
| none - clone, no-stall world | +119.4 | 9.75 | 1.39 | 9.65 |
| none - obca_k1, no-stall world | +0.0 | 0.00 | 0.00 | 0.00 |
| none - obca_k2, no-stall world | +0.0 | 0.00 | 0.00 | 0.00 |
| none - obca_forced, no-stall world | +25.3 | 2.26 | 0.59 | 2.18 |
| none - fixed_gated, no-stall world | +0.0 | 0.00 | 0.00 | 0.00 |
| none - conformal_k1, no-stall world | +0.0 | 0.00 | 0.00 | 0.00 |
| none - meangap_k1, no-stall world | +0.2 | 0.16 | 0.01 | 0.16 |
| none - standin, no-stall world | +22.8 | 2.96 | 0.49 | 2.92 |
| none - even_pull, no-stall world | +8.8 | 0.66 | 0.12 | 0.65 |
| none - push_less, no-stall world | +6.3 | 1.02 | 0.03 | 1.02 |
| none - random_1, no-stall world | +1.5 | 0.61 | 0.03 | 0.61 |
| none - random_2, no-stall world | +1.6 | 0.62 | 0.03 | 0.62 |
| none - range_hinge, no-stall world | +30.9 | 1.85 | 0.69 | 1.72 |
| none - perfect_band, no-stall world | +1.3 | 0.30 | 0.02 | 0.29 |

**NET in recorded conditions (information only in step 2)**

| difference | value | SE | SE starts | SE seeds |
|---|---|---|---|---|
| NET(clone), recorded | -74.6 | 2.03 | 0.97 | 1.78 |
| NET(obca_k1), recorded | +8.0 | 0.70 | 0.59 | 0.39 |
| NET(obca_k2), recorded | +8.5 | 0.74 | 0.59 | 0.45 |
| NET(obca_forced), recorded | +7.5 | 0.62 | 0.54 | 0.31 |
| NET(fixed_gated), recorded | +7.4 | 0.66 | 0.56 | 0.35 |
| NET(conformal_k1), recorded | +7.7 | 0.70 | 0.58 | 0.40 |
| NET(meangap_k1), recorded | +6.6 | 0.64 | 0.49 | 0.41 |
| NET(standin), recorded | +6.5 | 0.68 | 0.53 | 0.43 |
| NET(even_pull), recorded | -8.1 | 0.64 | 0.19 | 0.61 |
| NET(push_less), recorded | -9.5 | 0.61 | 0.19 | 0.58 |
| NET(random_1), recorded | -5.7 | 0.46 | 0.08 | 0.46 |
| NET(random_2), recorded | -4.0 | 0.20 | 0.07 | 0.19 |
| NET(range_hinge), recorded | +6.8 | 0.63 | 0.52 | 0.34 |
| NET(perfect_band), recorded | +7.5 | 0.66 | 0.57 | 0.33 |

## 5. Where the cost comes from, and the lower tail

From the same rollouts, mean over seeds [0, 1, 2, 3, 4], in return units per episode (not points). Tracking + energy = minus the mean return. Late peak: mean over starts (largest over starts in report.json). 10th percentile and worst-10% mean: of per-start return (more negative is worse). The best possible row is from the oracle's results.json, which has no 10th percentile or worst-10% mean.

**Stalling world, strong current**

| arm | tracking | energy | late peak | 10th pct | worst 10% |
|---|---|---|---|---|---|
| best possible | 58.3 | 32.1 | 1.03 | - | - |
| none | 236.0 | 39.4 | 2.15 | -464.7 | -509.2 |
| full_none | 96.0 | 30.9 | 1.32 | -226.1 | -260.8 |
| clone | 149.1 | 23.2 | 1.69 | -273.1 | -321.6 |
| obca_k1 | 91.0 | 29.8 | 1.28 | -215.7 | -250.3 |
| obca_k2 | 85.8 | 29.5 | 1.24 | -201.8 | -232.7 |
| obca_forced | 91.0 | 29.8 | 1.28 | -215.7 | -250.3 |
| fixed_gated | 110.4 | 31.0 | 1.44 | -268.5 | -316.0 |
| conformal_k1 | 98.8 | 30.5 | 1.34 | -238.3 | -277.1 |
| meangap_k1 | 156.5 | 32.7 | 1.76 | -376.2 | -426.5 |
| standin | 106.2 | 30.0 | 1.40 | -257.0 | -298.6 |
| even_pull | 235.2 | 37.0 | 2.16 | -474.8 | -523.8 |
| push_less | 234.6 | 33.9 | 2.14 | -463.8 | -509.2 |
| random_1 | 242.8 | 39.8 | 2.18 | -475.3 | -523.3 |
| random_2 | 236.8 | 39.8 | 2.15 | -467.7 | -512.7 |
| range_hinge | 85.5 | 28.6 | 1.24 | -195.2 | -226.5 |
| perfect_band | 102.7 | 31.6 | 1.35 | -246.9 | -281.4 |

**No-stall world, strong current**

| arm | tracking | energy | late peak | 10th pct | worst 10% |
|---|---|---|---|---|---|
| best possible | 28.7 | 36.7 | 0.71 | - | - |
| none | 37.4 | 35.1 | 0.79 | -99.5 | -112.6 |
| clone | 141.5 | 24.4 | 1.61 | -239.6 | -264.4 |
| obca_k1 | 37.4 | 35.1 | 0.79 | -99.5 | -112.6 |
| obca_k2 | 37.4 | 35.1 | 0.79 | -99.5 | -112.6 |
| obca_forced | 61.2 | 31.1 | 1.03 | -139.5 | -156.9 |
| fixed_gated | 37.4 | 35.1 | 0.79 | -99.5 | -112.6 |
| conformal_k1 | 37.4 | 35.1 | 0.79 | -99.5 | -112.6 |
| meangap_k1 | 37.6 | 35.1 | 0.79 | -99.9 | -113.0 |
| standin | 59.2 | 31.2 | 1.00 | -133.8 | -149.3 |
| even_pull | 46.3 | 33.2 | 0.89 | -110.4 | -124.1 |
| push_less | 47.0 | 30.5 | 0.87 | -103.7 | -116.8 |
| random_1 | 38.5 | 35.2 | 0.80 | -99.9 | -113.1 |
| random_2 | 38.1 | 35.7 | 0.80 | -100.1 | -113.1 |
| range_hinge | 66.3 | 30.5 | 1.08 | -147.3 | -165.6 |
| perfect_band | 39.5 | 34.1 | 0.81 | -101.1 | -114.2 |

## 6. Checks

- Step 2 as frozen: prespec.json 839ed716acb7a126..., code manifest 3222aedcfe80b72d... (both pinned in report.py): passed.
- Every lock's actors checked against the SHA-256 in lock.json: 340 actors, passed. Seed-worlds: no_stall/seed0 values, no_stall/seed1 values, no_stall/seed2 values, no_stall/seed3 values, no_stall/seed4 values, stall/seed0 values, stall/seed1 values, stall/seed2 values, stall/seed3 values, stall/seed4 values.
- The extra rollouts reproduce step 2's evaluation: 620 of 620 bank means of the return exactly equal to values.json; largest points difference 0.0e+00: yes.
- Tracking + energy = minus the return: largest error 1.0e-12: yes.
- Step 2's anchors equal the oracle's: largest difference 0.0e+00: yes.
- The oracle's best possible points equal those stated when this add-on was written (to 0.05): largest difference 0.038: yes.
- NET = total regret of none - total regret of x: largest difference 3.0e-14: yes.
- Pooled values from the per-start rollouts equal those from values.json: largest difference 2.8e-14: yes.
- Pooled values equal step 2's results.json: 73 compared, largest difference 1.7e-14: yes.
- Stalling world, obca_forced's trained actor is bit-identical to obca_k1's (same SHA-256): seed 0 yes, seed 1 yes, seed 2 yes, seed 3 yes, seed 4 yes (a reported fact).

## 7. Time

- 620 rollouts (one rollout = one actor on one situation's full bank of 1,000 starts, 100 steps): 145 s, 0.23 s each on average (largest 0.27 s).
- Loading the gate's data per seed-world: no_stall/seed0 7 s, no_stall/seed1 7 s, no_stall/seed2 7 s, no_stall/seed3 7 s, no_stall/seed4 7 s, stall/seed0 7 s, stall/seed1 7 s, stall/seed2 7 s, stall/seed3 7 s, stall/seed4 7 s.
- In all 224 s; peak memory 464 MB.

## Appendix: the other situations

**Stalling world, recorded conditions**

| arm | tracking | energy | late peak | 10th pct | worst 10% |
|---|---|---|---|---|---|
| best possible | 15.0 | 10.9 | 0.48 | - | - |
| none | 23.7 | 10.4 | 0.67 | -59.2 | -81.6 |
| full_none | 20.7 | 10.3 | 0.61 | -48.7 | -57.4 |
| clone | 41.7 | 8.0 | 0.96 | -79.8 | -92.4 |
| obca_k1 | 20.1 | 9.7 | 0.60 | -46.4 | -54.3 |
| obca_k2 | 19.8 | 9.8 | 0.59 | -46.2 | -53.8 |
| obca_forced | 20.1 | 9.7 | 0.60 | -46.4 | -54.3 |
| fixed_gated | 20.4 | 9.8 | 0.60 | -46.9 | -55.5 |
| conformal_k1 | 20.2 | 9.8 | 0.60 | -46.7 | -54.9 |
| meangap_k1 | 20.7 | 9.9 | 0.61 | -48.0 | -58.9 |
| standin | 20.5 | 9.7 | 0.61 | -47.1 | -55.2 |
| even_pull | 25.4 | 9.9 | 0.71 | -56.5 | -78.0 |
| push_less | 27.1 | 8.5 | 0.71 | -58.2 | -79.8 |
| random_1 | 25.0 | 10.8 | 0.69 | -62.4 | -84.9 |
| random_2 | 24.3 | 10.9 | 0.68 | -60.6 | -83.0 |
| range_hinge | 20.3 | 9.6 | 0.61 | -46.8 | -54.4 |
| perfect_band | 19.9 | 10.0 | 0.60 | -46.6 | -54.8 |

**Stalling world, big waves**

| arm | tracking | energy | late peak | 10th pct | worst 10% |
|---|---|---|---|---|---|
| best possible | 19.4 | 14.9 | 0.58 | - | - |
| none | 41.3 | 16.0 | 0.94 | -111.6 | -137.5 |
| full_none | 31.4 | 14.3 | 0.76 | -69.0 | -83.0 |
| clone | 64.0 | 12.0 | 1.14 | -116.3 | -136.9 |
| obca_k1 | 28.7 | 13.6 | 0.73 | -64.0 | -77.8 |
| obca_k2 | 28.1 | 13.7 | 0.72 | -63.2 | -76.4 |
| obca_forced | 28.7 | 13.6 | 0.73 | -64.0 | -77.8 |
| fixed_gated | 29.7 | 13.9 | 0.75 | -67.5 | -84.1 |
| conformal_k1 | 29.2 | 13.8 | 0.74 | -65.5 | -80.3 |
| meangap_k1 | 31.1 | 14.0 | 0.78 | -72.5 | -94.1 |
| standin | 29.7 | 13.5 | 0.75 | -66.3 | -81.5 |
| even_pull | 44.4 | 15.3 | 0.97 | -111.9 | -139.6 |
| push_less | 44.4 | 13.2 | 0.96 | -108.9 | -135.9 |
| random_1 | 42.6 | 16.3 | 0.97 | -112.4 | -138.4 |
| random_2 | 42.8 | 16.5 | 0.95 | -114.1 | -140.3 |
| range_hinge | 28.8 | 13.3 | 0.74 | -63.6 | -76.3 |
| perfect_band | 29.5 | 14.0 | 0.74 | -67.3 | -82.3 |

**Stalling world, big kicks**

| arm | tracking | energy | late peak | 10th pct | worst 10% |
|---|---|---|---|---|---|
| best possible | 141.9 | 14.9 | 1.80 | - | - |
| none | 194.0 | 16.8 | 2.31 | -364.3 | -471.0 |
| full_none | 183.4 | 15.7 | 2.28 | -337.2 | -436.7 |
| clone | 248.6 | 12.7 | 2.77 | -430.3 | -546.2 |
| obca_k1 | 176.6 | 14.5 | 2.20 | -325.5 | -420.1 |
| obca_k2 | 174.5 | 14.4 | 2.18 | -319.7 | -414.0 |
| obca_forced | 176.6 | 14.5 | 2.20 | -325.5 | -420.1 |
| fixed_gated | 181.3 | 14.9 | 2.23 | -337.7 | -439.6 |
| conformal_k1 | 179.3 | 14.8 | 2.22 | -331.3 | -431.3 |
| meangap_k1 | 185.5 | 15.3 | 2.26 | -346.7 | -454.0 |
| standin | 178.9 | 14.2 | 2.22 | -332.8 | -430.3 |
| even_pull | 200.8 | 15.9 | 2.38 | -371.9 | -483.2 |
| push_less | 194.3 | 13.9 | 2.29 | -356.7 | -465.2 |
| random_1 | 196.2 | 17.3 | 2.33 | -366.4 | -475.5 |
| random_2 | 194.9 | 17.3 | 2.32 | -364.5 | -473.5 |
| range_hinge | 174.8 | 13.7 | 2.18 | -320.7 | -414.4 |
| perfect_band | 179.1 | 15.0 | 2.23 | -329.9 | -427.9 |

**No-stall world, recorded conditions**

| arm | tracking | energy | late peak | 10th pct | worst 10% |
|---|---|---|---|---|---|
| best possible | 14.2 | 11.2 | 0.46 | - | - |
| none | 18.7 | 10.1 | 0.56 | -44.3 | -50.5 |
| clone | 43.0 | 7.9 | 0.97 | -82.5 | -95.0 |
| obca_k1 | 18.7 | 10.1 | 0.56 | -44.3 | -50.5 |
| obca_k2 | 18.7 | 10.1 | 0.56 | -44.3 | -50.5 |
| obca_forced | 19.3 | 9.8 | 0.59 | -45.3 | -52.2 |
| fixed_gated | 18.7 | 10.1 | 0.56 | -44.3 | -50.5 |
| conformal_k1 | 18.7 | 10.1 | 0.56 | -44.3 | -50.5 |
| meangap_k1 | 18.7 | 10.1 | 0.56 | -44.3 | -50.5 |
| standin | 19.5 | 9.8 | 0.59 | -45.6 | -52.5 |
| even_pull | 22.0 | 9.6 | 0.63 | -48.1 | -54.7 |
| push_less | 23.7 | 8.4 | 0.63 | -49.4 | -55.6 |
| random_1 | 19.8 | 10.4 | 0.57 | -45.5 | -51.9 |
| random_2 | 19.1 | 10.6 | 0.57 | -45.3 | -51.5 |
| range_hinge | 19.7 | 9.7 | 0.60 | -46.1 | -53.1 |
| perfect_band | 19.1 | 9.9 | 0.57 | -44.7 | -50.9 |

**No-stall world, big waves**

| arm | tracking | energy | late peak | 10th pct | worst 10% |
|---|---|---|---|---|---|
| best possible | 17.0 | 15.6 | 0.52 | - | - |
| none | 24.3 | 14.9 | 0.64 | -55.8 | -63.4 |
| clone | 66.7 | 11.6 | 1.16 | -117.7 | -134.6 |
| obca_k1 | 24.3 | 14.9 | 0.64 | -55.8 | -63.4 |
| obca_k2 | 24.3 | 14.9 | 0.64 | -55.8 | -63.4 |
| obca_forced | 27.0 | 13.6 | 0.70 | -60.2 | -69.6 |
| fixed_gated | 24.3 | 14.9 | 0.64 | -55.8 | -63.4 |
| conformal_k1 | 24.3 | 14.9 | 0.64 | -55.8 | -63.4 |
| meangap_k1 | 24.4 | 14.8 | 0.64 | -55.8 | -63.5 |
| standin | 27.3 | 13.6 | 0.70 | -60.6 | -70.0 |
| even_pull | 30.5 | 14.1 | 0.74 | -63.0 | -71.5 |
| push_less | 30.4 | 12.3 | 0.72 | -60.1 | -67.7 |
| random_1 | 25.6 | 14.9 | 0.66 | -57.3 | -65.0 |
| random_2 | 25.1 | 15.3 | 0.65 | -56.8 | -64.2 |
| range_hinge | 28.0 | 13.3 | 0.72 | -62.2 | -72.4 |
| perfect_band | 25.0 | 14.5 | 0.65 | -56.3 | -64.0 |

**No-stall world, big kicks**

| arm | tracking | energy | late peak | 10th pct | worst 10% |
|---|---|---|---|---|---|
| best possible | 128.4 | 16.7 | 1.60 | - | - |
| none | 143.6 | 15.0 | 1.81 | -260.4 | -331.5 |
| clone | 233.9 | 12.2 | 2.66 | -400.5 | -496.3 |
| obca_k1 | 143.6 | 15.0 | 1.81 | -260.4 | -331.5 |
| obca_k2 | 143.6 | 15.0 | 1.81 | -260.4 | -331.5 |
| obca_forced | 147.8 | 13.9 | 1.87 | -265.8 | -337.5 |
| fixed_gated | 143.6 | 15.0 | 1.81 | -260.4 | -331.5 |
| conformal_k1 | 143.6 | 15.0 | 1.81 | -260.4 | -331.5 |
| meangap_k1 | 143.6 | 15.0 | 1.81 | -260.5 | -331.5 |
| standin | 150.2 | 13.7 | 1.91 | -270.0 | -341.2 |
| even_pull | 155.2 | 14.3 | 1.96 | -277.1 | -350.4 |
| push_less | 150.8 | 12.7 | 1.87 | -266.6 | -337.5 |
| random_1 | 144.8 | 15.3 | 1.81 | -262.2 | -333.4 |
| random_2 | 144.1 | 15.5 | 1.81 | -262.7 | -333.0 |
| range_hinge | 151.6 | 13.4 | 1.94 | -271.9 | -343.5 |
| perfect_band | 144.5 | 14.7 | 1.82 | -261.4 | -332.5 |

## Files

- report.json: every number (per seed, per arm, per situation), the checks, the inputs' SHA-256.
- per_start.npz: the per-start return, tracking cost, energy cost and late peak of every rollout (keys world|seed|actor|situation|quantity).
- The code: outputs/shift/beyond-the-data/restructure/step2_report/report.py (README.md beside it).
