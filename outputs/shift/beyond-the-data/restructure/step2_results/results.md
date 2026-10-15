# Step 2 results: GO

EXPLORATORY: restructured BCA, step 2 (the policy pilot). Designed after seeing seeds 0-4 and after step 1; the specification was written and frozen before any step-2 actor was trained. Outside the gate; changes no pre-registered rule and no v6/v7/step-1 file.

Mode: official. Seeds [0, 1, 2, 3, 4]. results.json SHA-256 f57aec891fe22e5ef84c0ecca6851ae4079b2eb1f61bce46746a08d3a7024dcf. prespec.json 839ed716acb7a12646c49295c9319406ce86bba4267b2c9b6829bdb17cd8da04.

## Go / no-go (fixed in prespec.json before any step-2 actor was trained)

- G1 NET of obca_k1 +102.9 against the best world-blind rival (obca_forced) +77.7 + 10: yes
- G2 keeps the improvement in the stalling world (obca_k1 - none >= -3): recorded +8.0, big waves +21.2, big kicks +19.8: yes
- G3 stalling world, strong current, obca_k1 - matched even pull +100.8 >= 15: yes

## NET across worlds, strong current (mean over seeds of [stall gain over none] - [no-stall loss against none])

| arm | NET | stalling gain | no-stall loss |
|---|---|---|---|
| clone (baseline) | -50.7 | +68.6 | +119.4 |
| obca_k1 | +102.9 | +102.9 | +0.0 |
| obca_k2 | +106.6 | +106.6 | +0.0 |
| obca_forced | +77.7 | +102.9 | +25.3 |
| fixed_gated | +89.3 | +89.3 | +0.0 |
| conformal_k1 | +97.3 | +97.3 | +0.0 |
| meangap_k1 | +57.2 | +57.4 | +0.2 |
| standin | +69.9 | +92.7 | +22.8 |
| even_pull | -6.6 | +2.2 | +8.8 |
| push_less | -1.7 | +4.6 | +6.3 |
| random_1 | -6.3 | -4.8 | +1.5 |
| random_2 | -2.4 | -0.8 | +1.6 |
| range_hinge | +76.5 | +107.4 | +30.9 |
| perfect_band | +92.6 | +93.9 | +1.3 |

## Attribution (reported; decides only the wording of the claim)

- A1 NET(conformal_k1) +97.3 >= NET(obca_k1) +102.9 - 5: no (holds in 2 of 5 seeds; per seed -6.3, -3.7, -6.5, -7.9, -3.5). Wording: a larger dose did better: conformal_k1's smaller height did worse. Nothing is attributed to the Bayesian part: the switch is the same as obca_k1's in every seed-world, so the comparison differs only in the stalling-world height, and whether BCA's height did better because it is Bayesian or only because it is larger cannot be told here.
- A2 NET(fixed_gated) +89.3 >= NET(obca_k1) +102.9 - 5: no (holds in 0 of 5 seeds; per seed -21.9, -14.3, -8.9, -14.8, -8.3). Wording: a larger dose did better: the fixed height 1.0 did worse. Nothing is attributed to calibration: the switch is the same as obca_k1's in every seed-world, so the comparison differs only in the stalling-world height.
- A3 NET(meangap_k1) +57.2 >= NET(obca_k1) +102.9 - 5: no (holds in 0 of 5 seeds; per seed -48.4, -38.1, -44.4, -55.0, -42.6). Wording: a larger dose did better: meangap_k1's smaller height did worse. Nothing is attributed to calibration: the switch is the same as obca_k1's except in no_stall/seed2 (h = 0.048 there), so the comparison differs almost only in the stalling-world height.
- Dose: NET(obca_k2) +106.6 - NET(obca_k1) +102.9 = +3.7 (plateau): obca_k2 (twice BCA's dose) is within 5 of obca_k1: BCA's height reaches the plateau of the dose curve; this says it is large enough, not that it is calibrated.
- Credit to calibration: none on these seeds: in the stalling world the height of every rival (fixed_gated, conformal_k1, meangap_k1) is below BCA's in every seed, and their switches are (almost) BCA's, so no comparison separates a calibrated height from a larger one. A failed A is worded as dose (a larger dose did better), never as credit to the Bayesian part or to calibration; obca_k2 against obca_k1 says only where BCA's height lies on the dose curve.

Dose table, strong current, per seed (dose k*h in the stalling world / stalling gain over none / that seed's NET term; in the no-stall world these arms have dose 0 except meangap_k1 in seed 2):

| seed | meangap_k1 | fixed_gated | conformal_k1 | obca_k1 | obca_k2 |
|---|---|---|---|---|---|
| 0 | 0.66 / +58.7 / +58.7 | 1.00 / +85.3 / +85.3 | 1.73 / +100.8 / +100.8 | 6.02 / +107.2 / +107.2 | 12.04 / +111.7 / +111.7 |
| 1 | 0.68 / +77.5 / +77.5 | 1.00 / +101.4 / +101.4 | 1.45 / +112.0 / +112.0 | 2.42 / +115.7 / +115.7 | 4.84 / +117.3 / +117.3 |
| 2 | 0.53 / +57.4 / +56.6 | 1.00 / +92.1 / +92.1 | 1.10 / +94.5 / +94.5 | 2.06 / +101.0 / +101.0 | 4.13 / +102.1 / +102.1 |
| 3 | 0.49 / +45.8 / +45.8 | 1.00 / +85.9 / +85.9 | 1.20 / +92.8 / +92.8 | 1.61 / +100.7 / +100.7 | 3.21 / +107.1 / +107.1 |
| 4 | 0.54 / +47.5 / +47.5 | 1.00 / +81.8 / +81.8 | 1.17 / +86.5 / +86.5 | 1.80 / +90.1 / +90.1 | 3.59 / +94.8 / +94.8 |

## Version 6's marks restated for obca_k1 (reported)

- M1 analogue: obca_k1 - matched even pull +100.8 (positive in 5 of 5): yes
- M2 analogue: random surface 1 +107.7, 2 +103.8: yes
- M3 analogue: recorded vs none +8.0, vs clone +37.2: yes
- M4 analogue: obca_k1 - matched stand-in +10.2 >= -5: yes

## Points, strong current (mean over seeds)

| arm | stalling world | no-stall world |
|---|---|---|
| none | -14.3 | +94.9 |
| clone | +54.4 | -24.4 |
| obca_k1 | +88.6 | +94.9 |
| obca_k2 | +92.3 | +94.9 |
| obca_forced | +88.6 | +69.7 |
| fixed_gated | +75.0 | +94.9 |
| conformal_k1 | +83.0 | +94.9 |
| meangap_k1 | +43.1 | +94.8 |
| standin | +78.4 | +72.1 |
| even_pull | -12.1 | +86.1 |
| push_less | -9.7 | +88.7 |
| random_1 | -19.1 | +93.4 |
| random_2 | -15.1 | +93.4 |
| range_hinge | +93.1 | +64.0 |
| perfect_band | +79.7 | +93.6 |

## Meaning

GO: outcome-reading BCA carries over into trained policies across worlds on seeds 0-4. This is still exploratory (designed after seeing seeds 0-4 and step 1); it licenses drafting version 8 (fresh seeds 10-14), not a claim.
