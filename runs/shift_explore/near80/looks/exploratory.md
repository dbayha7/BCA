# Near-80 (version 8): Exploratory look, seeds 0-4

Version 8, near-80: the main test's setting with every recorded throttle from 70% to 90% also hidden

Mode: official. Seeds 0, 1, 2, 3, 4. Pre-registration 51e677ab76c8d2f7fa7779d21db7f53bd40d42ff5e6be278d4abfb34c3396d4a. This look's JSON SHA-256 4851f5e6076690fab5bc8d2added210a10a047418d74a4ea462075959af1fdde.

Seeds 0-4 never carry a claim: every claim below is exploratory, not a claim.

## The sentences (section 14 of the pre-registration)

- Exploratory, seeds 0-4, designed after seeing them; not a claim: On seeds 0-4, with every recorded command that had a throttle from 70% to 90% on any thruster left out of training and out of BCA's calibration rows, on top of the main test's removal of the densest 15% of commands, BCA's switched ramp met all three pre-set marks. Its switch turned caution on in 5 of 5 seeds where full throttle stalls and off in 5 of 5 where it does not. In strong current it gained 100.6 points over no caution where full throttle stalls (the full-log controller gained 95.5) and lost 0.0 where full throttle helps: a net of 100.6, 42.9 more than the best rule that cannot switch itself off (the matched range limit). It stayed within 3 points of no caution or better in the other three situations, and beat the matched even pull by 100.2. Where full throttle stalls and its switch was on, its controller is the same as the always-on ramp's, so that gain comes from where the ramp starts and from its height, not from the switch.
- Exploratory, seeds 0-4, designed after seeing them; not a claim: BCA's controller used throttles from 70% to 80% that it never saw in training, to good effect: they earned 8.6 points in strong current over the same controller held to 70%, and holding it to 80% would have added only 1.0.
- Exploratory, seeds 0-4, designed after seeing them; not a claim: For scale, in strong current: the stall-aware controller gains 29.3 points from the same throttles. The controller trained without caution gained -91.6 and lost 115.6 past 80%. The hand-set range limit gained 10.6 and lost 1.0. Version 6's BCA gained -91.4 and lost 115.3. A rule that only stays inside the recorded data scores 68.5 points, against 100 for the stall-aware controller.
- Exploratory, seeds 0-4, designed after seeing them; not a claim: On seeds 0-4, hiding the band cost the hand-set range limit only 4.3 points against random rows (at least 5 needed). By the pre-set rule, this version makes no claim about the band's cost. Its numbers are reported as description.
- Exploratory, seeds 0-4, designed after seeing them; not a claim: On seeds 0-4, in the main setting (only the densest 15% of commands removed), step 2's result held: BCA's switched ramp met all three pre-set marks, with a net across worlds of 102.9, 25.3 more than the best rule that cannot switch itself off (the always-on ramp).
- Exploratory, seeds 0-4, designed after seeing them; not a claim: Hiding the band changed strong-current points where full throttle stalls, against the main setting on the same seeds: BCA's switched ramp by -1.3, the hand-set range limit by -3.4, the controller trained without caution by 1.0.
- Stopped before the fresh seeds: on exploratory seeds 0-4, hiding the band cost the hand-set range limit only 4.3 points against random rows (at least 5 needed). The band does not change what a range limit can do, so this version adds nothing to the main test. Nothing on seeds 10-14 was generated for the near-80 and random-rows settings.
- This stop does not touch claim 4: the main setting ran on seeds 10-14 as planned, and its result is reported separately.

## Validity (section 9.1)

- The harm (full-log actor minus no caution, near-80, stalling world, strong current): 95.5 (at least 30): holds
- No caution's share of strong-current steps past 80%: 69.2% (at least 25%): holds
- Check 10 held in 5 of 5 seeds; it did not hold in 0 (at most 1): holds
- The hand-set range limit's band cost: 4.3 (at least 5): fails

## Stop rules (section 8.3)

- Every stop rule can be read on every seed: yes
- The harm is there: passed
- The band changes the rival's problem: 4.3 (at least 5): failed
- The noise floor: largest difference 2.0 (BCA's switched ramp), at most 5: passed
- The near-80 settings on seeds 10-14: stopped; claim 3 kept

## Marks

| claim | mark | value | needed | result |
|---|---|---|---|---|
| 1 (near-80) | net lead over the matched range limit | 42.9 | 10 | pass |
| 1 (near-80) | keeps the gain, recorded conditions | 6.2 | -3 | pass |
| 1 (near-80) | keeps the gain, big waves | 16.5 | -3 | pass |
| 1 (near-80) | keeps the gain, big kicks | 22.8 | -3 | pass |
| 1 (near-80) | beats the matched even pull | 100.2 | 15 | pass |
| 4 (main) | net lead over the always-on ramp | 25.3 | 10 | pass |
| 4 (main) | keeps the gain, recorded conditions | 8.0 | -3 | pass |
| 4 (main) | keeps the gain, big waves | 21.2 | -3 | pass |
| 4 (main) | keeps the gain, big kicks | 19.8 | -3 | pass |
| 4 (main) | beats the matched even pull | 100.8 | 15 | pass |
| 2 | unseen-throttle gain (positive in 5 seeds) | 8.6 | 5, and 4 seeds | pass |
| 2 | overshoot loss | 1.0 | at most 5 | pass |
| 3 | band cost, range limit minus BCA (positive in 3 seeds) | 1.1 | 5, and 4 seeds | fail |

## Claims

- claim1: exploratory, not a claim (pass)
- claim2: exploratory, not a claim (pass)
- claim3: exploratory, not a claim (invalid)
- claim4: exploratory, not a claim (pass)

## Brackets

| bracket | value |
|---|---|
| S | 5 |
| N | 5 |
| X | 100.6 |
| H | 95.5 |
| W | 0.0 |
| NET | 100.6 |
| D | 42.9 |
| R | the matched range limit |
| E | 100.2 |
| L | -6.2 |
| G | 8.6 |
| O | 1.0 |
| K2 | 5 |
| G0 | -91.6 |
| O0 | 115.6 |
| GZ | 10.6 |
| OZ | 1.0 |
| GV | -91.4 |
| OV | 115.3 |
| Q | 68.5 |
| A | 3.2 |
| B | 4.3 |
| C | 1.1 |
| K | 3 |
| M | 0 |
| P | 69.2% |
| V | 2.0 |
| arm | BCA's switched ramp |
| dB | -1.3 |
| dR | -3.4 |
| d0 | 1.0 |

## Points in strong current, main setting (mean over seeds)

| arm | stalling world | no-stall world |
|---|---|---|
| the controller trained without caution (none) | -14.3 | 94.9 |
| the cloned actor (clone) | 54.4 | -24.4 |
| the full-log controller (full_none) | 84.6 | - |
| the plumbing check (w0_check) | -14.3 | 94.9 |
| BCA's switched ramp (obca_k1) | 88.6 | 94.9 |
| the ramp at twice BCA's height (obca_k2) | 92.3 | 94.9 |
| the always-on ramp (obca_forced) | 88.6 | 69.7 |
| BCA's switch with height 1.0 (fixed_gated) | 75.0 | 94.9 |
| the split-conformal switch (conformal_k1) | 83.0 | 94.9 |
| the rank-test switch (meangap_k1) | 43.1 | 94.8 |
| the perfect band (perfect_band) | 79.7 | 93.6 |
| the hand-set range limit (range_hinge) | 93.1 | 64.0 |
| the matched even pull (even_pull) | -12.1 | 86.1 |
| the matched push-less penalty (push_less) | -9.7 | 88.7 |
| the first matched random surface (random_1) | -19.1 | 93.4 |
| the second matched random surface (random_2) | -15.1 | 93.4 |
| the matched range limit (standin) | 78.4 | 72.1 |

## Points in strong current, near-80 setting (mean over seeds)

| arm | stalling world | no-stall world |
|---|---|---|
| the controller trained without caution (none) | -13.2 | 93.2 |
| the cloned actor (clone) | 31.7 | -53.8 |
| the full-log controller (full_none) | 82.3 | - |
| the plumbing check (w0_check) | -13.2 | 93.2 |
| BCA's switched ramp (obca_k1) | 87.4 | 93.2 |
| the ramp at twice BCA's height (obca_k2) | 85.5 | 93.2 |
| the always-on ramp (obca_forced) | 87.4 | 41.6 |
| BCA's switch with height 1.0 (fixed_gated) | 82.7 | 93.2 |
| the split-conformal switch (conformal_k1) | 87.1 | 92.3 |
| the rank-test switch (meangap_k1) | 49.7 | 92.5 |
| the mean-difference switch (meandiff_k1) | 49.7 | 93.2 |
| the ramp started at the stall (knee_ramp) | 87.7 | 93.2 |
| the perfect band (perfect_band) | 84.9 | 92.9 |
| the hand-set range limit (range_hinge) | 89.8 | 44.8 |
| the range limit placed at the stall (knee_limit) | 78.8 | 79.8 |
| the matched even pull (even_pull) | -12.8 | 79.1 |
| the matched push-less penalty (push_less) | -5.3 | 81.3 |
| the first matched random surface (random_1) | -17.6 | 92.0 |
| the second matched random surface (random_2) | -16.2 | 91.7 |
| the matched range limit (standin) | 90.5 | 47.1 |
| version 6's BCA (version6_bca) | -13.0 | 93.1 |

## Points in strong current, random-rows setting (mean over seeds)

| arm | stalling world | no-stall world |
|---|---|---|
| the controller trained without caution (none) | -19.9 | 94.5 |
| the cloned actor (clone) | 48.7 | -49.1 |
| the plumbing check (w0_check) | -19.9 | 94.5 |
| BCA's switched ramp (obca_k1) | 90.6 | 94.5 |
| the hand-set range limit (range_hinge) | 94.1 | 56.3 |

## Band costs, stalling world, strong current

| arm | mean | per seed |
|---|---|---|
| the controller trained without caution | -6.7 | 0.9, -5.6, -8.4, -12.5, -7.8 |
| the cloned actor | 17.1 | 54.7, -1.5, 17.7, 10.9, 3.6 |
| the plumbing check | -6.7 | 0.9, -5.6, -8.4, -12.5, -7.8 |
| BCA's switched ramp | 3.2 | 14.7, 3.2, 8.3, -2.9, -7.3 |
| the hand-set range limit | 4.3 | 7.7, 4.3, 4.8, 5.0, -0.3 |

## Resources (from the ledger)

| group/phase kind | attempts | CPU-hours | peak memory (GiB) |
|---|---|---|---|
| exploratory/clone | 20 | 0.26 | 0.45 |
| exploratory/critic | 20 | 1.93 | 0.38 |
| exploratory/critic_second | 20 | 1.95 | 0.39 |
| exploratory/data_band | 10 | 0.05 | 0.25 |
| exploratory/data_main | 10 | 0.05 | 0.24 |
| exploratory/evaluate | 30 | 1.01 | 0.38 |
| exploratory/fixed_checks | 1 | 0.00 | 0.22 |
| exploratory/lock | 1 | 0.00 | 0.31 |
| exploratory/policy | 20 | 5.25 | 1.28 |
| exploratory/readouts | 20 | 0.12 | 0.37 |
| exploratory/reference_actors | 20 | 0.98 | 0.45 |
| exploratory/switch | 20 | 0.13 | 0.38 |
| exploratory/v6_band | 20 | 0.37 | 0.58 |

Total CPU-hours so far: 12.09.
