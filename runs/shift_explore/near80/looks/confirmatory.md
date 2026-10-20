# Near-80 (version 8): Confirmatory look, seeds 10-14

Version 8, near-80: the main test's setting with every recorded throttle from 70% to 90% also hidden

Mode: official. Seeds 10, 11, 12, 13, 14. Pre-registration 51e677ab76c8d2f7fa7779d21db7f53bd40d42ff5e6be278d4abfb34c3396d4a. This look's JSON SHA-256 1edd56c867e40cba7349505dddee0bcd814035499c7491237e938f3a5ca4f15b.

## The sentences (section 14 of the pre-registration)

- One simulated task, two worlds, five fresh seeds (10-14), four fixed test start sets of 1,000 starts each. In strong current the test states also lie outside the recorded currents, so the band's effect is read where states are new too.
- On fresh seeds 10-14, in the main setting, step 2's result did not hold: seed 10 failed its actor training phase (main setting, stalling world) and, by the pre-set rule, counts as failing every mark; seed 12 failed its actor training phase (main setting, stalling world) and, by the pre-set rule, counts as failing every mark; seed 13 failed its actor training phase (main setting, stalling world) and, by the pre-set rule, counts as failing every mark.
- Stopped before the fresh seeds: on exploratory seeds 0-4, hiding the band cost the hand-set range limit only 4.3 points against random rows (at least 5 needed). The band does not change what a range limit can do, so this version adds nothing to the main test. Nothing on seeds 10-14 was generated for the near-80 and random-rows settings.
- This stop does not touch claim 4: the main setting ran on seeds 10-14 as planned, and its result is reported separately.
- Seed 10, stalling world: the actor training (main setting) stopped (a built-in check failed). It was not rerun. Every mark that needs it fails.
- Seed 12, stalling world: the actor training (main setting) stopped (a built-in check failed). It was not rerun. Every mark that needs it fails.
- Seed 13, stalling world: the actor training (main setting) stopped (a built-in check failed). It was not rerun. Every mark that needs it fails.

## Validity (section 9.1)

- The harm (full-log actor minus no caution, near-80, stalling world, strong current): - (at least 30): fails
- No caution's share of strong-current steps past 80%: - (at least 25%): fails
- Check 10 held in 0 of 5 seeds; it did not hold in 5 (at most 1): fails
- The hand-set range limit's band cost: - (at least 5): fails

## Marks

| claim | mark | value | needed | result |
|---|---|---|---|---|
| 1 (near-80) | net lead over not available | - | 10 | fail |
| 1 (near-80) | keeps the gain, recorded conditions | - | -3 | fail |
| 1 (near-80) | keeps the gain, big waves | - | -3 | fail |
| 1 (near-80) | keeps the gain, big kicks | - | -3 | fail |
| 1 (near-80) | beats the matched even pull | - | 15 | fail |
| 4 (main) | net lead over not available | - | 10 | fail |
| 4 (main) | keeps the gain, recorded conditions | - | -3 | fail |
| 4 (main) | keeps the gain, big waves | - | -3 | fail |
| 4 (main) | keeps the gain, big kicks | - | -3 | fail |
| 4 (main) | beats the matched even pull | - | 15 | fail |
| 2 | unseen-throttle gain (positive in 0 seeds) | - | 5, and 4 seeds | fail |
| 2 | overshoot loss | - | at most 5 | fail |
| 3 | band cost, range limit minus BCA (positive in 0 seeds) | - | 5, and 4 seeds | fail |

## Claims

- claim1: not run
- claim2: not run
- claim3: not run
- claim4: fail

## Failed seeds

- claim 4, seed 10: the actor training stopped (a built-in check failed)
- claim 4, seed 12: the actor training stopped (a built-in check failed)
- claim 4, seed 13: the actor training stopped (a built-in check failed)

## Brackets

| bracket | value |
|---|---|
| S | 0 |
| N | 0 |
| X | - |
| H | - |
| W | - |
| NET | - |
| D | - |
| R | - |
| E | - |
| L | - |
| G | - |
| O | - |
| K2 | 0 |
| G0 | - |
| O0 | - |
| GZ | - |
| OZ | - |
| GV | - |
| OV | - |
| Q | - |
| A | - |
| B | - |
| C | - |
| K | 0 |
| M | 5 |
| P | - |
| V | 2.0 |
| arm | BCA's switched ramp |
| dB | - |
| dR | - |
| d0 | - |

## Points in strong current, main setting (mean over seeds)

| arm | stalling world | no-stall world |
|---|---|---|
| the controller trained without caution (none) | - | - |
| the cloned actor (clone) | - | - |
| the full-log controller (full_none) | - | - |
| the plumbing check (w0_check) | - | - |
| BCA's switched ramp (obca_k1) | - | - |
| the ramp at twice BCA's height (obca_k2) | - | - |
| the always-on ramp (obca_forced) | - | - |
| BCA's switch with height 1.0 (fixed_gated) | - | - |
| the split-conformal switch (conformal_k1) | - | - |
| the rank-test switch (meangap_k1) | - | - |
| the perfect band (perfect_band) | - | - |
| the hand-set range limit (range_hinge) | - | - |
| the matched even pull (even_pull) | - | - |
| the matched push-less penalty (push_less) | - | - |
| the first matched random surface (random_1) | - | - |
| the second matched random surface (random_2) | - | - |
| the matched range limit (standin) | - | - |

## Points in strong current, near-80 setting (mean over seeds)

| arm | stalling world | no-stall world |
|---|---|---|
| the controller trained without caution (none) | - | - |
| the cloned actor (clone) | - | - |
| the full-log controller (full_none) | - | - |
| the plumbing check (w0_check) | - | - |
| BCA's switched ramp (obca_k1) | - | - |
| the ramp at twice BCA's height (obca_k2) | - | - |
| the always-on ramp (obca_forced) | - | - |
| BCA's switch with height 1.0 (fixed_gated) | - | - |
| the split-conformal switch (conformal_k1) | - | - |
| the rank-test switch (meangap_k1) | - | - |
| the mean-difference switch (meandiff_k1) | - | - |
| the ramp started at the stall (knee_ramp) | - | - |
| the perfect band (perfect_band) | - | - |
| the hand-set range limit (range_hinge) | - | - |
| the range limit placed at the stall (knee_limit) | - | - |
| the matched even pull (even_pull) | - | - |
| the matched push-less penalty (push_less) | - | - |
| the first matched random surface (random_1) | - | - |
| the second matched random surface (random_2) | - | - |
| the matched range limit (standin) | - | - |
| version 6's BCA (version6_bca) | - | - |

## Points in strong current, random-rows setting (mean over seeds)

| arm | stalling world | no-stall world |
|---|---|---|
| the controller trained without caution (none) | - | - |
| the cloned actor (clone) | - | - |
| the plumbing check (w0_check) | - | - |
| BCA's switched ramp (obca_k1) | - | - |
| the hand-set range limit (range_hinge) | - | - |

## Band costs, stalling world, strong current

| arm | mean | per seed |
|---|---|---|
| the controller trained without caution | - | -, -, -, -, - |
| the cloned actor | - | -, -, -, -, - |
| the plumbing check | - | -, -, -, -, - |
| BCA's switched ramp | - | -, -, -, -, - |
| the hand-set range limit | - | -, -, -, -, - |

## Stopped phases

- policy.main.stall.s10: a built-in check failed
- policy.main.stall.s12: a built-in check failed
- policy.main.stall.s13: a built-in check failed

## Resources (from the ledger)

| group/phase kind | attempts | CPU-hours | peak memory (GiB) |
|---|---|---|---|
| confirmatory/clone | 10 | 0.12 | 0.45 |
| confirmatory/critic | 15 | 0.95 | 0.38 |
| confirmatory/critic_full | 10 | 0.48 | 0.37 |
| confirmatory/critic_second | 10 | 0.96 | 0.39 |
| confirmatory/data_main | 10 | 0.09 | 0.38 |
| confirmatory/evaluate | 4 | 0.10 | 0.38 |
| confirmatory/lock | 1 | 0.00 | 0.19 |
| confirmatory/policy | 7 | 1.71 | 1.19 |
| confirmatory/reference_actors | 10 | 0.47 | 0.46 |
| confirmatory/switch | 10 | 0.06 | 0.38 |
| exploratory/clone | 20 | 0.26 | 0.45 |
| exploratory/critic | 20 | 1.93 | 0.38 |
| exploratory/critic_second | 20 | 1.95 | 0.39 |
| exploratory/data_band | 10 | 0.05 | 0.25 |
| exploratory/data_main | 10 | 0.05 | 0.24 |
| exploratory/evaluate | 30 | 1.01 | 0.38 |
| exploratory/fixed_checks | 1 | 0.00 | 0.22 |
| exploratory/lock | 1 | 0.00 | 0.31 |
| exploratory/look_exploratory | 1 | 0.00 | 0.20 |
| exploratory/policy | 20 | 5.25 | 1.28 |
| exploratory/readouts | 20 | 0.12 | 0.37 |
| exploratory/reference_actors | 20 | 0.98 | 0.45 |
| exploratory/switch | 20 | 0.13 | 0.38 |
| exploratory/v6_band | 20 | 0.37 | 0.58 |

Total CPU-hours so far: 17.04.
