## Headline: stall/gapped (seeds [0, 1, 2, 3, 4])

Points of the rule, and in brackets the change from the actor alone (paired by seed); strong current also gives the share of the harm recovered; last two columns: share of steps switched in recorded conditions / strong current.

| Rule | Recorded conditions | Strong current | Harm recovered | Big waves | Big kicks | switched (recorded / strong) |
|---|---|---|---|---|---|---|
| never switch (the actor alone) | 86.0 | -15.0 | - | 70.3 | 57.0 | 0.00 / 0.00 |
| every step -> cloned actor | 56.0 (-29.9) | 53.9 (+68.9) | 69% | 43.2 (-27.2) | 5.1 (-51.8) | 1.00 / 1.00 |
| every step -> clip to recorded range | 93.0 (+7.0) | 91.0 (+106.0) | 107% | 90.0 (+19.7) | 84.9 (+28.0) | 0.06 / 0.77 |
| every step -> clip to full recorded range (min to max) | 86.6 (+0.6) | -13.9 (+1.1) | 1% | 71.2 (+0.9) | 72.0 (+15.1) | 0.01 / 0.08 |
| every step -> nearest recorded command | 72.9 (-13.1) | 48.1 (+63.0) | 63% | 66.0 (-4.3) | 37.9 (-19.1) | 1.00 / 1.00 |
| new state, 95% level -> clip to recorded range | 88.3 (+2.3) | 37.9 (+52.9) | 53% | 74.6 (+4.3) | 81.3 (+24.3) | 0.02 / 0.25 |
| new (state, command) pair, 95% level -> cloned actor | 78.6 (-7.3) | 27.7 (+42.7) | 43% | 65.3 (-5.0) | 15.3 (-41.6) | 0.05 / 0.35 |
| new (state, command) pair, 95% level -> clip to recorded range | 88.5 (+2.6) | 42.1 (+57.1) | 57% | 75.1 (+4.8) | 81.6 (+24.6) | 0.02 / 0.27 |
| new (state, command) pair, 95% level -> nearest recorded command | 84.0 (-1.9) | 22.9 (+37.9) | 38% | 70.0 (-0.3) | 42.6 (-14.3) | 0.04 / 0.39 |
| new (state, command) pair, 99% level -> clip to recorded range | 87.2 (+1.2) | 6.2 (+21.1) | 21% | 72.1 (+1.8) | 79.5 (+22.6) | 0.01 / 0.08 |
| random, at the rate of the 95% (state, command) check -> cloned actor | 85.0 (-0.9) | 13.2 (+28.2) | 28% | 68.7 (-1.7) | 49.7 (-7.3) | 0.05 / 0.35 |
| random, at the rate of the 95% (state, command) check -> clip to recorded range | 86.4 (+0.5) | 21.9 (+36.9) | 37% | 71.8 (+1.4) | 62.9 (+6.0) | 0.00 / 0.21 |
| random, at the rate of the 99% (state, command) check -> clip to recorded range | 86.0 (+0.1) | -3.5 (+11.5) | 12% | 70.8 (+0.5) | 61.7 (+4.7) | 0.00 / 0.06 |
| command outside the commands recorded at nearby states, 95% level -> clip to recorded range | 89.9 (+3.9) | 24.8 (+39.7) | 40% | 75.9 (+5.6) | 80.3 (+23.3) | 0.01 / 0.24 |
| random, at the rate of the 95% nearby-commands check -> clip to recorded range | 86.2 (+0.2) | 16.8 (+31.7) | 32% | 71.1 (+0.8) | 61.2 (+4.2) | 0.00 / 0.18 |
| perfect detector: the actor proposes past 80% (reference) -> clip to recorded range | 93.1 (+7.2) | 91.2 (+106.1) | 107% | 90.2 (+19.9) | 85.2 (+28.2) | 0.04 / 0.70 |
| exact one-step check (reference) -> cloned actor | 90.2 (+4.3) | 50.1 (+65.1) | 65% | 81.5 (+11.2) | 70.7 (+13.7) | 0.16 / 0.50 |
| exact one-step check (reference) -> clip to recorded range | 93.0 (+7.0) | 68.9 (+83.9) | 84% | 89.1 (+18.8) | 85.7 (+28.8) | 0.03 / 0.50 |

Harm on these starts (full-log critic's actor minus this actor, both alone, seeds [0, 1, 2, 3, 4]): 99.5 points.


## Headline: stall/td3bc_gapped (seeds [0, 1, 2, 3, 4])

Points of the rule, and in brackets the change from the actor alone (paired by seed); strong current also gives the share of the harm recovered; last two columns: share of steps switched in recorded conditions / strong current.

| Rule | Recorded conditions | Strong current | Harm recovered | Big waves | Big kicks | switched (recorded / strong) |
|---|---|---|---|---|---|---|
| never switch (the actor alone) | 86.4 | -17.6 | - | 70.9 | 52.8 | 0.00 / 0.00 |
| every step -> cloned actor | 56.0 (-30.4) | 53.9 (+71.5) | 70% | 43.2 (-27.8) | 5.1 (-47.7) | 1.00 / 1.00 |
| every step -> clip to recorded range | 93.4 (+7.0) | 89.7 (+107.2) | 105% | 90.9 (+20.0) | 84.3 (+31.5) | 0.06 / 0.75 |
| every step -> clip to full recorded range (min to max) | 87.7 (+1.3) | -15.1 (+2.4) | 2% | 72.4 (+1.5) | 71.1 (+18.2) | 0.02 / 0.14 |
| every step -> nearest recorded command | 73.4 (-13.0) | 45.9 (+63.5) | 62% | 66.8 (-4.1) | 37.4 (-15.5) | 1.00 / 1.00 |
| new state, 95% level -> clip to recorded range | 89.0 (+2.6) | 37.9 (+55.5) | 54% | 76.0 (+5.0) | 80.7 (+27.8) | 0.02 / 0.26 |
| new (state, command) pair, 95% level -> cloned actor | 79.8 (-6.6) | 28.5 (+46.1) | 45% | 66.5 (-4.4) | 15.6 (-37.2) | 0.05 / 0.36 |
| new (state, command) pair, 95% level -> clip to recorded range | 89.5 (+3.1) | 42.1 (+59.7) | 58% | 76.5 (+5.5) | 81.0 (+28.1) | 0.02 / 0.28 |
| new (state, command) pair, 99% level -> clip to recorded range | 87.9 (+1.5) | 4.9 (+22.5) | 22% | 73.0 (+2.1) | 78.7 (+25.8) | 0.01 / 0.09 |
| random, at the rate of the 95% (state, command) check -> cloned actor | 85.7 (-0.7) | 13.9 (+31.5) | 31% | 69.8 (-1.1) | 46.9 (-5.9) | 0.05 / 0.36 |
| random, at the rate of the 95% (state, command) check -> clip to recorded range | 86.8 (+0.4) | 21.5 (+39.1) | 38% | 72.3 (+1.4) | 59.1 (+6.2) | 0.00 / 0.22 |
| random, at the rate of the 99% (state, command) check -> clip to recorded range | 86.5 (+0.1) | -4.2 (+13.4) | 13% | 71.4 (+0.5) | 58.2 (+5.4) | 0.00 / 0.07 |
| command outside the commands recorded at nearby states, 95% level -> clip to recorded range | 90.8 (+4.4) | 27.9 (+45.5) | 45% | 78.6 (+7.7) | 79.6 (+26.8) | 0.02 / 0.25 |
| random, at the rate of the 95% nearby-commands check -> clip to recorded range | 86.7 (+0.3) | 15.8 (+33.4) | 33% | 72.1 (+1.2) | 57.8 (+5.0) | 0.00 / 0.19 |
| perfect detector: the actor proposes past 80% (reference) -> clip to recorded range | 93.6 (+7.2) | 89.8 (+107.4) | 105% | 91.1 (+20.2) | 84.5 (+31.7) | 0.03 / 0.68 |
| exact one-step check (reference) -> cloned actor | 91.0 (+4.6) | 48.7 (+66.3) | 65% | 83.8 (+12.8) | 69.6 (+16.8) | 0.17 / 0.48 |
| exact one-step check (reference) -> clip to recorded range | 93.5 (+7.1) | 64.9 (+82.4) | 81% | 90.5 (+19.6) | 84.8 (+32.0) | 0.03 / 0.48 |

Harm on these starts (full-log critic's actor minus this actor, both alone, seeds [0, 1, 2, 3, 4]): 102.1 points.


## Headline: stall/full (seeds [0, 1, 2, 3, 4])

Points of the rule, and in brackets the change from the actor alone (paired by seed); strong current also gives the share of the harm recovered; last two columns: share of steps switched in recorded conditions / strong current.

| Rule | Recorded conditions | Strong current | Harm recovered | Big waves | Big kicks | switched (recorded / strong) |
|---|---|---|---|---|---|---|
| never switch (the actor alone) | 91.3 | 84.5 | - | 86.5 | 68.4 | 0.00 / 0.00 |
| every step -> cloned actor | 56.0 (-35.3) | 53.9 (-30.6) | - | 43.2 (-43.3) | 5.1 (-63.3) | 1.00 / 1.00 |
| every step -> clip to recorded range | 91.4 (+0.1) | 89.7 (+5.1) | - | 86.7 (+0.2) | 80.1 (+11.7) | 0.04 / 0.68 |
| every step -> clip to full recorded range (min to max) | 92.0 (+0.7) | 85.6 (+1.1) | - | 87.3 (+0.8) | 83.8 (+15.4) | 0.01 / 0.05 |
| every step -> nearest recorded command | 71.9 (-19.4) | 41.4 (-43.1) | - | 64.4 (-22.1) | 33.2 (-35.2) | 1.00 / 1.00 |
| new state, 95% level -> clip to recorded range | 91.5 (+0.2) | 88.1 (+3.6) | - | 86.8 (+0.3) | 80.1 (+11.8) | 0.02 / 0.14 |
| new (state, command) pair, 95% level -> cloned actor | 82.8 (-8.5) | 73.7 (-10.8) | - | 76.6 (-9.9) | 17.2 (-51.1) | 0.05 / 0.21 |
| new (state, command) pair, 95% level -> clip to recorded range | 91.5 (+0.2) | 88.1 (+3.6) | - | 86.8 (+0.3) | 80.1 (+11.8) | 0.02 / 0.14 |
| new (state, command) pair, 99% level -> clip to recorded range | 91.4 (+0.1) | 85.8 (+1.3) | - | 86.7 (+0.2) | 80.1 (+11.7) | 0.01 / 0.03 |
| random, at the rate of the 95% (state, command) check -> cloned actor | 90.6 (-0.7) | 81.4 (-3.1) | - | 85.2 (-1.2) | 59.8 (-8.6) | 0.05 / 0.21 |
| random, at the rate of the 95% (state, command) check -> clip to recorded range | 91.3 (+0.0) | 85.4 (+0.9) | - | 86.5 (+0.0) | 70.3 (+2.0) | 0.00 / 0.10 |
| random, at the rate of the 99% (state, command) check -> clip to recorded range | 91.3 (+0.0) | 84.7 (+0.2) | - | 86.5 (+0.0) | 70.1 (+1.8) | 0.00 / 0.02 |
| command outside the commands recorded at nearby states, 95% level -> clip to recorded range | 91.5 (+0.2) | 85.7 (+1.2) | - | 86.7 (+0.3) | 79.4 (+11.1) | 0.01 / 0.04 |
| random, at the rate of the 95% nearby-commands check -> clip to recorded range | 91.3 (+0.0) | 84.8 (+0.3) | - | 86.5 (+0.0) | 69.6 (+1.3) | 0.00 / 0.03 |
| perfect detector: the actor proposes past 80% (reference) -> clip to recorded range | 91.6 (+0.3) | 91.1 (+6.6) | - | 87.0 (+0.5) | 80.5 (+12.1) | 0.01 / 0.44 |
| exact one-step check (reference) -> cloned actor | 92.1 (+0.8) | 90.4 (+5.9) | - | 88.1 (+1.7) | 75.1 (+6.7) | 0.22 / 0.20 |
| exact one-step check (reference) -> clip to recorded range | 91.8 (+0.5) | 92.3 (+7.8) | - | 87.3 (+0.8) | 81.4 (+13.1) | 0.02 / 0.33 |

## Headline: no_stall/no_stall_gapped (seeds [0, 1, 2, 3, 4])

Points of the rule, and in brackets the change from the actor alone (paired by seed); strong current also gives the share of the harm recovered; last two columns: share of steps switched in recorded conditions / strong current.

| Rule | Recorded conditions | Strong current | Harm recovered | Big waves | Big kicks | switched (recorded / strong) |
|---|---|---|---|---|---|---|
| never switch (the actor alone) | 93.5 | 95.1 | - | 88.0 | 93.7 | 0.00 / 0.00 |
| every step -> cloned actor | 47.8 (-45.7) | -24.9 (-120.0) | - | 10.9 (-77.0) | 11.7 (-82.0) | 1.00 / 1.00 |
| every step -> clip to recorded range | 91.5 (-2.0) | 33.3 (-61.8) | - | 79.6 (-8.3) | 72.4 (-21.3) | 0.07 / 0.78 |
| new (state, command) pair, 95% level -> cloned actor | 81.7 (-11.8) | 36.0 (-59.1) | - | 68.7 (-19.3) | 24.4 (-69.3) | 0.05 / 0.30 |
| new (state, command) pair, 95% level -> clip to recorded range | 92.2 (-1.3) | 69.4 (-25.7) | - | 85.2 (-2.7) | 72.9 (-20.8) | 0.02 / 0.15 |
| new (state, command) pair, 99% level -> clip to recorded range | 92.7 (-0.7) | 88.6 (-6.5) | - | 86.5 (-1.5) | 73.5 (-20.2) | 0.01 / 0.03 |
| random, at the rate of the 95% (state, command) check -> cloned actor | 92.7 (-0.8) | 75.7 (-19.4) | - | 85.7 (-2.2) | 83.7 (-10.0) | 0.05 / 0.30 |
| random, at the rate of the 95% (state, command) check -> clip to recorded range | 93.4 (-0.0) | 90.6 (-4.5) | - | 87.8 (-0.2) | 91.1 (-2.6) | 0.00 / 0.12 |
| random, at the rate of the 99% (state, command) check -> clip to recorded range | 93.5 (-0.0) | 94.3 (-0.8) | - | 87.9 (-0.0) | 91.5 (-2.2) | 0.00 / 0.02 |
| command outside the commands recorded at nearby states, 95% level -> clip to recorded range | 92.6 (-0.9) | 76.7 (-18.4) | - | 86.1 (-1.8) | 74.6 (-19.1) | 0.01 / 0.19 |
| random, at the rate of the 95% nearby-commands check -> clip to recorded range | 93.5 (-0.0) | 89.7 (-5.4) | - | 87.9 (-0.1) | 92.0 (-1.7) | 0.00 / 0.14 |
| perfect detector: the actor proposes past 80% (reference) -> clip to recorded range | 91.6 (-1.9) | 33.5 (-61.6) | - | 79.9 (-8.0) | 72.7 (-21.0) | 0.04 / 0.72 |
| exact one-step check (reference) -> clip to recorded range | 93.5 (+0.0) | 95.5 (+0.4) | - | 88.2 (+0.2) | 93.9 (+0.2) | 0.01 / 0.09 |

## no_stall/no_stall_gapped  (seeds [0, 1, 2, 3, 4], starts per bank [200])

Points: mean ± standard error over seeds; in brackets the share of steps the trigger fired.

| Rule | Recorded conditions | Strong current | Big waves | Big kicks |
|---|---|---|---|---|
| never switch (the actor alone) | 93.5 ± 0.1 [0.00] | 95.1 ± 0.3 [0.00] | 88.0 ± 0.5 [0.00] | 93.7 ± 0.2 [0.00] |
| every step -> cloned actor | 47.8 ± 2.2 [1.00] | -24.9 ± 9.9 [1.00] | 10.9 ± 4.9 [1.00] | 11.7 ± 1.0 [1.00] |
| every step -> clip to recorded range | 91.5 ± 0.2 [1.00] | 33.3 ± 1.7 [1.00] | 79.6 ± 0.6 [1.00] | 72.4 ± 0.2 [1.00] |
| new (state, command) pair, 95% level -> cloned actor | 81.7 ± 0.3 [0.05] | 36.0 ± 5.4 [0.30] | 68.7 ± 1.0 [0.09] | 24.4 ± 1.1 [0.23] |
| new (state, command) pair, 95% level -> clip to recorded range | 92.2 ± 0.1 [0.03] | 69.4 ± 1.7 [0.16] | 85.2 ± 0.3 [0.05] | 72.9 ± 0.2 [0.16] |
| new (state, command) pair, 99% level -> clip to recorded range | 92.7 ± 0.1 [0.01] | 88.6 ± 0.5 [0.03] | 86.5 ± 0.4 [0.02] | 73.5 ± 0.2 [0.13] |
| random, at the rate of the 95% (state, command) check -> cloned actor | 92.7 ± 0.1 [0.05] | 75.7 ± 2.4 [0.30] | 85.7 ± 0.4 [0.09] | 83.7 ± 0.4 [0.23] |
| random, at the rate of the 95% (state, command) check -> clip to recorded range | 93.4 ± 0.1 [0.03] | 90.6 ± 0.6 [0.16] | 87.8 ± 0.4 [0.05] | 91.1 ± 0.3 [0.16] |
| random, at the rate of the 99% (state, command) check -> clip to recorded range | 93.5 ± 0.1 [0.01] | 94.3 ± 0.3 [0.03] | 87.9 ± 0.5 [0.02] | 91.5 ± 0.1 [0.13] |
| command outside the commands recorded at nearby states, 95% level -> clip to recorded range | 92.6 ± 0.1 [0.01] | 76.7 ± 1.8 [0.19] | 86.1 ± 0.4 [0.02] | 74.6 ± 0.3 [0.10] |
| random, at the rate of the 95% nearby-commands check -> clip to recorded range | 93.5 ± 0.1 [0.01] | 89.7 ± 0.8 [0.19] | 87.9 ± 0.4 [0.02] | 92.0 ± 0.2 [0.10] |
| perfect detector: the actor proposes past 80% (reference) -> clip to recorded range | 91.6 ± 0.2 [0.04] | 33.5 ± 1.7 [0.72] | 79.9 ± 0.6 [0.18] | 72.7 ± 0.2 [0.14] |
| exact one-step check (reference) -> clip to recorded range | 93.5 ± 0.1 [0.01] | 95.5 ± 0.4 [0.09] | 88.2 ± 0.4 [0.04] | 93.9 ± 0.2 [0.02] |

Paired differences by seed (points, mean ± standard error):

| Difference | Recorded conditions | Strong current | Big waves | Big kicks |
|---|---|---|---|---|
| sa95:clone minus random95:clone | -11.0 ± 0.4 | -39.8 ± 3.5 | -17.0 ± 0.8 | -59.3 ± 0.8 |
| sa95:clip99 minus random95:clip99 | -1.2 ± 0.0 | -21.2 ± 1.1 | -2.6 ± 0.1 | -18.2 ± 0.3 |
| sa99:clip99 minus random99:clip99 | -0.7 ± 0.0 | -5.8 ± 0.3 | -1.4 ± 0.0 | -18.1 ± 0.1 |
| local95:clip99 minus rlocal95:clip99 | -0.9 ± 0.0 | -13.0 ± 1.1 | -1.7 ± 0.1 | -17.5 ± 0.2 |
| always:clone minus none | -45.7 ± 2.3 | -120.0 ± 10.0 | -77.0 ± 4.8 | -82.0 ± 0.9 |
| always:clip99 minus none | -2.0 ± 0.1 | -61.8 ± 1.8 | -8.3 ± 0.5 | -21.3 ± 0.2 |
| sa95:clone minus none | -11.8 ± 0.4 | -59.1 ± 5.3 | -19.3 ± 0.8 | -69.3 ± 1.0 |
| sa95:clip99 minus none | -1.3 ± 0.0 | -25.7 ± 1.5 | -2.7 ± 0.1 | -20.8 ± 0.2 |
| sa99:clip99 minus none | -0.7 ± 0.0 | -6.5 ± 0.4 | -1.5 ± 0.0 | -20.2 ± 0.2 |
| random95:clone minus none | -0.8 ± 0.1 | -19.4 ± 2.2 | -2.2 ± 0.1 | -10.0 ± 0.2 |
| random95:clip99 minus none | -0.0 ± 0.0 | -4.5 ± 0.5 | -0.2 ± 0.0 | -2.6 ± 0.1 |
| random99:clip99 minus none | -0.0 ± 0.0 | -0.8 ± 0.1 | -0.0 ± 0.0 | -2.2 ± 0.1 |
| local95:clip99 minus none | -0.9 ± 0.0 | -18.4 ± 1.9 | -1.8 ± 0.1 | -19.1 ± 0.1 |
| rlocal95:clip99 minus none | -0.0 ± 0.0 | -5.4 ± 0.8 | -0.1 ± 0.0 | -1.7 ± 0.1 |
| unrec:clip99 minus none | -1.9 ± 0.1 | -61.6 ± 1.8 | -8.0 ± 0.5 | -21.0 ± 0.2 |
| oracle:clip99 minus none | 0.0 ± 0.0 | 0.4 ± 0.1 | 0.2 ± 0.1 | 0.2 ± 0.0 |

Strong current: share of steps the actor proposes past 80%, share applied past 80%, and the share of past-80% proposals the trigger caught (recall) / share of firings on past-80% proposals (precision):

| Rule | proposed past 80% | applied past 80% | recall | precision |
|---|---|---|---|---|
| never switch (the actor alone) | 0.63 | 0.63 | 0.00 | 0.00 |
| every step -> cloned actor | 0.84 | 0.14 | 1.00 | 0.84 |
| every step -> clip to recorded range | 0.73 | 0.00 | 1.00 | 0.73 |
| new (state, command) pair, 95% level -> cloned actor | 0.64 | 0.44 | 0.44 | 0.94 |
| new (state, command) pair, 95% level -> clip to recorded range | 0.63 | 0.48 | 0.23 | 0.90 |
| new (state, command) pair, 99% level -> clip to recorded range | 0.63 | 0.60 | 0.05 | 0.95 |
| random, at the rate of the 95% (state, command) check -> cloned actor | 0.69 | 0.50 | 0.30 | 0.69 |
| random, at the rate of the 95% (state, command) check -> clip to recorded range | 0.64 | 0.54 | 0.16 | 0.64 |
| random, at the rate of the 99% (state, command) check -> clip to recorded range | 0.63 | 0.61 | 0.03 | 0.64 |
| command outside the commands recorded at nearby states, 95% level -> clip to recorded range | 0.63 | 0.45 | 0.29 | 0.99 |
| random, at the rate of the 95% nearby-commands check -> clip to recorded range | 0.64 | 0.52 | 0.18 | 0.64 |
| perfect detector: the actor proposes past 80% (reference) -> clip to recorded range | 0.72 | 0.00 | 1.00 | 1.00 |
| exact one-step check (reference) -> clip to recorded range | 0.64 | 0.57 | 0.11 | 0.79 |

Check against the gate (whole 1,000-start bank, no switching and capped at 80%): largest absolute difference recorded 0.0e+00, strong_current 0.0e+00, big_waves 0.0e+00, big_kicks 0.0e+00; whole-bank points of the actor alone: recorded 93.5 ± 0.1, strong_current 94.9 ± 0.3, big_waves 88.1 ± 0.4, big_kicks 91.2 ± 0.2

Thresholds (mean over seeds): state 3.72 ± 0.02 / 5.21 ± 0.05; (state, command) 4.26 ± 0.03 / 5.72 ± 0.04; nearby commands 0.278 ± 0.001 / 0.407 ± 0.002. Rate above threshold on held-out evaluation-split rows: state 0.050 / 0.010; (state, command) 0.051 / 0.010; nearby commands 0.051 / 0.010. Clip range (99% of recorded commands): low [-0.74, -0.64, -0.7, -0.67], high [0.52, 0.6, 0.56, 0.57]; full-range low [-1.0, -0.75, -0.78, -0.76]; training rows past 80%: [37, 34, 52, 28, 58]; held-out rows [16849, 16820, 16903, 16911, 16629].

## stall/full  (seeds [0, 1, 2, 3, 4], starts per bank [200])

Points: mean ± standard error over seeds; in brackets the share of steps the trigger fired.

| Rule | Recorded conditions | Strong current | Big waves | Big kicks |
|---|---|---|---|---|
| never switch (the actor alone) | 91.3 ± 0.5 [0.00] | 84.5 ± 2.0 [0.00] | 86.5 ± 0.7 [0.00] | 68.4 ± 0.6 [0.00] |
| every step -> cloned actor | 56.0 ± 2.1 [1.00] | 53.9 ± 2.4 [1.00] | 43.2 ± 3.4 [1.00] | 5.1 ± 2.9 [1.00] |
| every step -> clip to recorded range | 91.4 ± 0.5 [1.00] | 89.7 ± 0.8 [1.00] | 86.7 ± 0.7 [1.00] | 80.1 ± 1.0 [1.00] |
| every step -> nearest recorded command | 71.9 ± 0.6 [1.00] | 41.4 ± 1.8 [1.00] | 64.4 ± 0.5 [1.00] | 33.2 ± 0.5 [1.00] |
| every step -> clip to full recorded range (min to max) | 92.0 ± 0.5 [1.00] | 85.6 ± 2.1 [1.00] | 87.3 ± 0.7 [1.00] | 83.8 ± 0.6 [1.00] |
| new state, 95% level -> cloned actor | 83.2 ± 0.7 [0.05] | 73.8 ± 2.2 [0.20] | 76.9 ± 0.8 [0.08] | 17.5 ± 2.6 [0.23] |
| new state, 95% level -> clip to recorded range | 91.5 ± 0.5 [0.03] | 88.1 ± 1.5 [0.16] | 86.8 ± 0.7 [0.05] | 80.1 ± 1.0 [0.16] |
| new (state, command) pair, 95% level -> cloned actor | 82.8 ± 0.7 [0.05] | 73.7 ± 2.2 [0.21] | 76.6 ± 0.8 [0.08] | 17.2 ± 2.5 [0.23] |
| new (state, command) pair, 95% level -> clip to recorded range | 91.5 ± 0.5 [0.03] | 88.1 ± 1.5 [0.15] | 86.8 ± 0.7 [0.05] | 80.1 ± 1.0 [0.16] |
| new (state, command) pair, 99% level -> cloned actor | 88.2 ± 0.5 [0.01] | 80.2 ± 1.9 [0.06] | 82.5 ± 0.7 [0.02] | 21.4 ± 2.4 [0.18] |
| new (state, command) pair, 99% level -> clip to recorded range | 91.4 ± 0.5 [0.01] | 85.8 ± 2.0 [0.03] | 86.7 ± 0.7 [0.02] | 80.1 ± 1.0 [0.13] |
| random, at the rate of the 95% (state, command) check -> cloned actor | 90.6 ± 0.5 [0.05] | 81.4 ± 1.5 [0.21] | 85.2 ± 0.6 [0.08] | 59.8 ± 0.8 [0.23] |
| random, at the rate of the 95% (state, command) check -> clip to recorded range | 91.3 ± 0.5 [0.03] | 85.4 ± 1.7 [0.16] | 86.5 ± 0.7 [0.05] | 70.3 ± 0.6 [0.16] |
| random, at the rate of the 99% (state, command) check -> cloned actor | 91.1 ± 0.5 [0.01] | 83.7 ± 1.9 [0.06] | 86.1 ± 0.7 [0.02] | 61.6 ± 0.6 [0.18] |
| random, at the rate of the 99% (state, command) check -> clip to recorded range | 91.3 ± 0.5 [0.01] | 84.7 ± 1.9 [0.03] | 86.5 ± 0.7 [0.02] | 70.1 ± 0.7 [0.13] |
| command outside the commands recorded at nearby states, 95% level -> clip to recorded range | 91.5 ± 0.5 [0.01] | 85.7 ± 1.8 [0.04] | 86.7 ± 0.7 [0.01] | 79.4 ± 1.0 [0.09] |
| random, at the rate of the 95% nearby-commands check -> clip to recorded range | 91.3 ± 0.5 [0.01] | 84.8 ± 1.9 [0.04] | 86.5 ± 0.7 [0.01] | 69.6 ± 0.6 [0.09] |
| perfect detector: the actor proposes past 80% (reference) -> cloned actor | 88.6 ± 0.5 [0.02] | 74.7 ± 2.1 [0.42] | 82.6 ± 0.8 [0.04] | 32.1 ± 1.7 [0.14] |
| perfect detector: the actor proposes past 80% (reference) -> clip to recorded range | 91.6 ± 0.5 [0.01] | 91.1 ± 0.8 [0.44] | 87.0 ± 0.7 [0.04] | 80.5 ± 1.0 [0.11] |
| exact one-step check (reference) -> cloned actor | 92.1 ± 0.4 [0.22] | 90.4 ± 1.0 [0.20] | 88.1 ± 0.5 [0.21] | 75.1 ± 0.4 [0.22] |
| exact one-step check (reference) -> clip to recorded range | 91.8 ± 0.5 [0.02] | 92.3 ± 0.7 [0.33] | 87.3 ± 0.7 [0.04] | 81.4 ± 1.0 [0.09] |

Paired differences by seed (points, mean ± standard error):

| Difference | Recorded conditions | Strong current | Big waves | Big kicks |
|---|---|---|---|---|
| sa95:clone minus random95:clone | -7.8 ± 0.2 | -7.7 ± 1.7 | -8.6 ± 0.3 | -42.5 ± 2.0 |
| sa95:clip99 minus random95:clip99 | 0.2 ± 0.0 | 2.6 ± 1.0 | 0.3 ± 0.0 | 9.8 ± 0.5 |
| sa99:clone minus random99:clone | -3.0 ± 0.1 | -3.5 ± 0.4 | -3.5 ± 0.1 | -40.2 ± 2.0 |
| sa99:clip99 minus random99:clip99 | 0.1 ± 0.0 | 1.1 ± 0.2 | 0.2 ± 0.0 | 9.9 ± 0.5 |
| local95:clip99 minus rlocal95:clip99 | 0.2 ± 0.0 | 0.9 ± 0.2 | 0.3 ± 0.0 | 9.8 ± 0.5 |
| always:clone minus none | -35.3 ± 1.7 | -30.6 ± 3.5 | -43.3 ± 2.9 | -63.3 ± 2.8 |
| always:clip99 minus none | 0.1 ± 0.0 | 5.1 ± 2.3 | 0.2 ± 0.0 | 11.7 ± 0.7 |
| always:project minus none | -19.4 ± 0.4 | -43.1 ± 3.3 | -22.1 ± 0.5 | -35.2 ± 0.4 |
| always:clipmm minus none | 0.7 ± 0.0 | 1.1 ± 0.1 | 0.8 ± 0.0 | 15.4 ± 0.1 |
| state95:clone minus none | -8.1 ± 0.2 | -10.7 ± 2.4 | -9.6 ± 0.2 | -50.9 ± 2.4 |
| state95:clip99 minus none | 0.2 ± 0.0 | 3.6 ± 1.4 | 0.3 ± 0.0 | 11.8 ± 0.7 |
| sa95:clone minus none | -8.5 ± 0.2 | -10.8 ± 2.4 | -9.9 ± 0.2 | -51.1 ± 2.4 |
| sa95:clip99 minus none | 0.2 ± 0.0 | 3.6 ± 1.4 | 0.3 ± 0.0 | 11.8 ± 0.7 |
| sa99:clone minus none | -3.1 ± 0.1 | -4.3 ± 0.6 | -3.9 ± 0.2 | -47.0 ± 2.3 |
| sa99:clip99 minus none | 0.1 ± 0.0 | 1.3 ± 0.2 | 0.2 ± 0.0 | 11.7 ± 0.7 |
| random95:clone minus none | -0.7 ± 0.0 | -3.1 ± 0.8 | -1.2 ± 0.1 | -8.6 ± 0.5 |
| random95:clip99 minus none | 0.0 ± 0.0 | 0.9 ± 0.4 | 0.0 ± 0.0 | 2.0 ± 0.2 |
| random99:clone minus none | -0.2 ± 0.0 | -0.8 ± 0.3 | -0.4 ± 0.0 | -6.8 ± 0.4 |
| random99:clip99 minus none | 0.0 ± 0.0 | 0.2 ± 0.1 | 0.0 ± 0.0 | 1.8 ± 0.1 |
| local95:clip99 minus none | 0.2 ± 0.0 | 1.2 ± 0.3 | 0.3 ± 0.0 | 11.1 ± 0.6 |
| rlocal95:clip99 minus none | 0.0 ± 0.0 | 0.3 ± 0.1 | 0.0 ± 0.0 | 1.3 ± 0.1 |
| unrec:clone minus none | -2.7 ± 0.1 | -9.8 ± 2.5 | -3.9 ± 0.4 | -36.2 ± 1.6 |
| unrec:clip99 minus none | 0.3 ± 0.0 | 6.6 ± 2.1 | 0.5 ± 0.0 | 12.1 ± 0.7 |
| oracle:clone minus none | 0.8 ± 0.1 | 5.9 ± 1.4 | 1.7 ± 0.3 | 6.7 ± 0.2 |
| oracle:clip99 minus none | 0.5 ± 0.0 | 7.8 ± 1.9 | 0.8 ± 0.0 | 13.1 ± 0.6 |

Strong current: share of steps the actor proposes past 80%, share applied past 80%, and the share of past-80% proposals the trigger caught (recall) / share of firings on past-80% proposals (precision):

| Rule | proposed past 80% | applied past 80% | recall | precision |
|---|---|---|---|---|
| never switch (the actor alone) | 0.42 | 0.42 | 0.00 | 0.00 |
| every step -> cloned actor | 0.49 | 0.10 | 1.00 | 0.49 |
| every step -> clip to recorded range | 0.45 | 0.00 | 1.00 | 0.45 |
| every step -> nearest recorded command | 0.50 | 0.00 | 1.00 | 0.50 |
| every step -> clip to full recorded range (min to max) | 0.42 | 0.40 | 1.00 | 0.42 |
| new state, 95% level -> cloned actor | 0.44 | 0.36 | 0.36 | 0.75 |
| new state, 95% level -> clip to recorded range | 0.43 | 0.32 | 0.26 | 0.68 |
| new (state, command) pair, 95% level -> cloned actor | 0.44 | 0.35 | 0.37 | 0.75 |
| new (state, command) pair, 95% level -> clip to recorded range | 0.43 | 0.32 | 0.26 | 0.69 |
| new (state, command) pair, 99% level -> cloned actor | 0.43 | 0.41 | 0.12 | 0.82 |
| new (state, command) pair, 99% level -> clip to recorded range | 0.43 | 0.40 | 0.06 | 0.76 |
| random, at the rate of the 95% (state, command) check -> cloned actor | 0.44 | 0.36 | 0.21 | 0.44 |
| random, at the rate of the 95% (state, command) check -> clip to recorded range | 0.43 | 0.36 | 0.16 | 0.42 |
| random, at the rate of the 99% (state, command) check -> cloned actor | 0.43 | 0.41 | 0.06 | 0.42 |
| random, at the rate of the 99% (state, command) check -> clip to recorded range | 0.42 | 0.41 | 0.03 | 0.42 |
| command outside the commands recorded at nearby states, 95% level -> clip to recorded range | 0.42 | 0.39 | 0.08 | 0.80 |
| random, at the rate of the 95% nearby-commands check -> clip to recorded range | 0.42 | 0.41 | 0.04 | 0.43 |
| perfect detector: the actor proposes past 80% (reference) -> cloned actor | 0.42 | 0.07 | 1.00 | 1.00 |
| perfect detector: the actor proposes past 80% (reference) -> clip to recorded range | 0.44 | 0.00 | 1.00 | 1.00 |
| exact one-step check (reference) -> cloned actor | 0.42 | 0.32 | 0.28 | 0.58 |
| exact one-step check (reference) -> clip to recorded range | 0.44 | 0.15 | 0.63 | 0.86 |

Check against the gate (whole 1,000-start bank, no switching and capped at 80%): largest absolute difference recorded 0.0e+00, strong_current 0.0e+00, big_waves 0.0e+00, big_kicks 0.0e+00; whole-bank points of the actor alone: recorded 91.3 ± 0.4, strong_current 84.6 ± 2.0, big_waves 86.6 ± 0.7, big_kicks 64.3 ± 0.6

Thresholds (mean over seeds): state 3.73 ± 0.03 / 5.21 ± 0.04; (state, command) 4.26 ± 0.03 / 5.71 ± 0.04; nearby commands 0.278 ± 0.001 / 0.411 ± 0.004. Rate above threshold on held-out evaluation-split rows: state 0.049 / 0.009; (state, command) 0.049 / 0.009; nearby commands 0.050 / 0.011. Clip range (99% of recorded commands): low [-0.75, -0.63, -0.7, -0.67], high [0.51, 0.61, 0.55, 0.57]; full-range low [-1.0, -0.75, -0.78, -0.76]; training rows past 80%: [34, 45, 59, 42, 45]; held-out rows [16837, 16821, 16905, 16909, 16639].

## stall/gapped  (seeds [0, 1, 2, 3, 4], starts per bank [200])

Points: mean ± standard error over seeds; in brackets the share of steps the trigger fired.

| Rule | Recorded conditions | Strong current | Big waves | Big kicks |
|---|---|---|---|---|
| never switch (the actor alone) | 86.0 ± 0.7 [0.00] | -15.0 ± 1.8 [0.00] | 70.3 ± 0.6 [0.00] | 57.0 ± 0.5 [0.00] |
| every step -> cloned actor | 56.0 ± 2.1 [1.00] | 53.9 ± 2.4 [1.00] | 43.2 ± 3.4 [1.00] | 5.1 ± 2.9 [1.00] |
| every step -> clip to recorded range | 93.0 ± 0.3 [1.00] | 91.0 ± 0.5 [1.00] | 90.0 ± 0.3 [1.00] | 84.9 ± 0.2 [1.00] |
| every step -> nearest recorded command | 72.9 ± 0.5 [1.00] | 48.1 ± 1.7 [1.00] | 66.0 ± 0.3 [1.00] | 37.9 ± 0.6 [1.00] |
| every step -> clip to full recorded range (min to max) | 86.6 ± 0.7 [1.00] | -13.9 ± 1.7 [1.00] | 71.2 ± 0.6 [1.00] | 72.0 ± 0.5 [1.00] |
| new state, 95% level -> cloned actor | 78.8 ± 0.7 [0.05] | 24.1 ± 1.5 [0.32] | 65.3 ± 0.8 [0.08] | 15.5 ± 2.5 [0.23] |
| new state, 95% level -> clip to recorded range | 88.3 ± 0.6 [0.03] | 37.9 ± 2.4 [0.25] | 74.6 ± 0.6 [0.05] | 81.3 ± 0.3 [0.15] |
| new state, 95% level -> nearest recorded command | 84.1 ± 0.6 [0.04] | 20.3 ± 2.0 [0.34] | 69.8 ± 0.6 [0.06] | 42.6 ± 0.6 [0.19] |
| new state, 99% level -> cloned actor | 83.8 ± 0.7 [0.01] | -4.3 ± 1.7 [0.11] | 67.8 ± 0.8 [0.02] | 18.4 ± 2.5 [0.17] |
| new state, 99% level -> clip to recorded range | 86.9 ± 0.6 [0.01] | 3.1 ± 2.0 [0.07] | 71.8 ± 0.6 [0.01] | 79.0 ± 0.3 [0.12] |
| new (state, command) pair, 95% level -> cloned actor | 78.6 ± 0.7 [0.05] | 27.7 ± 1.2 [0.35] | 65.3 ± 0.8 [0.08] | 15.3 ± 2.5 [0.23] |
| new (state, command) pair, 95% level -> clip to recorded range | 88.5 ± 0.6 [0.03] | 42.1 ± 2.4 [0.28] | 75.1 ± 0.6 [0.05] | 81.6 ± 0.3 [0.16] |
| new (state, command) pair, 95% level -> nearest recorded command | 84.0 ± 0.5 [0.04] | 22.9 ± 2.1 [0.39] | 70.0 ± 0.6 [0.07] | 42.6 ± 0.6 [0.20] |
| new (state, command) pair, 99% level -> cloned actor | 83.2 ± 0.6 [0.01] | -1.9 ± 2.0 [0.13] | 67.5 ± 0.7 [0.02] | 18.0 ± 2.5 [0.18] |
| new (state, command) pair, 99% level -> clip to recorded range | 87.2 ± 0.6 [0.01] | 6.2 ± 2.3 [0.08] | 72.1 ± 0.6 [0.02] | 79.5 ± 0.3 [0.12] |
| random, at the rate of the 95% (state, command) check -> cloned actor | 85.0 ± 0.6 [0.05] | 13.2 ± 1.8 [0.35] | 68.7 ± 0.6 [0.08] | 49.7 ± 1.0 [0.23] |
| random, at the rate of the 95% (state, command) check -> clip to recorded range | 86.4 ± 0.7 [0.03] | 21.9 ± 1.9 [0.28] | 71.8 ± 0.5 [0.05] | 62.9 ± 0.3 [0.16] |
| random, at the rate of the 95% (state, command) check -> nearest recorded command | 85.5 ± 0.7 [0.04] | 13.2 ± 1.5 [0.39] | 69.6 ± 0.6 [0.07] | 54.9 ± 0.6 [0.20] |
| random, at the rate of the 99% (state, command) check -> cloned actor | 85.7 ± 0.7 [0.01] | -4.7 ± 1.9 [0.13] | 69.7 ± 0.6 [0.02] | 51.4 ± 0.5 [0.18] |
| random, at the rate of the 99% (state, command) check -> clip to recorded range | 86.0 ± 0.7 [0.01] | -3.5 ± 1.8 [0.08] | 70.8 ± 0.6 [0.02] | 61.7 ± 0.5 [0.12] |
| command outside the commands recorded at nearby states, 95% level -> cloned actor | 85.5 ± 0.6 [0.02] | 12.5 ± 4.6 [0.26] | 70.8 ± 0.9 [0.04] | 28.1 ± 2.1 [0.14] |
| command outside the commands recorded at nearby states, 95% level -> clip to recorded range | 89.9 ± 0.6 [0.01] | 24.8 ± 4.8 [0.24] | 75.9 ± 0.9 [0.03] | 80.3 ± 0.2 [0.11] |
| command outside the commands recorded at nearby states, 99% level -> cloned actor | 85.6 ± 0.7 [0.00] | -13.4 ± 2.0 [0.01] | 70.0 ± 0.6 [0.00] | 41.7 ± 1.6 [0.06] |
| command outside the commands recorded at nearby states, 99% level -> clip to recorded range | 86.3 ± 0.6 [0.00] | -12.0 ± 2.0 [0.01] | 70.6 ± 0.6 [0.00] | 68.5 ± 0.8 [0.05] |
| random, at the rate of the 95% nearby-commands check -> cloned actor | 85.6 ± 0.7 [0.02] | 7.4 ± 2.1 [0.27] | 69.6 ± 0.5 [0.04] | 53.1 ± 0.6 [0.14] |
| random, at the rate of the 95% nearby-commands check -> clip to recorded range | 86.2 ± 0.7 [0.01] | 16.8 ± 2.7 [0.24] | 71.1 ± 0.6 [0.03] | 61.2 ± 0.4 [0.11] |
| perfect detector: the actor proposes past 80% (reference) -> cloned actor | 85.0 ± 0.4 [0.06] | 58.8 ± 2.3 [0.79] | 74.4 ± 1.4 [0.18] | 26.5 ± 2.0 [0.19] |
| perfect detector: the actor proposes past 80% (reference) -> clip to recorded range | 93.1 ± 0.3 [0.04] | 91.2 ± 0.5 [0.70] | 90.2 ± 0.3 [0.15] | 85.2 ± 0.2 [0.14] |
| exact one-step check (reference) -> cloned actor | 90.2 ± 0.5 [0.16] | 50.1 ± 1.5 [0.50] | 81.5 ± 0.7 [0.22] | 70.7 ± 0.4 [0.18] |
| exact one-step check (reference) -> clip to recorded range | 93.0 ± 0.4 [0.03] | 68.9 ± 0.5 [0.50] | 89.1 ± 1.0 [0.13] | 85.7 ± 0.2 [0.12] |

Paired differences by seed (points, mean ± standard error):

| Difference | Recorded conditions | Strong current | Big waves | Big kicks |
|---|---|---|---|---|
| sa95:clone minus random95:clone | -6.4 ± 0.2 | 14.5 ± 1.6 | -3.3 ± 0.3 | -34.3 ± 2.0 |
| sa95:clip99 minus random95:clip99 | 2.1 ± 0.1 | 20.2 ± 1.3 | 3.3 ± 0.2 | 18.7 ± 0.2 |
| sa95:project minus random95:project | -1.5 ± 0.2 | 9.7 ± 1.5 | 0.4 ± 0.3 | -12.3 ± 0.1 |
| sa99:clone minus random99:clone | -2.5 ± 0.1 | 2.7 ± 1.6 | -2.3 ± 0.2 | -33.4 ± 2.3 |
| sa99:clip99 minus random99:clip99 | 1.1 ± 0.1 | 9.7 ± 1.2 | 1.3 ± 0.1 | 17.8 ± 0.2 |
| local95:clone minus rlocal95:clone | -0.1 ± 0.2 | 5.2 ± 2.7 | 1.2 ± 0.6 | -25.0 ± 2.0 |
| local95:clip99 minus rlocal95:clip99 | 3.7 ± 0.4 | 8.0 ± 2.5 | 4.8 ± 0.6 | 19.1 ± 0.5 |
| always:clone minus none | -29.9 ± 1.9 | 68.9 ± 3.8 | -27.2 ± 3.5 | -51.8 ± 3.1 |
| always:clip99 minus none | 7.0 ± 0.5 | 106.0 ± 2.0 | 19.7 ± 0.6 | 28.0 ± 0.4 |
| always:project minus none | -13.1 ± 0.7 | 63.0 ± 3.2 | -4.3 ± 0.6 | -19.1 ± 0.4 |
| always:clipmm minus none | 0.6 ± 0.0 | 1.1 ± 0.1 | 0.9 ± 0.0 | 15.1 ± 0.2 |
| state95:clone minus none | -7.2 ± 0.2 | 39.0 ± 2.6 | -5.0 ± 0.4 | -41.4 ± 2.7 |
| state95:clip99 minus none | 2.3 ± 0.1 | 52.9 ± 2.8 | 4.3 ± 0.1 | 24.3 ± 0.3 |
| state95:project minus none | -1.9 ± 0.2 | 35.3 ± 2.9 | -0.5 ± 0.1 | -14.4 ± 0.2 |
| state99:clone minus none | -2.2 ± 0.1 | 10.6 ± 1.4 | -2.5 ± 0.2 | -38.5 ± 2.7 |
| state99:clip99 minus none | 1.0 ± 0.0 | 18.0 ± 2.0 | 1.5 ± 0.1 | 22.1 ± 0.3 |
| sa95:clone minus none | -7.3 ± 0.2 | 42.7 ± 2.0 | -5.0 ± 0.4 | -41.6 ± 2.8 |
| sa95:clip99 minus none | 2.6 ± 0.1 | 57.1 ± 2.5 | 4.8 ± 0.1 | 24.6 ± 0.3 |
| sa95:project minus none | -1.9 ± 0.2 | 37.9 ± 3.0 | -0.3 ± 0.1 | -14.3 ± 0.3 |
| sa99:clone minus none | -2.7 ± 0.1 | 13.0 ± 1.5 | -2.8 ± 0.2 | -38.9 ± 2.7 |
| sa99:clip99 minus none | 1.2 ± 0.1 | 21.1 ± 2.0 | 1.8 ± 0.1 | 22.6 ± 0.3 |
| random95:clone minus none | -0.9 ± 0.1 | 28.2 ± 0.9 | -1.7 ± 0.4 | -7.3 ± 1.0 |
| random95:clip99 minus none | 0.5 ± 0.1 | 36.9 ± 1.6 | 1.4 ± 0.2 | 6.0 ± 0.3 |
| random95:project minus none | -0.4 ± 0.1 | 28.1 ± 1.6 | -0.7 ± 0.2 | -2.0 ± 0.2 |
| random99:clone minus none | -0.3 ± 0.1 | 10.3 ± 0.5 | -0.6 ± 0.2 | -5.5 ± 0.5 |
| random99:clip99 minus none | 0.1 ± 0.0 | 11.5 ± 0.9 | 0.5 ± 0.0 | 4.7 ± 0.1 |
| local95:clone minus none | -0.4 ± 0.2 | 27.5 ± 4.7 | 0.5 ± 0.7 | -28.8 ± 2.2 |
| local95:clip99 minus none | 3.9 ± 0.4 | 39.7 ± 4.8 | 5.6 ± 0.7 | 23.3 ± 0.6 |
| local99:clone minus none | -0.3 ± 0.1 | 1.6 ± 0.5 | -0.3 ± 0.1 | -15.3 ± 1.7 |
| local99:clip99 minus none | 0.4 ± 0.1 | 2.9 ± 0.8 | 0.3 ± 0.0 | 11.6 ± 0.3 |
| rlocal95:clone minus none | -0.3 ± 0.1 | 22.3 ± 2.5 | -0.7 ± 0.3 | -3.9 ± 0.3 |
| rlocal95:clip99 minus none | 0.2 ± 0.0 | 31.7 ± 3.2 | 0.8 ± 0.1 | 4.2 ± 0.2 |
| unrec:clone minus none | -1.0 ± 0.6 | 73.8 ± 3.6 | 4.1 ± 1.6 | -30.4 ± 2.1 |
| unrec:clip99 minus none | 7.2 ± 0.5 | 106.1 ± 2.0 | 19.9 ± 0.6 | 28.2 ± 0.4 |
| oracle:clone minus none | 4.3 ± 0.3 | 65.1 ± 2.9 | 11.2 ± 0.8 | 13.7 ± 0.3 |
| oracle:clip99 minus none | 7.0 ± 0.5 | 83.9 ± 1.9 | 18.8 ± 1.3 | 28.8 ± 0.4 |

Strong current: share of steps the actor proposes past 80%, share applied past 80%, and the share of past-80% proposals the trigger caught (recall) / share of firings on past-80% proposals (precision):

| Rule | proposed past 80% | applied past 80% | recall | precision |
|---|---|---|---|---|
| never switch (the actor alone) | 0.73 | 0.73 | 0.00 | 0.00 |
| every step -> cloned actor | 0.85 | 0.10 | 1.00 | 0.85 |
| every step -> clip to recorded range | 0.71 | 0.00 | 1.00 | 0.71 |
| every step -> nearest recorded command | 0.83 | 0.01 | 1.00 | 0.83 |
| every step -> clip to full recorded range (min to max) | 0.73 | 0.73 | 1.00 | 0.73 |
| new state, 95% level -> cloned actor | 0.77 | 0.56 | 0.40 | 0.96 |
| new state, 95% level -> clip to recorded range | 0.76 | 0.51 | 0.32 | 0.96 |
| new state, 95% level -> nearest recorded command | 0.77 | 0.43 | 0.44 | 0.97 |
| new state, 99% level -> cloned actor | 0.75 | 0.69 | 0.14 | 0.97 |
| new state, 99% level -> clip to recorded range | 0.75 | 0.68 | 0.09 | 0.97 |
| new (state, command) pair, 95% level -> cloned actor | 0.77 | 0.53 | 0.44 | 0.96 |
| new (state, command) pair, 95% level -> clip to recorded range | 0.75 | 0.49 | 0.35 | 0.96 |
| new (state, command) pair, 95% level -> nearest recorded command | 0.77 | 0.39 | 0.49 | 0.98 |
| new (state, command) pair, 99% level -> cloned actor | 0.75 | 0.68 | 0.17 | 0.97 |
| new (state, command) pair, 99% level -> clip to recorded range | 0.75 | 0.67 | 0.11 | 0.97 |
| random, at the rate of the 95% (state, command) check -> cloned actor | 0.80 | 0.58 | 0.35 | 0.80 |
| random, at the rate of the 95% (state, command) check -> clip to recorded range | 0.73 | 0.53 | 0.28 | 0.73 |
| random, at the rate of the 95% (state, command) check -> nearest recorded command | 0.79 | 0.49 | 0.39 | 0.78 |
| random, at the rate of the 99% (state, command) check -> cloned actor | 0.76 | 0.69 | 0.13 | 0.76 |
| random, at the rate of the 99% (state, command) check -> clip to recorded range | 0.73 | 0.67 | 0.08 | 0.73 |
| command outside the commands recorded at nearby states, 95% level -> cloned actor | 0.73 | 0.52 | 0.36 | 1.00 |
| command outside the commands recorded at nearby states, 95% level -> clip to recorded range | 0.70 | 0.47 | 0.33 | 0.99 |
| command outside the commands recorded at nearby states, 99% level -> cloned actor | 0.74 | 0.73 | 0.01 | 0.99 |
| command outside the commands recorded at nearby states, 99% level -> clip to recorded range | 0.74 | 0.72 | 0.01 | 0.99 |
| random, at the rate of the 95% nearby-commands check -> cloned actor | 0.78 | 0.62 | 0.27 | 0.79 |
| random, at the rate of the 95% nearby-commands check -> clip to recorded range | 0.73 | 0.56 | 0.24 | 0.73 |
| perfect detector: the actor proposes past 80% (reference) -> cloned actor | 0.79 | 0.09 | 1.00 | 1.00 |
| perfect detector: the actor proposes past 80% (reference) -> clip to recorded range | 0.70 | 0.00 | 1.00 | 1.00 |
| exact one-step check (reference) -> cloned actor | 0.75 | 0.35 | 0.62 | 0.94 |
| exact one-step check (reference) -> clip to recorded range | 0.72 | 0.23 | 0.67 | 0.97 |

Check against the gate (whole 1,000-start bank, no switching and capped at 80%): largest absolute difference recorded 0.0e+00, strong_current 0.0e+00, big_waves 0.0e+00, big_kicks 0.0e+00; whole-bank points of the actor alone: recorded 85.5 ± 0.6, strong_current -14.3 ± 1.9, big_waves 70.0 ± 0.6, big_kicks 52.5 ± 0.5

Thresholds (mean over seeds): state 3.73 ± 0.03 / 5.21 ± 0.04; (state, command) 4.26 ± 0.03 / 5.71 ± 0.04; nearby commands 0.278 ± 0.001 / 0.411 ± 0.004. Rate above threshold on held-out evaluation-split rows: state 0.049 / 0.009; (state, command) 0.049 / 0.009; nearby commands 0.050 / 0.011. Clip range (99% of recorded commands): low [-0.75, -0.63, -0.7, -0.67], high [0.51, 0.61, 0.55, 0.57]; full-range low [-1.0, -0.75, -0.78, -0.76]; training rows past 80%: [34, 45, 59, 42, 45]; held-out rows [16837, 16821, 16905, 16909, 16639].

## stall/td3bc_gapped  (seeds [0, 1, 2, 3, 4], starts per bank [200])

Points: mean ± standard error over seeds; in brackets the share of steps the trigger fired.

| Rule | Recorded conditions | Strong current | Big waves | Big kicks |
|---|---|---|---|---|
| never switch (the actor alone) | 86.4 ± 0.4 [0.00] | -17.6 ± 3.1 [0.00] | 70.9 ± 1.1 [0.00] | 52.8 ± 0.5 [0.00] |
| every step -> cloned actor | 56.0 ± 2.1 [1.00] | 53.9 ± 2.4 [1.00] | 43.2 ± 3.4 [1.00] | 5.1 ± 2.9 [1.00] |
| every step -> clip to recorded range | 93.4 ± 0.1 [1.00] | 89.7 ± 1.3 [1.00] | 90.9 ± 0.6 [1.00] | 84.3 ± 0.4 [1.00] |
| every step -> nearest recorded command | 73.4 ± 0.4 [1.00] | 45.9 ± 2.9 [1.00] | 66.8 ± 0.6 [1.00] | 37.4 ± 0.6 [1.00] |
| every step -> clip to full recorded range (min to max) | 87.7 ± 0.4 [1.00] | -15.1 ± 3.0 [1.00] | 72.4 ± 1.2 [1.00] | 71.1 ± 0.3 [1.00] |
| new state, 95% level -> cloned actor | 79.8 ± 0.4 [0.05] | 24.1 ± 2.8 [0.32] | 66.8 ± 0.7 [0.08] | 15.9 ± 2.6 [0.23] |
| new state, 95% level -> clip to recorded range | 89.0 ± 0.3 [0.03] | 37.9 ± 1.5 [0.27] | 76.0 ± 1.0 [0.05] | 80.7 ± 0.5 [0.15] |
| new (state, command) pair, 95% level -> cloned actor | 79.8 ± 0.4 [0.05] | 28.5 ± 2.4 [0.36] | 66.5 ± 0.6 [0.08] | 15.6 ± 2.6 [0.23] |
| new (state, command) pair, 95% level -> clip to recorded range | 89.5 ± 0.3 [0.03] | 42.1 ± 0.9 [0.29] | 76.5 ± 1.0 [0.05] | 81.0 ± 0.5 [0.16] |
| new (state, command) pair, 99% level -> cloned actor | 83.6 ± 0.4 [0.02] | -2.4 ± 3.6 [0.14] | 68.4 ± 0.8 [0.03] | 17.9 ± 2.6 [0.18] |
| new (state, command) pair, 99% level -> clip to recorded range | 87.9 ± 0.3 [0.01] | 4.9 ± 2.5 [0.09] | 73.0 ± 1.1 [0.02] | 78.7 ± 0.5 [0.13] |
| random, at the rate of the 95% (state, command) check -> cloned actor | 85.7 ± 0.3 [0.05] | 13.9 ± 1.7 [0.36] | 69.8 ± 1.0 [0.08] | 46.9 ± 0.9 [0.24] |
| random, at the rate of the 95% (state, command) check -> clip to recorded range | 86.8 ± 0.4 [0.03] | 21.5 ± 1.5 [0.29] | 72.3 ± 1.0 [0.05] | 59.1 ± 0.5 [0.15] |
| random, at the rate of the 99% (state, command) check -> cloned actor | 86.3 ± 0.4 [0.02] | -4.9 ± 2.1 [0.14] | 70.6 ± 1.1 [0.03] | 48.5 ± 0.8 [0.18] |
| random, at the rate of the 99% (state, command) check -> clip to recorded range | 86.5 ± 0.3 [0.01] | -4.2 ± 2.0 [0.09] | 71.4 ± 1.1 [0.02] | 58.2 ± 0.5 [0.13] |
| command outside the commands recorded at nearby states, 95% level -> clip to recorded range | 90.8 ± 0.3 [0.02] | 27.9 ± 5.0 [0.25] | 78.6 ± 1.3 [0.04] | 79.6 ± 0.6 [0.12] |
| random, at the rate of the 95% nearby-commands check -> clip to recorded range | 86.7 ± 0.4 [0.02] | 15.8 ± 3.3 [0.25] | 72.1 ± 1.1 [0.04] | 57.8 ± 0.3 [0.12] |
| perfect detector: the actor proposes past 80% (reference) -> cloned actor | 85.2 ± 0.5 [0.05] | 59.6 ± 2.6 [0.78] | 75.8 ± 0.9 [0.18] | 23.3 ± 2.2 [0.20] |
| perfect detector: the actor proposes past 80% (reference) -> clip to recorded range | 93.6 ± 0.1 [0.03] | 89.8 ± 1.3 [0.68] | 91.1 ± 0.5 [0.15] | 84.5 ± 0.4 [0.14] |
| exact one-step check (reference) -> cloned actor | 91.0 ± 0.2 [0.17] | 48.7 ± 2.3 [0.48] | 83.8 ± 0.4 [0.24] | 69.6 ± 0.3 [0.20] |
| exact one-step check (reference) -> clip to recorded range | 93.5 ± 0.1 [0.03] | 64.9 ± 2.5 [0.48] | 90.5 ± 0.6 [0.14] | 84.8 ± 0.4 [0.12] |

Paired differences by seed (points, mean ± standard error):

| Difference | Recorded conditions | Strong current | Big waves | Big kicks |
|---|---|---|---|---|
| sa95:clone minus random95:clone | -5.9 ± 0.4 | 14.6 ± 1.6 | -3.3 ± 0.4 | -31.3 ± 2.0 |
| sa95:clip99 minus random95:clip99 | 2.7 ± 0.2 | 20.6 ± 0.8 | 4.1 ± 0.1 | 21.9 ± 0.5 |
| sa99:clone minus random99:clone | -2.6 ± 0.2 | 2.5 ± 2.5 | -2.2 ± 0.3 | -30.6 ± 1.9 |
| sa99:clip99 minus random99:clip99 | 1.4 ± 0.0 | 9.1 ± 1.0 | 1.5 ± 0.1 | 20.5 ± 0.5 |
| local95:clip99 minus rlocal95:clip99 | 4.1 ± 0.1 | 12.1 ± 2.2 | 6.5 ± 0.9 | 21.8 ± 0.6 |
| always:clone minus none | -30.4 ± 1.8 | 71.5 ± 1.6 | -27.8 ± 4.3 | -47.7 ± 2.8 |
| always:clip99 minus none | 7.0 ± 0.4 | 107.2 ± 2.4 | 20.0 ± 0.9 | 31.5 ± 0.6 |
| always:project minus none | -13.0 ± 0.5 | 63.5 ± 2.5 | -4.1 ± 0.7 | -15.5 ± 0.7 |
| always:clipmm minus none | 1.3 ± 0.0 | 2.4 ± 0.3 | 1.5 ± 0.1 | 18.2 ± 0.2 |
| state95:clone minus none | -6.6 ± 0.3 | 41.7 ± 2.0 | -4.2 ± 0.5 | -36.9 ± 2.5 |
| state95:clip99 minus none | 2.6 ± 0.1 | 55.5 ± 1.8 | 5.0 ± 0.2 | 27.8 ± 0.5 |
| sa95:clone minus none | -6.6 ± 0.4 | 46.1 ± 2.1 | -4.4 ± 0.5 | -37.2 ± 2.6 |
| sa95:clip99 minus none | 3.1 ± 0.1 | 59.7 ± 2.2 | 5.5 ± 0.2 | 28.1 ± 0.6 |
| sa99:clone minus none | -2.8 ± 0.2 | 15.2 ± 1.9 | -2.5 ± 0.3 | -35.0 ± 2.6 |
| sa99:clip99 minus none | 1.5 ± 0.1 | 22.5 ± 0.7 | 2.1 ± 0.0 | 25.8 ± 0.5 |
| random95:clone minus none | -0.7 ± 0.1 | 31.5 ± 1.6 | -1.1 ± 0.3 | -5.9 ± 0.6 |
| random95:clip99 minus none | 0.4 ± 0.0 | 39.1 ± 2.0 | 1.4 ± 0.1 | 6.2 ± 0.2 |
| random99:clone minus none | -0.1 ± 0.0 | 12.7 ± 1.1 | -0.3 ± 0.1 | -4.4 ± 0.7 |
| random99:clip99 minus none | 0.1 ± 0.1 | 13.4 ± 1.3 | 0.5 ± 0.1 | 5.4 ± 0.2 |
| local95:clip99 minus none | 4.4 ± 0.1 | 45.5 ± 4.9 | 7.7 ± 1.1 | 26.8 ± 0.8 |
| rlocal95:clip99 minus none | 0.3 ± 0.0 | 33.4 ± 2.8 | 1.2 ± 0.2 | 5.0 ± 0.2 |
| unrec:clone minus none | -1.2 ± 0.4 | 77.2 ± 2.1 | 4.9 ± 1.6 | -29.6 ± 2.1 |
| unrec:clip99 minus none | 7.2 ± 0.4 | 107.4 ± 2.4 | 20.2 ± 0.9 | 31.7 ± 0.6 |
| oracle:clone minus none | 4.6 ± 0.2 | 66.3 ± 1.0 | 12.8 ± 0.8 | 16.8 ± 0.3 |
| oracle:clip99 minus none | 7.1 ± 0.4 | 82.4 ± 1.5 | 19.6 ± 0.7 | 32.0 ± 0.5 |

Strong current: share of steps the actor proposes past 80%, share applied past 80%, and the share of past-80% proposals the trigger caught (recall) / share of firings on past-80% proposals (precision):

| Rule | proposed past 80% | applied past 80% | recall | precision |
|---|---|---|---|---|
| never switch (the actor alone) | 0.73 | 0.73 | 0.00 | 0.00 |
| every step -> cloned actor | 0.85 | 0.10 | 1.00 | 0.85 |
| every step -> clip to recorded range | 0.69 | 0.00 | 1.00 | 0.69 |
| every step -> nearest recorded command | 0.82 | 0.02 | 1.00 | 0.82 |
| every step -> clip to full recorded range (min to max) | 0.73 | 0.72 | 1.00 | 0.73 |
| new state, 95% level -> cloned actor | 0.76 | 0.55 | 0.41 | 0.96 |
| new state, 95% level -> clip to recorded range | 0.75 | 0.49 | 0.34 | 0.95 |
| new (state, command) pair, 95% level -> cloned actor | 0.76 | 0.51 | 0.46 | 0.96 |
| new (state, command) pair, 95% level -> clip to recorded range | 0.75 | 0.47 | 0.37 | 0.96 |
| new (state, command) pair, 99% level -> cloned actor | 0.74 | 0.67 | 0.19 | 0.98 |
| new (state, command) pair, 99% level -> clip to recorded range | 0.74 | 0.65 | 0.12 | 0.97 |
| random, at the rate of the 95% (state, command) check -> cloned actor | 0.79 | 0.56 | 0.36 | 0.79 |
| random, at the rate of the 95% (state, command) check -> clip to recorded range | 0.72 | 0.51 | 0.29 | 0.71 |
| random, at the rate of the 99% (state, command) check -> cloned actor | 0.75 | 0.68 | 0.14 | 0.75 |
| random, at the rate of the 99% (state, command) check -> clip to recorded range | 0.72 | 0.66 | 0.09 | 0.72 |
| command outside the commands recorded at nearby states, 95% level -> clip to recorded range | 0.69 | 0.44 | 0.36 | 1.00 |
| random, at the rate of the 95% nearby-commands check -> clip to recorded range | 0.72 | 0.54 | 0.25 | 0.72 |
| perfect detector: the actor proposes past 80% (reference) -> cloned actor | 0.78 | 0.09 | 1.00 | 1.00 |
| perfect detector: the actor proposes past 80% (reference) -> clip to recorded range | 0.68 | 0.00 | 1.00 | 1.00 |
| exact one-step check (reference) -> cloned actor | 0.73 | 0.36 | 0.60 | 0.92 |
| exact one-step check (reference) -> clip to recorded range | 0.71 | 0.24 | 0.65 | 0.97 |

Check against the gate (whole 1,000-start bank, no switching and capped at 80%): largest absolute difference recorded 0.0e+00, strong_current 0.0e+00, big_waves 0.0e+00, big_kicks 0.0e+00; whole-bank points of the actor alone: recorded 86.1 ± 0.3, strong_current -16.8 ± 3.1, big_waves 70.9 ± 1.0, big_kicks 48.7 ± 0.5

Thresholds (mean over seeds): state 3.73 ± 0.03 / 5.21 ± 0.04; (state, command) 4.26 ± 0.03 / 5.71 ± 0.04; nearby commands 0.278 ± 0.001 / 0.411 ± 0.004. Rate above threshold on held-out evaluation-split rows: state 0.049 / 0.009; (state, command) 0.049 / 0.009; nearby commands 0.050 / 0.011. Clip range (99% of recorded commands): low [-0.75, -0.63, -0.7, -0.67], high [0.51, 0.61, 0.55, 0.57]; full-range low [-1.0, -0.75, -0.78, -0.76]; training rows past 80%: [34, 45, 59, 42, 45]; held-out rows [16837, 16821, 16905, 16909, 16639].
