# Study summary

Every entry is a mean over the 27 runs of its setting, method and data arm (9 dataset seeds and 3 training seeds in this folder). Written by `summarize_study.py` from `records.json` and `gates.json`.

## honeypot

Primary outcome, H: episodes ending in hazard 34 (lower is better).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 0.21% | 6.65% | 0.11% |
| support-aware width, calibrated part read at shuffled states | 0.00% | 0.06% | 0.00% |
| shuffled control of [support-aware width] | 0.69% | 5.48% | 0.35% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 0.00% | 3.18% | 0.00% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 0.16% | 4.62% | 0.10% |
| support-aware width, calibrated part read at shuffled states | 0.00% | 0.05% | 0.00% |
| shuffled control of [support-aware width] | 0.42% | 4.57% | 0.29% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 0.00% | 2.66% | 0.00% |

Average true value of an episode, over all start cells (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 5.02 | 4.97 | 5.07 |
| support-aware width, calibrated part read at shuffled states | 5.92 | 5.93 | 5.94 |
| shuffled control of [support-aware width] | 5.42 | 3.02 | 5.27 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 5.75 | 5.74 | 5.74 |

Average true value of an episode starting at a decision cell (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 5.36 | 4.72 | 5.53 |
| support-aware width, calibrated part read at shuffled states | 6.27 | 6.26 | 6.30 |
| shuffled control of [support-aware width] | 5.67 | 2.88 | 5.58 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 6.17 | 5.83 | 6.21 |

G1 (the gap misleads the baseline): 9 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| support-aware width, calibrated part read at shuffled states | PASS 9/3 | PASS 5/3 | PASS 9/3 | PASS 8/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 1.95% | 1.02% | 0.06 | 0.32 | 0.97 | 1.95 | 1.48 | 91.93% | 100.00% |
| baseline | targeted | 89.15% | 0.97% | 16.35 | 0.58 | 1.73 | 1.49 | 10.03 | 92.21% | 100.00% |
| baseline | placebo | 76.94% | 1.00% | 0.02 | 0.40 | 1.19 | 2.09 | 0.55 | 91.99% | 100.00% |
| support-aware width, calibrated part read at shuffled states | complete | 1.95% | 1.02% | 0.06 | 0.75 | 2.26 | 1.89 | 1.54 | 91.92% | 100.00% |
| support-aware width, calibrated part read at shuffled states | targeted | 89.15% | 0.97% | 16.35 | 114.28 | 342.83 | 1.37 | 10.60 | 92.25% | 100.00% |
| support-aware width, calibrated part read at shuffled states | placebo | 76.94% | 1.00% | 0.02 | 0.53 | 1.59 | 1.93 | 0.59 | 91.92% | 100.00% |

## bottleneck_r2_exit25_one_hot

Primary outcome, X: episodes from B reaching the goal over the unseen bridge (a cautious method lowers it).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 97.14% | 8.83% | 97.51% |
| support-aware width, calibrated part read at shuffled states | 100.00% | 0.00% | 99.80% |
| shuffled control of [support-aware width] | 99.77% | 3.73% | 99.17% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 100.00% | 6.95% | 99.77% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 86.77% | 7.72% | 91.31% |
| support-aware width, calibrated part read at shuffled states | 99.80% | 0.00% | 98.38% |
| shuffled control of [support-aware width] | 97.48% | 3.71% | 97.10% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 97.04% | 6.63% | 98.30% |

Average true value of an episode, over all start cells (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 3.96 | 3.26 | 3.93 |
| support-aware width, calibrated part read at shuffled states | 5.16 | 3.91 | 5.13 |
| shuffled control of [support-aware width] | 4.96 | 2.98 | 4.76 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 4.96 | 3.77 | 4.94 |

Average true value of an episode starting at a decision cell (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 4.33 | 2.57 | 4.34 |
| support-aware width, calibrated part read at shuffled states | 5.51 | 2.49 | 5.46 |
| shuffled control of [support-aware width] | 5.32 | 2.59 | 5.39 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 5.34 | 2.65 | 5.35 |

G1 (the gap misleads the baseline): 9 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| support-aware width, calibrated part read at shuffled states | PASS 9/3 | PASS 3/3 | PASS 9/3 | PASS 5/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 2.14% | 0.90% | 0.00 | 0.50 | 1.49 | 1.49 | 0.10 | 91.24% | 100.00% |
| baseline | targeted | 87.27% | 1.04% | 10.99 | 0.48 | 1.44 | 1.44 | -1.63 | 91.41% | 100.00% |
| baseline | placebo | 73.84% | 1.01% | 0.00 | 0.51 | 1.54 | 1.54 | 0.15 | 91.29% | 100.00% |
| support-aware width, calibrated part read at shuffled states | complete | 2.14% | 0.90% | 0.00 | 0.46 | 1.37 | 1.37 | -0.32 | 91.54% | 100.00% |
| support-aware width, calibrated part read at shuffled states | targeted | 87.27% | 1.04% | 10.99 | 77.97 | 233.91 | 233.91 | -2.44 | 91.61% | 100.00% |
| support-aware width, calibrated part read at shuffled states | placebo | 73.84% | 1.01% | 0.00 | 0.46 | 1.37 | 1.37 | -0.25 | 91.35% | 100.00% |

## bottleneck_r2_exit35_one_hot

Primary outcome, X: episodes from B reaching the goal over the unseen bridge (a cautious method lowers it).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 92.01% | 5.78% | 94.97% |
| support-aware width, calibrated part read at shuffled states | 99.77% | 0.28% | 99.51% |
| shuffled control of [support-aware width] | 98.71% | 0.41% | 98.99% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 96.49% | 4.31% | 99.78% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 82.98% | 5.51% | 89.69% |
| support-aware width, calibrated part read at shuffled states | 99.27% | 0.28% | 99.09% |
| shuffled control of [support-aware width] | 96.79% | 0.40% | 97.73% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 94.28% | 4.37% | 98.70% |

Average true value of an episode, over all start cells (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 3.98 | 3.67 | 3.96 |
| support-aware width, calibrated part read at shuffled states | 5.14 | 4.40 | 5.16 |
| shuffled control of [support-aware width] | 4.97 | 3.35 | 4.71 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 4.90 | 4.23 | 4.93 |

Average true value of an episode starting at a decision cell (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 4.33 | 3.45 | 4.37 |
| support-aware width, calibrated part read at shuffled states | 5.50 | 3.50 | 5.49 |
| shuffled control of [support-aware width] | 5.32 | 3.48 | 5.40 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 5.23 | 3.56 | 5.33 |

G1 (the gap misleads the baseline): 9 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| support-aware width, calibrated part read at shuffled states | PASS 9/3 | FAIL 1/3 | PASS 9/3 | PASS 7/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 2.14% | 0.90% | 0.00 | 0.49 | 1.48 | 1.48 | 0.08 | 91.33% | 100.00% |
| baseline | targeted | 87.27% | 1.04% | 10.99 | 0.48 | 1.45 | 1.45 | -0.84 | 91.51% | 100.00% |
| baseline | placebo | 73.84% | 1.01% | 0.00 | 0.49 | 1.48 | 1.48 | 0.11 | 91.20% | 100.00% |
| support-aware width, calibrated part read at shuffled states | complete | 2.14% | 0.90% | 0.00 | 0.44 | 1.32 | 1.32 | -0.34 | 91.47% | 100.00% |
| support-aware width, calibrated part read at shuffled states | targeted | 87.27% | 1.04% | 10.99 | 78.07 | 234.22 | 234.22 | -1.66 | 91.69% | 100.00% |
| support-aware width, calibrated part read at shuffled states | placebo | 73.84% | 1.01% | 0.00 | 0.44 | 1.33 | 1.33 | -0.32 | 91.63% | 100.00% |

## bottleneck_r2_exit45_one_hot

Primary outcome, X: episodes from B reaching the goal over the unseen bridge (a cautious method lowers it).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 77.07% | 1.72% | 80.83% |
| support-aware width, calibrated part read at shuffled states | 96.58% | 0.31% | 98.63% |
| shuffled control of [support-aware width] | 96.08% | 0.47% | 94.59% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 86.74% | 0.55% | 87.38% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 66.63% | 1.75% | 74.24% |
| support-aware width, calibrated part read at shuffled states | 96.58% | 0.31% | 98.63% |
| shuffled control of [support-aware width] | 94.40% | 0.45% | 93.34% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 81.55% | 0.57% | 85.27% |

Average true value of an episode, over all start cells (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 4.04 | 4.02 | 4.04 |
| support-aware width, calibrated part read at shuffled states | 5.15 | 4.81 | 5.17 |
| shuffled control of [support-aware width] | 4.98 | 4.16 | 4.82 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 4.90 | 4.65 | 4.92 |

Average true value of an episode starting at a decision cell (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 4.26 | 4.38 | 4.36 |
| support-aware width, calibrated part read at shuffled states | 5.45 | 4.50 | 5.46 |
| shuffled control of [support-aware width] | 5.29 | 4.48 | 5.37 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 5.14 | 4.47 | 5.19 |

G1 (the gap misleads the baseline): 9 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| support-aware width, calibrated part read at shuffled states | PASS 9/3 | FAIL 1/3 | PASS 9/3 | PASS 9/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 2.14% | 0.90% | 0.00 | 0.50 | 1.51 | 1.47 | 0.16 | 91.15% | 100.00% |
| baseline | targeted | 87.27% | 1.04% | 10.99 | 0.47 | 1.42 | 1.26 | 0.03 | 91.19% | 100.00% |
| baseline | placebo | 73.84% | 1.01% | 0.00 | 0.50 | 1.50 | 1.40 | 0.07 | 91.19% | 100.00% |
| support-aware width, calibrated part read at shuffled states | complete | 2.14% | 0.90% | 0.00 | 0.44 | 1.32 | 1.32 | -0.32 | 91.42% | 100.00% |
| support-aware width, calibrated part read at shuffled states | targeted | 87.27% | 1.04% | 10.99 | 78.12 | 234.35 | 223.94 | -0.82 | 91.49% | 100.00% |
| support-aware width, calibrated part read at shuffled states | placebo | 73.84% | 1.01% | 0.00 | 0.46 | 1.38 | 1.38 | -0.35 | 91.55% | 100.00% |
