# Study summary

Every entry is a mean over the 9 runs of its setting, method and data arm (6 dataset seeds and 3 training seeds in this folder). Written by `summarize_study.py` from `records.json` and `gates.json`.

## honeypot

Primary outcome, H: episodes ending in hazard 34 (lower is better).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 0.50% | 5.70% | 0.45% |
| support-aware width | 0.00% | 0.00% | 0.00% |
| shuffled control of [support-aware width] | 0.65% | 3.63% | 0.33% |
| sharpened baseline for [support-aware width] | 0.00% | 2.48% | 0.00% |
| support-aware width, calibrated part read at shuffled states | 0.00% | 0.00% | 0.16% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 0.00% | 2.52% | 0.16% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 0.36% | 4.29% | 0.37% |
| support-aware width | 0.00% | 0.00% | 0.00% |
| shuffled control of [support-aware width] | 0.56% | 3.47% | 0.33% |
| sharpened baseline for [support-aware width] | 0.00% | 2.14% | 0.00% |
| support-aware width, calibrated part read at shuffled states | 0.00% | 0.00% | 0.16% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 0.00% | 2.19% | 0.15% |

Average true value of an episode, over all start cells (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 5.08 | 5.04 | 4.99 |
| support-aware width | 5.90 | 5.95 | 5.91 |
| shuffled control of [support-aware width] | 5.29 | 5.06 | 5.56 |
| sharpened baseline for [support-aware width] | 5.77 | 5.76 | 5.74 |
| support-aware width, calibrated part read at shuffled states | 5.90 | 5.97 | 5.91 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 5.77 | 5.76 | 5.73 |

Average true value of an episode starting at a decision cell (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 5.35 | 4.93 | 5.44 |
| support-aware width | 6.24 | 6.32 | 6.30 |
| shuffled control of [support-aware width] | 5.60 | 5.06 | 5.76 |
| sharpened baseline for [support-aware width] | 6.10 | 5.90 | 6.19 |
| support-aware width, calibrated part read at shuffled states | 6.22 | 6.31 | 6.26 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 6.11 | 5.90 | 6.18 |

G1 (the gap misleads the baseline): 3 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| support-aware width | PASS 3/3 | PASS 3/3 | PASS 3/3 | FAIL 1/3 |
| support-aware width, calibrated part read at shuffled states | PASS 3/3 | PASS 3/3 | PASS 3/3 | PASS 2/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 2.26% | 1.06% | 0.02 | 0.29 | 0.87 | 1.81 | 0.47 | 90.08% | 100.00% |
| baseline | targeted | 89.19% | 1.12% | 11.67 | 0.52 | 1.55 | 1.37 | 9.99 | 90.56% | 100.00% |
| baseline | placebo | 76.78% | 1.03% | 0.00 | 0.38 | 1.14 | 1.83 | 0.56 | 90.69% | 100.00% |
| support-aware width | complete | 2.26% | 1.06% | 0.02 | 0.39 | 1.17 | 1.75 | 0.52 | 90.36% | 100.00% |
| support-aware width | targeted | 89.19% | 1.12% | 11.67 | 82.44 | 247.32 | 1.27 | 10.48 | 90.97% | 100.00% |
| support-aware width | placebo | 76.78% | 1.03% | 0.00 | 0.35 | 1.04 | 1.70 | 0.55 | 90.62% | 100.00% |
| support-aware width, calibrated part read at shuffled states | complete | 2.26% | 1.06% | 0.02 | 0.39 | 1.16 | 1.67 | 0.54 | 90.66% | 100.00% |
| support-aware width, calibrated part read at shuffled states | targeted | 89.19% | 1.12% | 11.67 | 82.49 | 247.46 | 1.26 | 10.48 | 90.80% | 100.00% |
| support-aware width, calibrated part read at shuffled states | placebo | 76.78% | 1.03% | 0.00 | 0.35 | 1.04 | 1.77 | 0.51 | 90.70% | 100.00% |

## bottleneck_r2_exit25_one_hot

Primary outcome, X: episodes from B reaching the goal over the unseen bridge (a cautious method lowers it).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 92.13% | 12.66% | 92.68% |
| support-aware width | 100.00% | 0.65% | 88.89% |
| shuffled control of [support-aware width] | 100.00% | 0.00% | 88.89% |
| sharpened baseline for [support-aware width] | 98.53% | 11.91% | 98.03% |
| support-aware width, calibrated part read at shuffled states | 100.00% | 0.00% | 88.89% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 98.54% | 11.55% | 98.77% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 78.96% | 11.34% | 83.07% |
| support-aware width | 100.00% | 0.65% | 88.89% |
| shuffled control of [support-aware width] | 97.20% | 0.00% | 88.89% |
| sharpened baseline for [support-aware width] | 94.54% | 11.61% | 96.13% |
| support-aware width, calibrated part read at shuffled states | 99.36% | 0.00% | 88.89% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 94.04% | 11.32% | 96.91% |

Average true value of an episode, over all start cells (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 3.95 | 3.37 | 4.04 |
| support-aware width | 5.17 | 3.89 | 5.05 |
| shuffled control of [support-aware width] | 5.00 | 2.15 | 4.75 |
| sharpened baseline for [support-aware width] | 4.96 | 3.92 | 5.00 |
| support-aware width, calibrated part read at shuffled states | 5.20 | 3.95 | 5.06 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 4.98 | 3.93 | 5.03 |

Average true value of an episode starting at a decision cell (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 4.29 | 2.62 | 4.43 |
| support-aware width | 5.54 | 2.52 | 5.20 |
| shuffled control of [support-aware width] | 5.36 | 2.49 | 5.18 |
| sharpened baseline for [support-aware width] | 5.36 | 2.81 | 5.36 |
| support-aware width, calibrated part read at shuffled states | 5.53 | 2.50 | 5.20 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 5.37 | 2.80 | 5.38 |

G1 (the gap misleads the baseline): 3 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| support-aware width | PASS 3/3 | FAIL 0/3 | PASS 2/3 | FAIL 1/3 |
| support-aware width, calibrated part read at shuffled states | PASS 3/3 | FAIL 0/3 | PASS 2/3 | PASS 3/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 0.87% | 0.85% | 0.00 | 0.48 | 1.43 | 1.43 | -0.19 | 90.95% | 100.00% |
| baseline | targeted | 94.10% | 1.02% | 9.27 | 0.46 | 1.39 | 1.39 | -1.78 | 91.28% | 100.00% |
| baseline | placebo | 73.61% | 0.73% | 0.00 | 0.47 | 1.41 | 1.41 | -0.34 | 91.67% | 100.00% |
| support-aware width | complete | 0.87% | 0.85% | 0.00 | 0.45 | 1.34 | 1.34 | -0.43 | 91.56% | 100.00% |
| support-aware width | targeted | 94.10% | 1.02% | 9.27 | 66.01 | 198.03 | 198.03 | -2.51 | 91.06% | 100.00% |
| support-aware width | placebo | 73.61% | 0.73% | 0.00 | 0.43 | 1.30 | 1.30 | -0.51 | 91.12% | 100.00% |
| support-aware width, calibrated part read at shuffled states | complete | 0.87% | 0.85% | 0.00 | 0.45 | 1.35 | 1.35 | -0.48 | 91.68% | 100.00% |
| support-aware width, calibrated part read at shuffled states | targeted | 94.10% | 1.02% | 9.27 | 66.09 | 198.27 | 198.27 | -2.61 | 90.94% | 100.00% |
| support-aware width, calibrated part read at shuffled states | placebo | 73.61% | 0.73% | 0.00 | 0.43 | 1.29 | 1.29 | -0.52 | 91.35% | 100.00% |

## bottleneck_r2_exit35_one_hot

Primary outcome, X: episodes from B reaching the goal over the unseen bridge (a cautious method lowers it).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 86.28% | 8.44% | 91.37% |
| support-aware width | 100.00% | 0.00% | 100.00% |
| shuffled control of [support-aware width] | 99.96% | 0.00% | 100.00% |
| sharpened baseline for [support-aware width] | 96.33% | 3.37% | 99.36% |
| support-aware width, calibrated part read at shuffled states | 100.00% | 0.70% | 100.00% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 96.27% | 3.31% | 99.38% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 77.53% | 7.43% | 80.82% |
| support-aware width | 99.37% | 0.00% | 100.00% |
| shuffled control of [support-aware width] | 97.22% | 0.00% | 100.00% |
| sharpened baseline for [support-aware width] | 92.78% | 3.05% | 98.57% |
| support-aware width, calibrated part read at shuffled states | 100.00% | 0.70% | 100.00% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 92.91% | 2.99% | 97.87% |

Average true value of an episode, over all start cells (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 3.94 | 3.66 | 4.01 |
| support-aware width | 5.17 | 4.32 | 5.12 |
| shuffled control of [support-aware width] | 5.02 | 3.67 | 4.87 |
| sharpened baseline for [support-aware width] | 4.91 | 4.25 | 4.99 |
| support-aware width, calibrated part read at shuffled states | 5.14 | 4.36 | 5.19 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 4.91 | 4.25 | 5.00 |

Average true value of an episode starting at a decision cell (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 4.36 | 3.43 | 4.46 |
| support-aware width | 5.52 | 3.49 | 5.52 |
| shuffled control of [support-aware width] | 5.38 | 3.49 | 5.44 |
| sharpened baseline for [support-aware width] | 5.28 | 3.52 | 5.43 |
| support-aware width, calibrated part read at shuffled states | 5.54 | 3.51 | 5.53 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 5.28 | 3.52 | 5.43 |

G1 (the gap misleads the baseline): 3 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| support-aware width | PASS 3/3 | FAIL 0/3 | PASS 3/3 | FAIL 1/3 |
| support-aware width, calibrated part read at shuffled states | PASS 3/3 | FAIL 0/3 | PASS 3/3 | FAIL 1/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 0.87% | 0.85% | 0.00 | 0.49 | 1.46 | 1.46 | -0.25 | 91.24% | 100.00% |
| baseline | targeted | 94.10% | 1.02% | 9.27 | 0.49 | 1.47 | 1.47 | -0.71 | 91.22% | 100.00% |
| baseline | placebo | 73.61% | 0.73% | 0.00 | 0.47 | 1.41 | 1.41 | -0.26 | 90.90% | 100.00% |
| support-aware width | complete | 0.87% | 0.85% | 0.00 | 0.42 | 1.27 | 1.27 | -0.40 | 91.13% | 100.00% |
| support-aware width | targeted | 94.10% | 1.02% | 9.27 | 65.77 | 197.30 | 197.30 | -1.56 | 91.38% | 100.00% |
| support-aware width | placebo | 73.61% | 0.73% | 0.00 | 0.42 | 1.25 | 1.25 | -0.43 | 90.49% | 100.00% |
| support-aware width, calibrated part read at shuffled states | complete | 0.87% | 0.85% | 0.00 | 0.43 | 1.30 | 1.30 | -0.46 | 91.01% | 100.00% |
| support-aware width, calibrated part read at shuffled states | targeted | 94.10% | 1.02% | 9.27 | 65.84 | 197.52 | 197.52 | -1.53 | 91.41% | 100.00% |
| support-aware width, calibrated part read at shuffled states | placebo | 73.61% | 0.73% | 0.00 | 0.41 | 1.23 | 1.23 | -0.48 | 90.75% | 100.00% |

## bottleneck_r2_exit45_one_hot

Primary outcome, X: episodes from B reaching the goal over the unseen bridge (a cautious method lowers it).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 71.28% | 4.76% | 82.55% |
| support-aware width | 82.51% | 0.00% | 88.75% |
| shuffled control of [support-aware width] | 75.06% | 0.00% | 91.61% |
| sharpened baseline for [support-aware width] | 78.22% | 3.70% | 92.96% |
| support-aware width, calibrated part read at shuffled states | 90.95% | 0.00% | 100.00% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 78.96% | 3.68% | 93.00% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 64.15% | 5.11% | 76.35% |
| support-aware width | 80.97% | 0.00% | 88.75% |
| shuffled control of [support-aware width] | 71.46% | 0.00% | 90.72% |
| sharpened baseline for [support-aware width] | 77.48% | 3.70% | 92.25% |
| support-aware width, calibrated part read at shuffled states | 89.38% | 0.00% | 100.00% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 78.67% | 3.68% | 92.31% |

Average true value of an episode, over all start cells (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 4.10 | 4.05 | 4.07 |
| support-aware width | 5.12 | 4.82 | 5.15 |
| shuffled control of [support-aware width] | 4.98 | 3.93 | 4.98 |
| sharpened baseline for [support-aware width] | 4.96 | 4.73 | 5.04 |
| support-aware width, calibrated part read at shuffled states | 5.16 | 4.82 | 5.18 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 4.98 | 4.74 | 5.05 |

Average true value of an episode starting at a decision cell (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 4.45 | 4.37 | 4.51 |
| support-aware width | 5.33 | 4.50 | 5.43 |
| shuffled control of [support-aware width] | 5.12 | 4.50 | 5.44 |
| sharpened baseline for [support-aware width] | 5.19 | 4.51 | 5.41 |
| support-aware width, calibrated part read at shuffled states | 5.41 | 4.50 | 5.54 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 5.20 | 4.51 | 5.41 |

G1 (the gap misleads the baseline): 3 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| support-aware width | PASS 3/3 | FAIL 0/3 | PASS 2/3 | PASS 2/3 |
| support-aware width, calibrated part read at shuffled states | PASS 3/3 | FAIL 0/3 | PASS 3/3 | PASS 3/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 0.87% | 0.85% | 0.00 | 0.47 | 1.40 | 1.39 | -0.14 | 91.16% | 100.00% |
| baseline | targeted | 94.10% | 1.02% | 9.27 | 0.50 | 1.50 | 1.27 | 0.31 | 91.52% | 100.00% |
| baseline | placebo | 73.61% | 0.73% | 0.00 | 0.49 | 1.47 | 1.54 | -0.08 | 91.16% | 100.00% |
| support-aware width | complete | 0.87% | 0.85% | 0.00 | 0.44 | 1.33 | 1.33 | -0.42 | 91.41% | 100.00% |
| support-aware width | targeted | 94.10% | 1.02% | 9.27 | 66.11 | 198.33 | 198.33 | -0.67 | 91.13% | 100.00% |
| support-aware width | placebo | 73.61% | 0.73% | 0.00 | 0.42 | 1.27 | 1.27 | -0.42 | 91.55% | 100.00% |
| support-aware width, calibrated part read at shuffled states | complete | 0.87% | 0.85% | 0.00 | 0.44 | 1.32 | 1.32 | -0.49 | 91.05% | 100.00% |
| support-aware width, calibrated part read at shuffled states | targeted | 94.10% | 1.02% | 9.27 | 66.10 | 198.30 | 198.30 | -0.67 | 91.26% | 100.00% |
| support-aware width, calibrated part read at shuffled states | placebo | 73.61% | 0.73% | 0.00 | 0.43 | 1.29 | 1.29 | -0.47 | 91.38% | 100.00% |
