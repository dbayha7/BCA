# Study summary

Every entry is a mean over the 9 runs of its setting, method and data arm (3 dataset seeds and 3 training seeds in this folder). Written by `summarize_study.py` from `records.json` and `gates.json`.

## honeypot

Primary outcome, H: episodes ending in hazard 34 (lower is better).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 0.39% | 3.50% | 0.45% |
| support-aware + one-sided + in the critic | 0.00% | 0.23% | 0.37% |
| shuffled control of [support-aware + one-sided + in the critic] | 0.77% | 2.61% | 0.20% |
| sharpened baseline for [support-aware + one-sided + in the critic] | 0.37% | 3.10% | 0.40% |
| support-aware width, calibrated part read at shuffled states | 0.00% | 0.18% | 0.00% |
| shuffled control of [support-aware width] | 0.77% | 0.42% | 0.17% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 0.00% | 0.57% | 0.00% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 0.34% | 2.64% | 0.22% |
| support-aware + one-sided + in the critic | 0.00% | 0.21% | 0.17% |
| shuffled control of [support-aware + one-sided + in the critic] | 0.65% | 2.16% | 0.20% |
| sharpened baseline for [support-aware + one-sided + in the critic] | 0.33% | 2.36% | 0.21% |
| support-aware width, calibrated part read at shuffled states | 0.00% | 0.17% | 0.00% |
| shuffled control of [support-aware width] | 0.67% | 0.37% | 0.17% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 0.00% | 0.45% | 0.00% |

Average true value of an episode, over all start cells (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 5.23 | 5.14 | 5.28 |
| support-aware + one-sided + in the critic | 5.28 | 5.22 | 5.29 |
| shuffled control of [support-aware + one-sided + in the critic] | 5.16 | 4.68 | 5.66 |
| sharpened baseline for [support-aware + one-sided + in the critic] | 5.35 | 5.24 | 5.40 |
| support-aware width, calibrated part read at shuffled states | 5.92 | 5.93 | 5.95 |
| shuffled control of [support-aware width] | 5.64 | 3.55 | 5.88 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 5.85 | 5.80 | 5.83 |

Average true value of an episode starting at a decision cell (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 5.61 | 5.18 | 5.57 |
| support-aware + one-sided + in the critic | 5.68 | 5.57 | 5.62 |
| shuffled control of [support-aware + one-sided + in the critic] | 5.41 | 5.32 | 5.96 |
| sharpened baseline for [support-aware + one-sided + in the critic] | 5.72 | 5.34 | 5.71 |
| support-aware width, calibrated part read at shuffled states | 6.28 | 6.22 | 6.29 |
| shuffled control of [support-aware width] | 5.87 | 4.05 | 6.18 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 6.18 | 6.15 | 6.20 |

G1 (the gap misleads the baseline): 3 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| support-aware + one-sided + in the critic | PASS 3/3 | PASS 3/3 | PASS 3/3 | FAIL 0/3 |
| support-aware width, calibrated part read at shuffled states | PASS 3/3 | FAIL 1/3 | PASS 3/3 | PASS 2/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 2.13% | 1.37% | 0.02 | 0.30 | 0.91 | 2.03 | 0.79 | 91.68% | 100.00% |
| baseline | targeted | 89.50% | 1.25% | 14.00 | 0.56 | 1.69 | 1.57 | 9.90 | 91.26% | 100.00% |
| baseline | placebo | 74.96% | 1.18% | 0.00 | 0.39 | 1.16 | 2.13 | 0.01 | 91.56% | 100.00% |
| support-aware + one-sided + in the critic | complete | 2.13% | 1.37% | 0.02 | 0.79 | 0.28 | 1.01 | 0.66 | 89.53% | 100.00% |
| support-aware + one-sided + in the critic | targeted | 89.50% | 1.25% | 14.00 | 93.09 | 274.88 | 0.00 | 8.84 | 88.84% | 100.00% |
| support-aware + one-sided + in the critic | placebo | 74.96% | 1.18% | 0.00 | 1.00 | 0.00 | 7.56 | 0.05 | 88.09% | 100.00% |
| support-aware width, calibrated part read at shuffled states | complete | 2.13% | 1.37% | 0.02 | 0.47 | 1.41 | 2.76 | 0.85 | 91.90% | 100.00% |
| support-aware width, calibrated part read at shuffled states | targeted | 89.50% | 1.25% | 14.00 | 98.00 | 294.01 | 1.46 | 10.38 | 91.25% | 100.00% |
| support-aware width, calibrated part read at shuffled states | placebo | 74.96% | 1.18% | 0.00 | 0.34 | 1.01 | 2.04 | 0.06 | 91.56% | 100.00% |

## bottleneck_r2_exit25_one_hot

Primary outcome, X: episodes from B reaching the goal over the unseen bridge (a cautious method lowers it).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 96.71% | 15.66% | 94.64% |
| support-aware + one-sided + in the critic | 96.67% | 0.65% | 97.89% |
| shuffled control of [support-aware + one-sided + in the critic] | 95.51% | 2.26% | 85.38% |
| sharpened baseline for [support-aware + one-sided + in the critic] | 96.85% | 15.28% | 95.08% |
| support-aware width, calibrated part read at shuffled states | 100.00% | 0.00% | 100.00% |
| shuffled control of [support-aware width] | 94.97% | 1.75% | 98.45% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 100.00% | 11.04% | 99.23% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 89.89% | 14.75% | 89.43% |
| support-aware + one-sided + in the critic | 93.66% | 0.65% | 97.13% |
| shuffled control of [support-aware + one-sided + in the critic] | 89.14% | 3.75% | 79.23% |
| sharpened baseline for [support-aware + one-sided + in the critic] | 91.37% | 14.52% | 92.28% |
| support-aware width, calibrated part read at shuffled states | 99.30% | 0.00% | 100.00% |
| shuffled control of [support-aware width] | 92.11% | 2.07% | 94.61% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 99.15% | 10.49% | 99.23% |

Average true value of an episode, over all start cells (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 4.16 | 3.38 | 4.12 |
| support-aware + one-sided + in the critic | 4.13 | 3.22 | 3.97 |
| shuffled control of [support-aware + one-sided + in the critic] | 4.27 | 2.54 | 4.00 |
| sharpened baseline for [support-aware + one-sided + in the critic] | 4.26 | 3.46 | 4.24 |
| support-aware width, calibrated part read at shuffled states | 4.91 | 3.93 | 4.91 |
| shuffled control of [support-aware width] | 4.58 | 3.19 | 4.44 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 5.04 | 3.85 | 5.03 |

Average true value of an episode starting at a decision cell (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 4.58 | 2.75 | 4.50 |
| support-aware + one-sided + in the critic | 4.67 | 2.41 | 4.61 |
| shuffled control of [support-aware + one-sided + in the critic] | 4.72 | 2.48 | 4.43 |
| sharpened baseline for [support-aware + one-sided + in the critic] | 4.68 | 2.77 | 4.62 |
| support-aware width, calibrated part read at shuffled states | 5.51 | 2.50 | 5.54 |
| shuffled control of [support-aware width] | 5.13 | 2.52 | 5.32 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 5.38 | 2.78 | 5.36 |

G1 (the gap misleads the baseline): 3 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| support-aware + one-sided + in the critic | PASS 3/3 | PASS 2/3 | PASS 3/3 | FAIL 0/3 |
| support-aware width, calibrated part read at shuffled states | PASS 3/3 | PASS 2/3 | PASS 3/3 | PASS 2/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 3.12% | 1.09% | 0.00 | 0.53 | 1.60 | 1.60 | -0.32 | 92.16% | 100.00% |
| baseline | targeted | 88.02% | 1.10% | 9.15 | 0.48 | 1.45 | 1.45 | -1.86 | 92.31% | 100.00% |
| baseline | placebo | 71.53% | 1.21% | 0.00 | 0.49 | 1.48 | 1.48 | -0.05 | 91.82% | 100.00% |
| support-aware + one-sided + in the critic | complete | 3.12% | 1.09% | 0.00 | 0.74 | 0.00 | 0.00 | -0.73 | 91.45% | 100.00% |
| support-aware + one-sided + in the critic | targeted | 88.02% | 1.10% | 9.15 | 63.27 | 185.50 | 185.50 | -2.08 | 91.28% | 100.00% |
| support-aware + one-sided + in the critic | placebo | 71.53% | 1.21% | 0.00 | 0.80 | 0.09 | 0.09 | -0.60 | 91.45% | 100.00% |
| support-aware width, calibrated part read at shuffled states | complete | 3.12% | 1.09% | 0.00 | 0.48 | 1.44 | 1.44 | -0.51 | 92.56% | 100.00% |
| support-aware width, calibrated part read at shuffled states | targeted | 88.02% | 1.10% | 9.15 | 64.87 | 194.60 | 194.60 | -2.49 | 92.70% | 100.00% |
| support-aware width, calibrated part read at shuffled states | placebo | 71.53% | 1.21% | 0.00 | 0.46 | 1.38 | 1.38 | -0.36 | 92.34% | 100.00% |

## bottleneck_r2_exit35_one_hot

Primary outcome, X: episodes from B reaching the goal over the unseen bridge (a cautious method lowers it).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 81.38% | 11.70% | 91.02% |
| support-aware + one-sided + in the critic | 81.32% | 0.00% | 84.61% |
| shuffled control of [support-aware + one-sided + in the critic] | 79.34% | 2.15% | 67.45% |
| sharpened baseline for [support-aware + one-sided + in the critic] | 82.56% | 11.68% | 91.20% |
| support-aware width, calibrated part read at shuffled states | 100.00% | 0.00% | 100.00% |
| shuffled control of [support-aware width] | 93.08% | 0.00% | 96.74% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 89.09% | 10.57% | 98.72% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 73.52% | 10.43% | 86.94% |
| support-aware + one-sided + in the critic | 76.97% | 0.00% | 83.29% |
| shuffled control of [support-aware + one-sided + in the critic] | 74.95% | 4.49% | 63.09% |
| sharpened baseline for [support-aware + one-sided + in the critic] | 75.17% | 10.42% | 87.27% |
| support-aware width, calibrated part read at shuffled states | 100.00% | 0.00% | 100.00% |
| shuffled control of [support-aware width] | 90.70% | 0.00% | 94.39% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 89.09% | 9.40% | 98.72% |

Average true value of an episode, over all start cells (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 3.97 | 3.69 | 4.04 |
| support-aware + one-sided + in the critic | 4.01 | 3.63 | 3.94 |
| shuffled control of [support-aware + one-sided + in the critic] | 4.15 | 2.85 | 3.95 |
| sharpened baseline for [support-aware + one-sided + in the critic] | 4.10 | 3.73 | 4.09 |
| support-aware width, calibrated part read at shuffled states | 4.93 | 4.36 | 5.11 |
| shuffled control of [support-aware width] | 4.76 | 3.24 | 4.71 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 4.95 | 4.26 | 4.95 |

Average true value of an episode starting at a decision cell (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 4.28 | 3.56 | 4.46 |
| support-aware + one-sided + in the critic | 4.47 | 3.45 | 4.60 |
| shuffled control of [support-aware + one-sided + in the critic] | 4.50 | 3.37 | 4.37 |
| sharpened baseline for [support-aware + one-sided + in the critic] | 4.39 | 3.55 | 4.53 |
| support-aware width, calibrated part read at shuffled states | 5.55 | 3.49 | 5.54 |
| shuffled control of [support-aware width] | 5.18 | 3.48 | 5.32 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 5.21 | 3.64 | 5.36 |

G1 (the gap misleads the baseline): 3 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| support-aware + one-sided + in the critic | PASS 3/3 | PASS 3/3 | PASS 2/3 | FAIL 1/3 |
| support-aware width, calibrated part read at shuffled states | PASS 3/3 | FAIL 0/3 | PASS 3/3 | PASS 2/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 3.12% | 1.09% | 0.00 | 0.51 | 1.53 | 1.53 | -0.24 | 92.00% | 100.00% |
| baseline | targeted | 88.02% | 1.10% | 9.15 | 0.46 | 1.38 | 1.38 | -1.26 | 92.23% | 100.00% |
| baseline | placebo | 71.53% | 1.21% | 0.00 | 0.49 | 1.47 | 1.47 | -0.14 | 92.10% | 100.00% |
| support-aware + one-sided + in the critic | complete | 3.12% | 1.09% | 0.00 | 0.76 | 0.00 | 0.00 | -0.47 | 91.26% | 100.00% |
| support-aware + one-sided + in the critic | targeted | 88.02% | 1.10% | 9.15 | 62.74 | 184.11 | 184.11 | -1.68 | 91.52% | 100.00% |
| support-aware + one-sided + in the critic | placebo | 71.53% | 1.21% | 0.00 | 1.00 | 0.35 | 0.35 | -0.69 | 90.33% | 100.00% |
| support-aware width, calibrated part read at shuffled states | complete | 3.12% | 1.09% | 0.00 | 0.47 | 1.40 | 1.40 | -0.50 | 92.05% | 100.00% |
| support-aware width, calibrated part read at shuffled states | targeted | 88.02% | 1.10% | 9.15 | 65.11 | 195.33 | 195.33 | -1.95 | 92.63% | 100.00% |
| support-aware width, calibrated part read at shuffled states | placebo | 71.53% | 1.21% | 0.00 | 0.43 | 1.29 | 1.29 | -0.45 | 92.32% | 100.00% |

## bottleneck_r2_exit45_one_hot

Primary outcome, X: episodes from B reaching the goal over the unseen bridge (a cautious method lowers it).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 82.06% | 2.78% | 63.92% |
| support-aware + one-sided + in the critic | 82.09% | 1.68% | 61.36% |
| shuffled control of [support-aware + one-sided + in the critic] | 66.75% | 1.06% | 33.88% |
| sharpened baseline for [support-aware + one-sided + in the critic] | 82.58% | 2.69% | 64.86% |
| support-aware width, calibrated part read at shuffled states | 99.37% | 0.00% | 100.00% |
| shuffled control of [support-aware width] | 97.00% | 0.00% | 94.56% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 91.35% | 1.25% | 76.13% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 73.24% | 2.63% | 59.23% |
| support-aware + one-sided + in the critic | 76.76% | 1.68% | 60.91% |
| shuffled control of [support-aware + one-sided + in the critic] | 60.56% | 2.32% | 30.82% |
| sharpened baseline for [support-aware + one-sided + in the critic] | 74.56% | 2.55% | 60.46% |
| support-aware width, calibrated part read at shuffled states | 99.37% | 0.00% | 100.00% |
| shuffled control of [support-aware width] | 91.76% | 0.00% | 91.65% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 90.53% | 1.25% | 73.97% |

Average true value of an episode, over all start cells (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 4.15 | 4.14 | 3.99 |
| support-aware + one-sided + in the critic | 4.16 | 4.15 | 4.19 |
| shuffled control of [support-aware + one-sided + in the critic] | 4.29 | 3.27 | 4.23 |
| sharpened baseline for [support-aware + one-sided + in the critic] | 4.23 | 4.21 | 4.15 |
| support-aware width, calibrated part read at shuffled states | 5.05 | 4.80 | 5.18 |
| shuffled control of [support-aware width] | 4.83 | 3.89 | 4.69 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 4.98 | 4.71 | 4.89 |

Average true value of an episode starting at a decision cell (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 4.52 | 4.37 | 4.32 |
| support-aware + one-sided + in the critic | 4.54 | 4.43 | 4.68 |
| shuffled control of [support-aware + one-sided + in the critic] | 4.50 | 4.40 | 4.51 |
| sharpened baseline for [support-aware + one-sided + in the critic] | 4.60 | 4.38 | 4.45 |
| support-aware width, calibrated part read at shuffled states | 5.54 | 4.49 | 5.51 |
| shuffled control of [support-aware width] | 5.17 | 4.45 | 5.28 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 5.31 | 4.47 | 5.10 |

G1 (the gap misleads the baseline): 3 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| support-aware + one-sided + in the critic | PASS 3/3 | FAIL 1/3 | FAIL 1/3 | FAIL 0/3 |
| support-aware width, calibrated part read at shuffled states | PASS 3/3 | FAIL 0/3 | PASS 3/3 | PASS 2/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 3.12% | 1.09% | 0.00 | 0.48 | 1.45 | 1.48 | -0.28 | 92.07% | 100.00% |
| baseline | targeted | 88.02% | 1.10% | 9.15 | 0.45 | 1.35 | 1.34 | -0.10 | 92.10% | 100.00% |
| baseline | placebo | 71.53% | 1.21% | 0.00 | 0.47 | 1.41 | 1.32 | -0.08 | 91.68% | 100.00% |
| support-aware + one-sided + in the critic | complete | 3.12% | 1.09% | 0.00 | 0.71 | 0.00 | 0.00 | -0.66 | 91.69% | 100.00% |
| support-aware + one-sided + in the critic | targeted | 88.02% | 1.10% | 9.15 | 62.90 | 184.84 | 158.18 | -0.65 | 90.01% | 100.00% |
| support-aware + one-sided + in the critic | placebo | 71.53% | 1.21% | 0.00 | 0.96 | 0.02 | 0.01 | -0.63 | 91.56% | 100.00% |
| support-aware width, calibrated part read at shuffled states | complete | 3.12% | 1.09% | 0.00 | 0.46 | 1.37 | 1.37 | -0.55 | 92.39% | 100.00% |
| support-aware width, calibrated part read at shuffled states | targeted | 88.02% | 1.10% | 9.15 | 65.02 | 195.07 | 195.07 | -0.88 | 92.17% | 100.00% |
| support-aware width, calibrated part read at shuffled states | placebo | 71.53% | 1.21% | 0.00 | 0.44 | 1.32 | 1.32 | -0.56 | 92.14% | 100.00% |
