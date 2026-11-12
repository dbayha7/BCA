# Study summary

Every entry is a mean over the 9 runs of its setting, method and data arm (3 dataset seeds and 3 training seeds in this folder). Written by `summarize_study.py` from `records.json` and `gates.json`.

## honeypot

Primary outcome, H: episodes ending in hazard 34 (lower is better).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 0.42% | 3.96% | 0.98% |
| support-aware + one-sided + in the critic | 0.17% | 0.72% | 0.24% |
| shuffled control of [support-aware + one-sided + in the critic] | 0.90% | 2.16% | 0.48% |
| sharpened baseline for [support-aware + one-sided + in the critic] | 0.20% | 2.92% | 0.69% |
| support-aware width, calibrated part read at shuffled states | 0.00% | 0.00% | 0.00% |
| shuffled control of [support-aware width] | 0.52% | 3.37% | 0.67% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 0.00% | 0.98% | 0.34% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 0.35% | 2.89% | 0.58% |
| support-aware + one-sided + in the critic | 0.15% | 0.65% | 0.19% |
| shuffled control of [support-aware + one-sided + in the critic] | 0.70% | 1.63% | 0.33% |
| sharpened baseline for [support-aware + one-sided + in the critic] | 0.18% | 2.23% | 0.40% |
| support-aware width, calibrated part read at shuffled states | 0.00% | 0.00% | 0.00% |
| shuffled control of [support-aware width] | 0.46% | 3.25% | 0.51% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 0.00% | 0.90% | 0.21% |

Average true value of an episode, over all start cells (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 5.09 | 4.94 | 5.05 |
| support-aware + one-sided + in the critic | 5.27 | 5.08 | 5.16 |
| shuffled control of [support-aware + one-sided + in the critic] | 5.00 | 5.07 | 5.69 |
| sharpened baseline for [support-aware + one-sided + in the critic] | 5.25 | 5.18 | 5.23 |
| support-aware width, calibrated part read at shuffled states | 5.98 | 5.93 | 5.99 |
| shuffled control of [support-aware width] | 5.58 | 4.69 | 5.90 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 5.82 | 5.75 | 5.79 |

Average true value of an episode starting at a decision cell (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 5.35 | 5.02 | 5.30 |
| support-aware + one-sided + in the critic | 5.55 | 5.52 | 5.64 |
| shuffled control of [support-aware + one-sided + in the critic] | 5.23 | 5.57 | 5.99 |
| sharpened baseline for [support-aware + one-sided + in the critic] | 5.54 | 5.34 | 5.50 |
| support-aware width, calibrated part read at shuffled states | 6.28 | 6.25 | 6.32 |
| shuffled control of [support-aware width] | 5.82 | 5.60 | 6.20 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 6.15 | 6.07 | 6.12 |

G1 (the gap misleads the baseline): 2 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| support-aware + one-sided + in the critic | PASS 3/3 | PASS 2/3 | PASS 3/3 | FAIL 1/3 |
| support-aware width, calibrated part read at shuffled states | PASS 3/3 | PASS 2/3 | PASS 3/3 | PASS 3/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 1.69% | 0.95% | 0.03 | 0.39 | 1.17 | 2.10 | 0.88 | 92.22% | 100.00% |
| baseline | targeted | 89.76% | 1.08% | 13.51 | 0.63 | 1.88 | 1.70 | 9.86 | 92.07% | 100.00% |
| baseline | placebo | 77.17% | 1.10% | 0.02 | 0.46 | 1.39 | 2.36 | -0.08 | 92.72% | 100.00% |
| support-aware + one-sided + in the critic | complete | 1.69% | 0.95% | 0.03 | 1.27 | 0.46 | 0.29 | 0.59 | 88.38% | 100.00% |
| support-aware + one-sided + in the critic | targeted | 89.76% | 1.08% | 13.51 | 86.55 | 254.08 | 0.00 | 8.74 | 89.16% | 100.00% |
| support-aware + one-sided + in the critic | placebo | 77.17% | 1.10% | 0.02 | 1.50 | 0.20 | 0.00 | -0.38 | 88.25% | 100.00% |
| support-aware width, calibrated part read at shuffled states | complete | 1.69% | 0.95% | 0.03 | 0.63 | 1.88 | 2.04 | 0.91 | 92.08% | 100.00% |
| support-aware width, calibrated part read at shuffled states | targeted | 89.76% | 1.08% | 13.51 | 94.27 | 282.81 | 1.50 | 10.40 | 92.00% | 100.00% |
| support-aware width, calibrated part read at shuffled states | placebo | 77.17% | 1.10% | 0.02 | 0.57 | 1.71 | 2.12 | -0.14 | 92.76% | 100.00% |

## bottleneck_r2_exit25_one_hot

Primary outcome, X: episodes from B reaching the goal over the unseen bridge (a cautious method lowers it).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 97.74% | 2.16% | 97.91% |
| support-aware + one-sided + in the critic | 96.99% | 1.45% | 98.46% |
| shuffled control of [support-aware + one-sided + in the critic] | 90.92% | 1.58% | 87.87% |
| sharpened baseline for [support-aware + one-sided + in the critic] | 97.88% | 2.10% | 98.02% |
| support-aware width, calibrated part read at shuffled states | 100.00% | 0.00% | 100.00% |
| shuffled control of [support-aware width] | 100.00% | 0.00% | 99.33% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 100.00% | 0.88% | 100.00% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 86.48% | 2.03% | 85.80% |
| support-aware + one-sided + in the critic | 90.47% | 1.44% | 94.36% |
| shuffled control of [support-aware + one-sided + in the critic] | 82.58% | 1.58% | 80.56% |
| sharpened baseline for [support-aware + one-sided + in the critic] | 88.13% | 1.98% | 87.31% |
| support-aware width, calibrated part read at shuffled states | 100.00% | 0.00% | 98.71% |
| shuffled control of [support-aware width] | 98.47% | 0.00% | 98.39% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 98.24% | 0.81% | 96.42% |

Average true value of an episode, over all start cells (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 4.07 | 2.61 | 4.11 |
| support-aware + one-sided + in the critic | 4.19 | 2.58 | 4.05 |
| shuffled control of [support-aware + one-sided + in the critic] | 4.13 | 1.88 | 4.06 |
| sharpened baseline for [support-aware + one-sided + in the critic] | 4.19 | 2.71 | 4.23 |
| support-aware width, calibrated part read at shuffled states | 5.00 | 3.27 | 5.07 |
| shuffled control of [support-aware width] | 4.78 | 2.27 | 4.62 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 4.88 | 3.10 | 5.00 |

Average true value of an episode starting at a decision cell (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 4.54 | 2.47 | 4.47 |
| support-aware + one-sided + in the critic | 4.69 | 2.46 | 4.76 |
| shuffled control of [support-aware + one-sided + in the critic] | 4.60 | 2.53 | 4.60 |
| sharpened baseline for [support-aware + one-sided + in the critic] | 4.70 | 2.48 | 4.61 |
| support-aware width, calibrated part read at shuffled states | 5.53 | 2.50 | 5.50 |
| shuffled control of [support-aware width] | 5.39 | 2.50 | 5.39 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 5.41 | 2.46 | 5.38 |

G1 (the gap misleads the baseline): 3 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| support-aware + one-sided + in the critic | PASS 3/3 | FAIL 1/3 | PASS 2/3 | FAIL 1/3 |
| support-aware width, calibrated part read at shuffled states | PASS 3/3 | FAIL 0/3 | PASS 2/3 | FAIL 1/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 1.39% | 0.92% | 0.00 | 0.49 | 1.46 | 1.46 | -0.12 | 90.50% | 100.00% |
| baseline | targeted | 86.11% | 1.03% | 12.50 | 0.48 | 1.44 | 1.42 | -0.91 | 90.30% | 100.00% |
| baseline | placebo | 76.56% | 0.97% | 0.00 | 0.48 | 1.45 | 1.45 | -0.01 | 90.71% | 100.00% |
| support-aware + one-sided + in the critic | complete | 1.39% | 0.92% | 0.00 | 0.61 | 0.00 | 0.00 | -0.52 | 89.66% | 100.00% |
| support-aware + one-sided + in the critic | targeted | 86.11% | 1.03% | 12.50 | 94.87 | 280.25 | 226.63 | -1.56 | 90.70% | 100.00% |
| support-aware + one-sided + in the critic | placebo | 76.56% | 0.97% | 0.00 | 0.64 | 0.00 | 0.00 | -0.49 | 88.99% | 100.00% |
| support-aware width, calibrated part read at shuffled states | complete | 1.39% | 0.92% | 0.00 | 0.41 | 1.22 | 1.22 | -0.34 | 90.74% | 100.00% |
| support-aware width, calibrated part read at shuffled states | targeted | 86.11% | 1.03% | 12.50 | 96.79 | 290.38 | 235.96 | 0.11 | 90.56% | 100.00% |
| support-aware width, calibrated part read at shuffled states | placebo | 76.56% | 0.97% | 0.00 | 0.45 | 1.35 | 1.35 | -0.31 | 90.60% | 100.00% |

## bottleneck_r2_exit35_one_hot

Primary outcome, X: episodes from B reaching the goal over the unseen bridge (a cautious method lowers it).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 93.70% | 5.78% | 96.10% |
| support-aware + one-sided + in the critic | 94.28% | 0.00% | 97.09% |
| shuffled control of [support-aware + one-sided + in the critic] | 78.33% | 0.11% | 76.13% |
| sharpened baseline for [support-aware + one-sided + in the critic] | 94.03% | 5.53% | 96.35% |
| support-aware width, calibrated part read at shuffled states | 100.00% | 0.00% | 100.00% |
| shuffled control of [support-aware width] | 98.42% | 0.00% | 98.56% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 99.23% | 3.90% | 99.40% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 86.11% | 4.59% | 88.36% |
| support-aware + one-sided + in the critic | 87.69% | 0.00% | 91.54% |
| shuffled control of [support-aware + one-sided + in the critic] | 72.12% | 1.29% | 70.96% |
| sharpened baseline for [support-aware + one-sided + in the critic] | 87.23% | 4.44% | 88.72% |
| support-aware width, calibrated part read at shuffled states | 100.00% | 0.00% | 99.29% |
| shuffled control of [support-aware width] | 97.28% | 0.00% | 97.96% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 97.97% | 2.35% | 98.01% |

Average true value of an episode, over all start cells (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 3.96 | 3.75 | 4.11 |
| support-aware + one-sided + in the critic | 4.09 | 3.79 | 4.10 |
| shuffled control of [support-aware + one-sided + in the critic] | 4.11 | 2.60 | 4.26 |
| sharpened baseline for [support-aware + one-sided + in the critic] | 4.06 | 3.85 | 4.17 |
| support-aware width, calibrated part read at shuffled states | 5.07 | 4.36 | 5.13 |
| shuffled control of [support-aware width] | 4.87 | 3.86 | 4.77 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 4.96 | 4.24 | 5.01 |

Average true value of an episode starting at a decision cell (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 4.40 | 3.43 | 4.53 |
| support-aware + one-sided + in the critic | 4.61 | 3.42 | 4.65 |
| shuffled control of [support-aware + one-sided + in the critic] | 4.48 | 3.44 | 4.60 |
| sharpened baseline for [support-aware + one-sided + in the critic] | 4.53 | 3.46 | 4.59 |
| support-aware width, calibrated part read at shuffled states | 5.48 | 3.50 | 5.50 |
| shuffled control of [support-aware width] | 5.28 | 3.49 | 5.40 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 5.45 | 3.49 | 5.41 |

G1 (the gap misleads the baseline): 3 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| support-aware + one-sided + in the critic | PASS 3/3 | FAIL 1/3 | PASS 3/3 | PASS 2/3 |
| support-aware width, calibrated part read at shuffled states | PASS 3/3 | FAIL 0/3 | PASS 3/3 | PASS 2/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 1.39% | 0.92% | 0.00 | 0.47 | 1.41 | 1.41 | 0.06 | 90.52% | 100.00% |
| baseline | targeted | 86.11% | 1.03% | 12.50 | 0.46 | 1.39 | 1.39 | -1.19 | 90.08% | 100.00% |
| baseline | placebo | 76.56% | 0.97% | 0.00 | 0.45 | 1.35 | 1.35 | 0.03 | 90.49% | 100.00% |
| support-aware + one-sided + in the critic | complete | 1.39% | 0.92% | 0.00 | 0.52 | 0.00 | 0.00 | -0.44 | 89.37% | 100.00% |
| support-aware + one-sided + in the critic | targeted | 86.11% | 1.03% | 12.50 | 86.84 | 256.83 | 256.83 | -1.37 | 89.59% | 100.00% |
| support-aware + one-sided + in the critic | placebo | 76.56% | 0.97% | 0.00 | 0.55 | 0.00 | 0.00 | -0.33 | 89.27% | 100.00% |
| support-aware width, calibrated part read at shuffled states | complete | 1.39% | 0.92% | 0.00 | 0.42 | 1.26 | 1.26 | -0.21 | 91.18% | 100.00% |
| support-aware width, calibrated part read at shuffled states | targeted | 86.11% | 1.03% | 12.50 | 89.06 | 267.19 | 267.19 | -1.79 | 90.43% | 100.00% |
| support-aware width, calibrated part read at shuffled states | placebo | 76.56% | 0.97% | 0.00 | 0.39 | 1.16 | 1.16 | -0.30 | 90.64% | 100.00% |

## bottleneck_r2_exit45_one_hot

Primary outcome, X: episodes from B reaching the goal over the unseen bridge (a cautious method lowers it).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 70.38% | 2.37% | 88.74% |
| support-aware + one-sided + in the critic | 65.60% | 0.62% | 76.22% |
| shuffled control of [support-aware + one-sided + in the critic] | 41.31% | 0.18% | 47.43% |
| sharpened baseline for [support-aware + one-sided + in the critic] | 70.96% | 2.36% | 89.10% |
| support-aware width, calibrated part read at shuffled states | 97.52% | 0.00% | 98.18% |
| shuffled control of [support-aware width] | 94.71% | 0.00% | 96.34% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 79.90% | 1.12% | 94.17% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 63.13% | 2.15% | 82.59% |
| support-aware + one-sided + in the critic | 59.65% | 0.62% | 73.29% |
| shuffled control of [support-aware + one-sided + in the critic] | 38.57% | 2.83% | 45.28% |
| sharpened baseline for [support-aware + one-sided + in the critic] | 64.26% | 2.15% | 82.90% |
| support-aware width, calibrated part read at shuffled states | 97.52% | 0.00% | 98.18% |
| shuffled control of [support-aware width] | 94.34% | 0.00% | 94.68% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 78.62% | 1.12% | 92.10% |

Average true value of an episode, over all start cells (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 4.15 | 4.11 | 4.22 |
| support-aware + one-sided + in the critic | 4.18 | 3.94 | 4.25 |
| shuffled control of [support-aware + one-sided + in the critic] | 4.31 | 2.58 | 4.29 |
| sharpened baseline for [support-aware + one-sided + in the critic] | 4.24 | 4.14 | 4.25 |
| support-aware width, calibrated part read at shuffled states | 5.14 | 3.80 | 5.18 |
| shuffled control of [support-aware width] | 4.97 | 3.21 | 4.66 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 4.95 | 4.65 | 5.04 |

Average true value of an episode starting at a decision cell (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 4.49 | 4.42 | 4.58 |
| support-aware + one-sided + in the critic | 4.53 | 4.44 | 4.56 |
| shuffled control of [support-aware + one-sided + in the critic] | 4.51 | 4.44 | 4.63 |
| sharpened baseline for [support-aware + one-sided + in the critic] | 4.58 | 4.43 | 4.60 |
| support-aware width, calibrated part read at shuffled states | 5.51 | 4.49 | 5.51 |
| shuffled control of [support-aware width] | 5.33 | 4.49 | 5.34 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 5.20 | 4.38 | 5.34 |

G1 (the gap misleads the baseline): 3 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| support-aware + one-sided + in the critic | PASS 3/3 | FAIL 1/3 | FAIL 0/3 | FAIL 1/3 |
| support-aware width, calibrated part read at shuffled states | PASS 3/3 | FAIL 0/3 | PASS 3/3 | FAIL 1/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 1.39% | 0.92% | 0.00 | 0.48 | 1.43 | 1.43 | -0.09 | 90.79% | 100.00% |
| baseline | targeted | 86.11% | 1.03% | 12.50 | 0.49 | 1.47 | 1.35 | -0.38 | 90.83% | 100.00% |
| baseline | placebo | 76.56% | 0.97% | 0.00 | 0.46 | 1.38 | 1.41 | -0.03 | 90.75% | 100.00% |
| support-aware + one-sided + in the critic | complete | 1.39% | 0.92% | 0.00 | 0.59 | 0.00 | 0.00 | -0.64 | 89.23% | 100.00% |
| support-aware + one-sided + in the critic | targeted | 86.11% | 1.03% | 12.50 | 87.59 | 258.86 | 198.47 | -0.81 | 90.36% | 100.00% |
| support-aware + one-sided + in the critic | placebo | 76.56% | 0.97% | 0.00 | 0.55 | 0.00 | 0.00 | -0.41 | 89.01% | 100.00% |
| support-aware width, calibrated part read at shuffled states | complete | 1.39% | 0.92% | 0.00 | 0.39 | 1.17 | 1.17 | -0.41 | 90.95% | 100.00% |
| support-aware width, calibrated part read at shuffled states | targeted | 86.11% | 1.03% | 12.50 | 91.60 | 274.80 | 236.42 | -1.02 | 90.95% | 100.00% |
| support-aware width, calibrated part read at shuffled states | placebo | 76.56% | 0.97% | 0.00 | 0.39 | 1.16 | 1.16 | -0.32 | 90.71% | 100.00% |
