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
| support penalty only, no calibrated width | 0.45% | 2.56% | 0.21% |
| shuffled control of [support penalty only, no calibrated width] | 1.29% | 6.88% | 0.62% |
| sharpened baseline for [support penalty only, no calibrated width] | 0.51% | 5.60% | 0.46% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 0.36% | 4.29% | 0.37% |
| support-aware width | 0.00% | 0.00% | 0.00% |
| shuffled control of [support-aware width] | 0.56% | 3.47% | 0.33% |
| sharpened baseline for [support-aware width] | 0.00% | 2.14% | 0.00% |
| support penalty only, no calibrated width | 0.32% | 1.69% | 0.19% |
| shuffled control of [support penalty only, no calibrated width] | 0.84% | 5.52% | 0.54% |
| sharpened baseline for [support penalty only, no calibrated width] | 0.36% | 4.30% | 0.38% |

Average true value of an episode, over all start cells (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 5.08 | 5.04 | 4.99 |
| support-aware width | 5.90 | 5.95 | 5.91 |
| shuffled control of [support-aware width] | 5.29 | 5.06 | 5.56 |
| sharpened baseline for [support-aware width] | 5.77 | 5.76 | 5.74 |
| support penalty only, no calibrated width | 5.10 | 5.11 | 5.06 |
| shuffled control of [support penalty only, no calibrated width] | 4.30 | 3.77 | 5.29 |
| sharpened baseline for [support penalty only, no calibrated width] | 5.08 | 5.06 | 4.99 |

Average true value of an episode starting at a decision cell (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 5.35 | 4.93 | 5.44 |
| support-aware width | 6.24 | 6.32 | 6.30 |
| shuffled control of [support-aware width] | 5.60 | 5.06 | 5.76 |
| sharpened baseline for [support-aware width] | 6.10 | 5.90 | 6.19 |
| support penalty only, no calibrated width | 5.37 | 5.22 | 5.50 |
| shuffled control of [support penalty only, no calibrated width] | 4.55 | 3.38 | 5.68 |
| sharpened baseline for [support penalty only, no calibrated width] | 5.35 | 4.94 | 5.43 |

G1 (the gap misleads the baseline): 3 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| support-aware width | PASS 3/3 | PASS 3/3 | PASS 3/3 | FAIL 1/3 |
| support penalty only, no calibrated width | PASS 3/3 | PASS 3/3 | PASS 2/3 | PASS 2/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 2.26% | 1.06% | 0.02 | 0.29 | 0.87 | 1.81 | 0.47 | 90.08% | 100.00% |
| baseline | targeted | 89.19% | 1.12% | 11.67 | 0.52 | 1.55 | 1.37 | 9.99 | 90.56% | 100.00% |
| baseline | placebo | 76.78% | 1.03% | 0.00 | 0.38 | 1.14 | 1.83 | 0.56 | 90.69% | 100.00% |
| support-aware width | complete | 2.26% | 1.06% | 0.02 | 0.39 | 1.17 | 1.75 | 0.52 | 90.36% | 100.00% |
| support-aware width | targeted | 89.19% | 1.12% | 11.67 | 82.44 | 247.32 | 1.27 | 10.48 | 90.97% | 100.00% |
| support-aware width | placebo | 76.78% | 1.03% | 0.00 | 0.35 | 1.04 | 1.70 | 0.55 | 90.62% | 100.00% |
| support penalty only, no calibrated width | complete | 2.26% | 1.06% | 0.02 | 0.12 | 0.35 | 0.19 | 0.46 | 0.78% | 100.00% |
| support penalty only, no calibrated width | targeted | 89.19% | 1.12% | 11.67 | 87.53 | 262.60 | 0.00 | 9.99 | 0.86% | 100.00% |
| support penalty only, no calibrated width | placebo | 76.78% | 1.03% | 0.00 | 0.00 | 0.00 | 0.00 | 0.51 | 0.86% | 100.00% |

## bottleneck_r2_exit25_one_hot

Primary outcome, X: episodes from B reaching the goal over the unseen bridge (a cautious method lowers it).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 92.13% | 12.66% | 92.68% |
| support-aware width | 100.00% | 0.65% | 88.89% |
| shuffled control of [support-aware width] | 100.00% | 0.00% | 88.89% |
| sharpened baseline for [support-aware width] | 98.53% | 11.91% | 98.03% |
| support penalty only, no calibrated width | 91.37% | 0.67% | 87.02% |
| shuffled control of [support penalty only, no calibrated width] | 94.64% | 0.00% | 84.84% |
| sharpened baseline for [support penalty only, no calibrated width] | 92.15% | 12.65% | 92.67% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 78.96% | 11.34% | 83.07% |
| support-aware width | 100.00% | 0.65% | 88.89% |
| shuffled control of [support-aware width] | 97.20% | 0.00% | 88.89% |
| sharpened baseline for [support-aware width] | 94.54% | 11.61% | 96.13% |
| support penalty only, no calibrated width | 78.30% | 0.61% | 82.89% |
| shuffled control of [support penalty only, no calibrated width] | 82.09% | 0.00% | 80.39% |
| sharpened baseline for [support penalty only, no calibrated width] | 78.97% | 11.35% | 83.06% |

Average true value of an episode, over all start cells (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 3.95 | 3.37 | 4.04 |
| support-aware width | 5.17 | 3.89 | 5.05 |
| shuffled control of [support-aware width] | 5.00 | 2.15 | 4.75 |
| sharpened baseline for [support-aware width] | 4.96 | 3.92 | 5.00 |
| support penalty only, no calibrated width | 3.93 | 3.29 | 4.02 |
| shuffled control of [support penalty only, no calibrated width] | 3.66 | 0.80 | 3.73 |
| sharpened baseline for [support penalty only, no calibrated width] | 3.96 | 3.37 | 4.04 |

Average true value of an episode starting at a decision cell (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 4.29 | 2.62 | 4.43 |
| support-aware width | 5.54 | 2.52 | 5.20 |
| shuffled control of [support-aware width] | 5.36 | 2.49 | 5.18 |
| sharpened baseline for [support-aware width] | 5.36 | 2.81 | 5.36 |
| support penalty only, no calibrated width | 4.28 | 2.35 | 4.37 |
| shuffled control of [support penalty only, no calibrated width] | 4.08 | 2.26 | 4.37 |
| sharpened baseline for [support penalty only, no calibrated width] | 4.30 | 2.62 | 4.43 |

G1 (the gap misleads the baseline): 3 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| support-aware width | PASS 3/3 | FAIL 0/3 | PASS 2/3 | FAIL 1/3 |
| support penalty only, no calibrated width | PASS 3/3 | FAIL 0/3 | PASS 2/3 | FAIL 0/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 0.87% | 0.85% | 0.00 | 0.48 | 1.43 | 1.43 | -0.19 | 90.95% | 100.00% |
| baseline | targeted | 94.10% | 1.02% | 9.27 | 0.46 | 1.39 | 1.39 | -1.78 | 91.28% | 100.00% |
| baseline | placebo | 73.61% | 0.73% | 0.00 | 0.47 | 1.41 | 1.41 | -0.34 | 91.67% | 100.00% |
| support-aware width | complete | 0.87% | 0.85% | 0.00 | 0.45 | 1.34 | 1.34 | -0.43 | 91.56% | 100.00% |
| support-aware width | targeted | 94.10% | 1.02% | 9.27 | 66.01 | 198.03 | 198.03 | -2.51 | 91.06% | 100.00% |
| support-aware width | placebo | 73.61% | 0.73% | 0.00 | 0.43 | 1.30 | 1.30 | -0.51 | 91.12% | 100.00% |
| support penalty only, no calibrated width | complete | 0.87% | 0.85% | 0.00 | 0.00 | 0.00 | 0.00 | -0.19 | 0.70% | 100.00% |
| support penalty only, no calibrated width | targeted | 94.10% | 1.02% | 9.27 | 69.52 | 208.56 | 208.56 | -1.78 | 0.82% | 100.00% |
| support penalty only, no calibrated width | placebo | 73.61% | 0.73% | 0.00 | 0.00 | 0.00 | 0.00 | -0.35 | 0.61% | 100.00% |

## bottleneck_r2_exit35_one_hot

Primary outcome, X: episodes from B reaching the goal over the unseen bridge (a cautious method lowers it).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 86.28% | 8.44% | 91.37% |
| support-aware width | 100.00% | 0.00% | 100.00% |
| shuffled control of [support-aware width] | 99.96% | 0.00% | 100.00% |
| sharpened baseline for [support-aware width] | 96.33% | 3.37% | 99.36% |
| support penalty only, no calibrated width | 88.53% | 1.41% | 93.18% |
| shuffled control of [support penalty only, no calibrated width] | 80.98% | 1.83% | 93.63% |
| sharpened baseline for [support penalty only, no calibrated width] | 86.23% | 8.46% | 91.46% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 77.53% | 7.43% | 80.82% |
| support-aware width | 99.37% | 0.00% | 100.00% |
| shuffled control of [support-aware width] | 97.22% | 0.00% | 100.00% |
| sharpened baseline for [support-aware width] | 92.78% | 3.05% | 98.57% |
| support penalty only, no calibrated width | 77.15% | 1.27% | 86.95% |
| shuffled control of [support penalty only, no calibrated width] | 69.70% | 2.05% | 87.24% |
| sharpened baseline for [support penalty only, no calibrated width] | 77.46% | 7.44% | 80.92% |

Average true value of an episode, over all start cells (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 3.94 | 3.66 | 4.01 |
| support-aware width | 5.17 | 4.32 | 5.12 |
| shuffled control of [support-aware width] | 5.02 | 3.67 | 4.87 |
| sharpened baseline for [support-aware width] | 4.91 | 4.25 | 4.99 |
| support penalty only, no calibrated width | 3.93 | 3.66 | 4.04 |
| shuffled control of [support penalty only, no calibrated width] | 3.75 | 1.27 | 3.91 |
| sharpened baseline for [support penalty only, no calibrated width] | 3.93 | 3.66 | 4.03 |

Average true value of an episode starting at a decision cell (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 4.36 | 3.43 | 4.46 |
| support-aware width | 5.52 | 3.49 | 5.52 |
| shuffled control of [support-aware width] | 5.38 | 3.49 | 5.44 |
| sharpened baseline for [support-aware width] | 5.28 | 3.52 | 5.43 |
| support penalty only, no calibrated width | 4.35 | 3.41 | 4.56 |
| shuffled control of [support penalty only, no calibrated width] | 4.18 | 3.32 | 4.61 |
| sharpened baseline for [support penalty only, no calibrated width] | 4.35 | 3.43 | 4.48 |

G1 (the gap misleads the baseline): 3 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| support-aware width | PASS 3/3 | FAIL 0/3 | PASS 3/3 | FAIL 1/3 |
| support penalty only, no calibrated width | PASS 3/3 | PASS 2/3 | PASS 3/3 | FAIL 1/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 0.87% | 0.85% | 0.00 | 0.49 | 1.46 | 1.46 | -0.25 | 91.24% | 100.00% |
| baseline | targeted | 94.10% | 1.02% | 9.27 | 0.49 | 1.47 | 1.47 | -0.71 | 91.22% | 100.00% |
| baseline | placebo | 73.61% | 0.73% | 0.00 | 0.47 | 1.41 | 1.41 | -0.26 | 90.90% | 100.00% |
| support-aware width | complete | 0.87% | 0.85% | 0.00 | 0.42 | 1.27 | 1.27 | -0.40 | 91.13% | 100.00% |
| support-aware width | targeted | 94.10% | 1.02% | 9.27 | 65.77 | 197.30 | 197.30 | -1.56 | 91.38% | 100.00% |
| support-aware width | placebo | 73.61% | 0.73% | 0.00 | 0.42 | 1.25 | 1.25 | -0.43 | 90.49% | 100.00% |
| support penalty only, no calibrated width | complete | 0.87% | 0.85% | 0.00 | 0.00 | 0.00 | 0.00 | -0.23 | 0.69% | 100.00% |
| support penalty only, no calibrated width | targeted | 94.10% | 1.02% | 9.27 | 69.52 | 208.56 | 208.56 | -0.72 | 0.82% | 100.00% |
| support penalty only, no calibrated width | placebo | 73.61% | 0.73% | 0.00 | 0.00 | 0.00 | 0.00 | -0.27 | 0.62% | 100.00% |

## bottleneck_r2_exit45_one_hot

Primary outcome, X: episodes from B reaching the goal over the unseen bridge (a cautious method lowers it).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 71.28% | 4.76% | 82.55% |
| support-aware width | 82.51% | 0.00% | 88.75% |
| shuffled control of [support-aware width] | 75.06% | 0.00% | 91.61% |
| sharpened baseline for [support-aware width] | 78.22% | 3.70% | 92.96% |
| support penalty only, no calibrated width | 70.19% | 0.00% | 83.04% |
| shuffled control of [support penalty only, no calibrated width] | 57.59% | 0.00% | 75.57% |
| sharpened baseline for [support penalty only, no calibrated width] | 71.25% | 4.76% | 82.62% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 64.15% | 5.11% | 76.35% |
| support-aware width | 80.97% | 0.00% | 88.75% |
| shuffled control of [support-aware width] | 71.46% | 0.00% | 90.72% |
| sharpened baseline for [support-aware width] | 77.48% | 3.70% | 92.25% |
| support penalty only, no calibrated width | 62.65% | 0.00% | 78.27% |
| shuffled control of [support penalty only, no calibrated width] | 50.22% | 0.00% | 70.51% |
| sharpened baseline for [support penalty only, no calibrated width] | 63.90% | 5.11% | 76.77% |

Average true value of an episode, over all start cells (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 4.10 | 4.05 | 4.07 |
| support-aware width | 5.12 | 4.82 | 5.15 |
| shuffled control of [support-aware width] | 4.98 | 3.93 | 4.98 |
| sharpened baseline for [support-aware width] | 4.96 | 4.73 | 5.04 |
| support penalty only, no calibrated width | 4.11 | 4.04 | 4.13 |
| shuffled control of [support penalty only, no calibrated width] | 3.93 | 0.78 | 3.84 |
| sharpened baseline for [support penalty only, no calibrated width] | 4.09 | 4.04 | 4.09 |

Average true value of an episode starting at a decision cell (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 4.45 | 4.37 | 4.51 |
| support-aware width | 5.33 | 4.50 | 5.43 |
| shuffled control of [support-aware width] | 5.12 | 4.50 | 5.44 |
| sharpened baseline for [support-aware width] | 5.19 | 4.51 | 5.41 |
| support penalty only, no calibrated width | 4.44 | 4.37 | 4.60 |
| shuffled control of [support penalty only, no calibrated width] | 4.27 | 4.26 | 4.53 |
| sharpened baseline for [support penalty only, no calibrated width] | 4.43 | 4.38 | 4.54 |

G1 (the gap misleads the baseline): 3 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| support-aware width | PASS 3/3 | FAIL 0/3 | PASS 2/3 | PASS 2/3 |
| support penalty only, no calibrated width | PASS 3/3 | FAIL 0/3 | PASS 2/3 | FAIL 0/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 0.87% | 0.85% | 0.00 | 0.47 | 1.40 | 1.39 | -0.14 | 91.16% | 100.00% |
| baseline | targeted | 94.10% | 1.02% | 9.27 | 0.50 | 1.50 | 1.27 | 0.31 | 91.52% | 100.00% |
| baseline | placebo | 73.61% | 0.73% | 0.00 | 0.49 | 1.47 | 1.54 | -0.08 | 91.16% | 100.00% |
| support-aware width | complete | 0.87% | 0.85% | 0.00 | 0.44 | 1.33 | 1.33 | -0.42 | 91.41% | 100.00% |
| support-aware width | targeted | 94.10% | 1.02% | 9.27 | 66.11 | 198.33 | 198.33 | -0.67 | 91.13% | 100.00% |
| support-aware width | placebo | 73.61% | 0.73% | 0.00 | 0.42 | 1.27 | 1.27 | -0.42 | 91.55% | 100.00% |
| support penalty only, no calibrated width | complete | 0.87% | 0.85% | 0.00 | 0.00 | 0.00 | 0.00 | -0.16 | 0.67% | 100.00% |
| support penalty only, no calibrated width | targeted | 94.10% | 1.02% | 9.27 | 69.52 | 208.56 | 97.66 | 0.36 | 0.84% | 100.00% |
| support penalty only, no calibrated width | placebo | 73.61% | 0.73% | 0.00 | 0.00 | 0.00 | 0.00 | -0.16 | 0.61% | 100.00% |
