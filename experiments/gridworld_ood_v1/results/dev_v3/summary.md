# Study summary

Every entry is a mean over the 9 runs of its setting, method and data arm (6 dataset seeds and 3 training seeds in this folder). Written by `summarize_study.py` from `records.json` and `gates.json`.

## honeypot

Primary outcome, H: episodes ending in hazard 34 (lower is better).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 0.50% | 5.70% | 0.45% |
| BCA as it was | 0.00% | 2.08% | 0.00% |
| shuffled control of [BCA as it was] | 0.00% | 1.98% | 0.34% |
| sharpened baseline for [BCA as it was] | 0.00% | 2.44% | 0.00% |
| support-aware width | 0.00% | 0.00% | 0.00% |
| shuffled control of [support-aware width] | 0.65% | 3.63% | 0.33% |
| sharpened baseline for [support-aware width] | 0.00% | 2.48% | 0.00% |
| support-aware + one-sided | 0.44% | 2.56% | 0.21% |
| shuffled control of [support-aware + one-sided] | 1.08% | 3.61% | 0.71% |
| sharpened baseline for [support-aware + one-sided] | 0.50% | 5.00% | 0.46% |
| support-aware + one-sided + in the critic | 0.21% | 1.09% | 0.32% |
| shuffled control of [support-aware + one-sided + in the critic] | 0.46% | 8.41% | 0.35% |
| sharpened baseline for [support-aware + one-sided + in the critic] | 0.24% | 4.10% | 0.41% |
| one-sided only | 0.50% | 5.46% | 0.43% |
| shuffled control of [one-sided only] | 0.28% | 4.05% | 0.41% |
| sharpened baseline for [one-sided only] | 0.49% | 5.52% | 0.43% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 0.36% | 4.29% | 0.37% |
| BCA as it was | 0.00% | 1.73% | 0.00% |
| shuffled control of [BCA as it was] | 0.00% | 1.78% | 0.31% |
| sharpened baseline for [BCA as it was] | 0.00% | 2.10% | 0.00% |
| support-aware width | 0.00% | 0.00% | 0.00% |
| shuffled control of [support-aware width] | 0.56% | 3.47% | 0.33% |
| sharpened baseline for [support-aware width] | 0.00% | 2.14% | 0.00% |
| support-aware + one-sided | 0.33% | 1.82% | 0.20% |
| shuffled control of [support-aware + one-sided] | 0.67% | 2.42% | 0.63% |
| sharpened baseline for [support-aware + one-sided] | 0.36% | 3.97% | 0.37% |
| support-aware + one-sided + in the critic | 0.15% | 1.03% | 0.32% |
| shuffled control of [support-aware + one-sided + in the critic] | 0.35% | 5.44% | 0.34% |
| sharpened baseline for [support-aware + one-sided + in the critic] | 0.17% | 3.46% | 0.35% |
| one-sided only | 0.35% | 4.17% | 0.36% |
| shuffled control of [one-sided only] | 0.18% | 3.39% | 0.36% |
| sharpened baseline for [one-sided only] | 0.35% | 4.28% | 0.36% |

G1 (the gap misleads the baseline): 3 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| BCA as it was | PASS 3/3 | FAIL 1/3 | PASS 3/3 | FAIL 0/3 |
| support-aware width | PASS 3/3 | PASS 3/3 | PASS 3/3 | FAIL 1/3 |
| support-aware + one-sided | PASS 3/3 | PASS 2/3 | PASS 2/3 | FAIL 1/3 |
| support-aware + one-sided + in the critic | PASS 3/3 | PASS 3/3 | PASS 3/3 | FAIL 1/3 |
| one-sided only | PASS 3/3 | FAIL 0/3 | PASS 2/3 | PASS 2/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 2.26% | 1.06% | 0.02 | 0.29 | 0.87 | 1.81 | 0.47 | 90.08% | 100.00% |
| baseline | targeted | 89.19% | 1.12% | 11.67 | 0.52 | 1.55 | 1.37 | 9.99 | 90.56% | 100.00% |
| baseline | placebo | 76.78% | 1.03% | 0.00 | 0.38 | 1.14 | 1.83 | 0.56 | 90.69% | 100.00% |
| BCA as it was | complete | 2.26% | 1.06% | 0.02 | 0.27 | 0.82 | 1.71 | 0.49 | 90.46% | 100.00% |
| BCA as it was | targeted | 89.19% | 1.12% | 11.67 | 0.48 | 1.43 | 1.26 | 10.47 | 91.00% | 100.00% |
| BCA as it was | placebo | 76.78% | 1.03% | 0.00 | 0.35 | 1.06 | 1.69 | 0.52 | 90.61% | 100.00% |
| support-aware width | complete | 2.26% | 1.06% | 0.02 | 0.39 | 1.17 | 1.75 | 0.52 | 90.36% | 100.00% |
| support-aware width | targeted | 89.19% | 1.12% | 11.67 | 82.44 | 247.32 | 1.27 | 10.48 | 90.97% | 100.00% |
| support-aware width | placebo | 76.78% | 1.03% | 0.00 | 0.35 | 1.04 | 1.70 | 0.55 | 90.62% | 100.00% |
| support-aware + one-sided | complete | 2.26% | 1.06% | 0.02 | 0.40 | 0.28 | 0.16 | 0.48 | 90.37% | 100.00% |
| support-aware + one-sided | targeted | 89.19% | 1.12% | 11.67 | 81.96 | 243.37 | 0.00 | 10.02 | 91.14% | 100.00% |
| support-aware + one-sided | placebo | 76.78% | 1.03% | 0.00 | 0.36 | 0.00 | 0.00 | 0.52 | 91.02% | 100.00% |
| support-aware + one-sided + in the critic | complete | 2.26% | 1.06% | 0.02 | 0.99 | 0.25 | 0.18 | 0.37 | 87.40% | 100.00% |
| support-aware + one-sided + in the critic | targeted | 89.19% | 1.12% | 11.67 | 76.23 | 223.84 | 0.04 | 8.99 | 86.92% | 100.00% |
| support-aware + one-sided + in the critic | placebo | 76.78% | 1.03% | 0.00 | 1.09 | 0.02 | 0.09 | 0.29 | 86.52% | 100.00% |
| one-sided only | complete | 2.26% | 1.06% | 0.02 | 0.29 | 0.00 | 0.00 | 0.47 | 90.16% | 100.00% |
| one-sided only | targeted | 89.19% | 1.12% | 11.67 | 0.52 | 0.00 | 0.00 | 10.01 | 90.71% | 100.00% |
| one-sided only | placebo | 76.78% | 1.03% | 0.00 | 0.38 | 0.00 | 0.00 | 0.54 | 90.95% | 100.00% |

## bottleneck_r2_exit25_one_hot

Primary outcome, X: episodes from B reaching the goal over the unseen bridge (a cautious method lowers it).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 92.13% | 12.66% | 92.68% |
| BCA as it was | 100.00% | 8.22% | 88.89% |
| shuffled control of [BCA as it was] | 100.00% | 8.32% | 88.89% |
| sharpened baseline for [BCA as it was] | 98.53% | 11.78% | 98.11% |
| support-aware width | 100.00% | 0.65% | 88.89% |
| shuffled control of [support-aware width] | 100.00% | 0.00% | 88.89% |
| sharpened baseline for [support-aware width] | 98.53% | 11.91% | 98.03% |
| support-aware + one-sided | 92.91% | 0.00% | 86.95% |
| shuffled control of [support-aware + one-sided] | 95.00% | 0.00% | 85.41% |
| sharpened baseline for [support-aware + one-sided] | 92.23% | 12.67% | 92.77% |
| support-aware + one-sided + in the critic | 94.36% | 0.00% | 98.29% |
| shuffled control of [support-aware + one-sided + in the critic] | 91.47% | 0.05% | 85.04% |
| sharpened baseline for [support-aware + one-sided + in the critic] | 93.13% | 12.55% | 93.06% |
| one-sided only | 93.06% | 12.71% | 83.74% |
| shuffled control of [one-sided only] | 92.41% | 9.47% | 93.62% |
| sharpened baseline for [one-sided only] | 92.21% | 12.64% | 92.78% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 78.96% | 11.34% | 83.07% |
| BCA as it was | 100.00% | 7.54% | 88.89% |
| shuffled control of [BCA as it was] | 99.32% | 7.76% | 88.89% |
| sharpened baseline for [BCA as it was] | 93.82% | 11.51% | 96.15% |
| support-aware width | 100.00% | 0.65% | 88.89% |
| shuffled control of [support-aware width] | 97.20% | 0.00% | 88.89% |
| sharpened baseline for [support-aware width] | 94.54% | 11.61% | 96.13% |
| support-aware + one-sided | 80.67% | 0.00% | 81.88% |
| shuffled control of [support-aware + one-sided] | 84.42% | 0.00% | 80.85% |
| sharpened baseline for [support-aware + one-sided] | 79.19% | 11.38% | 83.19% |
| support-aware + one-sided + in the critic | 84.63% | 0.00% | 96.39% |
| shuffled control of [support-aware + one-sided + in the critic] | 83.48% | 0.69% | 82.36% |
| sharpened baseline for [support-aware + one-sided + in the critic] | 80.77% | 11.36% | 84.49% |
| one-sided only | 79.38% | 11.63% | 76.00% |
| shuffled control of [one-sided only] | 80.16% | 7.56% | 84.87% |
| sharpened baseline for [one-sided only] | 79.63% | 11.34% | 83.23% |

G1 (the gap misleads the baseline): 3 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| BCA as it was | FAIL 0/3 | FAIL 1/3 | PASS 2/3 | FAIL 1/3 |
| support-aware width | PASS 3/3 | FAIL 0/3 | PASS 2/3 | FAIL 1/3 |
| support-aware + one-sided | PASS 3/3 | FAIL 0/3 | PASS 2/3 | PASS 2/3 |
| support-aware + one-sided + in the critic | PASS 3/3 | FAIL 1/3 | PASS 3/3 | FAIL 0/3 |
| one-sided only | FAIL 0/3 | FAIL 0/3 | PASS 2/3 | PASS 3/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 0.87% | 0.85% | 0.00 | 0.48 | 1.43 | 1.43 | -0.19 | 90.95% | 100.00% |
| baseline | targeted | 94.10% | 1.02% | 9.27 | 0.46 | 1.39 | 1.39 | -1.78 | 91.28% | 100.00% |
| baseline | placebo | 73.61% | 0.73% | 0.00 | 0.47 | 1.41 | 1.41 | -0.34 | 91.67% | 100.00% |
| BCA as it was | complete | 0.87% | 0.85% | 0.00 | 0.44 | 1.32 | 1.32 | -0.45 | 91.48% | 100.00% |
| BCA as it was | targeted | 94.10% | 1.02% | 9.27 | 0.43 | 1.28 | 1.28 | -2.46 | 91.05% | 100.00% |
| BCA as it was | placebo | 73.61% | 0.73% | 0.00 | 0.44 | 1.32 | 1.32 | -0.44 | 91.46% | 100.00% |
| support-aware width | complete | 0.87% | 0.85% | 0.00 | 0.45 | 1.34 | 1.34 | -0.43 | 91.56% | 100.00% |
| support-aware width | targeted | 94.10% | 1.02% | 9.27 | 66.01 | 198.03 | 198.03 | -2.51 | 91.06% | 100.00% |
| support-aware width | placebo | 73.61% | 0.73% | 0.00 | 0.43 | 1.30 | 1.30 | -0.51 | 91.12% | 100.00% |
| support-aware + one-sided | complete | 0.87% | 0.85% | 0.00 | 0.48 | 0.00 | 0.00 | -0.24 | 91.22% | 100.00% |
| support-aware + one-sided | targeted | 94.10% | 1.02% | 9.27 | 65.61 | 193.95 | 193.95 | -1.80 | 91.13% | 100.00% |
| support-aware + one-sided | placebo | 73.61% | 0.73% | 0.00 | 0.48 | 0.00 | 0.00 | -0.33 | 91.50% | 100.00% |
| support-aware + one-sided + in the critic | complete | 0.87% | 0.85% | 0.00 | 0.56 | 0.00 | 0.00 | -0.65 | 90.30% | 100.00% |
| support-aware + one-sided + in the critic | targeted | 94.10% | 1.02% | 9.27 | 63.87 | 188.14 | 188.14 | -3.05 | 89.22% | 100.00% |
| support-aware + one-sided + in the critic | placebo | 73.61% | 0.73% | 0.00 | 0.73 | 0.00 | 0.00 | -0.73 | 90.36% | 100.00% |
| one-sided only | complete | 0.87% | 0.85% | 0.00 | 0.49 | 0.00 | 0.00 | -0.21 | 91.05% | 100.00% |
| one-sided only | targeted | 94.10% | 1.02% | 9.27 | 0.46 | 0.00 | 0.00 | -1.81 | 91.15% | 100.00% |
| one-sided only | placebo | 73.61% | 0.73% | 0.00 | 0.48 | 0.00 | 0.00 | -0.33 | 91.48% | 100.00% |

## bottleneck_r2_exit35_one_hot

Primary outcome, X: episodes from B reaching the goal over the unseen bridge (a cautious method lowers it).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 86.28% | 8.44% | 91.37% |
| BCA as it was | 100.00% | 0.85% | 100.00% |
| shuffled control of [BCA as it was] | 100.00% | 0.82% | 100.00% |
| sharpened baseline for [BCA as it was] | 96.29% | 3.30% | 99.36% |
| support-aware width | 100.00% | 0.00% | 100.00% |
| shuffled control of [support-aware width] | 99.96% | 0.00% | 100.00% |
| sharpened baseline for [support-aware width] | 96.33% | 3.37% | 99.36% |
| support-aware + one-sided | 86.21% | 1.99% | 93.21% |
| shuffled control of [support-aware + one-sided] | 85.29% | 1.12% | 94.46% |
| sharpened baseline for [support-aware + one-sided] | 86.42% | 8.44% | 91.54% |
| support-aware + one-sided + in the critic | 89.86% | 0.00% | 94.63% |
| shuffled control of [support-aware + one-sided + in the critic] | 69.82% | 0.51% | 90.64% |
| sharpened baseline for [support-aware + one-sided + in the critic] | 86.69% | 8.11% | 92.63% |
| one-sided only | 87.14% | 8.11% | 91.83% |
| shuffled control of [one-sided only] | 89.76% | 6.50% | 92.21% |
| sharpened baseline for [one-sided only] | 86.39% | 8.37% | 91.40% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 77.53% | 7.43% | 80.82% |
| BCA as it was | 99.41% | 0.85% | 100.00% |
| shuffled control of [BCA as it was] | 100.00% | 0.82% | 100.00% |
| sharpened baseline for [BCA as it was] | 92.70% | 2.99% | 98.59% |
| support-aware width | 99.37% | 0.00% | 100.00% |
| shuffled control of [support-aware width] | 97.22% | 0.00% | 100.00% |
| sharpened baseline for [support-aware width] | 92.78% | 3.05% | 98.57% |
| support-aware + one-sided | 77.97% | 1.85% | 86.88% |
| shuffled control of [support-aware + one-sided] | 73.95% | 1.78% | 88.78% |
| sharpened baseline for [support-aware + one-sided] | 77.72% | 7.42% | 81.09% |
| support-aware + one-sided + in the critic | 79.07% | 0.00% | 92.96% |
| shuffled control of [support-aware + one-sided + in the critic] | 63.15% | 2.27% | 86.46% |
| sharpened baseline for [support-aware + one-sided + in the critic] | 78.08% | 7.19% | 82.80% |
| one-sided only | 79.11% | 7.31% | 82.54% |
| shuffled control of [one-sided only] | 78.93% | 6.21% | 83.18% |
| sharpened baseline for [one-sided only] | 77.68% | 7.37% | 80.87% |

G1 (the gap misleads the baseline): 3 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| BCA as it was | FAIL 0/3 | FAIL 0/3 | PASS 3/3 | FAIL 1/3 |
| support-aware width | PASS 3/3 | FAIL 0/3 | PASS 3/3 | FAIL 1/3 |
| support-aware + one-sided | PASS 3/3 | FAIL 1/3 | PASS 3/3 | FAIL 0/3 |
| support-aware + one-sided + in the critic | PASS 3/3 | PASS 2/3 | PASS 3/3 | FAIL 0/3 |
| one-sided only | FAIL 0/3 | FAIL 1/3 | PASS 2/3 | FAIL 1/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 0.87% | 0.85% | 0.00 | 0.49 | 1.46 | 1.46 | -0.25 | 91.24% | 100.00% |
| baseline | targeted | 94.10% | 1.02% | 9.27 | 0.49 | 1.47 | 1.47 | -0.71 | 91.22% | 100.00% |
| baseline | placebo | 73.61% | 0.73% | 0.00 | 0.47 | 1.41 | 1.41 | -0.26 | 90.90% | 100.00% |
| BCA as it was | complete | 0.87% | 0.85% | 0.00 | 0.42 | 1.27 | 1.27 | -0.40 | 90.95% | 100.00% |
| BCA as it was | targeted | 94.10% | 1.02% | 9.27 | 0.47 | 1.40 | 1.40 | -1.54 | 91.14% | 100.00% |
| BCA as it was | placebo | 73.61% | 0.73% | 0.00 | 0.42 | 1.25 | 1.25 | -0.42 | 90.58% | 100.00% |
| support-aware width | complete | 0.87% | 0.85% | 0.00 | 0.42 | 1.27 | 1.27 | -0.40 | 91.13% | 100.00% |
| support-aware width | targeted | 94.10% | 1.02% | 9.27 | 65.77 | 197.30 | 197.30 | -1.56 | 91.38% | 100.00% |
| support-aware width | placebo | 73.61% | 0.73% | 0.00 | 0.42 | 1.25 | 1.25 | -0.43 | 90.49% | 100.00% |
| support-aware + one-sided | complete | 0.87% | 0.85% | 0.00 | 0.49 | 0.00 | 0.00 | -0.26 | 91.34% | 100.00% |
| support-aware + one-sided | targeted | 94.10% | 1.02% | 9.27 | 65.53 | 193.60 | 193.60 | -0.81 | 91.35% | 100.00% |
| support-aware + one-sided | placebo | 73.61% | 0.73% | 0.00 | 0.45 | 0.00 | 0.00 | -0.27 | 90.66% | 100.00% |
| support-aware + one-sided + in the critic | complete | 0.87% | 0.85% | 0.00 | 0.56 | 0.00 | 0.00 | -0.59 | 91.13% | 100.00% |
| support-aware + one-sided + in the critic | targeted | 94.10% | 1.02% | 9.27 | 64.34 | 189.39 | 189.39 | -1.93 | 89.74% | 100.00% |
| support-aware + one-sided + in the critic | placebo | 73.61% | 0.73% | 0.00 | 0.76 | 0.00 | 0.00 | -0.68 | 89.74% | 100.00% |
| one-sided only | complete | 0.87% | 0.85% | 0.00 | 0.49 | 0.00 | 0.00 | -0.26 | 91.09% | 100.00% |
| one-sided only | targeted | 94.10% | 1.02% | 9.27 | 0.48 | 0.00 | 0.00 | -0.74 | 90.95% | 100.00% |
| one-sided only | placebo | 73.61% | 0.73% | 0.00 | 0.45 | 0.00 | 0.00 | -0.28 | 90.70% | 100.00% |

## bottleneck_r2_exit45_one_hot

Primary outcome, X: episodes from B reaching the goal over the unseen bridge (a cautious method lowers it).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 71.28% | 4.76% | 82.55% |
| BCA as it was | 90.67% | 0.00% | 99.02% |
| shuffled control of [BCA as it was] | 90.88% | 0.00% | 98.55% |
| sharpened baseline for [BCA as it was] | 78.22% | 3.71% | 93.12% |
| support-aware width | 82.51% | 0.00% | 88.75% |
| shuffled control of [support-aware width] | 75.06% | 0.00% | 91.61% |
| sharpened baseline for [support-aware width] | 78.22% | 3.70% | 92.96% |
| support-aware + one-sided | 70.08% | 0.60% | 83.04% |
| shuffled control of [support-aware + one-sided] | 65.28% | 0.00% | 73.80% |
| sharpened baseline for [support-aware + one-sided] | 71.55% | 4.76% | 82.84% |
| support-aware + one-sided + in the critic | 69.22% | 0.00% | 77.87% |
| shuffled control of [support-aware + one-sided + in the critic] | 38.84% | 0.01% | 34.63% |
| sharpened baseline for [support-aware + one-sided + in the critic] | 71.81% | 4.71% | 83.36% |
| one-sided only | 73.09% | 4.00% | 83.04% |
| shuffled control of [one-sided only] | 72.76% | 3.95% | 84.43% |
| sharpened baseline for [one-sided only] | 71.51% | 4.75% | 82.71% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 64.15% | 5.11% | 76.35% |
| BCA as it was | 88.33% | 0.00% | 99.02% |
| shuffled control of [BCA as it was] | 90.16% | 0.00% | 98.55% |
| sharpened baseline for [BCA as it was] | 77.48% | 3.71% | 92.41% |
| support-aware width | 80.97% | 0.00% | 88.75% |
| shuffled control of [support-aware width] | 71.46% | 0.00% | 90.72% |
| sharpened baseline for [support-aware width] | 77.48% | 3.70% | 92.25% |
| support-aware + one-sided | 62.46% | 0.63% | 77.97% |
| shuffled control of [support-aware + one-sided] | 57.32% | 0.00% | 68.60% |
| sharpened baseline for [support-aware + one-sided] | 64.58% | 5.11% | 77.34% |
| support-aware + one-sided + in the critic | 62.17% | 0.00% | 77.47% |
| shuffled control of [support-aware + one-sided + in the critic] | 36.03% | 0.84% | 33.77% |
| sharpened baseline for [support-aware + one-sided + in the critic] | 65.04% | 5.05% | 77.56% |
| one-sided only | 66.02% | 4.44% | 75.61% |
| shuffled control of [one-sided only] | 64.96% | 4.39% | 77.68% |
| sharpened baseline for [one-sided only] | 64.74% | 5.10% | 76.41% |

G1 (the gap misleads the baseline): 3 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| BCA as it was | FAIL 0/3 | FAIL 0/3 | PASS 3/3 | PASS 2/3 |
| support-aware width | PASS 3/3 | FAIL 0/3 | PASS 2/3 | PASS 2/3 |
| support-aware + one-sided | PASS 3/3 | FAIL 0/3 | PASS 3/3 | FAIL 0/3 |
| support-aware + one-sided + in the critic | PASS 3/3 | FAIL 1/3 | FAIL 1/3 | FAIL 1/3 |
| one-sided only | FAIL 0/3 | FAIL 0/3 | PASS 3/3 | PASS 2/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 0.87% | 0.85% | 0.00 | 0.47 | 1.40 | 1.39 | -0.14 | 91.16% | 100.00% |
| baseline | targeted | 94.10% | 1.02% | 9.27 | 0.50 | 1.50 | 1.27 | 0.31 | 91.52% | 100.00% |
| baseline | placebo | 73.61% | 0.73% | 0.00 | 0.49 | 1.47 | 1.54 | -0.08 | 91.16% | 100.00% |
| BCA as it was | complete | 0.87% | 0.85% | 0.00 | 0.44 | 1.32 | 1.32 | -0.44 | 91.36% | 100.00% |
| BCA as it was | targeted | 94.10% | 1.02% | 9.27 | 0.43 | 1.30 | 1.30 | -0.62 | 91.12% | 100.00% |
| BCA as it was | placebo | 73.61% | 0.73% | 0.00 | 0.43 | 1.28 | 1.28 | -0.43 | 91.36% | 100.00% |
| support-aware width | complete | 0.87% | 0.85% | 0.00 | 0.44 | 1.33 | 1.33 | -0.42 | 91.41% | 100.00% |
| support-aware width | targeted | 94.10% | 1.02% | 9.27 | 66.11 | 198.33 | 198.33 | -0.67 | 91.13% | 100.00% |
| support-aware width | placebo | 73.61% | 0.73% | 0.00 | 0.42 | 1.27 | 1.27 | -0.42 | 91.55% | 100.00% |
| support-aware + one-sided | complete | 0.87% | 0.85% | 0.00 | 0.47 | 0.00 | 0.00 | -0.15 | 91.32% | 100.00% |
| support-aware + one-sided | targeted | 94.10% | 1.02% | 9.27 | 65.57 | 193.70 | 121.60 | 0.29 | 91.49% | 100.00% |
| support-aware + one-sided | placebo | 73.61% | 0.73% | 0.00 | 0.50 | 0.00 | 0.00 | -0.09 | 91.32% | 100.00% |
| support-aware + one-sided + in the critic | complete | 0.87% | 0.85% | 0.00 | 0.55 | 0.00 | 0.00 | -0.61 | 90.27% | 100.00% |
| support-aware + one-sided + in the critic | targeted | 94.10% | 1.02% | 9.27 | 63.95 | 188.05 | 173.20 | -0.61 | 90.85% | 100.00% |
| support-aware + one-sided + in the critic | placebo | 73.61% | 0.73% | 0.00 | 0.72 | 0.00 | 0.00 | -0.90 | 90.34% | 100.00% |
| one-sided only | complete | 0.87% | 0.85% | 0.00 | 0.47 | 0.00 | 0.00 | -0.14 | 91.16% | 100.00% |
| one-sided only | targeted | 94.10% | 1.02% | 9.27 | 0.48 | 0.00 | 0.00 | 0.35 | 91.43% | 100.00% |
| one-sided only | placebo | 73.61% | 0.73% | 0.00 | 0.50 | 0.00 | 0.00 | -0.19 | 91.45% | 100.00% |
