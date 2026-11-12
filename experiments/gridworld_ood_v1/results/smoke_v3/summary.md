# Study summary

Every entry is a mean over the 2 runs of its setting, method and data arm (1 dataset seeds and 2 training seeds in this folder). Written by `summarize_study.py` from `records.json` and `gates.json`.

## honeypot

Primary outcome, H: episodes ending in hazard 34 (lower is better).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 0.83% | 4.05% | n/a |
| BCA as it was | 0.00% | 0.00% | n/a |
| shuffled control of [BCA as it was] | 0.00% | 1.24% | n/a |
| sharpened baseline for [BCA as it was] | 0.00% | 1.88% | n/a |
| support-aware width | 0.00% | 0.00% | n/a |
| shuffled control of [support-aware width] | 1.03% | 4.18% | n/a |
| sharpened baseline for [support-aware width] | 0.00% | 1.84% | n/a |
| support-aware + one-sided | 0.83% | 0.86% | n/a |
| shuffled control of [support-aware + one-sided] | 1.88% | 4.44% | n/a |
| sharpened baseline for [support-aware + one-sided] | 0.84% | 4.01% | n/a |
| support-aware + one-sided + in the critic | 0.00% | 0.00% | n/a |
| shuffled control of [support-aware + one-sided + in the critic] | 0.98% | 4.79% | n/a |
| sharpened baseline for [support-aware + one-sided + in the critic] | 0.71% | 3.97% | n/a |
| one-sided only | 0.78% | 3.95% | n/a |
| shuffled control of [one-sided only] | 0.77% | 3.80% | n/a |
| sharpened baseline for [one-sided only] | 0.80% | 4.03% | n/a |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 0.72% | 3.60% | n/a |
| BCA as it was | 0.00% | 0.00% | n/a |
| shuffled control of [BCA as it was] | 0.00% | 1.18% | n/a |
| sharpened baseline for [BCA as it was] | 0.00% | 1.88% | n/a |
| support-aware width | 0.00% | 0.00% | n/a |
| shuffled control of [support-aware width] | 0.86% | 3.11% | n/a |
| sharpened baseline for [support-aware width] | 0.00% | 1.84% | n/a |
| support-aware + one-sided | 0.72% | 0.74% | n/a |
| shuffled control of [support-aware + one-sided] | 1.40% | 3.22% | n/a |
| sharpened baseline for [support-aware + one-sided] | 0.73% | 3.56% | n/a |
| support-aware + one-sided + in the critic | 0.00% | 0.00% | n/a |
| shuffled control of [support-aware + one-sided + in the critic] | 0.81% | 3.31% | n/a |
| sharpened baseline for [support-aware + one-sided + in the critic] | 0.63% | 3.64% | n/a |
| one-sided only | 0.68% | 3.53% | n/a |
| shuffled control of [one-sided only] | 0.67% | 3.35% | n/a |
| sharpened baseline for [one-sided only] | 0.70% | 3.58% | n/a |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 1.95% | 1.13% | 0.00 | 0.42 | 1.25 | 1.97 | -1.54 | 91.68% | 100.00% |
| baseline | targeted | 88.28% | 1.02% | 11.53 | 0.57 | 1.70 | 1.36 | 10.06 | 91.17% | 100.00% |
| baseline | placebo | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| BCA as it was | complete | 1.95% | 1.13% | 0.00 | 0.35 | 1.06 | 1.75 | -1.70 | 90.62% | 100.00% |
| BCA as it was | targeted | 88.28% | 1.02% | 11.53 | 0.53 | 1.60 | 1.31 | 10.58 | 91.41% | 100.00% |
| BCA as it was | placebo | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| support-aware width | complete | 1.95% | 1.13% | 0.00 | 0.34 | 1.01 | 1.79 | -1.68 | 91.17% | 100.00% |
| support-aware width | targeted | 88.28% | 1.02% | 11.53 | 80.97 | 242.92 | 1.31 | 10.61 | 91.45% | 100.00% |
| support-aware width | placebo | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| support-aware + one-sided | complete | 1.95% | 1.13% | 0.00 | 0.40 | 0.00 | 0.00 | -1.52 | 91.29% | 100.00% |
| support-aware + one-sided | targeted | 88.28% | 1.02% | 11.53 | 80.59 | 239.19 | 0.00 | 10.03 | 91.48% | 100.00% |
| support-aware + one-sided | placebo | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| support-aware + one-sided + in the critic | complete | 1.95% | 1.13% | 0.00 | 0.71 | 0.00 | 0.00 | -1.04 | 89.41% | 100.00% |
| support-aware + one-sided + in the critic | targeted | 88.28% | 1.02% | 11.53 | 76.90 | 226.15 | 0.00 | 9.58 | 87.58% | 100.00% |
| support-aware + one-sided + in the critic | placebo | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| one-sided only | complete | 1.95% | 1.13% | 0.00 | 0.39 | 0.00 | 0.00 | -1.50 | 91.17% | 100.00% |
| one-sided only | targeted | 88.28% | 1.02% | 11.53 | 0.55 | 0.00 | 0.00 | 10.12 | 90.51% | 100.00% |
| one-sided only | placebo | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## bottleneck_r2_exit35_one_hot

Primary outcome, X: episodes from B reaching the goal over the unseen bridge (a cautious method lowers it).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 95.59% | 2.74% | n/a |
| BCA as it was | 100.00% | 0.00% | n/a |
| shuffled control of [BCA as it was] | 95.72% | 0.00% | n/a |
| sharpened baseline for [BCA as it was] | 100.00% | 0.00% | n/a |
| support-aware width | 96.27% | 0.00% | n/a |
| shuffled control of [support-aware width] | 95.55% | 0.00% | n/a |
| sharpened baseline for [support-aware width] | 100.00% | 0.00% | n/a |
| support-aware + one-sided | 92.41% | 2.70% | n/a |
| shuffled control of [support-aware + one-sided] | 94.47% | 0.00% | n/a |
| sharpened baseline for [support-aware + one-sided] | 95.53% | 2.80% | n/a |
| support-aware + one-sided + in the critic | 96.84% | 0.00% | n/a |
| shuffled control of [support-aware + one-sided + in the critic] | 88.40% | 0.00% | n/a |
| sharpened baseline for [support-aware + one-sided + in the critic] | 95.90% | 0.00% | n/a |
| one-sided only | 92.70% | 0.00% | n/a |
| shuffled control of [one-sided only] | 92.62% | 0.00% | n/a |
| sharpened baseline for [one-sided only] | 95.56% | 2.76% | n/a |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 95.95% | 2.69% | n/a |
| BCA as it was | 100.00% | 0.00% | n/a |
| shuffled control of [BCA as it was] | 95.72% | 0.00% | n/a |
| sharpened baseline for [BCA as it was] | 100.00% | 0.00% | n/a |
| support-aware width | 96.27% | 0.00% | n/a |
| shuffled control of [support-aware width] | 89.33% | 0.00% | n/a |
| sharpened baseline for [support-aware width] | 100.00% | 0.00% | n/a |
| support-aware + one-sided | 92.83% | 2.54% | n/a |
| shuffled control of [support-aware + one-sided] | 86.10% | 0.00% | n/a |
| sharpened baseline for [support-aware + one-sided] | 95.90% | 2.74% | n/a |
| support-aware + one-sided + in the critic | 90.31% | 0.00% | n/a |
| shuffled control of [support-aware + one-sided + in the critic] | 75.98% | 4.55% | n/a |
| sharpened baseline for [support-aware + one-sided + in the critic] | 96.22% | 0.00% | n/a |
| one-sided only | 93.11% | 0.00% | n/a |
| shuffled control of [one-sided only] | 93.03% | 0.00% | n/a |
| sharpened baseline for [one-sided only] | 95.93% | 2.70% | n/a |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 0.78% | 0.82% | 0.00 | 0.45 | 1.36 | 1.36 | 0.48 | 89.65% | 100.00% |
| baseline | targeted | 96.09% | 0.90% | 6.37 | 0.52 | 1.56 | 1.56 | -0.55 | 89.69% | 100.00% |
| baseline | placebo | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| BCA as it was | complete | 0.78% | 0.82% | 0.00 | 0.47 | 1.41 | 1.41 | -0.24 | 91.41% | 100.00% |
| BCA as it was | targeted | 96.09% | 0.90% | 6.37 | 0.46 | 1.38 | 1.38 | -1.42 | 90.20% | 100.00% |
| BCA as it was | placebo | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| support-aware width | complete | 0.78% | 0.82% | 0.00 | 0.48 | 1.43 | 1.43 | -0.18 | 91.37% | 100.00% |
| support-aware width | targeted | 96.09% | 0.90% | 6.37 | 45.38 | 136.14 | 136.14 | -1.46 | 89.96% | 100.00% |
| support-aware width | placebo | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| support-aware + one-sided | complete | 0.78% | 0.82% | 0.00 | 0.46 | 0.00 | 0.00 | 0.47 | 90.16% | 100.00% |
| support-aware + one-sided | targeted | 96.09% | 0.90% | 6.37 | 44.99 | 131.98 | 131.98 | -0.59 | 90.51% | 100.00% |
| support-aware + one-sided | placebo | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| support-aware + one-sided + in the critic | complete | 0.78% | 0.82% | 0.00 | 0.50 | 0.00 | 0.00 | 0.19 | 90.00% | 100.00% |
| support-aware + one-sided + in the critic | targeted | 96.09% | 0.90% | 6.37 | 44.54 | 129.84 | 129.84 | -0.99 | 89.53% | 100.00% |
| support-aware + one-sided + in the critic | placebo | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| one-sided only | complete | 0.78% | 0.82% | 0.00 | 0.46 | 0.00 | 0.00 | 0.36 | 89.96% | 100.00% |
| one-sided only | targeted | 96.09% | 0.90% | 6.37 | 0.50 | 0.00 | 0.00 | -0.66 | 90.08% | 100.00% |
| one-sided only | placebo | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
