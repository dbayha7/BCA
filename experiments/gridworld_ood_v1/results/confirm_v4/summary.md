# Study summary

Every entry is a mean over the 27 runs of its setting, method and data arm (9 dataset seeds and 3 training seeds in this folder). Written by `summarize_study.py` from `records.json` and `gates.json`.

## honeypot

Primary outcome, H: episodes ending in hazard 34 (lower is better).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 0.31% | 5.98% | 0.86% |
| support-aware width, calibrated part read at shuffled states | 0.05% | 0.24% | 0.30% |
| shuffled control of [support-aware width] | 0.16% | 7.02% | 0.17% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 0.00% | 2.33% | 0.41% |

Executed probability of the withheld move at the decision cells.

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 0.22% | 4.46% | 0.62% |
| support-aware width, calibrated part read at shuffled states | 0.05% | 0.23% | 0.22% |
| shuffled control of [support-aware width] | 0.14% | 6.94% | 0.16% |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 0.00% | 2.18% | 0.28% |

Average true value of an episode, over all start cells (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 4.96 | 4.90 | 5.00 |
| support-aware width, calibrated part read at shuffled states | 5.86 | 5.89 | 5.92 |
| shuffled control of [support-aware width] | 5.46 | 2.90 | 5.60 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 5.70 | 5.66 | 5.71 |

Average true value of an episode starting at a decision cell (not a gate).

| Method | complete | targeted | placebo |
|---|---|---|---|
| baseline | 5.29 | 4.80 | 5.30 |
| support-aware width, calibrated part read at shuffled states | 6.12 | 6.23 | 6.21 |
| shuffled control of [support-aware width] | 5.85 | 2.89 | 6.07 |
| sharpened baseline for [support-aware width, calibrated part read at shuffled states] | 6.08 | 5.90 | 6.05 |

G1 (the gap misleads the baseline): 9 of 3 datasets.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| support-aware width, calibrated part read at shuffled states | PASS 9/3 | PASS 7/3 | PASS 9/3 | PASS 7/3 |

What the width and the support score did. "Target" is the command that executes the withheld move with certainty; "best" is the truly best corner command at the same cell.

| Method | Data arm | Withheld-move rows scored unsupported | Fresh ordinary rows scored unsupported | Factors of ten below the floor, target | Width used, target | Penalty, target | Penalty, best | Critic error, target | Coverage on fresh rows | Runs with a usable width |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | complete | 2.16% | 1.02% | 0.03 | 0.36 | 1.08 | 2.06 | 0.69 | 91.01% | 100.00% |
| baseline | targeted | 89.44% | 1.13% | 13.77 | 0.59 | 1.77 | 1.52 | 9.87 | 91.02% | 100.00% |
| baseline | placebo | 76.23% | 1.12% | 0.00 | 0.34 | 1.03 | 2.07 | 0.22 | 91.11% | 100.00% |
| support-aware width, calibrated part read at shuffled states | complete | 2.16% | 1.02% | 0.03 | 0.57 | 1.70 | 2.01 | 0.72 | 91.16% | 100.00% |
| support-aware width, calibrated part read at shuffled states | targeted | 89.44% | 1.13% | 13.77 | 96.05 | 288.15 | 1.40 | 10.40 | 91.32% | 100.00% |
| support-aware width, calibrated part read at shuffled states | placebo | 76.23% | 1.12% | 0.00 | 0.31 | 0.94 | 1.98 | 0.23 | 91.11% | 100.00% |
