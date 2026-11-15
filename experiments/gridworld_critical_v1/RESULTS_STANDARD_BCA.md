# Standard BCA on the published map: first test

Prepared November 13 2026 for David, the morning of the research meeting. Standard BCA is the shared learner's plain `bca` arm: BCA's calibrated width, with the standard calibration, charged in the policy's choice. The pass marks are Amendment 11 of [PROTOCOL.md](PROTOCOL.md), written before any standard BCA policy was trained here. Run once. **A first test, not a confirmation**: a sixth look at the same three datasets.

Quick navigation: [Verdict](#1-verdict) · [Why it was run](#2-why-it-was-run) · [Scores](#3-scores) · [The marks](#4-the-marks) · [What it shows](#5-what-it-shows) · [Predictions](#6-predictions) · [Files](#7-files-and-commands)

## 1 Verdict

**Standard BCA does no harm and gains a little on the half-random data, but not enough to pass.** Two of three marks pass. The gain on the half-random data is +0.017, against a mark of +0.05.

## 2 Why it was run

David asked for the standard BCA score in the setup-validation table and, told that it had never been trained on this map, chose "Train it now". Version A had been carried over from the earlier testbed and tested here first; standard BCA, which the later versions build on, was skipped.

## 3 Scores

Means of nine runs on the three study datasets (seeds 6401 to 6403, training seeds 20 to 22). The score is the paper's: the chance of reaching the goal from the start, as a share of the best possible policy's.

| Policy | Optimal | Other starts | Half-random |
|---|---:|---:|---:|
| Baseline (TD3+BC, alpha 1.0) | 0.942 | 0.986 | 0.801 |
| **Standard BCA** | **0.943** | **0.986** | **0.817** |
| Shuffled control (same charge, read at shuffled states) | 0.943 | 0.986 | 0.808 |
| Decisive baseline (trains nothing) | 0.970 | 0.972 | 0.766 |
| Version A, for comparison | 0.989 | 0.987 | 0.875 |

On the half-random data, per dataset: standard BCA 0.798, 0.894, 0.760; baseline 0.771, 0.879, 0.752. Lowest run 0.625 (baseline 0.586). Lava endings 13.7% (baseline 14.5%).

## 4 The marks

| Mark | Result | Verdict |
|---|---|---|
| **K1** Helps where there is room | Half-random: higher than the baseline in 3 of 3 datasets, mean gain +0.017 (mark +0.05) | **FAIL** |
| **K2** Not a side effect | Higher than the shuffled control in 2 of 3 datasets; higher than the decisive baseline in 2 of 3 | **PASS** |
| **K3** No harm where the baseline is good | Optimal: score +0.001, lava endings 0.1 points lower. Other starts: unchanged | **PASS** |

By the rule written beforehand, standard BCA helps only if all three pass. K1 does not.

## 5 What it shows

- **No harm.** On optimal and random-starts data the scores equal the baseline's to two decimals.
- **A small, consistent gain on the half-random data**: higher in all three datasets, by 0.017 on average.
- **The width does not tell a fatal move from the right one.** At the first critical cell standard BCA charges a fatal move 0.06 and the way through 0.06 on optimal data (0.05 and 0.06 on random-starts data, 0.28 and 0.30 on the half-random data). This is what the width diagnostic of November 12 found.
- **The critic is untouched.** It rates a fatal move above the way through at the first critical cell in 9 of 9 runs on optimal data and 9 of 9 on random-starts data, as for the baseline.
- **The runner reproduces the baseline**: the baseline continued through it is identical to the saved one in 27 of 27 runs.

**What this means for the later versions.** Version A and Version C both add a check of how well the data support a command. That check is what tells a never-recorded fatal move from the recorded one; the calibrated width alone does not, on this map.

## 6 Predictions

Written in Amendment 11 before the run. K3 passes (right). K1 fails, with a gain under 0.05 (right: +0.017). K2 fails or ties (wrong: it passes, 2 of 3 against each control). A small gain on optimal data is possible (there is none to speak of: +0.001). The critic's ranking stays wrong in 9 of 9 runs (right).

## 7 Files and commands

| File | What it is |
|---|---|
| `PROTOCOL.md`, Amendment 11 | The method, the marks and the predictions, written before the run |
| `run_standard_bca_published.py` | The runner: `run_version_a_published.py` with the method switched to `bca` and its control `shuffled_width`. No earlier file is edited |
| `judge_standard_bca.py` | `judge_version_a.py` with the method names switched; writes `standard_bca.md`, `standard_bca.json`, `standard_bca_records.json` |
| `results/standard_bca_published/` | Policies, read-outs, receipts and the judged tables |
| `results/smoke_standard_bca/` | The smoke test on dataset seed 7; its scores were not looked at |

To re-derive every number without a GPU, from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B experiments/gridworld_critical_v1/judge_standard_bca.py --label standard_bca_published
```

Note: that command rewrites the three `standard_bca.*` files from the saved policies.
