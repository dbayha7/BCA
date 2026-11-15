# Version A on the published map: first test

Prepared November 12 2026 for David. The plan and the pass marks are Amendment 4 of [PROTOCOL.md](PROTOCOL.md), written before any Version A policy was trained on this map. Run once. **This is a first test, not a confirmation**: the marks were fixed before the run, but after the baselines on these datasets had been seen. How the baseline was validated is in [VALIDATION_PUBLISHED.md](VALIDATION_PUBLISHED.md).

Quick navigation: [Verdict](#1-verdict) · [What was run](#2-what-was-run) · [Scores](#3-scores) · [The marks](#4-the-marks) · [What Version A does here](#5-what-version-a-does-here) · [What to hold beside the pass](#6-what-to-hold-beside-the-pass) · [Next step](#7-next-step-yours-to-decide) · [Confirmation](#7b-confirmation-on-untouched-datasets) · [Limits](#8-limits) · [Files](#9-files-and-commands)

## 1 Verdict

**Version A passes all three marks. By the rule written beforehand, it helps on the published map.**

| Data recipe | Baseline | Version A |
|---|---:|---:|
| Optimal-policy runs | 0.942 | **0.989** |
| Same, from random start cells | 0.986 | 0.987 |
| Half optimal, half random moves | 0.801 | **0.875** |

The score is the paper's: the chance of reaching the goal from the start, as a share of the best possible policy's. Means of nine runs.

**Two things to hold beside it.** On the half-random data Version A's shuffled control does just as well (0.873), so that mark passes by a hair. And one run in nine collapses there. Section 6.

Terms used here:

- **Baseline**: TD3+BC with alpha 1.0, the plain learner validated on this map.
- **Version A**: the baseline plus a penalty on commands the data do not support. The support check is read at the true cell; the calibrated width is read at a shuffled cell.
- **Shuffled control**: the same penalty with all of its width read at shuffled cells.
- **Decisive baseline**: the baseline's own top choice at every cell, commanded with certainty. No training.
- **Critic**: the network that scores commands.

## 2 What was run

- **Map, noise, rewards, data recipes, score:** the paper's, as in the validation.
- **Data:** the three study datasets (seeds 6401, 6402, 6403), three recipes, 20,000 rows each.
- **Trained:** 27 Version A policies and 27 shuffled controls, training seeds 20, 21, 22. Each starts from the baseline's saved state after 10,000 updates and trains to 25,000, as in the earlier testbed. GPU, every calibration usable.
- **Check on the runner:** the baseline continued the same way reproduces the saved baselines bit for bit, in 27 of 27 runs.
- **Version A itself is unchanged** from the earlier testbed: strength 3, nothing tuned for this map.

## 3 Scores

Means of nine runs. "Per dataset" is the mean of three training seeds.

**Half optimal, half random moves (`mixed`)**

| Policy | Score | Lowest run | Per dataset | Ends in lava | Times out | Runs whose critic leaves the possible range |
|---|---:|---:|---|---:|---:|---|
| Baseline | 0.801 | 0.586 | 0.771, 0.879, 0.752 | 14.5% | 9.3% | 6 of 9 |
| Decisive baseline | 0.766 | 0.031 | 0.648, 0.989, 0.661 | 4.1% | 23.0% | |
| Shuffled control | 0.873 | 0.054 | 0.980, 0.975, 0.665 | 6.0% | 10.9% | 0 of 9 |
| **Version A** | **0.875** | 0.084 | 0.983, 0.972, 0.671 | 6.3% | 10.5% | 0 of 9 |

**Optimal-policy runs (`optimal`)**

| Policy | Score | Lowest run | Per dataset | Ends in lava | Times out |
|---|---:|---:|---|---:|---:|
| Baseline | 0.942 | 0.904 | 0.938, 0.949, 0.939 | 8.6% | 1.8% |
| Decisive baseline | 0.970 | 0.955 | 0.970, 0.971, 0.969 | 7.5% | 0.2% |
| Shuffled control | 0.983 | 0.958 | 0.986, 0.977, 0.986 | 6.5% | 0.0% |
| **Version A** | **0.989** | 0.975 | 0.990, 0.990, 0.986 | 5.9% | 0.1% |

**Optimal policy from random start cells (`other_starts`)**

| Policy | Score | Lowest run | Per dataset | Ends in lava | Times out |
|---|---:|---:|---|---:|---:|
| Baseline | 0.986 | 0.977 | 0.987, 0.986, 0.987 | 6.2% | 0.0% |
| Decisive baseline | 0.972 | 0.958 | 0.972, 0.972, 0.972 | 7.5% | 0.0% |
| Shuffled control | 0.974 | 0.958 | 0.977, 0.967, 0.977 | 7.3% | 0.0% |
| **Version A** | 0.987 | 0.958 | 0.982, 0.987, 0.991 | 6.1% | 0.0% |

The optimal policy itself ends in lava 4.9% of the time, pushed there by the noise.

## 4 The marks

| Mark | Result | Verdict |
|---|---|---|
| **K1** Helps where there is room | On `mixed`: higher than the baseline in 2 of 3 datasets; mean gain +0.074 against a mark of +0.05 | **PASS** |
| **K2** Not a side effect | On `mixed`: higher than the shuffled control in 2 of 3 datasets, and higher than the decisive baseline in 2 of 3 | **PASS**, by a hair against the shuffled control |
| **K3** No harm where the baseline is good | `optimal`: score +0.047, lava endings 2.7 points lower. `other_starts`: score unchanged, lava unchanged | **PASS** |

My predictions: K3 passes (right). Version A scores higher than the baseline on `mixed` (right, in 7 of 9 runs). K2 is the mark most at risk (right: it passed, but narrowly).

## 5 What Version A does here

**On the optimal-policy data it is better in every run.** 9 of 9 runs above the baseline, and 9 of 9 above the decisive baseline. Lava endings fall from 8.6% to 5.9%, against 4.9% for the best possible policy.

**The support check flags the moves the data never contain.** At the two critical cells, Version A's charge for a command:

| Data | Cell | Charge for a fatal move | Charge for the way through |
|---|---|---:|---:|
| `optimal` | first critical cell | 10.26 | 0.07 |
| `optimal` | second critical cell | 12.51 | 0.06 |
| `other_starts` | first critical cell | 8.05 | 0.06 |
| `other_starts` | second critical cell | 10.57 | 0.06 |
| `mixed` | first critical cell | 0.05 | 0.05 |
| `mixed` | second critical cell | 0.28 | 0.06 |

- With optimal data no row commands a fatal move at these cells, and the charge is more than 100 times that of the recorded move.
- With half-random data every move is recorded at most cells, so there is nothing for the support check to flag.

**It never commands a fatal move at the first critical cell.** The baseline does so 0.63% of the time on `optimal` data and 6.90% on `mixed`. Version A: 0.00% on all three recipes.

**On the half-random data it steadies the policy and the critic.**

| Dataset, training seed | Baseline | Decisive baseline | Shuffled control | Version A | Baseline critic's largest score |
|---|---:|---:|---:|---:|---:|
| 6401, 20 | 0.586 | 0.031 | 0.983 | 0.984 | 5.0 |
| 6401, 21 | 0.987 | 1.000 | 0.973 | 0.980 | 1.0 |
| 6401, 22 | 0.739 | 0.912 | 0.984 | 0.984 | 43.2 |
| 6402, 20 | 0.973 | 1.001 | 0.978 | 0.978 | 1.0 |
| 6402, 21 | 0.915 | 1.002 | 0.968 | 0.967 | 1.0 |
| 6402, 22 | 0.750 | 0.965 | 0.979 | 0.970 | 107.5 |
| 6403, 20 | 0.608 | 0.033 | 0.972 | 0.971 | 2.6 |
| 6403, 21 | 0.942 | 1.001 | 0.968 | 0.957 | 1.6 |
| 6403, 22 | 0.707 | 0.949 | 0.054 | 0.084 | 31.9 |

- **In 8 of 9 runs Version A scores between 0.957 and 0.984**, whatever the baseline did (0.586 to 0.987).
- **The critic's scores stay in range in 9 of 9 runs.** In the baseline they leave it in 6 of 9. No score can truly exceed 1.0.
- **It rescues the two runs where the baseline's top choices were wrong**: the decisive baseline scores 0.03 there, Version A 0.97 and 0.98.

**It does not fix the critic's ranking on optimal data.** At the first critical cell the critic still rates a fatal move above the way through in 9 of 9 runs, on `optimal` and on `other_starts`. Version A acts on the policy, not on the critic.

## 6 What to hold beside the pass

- **On the half-random data the shuffled control is as good as Version A.** 0.873 against 0.875. Run by run they differ by 0.03 at most, and Version A is higher in 4 of 9. K2 passed there because two dataset means came out 0.003 and 0.006 higher. That is a pass by the letter and should not be leaned on.
- **So on `mixed`, reading the penalty at the true cell is not what helps.** A penalty of this kind helps wherever it is read, which fits the table above: with half-random data the same five commands are recorded at nearly every cell.
- **Reading it at the true cell shows a small edge on the other two recipes.** Against the shuffled control: +0.006 on `optimal` (higher in 5 of 9 runs) and +0.013 on `other_starts` (higher in 8 of 9).
- **One run in nine collapses.** Dataset 6403, training seed 22: Version A 0.084, shuffled control 0.054, with 88% of episodes timing out. The baseline's critic had run away in that run (largest score 31.9). The baseline itself scored 0.707 there. That one run is why the third dataset is below the baseline.
- **Where the baseline is already right, commanding its top choice with certainty beats Version A.** In four `mixed` runs the decisive baseline scores 1.00 and Version A 0.96 to 0.98. Version A's lava endings are 6.3%, against 4.9% for the best possible policy.
- **On `other_starts` Version A adds nothing** (0.987 against 0.986). There was no room.
- **Which part of Version A does the work was not tested here.** In the earlier testbed the calibrated width was not the working part. Nothing in this run changes that or tests it.

## 7 Next step, yours to decide

By the rule written beforehand, a pass here is a reason to confirm on untouched datasets, nothing more.

**My recommendation:** a confirmation run on new datasets of this map, with the marks written before the datasets exist. Two changes I would make to the marks, both stated now:

- Judge "better than a control" on the average over datasets with an exact test, as in the earlier testbed, alongside the count of datasets. A 2-of-3 count can pass on a difference of 0.003.
- Use more than three datasets.

## 7b Confirmation on untouched datasets

Run November 12 2026 under Amendment 5 of [PROTOCOL.md](PROTOCOL.md), written before the datasets existed. David said: "Yes, confirm it". Run once.

**The run.** Fifteen candidate dataset seeds (6501 to 6515). All 45 datasets passed their checks and all fifteen seeds were valid by the baselines alone, so the first nine (6501 to 6509) were used. Training seeds 20, 21, 22. 135 baselines, then 81 Version A policies and 81 shuffled controls, on the GPU, every calibration usable. The baseline continued through the runner reproduces the saved one in 81 of 81 runs.

**Verdict: confirmed on optimal data, not on the half-random data.** By the rule written beforehand Version A needed C1, C2 and C3; C1 does not hold.

| Claim | Result | Verdict |
|---|---|---|
| **C1** Helps on the half-random data | 0.885 to 0.909: higher in 7 of 9 datasets, mean gain +0.024 (mark +0.05), p = 0.31 | does not hold |
| **C2** Helps on optimal data | 0.953 to 0.990: higher in 9 of 9 datasets, mean gain +0.038 (mark +0.02), p = 0.002 | **holds** |
| **C3** No harm where there is no room | `other_starts`: score +0.002, lava endings 0.2 points lower | **holds** |
| **D1** More than decisiveness | `optimal`: 9 of 9, +0.019, p = 0.002. `mixed`: 5 of 9, +0.036, p = 0.30 | holds on `optimal` only |
| **D2** The true cell matters | `optimal`: 9 of 9, +0.007, p = 0.002. `other_starts`: 9 of 9, +0.011, p = 0.002. `mixed`: 7 of 9, +0.061, p = 0.018 | **holds on all three** |

Means of 27 runs:

| Recipe | Baseline | Decisive baseline | Shuffled control | Version A | Version A runs below 0.50 |
|---|---:|---:|---:|---:|---|
| `optimal` | 0.953 | 0.971 | 0.984 | **0.990** | 0 of 27 |
| `other_starts` | 0.986 | 0.972 | 0.977 | 0.987 | 0 of 27 |
| `mixed` | 0.885 | 0.873 | 0.848 | 0.909 | 2 of 27 |

- **On optimal data the gain is confirmed.** Lava endings fall from 8.6% to 5.8%, against 4.9% for the best possible policy.
- **On the half-random data the gain is not confirmed.** Version A is ahead on average and in 7 of 9 datasets, but two of its 27 runs collapse (lowest 0.106) and the baseline has none below 0.658.
- **Reading the penalty at the true cell matters on every recipe.** In the first test the shuffled control tied on `mixed`; here it is behind by 0.061 and collapses in 4 of 27 runs.
- **The critic's scores stay in range in 27 of 27 `mixed` runs** with Version A; the baseline's leave it in 5 of 27.

My predictions: C2 and C3 hold (right). C1's comparison holds and its 0.05 gain is at risk (the gain fell short, and so did the test). D2 fails on `mixed` (wrong: it holds).

Seeds 6501 to 6515 are spent. Result files: `results/confirm_baselines/`, `results/confirm_version_a/` (`confirmation.md`, `confirmation.json`, `confirmation_records.json`), judged by `confirm_version_a_published.py` (12 tests in `test_confirm.py`).

## 8 Limits

- First test: three datasets, three training seeds each, development status.
- The marks were written after the baselines on these datasets had been seen.
- One map of the paper's three. The learner is ours; the paper's methods are tabular.
- The baseline's alpha was chosen for its competence, not for Version A.
- The control for decisiveness was changed before any run, the smoke test included, because the scaled version does not work on this map (Amendment 4 states why). The scaled version is in the result files, without a mark.
- I did not look at the smoke test's Version A scores before the run.

## 9 Files and commands

| File | What it is |
|---|---|
| `PROTOCOL.md`, Amendment 4 | The method, the controls, marks K1 to K3, the decision rule and the predictions, written before any Version A policy was trained here |
| `run_version_a_published.py` | Trains Version A and its shuffled control from the baselines' saved states; continues the baseline as a check |
| `judge_version_a.py`, `test_version_a.py` | Computes every number here from the saved policies; 7 tests of the marks' logic |
| `results/version_a_published/` | Policies and read-outs, `version_a.md`, `version_a.json`, `version_a_records.json` (every run), run receipts |
| `results/published_validation_2/` | The baselines the run starts from |
| `results/smoke_version_a/` | The smoke test on dataset seed 7 |

To re-derive every number without a GPU, from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B experiments/gridworld_critical_v1/judge_version_a.py --label version_a_published
```
