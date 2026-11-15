# Validation of the two tests on the published map

**A second validation, with the paper's own data recipes, is in [VALIDATION_PUBLISHED.md](VALIDATION_PUBLISHED.md) (November 12 2026).**

Prepared November 10 2026 for David. Plain baselines only; no BCA policy or control was trained. The pass marks are in [PROTOCOL.md](PROTOCOL.md) and were written before any training. The validation was run once.

Quick navigation: [Verdict](#1-verdict) · [What was run](#2-what-was-run) · [Competence](#3-competence-fails) · [The critic's errors](#4-the-critics-errors-are-as-intended) · [Behavior](#5-does-the-gap-change-what-the-baseline-does) · [The paper's noise](#6-the-papers-noise-makes-little-difference) · [Options](#7-options-for-you) · [Files](#8-files-and-commands)

## 1 Verdict

**The testbed does not pass validation. By the kill rule written beforehand, no method is trained on it as it stands.**

| Gate | Paper's noise (10%) | No noise |
|---|---|---|
| **V0** Baseline is competent with complete data | **Fail**, 0 of 9 runs | **Fail**, 0 of 9 runs |
| **V1-over** Critic overestimates the left-out fatal moves | Pass, 3 of 3 datasets | Pass, 3 of 3 |
| **V1-under** Critic underestimates the left-out way through | Pass, 3 of 3 | Pass, 3 of 3 |
| **V2-over** That overestimation changes what the baseline does | Fail, 1 of 3 | Fail, 1 of 3 |
| **V2-under** That underestimation changes what the baseline does | Pass, 2 of 3 | Pass, 3 of 3 |

In words:

- **Both tests produce the estimation error they are named for.** The critic overestimates the unseen fatal moves by about 0.7 and underestimates the unseen way through by 0.2 to 0.5, on a scale where reaching the goal is worth 1.
- **The plain baseline is not competent on this map.** Even with nothing left out, it reaches the goal from the start in 73% of episodes where 95% is possible, and takes about twice as long as it needs to.
- **The overestimation does not mislead the baseline.** The underestimation does.

## 2 What was run

- **Environment.** The published "Single Critical" map and rules ([PUBLISHED_SETUP.md](PUBLISHED_SETUP.md)), with our continuous-preference layer on top.
- **Data.** 24 datasets: 2 noise settings x 3 dataset seeds (6101 to 6103) x 4 arms. Every one passed the acceptance checks, when made and when re-read from disk.
- **Policies.** 72 plain baselines: 3 training seeds per dataset, 25,000 updates each. All 72 calibrations were usable; coverage on fresh rows was 86% to 94% against a 90% target.
- **Checks on the code.** 38 tests of the environment and datasets, and the 45 learner tests, all passing. A smoke run on dataset seed 7 came back clean. I did see its numbers and changed nothing after it.

## 3 Competence fails

Complete data, from the start cell, mean of 9 runs.

| | Paper's noise | No noise |
|---|---:|---:|
| Best any policy can do: reaches the goal | 95.1% | 100% |
| Baseline: reaches the goal | 72.6% | 81.7% |
| Baseline: ends in lava | 27.3% | 18.3% |
| Baseline: value, as a share of the best possible | 45% | 50% |
| Runs within 10 points of the best goal rate | 1 of 9 | 1 of 9 |
| Runs with at least 60% of the best value | 0 of 9 | 1 of 9 |

Why, from the baseline policies themselves:

- **Its commands are too soft for a map where one wrong step is fatal.** With complete data it still commands a fatal move 7% to 10% of the time at each of the two critical cells. Average command strength is 0.57 on a scale of 0 to 1.
- **Most of the lava comes from the second route.** From the start, 14% to 17% of episodes end in the lava beside the second critical cell and 4% to 10% in the lava beside the first. The two routes are equally short, so the policy uses both.
- **It is slow.** No episode times out, but the value is about half the best possible, which corresponds to roughly 13 wasted steps. The paper's reward has no cost per step, so only the discount pushes for speed.

The same learner was competent on yesterday's maps. Two things differ here: lava sits on both sides of the critical cells, and falling into it costs nothing beyond losing the goal (yesterday's hazards carried a penalty of half the goal reward).

## 4 The critic's errors are as intended

The critic's value for commanding a move with certainty at the critical cell, minus the true value. Mean over 3 training seeds, by dataset.

**Overestimation test: error at the fatal moves (left and right).** True value is 0.04 with the paper's noise and 0 without.

| Dataset | Paper's noise: left out | complete | No noise: left out | complete |
|---|---:|---:|---:|---:|
| 6101 | +0.65 | +0.31 | +0.73 | +0.28 |
| 6102 | +0.74 | +0.29 | +0.75 | +0.37 |
| 6103 | +0.71 | +0.33 | +0.66 | +0.25 |

**Underestimation test: error at the way through (down).** True value is about 0.76 with the paper's noise and 0.82 without.

| Dataset | Paper's noise: left out | complete | No noise: left out | complete |
|---|---:|---:|---:|---:|
| 6101 | -0.18 | +0.05 | -0.49 | +0.07 |
| 6102 | -0.39 | +0.20 | -0.51 | +0.03 |
| 6103 | -0.34 | +0.51 | -0.24 | +0.09 |

- **Leaving a move out more than doubles the overestimate**, from about +0.3 to about +0.7.
- **The critic overestimates the fatal moves even when they are in the data**, by about +0.3. That is part of why the baseline is not competent.
- **Leaving out the way through flips the sign**: the critic goes from slightly too high to clearly too low.

## 5 Does the gap change what the baseline does

At the critical cell, mean of 9 runs.

| Data arm | Paper's noise: commands a fatal move | commands down | No noise: commands a fatal move | commands down |
|---|---:|---:|---:|---:|
| Complete | 9.6% | 79.6% | 6.7% | 76.7% |
| Fatal moves left out ("over") | 5.5% | 88.2% | 11.1% | 75.2% |
| Way through left out ("under") | 23.5% | 30.5% | 10.9% | 25.7% |
| Placebo | 7.0% | 70.9% | 8.8% | 72.7% |

- **Overestimation does not lure the baseline.** Overestimated to about 0.72, the fatal moves come out nearly level with going down, which the critic values at about 0.78, but not ahead of it. No recorded command at that cell points at them either. So the baseline keeps going down.
- **Underestimation does stop it.** With the way through left out, the baseline commands down 26% to 30% of the time where it otherwise commands it 71% to 80%.
- **It does not then take the long route cleanly.** From the critical cell it ends in the neighboring lava in 52% of episodes with the paper's noise (26% without), against 10% to 22% otherwise. In the data for that arm the demonstrator at the critical cell says "up" while the cell above still says "down", and the policy dithers between them.

## 6 The paper's noise makes little difference

Every gate has the same verdict with and without the noise. With the noise, a left-out move still happens by accident about 5 times in 256 rows, and the critic is still wrong about it by nearly as much. So keeping the published noise does not weaken the tests.

## 7 Options for you

1. **Sharpen our own layer and validate again.** The published game stays as it is. What changes is how five preferences become a move: the sharpness of 2 was inherited from yesterday's four-move testbed. A sharper rule makes the same preferences more decisive, which is what this map needs. I would pick the value from baseline-only runs on dataset seed 7, write it down as an amendment, and validate on fresh datasets, as was done for the Bottleneck's second revision.
2. **Go back to yesterday's two environments.** They passed validation. They are not from a paper.
3. **Lower the competence mark.** I advise against it. A baseline that reaches the goal 73% of the time when 95% is possible is not competent, and any method's effect would be tangled up with that.

Whichever is chosen, the over arm has a second problem that sharpening will not obviously fix: the overestimation has to be large enough to beat the way through before it can mislead anyone.

### What happened to option 1

David chose it. A pilot on dataset seed 7 (baselines only, complete data only, six training seeds) tried sharpness 2, 3, 4 and 6 under a rule fixed beforehand: take the smallest value at which all six runs are competent in both noise settings.

| Sharpness | Paper's noise: runs competent | No noise: runs competent |
|---|---|---|
| 2 (the original) | 0 of 6 | 0 of 6 |
| 3 | 2 of 6 | 3 of 6 |
| 4 | 5 of 6 | 5 of 6 |
| 6 | 5 of 6 | 5 of 6 |

Sharpening helps a great deal, and at 4 and 6 the baseline no longer commands a fatal move at the critical cell at all. But no value reached six of six, so by the rule the fix did not qualify, and I stopped. The details and one runner defect found on the way are in Amendment 1 of [PROTOCOL.md](PROTOCOL.md). The validation has not been rerun.

## 8 Files and commands

| File | What it is |
|---|---|
| `PUBLISHED_SETUP.md` | The paper's map and rules, and what it leaves unstated |
| `transcribe_figure.py` | Reads the three maps from the paper's figure by color |
| `critical_env.py` | The published game, then our continuous-preference layer; exact evaluation |
| `critical_collect.py`, `critical_checks.py`, `build_datasets.py` | The four data arms, the acceptance checks, the dataset builder |
| `test_environment.py`, `test_datasets.py` | 21 and 17 tests |
| `PROTOCOL.md` | Pass marks and kill rules, written before training |
| `run_baselines.py`, `validate.py` | Train the baselines; judge the gates |
| `receipts/dev_manifest.json`, `receipts/dev_checks.json` | Dataset hashes and every check result |
| `results/dev_validation/` | `validation.json`, `records.json` (every policy), `validation_tables.md`, saved policies and snapshots |
| `results/smoke_1/` | The smoke run on dataset seed 7 |

To re-judge the saved policies without a GPU, from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B experiments/gridworld_critical_v1/validate.py --label dev_validation
```
