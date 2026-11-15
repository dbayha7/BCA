# Validation on the published map with the paper's own data

Prepared November 12 2026 for David. Plain baselines only; **no BCA policy or control was trained.** The rule and the pass marks are Amendment 2 of [PROTOCOL.md](PROTOCOL.md), written before any of these datasets existed. The validation was run once. The first validation, with our own data design, is in [VALIDATION.md](VALIDATION.md).

**Update, same day: the baseline now passes.** One setting of the baseline was changed, by a rule written beforehand, and the same gate was run again on fresh datasets. All 15 baselines on optimal data score 0.90 or more. Sections 8 and 9. Sections 1 to 7 describe the first run and are left as written. The first test of Version A on this map is in [RESULTS_VERSION_A.md](RESULTS_VERSION_A.md).

Quick navigation: [Verdict](#1-verdict) · [What was run](#2-what-was-run) · [Scores](#3-scores) · [Why the baseline falls short](#4-why-the-baseline-falls-short) · [What this means](#5-what-this-means) · [Next step](#6-next-step-yours-to-decide) · [Limits](#7-limits) · [The fix](#8-the-fix-one-setting-of-the-baseline) · [Second validation](#9-second-validation-the-baseline-passes) · [Files](#10-files-and-commands)

## 1 Verdict

**First run: the data pass. The baseline does not. By the rule written beforehand, no method is trained on this testbed as it stands.** (For the second run, see [section 9](#9-second-validation-the-baseline-passes).)

| Gate | Result | Verdict |
|---|---|---|
| W0 The data are what the paper describes | 15 of 15 datasets pass every check | pass |
| W1 The baseline is competent on optimal data | 0 of 5 dataset seeds valid; 3 were needed | **FAIL** |

**One thing to hold beside that verdict.** The baseline does not fall short because the map or the data are wrong. Plainly copying the data scores 1.00. The baseline loses its points because its critic overvalues commands the data never contain, which is the problem BCA is meant for. Sections 4 and 5.

Terms used here:

- **Baseline**: TD3+BC, the plain learner BCA is added to. It has a critic (a network that scores commands) and a policy that follows the critic while being pulled toward recorded commands.
- **Score**: the paper's measure. The chance of reaching the goal from the start cell within 200 steps, as a share of the optimal policy's chance (95.1%).
- **Critical cell**: a cell with lava on two sides, where one wrong move ends the run. The map has two. The first is row 6, column 7. The second is row 8, column 3.

## 2 What was run

**The paper's, unchanged:** the "Single Critical" map, five moves (stay, up, down, left, right), 10% noise, reward 1 at the goal, lava ends the run with nothing.

**The paper's three data recipes**, 20,000 rows each:

| Recipe | What it is |
|---|---|
| `optimal` | Episodes of the optimal policy from the start cell |
| `other_starts` | The same, each episode starting at a random open cell |
| `mixed` | Half the rows from the optimal policy, half from a policy that commands a random move |

Nothing is left out by hand. A row records the commanded move.

**Ours, because the paper does not state it or BCA needs it:** the learner outputs five continuous preferences instead of picking one of five moves; 20,000 rows; the random share of one half; discount 0.95; 200 steps.

**The run.** Five dataset seeds (6301 to 6305), so 15 datasets. Three training seeds (20, 21, 22), so 45 baselines, 25,000 updates each, on the GPU. Every calibration was usable. Results in `results/published_validation/`.

## 3 Scores

| Recipe | Score, mean | Lowest | Highest | Runs at 0.90 or more | Reaches the goal | Ends in lava | Times out |
|---|---:|---:|---:|---|---:|---:|---:|
| `optimal` | 0.753 | 0.565 | 0.945 | 2 of 15 | 71.6% | 22.0% | 6.4% |
| `other_starts` | 0.791 | 0.438 | 0.966 | 5 of 15 | 75.3% | 24.7% | 0.0% |
| `mixed` | 0.532 | 0.162 | 1.002 | 2 of 15 | 50.6% | 31.9% | 17.5% |
| The optimal policy | 1.000 | | | | 95.1% | 4.9% | 0.0% |

**Gate W1**, one dataset seed at a time. A seed is valid if all three of its baselines on `optimal` data score 0.90 or more.

| Dataset seed | Scores on `optimal` | Valid |
|---|---|---|
| 6301 | 0.875, 0.584, 0.744 | no |
| 6302 | 0.883, 0.619, 0.744 | no |
| 6303 | 0.834, 0.598, 0.754 | no |
| 6304 | 0.936, 0.565, 0.726 | no |
| 6305 | 0.945, 0.692, 0.796 | no |

- **On `optimal` data the scores swing with the training seed, not the dataset.** The first training seed scores 0.83 to 0.95 on every dataset; the second 0.57 to 0.69.
- A score can pass 1.000 by a hair: the optimal policy is the best for discounted reward, which is not quite the same as the highest goal rate.

## 4 Why the baseline falls short

Four findings, each read from the saved policies.

**1. The data are enough. Copying them scores 1.00.** A policy that commands, at every cell, the move the data command most often there scores 1.000 on `optimal`, 1.000 on `other_starts` and 0.997 on `mixed`. This check trains nothing and was not registered beforehand (`cloning_check.py`). So the whole shortfall, 21 to 47 points, comes from what the baseline's critic adds.

**2. At the first critical cell the critic rates a fatal move above the right one, in 15 of 15 runs.**

| Recipe | Rows at the cell | Rows that command a fatal move | Critic's error at the fatal moves (median) | Runs where the critic rates a fatal move above the way through | Policy commands a fatal move |
|---|---:|---:|---:|---|---:|
| `optimal` | 45 | 0 | +0.76 | 15 of 15 | 10.2% |
| `other_starts` | 942 | 0 | +0.67 | 15 of 15 | 15.0% |

- With `other_starts` data there are about 940 rows at this cell and **every one commands "down"**. No row commands "left" or "right", so the critic has to guess their value, and it guesses high. A fatal move is truly worth about 0.04.
- The result: from the start, 14.9% of episodes end in the lava to the right of this cell. Under the optimal policy that is 0.1%.
- **This is an out-of-distribution action being overvalued, and nothing was left out by hand to produce it.** It comes from the paper's own data recipe.
- At the second critical cell this does not happen: the critic rates the way through highest in 15 of 15 runs and the policy never commands a fatal move.

**3. The policy does not hold a recorded command even where the data are unanimous.** With `optimal` data, 12 cells have more than 1,000 rows that all command the same move. The policy commands that move 87.6% of the time there. On the 18 cells of the optimal route it commands the optimal move with certainty at only half. A softer command means a random move part of the time.

**4. With `mixed` data the critic's scores run away in 9 of 15 runs.** Nothing here can truly be worth more than 1.0. In nine runs the critic's largest score is between 1.6 and 215. Those runs score 0.40 on average; the six where the critic stays in range score 0.72.

**The common cause, my reading.** The paper's data contain five commands: the five moves. Our learner outputs continuous preferences, so nearly every command it can issue is one the data never contain. The critic's scores there are guesses, and the policy follows them. Our earlier data design hid this, because half of its rows were random preferences that covered the whole command space.

## 5 What this means

- **The map and the data are sound.** W0 passes and copying the data solves the task.
- **The baseline has the overestimation problem here, in a published setup, without any hand-made gap.** That is the situation BCA is meant for.
- **The gate I wrote stops us anyway, and it was built on a wrong assumption of mine.** I required the baseline to be competent on optimal data, taking optimal data to be the easy case. For a learner with continuous actions it is the hard case: only five commands are covered. The gate cannot tell "the testbed is broken" from "the baseline has the problem under study".
- **No BCA policy has been trained here, and none will be until you decide.** I saw the same pattern on the test seed before the study datasets were built and changed nothing in response.

## 6 Next step, yours to decide

**David's decision, November 12: not this.** "We need to get the plain baseline to work right." The gate stays, and the baseline was fixed instead (sections 8 and 9). What follows is what I proposed at the time.

**My recommendation:** replace gate W1 with "copying the data scores at least 0.90" (it scores 1.00), then run Version A against the baseline on fresh datasets of this testbed, with pass marks written first.

That changes a gate after seeing baseline results, so it needs your go-ahead. It would be a new amendment on new dataset seeds; seeds 6301 to 6305 have now been seen.

Other paths, for completeness:

| Option | What it is | Cost |
|---|---|---|
| Make the learner discrete too | Five moves and a table, as the paper's methods are | BCA needs a new host; the largest change |
| Stay with the earlier testbed | Honeypot and Bottleneck, where Version A is already replicated | Not a published setup |

## 7 Limits

- Development seeds, baselines only, run once.
- The map and the rules are the paper's; the learner is not. The paper's methods are tabular.
- Dataset size, the random share, the discount and the episode length are our choices; the paper does not state them.
- One of the paper's three maps. The paper runs its mixing experiment on the "Multiple Critical" map.
- The copying check and the cause in section 4 were not registered beforehand.
- Two smoke runs on dataset seed 7 came first, one on the CPU and one on the GPU. They check that the code runs.

## 8 The fix: one setting of the baseline

Done November 12 2026 under Amendment 3 of [PROTOCOL.md](PROTOCOL.md), written before the pilot. Baselines only.

**The cause, traced in the 45 saved baselines of the first run.** TD3+BC trains its policy on two pulls: follow the critic, and stay near the recorded command. The paper's data hold five commands, so at a cell where every row says "down" the critic's scores for the other four are guesses. The policy moved toward those guesses.

| What was measured | Result |
|---|---|
| Cells with 16 or more rows where the policy's top choice is a wrong move (`optimal`, `other_starts`) | 199 |
| Of those, the chosen move is one that no row commands there | 199 of 199 |
| Of those, the critic rates the chosen move above the recorded one | 199 of 199 |
| `optimal` data: route cells where the top choice is still the right move | 100%, in 15 of 15 runs |

- **A small pull weakens the command**, and our preference-to-move rule turns a weak command into random moves. This is the whole shortfall on `optimal` data.
- **A large pull changes the commanded move.** Three `other_starts` runs command "right" at the start cell, where every row says "down", and never reach the goal.

**The setting.** TD3+BC has one number for this trade, called alpha: how much weight the critic gets against the recorded commands. The published value is 2.5, chosen on robot-control benchmarks. Nothing else was changed: not the learner's code, the map, the data, our move rule, or the gate.

**The pilot.** Dataset seed 7 (not a study seed), six training seeds, 18 baselines per setting. The rule, fixed beforehand: use the largest alpha at which all six baselines on `optimal` data score 0.90 or more.

| Alpha | `optimal`: mean | Lowest | Runs at 0.90 or more | `other_starts`: mean | `mixed`: mean | `mixed`: runs whose critic leaves the possible range |
|---|---:|---:|---|---:|---:|---|
| 2.5 (published) | 0.716 | 0.661 | 0 of 6 | 0.890 | 0.478 | 3 of 6 |
| **1.0** | **0.948** | **0.923** | **6 of 6** | 0.984 | 0.761 | 3 of 6 |
| 0.5 | 0.993 | 0.981 | 6 of 6 | 0.989 | 0.857 | 2 of 6 |
| 0.25 | 0.997 | 0.985 | 6 of 6 | 0.995 | 0.990 | 0 of 6 |
| 0.1 | 0.999 | 0.999 | 6 of 6 | 0.998 | 0.978 | 0 of 6 |

- **The rule picks 1.0**, the largest setting that qualifies.
- **The cause is confirmed**: with nothing else changed, less weight on the critic takes the score from 0.72 to 0.95 and up.
- **My prediction was half wrong.** I wrote that 1.0 would not be enough and 0.5 would qualify. 1.0 qualifies.

## 9 Second validation: the baseline passes

Gates W0 and W1 exactly as before, on five new dataset seeds (6401 to 6405), training seeds 20, 21, 22, alpha 1.0. 45 baselines on the GPU, every calibration usable, run once. Results in `results/published_validation_2/`.

| Gate | Result | Verdict |
|---|---|---|
| W0 The data are what the paper describes | 15 of 15 datasets pass every check | pass |
| W1 The baseline is competent on optimal data | 5 of 5 dataset seeds valid; 3 were needed | **PASS** |

| Dataset seed | Scores on `optimal` | Valid |
|---|---|---|
| 6401 | 0.977, 0.904, 0.932 | yes |
| 6402 | 0.995, 0.908, 0.945 | yes |
| 6403 | 0.992, 0.908, 0.917 | yes |
| 6404 | 0.995, 0.913, 0.940 | yes |
| 6405 | 0.995, 0.919, 0.934 | yes |

| Recipe | Score, mean | Lowest | Runs at 0.90 or more | Reaches the goal | Ends in lava | Times out |
|---|---:|---:|---|---:|---:|---:|
| `optimal` | 0.945 | 0.904 | 15 of 15 | 89.9% | 8.7% | 1.4% |
| `other_starts` | 0.986 | 0.975 | 15 of 15 | 93.8% | 6.2% | 0.0% |
| `mixed` | 0.843 | 0.586 | 7 of 15 | 80.2% | 12.6% | 7.3% |
| The optimal policy | 1.000 | | | 95.1% | 4.9% | 0.0% |

**Three things to hold beside the pass.**

- **The margin is thin for one training seed.** The second training seed scores 0.904 to 0.919 on every dataset, just over the mark.
- **The critic is still wrong; the policy no longer follows it.** At the first critical cell the critic rates a fatal move above the right one in 15 of 15 runs on `optimal` and on `other_starts`, by about 0.7. The policy now commands a fatal move there 0.4% and 0.0% of the time. It was 10% and 15%.
- **The half-random data are still a weak spot.** On `mixed` the critic's scores leave the possible range in 6 of 15 runs, and those runs score 0.72 against 0.92 for the rest. In the pilot that stopped only at alpha 0.25, where the baseline is close to plain copying.

**Where this leaves the testbed.** The gate is met, so by the rules a method may be tested here once David says to go on. The room for a method is mostly on `mixed` data. On the other two recipes the baseline is at 0.95 and 0.99 of the best possible.

By the rule of Amendment 2, the study's datasets are the first three valid seeds: 6401, 6402, 6403. Seeds 6401 to 6405 have now been seen for baselines.

## 10 Files and commands

| File | What it is |
|---|---|
| `PROTOCOL.md`, Amendment 3 | The traced cause, the candidates, the rule and the prediction, written before the pilot |
| `run_published_alpha.py`, `test_alpha.py` | The runner that sets the cloning weight without editing the shared learner, and its 3 tests |
| `pilot_alpha.py`, `results/pilot_alpha.md`, `results/pilot_alpha.json`, `results/pilot_alpha_*/` | The pilot: 90 baselines on dataset seed 7 and the rule applied to them |
| `data/published2/`, `receipts/published2_*`, `results/published_validation_2/` | The second validation: 15 fresh datasets, 45 baselines at alpha 1.0, scores and gates |
| `PROTOCOL.md`, Amendment 2 | The recipes, what is ours, gates W0 and W1, written before any of these datasets was generated |
| `published_collect.py`, `published_checks.py`, `build_published.py`, `test_published.py` | The three recipes, their acceptance checks, the dataset builder and 24 tests |
| `data/published/`, `receipts/published_*` | The 15 datasets (ignored by git) and their check receipts |
| `run_published.py`, `results/published_validation/` | The 45 baselines: policies, critic read-outs, run receipts |
| `validate_published.py` | Scores and gates from the saved policies: `published_validation.json`, `records.json`, `published_tables.md` |
| `cloning_check.py`, `results/published_cloning_check.json` | The copying check of section 4. No training |
| `results/smoke_published_1/`, `results/smoke_published_gpu/` | The two smoke runs on dataset seed 7 |

To re-derive the scores and gates without a GPU, from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B experiments/gridworld_critical_v1/validate_published.py --label published_validation
```

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B experiments/gridworld_critical_v1/cloning_check.py --label published
```
