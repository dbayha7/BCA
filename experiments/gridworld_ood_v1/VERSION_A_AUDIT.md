# Version A in the GridWorld: every metric, the environment, and why it works

Prepared November 12 2026 for David. **This is a description of runs whose verdicts are already fixed.** It reads the saved policies of the three fresh-data runs of Version A (sections 13, 14 and 16 of [RESULTS_METHOD_CHANGES.md](RESULTS_METHOD_CHANGES.md)). Nothing was trained for it, and pooling those runs here confirms nothing new.

Quick navigation: [Short answer](#1-short-answer) · [What Version A is](#2-what-version-a-is) · [Is the environment good](#3-is-the-environment-good) · [Every metric](#4-every-metric-in-the-honeypot) · [Why it works](#5-why-it-works) · [Where it does not work](#6-where-it-does-not-work-the-bottleneck) · [Side effects](#7-side-effects) · [What this says about BCA](#8-what-this-says-about-bca) · [Limits](#9-limits-of-this-audit) · [Files](#10-files)

## 1 Short answer

**Version A works in the Honeypot, where a fatal move is missing from the data.** Over 45 runs on 15 fresh datasets it ends in the hazard 0.18% of the time, against 5.08% for the TD3+BC baseline, and it earns 97% of the best possible return, against 82%. It is better than the baseline in 15 of 15 datasets on hazard endings, on return and on bad endings.

**It works for two reasons, and they are different sizes.**

| Step | Hazard endings | Share of the improvement |
|---|---:|---:|
| TD3+BC baseline | 5.08% | |
| The same baseline with more decisive commands, no retraining | 1.70% | 69% |
| Version A | 0.18% | the other 31% |

1. **Decisive commands.** The baseline gives soft commands, so its moves are partly random, and some of that randomness lands on the fatal move. Version A's commands are decisive. Making the baseline's own commands as decisive, with no training at all, removes 69% of the hazard endings.
2. **A check for commands unlike any in the data.** This removes the rest. It is what handles the cases where the critic is truly fooled.

**The part of Version A that is BCA's own idea, a calibrated width that knows the state, is not what does the work.** At the fatal move that width is 0.59. The critic's error there is 9.88. Section 5 shows this number by number.

**Version A does not work in the Bottleneck, where a good move is missing from the data.** It never takes the unseen bridge. A penalty can push a policy away from an unseen move. It cannot tell a good unseen move from a bad one.

**The environment is built correctly.** It is a valid test of whether a policy avoids an overvalued unseen move. It is a weak test of calibrated uncertainty, for reasons listed in section 3.4.

## 2 What Version A is

Terms used throughout.

- **Command**: the four preferences the policy outputs at a cell, one per direction. The environment turns them into movement chances. A direction given less than a 5% chance is never executed.
- **Withheld move**: the one move at a decision cell that the data never contain. In the Honeypot it is the step into hazard 34 from each of the four cells around it.
- **Critic**: the network that scores how good a command is at a cell. **Critic error** is its score minus the true value, so a positive error means it rates the command too high.
- **Calibrated width**: BCA's estimate of how large the critic's one-step error is, set so that it covers 90% of errors on rows like the data.
- **Support score**: a small classifier trained to tell recorded commands from random ones. A low score means "no command like this was recorded at this cell".

Version A is the TD3+BC baseline plus one extra term in what the actor minimizes: 3 times a width.

| Part of the width | What it is | What it knows |
|---|---|---|
| The calibrated width | BCA's width, read at the state of another, randomly chosen row of the batch | The command, not the cell |
| The raise | Where the support score falls below its floor, the width is raised toward half the range of rewards, once per factor of ten below the floor | The cell and the command |

Version A changes the actor only. It does not change how the critic is trained.

## 3 Is the environment good?

### 3.1 What it is

A 10 by 10 grid with 89 cells an agent can stand on, 6 wall cells, 4 hazards and one goal. A step costs 0.1, the goal pays 10, a hazard costs 5 and ends the episode. The discount is 0.95 and an episode lasts at most 200 steps.

Every cell has 256 recorded rows: 128 uniformly random commands, and 128 noisy copies of the best command. The **targeted** data drop every command that could step into hazard 34 from the four cells around it. The **placebo** data drop a harmless move at the same cells instead.

### 3.2 Checks that pass

| Check | Result |
|---|---|
| Automated tests of the environment, data, learner and evaluator | 92 of 92 pass |
| Rows per cell, in all 15 targeted datasets | exactly 256 everywhere |
| Targeted and complete data identical outside the four decision cells | true in all 15 datasets |
| Commands in the targeted data that could execute the withheld move | 0 |
| Fewest executions of any other move at a decision cell | 28 (the mark is 16) |
| Best policy reaches the goal from every start cell | true |
| Best policy's hazard endings from the decision cells | 0% |
| Outcomes computed exactly, with no sampling noise | yes |

### 3.3 The gap does what it is meant to do

The baseline, by data arm, 45 runs each:

| | Fatal move in the data | Fatal move withheld | A harmless move withheld |
|---|---:|---:|---:|
| Hazard endings from the decision cells | 0.35% | **5.08%** | 0.80% |
| Critic's score for the fatal move (its true value is -5.00) | -4.25 | **+4.88** | -4.88 |
| Critic error at the fatal move | +0.75 | **+9.88** | +0.12 |

- **With the move in the data the critic knows it is bad.** With it withheld, the critic scores it +4.88, almost as high as the best move (+5.53). This is the overestimation the test is built around.
- **The effect is specific to the gap.** Withholding a harmless move at the same cells does not produce it.
- **The true cost of the fatal move is large.** At the decision cells the best move is worth 5.96 to 6.82 and the fatal move -5.00.

### 3.4 Four things that limit what this environment can show

1. **The problem is small.** The 5.08% is measured from episodes started at the four cells beside the hazard. Over all start cells the baseline ends in hazard 34 only 1.21% of the time. Only 10 of the 89 start cells have a best path that passes through a decision cell.
2. **The baseline's main weakness here is soft commands, and the data recipe causes it.** Half of every cell's rows are uniformly random commands. Behavior cloning pulls the policy toward their average, which is no preference at all. The baseline puts only 84.6% of its movement chance on best moves. Any method that makes commands decisive looks good here, whether or not it knows anything about uncertainty.
3. **The gap is large and clean, which is the easiest case for a support check.** 84% of random commands at a decision cell could execute the withheld move. So the targeted data cover only 16% of the command space at those cells. Cells are given as one-hot inputs and every cell has the same number of rows.
4. **A move with no rows gives a calibration method nothing to calibrate on.** The calibrated width is fitted on errors at recorded rows. At a move that was never recorded there are no errors to fit.

## 4 Every metric in the Honeypot

Fatal move withheld, means of 45 runs on 15 fresh datasets. *Sharpened baseline*: the baseline's own commands scaled to be as decisive as Version A's, with no training. *Shuffled control*: Version A's penalty with the width's values dealt to the wrong states.

### 4.1 What the policy achieves

| Metric | Baseline | Sharpened baseline | Version A | Shuffled control |
|---|---:|---:|---:|---:|
| Hazard-34 endings from the decision cells | 5.08% | 1.70% | **0.18%** | 4.97% |
| Chance per step of the fatal move at a decision cell | 3.79% | 1.57% | **0.17%** | 4.89% |
| Runs where the fatal move has no chance at any decision cell | 8 of 45 | 32 of 45 | **40 of 45** | 29 of 45 |
| Hazard-34 endings from all start cells | 1.21% | 0.49% | **0.04%** | 0.89% |
| Any hazard ending, all start cells | 1.72% | 0.71% | **0.13%** | 1.09% |
| Time-outs, all start cells | 0.00% | 0.00% | 0.00% | 31.77% |
| Goal reached, all start cells | 98.28% | 99.29% | **99.87%** | 67.14% |
| Worst start cell, chance of a bad ending | 17.29% | 10.16% | **3.86%** | 59.32% |
| Start cells that never reach the goal, per run | 0 | 0 | 0 | 28.8 |
| Return, all start cells (best possible 6.07) | 4.95 | 5.71 | **5.90** | 3.38 |
| Share of the best possible return | 81.6% | 94.0% | **97.2%** | 55.7% |
| Return from the decision cells | 4.92 | 5.98 | **6.23** | 3.66 |
| Return from the far start (best possible 3.02) | 1.63 | 2.50 | **2.78** | 0.44 |

The shuffled control is a broken policy: it times out from a third of start cells. Beating it means little. The comparison that matters is the sharpened baseline.

### 4.2 How the policy behaves

| Metric | Baseline | Sharpened baseline | Version A | Shuffled control |
|---|---:|---:|---:|---:|
| Decisiveness (average size of a preference, 0 to 1) | 0.565 | 0.779 | 0.779 | 0.950 |
| Movement chance placed on best moves | 84.6% | 94.8% | **97.4%** | 94.8% |
| Cells whose likeliest move is a best move | 99.7% | 99.7% | 99.9% | 96.6% |
| Cloning error (squared distance to recorded commands) | 0.493 | 0.587 | 0.546 | 0.684 |

- **The baseline almost always knows the best move.** Its likeliest move is a best move at 99.7% of cells. Its problem is how much chance it leaves on the other moves.
- **Version A is further from the recorded commands than the baseline**, because half the recorded commands are random.

### 4.3 What the critic believes

Read at the four decision cells. Columns are each method's own critic.

| Metric | Baseline | Version A | Shuffled control |
|---|---:|---:|---:|
| Critic's score for the fatal move (true value -5.00) | +4.88 | +5.40 | +3.84 |
| Critic error at the fatal move | **+9.88** | **+10.40** | +8.84 |
| The same, taking the lower of the two critics | +9.54 | +10.07 | +8.49 |
| Critic's score for the best move | 5.53 | 6.03 | 4.61 |
| Critic error at the best move | +0.05 | -0.25 | +0.31 |
| Cases where the critic scores the fatal move above the best move | 31.7% | 30.6% | 37.8% |
| Critic error, commands that could execute the fatal move | +4.49 | +4.43 | +4.16 |
| Critic error, commands that cannot | +0.25 | -0.24 | +0.50 |
| Critic error over the whole map, average size | 0.76 | 0.68 | 1.43 |
| The two critics disagree by, at the fatal move | 0.70 | 0.70 | 0.75 |
| The two critics disagree by, at the best move | 0.07 | 0.06 | 0.13 |

- **Version A does not fix the critic.** Its critic rates the fatal move as wrongly as the baseline's does (+10.40 against +9.88). Version A works by overruling the critic in the actor, not by correcting it.
- **Taking the lower of two critics does not help**: the error is still +9.54.
- **The two critics disagree ten times more at the fatal move than at the best move**, 0.70 against 0.07. That is a signal, but it is small beside an error of 9.88, and with the move in the data the disagreement is larger still (1.36).

### 4.4 What the uncertainty parts say

Read at the four decision cells, on the baseline's critic unless a column says otherwise.

| Metric | At the fatal move | At the best move |
|---|---:|---:|
| Calibrated width | 0.59 | 0.52 |
| Penalty it gives (3 times the width) | 1.78 | 1.57 |
| Support score, factors of ten below its floor | 13.8 | 0.003 |
| Width after the support raise | 95.6 | 0.54 |
| Penalty after the support raise | 287 | 1.63 |

| Metric | Value |
|---|---:|
| Calibrated width averaged over the whole map | 0.75 |
| Calibrated width at the fatal move when the move is in the data | 0.36 |
| Ordinary fresh rows the support score flags by mistake | 1.1% |
| Rows that could execute the fatal move that it flags | 89.5% |
| Corner commands at ordinary cells that it flags | 6.1% |
| Cells whose best command it flags | 1.6% |
| Coverage of the calibrated width on ordinary fresh rows (the aim is 90%) | 91.3% |
| Coverage of the calibrated width on rows that could execute the fatal move | 59.3% |
| Coverage of Version A's raised width on those rows | 97.8% |
| Calibrations that were usable | 100% |

- **The calibrated width is honest on rows like the data**: 91.3% coverage against an aim of 90%.
- **It is not honest where the data are missing**: 59.3% coverage on rows that could execute the fatal move.
- **It does grow where the move is missing**, from 0.36 to 0.59, but it stays below its own map average of 0.75.

### 4.5 Paired with the baseline

Each run compared with the baseline trained on the same data with the same seed. A start cell is "made worse" if its chance of a bad ending rises by more than 1 point.

| | Sharpened baseline | Version A | Shuffled control |
|---|---:|---:|---:|
| Start cells made worse, per run (of 89) | 0.71 | **0.58** | 31.8 |
| Start cells made better, per run | 20.2 | **22.4** | 14.6 |
| Start cells with a higher return, per run | 85.2 | 85.4 | 51.5 |
| Start cells with a lower return, per run | 0.29 | 0.53 | 33.0 |
| Datasets with fewer hazard-34 endings than the baseline | 15 of 15 | 15 of 15 | 8 of 15 |
| Datasets with a higher return than the baseline | 15 of 15 | 15 of 15 | 3 of 15 |
| Datasets with fewer bad endings than the baseline | 15 of 15 | 15 of 15 | 1 of 15 |

### 4.6 The other data arms

| Data | Metric | Baseline | Sharpened baseline | Version A |
|---|---|---:|---:|---:|
| Fatal move in the data | Hazard-34 endings from the decision cells | 0.35% | 0.00% | 0.03% |
| | Return, all start cells | 5.04 | 5.75 | 5.90 |
| | Bad endings, all start cells | 0.60% | 0.14% | 0.10% |
| A harmless move withheld | Hazard-34 endings from the decision cells | 0.80% | 0.32% | 0.18% |
| | Return, all start cells | 5.06 | 5.75 | 5.94 |
| | Bad endings, all start cells | 0.68% | 0.28% | 0.06% |

**Version A raises the return by about 0.9 in every arm, including when nothing is missing from the data.** So most of its return gain has nothing to do with the gap. It comes from decisive commands.

## 5 Why it works

### 5.1 What goes wrong for the baseline

Two things combine.

1. **The critic rates the fatal move almost as high as the best move**: +4.88 against +5.53, when the truth is -5.00 against +5.48. The fatal move has no rows, so nothing ever corrected its score.
2. **The baseline's commands are soft.** Because the critic sees little difference, the actor does not push the fatal direction's preference down. A soft command leaves it more than a 5% chance, so it sometimes executes.

The baseline almost never picks the fatal move as its top move (1.7% of cases). It leaks chance onto it.

### 5.2 The two kinds of case

One case per run and decision cell, 180 in all, split by what the baseline's critic believes there.

| | Critic still rates the best move higher | Critic rates the fatal move higher |
|---|---:|---:|
| Cases | 123 (68%) | 57 (32%) |
| Baseline: chance per step of the fatal move | 2.25% | 7.11% |
| Sharpened baseline: chance per step | 0.22% | **4.49%** |
| Version A: chance per step | 0.15% | **0.23%** |
| Sharpened baseline: cases where the fatal move can still happen | 2.4% | 17.5% |
| Version A: cases where it can still happen | 2.4% | 3.5% |

- **Where the critic is only unsure, decisiveness is enough.** A sharpened baseline does as well as Version A: 0.22% against 0.15%.
- **Where the critic is fooled, decisiveness is not enough, and the support check is.** A sharpened baseline still takes the fatal move 4.49% of the time. Version A takes it 0.23% of the time.
- **This is the whole of Version A's lead over a sharpened baseline.**

### 5.3 What each part of the width contributes

Cases where the critic scores the fatal move above the best move, before and after each penalty:

| | Share of cases |
|---|---:|
| Critic alone | 31.7% |
| After the calibrated width's penalty | 31.1% |
| After the support-aware width's penalty | 0.0% |

- **The calibrated width's penalty changes almost nothing.** It is 1.78 at the fatal move and 1.57 at the best move, a difference of 0.21, against a critic error of 9.88.
- **The support raise settles every case.** Its penalty is 287 at the fatal move and 1.63 at the best move.

### 5.4 The five cases Version A still gets wrong

In 5 of the 180 cases Version A can still execute the fatal move. In every one the chance is 5.0% to 6.7%, just over the 5% line below which a move is never executed. The pure fatal command is flagged strongly in all five (11 to 14 factors of ten below the floor). The policy's actual command sits at the edge of what was recorded, where the fatal direction is only slightly preferred.

### 5.5 The same picture from the development runs

These used seeds seen during development, so they are exploratory. Honeypot, fatal move withheld, hazard endings (section 12.4 of [RESULTS_METHOD_CHANGES.md](RESULTS_METHOD_CHANGES.md)):

| Method | Support check | Calibrated width | Hazard endings |
|---|---|---|---:|
| Baseline | no | no | 5.70% |
| BCA as it was | no | at the true state | 2.08% |
| BCA as it was, width at shuffled states | no | at a wrong state | 1.98% |
| Support check alone | yes | no | 2.56% |
| Support check and calibrated width | yes | at the true state | 0.00% |
| Version A | yes | at a wrong state | 0.00% |

Neither part is enough alone. Each time the calibrated width was read at the wrong state the result was the same, so what it adds is a pull toward decisive commands, not knowledge of the cell.

## 6 Where it does not work: the Bottleneck

Here the withheld move is the best one: crossing a bridge. A cheaper exit is available instead. Means of 18 runs on 6 fresh datasets.

| Low exit pays | Metric | Baseline | Sharpened baseline | Version A |
|---|---|---:|---:|---:|
| 4.5 | Crossings of the unseen bridge | 2.6% | 1.2% | 0.0% |
| | Return, all start cells | 4.13 | 4.68 | 4.30 |
| | Bad endings, all start cells | 0.24% | 0.18% | 4.81% |
| 3.5 | Crossings of the unseen bridge | 8.7% | 7.2% | 0.0% |
| | Return, all start cells | 3.72 | 4.25 | 4.36 |
| | Bad endings, all start cells | 0.37% | 0.05% | 0.79% |
| 2.5 | Crossings of the unseen bridge | 8.9% | 6.0% | 0.0% |
| | Return, all start cells | 2.99 | 3.47 | 3.60 |
| | Bad endings, all start cells | 4.53% | 4.24% | 4.88% |

The baseline critic's error at the crossing is -0.24, -1.22 and -1.38 at the three exit values, so it rates the crossing too low.

- **Version A never crosses.** The support check flags the crossing exactly as it flags the fatal move (10.8 factors of ten below the floor). It has no way to know this unseen move is good.
- **Its return is still above the baseline's**, because decisive commands help everywhere else on the map.
- **It ends badly more often than the baseline**, mostly by timing out. The gap is largest at exit 4.5: 4.81% against 0.24%.
- **With the bridge in the data it crosses more than the baseline**: 98% to 100% of the time, against 76% to 97%.

## 7 Side effects

- **In the Honeypot Version A does less harm than a sharpened baseline**: 0.58 start cells made worse per run, against 0.71.
- **The support score has false alarms.** It flags 1.1% of ordinary fresh rows and the best command at 1.6% of cells. Where it flags a best command, Version A is pushed off it.
- **In the Bottleneck some runs stall.** With the bridge withheld, 4.7% of episodes time out at exit 4.5, against 0.0% for the baseline. With the bridge in the data, the worst start cell has a 16% to 18% chance of a bad ending, against 5% to 10% for the baseline.
- **Version A's critic is no better than the baseline's** at the fatal move.

## 8 What this says about BCA

**What works in Version A is not BCA's calibrated uncertainty.**

| Ingredient | Is it BCA's own idea? | Does it carry the result? |
|---|---|---|
| A pull toward decisive commands | No. Scaling the baseline's commands does the same | Yes, 69% of the fall in hazard endings and 80% of the gain in return |
| A check for commands unlike any in the data | No. It is a classifier of recorded against random commands | Yes, the other 31%, and all of the lead over a sharpened baseline |
| A calibrated width that knows the state | Yes | No. It moves the critic's ranking in 0.6 points of cases out of 31.7 |

**Why the calibrated width cannot do this job.** It is fitted on the critic's errors at recorded rows, and it is honest there (91.3% coverage). The fatal move has no recorded rows. So the width has no error to learn from at exactly the place where the error is largest. It ends up at 0.59 against an error of 9.88.

**What the calibrated width is built for.** A move that is in the data, but thinly or noisily, so there are errors to calibrate on. This environment does not contain that case: every cell has 256 rows, and the withheld move has none.

## 9 Limits of this audit

- **Description, not confirmation.** These runs are spent. The three runs were judged under different rules (Amendments 6, 7 and 9), and one candidate dataset of the third was dropped before training. Pooling them is for description only.
- **Dependence.** Runs on the same dataset are not independent. Dataset counts (15, or 6 for the Bottleneck) are the honest sample size. No error bars are given because none would be defensible.
- **The critic and the widths are read at the 16 corner commands of each cell**, not at the policy's own command. The saved files hold nothing else.
- **Each method is scored against its own policy's true values.** So critic errors of different methods are measured against different futures.
- **The split in section 5.2 uses the baseline's critic** to sort cases. It is a description of what happened, not a controlled experiment.
- **One environment family.** One-hot cells, uniform coverage, and a move that is entirely absent.

## 10 Files

| File | What it is |
|---|---|
| `audit_version_a.py` | Computes everything here from the saved policies. Read-only |
| `results/version_a_audit.json` | Its full output: every metric, by task, data arm and method, plus the mechanism split and the environment facts |
| `results/confirm_v1/`, `results/confirm_v2/`, `results/confirm_v4/` | The three fresh-data runs it reads |

To reproduce, from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B experiments/gridworld_ood_v1/audit_version_a.py
```
