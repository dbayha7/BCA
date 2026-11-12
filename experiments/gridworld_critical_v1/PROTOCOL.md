# Protocol: validating the overestimation and underestimation tests on the published map

Written November 10 2026, before any policy was trained on this testbed. David chose the published setup ("Kumar critical states") and approved the build ("Yes, build it"). **This protocol covers validation with plain baselines only. No BCA policy or control is trained under it.**

Quick navigation: [What is copied and what is ours](#1-what-is-copied-and-what-is-ours) · [The two tests](#2-the-two-tests) · [What is trained](#3-what-is-trained) · [What is measured](#4-what-is-measured) · [Pass marks and kill rules](#5-pass-marks-and-kill-rules) · [Rules for running](#6-rules-for-running) · [Amendment 1: sharper move rule](#amendment-1-a-sharper-preference-to-move-rule-chosen-by-a-pilot) · [Limits](#7-limits-known-in-advance)

## 1 What is copied and what is ours

Copied from Kumar, Hong, Singh and Levine (ICLR 2022), as recorded in [PUBLISHED_SETUP.md](PUBLISHED_SETUP.md): the "Single Critical" 10 x 10 map, the five moves, the 10% noise, the reward of 1 for reaching the goal, and lava ending the run with nothing. The exact noise rule follows the code the paper cites.

Ours, and stated as ours:

1. **Continuous actions.** The policy outputs five preferences in [-1, 1]. softmax(2 x preferences) is the chance of commanding each move, and a move given under 5% is never commanded. BCA needs continuous actions; every published gridworld uses discrete ones.
2. **A left-out move.** One data arm never commands a chosen move at the critical cell. No published setup leaves a move out on purpose.
3. **How the data are collected.** Every cell gets 256 single-step rows: 128 with uniform random preferences and 128 copies of the best command with noise of standard deviation 0.1. The paper uses whole trajectories from the best policy, sometimes mixed with a random policy's.
4. **Two numbers the paper does not state.** Discount 0.95 and 200 steps per evaluation episode.

## 2 The two tests

Both act at the critical cell (row 6, column 7). From it, left and right land in lava, and down goes through to the goal side.

| Test | Left out of the data at the critical cell | What a misled learner believes | What it then does |
|---|---|---|---|
| **Overestimation** ("over" arm) | Left and right | Those moves are worth more than they are | Walks into lava |
| **Underestimation** ("under" arm) | Down | That move is worth less than it is | Refuses the way through and takes the long route |

Two more data arms: **complete** (nothing left out) and **placebo** (the move "up" left out at the same cell; it is the least useful harmless move there).

The arms are identical everywhere except at the critical cell. In the "under" arm the demonstrator's own command at that cell changes from down to up, because down may not be commanded. Nowhere else does the demonstrator change.

**Two noise settings.** The paper's 10% noise means a left-out move is never commanded but still happens by accident, about 2 rows in 100 at that cell. To see what that costs the test, everything is run twice: with the paper's noise (the primary setting) and with no noise (where a left-out move truly never happens).

## 3 What is trained

| Factor | Levels |
|---|---|
| Noise | 0.10 as published; 0.00 |
| Dataset seed | 6101, 6102, 6103 (never used before) |
| Data arm | complete, over, under, placebo |
| Training seed | 20, 21, 22 |
| Method | baseline only |

72 policies. The baseline is the TD3+BC learner of `experiments/gridworld_ood_v1/learner.py`, unchanged except that it now takes five action dimensions. Its 45 tests still pass. Each run is 25,000 updates. The scale and support networks are fitted and the width is calibrated after 10,000, 15,000 and 20,000 updates, exactly as before, so that a snapshot at 10,000 updates can start any later method from the same point. The baseline does not use any of it.

## 4 What is measured

Everything is exact: computed from each policy's preferences at every cell, under the rules of section 1, over 200 steps.

- **Competence.** From three starts (the start cell, the cell above the critical cell, the critical cell): the chance of reaching the goal, and the value. Each is compared with the best any policy can do under the same noise, also computed exactly.
- **Critic error at a move.** The critic's value for commanding that move with certainty at the critical cell, minus the true value of doing so and then following the policy. Positive means the critic overestimates.
- **What the policy commands.** At the critical cell: the chance it commands a fatal move (left or right), and the chance it commands down.
- Also reported: lava endings, time-outs and the chance of going through, from the critical cell.

## 5 Pass marks and kill rules

"Dataset" means the mean over that dataset's three training seeds. A gate passes with at least 2 of the 3 datasets, and all three are reported. Every gate is judged separately for each noise setting.

| Gate | Passes if | If it fails |
|---|---|---|
| **V0** Baseline competence, complete data | Every one of the 9 runs, from each of the three starts, reaches the goal within 10 points of the best possible rate and has at least 60% of the best possible value | **That noise setting is not a usable testbed. No method is trained on it** |
| **V1-over** The critic overestimates the left-out moves | In the over arm the critic's error at left and right (averaged) is at least +0.05, and at least 0.05 above the same error in the complete arm | The over arm is not an overestimation setting. It is reported as such and dropped |
| **V1-under** The critic underestimates the left-out move | In the under arm the critic's error at down is at most -0.05, and at least 0.05 below the same error in the complete arm | The under arm is not an underestimation setting. It is reported as such and dropped |
| **V2-over** The gap misleads behavior | The chance of commanding a fatal move is higher in the over arm than in the placebo arm, by more than the complete-versus-placebo difference | The critic is wrong but the baseline is not misled. No claim that a method "protects" can be tested here |
| **V2-under** The gap misleads behavior | The chance of commanding down is lower in the under arm than in the placebo arm, by more than the complete-versus-placebo difference | The critic is wrong but the baseline is not misled. No claim about the cost of caution can be tested here |

0.05 is 5% of the goal reward, and about 6% of the value of the way through.

A test is **valid** in a noise setting if V0 and its V1 gate pass. It can also **test behavior** if its V2 gate passes.

## 6 Rules for running

- The validation is run once. If a defect forces a rerun, the defect, the fix and the discarded outputs are recorded.
- No pass mark and no part of the environment is changed after outcomes are seen. If a gate fails, that is the result.
- A smoke test on dataset seed 7, which is not a study seed, checks that the code runs and that no saved number is non-finite. Its outcomes are not used to choose anything.
- No BCA policy and no control is trained until David has seen the validation and says to go on.

## Amendment 1: a sharper preference-to-move rule, chosen by a pilot

Written November 10 2026, after the validation above failed ([VALIDATION.md](VALIDATION.md)) and before any pilot run. David chose this option ("Yes, try that").

**What failed.** V0, baseline competence, in 0 of 9 runs in both noise settings. With nothing left out, the baseline reached the goal from the start in 73% of episodes where 95% is possible, and still commanded a fatal move 7% to 10% of the time at each critical cell. Its commands are too soft for a map where one wrong step is fatal.

**What changes.** One number in our own layer: the sharpness in `softmax(sharpness x preferences)`. It was 2, carried over from yesterday's four-move testbed. A larger value makes the same preferences more decisive. Nothing in the published game changes: not the map, the moves, the noise or the rewards. The 5% cutoff, the data recipe, the learner and every pass mark also stay as they are.

### Part A: the pilot, fixed before it is run

- **Runs.** Baselines only. Complete data only, so that no left-out arm is looked at while choosing. Dataset seed 7, which is not a study seed. Six training seeds (0 to 5). Both noise settings.
- **Candidates.** Sharpness 2 (the current value, as a reference), 3, 4 and 6. That is 48 policies.
- **Rule.** Take the smallest sharpness at which all six runs pass V0 in both noise settings. V0 is exactly as in section 5: from each of the three starts, within 10 points of the best possible goal rate and at least 60% of the best possible value.
- **If no candidate passes**, the fix has failed. I stop and report; I do not try further values.
- **Why the smallest.** It is the least departure from the inherited value, and it keeps the policy's own imprecision in play, which the tests depend on.

### Part B: what happens after the pilot

- The chosen value and the pilot table are recorded below before anything else is run.
- Fresh datasets are generated (seeds 6201 to 6203) with the chosen sharpness, and must pass the acceptance checks.
- The validation of sections 3 to 5 is run once on them, with the same pass marks, in both noise settings.
- Datasets 6101 to 6103 and the failed validation stay on disk and stay reported.

### Defect 1, found during the pilot

**What happened.** The pilot stopped after six of its eight settings. The runner generates fresh held-out rows for each run and required them to pass every acceptance check. One such held-out dataset (no noise, sharpness 4, seed 200007) failed the check "moves not left out here are in the data": the move "left" happened 14 times at the critical cell, and the mark is 16.

**Why it is a defect in the runner and not in the environment.** That check counts how often a move appears, so that training data are sure to show it. The expected count is 25.6, and it falls under 16 by chance about 1.3% of the time per move, at every sharpness. Held-out rows are only used to calibrate the width and to report coverage; 1,280 of them are sampled from the whole grid. Whether one move appears 14 or 16 times at one cell has no bearing on that.

**The fix.** Held-out rows must pass every check except the two that count appearances. Training data are still held to all checks, and a training dataset that fails is still rejected, never repaired. A test pins this down with the dataset that stopped the pilot.

**What is kept and what is rerun.** The six settings that finished are kept as they are: their held-out rows passed every check, so the fix changes nothing for them. The two settings that never started (no noise, sharpness 4 and 6) are run under the label `pilot_sharpness_2` with the same seeds. Nothing was discarded, and no outcome had been looked at when the fix was made.

### Outcome of the pilot: no candidate met the rule

Runs passing V0, of six, on dataset seed 7 with complete data. In brackets: the mean chance of reaching the goal from the start. The best possible is 95.1% with the paper's noise and 100% without.

| Sharpness | Paper's noise | No noise |
|---|---|---|
| 2 (the original) | 0 of 6 (79.5%) | 0 of 6 (77.5%) |
| 3 | 2 of 6 (86.5%) | 3 of 6 (96.4%) |
| 4 | 5 of 6 (91.2%) | 5 of 6 (99.0%) |
| 6 | 5 of 6 (92.4%) | 5 of 6 (83.3%) |

- **Sharper is much better, but no value reaches six of six in both noise settings**, which is what the rule required.
- **At 4 and 6 the baseline stops commanding a fatal move at the critical cell** (0.00% in every run, from 6% to 9% at sharpness 2 and 3). The one failing run in each case fails elsewhere: a start-cell goal rate of 80%, a value just under the mark, or, at sharpness 6 with no noise, one policy that never leaves the start.
- **By the rule, the fix has failed.** I stopped here. No further value was tried, no fresh datasets were generated, and the validation was not rerun.

For scale: V0 in the validation needs all nine runs to pass. If one run in six fails, as at sharpness 4, the chance that nine pass is about one in five.

## 7 Limits known in advance

- Development seeds. Three datasets with three training seeds each; seeds within a dataset are not independent copies of it.
- The map is the paper's. The data collection, the continuous actions and the left-out move are not.
- With the paper's noise, "left out" means never commanded. The move still happens by accident and its outcome is in the data, attached to other commands.
- The pass marks (10 points, 60%, 0.05) are judgment calls made today and not reviewed by David.
- The map has a second route of equal length, so refusing the critical cell costs nothing from the start cell. The cost shows only from cells near it.
