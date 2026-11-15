# Protocol: validating the overestimation and underestimation tests on the published map

Written November 10 2026, before any policy was trained on this testbed. David chose the published setup ("Kumar critical states") and approved the build ("Yes, build it"). **This protocol covers validation with plain baselines only. No BCA policy or control is trained under it.**

Quick navigation: [What is copied and what is ours](#1-what-is-copied-and-what-is-ours) · [The two tests](#2-the-two-tests) · [What is trained](#3-what-is-trained) · [What is measured](#4-what-is-measured) · [Pass marks and kill rules](#5-pass-marks-and-kill-rules) · [Rules for running](#6-rules-for-running) · [Amendment 1: sharper move rule](#amendment-1-a-sharper-preference-to-move-rule-chosen-by-a-pilot) · [Amendment 2: the paper's data recipes](#amendment-2-the-papers-own-data-recipes-and-competence-judged-per-dataset-written-before-any-of-its-datasets-was-generated) · [Amendment 3: the baseline's cloning weight](#amendment-3-the-baselines-cloning-weight-chosen-by-a-pilot-written-before-the-pilot) · [Amendment 4: Version A against the baseline](#amendment-4-version-a-against-the-baseline-on-the-published-map-written-before-any-version-a-policy-was-trained-here) · [Amendment 5: confirmation](#amendment-5-confirmation-of-version-a-on-untouched-datasets-written-before-any-of-its-datasets-was-generated) · [Amendment 6: first test of Version C](#amendment-6-first-test-of-version-c-written-before-any-version-c-policy-was-trained) · [Amendment 7: Version C2](#amendment-7-version-c2-the-cap-set-once-written-before-any-version-c2-policy-was-trained) · [Amendment 8: Version C3](#amendment-8-version-c3-no-charge-in-the-critics-target-written-before-any-version-c3-policy-was-trained) · [Amendment 9: instrumented run](#amendment-9-an-instrumented-run-of-version-c3-a-measurement-with-no-pass-marks-written-before-the-run) · [Amendment 10: Version C4](#amendment-10-version-c4-the-target-scored-without-the-noise-written-before-any-version-c4-policy-was-trained) · [Amendment 11: standard BCA](#amendment-11-standard-bca-on-the-published-map-written-before-any-standard-bca-policy-was-trained-here) · [Limits](#7-limits-known-in-advance)

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

## Amendment 2: the paper's own data recipes, and competence judged per dataset. Written before any of its datasets was generated

Written November 12 2026. David asked to return to the published map and get it through validation, and then said: "we need to utilize the published versions because why would we deviate". **Baselines only. No BCA policy or control is trained under this amendment.**

**What changes: everything of ours that the paper does not need.**

| Part | Before | Now |
|---|---|---|
| Data | 256 rows at every cell, half random commands, half the demonstrator's | **The paper's recipes**: runs of the optimal policy, and the two variants the paper describes |
| A move left out on purpose | The "over", "under" and "placebo" arms | **None.** The paper leaves no move out |
| Noise | 10% as published, and a no-noise variant | **10% only**, as published |
| Score | Goal rate and value against the best possible | **The paper's**: return as a share of the optimal policy's return |
| Sharpness of our preference-to-move rule | 2, then a pilot of 3, 4 and 6 | **2, untouched.** See below |

**The three data recipes.** The paper describes them in its section 5 and appendix E.1.

| Recipe | The paper's words | How it is built here |
|---|---|---|
| `optimal` | data from the optimal policy | Episodes of the optimal policy from the start cell |
| `other_starts` | "data generated from the optimal policy but with a different initial state distribution" | The same, each episode starting at a cell drawn evenly from the open cells |
| `mixed` | "interpolating the dataset with one generated by a random policy, where α is the proportion of random data" | Half the rows from the optimal policy and half from a policy that commands one of the five moves at random, both from the start cell |

In every recipe a row records the move that was commanded, written as the preference vector that commands it with certainty. The 10% noise then decides which move is executed.

**What is still ours, and why.**

1. **Continuous actions.** The paper's methods are tabular and choose among five moves. BCA is built on a neural learner with continuous actions, so the policy still outputs five preferences. This cannot be removed without rebuilding BCA as a different kind of method.
2. **Choices the paper does not state.** 20,000 rows per dataset. A random share of one half in `mixed`. Which other start cells `other_starts` uses: every open cell, evenly. Discount 0.95. Episodes of at most 200 steps, in the data and in evaluation; the last episode of a dataset is cut where the dataset is full. A cut is not an ending: the learner still looks ahead from the next cell. Only the goal and lava end an episode. The paper runs its mixing experiment on the "Multiple Critical" map; this testbed is the "Single Critical" map.

With the same seed, the optimal half of `mixed` is the first half of `optimal`, so the two differ only by the random rows.

**Why the sharpness stays at 2.** The first validation failed because the baseline's commands were too soft, and half of that data was uniformly random preferences. Under the paper's recipes every recorded command is a single clear move. If the baseline is still not competent, that is the result; no other sharpness is tried.

**What is trained.** Five candidate dataset seeds, 6301 to 6305, each with all three recipes. Training seeds 20, 21, 22. 45 TD3+BC baselines, 25,000 updates each, on the shared learner with five action dimensions. The learner's held-out rows (1,280 for its reference and 1,280 for its audit) are fresh rows of the same recipe, as before. Data label `published`. Results label `published_validation`.

**Pass marks, fixed before the data.** The score is the paper's: the chance of reaching the goal from the start cell within 200 steps, divided by the optimal policy's. Reaching the goal is the only reward and the paper states no discount, so this is the return as a share of the optimal return.

| Gate | Passes if | If it fails |
|---|---|---|
| **W0** The data are what the paper describes | Every dataset passes the acceptance checks of `published_checks.py` | That dataset's seed is dropped whole before training |
| **W1** The baseline is competent on optimal data | A seed is valid if all three of its baselines on the `optimal` recipe score at least 0.90. At least three of the five seeds are valid | **The testbed is not usable. No method is trained on it** |

Competence is judged per dataset, as in Amendment 9 of the earlier testbed: one weak dataset does not discard the rest, and the check never sees a method's result. The study's datasets are the first three valid seeds in increasing order, with all three recipes of each.

**Reported, without a pass mark.**

- The baseline's score on `other_starts` and on `mixed`. These are the conditions a method would be tested in, so there is nothing to pass.
- How often each move was commanded and executed at the two critical cells in each recipe.
- The critic's error at both critical cells, for the two fatal moves and for the way through. Positive means it rates the move too high. The first critical cell is row 6, column 7: lava left and right, the way through is down. The second is row 8, column 3: lava above and below, the way through is right.
- Lava endings and time-outs from the start cell, and the discounted value as a share of the best possible.

**A fact about the map, found on the test seed before any study dataset was generated.** With our discount of 0.95, the optimal route from the start cell runs down the left side and along row 8, through the **second** critical cell. The optimal policy reaches the goal 95.1% of the time from the start; about 4.5% of its runs end in the lava above or below that cell, pushed there by the noise. The first critical cell, which the earlier validation used, is on the route from other start cells: on the test seed `other_starts` has about 900 rows there and `optimal` about 40. So in `optimal` the moves the data never command at a well-visited cell are the two fatal moves at the second critical cell. This comes from the map and our discount, not from any trained policy. No pass mark depends on it, and both cells are reported.

**Rules.** Run once. No pass mark and no part of the environment is changed after outcomes are seen. A smoke test on dataset seed 7 checks only that the code runs. No BCA policy and no control is trained until David has seen this validation and says to go on. The earlier datasets, validation and pilot stay on disk and stay reported.

Limits known before the run: development seeds; the map and the rules are the paper's, the learner is not; the paper states neither dataset sizes nor the random share, so those are our choices; one map of the paper's three.

## Amendment 3: the baseline's cloning weight, chosen by a pilot. Written before the pilot

Written November 12 2026. The validation of Amendment 2 failed gate W1 ([VALIDATION_PUBLISHED.md](VALIDATION_PUBLISHED.md)). I proposed judging the testbed by copying the data instead. David declined: "we need to get the plain baseline to work right". **Gate W1 stays as written. Baselines only. No BCA policy or control is trained under this amendment.**

**Why the baseline falls short: one cause, traced in the 45 saved baselines of Amendment 2.** Nothing was retrained for this.

TD3+BC trains its policy on two terms: follow the critic, and stay near the recorded command. The paper's data hold five commands. At a cell where every row commands "down", the critic has no data on the other four commands, and its scores for them are guesses.

| What was measured | Result |
|---|---|
| `optimal` data: cells of the optimal route where the policy's highest preference is the optimal move | 100%, in 15 of 15 runs |
| The same policies, scored with the highest preference always commanded | 0.958 in 15 of 15 runs (0.753 as scored) |
| Where 100 or more rows all command one move: the policy's preference for it, and its highest other preference | +0.66 and -0.60. The rows say +1.00 and -1.00 |
| `optimal` and `other_starts`: cells with 16 or more rows where the highest preference is a wrong move | 199 |
| Of those, cells where no row commands the chosen move | 199 of 199 |
| Of those, cells where the critic rates the chosen move above the recorded one | 199 of 199, by 0.03 to 0.39 |
| `other_starts`: runs that never reach the goal | 3 of 15. Each commands "right" at the start cell, where 22 to 36 rows command "down" and none commands "right" |
| `mixed`: runs where the critic's scores leave the possible range | 9 of 15 |

**So every loss has the same source: the policy has moved toward a command that no row contains, because the critic's guess for it was higher.** A small move weakens the command, and our preference-to-move rule turns a weak command into random moves. A large move changes which move is commanded.

**Why the published setting allows it here.** TD3+BC multiplies the critic's term by a number, alpha, divided by the average size of the critic's scores. The published value is 2.5, chosen on robot-control benchmarks. Here the cloning term is an average over five preferences, so raising one preference from -1 to +1 costs 0.8. In the three failed `other_starts` runs a critic lead of 0.13 was enough to pay for it. The critic's guesses are off by more than that.

**The fix to test: a smaller alpha.** It is the one setting TD3+BC provides for this trade. Nothing else changes: the learner's code, the published game, the data recipes, our preference-to-move rule (sharpness 2) and gate W1.

**Not changed, and why.** Scoring with the highest preference always commanded would pass on `optimal` data (0.958). It would leave the three `other_starts` runs at 0.00, because it does not touch the cause.

**The pilot.** Dataset seed 7, which is not a study seed. All three recipes, six training seeds 0 to 5, 25,000 updates, baselines only. Candidates for alpha: 2.5, 1.0, 0.5, 0.25, 0.1. Results labels `pilot_alpha_<value>`.

**Rule, fixed now: use the largest alpha for which all six baselines on `optimal` data score at least 0.90.** The largest is the closest to the published value and keeps the most weight on the critic. If no candidate qualifies, I stop and report; there is no fresh validation.

Reported without a mark: the scores on the other two recipes, how many cells have a wrong highest preference, and whether the critic's scores stay in range.

**Prediction, written before the pilot.** 2.5 fails, as in the smoke runs. 1.0 is not enough. 0.5 qualifies.

**Then the validation, unchanged.** Gates W0 and W1 exactly as in Amendment 2, on five new dataset seeds 6401 to 6405, training seeds 20, 21, 22, with the alpha the pilot chose. Data label `published2`. Results label `published_validation_2`. Run once. No BCA policy and no control is trained until David has seen it and says to go on.

**How alpha is set.** The shared learner reads it from one constant when the update is first compiled. `run_published_alpha.py` sets that constant before anything is compiled, records it in the run's START file, and checks that the policy's objective uses it. `learner.py` is not edited.

Limits known before the pilot: one dataset seed in the pilot; alpha is chosen after seeing baseline results on seeds 7 and 6301 to 6305; a smaller alpha moves the baseline toward plain cloning, which leaves less for any method to add; the published alpha was never tuned for five-command data, and neither is ours beyond these five candidates.

## Amendment 4: Version A against the baseline on the published map. Written before any Version A policy was trained here

Written November 12 2026. The baseline passed gate W1 on fresh datasets under Amendment 3 ([VALIDATION_PUBLISHED.md](VALIDATION_PUBLISHED.md), section 9). David saw that result and said: "Yes, test it". **This is the first BCA training on this testbed. It is a first test, not a confirmation**: the marks below are written before any Version A policy exists here, but after seeing the baselines on these same datasets.

**The method, unchanged.** Version A as fixed in the earlier testbed (`support_blind_width`): the support raise is read at the true state, the calibrated width at a shuffled state, the penalty acts in the policy's objective only, strength 3. It starts from the baseline's saved state after 10,000 updates and trains to 25,000, with the reference refreshed after 15,000 and 20,000. Nothing in it is tuned for this map.

**The baseline it is added to** is the validated one: TD3+BC with alpha 1.0. One consequence, stated now: the learner multiplies the penalty by the same alpha, so against the cloning term the penalty weighs 2.5 times less than in the earlier testbed. The strength of 3 is not retuned.

**Controls.**

| Control | What it is | What it rules out |
|---|---|---|
| Shuffled control (`support_shuffled`) | The same penalty with its width read at shuffled states | That any penalty of this size would do |
| Decisive baseline | The baseline's own top choice at every cell, commanded with certainty. No training | That Version A only makes commands more decisive |

**A correction to this control, made before any Version A policy was trained, the smoke test included.** I first wrote the control used in the earlier testbed: the baseline's commands scaled up until their average strength matches Version A's. While testing the judging code I found that it does not work on this map. On the half-random data the baseline's preferences are often all negative, and scaling then pulls the top preference down toward the others instead of away from them, so the scaled baseline is less decisive, not more. A control that cannot do its job would make K2 too easy. The decisive baseline above is the strictest form of the same idea and cannot be beaten by decisiveness alone. The scaled version is still reported, without a mark.

**Data.** The study's datasets under Amendment 2's rule: dataset seeds 6401, 6402, 6403, all three recipes, training seeds 20, 21, 22. 27 baselines exist already (`results/published_validation_2`). Trained now: 27 Version A policies and 27 shuffled controls. The baseline is also continued from the same saved state as a check that the runner reproduces it. Results label `version_a_published`.

**The score** is the paper's, as before: the chance of reaching the goal from the start cell within 200 steps, as a share of the optimal policy's. A "dataset" is the mean over its three training seeds.

**Where the baseline stands on these three datasets.** `optimal` 0.942, `other_starts` 0.987, `mixed` 0.801. So the room for a method is on `mixed`, the half-random data.

**Marks, fixed now.**

| Mark | Passes if |
|---|---|
| **K1** Helps where there is room | On `mixed`, Version A scores higher than the baseline in at least 2 of 3 datasets, and its mean over the nine runs is at least 0.05 higher. 0.05 is a quarter of the gap between the baseline and the best possible |
| **K2** It is Version A's checks, not a side effect | On `mixed`, Version A scores higher than its shuffled control and higher than the decisive baseline, each in at least 2 of 3 datasets. A tie fails |
| **K3** No harm where the baseline is good | On `optimal` and on `other_starts`: Version A's mean score is no more than 0.02 below the baseline's, and its share of episodes ending in lava is no more than 1 point above the baseline's |

**Decision rule.** Version A helps on the published map if K1, K2 and K3 all pass. If K1 fails it does not help here, and I report that. If K3 fails it is rejected here whatever K1 says.

**Reported without a mark.**

- In how many `mixed` runs the critic's scores leave the possible range, baseline against Version A.
- At both critical cells: how often a fatal move is commanded, and the width Version A charges for the fatal moves against the way through.
- Scores on `optimal` and `other_starts` per dataset, lava endings and time-outs everywhere, command strength.
- Whether the continued baseline reproduces the saved one.

**Predictions, written before the run.** K3 passes. On `mixed` I expect Version A to score higher than the baseline, because off-corner commands are outside the data and the support raise should push the policy back onto recorded commands. I am not confident of the size. K2 is the mark most at risk: the decisive baseline may do as well wherever the critic stayed in range.

**Rules.** Run once. No mark is changed after outcomes are seen. A pass here would be a reason to confirm on untouched datasets, nothing more.

Limits known before the run: three datasets with three training seeds each; the marks were written after seeing the baselines on these datasets; one map; alpha was chosen for the baseline's competence, not for Version A; the calibrated width is fitted on held-out rows of the same recipe and carries no guarantee at commands the data never contain.

## Amendment 5: confirmation of Version A on untouched datasets. Written before any of its datasets was generated

Written November 12 2026, after the first test ([RESULTS_VERSION_A.md](RESULTS_VERSION_A.md)). David saw that result and said: "Yes, confirm it". **This is a confirmation: the method, the baseline, the controls, the claims and the marks are all fixed here, before the datasets exist.**

**Nothing in the method or the baseline changes.** Version A as in Amendment 4. TD3+BC with alpha 1.0. The shuffled control and the decisive baseline as in Amendment 4. The paper's map, noise, three data recipes and score.

**Datasets, and which are used.** Fifteen candidate dataset seeds, 6501 to 6515, each with all three recipes. The procedure is the one used in the earlier testbed's confirmations:

1. A seed whose three datasets do not all pass the acceptance checks is dropped (gate W0).
2. The baselines are trained on every remaining candidate. A seed is valid if all three of its baselines on `optimal` data score at least 0.90 (gate W1's rule, unchanged).
3. **The first nine valid seeds in increasing order are the confirmation's datasets.** This choice uses the baselines alone; no Version A policy exists when it is made. Fewer than nine valid seeds stops the run, and no Version A policy is trained.

**Training seeds 20, 21, 22**, the ones the baseline was validated with. I keep them on purpose. The baseline's score depends on the training seed more than on the dataset, and a new seed that happened to be weak would fail every dataset at once and stop the run. The datasets are what is untouched here. That is a limit, stated below.

**What is trained.** 135 baselines. Then, on the nine datasets: 81 Version A policies and 81 shuffled controls, and the baseline continued from its saved state as a check on the runner. Data label `published3`. Results labels `confirm_baselines` and `confirm_version_a`.

**How a comparison is judged.** For each dataset, take the mean score of its three training seeds for each policy, and the difference between two policies. A comparison "holds" if both of these are true:

- the first policy is higher in at least 6 of the 9 datasets;
- an exact one-sided test on the nine differences gives p of 0.05 or less. The test counts the share of all 512 ways of flipping the signs of the differences whose mean is at least the observed mean.

The first test showed why both are needed: a count alone passed on differences of 0.003.

**Claims.**

| Claim | Holds if |
|---|---|
| **C1** Version A helps on the half-random data | On `mixed`, Version A against the baseline holds, and the mean gain is at least 0.05 |
| **C2** Version A helps on optimal data | On `optimal`, Version A against the baseline holds, and the mean gain is at least 0.02 |
| **C3** No harm where there is no room | On `other_starts`, Version A's mean score is no more than 0.02 below the baseline's, and its share of episodes ending in lava is no more than 1 point above |
| **D1** More than decisiveness | Version A against the decisive baseline holds on `mixed` and on `optimal` |
| **D2** Reading the penalty at the true cell matters | Version A against its shuffled control holds, judged separately on each of the three recipes |

**Decision rule.**

- **Version A is confirmed on the published map if C1, C2 and C3 all hold.** If C1 or C2 fails, that part is not confirmed, and I report which. If C3 fails, Version A is not confirmed whatever the others say.
- D1 and D2 do not decide that. They say **why** it helps. If D1 fails on a recipe, the gain there is no more than decisiveness would give. If D2 fails on a recipe, a penalty read at the wrong cells does as well there.

**Predictions, written before the datasets exist.**

- C2 and C3 hold. C1: the comparison holds; the 0.05 mean gain is at risk, because about one `mixed` run in nine collapsed in the first test.
- D1 holds on `optimal`. On `mixed` it is uncertain: the decisive baseline scores 1.00 wherever the baseline's top choices are right.
- **D2 fails on `mixed`.** With half-random data the same five commands are recorded at nearly every cell, so the cell should not matter. It may hold on `other_starts` and is uncertain on `optimal`.

**Reported without a mark.** How many runs score below 0.50 for each policy. In how many `mixed` runs the critic's scores leave the possible range. Fatal commands and Version A's charges at both critical cells. Whether the continued baseline reproduces the saved one. The scaled baseline of the earlier testbed.

**Rules.** Run once. No claim, mark or dataset choice is changed after any outcome is seen. A smoke test is not repeated; the runners are the ones already checked. Seeds 6501 to 6515 are spent after this run.

Limits known before the run: the training seeds are reused from the validation, so only the datasets are new; alpha 1.0 was chosen on a pilot and passes the competence mark narrowly for one training seed; one map of the paper's three; the learner is ours, not the paper's; this run does not test which part of Version A does the work beyond D1 and D2.

## Amendment 6: first test of Version C. Written before any Version C policy was trained

Written November 12 2026. Version C is importance-weighted BCA that acts on the Q values; its definition, the four checks and what they found are in [BCA_IW_PLAN.md](BCA_IW_PLAN.md). David approved the version ("Yes, build it"), then, after seeing the checks and the two marks I asked to read differently, the training ("Yes, train it", and "proceed" after I explained why the two checks failed). **A first test, not a confirmation**: the marks are fixed here before any Version C policy exists, but the baseline's and Version A's results on these datasets have been seen.

**The method** is `version_c.py` as it stands: the corrected weight classifier, trained 10,000 updates before the run starts; the table of thresholds from `calibration/wbcp.py` at every refresh; the charge; its three uses, strength 3. It starts from the baseline's saved state after 10,000 updates and trains to 25,000, like Version A. Nothing is tuned on this map.

**Data.** The three study datasets (seeds 6401, 6402, 6403), three recipes, training seeds 20, 21, 22: 27 Version C policies. The baseline (`results/published_validation_2`) and Version A (`results/version_a_published`) on the same datasets and seeds are the comparisons. Results label `version_c_published`. A smoke test on dataset seed 7 checks only that the code runs.

**Marks, fixed now.** A "dataset" is the mean over its three training seeds. Means are over nine runs.

| Mark | Passes if |
|---|---|
| **V1** The Q values are corrected | At the first critical cell, on `optimal` and on `other_starts` data, the critic rates a fatal move above the way through in at most 1 of 9 runs. For the baseline and for Version A it is 9 of 9 |
| **V2** Better than the baseline | On `optimal`: mean score at least 0.02 higher, and higher in at least 2 of 3 datasets. On `mixed`: at least 0.05 higher, and higher in at least 2 of 3 datasets |
| **V3** At least as good as Version A where Version A is strong | On `optimal` and on `other_starts`: mean score no more than 0.02 below Version A's, and the share of episodes ending in lava no more than 1 point above |
| **V4** Better than Version A where Version A is weak | On `mixed`: mean score higher than Version A's, and higher in at least 2 of 3 datasets |

**Decision rule.** Version C passes its first test if V1, V2, V3 and V4 all pass; that is a reason to confirm it on untouched datasets, nothing more. **If V1 fails, the direct correction does not correct the Q values, and I report that plainly.** If V3 fails, acting on the critic costs more than it gains here, as Version B did in the earlier testbed.

**Reported without a mark.** Runs scoring below 0.50. Runs whose critic leaves the possible range. The critic's error at the fatal moves, and how often a fatal move is commanded, at both critical cells. The charge at the fatal moves and at the way through. Calibration use. The decisive baseline.

**Predictions, written before the run.** V1 passes. V2 passes on `optimal`; on `mixed` I am unsure. V3 is the mark most at risk: changing what the critic is trained toward did harm before. V4 is a coin flip.

Limits known before the run: three datasets with three training seeds; the marks follow two earlier looks at these datasets; Version C's update step matches the shared learner's to float rounding only, so its runs are compared with baseline and Version A runs from the shared step; the weight classifier's comparison set uses random cells, which suits a map of 100 cells and is not a general recipe; on the half-random data about 40% of commands missing by chance stay certifiable, so their Q values are not pushed down; the width still measures one-step error.

## Amendment 7: Version C2, the cap set once. Written before any Version C2 policy was trained

Written November 12 2026, after Version C failed its first test ([RESULTS_VERSION_C.md](RESULTS_VERSION_C.md)). David saw that result and the proposed fix and said: "Yes, fix and test". **Development, not confirmation**: the fix was chosen after seeing Version C fail on these same datasets.

**The one change.** In Version C the cap was half the range of the critic's own held-out targets, recomputed at every refresh, and it fed on itself: 0.50, then 4.4, then 30. **In Version C2 the cap is set once, at update 10,000, when the run is still the plain baseline, and is never recomputed.** Everything else is Version C exactly: the corrected weight classifier, the table of thresholds, the charge, its three uses, strength 3, the same starting states.

**Data and comparisons.** As in Amendment 6: the three study datasets (seeds 6401, 6402, 6403), three recipes, training seeds 20, 21, 22; 27 Version C2 policies; compared with the baseline and Version A on the same datasets and seeds. Results label `version_c2_published`. A smoke test on dataset seed 7 checks only that the code runs.

**Marks: V1 to V4 of Amendment 6, unchanged, and the same decision rule.** Version C2 passes if all four pass; that is a reason to confirm on untouched datasets, nothing more.

**Reported without a mark.** The cap, the residual unit and the range of the critic's scores at each refresh, to show whether the loop is gone. Everything Amendment 6 lists.

**Predictions, written before the run.** The cap no longer moves, by construction. V1 and V3 pass, as for Version C. On the half-random data I expect a clear improvement over Version C's 0.527; whether it beats the baseline by 0.05 (V2) and Version A (V4) I do not know. The soft spot found in the checks is still there: commands missing by chance on that recipe sit near the "cannot certify" line.

Limits known before the run: a third look at these three datasets; one change made after a failure; the rest as in Amendment 6.

## Amendment 8: Version C3, no charge in the critic's target. Written before any Version C3 policy was trained

Written November 12 2026, after Version C2 passed three of four marks ([RESULTS_VERSION_C.md](RESULTS_VERSION_C.md), section 7b). David saw that result and the proposed change and said: "Yes, test it". **Development, not confirmation**: a fourth look at these three datasets, with a change chosen after seeing Version C2 on them.

**The one change.** Version C2 used the charge in three places. **Version C3 drops the second: the critic's target is the plain one again.** The next step's value is no longer lowered by the charge. What stays:

1. the direct correction: at commands that cannot be certified, the critic's score may not stay above the recorded command's target minus the charge;
2. the policy chooses by Q value minus the charge.

The cap stays set once, as in Version C2. Everything else is unchanged.

**Why.** In Version C2 the critic's scores still drifted far off scale (fatal moves near -10 to -25) although the cap no longer moved. My reading was that the target charged the policy's next command after TD3's added noise, at almost every step, and that it added up over an episode. This run tests that reading: if it is right, the critic's scores should now stay near the scale of the true values.

**Data and comparisons.** As in Amendments 6 and 7: seeds 6401, 6402, 6403, three recipes, training seeds 20, 21, 22; 27 policies; against the baseline and Version A on the same datasets and seeds. Results label `version_c3_published`. A smoke test on dataset seed 7 checks only that the code runs.

**Marks: V1 to V4 of Amendment 6, unchanged, and the same decision rule.**

**Reported without a mark.** The range of the critic's scores and the residual unit at each refresh. The critic's error at the fatal moves. Everything Amendment 6 lists.

**Predictions, written before the run.** The critic's scores stay within a few units of the true scale: with the cap at 0.5 the direct correction can push a score at most 1.5 below a target. V1 passes. V3 passes. V2 passes on optimal data. On the half-random data I expect at least Version C2's 0.924 and fewer lava endings than its 12.1%; whether it beats Version A in 2 of 3 datasets (V4) I do not know.

Limits known before the run: a fourth look at these three datasets; the rest as in Amendment 6.

## Amendment 9: an instrumented run of Version C3. A measurement, with no pass marks. Written before the run

Written November 12 2026, after Version C3 ([RESULTS_VERSION_C.md](RESULTS_VERSION_C.md), section 7c). David said: "Yes, measure it". **No new method and no claim.** Version C3 is trained again exactly as before on the same three datasets and seeds, and should reproduce its policies. At each refresh extra read-outs are taken on the held-out rows. Results label `diag_version_c3`.

**The question.** In Version C3 the critic's scores still drift below the true values, step after step. Two fixes each removed about half of the drift, and both causes were guessed afterward. This run measures instead.

**What is recorded, per run, at updates 10,000, 15,000 and 20,000.**

| Read-out | What it tells |
|---|---|
| Share of the target's next commands that cannot be certified, with TD3's added noise and without it | Whether the noise carries the command the target scores into the region the direct correction pushes down |
| The critic's score at the next command with the noise minus without it | How much the target loses to that |
| Share of the direct correction's comparison commands that cannot be certified | How much of the command space is being pushed down |
| The critic's error at recorded commands, and at the policy's own next command, against the exact value of the current policy | The size of the drift itself, where it matters |

**What each outcome would mean.**

- Noisy next commands are often flagged and score much lower than un-noised ones: the direct correction leaks into the target through the noise. The fix is to score the target at the un-noised command, or to leave commands within the noise of a recorded one alone.
- The policy's own un-noised commands are often flagged: the weight classifier draws the line too tightly around recorded commands. The fix is in the classifier.
- Neither: the push-down spreads through the critic network itself, and the fix has to limit how far below a recorded command an uncertified one may be pushed.

Nothing is changed in response until David has seen the measurements.

## Amendment 10: Version C4, the target scored without the noise. Written before any Version C4 policy was trained

Written November 12 2026, after the instrumented run ([RESULTS_VERSION_C.md](RESULTS_VERSION_C.md), section 7d). David saw the measurement and said: "Yes, test it". **Development, not confirmation**: a fifth look at the same three datasets.

**The one change from Version C3.** TD3 adds a little noise to the next command before the critic's target scores it. In Version C4 the target scores the next command as the policy gives it, with no noise added. Everything else is Version C3: the corrected weight classifier, the table of thresholds, the direct correction, the charge in the policy's choice, the cap set once.

**Why.** The instrumented run measured it: with the noise the next command scores about 0.5 lower than without, at every step, and the critic's error at recorded commands grows from about -0.1 to about -3 (optimal data) and -14 (half-random data).

**Data, comparisons and marks.** As in Amendments 6 to 8: seeds 6401, 6402, 6403, three recipes, training seeds 20, 21, 22; against the baseline and Version A on the same datasets and seeds. **Marks V1 to V4 of Amendment 6, unchanged, and the same decision rule.** Results label `version_c4_published`. A smoke test on dataset seed 7 checks only that the code runs.

**Reported without a mark.** The read-outs of Amendment 9 at each refresh, to show whether the drift is gone: the critic's error at recorded commands, and its score at the next command with the noise minus without.

**Predictions, written before the run.** The critic's error at recorded commands stays within about half a unit of the true values on optimal and other-starts data. V1 and V3 pass. V2 passes. On the half-random data the score is at least Version C3's 0.956; the drift there may shrink less, because the policy's own next command is flagged 13% of the time at thin cells. V4: I do not know.

Limits known before the run: a fifth look at these three datasets; removing the noise also removes a smoothing that TD3 uses for stability; the rest as in Amendment 6.

## Amendment 11: standard BCA on the published map. Written before any standard BCA policy was trained here

Written November 13 2026, the morning of the research meeting. David asked for the standard BCA score in the setup-validation table ("need to include standard bca score as well") and, told that it had never been trained on this map, chose "Train it now". He also said: "dont get too ahead with the versions we need to focus on the setup validation". **Development, not confirmation**: a sixth look at the same three datasets, on which the baseline and five other methods have already been seen.

**Why it was missing.** Version A was carried over from the earlier testbed and tested here first (Amendment 4). Standard BCA, the method the later versions build on, was skipped on this map.

**The method.** The shared learner's plain `bca` arm, unchanged: BCA's calibrated width, with the standard (equal-weight) calibration, read at the true state and command, charged in the policy's choice with strength 3. No support check, nothing in the critic. Its control is the learner's own `shuffled_width` arm: the same charge read at shuffled states.

**How it is run.** Exactly as Version A was in Amendment 4: on the baseline with alpha 1.0, from the baseline's saved state after 10,000 updates to 25,000, with the reference refreshed after 15,000 and 20,000. Runner `run_standard_bca_published.py`, a wrapper that edits no earlier file. The baseline is continued the same way as a check that the runner reproduces it.

**Data.** The three study datasets (seeds 6401, 6402, 6403), three recipes, training seeds 20, 21, 22: 27 standard BCA policies and 27 controls. Results label `standard_bca_published`. A smoke test on dataset seed 7 checks only that the code runs; its scores are not looked at.

**Marks.** K1, K2 and K3 of Amendment 4, unchanged, with standard BCA in Version A's place and `shuffled_width` as its shuffled control. The same decision rule: standard BCA helps on the published map if all three pass. Judged by `judge_standard_bca.py`, which is `judge_version_a.py` with the method names switched.

**Reported without a mark.** The scores on all three recipes beside the baseline's, for the validation table. Whether the critic still rates a fatal move above the way through at the first critical cell. The width charged at the critical cells.

**Predictions, written before the run.** The width diagnostic of November 12 found BCA's calibrated width nearly flat on this map (about 0.05 everywhere, 0.20 on the half-random data) and unrelated to the critic's error. So I expect standard BCA to change little. K3 passes (no harm). K1 fails: the gain on the half-random data is under 0.05. K2 fails or ties: the shuffled control does as well. A small gain on optimal data is possible, as in the earlier testbed, where the width acted as a pull toward decisive commands. The critic's ranking at the first critical cell stays wrong in 9 of 9 runs, because standard BCA does not act on the critic.

Limits known before the run: a sixth look at these three datasets; three datasets with three training seeds each; the baseline's alpha was chosen for its own competence, not for BCA.

## 7 Limits known in advance

- Development seeds. Three datasets with three training seeds each; seeds within a dataset are not independent copies of it.
- The map is the paper's. The data collection, the continuous actions and the left-out move are not.
- With the paper's noise, "left out" means never commanded. The move still happens by accident and its outcome is in the data, attached to other commands.
- The pass marks (10 points, 60%, 0.05) are judgment calls made today and not reviewed by David.
- The map has a second route of equal length, so refusing the critical cell costs nothing from the start cell. The cost shows only from cells near it.
