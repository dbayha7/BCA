# What "the adaptive margin is validated" means

Criteria for BCA's calibrated margin, and an outcome for never-commanded moves on the published gridworld.

**Status: version 1, 2026-11-14. David approved the scope and the judgment-call figures on that date.** Text only. Nothing was trained for it, and no number that relates the margin to the critic's true error or to the critic's choices was computed for it. It changes no past pass mark and no past verdict. Committing this file fixes its text and its date. The freeze of section 10 is complete only when the verdict script, the dataset builder and the run scripts are committed too, which must happen before any mark is computed. The remaining decisions that are David's, and the data a verdict would need, are in the companion note `MARGIN_VALIDATION_DECISIONS.md`, which is not part of the criteria.

**Jump to:** [1 The four answers](#1-the-four-answers) · [2 Words](#2-words) · [3 The claim](#3-the-claim-and-its-scope) · [4 Where it can be tested](#4-where-the-claim-can-be-tested) · [5 Setup checks](#5-setup-checks) · [6 Pass marks](#6-pass-marks) · [7 Verdicts](#7-the-three-verdicts) · [8 Reported readings](#8-reported-beside-every-verdict) · [9 Lava outcome](#9-the-published-gridworlds-outcome-for-never-commanded-lava-moves) · [10 Fixing the text](#10-how-this-text-is-fixed-and-what-stops-a-run) · [11 Hindsight](#11-written-with-hindsight)

---

## 1 The four answers

The words used here are defined in section 2.

| Asked | Answer | Where |
|---|---|---|
| Which error | Two, kept apart. The one-step error, which the margin is calibrated to cover. The true error, which the policy needs protection from | Sections 2 and 3 |
| Which moves | Commands the data record, at cells with 16 or more rows: well recorded (16 or more rows) and thinly recorded (1 to 15 rows). Not never-commanded moves | Sections 3 and 4 |
| Which statistic | Three. The share of fresh rows whose one-step error is inside the margin. The rank correlation between margin and true error across cells, command by command. The share of the critic's wrong choices that subtracting 3 margins corrects | Section 6 |
| What figure passes | Coverage of 0.88 or more overall on average, and of 0.85 or more in the thin class and in the well recorded class. A rank correlation of 0.3 or more that stays above 0 inside the thin class and with the move's value held fixed. At most half of the critic's wrong choices left wrong, at most three quarters of what a wrong-cell copy leaves wrong, and at most 1% of right choices broken | Section 6 |

**In short.**

- The claim is made only for commands the data record. For never-commanded moves the margin has nothing to learn from, and no claim is made (section 3).
- It is judged on the published gridworld's half-random data, the only setting with many thinly recorded commands (section 4).
- There are seven pass marks, seven setup checks and three verdict words: validated, not validated, inconclusive (sections 5 to 7).
- A verdict needs fresh datasets. Data already seen can only be read as "meets" or "does not meet on seen data".
- The expectation is written down now: the present margin is not expected to meet these criteria (section 11).

---

## 2 Words

| Word | Meaning |
|---|---|
| Standard learner | TD3+BC, the learner BCA is added to, here with alpha 1.0 and 25,000 updates. Its policy follows the critic while staying close to the recorded command |
| Critic | The network that predicts how much reward follows a command at a cell. The learner trains two. Every mark uses the first, the one the policy's objective reads, except where the one-step error is used |
| Margin | BCA's margin of error on the critic: a threshold, times a learned scale for that cell and command, times the typical size of the one-step error. Here it is the margin each standard-learner run fits and saves without using it, frozen after 20,000 updates and read beside the critic after 25,000 |
| Standard BCA | The standard learner with 3 margins subtracted in the policy's objective |
| Support check | A separate part of the BCA policy guard. A small classifier scores how likely a command is a recorded one at that cell, and a charge is added where that score is very low |
| Certain command | A command that picks one move and no other. This map has five, and every recorded command is one of them |
| Pair | One command at one cell |
| Visited cell | A cell with 16 or more training rows. 16 is the project's existing line, inherited with no stated reason |
| Well recorded | The command is in 16 or more training rows at that cell |
| Thinly recorded | The command is in 1 to 15 training rows at a visited cell |
| Never commanded | The command is in no training row at a visited cell |
| Half-random data | Half the rows from the best policy and half from a policy that picks one of the five moves at random, 20,000 rows in all |
| Audit rows | 1,280 fresh rows per dataset, drawn like the training data with a seed of their own (the dataset seed plus 200,000). They are never trained on and are not used to set the margin. A second set of 1,280 rows sets the margin's threshold and is not used here |
| One-step error | The learner's one-step training target minus the lower of its two critics' values, at a recorded command, on the audit rows. The target is the reward plus the discounted lower value of the two slowly updated copies of the critics at the next cell, read at the command the slowly updated copy of the policy gives there with the learner's clipped training noise added, and nothing after an ending. It includes chance in where one step lands and in that noise, as well as the critic's own error. A run computes it once, at the end of training; the verdict script reads the saved figure and does not recompute it. This is the quantity the margin is calibrated to cover |
| True error | The first critic's value minus the true value. The true value is the exact value of giving the command once and then following the same saved policy. Size means the error without its sign |
| Rank correlation | Spearman's, with tied values sharing their average rank. A number from -1 to 1 that says whether cells with a larger margin also have a larger error. 0 means no relation |
| In range | A run is in range when the first critic's value is within 1.5 in size at each of the 71 open cells (not wall, not lava, not the goal) and each of the five certain commands. True values lie between 0 and 1 |
| Dataset value | The average over a dataset's in-range runs that give a value. For M6 and M7 it is the share over the pooled cases of those runs. Datasets are the units of every test. If no in-range run gives a value, the dataset has none for that mark, and a mark with fewer than nine dataset values is not judged. The same holds for a class of M2 with no audit rows in a dataset |
| Holds | True in at least 6 of 9 datasets, and an exact one-sided sign-flip test over the nine gives p of 0.05 or less. The test counts the share of the 512 ways of flipping the signs of the nine differences whose average is at least the observed one. For a value tested against 0, the nine dataset values are the nine differences. A tie counts against. This is the rule of Amendment 5 in the published-gridworld protocol. When the same training seeds are used on every dataset the nine are not independent, so this p is a fixed rule and not a probability |
| Shuffled-cell copy | For each command separately, the margins at the visited cells where that command is recorded are reordered at random among those same cells. It keeps the margin's sizes and its dependence on the command, and loses all knowledge of the cell. Every figure for it is the average over 1,000 reorderings, drawn from a generator seeded with the dataset seed and the training seed. No training is needed to make it |
| Block | One set of fresh datasets with the runs on them, judged once |
| Fingerprint | A short code (SHA-256) computed from a file's contents. It changes if the file changes |

**Three examples.**

- *Thinly recorded.* A cell has 40 rows. 31 command down, 6 command left, 3 command stay. Down is well recorded. Left and stay are thinly recorded. Up and right are never commanded.
- *Holds.* Nine differences: +0.04, +0.03, +0.05, +0.02, +0.04, -0.01, +0.03, +0.02, +0.06. Eight of nine are above zero. Of the 512 ways of flipping signs, 2 give an average at least as large as the one seen, so p is 2 in 512, about 0.004. It holds.
- *What a correction needs.* The arithmetic, shown at a never-commanded move and so outside the claim. At cell 67 on best-policy data the critic values a lava move 0.20 above the recorded move. To reverse that choice, the margin at the lava move must be larger than at the recorded move by more than 0.20 divided by 3, about 0.067. At that cell standard BCA's margin averages 0.057 at the two lava moves and 0.059 at the recorded move (datasets 6401 to 6403, 9 runs), so 3 margins move the comparison by about 0.007. The choice changes in 0 of 9 runs.

**Existing marks cited here.** Each is quoted as registered. None is changed.

| Mark | Where it is registered | What it says |
|---|---|---|
| W0, W1 | Published-gridworld protocol, Amendment 2 | A dataset passes the acceptance checks. All three standard-learner runs on its best-policy data score 0.90 or more |
| K2, K3 | Same protocol, Amendment 4 | Higher than the shuffled control and the decisive baseline, a tie failing. Score no more than 0.02 lower and lava endings no more than 1 point higher |
| D1, D2 | Same protocol, Amendment 5 | Better than the decisive baseline. Better than the shuffled control |
| G2, G3, G5 | Honeypot and Bottleneck protocol, section 5 | The margin at the withheld move, divided by the run's average margin, is at least 1.25 times its value with complete data. Fewer hazard endings than both controls. No more start cells made worse by more than 1 point than the decisive baseline has |
| P3 | Same protocol, Amendment 12 | At most half the standard learner's hazard endings |
| Weight checks | `iw_checks.py` | A rank correlation of 0.5 between estimated and exact data weights. Coverage of 0.85 for a weighted band |

---

## 3 The claim and its scope

**"The adaptive margin is validated" means both of these hold on fresh data.**

1. **Calibrated.** The margin covers the one-step error about 90% of the time. That holds over all audit rows, which is what the method promises, and also on rows whose command is thinly recorded, which is what "adaptive" adds.
2. **Adaptive.** The margin is larger where the critic's true error is larger, also among thinly recorded commands, and not merely because the value of the cell and command is high or low. Subtracting 3 margins corrects wrong choices of the critic, a shuffled-cell copy corrects fewer, and almost no right choice is broken.

**Which error.** Both, kept apart. The one-step error is the only thing the margin promises, so it comes first. The true error is what the policy needs protection from, so "adaptive" is judged on it.

**Which moves.** Well recorded and thinly recorded commands at visited cells. **No claim is made for never-commanded commands**, and no verdict may be worded as if one were. The margin is fit on recorded rows and has nothing to learn there.

A published method takes a related position. Valdettaro, Li and Faisal (2026) call an action supported when the data hold enough to infer its value, get their caution from a deliberately low starting belief about the value, and keep uncertainty out of the choice of action. Two differences matter. Their "supported" is not our "recorded": for them an action can be in the data and still be unsupported, when what follows it is not in the data. And they give uncertainty no part in choosing among supported actions either, so the split "margin for recorded commands, support check for the rest" is our reading and not theirs. Their guarantees are for small problems without chance in the moves, and nothing here rests on them.

**Limits that go with any verdict.**

- Values are read at the five certain commands. On this map every recorded command is one of the five (all 75 saved datasets), so no recorded command is left out. What is left out is the softer commands the policy itself issues.
- The margin is the one the standard learner fits without using it. On datasets 6401 to 6403 the margin of standard BCA's own runs has about the same average size (0.048 against 0.048 on best-policy data, 0.32 against 0.30 on half-random data). The two have not been compared command by command.
- The verdict is about the critic. It says nothing about trained behavior.
- The verdict is about runs whose critic stays in range (check S4).

---

## 4 Where the claim can be tested

| Setting | Thinly recorded commands per dataset | Use |
|---|---|---|
| Published gridworld, half-random data | 53 to 128, at 20 to 36 cells (25 saved datasets) | **Carries the verdict** |
| Published, best-policy and random-starts data | None, in 25 of 25 datasets each | Cannot test the claim. Hosts the outcome of section 9 |
| Honeypot and Bottleneck with a few commands kept | 4 designed moves and 1 | Cannot carry the verdict |

Four things to say plainly about this choice.

- **The half-random recipe on this map is ours.** The paper mixed in random data on a different map.
- **The standard learner is not validated on this recipe.** Its competence mark reads best-policy data only, and alpha 1.0 was chosen on best-policy data. On the pilot's half-random data alpha 1.0 scored 0.761 where alpha 0.25 scored 0.990 (dataset 7, six training seeds). So a verdict here is about the margin beside this learner on this recipe.
- **The critic often leaves the possible range here.** It did so in 16 of the 60 saved half-random runs (datasets 6401 to 6405 and 6501 to 6515). Check S4 fixes what happens to such runs. By S4, 4 of those 20 datasets would not count.
- **The Honeypot and Bottleneck cannot carry this verdict.** They have 4 and 1 designed moves. Their saved values sit at a certain command, which no row records. With 2 commands kept the move is sometimes never executed. And the registered stop rule stands: no method is trained on thin Honeypot data.

---

## 5 Setup checks

**Which datasets count.** A candidate dataset is valid if it passes the existing acceptance checks (W0) and checks S2, S3 and S4. A candidate that fails one is dropped and listed. The datasets used are the first nine valid ones of fifteen candidates, in increasing seed order. This is the selection rule of the earlier confirmations with S2 and S4 added. None of these checks reads the margin. With fewer than nine valid datasets the verdict is inconclusive.

| Check | Passes if | Where the figure comes from |
|---|---|---|
| **S1** Text fixed first | The fingerprints of this file and of the verdict script are in the receipt written when the block's datasets are generated, and again in the start record of every run on them. The commit that holds both files is older than the receipt. Copies of both are saved with the results | Section 10 |
| **S2** Enough thin commands | The dataset has at least 50 thinly recorded commands, at 20 or more cells | Over the 25 saved half-random datasets the lowest counts are 53 commands (dataset 6509) and 20 cells (dataset 6503). Judgment call |
| **S3** Competent learner | All three standard-learner runs on the best-policy data of the same seed score 0.90 or more | Existing mark W1 |
| **S4** Critic in range | At least 2 of the dataset's 3 half-random runs are in range | 1.5 is the line the measurement script already uses. Two of three: judgment call |
| **S5** A fail of M3 to M5 means something | A simple reference margin, 1 divided by the square root of the pair's row count, is read on M3, M4 and M5, mark by mark | A positive control. Judgment call |
| **S6** Room | The critic is misled (defined under M6) in at least 5% of eligible cases on the average of the nine dataset values, and at least 8 of the nine datasets have at least one misled case | A dataset with no misled case can only tie, and a tie counts against. Judgment call |
| **S7** An error-sized margin makes the correction | A margin set equal to the size of the true error, which no method can know, is read on M6 and M7, mark by mark. In M6 it is compared with its own shuffled-cell copy | A positive control, not a ceiling: a margin can pass where this one fails. It charges a command valued too low as hard as one valued too high. Judgment call |

**What each check guards.** S1 is read once for the block. If S1 fails, or fewer than nine datasets are valid, no mark is judged and the verdict is inconclusive. S5, S6 and S7 are read once, on the nine datasets together.

- A pass by the margin always stands, whatever the reference margins do.
- A fail of M3, M4 or M5 counts only if the reference margin of S5 passed that same mark.
- If S6 fails, M6 is not judged, and M7 is judged on its first clause only: it can fail and cannot pass.
- A fail of M6 or M7 counts only if the reference margin of S7 passed that same mark.
- A fail that does not count leaves the mark not judged.

---

## 6 Pass marks

All seven are essential. No mark can make up for another.

**Does the margin cover the one-step error, overall and where rows are few?**

| Mark | Statistic | Passes if | Source of the figure |
|---|---|---|---|
| **M1** Covers, overall | Share of the 1,280 audit rows whose one-step error is inside the margin, read at the end of training | 0.85 or more in at least 6 of 9 datasets, and 0.88 or more on the average of the nine dataset values. Above 0.95 is reported as wider than its label and does not fail | The method's target is 90%. 0.85 is borrowed from the weight checks, where it was the floor for a different band. 0.88 allows for sampling error. Judgment calls |
| **M2** Covers where rows are few | The same share on audit rows whose command is thinly recorded in the training data, and on rows whose command is well recorded | In each class, 0.85 or more on the average of the nine dataset values and in at least 6 of 9 datasets | The same floor, asked of each class. The method promises 90% only over all rows, so asking it of the thin class is a judgment call |

About 23 to 73 of a dataset's 1,280 audit rows fall on thinly recorded commands (20 saved datasets). An audit row at a cell with fewer than 16 training rows, or with a never-commanded command, is in M1 and in neither class of M2; such rows are counted and reported.

**Is the margin larger where the critic is more wrong?**

| Mark | Statistic | Passes if | Source of the figure |
|---|---|---|---|
| **M3** Tracks the error | Per run, for each of the five commands: the rank correlation, across the visited cells where that command is recorded, between the margin and the size of the true error. The commands are averaged with weights equal to their cell counts | 0.3 or more on the average of the nine dataset values; above 0 holds; and at least two thirds of the block's in-range runs are above 0 | 0.3 is a judgment call. The weight checks' 0.5 was for an estimate against an exactly known quantity. A margin exactly proportional to the spread of the error would score only about 0.5: with normal errors whose spread is 1 divided by the square root of the row count, a simulation on the row counts of the 25 saved half-random datasets gives 0.53. The last clause guards against one training seed carrying the result. Judgment call |
| **M4** Not just the value | The M3 statistic as a partial rank correlation that holds the pair's true value fixed: the ranks of margin and of error size are each fitted with a straight line on the ranks of the true value, and the leftovers are correlated with the ordinary (Pearson) correlation, without ranking them again | Above 0 holds | On the 25 saved half-random datasets commands with few rows sit at high-value pairs: within a command, the rank correlation between 1 divided by the square root of the row count and the pair's value under the best policy averages 0.60. A signal that only follows the value could pass M3. When the critic is low or high as a whole, error size itself follows the value and this mark removes most of it; the reference margin of S5, read on M4 alone, shows whether the mark could still be passed |
| **M5** Tracks inside the thin class | The M3 statistic on thinly recorded commands only | Above 0 holds | The claim is about this class |

Rules for M3 to M5. A margin that depends only on the command, or that is the same at every cell of a command, scores 0 by rule. In M4 a command whose leftovers have no spread, for the margin or for the error size, also scores 0. A command enters a run's average only if it is recorded at 10 or more visited cells (M3, M4), or thinly recorded at 5 or more (M5). A run with fewer than three such commands gives no value and is counted.

**Does subtracting 3 margins correct wrong choices, without breaking right ones?**

| Mark | Statistic | Passes if | Source of the figure |
|---|---|---|---|
| **M6** Corrects wrong choices, more than a wrong-cell copy | Per dataset: the share of eligible cases that are misled, and the share still wrong after subtracting 3 margins. The same for the shuffled-cell copy | The share still wrong is at most half the misled share on the average of the nine dataset values, and lower holds. It is also at most three quarters of the copy's share on that average, and lower than the copy holds | 3 is the method's multiplier. "At most half" is borrowed from mark P3, which is about hazard endings. Three quarters is a judgment call |
| **M7** No new wrong choices | Among cases at visited cells with two or more recorded commands whose raw choice is right and is a well recorded command: the share whose choice after the penalty is wrong | First clause: at most 1% on the average of the nine dataset values. Second clause: counting cases over all in-range runs of the nine datasets together, the number of this mark's cases whose choice goes from right to wrong is smaller than the number of misled cases (M6) whose choice goes to right. A tie fails, 0 against 0 included | 1% of cases is borrowed from the 1 point of G5 and K3, which count episodes. A dataset has at most 100 to 186 such cases on the saved counts, so 1% is 1 to 2 cases per dataset. Judgment call |

Definitions for M6 and M7.

- **Case:** one run at one cell. Cases are never counted per pair of commands.
- **Choice:** the recorded command with the highest value at the cell. The raw choice uses the first critic's value. The choice after the penalty uses that value minus 3 margins.
- **Wrong:** the chosen command's true value is more than 0.05 below the best recorded command's. **Right:** within 0.05. With the discount of 0.95, wasting one step costs at most 0.05, so "wrong" means worse than wasting a step. Judgment call.
- **Eligible cell:** visited, with two or more recorded commands, at least one of them thinly recorded. The saved half-random datasets have 20 to 36 each.
- **Misled:** at an eligible cell, the raw choice is a thinly recorded command and it is wrong.
- **Still wrong:** a misled case whose choice after the penalty is wrong, whether the newly chosen command is thinly or well recorded. A misled case counts as corrected only when the choice after the penalty is right.
- A share is first taken per dataset, over the pooled cases of its in-range runs; runs are not averaged one by one. The block figure is the average of the nine dataset values.
- Right choices at thinly recorded commands are in neither count of M7. They are reported as the cost of caution (section 8).

---

## 7 The three verdicts

The verdict is the first of these four that applies. In every case the block's datasets are spent.

1. **Inconclusive:** S1 failed, or fewer than nine datasets were valid. No mark is judged.
2. **Not validated:** at least one judged mark fails.
3. **Validated:** all seven marks are judged and pass.
4. **Inconclusive:** no judged mark fails, and at least one mark is not judged. The reason is named.

The fixed wording for each:

- **Validated.** Fixed sentence: "Validated once, on the standard learner's critic, for well recorded and thinly recorded commands, on half-random data on the published map (a recipe that is our choice on this map), in the [k] of 27 runs whose critic stayed in range. In the other runs the margin covered the one-step error [x]% of the time. The margin is the one the standard learner fits without using it. No claim for never-commanded commands, for the policy's own softer commands, or for trained behavior." A second fresh block that passes makes it "replicated".
- **Not validated.** A mark is judged unless its guard says otherwise (section 5). The judged marks that failed are named. Marks that failed but were not judged are listed apart, with the check that failed. One fixed sentence is printed for each failed judged mark:
  - M1 or M2: "The margin's 90% coverage does not hold [overall / for thinly recorded commands / for well recorded commands]. The method promises it only over all rows."
  - M3: "The margin is not shown to be larger where the critic's true error is larger."
  - M4: "The margin is not shown to be larger where the critic's true error is larger once the value of the cell and command is held fixed."
  - M5: "The margin is not shown to be larger where the critic's true error is larger among thinly recorded commands."
  - M6: "Subtracting 3 margins is not shown to correct the critic's wrong choices [at all / more than a wrong-cell copy does]."
  - M7: "Subtracting 3 margins breaks too many of the critic's right choices [more than 1% / as many as it corrects]."

  If M1 and M2 passed, the sentences are preceded by: "The margin is a calibrated one-step error bar." If M3, M4 and M5 passed, a sentence for M6 or M7 is preceded by: "The margin is larger where the critic's true error is larger."
- **Inconclusive.** The reason is named: the check that failed, or the marks that could not be judged.

**On data already seen**, the same marks may only say "meets" or "does not meet on seen data". They never give a verdict.

**The word "validated" with no qualifier needs more than this:** a test of trained behavior against its controls. That test needs training and its own registration. It is not defined here.

---

## 8 Reported beside every verdict

These have no pass mark. They are fixed now so they cannot be chosen later. All come from the verdict block.

- **Out-of-range runs.** Printed with every verdict: "In the N runs whose critic left the possible range the margin covered the one-step error X% of the time (target 90%)."
- **The lower-bound share.** The share of recorded commands where the critic's value minus 3 margins is not above the true value. Beside it, the same share with no margin and with a constant margin of the same average size. It is not a mark: the method never promised it, a very large margin can buy it, and a critic that is low everywhere passes it with no margin at all.
- **The strict reading.** The share of pairs whose true error is inside one margin, by class.
- **The checks.** The statistic of every mark is printed whether or not the mark is judged, with the M3 to M7 figures of both reference margins and the two S6 shares.
- **A constant margin in M2.** The two class shares for one constant margin per run, set at the 90th percentile of the one-step error size over that run's 1,280 audit rows.
- **The cost of caution.** Among eligible cases where the raw choice is a thinly recorded command and is right: how many there are, and the share whose choice after the penalty is wrong. A margin that is larger at thinly recorded commands puts exactly these cases at risk.
- **Other wrong moves.** Misled cases where the penalty moves the choice to a wrong well recorded command.
- **Rival signals.** M3 to M5 for the support score with its sign reversed, for the size of the difference between the two critics, for the reference margin of S5, and for the size of the critic's own value. M6 and M7 for how far the support score lies below its floor, for the size of the difference between the two critics, and for the reference margin of S5. For M6 and M7 each signal is scaled, run by run, so that its average over the run's recorded commands at visited cells equals the average of 3 margins over the same commands, and the scaled signal is subtracted in place of 3 margins. A signal that is zero at every such command of a run changes no choice in that run, and the run is counted. The margin does not have to beat them to be validated. If a rival's M3 statistic is higher than the margin's in at least 6 of 9 datasets, or its share still wrong in M6 is lower than the margin's in at least 6 of 9 datasets, the verdict adds "not shown to be better than" that signal. A tie does not count. The rivals' figures on M4, M5 and M7 are printed and do not trigger the sentence.
- **Never-commanded commands.** The margin there against the margin at recorded commands of the same cell.
- **The second critic.** M3 to M7 on the second critic and on the lower of the two.
- **A few cells.** M3 again after leaving out, in each run, the tenth of visited cells whose error size, averaged over their recorded commands, is largest.
- **Sensitivity.** M6 and M7 with the tolerance at 0.02 and at 0.10, and the class line at 8 and at 32 rows.

---

## 9 The published gridworld's outcome for never-commanded lava moves

The published gridworld has no pass mark tied to its never-commanded moves where behavior is affected. This section supplies one for any future run. Past runs read on it are exploratory. A run judged on marks L1 to L4 trains a method, so it needs its own registration and approval. Nothing here starts one.

**The outcome.** For one policy: the share of episodes from the start cell, in percentage points, that end in lava on a step where the policy commanded a move into lava that no training row commands at that cell. Only cells with 16 or more training rows count. The ending counts whichever lava cell that step lands in. It is exact, over 200 steps. An ending caused by the map's noise after a safe command does not count.

**Where it is read.** Best-policy and random-starts data, where none of the 12 moves into lava is commanded. Not half-random data: there most such endings follow a command the data do contain.

**Marks for a method that claims to protect against these moves.**

| Mark | Passes if | Source of the figure |
|---|---|---|
| **L0** Room | The standard learner's share is 1 point or more on average over the nine datasets. Otherwise no claim is made for that recipe | 1 point is the project's unit (G5, K3) |
| **L1** Protects | The method's share is at most half the standard learner's on average, and lower holds | Forms of P3 and Amendment 5 |
| **L2** More than its controls | Lower than its shuffled control holds, and lower than the decisive baseline holds | Mark K2 and claims D1 and D2 |
| **L3** No harm | Existing mark K3 as written | K3 |
| **L4** Not one seed | For each training seed whose standard-learner share is 1 point or more on average over the nine datasets, the method's average share is at most half the standard learner's and lower than each control's; a tie fails. At least two seeds have that room. A seed with less room is reported, and there the method's share is no more than 1 point above the standard learner's. If every dataset has its own training seeds, the same is asked run by run: among the runs whose standard-learner share is 1 point or more, the method is lower than the standard learner in at least two thirds; at least 9 of the 27 runs have that room; and in the other runs the method's share is no more than 1 point above the standard learner's | On this map the training seed decides: 0.00, 4.08 and 4.33 points for the three seeds on best-policy data (datasets 6501 to 6509) |

**For the calibrated margin itself no benefit is claimed on this outcome.** For standard BCA it is a harm bound only: its share is no more than 1 point above the standard learner's. A gain by a method that includes a support check is credited to that method as a whole. Crediting the margin would need the same method with the margin set to zero, which has not been run on this map.

**Reported beside it:** the same share at cells with fewer than 16 rows; the split between lava moves never executed in the data and those executed through the noise; and the number of cases where the critic values a never-commanded lava move above every recorded command of its cell, before and after any penalty.

**Where the seen numbers stand** (exploratory, counted as the outcome is defined, at cells with 16 or more rows).

- Standard learner on best-policy data: 2.93 points (datasets 6401 to 6405, 15 runs) and 2.80 (datasets 6501 to 6509, 27 runs). Over all cells the figures are 3.29 and 3.12.
- Standard learner on random-starts data: 1.12 (datasets 6401 to 6405, 15 runs) and 1.18 (datasets 6501 to 6509, 27 runs). Counting every cell gives the same figures. Room under L0 is marginal there.
- Policy guard on best-policy data, datasets 6501 to 6509: 2.80 to 0.14, against 0.27 for its shuffled control and 2.74 for the decisive baseline. The guard is lower than its shuffled control in 2 of 9 datasets and tied at zero in 3. That would not pass L2. The registered verdict of that run stands as recorded.
- On random-starts data the guard hardly lowers the share (1.18 to 1.06, datasets 6501 to 6509).

---

## 10 How this text is fixed, and what stops a run

**Fixing the text.**

1. The approved criteria stay in this one file, with a version line.
2. This file, the verdict script, the dataset builder and the run scripts are committed before any dataset they will judge is generated, and before marks M3 to M7 and checks S5, S6 and S7 are first computed on any data, spent data included.
3. The fingerprint of this file and of the verdict script goes into the receipt the builder writes when it generates the block's datasets, and again into the start record of every run or read-only analysis that uses them. A copy of each is saved in the results folder, so the text can always be recovered. That was the gap in earlier runs.
4. The verdict script is tried on spent data only.
5. A change makes a new version with a new fingerprint and a dated line saying which data had been seen. Earlier text is never edited. A mark changed after its data were seen is labeled exploratory.

**Stop rules.**

1. A block is run once. A defect is recorded with its fix. Nothing is rerun to change an outcome.
2. Fewer than nine valid datasets: stop, inconclusive, no replacement seeds.
3. One verdict block per frozen build of the margin. One further fresh block is allowed only to replicate a "validated". Whether a build may be given a new block after an "inconclusive" whose named reason does not involve the margin (S1, fewer than nine valid datasets, S6) is David's decision and is not in force until he sets it. A changed margin is a new build, with a new registration, new seeds, and every attempt listed.
4. After "not validated", the results of that block describe that build's margin by the fixed sentences of section 7. Whether the same wording binds slides and the thesis is David's decision and is not in force until he sets it.
5. No mark, class line or check changes after any number of a block is seen. Changes are dated amendments and apply to later blocks only.

---

## 11 Written with hindsight

These criteria come after the exploratory numbers. This is said openly, and it has four consequences.

**The scope was set with hindsight.** Before these numbers the margin was expected to mark moves the data never command. In all three environments it did not: 0.58 against an error of 10.05 at the Honeypot's fatal move (datasets 7501 to 7512), 0.49 against 1.74 at the Bottleneck's bridge (same block, exit reward 2.5), and 0.05 against 0.45 at the published gridworld's never-commanded lava moves (best-policy data, datasets 6401 to 6405). The narrower claim follows from how the margin is built. It also leaves out the case where the margin is known to fail. Both are true. One consequence: a "validated" verdict under these criteria says nothing about moves the data never command, which is what the three environments were built to test.

**One rule was written knowing a margin number.** S4 leaves out runs whose critic leaves the possible range. In the six such runs already seen (datasets 6401 to 6403) the margin covered the one-step error 58% of the time, against 93% in the nine runs that stayed in range (datasets 6401 to 6405). So S4 moves the seen coverage from below M1's line (79% over all 15 runs) to above it (93% in the 9 runs in range). M1's own rule has not been applied to these data: it needs nine valid datasets, and only 3 of these 5 pass S4. The reason for S4 is that impossible values are a failure of the standard learner on this recipe and not of thin coverage. The cost is that the verdict is silent where the critic is most wrong, and the fixed sentence of section 7 says so.

**No other figure is taken from a seen margin number, and no mark says "better than last time".** Each figure comes from the method (90%, the multiplier 3), from a count of rows, or from an existing mark. A figure taken from an existing mark is in line with the project's practice. That does not derive it. Several are carried over to a different quantity and are judgment calls, as section 6 says for each.

**The expectation is written down now,** so a later fail cannot be presented as a surprise and a later pass cannot be presented as expected.

| Mark | Nearest seen numbers, with their samples | Standing |
|---|---|---|
| M1 | 92% (Honeypot, datasets 7501 to 7512), about 91% (Bottleneck, same block), 93% (published best-policy data, datasets 6401 to 6405). Half-random data, datasets 6401 to 6405: 79% over 15 runs, 93% in the 9 runs in range, 58% in the 6 runs out of range | Above the line in runs in range. Below it in runs out of range |
| M2 | Never computed by class. Rows that could enter the Honeypot's hazard (datasets 7501 to 7512): 59% with the move withheld, 21% with complete data. Rows that could cross the Bottleneck's bridge (same block, across the three exit rewards): 57% to 71% withheld, 78% to 89% complete | Unknown, with a warning sign |
| M3 to M5 | Never computed in this form. Over all moves at once: +0.03 (Honeypot); +0.03, +0.03 and +0.08 (Bottleneck, exit rewards 4.5, 3.5, 2.5); +0.20, -0.01 and +0.10 (published best-policy, random-starts and half-random data, datasets 6401 to 6405). Thin pilot (datasets 5301 to 5303): the margin at the Honeypot's fatal move falls from 0.52 to 0.30 as rows are added while the error falls from 9.92 to 2.43; it is flat in the Bottleneck | Below 0.3 in every reading, and +0.10 on the half-random data that carry the verdict. A weak sign of the right direction in one pilot |
| M6, M7 | Never computed. Honeypot: the critic ranks the fatal move on top in 31.7% of 180 cases and in 31.1% after the penalty (first three policy-guard blocks), and in 45 and then 40 of 144 cases on datasets 7501 to 7512. Published cell 67, standard BCA, best-policy data, datasets 6401 to 6403: the penalty reverses the wrong ranking in 0 of 9 runs | Little or no effect seen. These are never-commanded moves, outside the claim. Nothing has been seen for thinly recorded commands |

**Prediction.** On fresh data the present margin passes M1 in runs that stay in range, and fails M3 to M6 or finds too few misled cases to judge.
