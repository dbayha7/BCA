# Decisions and data for the margin's validation criteria

Companion to `MARGIN_VALIDATION_CRITERIA.md`. **Not part of the criteria.** This note holds what changes as David decides, so that recording a decision does not change the text that will be fingerprinted.

Written 2026-11-14. Nothing here has been run, generated or trained, and nothing here starts a run.

**Jump to:** [1 Decisions](#1-decisions-that-are-yours) · [2 Spent data](#2-which-data-are-spent) · [3 What can be read without training](#3-what-can-be-read-on-spent-data-without-training) · [4 What a verdict would need](#4-what-a-verdict-would-need) · [5 Honeypot and Bottleneck](#5-one-time-readings-on-spent-honeypot-and-bottleneck-data) · [6 How it was written](#6-how-the-criteria-were-written) · [7 The verdict script](#7-the-verdict-script)

---

## 1 Decisions that are yours

Decisions 1, 3, 7, 8, 10 and 11 have a default, which is what the criteria say today. The others are open. Figures can be changed freely now. Once any of marks M3 to M7 or checks S5 to S7 has been computed on any data, a change is a dated amendment, applies to later blocks only, and is labeled as made with those data seen.

**Recorded so far.** Decision 1, the scope: accepted on 2026-11-14. Decision 2, the judgment-call figures: kept as written on 2026-11-14. The criteria already said both. They were then committed as version 1 on 2026-11-14, together with this note. The verdict script was written on 2026-11-15 (`gridworld_critical_v1/margin_verdict.py`). It was tested on made-up numbers only and has never been run on a saved run or dataset. It, the dataset builder and the run scripts are not yet committed, so the freeze of the criteria's section 10 is not complete. The four readings of the script that can change a verdict (section 7) were accepted on 2026-11-15. Decisions 3 to 11 are still to be made.

| # | Decision | Default in the criteria |
|---|---|---|
| 1 | **The scope.** Is "validated for recorded commands only" acceptable as the project's position? Its consequence: a "validated" margin says nothing about moves the data never command, which is what the three environments were built to test | Recorded commands only. No claim is made for never-commanded moves. **Decided: accepted by David on 2026-11-14** |
| 2 | **The judgment-call figures** (listed below) | As written. **Decided: kept as written by David on 2026-11-14** |
| 3 | **Are M1 and M2 judged on all runs or only on runs in range?** The 90% promise does not depend on the critic staying in range. On the seen half-random runs of datasets 6401 to 6405 the difference is 79% (all 15 runs) against 93% (the 9 runs in range) | In-range runs, with the other runs' coverage printed in the verdict sentence |
| 4 | **Training seeds for a fresh block.** The same three on every dataset, or three of its own for each dataset (27 in all) | Open. Own seeds make the nine datasets independent, which the test assumes. They also raise the chance that the competence check drops datasets |
| 5 | **Whether to spend a fresh block on the present margin,** whose marks M3 to M6 are predicted to fail ("not validated" if the reference margins of S5 and S7 pass, "inconclusive" if they do not), or to read the marks on spent data first and keep fresh datasets for a later build | Open |
| 6 | **Whether checks S5, S6 and S7, which do not use BCA's margin, are read on spent data after the freeze and before any fresh dataset is generated.** That would show whether this map can host the test at all. Reading them before the freeze would count as seeing a number | Open |
| 7 | **Whether the half-random recipe needs its own competence mark for the standard learner** before any margin is judged on it | No such mark today. Check S3 reads best-policy data |
| 8 | **Whether a pass figure is set for the cost of caution** at a thinly recorded good command. It is reported without a mark, because the number of such cases is not yet known | Reported only |
| 9 | **How the freeze is anchored.** Committing the criteria, the verdict script, the dataset builder and the run scripts, and whether the date is also recorded outside the repository | Open. The builder and run scripts are not in version control today |
| 10 | **Whether the wording of a "not validated" verdict also binds slides and the thesis** | Not in force until you set it |
| 11 | **Whether a build may get a new fresh block after an "inconclusive"** whose reason does not involve the margin (the text was not fixed first, fewer than nine valid datasets, or no room) | Not allowed until you set it |

**The judgment-call figures (decision 2).**

| Figure | Where | Note |
|---|---|---|
| 0.85 and 0.88 | M1, M2 | 0.85 comes from the weight checks, where it judged a different band. The method promises 90% only over all rows |
| 0.3 | M3 | A margin exactly proportional to the spread of the error would score about 0.5 on the saved row counts, so 0.5 would leave no room |
| 0.05 | Right and wrong in M6 and M7 | Worse than wasting one step. 0.02 and 0.10 are reported as sensitivity |
| Half, three quarters | M6 | "Half" is the form of mark P3, which is about hazard endings |
| 1% of cases | M7 | Borrowed from the 1 point of G5 and K3, which count episodes |
| 16 rows | Every class | Inherited with no stated reason. 8 and 32 are reported as sensitivity |
| 50 commands at 20 cells | S2 | The lowest seen counts are 53 and 20 |
| 1.5, two of three | S4 | 1.5 is the measurement script's line |
| The two reference margins | S5, S7 | Positive controls |
| 5%, eight of nine | S6 | Room for the decision marks |
| 10 cells and 5 cells per command, three commands | M3 to M5 | The smallest numbers for a rank correlation and for a run's average |
| Two thirds of the block's in-range runs | M3 | Guards against one training seed carrying the result |
| 1,000 reorderings | The shuffled-cell copy | The number of random reorderings averaged |
| 6 of 9, nine of fifteen | M1, M2, "holds", the rival signals, section 5 | The form of Amendment 5 |
| 1 point, half, two seeds, two thirds, 9 of 27 | L0, L1, L4 and the harm bound of section 9 | 1 point is the unit of G5 and K3. "Half" is the form of P3 |

---

## 2 Which data are spent

Every dataset already generated in both testbed folders is spent. It can be read as exploratory only.

- **Honeypot and Bottleneck:** dataset seeds 7, 5101 to 5103, 5201 to 5203, 5301 to 5303, 7101 to 7103, 7201 to 7203, 7301 to 7309, 7401 to 7412 and 7501 to 7512.
- **Published gridworld:** dataset seeds 7, 6101 to 6103, 6301 to 6305, 6401 to 6405 and 6501 to 6515.

Dataset seeds 6601 to 6615 appear in no data file, receipt, protocol or script of either folder.

---

## 3 What can be read on spent data without training

- **M1, and M3 to M7,** on the saved half-random standard-learner runs. Everything they need is saved: the critics' values, the margin, the policy's commands and the datasets. The reading uses the block's own rule on candidates 6501 to 6515: the first nine valid datasets in seed order. Datasets 6401 to 6405 are too few for "holds" (three of the five pass check S4) and are reported beside it, dataset by dataset, without a standing.
- **M2 cannot be read.** The runs kept only the overall coverage, not the audit rows one by one, and the final networks were not saved.
- **The outcome of section 9** of the criteria can be read for any saved policy.

By the criteria, marks M3 to M7 and checks S5 to S7 are not computed on any data, spent data included, until the text and the verdict script are committed. A reading on spent data can only say "meets" or "does not meet on seen data".

---

## 4 What a verdict would need

Only with your approval. Nothing is generated or trained by these documents.

- **Fresh datasets on the published gridworld:** fifteen candidate seeds that neither folder has used, for example 6601 to 6615. The builder makes all three recipes for each seed, so the best-policy data for check S3 and the random-starts data for the lava outcome come with them.
- **Standard-learner runs only,** with alpha 1.0 and 25,000 updates, as validated. No method is trained.
- **Three script changes,** each changing a fingerprint and nothing in training:
  1. the dataset builder writes the fingerprints of the criteria file and the verdict script into its receipt;
  2. the run script writes them into its start record and copies both files into the results folder;
  3. the run script saves the audit rows one by one: cell, command, the one-step target, the value of each of the two critics, the one-step error as the run computed it, and the margin. M2 needs them.

  The verdict script fixes the form of all three. The fingerprints go under the entry `margin_validation`, and the script's `record_for_receipt()` returns exactly what to write. The audit rows go in `audit_rows_<task>.npz` beside the run's other files, as arrays `cell`, `command`, `target`, `q1`, `q2`, `residual` and `margin`, one row of 1,280 per run.

  If decision 4 gives each dataset its own training seeds, a fourth change is needed: the run script takes its training seeds dataset by dataset. Today it uses 20, 21 and 22 on every dataset.
- **The verdict script** is written. It must be committed before the marks are first computed, and it refuses to run until it is (section 7).
- With standard-learner runs only, the lava outcome gives mark L0 and the readings reported beside it. Marks L1 to L4 need a trained method and are not part of this block.
- Optionally, saving the final networks, which would lift the limit that values are read only at the five certain commands.

Seen rates that bear on whether a block reaches nine valid datasets: by check S4, 4 of the 20 saved datasets 6401 to 6405 and 6501 to 6515 would not count, and all 15 of datasets 6501 to 6515 passed the competence rule.

---

## 5 One-time readings on spent Honeypot and Bottleneck data

These are exploratory and are not part of any verdict block. They use the thin-data pilot (datasets 5301 to 5303). The first two have already been read: the comparison (section 8) prints the margin at the designed move for each number of commands kept, and its saved Honeypot tables print the one-step coverage on fresh rows of the designed move. They are listed with that said. The third has not been computed and is listed so it cannot be chosen later.

- The one-step coverage on fresh rows of the designed move, beside the same reading with complete data.
- The margin at the designed move with 2 and 8 commands kept (thinly recorded), with 32 kept (well recorded by the 16-row line) and with complete data.
- In the Bottleneck: how often subtracting 3 margins reverses a critic that rightly prefers the bridge.
- Existing mark G2 as written, with its past results unchanged. Under the criteria a G2 pass does not count toward "validated": it asks for a ratio and can pass while the margin is many times too small.

In these two environments a row counts as recording the move when its command gives the move any chance. The saved values sit at the certain command, which no row records.

---

## 6 How the criteria were written

- Three designers each proposed a full set of criteria from a different angle: calibration, the policy's decision, and a skeptic's view of pre-registration. They were barred from computing any statistic that relates the margin to the critic's error.
- The three agreed on the scope, on keeping the two errors apart, and on judging coverage inside each class. They disagreed on where to judge it; the criteria follow the skeptic and use the published gridworld's half-random data, where the thin class has dozens of commands per dataset.
- Three reviewers then stress-tested the first draft under the same bar. The main changes they forced:
  - the one-step error is defined on the lower of the two critics, as the code computes it;
  - the bar for the rank correlation went from 0.5 to 0.3, because even a perfect margin would score only about 0.5;
  - a control was added for map position (M4), because thin coverage sits at high-value cells;
  - "misled after the penalty" now counts only cases that are still wrong, so moving the choice from one wrong command to another does not count as a correction;
  - two positive controls decide whether a fail means anything (S5, S7), and they can never cancel a pass;
  - the lava outcome's seen numbers were restated as the outcome is defined (2.93 and 2.80 points, not 3.29 and 3.12), and its per-seed mark no longer fails on a seed with no room;
  - the hindsight section now says that the scope and check S4 were both set knowing a margin number.
- Left out on purpose: a pass mark on "the true error is inside one margin" (the method never promised it); a ratio mark of the kind G2 uses; any mark for never-commanded moves; trained-behavior marks, which need their own registration.

---

## 7 The verdict script

`gridworld_critical_v1/margin_verdict.py`, with its tests in `test_margin_verdict.py`. Written 2026-11-15. Not committed. Never run on a saved run or dataset.

**What it does.**

- It reads a finished standard-learner run and judges the seven marks, the seven setup checks and the verdict of the criteria, and prints everything section 8 of the criteria lists.
- It has two modes, chosen by the data. On fresh datasets it gives a verdict, once. On datasets already seen it only says "meets" or "does not meet on seen data", in a file of its own.
- It locks itself. It opens no run until the criteria, the script and every script of the testbed are committed and unchanged. On 2026-11-15 that lock names 33 files.

**How it was checked.** Only on made-up numbers, because the criteria forbid computing the marks on real data before the commit.

- 80 tests. A made-up critic is built around the map's exact values, so the right answer is known: a margin equal to the critic's error must come out "validated", and a flat margin must not.
- Two rounds of independent review: 16 reviewers, each followed by a skeptic who tried to refute the findings. Three reviewers rebuilt the marks from the criteria text without reading the script, and compared: no disagreement on 580 cases for M3 to M5, 7,200 runs for M6 and M7, and 39,000 inputs for the pass rules and the verdict.
- The first round confirmed 42 findings and the second 23, many of them the same point found by several reviewers. None was a wrong number in a mark. Most were readings the script had chosen without saying so; the rest were gaps in the lock, in how a judged block is recognized, and in what the written report prints. The fixes after the second round were tested but not reviewed a third time.

**Readings the script had to fix.** Where the criteria are silent or can be read two ways, the script takes the literal, the strict or the plain reading. All 24 are listed at the top of the script. Four were put to you, because they can change a verdict or what must be committed. **Decided: all four accepted by David on 2026-11-15, as the script has them.** The script's text was not touched.

| # | Point | What the script does |
|---|---|---|
| 1 | **The lock** (reading 3). The criteria say the dataset builder and the run scripts must be committed before any mark is computed | It asks for every Python file in the testbed folder, the shared learner and the three calibration files. So the testbed's scripts must be committed before the first reading, on spent data too |
| 2 | **When there are too few wrong choices to judge** (reading 14). If check S6 fails, M7 is judged on its first clause only | A fail of M7 then counts only if the error-sized reference margin passed M7 in full. With no misled case that reference cannot pass, so the fail does not count and the verdict is "inconclusive", not "not validated" |
| 3 | **M2 with one class missing** (reading 8) | A class that fails on nine dataset values fails M2, even if the other class has no value |
| 4 | **A fixed sentence that reads badly.** Filled in as written, the sentence for M7 is "Subtracting 3 margins breaks too many of the critic's right choices more than 1%." | The script prints it as written. A better sentence would be a new version of the criteria |

Any change to the script after it is committed gives it a new fingerprint and is a new version, like a change to the criteria.
