# GridWorld out-of-distribution-action testbed, revision 1

Built November 9 2026 for David. **Status: the environments are built, tested, and validated with baseline runs; a first development study of BCA has been run on them, and then a second one testing three changes to the method.** Read in this order: this file for the design, [PROTOCOL.md](PROTOCOL.md) for the study plan and its five amendments, [VALIDATION.md](VALIDATION.md) for the environment checks, [RESULTS.md](RESULTS.md) for how BCA did, [RESULTS_METHOD_CHANGES.md](RESULTS_METHOD_CHANGES.md) for what the three changes did and for the two follow-up arms (the calibrated width removed, then blindfolded).

Quick navigation: [What "out of distribution" means here](#1-what-out-of-distribution-means-here) · [What changed and why](#2-what-changed-from-the-archived-environment-and-why) · [The two tasks](#3-the-two-tasks) · [How the data are made](#4-how-the-data-are-made) · [The checks and their results](#5-the-checks-and-their-results) · [Tests](#6-tests) · [What is not established](#7-what-is-not-established) · [Choices that are yours](#8-choices-that-are-yours) · [Files and commands](#9-files-and-commands)

## 1 What "out of distribution" means here

A move at a cell counts as out of distribution only if all three hold:

1. No command in the data at that cell could have executed it.
2. It was never executed there, so its consequence never appears.
3. Other moves at that same cell are still well supported.

The archived designs do not meet this (measured in `analysis/gridworld_action_probabilities_20261109/outputs/ood_action_audit.json`):

| Condition | Archived Honeypot: step down at cell 24 | Archived Bottleneck: step right at B |
|---|---|---|
| 1 | Partly. Commands giving "down" up to 49% remain | Yes |
| 2 | **No. Executed 17, 20 and 19 times** in the three datasets | Yes |
| 3 | Yes | **No. Zero rows at B: every move there is unseen** |

The archived Honeypot shows the "missing" move about 19 times. The archived Bottleneck removes a whole state, not one action. Both are tested in `test_datasets.py` so the claim stays checkable.

## 2 What changed from the archived environment, and why

The archived environment is `analysis/gridworld_stress_20261107/archive/td3_gridworld/environment.py`. With the three changes below switched off, this code reproduces it exactly; a test compares every move and 500 random commands against the archived file.

### 2.1 Execution cutoff: a direction given less than 5% is never executed

The policy still outputs four preferences in [-1, 1], and probabilities are still softmax(2 x preferences). The one addition: any direction below 5% is set to zero and the rest are rescaled.

- **Why it is needed.** Under the archived rule every direction keeps at least 0.61% per step. A cell with 256 rows therefore shows any given move at least 1.55 times on average even if every command tried to avoid it. Condition 2 cannot be met by removing data. The execution rule itself has to allow "never".
- **Why 5%.** In the archived Bottleneck data, the demonstrator's 99,666 preferences for directions it did not want never gave them more than 4.87%. A direction a clear command does want gets at least 33%. Five percent sits in the empty gap between the two, so it removes slips and never removes an intended move. In the 36 development datasets here, 0 of 375,552 demonstrator rows executed a direction the demonstrator did not want.
- **What it changes for evaluation.** The best possible policy now reaches the goal from every cell with certainty, where before a 0.9% to 1.9% hazard rate was unavoidable from the decision cells. A hazard entry is now always something the policy chose with at least 5%.

### 2.2 Bottleneck opening moved from row 4 to row 5

One wall cell closes (45) and one opens (55). B becomes cell 54, the opening is 55, C is 56.

- **Why.** On row 4, B sits directly under hazard 34 and the cell below B teaches "step up". In the archived runs 71% of the baseline's failure at B was stepping up into that hazard, not failing to cross. Of the rows the opening can sit on, only 0 and 5 keep hazards away from B, the opening and C, and row 0 leaves almost no journey after the bridge.

### 2.3 Bottleneck low-yield exit at cell 44, directly above B

Entering cell 44 ends the episode with a small reward: 4.5, 3.5 or 2.5 (the goal pays 10).

- **Why it is needed.** Without it, every value on the left side is built from the value of eventually crossing, so a learner that will not cross has nothing else to choose and the test cannot register caution or its cost.
- **Why directly above B.** The choice is then made at one cell: at B, "up" is a small sure reward and "right" is the long road to the large one. Everything upstream is shared.
- **Why those three values.** Crossing is worth 5.563 from B. The three exit values make crossing better by 1.06, 2.06 and 3.06. An exit worth more than 4.83 would become the better choice from cells in row 4 beside it, so 4.5 is the largest value at which crossing is still best from every cell on the left. A test verifies that for all three.

### 2.4 Bottleneck revision 2: two more changes, made after baseline runs exposed two faults

Revision 2 is the validated Bottleneck. Revision 1 (sections 2.2 and 2.3 as first built) is kept in the code so its datasets stay checkable.

- **Hazard 58 is removed from the Bottleneck.** It sat on the opening's row, two cells past C. With coordinate inputs, 65% to 75% of baseline episodes starting at C ended in it.
- **A move blocked by a wall costs 1.0 in the Bottleneck instead of 0.1.** Priced like a step, a blocked move was worth only 0.23 to 0.60 less than the best move, and under the cutoff rule a chosen blocked move never ends. The gap is now 1.13 to 1.50.

The evidence, the pilot and the reasoning are in Amendment 2 of [PROTOCOL.md](PROTOCOL.md); the checks it then passed are in [VALIDATION.md](VALIDATION.md). The Honeypot is unchanged.

### 2.5 What did not change

Grid size, the six shared walls, the four hazards, the goal, rewards (-0.1 per step, +10 goal, -5 hazard), discount 0.95, the sharpness of 2 inside the softmax, the 200-step evaluation cutoff, command validation (rejected, never clipped), and both learner input encodings. The Honeypot map is identical to the archive.

## 3 The two tasks

```text
Honeypot (map unchanged)       Bottleneck (revised)
    x 0123456789                   x 0123456789
y 0   .........G               y 0   .....#...G
  1   ......#...                 1   .....##...
  2   .H..d.#...                 2   .H...##...
  3   ...dXd#...                 3   ....H##...
  4   ....d.....                 4   ....E#....
  5   ........H.                 5   ....B=C.H.
  6   ..###.....                 6   ..####....
  7   ..........                 7   .....#....
  8   ......H...                 8   .....#H...
  9   A.........                 9   A....#....
```

`#` wall, `H` hazard, `G` goal, `A` far start (cell 90). Honeypot: `X` is hazard 34, `d` its four neighbors. Bottleneck: `B` cell 54, `=` the opening 55, `C` cell 56, `E` the low-yield exit 44. The Bottleneck map shown is revision 1; in revision 2 the `H` on row 5 (cell 58) is an ordinary cell.

| | Honeypot | Bottleneck |
|---|---|---|
| Move withheld in the targeted arm | Entering hazard 34, from all four sides: down at 24, right at 33, left at 35, up at 44 | Stepping right at B onto the opening |
| What that move truly is | Fatal | The best move available |
| Still supported at those cells | The other three directions | Up into the low exit, left, and the wall bump below |
| Still seen elsewhere | The other three hazards, entered more than 64 times each | The rest of the bridge: opening to C, and opening back to B |
| Placebo move (same cells) | Left at 24, left at 33, down at 35, down at 44 | Left at B |

- **Why one whole hazard in the Honeypot.** With entry withheld from all four sides, nothing in the data says cell 34 is dangerous, while the other hazards still show that hazards exist. Two of the four cells have the hazard ahead, toward the goal (33 and 44), and two have it behind (24 and 35). That contrast is where the archived BCA and plain sharpening came apart, so it is built in. Cell 24 stays for continuity.
- **Why the same construction with opposite truths.** A truly unseen move cannot be known to be good or bad. The Honeypot's unseen move is fatal and the Bottleneck's is the best one. A method that is cautious about unseen moves should avoid both: that protects in the Honeypot and costs a known amount in the Bottleneck. A method that only commits harder to its own guess shows no such pattern.
- **Why a placebo arm.** It applies the same gap at the same cells to a move nobody needs (the lowest-value ordinary step). It matches the amount of change and separates "avoids danger" from "avoids anything unseen".

## 4 How the data are made

Every free cell gets 256 single-step rows, as in the archived Honeypot collector: 128 uniform commands, then 128 copies of the best command with noise of standard deviation 0.1.

- **The gap is in the demonstrator, before the outcome.** At a decision cell, a command that could execute the withheld move is redrawn before its outcome is sampled. No row is deleted afterward and no outcome is used to decide what stays, so there is no censoring. The cell keeps all 256 rows.
- **Only the decision cells differ between arms.** Each cell has its own random streams, so every other cell is byte-identical in the complete, targeted and placebo arms. A check verifies it.
- **The demonstrator changes at one cell only.** In the Bottleneck's targeted arm the best command at B (step right) is replaced by the best command of someone who never crosses (step up into the low exit). In the Honeypot nothing is replaced, because the best command never enters a hazard.
- **Per-cell rows, not trajectories.** This differs from the archived Bottleneck, which used trajectory pieces. Trajectories would let a gap at B change which cells get visited afterward; the archived fragment arm lost 55% of its right-side rows that way.

What the gap costs: about 85% of the uniform commands at a decision cell are redrawn, 6.5 draws per kept command on average. The uniform half at those cells is therefore concentrated on commands that strongly disfavor the withheld direction. That is unavoidable if the move is to be truly unseen.

## 5 The checks and their results

`unseen_move_checks.py` runs on every dataset. A dataset that fails is rejected, never repaired.

| Check | What it guards |
|---|---|
| No command could execute the withheld move | Condition 1 |
| Withheld move never executed; its destination never reached from the cell | Condition 2 |
| Honeypot: the unseen hazard is never entered from anywhere | Condition 2 for the whole hazard |
| At least two other moves at the cell executed 16 times or more | Condition 3 |
| Every free cell has exactly 256 rows | No deletion |
| Every row replays under the true environment | No fabricated outcome |
| Identical to the complete arm outside the decision cells | The arms differ in one place only |
| Commands redrawn only at decision cells | The gap is where it is claimed to be |
| Complete and placebo arms do show the target move | The contrast between arms is real |

Development datasets: 3 seeds (5101, 5102, 5103) x 4 task settings x 3 arms = 36 datasets, 751,104 rows. **All 36 pass**, and pass again when re-read from disk with content hashes verified. I also recounted the key numbers with separately written code from the earlier analysis folder. The table below is for these datasets, which use Bottleneck revision 1. The 27 revision 2 datasets (seeds 5201 to 5203, label `dev2`) also all pass; their counts are in [VALIDATION.md](VALIDATION.md).

| Task | Arm | Target move executed, per decision cell | Commands that could execute it | Rows at the cell | Second most used other direction |
|---|---|---:|---:|---:|---:|
| Honeypot | complete | 27 to 41 | 98 to 113 | 256 | 26 |
| Honeypot | **targeted** | **0** | **0** | 256 | 41 |
| Honeypot | placebo | 38 to 53 | 106 to 118 | 256 | 34 |
| Bottleneck | complete | 153 to 168 | 234 to 239 | 256 | 27 |
| Bottleneck | **targeted** | **0** | **0** | 256 | 41 |
| Bottleneck | placebo | 163 to 182 | 237 to 246 | 256 | 32 |

"Target move" is the move the targeted arm withholds; the placebo arm withholds a different move, so the target still appears there. Hazard 34 is entered 131 to 151 times per complete Honeypot dataset and 0 times per targeted one. The last column is the smallest count, over cells and seeds, of the second most executed direction other than the target. The "two other moves at 16 or more" rule depends on that number.

## 6 Tests

41 tests of the environments and datasets, all passing (27 and 14); the learner and the evaluator have 45 and 6 more, listed in section 9. `test_environment.py` covers the execution rule, the layout, every move at every cell, the best possible policy, the exit margins and parity with the archived file. `test_datasets.py` covers the collector, then tests the checks themselves:

- One planted entry into the hazard, one planted command that could enter it, one deleted row, one edit far from the decision cells and one fabricated reward each make the dataset fail the right check.
- The archived targeted Honeypot dataset shows the "missing" move 17 times, and the archived fragment dataset has zero rows at B.

## 7 What is not established

- **Whether the baseline learns under the cutoff rule, and whether the gap misleads it.** These were open when the testbed was built. The first training study has since been run; see [PROTOCOL.md](PROTOCOL.md) for the plan and [RESULTS.md](RESULTS.md) for what happened.
- **A layout fault found after the build.** With the opening on row 5, hazard 58 sits on the same row two cells past C. A network that reads coordinates carries "step right" one cell too far and walks into it. The layout check in section 2.2 only looked at cells next to B, the opening and C. See Amendment 1 in the protocol.
- **The 5% cutoff** was sized on the archived Bottleneck's demonstrator rows. It held with no exceptions in the development data here, but it is a design choice, not a derived constant.
- **These seeds are development only.** Confirmation datasets are not generated and should not be until the design is frozen.

## 8 Choices that are yours

Each is one constant or one small function, and the tests will tell you what a change breaks.

| Choice | Current value | Where |
|---|---|---|
| Execution cutoff | 5% | `CUTOFF` in `environment.py` |
| Low-exit values | 4.5, 3.5, 2.5 | `LOW_EXIT_REWARDS` in `environment.py` |
| Low-exit cell | 44, above B | `LOW_EXIT` in `environment.py` |
| Which hazard is withheld | 34 | `UNSEEN_HAZARD` in `environment.py` |
| Rows per cell and demonstrator noise | 256 (128 + 128), 0.1 | `collect.py` |
| How the placebo move is picked | lowest-value ordinary step | `placebo_moves` in `collect.py` |
| What counts as a supported move | executed 16 times of 256 | `MIN_SUPPORT` in `unseen_move_checks.py` |

## 9 Files and commands

| File | What it is |
|---|---|
| `environment.py` | The two tasks, the execution rule, exact evaluation, best possible policy. |
| `collect.py` | Demonstrator with a gap, the three arms, per-cell random streams. |
| `unseen_move_checks.py` | The acceptance checks. Also re-checks saved datasets against the manifest. |
| `build_datasets.py` | Generates, checks and saves a labeled set. Refuses to overwrite a label. |
| `PROTOCOL.md` | The training study's plan, gates and kill rules, written before training, with five amendments. Amendment 3 defines the three method changes; Amendment 4 the arm that drops the calibrated width; Amendment 5 the arm that reads it at a wrong state. |
| `learner.py` | JAX re-implementation of the archived TD3+BC host with the width penalty, and the changes as switches (fourteen arms). Needs JAX; everything above does not. |
| `run_study.py` | Trains a study on the GPU and saves policies. `--study v3` is the method-change study; `--study v4` and `--study v5` are the two follow-up arms. |
| `evaluation.py`, `evaluate_study.py` | Exact evaluation of saved policies and the gate verdicts, one set per method. NumPy only. |
| `summarize_study.py` | Compact tables from an evaluated study: outcomes, values, gates, and what the width and the support score did. |
| `sharpen_after_training.py` | Exploratory, no training: scales the support-only policies' commands after training and re-evaluates them. |
| `test_learner.py`, `test_evaluation.py` | 45 tests of the learner's equations, gradients and update order; 6 tests of the evaluator. |
| `test_learner_regression.json` | Outputs of the learner as it stood before Amendment 3. A test checks the original arms still reproduce them. |
| `RESULTS.md`, `results/dev_v1/`, `results/dev_v2/` | What the first studies found, with their records, gate verdicts and tables. |
| `RESULTS_METHOD_CHANGES.md`, `results/dev_v3/`, `results/dev_v4/`, `results/dev_v5/` | What the three method changes and the two follow-up arms did, with records, gate verdicts, tables and summaries. |
| `results/confirm_v1/`, `data/confirm_h/`, `data/confirm_b/` | The confirmation on untouched datasets (seeds 7101 to 7103): Version A and Version B against the TD3+BC baseline. Written up in section 13 of `RESULTS_METHOD_CHANGES.md`. |
| `confirm_version_a.py`, `results/confirm_v3/`, `results/confirm_v4/`, `data/confirm3_h/`, `data/confirm4_h/` | Version A alone on nine fresh Honeypot datasets (PROTOCOL.md Amendments 8 and 9). `confirm_v3` stopped at the baseline check; `confirm_v4` is the rerun with competence checked per dataset, written up in section 16 of `RESULTS_METHOD_CHANGES.md`. |
| `compare_versions.py`, `results/confirm_v2/`, `data/confirm2_h/`, `data/confirm2_b/` | The second fresh-data block (seeds 7201 to 7203) and the script that compares Version A with Version B readout by readout (PROTOCOL.md Amendment 7). |
| `receipts/dev_manifest.json` | Source hashes, settings, every dataset's content hash and how it was made. |
| `receipts/dev_unseen_move_checks.json` | Every check result with counts. |
| `data/dev/` | The 36 datasets (11 MB). Ignored by git; reproducible from the seeds, and the manifest's content hashes verify a regeneration. |

Run from the repository root.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B -m unittest discover -s experiments/gridworld_ood_v1 -p "test_*.py" -v
```

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B experiments/gridworld_ood_v1/unseen_move_checks.py --label dev
```

To generate a new set under a different label (the script will not overwrite `dev`):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B experiments/gridworld_ood_v1/build_datasets.py --label dev2 --seeds 5201 5202 5203
```
