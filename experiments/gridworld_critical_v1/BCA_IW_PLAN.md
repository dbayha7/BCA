# Version C: importance-weighted BCA that acts on the Q values

Prepared November 12 2026 for David. **Status (updated November 12 2026, late): built, checked, and tested in four versions, C, C2, C3 and C4; none passes all four marks yet** ([RESULTS_VERSION_C.md](RESULTS_VERSION_C.md) opens with a table of all four). Every version corrects the critic's ranking at the critical cell and matches Version A on optimal and random-starts data. Version C failed on the half-random data because the cap fed on itself during training; C2 sets the cap once; C3 drops the charge from the critic's target and has the best average there (0.956); C4 scores the target without TD3's noise, which puts the critic's values back on the true scale, but one of its nine half-random runs collapsed. This document describes the first version; the later changes are in Amendments 7, 8 and 10 of [PROTOCOL.md](PROTOCOL.md). It answers: "we need to have an updated importance weighted version that is correctly applied. since bca is looking to impact q values directly we need to have a clear plan and version."

Quick navigation: [Why a new version](#1-why-a-new-version) · [The version](#2-the-version) · [Where it acts on Q](#3-where-it-acts-on-q) · [Checks before any training](#4-checks-before-any-training) · [The test](#5-the-test) · [What could go wrong](#6-what-could-go-wrong) · [Order of work](#7-order-of-work)

## 1 Why a new version

Three facts from this testbed, all measured:

- **The critic is wrong at commands the data never hold.** At the first critical cell it rates a fatal move above the right one in every run, by about 0.7.
- **Version A works around that; it does not fix it.** It keeps the policy away from those commands. The critic's scores stay wrong.
- **BCA's own width is not doing the work.** It is the same size everywhere (0.05), whether a command has 1,000 rows or none, and it does not track the critic's error. The part of Version A that works is a rule I set by hand.

Importance-weighted BCA is the principled form of that hand-set rule. It is already in this repository (`calibration/wbcp.py`, `experiments/shift_iw`). The GridWorld learner has been calling the calibration with all weights equal, so it was never switched on here.

## 2 The version

Terms:

- **Critic**: the network that gives every command a score, its Q value.
- **Held-out rows**: recorded rows kept out of training, used only to calibrate.
- **Score of a held-out row**: how far the critic was from its target on that row, divided by the scale network's prediction. This is BCA's score, unchanged.

**Step 1. The weight of a command.** For a command at a cell: how much less often the data hold it than a typical recorded command. A recorded command has a small weight. A command the data never hold has a very large one. It is estimated by the classifier the learner already trains (weight = exp(-classifier score)).

**Step 2. The threshold, per command.** BCA's calibration is run with every held-out row carrying its own weight and the command being asked about carrying its own weight as the extra point. This is the per-step method of `experiments/shift_iw`: a table of thresholds over a grid of weights, read at the command's own weight. The threshold rises with the weight. Past a point the calibration returns "cannot certify".

**Step 3. The charge.** The extra width that importance weighting adds:

    charge(cell, command) = [threshold at the command's weight - standard threshold] x scale x unit, never below zero

- A command the data hold as often as usual is charged **nothing**.
- A thinly recorded command is charged a little.
- A command that cannot be certified is charged the cap (half the range of the critic's targets), and more the further out it is.

No floor, no "one factor of ten" and no percentile set by hand. The boundary comes from the calibration.

## 3 Where it acts on Q

| | Where | What changes | What it is for |
|---|---|---|---|
| **1** | **The critic, at commands the data do not hold** | At every recorded cell the critic is also shown random commands. If a command cannot be certified, its Q value may not stay above the recorded command's target minus the charge. It is only ever pushed down, never up | **This is the direct part.** It corrects the Q values where they are wrong, which nothing so far does |
| **2** | The critic's target | The next step's Q value is lowered by the charge at the next command | Stops value from flowing through uncertified commands |
| **3** | The policy | It chooses by Q value minus the charge | Keeps the policy on commands the calibration can vouch for |

Strength 3 in all three places, as in every earlier version. Nothing else changes: the baseline (TD3+BC, alpha 1.0), the map, the data, the score.

## 4 Checks before any training

"Correctly applied" has to be shown, not assumed. On this map the exact weights are known, because the data hold exactly five commands and we can count them. Four checks on the saved baselines, with no policy trained:

1. **Equal weights give back standard BCA, bit for bit.** The same check `shift_iw` makes.
2. **The estimated weights match the exact ones** at recorded commands, and are far larger at commands no row holds.
3. **"Cannot certify" lands where it should**: on commands with no rows, and not on well-recorded ones.
4. **The weighted width covers what it claims** on held-out rows, at the 90% it is set to.

If check 2 or 3 fails, the classifier is the thing to fix, and nothing is trained until it passes.

### 4.1 What the checks found (November 12, `iw_checks.py`, 27 saved baselines, GPU)

**First run, with the classifier as the learner trains it: checks 2 and 3 failed.** On the half-random data only 13.5% of never-recorded commands were flagged, and the estimated weights tracked the exact ones at 0.44.

**The cause.** The classifier learned "data-like" by comparing recorded commands with uniformly random ones. A random command is almost never close to one of the five real moves. So it was never shown "a real move that is not recorded at this cell", and it rated every real move as data-like everywhere.

**Two corrections to how the weights are estimated** (`version_c.comparison_rows`). The classifier now compares each recorded row with three kinds of row, a third each:

1. the same cell with a command recorded at another row;
2. the same cell with a uniformly random command;
3. a random cell with a command recorded at another row. This makes the weight large at cells the data never visit.

It is trained for 10,000 updates on this comparison set before Version C starts, the same number the baseline's own classifier had.

**After the corrections:**

| Check | Optimal data | Random-starts data | Half-random data | As written |
|---|---|---|---|---|
| 1 Unit weights give standard BCA exactly | 9 of 9 runs | 9 of 9 | 9 of 9 | **pass** |
| 2a Never-recorded commands get the larger weight | 100.0% of pairings | 100.0% | 99.7% | **pass** |
| 2b Estimated weights track exact ones (half-random data) | | | rank correlation 0.78 | **pass** |
| 3 Never-recorded commands cannot be certified | 100.0% | 100.0% | 60.5% | fail |
| 3 Commands with 16 or more rows can be certified | 100.0% | 100.0% | 100.0% | |
| 4 Coverage of the weighted threshold alone | 88.3% (standard 91.4%) | 91.1% (90.5%) | 97.9% (91.8%) | fail |
| 4 Coverage of the band Version C uses | 93.6% | 91.2% | 97.9% | |

At both critical cells, with optimal and random-starts data: the fatal moves cannot be certified and the way through can, in 9 of 9 runs each.

**The two marks that fail as written, and why I think the marks are wrong, not the weights.**

- **Check 3 on the half-random data.** There, a command with no rows is missing by chance, not because it is outside the data. Every such command sits at a cell with at most 11 rows from the random policy (median 2 to 4). No cell with 25 or more of those rows misses any command. A correct estimate should give these commands a large weight, which it does (check 2a: 99.7%), and should not call them impossible.
- **Check 4 on optimal data.** I measured the weighted threshold on its own. Version C never uses it on its own: the charge is the extra over the standard width, so the band it uses is the larger of the two. That band covers 93.6%, above the standard band's 91.4%.

Both are changes to marks after seeing results, so they needed David's decision before anything was trained. **David's decision, November 12: "Yes, train it", and "proceed" after I explained both failures.** Check 3 is read on the commands whose absence is real, and check 4 on the band Version C uses. The marks as first written, and their failures, stay on record above. One soft spot stays open: on the half-random data about 40% of commands missing by chance remain certifiable, so their Q values are not pushed down. The test's own pass marks are Amendment 6 of [PROTOCOL.md](PROTOCOL.md).

### 4.2 One change to section 3 made while testing

In the critic's two uses the charge stops at the cap. Without that, a command far outside the data would be pushed to a Q value far below anything the critic is trained on. The policy's use keeps the rising charge, so the policy always has a slope back toward the data. In the critic's direct correction the commands it is shown are the same-cell comparison commands above (kinds 1 and 2), so the fatal move at a critical cell is among them.

## 5 The test

On the published map, with pass marks written before any Version C policy is trained.

- **Bar to beat: Version A.** Also against the baseline.
- **Score**: the paper's, as before.
- **A new mark for the Q values themselves**: at the critical cells, in how many runs the critic still rates a fatal move above the right one. The baseline and Version A: every run. Version C should bring that to zero. This is the mark that shows BCA acting on Q.
- First on the three study datasets, then a confirmation on untouched datasets if it passes.

## 6 What could go wrong

- **Too much caution adds up.** Lowering the target at every step can shrink every value. The charge is zero for well-recorded commands to prevent that; check 3 guards it.
- **The classifier's weights may be off.** Check 2 measures it against the exact weights.
- **An earlier version that changed the critic's target (Version B) did more harm than good** in the first testbed. It used the hand-set rule and never trained the critic at unrecorded commands. Part 1 above is what is different.
- **The width still measures one-step error**, not the error in the Q value over a whole episode. Calibrating to the returns the episodes show is the step after this one.

## 7 Order of work

1. The four checks of section 4, on saved baselines. No training.
2. Build Version C in the learner, with a test for every equation.
3. Write the pass marks, then train and judge on the three study datasets.
4. If it passes: confirm on untouched datasets.
