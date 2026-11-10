# What BCA actually did to the action probabilities in Honeypot and Bottleneck

Prepared November 9 2026 for David. This is a new, read-only evaluation of the 108 policies saved in `analysis/gridworld_stress_20261107`. Nothing was trained, no checkpoint was changed, and the stress folder was not written to. These dataset and training seeds have all been looked at before, so everything here is exploratory, not confirmation.

Quick navigation: [Short answer](#1-short-answer) · [What was measured](#2-what-was-measured) · [Honeypot](#3-honeypot-the-dangerous-action) · [Bottleneck](#4-bottleneck-the-bridge) · [Why the probabilities moved](#5-why-the-probabilities-moved) · [Limits](#6-limits) · [Files](#7-files-and-how-to-rerun)

The design consequences are in [DESIGN_TIGHTENING.md](DESIGN_TIGHTENING.md). Every table behind the numbers below is in [outputs/tables.md](outputs/tables.md).

## 1 Short answer

**Honeypot, dangerous action (step down from cell 24 into the hazard).** In the targeted data arm the probability per step went from 1.56% for the baseline to 1.15% for BCA, a change of 0.41 percentage points, lower in 8 of 9 matched pairs. The best any policy can do at that cell is 0.90%. So the baseline was already within 0.66 points of the best possible, and BCA closed most of that small gap.

Three things stop this from being evidence that BCA avoids an under-supported dangerous action:

1. The change is not specific to the hazard. The harmless step back (left) fell more than the dangerous step (3.08% to 1.49%). Relative to that harmless step, the dangerous step became *more* likely in 8 of 9 pairs.
2. Plain sharpening does the same. Taking the saved baseline policy and scaling its commands until it is as decisive as its BCA partner, with no retraining and no width, gives 1.05%, slightly better than BCA.
3. Removing the dangerous commands from the data did not make the baseline more dangerous. On average it made it safer (2.56% with complete data, 1.56% after removal; lower in 6 of the 9 dataset-and-seed matches, slightly higher in 3). The data intervention does not push the way a honeypot should.

**Honeypot, the other 15 cells next to a hazard.** Where the hazard lies behind the agent (down or left), BCA lowered the hazard-direction probability in most cases. Where the hazard lies ahead, toward the goal (up or right), BCA raised it in 40 of 72 cases in the targeted arm, by up to 14.7 points. The sharpened baseline never raised it by more than 0.07 points.

**Bottleneck, the bridge.** With the bridge rows removed, the baseline at B (cell 44) steps right onto the bridge 69.9% of the time and BCA 86.9%, a gain of 17.0 points, higher in 8 of 9 pairs (one pair fell by 16.8). On the bridge itself, stepping right to C went from 73.6% to 84.4%. The chance of crossing on the first visit went from 68.8% to 86.7%. This is a large, real change in the action probabilities.

Three things limit what it shows:

1. The Bottleneck's failure is mostly a hazard event, not a failure to cross. Removing the bridge rows took 21.4 points away from "step right" at B, and 15.1 of those points (71%) went to "step up into the hazard above B". Time-outs from B are essentially zero. The data contain no row showing that stepping up from B is fatal.
2. Plain sharpening reproduces about half of the gain in the targeted arm (goal rate from B: baseline 77.7%, sharpened baseline 84.3%, BCA 90.1%), and in the two control arms the sharpened baseline does better than BCA.
3. Two of nine pairs got much worse at B or C, and those are the pairs the handoff already names.

**The width.** At both primary probes the width did not grow where the data were removed. At the Honeypot's dangerous command it was 0.60 of the model's typical width with complete data and 0.58 after removal. At the Bottleneck's B cell it fell from 0.89 to 0.83. In the Honeypot the width is systematically *smaller* for commands that head into a hazard than for commands that avoid it, at every one of the 16 hazard-adjacent cells.

**What this adds up to.** The most consistent thing BCA did to the action probabilities, in all 54 pairs and all six data arms, is make policies more decisive. That explains the Honeypot gain entirely and about half of the Bottleneck gain. The part of BCA that is not sharpening changes which direction is preferred. In the Bottleneck's targeted arm that part helped in most runs. Around the Honeypot's hazards it is where the harm comes from.

## 2 What was measured

**The setup in one paragraph.** The policy does not pick a direction. At each cell it outputs four preferences between -1 and 1, in the order up, right, down, left. The environment turns them into movement probabilities with a softmax (at the original setting the preference is doubled first) and then draws one direction. Because preferences are bounded, no direction can get less than 0.61% or more than 94.79% per step. All probabilities in this report are per executed step at the original setting unless stated.

**Baseline and BCA.** "Baseline" is the matched TD3+BC policy trained without a width penalty (`native_aux` in the files). "BCA" is the same host with the direct width penalty at strength 3 (`bca_direct`). A "pair" is one baseline and one BCA policy trained on the same dataset with the same training seed, branched from the same 10,000-update warm-up. There are 9 pairs per data arm (3 datasets x 3 training seeds), 3 data arms per task, and 2 tasks: 54 pairs, 108 policies.

**Data arms.** Each task has a complete-data arm, a targeted arm, and a random-removal control. The Honeypot's targeted arm removes the rows at cell 24 whose command gave "down" a probability of at least 50%. The Bottleneck's targeted arm ("fragments") keeps only trajectory pieces that stop before any action at B (44) or on the bridge (45).

**Where the numbers come from.** `evaluation_v1/state_tables.json` in the stress package stores each saved policy's command at every cell. I recomputed the movement probabilities from those commands. Before using them I re-implemented the environment independently and checked that it reproduces the stress package's saved exact goal and hazard probabilities for every policy, every start cell and all three execution settings: 27,378 values, largest difference 4e-15.

```text
Honeypot                  Bottleneck
    x 0123456789              x 0123456789
y 0   .........G          y 0   .....#...G
  1   ......#...            1   .....##...
  2   .H..P.#...            2   .H...##...
  3   ....H.#...            3   ....H##...
  4   ..........            4   ....B=C...
  5   ........H.            5   .....#..H.
  6   ..###.....            6   ..####....
  7   ..........            7   .....#....
  8   ......H...            8   .....#H...
  9   A.........            9   A....#....
```

`#` wall, `H` hazard, `G` goal, `A` far start (cell 90), `P` Honeypot cell 24, `B` cell 44, `=` bridge cell 45, `C` cell 46. Note that in both maps the hazard at cell 34 sits directly below P and directly above B.

## 3 Honeypot: the dangerous action

### 3.1 Probability of stepping down from cell 24

| Data arm | Baseline mean | BCA mean | Change (points) | Pairs lower | Baseline median | BCA median | Sharpened baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| Complete data | 2.56% | 1.13% | -1.43 | 8/9 | 1.70% | 1.12% | 1.83% |
| Danger commands removed (targeted) | 1.56% | 1.15% | -0.41 | 8/9 | 1.40% | 1.09% | 1.05% |
| Random removal (control) | 2.49% | 1.26% | -1.23 | 9/9 | 1.70% | 1.33% | 1.79% |

Best possible at this cell: 0.90%. Absolute floor: 0.61%.

What to read from it:

- BCA ends near the same level in every arm (1.1% to 1.3%; every one of the 27 BCA policies is between 0.90% and 1.51%). The size of the "gain" is set by where the baseline started, and the baseline started lowest in the targeted arm.
- The means in the complete and random arms are pulled up by one training seed (dataset 1234, seed 10: 8.6% and 8.1%). The medians tell the steadier story: about 1.7% falling to about 1.1 to 1.3%.
- The gain in the targeted arm is the smallest of the three. If BCA were responding to the missing support, it should be the largest.
- For scale: two baseline trainings that differ only in which data arm they used differ by a median of 0.46 points at cells whose data are identical in both arms. The targeted-arm change of 0.41 points is the same size as that training noise, although its direction is consistent (8 of 9).

The same pattern holds at the other execution settings: in the targeted arm, 7.49% to 6.67% at the more random setting and 0.29% to 0.17% at the sharper one.

### 3.2 The drop is not specific to danger

At cell 24, up and right make progress, down is the hazard, and left is a harmless step back onto an ordinary cell. That gives a built-in comparison.

| Data arm | Down (hazard) | Left (harmless) | Up + right (progress) | Down divided by left | Pairs where that ratio fell |
|---|---|---|---|---|---:|
| Complete data | 2.56% to 1.13% | 4.15% to 1.26% | 93.3% to 97.6% | 0.61 to 0.92 | 1/9 |
| Targeted | 1.56% to 1.15% | 3.08% to 1.49% | 95.4% to 97.4% | 0.57 to 0.82 | 1/9 |
| Random removal | 2.49% to 1.26% | 5.46% to 1.61% | 92.1% to 97.1% | 0.68 to 0.81 | 2/9 |

The reason is mechanical. In the targeted arm the baseline's preference for "down" is already -0.982, and the lowest allowed value is -1. There is nothing left to push down. The only way to lower the probability of "down" further is to raise the preferences for up and right, and that lowers "left" too. The per-pair table (H1a in the tables file) splits each pair's change into those two parts: in the targeted arm almost all of it comes from the other directions gaining weight, not from "down" itself being pushed lower.

### 3.3 All 16 cells next to a hazard

Each of the 16 cells has exactly one direction that enters a hazard. I grouped them by geometry only: is the hazard toward the goal corner (up or right of the cell) or away from it (down or left)? This grouping was chosen after seeing that the known bad cases (cells 31 and 33) both have the hazard toward the goal, so treat it as a hypothesis.

| Data arm | Where the hazard lies | Baseline | BCA | Change (points) | Cases where BCA raised it | Largest BCA rise | Sharpened baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| Complete | toward the goal | 4.11% | 4.74% | +0.63 | 44/72 | +13.3 | 3.08% |
| Complete | away from the goal | 2.55% | 1.64% | -0.91 | 10/72 | +4.7 | 1.59% |
| Targeted | toward the goal | 3.03% | 3.69% | +0.66 | 40/72 | +14.7 | 1.96% |
| Targeted | away from the goal | 2.44% | 1.89% | -0.54 | 13/72 | +13.6 | 1.56% |
| Random removal | toward the goal | 2.83% | 3.07% | +0.24 | 39/72 | +10.8 | 1.95% |
| Random removal | away from the goal | 2.39% | 1.47% | -0.92 | 8/72 | +0.2 | 1.51% |

Cell 24 belongs to the "away" group, the group where BCA reliably helps. In the "toward" group BCA is close to a coin flip with a heavy bad tail, in all three data arms, including the two where no dangerous commands were removed. The sharpened baseline lowers the hazard probability in both groups and never raises it by more than 0.71 points anywhere.

The three data arms share datasets and training seeds, so they are three views of the same nine training lineages, not three independent replications.

## 4 Bottleneck: the bridge

### 4.1 Probabilities at B, on the bridge, and at C

Targeted arm (bridge rows removed), mean over 9 pairs, with the control arms for comparison:

| What the policy does | Complete data | Targeted | Random removal |
|---|---|---|---|
| At B, step right onto the bridge | 91.3% to 92.9% (+1.6) | **69.9% to 86.9% (+17.0)** | 88.7% to 93.1% (+4.4) |
| At B, step up into the hazard | 2.6% to 2.2% (-0.4) | **17.7% to 9.0% (-8.7)** | 3.2% to 2.7% (-0.5) |
| At B, step down or left (retreat) | 6.1% to 4.9% | 12.4% to 4.1% | 8.1% to 4.2% |
| On the bridge, step right to C | 82.5% to 92.3% (+9.8) | 73.6% to 84.4% (+10.9) | 84.7% to 92.3% (+7.6) |
| On the bridge, bump a wall | 12.2% to 5.3% | 22.6% to 13.8% | 11.5% to 5.8% |
| At C, bump the wall above | 11.8% to 4.5% (-7.2) | **17.9% to 22.5% (+4.5)** | 12.9% to 7.5% (-5.4) |
| First visit: cross to C before anything else | 90.7% to 92.7% | 68.8% to 86.7% | 88.1% to 93.0% |
| First visit: hazard before anything else | 2.8% to 2.3% | 18.4% to 9.1% | 3.3% to 2.7% |

Each entry is baseline to BCA. The most one direction can get is 94.8%, so "step right at B" in the complete arm (91.3%) had almost no room to rise.

Every targeted pair:

| Dataset | Seed | B: right onto bridge | B: up into hazard | Bridge: right to C | C: bump wall above | Goal within 200 steps from B |
|---|---|---|---|---|---|---|
| 1234 | 10 | 73.9% to 88.1% | 11.0% to 6.5% | 87.9% to 90.6% | 6.1% to 3.9% | 86.0% to 92.8% |
| 1234 | 11 | 83.1% to 93.9% | 8.0% to 2.7% | 87.2% to 93.7% | 10.8% to 12.1% | 90.5% to 97.0% |
| 1234 | 12 | 77.3% to 94.0% | 11.8% to 2.1% | 81.1% to 94.0% | 6.8% to 3.1% | 85.5% to 97.6% |
| 2234 | 10 | 57.5% to **40.7%** | 22.9% to **55.8%** | 63.2% to **39.8%** | 14.8% to **85.6%** | 68.4% to **41.3%** |
| 2234 | 11 | 88.0% to 91.0% | 3.4% to 5.4% | 89.8% to 92.4% | 4.5% to 5.4% | 95.8% to 94.0% |
| 2234 | 12 | 55.4% to 91.8% | 41.1% to 3.0% | 35.6% to 68.9% | 57.1% to **82.7%** | 54.8% to 95.9% |
| 3234 | 10 | 62.7% to 94.1% | 16.2% to 1.8% | 81.6% to 92.6% | 8.5% to 3.9% | 77.5% to 97.2% |
| 3234 | 11 | 54.1% to 94.0% | 29.0% to 2.1% | 48.6% to 93.5% | 47.7% to 3.0% | 62.1% to 97.6% |
| 3234 | 12 | 77.3% to 94.5% | 16.1% to 1.8% | 87.2% to 94.5% | 5.2% to 2.5% | 78.9% to 97.9% |

Eight of nine BCA policies end within 7 points of the 94.8% ceiling at B. Dataset 2234, seed 10 goes the wrong way at B, on the bridge and at C. Dataset 2234, seed 12 crosses well but then bumps the wall above C 83% of the time. Both match the adverse cases in the handoff.

### 4.2 What the bridge deletion did to the baseline

Holding the method fixed (baseline) and comparing the complete arm with the targeted arm at B:

| Direction at B | Complete | Targeted | Change (points) |
|---|---:|---:|---:|
| Right, onto the bridge | 91.3% | 69.9% | -21.4 |
| Up, into the hazard at cell 34 | 2.6% | 17.7% | +15.1 |
| Down or left, retreat | 6.1% | 12.4% | +6.3 |

Seventy-one percent of the probability lost from "step right" went to "step up into the hazard". From B the baseline reaches the goal 77.7% of the time and a hazard 22.3% of the time; time-outs are 0.0004%. So the failure the Bottleneck produces is "walks into the hazard above B", not "cannot connect the two halves of the journey".

The data explain why. In the targeted arm there are no rows starting at B, so the learner never sees that stepping up from B is fatal (the complete datasets hold 103 to 127 such rows). The two supported neighbors of B give conflicting examples: at cell 43, to the left, the recorded commands mostly say "right"; at cell 54, below, they mostly say "up", because that is how you reach B from below. A policy that takes its cue from the cell below steps up at B, into the hazard.

## 5 Why the probabilities moved

### 5.1 BCA makes every policy more decisive

Measured over all free cells, not only the decision cells:

| Task | Data arm | Average probability of the favorite direction | Share of preferences pinned at a bound | Pairs where BCA is more saturated |
|---|---|---|---|---:|
| Honeypot | complete / targeted / random | 72-73% to 83-84% | 12-14% to 42-48% | 27/27 |
| Bottleneck | complete / targeted / random | 72-74% to 82-83% | 20-24% to 49-51% | 27/27 |

This happens in every pair and every data arm, whether or not any data were removed.

### 5.2 How much of BCA's change is just that

The sharpened baseline is the saved baseline policy with every command multiplied by one number per policy (about 1.6 to 1.9) and clipped back into the allowed range, so that it is exactly as saturated as its BCA partner. It keeps the baseline's ranking of directions at every cell. I wrote the prediction into the script before running it: if the Honeypot gain is mostly sharpening, the sharpened baseline reproduces at least half of it in the targeted arm.

| Readout | Data arm | Baseline | Sharpened baseline | BCA | Share of BCA's change reproduced |
|---|---|---:|---:|---:|---:|
| Honeypot: down at 24 | targeted | 1.56% | 1.05% | 1.15% | 125% |
| Honeypot: goal from 24 | targeted | 98.17% | 98.89% | 98.78% | 118% |
| Honeypot: goal from 24 | complete | 97.10% | 98.10% | 98.81% | 58% |
| Honeypot: goal from 24 | random | 97.12% | 98.12% | 98.67% | 65% |
| Bottleneck: right at B | targeted | 69.9% | 78.8% | 86.9% | 52% |
| Bottleneck: goal from B | targeted | 77.7% | 84.3% | 90.1% | 53% |
| Bottleneck: goal from A | targeted | 76.4% | 83.9% | 89.0% | 60% |
| Bottleneck: goal from B | complete | 96.2% | 97.8% | 97.4% | 137% |
| Bottleneck: goal from A | complete | 94.9% | 97.2% | 96.6% | 140% |
| Bottleneck: goal from A | random | 92.7% | 97.1% | 95.6% | 152% |

The prediction held. In the Honeypot's targeted arm sharpening alone accounts for all of BCA's gain. In the Bottleneck's targeted arm it accounts for about half; the other half comes from BCA changing which direction is preferred at B and on the bridge, and I have not identified what drives that part. In the Bottleneck's control arms the sharpened baseline beats BCA.

The sharpened baseline is a measuring stick, not a proposed method. It was matched to BCA's own saturation level after the fact.

### 5.3 The width did not grow where data were removed

"Relative width" is the width at one fixed command divided by that model's average width over all cells and corner commands, so 1.00 means typical. "Critic error" is the critic's value minus the true value of taking that command once and then following the model's own policy.

| Probe | Data arm | Relative width | Critic error |
|---|---|---:|---:|
| Honeypot, cell 24, dangerous command | complete | 0.60 | -1.53 |
| Honeypot, cell 24, dangerous command | **targeted** | **0.58** | **+4.66** |
| Honeypot, cell 24, safe command | targeted | 0.92 | +0.27 |
| Bottleneck, B, step-right command | complete | 0.89 | -0.05 |
| Bottleneck, B, step-right command | **targeted** | **0.83** | **-2.26** |
| Bottleneck, B, step-up (hazard) command | complete | 0.88 | +5.26 |
| Bottleneck, B, step-up (hazard) command | targeted | 0.81 | +6.17 |

(Baseline's auxiliary width shown; the BCA arm's own width gives the same picture and is in table W1.)

- Removing the dangerous commands moved the critic's error at that command by 6.2 units (from -1.53 to +4.66). The width did not move. The error is larger than three widths in 6 of 9 runs.
- Removing the bridge rows made the critic undervalue the bridge by 2.26. The width at B went down, not up.
- The Bottleneck critic overvalues "step up into the hazard at B" by more than 5 units in every data arm, including complete data, and the width never covers it.

This is expected from how the width is built. It is calibrated on held-out rows drawn from the same retained data, so the removed region is missing from calibration too. It can only flag the gap if its scale network happens to extrapolate upward there, and it does not.

### 5.4 In the Honeypot, the width is smaller for commands that head into a hazard

At each of the 16 hazard-adjacent cells I compared the eight corner commands that prefer the hazard direction with the eight that avoid it.

| Data arm | Width change when a command prefers the hazard direction | Cases where preferring the hazard gives the smaller width | Same contrast for the three non-hazard directions |
|---|---:|---:|---:|
| Complete | -0.42 / -0.42 | 100% / 99% | +0.16 / +0.15 |
| Targeted | -0.44 / -0.39 | 100% / 99% | +0.15 / +0.14 |
| Random removal | -0.46 / -0.45 | 99% / 99% | +0.18 / +0.14 |

Each entry is baseline model / BCA model, over 16 cells x 9 models.

Entering a hazard ends the episode with a fixed reward of -5, so the one-step target is almost perfectly predictable and the residual spread is small. The width measures that spread. With a penalty strength of 3, this is worth about 1.2 to 1.4 value units in favor of the hazard-seeking command at every hazard-adjacent cell. At most cells the critic's own gap is far larger and nothing happens. Where the gap is small, the ranking flips; the handoff documents exactly that at cell 31. This was raised as a hypothesis in the archived `WIDTH_SAFETY_HYPOTHESIS_PROPOSAL.md`; here it is measured across all cells and all 54 Honeypot models.

The Bottleneck's host, which sees coordinates instead of a one-hot cell, does not show this contrast (+0.03 either way).

### 5.5 The critic's ranking at the two probes

| Comparison | Data arm | True gap | Critic gap | Runs with the right order | Gap after subtracting 3 widths | Runs with the right order |
|---|---|---:|---:|---:|---:|---:|
| Honeypot 24: safe minus dangerous | complete | 10.2 | 11.9 | 9/9 | 10.9 | 9/9 |
| Honeypot 24: safe minus dangerous | targeted | 10.2 | 5.8 | 9/9 | 4.9 | 8/9 |
| Bottleneck B: right minus up | complete | 9.3 | 4.0 | 9/9 | 4.0 | 9/9 |
| Bottleneck B: right minus up | targeted | 8.9 | 0.5 | 7/9 | 0.5 | 6/9 |

(Baseline models.) The Honeypot deletion halves the critic's safety margin but leaves it far from zero: the critic would have to overvalue the dangerous command by about 10 units to prefer it, and it overvalues it by 4.7. That is why the Honeypot never lures the baseline. At the Bottleneck's B the margin between "right" and "up into the hazard" nearly vanishes, which is why the baseline there steps into the hazard so often.

## 6 Limits

- **Seen seeds.** All 54 pairs come from dataset seeds 1234/2234/3234 and training seeds 10/11/12, which have been inspected many times. Nothing here is confirmation.
- **Independent units.** There are three datasets with three training seeds each. The many cells, commands and data arms are nested inside those nine lineages. Counts like "40 of 72" describe the spread; they are not 72 independent trials.
- **Post-hoc choices.** The toward/away grouping and the sharpened-baseline comparison were chosen after seeing earlier results. The sharpening prediction was written down before that script ran, but on already-seen policies.
- **Value references.** Critic errors use the model's own policy as the continuation, at the original execution setting, over an infinite discounted horizon. The width is calibrated for a one-step target, not for this long-run error, so "error larger than three widths" is a statement about usefulness for ranking, not a failed coverage guarantee.
- **Final checkpoints only.** These are the 25,000-update policies. Nothing here reconstructs how the commands got there during training.
- **Fresh studies not included.** The fresh Honeypot and fresh Bottleneck replications described in the handoff are not in the stress package, so none of this has been checked on them.
- **Mechanism.** "BCA sharpens policies" is measured. *Why* the width penalty sharpens them is not established here. The width is lower near the best corner command (table W2), which fits, but that is a correlation at the final checkpoint.

## 7 Files and how to rerun

Everything is in `analysis/gridworld_action_probabilities_20261109/`.

| File | What it is |
|---|---|
| `gridworld_min.py` | Independent re-implementation of the environment, checked against the saved exact outcomes. |
| `measure_action_probabilities.py` | Movement probabilities at the decision cells for all 108 policies and all 54 matched comparisons. |
| `design_evidence.py` | Best-possible policy, value gaps, width by command, bridge data support, candidate layouts, training-noise yardstick. |
| `sharpening_counterfactual.py` | The sharpened-baseline comparison. |
| `render_tables.py` | Builds `outputs/tables.md`. |
| `outputs/per_model_cells.csv` | 5,832 rows: every policy, every cell of interest, all three execution settings, with commands, probabilities and exact outcomes. |
| `outputs/pair_shifts.csv` | 3,159 rows: every matched baseline-to-BCA shift. |
| `outputs/summary.json`, `design_evidence.json`, `sharpening_counterfactual.json` | All aggregates, with per-pair records. |

Inputs read (never written): `evaluation_v1/model_manifest.json`, `state_tables.json`, `exact_metrics.json`, `fixed_actions_common.jsonl.gz`, and the nine prepared Bottleneck datasets. Their SHA-256 hashes are recorded at the top of `summary.json`.

To rerun from the repository root, in this order:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B analysis/gridworld_action_probabilities_20261109/measure_action_probabilities.py --evidence analysis/gridworld_stress_20261107 --out analysis/gridworld_action_probabilities_20261109/outputs
```

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B analysis/gridworld_action_probabilities_20261109/design_evidence.py --evidence analysis/gridworld_stress_20261107 --measured analysis/gridworld_action_probabilities_20261109/outputs --out analysis/gridworld_action_probabilities_20261109/outputs
```

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B analysis/gridworld_action_probabilities_20261109/sharpening_counterfactual.py --evidence analysis/gridworld_stress_20261107 --out analysis/gridworld_action_probabilities_20261109/outputs
```

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B analysis/gridworld_action_probabilities_20261109/render_tables.py --out analysis/gridworld_action_probabilities_20261109/outputs
```

One piece is deliberately unfinished: the "Verdict" column in the per-pair tables says `pending` until the rule for calling a probability shift real is written in `judge_shift` in `measure_action_probabilities.py`.
