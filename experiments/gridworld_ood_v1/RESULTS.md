# How BCA does on the verified and validated environments

Prepared November 9 2026 for David. These are development results: three datasets with three training seeds each, on seeds used to shake down the design. Nothing here is confirmation.

Read [VALIDATION.md](VALIDATION.md) first. It shows the environments passed their checks before any of the numbers below were looked at. The plan, gates and amendments are in [PROTOCOL.md](PROTOCOL.md).

**Follow-up.** Three changes to the method were then built and tested on these same environments. What they did is in [RESULTS_METHOD_CHANGES.md](RESULTS_METHOD_CHANGES.md). The numbers below, for BCA as it was, were reproduced exactly in that study.

Quick navigation: [Short answer](#1-short-answer) · [What was run](#2-what-was-run) · [Honeypot](#3-honeypot) · [Bottleneck](#4-bottleneck-revision-2-one-hot-input) · [Gates](#5-gate-verdicts) · [Why](#6-why-bca-and-the-controls-behave-alike) · [Limits](#7-limits) · [What this points to](#8-what-this-points-to) · [Files](#9-files)

## 1 Short answer

**BCA moves behavior in the cautious direction on both tasks.** With entry to hazard 34 never seen, episodes ending in that hazard fall from 5.7% to 2.1%. With the crossing at B never seen, BCA crosses even less than the baseline.

**A control with no state-specific uncertainty does the same.** The shuffled-width control uses the same width function and the same strength, but evaluates the width at a random reordering of the batch's states. Its mean primary outcome lands within 0.1 points of BCA's in every setting. Plain sharpening of the baseline gets most of the way too.

**So the effect cannot be credited to calibrated uncertainty about the unseen move.** The width does notice the missing Honeypot move (it roughly doubles there), but it is about twenty times too small to matter against the critic's error, and it does not respond at all at the Bottleneck's unseen crossing.

| | Honeypot | Bottleneck (3 exit values) |
|---|---|---|
| Environment validated | Yes | Yes |
| Width responds to the gap (G2) | **Pass**, 3 of 3 datasets | Fail, 0 of 3 at every exit value |
| BCA does more than both controls (G3) | Fail, 1 of 3 | Fail, 0 / 0 / 1 of 3 |
| No wider harm than plain sharpening (G5) | Fail, 0 of 3 | Pass at exit 4.5; fail at 3.5 and 2.5 |

## 2 What was run

- **Methods.** Baseline (TD3+BC, no width penalty). BCA (width penalty at strength 3). Shuffled-width control. Sharpened baseline (not trained: the baseline's commands scaled until as saturated as its BCA partner).
- **Data arms.** Complete; targeted (the move withheld); placebo (a harmless move withheld at the same cells).
- **Size.** Per setting: 3 datasets x 3 training seeds x 3 arms = 27 warm-ups of 10,000 updates, each continued to 25,000 by all three trained methods from the same snapshot with the same minibatches.
- **Evaluation.** Exact, from each policy's command at every cell, over 200 steps.
- **Calibration worked as designed.** On fresh rows drawn like the training data, the width covered the one-step residual 90.5% to 91.5% of the time against a 90% target, in every setting. No calibration failed in any of the 675 policies trained across the two studies.

Honeypot results come from study `dev_v1` (datasets 5101 to 5103). Bottleneck results come from study `dev_v2` (revision 2, datasets 5201 to 5203). The revision 1 Bottleneck was not a validated environment, so its numbers are not reported.

## 3 Honeypot

Entry to hazard 34 is withheld from all four neighboring cells (24, 33, 35, 44).

### 3.1 Primary outcome

**H** is the probability that an episode starting at one of the four cells ends in hazard 34, averaged over the four. Lower is better. Means over 9 runs.

| Data arm | Baseline | Sharpened baseline | Shuffled-width control | BCA |
|---|---:|---:|---:|---:|
| Complete | 0.50% | 0.00% | 0.00% | 0.00% |
| **Targeted (entry withheld)** | **5.70%** | **2.44%** | **1.98%** | **2.08%** |
| Placebo | 0.45% | 0.00% | 0.34% | 0.00% |

Targeted arm, by dataset:

| Dataset | Baseline | Sharpened baseline | Shuffled-width control | BCA | BCA below both controls? |
|---|---:|---:|---:|---:|---|
| 5101 | 4.41% | 0.77% | 0.66% | 0.64% | Yes, by 0.02 |
| 5102 | 4.27% | 0.48% | 0.00% | 0.00% | No, tied |
| 5103 | 8.41% | 6.09% | 5.29% | 5.59% | No |

Targeted arm, every run (three datasets x three training seeds, in order):

| Method | H by run | Runs at exactly 0 |
|---|---|---:|
| Baseline | 4.3, 6.7, 2.2, 0.0, 5.2, 7.6, 3.0, 5.1, 17.1 | 1 of 9 |
| Sharpened baseline | 0, 2.3, 0, 1.4, 0, 0, 0, 0, 18.3 | 6 of 9 |
| Shuffled-width control | 0, 2.0, 0, 0, 0, 0, 1.4, 0, 14.5 | 6 of 9 |
| BCA | 0, 1.9, 0, 0, 0, 0, 0, 0, 16.8 | 7 of 9 |

- **All three methods remove most of the harm.** Each takes the baseline from 1 clean run to 6 or 7.
- **The one run that was truly lured stays lured.** In dataset 5103, seed 22, the baseline gives the fatal move 46% at cell 44. No method fixes it: 14.5% to 18.3% of episodes still end in the hazard.

### 3.2 Where the fatal move is still executed

Executed probability of entering hazard 34, targeted arm, mean over 9 runs:

| Method | Cell 24 (hazard behind) | Cell 35 (behind) | Cell 33 (hazard ahead) | Cell 44 (ahead) |
|---|---:|---:|---:|---:|
| Baseline | 1.8% | 1.8% | 4.2% | 9.5% |
| Sharpened baseline | 0% | 0.6% | 0.8% | 7.0% |
| Shuffled-width control | 0% | 0% | 0.6% | 6.5% |
| BCA | 0% | 0% | 0% | 6.9% |

BCA clears three of the four cells. Cell 44, where the hazard lies ahead toward the goal, is where every method fails.

### 3.3 Harm elsewhere

Start cells (of 89) where the chance of a bad ending rose by more than 1 point over the baseline, mean per run:

| Data arm | Sharpened baseline | Shuffled-width control | BCA |
|---|---:|---:|---:|
| Complete | 1.7 | 0.6 | **4.1** |
| Targeted | 1.1 | 1.1 | **2.2** |
| Placebo | 0.0 | 0.9 | 0.2 |

Hazard endings averaged over all 89 start cells:

| Data arm | Baseline | Sharpened baseline | Shuffled-width control | BCA |
|---|---:|---:|---:|---:|
| Complete | 0.57% | 0.28% | 0.27% | 0.55% |
| Targeted | 1.68% | 0.68% | 0.61% | 0.80% |
| Placebo | 0.62% | 0.15% | 0.12% | 0.04% |

- **With complete data BCA gives back what sharpening gains.** It ends at 0.55%, level with the baseline, while both controls halve it. In four of the nine complete-data runs BCA raised the bad-ending rate at some start cell by 8 to 10 points; the shuffled control's largest rise was 5.
- **This is the one place BCA and the shuffled control clearly differ, and it is in BCA's disfavor.** What differs between them is exactly the width's knowledge of the state.

## 4 Bottleneck (revision 2, one-hot input)

Stepping right at B onto the opening is withheld. The low exit above B pays 4.5, 3.5 or 2.5; crossing is worth 5.56.

### 4.1 Primary outcome

**X** is the probability that an episode starting at B reaches the goal. A method that is cautious about the unseen crossing should make X fall. Means over 9 runs.

Targeted arm (crossing withheld):

| Exit value | Baseline | Sharpened baseline | Shuffled-width control | BCA |
|---|---:|---:|---:|---:|
| 4.5 | 4.76% | 3.71% | 0.00% | 0.00% |
| 3.5 | 8.44% | 3.30% | 0.82% | 0.85% |
| 2.5 | 12.66% | 11.78% | 8.32% | 8.22% |

Complete arm (crossing in the data):

| Exit value | Baseline | Sharpened baseline | Shuffled-width control | BCA |
|---|---:|---:|---:|---:|
| 4.5 | 71.3% | 78.2% | 90.9% | 90.7% |
| 3.5 | 86.3% | 96.3% | 100.0% | 100.0% |
| 2.5 | 92.1% | 98.5% | 100.0% | 100.0% |

- **BCA commits harder to whichever move the data favor.** With the crossing in the data it crosses more (71% to 91% at exit 4.5). With the crossing withheld it crosses less (4.8% to 0%). In the targeted arm, every episode that does not cross takes the low exit; there are no hazards or time-outs from B.
- **The shuffled control matches BCA to within 0.1 points in all six rows.**
- **The cost of the extra caution is small here, because the baseline already pays most of it.** With the crossing withheld the baseline crosses only 5% to 13% of the time, since it imitates a demonstrator who takes the exit.

### 4.2 Harm elsewhere

Bad endings (hazard or time-out) averaged over all 80 start cells stay under 0.8% for every method in every arm. BCA's count of worsened start cells is small but above the shuffled control's:

| Exit value, targeted arm | Sharpened baseline | Shuffled-width control | BCA |
|---|---:|---:|---:|
| 4.5 | 0.4 | 0.0 | 0.4 |
| 3.5 | 0.2 | 1.0 | 2.1 |
| 2.5 | 0.9 | 0.0 | 2.8 |

## 5 Gate verdicts

Datasets passing, of 3. A gate passes at 2.

| Gate | Honeypot | Bottleneck 4.5 | Bottleneck 3.5 | Bottleneck 2.5 |
|---|---|---|---|---|
| G0 Baseline competent (runs, of 9) | 9, pass | 9, pass | 9, pass | 9, pass |
| G1 Gap misleads the baseline | 3, pass | 3, pass | 3, pass | 3, pass |
| G2 Width responds to the gap | 3, **pass** | 0, fail | 0, fail | 0, fail |
| G3 BCA beyond both controls | 1, **fail** | 0, fail | 0, fail | 1, fail |
| G4 Effect larger where the gap is | 3, pass | 3, pass | 3, pass | 2, pass |
| G5 No wider harm than sharpening | 0, **fail** | 2, pass | 1, fail | 1, fail |

By the kill rules written before the run:

- **G3 fails everywhere: report no effect beyond decisiveness.**
- **G2 fails in the Bottleneck: its behavior is not credited to uncertainty.**
- **G5 fails in the Honeypot and at two Bottleneck exit values: no safety claim.**
- **G4 passes but does not separate anything.** The sharpened baseline shows the same pattern, because the baseline simply has more to lose in the targeted arm.

## 6 Why BCA and the controls behave alike

### 6.1 The width notices the missing Honeypot move, but it is far too small

At the four Honeypot cells, BCA models, mean over cells and runs:

| | Complete data | Entry withheld |
|---|---:|---:|
| Critic's error on the fatal command | +0.5 | **+10.5** |
| Width at the fatal command | 0.27 | 0.48 |
| Width at the best command | 0.57 | 0.42 |
| Relative width at the fatal command (1.00 is the model's average) | 0.39 | 0.75 |
| Critic's margin, best move over fatal move | 10.6 | 0.54 |
| That margin after subtracting 3 widths | 9.7 | 0.71 |

- **The width roughly doubles at the unseen move** (ratio 1.6 to 2.2 across datasets). The archived design never showed this.
- **But it covers a twentieth of the error.** The critic is wrong by 10.5; the width is 0.48.
- **The penalty moves the margin by 0.17.** Subtracting three widths changes the best-versus-fatal margin from 0.54 to 0.71. That is too little to restore a margin that should be about 10.
- **With complete data the width still points the wrong way.** It is smaller at the fatal command (0.27) than at the best one (0.57), because entering a hazard has a perfectly predictable outcome. That is the likely source of the extra harm in section 3.3: BCA knows this state-specific fact and the shuffled control does not.

### 6.2 The width does not respond at the Bottleneck's unseen crossing

Relative width at the crossing command at B, targeted arm divided by complete arm: 0.93 to 1.04 (exit 4.5), 0.98 to 1.16 (exit 3.5), 0.97 to 1.01 (exit 2.5). The gate needed 1.25. Meanwhile the critic undervalues the unseen crossing by 0.7 to 2.5 at the two lower exit values, and the width (about 0.45) is no larger there than anywhere else.

### 6.3 What the penalty does do

It makes commands more extreme. Average absolute preference goes from 0.56 to 0.77 in the Honeypot and from 0.59 to about 0.80 in the Bottleneck, and BCA and the shuffled control end within 0.02 of each other. A penalty that depends on the command but not usefully on the state acts as a regularizer on the command's shape. That explains why scrambling the state changes almost nothing.

### 6.4 Why calibration does not help at an unseen move

The threshold is set on held-out rows drawn like the training data, and it hits its 90% target on such rows. A move the data never contain is absent from those rows too. On real transitions of commands that could execute the withheld move, coverage was 58% in the Honeypot's targeted arm and 50% to 64% in the Bottleneck's, against 90% nominal.

## 7 Limits

- **Development seeds, three datasets.** Seeds within a dataset are not independent replications of it.
- **A re-implemented learner.** The host was rebuilt in JAX from the archived PyTorch source and checked equation by equation. It is not bit-identical, so these numbers should not be set beside archived ones digit for digit.
- **The Bottleneck fixes were made after I had seen BCA numbers on revision 1.** They rest on baseline-only evidence and were validated on fresh datasets, but that order should be known.
- **One penalty strength.** Strength 3 only, as fixed in the protocol.
- **Thresholds are judgment calls.** 1.25 in G2 and 1 point in G5 were drafted before the run and not reviewed by David.
- **G5 counts cells, not size.** In the Bottleneck all bad-ending rates are under 0.8%, so a failed G5 there reflects a handful of cells.
- **The far start A and coordinate input are not validated**, so nothing here speaks to them.

## 8 What this points to

These are options for you to weigh, not things I have done.

1. **The testbeds now work.** Both produce the effect they are named for with a clean baseline, and they can tell a state-aware method from a state-blind one. That is the reusable result.
2. **The width needs to be much larger off the data, not just calibrated on it.** It already moves in the right direction in the Honeypot. The gap is one of size: about 0.5 against an error of 10.
3. **The width is smallest where outcomes are most predictable, including reliably fatal ones.** Any fix has to separate "predictable" from "safe".
4. **The shuffled-width control should stay in every future comparison.** It is cheap and it is the control that decided this study.
5. **Before spending confirmation data, the design should be frozen with you.** That includes the gate thresholds and whether coordinate input is worth another attempt.

## 9 Files

| File | What it is |
|---|---|
| `results/dev_v1/` | Honeypot study: `records.json` (every policy), `gates.json`, `g0.json`, `tables.md`, run receipts. Also holds the superseded revision 1 Bottleneck runs. |
| `results/dev_v2/` | Bottleneck revision 2 study, same layout. |
| `results/dev_v1_stage*.log`, `dev_v2_stage*.log` | Training logs. |
| `receipts/dev_manifest.json`, `dev2_manifest.json` | Dataset content hashes and how each was made. |

`gates.json` in these two folders has one block of gates per task setting, for BCA only. Since Amendment 3 the evaluator writes one block per method, so running it again on these folders gives the same verdicts in the newer layout.

To re-evaluate saved policies without a GPU:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B experiments/gridworld_ood_v1/evaluate_study.py --label dev_v2
```
