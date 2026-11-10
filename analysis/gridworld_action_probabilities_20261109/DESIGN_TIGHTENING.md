# Tightening the Honeypot and Bottleneck designs

Prepared November 9 2026 for David. **Status: proposal only.** Nothing in this document has been run. It needs your approval before any training, and training is blocked anyway until a PyTorch runtime is available on Zeus. The evidence it rests on is in [REPORT.md](REPORT.md) and [outputs/tables.md](outputs/tables.md), and that evidence comes from seeds that have already been inspected.

> **Correction added later on November 9.** This proposal tightens the measurement and the controls, but it does not make the target actions out of distribution. A follow-up check (`ood_action_audit.py`, results in `outputs/ood_action_audit.json`) found that the Honeypot's "removed" move is still executed 17 to 20 times in the retained data, that no amount of command removal can prevent this under the archived execution rule, and that the Bottleneck removes a whole state instead of one action. Sections 4.1 and 5.3 below are therefore not sufficient. The revised construction (a cutoff in the execution rule, and a gap built into the demonstrator before outcomes are drawn) was approved by David and is built in `experiments/gridworld_ood_v1/`; its README is the current design. The archived environment was not modified, and nothing has been trained.

Quick navigation: [What each test must be able to show](#1-what-each-test-must-be-able-to-show) · [Audit of the current setup](#2-audit-of-the-current-setup) · [Changes that cost nothing](#3-changes-that-cost-nothing) · [Honeypot design](#4-honeypot-tightened-design) · [Bottleneck design](#5-bottleneck-tightened-design) · [Gates and kill rules](#6-gates-and-kill-rules-draft-for-your-approval) · [What stays fixed](#7-what-stays-fixed) · [Decisions for you](#8-decisions-for-you)

Words used throughout: the **baseline** is the matched TD3+BC policy with no width penalty. **BCA** is the same host with the width penalty at strength 3. The **width** is BCA's calibrated uncertainty estimate for one state and command. A **data arm** is one version of the training data (complete, targeted removal, or a control removal). The **sharpened baseline** is the saved baseline policy with its commands scaled up until it is as decisive as its BCA partner, with no retraining. **Points** means percentage points.

## 1 What each test must be able to show

A test is only useful if a working method and a non-working method would produce different results on it. So the first step is to write down, for each testbed, the chain of things that must be true and what BCA's own logic predicts.

| Step in the chain | Honeypot | Bottleneck |
|---|---|---|
| 1. The data gap makes the critic wrong at the target | Critic overvalues the dangerous command | Critic undervalues the bridge |
| 2. That error changes what the baseline does | Baseline chooses the dangerous direction more often than with complete data | Baseline crosses less often than with complete data, because something else now looks better |
| 3. The width is larger where the data are missing | Wider at the removed dangerous commands than with complete data | Wider at B and on the bridge than with complete data |
| 4. BCA's behavior changes for that reason | Dangerous direction falls by more than in the control arms and by more than plain sharpening gives | A penalty on uncertainty should make the unsupported bridge *less* attractive. The question is how much useful crossing BCA gives up |

Step 4 for the Bottleneck deserves a plain statement. The width penalty can only subtract. It cannot raise an undervalued bridge. If the width did its job at B, BCA would be *more* reluctant to cross than the baseline, and the Bottleneck would measure the cost of that caution. BCA crossing *more* often is therefore not something the width-as-uncertainty story predicts.

Where the current designs stand on that chain:

| Step | Honeypot today | Bottleneck today |
|---|---|---|
| 1. Critic wrong | Yes. Error on the dangerous command goes from -1.5 to +4.7 | Yes. Error on the bridge command goes from -0.05 to -2.3 |
| 2. Baseline misled | **No.** Baseline is not more dangerous (2.56% to 1.56% on average) | **Not for the stated reason.** The baseline crosses less, but 71% of the lost probability goes into the hazard above B |
| 3. Width responds | **No.** Relative width 0.60 to 0.58 | **No.** Relative width 0.89 to 0.83 |
| 4. BCA changes behavior for that reason | **No.** Smallest gain of the three arms; plain sharpening reproduces 125% of it | **Opposite sign to the prediction.** BCA crosses more; sharpening reproduces about half |

Both testbeds break at step 2 and step 3. Results at step 4 cannot be credited to the mechanism while the earlier steps fail. That is what "tightening" has to fix.

## 2 Audit of the current setup

Each row is one setup choice, why it was presumably made, what the evidence now says, and a verdict.

### 2.1 Honeypot

| Setup choice | Purpose | What the evidence says | Verdict |
|---|---|---|---|
| Decision cell is 24, where the hazard is directly below | One clear dangerous direction | The hazard is behind the agent (the goal is up and to the right). The best possible policy steps down 0.90% of the time; the baseline already sits at 1.56%. The whole measurable range is 0.66 points. The baseline's "down" preference is already -0.98 of an allowed -1.00 | **Change.** This cell cannot tell methods apart |
| Targeted arm removes rows at cell 24 whose command gives "down" at least 50% | Take away support for the dangerous command without hiding outcomes | Removes 16 to 21 of 256 rows (6-8%). 17 to 20 rows that ended in the hazard remain, and commands with up to 49% "down" remain. The critic's safety margin halves (11.9 to 5.8) but never reaches zero; it would need to overvalue the dangerous command by about 10 and overvalues it by 4.7 | **Strengthen.** The gap is too mild to mislead |
| Removing commands instead of outcomes | Avoid censoring | Correct in principle. Side effect: the remaining examples at that cell are safer, so the imitation part of the actor's objective pulls toward safety | **Keep, and add a control that separates the two effects** |
| Hazard entry is certain: step in, get -5, episode ends | Simple | A handful of rows is enough to learn a certain outcome. Missing support creates little real uncertainty about it. And because the outcome is so predictable, the width is *smaller* for hazard-seeking commands at all 16 hazard-adjacent cells (99-100% of cases) | **This is the deepest issue.** See section 4.3 |
| Control arm removes the same number of random rows at cell 24 | Match the amount of data removed | Sound. BCA's gain there (-1.23 points) is three times the targeted gain (-0.41) | **Keep** |
| Primary readout is goal rate from cell 24 over 200 steps | Task outcome | It moves by 0.6 points and mixes the decision at 24 with everything the policy does afterward | **Demote to secondary.** Lead with the per-step probability of the hazard direction |
| Wider check over 16 hazard-adjacent starts | Catch harm elsewhere | Essential, and it found the real pattern: BCA helps where the hazard is behind the agent and is a coin flip with a bad tail where the hazard is ahead | **Keep, and build the ahead/behind contrast into the design** |

### 2.2 Bottleneck

| Setup choice | Purpose | What the evidence says | Verdict |
|---|---|---|---|
| Opening on row 4, so B is cell 44 | A single crossing point | Cell 34, directly above B, is a hazard. The cell below B (54) teaches "step up", because that is how you reach B. A policy that copies its lower neighbor at B walks into the hazard. Baseline at B: 2.6% up with complete data, 17.7% with bridge rows removed | **Change.** The test is measuring a hidden hazard |
| Targeted arm stops every trajectory before its first action at B or on the bridge | Remove the connecting experience and nothing else | It removes much more. Rows on the right side of the wall fall by 55% and rows on the left by 12%. It also removes every row showing that stepping up from B is fatal (103 to 127 rows in the complete data, 0 in the targeted arm) | **Keep the idea, fix the control** |
| Control arm removes the same total number of random rows | Match the amount of data | Matches the total only. It keeps the left/right balance at 47/53 while the targeted arm is 64/36, and it keeps about 500 to 630 rows at each of B and the bridge | **Add a second control** matched on the left/right balance |
| No alternative to crossing | Simple | Every value on the left side depends on eventually crossing. Undervaluing the bridge lowers all of them together and leaves the ranking alone. All nine archived critics still rank "right" above "left" at B. There is nothing for the bridge to lose to, so the stated failure cannot happen | **Change.** This is why step 2 fails |
| C is cell 46, with a wall directly above | By-product of the map | The cell to the right of C teaches "step up". Copied at C, that is a wall bump. With bridge rows removed, two BCA runs bump that wall 83% and 86% of the time | **Change with the opening** |
| Far start A (cell 90) | Full journey | The baseline fails the complete-data qualification from A in 4 of 9 runs at 25,000 updates and 3 of 9 at 50,000 | **Local scope only** until the baseline qualifies |
| Host sees two coordinates instead of a one-hot cell | Let the policy generalize to cells with no rows | Needed, since B has no rows. But it is also why neighbors' commands leak into B and C | **Keep; it is what makes the layout matter** |

### 2.3 Shared

| Setup choice | What the evidence says | Verdict |
|---|---|---|
| Compare BCA with the baseline inside each data arm | BCA makes every policy more decisive in all 54 pairs, with or without a data gap. A within-arm gain cannot be credited to the gap | **Change the primary comparison** to the difference between arms |
| No control for decisiveness | The sharpened baseline reproduces all of the Honeypot gain and about half of the Bottleneck gain, and beats BCA in the Bottleneck's control arms | **Add it** |
| No direct check that the width responds to the gap | It does not, at either primary probe | **Add it as a gate** |
| No measure of training noise | Two baseline runs that differ only trivially differ by a median of 0.46 points at one cell (90th percentile 2.6 points) | **Add it** |
| Dataset seeds 1234/2234/3234, training seeds 10/11/12 | Inspected many times, including by this analysis | **Development only from now on** |

## 3 Changes that cost nothing

These need no new environment and no training. They apply to the existing policies today and to anything trained later.

1. **Lead with per-step probabilities.** For every decision cell report the probability of the direction in question, next to the best-possible value and the floor. Goal and hazard rates over 200 steps stay as secondary readouts.
2. **Always show the sharpened baseline.** A claim that BCA helps must be a claim that it helps more than making the baseline equally decisive.
3. **Compare arms, not methods within an arm.** The primary number is (BCA minus baseline in the targeted arm) minus (BCA minus baseline in the matched control arm).
4. **Check the width first.** Before reading any behavior, report the relative width at the removed region in the targeted arm against the same region in the complete arm.
5. **Report against training noise.** Use the spread between baseline runs at cells whose data did not change.
6. **State the predicted direction before running.** Section 1 does this.

Items 1, 2, 4 and 5 are already implemented in this folder's scripts and reported for the archived policies.

## 4 Honeypot: tightened design

### 4.1 Stage 1, same environment

**Use families of cells, not one cell.** The map already has eight cells where the hazard is ahead (toward the goal: 20, 31, 33, 44, 57, 68, 85, 96) and eight where it is behind (11, 22, 24, 35, 48, 59, 76, 87). Apply the command removal at all eight cells of a family at once. This gives two targeted arms.

- *Why families:* cell 24 was one arbitrary choice from the easy family. Eight cells per run give eight times the measurements per training run and remove the choice of cell as a free parameter. Cell 24 stays in, inside the "behind" family, so the history remains comparable.
- *Why the ahead/behind split:* it is defined by geometry alone, and it is where BCA and plain sharpening come apart. Sharpening lowers hazard probability in both families. A method that truly responds to missing support should too. The archived BCA does not.

**Remove more.** Use a threshold of 10% on the hazard direction instead of 50%.

- *Why:* at 50% the removal takes 6-8% of a cell's rows and leaves 17 to 20 hazard outcomes. At 10% it takes 30-36% and leaves 3 to 8. The 50% arm is kept for one family as a link to the old results, so there is a two-step dose.
- *What it costs:* about 85 rows per cell, 680 rows per family, 3% of a 22,784-row dataset.

**Match the control to each targeted arm.** For each targeted arm, a control arm that removes the same number of rows at the same cells, chosen at random. This is what the current control does at cell 24.

**Add one information-destroying control method.** Alongside baseline and BCA, train a third method: BCA with the width evaluated at a shuffled set of states inside each batch. It keeps a width-shaped pull on the command but removes what the width knows about the state.

- *Why:* if this control produces the same gains as BCA, the gains do not come from state-specific uncertainty. The sharpened baseline already suggests this, but it is a post-hoc transform. This is the trained version of the same question.

**Separate the critic from imitation (optional, adds runs).** Train with the critic seeing the reduced rows and the imitation term seeing the complete rows, and the reverse. This is the only way to tell "the critic was misled" apart from "the imitation examples became safer". It is a diagnostic configuration, not a method.

Stage 1 arms: complete, ahead-10%, behind-10%, behind-50%, and a matched random control for each targeted arm. That is 7 arms x 3 datasets x 3 training seeds = 63 warm-ups, each branched into baseline, BCA and shuffled-width: 189 final policies. A leaner version drops the 50% link arm and its control: 5 arms, 135 final policies.

### 4.2 What Stage 1 can and cannot settle

It can settle whether command removal in this environment can ever mislead the baseline (step 2), whether the width responds to a larger gap (step 3), and whether BCA does anything beyond decisiveness. It cannot create a lure that the environment does not contain.

### 4.3 Stage 2, only if Stage 1 fails at step 2: a hazard with an uncertain payoff

The current hazard always pays -5 and always ends the episode. Two consequences follow, and both undercut the test:

- A certain outcome needs very few examples to learn, so removing support barely creates uncertainty about it.
- A certain outcome has almost no one-step spread, so a width built from one-step spread is at its *smallest* exactly on the most reliably fatal commands.

A honeypot in the proper sense is an action that looks good in a small sample and is bad on average. That requires an uncertain payoff. The proposed change is one cell: replace one hazard with a **gamble cell** that ends the episode with a large reward most of the time and a large loss the rest of the time, set so that its average value is slightly below the safe route.

- *Why this is justified:* with an uncertain payoff, how wrong the critic is depends on how many examples it has, in a way that can be calculated in advance. If the safe route is worth `g` more than the gamble on average, the payoff has spread `s`, and `n` examples remain, the chance that the data make the gamble look better is roughly the chance that a normal variable falls below `-g * sqrt(n) / s`. Support becomes a dial with a known effect, instead of a deletion whose effect has to be discovered.
- *Worked example:* reward +10 with probability 0.7 and -5 with probability 0.3 gives an average of 5.5 and a spread of 6.9. If the safe route is worth 6.5, then with 128 examples about 5% of datasets are misled, with 47 about 16%, with 12 about 31%.
- *The mirror arm that keeps the test honest:* add a second version in which the gamble is slightly *better* than the safe route. A method that understands support should avoid the bad gamble when examples are few and take the good gamble when examples are many. A method that only dislikes noisy outcomes, or only sharpens, avoids both or takes both. The four cells of {bad gamble, good gamble} x {many examples, few examples} have different predicted patterns for each kind of method.
- *What it costs:* a new environment revision, which must be requalified from scratch (transition, reward and terminal tests; complete-data baseline competence). I recommend not paying that until Stage 1 shows the current hazard cannot do the job.

## 5 Bottleneck: tightened design

### 5.1 Move the opening from row 4 to row 5

One wall cell closes (45) and one opens (55). B becomes cell 54 and C becomes cell 56.

*Why row 5:* I checked every row the opening could sit on, using only the map and the best-possible policy.

| Opening on row | B | C | Hazard next to | Best command at B's supported neighbors | Neighbor whose command, copied at B, enters a hazard | Best possible goal / hazard rate from A |
|---|---|---|---|---|---|---|
| 0 | 4 | 6 | none | 14: up; 3: right | none | 98.9% / 1.1% |
| 4 (today) | 44 | 46 | **B** | 54: up; 43: right | **54** | 98.1% / 1.9% |
| 5 | 54 | 56 | none | 44: down; 53: right | none | 99.9% / 0.1% |
| 7 | 74 | 76 | C | 84: up; 73: right | none | 98.1% / 1.9% |
| 9 | 94 | 96 | C | 84: down; 93: right | none | 96.3% / 3.7% |

Rows 1, 2, 3, 6 and 8 are unusable because B or C would be a wall or a hazard.

- Only rows 0 and 5 keep hazards away from B, the bridge and C.
- Row 0 puts C three steps from the goal, so there is almost no journey left after the bridge.
- Row 5 keeps the journey shape. At B the neighbor from above says "down", which at B is a wall bump: a harmless delay instead of a death. C has all four directions open, so the wall-bump trap at today's C disappears.
- The best possible policy from A improves from 98.1% goal to 99.9%. A failed run is then more clearly a learning failure and less the map's own risk.

With this change, failing to cross shows up as delay, time-outs and lost return. That is what a failure to connect two halves of a journey should look like.

*The alternative:* keep row 4 and delete the hazard at cell 34 from the Bottleneck map. It is the smaller mental change, but the two tasks would no longer share one hazard set, and today's wall-bump trap at C would remain.

### 5.2 Give the agent something to lose the bridge to

Add a small end-of-episode reward on the left side, a **low-yield exit**, set so that walking to it from B is worth less than crossing truly is and more than the critic believes crossing is when the bridge rows are missing.

- *Why this is necessary, not optional:* today every value on the left side is built from the value of eventually crossing. Undervalue the bridge and they all fall together. The ranking does not change, so the agent heads for B regardless, and the archived critics confirm it. An option whose value does not depend on the bridge is the only thing an undervalued bridge can lose to. Without one, the Bottleneck cannot fail for its stated reason.
- *It restores the original idea.* The handoff notes that the implemented task dropped the low-yield versus high-yield choice from the original concept.
- *Set its size by rule, and as a dial.* True crossing value at the new B is about 5.3. With bridge rows removed, the archived critics put it at about 2.1 to 3.2. Three settings for the exit's value seen from B (about 4.8, 3.8 and 2.3) give true margins of about 0.5, 1.5 and 3.0. The result is then a curve: how large an undervaluation it takes to flip the choice. That is more informative than one hand-picked setting and cannot be accused of being tuned to a single outcome.
- *New primary readout:* from B, the probability of ending at the goal, at the low-yield exit, or timing out.
- *What it costs:* a new environment revision to requalify, like Stage 2 of the Honeypot.

### 5.3 Fix the control arms

Keep complete data, the fragment arm, and the count-matched random arm. Add a **side-matched** control: remove rows at random *within* each side of the wall until each side has as many rows as in the fragment arm, leaving B and the bridge with their proportional share. Fragment minus side-matched isolates "no rows at B and on the bridge" from "far fewer rows on the right".

### 5.4 Scope

B-start and C-start are the primary scope. The far start A is reported but not interpreted until the complete-data baseline passes its existing thresholds from A in every run. No threshold is relaxed.

## 6 Gates and kill rules (draft for your approval)

Your standing rule is that every new test has its kill rule written before it runs. These are drafts. The numeric thresholds are judgment calls and are yours to set. "Dataset" means the mean over that dataset's training seeds; there are three datasets, so "2 of 3" is the bar throughout, and all three are reported.

| Gate | Passes if | If it fails |
|---|---|---|
| G0 Environment and baseline | The new revision passes its contract tests, and the complete-data baseline meets the existing competence thresholds at the primary starts in every run | Stop. Fix the testbed. No BCA arm is trained |
| G1 The gap misleads the baseline | In the targeted arm the baseline's probability of the target direction is worse than in the matched control arm by more than the training-noise spread, in at least 2 of 3 datasets | **Kill rule for the testbed.** Shelve it as a behavior test. For the Honeypot this triggers Stage 2 |
| G2 The width responds | Relative width at the removed region is at least 1.25 times its complete-data value in at least 2 of 3 datasets | Behavior is still reported, but not credited to uncertainty. Label it a regularizer effect |
| G3 More than decisiveness | BCA beats both the sharpened baseline and the shuffled-width control on the primary readout in at least 2 of 3 datasets | **Kill rule for the claim.** Report "no benefit beyond decisiveness" |
| G4 Specific to the gap | The difference between arms has the predicted sign in at least 2 of 3 datasets | Report the effect as general, not gap-specific |
| G5 No wider harm | Over all hazard-adjacent cells, BCA does not raise the hazard-direction probability by more than the training-noise spread in more cells than the sharpened baseline does | Report the harm next to the benefit. No safety claim |

Applied to the archived policies, as a description and not a test:

- G1 fails for the Honeypot. It passes for the Bottleneck, but through the hazard above B.
- G2 fails for both.
- G3 fails for the Honeypot. For the Bottleneck it cannot be fully judged, because no shuffled-width control exists; BCA beats the sharpened baseline in the targeted arm and loses to it in both control arms.
- G5 fails for the Honeypot.

Confirmation needs dataset seeds and training seeds that have never been inspected. Dataset seeds 1234, 2234, 3234 and the fresh sets described in the handoff are all development from now on.

## 7 What stays fixed

Changing these would make the new results incomparable with the old ones for no gain in clarity:

- Grid size, rewards (-0.1 per step, +10 goal, -5 hazard), discount 0.95, execution setting 2, the 200-step evaluation cutoff.
- The host: TD3+BC with the archived networks, optimizer, batch size, target noise, delays and the 10,000-update shared warm-up before branching to 25,000.
- BCA's width construction and penalty strength 3. The point is to test the method as it is.
- How data are collected: half uniform commands, half noisy best commands.
- Removal by command or by position, never by outcome.
- The record-keeping in section 13 of the handoff, including step-level traces.

## 8 Decisions for you

1. **Honeypot.** Run Stage 1 in the current environment first (my recommendation, since it is cheap and tells us whether the present hazard can ever mislead), or go straight to the gamble cell?
2. **Bottleneck layout.** Move the opening to row 5 (my recommendation) or delete the hazard at cell 34?
3. **Bottleneck low-yield exit.** Add it? Without it I do not think the Bottleneck can test what it is named for.
4. **Thresholds** in section 6, in particular the 1.25 in G2 and what counts as "more than the training-noise spread".
5. **The rule for calling a probability shift real.** This one is set up for you to write: see `judge_shift` in `measure_action_probabilities.py`.
6. **Runtime and approval.** Any of the training above needs PyTorch on Zeus and your go-ahead. For sizing: a plain 25,000-update baseline run took about 112 seconds on one CPU thread in the archived logs. The matched BCA branches cost more per update and I did not find their timing in the files I read, so a short smoke run has to measure it before any launch estimate is trusted.
