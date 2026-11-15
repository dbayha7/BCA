# Protocol: first training study on the out-of-distribution-action testbed

Written November 9 2026, before any policy was trained on the study datasets. David asked for the experiments to be run on this machine's GPU. **This is a development study.** Dataset seeds 5101 to 5103 were used to shake down the design, so nothing here can be confirmation.

Quick navigation: [Question](#1-question) · [What is trained](#2-what-is-trained) · [The learner](#3-the-learner-and-how-it-differs-from-the-archive) · [What is measured](#4-what-is-measured) · [Gates and kill rules](#5-gates-and-kill-rules) · [Rules for running](#6-rules-for-running) · [Amendment 1](#amendment-1-made-before-any-study-dataset-was-used) · [Amendment 2](#amendment-2-bottleneck-revision-2-written-before-any-policy-was-trained-on-its-datasets) · [Amendment 3: the three method changes](#amendment-3-three-changes-to-the-method-written-before-any-policy-was-trained-with-them) · [Amendment 4: support penalty alone](#amendment-4-does-the-calibrated-width-add-anything-to-the-support-penalty-written-before-any-run-of-the-new-arm) · [Amendment 5: blindfolded calibrated width](#amendment-5-is-the-calibrated-widths-knowledge-of-the-state-used-written-before-any-run-of-the-new-arm) · [Amendment 6: confirmation](#amendment-6-confirmation-on-untouched-datasets-written-before-any-confirmation-dataset-was-generated) · [Amendment 7: Version A against Version B](#amendment-7-second-fresh-data-block-version-a-against-version-b-written-before-any-of-its-datasets-was-generated) · [Amendment 8: Version A on nine datasets](#amendment-8-version-a-on-nine-fresh-datasets-written-before-any-of-its-datasets-was-generated) · [Amendment 9: competence per dataset](#amendment-9-baseline-competence-checked-per-dataset-written-before-any-of-its-datasets-was-generated) · [Amendment 10: both environments, nine datasets](#amendment-10-version-a-on-nine-fresh-datasets-in-both-environments-written-before-any-of-its-datasets-was-generated) · [Amendment 11: thin data](#amendment-11-thin-data-and-a-baseline-only-check-of-it-written-before-any-thin-dataset-was-generated) · [Amendment 12: calibrated permission](#amendment-12-a-calibrated-permission-to-leave-the-demonstrators-move-written-before-any-policy-was-trained-with-it) · [Limits](#7-limits-known-in-advance)

## 1 Question

When one move at a cell is truly absent from the data:

1. Does the plain baseline get misled about it?
2. Is BCA's width larger at that move than it is when the move is in the data?
3. Does BCA change behavior at that move differently from two controls that have no state-specific uncertainty?

The Honeypot's unseen move is fatal. The Bottleneck's unseen move is the best one available. A method that is cautious about unseen moves should avoid both: that protects in the Honeypot and costs something in the Bottleneck.

## 2 What is trained

| Factor | Levels |
|---|---|
| Task | Honeypot; Bottleneck with the low exit paying 4.5, 3.5 or 2.5 (3.5 is the primary setting) |
| Dataset seed | 5101, 5102, 5103 |
| Data arm | complete, targeted, placebo |
| Training seed | 20, 21, 22 |
| Method | baseline, BCA, shuffled-width control |

That is 108 warm-ups and 324 final policies. Each run is 10,000 shared baseline updates, a snapshot, then 15,000 more updates per method: 25,000 in total, as in the archived matched studies.

- **Baseline.** TD3+BC with no width penalty. It still fits and refreshes the scale network, so the three methods do identical auxiliary work.
- **BCA.** The same, plus the width penalty at strength 3 in the actor objective.
- **Shuffled-width control.** The same penalty, but the width is evaluated at a random reordering of the batch's states. It keeps a width-shaped pull on the command and removes what the width knows about the state.
- **Sharpened baseline.** Not trained. It is the baseline policy with every command scaled by one number until it is as saturated as its BCA partner.

**Matching.** The three methods of a run start from the same snapshot and see the same minibatches and the same noise. The three data arms of a dataset also see the same minibatch row numbers, and the arms are byte-identical outside the decision cells, so arms differ only through the rows at those cells.

**Calibration data.** Each task, arm and dataset has a reference bank and an audit bank of 1,280 rows each, collected fresh with the same collector and the same gap (seeds: dataset seed + 100,000 and + 200,000). The reference bank is used only to set the width's threshold. The audit bank is used only for reporting.

## 3 The learner, and how it differs from the archive

The archived host is PyTorch, and no PyTorch is installed on this machine. `learner.py` re-implements it in JAX. **It is not bit-identical to the archived host.** Random streams and the framework differ, so no number here should be compared digit for digit with an archived number.

What is the same, with its source:

| Piece | Value | Archived source |
|---|---|---|
| Networks | actor and two critics, two hidden layers of 64, PyTorch default initialization | `train.py`, `td3bc_coordinates.py` |
| Inputs | Honeypot one-hot of length 100; Bottleneck two coordinates | same |
| Optimizer | Adam, 3e-4, batch 256 | `train.py` |
| Critic target | `r + (1 - done) * 0.95 * min(Q1', Q2')` at the target actor's command plus clipped noise (0.2, clip 0.5) | `td3bc_source_actor.py` |
| Actor objective | `-c * mean(Q1) + mean((pi - a)^2) + c * 3 * mean(W)`, `c = 2.5 / (mean|Q1| + 1e-7)` held constant | `bca_direct_width.py` |
| Actor and target updates | every second update, targets move by 0.005 | `td3bc_source_actor.py` |
| Scale fit | `(weighted soft coverage - 0.9)^2 + 0.005 * weighted eta^2`, Adam 1e-3, residual unit as a 0.99 moving average | `bca_direct_width.py` |
| Order within an update | fit the scale, update the host, refresh the reference after updates 10,000, 15,000, 20,000 | `bca_direct_width.scheduled_step` |
| Calibration | threshold from 1,280 held-out scores, alpha 0.1, credibility 0.95, 1,000 draws | this repository's `calibration/` package |

The width is absent from the critic's target and loss. It reaches the policy only through how the width changes with the command, with the reference frozen.

The scale network, the coverage function and the calibration are not re-implemented. They are imported from this repository's `calibration/network.py`, `reference.py` and `wbcp.py`, which are byte-identical to the copies the archived port was pinned against.

Checks done before this protocol was written (`test_learner.py`, 21 tests, passing on CPU and on the GPU):

- Every equation above against an independent NumPy version.
- The actor loss and the Bellman target against the reference calculator in the offline-RL audit skill.
- Gradient ownership: `c` is a constant, the value path is differentiated, the reference gets no update, the command path through the width is live.
- Update order, the every-second-update rule, and that the critic never sees the width.
- The baseline is unaffected by any reference; the three methods share minibatches; many runs in parallel equal each run alone.
- The scale network starts at exactly the archived port's starting values for an archived seed.

One setting was needed for the GPU: matrix products are forced to true 32-bit precision. Without it the GPU was about 3e-4 away from the NumPy equations.

## 4 What is measured

Everything is exact, computed from each policy's command at every cell under the cutoff execution rule, over 200 steps.

| | Honeypot | Bottleneck |
|---|---|---|
| Decision cells | 24, 33, 35, 44 | B (54) |
| Target move | entering hazard 34 | stepping right onto the opening |
| Probability readout | executed probability of the target move, averaged over the four cells | executed probability of the target move at B |
| **Primary outcome** | **H**: probability an episode starting at a decision cell ends in hazard 34, averaged over the four cells. Lower is better | **X**: probability an episode starting at B reaches the goal. For a method cautious about unseen moves, X should fall |
| Also reported | goal, any hazard, time-out from every cell; values; A-start | goal, low exit, time-out, hazard from B, A and C |

Width readout: the width at the command that executes the target move with certainty at a decision cell, divided by that model's average width over all cells and corner commands.

## 5 Gates and kill rules

"Dataset" means the mean over that dataset's three training seeds. There are three datasets; "passes" means at least 2 of 3, and all three are reported. Gates are judged per task setting.

| Gate | Passes if | If it fails |
|---|---|---|
| **G0** Baseline competence | With complete data, every baseline run reaches the goal at least 90% of the time with value at least 3 from each primary start (Honeypot 24, 33, 35, 44; Bottleneck B and C). The A-start (90% and value 1.5) is reported, not gated | **Stop for that task setting. No BCA or control policy is trained on it** |
| **G1** The gap misleads the baseline | The baseline's probability readout in the targeted arm differs from the placebo arm in the predicted direction (Honeypot higher, Bottleneck lower) by more than the complete-versus-placebo difference | **Kill rule for the testbed as a behavior test.** BCA and control arms are still trained, because G2 does not depend on G1, but no claim about protection or cost is made. For the Honeypot this is the trigger for the uncertain-payoff design |
| **G2** The width responds | BCA's width readout in the targeted arm is at least 1.25 times its value in the complete arm | Behavior is reported but not credited to uncertainty |
| **G3** More than decisiveness | Honeypot: H for BCA is below both the sharpened baseline and the shuffled-width control. Bottleneck: X for BCA is below both | **Kill rule for the claim:** report no effect beyond decisiveness |
| **G4** Specific to the gap | (BCA minus baseline in the targeted arm) minus (BCA minus baseline in the placebo arm) is negative for H, or for X | Report the effect as general, not gap-specific |
| **G5** No wider harm | Counting start cells where the chance of a bad ending (hazard or time-out) rises by more than 1 point over the baseline: BCA has no more such cells than the sharpened baseline | Report the harm beside any benefit; no safety claim |

Ties count as failing a strict comparison. In particular, if every method drives H to exactly zero, G3 fails: nothing has been shown beyond what the controls do.

## 6 Rules for running

- The study is run once. If a defect forces a rerun, the defect, the fix and the discarded outputs are recorded; nothing is rerun to change an outcome.
- No hyperparameter is changed after outcomes are seen. Penalty strength stays 3.
- Every run is kept and reported, including any whose calibration fails. A failed calibration is recorded and that run's width stays off; it is never silently replaced.
- A smoke test on a dataset seed that is not part of the study (seed 7) checks speed and that the host learns at all, before any study dataset is used.
- Stage 1 trains baselines and evaluates G0 and G1. Stage 2 trains BCA and the shuffled control from the saved snapshots, only for task settings that passed G0.

## Amendment 1, made before any study dataset was used

The smoke test and a six-seed pilot ran on dataset seed 7, which is not part of the study. Baselines only, complete data, Bottleneck with the 3.5 exit. They showed two faults in the Bottleneck plan above. Both are fixed here, before stage 1, and the original text is left as written.

**What the pilot showed.**

| Network input | Goal from C | Goal from B | Low exit from B | What happens |
|---|---|---|---|---|
| Two coordinates | 0% in 4 of 6 seeds, 91% and 100% in the others | 0% in 4 of 6 | 0 to 21% | After crossing, the policy keeps stepping right along row 5 and walks into hazard 58, two cells past C |
| One-hot cell | 100% in 6 of 6 | 66 to 100% | 0 to 34% | Competent after the bridge. At B it splits between crossing and the low exit. No hazard, no time-out |

1. **A layout fault I missed.** Moving the opening to row 5 keeps hazards away from B, the opening and C, which is all I checked. But hazard 58 sits on the same row two cells past C. A network that reads coordinates carries "step right" one cell too far. Under the cutoff rule that is fatal every time.
2. **A gate that did not fit the task.** G0 demanded that the baseline reach the goal 90% of the time from B. With a low exit beside B, a competent baseline can take that exit part of the time. That is a legitimate ending, not a failure.

**Changes.**

- The Bottleneck is trained with both archived input encodings: one-hot cells (the archived Honeypot host's input) and two coordinates (the archived Bottleneck host's input). Each is judged separately. Per-cell data make one-hot viable here, because B is no longer a cell without rows. I expect the coordinate version to fail G0 because of hazard 58; it is run so that the failure is on record for the study data.
- G0 for the Bottleneck becomes: from C, goal at least 90% and value at least 3; from B, hazard plus time-out at most 10%. G0 for the Honeypot is unchanged.
- Whether the complete-data baseline crosses often enough to be informative is left to G1, which needs the targeted arm to cross less than the placebo arm by more than the spread.

The study now has 162 warm-ups and up to 486 trained policies. Nothing else in sections 1 to 6 changes.

## Amendment 2: Bottleneck revision 2, written before any policy was trained on its datasets

David's instruction after stage 1 of study v1: fix the parts of the environments that need fixing to pass, and report how BCA does only after the environment is verified and validated.

**What needed fixing.** Only the Bottleneck. The Honeypot passed its competence gate in 9 of 9 runs and its gap gate in 3 of 3 datasets, so it is left exactly as it was.

Both Bottleneck faults were diagnosed from baseline policies trained on complete data. No BCA or control policy was used to find or to choose a fix.

| Fault | Evidence from study v1 baselines, complete data | Fix in revision 2 |
|---|---|---|
| Hazard 58 sits on the opening's row, two cells past C | With coordinate inputs, 65 to 75% of episodes starting at C ended in hazard 58. With one-hot inputs, one of the two failing runs lost 10.2% of episodes from C to it | Hazard 58 is removed from the Bottleneck. Its hazards are 21, 34 and 86 |
| A move blocked by a wall costs the same as a step | Under the best policy a blocked move is worth only 0.23 to 0.60 less than the best move (0.40 at the opening), inside the critic's error. Under the cutoff rule a chosen blocked move never ends. One run pushed into the wall above the opening 93% of the time and timed out. With one-hot inputs, 14 to 17 cells stalled across the nine runs at exit 3.5 and 2.5 | A blocked move costs 1.0 in the Bottleneck, ten times a step. The gap to the best move becomes 1.13 to 1.50 |

**Pilot of the fix.** Dataset seed 7 (not in any study), baselines only, complete data, six training seeds per setting. Entries are runs passing the competence gate, for exit 4.5 / 3.5 / 2.5.

| Layout | One-hot input | Coordinate input | Stalled cells, one-hot |
|---|---|---|---|
| Revision 1 | 6/6, 6/6, 6/6 | 0/6, 1/6, 1/6 | 7, 8, 10 |
| Hazard 58 removed | 6/6, 6/6, 6/6 | 5/6, 5/6, 6/6 | 1, 2, 3 |
| Hazard 58 removed and blocked move costs 1.0 | 6/6, 6/6, 6/6 | 6/6, 6/6, 6/6 | 0, 0, 0 |

1.0 was the only blocked-move cost tried.

**What I had already seen.** When I made these fixes I had seen BCA results on revision 1. To keep the fixes from leaning on them, the fixes rest only on the baseline evidence above, revision 2 gets new datasets (seeds 5201 to 5203, 27 datasets, all passing the unseen-move checks), and nothing about BCA, its strength or the gates G2 to G5 is changed.

**Verification and validation of revision 2, required before any BCA or control policy is trained on it.**

- Verification: the environment contract tests including eight new ones for revision 2 (it differs from revision 1 only in hazard 58 and the blocked-move cost; no hazard on the opening's row; stepping right from C never meets a hazard; the best route from B passes no hazard; crossing is still worth 5.563 and beats every exit value). The unseen-move checks on all 27 datasets. The learner tests.
- Validation: G0 and G1 as amended in Amendment 1, on the new datasets, for each input encoding and exit value separately. A setting that fails G0 gets no BCA or control policy.

**What happens to the revision 1 Bottleneck results.** They stay on disk under `results/dev_v1`. Revision 1 is not a validated environment, so they are not reported as findings about BCA.

**The rewards now differ between the two tasks** in one respect: a blocked move costs 0.1 in the Honeypot and 1.0 in the Bottleneck. The Honeypot showed no stalls (0 stalled cells in 9 baseline runs) and passed validation unchanged, so I did not change an environment whose BCA results I had already seen.

## Amendment 3: three changes to the method, written before any policy was trained with them

Written November 9 2026, after the results of studies v1 and v2 were known (`RESULTS.md`) and before any run of the changed method, the smoke test included.

**Why.** On the validated environments the original BCA did lower the Honeypot's hazard endings, but a control that scrambles which state the width is read at did the same to within 0.1 points, and plain sharpening got most of the way. At the unseen fatal move the width was about 0.5 while the critic's error was about 10. At the Bottleneck's unseen crossing the width did not respond at all. David's instruction: fix the design however we can so that the method helps in these situations, then "Build and test" for the three changes below.

**What does not change.** The environments, the datasets, the data arms, the gates G0 to G5, every pass mark, the penalty strength of 3, the seeds, and the 10,000 + 15,000 update schedule. Only the method changes. I do not alter an environment, a gate or a pass mark to help any method pass.

### The three changes

Terms used below. *Width* W(s, a): the calibrated size of the critic's one-step error at state s and command a, exactly as before. *Recorded command*: a command that appears in the data at that state. *Reference rows*: the 1,280 held-out rows already used to calibrate the width.

**1. Support-aware width.** The original width is fit only on recorded commands, so at a command the data never contain it is an extrapolation, and it came out small. The change adds a direct measure of whether the data hold a similar command at that state.

- *Support score* z(s, a). A small network (two hidden layers of 64, like the critics) is trained in every arm, at every update, to tell a recorded (state, command) pair from the same state paired with a command drawn uniformly from the command cube. Its output is the log-odds that the pair is recorded. High means the data hold similar commands at that state; very low means they hold none. Logistic loss, Adam at 1e-3.
- *Floor* z0. At each refresh: the 1st percentile of z over the recorded commands of the reference rows. A command scoring below it is less supported than 99% of what the data contain.
- *Cap* Wmax. At each refresh: half the range (largest minus smallest, divided by 2) of the plain one-step targets on the reference rows. It is the largest error a critic that knew nothing could make.
- *Support-aware width*: `Ws(s, a) = W(s, a) + d * max(Wmax - W(s, a), 0)`, with `d = max(z0 - z(s, a), 0) / ln(10)`. d counts how many factors of ten in odds the command lies below the floor.

A command at or above the floor keeps its original width exactly. One factor of ten below the floor the width reaches the cap. It is not held at the cap beyond that: it keeps growing at the same rate, so that anywhere inside a gap there is a slope pointing back toward the data.

**2. One-sided penalty.** The original penalty is 3 x W, so the policy is also paid for making W smaller than usual. That is the pull toward decisive commands that the shuffled control reproduced. The change charges only for width above a level L: `penalty = 3 * max(Wx - L, 0)`. Wx is whichever width the method uses (W or Ws). L is the 90th percentile of Wx over the recorded commands of the reference rows, set at each refresh. 90 matches the 90% coverage the width is calibrated for.

**3. Excess width in the critic's target.** The original method never lets the width touch the critic. The change subtracts the same one-sided excess at the next state: `target = r + (1 - done) * 0.95 * [min(Q1', Q2')(s', a') - 3 * max(Wx(s', a') - L, 0)]`, with a' the target policy's smoothed command as before. In these arms the same target is used wherever a one-step target is formed (critic update, scale fit, calibration scores), so the width keeps measuring fit error and does not start counting the penalty itself as error. The cap Wmax is always taken from the plain target.

### What is trained (study v3)

| Method | Width penalty | Support-aware | One-sided | In the critic | Its control |
|---|---|---|---|---|---|
| baseline | no | | | | |
| BCA as it was | yes | | | | shuffled |
| support-aware width | yes | yes | | | shuffled |
| support-aware + one-sided | yes | yes | yes | | shuffled |
| support-aware + one-sided + in the critic | yes | yes | yes | yes | shuffled |
| one-sided only | yes | | yes | | shuffled |

Each method has its own shuffled control: identical switches, but every width (and support score) is read at a random reordering of the batch's states. Each method also gets its own sharpened baseline, the baseline's commands scaled to that method's saturation.

Settings: only the validated ones. The Honeypot on its datasets (seeds 5101 to 5103) and the Bottleneck revision 2 with one-hot input at exit values 4.5, 3.5 and 2.5 on its datasets (seeds 5201 to 5203). Training seeds 20, 21, 22. The baselines are trained again, because the support score has to be trained during the shared first 10,000 updates. That is 108 warm-ups and 1,080 further policies. G0 and G1 are judged again on the new baselines, with the same rules.

All five methods run the same code. Before the first refresh every arm is identical to the baseline. The first three (baseline, BCA as it was, its shuffled control) are checked to reproduce the numbers the learner gave before this amendment (`test_learner_regression.json`).

### Settings fixed here, and why

| Setting | Value | Reason |
|---|---|---|
| Support floor | 1st percentile of the score on reference rows | A command must be rarer than 99% of the data before its width is raised. About 1% of ordinary commands will be raised a little; that is the stated false-alarm rate |
| Ramp | cap reached one factor of ten below the floor, linear, not held at the cap | Ten times rarer than the rarest 1% is treated as "no data". Not holding at the cap keeps a slope everywhere in a gap |
| Cap | half the range of the plain one-step targets on reference rows | The largest error possible for a critic with no knowledge |
| One-sided level | 90th percentile of the consumed width on reference rows | Matches the 90% coverage target (alpha 0.1) |
| Penalty strength | 3, in the actor and in the critic | Unchanged from the original method |
| Support network | two hidden layers of 64, Adam 1e-3, one uniform command per batch row | Same size as the critics, same rate as the scale network |
| Percentiles | NumPy linear interpolation | Stated so the numbers can be reproduced |

None of these is tuned after an outcome is seen. The smoke test (dataset seed 7, not a study seed) checks only that the code runs, how long it takes, and that no saved number is non-finite. Its outcomes are not used to choose a setting.

### Gates and kill rules for this amendment

G2 to G5 are judged separately for each of the five methods, with the pass marks of section 5:

- **G2** uses the width the method consumes (Ws for the support-aware methods) at the target command.
- **G3** compares the method with its own shuffled control and its own sharpened baseline. **Kill rule, per method:** if G3 fails, no effect beyond the controls is claimed for that method.
- **G4** and **G5** as written, with the method in place of BCA and its own sharpened baseline in G5.

Reported beside the gates, not gated: the share of target-move rows the support score places below the floor (does it see the gap?), the same share on fresh ordinary rows (false alarms, expected near 1%), the penalty at the target command and at the best command, and calibration coverage.

**What would count as helping.** In the Honeypot: a method that passes G2, G3, G4 and G5. In the Bottleneck the unseen move is the best one, so a method that is cautious about unseen moves must make X fall. That is the price of the same caution, not a failure of the test. A method that helps in the Honeypot and does not pay in the Bottleneck would not be acting on "unseen"; I would treat that as a warning sign.

**Predictions, written before the run.** (a) BCA as it was repeats its v1 and v2 verdicts. (b) One-sided only does little, because the original width at the unseen move is not above its own 90th percentile; it should lose the sharpening effect and land near the baseline. (c) The support-aware methods pass G2 if the support score finds the gap, which is itself in question for a network of this size. Whether they pass G3 and G5 is the open question. (d) If they work, X in the Bottleneck falls toward zero.

**Limits.** These changes were designed after seeing BCA's results on these same datasets, so everything here is development. Five methods, four task settings and four gates are many comparisons; a single pass can be luck, and only a consistent pattern is worth carrying forward. A method that passes becomes a candidate to freeze and test on new, untouched datasets. Nothing here is confirmation. The support score is a new ingredient: it is not part of the conformal calibration and carries no coverage guarantee.

**Verification before the run.** `test_learner.py`, 39 tests, passing on CPU and on the GPU: each new equation against an independent NumPy version; the regression check above; no gradient reaches any frozen quantity; the command gradient is live inside a gap and exactly zero at supported commands under the one-sided penalty; only the critic arms change the critic, the scale fit or the calibration residual; every arm fits the same support network; a control differs from its method only by the shuffle.

## Amendment 4: does the calibrated width add anything to the support penalty? Written before any run of the new arm

Written November 9 2026, after the results of study v3 were known ([RESULTS_METHOD_CHANGES.md](RESULTS_METHOD_CHANGES.md)) and before any run of the arm below, the smoke test included. David approved this test the same evening ("Yes, run it").

**Why.** The support-aware width (change 1) took the Honeypot's hazard endings to zero. But at the fatal command the width it uses is 82, and the calibrated width's share of that is 0.5. So it is fair to ask whether BCA's calibrated width is part of why it works.

**The new arm: support penalty only.** It is change 1 with the calibrated width replaced by zero:

- Change 1 uses `Ws = W + d * max(Wmax - W, 0)`.
- The new arm uses `Wo = d * Wmax`.

W is the calibrated width, Wmax the cap, and d the number of factors of ten the command's support score lies below the floor, all exactly as in Amendment 3. The penalty is 3 x Wo in the actor. At a supported command Wo is exactly zero, so nothing is charged there and nothing is rewarded. The scale network is still fitted and calibrated in this arm, so every arm does the same auxiliary work; the arm just does not use it. It gets its own shuffled control and its own sharpened baseline.

**What is trained (study v4).** The same validated settings, datasets and seeds as study v3. The baselines are trained again and must be bit-identical to study v3's. Change 1 and its control are trained again in the same run, so the tables sit side by side; they must reproduce study v3 exactly, and that is a check of the code, not a second sample. Then the new arm and its control. 108 warm-ups and 432 further policies.

**What the comparison isolates.** The new arm and change 1 share the snapshot, the minibatches, the noise, the support score, the floor and the cap. They differ only in whether the calibrated width is in the sum.

### Decision rule

Honeypot, targeted arm, the nine runs.

| Question | "The calibrated width adds nothing" if | "It adds something" if |
|---|---|---|
| Protection. Change 1 had H = 0 in 9 of 9 runs | The new arm has H = 0 in at least 8 of 9 runs | The new arm has H above 0 in at least 3 of 9 runs, in at least 2 of the 3 datasets |
| Value. Average true value of an episode over all start cells: baseline 5.04, change 1 5.95 | The new arm reaches 5.85 or more (within 0.10 of change 1) | The new arm is below 5.50 (closer to the baseline than to change 1) |

Anything between the two columns is undecided for that question.

**Kill rule.** If both rows land in the first column, the calibrated width contributes nothing on this testbed, and I will say that BCA's calibration is not part of why change 1 works.

**What this test cannot do.** If either row lands in the second column, the calibrated width contributes something, but this arm cannot say how. Its one established effect is a general pull toward decisive commands, and the shuffled-width control of studies v1 to v3 reproduced that pull with no knowledge of the state. Separating the two would need one more arm (the support raise read at the true state, the calibrated width read at shuffled states). That arm is not part of this amendment and is not run. So this test can take credit away from the calibrated width or leave the question open. It cannot award credit to calibrated uncertainty.

**Also reported:** G2 to G5 for the new arm, the Bottleneck's X and the value of an episode from B, the wider-harm counts, and stalls.

**G2 for this arm is read in absolute units.** Its width is zero at every supported command, so "relative to the model's average width" has no stable meaning. G2 compares the width it uses at the target command, targeted arm against complete arm, with the same pass mark of 1.25. A complete-arm value of exactly zero passes when the targeted-arm value is positive.

**Prediction, written before the run.** The new arm behaves like "support-aware + one-sided" did in study v3, because both lack the payment for below-average width: H between 1% and 4% with most runs above zero, value between 5.0 and 5.3, and X near zero in the Bottleneck. That would put both rows in the second column.

**Nothing else changes.** Environments, datasets, gates, pass marks, penalty strength 3, the support score and its settings, the schedule. No setting is tuned after an outcome. The smoke test on dataset seed 7 checks for defects only.

**Limits.** The same development datasets, now used for a third round. The arm was designed after seeing study v3.

**Verification before the run.** `test_learner.py`, 43 tests, passing on CPU and on the GPU: the new arm's width and actor objective against independent NumPy versions; exactly zero width and zero command gradient at supported commands, a live gradient inside a gap; no gradient to any frozen quantity; the earlier arms still reproduce the regression file. The evaluator, with this amendment's changes, re-derives every gate value of study v3 with a difference of exactly zero.

## Amendment 5: is the calibrated width's knowledge of the state used? Written before any run of the new arm

Written November 9 2026, after the results of study v4 were known (section 11 of [RESULTS_METHOD_CHANGES.md](RESULTS_METHOD_CHANGES.md)) and before any run of the arm below, the smoke test included. David approved this test the same evening ("Yes, run it").

**Why.** Study v4 showed that taking the calibrated width out of change 1 costs protection and value. It could not say why. The calibrated width could be contributing what it knows about each state, or only a general pull toward decisive commands, which needs no knowledge of the state. An unregistered check pointed to the second. This amendment tests it properly.

**The new arm: change 1 with the calibrated width blindfolded.**

- Change 1 uses `Ws = W(s, a) + d(s, a) * max(Wmax - W(s, a), 0)`.
- The new arm uses `Wb = W(s~, a) + d(s, a) * max(Wmax - W(s~, a), 0)`.

s is the true state of the row and a the policy's command there. s~ is the state of another row of the same minibatch, chosen by the same random reordering the shuffled controls use. So the calibrated width W still sees the true command but a wrong state. The support score d, its floor and the cap Wmax are read at the true state, exactly as in change 1. The penalty is 3 x Wb in the actor.

**What the comparison isolates.** The new arm and change 1 share the snapshot, the minibatches, the noise, the calibrated width function, the support score, the floor and the cap. They differ only in which state the calibrated width is read at.

**What is trained (study v5).** The same validated settings, datasets and seeds. Baselines again (they must be bit-identical to studies v3 and v4), change 1 and its shuffled control again (they must reproduce exactly; a check of the code, not a second sample), and the new arm. 108 warm-ups and 324 further policies. Shuffling everything in the new arm gives change 1's own shuffled control, so no extra control is trained.

### Decision rule

Honeypot, targeted arm, the nine runs. For reference: change 1 had H = 0 in 9 of 9 runs and an average value of 5.95; the support penalty alone had H = 2.56% and 5.10.

| Question | "The state knowledge is not used" if | "It is used" if |
|---|---|---|
| Protection | The new arm has H = 0 in at least 8 of 9 runs | The new arm has H above 0 in at least 3 of 9 runs, in at least 2 of the 3 datasets |
| Value, averaged over all start cells | The new arm reaches 5.85 or more | The new arm is below 5.50 |

Anything between the two columns is undecided for that question. The marks are the ones Amendment 4 used.

**Kill rule.** If both rows land in the first column: on this testbed the calibrated width's knowledge of the state plays no part in what change 1 achieves. I will say that BCA's calibration contributes a state-blind pull toward decisive commands and nothing else that these studies can measure, and that the method that works here is a support penalty plus that pull.

**If either row lands in the second column**, the calibrated width's knowledge of the state contributes. That would be the first evidence in these studies that state-specific calibrated uncertainty improves behavior. It would still be a development result, to be frozen and confirmed on untouched datasets before it is claimed.

Unlike Amendment 4, this test can come out either way.

**Also reported, without a pass mark:** G2 to G5 for the new arm (against change 1's shuffled control and its own sharpened baseline); the Bottleneck's X and values; stalls; and the number of start cells made worse per run in each data arm, beside change 1's. The last matters because the one earlier place where the width's knowledge of the state visibly changed behavior was extra harm with complete data in study v1. The blindfolded arm could therefore do less harm than change 1.

**Prediction, written before the run.** The new arm matches change 1: H = 0 in 8 or 9 of 9 runs and a value of 5.85 or more, so both rows land in the first column. My reasons are that the shuffled-width control matched BCA in studies v1 to v3, and that scaling commands after training recovered most of the gap in study v4.

**Nothing else changes.** Environments, datasets, gates, pass marks, penalty strength 3, the support score and its settings, the schedule. No setting is tuned after an outcome. The smoke test on dataset seed 7 checks for defects only.

**Limits.** The same development datasets, now used for a fourth round. The arm was designed after seeing study v4.

**Verification before the run.** `test_learner.py`, 45 tests, passing on CPU and on the GPU. New for this arm: the width with its two parts read at different states, and the actor objective built on it, against independent NumPy versions; and two identities of the full update. With no command unsupported the new arm is exactly the original shuffled-width control. With the calibrated width at zero it is exactly the support-penalty-only arm. The evaluator, with this amendment's changes, re-derives every gate and record value of study v4 with a difference of exactly zero.

## Amendment 6: confirmation on untouched datasets. Written before any confirmation dataset was generated

Written November 10 2026. David approved the confirmation ("Yes, confirm it") and asked that Q values be affected directly as well.

**What is confirmed.** Two versions of BCA with a support-aware width. Both were built and tested in development. Nothing about either is changed for this run.

| | Version A: best in development | Version B: also changes Q values |
|---|---|---|
| Arm name | `support_blind_width` (Amendment 5) | `full` (Amendment 3) |
| In the actor's objective | 3 x the support-aware width | 3 x the support-aware width above its usual level |
| In the critic's target (the Q values) | nothing | the same excess, subtracted at the next state |
| Development, Honeypot hazard endings | 0.00% | 1.09% |
| Development, Honeypot gates | passed G2, G3, G4, G5 | passed G2, G3, G4; failed G5 |

Each version is trained beside its own shuffled control (`support_shuffled`, `full_shuffled`) and judged against it, against its own sharpened baseline, and against the plain baseline.

**Data.** Fresh datasets on seeds 7101, 7102, 7103, for the Honeypot and for the Bottleneck revision 2 with one-hot input at exit values 4.5, 3.5 and 2.5. Training seeds 30, 31, 32. None of these seeds has been used for anything. 108 baselines and 432 further policies.

**Frozen.** The environments, the data recipe, the learner, every setting of Amendments 3 to 5, the gates and every pass mark. The source file hashes are recorded in the run's START receipts.

### Claims, fixed before the data

Each is judged separately for version A and for version B, with the gates exactly as written in section 5 and Amendment 3 (a gate passes with at least 2 of 3 datasets).

| Claim | Confirmed if | If not |
|---|---|---|
| **C0** The environments are still valid on fresh data | G0 and G1 pass in the setting | **No claim is made for that setting** |
| **C1** It protects in the Honeypot without wider harm | The version passes G2, G3, G4 and G5 | **The version is not confirmed.** I report which gate failed |
| **C2** It is cautious about the unseen bridge | X, the chance of crossing, is lower for the version than for the baseline in at least 2 of 3 datasets, at each of the three exit values | The Bottleneck result is reported as not confirmed |

G2 to G5 are also reported for the Bottleneck. Its G3 is known from development to fail on ties at zero, so it is not part of C2.

### Rules

- Run once. No pass mark, setting or gate is changed after an outcome is seen.
- Every run is kept and reported, including any whose calibration fails.
- A smoke test on dataset seed 7 checks only that this study's setup runs.
- These datasets are spent by this run. They cannot confirm anything else later.
- If any dataset of a seed fails the acceptance checks, that seed is dropped whole and the next unused seed (7104, then 7105) takes its place. The checks alone decide this, before any training.

## Amendment 7: second fresh-data block, Version A against Version B. Written before any of its datasets was generated

Written November 11 2026, after the first confirmation (`results/confirm_v1`) was known. David asked to "proceed and keep testing version A vs version B". The first block's seeds are spent, so this is a second block on seeds never used. It repeats the first block's claims and adds a direct comparison of the two versions, which the first block did not register.

**What is tested.** The same two versions, unchanged: Version A (`support_blind_width`, changes the actor only) and Version B (`full`, also subtracts the excess width in the critic's target, so it changes Q values). Each is trained beside its own shuffled control and the plain TD3+BC baseline.

**Data.** Fresh datasets on seeds 7201, 7202, 7203 (labels `confirm2_h` and `confirm2_b`), for the Honeypot and for the Bottleneck revision 2 with one-hot input at exit values 4.5, 3.5 and 2.5. Training seeds 40, 41, 42. 108 baselines and 432 further policies. Study `v7`, results label `confirm_v2`.

**Frozen.** Everything in Amendment 6: the environments, the data recipe, the learner, every setting, the gates and every pass mark. Only the seeds differ.

### Part 1: each version against the baseline

C0, C1 and C2 exactly as written in Amendment 6. Known before this run: C2 failed in the first block because the baseline already crossed 0% of the time in most datasets, so nothing could be lower. C2 is kept as written and may fail the same way.

### Part 2: Version A against Version B

Terms. *Targeted arm*: the data with the one move withheld. *Dataset mean*: the mean over the three training seeds of one dataset.

Each readout compares the two versions' dataset means in the targeted arm. A version **wins** a readout if it is strictly better in at least 2 of 3 datasets. A tie in a dataset counts for neither version. If neither reaches 2, the readout is **no difference shown**.

| Readout | Setting | Better means |
|---|---|---|
| **H1** Protection | Honeypot | fewer episodes ending in hazard 34 (H) |
| **H2** Side harm | Honeypot | fewer start cells made worse than the baseline (the count G5 uses) |
| **H3** Return | Honeypot | higher average true value of an episode over all start cells |
| **B1** Caution | Bottleneck, each exit value | fewer crossings of the unseen bridge (X) |
| **B2** Cost of caution | Bottleneck, each exit value | higher average true value of an episode over all start cells |

B1 and B2 are judged at each exit value. A version wins B1 or B2 overall if it wins at 2 or more of the 3 exit values and loses at none.

### Decision rule, fixed before the data

- **Version A.** It keeps the word "confirmed" only if it passes C1 again. If it fails, it is reported as "confirmed once, not replicated".
- **Version B.** If it passes C1 here, it is reported as "passed in 1 of 2 blocks", not as confirmed. If it fails, it is reported as not confirmed in either block.
- **Honeypot, head to head.** A version is the better one if it wins at least 2 of H1, H2 and H3 and loses none. Otherwise: no clear winner.
- **Bottleneck, head to head.** A version is the better one if it wins at least 1 of B1 and B2 overall and loses neither. Otherwise: no clear winner.
- **No pooling to rescue a claim.** The two blocks together (6 datasets per setting) are shown as a description only. The same Part 2 table for the first block is also a description, because it was not registered before that block ran.

### Rules

- Run once. No pass mark, setting or gate is changed after an outcome is seen.
- Every run is kept and reported, including any whose calibration fails.
- A smoke test on dataset seed 7 checks only that this study's setup runs.
- These datasets are spent by this run.
- If any dataset of a seed fails the acceptance checks, that seed is dropped whole and the next unused seed (7204, then 7205) takes its place. The checks alone decide this, before any training.
- `compare_versions.py` computes Part 2 from the saved records. It is written before this block's data exist and tried only on the first block.

## Amendment 8: Version A on nine fresh datasets. Written before any of its datasets was generated

Written November 11 2026, after the second block (`results/confirm_v2`) was known. David asked what Version A needs to work, and approved this run ("Yes, run it": Version A on 9 fresh datasets, with the pass mark fixed first).

**Why.** In the second block Version A cut hazard endings from 3.50% to 0.18% but failed G3. The baseline ends in the hazard only about 3.5% of the time, so its controls often land at zero too. Three datasets at a time cannot tell Version A apart from them reliably.

**What I told David, and a correction.** I said more datasets would settle it. Under G3 as written that is not assured. In the first two blocks the sharpened baseline reached exactly 0% in 3 of 6 datasets, and a tie fails G3 however many datasets there are. Claim R2 below is added for that reason. It is a second claim, not a replacement for the gates.

**What is tested.** Version A only (`support_blind_width`), unchanged, beside its own shuffled control (`support_shuffled`), its sharpened baseline and the plain TD3+BC baseline. Version B is not run.

**Data.** The Honeypot only. Fresh datasets on seeds 7301 to 7309 (label `confirm3_h`), training seeds 50, 51, 52. 81 baselines and 162 further policies. Study `v8`, results label `confirm_v3`. The Bottleneck is not rerun; its two blocks stand as reported.

**Frozen.** The environment, the data recipe, the learner, every setting and every gate's definition, as in Amendments 6 and 7. A "dataset" is still the mean over its three training seeds. Only the seeds and the number of datasets differ.

### Claims, fixed before the data

| Claim | Confirmed if | If not |
|---|---|---|
| **C0** The environment is valid on these data | G0 as written (every baseline run with complete data is competent, 27 of 27) and G1 in at least 6 of 9 datasets | **No claim is made** |
| **R1** The gates as written, on nine datasets | G2, G3, G4 and G5 each pass in at least 6 of 9 datasets. Each gate's per-dataset test is unchanged; 6 of 9 is the same two-thirds share as 2 of 3 | Version A is **not replicated**. I report which gate failed |
| **R2** Better than both controls on average | For each control (shuffled control, sharpened baseline) take the nine paired differences in hazard endings H, control minus Version A, one per dataset. The mean difference must be positive, and an exact one-sided sign-flip test over the nine datasets (all 512 sign patterns) must give p of 0.05 or less. This must hold for both controls | No average advantage over the controls is shown |

Terms. *Sign-flip test*: if Version A were no better than a control, each dataset's difference would be as likely to be negative as positive. The test counts how many of the 512 ways of flipping the nine signs give a mean at least as large as the one observed; p is that count divided by 512. A difference of exactly zero is unchanged by a flip, so ties add nothing either way.

### Decision rule, fixed before the data

- **R1 passes.** Version A is reported as replicated in the Honeypot (confirmed in two of three fresh-data runs).
- **R1 fails and R2 passes.** Reported in exactly these words: Version A has fewer hazard endings than both controls on average, and the per-dataset gates did not pass. It is not called confirmed or replicated.
- **R1 fails and R2 fails.** On this testbed Version A's protection is not shown to be more than a more decisive baseline. I stop claiming it for the Honeypot.
- R2 cannot rescue R1, and no result from the earlier blocks is pooled in to pass either claim.

### Rules

- Run once. No pass mark, setting or gate is changed after an outcome is seen.
- Every run is kept and reported, including any whose calibration fails.
- A smoke test on dataset seed 7 checks only that this study's setup runs.
- These datasets are spent by this run.
- If any dataset of a seed fails the acceptance checks, that seed is dropped whole and the next unused seed (7310, then 7311) takes its place. The checks alone decide this, before any training.
- `confirm_version_a.py` computes C0, R1 and R2 from the saved records. It is written before these data exist and tried only on the first two blocks. The pass or fail printed by `evaluate_study.py` uses the three-dataset mark and is not this study's verdict.

Limits known before the run: R2 has little power if most datasets tie at zero; three training seeds per dataset, as before; one-hot cells with uniform coverage; the Honeypot only.

## Amendment 9: baseline competence checked per dataset. Written before any of its datasets was generated

Written November 11 2026, after the run of Amendment 8 stopped at C0 (`results/confirm_v3`): 25 of 27 baselines were competent, both failures were in dataset 7306, and no Version A policy was trained. David approved a rerun on fresh seeds with baseline competence checked per dataset first ("Yes, run it").

**The one change.** G0 asks that the baseline be competent before any method is judged. Until now one run that was not competent stopped the whole setting. From this run on G0 is applied to each dataset. A dataset is **valid** if all three of its baseline runs with complete data are competent, by G0's marks exactly as written (section 5 and Amendment 1). A dataset that is not valid is dropped whole, the same way a dataset that fails its acceptance checks is dropped. This is decided from the baselines alone, before any Version A policy is trained.

**What does not change.** No mark of G0. Every dataset that is used still has every baseline run competent. The check never sees a Version A result. What changes is only that one dataset no longer discards the rest.

**Why the old form was wrong for this size.** "Every baseline run" is far harder to meet with 27 runs than with 9. Across the three fresh-data runs 43 of 45 Honeypot baselines were competent. I should have raised this before Amendment 8's data existed.

**Procedure.**

1. Twelve candidate datasets on seeds 7401 to 7412 (label `confirm4_h`), training seeds 60, 61, 62. None used before.
2. All 108 baselines are trained first, and G0 is judged for each dataset.
3. The study's datasets are the first nine valid seeds in increasing order. Valid seeds beyond the ninth are not used.
4. **If fewer than nine are valid, the run stops and no claim is made.** There is no further replacement.
5. Version A and its shuffled control are trained on the nine: 162 policies. Study `v9`, results label `confirm_v4`.

**Everything else is Amendment 8.** Version A only, the Honeypot only, the same learner, settings and gates.

| Claim | Confirmed if | If not |
|---|---|---|
| **C0** The environment is valid on these data | At least nine of the twelve candidates are valid, and G1 passes in at least 6 of the nine used | **No claim is made** |
| **R1**, **R2** | Exactly as in Amendment 8, judged on the nine datasets | As in Amendment 8 |

The decision rule is Amendment 8's, unchanged.

**Also reported.** Which candidates were dropped and their baseline results. Version A is not trained on a dropped dataset, so no result of Version A exists for it.

**Code, changed before the data.** `run_study.py` chooses the datasets by this rule at the start of stage 2 and writes the choice to `selected_datasets_honeypot.json`. `evaluate_study.py` judges G1 to G5 on the datasets that have method policies and writes the dropped datasets' baseline records to `records_dropped.json`. `confirm_version_a.py` reads the choice for C0. For every earlier study the first two behave exactly as before: re-evaluating `confirm_v2` reproduces its saved files byte for byte.

**Rules.** As in Amendment 8: run once; no pass mark, setting or gate is changed after an outcome is seen; every run is kept and reported; the smoke test on dataset seed 7 checks only that the setup runs; these datasets are spent by this run. If a candidate dataset fails its acceptance checks, that seed is dropped whole before training and counts as not valid.

Limits known before the run: the result describes datasets on which the baseline is competent, which is what G0 always required; R2 has little power if most datasets tie at zero; three training seeds per dataset; one-hot cells with uniform coverage; the Honeypot only.

## Amendment 10: Version A on nine fresh datasets in both environments. Written before any of its datasets was generated

Written November 12 2026, after the audit of Version A (`VERSION_A_AUDIT.md`). David asked to keep running experiments on the GridWorld and chose "honeypot and bottleneck".

**What is tested.** Version A only (`support_blind_width`), unchanged, beside its shuffled control, its sharpened baseline and the plain TD3+BC baseline. Nothing new is designed: this is the procedure of Amendment 9 applied to both environments. The Bottleneck has not yet been run at this sample size.

**Data.** Twelve candidate datasets per environment on seeds 7501 to 7512 (labels `confirm5_h` and `confirm5_b`), training seeds 70, 71, 72. The Honeypot, and the Bottleneck revision 2 with one-hot input at exit values 4.5, 3.5 and 2.5. Study `v10`, results label `confirm_v5`.

**Procedure.** As in Amendment 9, separately for each of the four settings: all baselines first; a dataset is valid in a setting if its three baselines with complete data are competent there (G0 as written, with Amendment 1 for the Bottleneck's B); the first nine valid seeds in increasing order are used; with fewer than nine valid, that setting stops and no claim is made for it. Then Version A and its shuffled control are trained on the datasets used.

**Frozen.** The environments, the data recipe, the learner, every setting and every gate's definition.

### Claims for the Honeypot, fixed before the data

C0, R1 and R2 exactly as in Amendment 9.

### Claims for the Bottleneck, fixed before the data

In the Bottleneck the withheld move is the best one, so avoiding it is caution, not help. X is the chance that an episode from B crosses the unseen bridge. Each claim is judged at each of the three exit values and holds only if it holds at all three.

| Claim | Confirmed if | If not |
|---|---|---|
| **C0** The environment is valid on these data | At least nine of the twelve candidates are valid in the setting, and G1 passes in at least 6 of the nine used | **No claim is made for that setting** |
| **U1** Cautious about the unseen bridge | The mean over the nine datasets of (baseline's X minus Version A's X) is positive, and an exact one-sided sign-flip test gives p of 0.05 or less | Caution is not shown |
| **U2** No wider harm | G5 passes in at least 6 of the nine datasets | Reported as harm |
| **U3** Brings the bridge back | The mean of (Version A's X minus the baseline's X) is positive, with p of 0.05 or less by the same test | **Version A does not relieve the underestimation** |

U1 and U3 cannot both hold. U3 is what relieving underestimation would mean.

**Also reported, without a pass mark.** How many datasets have X strictly lower than the baseline's (the old form of the caution claim, which failed twice on ties at zero); G2, G3 and G4; the return from all start cells and from B; bad endings; and X with the bridge in the data.

### Predictions, written before the run

- **Honeypot.** R1 and R2 pass if the baseline ends in the hazard about 5% of the time or more. If it is nearer 3.5%, G3 may fail on ties at zero, as in the second block.
- **Bottleneck.** Version A crosses 0% of the time in every run. So U1 passes wherever the baseline still crosses in several datasets, and U3 fails. U2 may fail at one or more exit values, because some Version A runs stalled there before.

### Decision rule, fixed before the data

- **Honeypot.** R1 passes: reported as passing again, three of four fresh-data runs. R1 fails: reported as failing in this run, with the gate that failed. The earlier runs are not pooled in either way.
- **Bottleneck.** U3 holds: Version A relieves the underestimation (not expected). U3 fails and U1 holds: Version A is cautious about an unseen move whether the move is bad or good. U2 fails at any exit value: the caution is reported with its harm.

### Rules

- Run once. No pass mark, setting or gate is changed after an outcome is seen.
- Every run is kept and reported, including any whose calibration fails.
- A smoke test on dataset seed 7 checks only that this study's setup runs.
- These datasets are spent by this run.
- If a candidate dataset fails its acceptance checks, that seed is dropped whole before training and counts as not valid.
- `confirm_both.py` computes every claim from the saved records. It is written before these data exist and tried only on runs whose outcomes are already known.

Limits known before the run: three training seeds per dataset; one-hot cells with uniform coverage; a move that is entirely absent from the data; at exit 4.5 the critic's underestimate was small in the second block, so that setting may be a weak underestimation test again.

## Amendment 11: thin data, and a baseline-only check of it. Written before any thin dataset was generated

Written November 12 2026, after the both-environments run (`results/confirm_v5`). David asked how to make BCA address both overestimation and underestimation, and approved building a two-sided BCA and a thin-data test ("Yes, build it"). **This is development, not confirmation.**

**Why thin data.** Every test so far withholds a move entirely. A move with no rows gives calibration nothing to fit, and nothing in the data can tell a good unseen move from a bad one. Thin data puts the move back a few times, so there is something to learn its direction from.

**The thin arm.** `collect_thin.py`, without changing `collect.py`. A thin dataset is the targeted dataset of the same task and seed, except that at each decision cell the first k uniformly random commands that could execute the withheld move are kept instead of redrawn. The demonstrator's rows are unchanged. No outcome decides what is kept. Dose 0 is the targeted dataset exactly, and doses are nested. Doses: 0, 2, 8 and 32 kept commands per decision cell. With complete data about 108 such commands are present per decision cell.

**Step 1, registered here: baselines only.** No BCA policy is trained in this step. It asks one question: at which doses is the plain baseline still wrong?

- Tasks: the Honeypot, and the Bottleneck revision 2 with one-hot input at exits 3.5 and 2.5. Exit 4.5 is left out because the critic did not underestimate there in the last run.
- Data arms: complete, and thin at doses 0, 2, 8, 32. Development seeds 5301, 5302, 5303. Training seeds 20, 21, 22. 135 baselines. Results label `thin_pilot`.
- Reference and audit banks as before: 1,280 fresh rows drawn with the same dose.

**A dose has room for a method** if the baseline, on the mean of its nine runs, is still wrong there in both ways:

| Environment | Behavior | Critic |
|---|---|---|
| Honeypot | hazard endings H at least twice the complete-data value, and at least 1 point above it | error at the withheld move above +2 |
| Bottleneck, each exit value | crossings X below half the complete-data value | error at the withheld move below -0.5 |

**Stop rule.** If no dose above 0 has room in an environment, thin data alone fixes the baseline there, and no method is trained on thin data in that environment. I report that as the answer.

**Step 2, not yet registered.** The two-sided BCA arm and its controls are specified in a further amendment, written after this step's results and before any method policy is trained. One control is already fixed: a baseline that is given the same rows the two-sided arm calibrates on. Without it, a gain could simply be the effect of more data.

**Verification before the run.** `test_thin.py`, 10 tests: dose 0 equals the targeted arm; exactly the dose is kept at every decision cell; kept rows are never the demonstrator's; nothing changes outside the decision cells; doses are nested; outcomes follow each row's own draw; generation is repeatable.

Limits known before the run: development seeds; three datasets; the kept commands are random ones, so the move is actually executed in only about a third of them; one-hot cells with uniform coverage.

## Amendment 12: a calibrated permission to leave the demonstrator's move. Written before any policy was trained with it

Written November 12 2026, after step 1 of Amendment 11 (`results/thin_pilot`). **Development, not confirmation.**

**What step 1 showed.** A few kept commands fix the Honeypot's overestimation for the plain baseline, so by the stop rule no method is trained on thin Honeypot data. In the Bottleneck the critic already rates the crossing above the low exit in most runs at every dose above 0, and the policy still takes the low exit, because TD3+BC pulls it toward the demonstrator's recorded move. Correcting the critic would not change that. David approved the following instead ("Yes, build it").

**The new arm, `permit`: Version A plus one rule.** Stop copying a recorded command when a move inside the data's support is better even after subtracting BCA's width.

1. **A proposal network** is trained beside the actor. It minimizes `-c * mean(Q1(s, mu(s))) + c * 3 * mean(W(s, mu(s)))`, with no cloning term. W is the support-aware width read at the true state: the calibrated width plus the raise where the support score is below its floor. So mu(s) is the best move the critic knows inside the data's support. It is used only in the rule below.
2. **The permission.** For a recorded row (s, a), the lower bound of the proposal is `Q1(s, mu(s)) - 1 * W(s, mu(s))`. If the critic's score of the recorded command, `Q1(s, a)`, is below that lower bound, the row is left out of the cloning term of the actor's objective. Otherwise it is cloned as before. The permission carries no gradient and is never given before a reference is ready.
3. **Everything else is Version A**: the same width penalty in the actor, the same critic, the same schedule.

The margin is 1 times the width, the calibrated interval's own lower bound. It is fixed here and not tuned. Where a move has no data the raise makes W very large, the lower bound very low, and the permission is never given.

**Controls.**

| Arm | What it is | What it isolates |
|---|---|---|
| `support_blind_width` (Version A) | no permission | the effect of the permission |
| `permit_plain` | the same permission, but the margin uses only the support raise and leaves the calibrated width out | whether BCA's calibrated width does any work in the permission |

Amendment 11 fixed a control in advance: a baseline given the same rows the new arm calibrates on. It is not needed. The new arm uses no rows beyond the reference bank that every arm already has.

**What is trained.** On the thin development datasets of Amendment 11 (seeds 5301 to 5303, training seeds 20 to 22). The learner's state gains a network, so the baselines are trained again under the new code, label `thin_dev2`, and must reproduce `thin_pilot`'s baseline policies. Then Version A, `permit` and `permit_plain`:

- Bottleneck at exits 3.5 and 2.5: doses 0, 2, 8, 32 and complete data.
- Honeypot: dose 0 and complete data only. By the stop rule of Amendment 11 nothing is trained on thin Honeypot data.

**Marks, fixed before the run.** A "dataset" is the mean over its three training seeds. Means are over the nine runs.

| Mark | Passes if |
|---|---|
| **P1** Brings the bridge back where the data support it | At dose 32, `permit` crosses more than Version A and more than the baseline in at least 2 of 3 datasets, at each exit value |
| **P2** Stays cautious without data | At dose 0, `permit`'s mean crossings are no more than 1 point above Version A's, at each exit value |
| **P3** Keeps the protection against overestimation | Honeypot, dose 0: `permit`'s mean hazard endings are no more than 0.5 points above Version A's, and at most half the baseline's |
| **P4** No wider harm | In every setting trained: `permit`'s mean bad endings over all start cells are no more than 0.5 points above Version A's, and its mean return over all start cells is no more than 0.1 below Version A's |
| **P5** The calibrated width earns its place | `permit` passes P3 and P4 where `permit_plain` fails one of them. Otherwise the calibrated width is reported as not needed for the permission |

**Kill rules.** If P1 fails at both exit values, the permission does not bring the bridge back, and I report that. If P3 fails, the arm is rejected whatever P1 says: it would have traded the overestimation result away.

**Predictions, written before the run.** P1 passes at exit 2.5, where the critic's lead for the crossing (about 2.0) is well above the width (about 0.5). At exit 3.5 the lead is about 1.0; I expect P1 to pass there too, but less clearly. At doses 2 and 8 the support score still flags the crossing, so `permit` crosses little. P2, P3 and P4 pass. P5: I expect `permit_plain` to do about as well as `permit`, which would mean the calibrated width is not needed.

**Verification before the run.** New tests in `test_learner.py`: the permission and the proposal objective against independent NumPy versions; no permission before a reference is ready; a row without permission is cloned exactly as before; `permit` is Version A exactly when no permission is given; `permit` and `permit_plain` are the same when the calibrated width is zero; every earlier arm still reproduces its pinned outputs.

Limits known before the run: development seeds, three datasets, large run-to-run variation in the Bottleneck; the design was made after seeing the baselines on these same datasets; one-hot cells with uniform coverage.

## 7 Limits known in advance

- Three datasets with three training seeds each. Seeds within a dataset are not independent replications of the dataset.
- Development seeds. A favorable result is a reason to freeze the design and generate untouched confirmation datasets, nothing more.
- The host is a re-implementation. It answers questions about this testbed; it does not re-verify archived numbers.
- The thresholds (1.25 in G2, 1 point in G5, 90% and value 3 in G0) are judgment calls drafted on November 9 and not amended by David before this run.
- The width is calibrated for a one-step target on data drawn like the training data. It carries no guarantee at a move the data never contain.
