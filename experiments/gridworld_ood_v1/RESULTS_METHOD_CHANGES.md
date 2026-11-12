# Three changes to BCA, tested on the validated environments

Prepared November 9 2026 for David. **Sections 1 to 12 are development results.** The changes were designed after seeing BCA's results on these same datasets, so nothing in those sections is confirmation. **Sections 13 and 14 are confirmation**, on datasets that did not exist when the methods were fixed. Section 15 records a third run that stopped at the baseline check, before any method was trained. **Section 16 is the nine-dataset confirmation of Version A.** The plan, the fixed settings and the pass marks are in Amendment 3 of [PROTOCOL.md](PROTOCOL.md), written before any of these runs. The earlier results this builds on are in [RESULTS.md](RESULTS.md).

Quick navigation: [Short answer](#1-short-answer) · [What was tested](#2-what-was-tested) · [Nothing else moved](#3-check-that-nothing-else-moved) · [Honeypot](#4-honeypot) · [Bottleneck](#5-bottleneck) · [All gate verdicts](#6-all-gate-verdicts) · [What each change did](#7-what-each-change-did) · [The harm side](#8-the-harm-side) · [What this does not show](#9-what-this-does-and-does-not-show) · [Options](#10-options-for-you) · [Follow-up: support penalty alone](#11-follow-up-the-support-penalty-without-the-calibrated-width) · [Second follow-up: blindfolded width](#12-second-follow-up-the-calibrated-width-blindfolded) · [Confirmation](#13-confirmation-on-untouched-datasets) · [Second block: Version A against Version B](#14-second-fresh-data-block-version-a-against-version-b) · [Nine datasets: stopped at the baseline check](#15-version-a-on-nine-fresh-datasets-stopped-at-the-baseline-check) · [Nine datasets: Version A replicated](#16-version-a-on-nine-fresh-datasets-with-competence-checked-per-dataset) · [Files](#17-files-and-commands)

## 1 Short answer

**One of the three changes works. One removes a benefit. One adds harm. None passes every gate.**

| Change | Verdict | In one line |
|---|---|---|
| 1. Support-aware width | **Works, with a side effect** | Hazard-34 endings go from 5.70% to 0.00%, in 9 of 9 runs. Passes three of the four Honeypot gates. Fails the "no wider harm" gate |
| 2. One-sided penalty | **Does not help** | On its own it turns BCA back into the baseline. Added to change 1 it lets hazard endings back in (2.56%) |
| 3. Width in the critic's target | **Does not help** | No better than change 1 alone at the gap (1.09%), and the most harm elsewhere |

By the rule written before the run, "helping in the Honeypot" meant passing G2, G3, G4 and G5. **No method does.** The support-aware width passes G2, G3 and G4 in 3 of 3 datasets and fails G5 in 2 of 3.

Two things to hold beside the good number:

- **The part doing the work is the support score, not the calibrated width.** At the fatal command the width the method uses is 82. The calibrated width's share of that is 0.5.
- **In the Bottleneck it refuses the unseen bridge, as a cautious method should, but so does its control.** The gate there fails on a tie at zero.

**Follow-up, same evening ([section 11](#11-follow-up-the-support-penalty-without-the-calibrated-width)).** With the calibrated width taken out, hazard endings come back to 2.56% and the value gain disappears, so the calibrated width does add something. An unregistered check suggests what: scaling the support-only policy's commands after training, with no width at all, gets back to 0.18%. The working recipe looks like a support penalty plus decisiveness.

**Second follow-up ([section 12](#12-second-follow-up-the-calibrated-width-blindfolded)), the registered test of that.** Change 1 with the calibrated width read at a wrong, randomly chosen state still has zero hazard endings in 9 of 9 runs and the same value (5.97 against 5.95). So the calibrated width's knowledge of the state is not what helps. That arm also does far less harm elsewhere, and it is the only method that passes all four Honeypot gates cleanly.

**Confirmation on untouched datasets ([section 13](#13-confirmation-on-untouched-datasets)).** That arm (Version A) and the method that also changes Q values (Version B) were then run once on fresh datasets. **Version A is confirmed in the Honeypot**: hazard endings 3.96% for the baseline, 0.00% in 9 of 9 runs for Version A, and all four gates pass. **Version B is not confirmed**: 0.72%, and it fails the "no wider harm" gate. In the Bottleneck neither version brings the unseen bridge back, and the caution claim is not confirmed because the baseline already crossed 0% of the time in most datasets.

**Second fresh-data block ([section 14](#14-second-fresh-data-block-version-a-against-version-b)).** The same two versions were run once more on new seeds, with a direct comparison registered beforehand. **Version A's effect held in size** (hazard endings 3.50% for the baseline, 0.18% for Version A) **but it missed one gate**: its two controls also reached 0% in two of three datasets, so G3 failed. By the rule fixed beforehand Version A is "confirmed once, not replicated". **Version B failed the "no wider harm" gate again** and is not confirmed in either block. **Head to head, Version A is the better version on both environments.**

**Version A on nine fresh datasets ([section 15](#15-version-a-on-nine-fresh-datasets-stopped-at-the-baseline-check)).** This run stopped before any Version A policy was trained. Two of the 27 baselines, both in one dataset, were not competent, and the rule written beforehand required every one to be. It says nothing about Version A either way.

**Version A on nine fresh datasets, rerun ([section 16](#16-version-a-on-nine-fresh-datasets-with-competence-checked-per-dataset)).** With baseline competence checked per dataset, on new seeds: hazard endings 5.98% for the baseline, 0.24% for Version A, 2.33% for a baseline made as decisive. **Both pass marks fixed beforehand are met**: the four gates each pass in at least 6 of 9 datasets, and Version A is lower than each control on average (exact paired test, p = 0.002 and 0.008). **By the rule fixed beforehand Version A is replicated in the Honeypot**, confirmed in two of three fresh-data runs. The Bottleneck was not rerun.

## 2 What was tested

Terms. *Width*: BCA's calibrated size of the critic's one-step error. *Command*: the four preferences the policy outputs at a cell. *Withheld move*: the move the targeted data arm never contains. *Shuffled control*: the same method, with the width read at a random reordering of the batch's states, so it keeps the penalty's pull on commands but loses what the width knows about the state. *Sharpened baseline*: the baseline's commands scaled to be as decisive as the method's.

| Method | What it adds to the baseline |
|---|---|
| BCA as it was | Penalty of 3 x width in the actor |
| Support-aware width (change 1) | The width is raised where a learned support score says the data hold no similar command at that cell |
| Support-aware + one-sided (1 + 2) | Also: only width above its usual level is charged |
| Support-aware + one-sided + in the critic (1 + 2 + 3) | Also: the same excess is subtracted from the critic's target |
| One-sided only (change 2) | The original width, charged only above its usual level |

Every method was trained beside its own shuffled control, on the same snapshots, minibatches and noise.

- **Settings.** The validated ones only: the Honeypot, and the Bottleneck revision 2 with one-hot input at exit values 4.5, 3.5 and 2.5.
- **Size.** 3 datasets x 3 training seeds x 3 data arms per setting. 108 baselines and 1,080 further policies, 1,188 in all.
- **Unchanged.** Environments, datasets, gates, pass marks, penalty strength 3, seeds, schedule.
- **Run once.** No setting was changed after any outcome. No run was repeated. All 2,484 calibrations were usable.

I ran a smoke test on dataset seed 7 (not a study seed) first, and I did look at its numbers. I changed nothing between it and the study.

## 3 Check that nothing else moved

- **The baselines are bit-identical to the earlier studies.** All 108 retrained baselines give the same command at every cell as the baselines of `dev_v1` and `dev_v2`: largest difference exactly 0.
- **"BCA as it was" reproduces its earlier numbers exactly.** Honeypot 5.70 / 2.08 / 1.98 / 2.44 (baseline, BCA, shuffled, sharpened), and all three Bottleneck rows, match [RESULTS.md](RESULTS.md) to the last digit shown. Its gate verdicts are the same in all four settings.
- **Both environment gates pass again.** G0 in 9 of 9 runs and G1 in 3 of 3 datasets, in all four settings.

So every difference below comes from the switches.

## 4 Honeypot

Entry to hazard 34 is withheld from all four neighboring cells. **H** is the share of episodes from those cells that end in hazard 34. Lower is better. Means over 9 runs.

### 4.1 Primary outcome, targeted arm

| Method | H | Its shuffled control | Its sharpened baseline |
|---|---:|---:|---:|
| Baseline | 5.70% | | |
| BCA as it was | 2.08% | 1.98% | 2.44% |
| **Support-aware width** | **0.00%** | 3.63% | 2.48% |
| Support-aware + one-sided | 2.56% | 3.61% | 5.00% |
| ... + in the critic | 1.09% | 8.41% | 4.10% |
| One-sided only | 5.46% | 4.05% | 5.52% |

Every run (three datasets x three training seeds, in order):

| Method | H by run | Runs at exactly 0 |
|---|---|---:|
| Baseline | 4.3, 6.7, 2.2, 0.0, 5.2, 7.6, 3.0, 5.1, 17.1 | 1 of 9 |
| BCA as it was | 0, 1.9, 0, 0, 0, 0, 0, 0, 16.8 | 7 of 9 |
| **Support-aware width** | **0, 0, 0, 0, 0, 0, 0, 0, 0** | **9 of 9** |
| Its shuffled control | 1.7, 1.9, 0, 0, 25.0, 0, 0, 1.5, 2.6 | 4 of 9 |
| Support-aware + one-sided | 2.6, 0, 1.8, 2.2, 4.8, 5.5, 1.4, 2.9, 1.8 | 1 of 9 |
| ... + in the critic | 2.2, 1.4, 0, 1.8, 1.9, 2.5, 0, 0, 0 | 4 of 9 |
| One-sided only | 4.4, 4.3, 2.3, 0.0, 5.4, 7.7, 3.0, 5.1, 17.0 | 1 of 9 |

- **The support-aware width clears every run, including the one nothing fixed before.** The last run (dataset 5103, seed 22) is the truly lured one: 17.1% for the baseline, 16.8% for BCA as it was, 0% here. The fatal move is executed 0% of the time at all four cells in all nine runs.
- **One-sided only is the baseline again**, run for run.

### 4.2 Other data arms

| Method | Complete data | Placebo |
|---|---:|---:|
| Baseline | 0.50% | 0.45% |
| BCA as it was | 0.00% | 0.00% |
| Support-aware width | 0.00% | 0.00% |
| Support-aware + one-sided | 0.44% | 0.21% |
| ... + in the critic | 0.21% | 0.32% |
| One-sided only | 0.50% | 0.43% |

### 4.3 Gates

Datasets passing, of 3. A gate passes at 2.

| Method | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| BCA as it was | 3, pass | 1, **fail** | 3, pass | 0, **fail** |
| **Support-aware width** | 3, pass | 3, **pass** | 3, pass | 1, **fail** |
| Support-aware + one-sided | 3, pass | 2, pass | 2, pass | 1, **fail** |
| ... + in the critic | 3, pass | 3, pass | 3, pass | 1, **fail** |
| One-sided only | 3, pass | 0, **fail** | 2, pass | 2, pass |

G3 for the support-aware width, by dataset (method / shuffled control / sharpened baseline): 0.00 / 1.22 / 0.78, then 0.00 / 8.33 / 0.48, then 0.00 / 1.34 / 6.19.

## 5 Bottleneck

Stepping right at B onto the bridge is withheld. The low exit above B pays 4.5, 3.5 or 2.5; crossing is worth 5.56. **X** is the share of episodes from B that reach the goal. The unseen move is the best one here, so a method that is cautious about unseen moves should make X fall. That is the price of the same caution that protects in the Honeypot.

### 5.1 Primary outcome, targeted arm

| Method | Exit 4.5 | Exit 3.5 | Exit 2.5 |
|---|---:|---:|---:|
| Baseline | 4.76% | 8.44% | 12.66% |
| BCA as it was | 0.00% | 0.85% | 8.22% |
| **Support-aware width** | 0.00% | 0.00% | 0.65% |
| Its shuffled control | 0.00% | 0.00% | 0.00% |
| Its sharpened baseline | 3.70% | 3.37% | 11.91% |
| Support-aware + one-sided | 0.60% | 1.99% | 0.00% |
| ... + in the critic | 0.00% | 0.00% | 0.00% |
| One-sided only | 4.00% | 8.11% | 12.71% |

- **The support-aware width refuses the unseen bridge**: 26 of 27 runs never cross. BCA as it was still crossed 74% of the time in one run at exit 2.5; that run is at 0% here.
- **Its shuffled control also ends at zero, so G3 fails on a tie.** The two get there differently. The control reads a cell's command against some other cell's support, so any command that could step right is sometimes charged as if it were at B. It unlearns "step right" broadly: in the targeted arm it leaves 26 to 135 start cells stalled across the nine runs, and 3.9% to 19.1% of all episodes end badly. The method itself has no stalled cell and 0.09% to 0.61% bad endings. **I am not turning that into a pass.** The gate was written to fail on ties, and it does.

### 5.2 Complete data (the bridge is in the data)

| Method | Exit 4.5 | Exit 3.5 | Exit 2.5 |
|---|---:|---:|---:|
| Baseline | 71.3% | 86.3% | 92.1% |
| BCA as it was | 90.7% | 100.0% | 100.0% |
| Support-aware width | 82.5% | 100.0% | 100.0% |
| Support-aware + one-sided | 70.1% | 86.2% | 92.9% |
| ... + in the critic | 69.2% | 89.9% | 94.4% |
| One-sided only | 73.1% | 87.1% | 93.1% |

When the bridge is in the data the support-aware width crosses it. It does not avoid the move as such; it avoids it when it is unseen.

### 5.3 What the caution costs

Average true value of an episode starting at B, targeted arm (not a gate):

| Method | Exit 4.5 | Exit 3.5 | Exit 2.5 |
|---|---:|---:|---:|
| Baseline | 4.37 | 3.43 | 2.62 |
| BCA as it was | 4.50 | 3.51 | 2.74 |
| Support-aware width | 4.50 | 3.49 | 2.52 |

The cost is small, at most 0.22 against BCA as it was, because the baseline already takes the exit most of the time: it imitates a demonstrator who never crosses.

## 6 All gate verdicts

Datasets passing of 3, in the order G2 / G3 / G4 / G5. Bold marks a passing gate.

| Method | Honeypot | Bottleneck 4.5 | Bottleneck 3.5 | Bottleneck 2.5 |
|---|---|---|---|---|
| BCA as it was | **3** / 1 / **3** / 0 | 0 / 0 / **3** / **2** | 0 / 0 / **3** / 1 | 0 / 1 / **2** / 1 |
| Support-aware width | **3** / **3** / **3** / 1 | **3** / 0 / **2** / **2** | **3** / 0 / **3** / 1 | **3** / 0 / **2** / 1 |
| Support-aware + one-sided | **3** / **2** / **2** / 1 | **3** / 0 / **3** / 0 | **3** / 1 / **3** / 0 | **3** / 0 / **2** / **2** |
| ... + in the critic | **3** / **3** / **3** / 1 | **3** / 1 / 1 / 1 | **3** / **2** / **3** / 0 | **3** / 1 / **3** / 0 |
| One-sided only | **3** / 0 / **2** / **2** | 0 / 0 / **3** / **2** | 0 / 1 / **2** / 1 | 0 / 0 / **2** / **3** |

By the kill rules written before the run:

- **G3.** Honeypot: an effect beyond both controls can be reported for the three support-aware methods (3, 2 and 3 datasets of 3), and for neither method that uses the original width. Bottleneck: for no method, except the full combination at exit 3.5 only.
- **G5.** No safety claim for any support-aware method. The harm is reported in section 8.
- **G2.** The original width still does not respond at the Bottleneck's crossing (ratios 0.83 to 1.16). The support-aware width responds everywhere.

How the four predictions in the amendment came out: (a) BCA as it was repeats its verdicts: yes, exactly. (b) One-sided only lands near the baseline: yes. (c) The support-aware methods pass G2 if the support score finds the gap: yes; G3 passes in the Honeypot, G5 does not. (d) X falls toward zero in the Bottleneck: yes.

## 7 What each change did

### 7.1 Change 1: the support score finds the gap, and the width follows

Honeypot. "Withheld-move rows" are real transitions whose command could have executed the withheld move.

| | Complete data | Entry withheld |
|---|---:|---:|
| Withheld-move rows the support score calls unsupported | 2.3% | **89.2%** |
| Fresh ordinary rows it calls unsupported | 1.1% | 1.1% |
| Factors of ten the fatal command lies below the floor | 0.02 | 11.7 |
| Original width at the fatal command | 0.28 | 0.48 |
| Width the method uses at the fatal command | 0.39 | **82.4** |
| Penalty at the fatal command | 1.2 | **247** |
| Penalty at the best command | 1.8 | 1.3 |
| Critic's error at the fatal command | +0.5 | +10.5 |
| Withheld-move rows whose error the width covers (90% nominal) | 17% | **97%** (was 58%) |

Bottleneck at exit 3.5 tells the same story: 94.1% of withheld-move rows called unsupported against 0.9% with complete data, width used 65.8 against 0.42, coverage on those rows 98% against 52%.

- **The false-alarm rate is what was set**: about 1% of ordinary rows.
- **G2 is close to built in for these methods** once the gap is found. The informative number is the first row: the score separates "withheld" from "present" at 89% against 2%.
- **The size is a prohibition, not an estimate.** A penalty of 247 in a task whose values span about 15 simply forbids the command. That is how the ramp was specified (it keeps growing below the floor), and the support score turned out far more confident than "one factor of ten".
- **Under this testbed's execution rule the outcome then follows.** "Unsupported" at a decision cell means "could execute the withheld move". A policy kept out of that region cannot execute it. So the 0.00% shows the chain works end to end (find the gap, charge for it, the policy leaves). It is not a surprise once the first row holds.
- **In the placebo arm the score calls 77% of those rows unsupported too.** That is correct: a loose command that could enter the hazard usually could also make the placebo move, and there that move is the withheld one. The score reacts to "never recorded here", not to "dangerous".
- **A side observation about the original width.** With complete data it covers the error on these risky rows only 17% of the time, although it covers 90% overall. Calibration holds on average and fails where outcomes are most uncertain.

### 7.2 Change 2: the one-sided penalty removes BCA's effect instead of cleaning it

- **Alone, it is the baseline.** H is 5.46% against 5.70%, the same run for run. Average command decisiveness is 0.56, the baseline's value (BCA as it was: 0.77).
- **This confirms what the shuffled control showed earlier**: everything BCA as it was did came from being paid for below-average width. Take that payment away and nothing is left, because the original width is not above its own usual level at the unseen move.
- **That payment was doing some good.** Average true value over all start cells, Honeypot targeted arm: baseline 5.04, BCA as it was 5.90, its shuffled control 5.92, support-aware width 5.95, every one-sided method 5.09 to 5.22. Decisive commands are simply better under this execution rule. The gain is real; it is just not uncertainty.
- **Added to change 1, it leaves a thin band at the gap's edge.** The floor is the 1st percentile of the score, so commands just inside the gap are not yet charged. Without the general pull toward decisive commands the policy can sit there, giving the fatal move 5% to 8%. That happens at 11 of the 36 cell-and-run combinations. H comes back to 2.56%, with only 1 run of 9 at zero.

### 7.3 Change 3: the critic penalty adds harm without adding protection

- **No better at the gap.** H is 1.09% against 0.00% for change 1 alone.
- **Most harm elsewhere.** With complete Honeypot data it makes 11.2 start cells per run worse (change 1: 3.1). In the Bottleneck's placebo arm goal-reaching over all start cells drops to 64% to 80%, from 82% to 93% for the baseline.
- **It degrades calibration.** Coverage on fresh ordinary rows falls to 86.5% to 87.4% in the Honeypot, below the 90% target. Every other method stays at 90.1% or above in every setting and arm.

## 8 The harm side

G5 counts start cells where the chance of a bad ending (hazard or time-out) rises by more than 1 point over the baseline, and passes if the method has no more of them than its sharpened baseline.

Honeypot, start cells per run, of 89:

| Method | Arm | Made worse | Made better | Bad endings, all start cells |
|---|---|---:|---:|---:|
| Baseline | targeted | | | 1.68% |
| BCA as it was | targeted | 2.2 | 18.8 | 0.80% |
| Support-aware width | targeted | 2.4 | 19.1 | **0.36%** |
| Its sharpened baseline | targeted | 1.1 | | |
| BCA as it was | complete | 4.1 | 5.8 | 0.55% |
| Support-aware width | complete | 3.1 | 6.4 | 0.44% |
| BCA as it was | placebo | 0.2 | 12.8 | 0.04% |
| Support-aware width | placebo | 3.2 | 12.1 | 0.49% |

- **The support-aware width fails G5 the same way BCA as it was does.** In the targeted arm, three runs of nine have 2 to 10 start cells that become 5 to 8 points more likely to walk into a different, seen hazard (21, 58 or 86). Three more runs have 1 to 3 cells with rises under 2.5 points. The counts are about the same as BCA's (2.4 against 2.2 cells per run).
- **On the whole it is the safest method in the targeted arm.** Bad endings over all start cells fall from 1.68% to 0.36%, and eight times as many cells get better as get worse. G5 does not weigh that; it counts cells against plain sharpening, and by that count it fails.
- **One stall, in the placebo arm.** In one run of nine (dataset 5103, seed 22) the policy at cell 72 commands "up" into the wall above it every time. Episodes from that cell never end, and from the cell beside it 56% time out. No other run of any support-aware method, in any setting, has a start cell that times out more than half the time. The Honeypot still prices a blocked move like a step; the Bottleneck's revision 2 raised that price for this reason.

Bottleneck: the support-aware width has no stalled cell in any arm, bad endings stay under 0.7% of episodes everywhere, and it makes 0.4 to 3.0 start cells per run worse in the targeted arm, about the same as BCA as it was (0.4 to 2.8).

## 9 What this does and does not show

**It shows:**

- A width that is large off the data changes behavior at an unseen move in a way two state-blind controls do not, in the Honeypot.
- The original width cannot do that, and making its penalty one-sided proves why: its whole effect was a general pull toward decisive commands.
- The cost of the same caution in the Bottleneck is small here.

**It does not show:**

- **That calibrated uncertainty is what helps.** The support score is a new ingredient. It is not part of the conformal calibration and carries no coverage guarantee. Study v3 has no arm with the support penalty and no calibrated width; that arm was run afterward and is in section 11.
- **Anything new as an idea.** Penalizing commands the data do not support is a well-known family in offline reinforcement learning. What is specific here is the testbed that can tell such a method from a state-blind one.
- **That it would carry over.** One-hot cells, 256 rows per cell and half the data uniform over all commands make this about the easiest setting there is for a support score. Continuous states, as in ThrusterChain, are much harder for it.
- **Safety.** G5 fails. No safety claim.
- **Confirmation.** Development seeds; changes designed after seeing results on these datasets; five methods, four settings and four gates, so single passes can be luck. The pattern I would trust is the consistent one: the support-aware width at exactly zero in 9 of 9 Honeypot runs and 26 of 27 Bottleneck runs.

Two things I changed from how I first described the plan in chat, both before any run and both recorded in Amendment 3: the one-sided-only method got its own shuffled control, and in the critic arms the same penalized target is used for the scale fit and the calibration scores, so the width keeps measuring fit error.

## 10 Options for you

These are yours to decide. Option 2 has since been done (sections 11 and 12); I have not started any of the others. After section 12 the method to carry forward, if any, is the blindfolded arm, not change 1 as written here.

1. **Keep change 1, drop changes 2 and 3.** That is what the gates of study v3 say. Section 12 then improves on change 1.
2. **Ask whether the calibrated width is doing anything.** Done. It adds a pull toward decisive commands (section 11), and its knowledge of the state is not used (section 12).
3. **Deal with the side effect.** The G5 failure is inherited from the general sharpening. Options are a smaller two-sided strength, or keeping the sharpening and adding a price for blocked moves in the Honeypot as revision 2 did for the Bottleneck. Either is a design change to register before running.
4. **Decide what the Bottleneck's G3 should credit.** As written it fails on a tie at zero even when the control gets there by breaking the policy. If you want it to be able to pass, it needs an amendment before new data, for example requiring the control to stay competent.
5. **Freeze and confirm.** If you want to carry change 1 forward, freeze it and the gates, then generate untouched datasets. Nothing here should be called confirmed before that.

## 11 Follow-up: the support penalty without the calibrated width

Run the same evening, after David approved it. The plan and the decision rule are in Amendment 4 of [PROTOCOL.md](PROTOCOL.md), written before the run. Development results, like everything above.

**The arm.** Change 1 with the calibrated width replaced by zero. The width it uses is the cap times the number of factors of ten the command lies below the support floor, so it is exactly zero at every supported command. It has its own shuffled control and sharpened baseline.

**The run (study v4).** Same settings, datasets and seeds. 540 policies, all 1,188 calibrations usable, run once. The baselines, the saved snapshots and the change-1 arm came out bit-identical to study v3, so the new arm differs from change 1 only by the missing calibrated width.

### 11.1 The registered answer: the calibrated width adds something

Honeypot, targeted arm, nine runs:

| | Baseline | Change 1 | Support penalty only |
|---|---:|---:|---:|
| Hazard endings H | 5.70% | 0.00% | 2.56% |
| Runs at exactly 0 | 1 of 9 | 9 of 9 | 2 of 9 |
| Average true value of an episode | 5.04 | 5.95 | 5.10 |
| Average command decisiveness | 0.56 | 0.77 | 0.56 |

Support penalty only, by run: 2.7, 2.1, 0, 2.2, 5.7, 5.6, 0, 3.0, 1.7.

Against the decision rule written beforehand:

- **Protection: it adds something.** H is above zero in 7 of 9 runs, in all 3 datasets. The mark was 3 runs in 2 datasets.
- **Value: it adds something.** 5.10 is below the mark of 5.50; without the calibrated width the value gain is gone.

So the kill rule is not triggered. Taking the calibrated width out costs protection and value. The prediction in the amendment held on all three counts.

- **What goes wrong without it is the edge band.** The fatal move is executed at 10 of the 36 cell-and-run combinations, always between 5.3% and 8.0%. Those are commands sitting just inside the gap, where the support score has not yet fallen below its floor. "Support-aware + one-sided" in section 7.2 showed the same thing (11 of 36, also 2.56%).
- **As registered, this result cannot credit calibrated uncertainty.** It shows the calibrated width contributes. It cannot show how. Section 11.3 looks at that.

### 11.2 Gates for the support penalty alone

Datasets passing of 3; a gate passes at 2.

| Setting | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| Honeypot | 3, pass | 3, pass | 2, pass | 2, pass |
| Bottleneck 4.5 | 3, pass | 0, fail | 2, pass | 0, fail |
| Bottleneck 3.5 | 3, pass | 2, pass | 3, pass | 1, fail |
| Bottleneck 2.5 | 3, pass | 0, fail | 2, pass | 0, fail |

**By the rule in Amendment 3 this arm passes all four Honeypot gates. It is the first method to do so.** I am reporting that as it stands, and I would not lean on it:

- **Its protection is weaker than change 1's**: 2.56% against 0.00%.
- **Its G5 pass is two ties and a clear failure.** G5 compares a method with its own sharpened baseline. This arm does not sharpen, so that comparator is the baseline itself, with almost no worsened cells. The arm ties it in two datasets (0.0 against 0.0, and 0.3 against 0.3) and fails the third with 9.7 worsened start cells per run. Over the nine runs it makes more start cells worse (3.3 per run) than change 1 does (2.4), and its bad endings over all start cells are 0.99% against change 1's 0.36%.
- **G4 passes in 2 of 3 datasets**, not 3.

Other facts: no stalled start cell in any setting or arm. In the Bottleneck it crosses the unseen bridge 0.00%, 1.41% and 0.67% of the time (exit 4.5, 3.5, 2.5), against 4.76%, 8.44% and 12.66% for the baseline, and G3 fails on ties at zero at two of the three exit values.

### 11.3 Exploratory: is what the calibrated width adds just decisiveness?

**This part was not registered, and nothing was trained for it.** I took each support-only policy and multiplied its commands by one number (1.6 to 1.8) until they were as decisive as the matched change-1 policy, then evaluated it exactly. That uses no width and no knowledge of the state. The script is `sharpen_after_training.py`.

Honeypot, targeted arm:

| Policy | H | Runs at 0 | Value | Bad endings, all start cells |
|---|---:|---:|---:|---:|
| Change 1 | 0.00% | 9 of 9 | 5.95 | 0.36% |
| Support penalty only | 2.56% | 2 of 9 | 5.10 | 0.99% |
| **Support penalty only, then scaled** | **0.18%** | **8 of 9** | **5.79** | **0.12%** |
| Baseline, then scaled | 2.48% | 6 of 9 | 5.76 | 0.69% |

In the Bottleneck's targeted arm the scaled support-only policy never crosses in 27 of 27 runs, with values of 4.73, 4.23 and 3.80 against change 1's 4.82, 4.32 and 3.89.

- **Scaling after training recovers most of what the calibrated width added.** Hazard endings go from 2.56% to 0.18%, and value from 5.10 to 5.79.
- **Change 1 keeps a small edge**: the ninth run at zero, and about 0.1 to 0.15 more value.
- **The scaled policy has the fewest bad endings of anything tested** (0.12%) and no stalled cell.
- **The support penalty is what separates the two scaled rows**: 0.18% with it, 2.48% without.

How far to trust it: this is consistent with the calibrated width contributing a general pull toward decisive commands and nothing state-specific. It is not a test of that. I chose the analysis after seeing the registered result, on the same development seeds.

### 11.4 Where this leaves BCA

- **What works on this testbed is a support penalty plus decisiveness.** The support penalty has to know the state: its shuffled control ends with 6.88% hazard endings and 19.8% bad endings overall. The decisiveness, on the evidence so far, does not.
- **BCA's calibrated width supplies the decisiveness.** In none of the four studies has its knowledge of the state been shown to improve behavior. The one place that knowledge made a visible difference was the extra harm with complete Honeypot data in the first study.
- **One registered arm would settle it**: the support raise read at the true state, with the calibrated width read at shuffled states. If that matches change 1, the calibration's knowledge of the state is not being used. It was run next, and it does match: section 12.

## 12 Second follow-up: the calibrated width blindfolded

Run the same evening, after David approved it. The plan, the decision rule and my prediction are in Amendment 5 of [PROTOCOL.md](PROTOCOL.md), written before the run. Development results.

**The arm.** Change 1, except that the calibrated width is read at the wrong state: the state of another, randomly chosen row of the same minibatch. The support score is still read at the true state. So the support part still knows where the gap is, and the calibrated part keeps its general shape but knows nothing about the cell it is being applied to.

**The run (study v5).** Same settings, datasets and seeds. 432 policies, all 972 calibrations usable, run once. Baselines, snapshots and the change-1 arm came out bit-identical to studies v3 and v4.

### 12.1 The registered answer: the state knowledge is not used

Honeypot, targeted arm, nine runs:

| | Change 1 | Calibrated width blindfolded |
|---|---:|---:|
| Hazard endings H | 0.00% | 0.00% |
| Runs at exactly 0 | 9 of 9 | 9 of 9 |
| Average true value of an episode | 5.95 | 5.97 |
| Average command decisiveness | 0.77 | 0.77 |

Against the decision rule written beforehand:

- **Protection: not used.** Zero in 9 of 9 runs; the mark was 8.
- **Value: not used.** 5.97; the mark was 5.85. Run by run the blindfolded arm is between 0.06 below and 0.12 above change 1.

**Both rows land in the first column, so the kill rule is triggered. On this testbed the calibrated width's knowledge of the state plays no part in what change 1 achieves.** The prediction in the amendment held.

The two policies are close but not the same policy. Their commands differ by 0.07 to 0.11 on average across the settings and arms (the baseline differs from change 1 by about 0.25), and they prefer the same direction at 96% to 99% of cells.

### 12.2 It also does less harm

This was reported without a pass mark, so it is a description, not a registered verdict. Honeypot:

| Data arm | Start cells made worse per run: change 1 | blindfolded | Bad endings, all start cells: baseline | change 1 | blindfolded |
|---|---:|---:|---:|---:|---:|
| Complete | 3.1 | 0.3 | 0.57% | 0.44% | 0.25% |
| Targeted | 2.4 | 0.2 | 1.68% | 0.36% | 0.07% |
| Placebo | 3.2 | 1.4 | 0.62% | 0.49% | 0.13% |

- **The side effect that made change 1 fail G5 mostly disappears.** The largest single rise at any start cell is 5.5, 1.3 and 5.2 points in the three arms, against 15.7, 7.6 and 99.6 for change 1.
- **No stalled cell in the Honeypot.** Change 1 had one stalled run in the placebo arm.
- **This matches the first study.** There the width was smaller at the fatal command than at the best one, because walking into a hazard has a perfectly predictable outcome. A penalty that knows the state rewards such commands at the cells beside a hazard; blindfolded, it cannot. I offer that as the likely reason. This study did not test it.

### 12.3 Gates for the blindfolded arm

Datasets passing of 3; a gate passes at 2.

| Setting | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm |
|---|---|---|---|---|
| **Honeypot** | 3, pass | 3, pass | 3, pass | 2, **pass** |
| Bottleneck 4.5 | 3, pass | 0, fail | 3, pass | 3, pass |
| Bottleneck 3.5 | 3, pass | 0, fail | 3, pass | 1, fail |
| Bottleneck 2.5 | 3, pass | 0, fail | 2, pass | 3, pass |

- **It passes all four Honeypot gates, three of them in 3 of 3 datasets.** Its G5 passes are clear ones (0.0 worsened cells against 1.3 for plain sharpening, and 0.0 against 1.7); the one dataset it loses is 0.7 against 0.3.
- **Bottleneck:** it never crosses the unseen bridge in 26 of 27 runs (one run at 6.3%), and with the bridge in the data it crosses 91%, 100% and 100% of the time. G3 fails on ties at zero, as for every support-aware method. G5 passes at two exit values and fails at 3.5.
- **Two stalled cells in the Bottleneck**, one each in two of the 81 runs (exit 3.5 with complete data, exit 4.5 placebo). Change 1 had none there. Bad endings stay at or under 0.20% of episodes in every Bottleneck arm.
- **One small lapse in the Honeypot's placebo arm**: one run of nine has 1.5% hazard-34 endings (0.16% on average), where change 1 had none. In that arm entry to hazard 34 is in the data.
- **Calibration is unaffected**: 90.7% to 91.7% coverage on fresh rows.

### 12.4 What the three follow-up arms say together

| Arm | Support penalty | Calibrated width | Hazard endings | Value | Cells made worse |
|---|---|---|---:|---:|---:|
| Baseline | no | no | 5.70% | 5.04 | |
| BCA as it was | no | at the true state | 2.08% | 5.90 | 2.2 |
| Support penalty only | yes | no | 2.56% | 5.10 | 3.3 |
| Change 1 | yes | at the true state | 0.00% | 5.95 | 2.4 |
| **Blindfolded** | yes | at a wrong state | **0.00%** | **5.97** | **0.2** |

Honeypot, targeted arm, means of nine runs.

- **The support penalty needs the state. The width does not.** Shuffling the support score's state wrecks the policy (section 11.4). Shuffling the calibrated width's state loses nothing and removes most of the harm.
- **What the calibrated width contributes is its shape as a function of the command**: a pull toward decisive commands. That pull is useful here, and it is not uncertainty about the state.
- **The best method tested is a support penalty plus a state-blind pull toward decisive commands.** It is the only one that passes all four Honeypot gates cleanly.
- **For BCA's central claim this is a negative result on this testbed.** Across five studies, state-specific calibrated uncertainty was never what improved behavior, and where its state knowledge made a visible difference it was harm.

Limits, as before: development seeds used for a fourth round; an arm designed after seeing the previous result; one-hot cells with uniform coverage, the easiest setting for a support score; and nothing here tests whether the width is needed at all, or whether any plain decisiveness regularizer would do the same job.

## 13 Confirmation on untouched datasets

Run November 10 2026, after David approved it and asked that Q values be changed directly as well. The claims and pass marks are in Amendment 6 of [PROTOCOL.md](PROTOCOL.md), written before any of these datasets existed. **This section is confirmation, not development.**

**The two versions.** Neither was changed for this run.

| | Version A | Version B |
|---|---|---|
| What it is | The blindfolded arm of [section 12](#12-second-follow-up-the-calibrated-width-blindfolded) | Changes 1 + 2 + 3 of [section 2](#2-what-was-tested) |
| Changes the actor (the part that picks commands) | yes | yes |
| Changes the Q values (the critic's training target) | no | yes |
| In development | best: 0.00%, all four Honeypot gates | 1.09%, failed G5 |

**The run (study v6, `results/confirm_v1`).** Fresh datasets on seeds 7101 to 7103 and training seeds 30 to 32, none used before. All 36 datasets pass their checks, and the withheld move is never executed in any targeted dataset. 108 TD3+BC baselines and 432 further policies, run once. Every calibration was usable, and the code was identical in both stages.

### 13.1 The environments held up on fresh data (claim C0)

| Setting | G0: baseline competent, runs of 9 | G1: the gap misleads the baseline, datasets of 3 |
|---|---|---|
| Honeypot | 9, pass | 2, pass |
| Bottleneck 4.5 | 9, pass | 3, pass |
| Bottleneck 3.5 | 9, pass | 3, pass |
| Bottleneck 2.5 | 9, pass | 3, pass |

**C0 holds in all four settings.**

The baseline critic's error at the withheld move (its value minus the move's true value), means of nine runs:

| Setting | Move in the data | Move withheld |
|---|---:|---:|
| Honeypot, the fatal move | +0.88 | **+9.86** |
| Bottleneck 4.5, the best move | -0.09 | **-0.38** |
| Bottleneck 3.5, the best move | +0.06 | **-1.19** |
| Bottleneck 2.5, the best move | -0.12 | **-0.91** |

So the Honeypot is an overestimation test and the Bottleneck is an underestimation test, as designed. The critic overestimates in 9 of 9 Honeypot runs and underestimates in 8 of 9 runs at each Bottleneck exit value.

### 13.2 Honeypot (claim C1)

Targeted arm, nine runs. *Bad endings* are episodes that end in any hazard or run out of time, over all start cells.

| | Hazard endings H | Runs at exactly 0 | Average true value | Bad endings |
|---|---:|---:|---:|---:|
| TD3+BC baseline | 3.96% | 2 of 9 | 4.94 | 1.69% |
| **Version A** | **0.00%** | **9 of 9** | **5.93** | **0.09%** |
| Version A's shuffled control | 3.37% | 5 of 9 | 4.69 | 16.08% |
| Version A's sharpened baseline | 0.98% | 5 of 9 | 5.75 | 0.45% |
| **Version B** | **0.72%** | 6 of 9 | 5.08 | 0.56% |
| Version B's shuffled control | 2.16% | 2 of 9 | 5.07 | 5.57% |
| Version B's sharpened baseline | 2.92% | 4 of 9 | 5.18 | 1.27% |

Gates, datasets passing of 3 (a gate passes at 2):

| | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm | C1 |
|---|---|---|---|---|---|
| Version A | 3, pass | 2, pass | 3, pass | 3, pass | **confirmed** |
| Version B | 3, pass | 2, pass | 3, pass | 1, **fail** | **not confirmed** |

- **Version A is confirmed.** It has no hazard-34 ending in any of its 27 runs across the three data arms. It makes no start cell worse than the baseline in any dataset. The one dataset it loses in G3 is a tie at zero with its sharpened baseline.
- **Version B is not confirmed.** It cuts hazard endings about five times, but it makes 13.7, 0.0 and 3.0 start cells worse in the three datasets, against 0.0 for its sharpened baseline.
- **Version B barely moves the Q value at the fatal move.** The critic's error there is +9.86 for the baseline and +8.74 for Version B (+10.40 for Version A). The fatal move has no rows in the data, so the critic is never trained on it. The penalty in the target lowers only the Q values of moves that are in the data.

### 13.3 Bottleneck (claim C2)

How often an episode from B crosses the unseen bridge (X), targeted arm, nine runs:

| Low exit pays | Baseline | Version A | Version B | Baseline with the bridge in the data |
|---|---:|---:|---:|---:|
| 4.5 | 2.37% | 0.00% | 0.62% | 70.38% |
| 3.5 | 5.78% | 0.00% | 0.00% | 93.70% |
| 2.5 | 2.16% | 0.00% | 1.45% | 97.74% |

C2 needed X to be lower than the baseline's in at least 2 of 3 datasets, at every exit value:

| Low exit pays | Version A lower in | Version B lower in |
|---|---|---|
| 4.5 | 1 of 3 | 1 of 3 |
| 3.5 | 1 of 3 | 1 of 3 |
| 2.5 | 2 of 3 | 2 of 3 |

**C2 is not confirmed for either version.** At exits 4.5 and 3.5 the baseline already crossed exactly 0% of the time in 2 of 3 datasets, and a tie does not count as lower. Version A crossed 0% of the time in 27 of 27 runs. Version B crossed more than the baseline in one dataset at exit 4.5 (1.86% against 0.00%).

- **Neither version brings the bridge back.** The best move was never in the data. The baseline mostly avoids it and both versions avoid it more.
- **The return from the decision cell does not change.** Baseline 4.42, 3.43 and 2.47 at the three exit values; Version A 4.49, 3.50 and 2.50; Version B 4.44, 3.42 and 2.46. Every method takes the low exit.
- **With the bridge in the data, Version A crosses more than the baseline**: 97.52%, 100.00% and 100.00% against 70.38%, 93.70% and 97.74%. Version B crosses 65.60%, 94.28% and 96.99%.

The other gates in the Bottleneck, reported as Amendment 6 said (they are not part of C2):

| Setting | Version | G2 | G3 | G4 | G5 no wider harm |
|---|---|---|---|---|---|
| Bottleneck 4.5 | A | 3, pass | 0, fail | 3, pass | 1, **fail** |
| Bottleneck 4.5 | B | 3, pass | 1, fail | 0, fail | 1, **fail** |
| Bottleneck 3.5 | A | 3, pass | 0, fail | 3, pass | 2, pass |
| Bottleneck 3.5 | B | 3, pass | 1, fail | 3, pass | 2, pass |
| Bottleneck 2.5 | A | 3, pass | 0, fail | 2, pass | 1, **fail** |
| Bottleneck 2.5 | B | 3, pass | 1, fail | 2, pass | 1, **fail** |

- **G3 fails on ties at zero**, as it did in development.
- **Both versions fail "no wider harm" at exits 4.5 and 2.5.** The largest case is Version A at exit 4.5 in dataset 7101: 23.0 start cells made worse (its shuffled control 23.3, its sharpened baseline 1.3). One of that dataset's three runs leaves 27 more start cells that never reach the goal.
- **With the bridge in the data, Version A ends badly more often than the baseline at two exit values**: 0.61%, 0.70% and 0.82% of episodes at exits 4.5, 3.5 and 2.5, against 0.58%, 0.14% and 0.04%.

Targeted arm, all start cells, means of nine runs:

| Low exit pays | Bad endings: baseline | Version A | Version B | Average true value: baseline | Version A | Version B |
|---|---:|---:|---:|---:|---:|---:|
| 4.5 | 0.23% | 9.60% | 5.12% | 4.11 | 3.80 | 3.94 |
| 3.5 | 0.06% | 1.35% | 1.58% | 3.75 | 4.36 | 3.79 |
| 2.5 | 8.67% | 9.62% | 8.83% | 2.61 | 3.27 | 2.58 |

### 13.4 What the confirmation says

- **Overestimation (Honeypot): Version A is confirmed.** It stops the overvalued fatal move in every run, raises the return, and does no wider harm there.
- **Changing Q values directly (Version B) is not confirmed.** It protects less and harms more than Version A, and the Q value at the unseen move stays too high.
- **Underestimation (Bottleneck): neither version helps.** A penalty can push a policy away from an unseen move. It cannot bring back an unseen move that is good. The caution claim itself is not confirmed, because of ties at zero.
- **Version A is not free of harm in the Bottleneck** when the bridge is withheld: it fails the "no wider harm" gate at two of the three exit values.

Limits: three datasets with three training seeds each; G1 passed in 2 of 3 Honeypot datasets; one-hot cells with uniform coverage; these seeds are spent and cannot confirm anything else.

## 14 Second fresh-data block: Version A against Version B

Run November 11 2026, after David asked to keep testing Version A against Version B. Amendment 7 of [PROTOCOL.md](PROTOCOL.md), written before any of these datasets existed, repeats the claims of section 13 and adds a direct comparison of the two versions. **This section is confirmation, not development.**

**The run (study v7, `results/confirm_v2`).** Fresh datasets on seeds 7201 to 7203 and training seeds 40 to 42, none used before. All 36 datasets pass their checks, and the withheld move is never executed in any targeted dataset. 108 TD3+BC baselines and 432 further policies, run once on the GPU. Every calibration was usable. The code was identical in both stages and unchanged from the first block apart from the seeds.

**One incident before the run.** The first smoke attempt (`smoke_v7`) fell back to the CPU because WSL had lost the GPU. It was stopped before any registered training began, and the smoke test was repeated on the GPU as `smoke_v7b`. The gate verdicts were then computed with the Windows interpreter, because the same script was very slow inside WSL. That evaluation is deterministic and trains nothing.

### 14.1 The environments held up again (claim C0)

| Setting | G0: baseline competent, runs of 9 | G1: the gap misleads the baseline, datasets of 3 |
|---|---|---|
| Honeypot | 9, pass | 3, pass |
| Bottleneck 4.5 | 9, pass | 3, pass |
| Bottleneck 3.5 | 9, pass | 3, pass |
| Bottleneck 2.5 | 9, pass | 3, pass |

**C0 holds in all four settings.**

The baseline critic's error at the withheld move, means of nine runs:

| Setting | Move in the data | Move withheld |
|---|---:|---:|
| Honeypot, the fatal move | +0.79 | **+9.90** |
| Bottleneck 4.5, the best move | -0.28 | -0.10 |
| Bottleneck 3.5, the best move | -0.24 | **-1.26** |
| Bottleneck 2.5, the best move | -0.32 | **-1.86** |

The critic overestimates in 9 of 9 Honeypot runs. It underestimates in 9 of 9 runs at exits 3.5 and 2.5, and in 6 of 9 at exit 4.5. **At exit 4.5 the error is small and no larger than with the move in the data, so that setting is a weak underestimation test in this block.**

### 14.2 Honeypot (claim C1)

Targeted arm, nine runs:

| | Hazard endings H | Runs at exactly 0 | Average true value | Bad endings |
|---|---:|---:|---:|---:|
| TD3+BC baseline | 3.50% | 1 of 9 | 5.14 | 1.10% |
| **Version A** | **0.18%** | 8 of 9 | **5.93** | **0.05%** |
| Version A's shuffled control | 0.42% | 7 of 9 | 3.55 | 30.85% |
| Version A's sharpened baseline | 0.57% | 8 of 9 | 5.80 | 0.19% |
| **Version B** | **0.23%** | 8 of 9 | 5.22 | 0.16% |
| Version B's shuffled control | 2.61% | 2 of 9 | 4.68 | 11.08% |
| Version B's sharpened baseline | 3.10% | 2 of 9 | 5.24 | 0.94% |

Gates, datasets passing of 3 (a gate passes at 2):

| | G2 width responds | G3 beats both controls | G4 specific to the gap | G5 no wider harm | C1 |
|---|---|---|---|---|---|
| Version A | 3, pass | 1, **fail** | 3, pass | 2, pass | **fails** |
| Version B | 3, pass | 3, pass | 3, pass | 0, **fail** | **fails** |

Where Version A lost G3 (hazard endings by dataset):

| Dataset | Baseline | Version A | Its shuffled control | Its sharpened baseline | G3 |
|---|---:|---:|---:|---:|---|
| 7201 | 1.29% | 0.00% | 0.00% | 0.00% | fail, a tie at zero |
| 7202 | 3.04% | 0.55% | 0.00% | 0.00% | fail |
| 7203 | 6.17% | 0.00% | 1.27% | 1.71% | pass |

- **By the rule fixed beforehand, Version A is "confirmed once, not replicated".** The size of its effect held. What failed is telling it apart from its controls.
- **The shuffled control reached low hazard endings by breaking the policy**: 30.85% of its episodes end badly and its value is 3.55. G3 does not look at that.
- **The sharpened baseline did not break**: 0.57% hazard endings and a value of 5.80. In this block the test could not separate Version A from a baseline that is simply made more decisive.
- **Version A in the other arms**: no hazard-34 ending in any of the 18 runs with complete or placebo data. Its one non-zero run is dataset 7202 in the targeted arm, at 1.7%.
- **Version B is not confirmed in either block.** It makes 2.0, 0.3 and 2.3 start cells worse in the three datasets, against 0.0 for its sharpened baseline.
- **Version B's Q value at the fatal move is still too high**: the critic's error is +8.84, against +9.90 for the baseline (+10.38 for Version A).

### 14.3 Bottleneck (claim C2)

How often an episode from B crosses the unseen bridge (X), targeted arm, nine runs:

| Low exit pays | Baseline | Version A | Version B | Baseline with the bridge in the data |
|---|---:|---:|---:|---:|
| 4.5 | 2.78% | 0.00% | 1.68% | 82.06% |
| 3.5 | 11.70% | 0.00% | 0.00% | 81.38% |
| 2.5 | 15.66% | 0.00% | 0.65% | 96.71% |

Datasets where the version's X is lower than the baseline's:

| Low exit pays | Version A lower in | Version B lower in |
|---|---|---|
| 4.5 | 1 of 3 | 1 of 3 |
| 3.5 | 3 of 3 | 3 of 3 |
| 2.5 | 3 of 3 | 3 of 3 |

**C2 is not confirmed for either version.** It holds at exits 3.5 and 2.5 and fails at exit 4.5, where the baseline already crossed 0% of the time in two of three datasets. Version A crossed 0% of the time in 27 of 27 runs, Version B in 24 of 27.

- **Neither version brings the bridge back**, as in the first block.
- **The caution costs a little at the decision cell here.** In this block the baseline still crosses sometimes, and crossing is the better move. The return from B is 4.37, 3.56 and 2.75 for the baseline at the three exit values, 4.49, 3.49 and 2.50 for Version A, and 4.43, 3.45 and 2.41 for Version B.
- **With the bridge in the data, Version A crosses more than the baseline**: 99.37%, 100.00% and 100.00% against 82.06%, 81.38% and 96.71%. Version B matches the baseline: 82.09%, 81.32% and 96.67%.

The other gates in the Bottleneck (not part of C2):

| Setting | Version | G2 | G3 | G4 | G5 no wider harm |
|---|---|---|---|---|---|
| Bottleneck 4.5 | A | 3, pass | 0, fail | 3, pass | 2, pass |
| Bottleneck 4.5 | B | 3, pass | 1, fail | 1, fail | 0, **fail** |
| Bottleneck 3.5 | A | 3, pass | 0, fail | 3, pass | 2, pass |
| Bottleneck 3.5 | B | 3, pass | 3, pass | 2, pass | 1, **fail** |
| Bottleneck 2.5 | A | 3, pass | 2, pass | 3, pass | 2, pass |
| Bottleneck 2.5 | B | 3, pass | 2, pass | 3, pass | 0, **fail** |

- **Version A passes "no wider harm" at all three exit values in this block.** It failed at two of them in the first block. The gate is judged on the targeted arm.
- **With the bridge in the data, Version A still ends badly more often than the baseline**: 1.46%, 2.02% and 2.02% of episodes at exits 4.5, 3.5 and 2.5, against 0.17%, 0.49% and 0.49%. Five of its 27 runs there have 2 to 8 start cells that never reach the goal.
- **Version B fails "no wider harm" at all three exit values.**

Targeted arm, all start cells, means of nine runs:

| Low exit pays | Bad endings: baseline | Version A | Version B | Average true value: baseline | Version A | Version B |
|---|---:|---:|---:|---:|---:|---:|
| 4.5 | 0.25% | 0.01% | 0.44% | 4.14 | 4.80 | 4.15 |
| 3.5 | 0.69% | 0.23% | 0.98% | 3.69 | 4.36 | 3.63 |
| 2.5 | 0.38% | 0.14% | 0.24% | 3.38 | 3.93 | 3.22 |

### 14.4 Version A against Version B, the registered comparison

Targeted arm, dataset means in dataset-seed order. A version wins a readout if it is strictly better in at least 2 of 3 datasets; a tie counts for neither.

| Readout | Version A by dataset | Version B by dataset | A better in | B better in | Winner |
|---|---|---|---:|---:|---|
| H1 Protection: hazard endings | 0.00% / 0.55% / 0.00% | 0.00% / 0.69% / 0.00% | 1 | 0 | no difference shown |
| H2 Side harm: start cells made worse | 0.00 / 0.67 / 0.00 | 2.00 / 0.33 / 2.33 | 2 | 1 | **Version A** |
| H3 Return: value over all start cells | 5.94 / 5.97 / 5.87 | 5.32 / 5.28 / 5.08 | 3 | 0 | **Version A** |
| B1 Caution: crossings, exit 4.5 | 0.00% / 0.00% / 0.00% | 0.00% / 1.96% / 3.07% | 2 | 0 | Version A |
| B1 Caution: crossings, exit 3.5 | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% | 0 | 0 | no difference shown |
| B1 Caution: crossings, exit 2.5 | 0.00% / 0.00% / 0.00% | 0.00% / 1.95% / 0.00% | 1 | 0 | no difference shown |
| B2 Cost of caution: value, exit 4.5 | 4.80 / 4.82 / 4.79 | 4.19 / 4.19 / 4.07 | 3 | 0 | Version A |
| B2 Cost of caution: value, exit 3.5 | 4.40 / 4.28 / 4.38 | 3.72 / 3.62 / 3.54 | 3 | 0 | Version A |
| B2 Cost of caution: value, exit 2.5 | 3.89 / 3.92 / 3.97 | 3.16 / 3.32 / 3.19 | 3 | 0 | Version A |

- **Honeypot: Version A is the better version.** It wins H2 and H3 and loses none.
- **Bottleneck: Version A is the better version.** B1 has no overall winner (Version A wins at one exit value; the rule needs two). Version A wins B2 at all three exit values.

The same table for the first block gives the same two verdicts. That is a description only, because the comparison was not registered before that block ran.

### 14.5 Both blocks together

This is a description only. Amendment 7 rules out pooling to rescue a claim.

| | C1 in the first block | C1 in the second block |
|---|---|---|
| Version A | passes | fails G3 |
| Version B | fails G5 | fails G5 |

Honeypot, targeted arm, 18 runs per method:

| | Hazard endings H | Runs at exactly 0 | Average true value | Bad endings |
|---|---:|---:|---:|---:|
| TD3+BC baseline | 3.73% | 3 of 18 | 5.04 | 1.40% |
| Version A | 0.09% | 17 of 18 | 5.93 | 0.07% |
| Version B | 0.48% | 14 of 18 | 5.15 | 0.36% |

Across the six Honeypot datasets, Version A has fewer hazard endings than Version B in 3 and ties in 3. It makes fewer start cells worse in 4, more in 1, and ties in 1. Its return is higher in all 6.

### 14.6 Where this leaves the two versions

- **Version A.** Its effect was the same size on two sets of new data, and it passed the "no wider harm" gate in the Honeypot both times. It is "confirmed once, not replicated", because the second block could not tell it apart from its controls.
- **Version B, which changes Q values.** Not confirmed in either block. It fails the "no wider harm" gate each time, and the Q value at the unseen move stays too high.
- **Head to head.** Version A is the better version on both environments, in both blocks.
- **Underestimation.** Unchanged: neither version brings back an unseen move that is good.

Limits: three datasets per block; hazard rates this close to zero let a single run flip a dataset's verdict; one-hot cells with uniform coverage; these seeds are spent and cannot confirm anything else.

## 15 Version A on nine fresh datasets: stopped at the baseline check

Run November 11 2026 under Amendment 8 of [PROTOCOL.md](PROTOCOL.md), written before any of these datasets existed. David asked what Version A needs to work and approved a larger run: Version A alone, in the Honeypot, on nine fresh datasets, with the pass marks fixed first.

**No Version A policy was trained. This run says nothing about Version A, for or against.**

**What was run (study v8, `results/confirm_v3`).** 27 fresh Honeypot datasets on seeds 7301 to 7309, all passing their checks, with the withheld move never executed. 81 TD3+BC baselines on training seeds 50 to 52, run once on the GPU. Every calibration was usable.

**Why it stopped.** Claim C0 needed every baseline run with complete data to be competent, and G1 to pass in at least 6 of 9 datasets.

| Check | Result | Mark | Verdict |
|---|---|---|---|
| G0: baseline competent with complete data | 25 of 27 runs | 27 of 27 | **fail** |
| G1: the gap misleads the baseline | 8 of 9 datasets | 6 of 9 | pass |

Both failing runs are in dataset 7306. The other eight datasets have 24 of 24 competent runs.

| Run | Start cell | Reaches the goal | Mark | Value | Mark |
|---|---|---:|---:|---:|---:|
| Dataset 7306, training seed 50 | 44 | 64.5% | 90% | 0.52 | 3 |
| Dataset 7306, training seed 52 | 35 | 88.4% | 90% | 4.17 | 3 |

By the stop rule of the protocol's section 5 (if G0 fails, no method is trained on that setting) and by Amendment 8 (if C0 fails, no claim is made), the run ended there.

**The baseline on these data, for the record.** Hazard endings: 0.58% with complete data, 6.36% with the fatal move withheld, 1.09% in the placebo arm. The critic's error at the fatal move is +1.10 with the move in the data and +10.19 with it withheld, too high in 27 of 27 runs.

**A weakness in the mark, which I should have flagged before the run.** "Every baseline run" is much harder to meet with 27 runs than with 9. Across the three fresh-data runs, 43 of 45 Honeypot baselines were competent. I scaled the gates to nine datasets and left this one as written.

**These seeds cannot be reused for confirmation.** Their baseline results have been seen. Training Version A on them now, or changing the competence mark for them, would break the rules this study runs under.

## 16 Version A on nine fresh datasets, with competence checked per dataset

Run November 11 2026 under Amendment 9 of [PROTOCOL.md](PROTOCOL.md), written before any of these datasets existed. It is the run of section 15 again on new seeds, with one change: baseline competence is judged for each dataset from the baselines alone, and a dataset that fails is dropped before any Version A policy is trained. **This section is confirmation, not development.**

**The run (study v9, `results/confirm_v4`).** Twelve candidate datasets on seeds 7401 to 7412. All 36 files pass their checks, and the withheld move is never executed. 108 TD3+BC baselines, then Version A and its shuffled control on the nine datasets used: 162 policies. Training seeds 60 to 62. Run once on the GPU. Every calibration was usable, and the code was identical in both stages.

### 16.1 Which datasets were used (claim C0)

| | |
|---|---|
| Candidate datasets | 12 |
| Valid: all three baselines competent | 11 |
| Dropped | 7404. One baseline reaches the goal 89.8% of the time from start cell 44; the mark is 90% |
| Used: the first nine valid seeds | 7401, 7402, 7403, 7405, 7406, 7407, 7408, 7409, 7410 |
| Valid and not needed | 7411, 7412 |
| G1: the gap misleads the baseline | 9 of 9 datasets |

**C0 holds.** All 27 baselines of the nine datasets are competent. The baseline critic's error at the fatal move is +0.69 with the move in the data and +9.87 with it withheld, too high in 27 of 27 runs.

### 16.2 What Version A did

Targeted arm, 27 runs per row:

| | Hazard endings H | Runs at exactly 0 | Largest single run | Average true value | Bad endings |
|---|---:|---:|---:|---:|---:|
| TD3+BC baseline | 5.98% | 5 of 27 | 20.8% | 4.90 | 1.94% |
| **Version A** | **0.24%** | **23 of 27** | 1.9% | **5.89** | **0.17%** |
| Version A's shuffled control | 7.02% | 17 of 27 | 25.0% | 2.90 | 39.12% |
| Version A's sharpened baseline | 2.33% | 19 of 27 | 17.7% | 5.66 | 0.97% |

Hazard endings by dataset:

| Dataset | Baseline | Version A | Shuffled control | Sharpened baseline |
|---|---:|---:|---:|---:|
| 7401 | 1.84% | 0.00% | 2.83% | 0.00% |
| 7402 | 5.31% | 0.00% | 0.54% | 1.43% |
| 7403 | 4.88% | 0.00% | 8.33% | 0.72% |
| 7405 | 10.47% | 0.55% | 8.33% | 4.22% |
| 7406 | 7.94% | 0.64% | 8.33% | 5.90% |
| 7407 | 6.02% | 0.00% | 8.33% | 1.19% |
| 7408 | 6.73% | 0.45% | 1.51% | 5.10% |
| 7409 | 6.20% | 0.54% | 8.33% | 2.37% |
| 7410 | 4.40% | 0.00% | 16.67% | 0.00% |

### 16.3 The verdict against the marks fixed beforehand

**R1, the gates as written.** Each must pass in at least 6 of 9 datasets.

| Gate | Datasets passing | Verdict |
|---|---|---|
| G2 width responds | 9 of 9 | pass |
| G3 beats both controls | 7 of 9 | pass |
| G4 specific to the gap | 9 of 9 | pass |
| G5 no wider harm | 7 of 9 | pass |

**R2, the average.** Paired differences in hazard endings across the nine datasets, control minus Version A.

| Control | Mean difference | Version A lower / tie / control lower | p (exact sign-flip test) | Verdict |
|---|---:|---|---:|---|
| Shuffled control | +6.78 points | 9 / 0 / 0 | 0.0020 | pass |
| Sharpened baseline | +2.08 points | 7 / 2 / 0 | 0.0078 | pass |

**R1 passes and R2 passes. By the decision rule fixed beforehand, Version A is replicated in the Honeypot: confirmed in two of three fresh-data runs** (section 13 passed, section 14 failed G3, this run passed).

- **The two datasets that fail G3 are ties at zero** with the sharpened baseline (7401 and 7410). Version A is never above the sharpened baseline in any dataset.
- **The two datasets that fail G5**: Version A makes 1.3 start cells worse in dataset 7401 and 3.7 in dataset 7409, against 0.0 for the sharpened baseline. Over all nine it makes 0.9 start cells worse per run in the targeted arm, against 1.1 for the sharpened baseline.
- **The shuffled control is broken, so beating it means little.** 39.12% of its episodes end badly, and seven of its 27 runs walk into the hazard from one start cell every time. The comparison that carries weight is the sharpened baseline: 2.33% against 0.24%.
- **Other data arms.** With complete data Version A has 0.05% hazard endings against 0.31% for the baseline. In the placebo arm it has 0.30% against 0.86%, with one run at 5.1%.

### 16.4 What to hold beside this result

- **The baseline was weaker on these seeds**: 5.98% hazard endings, against 3.96% and 3.50% in the first two blocks. That is what gave the test room to separate Version A from a more decisive baseline. The three candidates that were not used had baseline rates of 9.97%, 10.76% and 6.68%.
- **The competence rule was changed after the run of section 15 stopped.** It was fixed before these data existed, no mark of G0 changed, and it is decided before any Version A result exists. One candidate was dropped by it.
- **The Honeypot only.** The Bottleneck was not rerun. Nothing here changes its result: Version A does not bring back an unseen move that is good.
- **Every fresh-data run of Version A together** (a description): over 45 runs the baseline has 5.08% hazard endings (8 runs at zero), Version A 0.18% (40 runs at zero), and the sharpened baseline 1.70% (32 runs at zero).

Limits: three training seeds per dataset; one-hot cells with uniform coverage; these seeds are spent.

## 17 Files and commands

| File | What it is |
|---|---|
| `PROTOCOL.md`, Amendment 9 | Baseline competence judged per dataset, and the procedure for the rerun, written before its datasets were generated |
| `run_study.py --study v9`, `results/confirm_v4/` | The run of section 16: records, gate verdicts, `summary.md`, `version_a.md`, the dataset choice (`selected_datasets_honeypot.json`), and the baselines of the three candidates not used (`records_dropped.json`) |
| `data/confirm4_h/`, `receipts/confirm4_h_*` | Its twelve candidate datasets (seeds 7401 to 7412) and their check receipts |
| `PROTOCOL.md`, Amendment 8 | The nine-dataset run's claims, pass marks and decision rule, written before its datasets were generated |
| `run_study.py --study v8`, `results/confirm_v3/`, `confirm_version_a.py` | The run of section 15 (baselines only) and the script that would have judged it |
| `data/confirm3_h/`, `receipts/confirm3_h_*` | Its datasets (seeds 7301 to 7309) and their check receipts |
| `PROTOCOL.md`, Amendment 7 | The second block's claims, the Version A against Version B comparison and its decision rule, written before its datasets were generated |
| `run_study.py --study v7`, `results/confirm_v2/`, `results/confirm_v2_*.log` | The second block of section 14: records, gate verdicts, `summary.md`, `versions.md`, receipts and logs |
| `compare_versions.py` | The registered comparison of section 14.4. Reads the saved records and trains nothing |
| `data/confirm2_h/`, `data/confirm2_b/`, `receipts/confirm2_*` | The second block's datasets (seeds 7201 to 7203) and their check receipts |
| `PROTOCOL.md`, Amendments 3, 4 and 5 | The three changes and the two follow-up arms, with their fixed settings, gates, decision rules and kill rules, each written before its run |
| `PROTOCOL.md`, Amendment 6 | The confirmation's claims and pass marks, written before its datasets were generated |
| `run_study.py --study v6`, `results/confirm_v1/`, `results/confirm_v1_stage*.log` | The confirmation run of section 13: records, gate verdicts, summary, receipts and logs |
| `data/confirm_h/`, `data/confirm_b/`, `receipts/confirm_*` | The confirmation datasets (seeds 7101 to 7103) and their check receipts |
| `learner.py` | All fourteen arms, selected by one switch table |
| `test_learner.py`, `test_learner_regression.json` | 45 tests; the regression file pins the learner's outputs from before the changes |
| `run_study.py --study v3`, `--study v4`, `--study v5` | Train the three studies |
| `evaluate_study.py`, `summarize_study.py` | Gate verdicts per method; compact tables |
| `sharpen_after_training.py` | The exploratory analysis of section 11.3 |
| `results/dev_v3/`, `results/dev_v4/`, `results/dev_v5/` | `records.json` (every policy), `gates.json`, `g0.json`, `tables.md`, `summary.md`, run receipts. `dev_v4` also has `exploratory_sharpening.json` |
| `results/dev_v3_stage*.log`, `dev_v4_stage*.log`, `dev_v5_stage*.log` | Training logs |
| `results/smoke_v3/`, `results/smoke_v4/`, `results/smoke_v5/` | The smoke tests on dataset seed 7 |

To re-evaluate the saved policies without a GPU, from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B experiments/gridworld_ood_v1/evaluate_study.py --label dev_v3
```

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B experiments/gridworld_ood_v1/summarize_study.py --label dev_v3
```

The same two commands with `--label dev_v4` re-derive section 11, with `--label dev_v5` section 12, with `--label confirm_v1` section 13 and with `--label confirm_v2` section 14. With `--label confirm_v4` they re-derive section 16, and its verdict comes from:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B experiments/gridworld_ood_v1/confirm_version_a.py --label confirm_v4
```

For the comparison in section 14.4:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B experiments/gridworld_ood_v1/compare_versions.py --label confirm_v2
```

For section 11.3:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B experiments/gridworld_ood_v1/sharpen_after_training.py --label dev_v4
```
