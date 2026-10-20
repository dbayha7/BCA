# Does BCA work? The decision, 20 October 2026

Written from the records only (the line-ending-safe clone at `B:\`, commit `52914e4`, plus the fetched branches), the
five notes beside this file (`verdict.md`, `refusal.md`, `credit.md`, `process.md`, `steelman.md`, each with its
arithmetic scripts), and one parallel preview of the pooled groups on spent seeds
(`C:\Users\DAVIDB~1\AppData\Local\Temp\claude\C--Users-David-Bayha-BCA\9ab1f036-9335-42af-93a3-a9c39c11b8cc\scratchpad\v9preview\PREVIEW.md`).
Nothing was trained, run or approved. The WSL watcher was not alive at the first writing; at the revision (the same
day, after three reviews: `review_stats.md`, `review_fairness.md`, `review_feasibility.md`) it was alive and the
study machine's clone was read, never written. Every change made for the revision and every reviewer's finding
with its verdict is in `REVISION_LOG.md`; `c3_rule.py` / `c3_rule.out` (the twin rules' error rates and power on
certified seeds) and `ledger_v8_main.py` (version 8's CPU seconds by phase) were added for it. The text as first
written is kept as `DECISION_v1_before_reviews.md`. Two small scripts were added for this document: `power_v9.py` (certification odds at 5 and 10 seeds; the
lead mark under refusals) and `c2_rule.py` (the power of claim C2's rule); their outputs are `power_v9.out` and
`c2_rule.out`.

**Labels, used exactly.** *Pre-registered*: a mark fixed and approved before the data were seen, read on fresh seeds.
*Exploratory*: seeds 0-4, which every design has seen; it can license a design, never a claim. *Description only*: a
number from a subset chosen after the fact (seeds 11 and 14 are the two of five fresh seeds where BCA could act).
*Preview*: the parallel recomputation on spent seeds in a Windows copy of the project's pinned package versions, not
the pinned Linux environment; exact for row counts, recorded-critic heights on seeds 10-14 reproduced version 8's
numbers to zero difference.

**Words.** *BCA* (Bayesian Conformal Aggregation): a threshold on held-out critic errors set by a weighted Bayesian
quantile (miscoverage 0.1, credibility 0.95, 1,000 draws); with too few rows it refuses (threshold infinite).
*The switched ramp* (`obca_k1`): the pipeline that worked on seeds 0-4: BCA's *switch* turns a training penalty on
when the lower 5% bound of its posterior gap (full-throttle group minus the rest) clears a margin of 0.10; BCA's
*height* is the difference of the two groups' thresholds; the ramp is zero up to the data's edge, rises across the
empty stretch, flat beyond. *NET across worlds*: points gained over no caution where full throttle stalls minus points
lost where it helps, strong current (0 = the recording controller, 100 = the stall-aware reference). *The three
levels*: L1, BCA's output raises return on fresh seeds without pre-set harm; L2, BCA beats the best rule with no BCA
in it; L3, BCA's distinctive parts (the Bayesian quantile, eta, the importance weights) earn measurable credit.

---

## 1. The answer today, in plain words

| level | verdict | the numbers that carry it (label) |
|---|---|---|
| **L1** | **Not shown.** Strongly suggested. | The only pre-registered fresh-seed test that trained anything with BCA (near-80 version 8, claim 4) **failed by its pre-set rule**: BCA refused to set its height on 3 of 5 fresh seeds (10, 12, 13: held-out full-throttle groups of 19, 22, 18 rows against a floor of about 29), the approved code stopped those seeds before any arm trained, and the marks are pooled over all five (pre-registered). Where BCA could act it met every mark: nets 106.3 and 109.5 with leads of 29.7 and 28.1 over the best world-blind rival, keeps 8.1-23.0, exactly 0.0 loss in every no-stall situation (seeds 11, 14: description only); 102.9 on seeds 0-4 with a lead of 25.3 (exploratory). Every other pre-registered BCA return test was stopped by its own checks before a BCA return was read (H3, version 6/7's fresh seeds, weighting version 5's return gates) or shelved (hosts version 3). |
| **L2** | **Not shown; the records lean against it.** | A rule with no BCA in it, the plain mean-difference switch (same 0.10 margin) turning on the hand-set range limit, assembled from recorded arms: 107.4 vs BCA's 102.9 (seeds 0-4), 108.6 / 110.0 vs 106.3 / 109.5 (seeds 11 / 14), 103.0 vs 100.6 (near-80), 114.0 vs 110.5 (random rows); ahead by 3.2 on average over 17 seed-settings, in 12 of 17, within the per-seed spread of 4-5 (description only; never trained as one arm). |
| **L3** | **Not shown.** Contradicted for the switch, reduced to a dose for the height, absent for eta; coverage credit only for the weights. | The switch: the plain difference of means makes BCA's on/off call in **40 of 40** recorded seed-worlds (split conformal 37, rank test 36). The height: the top or second-top held-out score of a 30-53-row group (verified by rerunning `wbcp.calibrate`); in the main setting points rise with height up to about BCA's and are flat beyond it (the dose line, about 17.3 points per unit of log-dose, within-seed R² 0.63; the near-80 setting is not monotone: the split-conformal switch beats BCA on seeds 0 and 3); twice BCA's height does as well (+3.7 main 0-4, -1.9 near-80, -1.2 / +1.7 on 11 / 14): "large enough", not "calibrated" (step 2's own wording). Eta: not in the restructured pipeline at all; 0.2% of the band's extra width where it was (version 6, exploratory, as cited by the root-cause analysis). Weights: pre-registered coverage marks W1/W2 passed on 5 of 5 qualities; no return credit anywhere. The credibility's one measurable effect on fresh seeds was the refusal: a draw is infinite with chance 0.9 to the power of the group size whatever the scores say; certification is 0% at 18-22 rows, 70% at 29, 90% at 30, about 100% from 33. |

**Three things verified today that the earlier (lost) analysis had wrong or open.**
1. The chance that all five fresh seeds certify under the present evidence rule was **16-27%** (16% from the 10
   observed main-setting groups, 27% from the gate's 30 stalling-world logs), not 16-20%. The gate's 30 logs (on
   `B:\` in the stage2_gpu payload) kept 11-60 rows past 80%, mean 36.5, 7 of 30 below 29, and **exactly one (seed
   112, 11 rows) at 14 or fewer**; the 10 recorded main-setting seeds have none below 18. Claim 4 was more likely
   than not to fail by refusal before it started.
2. With BCA's groups pooled from every held-out row outside the ball, the chance that five fresh seeds all certify
   is **about 80%** (per seed 0.957 on the gate's spread; 65% for ten seeds), not 86-99.5%. The 93% figure is the
   chance for seeds 10-14's own known counts, not for new seeds. The preview's exact counts on all 20 known
   seed-worlds are 35-135 rows (1.8-2.3 times version 8's), and none of them refuses.
3. The heights shrink more than 5-35%: on seeds 11 and 14 the pooled height is **0.73 and 0.49** of the recorded one
   (2.29 to 1.67; 4.60 to 2.26); on 10, 12, 13 the pooled heights are 2.78, 1.85, 1.85 where version 8 had none
   (preview, recorded second critics). On four of the five seeds BCA's pooled height sits between the plain 90%-quantile
   gap (1.0-2.5 on those seeds) and twice it; on seed 14 it is above twice the gap (2.26 against 1.97); it is still
   an order statistic (the largest of 35-38 scores, down to the eighth of 135).

**So, today:** nothing about "BCA works" is shown at any level by a pre-registered test; the failure on record is a
refusal of BCA's own evidence rule, foreseeable from the gate's logs; everything favourable is exploratory or
description only; and the records say the favourable part belongs to "read held-out outcomes against a fixed
margin, then apply an ample penalty at the data's edge", which a plain rule does as well. One test settles L1, L2 at a
5-point margin, and the testable part of L3. It is specified below and is largely built.

---

## 2. The deciding test: near-80 version 9, main setting, fresh seeds

### 2.1 What it is

Step 2's confirmation (version 8's claim 4, the same host, protocol, matching, situations, points and three marks)
rerun on fresh seeds in the main setting only, with three changes to the evidence and the arms and none to BCA's
rule, the training rows, the removal ball, the test start sets or any rival's rule:

1. **BCA's two groups are pooled** from every held-out recorded row outside the ball (the refusal fix).
2. **A refusal fails BCA's arms on that seed, never the seed**; every other arm trains and is scored.
3. **Three plain twins are trained as real arms** (the mean-difference switch with the plain 90% gap, with twice
   it, and turning on the hand-set range limit), so that L2 and the height's share of L3 are read on the same seeds
   by heights alone: the mean-difference switch made BCA's on/off call in 40 of 40 recorded seed-worlds and in 10 of
   10 under pooling (preview).

### 2.2 The refusal fix, and why it is fair

**The rule.** Step 1's gap rule, unchanged, fixes the full-throttle edge `t_lo` on the kept training commands. BCA's
switch then reads every recorded row the learners do not train on: the whole calibration split (20,000 rows, the
1,000-row bank included) and the whole evaluation split (30,000 rows), keeping those whose command the ball would
keep. Rows above `t_lo` join the training-split survivors to form the **full-throttle group**; rows at or below
`t_lo` form the **rest**. BCA's second critic, which trains on none of these rows, scores both groups exactly as it
scored the bank in version 8; BCA's certificate (alpha 0.1, credibility 0.95, 1,000 draws, margin 0.10, the same
tags), the split-conformal, rank-test and mean-difference switches all read these two groups and no others. The rest
group is pooled by the same rule (one rule for both groups; its threshold moved from about -0.03 to -0.04 in the
preview, so the heights barely feel it).

**Counted before training.** Both group sizes depend on commands only. The data phase writes them to the seed's data
record before any learner trains, with the exact chance that a group of that size gives a finite height. No seed is
skipped or replaced on its count.

**Why it is fair (five checks).**
- It changes BCA's evidence, not BCA: alpha, credibility, draws, margin, the gap rule, the score and the second critic
  are unchanged; more rows of the same kind are read.
- Nothing scored reads these rows: points come from exact rollouts on the four fixed test start sets, and no arm
  trains on a held-out row (a leak check says so per seed).
- Every switch arm reads the same two groups, so no rival is disadvantaged; the mean-difference and split-conformal
  switches make the same calls under pooling as before on all 10 fresh seed-worlds (preview).
- It is the natural rule ("every recorded row the learners do not train on"), not a number tuned to the failing
  seeds; the alternatives that would tune BCA (a lower credibility, a fallback height) are rejected for that reason.
- Its honest consequence is stated up front: with 35-135 rows BCA's height is a lower order statistic than before,
  so the heights shrink (0.49-0.73 of version 8's on seeds 11 and 14). If BCA's gain on seeds 0-4 came partly from an
  inflated dose, this test shows it. Both heights are reported on seeds 0-4.

**What it buys.** Per fresh seed, certification rises from 0.77 to about 0.96 (the gate's spread of group sizes,
leak factor 2.0, about 1 refusal in 23 seeds; this factor is used throughout this document; the preview's factors
run 1.8-2.3, 1 refusal in 17 to 1 in 38). With ten seeds: no refusal with chance 0.65, at most one 0.94, two or
more 0.065 (across factors 1.8-2.3: 0.55-0.77, 0.89-0.97 and 0.03-0.11). Under the refusal rule below, one refusal
costs BCA about 7-10 points of its pooled lead and does not void the test.

**A second held-out log per seed and world (option C) is offered, default no.** One more log from the same recorder
(500/200/300 episodes, a declared stream, never trained on) would take the group to 70-270 rows and the chance of
any refusal in ten seeds to about 0.2%. It costs a generator entry, a leak check and the sentence "BCA read a second
log"; it moves BCA's height further toward the plain 90% point. The owner can switch it on before the freeze (choice
3 of the appendix).

### 2.3 The refusal rule

A refusal is BCA's own output ("cannot certify"), not a missing number, so it is scored as what BCA's controller then
does: **no ramp**. In a seed-world where BCA's switch is on and its height infinite, BCA's height-dependent arms
(`obca_k1`, `obca_k2`, the always-on ramp `obca_forced`, and the noise-floor copy `obca_k1_alone`) are not trained;
their numbers in that seed-world are the no-caution actor's; the controls matched to BCA's strength are matched to
zero strength, which is also the no-caution actor. The arm table is built per arm (no whole-table refusal), the other
arms train, the no-stall world's phase and both evaluations run. The seed is written down and never replaced. The
sentence is fixed (appendix, section 14). The literal reading of an infinite height as an infinite penalty (a hard
clip at the data's edge, which the hand-set range limit approximates and which scored 93 against BCA's 88 in the
stalling world on seeds 0-4) is **not** used, because it would reward refusing; it is reported as a composite only.

**Effect on the marks** (with the hand-set range limit added to the world-blind rivals, see 2.5). **C1 is read from its
three pooled marks only; there is no rule on the refusal count.** What a refusal does to the lead is an expectation,
not a rule: resampling the seven recorded main-setting seeds (`review_stats_calc.out`), the pooled lead over the best
world-blind rival with ten seeds is about 26 (sd 1.3) with no refusal, 19 (sd 2.5) with one, 8 (sd 2.4) with two and
-2 with three, against the mark of 10. So one refusal is expected to be survived, two are expected to fail the lead
mark (about 1 chance in 4 of passing at the recorded effect), three to fail it. Whatever the count, the look names
the mark that failed and reports, description only, whether the three marks pass on the certified seeds alone. The
parallel build currently makes any refusal fail C1 and C2 outright (version 8's "every seed is needed"); under the
pooled rule alone that would void the test with chance about 0.35 in ten seeds. This document fixes the no-caution
reading for C1 instead.

**Effect on C2 and C3** (2.6). A refused seed-world would enter a paired twin difference at about -100 (the
no-caution actor against a twin that scored about 100), so one refusal would decide the twin claims by itself.
C2 and C3 are therefore read on the certified seeds only; the refusal is charged in C1 and named in C2's sentence.
Reading them on the certified seeds is not outcome selection: a refusal depends on the group's size and on BCA's
random draws, both written before any arm trains. At least 8 certified seeds are needed; with fewer, C2 and C3 are
not read (chance about 0.01 at ten seeds).

### 2.4 Seeds: ten fresh seeds, 40-49

- **Block:** seeds 40-49 of the main generator (master seed 20261012). No registry, record or branch names 40-59
  (the build checked this repository, the near-80 branches and the placement design). Seeds 10-14 are spent, 5-9 are
  version 6/7's, 15-19 stay named for version 8's fallback, 20-34 belong to the placement design. Seeds 50-59 are
  reserved now for a version 10 and nothing else. No replacement block, no extension on an outcome.
- **Why ten, not five:** (i) one refusal does not decide C1 (above; with five seeds one refusal already pushes the
  expected lead to about 8, a coin flip against the mark of 10, and C2 would be read on four seeds); (ii) power
  for C2 (2.6): a true 10-point edge over a twin is found with chance 1.00 (per-seed spread 5) or 0.79 (spread 10),
  a 5-point edge with 0.48, and a zero edge is called "ahead" with chance 0.00 at spread 5 and 0.03 at spread 10;
  (iii) the composite's recorded 3-4.5-point edge over BCA reads "tie" with chance about 0.7-0.9 (2.6); (iv) cost:
  about 18 CPU-hours, about 5-5.5 hours of wall time on the desktop in one sitting, no cluster needed.
- **Why not twenty:** more seeds do not decide L2 at the recorded effect size. The composite's recorded edge over BCA
  is 3-4.5 points (3.2 over 17 seed-settings, 3.6 over the seven main-setting seeds, 4.5 over seeds 0-4). With a
  5-point margin the likely reading of such an edge is "tie" at any seed count (0.73 with ten seeds, 0.67 with
  twenty at the recorded spread; 0.90-0.96 for a 3-point edge), and the chance of reading "ahead" for a true
  5-point edge stays near 0.5 whatever the count, because the mean itself must clear the margin. Deciding a
  3-4-point edge needs a smaller margin, which is a different claim (below the per-seed spread of 4-5 and inside the
  training noise), not more seeds. Twenty seeds raise the sd-10 power for a 10-point edge from 0.79 to 0.91, double
  the cost and need the cluster entry; they are the right size for a version 10 only if a 10-point question is
  still open after version 9.

### 2.5 Arms (main setting, both worlds)

Version 8's claim-4 arm set, unchanged: step 2's six ramp arms (`obca_k1`, `obca_k2` at twice BCA's height,
`obca_forced` always on, `fixed_gated` BCA's switch with height 1.0, `conformal_k1` the split-conformal switch,
`meangap_k1` the rank-test switch), the plumbing check `w0_check`, the matched controls (even pull, push-less, two
random surfaces), the matched range limit `standin`, the hand-set range limit `range_hinge`, the perfect band, the
no-caution, cloned and full-log actors.

**New, trained as real arms by their own rule in every seed-world:**

| arm | switch | height / penalty | why it is there |
|---|---|---|---|
| `meandiff_q90x1` (**C2 twin 1**: the mean-difference switch with the plain 90%-quantile gap) | on when mean(full-throttle group) - mean(rest) > 0.10 (the same 0.10 as BCA's margin, carried unchanged) | the plain 90%-quantile gap of the same two groups (split conformal's statistic; no Bayesian step) | the plain height on the near side of BCA's, with BCA's own call: it differs from BCA's switched ramp in height only, so BCA ahead of it is a dose reading. The split-conformal switch cannot play this part: it differs in its switch too (wrongly on in the no-stall world of seeds 10 and 13 under pooling, gaps 0.121 and 0.116 against its 0.10 margin; a wrong "on" cost 4.4 points on near-80 seed 3). |
| `meandiff_q90x2` (**C2 twin 2**: the same switch with twice that gap) | the same | twice the plain 90%-quantile gap (step 2's own k = 2 factor) | the plain height on the far side of BCA's on four of the five preview seeds: under pooling BCA's heights were 1.67-2.78 on seeds 10-14 against plain gaps of 0.99-2.48 and twice them 1.98-4.96, so the ladder plain x1 < BCA x1 < plain x2 holds on seeds 10-13; on seed 14 BCA's 2.26 is above twice the gap (1.97). If BCA beats plain x1 but not plain x2, its edge is a dose. |
| `meandiff_limit` (**C2 twin 3**: the same switch turning on the hand-set range limit) | the same | when on, the hand-set range limit's own penalty (coefficient 1.0, `range_hinge`'s program); when off, no caution | the no-BCA composite that led BCA by 3.2 points on average in the records, now trained as one arm |
| `obca_k1_alone` (validity, and C2's BCA side) | BCA's | BCA's height, trained alone instead of in step 2's six-arm batch | the twins train alone, so C2 and C3 compare them with this copy (the same training mode); the noise floor `obca_k1` minus `obca_k1_alone` is the validity check that the batch did not move BCA's points by more than 5 (if it did, C2 and C3 are dropped). Both differences, against `obca_k1` and against the copy, are reported. |
| `conformal_k1` (reported) | the plain 90%-quantile gap > 0.10 | the plain 90%-quantile gap | step 2's arm, kept and reported with its no-stall false alarms counted; it differs from BCA in its switch, so it is not a twin |
| `meandiff_k1` (reported) | the mean-difference switch | the mean gap itself (0.4-0.8) | version 8's arm, the bottom of the dose ladder; BCA beats it by 40+ points by dose alone, so it is not a twin |

**Reported composites, no training:** BCA's switch turning on the hand-set range limit; "refusal means the range
limit" (BCA's height when finite, the range limit when it refuses). Both from trained arms.

**One change to C1's rival set (its marks are unchanged):** the hand-set range limit joins the world-blind rivals for
the net-lead mark (it cannot switch itself off, and with refusals it is the rival that keeps the lead mark from
collapsing). On every recorded main-setting set it is below the always-on ramp (76.5 against 77.7 on seeds 0-4;
64.0 against 76.6 on seed 11; 79.3 against 81.5 on seed 14), so nothing changes where BCA acts; in the near-80
setting, which version 9 does not run, it is above the always-on ramp (54.7 against 49.1). Version 8 listed it as
"reported; used in claim 3" (appendix 15.6 records the departure).

### 2.6 Claims, marks, tests and the multiplicity rule (fixed now)

**C1 (L1): version 8's claim 4, its three marks unchanged; the world-blind rival set gains the hand-set range limit
(appendix 15.6).** Pooled over the ten seeds, main setting, strong current: net lead at least 10 over the best
world-blind rival; BCA minus no caution at least -3 in recorded conditions, big waves and big kicks (stalling world);
BCA minus the matched even pull at least 15. All three needed. A refused seed-world enters every pooled mean with the
no-caution actor's numbers (2.3). A seed that fails one of the gate's own data checks fails every mark, as in version
8. A plumbing stop on a fresh seed also fails every mark that needs the stopped phase (version 8's rule, kept so that
no stop can be rerun into a pass), but it is not a reading of BCA's return: the fail sentence then says that a stop
decided it, the marks on the seeds that ran are reported as description, and the verdict of section 3 is "no
reading".

**C2 (L2 and the height's part of L3): BCA's switched ramp beats each plain twin.** For each twin T, the paired
per-seed difference d = NET(`obca_k1_alone`) - NET(T) across worlds, strong current, over the certified seeds (the
seeds where BCA set a finite height in both worlds; n of them, at least 8). **"BCA ahead of T"** when the mean of d
is at least **5.0 points** and d is positive in at least **ceil(0.8 n)** seeds: 8 of 10, 8 of 9, 7 of 8 (a coin gives
that many with chance 0.055, 0.020, 0.035). **C2 holds when BCA is ahead of all three twins** (multiplicity rule:
`all_three`; a conjunction keeps the error rate at the per-twin level, so no Holm step is applied). The BCA side is
the copy trained alone, like the twins; the difference against the batch-trained `obca_k1` is reported beside it.

**C3 (the mirror of C2, one claim per twin): a plain twin beats BCA's switched ramp.** **"T ahead of BCA"** when the
mean of d is at most **-5.0** and d is positive in at most **1** of the n certified seeds (coin chance 0.011, 0.020,
0.035 for n = 10, 9, 8; stricter than C2's sign count because three twins give three chances). The three readings
form a family: under a true tie all three read "tie" except with chance 0.001 (per-seed spread 5), 0.014 (7.5),
0.023 (10), 0.028 (15) that some twin reads ahead (`c3_rule.out`). Otherwise the twin's reading is **"tie"**, written
"BCA's switched ramp was not shown to beat T by 5 points" (never "did as well" or "interchangeable": a true
5-point edge reads "tie" about half the time). The one-sided 95% lower bound of each mean is reported, not used.

- **Why 5 points:** the project's smallest mark unit (version 8's claims 2 and 3: "5 points, and 4 of 5 seeds"; step
  2's 5-point wording rule; round 1's bar), and about one per-seed standard deviation of the paired differences
  within the main setting (1.9-5.2). A 5-point mean is a difference an operator would notice.
- **Why ceil(0.8 n):** the sign part at about the 5% level for ten seeds (a coin gives 8 or more of 10 with chance
  0.055); version 8's 4 of 5 (chance 0.19) was carried by the mean.
- **Power per twin (`c3_rule.out`, ten certified seeds):** true edge 0 (spread 5): "BCA ahead" 0.001; 3: 0.10; 5:
  0.48; 10: 1.00; 5 (spread 10): 0.31; 10 (spread 10): 0.79; the main 0-4 BCA-conformal spread (5.6, sd 1.9): 0.84.
  A zero edge reads "BCA ahead" with chance 0.00 at spread 5 and 0.03 at spread 10 (0.044 at 15). For C3, the
  composite's recorded edge over BCA (-4.5, sd 5.2) reads "T ahead" with chance 0.27 and "tie" 0.73; -3.2 reads
  "T ahead" 0.09; -10 (sd 5) 0.98. So the test finds a 10-point edge either way, finds a 5-point edge about half
  the time, and invents one with chance at most about 0.03 per twin.
- **Power of the conjunction (C2 as a whole, shared BCA side):** three true 5-point edges: 0.23 (spread 5) or 0.09
  (spread 10); three true 10-point edges: 0.99 or 0.56; the predicted pattern (6, 0, -3.5): 0.00. C2 is a demanding
  claim by design: it credits BCA's height only when it beats a plain height on both sides of it and the range limit.
- **Validity for C2 and C3:** the noise floor (`obca_k1` minus `obca_k1_alone`, pooled, stalling, strong current)
  within 5 points; otherwise both are dropped and only reported.

**Multiplicity across claims:** C1, C2 and C3 are read separately; none rescues another (version 8's rule 9.6).

**L3, reported with no mark:** the switch-call table (BCA, mean difference, split conformal, rank test) over the 20
seed-worlds; each height's rank among its group's scores; the dose ladder (mean gap, 1.0, plain x1, BCA, plain x2,
2 x BCA, the range limit) per seed; the refusal count and its sentence; the two composites; the per-twin readings
against the batch-trained `obca_k1`; the fresh start bank readout if chosen (choice 10).

### 2.7 Validity checks and stops kept, and what is description only

- **Kept per fresh seed:** the gate's five world checks on the ball and its leak checks; the training-file check;
  new leak checks for the pooled groups (no training row other than a survivor in the groups, no ball row, every
  held-out row outside the ball exactly once, the outcome-free count equals the scored sizes). A failure fails the
  seed's marks as in version 8 and is never replaced.
- **Plumbing, after approval:** unit tests, the 40-step smoke of every phase kind, and the bit-for-bit reproduction
  on seed 0 of version 8's items plus version 8's own switch computed with the old groups (survivors and bank) before
  the pooled one. A reproduction difference ends the version.
- **Before approval, with the frozen code (no training):** the no-training checks on seeds 0-4 as in version 8, the pooled
  counts on seeds 0-4 (no critic is needed), and the pooled switch, heights and calls on the spent seeds 10-14 from
  version 8's recorded second critics (on the desktop under `runs/shift_explore/near80/learners/main/`; nothing is
  trained). Step 1's second critics for seeds 0-4 are **not** on the desktop (step 1 ran on another machine, "vm",
  14 October), so their pooled heights cannot be read without retraining ten second critics, about 1 CPU-hour of
  exploratory training on seen seeds: offered as choice 11, default no. Kill rule: BCA's and the mean-difference
  switch's pooled calls on seeds 10-14 equal version 8's recorded calls (on in 5 of 5 stalling, off in 5 of 5
  no-stall seed-worlds); otherwise the design is redrafted and no approval is asked. The number of redrafts is
  written in section 0 of the pre-registration. The pooled heights, the plain gaps and the twins' calls on seeds
  10-14 are pinned in the pre-registration as exploratory (spent seeds).
- **No outcome-based stop on fresh seeds**, no exploratory stage, no harm stop (the main setting's harm is 98.9
  points). The budget guard and the safe stop stay.
- **Description only:** the per-twin readings beyond C2's and C3's pass/fail, and the readings against the
  batch-trained `obca_k1`; the lower bounds; the marks on the certified seeds alone; the dose ladder; the switch-call
  table; the height ranks; the refusal sentence; the two composites; seeds 10-14's old and new heights; the readouts
  (held to 70%, capped at 80%, as version 8); the noise floor's value; the fresh start bank readout if chosen.

### 2.8 Predicted outcome, written now so that surprises show

C1 passes with 0 or 1 refusal (chance of 0-1 refusals about 0.94; of two or more about 0.065, in which case the
lead mark is expected to fail). C2 fails: against the plain gap x1 BCA reads "ahead" or "tie" (the pooled heights
are 1.1-2.3 times the plain gap, worth about 2-14 points on the dose line, mean about 6, but the line's within-seed
R² is only 0.63); against the plain gap x2 "tie" (its heights are 0.9-1.8 times BCA's, on the plateau: twice BCA's
own height scored 1.2 below BCA on seed 11 and 1.7 above on seed 14); against the gated range limit "tie" (the
composite's recorded edge is 3-4.5, read "tie" with chance about 0.73). C3 is not expected to fire (chance about
0.27 for the gated range limit at the recorded edge, less for the others). Expected verdict: **"BCA works as a
switched harm gate on ThrusterChain (L1, pre-registered). It was not shown to beat a plain switch with a plain height
of either size, or the hand-set range limit, by 5 points (L2 not shown). Its Bayesian parts earn no return credit
(L3 not shown)."** If C1 fails with at most one refusal on a mark read from the numbers, step 2's result does not
carry to fresh seeds, and that is the answer.

---

## 3. The decision rule: every outcome and the verdict the project will state

| outcome | the verdict about BCA, stated as such | what the project does next |
|---|---|---|
| **No reading**: the reproduction differs, a pre-approval check fails, or on a fresh seed a gate data check fails or a phase stops (C1 then fails by the pre-set rule, and its sentence says a stop decided it) | "No reading. The verdict of section 1 stands unchanged." With a stop: "C1 failed by the pre-set rule because seed [s]'s [phase] stopped; this is not a reading of BCA's return. On the seeds that ran, the marks read [description]." | Fix the cause (an exception is retried after an amendment, a pre-set stop is final); the touched seeds are written down as spent; a version 10 on seeds 50-59 if needed. |
| **C1 passes, 0-1 refusals, C2 passes** | "BCA works on ThrusterChain, pre-registered: its switched ramp raised return on fresh seeds without pre-set harm (L1), and its height beat a plain height on both sides of it and the hand-set range limit by at least 5 points (L2 and the height's part of L3). The switch's credit remains unshown: the plain switch made the same call." | Write it up as the project's first pre-registered BCA return result. Next questions, in order: the weights under a known shift (the placement design's module W), other hosts (CQL, IQL, ReBRAC: no ThrusterChain return exists). |
| **C1 passes, 0-1 refusals, C2 fails, no C3** (the predicted pattern, whether BCA is ahead of plain x1 or tied with it) | "BCA works as a conformal harm switch on ThrusterChain (L1, pre-registered). It was not shown to beat a plain switch with a plain height of either size, or the hand-set range limit, by 5 points (per twin: mean [d], one-sided 95% lower bound [lb], ahead in [k] of [n]). No credit beyond a plain rule is shown (L2 not shown, L3 not shown)." If BCA was ahead of plain x1 only, the sentence adds: "Its gain over the plain height was matched by twice that height and by the range limit: a dose, not a calibration." | State the working method as "read held-out outcomes against a fixed margin; apply an ample penalty at the data's edge"; BCA is one instance of it. Stop attribution work on ThrusterChain. The only open BCA-specific question is the weights under a known shift (module W), which this test bed cannot ask. A smaller margin would be a new claim, not a rerun. |
| **C1 passes, 0-1 refusals, C3 holds for some twin** | "L1 shown, L2 contradicted at the pre-registered level: a rule with no BCA in it ([T]) beat BCA's switched ramp by at least 5 points on fresh seeds (mean [d], positive in [k] of [n]). On ThrusterChain the recommended method is that rule." | The report names the rule; BCA's parts are dropped on this test bed; the name BCA is not attached to the recommended method. |
| **C1 passes with 2 or more refusals** (the marks pass even with those seeds charged as no caution) | "BCA works where it can act (L1 at the pre-set marks, with [F] of 10 seeds charged as no caution) but cannot be deployed on logs of this size as built: even reading every held-out row, its evidence rule refused on [F] of 10 fresh seeds." C2 and C3 read on the certified seeds as above. | A declared non-BCA fallback becomes part of the method (a method change; a version 10 with the fallback as the primary) before any deployment claim. |
| **C1 fails, 0-1 refusals, a mark missed on the numbers** | "L1 contradicted at the pre-registered level: step 2's exploratory result did not reproduce on fresh seeds; the [mark] mark failed by [x]. On the certified seeds alone the marks read [description]. 'BCA works' is not supported on ThrusterChain." | Close the ThrusterChain return claims for BCA. The C2 and C3 readings are description. The hosts version 3 result (TD3+BC medium-expert +6.54 return units, lower bound +1.40, p 0.0267, Holm 0.1069, shelved; the same record flags harm: standard BCA raised F2 on recorded conditions by +6.84 percentage points, lower bound +5.76, big waves +6.60, big kicks +2.22) stays the only positive BCA-as-built return effect on record, on another test bed, carrying its own harm flag, and is pursued only if the owner chooses. |
| **C1 fails with 2 or more refusals** | "BCA as built cannot be deployed on logs of this size: even reading every held-out row, its evidence rule refused on [F] of 10 fresh seeds, and with those seeds charged as no caution the [mark] mark failed by [x]. Where it acted its marks read [description], description only." | Either a declared non-BCA fallback becomes part of the method (version 10 with the fallback as the primary) or the ThrusterChain line closes. The C2 and C3 readings are description. |
| **C1 fails, C2 passes** | "No 'works' claim: BCA was the best of the rules tested by at least 5 points, but it did not reach the pre-set bar on fresh seeds. The bar or the pipeline failed, not BCA relative to its rivals." | Report which mark failed (lead, keeps the gain, even pull) and by how much; no claim; a version 10 only if the failed mark is a plumbing-like artefact (for example matching) rather than return. |
| **C2 and C3 dropped** (noise floor above 5) **or not read** (fewer than 8 certified seeds) | C1 read as usual. "C2 and C3 were not read: [training noise exceeds their margin / only [n] seeds certified]." | The per-twin differences are reported; a larger batch or more seeds would be a version 10 choice; L2 and L3 stay "not shown". |

The sentences for each cell are in the appendix (section 14) and go into `prereg.json` word for word. No sentence
moves after a number exists.

---

## 4. Cost, time, what is built first, and the owner's steps

### 4.1 Cost and time

| | desktop (Framework Desktop, WSL, as versions 6-8) | one Orion node (hpc.charlotte.edu) |
|---|---|---|
| ten fresh seeds, both worlds, every arm | about 16.5-17 CPU-hours: version 8's ledger gives 1.52 CPU-hours per main-setting seed (seeds 11 and 14: 1.53 and 1.51; `ledger_v8_main.py`), plus about 0.14 per seed for the five programs run alone (four lone ramp calls of about 63 s each in the stalling world and about 40 s in the no-stall world, and the hinge) and the pooled scores (seconds) | the same CPU-hours |
| wall time, run | about 5-5.5 h in one sitting under version 7's memory rule (2 actor phases at once: ten stalling-world phases of about 47 min and ten no-stall phases of about 13 min, after a lead-in of about 20 min); about 2.5 h if the actor reservation is set to 1.25 x its measured peak (choice 5) | about 1.5-2 h at 24 lanes, plus the queue |
| plumbing | about 1 h of wall (version 8: 62.5 min, 1.23 CPU-hours; version 9's reproduction adds the lone programs, about 5 min) | about 1 h, plus a probe first |
| budget guard | the forecast constant stays 2.2 CPU-hours per seed (a safe margin over the 1.65 expected; the guard pauses at twice the forecast, 44 CPU-hours) and the cap is 2.5 x 22 = 55 CPU-hours | the same |
| extra work before it can run | none: the desktop entry exists | the machine table, `place_committed_copies.py`, a seed-0 probe (CPU model, `avx512f`, retrain the gate's seed-0 critics and compare fingerprints), and a decision on the reproduction reference (step 2's saved files exist only on the desktop; step 1's second critics exist on no machine the project still has); about a day of work and one open point (`process.md` 4.4). The two cluster scripts drafted in the build are untested and are left out of version 9's manifest (BUILD_LIST) |

**Decision: the desktop runs version 9.** Ten seeds cost one evening there; the cluster buys nothing for ten seeds
and adds a day of entry work and an unresolved bit-for-bit question. The cluster entry is worth building for a
version 10 of twenty seeds or for the weighting world.

### 4.2 What is built and checked first (the build is well advanced; `BUILD_LIST.md` has every file and line)

A parallel step of this workflow has drafted version 9 on branch `claude/near80-v9` of the `B:\` clone: 26 files
edited in place under `outputs/shift/beyond-the-data/near80/` and `near80/code/`, three new (`prereg_draft.json`,
two cluster scripts), one deleted (`code/v6bca.py`), uncommitted. It holds the pooled groups (`near80lib.
pooled_groups`, `learner_phases.switch_scores_pooled`), the outcome-free count in the data record, the per-arm
refusal, `meandiff_k1`, `meandiff_limit`, `obca_k1_alone`, host profiles, and placeholders that this document fixes.
In order:

1. **Fill every placeholder** (`BUILD_LIST.md` section 1): `near80_constants.py` (`PLACEHOLDERS` emptied; `FRESH` =
   40-49; `MARKS_C2` = 5.0, ceil(0.8 n) with 8 for ten seeds, `all_three`; new `MARKS_C3` = -5.0, at most 1 positive;
   `MIN_CERTIFIED` = 8; `OWNER_ANSWER` after the owner answers section 16); `prereg_draft.json` by hand (5 marks);
   `PREREGISTRATION.md` (14 marks, line 3's DRAFT line and line 1's title to the FINAL forms) and `README.md` (1
   mark, line 3); the first docstring line of every code file ("DRAFT" is refused by the freeze).
2. **Changes this document asks of the build** (`BUILD_LIST.md` sections 2-6): (a) the twins `meandiff_q90x1` and
   `meandiff_q90x2` as lone programs, `TWIN_ARMS` = (`meandiff_q90x1`, `meandiff_q90x2`, `meandiff_limit`),
   `conformal_k1` and `meandiff_k1` reported; (b) the refusal reading: C1 reads a refused seed's BCA-dependent arms
   and matched rivals as the no-caution actor (not "fails every mark that needs BCA"); C2 and C3 read the certified
   seeds with `obca_k1_alone` as BCA's side; (c) the lead computed in `pooled.py` over `WORLD_BLIND + (range_hinge,)`
   with step 2's own lead reported beside it (step 2's `go_no_go` is pinned and cannot change); (d) the mirrored
   reading, the tie wording, the one-sided 95% lower bound, the certified-seed marks, the switch-call table, the
   height ranks, the dose ladder and the binomial certification chance; (e) the pre-approval facts: the counts on
   seeds 0-4 and the pooled switch on the spent seeds 10-14 from version 8's recorded second critics, with the kill
   rule; (f) section 14 of the pre-registration in the extractor's labels, with the new sentences (C3, the C1 last
   sentence, the certified-seed reading, "not read"); (g) version 9 stays in place in `near80/` (choice 7; version
   8's chain is checkable at `52914e4`), and `claude/near80-v9` is merged into `claude/main-bca-connection-x4o46s`
   before the freeze so the owner's `git pull --ff-only` and the import's one-commit rule work; (h) the results step
   prints `git add -f` (`.gitignore` line 7 `data/` hides `results-<utc>/files/data/`, the gap that cost version 8
   a second commit); (i) the two cluster scripts leave `TOP_FILES` and the folder (drafted, untested); (j) the
   `__pycache__` folders under `near80/` are deleted (the freeze refuses them). Option C (the second held-out log)
   only if the owner chooses it.
3. **Tests before the freeze** (`BUILD_LIST.md` section 7): `test_approval.py` and `test_runner.py` still carry
   version 8's names (`K.CONFIRMATORY`, `K.RESERVED`, `K.BAND_*`, `no_training_checks`, ...; 13 and 10 lines) and
   `test_v9.py`, which the manifest lists, does not exist; `test_near80.py` and `test_pooled.py` have been
   retargeted. `test_v9.py` covers the pooled groups (every held-out row outside the ball exactly once; no training
   row other than a survivor; no ball row; count equals scored sizes), the per-arm refusal and the look's two
   readings, C2's and C3's rules on synthetic nets (ties, the sign thresholds by n, the noise floor, fewer than 8
   certified), the new sentences, and the stub end-to-end run with a refused seed. Then `freeze.py check` and
   `approve_local_near80.py --check` on a fresh clone, inside WSL with the pinned environment (the freeze's unit-test
   step uses it).
4. **On the desktop, with the frozen code, before approval (nothing trained):** the gate's checks on seeds 0-4; the
   pooled counts on seeds 0-4; the pooled switch, heights and calls on seeds 10-14 from version 8's recorded second
   critics, and the kill rule; the facts. About 10-15 CPU-minutes.
5. **Independent review** of the draft and the frozen code ("do not approve yet" until it passes), then the freeze
   and the fingerprint.
6. **Approval at the owner's terminal, import, plumbing (1 h), the run (5-5.5 h), results, the independent check.**
   Machine time about 6.5 hours in one sitting. With the build this far along, results are a few days away once the
   owner's answers are in.

### 4.3 The owner's steps

1. **Answer the choices** of the appendix's section 16 (defaults: ten seeds 40-49; the no-caution reading of a
   refusal for C1 and the certified seeds for C2 and C3; no second held-out log; the desktop; version 7's memory
   rule; 10 lanes; version 9 in place in `near80/`; the twins' factors 1 and 2; results committed from the desktop;
   no fresh start bank; no retrain of seeds 0-4's second critics). "Accept the defaults" is enough; it is recorded
   with its time.
2. **When the session says the final version is frozen and independently re-checked:** in PowerShell
   `wsl -d Ubuntu-24.04 -u root`, then inside Ubuntu (the branch is `claude/main-bca-connection-x4o46s`, the one
   your clone is on):

       cd /home/user/bca && git pull --ff-only
       python3 -I outputs/shift/beyond-the-data/near80/approve_local_near80.py --check
       python3 -I outputs/shift/beyond-the-data/near80/approve_local_near80.py
       git add outputs/shift/beyond-the-data/near80/approval_near80_<16>.json outputs/shift/beyond-the-data/near80/approval_near80_<16>.json.sha256
       git -c user.name="David Bayha" -c user.email=35092758+dbayha7@users.noreply.github.com commit -m "Near-80 v9: approve <16>"
       git push origin claude/main-bca-connection-x4o46s

   Tell the session; it imports and pushes the import record; then `git -C /home/user/bca pull --ff-only`.
3. **Run:** `sh outputs/shift/beyond-the-data/near80/run_near80.sh selftest` (must end READY), then in `tmux`:
   `run_near80.sh plumbing --lanes 10` (about 1 h, must end PASSED), then `run_near80.sh run --lanes 10` (continues
   after any restart; one Ctrl-C is a safe stop); `run_near80.sh status` from a second window.
4. **After "Finished":** commit `near80/results-<utc>` as yourself with `git add -f` (the runner prints the
   commands; `-f` because `.gitignore`'s `data/` line hides the data records) and run `push_runfiles_near80.sh`,
   exactly as version 8 (`process.md` 7.1).


No agent types an approval, and no message counts as one.

---

## 5. Appendix: draft pre-registration text for version 9 (near-80 style)

> This is a draft in the form of version 8's `PREREGISTRATION.md`, to be completed by the build (section 4 of this
> appendix needs the frozen code's numbers) and frozen by `freeze.py`. Where a section says "as version 8", version
> 8's text applies unchanged and is copied into the final document.

# Step 2 confirmed with pooled evidence: pre-registration (version 9, near-80 main)

## 0. Status, and what this was written after

**Seeds.**
- Seeds 0-4 are seen. Everything on them is exploratory.
- Seeds 40-49 are fresh. Only they can carry a claim. Nobody has generated, loaded or looked at them. Seeds 50-59 are
  reserved for a later version and are refused by this one.
- Seeds 10-14 are spent (version 8). Seeds 5-9 belong to version 6/7. Seeds 15-19 stay named for version 8's fallback;
  20-34 belong to the placement design. This version never touches them.

**Written after:**
- version 8's run and its independent check (20 October 2026): claim 4 failed by its pre-set rule because BCA's
  certificate refused on fresh seeds 10, 12 and 13 (held-out full-throttle groups of 19, 22 and 18 rows); claims 1-3
  did not run (the exploratory stop);
- five analyses of the records (verdict, refusal, credit, process, steelman) and a preview of the pooled groups on
  the spent seeds 0-4 and 10-14 (counts exact on all 20 seed-worlds; heights on 10-14 from the recorded second
  critics). Nothing was trained;
- your answer to section 16 ([date, time], "[words]").

**Not seen by anyone:** any number from seeds 40-49.

**Redrafts before the freeze:** [number] (the kill rule of section 8.1 on the spent seeds 10-14).

## 1. Words used

As version 8, plus: **the pooled groups** (section 3); **a refusal** (BCA's switch on with an infinite height);
**the plain twins** (section 5.3); **ahead / tie / behind** (section 9.3); **the noise floor** (section 9.4).

## 2. The short version

- **What changes:** BCA's switch reads every held-out recorded row outside the ball (the calibration and evaluation
  splits), not the 1,000-row bank; its full-throttle group grows from 18-60 rows to about 35-135, past the 29-row
  floor of its own certificate on every seed seen so far. A refusal fails BCA's arms on that seed, never the seed.
- **What is tested:** step 2's three marks on ten fresh seeds (C1); whether BCA's switched ramp beats each of three
  plain twins trained beside it by 5 points (C2): the mean-difference switch with the plain 90% gap, with twice that
  gap, and turning on the hand-set range limit; and, the mirror, whether a twin beats BCA by 5 points (C3). C2 and C3
  are read on the seeds where BCA set a height.
- **What does not change:** TD3+BC in step 2's actor-only protocol; the gate's recorder, sizes and physics; the ball;
  the four fixed test start sets; the matching rules; the approval, identity and manifest rules.
- **Cost:** about 18 CPU-hours in all (about 1.65 per seed plus about 1.5 of plumbing), about 5-5.5 hours of wall time
  on your Framework Desktop in one sitting, plus about an hour of plumbing.

## 3. The evidence change, exactly

### 3.1 The rule

Step 1's gap rule, unchanged, fixes `t_lo` and `t_up` on the setting's kept training commands (`step1lib.gap_rule`,
upper quantile 0.99, least gap 0.05, at least 9 rows). BCA's switch reads every recorded row that the learners do not
train on: the seed's calibration split (20,000 rows, the bank included) and evaluation split (30,000 rows), keeping
the rows whose command the ball would keep. The **full-throttle group** is those rows with throttle index above
`t_lo`, together with the survivors (the kept training rows above `t_lo`, which the second critic does not train on).
The **rest** is those rows with throttle index at or below `t_lo`. The second critic scores both groups exactly as it
scored the bank in version 8 (one-sided overestimates, the stall-aware controller at the next state, gamma 0.98).
BCA's certificate (alpha 0.1, credibility 0.95, 1,000 draws, margin 0.10, the same tags), the split-conformal switch,
the rank test and the mean-difference switch all read these two groups and no others.

### 3.2 Counted before training

Both groups' sizes depend on commands only. The data phase writes them in the seed's data record, by source
(survivors, calibration rows, evaluation rows), with the exact chance that a group of that size gives a finite height
(the chance that at most 50 of 1,000 draws are infinite, each infinite with chance 0.9 to the power of the size: 0% up
to 24 rows, 70% at 29, 90% at 30, 98% at 31, 100% from 33), before any learner of that seed trains. No seed is
skipped or replaced on its count. The look prints the count per seed.

### 3.3 Where it applies

To the switch arms only. The training rows, the ball, the second critic's training set, the normalisation, the test
start sets and every rival's rule are unchanged. No scored number reads a held-out row.

### 3.4 What it does on the seen seeds (preview on spent seeds; exploratory)

Pooled full-throttle groups: seeds 0-4 stalling 70, 106, 121, 100, 110 (version 8: 35, 47, 59, 43, 47); seeds 10-14
stalling 35, 109, 46, 38, 58 (version 8: 19, 53, 22, 18, 30); no-stall 37-135. Rest 41,476-42,997 (version 8:
825-878). No refusal on any of the 20 seed-worlds. Heights on seeds 10-14, stalling world: 2.78, 1.67, 1.85, 1.85, 2.26
(version 8: infinite, 2.29, infinite, infinite, 4.60). BCA's, the mean-difference and the split-conformal calls are
unchanged on all 10 fresh seed-worlds; the rank test adds one wrong "on" (seed 11, no-stall). The pooled switch, heights
and calls on the spent seeds 10-14 are recomputed by the frozen code in the pinned environment from version 8's
recorded second critics before approval (section 8.1) and pinned here: [to be filled]. Seeds 0-4 have no second
critic on the desktop (step 1 ran on another machine), so only their pooled counts are pinned (choice 11 offers the
retrain).

### 3.5 A second held-out log (choice 3, default no)

If chosen: one more log per seed and world from the same recorder (500, 200 and 300 episodes), stream tag [declared
here], never trained on, read whole by the switch arms by the rule of 3.1; a leak check that no row of it enters any
training set. It is left out by default.

## 4. The facts on seeds 0-4, before any training

[Filled by `facts_near80.py` with the frozen code: the pooled counts and sizes by source on seeds 0-4, both worlds,
and the gate's checks; on the spent seeds 10-14, both worlds, from version 8's recorded second critics and data
records: BCA's pooled certificate (switch, height, 5% bound), the plain 90% gap and twice it, the mean-difference,
split-conformal and rank-test statistics and calls, each height's rank among the group's scores. Nothing trained.]
Kill rule (section 8.1): BCA's and the mean-difference switch's pooled calls on seeds 10-14 equal version 8's
recorded calls (on in 5 of 5 stalling, off in 5 of 5 no-stall seed-worlds); otherwise this design is redrafted and no
approval is asked. Redrafts so far: [number].

## 5. Arms

### 5.1 The host

As version 8 (TD3+BC, actor-only, step 2's protocol; alpha 5.0; 20,000 actor steps in chunks of 4,000; critics of
100,000 updates).

### 5.2 Main setting, both worlds: step 2's arm set

As version 8 section 5.4 for seeds 10-14, restated for seeds 40-49: `none`, `clone`, `full_none` (stalling world),
`w0_check`, `obca_k1`, `obca_k2`, `obca_forced`, `fixed_gated`, `conformal_k1`, `meangap_k1`, `even_pull`,
`push_less`, `random_1`, `random_2`, `standin`, `range_hinge`, `perfect_band`.

### 5.3 New arms

- **`meandiff_q90x1`, the mean-difference switch with the plain 90%-quantile gap (twin 1).** Switch: on when the
  full-throttle group's mean score exceeds the rest's by more than 0.10. Height: the plain 90%-quantile gap of the
  same two groups. Trained alone, in its own call, after step 2's programs.
- **`meandiff_q90x2`, the same switch with twice that gap (twin 2).** Trained alone.
- **`meandiff_limit`, the mean-difference switch turning on the hand-set range limit (twin 3).** The same switch;
  when on, the hand-set range limit's own program (the room's hinge at the recorded range, coefficient 1.0); when
  off, no caution (a height-0 ramp checked equal to the no-caution actor).
- **`meandiff_k1`** (reported only): the same switch with the mean gap as its height (version 8's arm). Trained alone.
- **`conformal_k1`** (reported only, step 2's arm): the split-conformal switch; its no-stall false alarms are counted.
- **`obca_k1_alone`** (validity, and C2's and C3's BCA side): `obca_k1`'s switch and height trained alone in a
  program of one.
- **The twins for C2 and C3** are `meandiff_q90x1`, `meandiff_q90x2` and `meandiff_limit`. The order of the lone
  calls is fixed: `meandiff_k1`, `meandiff_q90x1`, `meandiff_q90x2`, `obca_k1_alone`, then `meandiff_limit`.

### 5.4 The refusal rule

`step2lib.arm_table` refuses a switched-on arm whose height is not finite. Version 8 let that refusal stop the whole
actor phase. This version reads it per arm: an arm whose switch is on with a height that is not finite is **refused**
(recorded, never trained); every other arm trains and is scored. The refused arms can only be `obca_k1`, `obca_k2`,
`obca_forced` (which takes the stalling world's height in both worlds) and `obca_k1_alone`; a refusal of any other arm
is a plumbing stop. The seed is never replaced.

**In C1's marks**, a refused arm's numbers are **the no-caution actor's** in that seed-world, and the controls matched
to BCA's strength are matched to zero strength (also the no-caution actor's numbers). Nothing is "not available",
so no mark fails by a missing number; the refusal's cost enters the pooled means as it would enter a deployment.

**In C2 and C3**, a seed where BCA refused in either world is left out: the twin claims are read on the certified
seeds (at least 8), with the sign thresholds of section 9.3 for their number. The refusal is named in C2's sentence.
A refused seed enters no twin difference, because the no-caution actor against a twin would be a difference of about
-100 and would decide the claim by itself.

### 5.5 Fixed references and composites

As version 8 section 5.5. Reported composites from trained arms: BCA's switch turning on the hand-set range limit;
"refusal means the range limit" (BCA's switched ramp where it set a height, the range limit where it refused).

### 5.6 Left out, and why

The near-80 and random-rows settings (version 8's exploratory stop found the band changes nothing for a range
limit); version 6's BCA (nets 0.13 against no caution in version 8); a fallback height or a lower credibility (they
change BCA); a hard clip as the reading of an infinite height (it would reward refusing).

## 6. Situations

As version 8 (recorded conditions, strong current, big waves, big kicks; four fixed test start sets of 1,000 starts;
strong current scored).

## 7. Measures

As version 8 (points, NET across worlds, the readouts held to 70% and capped at 80%), plus per seed-world: both
groups' sizes by source, the certification chance, each switch's statistic and call, each ramp arm's height and its
rank among the full-throttle group's scores.

## 8. The tests, in order (kill rules first)

### 8.1 The no-training checks (before approval, seeds 0-4; on each fresh seed when its data are built)

| # | check | pass |
|---|---|---|
| 1 | the gate's five world checks on the ball; the ball's leak checks | all pass |
| 2 | the training file equals `dataset.d4rl_dict` of the kept rows (six arrays, SHA-256) | equal |
| 3 | the pooled groups: every held-out row outside the ball appears exactly once; no training row other than a survivor; no ball row; the outcome-free count equals the sizes the switch scores | all pass |
| 4 | BCA's gap rule is defined; `t_lo` in [0.78, 0.80] (seen 0.782-0.791) | defined, in range |
| 5 | (seeds 0-4 only, before approval) the pooled counts on seeds 0-4; on the spent seeds 10-14, from version 8's recorded second critics, the pooled switch makes version 8's recorded calls (BCA's and the mean-difference switch's) | on 5 of 5 stalling, off 5 of 5 no-stall, for both switches |

A fresh seed that fails check 1 or 2 is a failed seed: it fails every mark and is never replaced. A failure of check
3 or 4 on a fresh seed is a plumbing stop of that seed-world: every mark that needs it fails by the pre-set rule, the
look says that a stop and not a return decided it, and the marks on the seeds that ran are reported as description.
Check 5 failing on seeds 10-14 means a redraft and no approval; the number of redrafts is written in section 0.

### 8.2 Plumbing, after approval (scratch only)

As version 8: the unit tests, a 40-step smoke of every phase kind, and the reproduction check on seed 0 in both
worlds, bit for bit: the gate's critic, cloned actor and no-caution actor; version 8's switch computed with the old
groups (survivors and bank) against step 1's record; step 2's saved actors for the plumbing check, BCA's switched ramp
and the hand-set range limit. Only a reproduction difference is final.

### 8.3 No exploratory stage

Seeds 0-4 are read only by the no-training checks of 8.1 and the reproduction check of 8.2. There is no harm stop
(the main setting's harm was 98.9 points on seeds 0-4) and no outcome-based stop before the fresh seeds.

### 8.4 The confirmatory stage: seeds 40-49

The main setting's data for all 20 seed-worlds (the checks of 8.1 and the outcome-free count); the critics, the
full-log critic and BCA's second critic; the switches on the pooled groups; the cloned and reference actors; the
actor phases with matching and the refusal rule, each seed-world locked before any score is computed; the
evaluations; the look. **Nothing on seeds 40-49 stops early on an outcome.** Plumbing stops only; a stopped phase is
written once and never rerun.

## 9. Pass marks, with reasons and power

Each mark is pooled: the mean of each seed's paired difference. Stalling world, strong current, unless stated. C1 is
pooled over all ten seeds; a refused seed-world enters it with the no-caution actor's numbers for the refused arms
(5.4). C2 and C3 are pooled over the certified seeds (BCA's height finite in both worlds), at least 8 of them. A failed
seed (8.1) fails every mark; a stopped phase fails every mark that needs it, and its sentence says so.

### 9.1 Validity

- **For C2 and C3:** the noise floor: `obca_k1` minus `obca_k1_alone`, pooled over the certified seeds, within 5
  points either way. If not, C2 and C3 are dropped and reported only (their 5-point margin would sit inside the
  training noise). C2 and C3 read `obca_k1_alone` as BCA's side, the same training mode as the twins; the differences
  against the batch-trained `obca_k1` are reported beside them. At least 8 certified seeds; otherwise C2 and C3 are
  not read. C1 has no validity condition beyond the checks of 8.1.

### 9.2 C1: step 2 confirmed in the main setting

| mark | pass | reason | on the main setting, seeds 0-4 (exploratory) |
|---|---|---|---|
| net lead | BCA's net across worlds is at least the best world-blind rival's plus 10; the world-blind rivals are the matched range limit, the matched even pull, the two random surfaces, the always-on ramp and the hand-set range limit | step 2's margin | +25.3 over the always-on ramp (standard error 2.3); the hand-set range limit nets 76.5 |
| keeps the gain | BCA minus no caution is at least -3 in recorded conditions, big waves and big kicks | version 6's margin | +8.0, +21.2, +19.8 |
| beats the even pull | BCA minus the matched even pull is at least 15 | version 6's margin | +100.8 |

- Under the refusal rule with ten seeds the pooled lead is about 28.5 with no refusal, 18.0 with one and 7.5 with
  two by the planning formula; resampling the seven recorded main-setting seeds gives 26 (sd 1.3), 19 (sd 2.5), 8 (sd
  2.4) and -2 with three. One refusal is expected to be survived, two to fail the lead mark (about 1 chance in 4 of
  passing). This is an expectation; the mark is read from the numbers. Chance of no refusal 0.65, of at most one
  0.94 (the gate's 30 logs, leak factor 2.0; 0.55-0.77 and 0.89-0.97 across 1.8-2.3).
- The marks on the certified seeds alone are reported beside C1, description only.
- Prediction: passes with 0 or 1 refusal; the heights shrink to about 0.5-0.75 of version 8's, so BCA's own
  stalling-world points may fall toward the plain switch's; the lead mark is about the switch, not the height, and
  is expected to hold.

### 9.3 C2 and C3: BCA's switched ramp against each plain twin

For each twin T in {the mean-difference switch with the plain 90% gap, with twice it, turning on the hand-set range
limit}: d = NET(BCA's switched ramp trained alone) minus NET(T), per certified seed; n certified seeds.

| reading | rule | n = 10 / 9 / 8 |
|---|---|---|
| BCA ahead of T (C2's part) | mean of d at least 5.0, and d positive in at least ceil(0.8 n) seeds | 8 / 8 / 7 (coin chance 0.055 / 0.020 / 0.035) |
| T ahead of BCA (C3 for T) | mean of d at most -5.0, and d positive in at most 1 seed | 1 / 1 / 1 (coin chance 0.011 / 0.020 / 0.035) |
| tie | otherwise; written "not shown to beat T by 5 points" | |

**C2 holds when BCA is ahead of all three twins** (one claim; `all_three`). **C3 holds for a twin when that twin is
ahead of BCA** (one claim per twin; the three form a family whose chance of any false "ahead" under a true tie is
0.001 / 0.014 / 0.023 / 0.028 at per-seed spreads 5 / 7.5 / 10 / 15). The three readings are reported in every
case, with the one-sided 95% lower bound of each mean (reported, not used) and the same differences against the
batch-trained `obca_k1`.

- Why 5: the project's smallest mark unit (version 8's claims 2 and 3, step 2's wording rule, round 1's bar), about
  one per-seed standard deviation of the paired differences within the main setting (1.9-5.2).
- Why ceil(0.8 n) and at most 1: the sign parts at about the 5% and 1-3% levels; C3's is stricter because three twins
  give three chances.
- Power (ten certified seeds, simulated, `c3_rule.out`): a true BCA edge of 10 points is read "ahead" with chance
  1.00 (spread 5) or 0.79 (spread 10); 5 points: 0.48 (0.31 at spread 10); 3 points: 0.10; 0: 0.001 (0.028 at spread
  10). A true twin edge of 10 is read "T ahead" with 0.98 (spread 5) or 0.51 (spread 10); the composite's recorded
  4.5-point edge (sd 5.2) with 0.27, "tie" 0.73; 3.2 points with 0.09. The conjunction C2: three true 5-point edges
  0.23 (spread 5) or 0.09 (spread 10); three true 10-point edges 0.99 or 0.56.
- Prediction: C2 fails and C3 does not fire; "ahead" or "tie" against the plain gap, "tie" against twice it,
  "tie" against the gated range limit.

### 9.4 Rules about the marks

- No mark moves after its numbers exist.
- C1, C2 and C3 are read separately. None rescues another (version 8's rule 9.6).
- Seeds 0-4 never carry a claim.

## 10. Confirmation on fresh seeds 40-49

- **Data:** new logs of both worlds with the gate's recorder, sizes and settings (stall slope 1.5), as version 8's
  amendment 1 builds them. Then the ball, its leak checks, the gate's five world checks, the pooled count.
- **A fresh seed that fails** check 1 or 2 fails every mark and is never replaced.
- **Learners:** every one made fresh: critics, the full-log critic, BCA's second critics, cloned actors, the reference
  actors, every arm of section 5.
- **Frozen from this pre-registration:** the pooled rule, the refusal rule, every arm, ladder, matching rule and
  mark, the seeds, the sentences of section 14.
- **The confirmatory look:** the validity condition of 9.1, then C1, then C2 and C3; the code fills in section 14.

## 11. Approval

As version 8 (the owner's own terminal on the desktop, `approve_local_near80.py --check` then without `--check`,
the commit under the owner's identity, the session's import, every phase re-checking the chain). The folder is
`outputs/shift/beyond-the-data/near80/`: version 9 supersedes version 8 in place (choice 7); version 8's `prereg.json`
moves to `prereg_history/`, its approval, import, amendment and results files stay, and its chain is checkable at
commit `52914e4`. The branch is `claude/main-bca-connection-x4o46s`.

## 12. Order of work

1. The draft and the code that trains nothing (this document, the constants, the pooled groups, the refusal rule, the
   new arms, the unit tests; `BUILD_LIST.md`).
2. Your answers to section 16.
3. The final version: the facts with the frozen code (the counts on seeds 0-4; the pooled switch on the spent seeds
   10-14 from version 8's recorded second critics, with the kill rule; about 10-15 CPU-minutes, nothing trained), the
   independent re-check, the freeze (inside WSL, in the pinned environment), the fingerprint.
4. Your approval; the session's import.
5. Plumbing (about 1 hour).
6. The confirmatory stage on seeds 40-49 (about 5-5.5 hours of wall time).
7. Results committed with their fingerprints (`git add -f`); run files to `claude/near80-v9-run-files`; the
   independent check.

## 13. Compute

On the Framework Desktop in Ubuntu under WSL, on the processor alone, one processor per phase.

| part | CPU-hours |
|---|---|
| the facts before approval | about 0.2 (nothing trained) |
| plumbing | about 1.3-1.5 (version 8: 1.23; the lone programs added to the reproduction) |
| confirmatory, seeds 40-49, main setting, both worlds | about 16.5-17 (version 8's ledger: 1.52 per seed; the five programs run alone add about 0.14 per seed) |
| **total** | **about 18** |

Memory: version 7's rule, unchanged (choice 5): at most 2 actor phases at once, so about 5-5.5 hours of wall time with
10 lanes in one sitting (ten stalling-world actor phases of about 47 min, ten no-stall of about 13 min, a lead-in of
about 20 min). Budget guard as version 8 with a forecast of 2.2 CPU-hours per seed (22 for ten; a pause at twice
it) and a cap of 2.5 times the forecast, 55 CPU-hours.

## 14. The claim, word for word, for each outcome

The code fills in the brackets. Numbers are rounded to one decimal. Every bracket is pooled over the fresh seeds, in
strong current, stalling world, main setting, unless stated. The labels of this section are the anchors
`claims_text.extract_section14` reads (`BUILD_LIST.md` section 5 lists the anchors to add for the new sentences).

| bracket | meaning |
|---|---|
| S, N | seeds where BCA's switch was on (stalling world), off (no-stall world) |
| X, H | BCA minus no caution; the full-log actor minus no caution (the harm) |
| W | no-stall world: no caution minus BCA |
| NET, D, R | BCA's net across worlds; its lead over the best world-blind rival; that rival's name |
| E, L | BCA minus the matched even pull; BCA's largest loss to no caution elsewhere |
| K, F | seeds where BCA set a finite height in both worlds (the certified seeds); seeds where it refused |
| NC | the number of certified seeds C2 and C3 read |
| DC, RC, LC, EC | the lead, the best rival, the largest loss and the even-pull margin on the certified seeds alone (description) |
| D1, K1, LB1 | BCA's switched ramp trained alone minus the mean-difference switch with the plain 90% gap, net across worlds, pooled over the certified seeds; seeds where it was positive; its one-sided 95% lower bound |
| D2, K2, LB2 | the same against the mean-difference switch with twice the plain 90% gap |
| D3, K3, LB3 | the same against the mean-difference switch with the hand-set range limit |
| DT, KT, LBT | the same for the twin a clause names |
| KMIN | ceil(0.8 NC): the seeds C2's sign part requires |
| A | seed-worlds, of 20, where the mean-difference switch made BCA's on/off call |
| P | no caution's share of steps past 80% |
| V | the noise floor: the size of BCA's switched ramp minus the same ramp trained alone, pooled |
| NSEEDS, SEEDS | the number of fresh seeds and their range |
| NMIN, NMAX, NREACH, NSW | the pooled full-throttle group's smallest and largest size over the seed-worlds, how many of them reached 29 rows, and how many seed-worlds there are |
| N8MIN, N8MAX | version 8's groups' smallest and largest size over the same seed-worlds (for scale) |

**Always attached:** "One simulated task, two worlds, [NSEEDS] fresh seeds ([SEEDS]), four fixed test start sets of
1,000 starts each. BCA's two groups were pooled over every held-out recorded row outside the removal ball, and every
switch read the same groups. In strong current the test states lie outside the recorded currents."

**On the plumbing's seed 0,** every claim sentence below (sections 14.1-14.2) starts with "Plumbing, seed 0, a check
of the code; not a claim:".

### 14.1 C1: version 8's claim 4 marks

**All three marks pass:**
> "On fresh seeds [SEEDS], in the main setting (only the densest 15% of commands removed) with BCA's two groups pooled
> over every held-out row outside the ball, step 2's result held: BCA's switched ramp met all three pre-set marks.
> Its switch turned caution on in [S] of [NSEEDS] seeds where full throttle stalls and off in [N] of [NSEEDS] where it
> does not. In strong current it gained [X] points over no caution where full throttle stalls (the full-log controller
> gained [H]) and lost [W] where full throttle helps: a net of [NET], [D] more than the best rule that cannot switch
> itself off ([R]). It stayed within 3 points of no caution or better in the other three situations, and beat the
> matched even pull by [E]. It set a height on [K] of the [NSEEDS] seeds[; on [F] it refused: [refusal clauses]].
> Where full throttle stalls and its switch was on, its controller is the same as the always-on ramp's, so that gain
> comes from where the ramp starts and from its height, not from the switch."

**Any mark fails:**
> "On fresh seeds [SEEDS], in the main setting with BCA's two groups pooled over every held-out row outside the ball,
> step 2's result did not hold: BCA's switched ramp did not meet its pre-set marks: [one clause per missed mark].
> [last sentence]"

The clauses:
- "its net across worlds of [NET] was [D] above the best rule that cannot switch itself off ([R]), below the 10
  required"
- "it lost [L] points to no caution in [situation], more than the 3 allowed"
- "it beat the matched even pull by [E] in strong current, below the 15 required"
- "seed [s] failed [check] and, by the pre-set rule, counts as failing every mark"
- "on seed [s] BCA could not set its height from [n] held-out full-throttle rows ([world]; chance of a finite height
  [p]%), so its switched ramp, the ramp at twice its height, the always-on ramp and its noise-floor copy count there
  as the controller trained without caution, and the rivals matched to its strength as matched to none; every other
  arm ran"
- "seed [s]'s [phase] stopped ([reason]) and, by the pre-set rule, every mark that needs it fails"

The last sentence:
- "Step 2's exploratory result was not reproduced on these seeds." (when a mark was missed on its numbers)
- "A stop, not a return, decided this: no reading of BCA's return is claimed, and the marks on the seeds that ran are
  reported as description." (when every missed mark is told by a stop clause or a failed-seed clause alone)

**Always after C1, the certified seeds:**
> "Description only: on the [NC] seed[s] where BCA set a height in both worlds, the three marks read [pass / fail]
> (lead [DC] over [RC], largest loss [LC], even-pull margin [EC])."

**Validity fails:**
> "On fresh seeds [SEEDS], the main setting did not produce the harm it was built to test ([reason]). It supports no
> conclusion about BCA either way."

The reasons: "the controller trained without caution lost only [H] points to the full-log controller (at least 30
needed)", or "the controller trained without caution went past 80% on only [P]% of strong-current steps (at least 25%
needed)".

### 14.2 C2 and C3: BCA's switched ramp against each plain twin

**Holds (every twin beaten):**
> "On the [NC] fresh seeds where BCA set its height, BCA's switched ramp beat each plain twin that read the same two
> groups and made its call by the same mean difference, paired by seed on the net across worlds: the plain 90% gap by
> [D1] points (positive in [K1] of [NC] seeds), twice that gap by [D2] ([K2] of [NC]) and the hand-set range limit by
> [D3] ([K3] of [NC]); at least 5.0 points and [KMIN] of [NC] seeds were required for each. Something in BCA's height,
> beyond a plain switch and a plain height on either side of it, earned return here. The plain mean-difference switch
> made BCA's on/off call in [A] of 20 seed-worlds, so nothing is credited to the switch."

**Fails:**
> "On the [NC] fresh seeds where BCA set its height, BCA's switched ramp did not beat every plain twin by the pre-set
> margin: [one clause per twin]. [last sentence]"

The clauses:
- "it was ahead of [twin] by [DT] points (positive in [KT] of [NC] seeds)"
- "it was not shown to beat [twin] by 5 points (mean [DT], one-sided 95% lower bound [LBT], positive in [KT] of [NC]
  seeds)"
- "[twin] was ahead of it by [DT] points (positive in only [KT] of [NC] seeds)"
- "seed [s] was left out of this reading because BCA refused there (C1 carries its cost)"

The last sentence:
- "Its gain over the plain height was matched by twice that height and by the hand-set range limit: a dose, not a
  calibration. Nothing is credited to BCA's Bayesian step or to its height." (BCA ahead of the plain gap and tied
  with the other two)
- "No difference of 5 points or more from a plain switch with a plain height, of either size, or from the hand-set
  range limit was shown. No credit beyond a plain rule is shown." (tied with all three)
- "A rule with no BCA in it, [twin], did better by the pre-set margin (C3)." (any twin ahead)
- "No credit beyond a plain rule is shown." (otherwise)

**A twin ahead (C3 holds for it):**
> "On the [NC] fresh seeds where BCA set its height, [twin] beat BCA's switched ramp by [DT] points on the net
> across worlds, paired by seed (BCA ahead in only [KT] of [NC] seeds; at most 1 was allowed and at least 5.0 points
> were required). A rule with no BCA in it did better by the pre-set margin."

**No twin ahead (C3 fails):**
> "No plain twin beat BCA's switched ramp by the pre-set margin."

**Each separately, a twin beaten:**
> "(Not used: the multiplicity rule is `all_three`.)"

**Each separately, a twin not beaten:**
> "(Not used: the multiplicity rule is `all_three`.)"

**Dropped by the noise floor:**
> "C2 and C3 were not read: BCA's switched ramp trained alone differed from its batch-trained copy by [V] points (at
> most 5 allowed), so the 5-point margin sits inside the training noise. The differences from the three twins are
> reported ([D1], [D2], [D3]), not claimed."

**Not read (too few certified seeds):**
> "C2 and C3 were not read: BCA set its height on only [NC] seeds (at least 8 needed). The differences from the
> three twins are reported ([D1], [D2], [D3]), not claimed."

**Margins not fixed:**
> "(Cannot occur: the margins are fixed in this document; kept for the extractor.)"

**If a seed failed:**
> "By the pre-set rule, C2 and C3 fail because seed [seeds] failed (C1 says why); the numbers of the other seeds are
> reported as description only."

**If every seed failed:**
> "By the pre-set rule, C2 and C3 fail because seed [seeds] failed (C1 says why)."

### 14.3 The pooled groups (reported)

**Always, for the pooled groups:**
> "Before any training, the pooled full-throttle group had [NMIN] to [NMAX] rows over the [NSW] seed-worlds, and
> [NREACH] of them reached 29 rows, the smallest size BCA's certificate can certify; version 8's groups would have had
> [N8MIN] to [N8MAX] rows."

### 14.4 Stops

- **Reproduction fails:** "Stopped: the new code did not reproduce [the main test's removal / critic / cloned actor /
  no-caution actor, or step 2's plumbing-check / switched-ramp / hand-set range-limit actor] on seed 0. No result of
  this version stands until it does."
- **BCA refuses on a seed-world:** "Seed [s], [world]: BCA could not set its height from [n] held-out full-throttle
  rows (chance of a finite height [p]%). In C1 its switched ramp, the ramp at twice its height, the always-on ramp
  and its noise-floor copy count there as the controller trained without caution; the seed is left out of C2 and C3;
  every other arm ran."
- **A phase stops:** "Seed [s], [world]: [phase] stopped ([reason]). It was not rerun. By the pre-set rule every mark
  that needs it fails; this is a stop, not a reading of BCA's return."

### 14.5 How the code fills in the sentences

As version 8 section 14.6 and the build's section 14.5, with: [twin] written "the mean-difference switch with the
plain 90% gap", "the mean-difference switch with twice the plain 90% gap", "the mean-difference switch with the
hand-set range limit"; [R] as version 8, plus "the hand-set range limit"; [p] with one decimal as a percentage; [A]
a whole number of 20; [KMIN] = ceil(0.8 NC); [LBT] the one-sided 95% lower bound of the mean of d (Student t on the
certified seeds); C2's and C3's clauses in the twins' fixed order, then the refused seeds' clauses in seed order;
C1's last sentence is the first form when any clause of the net lead, the keeps or the even pull appears, otherwise
the second; C2's last sentence by the four cases in that order; "Not read" takes precedence over "Dropped", both
over "Holds" / "Fails"; C3's sentence is written once per twin ahead, else the "No twin ahead" sentence once;
the certified-seed sentence is written whenever C1 is written on fresh seeds.

## 15. Departures from version 8, fixed now

1. **BCA's groups are pooled** over every held-out row outside the ball (section 3). Version 8 read the survivors and
   the 1,000-row bank.
2. **A refusal is read per arm**: in C1 as no caution (5.4); in C2 and C3 the seed is left out. Version 8 stopped the
   whole actor phase and both evaluations, and the seed failed every mark.
3. **Main setting only; no exploratory stage; no harm stop.**
4. **Ten fresh seeds, 40-49.**
5. **Four new trained arms** (`meandiff_q90x1`, `meandiff_q90x2`, `meandiff_limit`, `obca_k1_alone`) and two claims
   about them (C2 and its mirror C3), with step 2's 5-point unit and sign counts. Version 8 reported the switched
   rivals with no claim (its departure 10); the reason given there, that facts about each arm's switch and height
   were unknown in the near-80 setting, does not apply in the main setting, where they are recorded on seven seeds.
   The split-conformal switch is reported, not a twin: it differs from BCA in its switch as well as its height.
6. **The hand-set range limit joins the world-blind rivals** for the net-lead mark; version 8 listed it as
   "reported; used in claim 3". The marks themselves are unchanged. The lead is computed in version 9's own look
   over the enlarged set; step 2's lead over its own set is reported beside it.
7. **Both groups' sizes are counted and written before training**, with the certificate's floor stated (29 rows).
8. **Version 9 supersedes version 8 in place** (`near80/`), with a new fingerprint; version 8's `prereg.json` moves to
   `prereg_history/`, its approval, import, amendment and results files stay, and its chain is checkable at
   `52914e4`.
9. **C2's and C3's BCA side is the copy trained alone**, the twins' own training mode; the noise floor remains the
   validity check.

## 16. Choices

1. **Seed count and block.** Default: **ten seeds, 40-49**. Or five (40-44; one refusal then decides C1), fifteen or
   twenty (40-54 or 40-59; the cluster entry is then needed, about a day of work and a probe).
2. **What a refusal means for the marks.** Default: **in C1, BCA's arms count as no caution in that seed-world; in C2
   and C3 the seed is left out** (5.4). Or version 8's rule (every seed is needed; a refusal fails C1, C2 and C3),
   with a 0.35 chance under the default evidence that one refusal voids the test. Or, a method change: a declared
   fallback (the hand-set range limit, a hard clip at the data's edge) when BCA refuses, scored as that fallback;
   rejected here because it would reward refusing (the range limit scored 93.1 against BCA's 88.6 in the stalling
   world on seeds 0-4 and would net about 107); it is reported as the composite "refusal means the range limit".
3. **A second held-out log per seed and world.** Default: **no**. Yes brings the chance of any refusal in ten seeds
   from 0.35 to about 0.002 and moves BCA's height further toward the plain 90% point.
4. **The machine.** Default: **the desktop**. Or one Orion node (needs the machine table, the probe and a decision on
   the reproduction reference; `process.md` 4).
5. **The actor phase's memory reservation.** Default: **version 7's rule** (2 at once; about 5-5.5 hours of wall). Or
   1.25 times the measured peak (9 at once; about 2.5 hours; changes `TestMemoryRule`).
6. **Lanes.** Default: **10**.
7. **The folder.** Default: **version 9 in place in `near80/`**, as the build has it (version 8's `--check` then
   passes only at commit `52914e4`; its chain is recorded in that commit's report). Or a new folder `near80_v9/`
   (eight path sites to move and the history logic to drop; about half a day).
8. **The twins' factors.** Default: **the plain 90% gap and twice it** (step 2's k = 1 and k = 2). Or a hand-set
   height of 3.0 for the far twin (on the recorded plateau; a number chosen after seeing seeds 0-4, labelled as such).
9. **Where results are committed from.** Default: **the desktop**, as versions 6-8.
10. **A fresh start bank per seed.** Default: **no**. The four fixed test start sets (`bank_starts`, one stream
    `[20261012, 2, 9]` for every seed) were pre-registered in version 6 and are the same for every arm, so they
    rig nothing; but every design number was chosen on them. Yes adds one fresh bank of 1,000 strong-current starts
    per seed (stream `[20261012, 2, 9, 1000 + seed]`), scored for every arm as a description-only readout: about 1.5
    minutes of evaluation per seed-world and its anchors, plus code in `evaluate_phase.py`.
11. **Retrain step 1's second critics on seeds 0-4** to pin their pooled heights beside seeds 10-14's. Default: **no**
    (about 1 CPU-hour of exploratory training on seen seeds; the kill rule reads seeds 10-14, which need no training).

## 17. What could fool us, and what we expect

**Limits, with the guard against each:**
1. **The heights shrink under pooling** (0.49-0.73 of version 8's on seeds 11 and 14), so BCA's stalling-world points
   may fall. Guard: this is the test's purpose; both heights are reported on seeds 0-4; C1's lead mark is about the
   switch.
2. **A fresh seed with very few leaked rows** (the gate's 30 logs include one with 11; pooled, about 22) refuses even
   under pooling. Guard: the outcome-free count, the refusal rule, ten seeds; the chance is about 1 in 23 per seed (leak factor 2.0;
   1 in 17 to 1 in 38 across 1.8-2.3).
3. **The split-conformal switch is wrongly on in some no-stall seed-worlds** (0.121 and 0.116 on seeds 10 and 13
   under pooling; its margin is step 2's 0.10). Guard: it is reported, not a twin; the twins make BCA's call by the
   mean difference, so their readings are about heights.
4. **The rank test's false alarms grow with 42,000 rest rows** (one new on seed 11). Guard: reported only.
5. **The 2 x plain twin may sit past the dose plateau on some seeds** (as twice BCA's height did in the near-80
   setting). Guard: the ladder is reported per seed; the prediction says "tie or plain ahead".
6. **Training noise.** Guard: the noise floor (9.1) drops C2 if it exceeds the margin.
7. **Seeds 11 and 14 were a selected pair.** Guard: no number from them is a mark; the predictions use seeds 0-4's
   spreads and the preview.
8. **Strong current lies outside the recorded currents.** Guard: stated in every claim, as version 8.
9. **The kill rule is a check on spent seeds** (10-14) of BCA's and the mean-difference switch's calls only; it is
   not a fresh-seed test. Guard: the number of redrafts is written in section 0; the twins' calls are pinned beside.
10. **The 0.10 margin was placed in step 1 after seeing seeds 0-4** on BCA's own statistic and is carried to every
    switch arm unchanged. Guard: labelled; retuned nowhere; the switch-call table shows every disagreement.
11. **The four fixed start banks are the ones every design number was chosen on.** Guard: symmetric across arms and
    pre-registered since version 6; choice 10 offers a fresh bank per seed as a readout.

**Predictions, written before any training (not marks):** C1 passes with 0 or 1 refusal; C2 fails with BCA ahead of
or tied with the plain gap, tied with twice it, tied with the gated range limit; C3 does not fire; the pooled heights
on the fresh seeds lie between 1.5 and 3.0; the no-caution actor scores near -14 in strong current where full throttle
stalls; the mean-difference switch makes BCA's call in 19 or 20 of the 20 seed-worlds.

## 18. Files

As version 8's table, in `outputs/shift/beyond-the-data/near80/` (in place), with: `code/near80_constants.py`
(seeds 40-49, the pooled rule, the refusal rule, the twins, `MARKS_C2` 5.0 / ceil(0.8 n) / `all_three`, `MARKS_C3`
-5.0 / at most 1, `MIN_CERTIFIED` 8), `code/near80lib.py` (`pooled_groups`, `pooled_count_record`,
`certification_expectation` with the binomial chance), `code/learner_phases.py` (`switch_scores_pooled`,
`switch_scores_step1` for the reproduction check), `code/policy_phase.py` (the per-arm refusal, the new arms, the
programs run alone), `code/pooled.py` and `code/claims_text.py` (section 14), `code/facts_near80.py` (the counts on
seeds 0-4 and the spent-seed switch facts), `code/test_v9.py` and the retargeted tests, all listed with lines in
`BUILD_LIST.md`. The cluster scripts are not part of version 9.

---

### Files of this decision

- This document (and `DECISION_v1_before_reviews.md`, the text before the reviews), `power_v9.py` / `power_v9.out`,
  `c2_rule.py` / `c2_rule.out`, `c3_rule.py` / `c3_rule.out`, `ledger_v8_main.py`, `review_stats.md`,
  `review_fairness.md`, `review_feasibility.md`, `review_stats_calc.py` / `.out`, `REVISION_LOG.md`,
  `OWNER_SUMMARY.md`, `BUILD_LIST.md`, beside the five notes and their scripts, in
  `C:\Users\DAVIDB~1\AppData\Local\Temp\claude\C--Users-David-Bayha-BCA\9ab1f036-9335-42af-93a3-a9c39c11b8cc\scratchpad\decide\`.
- The preview this document cites:
  `C:\Users\DAVIDB~1\AppData\Local\Temp\claude\C--Users-David-Bayha-BCA\9ab1f036-9335-42af-93a3-a9c39c11b8cc\scratchpad\v9preview\PREVIEW.md`
  (and `preview_tables.md`, `preview.json`).
- The build this document fixes values for: branch `claude/near80-v9` of the `B:\` clone (HEAD `52914e4`; 26 files
  modified under `outputs/shift/beyond-the-data/near80/` and its `code/`, three untracked, one deleted; uncommitted at
  the time of the revision).
- Nothing under the repository, `runs\` or the owner's clones was written by this step.
