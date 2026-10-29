# The main setting on fresh seeds: pre-registration (version 9, near-80 main)

**DRAFT, not frozen, awaiting the owner. Nothing has been trained.**

No learner, critic or actor has been trained for this version, and nothing has been frozen. This version was built by
the session on 20 October 2026, after version 8's run on the Framework Desktop ended (19 October), after the check of
its results (`../near80_results_check/REPORT.md`), and after `DECISION.md` (the same day, revised after three
independent reviews) fixed every value below, and your answers to section 16 (20 October 2026, 18:29 UTC, and on
choice 9 at about 19:20 UTC) are recorded in sections 0 and 16. One item stays open until the freeze and carries a
bracketed mark the freeze step (`code/freeze.py`) refuses: the facts the frozen code computes before approval
(section 4.2, with the redraft count of section 0). The freeze also refuses while the status line above says DRAFT.
Version 8's files stay in this folder as history (section 18).

**The one question this answers:** on fresh seeds, in the main test's setting (the densest 15% of recorded commands
removed), does BCA's switched ramp meet step 2's three marks once its two calibration groups are pooled over every
held-out row, and does its height earn anything, by 5 points, that a plain switch with a plain height reading the
same groups would not?

---

## 0. Status, and what this was written after

**Seeds.**
- Seeds 0-4 are seen by everything. Version 9 uses seed 0 for its plumbing only (nothing on it is a claim).
- Seeds 40-49 are fresh. Only they can carry a claim. Nobody has generated, loaded or looked at them. Seeds 50-59 are
  reserved for a later version and are refused by this one.
- One stub run on 20 October 2026 (the cluster entry's smoke, `V9_BUILD_NOTES.md` section 4) ran on a synthetic
  fixture under the label seed 40 before the rule that every stub uses seed 0: it generated nothing from the
  generator's seed 40, read no log of it and trained nothing; every stub since uses seed 0. The same smoke read five
  pinned `runs/` folders of the desktop clone (the gate's data, the room harness, step 2's saved actors, version 6's
  run folder), none of which holds a fresh seed.
- Seeds 10-14 are spent (version 8's fresh seeds). This version reads them once, before approval, with version 8's
  recorded second critics (section 8.1, check 5; nothing trained). Seeds 5-9 belong to version 6/7; 15-19 stay named
  for version 8's fallback; 20-34 belong to the placement design; 100-119 and 900-909 to the gate and stage 0. This
  version never touches any of them.

**Written after:**
- version 8's run and its independent check (19-20 October 2026): claim 4 failed by its pre-set rule because BCA's
  certificate refused on fresh seeds 10, 12 and 13 (held-out full-throttle groups of 19, 22 and 18 rows against a
  floor of about 29), the approved code stopped the whole actor phase of those seeds, and claims 1-3 did not run (the
  band's stop rule on seeds 0-4); the switch's on/off call was right in 10 of 10 fresh seed-worlds;
- five analyses of the records (the verdict, the refusal, the credit, the process, the steelman) and a preview of the
  pooled groups on the spent seeds 0-4 and 10-14 (counts exact on all 20 seed-worlds; heights on 10-14 from the
  recorded second critics; nothing trained): the group is the removal ball's leakage; it reads no outcome; its only
  size guard was the gap rule's "at least 9 rows"; BCA's height at these sizes is the top-1 or top-2 held-out tail
  score; a plain mean-difference test with the same 0.10 margin makes BCA's on/off call in all 40 recorded
  seed-worlds and in 10 of 10 under pooling;
- `DECISION.md` of 20 October 2026, revised after a statistics review, a fairness review and a feasibility review
  (`REVISION_LOG.md`), which fixed the seeds, the claims, the margins and the refusal reading below; both are kept,
  unpinned, under `decision/` in this folder with the credit and power calculations and the preview they rest on
  (section 18), and its verdict table is copied word for word into section 14.6;
- your answer to section 16, given on 20 October 2026 at 18:29 UTC in the chat with the cloud assistant: "Accept the
  defaults and cluster", that is every default, with the university cluster as the run machine (choice 4) and the
  desktop kept as the fallback; and your second answer, at about 19:20 UTC in the same chat, on choice 9 (where
  results are committed from), which the first answer had been read as giving to the cluster too: "committed from
  desktop after copying over from cluster", that is the default, the desktop, after the results folder and the output
  root are copied from the cluster over your own SSH session and checked by SHA-256 on the desktop (12.2, step 4).
  Both are recorded, with their times and the place, in `OWNER_ANSWER` of `code/near80_constants.py`.

**Not seen by anyone:** any learner trained with the pooled groups, and any number from seeds 40-49.

**Redrafts before the freeze:** TODO(FREEZE: the number of redrafts after the kill rule of section 8.1 on the spent
seeds 10-14; 0 when the recomputed calls equal version 8's at the first try).

**So:** every mark below is fixed before any learner sees the fresh seeds. Nothing is tuned after the freeze.

---

## 1. Words used

- **Throttle**: the size of one thruster's command, from 0% to 100% (|a| from 0 to 1), whatever its sign.
- **The stall**: past 80% throttle a thruster's push falls while the throttle still costs energy. The stalling world
  has it; the no-stall world keeps the push rising.
- **The ball**: the main test's removal, the densest 15% of recorded commands (all at full throttle). It leaves the
  training rows and the calibration bank. Held-out rows are never touched.
- **The main setting**: the ball alone (version 6's main test, step 2's setting). Version 8's near-80 band and
  random-rows settings are not in this version.
- **The gap rule**: step 1's outcome-free rule on the ball-kept training commands: the largest gap among the top 1%
  of throttle indexes; `t_lo` is its lower end. A row is in the full-throttle group when its throttle index is above
  `t_lo`.
- **BCA's two groups**: the rows BCA scores with its second critic to decide its switch and height: the full-throttle
  group and the rest. Version 8's groups were the training-split survivors (ball-kept rows above `t_lo`) plus the
  calibration bank. Version 9 pools every held-out recorded row outside the ball into them (**the pooled groups**,
  section 3).
- **BCA's certificate**: 1,000 posterior draws per group; the switch is on when the 5% quantile of the gap between
  the groups' thresholds is above 0.10; the height is the difference of the deployed thresholds. A draw is infinite
  with probability 0.9 to the power of the group's size, and the group is certified only when at least 95% of the
  draws are finite: at least 29 rows (0.9^29 < 0.05), certain only from about 33-36 rows.
- **A refusal**: a certificate that is on with an infinite height (the full-throttle group was not certified). Step
  2's code refuses to build a switched-on arm from it; this version reads that refusal per arm (section 5.4).
- **The plain twins**: three plain switches that read the same two groups as BCA and make its on/off call by the mean
  difference of the groups' scores (the same 0.10 margin): with the plain 90%-quantile gap as the ramp's height, with
  twice that gap, and turning on the hand-set range limit (section 5.3).
- **The certified seeds**: the fresh seeds where BCA set a finite height in both worlds (neither refused nor failed).
- **Ahead, tie, behind**: the three readings of BCA against one twin (section 9.3).
- **The noise floor**: BCA's switched ramp trained in step 2's batch minus the same ramp trained alone (section 9.1).
- **Net across worlds**: step 2's measure, an arm's gain over no caution in the stalling world minus its loss to no
  caution in the no-stall world, strong current.
- **Points**: the exact 100-step return on a fixed test start set, scaled so that the recording controller scores 0
  and the world's top controller 100.

---

## 2. The short version

- **What changes against version 8:** one setting (the main setting), ten fresh seeds from an unused block, and BCA's
  two groups pooled over every held-out row outside the ball: its full-throttle group grows from 18-60 rows to about
  35-135, past the 29-row floor of its own certificate on every seed seen so far. A refusal fails BCA's arms on that
  seed, never the seed. Everything else of step 2's protocol stays: the critics, the cloned and no-caution actors,
  the switched ramp, the always-on ramp, the hand-set range limit, the matched rivals, the lock before any score.
- **What is tested:** step 2's three marks on ten fresh seeds (C1); whether BCA's switched ramp beats each of three
  plain twins trained beside it by 5 points (C2): the mean-difference switch with the plain 90% gap, with twice that
  gap, and turning on the hand-set range limit; and, the mirror, whether a twin beats BCA by 5 points (C3). C2 and C3
  are read on the seeds where BCA set a height.
- **Why pooling:** version 8's group had 18-53 rows on fresh seeds; a group below 29 rows cannot be certified. The
  calibration and evaluation splits hold about 50,000 recorded rows per seed-world, of which a few dozen are outside
  the ball and above `t_lo`. Pooling them adds those rows to the group and about 42,000 rows to the rest. It reads no
  outcome, touches no training row, and no scored test reads the pooled rows (the four fixed test start sets are
  rollouts).
- **What does not change:** TD3+BC in step 2's actor-only protocol; the gate's recorder, sizes and physics; the ball;
  the four fixed test start sets; the matching rules; BCA's certificate, its draws and its tags; the approval,
  identity and manifest rules.
- **Cost:** on the university cluster (your choice; section 12.2) about 24 CPU-hours in all (about 2.25 per seed on
  its Cascade Lake nodes plus about 1.5-2 of plumbing) and about 1.7 hours of wall time, every seed at once; on your
  Framework Desktop (the fallback) about 18 CPU-hours (about 1.65 per seed plus about 1.5 of plumbing) and about
  5-5.5 hours of wall time in one sitting; plus about an hour of plumbing on either (section 13).

---

## 3. The evidence change, exactly

### 3.1 The rule

The main test's ball, exactly as before (`shift.define_gap`, `shift.make_removal(log, 'gap', gap)`), bit for bit, in
both worlds. Step 1's gap rule, unchanged, fixes `t_lo` and `t_up` on the ball-kept training commands
(`step1lib.gap_rule`, upper quantile 0.99, least gap 0.05, at least 9 rows). BCA's switch reads every recorded row
that the learners do not train on: the seed's calibration split (20,000 rows, the bank included) and evaluation split
(30,000 rows), keeping the rows whose command is outside the ball (the ball's rows are hidden everywhere, as version 8
hid them from the bank). For one seed and world:

1. **The held-out rows** are every row of the calibration split and of the evaluation split whose command is outside
   the ball.
2. **The full-throttle group** is the training-split survivors (ball-kept training rows with throttle index above
   `t_lo`, which step 1 already scored and holds out from BCA's second critic) together with the held-out rows with
   throttle index above `t_lo`.
3. **The rest** is the held-out rows with throttle index at or below `t_lo`.

Version 8's groups (the survivors and the calibration bank) are a subset of these: the bank is part of the
calibration split. The second critic is unchanged: it trains on the ball-kept training rows minus the survivors, so
it has seen no full-throttle command, and every row it scores is held out from it. It scores both groups exactly as
it scored the bank in version 8 (one-sided overestimates, the stall-aware controller at the next state, gamma 0.98).
BCA's certificate (alpha 0.1, credibility 0.95, 1,000 draws, margin 0.10, the same tags), the split-conformal switch,
the rank test and the mean-difference switch all read these two groups and no others (`policy_phase.arm_row_from`
reads one switch record per world). The rest is pooled by the same rule: one rule for both groups. The code:
`near80lib.pooled_groups`, `learner_phases.switch_scores_pooled`.

### 3.2 Counted before training

Both groups' sizes depend on commands only. The data phase writes them in the seed's data record, by source (the
survivors, the calibration rows, the evaluation rows), with the exact chance that a group of that size gives a finite
height (the chance that at most 50 of 1,000 draws are infinite, each infinite with chance 0.9 to the power of the
size: 0% up to 24 rows, 40% at 28, 70% at 29, 90% at 30, 98% at 31, about 100% from 33), before any learner of that
seed trains (`near80lib.pooled_count_record`). No seed is skipped or replaced on its count. The look prints the
count per seed (section 14.3). The same phase records the pooled groups' leak checks (`near80lib.leak_checks`) and
stops the seed-world when one fails, or when the gap rule is undefined or its lower end lies outside [0.78, 0.80]
(`near80lib.gap_rule_problem`; section 8.1, checks 3 and 4).

### 3.3 Where it applies, and what it does not change

To the switch arms only. The training rows, the training file and its checks (`near80lib.training_file_check`); the
ball; the second critic's training set; the normalisation; the test start sets (nothing recorded there) and the
points; every rival's own rule (the hand-set range limit's recorded range, the matched rivals' ladders and matcher,
the always-on ramp's height, which is BCA's stalling-world height, the perfect band); step 2's programs (BCA's
switched ramp and the plumbing check train in step 2's own batches, section 5.5, so the plumbing's reproduction check
on seed 0 still reproduces step 2's saved actors bit for bit when given step 1's own switch entries). No scored number
reads a held-out row.

### 3.4 Why it is fair (five checks)

- It changes BCA's evidence, not BCA: alpha, credibility, draws, margin, the gap rule, the score and the second critic
  are unchanged; more rows of the same kind are read.
- Nothing scored reads these rows: points come from exact rollouts on the four fixed test start sets, and no arm
  trains on a held-out row (a leak check says so per seed, section 8.1 check 3).
- Every switch arm reads the same two groups, so no rival is disadvantaged; the mean-difference and split-conformal
  switches make the same calls under pooling as before on all 10 fresh seed-worlds (the preview).
- It is the natural rule ("every recorded row the learners do not train on"), not a number tuned to the failing
  seeds; the alternatives that would tune BCA (a lower credibility, a fallback height) are rejected for that reason.
- Its honest consequence is stated up front: with 35-135 rows BCA's height is a lower order statistic than before,
  so the heights shrink (0.49-0.73 of version 8's on seeds 11 and 14). If BCA's gain on seeds 0-4 came partly from an
  inflated dose, this test shows it.

### 3.5 What it does on the seen seeds (the preview on spent seeds; exploratory)

Pooled full-throttle groups: seeds 0-4 stalling 70, 106, 121, 100, 110 (version 8: 35, 47, 59, 43, 47); seeds 10-14
stalling 35, 109, 46, 38, 58 (version 8: 19, 53, 22, 18, 30); no-stall 37-135. Rest 41,476-42,997 (version 8:
825-878). No refusal on any of the 20 seed-worlds. Heights on seeds 10-14, stalling world: 2.78, 1.67, 1.85, 1.85, 2.26
(version 8: infinite, 2.29, infinite, infinite, 4.60). BCA's, the mean-difference and the split-conformal calls are
unchanged on all 10 fresh seed-worlds; the rank test adds one wrong "on" (seed 11, no-stall). The pooled switch,
heights and calls on the spent seeds 10-14 are recomputed by the frozen code in the pinned environment from version
8's recorded second critics before approval (section 8.1, check 5) and pinned in section 4. Seeds 0-4 have no second
critic on the desktop (step 1 ran on another machine), so only their pooled counts are pinned (choice 11 offers the
retrain).

### 3.6 A second held-out log (choice 3, default no)

If chosen: one more log per seed and world from the same recorder (500, 200 and 300 episodes), a declared stream,
never trained on, read whole by the switch arms by the rule of 3.1; a leak check that no row of it enters any
training set. It is left out by default.

---

## 4. The facts before any training

### 4.1 From version 8's run (fresh seeds 10-14, main setting, stalling world)

| seed | version 8's full-throttle group | finite share of the draws | certified | switch | height |
|---|---|---|---|---|---|
| 10 | 19 | 0.879 | no | on | infinite |
| 11 | 53 | 0.997 | yes | on | 2.289 |
| 12 | 22 | 0.903 | no | on | infinite |
| 13 | 18 | 0.85 | no | on | infinite |
| 14 | 30 | 0.964 | yes | on | 4.595 |

On seeds 0-4 the same group had 35, 47, 59, 43 and 47 rows, all certified. The switch's on/off call was right in 10
of 10 fresh seed-worlds (on in the stalling world, off in the no-stall world). Version 8's ledger: 1.53 and 1.51
CPU-hours for the two main-setting seeds that ran every phase (11 and 14).

### 4.2 The frozen code's facts (nothing trained; `pre_approval/pooled_facts_seeds0_4_spent10_14.json`)

Filled by `facts_near80.py` with the frozen code, in the pinned environment, before approval: the pooled counts and
sizes by source on seeds 0-4, both worlds, with the gate's checks; on the spent seeds 10-14, both worlds, from version
8's recorded second critics and data records (each regenerated log and critic checked against its record): BCA's
pooled certificate (switch, height, 5% bound), the plain 90% gap and twice it, the mean difference, the
split-conformal and rank-test statistics and calls, each height's rank among the group's scores, and the comparison
with version 8's recorded calls.

TODO(FREEZE: the table of those facts, copied from the record: per seed-world the pooled sizes, BCA's call and height,
the plain gap, the mean difference, the four switches' calls and the ranks; and the line "kill rule: passed on 10 of
10 spent seed-worlds".)

**Kill rule (section 8.1, check 5):** BCA's and the mean-difference switch's pooled calls on seeds 10-14 equal version
8's recorded calls (on in 5 of 5 stalling, off in 5 of 5 no-stall seed-worlds); otherwise this design is redrafted and
no approval is asked. The number of redrafts is written in section 0.

---

## 5. Arms

### 5.1 The host

TD3+BC in step 2's actor-only protocol, unchanged: each world's critic trained on its own kept rows with the gate's
call (100,000 updates), then frozen; actors from that world's cloned actor, 20,000 steps in 4,000-step chunks, alpha
5, the gate's fixed batch stream; arms of one kind side by side, one processor per phase.

### 5.2 The main setting, both worlds: step 2's arm set

Version 8's claim-4 arm set, unchanged, restated for seeds 40-49:

| arm | role |
|---|---|
| BCA's switched ramp, k = 1 (`obca_k1`) | **primary** (C1); the noise floor's batch side |
| the ramp with height 0 and k = 0 (`w0_check`) | plumbing check: must equal the no-caution actor within 0.00001 |
| the ramp at k = 2 (`obca_k2`), BCA's switch with height 1.0 (`fixed_gated`), the rank-test switch (`meangap_k1`) | reported only (step 2's) |
| the split-conformal switch (`conformal_k1`) | reported only (step 2's); its no-stall false alarms are counted; not a twin, because it differs from BCA in its switch as well as its height |
| the always-on ramp (`obca_forced`, at BCA's stalling-world height) | world-blind rival |
| the matched range limit (`standin`); the matched even pull and the two matched random surfaces | world-blind rivals |
| the hand-set range limit (`range_hinge`) | **world-blind rival** (new in this role: section 9.2) |
| the matched push-less penalty | reported |
| the perfect band | reference |
| the no-caution (`none`), cloned (`clone`) and full-log (`full_none`) actors | references |

### 5.3 New arms (every one trained by its own rule in every seed-world)

| arm | switch | height / penalty | why it is there |
|---|---|---|---|
| `meandiff_q90x1`, **twin 1**: the mean-difference switch with the plain 90% gap | on when mean(full-throttle group) - mean(rest) > 0.10 (the same 0.10 as BCA's margin, carried unchanged) | the plain 90%-quantile gap of the same two groups (split conformal's statistic; no Bayesian step), k = 1 | the plain height on the near side of BCA's, with BCA's own call: it differs from BCA's switched ramp in height only, so BCA ahead of it is a dose reading. The split-conformal switch cannot play this part: it differs in its switch too (wrongly on in the no-stall world of seeds 10 and 13 under pooling, gaps 0.121 and 0.116 against its 0.10 margin). |
| `meandiff_q90x2`, **twin 2**: the same switch with twice that gap | the same | twice the plain 90%-quantile gap (step 2's own k = 2 factor) | the plain height on the far side of BCA's on four of the five preview seeds (pooled heights 1.67-2.78 on seeds 10-14 against plain gaps of 0.99-2.48 and twice them 1.98-4.96; on seed 14 BCA's 2.26 is above twice the gap, 1.97). If BCA beats plain x1 but not plain x2, its edge is a dose. |
| `meandiff_limit`, **twin 3**: the same switch turning on the hand-set range limit | the same | when on, the hand-set range limit's own penalty (the room's hinge at the recorded range, coefficient 1.0, `range_hinge`'s program, run alone); when off, no caution (the ramp with height 0, checked equal to the no-caution actor) | the no-BCA composite that led BCA by 3.2 points on average in the records, now trained as one arm |
| `obca_k1_alone` (validity, and C2's and C3's BCA side) | BCA's | BCA's height, trained alone instead of in step 2's six-arm batch | the twins train alone, so C2 and C3 compare them with this copy (the same training mode); the noise floor `obca_k1` minus `obca_k1_alone` is the validity check that the batch did not move BCA's points by more than 5 |
| `meandiff_k1` (reported only) | the mean-difference switch | the mean gap itself (0.4-0.8), k = 1 | version 8's arm, the bottom of the dose ladder; BCA beats it by 40+ points by dose alone, so it is not a twin |

None of the twins uses BCA's posterior or its certificate. **The order of the lone calls is fixed:** after step 2's
batches, `meandiff_k1`, `meandiff_q90x1`, `meandiff_q90x2`, `obca_k1_alone`, each a ramp program of one, then
`meandiff_limit` (the hinge program, or a height-0 ramp). **Matching:** as step 2, blind to returns, to the primary
arm's stalling-world strength, with version 6's ladders and matcher; the no-stall world uses the stalling world's
locked settings.

**Reported composites, no training:** BCA's switch turning on the hand-set range limit; "refusal means the range
limit" (BCA's switched ramp where it set a height, the range limit where it refused). Both from trained arms.

### 5.4 The refusal rule

`step2lib.arm_table` refuses a switched-on arm whose height is not finite. Version 8 let that refusal stop the whole
actor phase. This version reads it per arm: an arm whose switch is on with a height that is not finite is **refused**
(recorded in the switch record, `refused: true`, and in the arm table; never trained; its actor missing); every other
arm trains and is scored. The refused arms can only be `obca_k1`, `obca_k2`, `obca_forced` (which takes the stalling
world's height in both worlds) and `obca_k1_alone`; a refusal of any other arm is a plumbing stop. With no BCA actor
there is no strength to match, so the matched rivals and the stand-in of that seed are unmatched, as in version 8
when BCA's switch stayed off. A refused arm's place in step 2's six-arm batch is filled by a height-0 pad, never saved
or scored, so the other members keep step 2's batch size. The seed is never replaced; nothing of it stops.

**In C1's marks**, a refused arm's numbers are **the no-caution actor's** in that seed-world, and the controls matched
to BCA's strength (the matched range limit, the matched even pull, the two matched random surfaces) are matched to
zero strength, which is also the no-caution actor's numbers. Nothing is "not available", so no mark fails by a
missing number; the refusal's cost enters the pooled means as it would enter a deployment (for the keeps mark a
refused seed enters at 0, which is no cost; the lead and even-pull marks carry the cost). (The literal reading of an
infinite height as an infinite penalty, a hard clip at the data's edge which the hand-set range limit approximates,
is not used: it would reward refusing, since the range limit scored 93.1 against BCA's 88.6 in the stalling world on
seeds 0-4; it is reported as a composite only.)

**In C2 and C3**, a seed where BCA refused in either world is left out: the twin claims are read on the certified
seeds (at least 8), with the sign thresholds of section 9.3 for their number. The refusal is named in C2's sentence.
A refused seed enters no twin difference, because the no-caution actor against a twin would be a difference of about
-100 and would decide the claim by itself. Reading them on the certified seeds is not outcome selection: it is
selection on the group's size and BCA's draws, both written before any arm trains, never on a return; a seed with a
small group is also a seed where BCA's height is its top order statistic, so the certified seeds may differ in the
height ratio, which the dose ladder reports.

**What a refusal is expected to do** (an expectation, not a rule; C1 is read from its marks only): resampling the
seven recorded main-setting seeds, the pooled lead over the best world-blind rival with ten seeds is about 26 with no
refusal, 19 with one, 8 with two and -2 with three, against the mark of 10. So one refusal is expected to be
survived, two are expected to fail the lead mark (about 1 chance in 4 of passing). The chance of no refusal in ten
seeds is about 0.65, of at most one 0.94 (the gate's 30 logs, leak factor 2.0; 0.55-0.77 and 0.89-0.97 across
factors 1.8-2.3).

### 5.5 One training program

The code trains the ramp arms of one batch side by side, and the batch size changes the last bits of each actor. So
BCA's switched ramp and the plumbing check train in step 2's own batches: where full throttle stalls, the plumbing
check alone, then the six step-2 ramp arms with BCA's switched ramp first; in the no-stall world, one batch of seven
with the plumbing check first. Then, in calls that leave those programs' bits unchanged, the lone calls of section
5.3 in their fixed order.

### 5.6 Fixed references (no seed, nothing trained)

The recording controller, the stall-aware and rising-thrust controllers (the points' anchors, in both worlds).

### 5.7 Left out, and why

The near-80 and random-rows settings (version 8's exploratory stop found the band changes nothing for a range limit:
a hand-set range limit lost only 4.3 points to it); version 6's BCA (it depends on the band setting's readouts, and
netted 0.13 against no caution in version 8); a fallback height or a lower credibility (they change BCA); a hard clip
as the reading of an infinite height (it would reward refusing); the split-conformal switch as a twin (it differs
from BCA in its switch).

---

## 6. Situations

The main test's four fixed test start sets of 1,000 starts, 100 steps each, in both worlds: recorded conditions, big
kicks, big waves, strong current (current 0.7 to 1.1, above anything recorded). **The marks are read in strong
current.**

---

## 7. Measures

For every scored actor, world and situation: points on the full test start set; the unseen-throttle readout (points
as trained minus points held to 70%) and the overshoot readout (points capped at 80% minus as trained), carried over
from version 8 and reported only; the zone shares; the lower tail (the worst 100 of 1,000 starts); net across worlds;
strengths and matches; time and peak memory per phase. Per seed and world: both groups' sizes by source, the
certification chance, BCA's switch, height, 5% bound, ramp ends and group sizes, with the refusal when there is one;
each switch's statistic and call (BCA, the mean difference, the split conformal, the rank test); each ramp arm's
height and its rank among the full-throttle group's scores.

---

## 8. The tests, in order (kill rules first)

Every threshold is fixed now. None moves after its numbers exist, except through a labeled post-hoc
pre-registration that you approve.

### 8.1 The no-training checks (before approval on seeds 0-4 and 10-14; on each fresh seed when its data are built)

| # | check | pass |
|---|---|---|
| 1 | the gate's five world checks on the ball (stalling world); the ball's leak checks | all pass |
| 2 | the training file equals `dataset.d4rl_dict` of the kept rows (six arrays, SHA-256) | equal |
| 3 | the pooled groups: every held-out row outside the ball appears exactly once; no training row other than a survivor; no ball row; the outcome-free count equals the sizes the switch scores | all pass |
| 4 | BCA's gap rule is defined on the ball-kept training commands; `t_lo` in [0.78, 0.80] (seen 0.782-0.791) | defined, in range |
| 5 | (before approval only) the pooled counts on seeds 0-4; on the spent seeds 10-14, from version 8's recorded second critics, the pooled switch makes version 8's recorded calls (BCA's and the mean-difference switch's) | on in 5 of 5 stalling, off in 5 of 5 no-stall seed-worlds, for both switches |

A fresh seed that fails check 1 or 2 is a failed seed: it fails every mark and is never replaced. A failure of check
3 or 4 on a fresh seed is a plumbing stop of that seed-world: every mark that needs it fails by the pre-set rule, the
look says that a stop and not a return decided it, and the marks on the seeds that ran are reported as description.
Check 5 failing on seeds 10-14 means a redraft and no approval; the number of redrafts is written in section 0.

### 8.2 Plumbing, after approval (scratch only)

- The unit tests (five files).
- A 40-step smoke of every phase kind on seed 0, in scratch, with the pooled groups.
- The fresh-seed code path on seed 0: it must reproduce the gate's seed-0 data record, and the record written must
  read back (amendment 1's lesson).
- **The reproduction check:** on seed 0 in both worlds, the near-80 code must reproduce bit for bit the gate's
  critic, cloned actor and no-caution actor, step 1's second critic and certificate (on version 8's groups: the
  survivors and the bank, the only groups step 1 computed), and, through the official policy function given step 1's
  own switch entries, step 2's saved plumbing-check, switched-ramp and hand-set range-limit actors. The pooled
  groups' certificate of seed 0 is computed beside it and reported, not compared. If any item differs, nothing
  official starts on that host, ever, in this version: a difference on the cluster (whose processors differ from the
  desktop's that made the reference files) moves the whole of version 9 to the desktop fallback (12.1), with its own
  plumbing from a clean output root; a difference on the desktop too ends this version.
- The first peak-memory measurements of each phase kind.
- Only a reproduction difference is final. Any other failure can be retried once its cause is fixed.

### 8.3 No exploratory stage

Seeds 0-4 are read only by the no-training checks of 8.1 and the reproduction check of 8.2. There is no harm stop
(the main setting's harm was 98.9 points on seeds 0-4) and no outcome-based stop before the fresh seeds.

### 8.4 The confirmatory stage: seeds 40-49

In order, per seed: the main setting's data of both worlds (the checks of 8.1 and the outcome-free count); the
critics, the full-log critic and BCA's second critics; the switches on the pooled groups; the cloned, no-caution and
full-log actors; the actor phases with matching and the refusal rule (5.4), each seed-world locked (every actor's
fingerprint written) before any score; the evaluations; then the look. **Nothing on seeds 40-49 stops early on an
outcome.**

**Plumbing stops, per phase:** the height-0 check, a non-finite network, a saved input whose fingerprint does not
match, a training file that fails its check, or a pooled-group check. A stopped phase is written once and never
rerun. Every mark that needs it fails by the pre-set rule, and the sentence says that a stop and not a return decided
it.

---

## 9. Pass marks, with reasons and power

Each mark is pooled: the mean of each seed's paired difference. Stalling world, strong current, main setting, unless
stated. C1 is pooled over all ten seeds; a refused seed-world enters it with the no-caution actor's numbers for the
refused arms and for the controls matched to BCA (5.4). C2 and C3 are pooled over the certified seeds (BCA's height
finite in both worlds), at least 8 of them. A failed seed (8.1) fails every mark; a stopped phase fails every mark
that needs it, and its sentence says so.

### 9.1 Validity

- **For every claim:** the harm is at least 30 points (the full-log actor beats the no-caution actor by at least 30,
  pooled), and the no-caution actor goes past 80% on at least 25% of strong-current steps, pooled; both over the
  seeds that did not fail (a refused seed's no-caution and full-log numbers count: they do not depend on BCA). On
  the main setting on seeds 0-4 these were 98.9 points and 73.2%.
- **For C2 and C3:** the noise floor, `obca_k1` minus `obca_k1_alone`, pooled over the certified seeds, within 5
  points either way. If not, C2 and C3 are dropped and reported only (their 5-point margin would sit inside the
  training noise). C2 and C3 read `obca_k1_alone` as BCA's side, the same training mode as the twins; the differences
  against the batch-trained `obca_k1` are reported beside them. At least 8 certified seeds; otherwise C2 and C3 are
  not read. When no certified seed is left to read the noise floor, it cannot be read, and C2 and C3 are written as
  their other rules say. The two conditions above (the harm and the past-80 share) are version 8's and step 2's,
  kept unchanged for every claim; `DECISION.md`'s appendix 9.1 said C1 had no validity condition while its 14.1
  carried the "Validity fails" sentence; this document keeps the condition (departure 15.12).

### 9.2 C1: step 2 confirmed in the main setting (version 8's claim 4 marks, unchanged)

| mark | pass | reason | on the main setting, seeds 0-4 (exploratory) |
|---|---|---|---|
| net lead | BCA's net across worlds is at least the best world-blind rival's plus 10; the world-blind rivals are the matched range limit, the matched even pull, the two matched random surfaces, the always-on ramp and the hand-set range limit (ties go to the first in that order) | step 2's margin | +25.3 over the always-on ramp (standard error 2.3); the hand-set range limit nets 76.5 |
| keeps the gain | BCA minus no caution is at least -3 in recorded conditions, big waves and big kicks | version 6's margin | +8.0, +21.2, +19.8 |
| beats the even pull | BCA minus the matched even pull is at least 15 | version 6's margin | +100.8 |

- **One change to the rival set (the marks are unchanged):** the hand-set range limit joins the world-blind rivals for
  the net-lead mark (it cannot switch itself off, and with refusals it is the rival that keeps the lead mark from
  collapsing). Version 8 listed it as "reported; used in claim 3". On every recorded main-setting set it is below the
  always-on ramp (76.5 against 77.7 on seeds 0-4; 64.0 against 76.6 on seed 11; 79.3 against 81.5 on seed 14), so
  nothing changes where BCA acts; in the near-80 setting, which version 9 does not run, it was above it (54.7
  against 49.1). The lead is computed in this version's look over the enlarged set; step 2's own lead over its own
  five rivals (`step2lib.go_no_go`, pinned) is reported beside it.
- Under the refusal rule with ten seeds the pooled lead is about 26 (sd 1.3) with no refusal, 19 (sd 2.5) with one, 8
  (sd 2.4) with two and -2 with three (resampling the seven recorded main-setting seeds). One refusal is expected to
  be survived, two to fail the lead mark (about 1 chance in 4 of passing). This is an expectation; the mark is read
  from the numbers.
- The marks on the certified seeds alone are reported beside C1, description only.
- Prediction: passes with 0 or 1 refusal; the heights shrink to about 0.5-0.75 of version 8's, so BCA's own
  stalling-world points may fall toward the plain switch's; the lead mark is about the switch, not the height, and
  is expected to hold.

### 9.3 C2 and C3: BCA's switched ramp against each plain twin

For each twin T in {the mean-difference switch with the plain 90% gap, with twice it, turning on the hand-set range
limit}: d = NET(BCA's switched ramp trained alone, `obca_k1_alone`) minus NET(T), the net across worlds in strong
current, per certified seed; n certified seeds.

| reading | rule | n = 10 / 9 / 8 |
|---|---|---|
| BCA ahead of T (C2's part) | the mean of d is at least 5.0, and d is positive in at least ceil(0.8 n) seeds | 8 / 8 / 7 (a coin gives that many with chance 0.055 / 0.020 / 0.035) |
| T ahead of BCA (C3 for T) | the mean of d is at most -5.0, and d is positive in at most 1 seed | 1 / 1 / 1 (coin chance 0.011 / 0.020 / 0.035) |
| tie | otherwise; written "BCA's switched ramp was not shown to beat T by 5 points" (never "did as well" or "interchangeable": a true 5-point edge reads "tie" about half the time) | |

**C2 holds when BCA is ahead of all three twins** (one claim; the multiplicity rule is `all_three`: a conjunction
keeps the error rate at the per-twin level, so no Holm step is applied). **C3 holds for a twin when that twin is
ahead of BCA** (one claim per twin; the three form a family whose chance of any false "ahead" under a true tie is
0.001 / 0.014 / 0.023 / 0.028 at per-seed spreads 5 / 7.5 / 10 / 15). The three readings are reported in every
case, with the one-sided 95% lower bound of each mean (Student t on the certified seeds; reported, not used) and the
same differences against the batch-trained `obca_k1`.

- **Why 5 points:** the project's smallest mark unit (version 8's claims 2 and 3: "5 points, and 4 of 5 seeds"; step
  2's 5-point wording rule; round 1's bar), and about one per-seed standard deviation of the paired differences
  within the main setting (1.9-5.2). A 5-point mean is a difference an operator would notice.
- **Why ceil(0.8 n) and at most 1:** the sign parts at about the 5% and 1-3% levels; C3's is stricter because three
  twins give three chances. Version 8's 4 of 5 (chance 0.19) was carried by the mean.
- **Power per twin (ten certified seeds, simulated):** a true BCA edge of 10 points is read "ahead" with chance 1.00
  (spread 5) or 0.79 (spread 10); 5 points: 0.48 (0.31 at spread 10); 3 points: 0.10; 0: 0.001 (0.028 at spread 10,
  0.044 at 15); the main-setting BCA-conformal spread on seeds 0-4 (5.6, sd 1.9): 0.84. A true twin edge of 10 is
  read "T ahead" with 0.98 (spread 5) or 0.51 (spread 10); the composite's recorded 4.5-point edge (sd 5.2) with
  0.27, "tie" 0.73; 3.2 points with 0.09. A true twin edge of 5 points reads "T ahead" with 0.38 (against 0.48 for
  the same edge on BCA's side): the price of C3's stricter sign count, chosen because three twins give three chances
  (the family rate above). So the test finds a 10-point edge either way, finds a 5-point edge about half the time,
  and invents one with chance at most about 0.03 per twin.
- **Power of the conjunction (C2 as a whole, shared BCA side):** three true 5-point edges: 0.23 (spread 5) or 0.09
  (spread 10); three true 10-point edges: 0.99 or 0.56; the predicted pattern (6, 0, -3.5): 0.00. C2 is a demanding
  claim by design: it credits BCA's height only when it beats a plain height on both sides of it and the range limit.
- **What the twins separate:** where BCA and a twin make the same on/off call (the mean-difference switch made BCA's
  call in 40 of 40 recorded seed-worlds), the comparison is of heights only; the gated range limit replaces the ramp
  by a hard limit at the data's edge.
- Prediction: C2 fails and C3 does not fire; "ahead" or "tie" against the plain gap, "tie" against twice it, "tie"
  against the gated range limit.

### 9.4 Rules about the marks

- No mark moves after its numbers exist.
- C1, C2 and C3 are read separately. None rescues another (version 8's rule 9.6).
- Seeds 0-4 never carry a claim; the plumbing's seed 0 never carries a claim.

---

## 10. Confirmation on fresh seeds 40-49

- **Data:** new logs of both worlds with the gate's recorder, sizes and settings (stall slope 1.5), as version 8's
  amendment 1 builds them. Then the ball, its leak checks, the gate's five world checks (required for every claim),
  the training file, the pooled count.
- **A fresh seed that fails** check 1 or 2 fails every mark and is never replaced. A stopped phase fails every mark
  that needs it (a stop, not a reading). A refused seed is read by the refusal rule (5.4).
- **Learners:** every seed-world is made fresh: critics, the full-log critic, BCA's second critics, cloned actors, the
  reference actors, every arm of section 5.
- **Frozen from this pre-registration:** the pooled rule, the refusal rule, every arm, ladder, matching rule and
  mark, the seeds, the sentences of section 14.
- **The confirmatory look:** the validity conditions of 9.1, then C1, then C2 and C3; the code fills in section 14.
- **Old code that refuses these seeds is restated, never edited.** Version 6's fresh-seed code and step 2's runner
  are restated for the fresh seeds in new code.

---

## 11. Approval

The route of versions 6, 7 and 8, on the Framework Desktop, inside Ubuntu as root, from `/home/user/bca`:
`python3 -I outputs/shift/beyond-the-data/near80/approve_local_near80.py --check`, then the same without `--check`.
The script checks every file against the code manifest, refuses any file the manifest does not list other than
version 8's history (section 18), the build notes and the `decision/` folder, refuses a draft, and asks you to type the first 16 characters
of `prereg.json`'s fingerprint. The session imports your approval (`code/import_approval.py import`) and pushes the
import record. Every phase refuses without it. The folder is `outputs/shift/beyond-the-data/near80/`: version 9
supersedes version 8 in place (choice 7); version 8's `prereg.json` moves to `prereg_history/`, its approval, import,
amendment and results files stay, and its chain is checkable at commit `52914e4`. The branch is
`claude/main-bca-connection-x4o46s`.

**Approving accepts every choice of section 16 as recorded there** (the defaults, with the cluster as the run
machine (choice 4) and the desktop as the fallback; the results copied from the cluster to the desktop and committed
from there (choice 9, the default)). Any other choice makes a new fingerprint. The approval itself is typed on the
desktop, whatever machine runs; the cluster clone fetches the approved commit afterwards.

---

## 12. Hosts and order of work

### 12.1 The desktop: the fallback

`run_near80.sh`, as version 8: root inside WSL, tmux, one runner, 1-16 lanes, version 7's memory rule. Version 8's
checks on version 7's run are gone (it finished on 16 October 2026). By your answer (choice 4) the desktop is the
fallback: it runs version 9 if the cluster's plumbing does not reproduce the reference files (8.2), or if the cluster
is not available; ten seeds cost one evening there (section 13). The approval is typed on the desktop in either case
(section 11).

### 12.2 The cluster (hpc.charlotte.edu): the run machine

**The entry:** `run_near80_cluster.sh` with its job script `near80_slurm_job.sh`, both pinned in the code manifest
and approved with it. On a login node, never as root, from a git clone of the approved commit under
`/scratch/dbayha` (`NEAR80_HOST=cluster`, `NEAR80_ROOT`, `NEAR80_PYTHON`, `BCA_SHIFT_DEPS`), with Slurm account
`staruser`, partition `Orion` and `--constraint=caslake` (one processor type for the whole run: the 81 Cascade Lake
nodes of 48 processors, whose OpenBLAS kernel is the desktop's):

1. `selftest` on the login node (READY or NOT READY; runs nothing), then `plumbing`: one job of 4 processors, 16 GB,
   up to 2 hours (section 8.2: the five unit test files, the 40-step smoke, the fresh-seed path, the reproduction
   check).
2. `run`: one job per fresh seed (8 processors, 24 GB, up to 3 hours: its 15 phases of both worlds up to the actor
   training; the job's runner starts only its seed's phases and ends once only lock-dependent phases remain), then
   one group job (16 processors, 24 GB, up to 1 hour) for the lock, the 20 evaluations and the look, submitted with
   `--dependency=afterany` on the seed jobs. Every job runs the same frozen runner with the same approval import and
   manifest checks as the desktop; the job's Slurm memory stands for the machine's in version 7's memory rule; one
   phase per allocated processor.
3. `status`, `plan`, `report`, `retry`, `budget-continue` on the login node, as on the desktop.
4. **Results (your answer to choice 9: committed from the desktop after the copy from the cluster):** `results` on
   the login node copies every record (no model file) into `near80/results-<utc>/` in the cluster clone with
   `SHA256SUMS`, writes `OUTPUT_ROOT_SHA256SUMS` there (the SHA-256 of every file of the output root
   `runs/shift_explore/near80_v9/`, model files included) and prints the steps that follow. Nothing is committed or
   pushed from the login node. In your own SSH session you copy the results folder and the output root to the
   desktop (`scp -r`, or `tar` over SSH) into the desktop clone's `near80/` and `runs/shift_explore/`, check both
   lists there (`sha256sum -c`), and then, on the desktop, commit and push the results folder under your own
   identity (`git add -f`; the data records sit under `data/` folders that `.gitignore` hides) and run
   `push_runfiles_near80.sh` (it needs a terminal and your GitHub sign-in), which pushes the run files to
   `claude/near80-v9-run-files`: exactly as version 8's records were committed.

**What the entry provides that the desktop assumed:** a stand-in `HOME` under `/scratch/dbayha` (nothing is written
under `/users`); the pinned Python environment rebuilt from the package's lock file with the same tool (uv 0.11.32,
Python 3.10.20, the 67 hash-checked wheels; numpy 2.2.6, jax 0.6.2 on the processor); the shift dependencies at
the literal path the gate's pinned pre-registration records (`/root/.cache/bca-shift/py`), provided inside an
unprivileged user namespace by the job script (a tmpfs over `/root`, the dependency folder bound at that path, the
real user id mapped back), since the gate's own check refuses any other path; the gate's data (`runs/shift`), the
room harness, step 2's saved actors and version 6's run folder (`runs/shift_explore/stage2`) copied read-only from
the desktop clone (the self-test checks them); version 6's pinned libraries rebased onto the clone's path in the
process after their bytes are verified. The identity rule is the desktop's.

**Tested on the cluster on 20 October 2026** (the record is in `V9_BUILD_NOTES.md`): the environment is byte for byte
the desktop's (the package list, the wheels, the shift dependencies); a numerics probe and a paired BCA-free driver
(the critic, the cloned actor, the no-caution actor, two evaluations) gave bit-identical results on the desktop and
on a Cascade Lake, a Genoa and a Turin node; the four unit test files of version 8 pass on an Orion node; the entry's
stub run (stand-ins, nothing trained, a synthetic fixture) finished all 19 phases as real Slurm jobs through the
patched job script. Not covered by any cluster test: BCA's certificate, the second critic, the matching, the lock and
the look; the plumbing's reproduction check (8.2) decides, and a difference there moves the run to the desktop.

### 12.3 Order

1. **Done:** this document, its machine-readable twin (`prereg_draft.json`), the code and its tests, `DECISION.md`,
   the cluster entry and its test.
2. **Done:** your answers to section 16 ("Accept the defaults and cluster", 20 October 2026, 18:29 UTC; on choice
   9, "committed from desktop after copying over from cluster", about 19:20 UTC), recorded in `OWNER_ANSWER`.
3. The session freezes (`code/freeze.py write`, inside WSL in the pinned environment): the draft into history, the
   manifest, the five unit test files, the facts of section 4.2 (the counts on seeds 0-4; the pooled switch on the
   spent seeds 10-14 from version 8's recorded second critics, with the kill rule; about 10-15 CPU-minutes, nothing
   trained), `prereg.json`; then an independent re-check and the fingerprint.
4. **You approve** (section 11, on the desktop). The session imports it; the cluster clone fetches the commit.
5. Plumbing on the cluster (section 8.2; one job, about 1-1.5 hours). If its reproduction check differs: the desktop
   fallback (12.1), from step 5 again.
6. The confirmatory stage on seeds 40-49 (about 1.7 hours of wall time on the cluster; about 5-5.5 on the desktop)
   and the look.
7. The results folder and the output root copied from the cluster to the desktop over your own SSH session and
   checked by SHA-256 there (12.2, step 4); the results committed with their fingerprints from the desktop's clone
   (`git add -f`: `.gitignore`'s `data/` line hides the data records); the model files to the run-files branch
   `claude/near80-v9-run-files` from the desktop; the independent check.

---

## 13. Compute

One processor per phase, on the processor alone, on either host. **The desktop's numbers** (the Framework Desktop in
Ubuntu under WSL), from version 8's ledger of the main setting (`results-20261020T115246Z`): 1.53 and 1.51 CPU-hours
per seed for the two seeds that ran every phase (critic 5.7 minutes, full-log critic 5.7, second critic 5.7 per
world, switch 0.3, clone 0.7, reference actors 3.5 and 2.2, the stalling-world actor phase 42 and the no-stall one
9.3, the evaluation 1.5 per world). The five programs run alone (four lone ramp calls of about 63 s in the stalling
world and about 40 s in the no-stall world, and the hinge) add about 0.14 per seed; the pooled scores (about 42,000
rows) seconds. **The cluster's numbers** (a Cascade Lake node of Orion, measured on 20 October 2026 with the
package's own phases at the official sizes, BCA-free): the data phase 0.64, the critics 1.25, the clone 1.16 and the
actor training 1.41 times the desktop ledger's seconds (the ledger was taken under the desktop's own ten-lane load;
an idle desktop core is 1.62 times faster than a caslake core), so about 2.05 CPU-hours per seed for version 8's
phases and about 0.2 for the lone programs; the two actor phases are 59 and 13 minutes there.

| part | desktop, CPU-hours | cluster (caslake), CPU-hours |
|---|---|---|
| the facts before approval (on the desktop, nothing trained) | about 0.2 | - |
| plumbing | about 1.3-1.5 (version 8: 1.23; the lone programs added to the reproduction) | about 1.5-2 |
| confirmatory, seeds 40-49, main setting, both worlds | about 16.5-17 (about 1.65 per seed) | about 22.5 (about 2.25 per seed) |
| **total** | **about 18** | **about 24** |

- **Wall time on the cluster** (12.2): every seed runs at once, so the wall is one seed's longest chain (data, critic,
  reference actors, the stalling-world actor phase, the no-stall one: about 96 minutes on caslake with the lone
  programs) plus the group job's 20 evaluations over 16 lanes (about 4 minutes) and two queue waits (seconds on the
  day of the test, with about 2,000 idle processors): **about 1.7 hours for the ten seeds**, the same for twenty. The
  plumbing job about 1-1.5 hours. Allocated processor-hours are about 5 times the CPU-hours used (10 jobs of 8
  processors for about 1.6 hours), which costs nothing that matters on Orion.
- **Wall time on the desktop** (the fallback): version 7's memory rule, unchanged (choice 5): at most 2 actor phases
  at once, so about 5-5.5 hours with 10 lanes in one sitting (ten stalling-world actor phases of about 47 minutes,
  ten no-stall of about 13, a lead-in of about 20); about 2.5 hours if the actor reservation were set to 1.25 times
  its measured peak (choice 5's alternative). The plumbing about an hour.
- **Lanes:** 10 by default on the desktop (choice 6); on the cluster 8 per seed job (a seed's width is 7 phases), 16
  for the group job, 4 for the plumbing.
- **Budget guard:** as version 8, with a forecast of 2.2 CPU-hours per seed (22 for ten; a margin over the desktop's
  1.65 and near the cluster's 2.25); a stage that uses more than twice its forecast (44) pauses and reports; the cap
  is 2.5 times the forecast, 55 CPU-hours; the cluster's expected 22.5 lies inside both. A pause lasts until you type
  `budget-continue fresh`.

---

## 14. The claim, word for word, for each outcome

The code fills in the brackets. Numbers are rounded to one decimal. Every bracket is pooled over the fresh seeds, in
strong current, stalling world, main setting, unless stated:

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
> matched even pull by [E]. It set a height on [K] of the [NSEEDS] seeds and refused on [F] (a refused seed's
> switched ramp, the ramp at twice its height, the always-on ramp and its noise-floor copy count there as the
> controller trained without caution; section 14.4 names each refusal). Where full throttle stalls and its switch was
> on, its controller is the same as the always-on ramp's, so that gain comes from where the ramp starts and from its
> height, not from the switch."

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
- "Step 2's exploratory result was not reproduced on these seeds." (written when a mark was missed on its numbers)
- "A stop, not a return, decided this: no reading of BCA's return is claimed, and the marks on the seeds that ran
  are reported as description." (written when every missed mark is told by a stop clause or a failed-seed clause
  alone)

**Always after C1, the certified seeds:**
> "Description only: on the [NC] seeds where BCA set a height in both worlds, the three marks read [reading] (lead
> [DC] over [RC], largest loss [LC], even-pull margin [EC])."

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
> made BCA's on/off call in [A] of 20 seed-worlds; where it did, the comparison is of heights alone, and nothing is
> credited to the switch."

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
  calibration. Nothing is credited to BCA's Bayesian step or to its height." (written when BCA was ahead of the plain
  gap and tied with the other two)
- "No difference of 5 points or more from a plain switch with a plain height, of either size, or from the hand-set
  range limit was shown. No credit beyond a plain rule is shown." (written when BCA tied with all three)
- "A rule with no BCA in it, [twin], did better by the pre-set margin (C3)." (written when any twin was ahead; the
  first such twin in the twins' order is named)
- "No credit beyond a plain rule is shown." (written otherwise)

**A twin ahead (C3 holds for it):**
> "On the [NC] fresh seeds where BCA set its height, [twin] beat BCA's switched ramp by [DT] points on the net
> across worlds, paired by seed (BCA ahead in only [KT] of [NC] seeds; at most 1 was allowed and at least 5.0 points
> were required). A rule with no BCA in it did better by the pre-set margin."

**No twin ahead (C3 fails):**
> "No plain twin beat BCA's switched ramp by the pre-set margin."

**Each separately, a twin beaten:**
> "(Not used: the multiplicity rule is all_three.)"

**Each separately, a twin not beaten:**
> "(Not used: the multiplicity rule is all_three.)"

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

- Numbers are written with one decimal ("{:.1f}"), with a minus sign when negative and no plus sign; "-0.0" is
  written "0.0". [P] is a share of steps written as a percentage; [p] is a chance written as a percentage with one
  decimal. [S], [N], [K], [F], [NC], [K1], [K2], [K3], [KT], [KMIN], [A], [NSEEDS], [NMIN], [NMAX], [NREACH], [NSW],
  [N8MIN], [N8MAX], [n] and [s] are whole numbers. [SEEDS] is the seeds' range, "40-49". [KMIN] = ceil(0.8 NC).
  [LBT] (and [LB1]-[LB3]) is the one-sided 95% lower bound of the mean of d, Student t on the certified seeds.
- A number that cannot be computed, because a seed failed, a phase stopped or an arm is missing, is written "not
  available"; the seed fails every mark that needs it, and its own clause tells it: a missing BCA or twin number on a
  seed that did not refuse fails that seed for every claim, so C1 reads "no reading" even when only a twin's number
  is missing; a matched rival that could not be matched fails C1's marks only. A refused seed-world's BCA-dependent
  arms and the controls matched to BCA are not missing in C1: they take the no-caution actor's numbers (section 5.4).
- [R] (and [RC]) is written as "the always-on ramp", "the matched range limit", "the matched even pull", "the first
  matched random surface", "the second matched random surface" or "the hand-set range limit"; ties go to the first in
  the order of section 9.2. [twin] is "the mean-difference switch with the plain 90% gap", "the mean-difference switch
  with twice the plain 90% gap" or "the mean-difference switch with the hand-set range limit".
- [situation] is "recorded conditions", "big waves" or "big kicks". [world] is "stalling world", "no-stall world" or,
  for a refusal in both, "stalling world and no-stall world". [phase] is the phase's plain name in the runner's
  phase table. [reading] is "pass" or "fail" (the three marks on the certified seeds alone), or "not available" when
  no seed certified.
- [one clause per missed mark] lists the clauses of the missed marks, joined by "; ", in this order: the net lead;
  keeping the gain (one clause per situation that misses, in the order recorded conditions, big waves, big kicks);
  beating the even pull; then one clause per failed seed, in seed order (the stopped-phase clause for a seed failed by
  a stop, one per stopped phase; the failed-seed clause otherwise); then one clause per refused seed, in seed order.
  A missed mark whose number is not available because a seed failed is told by that seed's own clause only.
- C1's last sentence is the first form when any clause of the net lead, the keeps or the even pull appears, otherwise
  the second.
- [one clause per twin] lists the twins' clauses in the twins' fixed order (the plain 90% gap, twice it, the hand-set
  range limit), each by its reading (ahead, tie, behind), then the refused seeds' clauses in seed order. C2's last
  sentence is chosen by the four cases in their order. C3's "A twin ahead" sentence is written once per twin ahead, in
  the twins' order, else the "No twin ahead" sentence once; C3's sentences follow C2's.
- [check] is "the ball's leak checks", "the gate's five world checks on the ball alone", "its [phase] phase" for a
  stopped phase, "its evaluation (numbers missing)" for a seed whose evaluation lacks a number, or "its matching (a
  rival could not be matched or scored)" for a seed whose matched rivals could not be matched and BCA did not refuse
  (that seed fails C1's marks only).
- [reason] in C1's validity sentence is the harm reason when the harm is below 30 points, the past-80 reason when the
  share is below 25%, and both, joined by "; ", when both fail.
- In the reproduction stop, the bracket lists every item that differed, written "A", "A and B" or "A, B and C".
- C2 and C3 are written only when C1's validity holds, in this order of precedence: the margins not fixed (cannot
  occur); a seed failed (the "If a seed failed" sentence, or "If every seed failed" when no seed is left: a failed
  seed fails every mark by the pre-set rule); "Not read" (fewer than 8 certified seeds); "Dropped" by the noise
  floor; then "Holds" or "Fails" and C3's sentences. Wherever a sentence says "seed [seeds]", it reads "seed 41" for
  one seed, and "seeds 41 and 43" or "seeds 40, 41 and 43" for more.
- The certified-seed sentence is written whenever C1 is written on the fresh seeds (not on the plumbing's seed 0).
- On the plumbing's seed 0, every claim sentence starts with the plumbing prefix and "fresh seeds [SEEDS]" reads
  "the plumbing's seed 0". The sentences of sections 14.3 and 14.4 are written as they stand.
- Marks and conditions are decided on the unrounded numbers, so a sentence can show 5.0 next to a mark of at least 5
  that was missed.

### 14.6 The verdict the project states for each outcome

`decision/DECISION.md` section 3, copied here word for word so that the verdicts sit inside the fingerprint, with one
row added at the end for the validity conditions of 9.1. The brackets of this table ([s], [phase], [description],
[d], [lb], [k], [n], [T], [F], [mark], [x]) are the write-up's own places for the look's numbers and sentences,
filled by hand from sections 14.1-14.4; the code fills nothing here. "Section 1" in the first and the last row is
`DECISION.md`'s section 1, the project's verdict on BCA as it stood before this test.

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
| **Validity fails** (the harm below 30 points or the past-80 share below 25%, section 9.1; the sentence of 14.1 "Validity fails" is written, and C2 and C3 are not written) | "No reading. The verdict of section 1 stands unchanged." | The test bed produced no harm to guard against, so it supports no conclusion about BCA either way; a version 10 only with a setting that does (the main setting produced 98.9 points of harm on seeds 0-4). |

---

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
10. **The pre-approval facts read the spent seeds 10-14** (version 8's recorded second critics; nothing trained) with
    the kill rule of 8.1; version 8's no-training checks read seeds 0-4 only.
11. **The run machine is the university cluster** (12.2, your answer to choice 4): one Slurm job per fresh seed and
    a group job, the same frozen code, approval import and manifest checks; version 8 ran on the desktop, which
    stays the fallback. The results folder and the output root are copied from the cluster to the desktop over
    your own SSH session, checked by SHA-256 there, and committed from the desktop's clone (choice 9, the default,
    your second answer), as version 8's records were.
12. **The harm and past-80 validity conditions of version 8 are kept for every claim** (9.1). `DECISION.md` left
    them out of its appendix 9.1 ("C1 has no validity condition beyond the checks of 8.1") and kept them in its 14.1
    (the "Validity fails" sentence); this document keeps the condition, which binds no side (a test bed that
    produces no harm supports no conclusion either way) and cannot bind on the record (harm 98.9, per seed 95-103;
    past-80 share 73%).

---

## 16. Choices (defaults, and what your answer takes)

Your answer, 20 October 2026, 18:29 UTC, in the chat with the cloud assistant: **"Accept the defaults and cluster"**,
and, asked whether "cluster" also meant choice 9 (where results are committed from), your second answer in the same
chat at about 19:20 UTC: **"committed from desktop after copying over from cluster"**. Both are recorded in
`OWNER_ANSWER` of `code/near80_constants.py` with their times, the place and the choice taken for each number:
every default, except that choice 4 (the machine) takes the university cluster, with the desktop kept as the
fallback; choice 9 takes its default, the desktop, after the copy from the cluster.

1. **The seed count and block.** Default: **ten seeds, 40-49**. Or five (40-44; one refusal then decides C1), fifteen
   or twenty (40-54 or 40-59; the cluster entry of 12.2 carries them in the same wall time; a version 10). Taken:
   the default.
2. **What a refusal means for the marks.** Default: **in C1, BCA's arms count as no caution in that seed-world; in C2
   and C3 the seed is left out** (5.4). Or version 8's rule (every seed is needed; a refusal fails C1, C2 and C3),
   with a 0.35 chance under the default evidence that one refusal voids the test. Or, a method change: a declared
   fallback (the hand-set range limit, a hard clip at the data's edge) when BCA refuses, scored as that fallback;
   rejected here because it would reward refusing (the range limit scored 93.1 against BCA's 88.6 in the stalling
   world on seeds 0-4 and would net about 107); it is reported as the composite "refusal means the range limit".
3. **A second held-out log per seed and world.** Default: **no**. Yes brings the chance of any refusal in ten seeds
   from about 0.35 to about 0.002 and moves BCA's height further toward the plain 90% point.
4. **The machine.** Default: **the desktop**. Or the university cluster hpc.charlotte.edu through the entry of 12.2
   (one Slurm job per fresh seed on the Cascade Lake nodes and a group job; tested in stub mode on the cluster on 20
   October 2026; the plumbing's reproduction check decides the bit-for-bit question, and a difference there moves
   the run to the desktop). **Taken: the cluster; the desktop is the fallback.**
5. **The actor phase's memory reservation.** Default: **version 7's rule** (2 at once; about 5-5.5 hours of wall). Or
   1.25 times the measured peak (9 at once; about 2.5 hours; changes `TestMemoryRule`).
6. **Lanes.** Default: **10**.
7. **The folder.** Default: **version 9 in place in `near80/`** (version 8's `--check` then passes only at commit
   `52914e4`; its chain is recorded in that commit's report). Or a new folder `near80_v9/` (eight path sites to move
   and the history logic to drop; about half a day).
8. **The twins' factors.** Default: **the plain 90% gap and twice it** (step 2's k = 1 and k = 2). Or a hand-set
   height of 3.0 for the far twin (on the recorded plateau; a number chosen after seeing seeds 0-4, labelled as such).
9. **Where results are committed from.** Default: **the desktop**, as versions 6-8. Or the cluster's login node, in
   your own SSH session under your own identity. **Taken: the default, the desktop** (your second answer, 20
   October 2026, about 19:20 UTC: "committed from desktop after copying over from cluster"): the results folder and
   the output root are copied from the cluster over your own SSH session, checked by SHA-256 on the desktop, and
   committed and pushed from the desktop's clone (12.2, step 4); nothing is committed or pushed from the login node.
10. **A fresh start bank per seed.** Default: **no**. The four fixed test start sets (`bank_starts`, one stream
    `[20261012, 2, 9]` for every seed) were pre-registered in version 6 and are the same for every arm, so they rig
    nothing; but every design number was chosen on them. Yes adds one fresh bank of 1,000 strong-current starts per
    seed (stream `[20261012, 2, 9, 1000 + seed]`), scored for every arm as a description-only readout: about 1.5
    minutes of evaluation per seed-world and its anchors, plus code in `evaluate_phase.py`.
11. **Retrain step 1's second critics on seeds 0-4** to pin their pooled heights beside seeds 10-14's. Default: **no**
    (about 1 CPU-hour of exploratory training on seen seeds; the kill rule reads seeds 10-14, which need no training).

---

## 17. What could fool us, and what we expect

**Limits, with the guard against each:**
1. **The heights shrink under pooling** (0.49-0.73 of version 8's on seeds 11 and 14), so BCA's stalling-world points
   may fall. Guard: this is the test's purpose; both heights are reported on seeds 10-14; C1's lead mark is about the
   switch.
2. **A fresh seed with very few leaked rows** (the gate's 30 logs include one with 11; pooled, about 22) refuses even
   under pooling. Guard: the outcome-free count, the refusal rule, ten seeds; the chance is about 1 in 23 per seed
   (leak factor 2.0; 1 in 17 to 1 in 38 across 1.8-2.3).
3. **The split-conformal switch is wrongly on in some no-stall seed-worlds** (0.121 and 0.116 on seeds 10 and 13
   under pooling; its margin is step 2's 0.10). Guard: it is reported, not a twin; the twins make BCA's call by the
   mean difference, so their readings are about heights.
4. **The rank test's false alarms grow with 42,000 rest rows** (one new on seed 11). Guard: reported only.
5. **The 2 x plain twin may sit past the dose plateau on some seeds** (as twice BCA's height did in the near-80
   setting). Guard: the ladder is reported per seed; the prediction says "tie or plain ahead".
6. **Training noise.** Guard: the noise floor (9.1) drops C2 and C3 if it exceeds the margin.
7. **Seeds 11 and 14 were a selected pair.** Guard: no number from them is a mark; the predictions use seeds 0-4's
   spreads and the preview.
8. **Strong current lies outside the recorded currents.** Guard: stated in every claim, as version 8.
9. **The kill rule is a check on spent seeds** (10-14) of BCA's and the mean-difference switch's calls only; it is
   not a fresh-seed test. Guard: the number of redrafts is written in section 0; the twins' calls are pinned beside.
10. **The 0.10 margin was placed in step 1 after seeing seeds 0-4** on BCA's own statistic and is carried to every
    switch arm unchanged. Guard: labelled; retuned nowhere; the switch-call table shows every disagreement.
11. **The four fixed start banks are the ones every design number was chosen on.** Guard: symmetric across arms and
    pre-registered since version 6; choice 10 offers a fresh bank per seed as a readout.
12. **Rivals getting worse is not BCA getting better.** Guard: each arm's own points are reported in both worlds.

**Predictions, written before any training (not marks):** C1 passes with 0 or 1 refusal; C2 fails with BCA ahead of
or tied with the plain gap, tied with twice it, tied with the gated range limit; C3 does not fire; the pooled heights
on the fresh seeds lie between 1.5 and 3.0; the no-caution actor scores near -14 in strong current where full throttle
stalls; the mean-difference switch makes BCA's call in 19 or 20 of the 20 seed-worlds. The expected verdict: "BCA
works as a switched harm gate on ThrusterChain (L1, pre-registered). It was not shown to beat a plain switch with a
plain height of either size, or the hand-set range limit, by 5 points (L2 not shown). Its Bayesian parts earn no
return credit (L3 not shown)."

---

## 18. Files

**Version 9** (in `outputs/shift/beyond-the-data/near80/`; nothing trains before your approval is imported):

| file | what it holds |
|---|---|
| `PREREGISTRATION.md` | this document |
| `README.md` | a one-page summary |
| `prereg_draft.json` | the machine-readable draft (the freeze moves it into `prereg_history/`) |
| `approve_local_near80.py` | the approval script you run at your own terminal (standard library only) |
| `run_near80.sh` | the desktop entry (the fallback) |
| `run_near80_cluster.sh` | the cluster entry (12.2): the self-test on the login node, then `sbatch` jobs |
| `near80_slurm_job.sh` | the job script the cluster entry submits (the pinned dependency path inside a user namespace, the job's memory, one runner per job) |
| `push_runfiles_near80.sh` | pushes the run files (model files included) to the branch `claude/near80-v9-run-files`, from the desktop (after the cluster's output root was copied there and checked, 12.2 step 4) |
| `V9_BUILD_NOTES.md` | the build notes (every change, test, estimate and the items the freeze fills); not pinned |
| `decision/` | copies of the session's decision files, kept as history, not pinned (the approval script and the freeze allow the folder as they allow the build notes; nothing reads it): `DECISION.md` (its section 3 is section 14.6 above) and `REVISION_LOG.md`, `credit.md`, `c3_rule.py` / `.out` and `review_stats_calc.py` / `.out` (the power and credit calculations behind sections 9 and 17), `PREVIEW.md` (the preview of 3.5), and a `README.md` naming them |
| `pre_approval/pooled_facts_seeds0_4_spent10_14.json` | the facts of section 4.2, written by the freeze (nothing trained) |
| `code/` | the code: `near80_constants.py` (seeds 40-49, the pooled rule, the refusal rule, the twins, `MARKS_C2` 5.0 / ceil(0.8 n) / `all_three`, `MARKS_C3` -5.0 / at most 1, `MIN_CERTIFIED` 8, `T_LO_RANGE`), `near80lib.py` (`pooled_groups` with `leak_checks`, `pooled_count_record`, `gap_rule_problem`, `certification_expectation` with the binomial chance), `facts_near80.py` (the counts on seeds 0-4 and the spent-seed switch facts), `learner_phases.py` (`switch_scores_pooled`; `switch_scores_step1` for the reproduction check), `policy_phase.py` (the per-arm refusal, the new arms, the programs run alone), `pooled.py` and `claims_text.py` (the look and section 14), the approval chain (`approval_check.py`, `import_approval.py`, `near80_prereg.py`, `freeze.py`), the runner (`near80_runner.py`, `phases.py`, `phase_main.py`, `settingdata.py`, `evaluate_phase.py`, `ledger.py`, `stubs.py`) and five unit test files (`test_near80.py`, `test_approval.py`, `test_runner.py`, `test_pooled.py`, `test_v9.py`). `near80_constants.py` honours a test-only environment variable (`NEAR80_STUB_FRESH_LABELS`) that swaps the fresh seeds for the labels 1000-1009 inside the unit tests; every official phase refuses a constants module whose seeds differ from the pinned ones. |

**Version 8's history, kept unchanged:** `prereg_history/prereg_superseded_51e677ab76c8d2f7.json` (its approved
`prereg.json`, moved there by the freeze), `code_manifest_history/code_manifest_v6.json` (its manifest),
`approval_near80_51e677ab76c8d2f7.json`, `import_near80_51e677ab76c8d2f7.json`, `amendment_1.json`,
`approval_near80_amendment1_52fe05b92900290b.json`, `import_near80_amendment1_52fe05b92900290b.json` (each with its
`.sha256`), `pre_approval/no_training_checks_seeds0_4.json`, and `results-20261020T115246Z/`. Nothing of version 9
reads them.

**Outputs** go to `runs/shift_explore/near80_v9/` under the clone that runs (the cluster clone under
`/scratch/dbayha`, or the desktop's; write-once; version 8's `runs/shift_explore/near80/` on the desktop is read by
the facts of section 4.2 and never written). Committed copies of results go to `near80/results-*/` with their
fingerprints (`git add -f`); the model files to the branch `claude/near80-v9-run-files`.
