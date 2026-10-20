# The main setting on fresh seeds: pre-registration (version 9, near-80 main)

**DRAFT, not frozen, awaiting the owner. Nothing has been trained.**

No learner, critic or actor has been trained for this version, and nothing has been frozen. This draft was built by
the session on 20 October 2026, after version 8's run on the Framework Desktop ended (19 October) and after the
check of its results (`../near80_results_check/REPORT.md`). Its open items are marked `TODO(DECISION.md: ...)`; a
parallel workflow (`DECISION.md`, `BUILD_LIST.md`) fixes them, and the freeze step (`code/freeze.py`) refuses to
freeze while any remains. Version 8's files stay in this folder as history (section 18).

**The one question this answers:** on fresh seeds, in the main test's setting (the densest 15% of recorded commands
removed), does BCA's switched ramp meet step 2's three marks once its two calibration groups are pooled over every
held-out row, and does its switch and height earn anything that a plain switch would not?

---

## 0. Status, and what this was written after

**Seeds.**
- Seeds 0-4 are seen by everything. Version 9 uses seed 0 for its plumbing only (nothing on it is a claim).
- Seeds 5-9 belong to version 6/7; 10-14 to version 8; 15-19 and 20-34 are named by other designs; 100-119 and
  900-909 by the gate and stage 0. Version 9 never touches any of them.
- Version 9's fresh seeds come from the block 40-59, which no study of this generator has used or named. The first
  five are listed in the code (`FRESH = (40, 41, 42, 43, 44)`); the count is `TODO(DECISION.md: the number of fresh
  seeds, 5, 10, 15 or 20)`. Nobody has generated, loaded or looked at them.

**Written after:**
- version 8's run (19 October 2026): claim 4 failed by the pre-set rule because on fresh seeds 10, 12 and 13 the
  held-out full-throttle group had 19, 22 and 18 rows, BCA's certificate did not certify it (its height was infinite),
  and the approved code stopped the whole actor phase of those seeds (`policy_phase.py`, `step2lib.arm_table`);
  claims 1-3 did not run (the band's stop rule on seeds 0-4);
- the check of those results (`near80_results_check/REPORT.md`): the stop is a genuine limit of the method, not a
  defect; it was foreseeable from the gate's own records (7 of 30 stalling-world seeds have fewer than 29 rows past
  80% after the removal); the switch's on/off call was right in 10 of 10 fresh seed-worlds;
- the lost analysis of version 8's records: the group is the removal ball's leakage (training-split full-throttle
  commands the ball missed, plus 0-3 calibration-bank rows); it reads no outcome; its only size guard was the gap
  rule's "at least 9 rows"; BCA's height at these sizes equals the top-1 or top-2 held-out tail score; a plain
  mean-difference test with the same 0.10 margin makes BCA's on/off call in all 20 recorded seed-worlds and, turning
  on the hand-set range limit, nets as much or more than BCA's switched ramp.

**Not seen by anyone:** any learner trained with the pooled groups, and any number from the fresh seeds.

**So:** every mark below is fixed before any learner sees the fresh seeds, except the items marked
`TODO(DECISION.md: ...)`, which DECISION.md fixes before the freeze. Nothing is tuned after the freeze.

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
  calibration bank. Version 9 pools every held-out recorded row outside the ball into them (section 3).
- **BCA's certificate**: 1,000 posterior draws per group; the switch is on when the 5% quantile of the gap between
  the groups' thresholds is above 0.10; the height is the difference of the deployed thresholds. A draw is infinite
  with probability 0.9 to the power of the group's size, and the group is certified only when at least 95% of the
  draws are finite: at least 29 rows (0.9^29 < 0.05), certain only from about 33-36 rows.
- **A refusal**: a certificate that is on with an infinite height (the full-throttle group was not certified). Step
  2's code refuses to build a switched-on arm from it.
- **The twins**: three plain switches that read the same two groups as BCA and switch the same penalty: the
  split-conformal switch (step 2's), the mean-difference switch with the ramp, and the mean-difference switch
  turning on the hand-set range limit (section 5).
- **Net across worlds**: step 2's measure, BCA's gain over no caution in the stalling world minus its loss to no
  caution in the no-stall world, strong current.
- **Points**: the exact 100-step return on a fixed test start set, scaled so that the recording controller scores 0
  and the world's top controller 100.

---

## 2. The short version

- **What changes against version 8:** one setting (the main setting), fresh seeds from an unused block, and BCA's two
  groups pooled over every held-out row outside the ball. Everything else of step 2's protocol stays: the critics,
  the cloned and no-caution actors, the switched ramp, the always-on ramp, the hand-set range limit, the matched
  rivals, the lock before any score.
- **Why pooling:** version 8's group had 18-53 rows on fresh seeds; a group below 29 rows cannot be certified. The
  calibration and evaluation splits hold about 50,000 recorded rows per seed-world, of which a few dozen are outside
  the ball and above `t_lo`. Pooling them adds those rows to the group and about 50,000 rows to the rest. It reads no
  outcome, touches no training row, and no scored test reads the pooled rows (the four fixed test start sets are
  rollouts).
- **If BCA still refuses on a seed:** that counts as BCA failing that seed, for the arms that depend on BCA's height
  only. Every other arm trains and is scored; nothing of the seed stops.
- **Two claims:** C1, version 8's claim 4 marks unchanged (step 2's three go marks); C2, BCA's switched ramp beats
  each plain twin, paired by seed, on the net across worlds, with margins `TODO(DECISION.md: the margin and the seed
  count)` and a multiplicity rule `TODO(DECISION.md: all three twins, or each separately)`.
- **Cost:** about 2.2 CPU-hours per fresh seed on the desktop (version 8's ledger, plus the two new arm calls and
  the pooled scores) and about 2.5 for the plumbing; section 13 gives 5, 10, 15 and 20 seeds on the desktop and on
  the cluster.

---

## 3. The data and the pooled groups, exactly

### 3.1 The removal

The main test's ball, exactly as before (`shift.define_gap`, `shift.make_removal(log, 'gap', gap)`), bit for bit, in
both worlds. The training rows a learner sees, the calibration bank, the test start sets and every rival's own rule
are unchanged.

### 3.2 The pooled groups

For one seed and world, with `t_lo` from the gap rule on the ball-kept training commands:

1. **The held-out rows** are every row of the calibration split and of the evaluation split whose command is outside
   the ball. The ball's rows are hidden everywhere, as version 8 hid them from the bank.
2. **The full-throttle group** is the training-split survivors (ball-kept training rows with throttle index above
   `t_lo`, which step 1 already scored and holds out from BCA's second critic) together with the held-out rows with
   throttle index above `t_lo`.
3. **The rest** is the held-out rows with throttle index at or below `t_lo`.

Version 8's groups (the survivors and the calibration bank) are a subset of these: the bank is part of the
calibration split. The second critic is unchanged: it trains on the ball-kept training rows minus the survivors, so
it has seen no full-throttle command, and every row it scores is held out from it. Every switch arm reads the same
two groups (`policy_phase.arm_row_from` reads one switch record per world). The code: `near80lib.pooled_groups`,
`learner_phases.switch_scores_pooled`.

### 3.3 What it does not change

- The training rows, the training file and its checks (`near80lib.training_file_check`).
- The test start sets (nothing recorded there) and the points.
- Every rival's own rule: the hand-set range limit's recorded range, the matched rivals' ladders and matcher, the
  always-on ramp's height (BCA's stalling-world height), the perfect band.
- Step 2's programs: BCA's switched ramp and the plumbing check train in step 2's own batches (section 5.3), so the
  plumbing's reproduction check on seed 0 still reproduces step 2's saved actors bit for bit, when given step 1's own
  switch entries.

### 3.4 The outcome-free count

Before any training, when a seed's data is made, the code counts the pooled groups (`near80lib.pooled_count_record`,
written into the data record): the full-throttle group's size, the rest's size, the survivors, and whether the group
reaches 29 rows. It reads commands and split labels only. It is reported (section 14.3); no mark reads it; it fails
no seed. For scale it also counts version 8's groups.

---

## 4. The facts from version 8 (nothing trained for this version)

From version 8's results (`results-20261020T115246Z/`, fresh seeds 10-14, main setting, stalling world):

| seed | version 8's full-throttle group | finite share of the draws | certified | switch | height |
|---|---|---|---|---|---|
| 10 | 19 | 0.879 | no | on | infinite |
| 11 | 53 | 0.997 | yes | on | 2.289 |
| 12 | 22 | 0.903 | no | on | infinite |
| 13 | 18 | 0.85 | no | on | infinite |
| 14 | 30 | 0.964 | yes | on | 4.595 |

On seeds 0-4 the same group had 35, 47, 59, 43 and 47 rows, all certified. The switch's on/off call was right in 10
of 10 fresh seed-worlds (on in the stalling world, off in the no-stall world). The freeze step records the pooled
groups' sizes on seeds 0-4 (`pre_approval/pooled_group_counts_seeds0_4.json`), outcome-free, nothing trained.

---

## 5. Arms

### 5.1 The host

TD3+BC in step 2's actor-only protocol, unchanged: each world's critic trained on its own kept rows with the gate's
call (100,000 updates), then frozen; actors from that world's cloned actor, 20,000 steps in 4,000-step chunks, alpha
5, the gate's fixed batch stream; arms of one kind side by side, one processor per phase.

### 5.2 The main setting, both worlds

| arm | role |
|---|---|
| BCA's switched ramp, k = 1 (`obca_k1`) | **primary** (C1 and C2) |
| the ramp with height 0 and k = 0 (`w0_check`) | plumbing check: must equal the no-caution actor within 0.00001 |
| the ramp at k = 2, BCA's switch with height 1.0, the rank-test switch | reported only (step 2's) |
| the split-conformal switch (`conformal_k1`) | step 2's; **a twin of C2** |
| new: the mean-difference switch with the ramp (`meandiff_k1`) | **a twin of C2** |
| new: the mean-difference switch turning on the hand-set range limit (`meandiff_limit`) | **a twin of C2** |
| new: BCA's switched ramp trained alone (`obca_k1_alone`) | the noise-floor copy (section 8.5); reported only |
| the always-on ramp (at BCA's stalling-world height) | world-blind rival |
| range limit, matched; even pull and two random surfaces, matched | world-blind rivals |
| push less, matched | reported |
| the hand-set range limit (`range_hinge`) | reported |
| perfect band | reference |
| no-caution, cloned and full-log actors | references |

- **The twins read the same groups as BCA.** The split-conformal switch is on when the 90% quantile of the
  full-throttle group's scores minus the rest's is above 0.10, with that gap as its height. The mean-difference
  switch is on when the mean of the full-throttle group's scores minus the mean of the rest's is above 0.10, with
  that difference as its height (version 8's reported `meandiff_k1`). Turning on the hand-set range limit: when the
  mean-difference switch is on, the arm is the hand-set range limit (the room's hinge at the recorded range,
  coefficient 1.0, trained alone in its own program); when off, the ramp with height 0 (checked equal to the
  no-caution actor). None of the three uses BCA's posterior or its certificate.
- **The noise-floor copy** is BCA's switched ramp with BCA's own switch and height, trained alone in a ramp program
  of one instead of step 2's six-arm program. It measures what the program's batch does to an actor's bits and so
  to its points (section 8.5).
- **Matching:** as step 2, blind to returns, to the primary arm's stalling-world strength, with version 6's ladders
  and matcher; the no-stall world uses the stalling world's locked settings.

### 5.3 One training program

The code trains the ramp arms of one batch side by side, and the batch size changes the last bits of each actor. So
BCA's switched ramp and the plumbing check train in step 2's own batches: where full throttle stalls, the plumbing
check alone, then the six step-2 ramp arms with BCA's switched ramp first; in the no-stall world, one batch of seven
with the plumbing check first. Then, in calls that leave those programs' bits unchanged: the mean-difference ramp
alone, the mean-difference range limit alone (a hinge, or a height-0 ramp), and the noise-floor copy alone. A refused
arm's place in the six-arm batch is filled by a height-0 pad, never saved or scored, so the other members keep step
2's batch size.

### 5.4 Fixed references (no seed, nothing trained)

The recording controller, the stall-aware and rising-thrust controllers (the points' anchors, in both worlds).

### 5.5 Left out, and why

Version 8's band and random-rows settings (the band's stop rule failed on seeds 0-4: a hand-set range limit lost only
4.3 points to it); version 6's BCA (it depends on the band setting's readouts); the readouts at band rows (no band).

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
strengths and matches; BCA's switch, height, 5% bound, ramp ends and group sizes per seed and world, with the refusal
when there is one; time and peak memory per phase.

---

## 8. The tests, in order

Every threshold is fixed now (the `TODO(DECISION.md: ...)` items are fixed before the freeze). None moves after its
numbers exist, except through a labeled post-hoc pre-registration that you approve.

### 8.1 The gate's checks on each fresh seed

When a fresh seed's data is made: the gate's ball must be found, the ball's leak checks must pass, and in the
stalling world the gate's five world checks must pass. A seed that fails is a failed seed for every mark, written
down and never replaced (section 10).

### 8.2 The outcome-free count

Section 3.4, in the same data phase, before any learner: reported in the look and in section 14.3's sentence. It
neither stops nor fails anything.

### 8.3 Plumbing, after approval (scratch only)

- The unit tests (five files).
- A 40-step smoke of every phase kind on seed 0, in scratch, with the pooled groups.
- The fresh-seed code path on seed 0: it must reproduce the gate's seed-0 data record, and the record written must
  read back (amendment 1's lesson).
- **The reproduction check:** on seed 0 in both worlds, the near-80 code must reproduce bit for bit the gate's
  critic, cloned actor and no-caution actor, step 1's second critic and certificate (on version 8's groups: the
  survivors and the bank, the only groups step 1 computed), and, through the official policy function given step 1's
  own switch entries, step 2's saved plumbing-check, switched-ramp and hand-set range-limit actors. The pooled
  groups' certificate of seed 0 is computed beside it and reported, not compared. If any item differs, nothing
  official starts, ever, in this version.
- The first peak-memory measurements of each phase kind.
- Only a reproduction difference is final. Any other failure can be retried once its cause is fixed.

### 8.4 The refusal rule

If, on a fresh seed, BCA's certificate is on with a height that is not finite in either world, BCA has refused on
that seed. The switch record says so (`refused: true`). The policy phase then leaves out the arms that depend on
BCA's height, BCA's switched ramp, the ramp at k = 2, the always-on ramp and the noise-floor copy, and trains and
scores every other arm. With no BCA actor there is no strength to match, so the matched rivals and the stand-in of
that seed are unmatched, as in version 8 when BCA's switch stayed off. By the pre-set rule the seed fails every mark
that needs BCA (C1's three marks and C2's), and only those: its own clause in section 14 says so, and the other arms'
numbers are reported. Nothing of the seed stops. (Version 8 stopped the whole actor phase.)

### 8.5 The noise floor

Pooled over the fresh seeds that neither failed nor refused, stalling world, strong current: BCA's switched ramp
minus the noise-floor copy (the same arm trained alone) is at most 5 points in size. If it is larger, C2 is dropped
and only reported: its margin would sit inside the noise of the training program's batch. The number is reported
either way. When no seed is left to read it, it cannot be read, and C2 is written as its other rules say. (Version
8's noise floor compared the random-rows and main settings; without the random-rows setting, this is the adapted
form.)

### 8.6 The fresh seeds' run

In order, per seed: the data of both worlds (8.1, 8.2); the critics, the full-log critic, BCA's second critics; the
switches on the pooled groups; the cloned, no-caution and full-log actors; the policy phases with matching (8.4),
each seed-world locked (every actor's fingerprint written) before any score; the evaluations; then the look.

**Plumbing stops, per phase:** the height-0 check, a non-finite network, a saved input whose fingerprint does not
match, or a training file that fails its check. A stopped phase is written once and never rerun. Every mark that
needs it fails.

**Nothing on the fresh seeds stops early on an outcome.**

---

## 9. Pass marks, with reasons and power

Each mark is pooled over the fresh seeds: the mean of each seed's paired difference. Stalling world, strong current,
unless stated. Every seed is needed. A failed, stopped or refused seed fails every mark that needs it.

### 9.1 Validity conditions (without them, no claim either way)

- The harm is at least 30 points: the full-log actor beats the no-caution actor by at least 30, pooled.
- The no-caution actor goes past 80% on at least 25% of strong-current steps, pooled.
- Both are computed over the seeds that did not fail (a refused seed's no-caution and full-log numbers count: they
  do not depend on BCA). On the main setting on seeds 0-4 these were 98.9 points and 73.2%.

### 9.2 C1: version 8's claim 4 marks, unchanged

Step 2's three go marks on the fresh seeds, main setting:

| mark | pass | reason |
|---|---|---|
| net lead | BCA's net across worlds is at least the best world-blind rival's plus 10 | step 2's margin |
| keeps the gain | BCA minus no caution is at least -3 in recorded conditions, big waves and big kicks | version 6's margin |
| beats the even pull | BCA minus the matched even pull is at least 15 | version 6's margin |

### 9.3 C2: BCA's switched ramp beats each plain twin

For each twin, paired by seed: BCA's net across worlds minus the twin's net across worlds, pooled, is at least
`TODO(DECISION.md: the margin)` points, and positive in at least `TODO(DECISION.md: the seed count)` of the fresh
seeds. The multiplicity rule, `TODO(DECISION.md: 'all_three' or 'each_separately')`, says whether C2 is one claim
that holds only when every twin is beaten, or one claim per twin, each read on its own. The code reads them from
`MARKS_C2` in `code/near80_constants.py`; the freeze refuses while any is `None`.

- **Why the net across worlds:** a switch earns its keep in both worlds, on in the stalling world and off in the
  no-stall world; the net rewards both calls at once.
- **What the twins separate:** where BCA and a twin make the same on/off call, the comparison is of heights only; the
  mean-difference switch with the range limit replaces the ramp by a hard limit at the data's edge.
- **The refused seeds:** a refused seed counts as BCA failing that seed against every twin.
- **The noise floor** (section 8.5) can drop C2.

### 9.4 Rules about the marks

- No mark moves after its numbers exist.
- The two claims are read separately. Neither rescues the other.
- The plumbing's seed 0 never carries a claim.

---

## 10. The fresh seeds

- **Data:** new logs of both worlds, with the gate's recorder, sizes and settings (stall slope 1.5). Then the ball,
  its leak checks and the gate's five world checks (required for every claim), the training file, the outcome-free
  count.
- **A fresh seed that fails** a data check, or whose learner fails, fails every mark. A refused seed fails the marks
  that need BCA (section 8.4). A failed seed is written down and never replaced.
- **Learners:** every seed-world is made fresh: critics, BCA's second critics, cloned actors, the full-log actors,
  every arm of section 5.
- **Frozen from this pre-registration:** the pooled groups' rule, every arm, ladder, matching rule and mark, the seed
  list.
- **Old code that refuses these seeds is restated, never edited.** Version 6's fresh-seed code and step 2's runner
  are restated for the fresh seeds in new code.

---

## 11. Approval

The route of versions 6, 7 and 8, on the Framework Desktop, inside Ubuntu as root, from `/home/user/bca`:
`python3 -I outputs/shift/beyond-the-data/near80/approve_local_near80.py --check`, then the same without `--check`.
The script checks every file against the code manifest, refuses any file the manifest does not list other than
version 8's history (section 18), refuses a draft, and asks you to type the first 16 characters of `prereg.json`'s
fingerprint. The session imports your approval (`code/import_approval.py import`) and pushes the import record.
Every phase, on the desktop or on the cluster, refuses without it.

**Approving accepts every default** in section 16. Any other choice makes a new fingerprint.

---

## 12. Hosts and order of work

### 12.1 The desktop

`run_near80.sh`, as version 8: root inside WSL, tmux, one runner, 1-16 lanes, version 7's memory rule. Version 8's
checks on version 7's run are gone (it finished on 16 October 2026).

### 12.2 The cluster (hpc.charlotte.edu)

`run_near80_cluster.sh` submits Slurm jobs (account `staruser`, partition `Orion`, 48-CPU nodes, 30-day limit) that
run the same frozen code from a clone under `/scratch/dbayha`, never as root, with the same approval import and
manifest checks, and a git identity rule that refuses the approver's and the cloud's identities. One job runs the
plumbing; then one job per fresh seed (its 17 phases, both worlds); then one job for the group's lock and look. Each
job's runner starts only its own seed's phases and shares one output root; every phase holds its own lock, so no
phase runs twice. The runner reads the job's memory from Slurm and reserves within it. Results are gathered and
pushed with `run_near80.sh results` and `push_runfiles_near80.sh` from the cluster (the owner's GitHub sign-in on the
cluster), to the branches of section 18. What the desktop entry assumed and the cluster entry changes is listed in
`V9_BUILD_NOTES.md` and in `run_near80_cluster.sh`'s header.

### 12.3 Order

1. **Done:** this draft, its machine-readable twin (`prereg_draft.json`), the code and its tests.
2. **DECISION.md** fixes the open items (seed count, C2's margins, the multiplicity rule, the owner's answer).
3. The session freezes (`code/freeze.py write`): the draft into history, the manifest, the tests, the outcome-free
   counts on seeds 0-4 (nothing trained), `prereg.json`.
4. **You approve** (section 11). The session imports it.
5. Plumbing (section 8.3), on the host you choose.
6. The fresh seeds' run and the look.
7. Results committed with their fingerprints; model files to the run-files branch.

---

## 13. Compute

From version 8's ledger of the main setting on seeds 10-14 (one processor; `results-20261020T115246Z`): critic 5.7
minutes, full-log critic 5.7, second critic 5.7 per world, switch 0.3, clone 0.7, reference actors 3.5 (stalling) and
2.2, the stalling-world actor phase 42 and the no-stall one 9.3, the evaluation 1.5 per world; about 1.8 CPU-hours per
seed. Version 9 adds three ramp calls alone and one hinge (about 0.3 CPU-hours) and the pooled scores (about 50,000
rows, seconds). So about 2.2 CPU-hours per fresh seed, about 2.5 for the plumbing (the reproduction check at full
length is most of it).

| fresh seeds | CPU-hours (plumbing included) | desktop wall time (2 policy phases at once) | cluster wall time (one 48-CPU node, one job per seed) |
|---|---|---|---|
| 5 | about 13-14 | about 6-7 hours in two sittings | about 2.5 hours (the plumbing) + about 1.5 hours |
| 10 | about 24-25 | about 11-13 hours in three sittings | about 2.5 + 1.5 hours |
| 15 | about 35-36 | about 16-19 hours in four sittings | about 2.5 + 1.5 hours (two nodes: the same) |
| 20 | about 46-47 | about 21-25 hours in five sittings | about 2.5 + 1.5 hours |

- **Lanes:** 10 by default on the desktop; on the cluster a seed's job asks for 10 processors (its chain runs at most
  a few phases at once) and the plumbing for 10.
- **Memory:** version 7's rule, unchanged on the desktop (at most 2 policy phases at once). On the cluster each job
  asks Slurm for 24 GB and the runner reserves within it (7 GiB per unmeasured kind).
- **Budget guard:** a forecast of 2.2 CPU-hours per seed times the seed count; a stage that uses more than twice its
  forecast pauses and reports; the cap is 2.5 times the forecast. A pause lasts until you type
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
| D1, K1 | BCA's net across worlds minus the split-conformal switch's, pooled; seeds where it was positive |
| D2, K2 | the same against the mean-difference switch with the ramp |
| D3, K3 | the same against the mean-difference switch with the hand-set range limit |
| DT, KT | the same for the twin a clause names |
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
> over every held-out row, BCA's switched ramp met all three pre-set marks. Its switch turned caution on in [S] of
> [NSEEDS] seeds where full throttle stalls and off in [N] of [NSEEDS] where it does not. In strong current it gained
> [X] points over no caution where full throttle stalls (the full-log controller gained [H]) and lost [W] where full
> throttle helps: a net of [NET], [D] more than the best rule that cannot switch itself off ([R]). It stayed within 3
> points of no caution or better in the other three situations, and beat the matched even pull by [E]."

**Any mark fails:**
> "On fresh seeds [SEEDS], in the main setting with BCA's two groups pooled over every held-out row, BCA's switched
> ramp did not meet its pre-set marks: [one clause per missed mark]."

The clauses:
- "its net across worlds of [NET] was [D] above the best rule that cannot switch itself off ([R]), below the 10
  required"
- "it lost [L] points to no caution in [situation], more than the 3 allowed"
- "it beat the matched even pull by [E] in strong current, below the 15 required"
- "seed [s] failed [check] and, by the pre-set rule, counts as failing every mark"
- "on seed [s] BCA's certificate did not certify its pooled full-throttle group of [n] rows ([world]; the height was
  infinite), so by the pre-set rule BCA fails this seed; the other arms were trained and scored"

**Validity fails:**
> "On fresh seeds [SEEDS], the main setting did not produce the harm it was built to test ([reason]). It supports no
> conclusion about BCA either way."

The reasons: "the controller trained without caution lost only [H] points to the full-log controller (at least 30
needed)", or "the controller trained without caution went past 80% on only [P]% of strong-current steps (at least 25%
needed)".

### 14.2 C2: BCA's switched ramp against each plain twin

**Holds (every twin beaten):**
> "On fresh seeds [SEEDS], BCA's switched ramp beat each plain switch that read the same two groups, paired by seed on
> the net across worlds: the split-conformal switch by [D1] points (positive in [K1] of [NSEEDS] seeds), the
> mean-difference switch with the ramp by [D2] ([K2] of [NSEEDS]) and the mean-difference switch with the hand-set
> range limit by [D3] ([K3] of [NSEEDS]); at least TODO(DECISION.md: the margin) points and TODO(DECISION.md: the seed
> count) of [NSEEDS] seeds were required for each."

**Fails:**
> "On fresh seeds [SEEDS], BCA's switched ramp did not beat every plain switch that read the same two groups: [one
> clause per twin not beaten]."

The clauses:
- "[twin], by [DT] points on the net across worlds (positive in [KT] of [NSEEDS] seeds; at least TODO(DECISION.md:
  the margin) points and TODO(DECISION.md: the seed count) seeds were required)"
- "on seed [s] BCA's certificate did not certify its pooled full-throttle group of [n] rows ([world]), so by the
  pre-set rule BCA fails this seed against every twin"

**Each separately, a twin beaten:**
> "On fresh seeds [SEEDS], BCA's switched ramp beat [twin] by [DT] points on the net across worlds, paired by seed
> (positive in [KT] of [NSEEDS] seeds; at least TODO(DECISION.md: the margin) points and TODO(DECISION.md: the seed
> count) seeds were required)."

**Each separately, a twin not beaten:**
> "On fresh seeds [SEEDS], BCA's switched ramp did not beat [twin]: [DT] points on the net across worlds, paired by
> seed (positive in [KT] of [NSEEDS] seeds; at least TODO(DECISION.md: the margin) points and TODO(DECISION.md: the
> seed count) seeds were required)."

**Dropped by the noise floor:**
> "The comparison with the plain switches was dropped, because BCA's switched ramp trained alone differed from the
> same ramp trained in step 2's program by [V] points, pooled (at most 5 allowed): C2's margin would sit inside the
> noise of the training program. Its numbers are reported as description."

**Margins not fixed:**
> "C2's margins were not fixed when this look was written (DECISION.md): no claim is read. The numbers against each
> plain switch are reported as description."

**If a seed failed:**
> "By the pre-set rule, C2 fails because seed [seeds] failed (C1 says why); the numbers of the other seeds are
> reported as description only."

**If every seed failed:**
> "By the pre-set rule, C2 fails because seed [seeds] failed (C1 says why)."

### 14.3 The pooled groups (reported)

**Always, for the pooled groups:**
> "Before any training, the pooled full-throttle group had [NMIN] to [NMAX] rows over the [NSW] seed-worlds, and
> [NREACH] of them reached 29 rows, the smallest size BCA's certificate can certify; version 8's groups would have had
> [N8MIN] to [N8MAX] rows."

### 14.4 Stops

- **Reproduction fails:** "Stopped: the new code did not reproduce [the main test's removal / critic / cloned actor /
  no-caution actor, or step 2's plumbing-check / switched-ramp / hand-set range-limit actor] on seed 0. No result of
  this version stands until it does."
- **BCA refuses on a seed-world:** "Seed [s], [world]: BCA's certificate did not certify its pooled full-throttle group
  of [n] rows (the height was infinite). By the pre-set rule BCA fails this seed; every other arm was trained and
  scored."
- **A phase stops:** "Seed [s], [world]: [phase] stopped ([reason]). It was not rerun. Every mark that needs it
  fails."

### 14.5 How the code fills in the sentences

- Numbers are written with one decimal ("{:.1f}"), with a minus sign when negative and no plus sign; "-0.0" is
  written "0.0". [P] is a share of steps written as a percentage. [S], [N], [K1], [K2], [K3], [KT], [NSEEDS], [NMIN],
  [NMAX], [NREACH], [NSW], [N8MIN], [N8MAX], [n] and [s] are whole numbers. [SEEDS] is the seeds' range, "40-44".
- A number that cannot be computed, because a seed failed, a phase stopped, BCA refused or an arm is missing, is
  written "not available". A mark that needs it fails.
- [R] is written as "the always-on ramp", "the matched range limit", "the matched even pull", "the first matched
  random surface" or "the second matched random surface"; ties go to the first in step 2's order. [twin] is "the
  split-conformal switch", "the mean-difference switch with the ramp" or "the mean-difference switch with the
  hand-set range limit".
- [situation] is "recorded conditions", "big waves" or "big kicks". [world] is "stalling world", "no-stall world" or,
  for a refusal in both, "stalling world and no-stall world". [phase] is the phase's plain name in the runner's
  phase table.
- [one clause per missed mark] lists the clauses of the missed marks, joined by "; ", in this order: the net lead;
  keeping the gain (one clause per situation that misses); beating the even pull; then one clause per failed seed, in
  seed order; then one clause per refused seed, in seed order. [one clause per twin not beaten] lists the twins'
  clauses in the order split-conformal, mean-difference ramp, mean-difference range limit, then the refused seeds'
  clauses. A missed mark whose number is not available because a seed failed or refused is told by that seed's own
  clause only.
- [check] is "the ball's leak checks", "the gate's five world checks on the ball alone", "its [phase] phase" for a
  stopped phase, "its evaluation (numbers missing)" for a seed whose evaluation lacks a number, or "its matching (a
  rival could not be matched or scored)" for a seed whose matched rivals could not be matched (that seed fails C1's
  marks only).
- [reason] in C1's validity sentence is the harm reason when the harm is below 30 points, the past-80 reason when the
  share is below 25%, and both, joined by "; ", when both fail.
- In the reproduction stop, the bracket lists every item that differed, written "A", "A and B" or "A, B and C".
- C2 is written only when C1's validity holds, in this order of precedence: the margins not fixed; dropped by the
  noise floor; a seed failed (the "If a seed failed" sentence, or "If every seed failed" when no seed is left); then
  "Holds" or "Fails" under the rule 'all_three', or one "Each separately" sentence per twin under 'each_separately'.
  Wherever a sentence says "seed [seeds]", it reads "seed 41" for one seed, and "seeds 41 and 43" or "seeds 40, 41
  and 43" for more.
- On the plumbing's seed 0, every claim sentence starts with the plumbing prefix and "fresh seeds [SEEDS]" reads
  "the plumbing's seed 0". The sentences of sections 14.3 and 14.4 are written as they stand.
- Marks and conditions are decided on the unrounded numbers, so a sentence can show 5.0 next to a mark of at least 5
  that was missed.

---

## 15. Departures from version 8, fixed now

1. **One setting**, the main setting; no band, no random rows, no band readouts, no version 6 arm.
2. **BCA's two groups pooled** over every held-out recorded row outside the ball (section 3).
3. **The refusal rule:** BCA refusing fails the seed for the BCA-dependent arms only; nothing stops (section 8.4).
4. **The outcome-free count** of the pooled groups, recorded before any training (section 3.4).
5. **Two new trained arms**, the twins of C2, plus the noise-floor copy.
6. **Fresh seeds from an unused block** (40-59), the count fixed by DECISION.md.
7. **No exploratory stage**, no exploratory look and no stop rules on seeds 0-4; seed 0 serves the plumbing only.
8. **The noise floor** adapted to the training program's batch (section 8.5).
9. **A cluster entry** beside the desktop entry (section 12.2).

---

## 16. Choices (defaults; the owner's acceptance is `TODO(DECISION.md: the owner's answer)`)

1. **The setting.** Default: **the main setting only**. Or the near-80 band as well (version 8's claims 1-3 on fresh
   seeds; about +2.5 CPU-hours per seed).
2. **The seeds.** Default: **the first N of the block 40-59**, N fixed by DECISION.md. Or another unused block.
3. **The pooled groups.** Default: **every held-out row outside the ball** (the calibration and evaluation splits).
   Or the calibration split alone (about half the rows).
4. **The refusal rule.** Default: **the BCA-dependent arms only** fail; every other arm trains. Or version 8's stop.
5. **The noise floor.** Default: **BCA's switched ramp trained alone**, at most 5 points. Or none.
6. **Hosts.** Default: **TD3+BC, actor-only**.
7. **Lanes.** Default: **10**.
8. **The name.** Default: **"version 9, near-80 main"**.
9. **Where it runs.** Default: **the desktop**; the cluster entry is ready (section 12.2).

---

## 17. What could fool us, and what we expect

- **The pooled rest is large** (about 50,000 rows): BCA's threshold on the rest becomes tight, so the height may
  grow. Guard: the heights are reported per seed and world, and C2 compares BCA with switches that read the same
  rest.
- **The full-throttle group may still be small** on some seed: a refusal. Guard: the refusal rule and the count.
- **A twin with the same call and a smaller height is a dose comparison**, as step 2 said. Guard: C2 credits nothing
  to BCA's posterior; it says only that the switched ramp beat each twin's actor.
- **Rivals getting worse is not BCA getting better.** Guard: each arm's own points are reported in both worlds.
- **The batch changes bits.** Guard: the noise floor.

**Predictions (not marks):** the pooled groups certify on most fresh seeds; the mean-difference switch makes BCA's
on/off call on every seed; C2's outcome depends on the heights, which nobody has seen.

---

## 18. Files

**Version 9** (in `outputs/shift/beyond-the-data/near80/`; nothing trains before your approval is imported):

| file | what it holds |
|---|---|
| `PREREGISTRATION.md` | this document |
| `README.md` | a one-page summary |
| `prereg_draft.json` | the machine-readable draft (the freeze moves it into `prereg_history/`) |
| `approve_local_near80.py` | the approval script you run at your own terminal (standard library only) |
| `run_near80.sh` | the desktop entry |
| `run_near80_cluster.sh`, `near80_slurm_job.sh` | the cluster entry (sbatch) and its job script |
| `push_runfiles_near80.sh` | pushes the run files (model files included) to the branch `claude/near80-v9-run-files` |
| `V9_BUILD_NOTES.md` | the build notes (every change, test, estimate and placeholder); not pinned |
| `code/` | the code (section 18 of version 8's document lists the roles; `near80lib.py` now holds the pooled groups and the count, `facts_near80.py` the counts on seeds 0-4, `test_v9.py` the new tests) |

**Version 8's history, kept unchanged:** `prereg_history/prereg_superseded_51e677ab76c8d2f7.json` (its approved
`prereg.json`, moved there by the freeze), `code_manifest_history/code_manifest_v6.json` (its manifest),
`approval_near80_51e677ab76c8d2f7.json`, `import_near80_51e677ab76c8d2f7.json`, `amendment_1.json`,
`approval_near80_amendment1_52fe05b92900290b.json`, `import_near80_amendment1_52fe05b92900290b.json` (each with its
`.sha256`), `pre_approval/no_training_checks_seeds0_4.json`, and `results-20261020T115246Z/`. Nothing of version 9
reads them.

**Outputs** go to `runs/shift_explore/near80_v9/` (write-once; version 8's `runs/shift_explore/near80/` is never
written). Committed copies of results go to `near80/results-*/` with their fingerprints; the model files to the branch
`claude/near80-v9-run-files`.
