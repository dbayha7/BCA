# The deciding test: restructured BCA on fresh seeds with every held-out row: pre-registration (version 9)

**DRAFT. Not frozen. Nothing has been trained.**

No learner, critic or actor has been trained for this version, and no fresh seed has been generated. The
machine-readable twin is `prereg_draft.json` beside this file. You approve the fingerprint of the final `prereg.json`
as section 11 says.

**Your request** (20 October 2026): "WE ARE LONG OVERDUE SOLVING THE BIG MAIN PROBLEM OF BCA WORKING OR NOT. TOP
PRIORITY NOW".

**The one question this answers:** on fresh seeds, and with enough held-out rows that BCA can act, does BCA's switched
ramp work? Does BCA beat the same pipeline with BCA's part swapped for a plain switch turning on a hand-set limit? Does
its height beat plain split conformal's? (On this task, with equal weights, no test can credit BCA's Bayesian part
itself: section 9.3.)

---

## 0. Status, and what this was written after

**Seeds.**
- Seeds 35-49 of the main generator (master 20261012) are fresh. Only they can carry a claim. Nobody has generated,
  loaded or looked at them.
- Seeds 0-4 are seen, and seeds 10-14 are spent. They enter only through the preview, which trains no actor
  (section 4), and the plumbing.
- Seeds 5-9, 15-19, 20-34, 100-119 and 900-909 are refused in every mode.

**Written after:**
- version 8's results (`near80/results-20261020T115246Z/`): claim 4 failed by its rule, because seeds 10, 12 and 13
  stopped on BCA's refusal with 19, 22 and 18 held-out full-throttle rows. Claims 1-3 were never run (stop rule 8.3:
  4.3 < 5);
- the checkers' report on version 8 (`near80_results_check/REPORT.md`, 52914e4);
- five read-only analyses of the records (verdict, refusal, credit, process, steelman; 20 October 2026), the decision
  note that fixed this version's claims, marks and verdict table (draft 1), its three reviews and draft 2;
- a first cloud preview on seeds 0-4 and 10-14 (both groups pooled, no top-up, no actor trained), read after draft 1
  and before draft 2: it rebuilt all 20 recorded version 8 switch entries exactly; BCA and the mean-difference switch
  made the right call in 20 of 20 seed-worlds, split conformal (step 2's rank) in 18 and the rank test in 15; BCA
  refused in none (6 of 20 with version 8's groups); one full-throttle group was below 36 (stalling seed 10, 35 rows,
  which certified); BCA's stalling-world heights shrank (seeds 0-4: mean 2.78 to 1.69) and stayed above split
  conformal's by 0.21 to 1.27 (section 4, Table 1);
- the cloud's outcome-free counts of the pooled group on seeds 0-4, 10-14 and the gate's records: the pooled
  full-throttle group held 35-146 rows (median 78) over 35 stalling seed-worlds and 37-135 (median 69) over 15
  no-stall seed-worlds; none was below 29 and one below 36 (stalling seed 10, 35 rows), against 13 below 29 and 25
  below 36 with version 8's groups (section 4);
- the pinned preview, with this version's code: [TODO(DECISION.md): PREVIEW SUMMARY: switch calls of every rule,
  heights, counts by source, top-up blocks].

**Not seen by anyone:** any number from seeds 35-49.

**So:** every claim, mark, bar, test and verdict sentence below was fixed before any fresh seed exists. The previews
could only have stopped this draft (section 4). The changes made after the first preview are listed in section 15
with their reasons.

---

## 1. Words used

Version 8's section 1 holds, except for the band words, which do not apply here. New or changed words:
- **Held-out rows (H)**: the recorded rows of the calibration split (20,000) and the evaluation split (30,000) that lie
  outside the ball, plus any top-up rows. No critic or actor trains on them.
- **Top-up blocks**: 100 new recorded episodes each (10,000 rows), made by the same recording controller in the same
  world, from a new registered random stream. Their rows are held-out rows too. They are never trained on.
- **BCA's full-throttle group**: the survivors, plus every held-out row past the gap rule's lower end. **The rest**:
  every held-out row at or below it.
- **The count**: the size of BCA's full-throttle group, from commands only, written before any critic exists.
- **Refused**: BCA's threshold R_tail is +infinity (fewer than 95% of its 1,000 draws cross), so a BCA arm that needs
  that height has no finite height.
- **Split-conformal switch (textbook)**: in each group, the ⌈(n+1)·0.9⌉-th smallest held-out score. It is on when
  the full-throttle group's value minus the rest's is above 0.10; its height is that gap.
- **Mean-difference switch**: on when the mean held-out score of the full-throttle group minus the rest's is above
  0.10.
- **The plain switch with the hand-set limit**: in each world, the hand-set range limit's actor where the
  mean-difference switch is on, and the no-caution actor where it is off. It is a deployment rule over two trained
  actors, so it needs no training of its own.
- **D, L, U**: a seed's paired difference of NET, and the two-sided 95% interval of their mean over the fresh seeds.
- **Missing for a reason of BCA's own / not BCA's**: section 9.0.

## 2. The short version

- **What changes:**
  - BCA's groups get every held-out recorded row outside the ball; the full-throttle group is topped up to 36 if
    needed. 36 is BCA's own certifiable size.
  - A refusal now fails only BCA's arms on that seed. It no longer stops the seed.
- **What does not change:** BCA itself, the data the learners train on, the four test start sets, and step 2's
  three marks.
- **What is tested:** step 2's result on 15 fresh seeds (claim 1). Then two paired comparisons: whether BCA's height
  beats plain split conformal's by 5 points (claim 2a, credited as a dose), and whether BCA beats a plain switch with
  a hand-set limit by 5 points or is ruled out from doing so (claim 2b).
- **What the result will be called:** section 14 gives the sentence for every outcome, written now. The expected one
  is "BCA works as a harm switch, but is not needed".
- **Cost:** about 26 processor-hours, or about 9 hours on your Framework Desktop in one sitting.

## 3. The evidence change, exactly

### 3.1 BCA's two groups

- **Held-out rows H** = `split_rows('calibration') ∪ split_rows('evaluation')`, minus the rows inside the ball
  (`gap.contains`, the rule that empties the bank of ball rows), plus the top-up rows (3.2), minus theirs inside the
  ball.
- **Full-throttle group** = the survivors + the rows of H whose throttle index t = max_k |a_k| is above t_lo.
  **Rest** = the rows of H with t ≤ t_lo.
- The scores are version 8's: one-sided one-step over-predictions of the second critic, with no noise. The second
  critic trains on the kept training rows minus the survivors, unchanged.
- The certificate is version 8's: `step1lib.certificate`, which calls `wbcp.calibrate` with α 0.1, credibility 0.95,
  1,000 draws, a 5% statistic and margin 0.10, under the same random tags.

### 3.2 The count and the top-up

- At data time, before any critic exists, the count is formed per seed-world. It reads the stored commands, the ball
  and the gap rule only.
- It is written to the data record beside BCA's certifiable bar (`reference.certifiable`: 36 at 0.1 / 0.95 / 1,000)
  and the exact floor (29).
- If the count is below 36, top-up blocks 1, 2, ... are added to that world until its count reaches 36, at most 10
  blocks. Each world takes its own number of blocks.
- Block k's starts and noise come from `default_rng([20261012, seed, 61, k])` and its coin from
  `default_rng([20261012, seed, 62, k])`. Neither depends on the world. Both were checked unused on every branch at
  the freeze: [TODO(DECISION.md): TAG CHECK]. They are simulated exactly as `dataset.generate` does (same recorder,
  physics, length and float32 commands).
- The top-up rows' arrays are hashed into the data record. Nothing else changes: the training rows, bank, evaluation
  split and test start sets are bit-identical to what version 8's code would build.

### 3.3 A refusal

- In a world where BCA's switch is on and R_tail is +infinity, BCA's switched ramp and the twice-height ramp are
  recorded as refused in that world, with no actor and no number.
- If the stalling world's R_tail is +infinity, whatever the switch, the always-on ramp is refused in both worlds,
  because it uses the stalling world's height.
- If the stalling world's switched ramp is refused, the matched rivals are recorded as unmatched, because they have
  no target.
- Every other arm trains and is scored. BCA's switch with height 1.0 is never refused (its height is finite).
- A refused arm's place in step 2's six-arm program is filled with a height-0 pad. The pad is checked equal to no
  caution, and is never saved or scored. So every other arm's bits are unchanged.
- In a world where BCA's switch is off, its height there is 0, as in version 8.

### 3.4 What every switch reads

BCA's certificate, the textbook split-conformal switch, the mean-difference switch, the rank-test switch, the exact
Wilks twin (reported only) and the fixed-height arms all read the same two groups.

## 4. What was seen before freezing (the previews; no actor trained)

**Table 1, the first cloud preview (20 Oct).** Written 20 October 2026 at 14:36 UTC (`PREVIEW.md` and `preview.json`
in the cloud session's preview folder `v9preview/`), after draft 1 of the decision note (14:17 UTC) and before draft 2.
- Main seeds 0-4 (seen) and 10-14 (spent), main setting, both worlds. Both groups pooled, no top-up. No actor was
  trained and no official phase ran.
- It first rebuilt all 20 recorded version 8 switch entries of these seed-worlds with version 8's own code: every one
  equals its record, with 0 differences in any field. Step 1's 10 second critics of seeds 0-4 were reproduced in the
  cloud bit for bit: each file's SHA-256 equals the one step 1's `results.json` records.
- Split conformal here is step 2's rank, as in version 8. The textbook rank of section 1 is in Table 2.
- A call is right when it is on in the stalling world and off in the no-stall world. A wrong call is in bold.
- The height column is R_tail - R_rest. Where BCA's switch is off, the height used is 0, as in version 8.

| seed | world | full-throttle group: survivors + bank + other calibration rows + evaluation rows = n | rest | BCA's q05 | BCA's call | R_tail | R_rest | height | split conformal: call (q90 gap) | mean difference: call (mean gap) | rank test: call (p) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | stalling | 34 + 1 + 17 + 18 = 70 | 42089 | 1.135 | on | 2.455 | -0.035 | 2.490 | on (1.537) | on (0.605) | on (9.6e-19) |
| 0 | no-stall | 37 + 1 + 17 + 27 = 82 | 42124 | 0.044 | off | 0.317 | -0.035 | 0.352 | off (0.087) | off (0.047) | off (0.062) |
| 1 | stalling | 45 + 2 + 28 + 31 = 106 | 42027 | 1.020 | on | 1.674 | -0.041 | 1.715 | on (1.227) | on (0.596) | on (3.1e-40) |
| 1 | no-stall | 34 + 2 + 24 + 24 = 84 | 42026 | 0.012 | off | 0.035 | -0.043 | 0.078 | off (0.041) | off (-0.013) | off (0.51) |
| 2 | stalling | 59 + 0 + 24 + 38 = 121 | 42447 | 0.879 | on | 1.347 | -0.033 | 1.379 | on (1.093) | on (0.512) | on (4.8e-45) |
| 2 | no-stall | 52 + 0 + 21 + 32 = 105 | 42469 | 0.006 | off | 0.033 | -0.042 | 0.075 | off (0.026) | off (0.021) | **on** (0.022) |
| 3 | stalling | 42 + 1 + 21 + 36 = 100 | 42176 | 1.010 | on | 1.574 | -0.038 | 1.611 | on (1.164) | on (0.520) | on (9.1e-35) |
| 3 | no-stall | 28 + 1 + 14 + 23 = 66 | 42210 | 0.017 | off | 0.107 | -0.039 | 0.146 | off (0.069) | off (0.030) | **on** (0.0054) |
| 4 | stalling | 45 + 2 + 28 + 35 = 110 | 41505 | 0.879 | on | 1.219 | -0.038 | 1.256 | on (1.049) | on (0.538) | on (1.8e-48) |
| 4 | no-stall | 58 + 2 + 25 + 50 = 135 | 41476 | -0.011 | off | 0.016 | -0.030 | 0.046 | off (0.017) | off (0.000) | **on** (0.034) |
| 10 | stalling | 18 + 1 + 5 + 11 = 35 | 42997 | 1.247 | on | 2.736 | -0.040 | 2.775 | on (2.480) | on (0.779) | on (7.2e-15) |
| 10 | no-stall | 15 + 3 + 6 + 13 = 37 | 42997 | 0.017 | off | 0.177 | -0.042 | 0.219 | **on** (0.121) | off (0.028) | off (0.14) |
| 11 | stalling | 51 + 2 + 21 + 35 = 109 | 41788 | 0.908 | on | 1.630 | -0.041 | 1.671 | on (1.200) | on (0.507) | on (3.6e-36) |
| 11 | no-stall | 30 + 2 + 16 + 15 = 63 | 41818 | 0.014 | off | 0.086 | -0.036 | 0.122 | off (0.065) | off (0.034) | **on** (0.017) |
| 12 | stalling | 21 + 1 + 9 + 15 = 46 | 42547 | 1.165 | on | 1.813 | -0.038 | 1.851 | on (1.639) | on (0.645) | on (2.4e-20) |
| 12 | no-stall | 20 + 1 + 10 + 16 = 47 | 42565 | -0.001 | off | 0.041 | -0.037 | 0.079 | off (0.008) | off (0.008) | off (0.35) |
| 13 | stalling | 18 + 0 + 5 + 15 = 38 | 42762 | 0.646 | on | 1.810 | -0.036 | 1.845 | on (1.408) | on (0.396) | on (1e-10) |
| 13 | no-stall | 36 + 0 + 11 + 29 = 76 | 42734 | 0.062 | off | 0.131 | -0.035 | 0.165 | **on** (0.116) | off (0.012) | off (0.19) |
| 14 | stalling | 30 + 0 + 13 + 15 = 58 | 41899 | 0.906 | on | 2.220 | -0.036 | 2.256 | on (0.987) | on (0.594) | on (3.3e-22) |
| 14 | no-stall | 31 + 0 + 13 + 20 = 64 | 41922 | 0.011 | off | 0.159 | -0.033 | 0.192 | off (0.044) | off (0.054) | **on** (0.0017) |

Source: `preview.json` `per_seed_world` (`v9`), one row per seed-world.

What Table 1 shows:
- **Right calls, of 20:** BCA 20; the mean difference 20; split conformal 18 (false alarms in no-stall seeds 10 and
  13); the rank test 15 (false alarms in no-stall seeds 2, 3, 4, 11 and 14). BCA and the mean difference made the
  same call in all 20. With version 8's groups the counts were 20, 20, 18 and 18; the rank test's three new false
  alarms (no-stall seeds 3, 4 and 11) are the only calls that pooling changed.
- **No refusal.** BCA refused in 0 of 20 seed-worlds, against 6 of 20 with version 8's groups (stalling seeds 10, 12
  and 13; no-stall seeds 3, 10 and 12).
- **One group below 36:** stalling seed 10, with 35 rows. It certified, with a finite share of 0.980. No group was
  below 29.
- **The rest group** held 41,476-42,997 rows (version 8: 825-878). R_rest moved little: -0.043 to -0.030 pooled,
  against -0.039 to -0.014 with version 8's groups.
- **Distinct episodes.** The full-throttle groups' rows lay in 0.88-1.00 as many distinct episodes as rows; every rest
  group spans all 500 held-out episodes. The one-row-per-episode variant (one row per episode, drawn with
  `default_rng([20261020, 9901, seed, world, 1 or 2])`) changed no call and moved the stalling heights by -0.21 to
  +0.13.
- **BCA's heights shrank**, stalling world: seeds 0-4 from 6.02 / 2.42 / 2.06 / 1.61 / 1.80 to 2.49 / 1.72 / 1.38 /
  1.61 / 1.26 (mean 2.78 to 1.69); seeds 11 / 14 from 2.29 / 4.60 to 1.67 / 2.26; seeds 10 / 12 / 13, refused before,
  2.78 / 1.85 / 1.85.
- **BCA's height stayed above split conformal's** on every stalling seed, by 0.21 to 1.27 (mean 0.51), against 0.41
  to 4.29 (mean 1.77, over the 7 seeds that certified) with version 8's groups.

**The outcome-free counts, every seed-world on record** (the preview's `count_gate.py`; it reads the stored
commands, the ball and the gap rule only: no reward, no next state, no critic). The seed-worlds are the gate's data
records (stalling seeds 0-4, 100-119 and 900-904; no-stall seeds 0-4 and 100-104) and the spent seeds 10-14 in both
worlds.

| world | seed-worlds | version 8's groups: min / median / max | below 29 | below 36 | pooled: min / median / max | below 29 | below 36 |
|---|---|---|---|---|---|---|---|
| stalling | 35 | 11 / 36 / 63 | 10 | 17 | 35 / 78 / 146 | 0 | 1 (seed 10: 35) |
| no-stall | 15 | 18 / 32 / 60 | 3 | 8 | 37 / 69 / 135 | 0 | 0 |

- The pooled group is 1.67-3.36 times version 8's in the stalling world (median 2.20) and 1.88-2.38 times in the
  no-stall world (median 2.16).
- Expected refusals, the exact binomial at 1,000 draws summed over the seed-worlds: 10.0 (stalling) and 3.65
  (no-stall) with version 8's groups; 2.5e-6 and 4.6e-9 pooled.
- A lognormal model of the pooled stalling count (a model, not a record) puts P(all certify) without the top-up at
  about 0.96, 0.93 and 0.90 for 5, 10 and 15 fresh seeds.

Source: `preview.json` `outcome_free_counts`.

**The scores read before freezing.** Every scored number anyone has seen, with its label. All of them were read before
the decision note's draft 1 fixed the claims, marks, bar and verdict table. None is from a fresh seed.
- **Main setting, seeds 0-4 (exploratory):** step 2's records (`restructure/step2_results/`) and version 8's own
  evaluation of these seeds. Step 2's three marks: net lead +25.3, keeps the gain +8.0 / +21.2 / +19.8, beats the even
  pull +100.8. Computed from recorded arms: BCA's switched ramp netted 102.9 and the plain switch with the hand-set
  limit 107.4; BCA's ramp led the split-conformal ramp by 5.6 on average. Every height in these records comes from
  version 8's groups, not the pooled ones.
- **Spent seeds 10-14 (version 8, `near80/results-20261020T115246Z/`):** seeds 10, 12 and 13 have no scores (their
  stalling-world actor phases stopped on BCA's refusal, and no evaluation record exists for either world). Seeds 11
  and 14 were scored in both worlds and read as description only: BCA's switched ramp netted 106.3 / 109.5, the plain
  switch with the hand-set limit 108.6 / 110.0 (computed from recorded arms), BCA's ramp led the split-conformal ramp
  by 27.6 / 25.4, and the full-log actor led no caution by 97.9 / 98.3 in the stalling world.
- **The near-80 and random-rows settings, seeds 0-4 (version 8, exploratory):** the hand-set range limit's band cost
  was 4.3 against the 5 needed, so those settings stopped (stop rule 8.3); in the near-80 setting the plain switch
  with the hand-set limit netted 103.0 against BCA's 100.6 (computed).
- **Across these records**, BCA was ahead of the plain switch with the hand-set limit in 5 of 17 seed pairs (mean
  difference -3.2, computed). These numbers feed only the planning guides and predictions of sections 9.2-9.4 and 17,
  never a mark.

**Table 2, the pinned preview.** [TODO(DECISION.md): TABLE 2, the pinned preview, the same columns plus: the top-up
blocks; distinct episodes; step 2's uncorrected split-conformal gap; the Wilks twin; the one-row-per-episode
variant.]

The pre-set no-go rules: no approval is asked, and the design is redrafted, if BCA's pooled switch makes a wrong call
on any of these 20 seed-worlds, or if any stays below 36 rows after the top-up. A redraft may change only the
evidence rule. [TODO(DECISION.md): RESULT: passed / the draft was redrafted because ...]

## 5. Arms

**Host:** TD3+BC in step 2's actor-only protocol, unchanged (version 8, section 5.1).

| arm | role |
|---|---|
| BCA's switched ramp, k = 1 | **primary** (claims 1, 2a, 2b) |
| the ramp with height 0 | plumbing check: must equal the no-caution actor within 0.00001 |
| the split-conformal switch (textbook), in step 2's `conformal_k1` slot | **claim 2a's comparison** |
| the plain switch with the hand-set limit (made from the hand-set range limit and no caution, not trained) | **claim 2b's comparison** |
| the ramp at k = 2; BCA's switch with height 1.0; the rank-test switch | reported only: no test, no claim |
| new: the mean-difference switch with ramp heights 2.5 and 5.0 (their own program), trained only in a world where that switch is on | reported only; they feed claim 2a's dose sentence |
| the always-on ramp (BCA's stalling-world height) | world-blind rival |
| range limit, even pull and two random surfaces, all matched | world-blind rivals |
| push less, matched | reported |
| hand-set range limit | reported; feeds claim 2b |
| perfect band | reference |
| no-caution, cloned and full-log actors | references |

- **Matching:** blind to returns, to the primary arm's stalling-world strength, with version 6's ladders and matcher.
  The no-stall world uses the stalling world's locked settings, as in step 2.
- **One program per batch, as in version 8.** The plumbing check trains alone. Step 2's six ramp arms train together,
  with BCA's switched ramp first. The two fixed-height arms train together where their switch is on. The hand-set
  range limit trains alone.

## 6. Situations

Version 8's section 6, unchanged: recorded conditions, big kicks, big waves and strong current. **The marks are read
in strong current.**

## 7. Measures

These are version 8's measures 1, 3, 5, 6, 7 and 8 (points, zone shares, NET, strengths, the switch fields, the
worst-100 tail, time and memory), plus:
- the count by source, the top-up blocks and the distinct episodes;
- every switch's call and height in every seed-world;
- the paired differences D of claims 2a and 2b in all four situations, with their stalling-world and no-stall parts.

The band readouts (version 8's measures 2, 4, 9 and 10) do not apply.

## 8. The tests, in order

### 8.1 The no-training checks, per fresh seed, at data time

- The gate's five world checks on the ball.
- The ball's leak checks.
- New: no held-out or top-up row is a training row, and none lies in the ball.
- New: the count is unchanged when rewards and next states of the recorded rows are scrambled.
- New: the top-up blocks' hashes, and their streams' tags.
- The training-file check.

A fresh seed that fails one of these is a **failed seed**, missing for a reason that is not BCA's (section 9.0).

### 8.2 Plumbing, after approval (scratch only)

These are version 8's, plus three items:
- the unit tests;
- a 40-step smoke of every phase kind, including a forced top-up and a forced refusal;
- the first memory peaks;
- **the reproduction check**:
  - (a) with version 8's groups and step 2's own arm table, the runner reproduces step 2's saved plumbing-check,
    switched-ramp and hand-set range-limit actors on seed 0, in both worlds, bit for bit;
  - (b) the pooled switch reproduces the pinned preview's record for seed 0, both worlds;
  - (c) a written fresh-path data record is read back through the phase wrapper.

  If any of them differs, nothing official starts. A named code defect may be fixed through the amendment route; with
  no named defect, the version ends.

`run_near80_v9.sh plumbing-then-run` starts the confirmatory stage only after the plumbing has passed.

### 8.3 The confirmatory stage: seeds 35-49

The steps:
- data, counts and top-ups;
- the critics and second critics;
- the switches;
- the cloned, no-caution and full-log actors;
- the actor phases;
- each seed-world locked before any score is computed;
- the evaluations;
- the confirmatory look.

There is no exploratory training stage, and nothing on seeds 35-49 stops early on an outcome.

## 9. Pass marks, with reasons and power

Each number is computed per fresh seed. Stalling world, strong current, unless stated. The marks are read on
unrounded numbers.

### 9.0 Missing seeds and imputation

- **Missing for a reason of BCA's own:** a refusal; BCA's switch off in the stalling world; BCA's switched-ramp actor
  not finite. Any of these fails claim 1.
- **Missing for a reason that is not BCA's:** a failed seed (8.1); a plumbing stop of a phase; a rival-side failure (a
  matched rival unmatched although BCA's strength is positive, a rival network not finite, an error inside the
  stand-in).
- **Imputation:** each missing per-seed number a claim reads is replaced by that quantity's least favourable observed
  value over the other fresh seeds: the smallest observed lead for a pass or "shown", the largest for "excluded". At
  most 2 of 15 seeds may be missing for reasons not BCA's for claim 1; each claim 2 contrast needs its own numbers on
  at least 13 of 15 seeds.
- A crash, a killed attempt or an out-of-memory attempt is rerun from scratch and makes no seed missing.

### 9.1 Validity (without it, no claim either way)

- The harm (full-log actor minus no caution) must be at least 30 points, pooled over seeds whose numbers exist.
- The no-caution actor must go past 80% on at least 25% of steps.
- On seeds 0-4: 98.9 points, and 73.2% past 80% (version 8's text).

### 9.2 Claim 1: step 2's result on fresh seeds

Step 2's three go marks, unchanged:

| mark | pass | reason | on seeds 0-4 (exploratory) |
|---|---|---|---|
| net lead | BCA's net across worlds is at least the best world-blind rival's plus 10 | step 2's margin; about 4 standard errors | +25.3 (standard error 2.2) |
| keeps the gain | BCA minus no caution is at least -3 in recorded conditions, big waves and big kicks | version 6's margin; about the noise of 1,000 starts | +8.0, +21.2, +19.8 |
| beats the even pull | BCA minus the matched even pull is at least 15 | version 6's margin | +100.8 (standard error 4.8) |

- Any number missing for a reason of BCA's own fails every mark of claim 1.
- More than 2 seeds missing for reasons not BCA's make claim 1 unreadable. If claim 1 fails only through imputed
  values, it is reported as not readable, not as BCA's failure.
- **Planning:** at the pooled heights the net lead is expected at about 21-27 (a guide from 7 recorded seeds).

### 9.3 Claim 2a: BCA's height against plain split conformal's (a dose)

- **What is compared:** per seed, D = NET(BCA's switched ramp) minus NET(the split-conformal switch). Same rows, same
  margin, same ramp; only BCA's certificate is swapped for the textbook 90% point.
- **The test:** D̄ and its two-sided 95% interval [L, U] over the fresh seeds, with t(0.975, n - 1) (2.1448 at 15).
  - **Shown** if D̄ ≥ 5 and L > 0.
  - **Not shown** otherwise. An upper end below 5 is printed but is not read as ruling a 5-point lead out: the
    differences can be strongly skewed (seeds 11 / 14 gave +27.6 / +25.4), and under such shapes no 15-seed rule
    bounds the mean from above at 2.5%.
- **Why 5:** the project's standing bar for attributing a gain (step 2's attribution rule, version 6's tolerance
  unit, version 8's claims 2 and 3).
- **What it credits:** BCA's height is an upper credible limit of the full-throttle group's 90% point, so it is larger
  than split conformal's by construction; with equal weights it is Wilks' distribution-free upper tolerance limit. A
  pass credits a larger, data-chosen height (a dose). By step 2's attribution rule, nothing is credited to BCA's
  Bayesian part or to calibration.
- **Power** (planning): a guide from the recorded dose curves at the pooled heights puts the mean D at about 5.3-6.7,
  so "shown" has chance about 0.65-0.99; at a true lead of 0 it has chance at most 0.011 under every recorded shape.

### 9.4 Claim 2b: BCA against the plain switch with the hand-set limit

- **What is compared:** per seed, D = NET(BCA's switched ramp) minus NET(the plain switch with the hand-set limit).
- **The test:** as claim 2a, with three outcomes: **shown** if D̄ ≥ 5 and L > 0; **excluded** if U < 5; **unresolved**
  otherwise. Under the recorded shape of these differences, P(excluded) at a true lead of exactly 5 is 0.021.
- **Readings of an excluded claim 2b** (fixed now): U < 0, "the plain rule did better"; 0 ≤ U < 5 and L > -5, "BCA and
  the plain rule were within 5 points of each other"; 0 ≤ U < 5 and L ≤ -5, "the plain rule was within 5 points of BCA
  or ahead".
- **Why this comparison:** it is the best pipeline without BCA on record. It netted 107.4 against BCA's 102.9 on seeds
  0-4 (exploratory, computed from recorded arms).
- **Power:** if the recorded mean of about -3 holds, or the pooled-height guide's -8, it is excluded with chance about
  1.00. **"Shown" is practically unreachable**: the plain rule scores 89.8-96.4 of 100 in the stalling world, so BCA
  would need about 98-100.

### 9.5 Rules about the marks

- No mark moves after its numbers exist.
- Claims 1, 2a and 2b are read separately. None rescues another.
- Every statement about one contrast is read off its one two-sided 95% interval; each one-sided statement errs at
  most 2.5%. Any pair of statements about the two contrasts errs at most 5% family-wise; "both shown" holds at 2.5%
  by intersection-union.
- Claim 2 passes only if 2a and 2b are both shown.

## 10. Stops, failed seeds and refusals (the complete list)

Only the events below change what runs or what a claim reads. A unit test scans the code and checks that every stop,
halt and refusal appears in this list.

- **A plumbing failure before any official phase:**
  - a reproduction difference: nothing official starts; a named code defect may go through the amendment route, and
    plumbing reruns; otherwise the version ends;
  - any other failure: retried after its cause is fixed (version 8, section 8.2).
- **A failed seed** (a check of section 8.1): missing for a reason that is not BCA's; imputed (9.0); never replaced.
- **A refusal** (section 3.3): not a stop; the seed fails claim 1; claim 2 imputes; every other arm's numbers are
  reported.
- **BCA's own switched-ramp actor not finite; BCA's switch off where full throttle stalls:** the seed fails claim 1.
- **A rival-side failure:** missing for a reason that is not BCA's; imputed.
- **A plumbing stop of a phase** (a training file, a fingerprint, a failed critic, cloned or reference actor, the
  height-0 check, a built-in check, including a gap rule that differs from step 1's): written once and never rerun;
  missing for a reason that is not BCA's; imputed.
- **An interrupted attempt** (out of memory, killed): rerun from scratch.
- **A failed attempt, or a halt of the runner:** the run halts; nothing is deleted; the cause is fixed (a code defect
  only through the amendment route), and the phase is retried from scratch under the same seed and inputs. Recorded.
- **The budget guard:** a pause, lifted by `run_near80_v9.sh budget-continue <stage>`. No result is read differently.
- **The count below 36 after 10 top-up blocks:** recorded. The certificate runs as usual.
- **Validity fails:** no conclusion either way.

## 11. Approval

Version 8's section 11, with version 9's names:
- `approve_local_near80_v9.py --check`, then without `--check`, at your own terminal, in Ubuntu as root, at
  `/home/user/bca`;
- it writes `approval_near80v9_<16>.json` and its `.sha256`;
- you commit and push them under your own identity;
- the session imports with `code/import_approval.py import` and pushes `import_near80v9_<16>.json`;
- every training phase refuses without it.

The approval script also requires the pinned preview's result file, by SHA-256. Its amendment mode is kept from
version 8, for defects only; `code/near80_constants.py` is never amended.

No agent writes your approval file or makes that commit, and no message counts as approval.

## 12. Compute

| part | processor-hours | wall-clock (version 8's memory rule) |
|---|---|---|
| preview (cloud) | about 0.3-0.5 | no desktop time |
| plumbing | about 1.2-1.5 | about 1 hour |
| confirmatory, seeds 35-49 | about 24 | about 7.6-8 hours |
| **total on the desktop** | **about 26** | **about 9 hours, one sitting** |

- **Memory:** version 8's rule, unchanged. An actor phase reserves the larger of 7 GiB and twice its measured peak, so
  at most 2 run at once.
- **Budget guard:**
  - forecasts: plumbing 2.5, confirmatory 1.7 × 15 processor-hours;
  - pause factor 2;
  - cap 45 processor-hours.

## 13. The claims, word for word, for each outcome

| bracket | meaning |
|---|---|
| S, N | seeds where BCA's switch was on (stalling world), off (no-stall world), of 15 |
| NET, D, R, E, L | as version 8, section 14 |
| Da, La, Ua; Db, Lb, Ub | claim 2a's and 2b's mean paired difference and its 95% interval |
| Da_s, Da_n | claim 2a's mean difference in the stalling and the no-stall world |
| hB, hC | mean height of BCA and of the split-conformal switch, stalling world |
| F, hF | the better fixed height's mean NET minus BCA's, and that height (2.5 or 5.0) |
| K2 | twice BCA's height: mean NET minus BCA's |
| nT | median size of BCA's full-throttle group, stalling world; [k] top-up blocks used in all |
| m | the number of seeds whose missing numbers were imputed, and why |

**Always attached:** "One simulated task, two worlds, fifteen fresh seeds (35-49), four fixed test start sets of
1,000 starts each. In strong current the test states lie outside the recorded currents. BCA's groups used every
held-out recorded row outside the hidden region (full-throttle group: median [nT] rows), with [k] blocks of new
held-out episodes added before training." If m > 0: "On [m] seeds a number was missing ([reasons]); by the pre-set
rule it was replaced by the least favourable value observed on the other seeds."

### 13.1 Claim 1

**All three marks pass:**
> "On fresh seeds 35-49, in the main setting, step 2's result held. BCA's switched ramp met all three pre-set marks.
> Its switch turned caution on in [S] of 15 seeds where full throttle stalls and off in [N] of 15 where it does not.
> In strong current it netted [NET] across the two worlds, [D] more than the best rule that cannot switch itself off
> ([R]). It stayed within 3 points of no caution or better in the other three situations, and beat the matched even
> pull by [E]. Where full throttle stalls and its switch was on, its controller is the same as the always-on ramp's,
> so that gain comes from where the ramp starts and from its height, not from the switch. Pre-registered history:
> this claim failed in version 8 with the as-built held-out group (BCA refused on 3 of 5 seeds); it passed here with
> every held-out row plus [k] new held-out blocks."

**Any mark fails on BCA's own numbers:**
> "On fresh seeds 35-49, in the main setting, step 2's result did not hold: [one clause per missed mark, as version
> 8]."

**The clause for a refused seed:**
> "on seed [s], BCA could not set its height even with every held-out row plus [k] new held-out blocks ([n] rows): it
> refused, and by the pre-set rule claim 1 fails."

**Not readable:**
> "Claim 1 could not be read, because [m] seeds lack numbers for reasons that are not BCA's ([reasons]). This is not
> evidence about BCA."

### 13.2 Claim 2a

- **Shown:**
  > "BCA's switched ramp beat the same ramp with split conformal's switch and height, read from the same held-out rows
  > with the same 0.10 margin, by an average of [Da] points of net in strong current (an average of at least 5 and a
  > lower end above 0 were required; 95% interval [La] to [Ua]): [Da_s] in the stalling world and [Da_n] in the
  > no-stall world. Its height averaged [hB] against split conformal's [hC]. BCA's height is an upper limit of the
  > same rows' 90% point; with equal weights it is Wilks' distribution-free upper tolerance limit, larger by
  > construction. So a larger dose did better. By the pre-set rule nothing is credited to BCA's Bayesian part or to
  > calibration. Twice BCA's height scored [K2] relative to BCA. [Dose clause.]"
  - Dose clause, if F > 2: "A hand-picked fixed height of [hF] on the plain switch did better by [F] points, so the
    credit is for a large enough height, not for how BCA chooses it."
  - Dose clause, otherwise: "No hand-picked fixed height on the plain switch beat BCA's by more than 2 points ([F])."
- **Not shown:**
  > "BCA's switched ramp led the same ramp with split conformal's switch and height by an average of [Da] points (95%
  > interval [La] to [Ua]). By the pre-set rule this does not show a credit of 5 points for BCA's height; it does not
  > rule one out either."
- **Too few seeds:**
  > "Claim 2a is not shown by the pre-set rule, because seeds [s] have no [arm] number ([reason])."

### 13.3 Claim 2b

- **Shown:**
  > "BCA's switched ramp beat the plain switch with the hand-set limit (a mean-difference switch on the same rows and
  > margin, turning on the hand-set range limit) by an average of [Db] points of net in strong current (an average of
  > at least 5 and a lower end above 0 were required; 95% interval [Lb] to [Ub]). That plain rule needs no BCA."
- **Excluded:**
  > "A BCA lead of 5 points or more over the plain switch with the hand-set limit is ruled out by the pre-set rule:
  > BCA's mean lead was [Db] (95% interval [Lb] to [Ub]). [Reading.]"
  - Reading, if Ub < 0: "The plain rule did better."
  - Reading, if 0 ≤ Ub < 5 and Lb > -5: "BCA and the plain rule were within 5 points of each other."
  - Reading, otherwise: "The plain rule was within 5 points of BCA or ahead."
- **Unresolved:**
  > "BCA's switched ramp led the plain switch with the hand-set limit by an average of [Db] points (95% interval [Lb]
  > to [Ub]). By the pre-set rule this neither shows nor rules out a lead of 5 points."
- **Too few seeds:**
  > "Claim 2b is unresolved by the pre-set rule, because seeds [s] have no [arm] number ([reason])."

### 13.4 Validity fails

> "On fresh seeds 35-49, the main setting did not produce the harm it was built to test ([reason]). It supports no
> conclusion about BCA either way."

### 13.5 Stops

These are version 8's sentences for a reproduction difference and for a stopped phase, plus:

> "Seed [s], [world]: BCA refused (R_tail infinite with [n] held-out full-throttle rows, after [k] top-up blocks). Its
> arms that need that height have no numbers on this seed. Every other arm trained and is reported."

## 14. The verdict, word for word

After the claims, the look writes exactly one verdict, the first row that applies in this order. These are section 3
of DECISION.md's sentences, with the brackets of section 13.

| when | the verdict |
|---|---|
| validity fails | "No conclusion about BCA either way." |
| claim 1 is not readable (reasons not BCA's) | "The deciding test could not read its first claim, because of [reasons] on seeds [s]. This is not evidence about BCA. As built, BCA failed its pre-registered test; with enough held-out rows it is not shown either way." |
| claim 1 fails on BCA's numbers | "BCA does not work on ThrusterChain's main setting. On fresh pre-registered seeds, with every held-out row available, its switched ramp failed [the missed marks]." |
| claim 1 fails by a refusal only | "BCA could not act on seed [s] even with every held-out row plus [k] new held-out blocks ([n] rows). It refused, and by the pre-set rule claim 1 fails." |
| claim 1 fails because BCA's switch stayed off where full throttle stalls | "BCA's switch stayed off on seed [s], where full throttle stalls; by the pre-set rule claim 1 fails." |
| claim 1 passes; 2a shown; 2b shown | "BCA works on ThrusterChain and beat the best plain rule: the plain switch with a hand-set range limit by an average of [Db] points (at least 5 required; 95% interval [Lb] to [Ub]), and split conformal's height by [Da], a larger-dose effect. Nothing is credited to BCA's Bayesian part." |
| claim 1 passes; 2a shown; 2b excluded | "BCA works as a harm switch on ThrusterChain, and its larger upper-limit height did better than split conformal's 90% point (by an average of [Da] points; a dose effect, not credit for its Bayesian part). But BCA is not needed: a BCA lead of 5 points or more over the plain switch with a hand-set limit is ruled out, and [reading] (mean [Db], 95% interval [Lb] to [Ub])." |
| claim 1 passes; 2a shown; 2b unresolved | "BCA works as a harm switch on ThrusterChain, and its larger upper-limit height did better than split conformal's 90% point (by an average of [Da] points; a dose effect). Whether BCA beats the plain switch with a hand-set limit by 5 points is neither shown nor ruled out (mean [Db], 95% interval [Lb] to [Ub])." |
| claim 1 passes; 2a not shown; 2b shown | "BCA's switched ramp works on ThrusterChain and beat the plain switch with a hand-set limit by an average of [Db] points (at least 5 required; 95% interval [Lb] to [Ub]). Its height did not beat split conformal's by the 5-point bar. The credit belongs to BCA's pipeline as a whole, not to its Bayesian part." |
| claim 1 passes; 2a not shown; 2b excluded | "BCA works as a harm switch on ThrusterChain, but nothing specific to BCA earned credit. Its height did not beat split conformal's by the 5-point bar (mean [Da], 95% interval [La] to [Ua]), and a BCA lead of 5 points or more over the plain switch with a hand-set limit is ruled out: [reading] (mean [Db], [Lb] to [Ub])." |
| claim 1 passes; 2a not shown; 2b unresolved | "BCA works as a harm switch on ThrusterChain, but no BCA-specific credit was shown. Its height did not beat split conformal's by the 5-point bar, and a 5-point lead over the plain switch with a hand-set limit is neither shown nor ruled out (mean [Db], [Lb] to [Ub])." |

## 15. Departures from version 8, fixed now

1. Main setting only. No band, no random-rows control, no near-80 stop rules, no exploratory training stage.
2. **BCA's groups read every held-out row outside the ball**; the full-throttle group is topped up to 36 from new
   held-out episodes, counted from commands only before training. (Changed after the first preview: draft 1 kept
   version 8's rest group. The first preview showed no switch call of BCA, the mean difference or split conformal
   changes with the pooled rest, and the build already used it.)
3. **A refusal fails BCA's arms only.** It is not a stop. Every other arm trains.
4. The split-conformal arm uses the textbook rank ⌈(n+1)·0.9⌉. Step 2's uncorrected gap is reported. The plumbing
   reproduction trains step 2's own arm table.
5. New arms: the mean-difference switch at fixed ramp heights 2.5 and 5.0 (reported only, trained only where that
   switch is on). The plain switch with the hand-set limit is made from two trained arms. Near-80's mean-gap-height
   switch and the knee arms are left out.
6. New claims 2a and 2b, with a 5-point bar and a two-sided 95% interval. Claim 2a has no "excluded" outcome
   (changed after the reviews, from a shape analysis of version 8's records).
7. **Missing seeds:** a reason of BCA's own fails claim 1; a reason that is not BCA's (a failed data check, a plumbing
   stop, a rival-side failure) is imputed with the least favourable observed value, up to 2 of 15 seeds. Version 8
   failed every mark of claim 1 on any failed seed, and counted an error inside the stand-in against BCA.
8. A validity condition on the main setting itself: harm at least 30, past 80% at least 25%.
9. Fresh seeds 35-49; refused 0-34, 100-119 and 900-909 (0-4 and 10-14 enter only the previews and the plumbing).
10. A complete stop table, tested against the code by a scan of every stop, halt and refusal (version 8's section 10
    omitted one stop, and one stop had free text).
11. The preview runs in the cloud before the freeze, pinned by fingerprint. Plumbing reads a fresh record back and
    ties the frozen code to the preview. `plumbing-then-run` starts the run only after the plumbing passed.
12. The results step refuses if `.gitignore` would drop any file it wrote.

## 16. Choices (defaults)

1. Fresh seeds: **15 (35-49)**, or 10 (35-44).
2. Evidence: **the calibration and evaluation splits plus the top-up**, or the calibration split plus the top-up.
3. Memory rule: **version 8's**, or twice the measured peak.
4. The two fixed-height arms: **yes**, or no.
5. An exploratory retrain of seeds 0-4 at pooled heights (reported only): **no**, or yes.
6. Branch: **`claude/near80-v9`**.

## 17. What could fool us, and what we expect

**Limits, with the guard against each:**
1. **More rows lower BCA's height.**
   - BCA's height moves from the group's largest row toward its 95th percentile (seen seeds: mean 2.78 to 1.69).
   - Guard: every height is reported, and the sentences say what claim 2a credits.
2. **Within-episode dependence** could make the pooled posterior too narrow.
   - Guard: distinct episodes and a one-row-per-episode variant are reported; below 80% distinct episodes in a
     stalling full-throttle group, the claim 2 sentences cite the variant.
3. **The 0.10 margin was set after seeing seeds 0-4** (step 1).
   - Guard: it is the same for every switch, and labelled.
4. **The fixed heights 2.5 and 5.0 were chosen after seeing the dose curve on seeds 0-4.**
   - Guard: they are reported only, and their only use, the dose sentence, can only take credit away from BCA.
5. **The plain switch with the hand-set limit is not a trained arm.**
   - It is exact by definition: it deploys one of two trained actors per world. A switched-off ramp equals no caution
     within 0.00001 (the plumbing check).
6. **The certificates read evaluation-split outcomes for the first time.**
   - Guard: no scored number reads the evaluation split; scores come from rollouts on the fixed test start sets.
7. **The top-up adds data beyond the recorded log on some seeds.**
   - Guard: it is fixed in advance, reads no outcome, is given to every switch, and the blocks used are reported.
8. **In the stalling world BCA's actor is the always-on ramp's.**
   - Guard: claim 1's sentence says so.
9. **With equal weights BCA is Wilks' tolerance limit.**
   - Guard: the Wilks twin is reported, and claim 2a's sentence says a pass credits a dose, not the Bayesian part.
10. **Strong current lies outside the recorded currents.**
    - Guard: stated in every claim.
11. **Claim 2a's differences can be strongly skewed.**
    - Guard: claim 2a cannot be "excluded"; its upper bound is printed but not read as a negative.
12. **Imputation could hide a bad seed.**
    - Guard: it applies only to reasons that are not BCA's, uses the least favourable observed value, is capped at 2
      seeds, and every imputed seed is named.

**Predictions, written before any training (not marks):**
- Claim 1 passes.
- Claim 2b is excluded, most likely with the plain rule ahead (mean D around -3 to -8).
- Claim 2a's mean D is near the bar (about 5-7): about even to likely shown, as a dose.
- The switch is right in 30 of 30 seed-worlds.
- No seed refuses.

## 18. Files

Version 8's layout under `near80_v9/`:
- `PREREGISTRATION.md`, `README.md`, `prereg.json`, `code_manifest.json`;
- `approve_local_near80_v9.py`, `run_near80_v9.sh`, `push_runfiles_near80_v9.sh`, `preview/preview_v9.py`;
- `code/`, with version 8's files changed as `decide/BUILD_LIST.md` lists, plus `arms_v9.py`, `topup.py`,
  `test_v9.py` and `test_pooled_v9.py`;
- `pre_approval/`: the counts, the pinned preview and the frozen no-training checks;
- `prereg_history/`.

---

## Adaptations of the decision note's text

This document is appendix A of the decision note (`DECISION.md`, draft 2, 20 October 2026), word for word, except:
1. **File names follow the package** (`code/near80_constants.py`):
   - `approve_local_v9.py` is `approve_local_near80_v9.py` (sections 11 and 18);
   - `run_v9.sh` is `run_near80_v9.sh` (sections 8.2, 10 and 18);
   - `push_runfiles_v9.sh` is `push_runfiles_near80_v9.sh` (section 18);
   - `approval_v9_<16>.json` is `approval_near80v9_<16>.json`, and `import_v9_<16>.json` is
     `import_near80v9_<16>.json` (section 11).
2. **Left out:** the appendix's opening note to the build session.
3. **Filled from the first cloud preview:** section 0's first-preview summary and counts, and section 4's Table 1.
   Section 4 also gives the outcome-free counts and the scores read before freezing, as its heading asks.
4. **Left open**, each with the package's open marker, so that `code/freeze.py` refuses while any remains: section
   0's pinned-preview summary, section 3.2's tag check, and section 4's Table 2 and result. The pinned preview and the
   freeze fill them.
5. **Kept as written:** the references to `DECISION.md` (section 14) and `decide/BUILD_LIST.md` (section 18), the
   decision note and its build list; and the file names of section 18 that the package does not hold yet
   (`preview/preview_v9.py`, `code/arms_v9.py`, `code/topup.py`, `code/test_pooled_v9.py`).
