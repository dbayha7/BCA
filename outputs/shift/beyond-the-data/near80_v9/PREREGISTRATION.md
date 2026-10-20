# The deciding test: pre-registration (version 9, near-80)

**DRAFT, not frozen, awaiting the owner. Nothing has been trained.**

No learner, critic or actor was trained for this version, and no BCA certificate was computed on any of its fresh
seeds. Every value the owner's decision fixes is an open marker below and in `code/near80_constants.py`
(the claims block `CLAIMS`, the seed count `N_FRESH` and the owner's answer `OWNER_ANSWER`). `freeze.py` refuses to
freeze while any of them is left (`freeze.todo_problems`). The machine-readable twin of this document is
`prereg_draft.json` beside it; the final `prereg.json` is written by `code/freeze.py write` only after the values are
filled, and you approve its fingerprint as section 11 says.

**The request:** "build now we need to keep pushing for good results. ASAP" (20 October 2026), after version 8's
claim 4 failed on fresh seeds 10-14.

**The one question this answers:** in the main test's setting, on seeds nobody has seen, with BCA's two groups built
from every held-out recorded row outside the ball, does BCA's switched ramp still meet step 2's three marks (C1), and
does it beat the plain rules that read the same groups (C2)?

---

## 0. Status, and what this was written after

**Version 8 (near-80) is finished.** Its runner gathered its records on 20 October 2026
(`near80/results-20261020T115246Z/RESULTS.json`). Claim 4 (step 2 confirmed in the main setting, fresh seeds 10-14)
is pre-registered and failed. Claims 1-3 never ran. The scores of seeds 11 and 14 have been read (description only).

**Why claim 4 failed** (`decide/refusal.md`, records only): on fresh seeds 10, 12 and 13 the stalling world's
held-out full-throttle group had 19, 22 and 18 rows. BCA's tail threshold is +infinity below about 29 rows, so BCA
refused there. Step 2's arm table refuses any switched-on arm with an infinite height, and the whole actor phase of
that seed stopped: every arm was lost, the rivals included. The switch's on/off call itself was right in every
seed-world that recorded one.

**What this version changes, and only this** (`decide/refusal.md` section 8):
1. BCA's two groups pool every held-out recorded row outside the ball: the whole calibration split and the whole
   evaluation split, fixed in advance. Every switch arm reads the same two groups. The training rows, the test start
   sets and every rival's own rule are unchanged. No scored number reads the pooled rows (section 3).
2. The pooled full-throttle group is counted per seed-world from commands alone, recorded before any training, and
   reported (section 4).
3. A refusal by BCA fails only the arms that need BCA, on that seed; every other arm trains and is scored (section 8.4).
4. The plain twins are trained arms: the mean-difference switch (0.10 margin) with the ramp, and the same switch
   turning on the hand-set range limit (section 5).
5. The claims are a configurable block (section 9): C1 is version 8's claim 4 with its marks unchanged; C2 compares
   BCA's switched ramp with each plain twin, paired by seed, with margins and a multiplicity rule fixed by
   DECISION.md.

**Seeds** (section 10). Seeds 0-4 are seen; the plumbing uses seed 0 and the frozen no-training checks read seeds 0-4
(commands only). The confirmatory seeds are the first TODO(DECISION.md) seeds of 35-49 (5, 10 or 15). Nobody has
generated, loaded or looked at any seed 35-49.

---

## 1. Words used

- **The main setting:** the main test's data change, unchanged: the densest 15% of recorded commands (the "ball",
  `experiments/shift/shift.py` `define_gap`) removed from training and from BCA's calibration bank. No other removal.
- **Throttle t(a):** the largest thruster setting of a command, max_k |a_k|.
- **Survivors:** the main setting's kept training rows past t_lo (step 1's gap rule on the kept training commands).
  They are held out from BCA's second critic.
- **Held-out rows H:** every row of the calibration split and the evaluation split whose command lies outside the
  ball. The calibration bank is part of the calibration split.
- **BCA's two groups:** full-throttle group = survivors + the rows of H past t_lo; rest = the rows of H at or below
  t_lo.
- **BCA refused:** BCA's switch is on in a seed-world but its height is not finite (the full-throttle group was too
  small to certify).
- **The plain twins:** the mean-difference switch with the ramp (`meandiff_k1`) and the mean-difference switch with
  the hand-set range limit (`meandiff_limit`).
- **Net across worlds:** strong-current points where full throttle stalls minus points lost where it helps, against
  no caution (step 2's `net`).
- **Strong current:** the scored situation; the other three (recorded conditions, big waves, big kicks) are read by
  C1's keeping mark.

---

## 2. The short version

- Main setting only, both worlds, on the confirmatory fresh seeds. No band, no random rows, no version 6 band.
- BCA's switch, height and every other switch arm read two pooled held-out groups (section 3).
- The pooled full-throttle group is counted before any training (section 4).
- When BCA refuses on a seed, its height arms and its matched rivals fail there; everything else trains (section 8.4).
- C1: step 2's three marks, as version 8's claim 4. C2: BCA's switched ramp against the split-conformal switch and
  both plain twins, paired by seed. The decision rule says which outcome is read as BCA-specific credit (section 9.4).

---

## 3. BCA's two groups, exactly

### 3.1 The rule

- H = every row of `log.split_rows("calibration")` and `log.split_rows("evaluation")` with `gap.contains(a)` false,
  on the stored float32 command at its exact float64 value (`near80lib.heldout_rows`). This is the same rule that
  empties the bank of its ball rows (`shift.make_removal`).
- t_lo and t_up come from step 1's gap rule on the kept training commands (upper quantile 0.99, minimum gap 0.05,
  minimum tail 9), unchanged.
- Full-throttle group = the survivors, then the rows of H with t > t_lo. Rest = the rows of H with t <= t_lo
  (`near80lib.pooled_groups`).
- Both groups are scored once, by the switch phase, with BCA's second critic (`learner_phases.switch_scores`). The
  certificate is unchanged: `step1lib.certificate` at alpha 0.1, credibility 0.95, 1,000 draws, q05 above 0.10, with
  the same random tags.

### 3.2 What is checked (`near80lib.group_checks`; a failure is a recorded stop)

1. No group row lies inside the ball.
2. The only training rows in the groups are survivors; the survivors are exactly the kept training rows past t_lo.
3. Every other group row is a held-out row (calibration or evaluation split).
4. The two groups are disjoint and together equal H plus the survivors.
5. Every full-throttle row is past t_lo; every rest row is at or below it.
6. The switch phase's held-out rows hash to the data record's `heldout_rows_sha256`, and its group sizes equal the
   outcome-free count of section 4.

### 3.3 Why no scored number reads these rows

Every score is a simulator rollout from the four fixed test start sets (`thruster_chain.bank_starts`), which are not
recorded rows. No learner trains on an H row: the critics and actors train on the main setting's kept training rows,
and BCA's second critic also leaves out the survivors.

### 3.4 Reported only

- Each group's rows by source (survivors, bank, other calibration rows, evaluation rows) and its distinct episodes.
- BCA's certificate on one row per episode (a reported variant).
- BCA's certificate on version 8's bank-only groups (survivors + bank past t_lo; bank at or below), for comparison.

---

## 4. The outcome-free count, before any training

- The data phase of every seed-world counts the pooled full-throttle and rest groups from commands alone
  (`near80lib.outcome_free_count`; no reward, next state or learner output is read) and records the count in its data
  record, before any critic of that seed is trained.
- The count is written beside BCA's certifiable bar (36 rows at 0.1 / 0.95 / 1,000 draws,
  `calibration.reference.certifiable`) and the exact floor of about 29 rows.
- It is a **reported** check, never a stop: section 14.1's first sentence writes it in every look.
- The switch phase must find exactly the counted sizes (check 6 of section 3.2).

---

## 5. Arms

### 5.1 The host

TD3+BC, actor only, as step 2 and version 8 (alpha 5.0; 20,000 actor steps in chunks of 4,000; 100,000 critic
updates). Step 2's code is imported read-only through its verified manifest.

### 5.2 Main setting, both worlds

Version 8's claim-4 arms, unchanged: `none`, `clone`, `full_none` (stalling world only), `w0_check`, `obca_k1` (BCA's
switched ramp), `obca_k2`, `obca_forced`, `fixed_gated`, `conformal_k1` (the split-conformal switch), `meangap_k1`,
`perfect_band`, `range_hinge` (the hand-set range limit), `even_pull`, `push_less`, `random_1`, `random_2` and
`standin` (the matched range limit).

New, reading the same pooled groups:
- `meandiff_k1`: switch on when mean(full-throttle overestimate) minus mean(rest) > 0.10; the ramp at that
  difference, k = 1. Trained in a program of its own.
- `meandiff_limit`: the same switch turning on the hand-set range limit; switched off it is the no-caution actor.
  Trained in a hinge program of its own when on, and reported against `range_hinge`.

### 5.3 Programs

BCA's switched ramp and the plumbing check train in step 2's own programs: in the stalling world `w0_check` alone,
then six ramp arms with `obca_k1` first; in the no-stall world one batch of seven, `w0_check` first. A refused arm's
place keeps a height-0 copy (`refused_pad_<arm>`), checked equal to no caution, never saved or scored, so every other
arm trains in the program it would have had.

---

## 6. Situations

Four fixed test start sets of 1,000 starts each (`thruster_chain.bank_starts`): recorded conditions, strong current,
big waves, big kicks. Strong current is scored; the other three are read by C1's keeping mark.

---

## 7. Measures

Points (0 for no caution's floor, 100 for the stall-aware controller), per arm, world, seed and situation; the net
across worlds; held-to-70% and capped-at-80% points and the share of steps past 80% (reported).

---

## 8. The tests, in order

### 8.1 The no-training checks (before the freeze; seeds 0-4; commands only)

`code/facts_v9.py` on seeds 0-4, both worlds: the gate's data checks (the ball's leak checks; in the stalling world
T1-T5), step 1's gap rule defined, the pooled groups' checks (section 3.2), and the bank-only groups against step 1's
recorded group sizes. Every check must pass, or the design is redrafted (section 14.5).

### 8.2 Plumbing, after approval (scratch only)

The unit tests, a 40-step smoke of every phase kind on seed 0, and the reproduction check on seed 0: the main test's
removal, critic and cloned actor, the no-caution actor and step 2's plumbing-check, switched-ramp and hand-set
range-limit actors must be reproduced. The pooled certificate on seed 0 is recorded (reported only).

### 8.3 The confirmatory stage: the fresh seeds

Every phase of the main setting on every confirmatory seed, both worlds; the lock before any score; the evaluation;
the confirmatory look.

### 8.4 The refusal rule

When BCA refuses in a seed-world, the arms that read BCA's height (`obca_k1`, `obca_k2`, `obca_forced`) and the rivals
matched to its strength (`standin`, `even_pull`, `push_less`, `random_1`, `random_2`) fail that seed. `obca_forced`
takes the stalling world's height in both worlds, so a stalling-world refusal fails it in both. Every other arm trains
and is scored. A refused seed fails every mark that reads a BCA arm or a matched rival, in both claims; it is never
replaced. No whole seed is stopped for a refusal.

---

## 9. Claims, with their marks

### 9.0 Validity (without it, no claim either way)

Version 8's rule, carried from the near-80 setting to the main setting (`near80_constants.CLAIMS["validity"]`):
pooled over the fresh seeds that did not fail, stalling world, strong current, the full-log controller must beat the
controller trained without caution by at least 30 points, and the controller trained without caution must go past 80%
on at least 25% of steps. On record: about 98-99 points and 73% on seeds 0-4. When validity fails, neither C1 nor C2 is
read either way and the decision is "no conclusion" (section 14.2's validity sentence). A failed seed never makes
validity fail.

### 9.1 C1: step 2 confirmed in the main setting (version 8's claim 4)

Step 2's three go marks (`step2lib.go_no_go`), unchanged:
- BCA's net across worlds at least 10 points above the best rule that cannot switch itself off;
- in recorded conditions, big waves and big kicks, BCA loses at most 3 points to no caution;
- BCA beats the matched even pull by at least 15 points in strong current.

### 9.2 C2: BCA's switched ramp beats each plain twin

Paired by seed, on the net across worlds, BCA's switched ramp against each of: the split-conformal switch
(`conformal_k1`), the mean-difference switch with the ramp (`meandiff_k1`), and the mean-difference switch with the
hand-set range limit (`meandiff_limit`).

| value | setting |
|---|---|
| margins (conformal_k1, meandiff_k1, meandiff_limit) | TODO(DECISION.md) |
| statistic (mean, paired_t or interval) | TODO(DECISION.md) |
| alpha | TODO(DECISION.md) |
| at least this many seeds with BCA ahead | TODO(DECISION.md) |
| multiplicity rule (all, holm or bonferroni) | TODO(DECISION.md) |
| noise floor (largest standard error allowed) | TODO(DECISION.md) |

### 9.3 Rules about the marks

- Marks are decided on the unrounded numbers.
- A seed whose data or a needed phase failed, or where BCA refused, fails every mark that needs it, in both claims; it
  is never replaced.
- A seed with no failure or stop record whose evaluation lacks a number a claim reads fails that claim's marks
  ("its evaluation (numbers missing)").

### 9.4 The decision rule

TODO(DECISION.md): "C2" (C2 holds) or "C1_and_C2" (C1 and C2 both hold) is read as BCA-specific credit shown.

---

## 10. Seeds

- Main generator (master seed 20261012). Exploratory: 0-4 (plumbing and the frozen no-training checks only).
- Confirmatory: the first TODO(DECISION.md) of 35-49.
- Refused always: 5-9 (versions 6/7), 10-14 (version 8, spent), 15-34 (named elsewhere: near-80's fallback block and
  the placement design), 100-119 (the gate's confirmation seeds), 900-909 (Stage 0), and the rest of 35-49.
- The registry is `near80_constants.SEED_REGISTRY`; it was checked against every registry and record of every branch
  on 20 October 2026, including `origin/claude/placement-test-design`'s `placement_test/seed_registry.json`.

---

## 11. Approval

As version 8: `freeze.py write` writes `prereg.json` and its code manifest; an independent review; then, at your
desktop terminal, `python3 -I outputs/shift/beyond-the-data/near80_v9/approve_local_near80_v9.py --check` and the same
without `--check`, which writes `approval_near80v9_<first 16>.json` under your own git identity; then the session
imports it (`code/import_approval.py`), writing `import_near80v9_<first 16>.json`. Nothing trains before the import
record is pushed. The approval is yours alone: no agent writes, fakes or imports it.

---

## 12. Order of work

1. DECISION.md fills every open marker.
2. The session reruns the tests and `freeze.py check`; then `freeze.py write`.
3. An independent review; then your approval (section 11) and the import.
4. On the desktop: `run_near80_v9.sh plumbing`, then `run_near80_v9.sh run`.
5. `run_near80_v9.sh results`; the report.

---

## 13. Compute

From version 8's ledger (main setting, fresh seeds 11 and 14, both worlds' actors trained): 1.51-1.54 CPU-hours per
seed (stalling-world policy 41-42 minutes, no-stall 9-10; three critics about 5.7 minutes each). The two plain-twin
programs add about 1.6 minutes per world. Forecast: 1.6 CPU-hours per fresh seed, plus about 2.5 for the plumbing.

| fresh seeds | CPU-hours | wall time at 10 lanes (at most 2 policy phases at once) |
|---|---|---|
| 5 | about 8 + 2.5 | about 3 hours, plus 1-2.5 for the plumbing |
| 10 | about 16 + 2.5 | about 5-5.5 hours, plus the plumbing |
| 15 | about 24 + 2.5 | about 7.5-8 hours, plus the plumbing |

A stage past twice its forecast, or the run past the cap, pauses; `run_near80_v9.sh budget-continue <stage>` lifts it.

---

## 14. The claim, word for word, for each outcome

The code fills in the brackets. Numbers are rounded to one decimal. Every bracket is pooled over the group's seeds,
in strong current, unless stated:

| bracket | meaning |
|---|---|
| SEEDS, NS | the group's seeds, written "35-39"; how many |
| S, N | seeds where BCA's switch was on (stalling world), off (no-stall world) |
| GMIN, GMAX, NSW, GB | the smallest and largest pooled full-throttle group; the seed-worlds counted; how many were below 36 rows |
| RF | seeds where BCA refused |
| X, H | BCA minus no caution; the full-log actor minus no caution (stalling world) |
| P | no caution's share of strong-current steps past 80%, stalling world, as a percentage |
| W | no-stall world: no caution minus BCA |
| NET, D, R | BCA's net across worlds; its lead over the best world-blind rival; that rival's name |
| E, L | BCA minus the matched even pull; BCA's largest loss to no caution in the other three situations |
| T | a plain twin's name |
| DIFF, MARGIN | BCA's net minus the twin's, mean over seeds; the pre-set margin |
| KPOS | seeds where BCA's net was above the twin's |
| PV, LEVEL | the one-sided paired t-test's p-value; the level it is compared with |
| LO, HI, CONF | the two-sided t-interval of DIFF; its confidence, in percent |
| RESULT | "met" or "missed"; for the interval, "shown", "excluded" or "unresolved" |
| MULT | the multiplicity rule, in words |
| V, NF | the largest standard error of a paired difference across seeds; the noise floor's limit |
| NETRAW, N1, N2, N3, NZ, NK2 | nets across worlds over every seed: BCA's switched ramp, the split-conformal switch, the mean-difference switch with the ramp, the mean-difference switch with the hand-set range limit, the hand-set range limit, the ramp at twice BCA's height |
| RULE | the decision rule, in words |

**Always attached:** "One simulated task, two worlds, [NS] fresh seeds ([SEEDS]), four fixed test start sets of 1,000
starts each. In strong current the test states also lie outside the recorded currents."

**In the plumbing look,** every claim sentence below starts with "Plumbing on seen seeds, not a claim:".

### 14.1 The count and the refusals

**Always written first, the outcome-free count:**
> "Before any learner was trained, BCA's full-throttle group, pooled from every held-out recorded row outside the
> ball, held [GMIN] to [GMAX] rows over [NSW] seed-worlds; [GB] were below the 36 rows BCA needs to certify a group."

**If BCA never refused:**
> "BCA did not refuse on any of the [NS] seeds."

**If BCA refused:**
> "BCA refused on [RF] of [NS] seeds (seed [seeds]): its switch was on but its full-throttle group was too small to
> certify. By the pre-set rule every arm that needs BCA's height or its strength failed there, and every other arm
> was trained and scored."

### 14.2 C1: step 2 confirmed in the main setting

**Holds:**
> "On fresh seeds [SEEDS], in the main setting (the densest 15% of commands removed), step 2's result held: BCA's
> switched ramp met all three pre-set marks. Its switch turned caution on in [S] of [NS] seeds where full throttle
> stalls and off in [N] of [NS] where it does not. In strong current it gained [X] points over no caution where full
> throttle stalls (the full-log controller gained [H]) and lost [W] where full throttle helps: a net of [NET], [D]
> more than the best rule that cannot switch itself off ([R]). It beat the matched even pull by [E]."

**Fails:**
> "On fresh seeds [SEEDS], in the main setting, step 2's result did not hold: [clauses]."

The clauses:
- "its net across worlds of [NET] was [D] above the best rule that cannot switch itself off ([R]), below the 10
  required"
- "it lost [L] points to no caution in [situation], more than the 3 allowed"
- "it beat the matched even pull by [E] in strong current, below the 15 required"
- "seed [s] failed [check] and, by the pre-set rule, counts as failing every mark"

**Validity fails:**
> "On fresh seeds [SEEDS], the main setting did not produce the harm it was built to test ([reason]). It supports no
> conclusion about BCA either way."

The reasons:
- "the controller trained without caution lost only [H] points to the full-log controller (at least 30 needed)"
- "the controller trained without caution went past 80% on only [P]% of strong-current steps (at least 25% needed)"

### 14.3 C2: BCA against the plain twins

**Holds:**
> "On fresh seeds [SEEDS], BCA's switched ramp beat every plain rule that reads the same pooled groups, paired by
> seed ([MULT]): [comparisons]."

**Fails:**
> "On fresh seeds [SEEDS], BCA's switched ramp did not beat every plain rule that reads the same pooled groups,
> paired by seed ([MULT]): [comparisons]."

The comparisons, one per plain twin:
- by the mean: "against [T], BCA's net minus the twin's was [DIFF] points on average (BCA ahead in [KPOS] of [NS]
  seeds; margin [MARGIN]): [RESULT]"
- by the paired t-test: "against [T], BCA's net minus the twin's was [DIFF] points on average (BCA ahead in [KPOS]
  of [NS] seeds; one-sided paired t-test against a margin of [MARGIN], p = [PV] at level [LEVEL]): [RESULT]"
- by the interval: "against [T], BCA's net minus the twin's was [DIFF] points on average ([CONF]% interval [LO] to
  [HI]; BCA ahead in [KPOS] of [NS] seeds; bar [MARGIN]): [RESULT]"

**If a seed failed:**
> "By the pre-set rule, C2 fails because seed [seeds] failed (C1 says why); the numbers of the other seeds are
> reported as description only."

**If every seed failed:**
> "By the pre-set rule, C2 fails because seed [seeds] failed (C1 says why)."

**Dropped by the noise floor:**
> "C2 was not decided: BCA's paired difference against [T] varied too much across seeds (standard error [V] points;
> at most [NF] allowed). Its numbers are reported as description."

**Always added, for scale:**
> "For scale, nets across worlds in strong current: BCA's switched ramp [NETRAW], the split-conformal switch [N1], the
> mean-difference switch with the ramp [N2], the mean-difference switch with the hand-set range limit [N3], the
> hand-set range limit [NZ], the ramp at twice BCA's height [NK2]."

### 14.4 The decision

**Shown:**
> "By the pre-set decision rule ([RULE]), this test shows BCA-specific credit in the main setting."

**Not shown:**
> "By the pre-set decision rule ([RULE]), this test does not show BCA-specific credit in the main setting."

### 14.5 Stops (exactly the one that applies)

- **A no-training check fails on seeds 0-4:** "On seeds 0-4, before any training, version 9's data failed [check]
  ([value]; needed [threshold]). The design was redrafted. Nothing was trained."
- **Reproduction fails:** "Stopped: the new code did not reproduce [the main test's removal / critic / cloned actor /
  no-caution actor, or step 2's plumbing-check / switched-ramp / hand-set range-limit actor] on seed 0. No result of
  this version stands until it does."
- **A phase stops:** "Seed [s], [world]: [phase] stopped ([reason]). It was not rerun. Every mark that needs it
  fails."

### 14.6 How the code fills in the sentences

- Numbers are written with one decimal ("{:.1f}"), with a minus sign when negative and no plus sign; "-0.0" is
  written "0.0". [P] is a share of steps written as a percentage. [S], [N], [NS], [RF], [KPOS], [GMIN], [GMAX], [NSW], [GB] and [s] are whole numbers; [PV] and
  [LEVEL] are written with four significant digits.
- A number that cannot be computed, because a seed failed, a phase stopped or an arm is missing, is written "not
  available". A mark that needs it fails.
- [R] is written as the plain name of step 2's world-blind rival with the best net; ties go to the first in step 2's
  order: the matched range limit, the matched even pull, the first random surface, the second random surface, the
  always-on ramp. [T] is "the split-conformal switch", "the mean-difference switch with the ramp" or "the
  mean-difference switch with the hand-set range limit".
- [situation] is "recorded conditions", "big waves" or "big kicks". [world] is "stalling world" or "no-stall world".
  [phase] is the phase's plain name in the runner's phase table.
- [clauses] lists the clauses of the missed marks, joined by "; ", in this order: the net lead; keeping the gain (one
  clause per situation that misses, in the order recorded conditions, big waves, big kicks); beating the even pull;
  then one clause per failed seed, in seed order. A missed mark whose number is not available because of a failed seed
  is told by that seed's own clause only.
- [check] is the first check that failed: a no-training check by its name; "its [phase] phase" for a stopped phase;
  "BCA's height (BCA refused: ...)" for a seed where BCA refused; "its evaluation (numbers missing)" for a seed whose
  evaluation lacks a number a claim reads; "its matching (a rival could not be matched or scored)" when a matched
  rival could not be matched.
- [comparisons] writes one comparison per plain twin of the claims block, in its order (as drafted: the
  split-conformal switch, the mean-difference switch with the ramp, the mean-difference switch with the hand-set range
  limit), joined by "; ",
  in the form of the claims block's statistic. [MULT] is "every comparison at its own level; C2 holds only if every
  one passes", "Holm's step-down rule over the comparisons" or "Bonferroni over the comparisons".
- When validity (section 9.0) fails, its sentence replaces every C1 and C2 sentence and no decision sentence is
  written; [reason] is the harm reason when the harm is below 30 points, the past-80 reason when the share is below
  25%, and both, joined by "; ", when both fail. The count, the refusal and the "for scale" sentences are written as
  usual.
- When a seed failed for C2, its "If a seed failed" sentence replaces the "Holds" or "Fails" sentence; when every
  seed failed, its "If every seed failed" sentence. The noise floor's sentence replaces both when the largest
  standard error is above the limit. While the claims block holds an open value, C2 and the decision are not decided
  and no C2 or decision sentence is written.
- Wherever a sentence says "seed [seeds]", it reads "seed 2" for one seed, and "seeds 2 and 3" or "seeds 1, 2 and 3"
  for more.
- In the reproduction stop, the bracket lists every item that differed, written "A", "A and B" or "A, B and C".
- In the plumbing look the "Always attached" sentence is not written, every claim sentence starts with the plumbing
  prefix, and "fresh seeds" reads "seeds". The stop sentences are written as they stand.
- Exactly one stop sentence is written, the first that applies in this order: a no-training check, reproduction.
  Each stopped phase adds its own sentence.

---

## 15. Departures from version 8, fixed now

- BCA's groups are pooled from every held-out recorded row outside the ball (version 8: the survivors and the
  calibration bank only).
- A refusal by BCA fails only the arms that need it (version 8: the whole actor phase of the seed stopped).
- The plain twins are trained arms; C2 is new.
- The near-80 band, the random-rows control, version 6's band, the readouts and version 8's exploratory stop rules are
  removed. Seeds 0-4 carry no phase of the official run.
- The data record's run sizes are read back under the key `run_sizes` (the cause of version 8's amendment 1).

---

## 16. Choices (open until DECISION.md)

| choice | default | the owner's answer |
|---|---|---|
| 1. number of fresh seeds (5, 10 or 15; the first of 35-49) | none | TODO(DECISION.md) |
| 2. C2's margins, statistic, alpha, seeds-ahead rule, multiplicity and noise floor | none | TODO(DECISION.md) |
| 3. the decision rule (C2, or C1 and C2) | none | TODO(DECISION.md) |
| 4. lanes | 10 | TODO(DECISION.md) |

---

## 17. What could fool us, and what we expect

- **BCA's height shrinks with more rows.** At version 8's group sizes BCA's height is the largest or second largest
  held-out tail score; pooled groups give it more rows, so its height and part of its recorded gain can shrink
  (`decide/refusal.md` section 0, item 4). C2 is there to tell BCA's own credit from what any switch on the same
  groups would get.
- **The evaluation split is now read by the switch.** No scored number reads it (section 3.3); the checks of section
  3.2 hold it out of every training file.
- **A refusal is still possible.** The count of section 4 says, before training, how likely it is; a refusal fails
  BCA's arms on that seed, and the sentence of section 14.1 says so.

---

## 18. Files

- `PREREGISTRATION.md`, `README.md`, `prereg_draft.json` (this draft), `approve_local_near80_v9.py`,
  `run_near80_v9.sh`, `push_runfiles_near80_v9.sh`.
- `code/`: the files of `near80_constants.CODE_FILES`, five unit test files among them (`test_v9.py` is new).
- Outputs: `runs/shift_explore/near80_v9/` (write-once); committed copies in `near80_v9/results-*/`.
