# Confirmation of the OOD-action overestimation comparison: fresh seeds, three doses, both worlds (written before any number exists)

Written 2026-11-01, 22:53 EST, after the exploratory comparison (`../overestimation_v1/OVERESTIMATION_RULE.md`, runs
`oe_run/attempt_2` and `oe_run_a2/attempt_1`) read PROMISING / PLACEBO MIXED on three seeds at one dose. This package
asks whether that holds on seeds no run has touched, and how it moves with the dose. One sealed package, one
approval, one run. Pre-registered: every bar below is fixed before the first fresh-seed number exists.

## The claims under test

| Claim | What was seen (exploratory, kappa 63) | What confirms it here |
|---|---|---|
| C1, honeypot: where the withheld region is a trap (stall3), the BCA actor's return beats the native actor's from the same fork | dJ +11.9, +41.3, +22.8 | at kappa 63, dJ ≥ +2.0 in every fresh seed |
| C2, placebo: where the withheld region is harmless (no-stall), BCA's avoidance of it costs little | dJ −0.3, +3.4, −6.7 (one seed past the bar) | at kappa 63, dJ ≥ −2.0 in every fresh seed (CALIBRATED), or the cost is bounded: mean dJ ≥ −4.0 and no seed below −8.0 (COST BOUNDED) |
| C3, mechanism-side: the pull out of W is world-blind | dEXPO −0.00/−0.06/−0.24 (stall3), −0.25/−0.08/−0.32 (no-stall) | reported, no bar: dEXPO by dose in both worlds |
| D, the dose axis | one dose | reported, no bar: exposure, dJ, de_own at kappa 21, 63, 189 in both worlds; monotonicity flags |

The dose being confirmed is 63. The other two doses are a profile, read after the fact; the best-looking dose on
this run is not a confirmed dose and the rule says so now.

## Decisions fixed here

| Decision | Value | Why |
|---|---|---|
| Host, data, worlds, starts, probes, reference | exactly the exploratory package's (`../overestimation_v1/`): Unifloral TD3+BC through the native screen's worker and its seven bindings; the formal G2 v2 CUTOFF datasets stall3 (c2a6d616…) and no-stall (e3549869…) by digest; the 128 E2 starts; 48 visited states; 192 probe rows with logged and three withheld a0 magnitudes; the policy's own exact unclipped continuation | the comparison is confirmed as run, not redesigned |
| Fresh seeds | three training rows from the reserved namespace `knee_s10_candidate_v2/confirmation/` (`cf_seeds.json`, audited against every integer on record, the screen's and the exploratory package's); fresh `visited_states` and `dataset_rows` seeds from the same namespace | no run has drawn from this namespace |
| What is not fresh | the 128 start states (the sealed formal G2 v2 bank) and the data-plan seeds (the screen's, which fix the CUTOFF rows and must match the sealed collection record) | they are states and masks, selected by no outcome; declared |
| Doses | kappa ∈ {21, 63, 189}: the confirmed dose and one third and three times it | one axis, three points, no search |
| Training | per seed and world: one 10k native warmup, one native branch 10k→20k, three BCA branches 10k→20k from the same fork, each paired with the native branch (same minibatch stream, same warmup) | Stage 1's protocol; the dose screen's branch |

## Units, in order (fail-fast)

Per world (stall3, then no-stall; ids suffixed `_<world>`): `warmup_s{i}`, `branch_s{i}`, `bca_branch_s{i}_k{kappa}`
for kappa in 21, 63, 189, then `eval_native_s{i}` and `eval_bca_s{i}_k{kappa}` (kind `cf_evaluate`, the
exploratory `measure` with the fresh seeds). Thirty training jobs (about 8 minutes each on this machine: about 4
hours on Zeus, about 7.5 on Titan, up to about 13 at the slowest pace either machine has shown) and twenty-four evaluations (about 25 s each). Then the decision (`cf_math.decide`), `decision.json`,
`CF_RESULT.json` always written.

## Reads

Per unit, the exploratory package's reads (`oe_math.unit_reads`): J, W exposure, own-action error e_own, e_data,
e_W, U_data, U_W, U_own, the penalized score's error at the unit's own kappa, the I1 and I2 instrument checks (A1
tolerances: 1e-6). Paired per seed, world and dose, BCA minus native: dJ, de_own, dEXPO, de_W.

## Readings

Validity first: INCOMPLETE > INCONCLUSIVE (any instrument miss or non-finite read) > NOT REACHED (native stall3 W
exposure < 0.01 in every seed). Then the reading is "C1 / C2" at kappa 63:

| C1 (stall3, kappa 63) | When |
|---|---|
| CONFIRMED | dJ ≥ +2.0 in every fresh seed |
| NOT CONFIRMED | otherwise (the three dJ are reported either way) |

| C2 (no-stall, kappa 63) | When |
|---|---|
| CALIBRATED | dJ ≥ −2.0 in every fresh seed |
| COST BOUNDED | not CALIBRATED, and mean dJ ≥ −4.0 and no seed below −8.0 |
| OVER-CORRECTION | dEXPO ≤ −0.05 in two or more seeds and dJ ≤ −2.0 in two or more seeds, and not COST BOUNDED |
| PLACEBO WITHOUT STAKE | native W exposure < 0.01 in every seed |
| PLACEBO MIXED | anything else |

Precedence inside C2: PLACEBO WITHOUT STAKE > CALIBRATED > COST BOUNDED > OVER-CORRECTION > PLACEBO MIXED.

The dose profile is reported with two flags and no bar: `exposure_monotone` (W exposure non-increasing in kappa in
every seed and world) and `honeypot_gain_monotone` (stall3 dJ non-decreasing in kappa in every seed). A profile
is a description of this run.

## Kill rules and caps

Training: 1,800 s per job, 13 h in all (amendments A1 and A2; was 5 h). Evaluation: 900 s and 200,000 exact transitions per job, 2 h and
3,000,000 in all. A failed job ends the attempt INCOMPLETE. No bar, dose, slot or seed changes after the approval.

## Attempts

One attempt. After INCOMPLETE, a second attempt under a new approval naming attempt 2 reuses every completed
training unit (same id, kind, seed, arm, world, kappa) and reruns every evaluation. After INCONCLUSIVE, the same,
only when every evaluation of the previous attempt recorded its I1 identity within 1e-6 relative. No attempt 3.
`--decide` recomputes the decision from the saved reports.

## What the user's approval states

`CF_APPROVAL.json`, written at the user's terminal, must contain these sentences, the manifest's sha256 and the attempt:

1. "I approve the confirmation of the OOD-action overestimation comparison on the knee_s10_v2 CUTOFF data, Unifloral TD3+BC host, doses kappa 21, 63 and 189, both worlds, attempt {n}"
2. "the BCA arms at kappa 21, 63 and 189 are approved"
3. "the confirmation seeds are fresh: no run has used them"
4. "the dose being confirmed is 63; the other doses are a profile"

## Amendments

| # | When | Change | Unchanged |
|---|---|---|---|
| A1 | 2026-11-02, after attempt 1 ended INCOMPLETE (04:12 UTC: the user interrupted `warmup_s0_stall3` after 286 s, before any checkpoint, report or evaluation existed; no number from the fresh seeds was produced) | **Total training cap 5 h -> 9 h**, set by the user before the attempt-2 approval. The rule's 5 h assumed about 8 minutes per job (the overestimation runs' 480 to 530 s). On the same machine the r5 screen's real warmups took 880 s and 1,286 s (the second overlapped this package's attempt 1 on the GPU), so the 30 jobs could need about 7 to 8.5 h; at the 5 h cap the last allowed attempt would end INCOMPLETE on time alone. A kill rule only: 9 h covers about 1,080 s per job. Approval sentence 3 ("the confirmation seeds are fresh: no run has used them") stands in the sense that no result from them exists: attempt 1's 286 s of seed-0 training produced no checkpoint and is rerun from the start. | the per-job cap (1,800 s), the evaluation caps, the doses, the seeds, the reads, the bars, the readings, the two-attempt limit, the approval words. |
| A2 | 2026-11-02, before the attempt-2 approval; the user moved the run to Titan ("I want this done on hpc cluster or titan", then "Titan, after r5") | **Change 4, device kinds.** knee_edge_ood_v3 pins Zeus's GPU kind in every research training bind, so on Titan the first training job would refuse and end the last attempt INCOMPLETE. The worker now sets the recorded kind to the one allowed kind the machine has (`device_kinds.py`, the same list as the r5 screen's A2: Zeus's laptop GPU or Titan's "NVIDIA GeForce RTX 5070 Ti"), right after the worker environment is set; every report keeps the runtime's device kinds. Attempt 1 left no completed unit, so nothing from Zeus is reused. **Training cap 9 h -> 13 h:** Titan runs about 1.8 times slower per job than Zeus (the r5 dry run: about 900 s), about 7.5 h for the 30 jobs; 13 h covers about 1,550 s per job, the slowest pace either machine has shown under load. Kill rule only. Runs on Titan after the r5 screen finishes, never alongside it. | the evaluation caps, the doses, the seeds, the reads, the bars, the readings, the two-attempt limit, the approval words. |

## Integrity

`CF_MANIFEST.json` hashes this package, the exploratory package's files (whose `oe_eval.measure` and
`oe_math.unit_reads` this package runs), and the native screen's manifest body; it pins knee_edge_ood_v3's A4 and
exploratory manifests. Every job's record is the v3 dependency closure plus those files; the worker refuses on any
change before and after its work, and on any importable file added to this folder, the exploratory package's or
the native screen's.

## What this can and cannot say

It can say whether the honeypot gain and the placebo cost at kappa 63 reproduce on seeds no run has touched, and
how exposure, return and own-action error move across three doses in both worlds. It cannot say which dose is
best (a profile is not a confirmation), anything about the mechanism (U's state-dependence versus the penalty's
gradient), or anything about the knee departure. Three fresh seeds are three; a CONFIRMED reading is a replication
at the pre-registered bar, not a protocol claim.

## Ledger entry

Confirmation v1: "C1 / C2" at kappa 63 with the three dJ in each world, the dose profile (exposure, dJ, de_own by
dose and world), the two monotonicity flags, GPU hours and exact transitions.
