# Confirmation v2 of the OOD-action overestimation comparison: kappa 189, five fresh seeds, both worlds (written before any number exists)

Written 2026-11-03, after confirmation v1 closed INCONCLUSIVE on its last attempt (`../confirmation_v1/`,
`CF_READOUT_attempt3_20261102.md`). In v1's dose profile, kappa 189 was the only dose with one sign in every seed of
each world: dJ +11.2, +121.5, +9.8 in stall3 and −41.4, −98.3, −20.5 in no-stall. A profile is not a confirmation:
v1's rule said the best-looking dose is not a confirmed dose. This package makes kappa 189 the dose being confirmed, on
five seeds no run has touched, and measures its no-stall cost under its own bar. One sealed package, one approval, one
run, on Titan. Pre-registered: every bar below is fixed before the first v2 number exists.

## The claims under test

| Claim | What v1 saw at kappa 189 (profile, three seeds) | What confirms it here (kappa 189, five fresh seeds) |
|---|---|---|
| C1, honeypot: where the withheld region W is a trap (stall3), the BCA actor's return beats the native actor's from the same fork | dJ +11.2, +121.5, +9.8 | dJ ≥ +2.0 in every one of the five seeds |
| C2, placebo: where W is harmless (no-stall), BCA's avoidance of it costs little | dJ −41.4, −98.3, −20.5 (would read OVER-CORRECTION) | CALIBRATED: dJ ≥ −2.0 in every seed; or COST BOUNDED: mean dJ ≥ −4.0 and no seed below −8.0 |
| D, description | — | reported, no bar: dJ, dEXPO, de_own, W exposure overall and at deep load, the share of the 128 starts on which BCA's return is lower, per seed and world |

C2 is measured, not expected to pass: v1's three kappa-189 no-stall seeds would read OVER-CORRECTION. The package asks
whether the stall3 gain reproduces at an every-seed bar on five fresh seeds, and what it costs where W is harmless.

## Decisions fixed here

| Decision | Value | Why |
|---|---|---|
| Host, data, worlds, starts, probes, reference | exactly confirmation v1's (and the exploratory package's, `../overestimation_v1/`): Unifloral TD3+BC through the native screen's worker and its seven bindings; the formal G2 v2 CUTOFF datasets stall3 (c2a6d616…) and no-stall (e3549869…) by digest; the 128 E2 starts; 48 visited states; 192 probe rows with logged and three withheld a0 magnitudes; the policy's own exact unclipped continuation | the comparison is confirmed as run, not redesigned |
| Dose | kappa 189 only | the claim is about 189; a second dose would be a second profile, which this package does not need |
| Fresh seeds | five training rows from the new namespace `knee_s10_candidate_v2/confirmation_v2/` (`cf_seeds.json`, audited clean against every integer on record: 3,943 scanned, 27 named values distinct, no collision), confirmation v1's seeds included; fresh `visited_states` and `dataset_rows` seeds from the same namespace | v1's namespace is spent; no run has drawn from this one |
| What is not fresh | the 128 start states (the sealed formal G2 v2 bank) and the data-plan seeds (the screen's, which fix the CUTOFF rows and must match the sealed collection record) | they are states and masks, selected by no outcome; declared, as in v1 |
| Training | per seed and world: one 10k native warmup, one native branch 10k→20k, one BCA branch 10k→20k at kappa 189 from the same fork, paired with the native branch (same minibatch stream, same warmup) | Stage 1's protocol; v1's branch |
| Independent unit | the training seed (five). Starts, visited states and probe rows are shared across arms within a seed: paired, not replicates | v1's design; five seeds make the every-seed bar harder to pass by luck |
| I1 tolerance | **1e-5 relative** (v1: 1e-6), applied by the decision to each evaluation's recorded `I1_max_rel`; the sealed evaluator is unchanged and still writes its own 1e-6 flag. I2 unchanged | v1 attempt 3's verdict was set by two I1 misses of 1.11e-6 and 1.86e-6 (2 of 24 evaluations, both stall3), about 0.0007 in return; I1 compares the reference Q^π with the rollout return at the 16 visited states with t = 0, and the misses are the float32 GPU batch effect the exploratory package's amendment A1 named. 1e-5 relative is about 0.004 on returns of a few hundred, more than two orders below the 2.0 bar. Chosen from v1's record, before any v2 number |
| OVER-CORRECTION count | dEXPO ≤ −0.05 and dJ ≤ −2.0 each in **3 or more of the 5 seeds** (v1: 2 or more of 3) | the same majority |
| Machine | Titan ("NVIDIA GeForce RTX 5070 Ti"); Zeus's laptop GPU stays allowed (`device_kinds.py`, carried from v1's amendment A2) | the user's choice |

## Units, in order (fail-fast)

Per world (stall3, then no-stall; ids suffixed `_<world>`): `warmup_s{i}`, `branch_s{i}`, `bca_branch_s{i}_k189` for
i in 0 to 4, then `eval_native_s{i}` and `eval_bca_s{i}_k189` (kind `cf_evaluate`, the exploratory `measure` with v2's
seeds). Thirty training jobs (about 900 s each on Titan by v1's dry run: about 7.5 hours; up to about 1,550 s at the
slowest pace either machine has shown) and twenty evaluations (about 25 to 55 s each). Then the decision
(`cf_math.decide`), `decision.json`, `CF_RESULT.json` always written.

## Reads

Per unit, the exploratory package's reads (`oe_math.unit_reads`): J, W exposure, own-action error e_own, e_data, e_W,
U_data, U_W, U_own, the penalized score's error at kappa 189, I2. I1 at this rule's tolerance. Added, no bar: W
exposure at deep load (`rates.by_band.deep`). Paired per seed and world, BCA minus native: dJ, de_own, dEXPO, de_W, and
the share of the 128 starts on which BCA's return is lower (higher).

## Readings

Validity first: INCOMPLETE > INCONCLUSIVE (an I1 miss beyond 1e-5 relative, an I2 miss, or a non-finite read) > NOT
REACHED (native stall3 W exposure < 0.01 in every seed). Then the reading is "C1 / C2" at kappa 189:

| C1 (stall3) | When |
|---|---|
| CONFIRMED | dJ ≥ +2.0 in every one of the five seeds |
| NOT CONFIRMED | otherwise (the five dJ are reported either way) |

| C2 (no-stall) | When |
|---|---|
| CALIBRATED | dJ ≥ −2.0 in every seed |
| COST BOUNDED | not CALIBRATED, and mean dJ ≥ −4.0 and no seed below −8.0 |
| OVER-CORRECTION | dEXPO ≤ −0.05 in 3 or more seeds and dJ ≤ −2.0 in 3 or more seeds, and not COST BOUNDED |
| PLACEBO WITHOUT STAKE | native W exposure < 0.01 in every seed |
| PLACEBO MIXED | anything else |

Precedence inside C2: PLACEBO WITHOUT STAKE > CALIBRATED > COST BOUNDED > OVER-CORRECTION > PLACEBO MIXED. The
description (D) is reported with no bar and changes no reading. The readout keeps the verdict, what the data show and
the open questions apart.

## Kill rules and caps

Training: 1,800 s per job, 13 h in all (v1's caps after its amendments A1 and A2, for the same thirty jobs).
Evaluation: 900 s and 200,000 exact transitions per job, 2 h and 3,000,000 in all. A failed job ends the attempt
INCOMPLETE. No bar, dose, slot, seed or tolerance changes after the approval.

## Attempts

One attempt. After INCOMPLETE, a second attempt under a new approval naming attempt 2 reuses every completed training
unit (same id, kind, seed, arm, world, kappa) and reruns every evaluation. After INCONCLUSIVE, the same, only when every
evaluation of the previous attempt recorded its I1 identity within 1e-5 relative (the cause was I2 or a non-finite
read). A third attempt only after an INCOMPLETE attempt 2; no attempt 4. `--decide` recomputes the decision from the
saved reports. While a run is live, no branch switch, stash or pull in its checkout (GitHub Desktop included): v1's
attempt 2 was ended that way.

## What the user's approval states

`CF_APPROVAL.json`, written at the user's terminal (`../approve_confirmation_v2.py`), must contain these sentences,
the manifest's sha256 and the attempt:

1. "I approve confirmation v2 of the OOD-action overestimation comparison on the knee_s10_v2 CUTOFF data, Unifloral TD3+BC host, kappa 189, both worlds, five fresh seeds, attempt {n}"
2. "the BCA arm at kappa 189 is approved"
3. "the confirmation v2 seeds are fresh: no run has used them"
4. "kappa 189 was chosen from confirmation v1's profile, so this run, not that one, is its confirmation"
5. "the I1 tolerance is 1e-5 relative"

## Integrity

`CF_MANIFEST.json` hashes this package (this rule, the seeds, the code), the exploratory package's files (whose
`oe_eval.measure` and `oe_math.unit_reads` this package runs), and the native screen's manifest body; it pins
knee_edge_ood_v3's A4 and exploratory manifests. Every job's record is the v3 dependency closure plus those files; the
worker refuses on any change before and after its work, and on any importable file added to this folder, the
exploratory package's or the native screen's. This package imports nothing from `../confirmation_v1/`; it reads v1's
`cf_seeds.json` only to keep its own seeds disjoint.

## What this can and cannot say

It can say whether the kappa 189 honeypot gain reproduces on five seeds no run has touched, at an every-seed bar, and
what kappa 189 costs where the withheld region is harmless. It cannot say that 189 is the best dose (one dose is run),
anything about the mechanism (U's state-dependence against the penalty's gradient), anything about other hosts, or
anything about the knee departure. Five fresh seeds are five; a CONFIRMED reading is a replication at the
pre-registered bar, not a protocol claim.

## Ledger entry

Confirmation v2: "C1 / C2" at kappa 189 with the five dJ in each world, the description (dEXPO, de_own, deep-load
exposure, share of starts lower, by world), GPU hours and exact transitions.
