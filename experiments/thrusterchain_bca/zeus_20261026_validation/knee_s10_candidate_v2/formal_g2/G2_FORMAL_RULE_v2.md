# Candidate knee_s10_v2: the formal G2 rule (written before the formal collection exists)

This is the formal exact gate G2 of OOD definition v4 r3 (`../../ood_definition_v4/DRAFT_DEFINITION.md`, sections 5
and 6a) for candidate knee_s10_v2: knee_s10_v1 with one change, the anchor prefix rule (`../prefix_rule.py`,
`../G2_DEV_RULE_v2.md`). Everything else is v1's, by reference and by hash: the knee bed re-collected with the
high-load recorder window widened to [-1.0, -0.6] (`../../knee_s10_candidate_v1/recorder_wide.py`), scored at
sigma = s10 (a0 = -0.9), under the support-projected LQR continuation pi_ref version 2
(`../../knee_s10_candidate_v1/pi_ref.py`, gains pinned bit for bit). It costs no training.

Written 2026-11-01 after v1's formal G2 read NOT DATA-OOD (no-stall D3 anchors 84 of 96; its stall3-run prefixes
carried anchors outside the no-stall data) and after v2's development screen read PASS (48 of 48 anchors inside both
worlds' data, H2 2.198; `../g2_development_result_v2.json`, seen data, never confirmatory).

## What v2 inherits and what it owns

- **Measurements, D4 bounds check, label code**: v1's sealed formal driver
  `../../knee_s10_candidate_v1/formal_g2/g2_formal.py` (sha256 b2ea3b13...), imported unchanged; its preflight and
  execute are never called. The v2 driver `g2_formal_v2.py` owns only the v2 seeds, the anchors (built with the v2
  prefix in both worlds), the preflight, the report and the ledger entry.
- **D4 bounds**: v1's sealed two-sided six-SD bounds `d4_bounds.json` (sha256 060460da...), copied byte for byte;
  they are a property of the unchanged recorder at 1,280 episodes. v1's `d4_bounds.py` is in the manifest.
- **Seeds**: `g2_seeds_v2.py`, `g2_seeds_v2.json`: a new namespace (`knee_s10_candidate_v2/`), so new collection
  entropy and new anchor and start streams, never the development entropy 424263 or stream 707070707 and never v1's
  formal values, audited against every integer on record (raw and derived) before sealing.
- **96 gated anchors** (the t in {0, 25, 50} items of a 128-anchor bank) and 128 start states, as v1.
- A sealed manifest, the user's own approval naming it, one attempt, exact simulator-step accounting, and a report
  that is always written.

## What the user's approval states

The approval file (`G2_FORMAL_APPROVAL.json`, written by the user at their terminal, committed under their identity)
carries, in the user's own words, three decisions that the definition leaves to them:

1. definition v4 r3 is adopted for this candidate (v1-v3 sealed rules unchanged);
2. the primary scoring set is **all 96 gated anchors**; the subset where pi_ref's own command exceeds the knee is
   reported, never decides. If the user wants the subset as primary instead, this rule is edited and re-sealed
   before any run;
3. the attempt number.

## Checks and bars (per world unless stated)

| id | Check | Bar | On a miss |
|---|---|---|---|
| P1 | every prefix command inside CUTOFF's retained support for its state's load band and none past the knee; each anchor's prefix replayed in nostall gives the same observations and state as in stall3, bit for bit | exact, all 128 anchors | INCOMPLETE (the rule was not implemented; an instrument fault) |
| D1 | `masks.arm_masks` CUTOFF holds 0 W rows; `region_independent.withheld_region` agrees with `knee_region.past_knee_a0` row for row; CUTOFF is W's complement | exact | NOT DATA-OOD |
| D2 | no retained high-load a0 <= s10; distance from s10 to the nearest retained high-load a0 | 0 rows; >= 0.0999 | NOT DATA-OOD |
| D3 | L1.5 as amended (`step04_math.l15`): the 128 starts and the 96 gated anchors against the 0.99 quantile of their own group | starts >= 95%, anchors >= 90% (87 of 96) | NOT DATA-OOD |
| D4 | twin density of fitting rows with executed a0 in [-0.95, -0.85], per high-load band, against the sealed two-sided bounds in `d4_bounds.json` | mid in [2.328, 2.587], deep in [2.114, 2.813] | below lower: NOT TESTABLE; above upper: INCONCLUSIVE (the realized data are not the designed condition; collection is seeded, so no rerun) |
| B1 | share of fitting-row next states (t < 99) where pi_ref's command lies inside CUTOFF's retained support for the state's load band | >= 0.99 | NOT TESTABLE |
| H1 | max over the S9 slots and the 96 anchors of \|Q_stall - Q_nostall\| / max(1, \|Q\|), exact, under pi_ref | <= 1e-9 | INCONCLUSIVE (an instrument fault: the identity is exact algebra) |
| H2 | median over the 96 gated anchors of the exact gap rho_j = Q_nostall(s10) - Q_stall(s10) under pi_ref | >= R_req = (0.5 + 0.3 x 0.15) / 0.7 = 0.7786 | NOT TESTABLE |

Every check reports its value whether or not it decides. Non-finite values never meet a bar.

## Labels

Every label computed is reported; the headline is the first in this order: INCOMPLETE, INCONCLUSIVE, NOT DATA-OOD,
NOT TESTABLE, PASS.

| Label | When | Leads to |
|---|---|---|
| INCOMPLETE | the run raised (P1 included), a cap bound, or the integrity check failed | A second attempt is allowed, with at most one code-only fix, under a new approval naming attempt 2. |
| INCONCLUSIVE | H1 fails, or D4 is above its upper bound | For H1 only: fix the instrument and rerun once (attempt 2). For D4: no rerun; the recorder's realized data differ from its design, and that is a finding about the recorder. |
| NOT DATA-OOD | D1, D2 or D3 fails | Stop this candidate (this prefix rule at this recorder and sigma). |
| NOT TESTABLE | D4 below its lower bound, B1 or H2 fails | Stop this recorder. A different recorder is a new candidate in the ledger. If H2 is the cause, it says the value gap at s10 lives in states only a stalled prefix produces, which are outside the no-stall data by construction: a finding about this candidate family, recorded, not a reason to move a bar or a scoring set. |
| PASS | every check passes | The candidate is testable under definition v4 r3. Next, each with its own approval: G1 (the supervised representation positive control on exact Q^{pi_ref} labels), the variance pilot, G3. |

**No label here says the critic over-values anything.** G2 is exact arithmetic about the data and pi_ref. It says
whether the withholding test can be run on this candidate, nothing about what any learner does.

## Attempts

- Attempt 1 needs the approval naming attempt 1. There is no retry after NOT DATA-OOD, NOT TESTABLE, D4-upper
  INCONCLUSIVE, or PASS.
- Attempt 2 is allowed only after INCOMPLETE or an H1 INCONCLUSIVE, needs an approval naming attempt 2, and may carry
  at most one code change (one manifest change). There is no attempt 3.
- A preflight failure (approval, manifest, seeds or bounds file invalid) computes nothing and is not an attempt.

## Reported only, never used to decide

- rho(s10) by anchor time and split by whether pi_ref's unprojected a0 exceeds the knee at the first continuation
  state; its quantiles; rho at s05, s20 and p10.
- The predictors' rho_hat and item 5 (max-over-k and |min_k| forms) and the item 6 sign control.
- pi_ref's mean return on the 128 starts in stall3, against the zero policy.
- Pooled-high density, the load-band shares and row counts against their Monte Carlo bounds.
- The exact simulator steps and wall-clock of every unit.

## Caps

1,000,000 exact simulator steps and 1,800 s for the attempt. The expected spend is about 500,000 steps: collection
2 x 1,280 x 100; anchor prefixes in both worlds 2 x 32 x (0 + 25 + 50 + 75); continuations 2 x 14 slots x 32 x
(100 + 75 + 50); one step per anchor for the projection readout; returns 2 x 128 x 100.

## Integrity

Before anything runs and again after, every file in `G2_FORMAL_MANIFEST.json` (this package, the v2 prefix rule and
its development record, v1's pi_ref, recorder, formal driver, seeds and bounds, the definition, and the
knee_edge_ood_v3 modules imported) must equal its sealed hash; the approval must name the manifest's hash. The
datasets are saved with their content digests (`*.npz`, git-ignored) and the report records them, so the later gates
(G1, the pilot, G3) run on exactly these rows.

## Ledger entry

Candidate knee_s10_v2 (v1 with the in-support anchor prefix), sigma s10, mode exact (pi_ref v2). Development: G2 v2
PASS (v1's development entropy and stream, outcomes known). Formal G2: this attempt's headline.
