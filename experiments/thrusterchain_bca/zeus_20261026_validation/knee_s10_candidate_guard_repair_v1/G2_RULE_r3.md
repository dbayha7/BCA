# Candidate knee_s10_v1: development G2 rule, revision 3 (written before the r3 screen ran)

**Supersedes G2_RULE.md for the next development run only.** G2_RULE.md and its result, g2_development_result.json
(INCONCLUSIVE, H1 at 43 of 48), stay on record unchanged. That result also stays in the candidate ledger as a
development item-5 miss, together with v2 step 0.4's.

This is still a development screen. It reuses the same test entropy (424263) and test-only stream (707,070,707), whose
outcomes under v1 are already known. It can shape the formal rule. It never admits the candidate.

## Changes from G2_RULE.md, and why

The reasons come from the review (wf_e1964e57-92d) and from the definition's own estimand. None is chosen because it
passes.

1. **pi_ref version 2 stays inside both arms' data** (`pi_ref.py`).
   - Side commands are clipped to +-0.7.
   - At high load, a0 is clipped to [-KNEE, -0.6].
   - Why: v1's commands fell outside the recorded data at 78% of next states (side commands) and at 36-51% of
     high-load next states (a0 above -0.6). Policy-evaluation critics could then reach Q^{pi_ref} only by
     extrapolating. This is a property of pi_ref, not of any outcome.
2. **"Hidden" is established exactly, not by fitted predictors.**
   - Under a fixed pi_ref, both Q_stall and Q_nostall are exact, and the withholding estimand theta already uses the
     no-stall world as the smooth counterfactual (DRAFT_DEFINITION.md section 4, E1).
   - So H1 is the L2.2 identity: stall equals no-stall at every supported S9 slot, to 1e-9 relative. Per-tuple data
     on the support cannot tell the worlds apart.
   - The gap at sigma is exact: rho_j = Q_nostall(s10) - Q_stall(s10).
   - The predictors P1-P3 (rho_hat, item 5, item 6) are reported, never used to decide. They were a proxy for "what a
     smooth extrapolation predicts", and theta's no-stall difference replaces that proxy.
   - Development note: the coherent item-5 form, |min_k P_k - Q_nostall| at s10, also gives 43 of 48 on the v1 run. So
     keeping item 5 as a gate with any defensible form would not have changed v1's label. Dropping it is a change of
     estimator, decided here, and it applies to the formal G2 only if the user adopts definition r3.
3. **H2 uses the same band as L1** (V0(f)'s form, c(rho + tol)): R_req = (0.5 + 0.3 x 0.15) / 0.7 = 0.7786, as before.
4. **New gated row B1, bootstrap support.** At the next states of all fitting rows with t < 99, in both worlds, pi_ref's
   command must lie inside CUTOFF's retained support for that state's load band at >= 99% of rows:
   - high load: a0 in [-KNEE, -0.6] and |a1..a3| <= 0.7;
   - low load: |a0| <= KNEE and |a1..a3| <= 0.7.

   The twin's support contains CUTOFF's.
5. **D1 uses the sealed code.**
   - CUTOFF comes from `masks.arm_masks`.
   - `region_independent.withheld_region` must agree with `knee_region.past_knee_a0` row for row.
   - D2 measures distance to float32 s10 and also requires that no retained high-load a0 is <= s10.

## Checks and bars

| id | Check | Bar |
|---|---|---|
| D1 | arm_masks CUTOFF holds 0 W rows; the two independent W predicates agree row for row | exact |
| D2 | no CUTOFF high-load a0 <= s10; distance from s10 to the nearest retained a0 | none; >= 0.0999 |
| D3 | L1.5 as amended, each world, all fitting rows | starts >= 95%, anchors >= 90% |
| D4 | twin density at a0 in [-0.95, -0.85], per high-load band (development lower bars; the formal G2 needs two-sided six-SD Monte Carlo bounds for this recorder) | mid >= 2.00, deep >= 1.82 |
| B1 | bootstrap support of pi_ref at fitting-row next states, each world | >= 99% |
| H1 | max over S9 slots and anchors of \|Q_stall - Q_nostall\| / max(1, \|Q\|) | <= 1e-9 |
| H2 | median over the 48 gated anchors of the exact rho_j(s10) | >= 0.7786 |

## Labels and what each leads to

| Label | When | Leads to |
|---|---|---|
| PASS | every row passes | The candidate may be sealed for a formal G2 (new entropy, new streams, 96 gated anchors, two-sided D4 bounds, pi_ref committed) once the user adopts definition r3. |
| NOT TESTABLE | D4, B1 or H2 fails | Stop this recorder. |
| NOT DATA-OOD | D1, D2 or D3 fails | Stop. |
| INCONCLUSIVE | H1 fails | An instrument fault. The identity is exact algebra, so a miss is a bug: fix it and rerun once. |

## Reported only, never used to decide

- rho split by whether pi_ref's a0 is projected at the first continuation state.
- The projection-binding shares along the continuation.
- The predictors' rho_hat and item 5, in both the max-over-k and the |min_k| forms.
- Item 6: max over P1 and P3; P2 at p10 separately, marked as outside its fitting range; and the exact sign share.
- pi_ref's mean return.
