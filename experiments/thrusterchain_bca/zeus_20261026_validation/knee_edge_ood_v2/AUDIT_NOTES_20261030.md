# Audit notes, knee-edge OOD line, 2026-10-30 (read-only audits; not a protocol document)

Two independent multi-agent audits (code path, recomputed numbers, protocol reading, adversarial challenger) were run
after each stop. Both recomputed every number from the sealed records with numpy-only scripts, opened no checkpoint and
decoded no dataset. Their conclusions, for the write-up:

## v1 step 0.3 (A2): INCONCLUSIVE on G0.5, genuine

- Code matches PROTOCOL.md lines 554-601 on every point: the seven float32 G0.5 points, the -1 and -0.5 columns found
  by bit match, the constant 0.22499999105930302 = 0.3(1 - T15(-1)^2), a1..a3 = 0, raw observations normalized inside
  the network from the common fixture, steep selection R_lin >= 1.0 from the saved slots, strict `share < 0.25`.
- F recomputed from the saved Q1 equals the stored F exactly; steep masks rebuilt from the saved Q^pi match (16 per
  seed); medians and shares reproduce 0.658/0.4375, 1.164/0.25, 1.003/0.125. Seed 2's steep F sorted:
  -1.12, -0.019, 0.528, 0.686, ...: two more probes would be needed for 25%, so the label does not hinge on < vs <=.
- The known-answer reproduction runs the same `policy.q` path as F; the new Q1 at -0.5 is bit-identical to the saved
  slot-1 Q1 on the same rows.
- Limit noted: the step never asserts the training data's stall slope; it rests on provenance (every evaluation-side
  check is consistent with 1.5).
- Challenger: not refuted. Fragility, not defect: with a fold constant of 0 the label would differ; the constant is
  verified.

## v2 step 0.4 (A3): INCONCLUSIVE on L2 item 5, genuine; item 6 recorded failing in every seed

- Predictors: P1 quadratic LSQ on S9, P2 line on the five edge slots in [-KNEE, -0.69999999], P3 cubic on S9; x values
  are the sealed float32 slot values; fits use Q_stall at S9, which equals Q_nostall there to exactly 0.0 at all 48
  gated anchors in all seeds. item5_error = max over k |P_k - Q_nostall| at s05/s10/s20 with 0.10/0.15/0.30; share over
  valid anchors (48/48 valid) >= 0.90, so 44 needed. Recomputed errors match the record to <= 3.1e-13; shares 44/42/42.
- Failing anchors: seed 0 -> 9, 11 (t=0), 29, 30 (t=25); seed 1 -> 18, 21, 28, 29, 30, 31 (all t=25); seed 2 -> 2, 4, 9
  (t=0), 18, 28, 30 (t=25). Anchor 30 fails in every seed. No t=50 anchor fails. Errors grow about linearly with depth
  (1 : 1.5 : 3, the tolerance ratio), so no single slot is the binding one.
- Cause: the global-span quadratic P1 (secondarily P3) misfits Q^pi over [-KNEE, KNEE] at a few state-specific anchors
  (in-sample S9 residual 0.07-0.70 there, against 0.02-0.03 elsewhere). The local edge line P2 meets item 5's 0.10/0.15
  tolerance at every one of the 144 gated anchor-seeds (max P2 error at s05 0.05). The no-stall Q continues the edge
  trend smoothly across the knee (second derivative about -2.5 on both sides). So the miss is the designed consequence
  of including the global fits in the max over k (L2.3 "harder to pass, never easier"), an estimator-design limitation,
  not a property of the environment and not a code defect.
- rho_hat under-estimates the true rho slightly (medians -0.013 to -0.05), the conservative direction for items 1-4.
  True rho medians: s05 1.12/0.95/1.25, s10 2.28/1.95/2.55, s20 4.64/4.05/5.17.
- Item 6 (sign control, max over k rho_hat(p10) < 0 at >= 90%): shares 0.48/0.27/0.60, fails in every seed. Physics
  confirms the intended sign: at these high-load anchors positive thrust is harmful, the fold at +0.9 helps, and the
  true rho(p10) is negative at 96-100% of anchors (medians about -3.4). P1 and P3 each give rho_hat(p10) < 0 at 96-100%
  and would pass alone. The failure is entirely P2: a line fitted on [-0.8, -0.7] and extrapolated 1.6 units to +0.9
  overshoots by a median +3.3 and attains the max at 48/48 anchors in every seed. Under revision 1's wording (min over
  k) item 6 would pass in every seed; revision 2 switched the control to max over k for conservatism without a
  numerical check, and no design screen ever evaluated p10. A design error in the sign control, sealed in r13.
- Had item 5 passed, the label would have been FAIL-consistent on item 6 ("not shown model-relevant").
- Wording gap with no effect here: the share is taken over valid anchors; the text says "of anchors" (48/48 valid).
- Challenger: not refuted; under the per-slot reading seeds 1 and 2 still fail (s10 at 0.875). It notes that an
  exploratory Stage 1 run (`stage1_exploratory.py`, outside the protocol, by the user's decision) was launched in this
  package after the step 0.4 report, so "no continuation" is no longer true of the package as a whole.

## What the step 0.4 record establishes, label aside

Under the three reference continuations: the L2.2 premise (bit-identical worlds on the support, exact identities, T3
thrust, twin divergence); a true one-step gap rho = Q_nostall - Q_stall with medians about 1.0-1.25 at s05, 1.95-2.55 at
s10, 4.0-5.2 at s20; a sign flip to about -3.4 at p10; items 1-4 above their bars in every seed; reach-forecast medians
0.84-1.14 (s05) and 1.73-2.32 (s10) against Stage 1's 0.60 and 0.65. Not established: the pre-registered validity
rule (two anchors short in two seeds), the pre-registered sign control, and anything about Stage 1 learners.

Full records: the workflow journals under
`C:\Users\David\.claude\projects\C--Users-David-dev-BCA\be215736-77cd-47c3-bb51-9f394d41a6ce\subagents\workflows\`
(wf_3c59420a-436 for A2, wf_cf1fcc2a-e1f for step 0.4).
