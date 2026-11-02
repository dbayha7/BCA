# What we are testing, and what to do next (2026-10-31)

This note answers the user's request: "Keep pushing to get this experiment setup correct and really think about what
we are doing here." It draws on a design panel (workflow `wf_f375f877-2cb`: a mechanism audit, a question audit, a
steelman of the current path, an adversarial critic, a synthesis and two checkers) and on a new readout whose rule was
written first (`bca_leverage_readout_v1/`). Everything here is exploratory. Nothing was trained or approved, and no
checkpoint was opened.

## 1. The question the study is meant to answer

"Whether placing caution where a learned critic is wrong, based on calibrated uncertainty, beats spreading caution
evenly or simple data-only rules." (docs/session-history/2026-10-14_cloud_session_01Ro4BnWxcsoeFi9CGcTQ5bx.md, Part 1)

## 2. What the recent work actually tests

- **The OOD gates test the benchmark, not BCA.** This covers the knee edge v1/v2/v3, the definition v4 and the
  knee_s10 candidate. They ask whether withholding labels makes this host's critic over-value a region. That is a
  property of TD3+BC and the data. BCA's uncertainty U appears in no gate.
- **The sealed Stage 2 does not test the stated question.** It compares BCA with native and with a "distance" arm,
  and that arm penalizes moving away from the frozen 10k actor, which already acts past the knee on about half of
  high-load rows. So it is a proximal control, not a data-only rule. Stage 2 has no even-caution arm and no data-only
  arm. Its main score, BCA's gain in CUTOFF minus its gain in RANDOM, rewards flagging novelty, not flagging where the
  critic is wrong. RANDOM is harmed as much as CUTOFF (delta_clipall 285-303 in both stall cells).

## 3. What BCA is, from its code

- **U(s, a) = R x eta(s, a) x u.**
  - eta is a single 2x64 ReLU network on the normalized observation and the raw action. It is fit to one-step
    twin-min TD residuals on training rows only (`uncertainty_source/historical/td3_bc_bca.py`).
  - R is one conformal quantile over 128 held-out rows, with unit weights (`calibration/reference.py`
    `freeze_reference`).
  - u is a running residual scale.
- **No part of it grows with distance from the data or with shift.** The weighted-conformal path exists but is not
  used. There is no ensemble and no density feature.
- **The actor sees only the action gradient of U.** So a U that varies with the state but not with the action leaves
  the actor exactly as native.

## 4. The evidence

1. **Central quartile (on file).** Within one state, U varies by 0.5-0.8% across a0; across states it varies by
   9-12.5%. U is a per-state offset. Its accuracy gain came from subtracting a near-constant 3U of about 15.
2. **Leverage readout, new (`bca_leverage_readout_v1/`, rule written first): NO LEVERAGE.**
   - **Data:** the gradient audits saved during the exploratory Stage 1, 768 logged rows per run, at the actor's own
     actions, on high-load rows where the actor is past the knee.
   - **Results at 20k, in the stall world:**

     | Readout | CUTOFF | RANDOM |
     |---|---|---|
     | Share of outward pushes that BCA reverses at its configured dose (kappa 3) | 1.5-3.1% | 1.8-3.1% |
     | BCA's pull on a0 relative to the critic's | 2.6-5.7% | 5.3-12.2% |
     | Share of rows where BCA points inward at all (the ceiling at any dose) | 0.29-0.69 | 0.69-0.99 |
     | Median dose that would cancel native's push (method grid max: 10) | 35-76 | 25-48 |
     | Share reversed at kappa = 10 | 4.6-9.8% | 8.0-19.2% |

   - **Placement:** stall-world inward share minus no-stall inward share is mixed. CUTOFF gives +0.32 / -0.14 /
     -0.06 by seed; RANDOM gives -0.03 / +0.12 / +0.45.
   - **Limits:**
     - these are logged training states, not the evaluation anchors;
     - the analysis is linear in the gradient and ignores the actor network and curvature;
     - there are only 3 seeds.
3. **The bed works as a harm bed.** Under the native host, in both data arms, the critic over-values past the knee,
   the actor goes there (about 70% of high-load steps) and loses about 295 return. The values are exact and the
   no-stall world gives the counterfactual.

## 5. What this means

- **Stage 2 on this bed would most likely show BCA about equal to native.** BCA as built, at its configured strength,
  cannot steer this actor away from the knee, with or without labels. More benchmark engineering would not change that.
- **Where the data is withheld (CUTOFF), BCA's push has no consistent direction.** That is expected from a method with
  no mechanism for missing data.
- **With labels (RANDOM), BCA's push mostly points the right way (69-99% inward) but is about 10-25 times too weak.**
  RANDOM is exactly the "labelled but wrong" case where the study's claim lives. A data-only rule cannot see it there,
  because the data exists.
- **The OOD withholding program and the BCA question come apart.** The first is a valid question about the host and
  the data, and a separate, optional program. It is not a precondition for testing BCA's claim.

## 6. The experiment structure the question needs

Each claim gates the next.
1. **Leverage (L):** at the dose to be tested, BCA reverses a material share of the actor's harmful pushes. This is
   checked first, cheaply, on audits or checkpoints.
2. **Room (R):** on this bed, an oracle-placed penalty (on the past-knee region in the stall world only) beats even
   caution (a lower TD3+BC alpha) and a true data-only rule at matched no-stall cost. This is the program's own rule
   ("If even perfect placement can't win, no result means anything"; H3 failed on an earlier host).
3. **Placement (P):** within a data arm, BCA pushes inward more in the stall world than in the no-stall world, where
   going past the knee helps.
4. **Efficacy (E):** in a labelled-but-wrong cell, BCA beats even caution and the data-only rule at matched no-stall
   cost, positive in every seed.
   - RANDOM is labelled only at onset depth at high load (deep labels only at low load).
   - The knee_s10 wide-recorder twin is labelled deep at high load. It is the cleaner cell if its native critic
     still fails to learn there.

## 7. Decisions for you

1. **Which claim is primary?** Recommended: "caution placed by calibrated uncertainty beats even and data-only caution"
   (the stated question). The OOD-withholding attribution becomes a separate, optional program.
2. **What to do about the leverage verdict?** Recommended:
   - Record it as the finding for BCA as built: on this host, U acts as a per-state offset with about 3% of the needed
     action leverage.
   - If BCA work continues here, the next compute is a pre-registered leverage screen of BCA at a higher dose, fixed
     now by a rule (for example, kappa set so BCA's median pull equals the critic's), on fresh seeds in the RANDOM and
     wide-twin cells. That screen comes before any Stage 2.
   - A method change that gives U real action resolution would be a new method with its own screen. The Q1-aligned
     variant already failed one (`action_sensitive_bca_dev_v1`).
3. **The OOD v4 / knee_s10 formal check:** recommended to pause. It does not gate any BCA claim.
4. **Comparison arms for any future Stage 2:** add even caution and a true data-only rule, each at matched no-stall
   cost; relabel "distance" as a proximal control.

## 8. Files

- `bca_leverage_readout_v1/RULE.md`, `leverage.py` and `leverage_result.json`: the new readout.
- `ood_definition_v4/DRAFT_DEFINITION.md` (r3) and `knee_s10_candidate_v1/`: the OOD program, paused, with every
  development result on record.
- The panel's reports are in the session scratchpad and the workflow journal `wf_f375f877-2cb`.
