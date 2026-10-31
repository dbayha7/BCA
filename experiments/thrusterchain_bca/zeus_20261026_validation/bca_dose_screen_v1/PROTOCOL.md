# BCA dose screen v1: does BCA as built help at a higher dose? (protocol, written before any number it judges)

Written 2026-10-31 after the user chose "Higher dose and change the method". This is the higher-dose track. The
method-change track is separate. **Exploratory:** it runs from the knee-edge exploratory Stage 1's 10k forks
(development data, outcomes known), so it supports no protocol claim. It needs the user's own terminal approval
(`SCREEN_APPROVAL.json`) and is user-run in WSL.

## The concept this screen stays on (the user's instruction: "Make sure we are acting on q values")

BCA acts on Q values. U is BCA's calibrated uncertainty about the critic's own Q values, and the BCA actor maximizes
the penalized value Q1 - kappa x U. BCA works here only if the penalized Q ranks actions correctly where raw Q1 ranks
them wrongly, measured against exact Q^pi. So the decision below requires the effect to come through the Q values: a
behavioral change without a Q-value correction does not count as EFFECT.

## Why

The leverage readout (`../bca_leverage_readout_v1/`, verdict NO LEVERAGE) showed BCA as built, at kappa 3, reversing
1.5-3.1% of the actor's outward pushes past the knee. With labels (RANDOM), its push mostly points inward (0.69-0.99)
but is about 10-25 times too weak; without labels (CUTOFF), it has no consistent direction. This screen asks whether
BCA at a dose strong enough to matter reduces harmful entry past the knee, and whether it does so selectively: more
where entry is harmful (the stall world) than where it helps (the no-stall world).

## Dose rule (fixed now, computed by `dose.py` from pre-branch data)

- **kappa_A** is the median, over the 12 exploratory forks, of each fork's median "dose needed" at its 10k warmup
  gradient audit. The dose needed is defined in `../bca_leverage_readout_v1/RULE.md` readout 4: on high-load rows
  where the actor is past the knee, native pushes outward and BCA points inward, the kappa at which BCA would just
  cancel native's push. Rounded to the nearest integer.
- **Why this rule:** it is the dose at which BCA cancels the typical native outward push where its direction is right.
  It uses the 10k fork states only, which are the states every branch starts from, and no branch outcome.
- **Known limits** (`DOSE.json` records each per fork, from the 10k audits):
  - Where BCA points the wrong way, kappa_A also strengthens its outward push 21-fold.
  - Half of the pooled forks are no-stall forks, where entry helps.
  - The per-fork doses spread widely, so one kappa under-doses some forks and over-doses others.
  - At kappa_A the penalty's action gradient outweighs native's on most probe rows. So this screen tests "BCA's
    penalty at a dominant strength", and selectivity is what separates information from mere strength.
- **Disclosure:** the rule was chosen after the leverage readout printed these per-fork values, but before any branch
  at that dose existed.

## Design

- **Forks:** the 12 exploratory 10k warmups (3 seeds x {CUTOFF, RANDOM} x {stall3, nostall}) in
  `knee_edge_ood_v2/stage1_exploratory_run/attempt_1/warmup_*`. Checkpoints are authenticated against their reports.
- **New units:** one BCA branch per fork, 10k to 20k, at kappa_A, with the fork's sealed Stage 1 branch seeds. These
  are the same seeds its existing native branch used. Each pair uses shared random numbers: the same fork state, the
  same branch seeds, minibatch stream and target noise (kappa does not change how random numbers are consumed). The
  worker requires the minibatch hash, the data binding and the warmup report to equal the native branch's. The two
  actors still diverge after 10k, so the per-seed difference includes that divergence.
- **Gradient audits:** at 15k and 20k, with the applied penalty.
- **Comparator:** the existing native branch of the same fork (`branch_*`), with its evaluation already on record
  (`evaluate_s*`). The paired difference is BCA minus native.
- **Evaluation:** per seed, the same measurements as the exploratory evaluation on the same banks and anchors:
  - free-running returns J and entry rates on E2 and E1 in the run's own world, and for stall runs also in the
    no-stall world;
  - anchor values (Q^pi, Q1) at E2 and E1;
  - visited states;
  - plus U at the gated slots of the E2 and E1 anchors.
  - No BC-only actor (it already exists).
- **Caps:** 1,800 s per branch, 6 h of training, 1 h and 18,000,000 exact simulator steps of evaluation.

## Readouts, per seed i and data arm A (CUTOFF, RANDOM), at 20k

| Readout | Definition |
|---|---|
| Delta_entry(world) | R_deep- entry rate on E2 (pi, own world), BCA minus native |
| Delta_J(world) | E2 return (pi, own world), BCA minus native |
| Sel | Delta_J(stall3) - Delta_J(nostall): the gain where entry hurts, minus the change where it helps |
| Placement | Delta_entry(stall3) - Delta_entry(nostall) |
| Applied reversal (descriptive only) | in the BCA branch's 20k audit, among high-load rows where the BCA actor is still past the knee and native's push is outward, the share whose net push with the applied penalty is inward. It is biased by survivorship, so it is reported beside the predicted reversal at kappa_A from the 10k audits (`DOSE.json`). |
| Q-value readouts (BCA stall3 run, 48 gated E2 anchors, against exact Q^pi under the run's own clipped continuation) | true contrast C_true = Q^pi(slot) - Q^pi(edge_knee); raw C = Q1(slot) - Q1(edge_knee); penalized C = raw C - kappa_A x [U(slot) - U(edge_knee)]. KC and penalized KC (C - C_true). On the anchors where C_true < 0 (the past-knee slot is truly worse), the share ranked correctly (C < 0) by raw Q1 and by the penalized Q. **q_rank_gain_s10** = penalized share minus raw share at s10, within the BCA run. |
| Reported only | the same raw readouts for the native runs; U at the anchors; delta_clip; E1 returns; the pairing receipt |

## Decision, per data arm (fixed now; revised before sealing after review `wf_04029561-cff`)

In precedence order:

| Verdict | Rule |
|---|---|
| INCOMPLETE ARM | Some seed has no evaluation, because a branch failed (a dose failure) or a cap bound. |
| EFFECT (selective, through Q values) | In every seed: Delta_entry(stall3) <= -0.10; Placement <= -0.05 (entry falls at least 0.05 more where it hurts); Delta_J(stall3) >= +10 (above within-run moves of up to 8.1 between 15k and 20k); Delta_J(nostall) >= -5; AND q_rank_gain_s10 >= 0.20 (the penalized Q ranks s10 below the knee where the truth does on at least 20 points more anchors than raw Q1). Added at the user's instruction before sealing. |
| BEHAVIORAL EFFECT NOT THROUGH Q | Every behavioral part of EFFECT holds in every seed, but q_rank_gain_s10 does not. |
| NON-SELECTIVE REDUCTION | Delta_entry(stall3) <= -0.10 in every seed, without the rest of EFFECT. |
| HARM | In at least 2 seeds: Delta_entry(stall3) >= +0.05, or Delta_J(nostall) <= -20. |
| NO EFFECT | Delta_entry(stall3) > -0.05 in every seed. |
| MIXED | Anything else. |

Non-finite values never meet a bar. With three forks and unanimity required, this screens only for large effects.

## What each verdict leads to

- **EFFECT in RANDOM:** at this dose, BCA-as-built steers selectively where data exists but the critic is wrong. The
  next step is the efficacy comparison against even caution (a lower TD3+BC alpha) and a true data-only rule, at
  matched no-stall cost, on fresh seeds. It needs a new protocol and approval, and must include a generic penalty of
  matched strength. This screen has none, so EFFECT here shows selectivity, not that BCA's information beats strength.
- **EFFECT in CUTOFF only:** at this dose BCA acts as a penalty at the edge of the data. Expect parity with a
  data-only rule.
- **NON-SELECTIVE REDUCTION:** a strong penalty reduces entry everywhere. That is not evidence of BCA's information.
- **HARM:** the dose makes things worse.
- **NO EFFECT in both arms:** dose alone does not rescue BCA as built. The method-change track is the path.
- **MIXED or INCOMPLETE ARM:** report it, and make no claim.

## Failures

- **A preflight failure** happens before the attempt exists (`screen_run/preflights/`). It uses up nothing.
- **A failed BCA branch** is a dose failure. The other forks still run, and that seed's evaluation is skipped.
- **Anything that raises** still produces `SCREEN_RESULT.json` (INCOMPLETE, with the traceback).
- **`screen.py --decide`** recomputes the decision from the saved reports. It is JSON only.
- **No retries.**

## Integrity and accounting

- **Integrity:** the worker verifies every file of this package, the v3 package's dependency closure, rerun1's
  pinned sources, the payload modules and the supervisor before any import, and again after the work.
- **Accounting:** exact simulator steps on every report; one attempt; every unit's spend recorded.
- **Never:** no sealed file is edited, and no agent opens a checkpoint.
