# Shelving note: the Q-error method-change track on the knee bed (2026-10-31)

- **Result:** `SIGNAL_RESULT_20261031T231314Z.json`. **Verdict:** NO SIGNAL, under `RULE.md`, which was committed in
  f79c4eb6 at 15:01 EDT, before the run at 19:13 EDT.
- **What is shelved:** the method-change track. That is the line that fits BCA's U to the critic's Q error, whether as
  a one-step signed head or as a propagated (BCA-PE) head.
- **Where:** this bed only. That means the ThrusterChain stall-knee bed on the Unifloral TD3+BC host, with the
  knee-edge CUTOFF and RANDOM data.
- **What is not shelved:** the BCA concept as a whole, and other beds.

## Terms

- **Q1:** the critic head the actor maximizes.
- **delta1:** Q1's own one-step error on a logged transition, Q1(s,a) - r - 0.98 (1 - done) Q1(s', pi(s')).
- **Edge rows and onset rows:** high-load training rows whose logged a0 is just inside the knee ([-0.8, -0.7]) or just
  past it ([-0.9, -0.8)).
- **I_w:** mean delta1 on onset rows minus mean delta1 on edge rows.
- **D:** I_w in the stall world minus I_w in the no-stall twin, in the RANDOM arm. RANDOM is the arm that has labelled
  rows past the knee.
- **KC_s05:** at the 48 evaluation anchors, how much Q1 over-values the action -0.85 relative to the knee edge,
  measured against exact Q^pi.

## The idea

- **What BCA should do.** BCA should put caution where the critic's Q values are wrong.
- **The question.** On this bed the critic over-values actions just past the knee. If its own one-step error shows that
  over-valuation where labelled data exists, then a U fit to Q error can learn where the critic is wrong.

## What we tested (and the kill rule)

- **Measurement.** Forward passes only, on the 12 exploratory runs (3 seeds x CUTOFF/RANDOM x stall/no-stall), at
  20k to decide and at 10k as a report.
- **The data.** The checkpoints come from the knee-edge v2 exploratory Stage 1 run, which supports no claim.
  - All three seeds share one dataset and one anchor set.
  - delta1 was measured on the rows the critic trained on, with no held-out residuals.
- **Kill rule, as written before the run:**
  - SIGNAL if D >= 0.20 in at least 2 of 3 seeds.
  - Otherwise NO SIGNAL, which means "shelve the method-change track on this bed".
- **Integrity.** The result records the rule and code hashes, and both equal the committed files. The imported v3
  modules were committed earlier (fc12abf0) and were clean at run time.

## What happened

- **The decision.** D was 0.172, 0.207 and 0.172. It was positive in all three seeds, and the no-stall control's I_w
  was only 0.003-0.021. It reached 0.20 in one seed, not two, so the verdict is **NO SIGNAL**.
- **Validity.** It passed: Q1 at every anchor matched the saved evaluation exactly, every delta1 was finite, the
  restored states matched, and every RANDOM bin was populated. CUTOFF has no onset rows by design.
- **Reported, not deciding.**
  - At the 10k fork, D was 0.326, 0.321 and 0.302, so it was smaller at 20k in every seed.
  - The deep bin (a0 < -0.9, high load, only 28 rows) had mean delta1 of +0.61 to +0.87 in stall against -0.44 to
    -0.54 in no-stall, with SD 2.0-2.5.
- **Ruled out.** None of these reverse or soften the verdict, and none is done:
  - reading 10k as deciding;
  - pooling the seeds into one test;
  - counting seed 1 as partial support;
  - re-binning;
  - rerunning.

## Why (descriptive, analyzed after the result)

The rule assumed that the critic's over-valuation past the knee would show in its one-step error on the labelled
rows. It shows only partly.

- **The error is one-step and local at the anchors.** In the stall world, -0.85 delivers the same thrust as -0.65, so
  the two actions share a next state. That means Q^pi(-0.85) - Q^pi(-0.65) = -0.09 exactly, from the action cost alone.
  - Q1's error on that twin pair is a pure one-step error. It is 0.53-0.66 in RANDOM and 0.60-0.67 in CUTOFF, which
    is about 0.8-1.1 times KC_s05 (`post_hoc/twin_split.py`).
  - The interpolation it uses reproduces the exact Q^pi identity to within 0.003.
  - So at the anchors the over-valuation does not arrive through the continuation, as cause 1 in
    `METHOD_DESIGN_NOTES.md` supposed. The critic is wrong about the immediate step.
- **The labelled rows don't show it at that size.** On the critic's own training rows, the stall-specific one-step
  contrast is D of about 0.2, roughly a third of the anchor error.
- **The labels didn't fix the anchors.** RANDOM's anchor over-valuation is about the same as CUTOFF's, although
  RANDOM has 9,530 labelled onset rows.
- **The simplest reading.** The critic fits its labelled training rows but stays wrong at the anchor states. So a U
  fit to in-sample residuals on training rows sees the small number, not the error that matters.
- **Open, and not tested here.** Whether held-out (cross-fitted) residuals would see the larger error, or whether the
  anchor states simply lie away from the labelled rows' states, is still open. Testing it on this bed would be a
  rescue designed after seeing D.
- **Correction to the record.** RULE.md's reference range "KC_s05 0.57-0.75" is CUTOFF's range. RANDOM's is
  0.59-0.64 (common anchors). The reasoning behind the bar is unchanged, and nothing is re-scored.

## Where the knee bed stands

These three exploratory readouts all point the same way. None supports a claim.

| Readout | Result |
|---|---|
| Leverage | NO LEVERAGE: at kappa 3, BCA's push reverses 1.5-3.1% of the actor's outward pushes. |
| Dose screen at kappa 63 | CUTOFF MIXED, RANDOM MIXED, so no claim. Descriptively, on fixed networks, an even penalty corrected more of the critic's wrong rankings than BCA's U at matched false caution. |
| This check | NO SIGNAL. |

On this bed there is no evidence that caution placed by calibrated uncertainty beats even caution.

## Lessons for a next bed's rule

- **Measure the identifiable error where it matters.** That means at the anchor states, or on held-out rows, not only
  in-sample.
- **Add a positive control** before the rule is trusted.
- **Make the code and the text of the rule agree** on what counts as a missing bin.
- **Set bars on selectivity and signal-to-noise, then check them on hypothetical estimators.** An oracle must pass,
  and raw Q1 and a uniform ramp must fail.
