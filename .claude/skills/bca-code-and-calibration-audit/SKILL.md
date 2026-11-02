---
name: bca-code-and-calibration-audit
description: Audit an implemented Bayesian-calibrated actor penalty, its residual model, calibration, gradients and evaluation claims against source code and saved evidence.
---

# BCA code and calibration audit

Trace the actual implementation rather than substituting a generic conformal or Bayesian algorithm. Use [the dated project reference](references/thrusterchain-bca-2026-10-27.md) only as a lead; reverify every identity and setting in the checkout being examined.

1. Identify the host implementation and immutable source/config/checkpoint identities. Trace the actor loss, Q normalization coefficient, BC term, critic loss and target, including stop-gradients. State precisely whether BCA changes the actor objective, critic training, TD targets or some combination.
2. Trace the residual target and its sign: the particular critic head or twin-min, one-step or multi-step operator, terminal mask, policy and action smoothing. Distinguish this target from simulator-estimated same-policy full-return Q error; neither an absolute residual nor a posterior draw alone identifies over- versus underestimation.
3. Trace local scale inputs, preprocessing, network output/positivity, numerical floors and residual-unit updates. For calibration, record data roles, score formula, empirical quantile convention, optional importance weights, test atom, Bayesian draw distribution/count, threshold selection, alpha and beta meanings. Verify the actual caller's arguments, not merely general library defaults.
4. Check readiness, refresh and freeze rules, invalid/infinite threshold behavior, whether action gradients remain live while reference parameters are frozen, and checkpoint/resume state. Review zero-penalty baseline parity and finite-difference/gradient routing tests without loosening tolerances to force a pass.
5. Separate bootstrap calibration draws from critic heads, and empirical residual calibration from claims about true-Q coverage or shifted-distribution guarantees. Flag violated assumptions and unknowns explicitly.
6. Analyze Q-unit scales and action-level penalty differences. A constant offset across actions cannot change ranking. Check near-zero denominators, floors and stop-gradient treatment; do not assert scale invariance or choose a universal coefficient without evidence.
7. Map each claim to source lines and a specific test or receipt. Distinguish magnitude calibration, error direction detection, action selection and downstream policy return. Do not start experiments, install dependencies, alter methods or access restricted systems without separate authorization.

**Deliverable:** source-anchored actor/critic equations, calibration pseudocode and update order; tests and failure behavior; a claim/evidence table distinguishing validated mechanics from untested efficacy.

