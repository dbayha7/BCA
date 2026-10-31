# Q-error signal check v1: can Q1's own one-step error see the knee where labels exist? (rule, written first)

Written 2026-10-31, before any number below was computed. Exploratory: it uses the knee-edge exploratory checkpoints
(development data). It runs forward passes only. Nothing is fit and nothing is trained. It is user-run in WSL.

## Why this check comes first

The user's main concept: **BCA acts on Q values.** U should be a calibrated uncertainty about the critic's own Q
error, E = Q1 - Q^pi, and the actor maximizes Q1 - kappa x U.

The method-design panel (`wf_bdf1de0d-a7e`) proposed fitting U to E, propagated through a Bellman recursion. Its
checkers showed that most of E is not identifiable from the data. Under the actor's own continuation, E in the stall
world is about 200, built along paths the actor takes off the data, where no row carries a label.

What the data does identify, at every logged transition, is Q1's own one-step error:

  delta1(s, a) = Q1(s, a) - r - gamma (1 - done) Q1(s', pi(s'))

where Q1 is the head the actor maximizes, pi is the online actor (deterministic), and gamma is the host's discount
(0.98, `args.gamma`). The stall identity gives delta1(s, a) - delta1(s, T(a)) = E(s, a) - E(s, T(a)), with the same
next state. So where the critic over-values a labelled past-knee action, its one-step error there should carry that
over-valuation.

**If Q1's one-step error does not show the kink at RANDOM's labelled rows, then no U fit to Q-value error can learn
where the critic is wrong in this region from data, whatever the method.** That would settle the method-change track
on this bed before anything is built.

## Measurement

For each exploratory run (3 seeds x {CUTOFF, RANDOM} x {stall3, nostall}), at the 20k checkpoint (decides) and the 10k
fork (reported):
- delta1 on every training transition the critic trained on (`context.train_transition`);
- bins by the logged a0 at high load (load g >= 0.4):
  - **edge**, in support: [-0.8, -0.7];
  - **onset**, past the knee: [-0.9, -0.8);
  - **deep**: < -0.9.

  Low-load past-knee rows are reported separately.
- **I_w = mean delta1 (onset, high load) - mean delta1 (edge, high load).** This is a local contrast, so the
  per-state level largely cancels.
- **Reported:**
  - mean, median and SD of delta1 per bin;
  - row counts;
  - the deep and low-load bins;
  - gamma;
  - the same for CUTOFF, which has no onset rows, and that is reported as such.

## Decision (RANDOM arm, 20k)

D_i = I_w(stall3) - I_w(nostall) for seed i.

| Verdict | Rule | Leads to |
|---|---|---|
| SIGNAL | D_i >= +0.20 in at least 2 of 3 seeds | Q1's own error carries the kink where labels exist, so a U fit to Q-value error can learn it. Build the Q-error BCA variant (one-step signed head primary, propagated head reported), with its own screen and rule. |
| NO SIGNAL | D_i < +0.20 in at least 2 of 3 seeds | The critic's own one-step error does not register its over-valuation past the knee, even with labels. Every U fit to Q-value error from these data is blind there. Shelve the method-change track on this bed ("The idea / What we tested / What happened / Why"). |

**Why 0.20.** If Q1 ignores the kink entirely, the one-step error at onset rows carries up to the onset
contrast error, KC_s05, which is 0.57-0.75 in the exploratory evaluation. 0.20 is about a third of that. It is also
about the change needed to flip the median s10 ranking of raw Q1 (raw contrast 0.19-0.26).

A validity check that fails, or a missing bin in RANDOM, makes the result INVALID; report it, with no rerun under
this rule.

## Validity checks

- Q1 at the 48 gated E2 anchors' 14 slots, recomputed from the restored policy, matches `evaluate_s*/report.json`
  `anchors_e2.q1` to relative 1e-5.
- Every delta1 is finite.
- The restored state digests match the reports (`stage1_worker.policies` enforces this).
