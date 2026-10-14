# How to restructure BCA so it contributes more

Tags: [measured] from a saved file; [inferred] reasoned from measurements; [speculative] a guess. All seed 0-4 results are exploratory. Paths: CAL, TB, AA, CONS, JB are the calibration, testbed, action_aware, consumer and judge_builder folders under scratchpad/restructure/; IMP is scratchpad/impact/.

## 1. The core problem

BCA's own part (the scale network and its threshold) learns only from recorded steps, so its width hardly moves between a harmful past-80% command and the 80% cap: 0.31-0.36 at both, against a false preference of 0.63-0.89 (IMP/decompose_exact.json) [measured]. In the weighted version, 86% of the extra width came from the classifier's refusal term and 0.2% from BCA's network: novelty detection, delivered through a soft penalty that still left the actor at 95%+ throttle on 22% of strong-current steps (IMP/actor_readout_seed*.json) [measured]. In ThrusterChain the recorded commands are the same in both worlds, so any signal computed from where the data lie (BCA as built, the classifier, the range hinge) costs in the no-stall world what it earns in the stalling world. What differs is the outcome of the 28-59 full-throttle steps that survived the gap (the removal of most full-throttle commands from the log), and BCA as built drowns them: about 0.1% of its rows, 0-2 in its calibration bank (CAL/e2b_past80.json) [measured].

## 2. The restructured BCA: make it read outcomes

Stop asking BCA's uncertainty to "see" the command: no ensemble or posterior width we could build responds to it (two critic heads change by 0.004-0.006 between command and cap, a last-layer Bayesian posterior by at most 0.0008; AA/summary.json:stall.E2_gap_two_heads, NL_*) [measured]. Use instead what only BCA reads: residuals, i.e. how far the critic's prediction missed the one-step outcome on a recorded step.

1. **Score held-out residuals, one-sided.** Score each recorded step by how much the critic overestimated it, using a critic that never trained on that step (cross-fitting). Residuals on training rows fade as the critic fits them: at seed 0, stalling world, the in-sample residual at added full-throttle rows goes -0.24, -0.06, +0.10 as their share rises 0.5%, 2%, 5%, while a critic that never saw them stays at -0.49 to -0.50; no-stall, +0.05 to +0.06, like ordinary rows (TB/summary.json:E1.0.sweep.*.{insample_trace,heldout_gap}) [measured].
2. **Drop the scale network and the classifier** (with its threshold table and refusal term): 0.2% and 86% of the signal (IMP/decompose_exact.json) [measured].
3. **Calibrate where the actor goes.** Split recorded steps by command size at the empty gap in recorded throttle. The rule reads commands only; it found the gap at 0.78-0.79 and picked exactly the survivors in 10 of 10 seed-worlds (CAL/e4_tailband.json) [measured]. Run BCA's own Bayesian weighted conformal threshold separately on the full-throttle group and the rest.
4. **A world-level certificate.** Caution switches on only if BCA's posterior says the critic is credibly more over-optimistic on full-throttle steps than on the rest, at a margin fixed in advance and stricter than seen here. In-sample so far: 5% bound +0.28 to +0.46 (stalling) and -0.015 to -0.029 (no-stall) (CAL/e11_credible_gap.json:[*].B.post_q05_gap) [measured]. Thin: the posterior median is positive in 4 of 5 no-stall seeds (same file, B.post_q50_gap) [measured]. Caution height: 0.60-1.27 value units; with scores shuffled, 0.03-0.07 (CAL/e4_tailband.json) [measured].
5. **Consumer: deployment check first, training penalty second.** When certified and the command is past the gap, clip to the recorded range. On saved actors, stalling world, strong current: 91.1 against 68.9 for random switching at its own rate (+22.2; mark M5 needs 10), +7.1 in recorded conditions, no cost in the no-stall world (gate off 5 of 5). Net across worlds (stalling gain minus no-stall loss): 106 (CAL/e12_gated_deploy.json) [measured, in-sample]. Rivals: v6's check about 27, clip-always 44.2, classifier 48.7, an 80% cap placed at the true knee 81.2 (CONS/summary.txt; results-2026-10-13/deploy_summary.json) [measured]. Forcing the gate on costs 61.7 no-stall points (CAL/e14_ungated.json) [measured]. As a training penalty, strong-current states whose best choice is still past 80%: 44.7% at strength k=1, 25.8% at k=2, against 38.5% for the matched hinge and 64.4% for v6 (CAL/e6_bestresponse_pooled.json:stall_T3a; e6b_matched_hinge.json) [measured proxy]. So mark M4 (within 5 points of the matched hinge) likely still fails at k=1 and may pass at k=2 [speculative].

**Which part is BCA's own.** Within one world the clip point is the recorded edge, the hinge's information: an oracle past-80% detector with the same clip scores 91.2 in a comparable run (AA/box_shares.json) [measured]. BCA's residuals supply the per-world yes/no and the height. But a plain Mann-Whitney test on the same residuals gives the same yes/no in 10 of 10 seed-worlds (stalling p at most 0.0003, no-stall 0.51-0.99; JB/gate_ablation.json) [measured]. So the honest claim is "BCA's residual score reads outcomes that range-based rivals cannot", not "the Bayesian aggregation adds value". The posterior's possible own share, the calibrated height, is untested against plain split conformal [inferred].

## 3. Where to test it

ThrusterChain can show this, but only as a claim across worlds; within one world BCA can at best match a range limit (PLAN.md:535). The fair change, written before any fresh seed: keep M4 and M5, and add a mark on net across worlds against world-blind rivals (cap ladder 0.75-0.95 with 0.80 labelled as set at the knee, clip-always, classifier, range hinge), with attribution arms (gate forced on, outcomes permuted, shuffled groups, plain rank test, plain split conformal). Survivor rows must be selected without the knee. "BCA identifies harm in general" needs a mixed world where missing commands help in some states and harm in others: new physics, 25-40 process-hours [inferred]. Later.

## 4. Staged plan

**Step 0, done** (read-only, about 2.2 CPU-hours): the results above, plus two shelved routes. A run-time classifier guard nets 45.1, the same as clip-always (CONS/summary.txt) [measured]; a range penalty scaled by the critic's slope is a range limit stated as a prior [inferred]. One more needs no approval (10 minutes): the certificate on the saved trace critics. I expect it to switch off in the stalling world at 2-5% shares, because in-sample residuals fade [inferred]. Two stray files written into the repo during this work were deleted; I confirmed both are gone.

**Step 1, the decisive test (needs approval).** Train 10 fixed-controller critics (seeds 0-4, both worlds) on the gapped log minus its survivors, exact gate protocol: about 70 CPU-minutes, 7 minutes on 10 processors. Then, read-only (10 minutes), recompute the certificate, height and deployment check from the survivors' held-out residuals. Go/no-go, fixed now: certificate on in 5 of 5 stalling seeds and off in 5 of 5 no-stall seeds; deployment net at least 90; with outcomes permuted, at most half that; plain rank test reported beside it.

**Step 2, pilot on seeds 0-4 (needs approval; exploratory, labelled post hoc).** Actor-only policy test with the new band at k=1 and k=2, gate forced on, shuffled groups, the re-matched hinge and existing controls, both worlds: 2-3 process-hours per seed, 10-15 in all [inferred].

**Step 3, a new labelled post-hoc pre-registration (v8)**, one primary, every setting frozen, confirmed on unseen seeds 10-14: deployment channel first (about 1 process-hour per seed plus 2 cross-fit critics; 5-6 in all [inferred]), then the policy test, then the full host.

**Seeds 5-9 and v7.** Let v7 run seeds 5-9 as pre-registered: it is the confirmatory test of BCA as built, and withdrawing it after seeds 0-4 failed M4 and M5 would look like dodging. Cost: about 50 process-hours of laptop time (PLAN.md:516). If you need the laptop, withdraw in writing before any seed 5-9 result exists. v7's open question 7 still needs your answer before its step 8 (stage2_v7/README.md).

## 5. What would make us stop

If, in step 1, held-out residuals do not switch the certificate on in all 5 stalling seeds and off in all 5 no-stall seeds, or the net falls below 90, or permuted outcomes keep more than half of it, BCA's own machinery cannot carry this in ThrusterChain. We then report BCA as no better than a range limit here and move the claim to a mixed world, rather than tune further. Milder: if in step 2 the posterior's height does no better than plain split conformal, drop "Bayesian aggregation" from the claim and keep "residual calibration".

## 6. Decisions needed from you

1. Approve step 1: 10 critic trainings, about 70 CPU-minutes, with the go/no-go above.
2. Accept the claim change: a declared cross-world mark and attribution arms, M4 and M5 unchanged.
3. Seeds 5-9: let v7 run (recommended), or withdraw it formally now.
4. If step 1 passes: approve the step 2 pilot (10-15 process-hours of training).
5. Then approve the v8 text and fresh seeds 10-14.
6. Shelve the critic-ensemble width, the classifier guard and the critic-scaled range penalty as negative results and comparators.
