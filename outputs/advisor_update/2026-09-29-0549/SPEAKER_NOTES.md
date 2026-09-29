# BCA advisor update

Snapshot: September 29, 2026, 1:49 a.m. EDT / 05:49 UTC.

## Slide 1: BCA experiment update

We have four additional successful cluster training closures since the previous deck. We still have seven independently accepted training pairs and one completed OOD seed. The research question is whether BCA helps with poorly represented actions. The first Hopper panel does not demonstrate that benefit. The slides address why the study matters, what it measures, how observations compare with expectations, and why each follow-up is justified. Training closures, scientific acceptance, simulator outcomes and synthetic mathematics tests remain separate. Snapshot: September 29, 2026, 1:49 a.m. EDT / 05:49 UTC. Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/advisor_update/2026-09-29-0549/cluster_snapshot.json and local_snapshot.json. Runtime status captured in OOD_STATUS_SNAPSHOT.md (05:39 UTC source update).

## Slide 2: Why this OOD study matters

Advisor questions: Why do this study, and what does it accomplish? Ordinary return curves combine action selection, visitation and later behavior. The OOD study restores the same state, imposes a fixed action and measures what happens under each frozen continuation. Warning quality and behavioral benefit answer different questions. If width predicts harmful actions but the policy does not improve, investigate how the host consumes the width. If width barely separates harm, investigate the score's relationship to behavioral loss. These are diagnostic leads, not identified causes. The intended benefit is a clearer mechanism test, regardless of whether BCA wins. Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/superpowers/plans/2026-09-28-ood-robustness-v2.md.

## Slide 3: Current comparison and scope

The current standard study is distinct from earlier native and full-width experiments. Both Bayesian and conformal components remain, and fitting importance factors are equal. Bayesian random masses remain active. The frozen matrix has 280 physical runs and 315 actor trajectories because the IQL paired branch shares Q/V with an additional actor. Five declared seeds remain required. Do not treat those actors as independent seed replicates.
Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/experiments/standard_bca/README.md
Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/experiments/standard_bca/RUNTIME.md

## Slide 4: The OOD action experiment

The simulator supplies evaluation outcomes after training. It does not train the actor or calibrator during these rollouts. We restore complete states, fix candidate actions and randomness before observing outcomes, and compare continuations from the same first action. Legal and distant are not synonyms for harmful. The v1 panel uses host-action harm reference. Prospective v2 uses a nearest-recorded common anchor and explicit support bands.
Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/superpowers/plans/2026-09-28-ood-robustness-v2.md

## Slide 5: Where BCA enters each host

The width is translated into a host-specific training adjustment, rather than being subtracted universally from every Q-value. TD3+BC and ReBRAC modify actor behavior-cloning strength. ReBRAC critic BC is unchanged. CQL attaches a transition-level multiplier to its conservative critic gap using a recorded-action width, which must not be called separate candidate-action uncertainty. IQL shrinks the capped actor advantage weight's excess above one, using shared nuisance Q/V and fixed gain 1.
Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/WEIGHTED_CONFORMAL_DESIGN.md

## Slide 6: Live progress and execution barriers

Snapshot: September 29, 2026, 1:49 a.m. EDT / 05:49 UTC. Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/advisor_update/2026-09-29-0549/cluster_snapshot.json and local_snapshot.json. Runtime status captured in OOD_STATUS_SNAPSHOT.md (05:39 UTC source update). Cluster controller PID 763309 and worker PID 893496 match their saved command lines, start ticks and process-group/session identities. The 68 completed runs comprise 34 ReBRAC and 34 IQL, with all completed worker exits zero and learner completed. Seed counts are 28, 28 and 12. Active run is rebrac-maze2d-host-s202609173 at 425,000 updates. Current worker/controller final exits are pending. The local controller still has actual exit 1 from September 27 after CQL final evaluation because saved episode field names differ from the result validator. Across lanes, 70/280 physical runs have successful closure receipts. Fourteen runs / seven pairs have independent scientific acceptance. Fifty-six cluster closures await full audits. There is no completed five-seed OOD cell and no v2 outcomes. The latest isolated callback/state-binding component passed 41 synthetic fixtures, but no actual production dispatcher or complete native import/state transition is implemented or accepted. Component tests are engineering progress only. The real checkpoint/simulator path remains the critical blocker. No completion ETA is supported.

## Slide 7: Seven accepted training comparisons

All rows use final twenty-episode normalized-score means. Curve difference is the arithmetic mean across 200 periodic evaluation banks, not the periodic score at 1M. The first seed suffix is 171, etc. Every row retains all original episodes and is based on one paired training seed. Fourteen physical runs form these seven pairs. ReBRAC Hopper and Walker each have 3/5 accepted seeds, while TD3 Hopper has 1/5. Final and learning-time comparisons differ. The large third Walker difference coexists with a highly variable host final bank, whose episode SD is 39.12 versus BCA 15.02. This does not establish a training cause or OOD benefit.
Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/advisor_update/2026-09-29-0346/accepted_training_pairs.csv

## Slide 8: Learning curves through 1M updates

Curves retain every saved periodic bank and show no smoothing. Embedded chart values round to eight decimal places for spreadsheet portability; exact source values remain in data.json. TD3 Hopper has one accepted training seed. The two ReBRAC panels show the arithmetic mean of the three accepted training seeds at each update. They are descriptive means with no training-seed interval, and the paired final contrasts on the preceding slide show important variation. Periodic bank means have ten episodes; final evaluation uses a separate twenty-episode bank. Final endpoints in the preceding slide therefore need not equal these final plotted points.
Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/advisor_update/2026-09-29-0346/data.json

## Slide 9: Cluster endpoints vary by dataset and seed

These are all seven datasets from the already saved two-seed cluster extraction at September 28 22:37 UTC. They are preliminary saved evaluation summaries, not a fresh full scientific acceptance. ReBRAC Hopper and Walker rows now have accepted audits; remaining rows including IQL still await full audits. IQL comparisons use paired actors sharing Q/V. Do not combine normalized deltas across tasks into a single inferential average. ReBRAC Maze changes sign dramatically between seeds, and the small IQL differences are mixed. This motivates retaining the full declared seed set.
Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/advisor_progress/2026-09-29/cluster_pair_results.csv

## Slide 10: First OOD result: weak harm ranking

This accepted TD3+BC Hopper result is standard BCA at seed 202609171. It contains 5,120 rollouts, 256 captured rows and 64 shared reset blocks, with 1.28 million outcome environment steps. Under BCA continuation, harm means a loss greater than 1 raw reward unit relative to the host first action at the same state. There are 276 harmful alternatives among 2,304 and 111/256 panels with both classes. The plotted metric is the equal-four-stratum mean of within-state AUROC. Constant scores give 0.5 and the precommitted random score realizes 0.513 in this panel. There is no significance or replicated benefit claim. Pooled width AUROC 0.431776 measures a different comparison and is deliberately not substituted.
Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/advisor_progress/2026-09-29/ADVISOR_BRIEF.md

## Slide 11: First OOD result: behavior after action stress

The first row isolates first-action choice under the same BCA continuation, on the accepted v1 panel. The other two rows are the separately labelled post-hoc support-distant bridge using 233 actions across 193 states, selected using saved support labels. They are not prospective v2 outcomes. A equals BCA return minus host return after the exact same imposed action. D compares each continuation's loss relative to a common familiar anchor. BCA is lower by about 5.046 raw return units after the distant action. D is about -0.019, almost zero, indicating no degradation advantage against that anchor. Distant actions actually improve the average relative to the anchor under both policies, so these quantities cannot be described as universal harmful attacks.
Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/advisor_progress/2026-09-29/ADVISOR_BRIEF.md

## Slide 12: Why each OOD follow-up change is needed

Advisor question: Why do minor changes make sense, and what should they accomplish? V2 was specified after inspecting v1 and before new v2 outcomes. It is not a retroactive preregistration. It holds coverage and training recipes fixed. The first tranche remains 20 pairs: TD3+BC and ReBRAC, Hopper and Walker, five seeds. CQL/IQL and other datasets require additional adapters. Pure selection and known-answer tests do not count as scientific completion. The single seed collision amendment was explicitly approved and changes only 404735174 to 404735181. All original streams/failures remain.
Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/superpowers/plans/2026-09-28-ood-robustness-v2.md
Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/OOD_V2_STREAM_AMENDMENT_20260928.md
Advisor explanation: these changes improve the experiment's ability to distinguish explanations. They do not assert that the next result will favor BCA. Support bands ask whether performance depends on action support. A common recorded-action anchor prevents unequal reference actions from changing the degradation comparison. Separate state strata distinguish action-only from joint state/action shift. Five training seeds measure training variation. The approved candidate-seed amendment changes 404735174 to 404735181 solely to remove overlap with an existing evaluation seed. Coverage, recipes, checkpoints and prior outcomes remain unchanged. Every scientific expectation for v2 remains untested because it has no outcomes.

## Slide 13: How to interpret good and bad outcomes

A useful prediction interval and a useful control policy answer different questions. Coverage can be correct while width cannot rank harmful actions, because Bellman-target residuals may be a poor proxy for finite-horizon harm. A favorable ranking alone does not prove a better policy. A lower within-state AUROC does not falsify a conformal coverage theorem, whose response, population and assumptions differ. A favorable performance result still needs all declared seeds and the protocol's intervals. V2 requires both adjusted lower bounds positive for stressed-return advantage and degradation advantage, under its explicit scope.
Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/superpowers/plans/2026-09-28-ood-robustness-v2.md
Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/WEIGHTED_CONFORMAL_DESIGN.md

## Slide 14: What was wrong with the earlier IW path

The standard no-IW experiment intentionally uses equal importance factors. Its identity does not change. The separate earlier IW implementations changed scale-fitting weights but refreshed radius with equal calibration masses and query mass 1. The intended method was weighted conformal prediction, so describing those variants as fully weighted conformal would overstate the implementation. Existing generic low-level routines accepting weights do not prove the actual host path supplies valid ratios. Current four hosts use one cached scalar radius. This implementation gap is independent evidence and does not prove it caused the unfavorable Hopper OOD behavior.
Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/WEIGHTED_CONFORMAL_DESIGN.md

## Slide 15: Components needed for the IW correction

The mathematical reference preserves both components. The conformal floor uses target/source covariate ratios at calibration examples and the query. Finite-sample marginal coverage needs the appropriate fixed score, sampling/independence, overlap and unchanged conditional response law. Density estimates add estimation error; policy density alone omits the behavior denominator. Deterministic TD3/ReBRAC policy targets can be singular relative to continuous behavior. Smoothing defines a new target and cannot silently justify the original. Reusing a calibration bank to influence later policies creates feedback; freezing afterward does not undo it. Fresh independent validation is needed. Infinity remains unbounded in coverage reporting, and host behavior must be separately declared.
Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/WEIGHTED_CONFORMAL_DESIGN.md
Primary source: https://www.stat.berkeley.edu/~ryantibs/papers/weightedcp.pdf, equations (5)-(7), Corollary 1.

## Slide 16: Weighted threshold: synthetic validation

This is a controlled scalar simulation, not an RL or OOD-policy result. In 2,000 independent trials, each bank has 199 calibration examples. Source has 20% hard examples and target has 80%, while conditional responses remain fixed: easy Uniform(0,1), hard Uniform(0,6). The predictor is zero, scale one, and exact target/source weights are 0.25 and 4. The conformal calculation applies weights to both calibration examples and query. Coverage counts are 1820,1221,1814,1879. Exact 95% binomial intervals: [89.66,92.22], [58.87,63.19], [89.34,91.94], [92.81,94.95] percent. Means of finite radii are 2.999,2.999,5.295,5.590. Extra Bayesian coverage comes with wider bands and does not establish greater efficiency or a 95%-confidence population-risk bound. Fifteen tests and independent reconstruction passed; original numerical failures remain preserved.
Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/weighted_conformal/2026-09-29-v2/READOUT.md

## Slide 17: Next execution steps and acceptance criteria

No timing promise is made because the native/runtime blocker is unresolved. The immediate OOD priority is a source-bound runtime integration followed by actual checkpoint and simulator gates, then the declared first tranche. A new synthetic fixture does not close this step. Local CQL recovery should preserve the exit-1 record and reuse saved final evaluations and checkpoints rather than repeat training. A separately documented schema-only saved-artifact validation is the next recovery proposal, not a completed recovery. Cluster training continues under its original controller; independent audits proceed in fixed order as resources allow. IW proceeds first through a frozen-score known-ratio real-data/simulator sampling contract, not by selecting weights from final OOD outcomes.
Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/STANDARD_BCA_STATUS.md
Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/OOD_ACTION_COLLECTION_STATUS.md
Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/advisor_update/2026-09-29-0346/cluster_snapshot.json
Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/WEIGHTED_CONFORMAL_DESIGN.md
Update at 05:49 UTC: cluster is at 68 successful closures and continues under its original controller. The latest callback/state binder is a tested component, but the integrated production dispatcher is not ready. Existing isolated tests do not authorize a completion claim. Finish and review the real execution path, then pass the actual checkpoint/state/action/reward checks before collecting precommitted v2 outcomes. The local CQL schema failure is a separate recovery track. This slide update has not started or restarted either track. Snapshot: September 29, 2026, 1:49 a.m. EDT / 05:49 UTC. Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/advisor_update/2026-09-29-0549/cluster_snapshot.json and local_snapshot.json. Runtime status captured in OOD_STATUS_SNAPSHOT.md (05:39 UTC source update).

## Slide 18: Expectations and what the evidence shows

Advisor questions: Did the results meet expectations? If not, why? Hypothesis 1 is that larger residual bands help distinguish harmful actions. The first TD3+BC Hopper seed has equal-stratum mean within-state harm AUROC 0.529, close to 0.5, so this panel gives weak support. Hypothesis 2 is that BCA-trained continuation is more robust after action stress. The post-hoc distant-action subset has BCA-minus-host return −5.045585 and degradation advantage −0.018709 against the common anchor. It does not demonstrate the expected benefit, but it is one seed and post hoc. The negative evidence stays. Why could this happen? Residuals against bootstrapped targets may not align with simulator harm, host-specific attachment may use widths ineffectively, or sampling/population mismatch may matter. These remain hypotheses. Neither weak AUROC nor low return identifies the cause or disproves a coverage theorem. The weighted-conformal toy test meets its deliberately limited expectation: 90.70% target coverage under known ratios versus nominal 90%. That is not RL coverage or a policy result. The prospective v2 comparison has no outcomes, so its expectation is pending. Sources: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/advisor_progress/2026-09-29/ADVISOR_BRIEF.md; C:/Users/David Bayha/Documents/GitHub/BCA/outputs/weighted_conformal/2026-09-29-v2/READOUT.md; C:/Users/David Bayha/Documents/GitHub/BCA/docs/WEIGHTED_CONFORMAL_DESIGN.md.

## Slide 19: Appendix: training, calibration and evaluation

These row counts are for the accepted first TD3+BC Hopper pair only, not universal across tasks. Both methods use the same 998895 training rows and 1103 held-out rows. BCA uses a 1024-row training reference. Normalization fits the training complement. The calibrator fits detached host-target prediction errors. Posterior refreshes use reserved offline transitions, with 198 refreshes through 1M. Evaluation observes simulator rewards with frozen parameters; the OOD rollout may log widths but does not refit them. The reuse of a held-out bank to influence later training prevents treating a later checkpoint as automatically independent of that bank.
Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/standard_bca/td3_bc/hopper/bca_noiw/s202609171/README.md

## Slide 20: Appendix: weighted-threshold arithmetic

Illustration only at alpha 0.4 to keep four scores readable. Production alpha remains 0.1. Scores are normalized absolute errors against a frozen host target, not known true-Q errors. Weighted conformal's finite-sample target/source construction needs the query's mass at infinity. Using a common normalization cancels; normalizing query separately does not. If observed cumulative mass cannot reach the cutoff, return infinity and retain it. Width is scale times the maximum of the conformal and Bayesian radii.
Repository evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/WEIGHTED_CONFORMAL_DESIGN.md
Primary source: https://www.stat.berkeley.edu/~ryantibs/papers/weightedcp.pdf
