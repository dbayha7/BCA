# BCA advisor update

15-slide presentation followed by 11 supporting slides.

## 1. Can BCA identify harmful unfamiliar actions?

Why this study: episode return alone cannot tell whether BCA’s uncertainty mechanism works. A policy can improve for unrelated reasons, and a useful residual interval need not identify harmful choices. Standard BCA is the baseline with equal fitting weights and both Bayesian/conformal components. Current evidence is limited: a completed original TD3+BC Hopper pilot supplies action outcomes; reviewed TD3+BC/ReBRAC Walker runs supply training diagnostics; CQL diagnostics are preliminary; the controlled weighted-CP experiment is synthetic. No new outcome or live progress check was performed to prepare this update.

Source records: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/weighted_conformal/2026-09-29-v2/results.json; docs/WEIGHTED_CONFORMAL_DESIGN.md.
This update reuses saved evidence; it adds no training, simulation or synthetic trials.

## 2. The OOD test compares actions at the same state

What was done: the completed original Hopper pilot used ten candidates per captured row, including the host reference, with both frozen host and BCA continuations. The primary harm-ranking readout fixes BCA as the later policy. There are 2,304 alternatives under that continuation and 64 shared reset blocks. States and alternatives are dependent, so they are not independent training seeds. The common-state comparison isolates a local first-action contrast; it does not identify a whole-policy training cause. Separate revised OOD runs must not be marked complete using this pilot. The saved September 29 snapshot had 5 active and 0 complete of 20 first-tranche revised pairs; that is a dated snapshot, not a current live count. Training/calibration remain offline; simulator interaction is diagnostic evaluation.

Source records: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/weighted_conformal/2026-09-29-v2/results.json; docs/WEIGHTED_CONFORMAL_DESIGN.md.
This update reuses saved evidence; it adds no training, simulation or synthetic trials.

## 3. BCA’s harm warnings are weak in this pilot

Why important: a warning is useful only if larger values tend to identify worse alternatives at the same state. AUROC measures pairwise harmful-versus-nonharmful ordering; 0.5 is chance. Width is 0.529327, distance 0.565805 and a fixed random score 0.513251. Average precision emphasizes the highest warnings: width 0.510392, distance 0.542025, random 0.432432 and constant-score baseline 0.274558. Thus there is some enrichment at the top, but overall separation is weak; the metrics answer different questions. The AP baseline is computed in the same eligible-row/equal-group aggregation, so it is not the pooled 11.98% harmful rate. Only 111 of 256 rows contain both classes; the other 145 have undefined AUROC and are not assigned zero. These descriptive results do not establish a multi-seed benefit or statistical superiority. Next: reproduce the paired ranking across declared seeds and measure coverage separately before attributing weak ranking to calibration shift.

Source records: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/weighted_conformal/2026-09-29-v2/results.json; docs/WEIGHTED_CONFORMAL_DESIGN.md.
This update reuses saved evidence; it adds no training, simulation or synthetic trials.

## 4. Unfamiliarity raises risk, but does not define harm

Why important: equating OOD with harm would make an uncertainty method look good merely for flagging novelty. Arithmetic: near recorded actions, 231/2071 = 11.153%; farther actions, 45/233 = 19.313%. There are more near actions, so 231/276 harmful alternatives are near. Also 188/233 distant alternatives are not harmful under the specified continuation. Distance is minimum RMS action distance to recorded actions among 32 nearby reference states, using threshold 0.21134613219046514; it is not a true density or exact OOD label. Meaning: novelty identifies a harder group, but a blanket penalty on all distant actions could suppress useful moves and miss familiar bad moves. Next: report coverage and harm ranking separately within support-distance groups while retaining paired states.

Source records: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/weighted_conformal/2026-09-29-v2/results.json; docs/WEIGHTED_CONFORMAL_DESIGN.md.
This update reuses saved evidence; it adds no training, simulation or synthetic trials.

## 5. BCA has not improved tested-action regret

Why important: even a statistically well-calibrated interval is not useful for control unless it helps decisions. Candidate-set regret compares each chosen first action with the best of ten tested alternatives from the same state, holding the later policy fixed. Under BCA continuation, host-first mean regret is 6.333291153 and BCA-first 6.452881609; BCA minus host is +0.119590456, so there is no descriptive mean reduction. Under host continuation, corresponding values are 8.242196683 and 8.489875844, a +0.247679161 difference. This is not regret against the unknown optimal policy. One seed and correlated states cannot establish a general worsening. Next: replicate this paired comparison; if coverage improves but regret does not, investigate how uncertainty is consumed rather than declaring weighted CP successful for control.

Source records: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/weighted_conformal/2026-09-29-v2/results.json; docs/WEIGHTED_CONFORMAL_DESIGN.md.
This update reuses saved evidence; it adds no training, simulation or synthetic trials.

## 6. BCA is fitting and changing the training penalty

Why important: weak behavioral results could come from an inactive method or from an active method whose signal is not useful. Independent checkpoint counters confirm 1,000,000 accepted scale fits for both reviewed Walker runs. Their last 1,000-update blocks have actor-BC multipliers 1.331356037 for TD3+BC and 1.377447963 for ReBRAC. Arithmetic: (1.331356037−1)×100 = 33.1356% more BC coefficient and (1.377447963−1)×100 = 37.7448%. This is increased loss strength, not the probability of following the old policy. Final scale objectives are 0.0109333582 and 0.0100045581. The graphs show full block means, with skipped actor placeholder zeros excluded. Meaning: the mechanism operates in these Walker runs; accepted optimization steps do not prove coverage or harm detection. Next: evaluate fresh coverage and decision metrics, not only fitting success. Do not use these Walker diagnostics as a causal explanation of the separate Hopper pilot.

Source records: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/weighted_conformal/2026-09-29-v2/results.json; docs/WEIGHTED_CONFORMAL_DESIGN.md.
This update reuses saved evidence; it adds no training, simulation or synthetic trials.

## 7. The Bayesian radius controls these bands

Why important: both BCA components remain in the implementation, but only the larger radius determines the band at a given refresh. The conformal component is smaller at 198/198 logged refreshes in each reviewed Walker run. A change to the conformal floor can therefore be numerically masked by the Bayesian radius. Taking their maximum cannot make the band smaller than the conformal floor; this does not validate a Bayesian population-risk theorem. Meaning: we must report each radius, the effective width, and the actual training multiplier to know whether a correction reaches the host. A positive global radius rescales widths without changing within-state action ordering. Next: keep both components, check which one controls the final band, and distinguish coverage improvement from changes in harm ranking.

Source records: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/weighted_conformal/2026-09-29-v2/results.json; docs/WEIGHTED_CONFORMAL_DESIGN.md.
This update reuses saved evidence; it adds no training, simulation or synthetic trials.

## 8. The next question is calibration under shift

Why this follows the current study: standard BCA provides an equal-fitting-weight comparison, while the code audit identifies a missing weighted-conformal threshold in the earlier fitting-IW variants. These are two different observations. Weak harm ranking alone cannot prove covariate shift or undercoverage. The clean next experiment freezes the model, measures fresh response coverage under a declared shift, and then tests the behavioral consequences separately. The current response is a host-specific one-step Bellman target, not true Q or a safety label. Keep nominal coverage at 90%; it does not specify a 90% probability of copying the old policy.

Source records: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/weighted_conformal/2026-09-29-v2/results.json; docs/WEIGHTED_CONFORMAL_DESIGN.md.
This update reuses saved evidence; it adds no training, simulation or synthetic trials.

## 9. Our earlier IW path did not complete weighted CP

Current versus missing: all four host paths refresh through fit_posterior with one group; partitioned_posterior supplies indicator masses and query mass one. With one group those masses are equal even when fitting IW is on. Lower-level unequal-mass support exists, but the hosts do not supply the intended source/target ratios. One cached scalar is generally insufficient for query-dependent weighted CP. CQL policy density lacks the behavior denominator; TD3/ReBRAC affinity and IQL advantage weights are not automatically density ratios. The calibration bank influences later training, so freezing the final model does not restore independent split-calibration validity. The required correction changes the weighted quantile and query mass, and separately resolves sampling and ratio assumptions. This is not a claim that the intended equal-weight standard-BCA treatment contains a failed IW intervention. Reference: Tibshirani et al. (2019), Conformal Prediction Under Covariate Shift, https://www.stat.berkeley.edu/~ryantibs/papers/weightedcp.pdf ; local docs/WEIGHTED_CONFORMAL_DESIGN.md.

Source records: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/weighted_conformal/2026-09-29-v2/results.json; docs/WEIGHTED_CONFORMAL_DESIGN.md.
This update reuses saved evidence; it adds no training, simulation or synthetic trials.

## 10. A controlled test isolates the effect of the shift

Purpose: test whether target/source weighting repairs an intentionally controlled calibration-population mismatch. Prediction center is zero and scale one, so absolute prediction error equals Y. Easy response Y is uniform on [0,1]; hard response Y is uniform on [0,6]. Source type probabilities are (0.8,0.2), target (0.2,0.8), and conditional response distributions are unchanged. Known ratios are 0.25 and 4. Each of 2,000 independent trials samples a fresh bank of 199 calibration examples and a fresh query. All compared methods reuse the fixed examples and score construction. The reference also passed 15 boundary/arithmetic tests, and an independent reader checked all 8,000 radii and coverage outcomes. The accepted rational-ratio correction and its earlier numerical attempt remain archived; this deck uses the accepted v2 results without rerunning trials. This experiment establishes behavior in its stated mathematical setting only.

Source records: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/weighted_conformal/2026-09-29-v2/results.json; docs/WEIGHTED_CONFORMAL_DESIGN.md.
This update reuses saved evidence; it adds no training, simulation or synthetic trials.

## 11. Weighting restores coverage in the synthetic test

What the plot shows: ordinary CP covers 1820/2000 = 91.00% without shift, close to the 90% target. After changing only the easy/hard mixture, it covers 1221/2000 = 61.05%. Correct weighted CP covers 1814/2000 = 90.70%, an increase of 593 covered queries or 29.65 percentage points over shifted ordinary CP. Retaining the Bayesian maximum covers 1879/2000 = 93.95%. Respective 95% binomial intervals are [89.660%,92.218%], [58.873%,63.195%], [89.342%,91.937%], and [92.814%,94.955%]. Why important: it demonstrates the missing threshold correction can repair coverage in a clean shift where the assumptions and ratios are known. Meaning: the reference implementation behaves as intended in this controlled setting. It does not establish that real policy queries have the same assumptions or that these wider bands improve RL decisions. Next: a frozen-model real-data test with a declared source/target sampling design.

Source records: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/weighted_conformal/2026-09-29-v2/results.json; docs/WEIGHTED_CONFORMAL_DESIGN.md.
This update reuses saved evidence; it adds no training, simulation or synthetic trials.

## 12. Coverage recovery requires wider error bands

Why important: coverage alone rewards arbitrarily wide intervals. We also need sharpness and the effect on decisions. Ordinary CP mean half-width is 2.998574625; weighted CP is 5.294714196; weighted plus Bayesian maximum is 5.589847011. The latter is about 5.57% wider than weighted CP and covers 65 additional queries, raising coverage by 3.25 percentage points. No interval was infinite in these trials, but other supported inputs can require infinity. The wider interval is appropriate here because hard cases are more frequent under the target mixture. Higher-than-target coverage is not automatically better; it may lead to unnecessarily strong penalties. Next: report coverage, width, infinite-band frequency, and actual consumed training dose together; then evaluate regret and harm ranking separately.

Source records: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/weighted_conformal/2026-09-29-v2/results.json; docs/WEIGHTED_CONFORMAL_DESIGN.md.
This update reuses saved evidence; it adds no training, simulation or synthetic trials.

## 13. The arithmetic explains the coverage change

This arithmetic explains the experiment rather than replacing its measured results. For R between 1 and 6, all uniform [0,1] errors are covered and a fraction R/6 of uniform [0,6] errors are covered. Source mixture coverage at R=3 is 0.8+0.2×0.5=0.9. Target mixture coverage at R=3 is 0.2+0.8×0.5=0.6. Solving target coverage 0.2+0.8R/6=0.9 yields R=6×0.7/0.8=5.25. The empirical radius changes trial by trial; 5.2947 is its mean, not a fixed radius used in all 2,000 tests. At center zero and scale one, a band of half-width 3 is [−3,3], while half-width 5.25 gives [−5.25,5.25]; a target value 4 is missed by the first and contained by the second. Weighted CP has not discovered true Q or corrected the point prediction.

Source records: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/weighted_conformal/2026-09-29-v2/results.json; docs/WEIGHTED_CONFORMAL_DESIGN.md.
This update reuses saved evidence; it adds no training, simulation or synthetic trials.

## 14. Next, validate coverage before changing learning

What must happen to execute: choose the target population before selecting ratios or seeing outcomes. A controlled common-state action-shift design with known sampling densities is an appropriate first real-data mechanism test, but it does not by itself establish offline deployment coverage. Target support must be within source support, and Y|X must stay fixed under the chosen frozen Bellman-target construction. Query-dependent radii require a sorted weighted bank/cache, versioned checkpoints, model/ratio identities, and correct normalization and RNG handling. Infinite intervals remain infinite in coverage reports; a policy fallback is a separate declared behavior. The first training comparison should change the conformal threshold while holding scale fitting and host consumption fixed. If the Bayesian observed-score weighting is also changed, identify that arm separately so the causal contrast remains interpretable. Keep alpha=0.1; do not tune coverage for return. A later fitting-IW factorial can then isolate additional effects.

Source records: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/weighted_conformal/2026-09-29-v2/results.json; docs/WEIGHTED_CONFORMAL_DESIGN.md.
This update reuses saved evidence; it adds no training, simulation or synthetic trials.

## 15. What the next results will tell us

The advisor-facing conclusion is conditional. The original Hopper pilot has limited harm-ranking evidence and no mean tested-action regret improvement. Reviewed Walker diagnostics show BCA is active, not that its uncertainty is correct. The synthetic known-ratio experiment validates an important threshold mechanism and motivates a direct real-data coverage test. It does not prove that IW fixes RL. If coverage improves while decision metrics do not, possible limitations include the distinction between one-step Bellman response uncertainty and long-horizon harm, action ordering in the scale model, a Bayesian radius masking the conformal correction, or the host’s consumption rule. Those require separated tests; changing all components at once would prevent attribution. If both calibration and decisions improve across seeds, that supports a benefit in the specified populations, not broad OOD safety.

Source records: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/weighted_conformal/2026-09-29-v2/results.json; docs/WEIGHTED_CONFORMAL_DESIGN.md.
This update reuses saved evidence; it adds no training, simulation or synthetic trials.

## 16. Warning quality changes with the situation

APPENDIX: supporting diagnostics and implementation details.

The four primary strata retain both policy collectors at reset and after100 collector steps. Width AUROC and support AUROC differ by stratum. Captured reset states under the collectors can coincide; they are not independent samples. The largest width ranking here is at host-collected step100 states, while the host-reset and BCA-step100 strata are below0.5. Do not select the favorable stratum as the overall result. Across all2560 fixed state-action slots the saved width min/median/max are1.352396965/1.746680200/2.478725672. No panel has all nine alternative widths tied, so the near-chance primary summary is not explained by an all-action constant warning. This does not identify the residual model's causal failure. One positive global radius scales widths without changing within-state ordering.
Saved closed exports; no live simulator archives or checkpoints queried.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/summary.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/panels.json

## 17. TD3+BC: inspect Q and Bellman loss

APPENDIX: supporting diagnostics and implementation details.

TD3+BC Walker, seed202609171, audited1000-update block means. Q1 at proposed actor actions. TD3 Q is evaluated after critic update before actor update, on active actor rows; ReBRAC Qmin is pre-critic and retained across all host rows. TD3 and ReBRAC Q summaries are not the same estimand and must not be compared as a shared optimism curve. Critic loss is the host's twin-loss training objective, not measured true-Q error or OOD harm. In these last blocks BCA Q is lower but critic loss is higher. That pattern motivates investigation but does not prove corrected overestimation or excessive pessimism. Actor loss and BC accounting are also available in the saved blocks: TD3 bc_loss includes consumed dose, while ReBRAC bc_mse_policy is an unweighted coordinate sum.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/local_exports.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/cluster_snapshot.json.gz

## 18. TD3+BC: lower Q does not establish better Q

On Walker, BCA lowers the final-block mean Q estimate from 207.10 to 191.92. This is consistent with more conservative values, but their accuracy is still unknown.

The critic fitting loss rises from 55.42 to 61.31. That loss compares predictions with a moving Bellman target; it does not measure error against known true returns.

We need frozen-target coverage and paired action outcomes to decide whether the change is helpful. A lower Q estimate alone cannot explain improved or worse decisions.

TD3 Walker final block Q1 at proposed actor actions changes from host 207.100791 to BCA 191.923997. The critic fitting objective changes from 55.421106 to 61.308944. Both observations are worth showing because return alone cannot reveal whether the critic values or learning signal changed. However, the policies and targets also change during training, so these logs do not compare two predictors against the same truth. Lower Q might reflect reduced overestimation, excessive pessimism or simply different actions/states evaluated by the logging path. Higher fitting loss is not a measurement of OOD harm or a proof of worse true values. Next: compare frozen predictors at common inputs against an explicitly defined fixed response, and use independent simulator outcomes for action quality. The weighted-CP response is the declared frozen one-step Bellman target; its coverage is not true long-horizon Q coverage. The first radius experiment should keep this response definition unchanged to isolate the distribution correction.

Sources: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/standard_bca/*/walker2d/*/s202609171/verified-v1/; analysis/current_results_20260929/inputs/cluster_snapshot.json.gz
Evidence scope: dated saved exports. No new simulator outcome or training run.

Source records: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/weighted_conformal/2026-09-29-v2/results.json; docs/WEIGHTED_CONFORMAL_DESIGN.md.
This update reuses saved evidence; it adds no training, simulation or synthetic trials.

## 19. ReBRAC: inspect Q and Bellman loss

APPENDIX: supporting diagnostics and implementation details.

ReBRAC Walker, seed202609171, audited1000-update block means. Minimum twin Q at recorded actions. TD3 Q is evaluated after critic update before actor update, on active actor rows; ReBRAC Qmin is pre-critic and retained across all host rows. TD3 and ReBRAC Q summaries are not the same estimand and must not be compared as a shared optimism curve. Critic loss is the host's twin-loss training objective, not measured true-Q error or OOD harm. In these last blocks BCA Q is lower but critic loss is higher. That pattern motivates investigation but does not prove corrected overestimation or excessive pessimism. Actor loss and BC accounting are also available in the saved blocks: TD3 bc_loss includes consumed dose, while ReBRAC bc_mse_policy is an unweighted coordinate sum.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/local_exports.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/cluster_snapshot.json.gz

## 20. ReBRAC: the value tradeoff is also unresolved

On Walker, BCA lowers the final-block mean minimum Q from 232.63 to 216.31, while critic loss rises from 20.30 to 24.75.

BCA increases the actor’s behavior-cloning strength, while the critic’s behavior-cloning coefficient stays fixed. The change can influence later targets and values indirectly.

To assess the impact, compare fresh coverage and action outcomes at matched states. These logged losses cannot show that BCA corrected overestimation or detected OOD harm.

ReBRAC Walker final block twin-minimum Q at recorded dataset actions falls from 232.634650 to 216.314605, while critic loss rises from 20.296611 to 24.752905. These are the same qualitative directions as TD3 but use a different query point and logging order; cross-host Q magnitudes cannot rank relative optimism. ReBRAC's actor BC multiplier changes while its critic BC remains part of the original host. A weighted-CP experiment must preserve the host's regularized target definition and critic BC to avoid a confounded comparison. Fixed-target residual tests can assess fitting at a common response. Separate OOD interventions assess whether candidate actions actually cause harm. Neither a training-loss increase nor lower Q by itself establishes why BCA performance is weak. The proposed improvement is better isolation and measurement, followed by a specific weighted-threshold intervention if the population contract is satisfied.

Sources: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/standard_bca/*/walker2d/*/s202609171/verified-v1/; analysis/current_results_20260929/inputs/cluster_snapshot.json.gz
Evidence scope: dated saved exports. No new simulator outcome or training run.

Source records: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/weighted_conformal/2026-09-29-v2/results.json; docs/WEIGHTED_CONFORMAL_DESIGN.md.
This update reuses saved evidence; it adds no training, simulation or synthetic trials.

## 21. CQL: inspect its critic and consumed penalty

APPENDIX: supporting diagnostics and implementation details.

CQL Walker seed202609171, preliminary frozen cluster export pending full independent audit.1000 retained last-row observations at1000-update intervals; not full scan means or a million per-update rows. Q1 is on recorded actions. Bellman objective and conservative objective are scalar losses, not gradient norms. BCA width at the recorded transition changes the multiplier on the conservative gap, not a separate warning for each sampled negative action. The fourth chart is that logged multiplier; a nonzero or large multiplier is not coverage or detection. Both Hopper and Walker have198 successful refresh records, but numerical historical CQL radii are unavailable in this export. Sparse accepted scale snapshots do not establish every update accepted; checkpoint review is pending. No counters or radius traces are fabricated.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/local_exports.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/cluster_snapshot.json.gz

## 22. CQL’s early diagnostics need a complete review

In the final retained Walker snapshot, BCA lowers Q1 from 177.15 to 163.12 and increases Bellman loss from 35.60 to 37.20.

The conservative objective changes from 13.68 to 13.21 with a BCA multiplier of 1.345. One recorded-action width scales the transition’s conservative gap.

These are sparse snapshots, and independent checkpoint review remains pending. We must verify the full run and paired outcomes before attributing a benefit or failure to BCA.

CQL Walker final sparse snapshot values are host/BCA Q1 177.151443/163.115646, Q1 fitting loss 35.597420/37.195969, and conservative objective 13.682011/13.208880. The BCA critic-dose mean is 1.345180 and scale objective 0.011620161, with this sampled row accepting a fit. These are one of 1,000 sparse retained last rows, not means over all updates. They cannot independently prove one million accepted scale fits. There are 198 successful refresh records, but historical numerical radii are absent from this extracted record, and the full independent checkpoint review remains pending. Why these metrics matter: they show that the intervention reaches the critic, but neither scalar loss magnitude nor lower Q identifies a gradient mechanism or true-value improvement. Current CQL attaches a detached recorded-transition multiplier to the conservative gap. It does not assign a distinct calibrated interval to every sampled negative action in that gap. The first weighted-CP implementation must make the actual query point explicit while preserving the original dual-gap definition. Candidate-specific penalties would be a separate host-method change. Improvement begins with schema/counter review and frozen-input diagnostic comparisons, not silently relabeling the existing treatment.

Sources: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/standard_bca/*/walker2d/*/s202609171/verified-v1/; analysis/current_results_20260929/inputs/cluster_snapshot.json.gz
Evidence scope: dated saved exports. No new simulator outcome or training run.

Source records: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/weighted_conformal/2026-09-29-v2/results.json; docs/WEIGHTED_CONFORMAL_DESIGN.md.
This update reuses saved evidence; it adds no training, simulation or synthetic trials.

## 23. Fresh coverage is the missing real-data measurement

Coverage asks how often a new stated target lies inside its prediction band. The current BCA target is a one-step Bellman response, not known true long-run Q.

We have not verified fresh OOD response coverage for these real-RL results. Successful scale fitting, wide bands, and completed radius refreshes do not supply that measurement.

Freeze the model, use a fresh calibration/test design, and report coverage by the declared population alongside width and unsupported queries. Keep trajectory dependence explicit.

Do not replace the missing calibration-validity test with fitted scale loss, successful updates, radius plots or behavioral returns. The required observable is indicator{|Y-m(X)| <= width(X)} on a fresh appropriately held-out population after the scoring function is frozen, with an explicitly declared one-step response/target, sampling design and assumptions. The target is the specified bootstrapped Bellman response, not true infinite-horizon Q. Nominal coverage90% is a configured target, not a measured90% outcome or90% imitation of the old policy. Nominal Bayesian credibility is also not population coverage. Report empirical coverage, interval width/sharpness, sampling uncertainty and predeclared support-stratum diagnostics separately. Conditional group checks are empirical; the marginal theorem is not automatically conditional coverage. Current behavioral archives do not establish fresh residual coverage, and ReBRAC's next-action target contract remains a separate compatibility constraint. Exact weighted-conformal correction needs justified density ratios/support and query mass; fitting weights alone are not sufficient.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/WEIGHTED_CONFORMAL_DESIGN.md
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/superpowers/plans/2026-09-28-ood-robustness-v2.md
Saved closed exports; no live simulator archives or checkpoints queried.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/summary.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/panels.json

Source records: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/weighted_conformal/2026-09-29-v2/results.json; docs/WEIGHTED_CONFORMAL_DESIGN.md.
This update reuses saved evidence; it adds no training, simulation or synthetic trials.

## 24. Weighted CP changes how errors set the band

APPENDIX: supporting diagnostics and implementation details.

The additive reference uses frozen X=(s,a), center m(X), positive scale sigma(X), and a specified frozen one-step Bellman response Y using observed reward/next state/terminal flag and fixed target-randomness law. Score S_i=|Y_i-m(X_i)|/sigma(X_i). With w(x)=dQ_X/dP_X(x), use calibration masses w(X_i) and query mass w(x), normalized by their combined total. The conformal radius is the 1-alpha quantile of the observed-score masses plus the query's mass at positive infinity. Common positive rescaling of all weights cancels; separately normalizing query and bank weights is wrong. As a toy arithmetic example only, scores [1,2,3,4], masses [1,1,1,7], query mass 1 and alpha .4 produce cutoff .6*11=6.6, so radius 4. Equal masses instead produce radius 3. Query mass 100 produces cutoff 66 above observed mass 10, so radius infinity. This toy alpha does not change the study alpha .1. A full radius 4 with scale 2 and center 10 yields half-width 8 and interval [2,18]. Source for the conformal correction: Tibshirani et al. (2019), https://www.stat.berkeley.edu/~ryantibs/papers/weightedcp.pdf, equations (5)-(7), Corollary 1 and split-conformal discussion. Bayesian masses and credibility calculation remain a separately labeled construction; the maximum cannot narrow a conformal band, but it supplies no automatic Bayesian population-risk guarantee.

Sources: docs/WEIGHTED_CONFORMAL_DESIGN.md; calibration/reference.py; calibration/posterior.py; experiments/weighted_conformal/reference.py
Evidence scope: dated saved exports. No new simulator outcome or training run.

## 25. Weighted CP needs a defensible target population

The target must have support in the source, and the conditional response law must stay the same. If state visitation changes, an action-only ratio is insufficient.

Deterministic actor outputs can be point masses relative to continuous source actions. A stochastic neighborhood would be a different target and must be declared explicitly.

Use fresh data after freezing the score, validate estimated ratios separately, and retain infinite intervals. Weight clipping and repeated bank reuse require additional justification.

Weighted CP requires more than plugging arbitrary weights into a quantile. We must name the source and target distributions, ensure target support is contained in source support, retain the same conditional response distribution, and justify the calibration/query sampling contract. A policy-action ratio only handles action shift under a justified common state population; changed state visitation requires its state component as well. Deterministic TD3/ReBRAC query actions can be singular relative to continuous source actions. Smoothing an estimated behavior density does not solve that measure mismatch. A declared stochastic neighborhood is a different target and cannot be quietly called deterministic-policy coverage. For an estimated-ratio stage, fit source-versus-target classification on separate covariates, correct class-prior odds, freeze the estimator, and assess overlap/weight concentration. Do not choose ratios using harm labels. Clipping or ESS tempering cannot silently enter the conformal floor as though the original ratio theorem survived unchanged. Freezing after adaptive bank reuse does not restore independence, and episode separation alone does not make within-trajectory transitions IID. Infinite intervals must remain infinite in coverage reports; a policy fallback when an interval is unbounded is a separate engineering and behavioral decision. These are gates for the next experiment, not excuses to relabel the old data.

Sources: docs/WEIGHTED_CONFORMAL_DESIGN.md; calibration/reference.py; calibration/posterior.py; experiments/weighted_conformal/reference.py
Evidence scope: dated saved exports. No new simulator outcome or training run.

Source records: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/weighted_conformal/2026-09-29-v2/results.json; docs/WEIGHTED_CONFORMAL_DESIGN.md.
This update reuses saved evidence; it adds no training, simulation or synthetic trials.

## 26. Each host keeps its original BCA attachment initially

APPENDIX: supporting diagnostics and implementation details.

Implementation is not a blanket change to every action penalty. For TD3+BC and ReBRAC, BCA acts through a detached actor behavior-cloning multiplier; ReBRAC critic BC is unchanged. For CQL, it multiplies each recorded transition's conservative gap while the dual update retains its original unweighted gap. That is not the same as an individual bound on every negative action. For IQL, BCA shrinks capped AWR-weight excess above one, with shared nuisance Q/V and fixed gain one. Advantage-based actor weights are not automatically probability ratios for calibration. The query-aware implementation must store a sorted weighted score bank, the ratio estimator identity, frozen score/target snapshots, units and stochastic target keys. It must evaluate the query ratio at the exact same input for which the host consumes the width. Separate tests should cover normalization, state/action alignment, checkpoint restore equivalence, array shapes, and infinity fallback. Retain both radius components. Do not use the intended weighted-CP correction as an opportunity to alter host losses or select a better recipe by score.

Sources: docs/WEIGHTED_CONFORMAL_DESIGN.md; calibration/reference.py; calibration/posterior.py; experiments/weighted_conformal/reference.py
Evidence scope: dated saved exports. No new simulator outcome or training run.
