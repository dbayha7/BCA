# Result interpretation and weighted CP speaker notes

## Slide 1: Research objective and the next hypothesis

The research objective is to determine whether BCA supplies calibrated and decision-useful uncertainty for actions that differ from those represented in its calibration data. These are separate questions: operational fitting, response coverage, harm ranking, and decisions. The current standard-BCA study supplies an equal-fitting-weight baseline. The old IW recipes changed the scale-fitting objective but did not fully implement the intended target/source weighted conformal threshold. This motivates an explicit correction and a controlled comparison. Weak pilot harm ranking does not itself prove covariate shift caused a coverage failure. The planned weighted-CP stage must measure that mechanism directly, then ask whether improved coverage helps decisions. All empirical examples in this deck retain their host, dataset and training-seed identities.

Sources: outputs/advisor_metrics/2026-09-29/METRICS_OVERVIEW.md; docs/WEIGHTED_CONFORMAL_DESIGN.md
Evidence scope: dated saved exports. No new simulator outcome or training run.


## Slide 2: The study needs more than a return curve

Research objective: Determine whether BCA produces useful prediction-error bands and whether their use by the host improves action decisions. Whole-policy return is only one end result. This deck adds the missing evidence chain: empirical coverage, width/sharpness, warning discrimination, decision regret, scale fitting, component radii, consumed penalty, and critic behavior. Fresh residual coverage is not established. The prior slide deck overemphasized return and omitted already-available calibration diagnostics and secondary pilot metrics. Pilot statistics are one TD3+BC Hopper training seed; training diagnostics below use seed202609171 on Walker. Those are different cells and cannot be merged into a same-run causal explanation. Revised OOD snapshot still has zero completed pairs.
Saved closed exports; no live simulator archives or checkpoints queried.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/summary.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/panels.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/local_exports.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/cluster_snapshot.json.gz


## Slide 3: Warnings: some enrichment, weak separation

Original TD3+BC Hopper pilot, seed202609171, frozen BCA continuation. Harm means losing more than1 raw reward unit relative to host first action in the same state over250 total transitions. Each panel has nine alternatives. AUROC and average precision are computed within eligible panels, then averaged within the four collector/capture strata; the displayed mean gives equal weight to each stratum. There are111/256 eligible panels. All145 single-class panels are N/A for these rankings. AUROC asks how often a harmful action scores above a nonharmful action;0.5 is chance, ties half. AP summarizes precision as high-score actions are selected; higher is better, with tied-score groups handled together. Constant-score AP=.27456 is the matched eligible-panel baseline, not the11.98% pooled harm frequency. AP shows enrichment over constant/random, so it is inaccurate to say the signal contains no information. Width trails support distance on both metrics and AUROC is near chance; no significance or replicated claim. Within-state and pooled statistics remain separate.
Saved closed exports; no live simulator archives or checkpoints queried.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/summary.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/panels.json


## Slide 4: Harm ranking has signal, but limited separation

Why AUROC matters: a useful warning should rank a harmful alternative above a harmless alternative at the same captured state. Equal-stratum within-state width AUROC is 0.529327, compared with support distance 0.565805, fixed random 0.513251 and constant 0.5. That is weak descriptive separation, not a statistically established failure across seeds. Why AP matters: when warnings are ranked from largest to smallest, AP summarizes enrichment of harmful actions. Width AP 0.510392 exceeds constant 0.274558 but falls below distance 0.542025. Thus the pilot contains some signal; calling it completely uninformative would be wrong. AP uses eligible panels and equal stratum weights, so the constant baseline differs from the pooled 11.98% harm frequency. Why eligible counts matter: 111 of 256 panels contain both classes; the remaining 145 cannot identify binary ranking. Pooled AUROC 0.431776 mixes states and is not a replacement for the primary action-ranking quantity. Expected useful behavior is repeatable within-state ranking and top-warning enrichment, accompanied by uncertainty intervals that respect reset-block dependence. Next: complete the declared seeds, preserve all strata, and measure fresh shifted residual coverage. Weighted CP could address population mismatch in coverage, but the pilot does not identify that mismatch as the cause of weak ranking.

Sources: outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/summary.json; outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json
Evidence scope: dated saved exports. No new simulator outcome or training run.


## Slide 5: Unfamiliar actions and harmful actions differ

The same primary BCA-continuation pilot alternatives are used: near2071 with231 harmful, distant233 with45 harmful. Distant/nonharmful188 and near/nonharmful1840. Harm threshold >1 and support threshold .21134613219046514 are unchanged. Support is minimum RMS action distance among32 nearby recorded observations, not exact policy density or true OOD membership. The descriptive rates are231/2071=11.153...% and45/233=19.313...%. Support-distant actions have a higher observed harm rate, yet231/276 harmful alternatives are near and188/233 distant alternatives are not harmful. These correlated actions and shared resets are not independent training seeds. This motivates balanced support bands and direct harm measurements, not labelling every far action dangerous.
Saved closed exports; no live simulator archives or checkpoints queried.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/summary.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/panels.json


## Slide 6: Distance identifies a harder group, not every bad move

These counts concern nine alternatives per captured row under the frozen BCA continuation, with harmful defined as loss greater than 1 raw reward unit against the host first action. The 2,304 alternatives contain 276 harmful outcomes. Near actions have 231/2,071 harm, or 11.153%; distant actions have 45/233, or 19.313%. The difference makes support shift a useful study dimension but cannot establish causation or a repeated seed-level effect. Most harmful alternatives are near because that group is much larger, and 188 of 233 distant alternatives are not harmful. These two facts guard against interpreting all OOD actions as bad. The support measure is minimum RMS action distance to recorded actions among 32 nearby reference states, with the frozen threshold 0.21134613219046514. It is neither true behavior density nor a calibrated density ratio. Improvement: balanced, predeclared support strata and matched same-state interventions can locate where warning quality changes. For weighted CP, use a justified population ratio; do not insert this distance as though it were Q_X/P_X. No new support threshold or penalty is tuned from these outcomes.

Sources: outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/summary.json; outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json
Evidence scope: dated saved exports. No new simulator outcome or training run.


## Slide 7: Warning quality changes with the situation

The four primary strata retain both policy collectors at reset and after100 collector steps. Width AUROC and support AUROC differ by stratum. Captured reset states under the collectors can coincide; they are not independent samples. The largest width ranking here is at host-collected step100 states, while the host-reset and BCA-step100 strata are below0.5. Do not select the favorable stratum as the overall result. Across all2560 fixed state-action slots the saved width min/median/max are1.352396965/1.746680200/2.478725672. No panel has all nine alternative widths tied, so the near-chance primary summary is not explained by an all-action constant warning. This does not identify the residual model's causal failure. One positive global radius scales widths without changing within-state ordering.
Saved closed exports; no live simulator archives or checkpoints queried.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/summary.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/panels.json


## Slide 8: Warning quality depends on the state group

All four groups contain 64 captured rows. Exact width AUROCs are 0.448775 at host reset, 0.640231 at host step 100, approximately 0.557 at BCA reset and approximately 0.471 at BCA step 100; the preceding result slide retains exact rounded values drawn from the same JSON. Support AUROCs are approximately 0.501, 0.652, 0.588 and 0.522, respectively. The favorable host-step-100 subgroup cannot justify a broad OOD claim. Eligibility differs across groups, and reset origins create dependence. The two reset groups use a common initial reset population, so their differences should not be explained as an isolated state-visitation effect without checking the panel construction. Widths range from 1.352397 to 2.478726, with median 1.746680 and no panel tying all alternative widths. This rules out a trivial all-tied explanation for the entire pilot. A positive global radius multiplies all action widths at a state and cannot change their order. A properly query-dependent conformal radius may change ordering, but weighted CP promises a coverage property under assumptions, not better harm ranking. Improvements should distinguish scale-model ordering, action-shift weighting and state-visitation shift rather than changing all three simultaneously.

Sources: outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/summary.json; outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json
Evidence scope: dated saved exports. No new simulator outcome or training run.


## Slide 9: Action choice: regret among tested actions

Secondary exploratory endpoint declared after collection and before the readout. For a state and fixed continuation c, regret(a)=max over ten tested precommitted unique actions G_c(s,a') minus G_c(s,a). It is not optimal-policy regret, true-Q error or a guarantee about untested actions. All256 captured rows are averaged, including repeated reset blocks. BCA continuation means the same later BCA policy is used to evaluate both candidate first actions. Host continuation likewise holds the later host policy fixed. Mean regret values are6.333291153 and6.452881609 under BCA continuation, and8.242196683 and8.489875844 under host continuation. Their differences equal the negative paired first-action advantage, an arithmetic consistency check. Lower is better, but the small observed differences in one correlated panel are not a significant general disadvantage. This is a local decision metric derived from measured outcomes, separate from whole-policy average return.
Saved closed exports; no live simulator archives or checkpoints queried.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/summary.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/panels.json


## Slide 10: Better average action choice remains unestablished

Regret here is max_a G(s,a; fixed continuation) minus G for the selected first action, restricted to ten tested actions. It is not regret against the unknown globally optimal action or policy. Under BCA continuation, host-first regret is 6.333291 and BCA-first regret 6.452882, a difference of 0.119590. Under host continuation they are 8.242197 and 8.489876, a difference of 0.247679. These are descriptive, exploratory means in one training seed, with correlated states/actions and no new significance claim. Both directions fail to establish the hoped-for mean reduction. They do not identify whether the cause is the scale model, the radius, the host attachment, learned policy differences or the candidate panel. The useful next step is paired replication under the frozen protocol, followed by a separated test of coverage correction and host consumption if warranted. A weighted-CP band can attain better target coverage while still ranking harm poorly or encouraging excessive conservatism. We will therefore keep regret alongside coverage, AUROC/AP, lost good actions and return.

Sources: outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/summary.json; outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json
Evidence scope: dated saved exports. No new simulator outcome or training run.


## Slide 11: Calibration is fitting and changing the penalty

Standard BCA Walker, seed202609171. Both TD3+BC and ReBRAC audited final checkpoint counters independently record1000000 accepted scale fits and500000 actor updates. Each plot shows1000 block means covering1000 host updates each. Actor BC multipliers use actual actor rows, excluding skipped-update zeros; genuine active zeros remain. Final block TD3 multiplier1.331356037 andReBRAC1.377447963. Scale-loss final block .0109333582/.0100045581. These are different host-specific models; the scale objectives are not a cross-host quality ranking. Width-dependent multipliers modulate actor behavior-cloning strength; ReBRAC's critic BC stays at its host value. Standard equal fitting factors do not create an importance-weight degeneracy diagnosis. A learned positive dose and accepted updates establish execution, not calibrated coverage, OOD detection or benefit.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/local_exports.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/cluster_snapshot.json.gz


## Slide 12: BCA is active, but activity does not prove usefulness

The checkpoint counters independently establish one million accepted scale fits in each reviewed Walker run, with one million critic/host updates and 500,000 delayed actor updates. This rules out complete fit abstention for these two examples, not for every host or historic recipe. The final block scale-fitting objectives are TD3 0.0109333582 and ReBRAC 0.0100045581. Their magnitudes do not establish true-Q accuracy, calibration validity or useful risk ordering. Final block actor BC multipliers are 1.331356 and 1.377448: about 33.14% and 37.74% greater imitation coefficients than the same host coefficient before multiplication. These are mean multipliers, not the fraction of actions copied, a coverage level, or percentages of reliance on the old policy. They also do not measure gradient norms. A matched fixed-strength control is a future separately declared comparison, not an already measured explanation or permission to choose a winning constant. The immediate weighted-CP comparison should hold the scale-fitting rule and host attachment fixed, so any coverage change can be attributed to threshold weighting. Fitting IW can be evaluated in a later factorial comparison.

Sources: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/standard_bca/*/walker2d/*/s202609171/verified-v1/; analysis/current_results_20260929/inputs/cluster_snapshot.json.gz
Evidence scope: dated saved exports. No new simulator outcome or training run.


## Slide 13: Bayesian and conformal components are retained

These are the198 saved refreshes from10k to995k in each reviewed standard-BCA Walker run; no interpolated or fabricated final1M refresh. The effective radius equals max(Bayesian,conformal) at every saved refresh. The conformal floor is nonbinding at198/198 for both runs; its zero contribution to the max in these observations does not authorize removing it. Bayesian mass draws remain active. TD3 final saved refresh Bayes1.499558091,conformal1.372334599, frozen residual unit5.178395748. ReBRAC units were not in this refresh export, so no unit history is invented. Radii are in normalized residual-score units, not directly Q-value width. Width also depends on the state-action scale and stored unit. Radius evolution is not empirical coverage.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/local_exports.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/cluster_snapshot.json.gz


## Slide 14: The Bayesian radius controls these observed bands

The effective radius is max(Bayesian, conformal). In both reviewed TD3 and ReBRAC Walker training histories, the conformal floor is lower at all 198 logged refreshes. This establishes which component sets the observed scalar maximum, but does not justify removing the floor. A larger band can cover more responses while being less informative, so both coverage and width are necessary. This result is especially relevant to weighted CP: correcting an inactive conformal floor may leave the final max unchanged. We must inspect conformal-only and full-band diagnostics, and separately specify whether the Bayesian observed-score masses also use valid target weights. Otherwise a null decision change could simply mean the corrected component never affects the consumed band. Positive global scaling cannot repair within-state action ordering. Query-specific conformal radii can vary with the query ratio, but improved ranking remains an empirical question. The Bayesian observed-support bootstrap remains an additional construction; taking a maximum does not validate a finite-draw 95%-confidence population-risk theorem. Keep both components while testing their distinct contributions.

Sources: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/standard_bca/*/walker2d/*/s202609171/verified-v1/; analysis/current_results_20260929/inputs/cluster_snapshot.json.gz
Evidence scope: dated saved exports. No new simulator outcome or training run.


## Slide 15: TD3+BC: inspect Q and Bellman loss

TD3+BC Walker, seed202609171, audited1000-update block means. Q1 at proposed actor actions. TD3 Q is evaluated after critic update before actor update, on active actor rows; ReBRAC Qmin is pre-critic and retained across all host rows. TD3 and ReBRAC Q summaries are not the same estimand and must not be compared as a shared optimism curve. Critic loss is the host's twin-loss training objective, not measured true-Q error or OOD harm. In these last blocks BCA Q is lower but critic loss is higher. That pattern motivates investigation but does not prove corrected overestimation or excessive pessimism. Actor loss and BC accounting are also available in the saved blocks: TD3 bc_loss includes consumed dose, while ReBRAC bc_mse_policy is an unweighted coordinate sum.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/local_exports.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/cluster_snapshot.json.gz


## Slide 16: TD3 estimates fall while its fitting loss rises

TD3 Walker final block Q1 at proposed actor actions changes from host 207.100791 to BCA 191.923997. The critic fitting objective changes from 55.421106 to 61.308944. Both observations are worth showing because return alone cannot reveal whether the critic values or learning signal changed. However, the policies and targets also change during training, so these logs do not compare two predictors against the same truth. Lower Q might reflect reduced overestimation, excessive pessimism or simply different actions/states evaluated by the logging path. Higher fitting loss is not a measurement of OOD harm or a proof of worse true values. Next: compare frozen predictors at common inputs against an explicitly defined fixed response, and use independent simulator outcomes for action quality. The weighted-CP response is the declared frozen one-step Bellman target; its coverage is not true long-horizon Q coverage. The first radius experiment should keep this response definition unchanged to isolate the distribution correction.

Sources: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/standard_bca/*/walker2d/*/s202609171/verified-v1/; analysis/current_results_20260929/inputs/cluster_snapshot.json.gz
Evidence scope: dated saved exports. No new simulator outcome or training run.


## Slide 17: ReBRAC: inspect Q and Bellman loss

ReBRAC Walker, seed202609171, audited1000-update block means. Minimum twin Q at recorded actions. TD3 Q is evaluated after critic update before actor update, on active actor rows; ReBRAC Qmin is pre-critic and retained across all host rows. TD3 and ReBRAC Q summaries are not the same estimand and must not be compared as a shared optimism curve. Critic loss is the host's twin-loss training objective, not measured true-Q error or OOD harm. In these last blocks BCA Q is lower but critic loss is higher. That pattern motivates investigation but does not prove corrected overestimation or excessive pessimism. Actor loss and BC accounting are also available in the saved blocks: TD3 bc_loss includes consumed dose, while ReBRAC bc_mse_policy is an unweighted coordinate sum.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/local_exports.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/cluster_snapshot.json.gz


## Slide 18: ReBRAC shows the same unresolved value tradeoff

ReBRAC Walker final block twin-minimum Q at recorded dataset actions falls from 232.634650 to 216.314605, while critic loss rises from 20.296611 to 24.752905. These are the same qualitative directions as TD3 but use a different query point and logging order; cross-host Q magnitudes cannot rank relative optimism. ReBRAC's actor BC multiplier changes while its critic BC remains part of the original host. A weighted-CP experiment must preserve the host's regularized target definition and critic BC to avoid a confounded comparison. Fixed-target residual tests can assess fitting at a common response. Separate OOD interventions assess whether candidate actions actually cause harm. Neither a training-loss increase nor lower Q by itself establishes why BCA performance is weak. The proposed improvement is better isolation and measurement, followed by a specific weighted-threshold intervention if the population contract is satisfied.

Sources: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/standard_bca/*/walker2d/*/s202609171/verified-v1/; analysis/current_results_20260929/inputs/cluster_snapshot.json.gz
Evidence scope: dated saved exports. No new simulator outcome or training run.


## Slide 19: CQL: inspect its critic and consumed penalty

CQL Walker seed202609171, preliminary frozen cluster export pending full independent audit.1000 retained last-row observations at1000-update intervals; not full scan means or a million per-update rows. Q1 is on recorded actions. Bellman objective and conservative objective are scalar losses, not gradient norms. BCA width at the recorded transition changes the multiplier on the conservative gap, not a separate warning for each sampled negative action. The fourth chart is that logged multiplier; a nonzero or large multiplier is not coverage or detection. Both Hopper and Walker have198 successful refresh records, but numerical historical CQL radii are unavailable in this export. Sparse accepted scale snapshots do not establish every update accepted; checkpoint review is pending. No counters or radius traces are fabricated.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/local_exports.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/cluster_snapshot.json.gz


## Slide 20: CQL changes its critic, but the mechanism is unverified

CQL Walker final sparse snapshot values are host/BCA Q1 177.151443/163.115646, Q1 fitting loss 35.597420/37.195969, and conservative objective 13.682011/13.208880. The BCA critic-dose mean is 1.345180 and scale objective 0.011620161, with this sampled row accepting a fit. These are one of 1,000 sparse retained last rows, not means over all updates. They cannot independently prove one million accepted scale fits. There are 198 successful refresh records, but historical numerical radii are absent from this extracted record, and the full independent checkpoint review remains pending. Why these metrics matter: they show that the intervention reaches the critic, but neither scalar loss magnitude nor lower Q identifies a gradient mechanism or true-value improvement. Current CQL attaches a detached recorded-transition multiplier to the conservative gap. It does not assign a distinct calibrated interval to every sampled negative action in that gap. The first weighted-CP implementation must make the actual query point explicit while preserving the original dual-gap definition. Candidate-specific penalties would be a separate host-method change. Improvement begins with schema/counter review and frozen-input diagnostic comparisons, not silently relabeling the existing treatment.

Sources: outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json; outputs/standard_bca/*/walker2d/*/s202609171/verified-v1/; analysis/current_results_20260929/inputs/cluster_snapshot.json.gz
Evidence scope: dated saved exports. No new simulator outcome or training run.


## Slide 21: Coverage remains an unanswered question

Do not replace the missing calibration-validity test with fitted scale loss, successful updates, radius plots or behavioral returns. The required observable is indicator{|Y-m(X)| <= width(X)} on a fresh appropriately held-out population after the scoring function is frozen, with an explicitly declared one-step response/target, sampling design and assumptions. The target is the specified bootstrapped Bellman response, not true infinite-horizon Q. Nominal coverage90% is a configured target, not a measured90% outcome or90% imitation of the old policy. Nominal Bayesian credibility is also not population coverage. Report empirical coverage, interval width/sharpness, sampling uncertainty and predeclared support-stratum diagnostics separately. Conditional group checks are empirical; the marginal theorem is not automatically conditional coverage. Current behavioral archives do not establish fresh residual coverage, and ReBRAC's next-action target contract remains a separate compatibility constraint. Exact weighted-conformal correction needs justified density ratios/support and query mass; fitting weights alone are not sufficient.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/WEIGHTED_CONFORMAL_DESIGN.md
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/superpowers/plans/2026-09-28-ood-robustness-v2.md
Saved closed exports; no live simulator archives or checkpoints queried.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/summary.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/panels.json


## Slide 22: What the combined metrics mean for BCA

Advisor questions: Why do the experiment, what does it accomplish, do findings match expectations, and what next? We expected useful calibrated warnings and better decisions. Current evidence establishes active calibration machinery in the reviewed training examples and some average-precision enrichment in the pilot; it does not establish reliable harm separation, reduced action regret, empirical coverage or replicated OOD robustness. The raw weak performance cannot identify one causal fault. Finish the five active revised pairs and all declared seeds, report AUROC/AP and complete-tie risk retention, support-stratified harm and decision regret separately from primary stressed-return/degradation endpoints. Regret remains exploratory and no score threshold is selected from these outcomes. Add a separate accepted fresh residual-coverage test with its own target/independence contract. Diagnose the failure before the next treatment: poor harm ranking concerns residual target/scale; good warnings but weak choices concerns host use; population coverage shift with defensible support/ratios motivates corrected weighted conformal. Preserve all original attempts, alpha and both components; no tuning or training launched for this slide update.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/superpowers/plans/2026-09-28-ood-robustness-v2.md
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/WEIGHTED_CONFORMAL_DESIGN.md
Saved closed exports; no live simulator archives or checkpoints queried.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/summary.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/panels.json


## Slide 23: Why these results lead to weighted conformal prediction

Suggested advisor wording: Standard BCA is operational, but the pilot does not yet show strong harm ordering or reduced tested-action regret. That motivates asking whether the calibration errors represent the action population on which we use the bands. Weighted conformal prediction supplies a principled way to adjust a conformal threshold for a declared population shift when its assumptions hold. We will first test that coverage mechanism with the scale predictor and host definition held fixed, then examine whether any coverage improvement helps action ranking or regret. The current data motivate the question, not its answer. Weak AUROC does not establish a coverage violation; no verified fresh OOD residual-coverage measurement is yet available. The code audit provides independent motivation: older fitting-IW variants never connected target/source ratios and query mass to the conformal threshold. Correcting that implementation gap is necessary to test the intended method, even if it ultimately does not improve behavior. Both negative and positive results are informative because they distinguish calibration validity from decision usefulness.

Sources: outputs/advisor_metrics/2026-09-29/METRICS_OVERVIEW.md; docs/WEIGHTED_CONFORMAL_DESIGN.md
Evidence scope: dated saved exports. No new simulator outcome or training run.


## Slide 24: Current IW fitting leaves the weighted-CP step incomplete

This is an implementation gap relative to the intended weighted-conformal method, not a claim that standard equal-weight BCA secretly contains a failed importance-weighting treatment. The clean repository's four host paths refresh through fit_posterior with one group. partitioned_posterior uses a group indicator as calibration mass and query mass one. Therefore enabling fitting IW does not turn the conformal threshold into target/source weighted CP. posterior_radius has lower-level unequal-mass support, but the host path does not supply the needed ratios. A query-dependent conformal threshold also cannot generally be represented by the single cached scalar currently used. CQL policy density alone lacks the behavior-density denominator; TD3/ReBRAC affinity and IQL advantage weights are not automatically density ratios. The calibration bank influences later actor/critic updates, so freezing the final model does not undo this dependence. The first valid empirical test requires a fresh calibration/test design after freezing the score construction, or a separately justified adaptive method. Source: docs/WEIGHTED_CONFORMAL_DESIGN.md and calibration/reference.py; calibration/posterior.py.

Sources: docs/WEIGHTED_CONFORMAL_DESIGN.md; calibration/reference.py; calibration/posterior.py; experiments/weighted_conformal/reference.py
Evidence scope: dated saved exports. No new simulator outcome or training run.


## Slide 25: Weighted CP changes how errors set the band

The additive reference uses frozen X=(s,a), center m(X), positive scale sigma(X), and a specified frozen one-step Bellman response Y using observed reward/next state/terminal flag and fixed target-randomness law. Score S_i=|Y_i-m(X_i)|/sigma(X_i). With w(x)=dQ_X/dP_X(x), use calibration masses w(X_i) and query mass w(x), normalized by their combined total. The conformal radius is the 1-alpha quantile of the observed-score masses plus the query's mass at positive infinity. Common positive rescaling of all weights cancels; separately normalizing query and bank weights is wrong. As a toy arithmetic example only, scores [1,2,3,4], masses [1,1,1,7], query mass 1 and alpha .4 produce cutoff .6*11=6.6, so radius 4. Equal masses instead produce radius 3. Query mass 100 produces cutoff 66 above observed mass 10, so radius infinity. This toy alpha does not change the study alpha .1. A full radius 4 with scale 2 and center 10 yields half-width 8 and interval [2,18]. Source for the conformal correction: Tibshirani et al. (2019), https://www.stat.berkeley.edu/~ryantibs/papers/weightedcp.pdf, equations (5)-(7), Corollary 1 and split-conformal discussion. Bayesian masses and credibility calculation remain a separately labeled construction; the maximum cannot narrow a conformal band, but it supplies no automatic Bayesian population-risk guarantee.

Sources: docs/WEIGHTED_CONFORMAL_DESIGN.md; calibration/reference.py; calibration/posterior.py; experiments/weighted_conformal/reference.py
Evidence scope: dated saved exports. No new simulator outcome or training run.


## Slide 26: A valid ratio needs a defined population

Weighted CP requires more than plugging arbitrary weights into a quantile. We must name the source and target distributions, ensure target support is contained in source support, retain the same conditional response distribution, and justify the calibration/query sampling contract. A policy-action ratio only handles action shift under a justified common state population; changed state visitation requires its state component as well. Deterministic TD3/ReBRAC query actions can be singular relative to continuous source actions. Smoothing an estimated behavior density does not solve that measure mismatch. A declared stochastic neighborhood is a different target and cannot be quietly called deterministic-policy coverage. For an estimated-ratio stage, fit source-versus-target classification on separate covariates, correct class-prior odds, freeze the estimator, and assess overlap/weight concentration. Do not choose ratios using harm labels. Clipping or ESS tempering cannot silently enter the conformal floor as though the original ratio theorem survived unchanged. Freezing after adaptive bank reuse does not restore independence, and episode separation alone does not make within-trajectory transitions IID. Infinite intervals must remain infinite in coverage reports; a policy fallback when an interval is unbounded is a separate engineering and behavioral decision. These are gates for the next experiment, not excuses to relabel the old data.

Sources: docs/WEIGHTED_CONFORMAL_DESIGN.md; calibration/reference.py; calibration/posterior.py; experiments/weighted_conformal/reference.py
Evidence scope: dated saved exports. No new simulator outcome or training run.


## Slide 27: Weighted CP restores coverage in the controlled shift

This figure reuses outputs/weighted_conformal/2026-09-29-v2/results.json with no new samples. Ordinary CP gives 91.00% coverage on its source population but 61.05% on the constructed target shift. Correctly weighted CP gives 90.70%, while the weighted-plus-Bayesian maximum gives 93.95%. Error bars are the saved 95% binomial intervals, not standard deviations across seeds. Source hard-group frequency is .2 and target .8, with unchanged conditional response. Known ratios are .25 and 4. This directly supports the intended population-correction mechanism under the synthetic assumptions. It does not identify the cause of weak real pilot harm ranking or establish a policy improvement. Alpha remains .1. See the following analysis slide for meaning and implementation gates.


## Slide 28: Coverage recovery comes with wider bands

The accepted synthetic result has mean finite radii 2.998574625, 2.998574625, 5.294714196 and 5.589847011 for ordinary/source, ordinary/shifted, weighted-CP/shifted and weighted-plus-Bayesian/shifted. The frozen score has scale one and center zero, so these are also half-widths. Full interval lengths are twice these values. All 2,000 bands are finite in each arm; infinity is tested separately in boundary checks. These means have no uncertainty bars in this figure. Why important: the target population contains more hard/high-error responses, so widening is part of the observed coverage recovery. The extra Bayesian maximum raises coverage and width simultaneously. That is not automatically more efficient or more useful for the host policy. Compare adequate coverage at usable widths, then risk ranking and regret, rather than maximizing coverage alone. Source: outputs/weighted_conformal/2026-09-29-v2/results.json. No new scalar trials, model query, simulator steps or learning updates.


## Slide 29: A controlled shift validates the correction mechanism

The synthetic source contains 20% high-error and 80% low-error examples; the target contains 80% high-error and 20% low-error examples. Conditional responses remain Uniform(0,1) for easy examples and Uniform(0,6) for hard examples, with fixed center zero and scale one. Correct target/source ratios are .25 and 4. Ordinary source coverage is 91.00%, 95% interval 89.66â€“92.22%, with mean finite radius 2.999. Under the target shift ordinary coverage is 61.05%, interval 58.87â€“63.19%, same radius. Weighted CP gives 90.70%, interval 89.34â€“91.94%, radius 5.295. The weighted-plus-Bayesian maximum gives 93.95%, interval 92.81â€“94.95%, radius 5.590. This is the strongest existing direct evidence for the proposed coverage mechanism because the shift and ratios are known by construction. It does not establish that our RL pilot has this cause or these ratios. The wider Bayesian maximum may be less efficient despite greater coverage; no harm-ranking benefit is measured here. Fifteen unit checks include exact finite-population enumeration, query mass, support refusal, ties, common rescaling and infinity boundaries. An independent reader reconstructs all 8,000 radii across four arms. The corrected v2 arithmetic and prior failures are preserved; this slide reuses accepted results and adds no run. Source: outputs/weighted_conformal/2026-09-29-v2/READOUT.md and independent_verification.json.

Sources: outputs/weighted_conformal/2026-09-29-v2/READOUT.md; independent_verification.json; results.json
Evidence scope: dated saved exports. No new simulator outcome or training run.


## Slide 30: Each host keeps its original BCA attachment initially

Implementation is not a blanket change to every action penalty. For TD3+BC and ReBRAC, BCA acts through a detached actor behavior-cloning multiplier; ReBRAC critic BC is unchanged. For CQL, it multiplies each recorded transition's conservative gap while the dual update retains its original unweighted gap. That is not the same as an individual bound on every negative action. For IQL, BCA shrinks capped AWR-weight excess above one, with shared nuisance Q/V and fixed gain one. Advantage-based actor weights are not automatically probability ratios for calibration. The query-aware implementation must store a sorted weighted score bank, the ratio estimator identity, frozen score/target snapshots, units and stochastic target keys. It must evaluate the query ratio at the exact same input for which the host consumes the width. Separate tests should cover normalization, state/action alignment, checkpoint restore equivalence, array shapes, and infinity fallback. Retain both radius components. Do not use the intended weighted-CP correction as an opportunity to alter host losses or select a better recipe by score.

Sources: docs/WEIGHTED_CONFORMAL_DESIGN.md; calibration/reference.py; calibration/posterior.py; experiments/weighted_conformal/reference.py
Evidence scope: dated saved exports. No new simulator outcome or training run.


## Slide 31: Implementation sequence and comparison plan

Stage 1 is the next scientific prerequisite: declare a valid population and frozen response, with source/target support and independence justified. A controlled simulator design can use known action densities under a common declared state distribution, but that tests a calibration mechanism rather than proving offline deployment coverage. Stage 2 is engineering: connect calibration ratios and query mass to the host, preserve explicit infinite radii, version score-bank caches/checkpoints and compare CPU/JAX arithmetic. Stage 3 compares ordinary CP and corrected weighted CP on the same frozen predictor/scale with untouched calibration and target tests. Measure marginal coverage with sampling-aware uncertainty, width/sharpness, unbounded fraction, ratio concentration and support subgroups. Stage 4 only follows a frozen comparison plan: standard BCA versus corrected conformal threshold while holding scale fitting and host attachment fixed. Because the Bayesian component often dominates the max, report whether corrected CP actually changes the consumed band. A separate arm can then test target-weighting of Bayesian observed-score masses under a declared interpretation. Stage 5 considers fitting IW in a factorial extension. Keep the old fitting-IW recipe as a separately named reference if it is included; never rename it weighted CP. Do not choose alpha, ratios or fixed strength by favorable return. This deck is a proposal and analysis, not a new training dispatch.

Sources: docs/WEIGHTED_CONFORMAL_DESIGN.md; calibration/reference.py; calibration/posterior.py; experiments/weighted_conformal/reference.py
Evidence scope: dated saved exports. No new simulator outcome or training run.


## Slide 32: What the next results would mean for BCA

These outcomes are deliberately falsifiable. Good weighted-CP results would support a specific correction under the declared covariate shift, not a universal OOD safety claim. If coverage improves but AUROC/AP or tested-action regret does not, that separates a statistically useful calibration correction from a behavioral benefit. If coverage is obtained mainly through extreme widths or infinity, report that limitation rather than counting it as a practical win. If standard BCA already covers adequately on the declared target population, the particular coverage-mismatch explanation becomes weaker, even if ranking remains poor. Conversely, poor coverage with estimated ratios should trigger an audit of ratio error, support, dependence, response consistency and code before concluding the theorem failed. Negative results still advance the research because each rules out or narrows a mechanism. The existing one-seed pilot, Walker training examples and synthetic known-ratio experiment must remain distinct lines of evidence. The revised OOD queue snapshot in the preceding slides is as of Sep 29 approximately 9 a.m. EDT and is not a new live status check.

Sources: outputs/advisor_metrics/2026-09-29/METRICS_OVERVIEW.md; docs/WEIGHTED_CONFORMAL_DESIGN.md
Evidence scope: dated saved exports. No new simulator outcome or training run.
