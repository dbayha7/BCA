# BCA advisor update: metrics beyond return

## 1. The study needs more than a return curve


| Question | Metric | Evidence available |
| --- | --- | --- |
| Are bands reliable? | Fresh empirical residual coverage | Not yet verified |
| Are warnings useful? | Within-state AUROC and average precision | Completed Hopper pilot |
| Is unfamiliarity linked to harm? | Support distance versus harm rate | Completed Hopper pilot |
| Are choices better? | Regret among tested actions | Exploratory pilot readout |
| Is BCA working during training? | Scale fits, width, radii and penalty dose | Reviewed TD3/ReBRAC; CQL preliminary |
| What changes in the critic? | Q estimates and Bellman/conservative loss | Training diagnostics; not true-Q error |

Research objective: Determine whether BCA produces useful prediction-error bands and whether their use by the host improves action decisions. Whole-policy return is only one end result. This deck adds the missing evidence chain: empirical coverage, width/sharpness, warning discrimination, decision regret, scale fitting, component radii, consumed penalty, and critic behavior. Fresh residual coverage is not established. The prior slide deck overemphasized return and omitted already-available calibration diagnostics and secondary pilot metrics. Pilot statistics are one TD3+BC Hopper training seed; training diagnostics below use seed202609171 on Walker. Those are different cells and cannot be merged into a same-run causal explanation. Revised OOD snapshot still has zero completed pairs.
Saved closed exports; no live simulator archives or checkpoints queried.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/summary.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/panels.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/local_exports.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/cluster_snapshot.json.gz

## 2. Warnings: some enrichment, weak separation

Width carries some signal, but the simple distance warning is stronger here.


Original TD3+BC Hopper pilot, seed202609171, frozen BCA continuation. Harm means losing more than1 raw reward unit relative to host first action in the same state over250 total transitions. Each panel has nine alternatives. AUROC and average precision are computed within eligible panels, then averaged within the four collector/capture strata; the displayed mean gives equal weight to each stratum. There are111/256 eligible panels. All145 single-class panels are N/A for these rankings. AUROC asks how often a harmful action scores above a nonharmful action;0.5 is chance, ties half. AP summarizes precision as high-score actions are selected; higher is better, with tied-score groups handled together. Constant-score AP=.27456 is the matched eligible-panel baseline, not the11.98% pooled harm frequency. AP shows enrichment over constant/random, so it is inaccurate to say the signal contains no information. Width trails support distance on both metrics and AUROC is near chance; no significance or replicated claim. Within-state and pooled statistics remain separate.
Saved closed exports; no live simulator archives or checkpoints queried.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/summary.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/panels.json

## 3. Unfamiliar actions and harmful actions differ

Why test both? BCA must distinguish bad actions, not merely unfamiliar ones.
- 231 of 276 harmful actions were still support-near.
- 188 of 233 distant actions were not harmful.
- Novelty is a useful descriptor, not a harm label.


The same primary BCA-continuation pilot alternatives are used: near2071 with231 harmful, distant233 with45 harmful. Distant/nonharmful188 and near/nonharmful1840. Harm threshold >1 and support threshold .21134613219046514 are unchanged. Support is minimum RMS action distance among32 nearby recorded observations, not exact policy density or true OOD membership. The descriptive rates are231/2071=11.153...% and45/233=19.313...%. Support-distant actions have a higher observed harm rate, yet231/276 harmful alternatives are near and188/233 distant alternatives are not harmful. These correlated actions and shared resets are not independent training seeds. This motivates balanced support bands and direct harm measurements, not labelling every far action dangerous.
Saved closed exports; no live simulator archives or checkpoints queried.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/summary.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/panels.json

## 4. Warning quality changes with the situation

Widths vary: 1.35–2.48. No panel has all alternative widths tied.
Changing one global radius cannot repair which action is ranked as riskier.

| Collected states | Eligible / 64 | Width AUROC | Distance AUROC |
| --- | --- | --- | --- |
| Host · reset | 23 | 0.449 | 0.501 |
| Host · step 100 | 34 | 0.640 | 0.652 |
| BCA · reset | 25 | 0.557 | 0.588 |
| BCA · step 100 | 29 | 0.471 | 0.522 |

The four primary strata retain both policy collectors at reset and after100 collector steps. Width AUROC and support AUROC differ by stratum. Captured reset states under the collectors can coincide; they are not independent samples. The largest width ranking here is at host-collected step100 states, while the host-reset and BCA-step100 strata are below0.5. Do not select the favorable stratum as the overall result. Across all2560 fixed state-action slots the saved width min/median/max are1.352396965/1.746680200/2.478725672. No panel has all nine alternative widths tied, so the near-chance primary summary is not explained by an all-action constant warning. This does not identify the residual model's causal failure. One positive global radius scales widths without changing within-state ordering.
Saved closed exports; no live simulator archives or checkpoints queried.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/summary.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/panels.json

## 5. Action choice: regret among tested actions

The BCA first action does not reduce average tested-action regret in this pilot.


Secondary exploratory endpoint declared after collection and before the readout. For a state and fixed continuation c, regret(a)=max over ten tested precommitted unique actions G_c(s,a') minus G_c(s,a). It is not optimal-policy regret, true-Q error or a guarantee about untested actions. All256 captured rows are averaged, including repeated reset blocks. BCA continuation means the same later BCA policy is used to evaluate both candidate first actions. Host continuation likewise holds the later host policy fixed. Mean regret values are6.333291153 and6.452881609 under BCA continuation, and8.242196683 and8.489875844 under host continuation. Their differences equal the negative paired first-action advantage, an arithmetic consistency check. Lower is better, but the small observed differences in one correlated panel are not a significant general disadvantage. This is a local decision metric derived from measured outcomes, separate from whole-policy average return.
Saved closed exports; no live simulator archives or checkpoints queried.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/summary.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/panels.json

## 6. Calibration is fitting and changing the penalty

Both accepted 1,000,000 scale fits. Final block strength: ×1.331 / ×1.377.


Standard BCA Walker, seed202609171. Both TD3+BC and ReBRAC audited final checkpoint counters independently record1000000 accepted scale fits and500000 actor updates. Each plot shows1000 block means covering1000 host updates each. Actor BC multipliers use actual actor rows, excluding skipped-update zeros; genuine active zeros remain. Final block TD3 multiplier1.331356037 andReBRAC1.377447963. Scale-loss final block .0109333582/.0100045581. These are different host-specific models; the scale objectives are not a cross-host quality ranking. Width-dependent multipliers modulate actor behavior-cloning strength; ReBRAC's critic BC stays at its host value. Standard equal fitting factors do not create an importance-weight degeneracy diagnosis. A learned positive dose and accepted updates establish execution, not calibrated coverage, OOD detection or benefit.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/local_exports.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/cluster_snapshot.json.gz

## 7. Bayesian and conformal components are retained

Conformal is below Bayesian at 198/198 refreshes in each run.


These are the198 saved refreshes from10k to995k in each reviewed standard-BCA Walker run; no interpolated or fabricated final1M refresh. The effective radius equals max(Bayesian,conformal) at every saved refresh. The conformal floor is nonbinding at198/198 for both runs; its zero contribution to the max in these observations does not authorize removing it. Bayesian mass draws remain active. TD3 final saved refresh Bayes1.499558091,conformal1.372334599, frozen residual unit5.178395748. ReBRAC units were not in this refresh export, so no unit history is invented. Radii are in normalized residual-score units, not directly Q-value width. Width also depends on the state-action scale and stored unit. Radius evolution is not empirical coverage.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/local_exports.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/cluster_snapshot.json.gz

## 8. TD3+BC: inspect Q and Bellman loss

Final block Q: 207.10 → 191.92; critic loss: 55.42 → 61.31.


TD3+BC Walker, seed202609171, audited1000-update block means. Q1 at proposed actor actions. TD3 Q is evaluated after critic update before actor update, on active actor rows; ReBRAC Qmin is pre-critic and retained across all host rows. TD3 and ReBRAC Q summaries are not the same estimand and must not be compared as a shared optimism curve. Critic loss is the host's twin-loss training objective, not measured true-Q error or OOD harm. In these last blocks BCA Q is lower but critic loss is higher. That pattern motivates investigation but does not prove corrected overestimation or excessive pessimism. Actor loss and BC accounting are also available in the saved blocks: TD3 bc_loss includes consumed dose, while ReBRAC bc_mse_policy is an unweighted coordinate sum.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/local_exports.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/cluster_snapshot.json.gz

## 9. ReBRAC: inspect Q and Bellman loss

Final block Q: 232.63 → 216.31; critic loss: 20.30 → 24.75.


ReBRAC Walker, seed202609171, audited1000-update block means. Minimum twin Q at recorded actions. TD3 Q is evaluated after critic update before actor update, on active actor rows; ReBRAC Qmin is pre-critic and retained across all host rows. TD3 and ReBRAC Q summaries are not the same estimand and must not be compared as a shared optimism curve. Critic loss is the host's twin-loss training objective, not measured true-Q error or OOD harm. In these last blocks BCA Q is lower but critic loss is higher. That pattern motivates investigation but does not prove corrected overestimation or excessive pessimism. Actor loss and BC accounting are also available in the saved blocks: TD3 bc_loss includes consumed dose, while ReBRAC bc_mse_policy is an unweighted coordinate sum.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/local_exports.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/cluster_snapshot.json.gz

## 10. CQL: inspect its critic and consumed penalty



CQL Walker seed202609171, preliminary frozen cluster export pending full independent audit.1000 retained last-row observations at1000-update intervals; not full scan means or a million per-update rows. Q1 is on recorded actions. Bellman objective and conservative objective are scalar losses, not gradient norms. BCA width at the recorded transition changes the multiplier on the conservative gap, not a separate warning for each sampled negative action. The fourth chart is that logged multiplier; a nonzero or large multiplier is not coverage or detection. Both Hopper and Walker have198 successful refresh records, but numerical historical CQL radii are unavailable in this export. Sparse accepted scale snapshots do not establish every update accepted; checkpoint review is pending. No counters or radius traces are fabricated.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/local_exports.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/analysis/current_results_20260929/inputs/cluster_snapshot.json.gz

## 11. Coverage remains an unanswered question


| Measure | What it answers | Current evidence |
| --- | --- | --- |
| Empirical residual coverage | How often does the band contain its stated target? | Not verified on fresh OOD samples |
| Band width / sharpness | How much uncertainty does it report? | Pilot widths 1.35–2.48; not coverage |
| Coverage by support band | Where does calibration fail under shift? | Requires a separate accepted readout |
| Fit counts and radius refreshes | Did the calibration machinery operate? | Verified for reviewed TD3/ReBRAC runs |
| True-Q prediction error | Does Q match actual long-run value? | Not supplied by training Bellman loss |

Do not replace the missing calibration-validity test with fitted scale loss, successful updates, radius plots or behavioral returns. The required observable is indicator{|Y-m(X)| <= width(X)} on a fresh appropriately held-out population after the scoring function is frozen, with an explicitly declared one-step response/target, sampling design and assumptions. The target is the specified bootstrapped Bellman response, not true infinite-horizon Q. Nominal coverage90% is a configured target, not a measured90% outcome or90% imitation of the old policy. Nominal Bayesian credibility is also not population coverage. Report empirical coverage, interval width/sharpness, sampling uncertainty and predeclared support-stratum diagnostics separately. Conditional group checks are empirical; the marginal theorem is not automatically conditional coverage. Current behavioral archives do not establish fresh residual coverage, and ReBRAC's next-action target contract remains a separate compatibility constraint. Exact weighted-conformal correction needs justified density ratios/support and query mass; fitting weights alone are not sufficient.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/WEIGHTED_CONFORMAL_DESIGN.md
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/superpowers/plans/2026-09-28-ood-robustness-v2.md
Saved closed exports; no live simulator archives or checkpoints queried.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/summary.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/panels.json

## 12. What the combined metrics mean for BCA


| Evidence | What it establishes | Next step and why |
| --- | --- | --- |
| Fits accepted; multiplier changes | BCA is active during training | Evaluate whether that signal is useful |
| AP enrichment; near-chance AUROC | Some warning information, limited separation | Finish support-balanced, five-seed tests |
| No lower tested-action regret | No demonstrated choice benefit in the pilot | Check warning-to-host influence separately |
| Fresh coverage absent | Calibration validity under shift is unresolved | Measure coverage against a declared target |
| IW only changed fitting previously | Intended weighted-conformal stage was missing | Validate weighted quantile and query mass |

Advisor questions: Why do the experiment, what does it accomplish, do findings match expectations, and what next? We expected useful calibrated warnings and better decisions. Current evidence establishes active calibration machinery in the reviewed training examples and some average-precision enrichment in the pilot; it does not establish reliable harm separation, reduced action regret, empirical coverage or replicated OOD robustness. The raw weak performance cannot identify one causal fault. Finish the five active revised pairs and all declared seeds, report AUROC/AP and complete-tie risk retention, support-stratified harm and decision regret separately from primary stressed-return/degradation endpoints. Regret remains exploratory and no score threshold is selected from these outcomes. Add a separate accepted fresh residual-coverage test with its own target/independence contract. Diagnose the failure before the next treatment: poor harm ranking concerns residual target/scale; good warnings but weak choices concerns host use; population coverage shift with defensible support/ratios motivates corrected weighted conformal. Preserve all original attempts, alpha and both components; no tuning or training launched for this slide update.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/superpowers/plans/2026-09-28-ood-robustness-v2.md
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/WEIGHTED_CONFORMAL_DESIGN.md
Saved closed exports; no live simulator archives or checkpoints queried.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/summary.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/ood/td3_bc/hopper/s202609171/real-harm-report-v1/panels.json
