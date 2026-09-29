# BCA advisor update — September 29, 2026
Data snapshot: approximately 9:00 a.m. EDT. Original pilot evidence is labelled separately.

## Slide 1: Why are we doing this study?

Does BCA help when an action has little support in the dataset?

Expected result: useful warnings AND better behavior, repeated across seeds.

- Why: training reward alone cannot tell us how the agent handles unfamiliar actions.
- Test: do BCA warnings identify harmful actions, and does its policy handle them better?
- Benefit: locate the limitation before deciding what to change.

**Speaker notes**

Advisor question: Why do this, and what does it accomplish? Our objective is to test whether BCA improves decisions involving poorly represented actions. A high whole-policy return does not identify OOD handling, and a small Bellman error does not establish true long-run value accuracy. BCA fits prediction errors against bootstrapped targets, not known true Q-values. We separate warning quality, first-action choice and behavior after the same imposed action. The hypothesis is that learned residual scale can guide useful host-specific caution; this is not an established causal mechanism. Standard BCA keeps Bayesian and conformal components with equal calibration-fitting importance factors.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/superpowers/plans/2026-09-28-ood-robustness-v2.md
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/WEIGHTED_CONFORMAL_DESIGN.md

## Slide 2: What does the experiment actually do?


| Step | What we do | Question answered |
| --- | --- | --- |
| 1. Freeze | Use trained host and BCA policies | Which learned policy are we testing? |
| 2. Match | Restore the same state and first action | Is the starting situation comparable? |
| 3. Stress | Try near, moderate and distant actions | Does action support matter? |
| 4. Measure | Let each policy continue; record return | What actually happens afterward? |
| 5. Compare | Check behavior, warning scores and seeds | Is any benefit useful and repeatable? |

**Speaker notes**

Advisor question: What does the experiment accomplish? Policies and the calibrator are frozen during OOD evaluation. Calibration itself uses reserved offline transitions, not these simulator rollouts. We capture full simulator states under both collectors, precommit actions/randomness/widths, then restore each state and impose the same first action under host and BCA continuations. Return is raw and undiscounted over up to 250 total transitions including the first action; natural terminations and original time limits remain. Action-support distance is a proxy: minimum RMS action distance to recorded actions among 32 nearby recorded observations. The predeclared q95/q99 support bands are not proof of exact OOD status, and distant is not synonymous with harmful. First-action quality is also compared with later policy held fixed.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/superpowers/plans/2026-09-28-ood-robustness-v2.md

## Slide 3: Why these experimental choices?


| Choice / change | Why it is needed | What we expect to learn |
| --- | --- | --- |
| Standard BCA before IW | Avoid changing two mechanisms at once | Does basic BCA help? |
| Three action-support bands | A few distant examples are insufficient | Does robustness vary with support? |
| Shared familiar-action anchor | Relative losses need a common reference | Less degradation or just a weaker baseline? |
| Separate action and state shift | Unfamiliar states can confound the test | Which type of shift matters? |
| Five training seeds | One favorable run may be accidental | Does the effect repeat? |

**Speaker notes**

Advisor question: Justify each change and say why it makes sense. The revised protocol was declared after inspecting the first study, before revised outcomes; it is a prospective follow-up, not original preregistration. Near/moderate/strong action bands improve representation of support levels. Both policies use the same nearest-recorded familiar-action anchor, and we report absolute stressed return as well as relative degradation, so a weak anchor baseline cannot masquerade as robustness. State-near is the primary action-shift population; state-distant is a separate joint-shift sensitivity. Five independent training seeds are the replication unit; correlated rollouts do not create extra training seeds. Coverage stays alpha 0.1; both BCA components and scientific tolerances stay unchanged. The explicitly approved Walker candidate seed correction changed only 404735174 to 404735181 to remove a stream collision, without selecting based on outcomes. Engineering corrections preserve original failures and completed outcomes.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/superpowers/plans/2026-09-28-ood-robustness-v2.md
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/OOD_V2_STREAM_AMENDMENT_20260928.md

## Slide 4: Where are we now?

CQL’s output check is corrected; its finished training was preserved.

Current bottleneck: finish simulator collection and verify saved outcomes.


| Workstream | Progress | What that means |
| --- | --- | --- |
| Training grid | 94 / 280 results produced | Eight training pairs independently reviewed |
| Original OOD pilot | One TD3+BC Hopper seed complete | Initial evidence; not the revised experiment |
| Revised OOD tranche | 5 active / 20 pairs; 0 complete | TD3+BC and ReBRAC, Hopper and Walker |
| Resources | 2 A100 training lanes + CPU OOD | Cluster and local execution both in use |

**Speaker notes**

These are frozen, sourced counts, not a live claim. Training extraction was September 29 12:59–13:02 UTC; revised OOD snapshot was 12:52 UTC, equivalent to approximately 8:52–9:02 a.m. EDT. There are 94 produced physical training results out of 280: 93 successful processes plus one reviewed recovery retaining original CQL exit 1. Eight host/BCA training pairs have full independent reviews; produced results are not all scientifically accepted. There are 115 actor trajectories because IQL BCA physical runs include paired actors sharing nuisance Q/V; 46 saved within-seed comparisons are available. Training grid: four hosts, seven datasets, five seeds. Revised OOD first tranche: TD3+BC/ReBRAC × Hopper/Walker × five seeds =20 pairs, of which five are active and none complete; fifteen unstarted. One earlier TD3+BC Hopper pilot seed is complete and remains separate. Two A100 training lanes and four cluster CPU OOD workers, plus one local CPU OOD worker, were active in this snapshot. CQL's result-field validator was corrected after training finished, without rerunning it. The local OOD continuation preserves completed outcomes after a database-lock failure; active scientific databases must not be audited while writers are running.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/current_results/2026-09-29/plot_data.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/current_results/2026-09-29/ANALYSIS.md
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/OOD_ACTION_COLLECTION_STATUS.md

## Slide 5: New training results: mixed, not a clear win

TD3+BC Walker ends higher, but its average learning curve is lower.

Meaning: a better endpoint is not yet evidence of a better OOD mechanism.


| Pair | Host final | BCA final | Final change | Curve change |
| --- | --- | --- | --- | --- |
| CQL Hopper | 62.98 | 56.79 | −6.19 | +0.07 |
| CQL Walker | 80.13 | 63.88 | −16.25 | −0.72 |
| TD3+BC Walker | 77.25 | 82.90 | +5.65 | −2.22 |

**Speaker notes**

These three pairs use training seed 202609171, one million host updates, a separate 20-episode final bank, and 200 equally spaced periodic banks from 5k to 1M. Average-curve difference is the arithmetic mean of periodic-bank means, not the final periodic evaluation or selected best checkpoint. TD3+BC Walker is independently reviewed. CQL pairs are preliminary saved-evaluation comparisons pending full independent audits; CQL Hopper's host ran on local RTX5070Ti while its BCA counterpart ran on A100, a further qualification. All values are normalized scores for the named dataset and are not aggregated across datasets. Final endpoint and learning-time behavior answer different questions. TD3 Walker BCA wins only eight of twenty paired final resets despite a higher mean; those episodes are not independent training seeds. None of these training evaluations establish action-stress robustness.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/current_results/2026-09-29/plot_data.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/current_results/2026-09-29/ANALYSIS.md

## Slide 6: The learning curves explain that distinction

BCA does not consistently stay above its host across training.


**Speaker notes**

Each native editable chart uses every saved periodic-bank mean, no smoothing or selected checkpoints. Host is navy; BCA is teal. All use training seed 202609171. Each periodic bank contains ten episodes in these hosts, with evaluation seeds pinned. Axes show normalized score and thousands of host updates. The final reserved bank in the prior slide is separate, so its mean need not equal the final point plotted here. CQL saved summaries remain preliminary, and CQL Hopper has different training hardware across arms. These plots describe policy performance while learning, not harm ranking or conformal coverage.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/current_results/2026-09-29/plot_data.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/current_results/2026-09-29/ANALYSIS.md

## Slide 7: First OOD pilot: benefit is not established

0.5 = chance ranking

BCA first-action gain: −0.12 raw return under the same later policy.

- Width is only slightly above chance.
- The expected warning benefit is not convincing yet.
- One seed cannot establish a general failure or success.

**Speaker notes**

Advisor questions: How did we establish weak evidence, and does it meet expectations? This is the completed original TD3+BC Hopper study, training seed 202609171, not the new revised protocol. There are 5120 rollouts, 1280000 outcome steps, 256 captured state rows and 64 shared reset blocks. Harm is loss >1 raw reward versus the host first action under BCA continuation. There are 276 harmful alternatives among 2304 and 111/256 panels with both classes. The chart is the equal-four-stratum mean of within-state AUROC; constant gives 0.5, fixed random realizes .513251. Pooled width AUROC .431776 answers a different question. These descriptive differences are not significance claims. Expected wider bands to rank harmful actions and BCA behavior to improve; observed ranking is near chance and no average first-action gain. BCA minus host first-action return under the same BCA continuation is -.120. Separate post-hoc distant-action bridge (233 actions in193 states) has absolute continuation return difference -5.046 and relative degradation advantage -.019; it is unbalanced and is not revised OOD evidence. The reason for weak performance is not identified: residual target, scale ranking, host attachment and distribution shift remain competing explanations.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/advisor_progress/2026-09-29/ADVISOR_BRIEF.md

## Slide 8: Why we need all five training seeds

Mean change: +11.88

- One large gain drives the mean.
- Another seed favors the host.
- Finish the declared seed set before claiming reliability.

**Speaker notes**

All three plotted ReBRAC Walker pairs have independent reviews. Training seeds are 202609171, 202609172,202609173. Values are differences between separate-final-bank means, BCA minus host, in normalized score units. Arithmetic mean difference is11.8775121, but one seed contributes33.6399399 while another is negative. This does not provide a five-seed replicated effect or establish OOD performance. Twenty final episodes per run improve evaluation precision but do not create twenty independently trained models. The example is about training variability and responsible interpretation; it is not evidence of why a host attachment succeeds or fails. ReBRAC Hopper's three reviewed seed differences average approximately-.03, nearly tied.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/current_results/2026-09-29/plot_data.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/current_results/2026-09-29/ANALYSIS.md

## Slide 9: What would good or bad OOD results mean?

An OOD outcome tells us what to investigate; it does not identify the training cause.


| Observed result | Meaning | Next question |
| --- | --- | --- |
| Higher stressed return AND less degradation | Evidence of useful action-stress robustness | Does it repeat across all five seeds? |
| Good warnings; weaker behavior | Warning may not be used effectively | Where does the host consume the width? |
| Warnings rank harm poorly | Residual scale may be a weak harm proxy | Is the error target aligned with decision harm? |
| Lower return after the same action | No benefit in that measured setting | Which mechanism explains the loss? |

**Speaker notes**

Advisor question: What do results mean for BCA? Primary action-stress performance A=G_BCA(s,a)-G_host(s,a) after the same imposed action. Relative degradation D=[G_host(s,a0)-G_host(s,a)]-[G_BCA(s,a0)-G_BCA(s,a)], with shared recorded-action anchor a0. We report both and each anchor return; D alone can be favorable because the baseline is weak. Strong support distance at state-near states is primary. The frozen protocol requires all five seeds and positive multiplicity-adjusted lower bounds for both A and D for a directional robustness claim (eight endpoints; two-sided99.375% intervals); ordinary95% intervals are descriptive. Within-state width ranking is secondary, pooled ranking distinct, and regret among tested actions is exploratory, not regret against the optimal policy. Fresh residual coverage is a separate unavailable measurement: weak harm ranking does not disprove a conformal theorem. Warning-good/behavior-bad is a lead about host use, not identified causality.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/superpowers/plans/2026-09-28-ood-robustness-v2.md
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/WEIGHTED_CONFORMAL_DESIGN.md

## Slide 10: What is wrong with the current IW approach?

Why it could help: calibrate for the population where predictions will be used.

Why it may not: weighting cannot create missing evidence or guarantee better actions.


| Component | Current implementation | Needed for intended method |
| --- | --- | --- |
| Scale fitting | Some earlier arms reweighted fitting errors | Keep fixed first to isolate the correction |
| Conformal threshold | Held-out scores effectively had equal masses | Use target/source weights and the query’s mass |
| Validation | CPU/reference mathematics checked | Validate real-data assumptions and host integration |

**Speaker notes**

Advisor questions: Which component was wrong relative to intent, what changes, why? Earlier importance weighting changed the scale-fitting loss, but the radius stage remained effectively an equal-mass held-out conformal calculation with query mass1 and a global radius cache. That is not the intended covariate-shift weighted conformal prediction. The correction uses score S_i=|Y_i-m(X_i)|/sigma(X_i), weights w(X_i)=dQ_X/dP_X, and query weight w(x) at infinity in the weighted1-alpha quantile. Target Y is the frozen one-step bootstrap response, not true Q. Both Bayesian and conformal components remain and their maximum is retained. Finite calibration mass insufficient for the level must yield infinity, not an arbitrary cap. Exact-ratio toy/reference and CPU boundary checks pass, not four-host real-data integration. Real target/source population, support overlap, unchanged Y|X, justified ratios, independently frozen scoring/calibration, query-aware host integration and infinite-radius behavior remain requirements. Estimated ratios are approximate, and deterministic policy targets can be singular. First test radius correction with fitting fixed; fitting IW is a separate comparison. It cannot fix absent support or automatically repair within-state harm ranking.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/WEIGHTED_CONFORMAL_DESIGN.md

## Slide 11: Next steps: finish, diagnose, then change

No coverage tuning yet. Importance weighting remains a testable hypothesis.


| Next action | Why | Deliverable |
| --- | --- | --- |
| Finish and verify active OOD pairs | Obtain real, comparable outcomes | Stressed return, degradation and warning plots |
| Complete the five-seed tranche | Separate repeatable effects from randomness | All seed effects and uncertainty |
| Diagnose the observed limitation | Avoid changing the wrong component | Target/scale, host attachment or population shift |
| Test one justified change at a time | Make any improvement interpretable | Standard BCA versus a declared correction |

**Speaker notes**

Advisor question: What do we do next, and why? Finish the five active revised OOD pairs and independently validate saved outcomes before presenting readouts. Finish all20 declared first-tranche pairs with five seeds per cell; CQL/IQL and other-dataset OOD adapters remain separate work, so training completion is not OOD completion. Report A,strong and D,strong, familiar baselines, missing support quotas/states, uncertainty and within-state ranking. Preserve all negative findings and original failures. The next method depends on evidence: weak ranking motivates investigating residual target/scale; useful warnings but weaker policies motivates controlled host-attachment tests; justified population shift motivates query-aware weighted conformal. Each intervention needs a declared hypothesis, a matching control and a pass/fail reading. No claim IW must improve results, no coverage or strength tuning yet. A responsible final statement is: current evidence does not demonstrate the desired BCA OOD benefit, and the revised experiment is designed to identify the limitation rather than select favorable cases.
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/superpowers/plans/2026-09-28-ood-robustness-v2.md
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/docs/WEIGHTED_CONFORMAL_DESIGN.md
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/current_results/2026-09-29/plot_data.json
Evidence: C:/Users/David Bayha/Documents/GitHub/BCA/outputs/current_results/2026-09-29/ANALYSIS.md
