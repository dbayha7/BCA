# What the retained data and saved uncertainty outputs actually show

This is a retrospective analysis of the completed knee/stall confirmation study: two worlds, three training seeds, native TD3+BC and BCA at kappa 21, 63 and 189. It uses only existing data, training receipts and evaluation arrays. It is not the Honeypot/Bottleneck Path A study. No models, simulator, calibration or optimizer were run, and no original result or criterion was changed.

**The main finding is a separation of roles.** The native baseline has much more harmful optimism in stall than in no-stall. U usually assigns slightly more width to a withheld action than to the logged alternative at the same state, but is much less reliable at identifying which withheld action is harmful. Increasing the downward penalty can therefore improve stall behavior while suppressing useful behavior in no-stall. The retained worlds are not identical information: filtering removed transition rows, but preserved their consequences in later states.

## 1. Start with what the native baseline does

Means below give equal weight to the three training seeds. Return is discounted with gamma 0.98 and horizon 100; higher is better. Exposure is actual executed first-thruster commands beyond the fixed cutoff.

| Native baseline measurement | Stall | No-stall |
|---|---:|---:|
| Policy return | -366.142 | -62.826 |
| Executed withheld-region exposure | 71.54% | 81.10% |
| Own-action raw Q signed error | +188.402 | +6.133 |
| Own-action raw Q MAE | 188.402 | 6.730 |
| Raw-Q candidate-bank regret | 2.66449 | 0.02245 |
| Raw-Q-selected withheld action is worse than logged | 197 / 576 | 28 / 576 |
| Raw-Q-selected withheld action is better than logged | 34 / 576 | 171 / 576 |

Own-action errors compare Q1 with the actual deployed policy's forced-first-action continuation return at 48 of its visited states per seed. All 144 stall own-action errors are positive; no-stall has 124 positive and 20 negative. These are not the same state distribution as the dataset candidate banks: native stall signed error there averages +18.805 on logged actions and +19.696 on withheld actions. Sampling location matters substantially.

The bank results reveal an actionable baseline limitation. In stall, native raw Q selects a withheld action in 77/192 banks in every seed; 66, 64 and 67 of those choices are worse than the logged alternative. In no-stall, the corresponding harmful counts are only 9, 8 and 11, while 57, 56 and 58 choices are useful. Withheld does not mean bad: the bank-best action is withheld in 275/576 stall and 208/576 no-stall evaluations.

Each bank contains the logged complete action plus three versions with first-thruster magnitude 0.85, 0.9 and 1.0. Other action components are unchanged. These are retrospective argmax choices, not the deployed actor's behavior. "Harmful" means lower true continuation return than this logged alternative, not unsafe or globally suboptimal. The 576 evaluations reuse 192 states across three models; they are not 576 independent states.

## 2. What information differs between the retained worlds?

Both archives contain 768 fitting episodes and 76,800 fitting transitions. Their initial states, random seeds, requested/executed actions, split identities, load-band labels and removal masks match exactly. Each CUTOFF training stream retains the same 47,978 row IDs, with no above-cutoff actions.

| Actual learner stream | Rows per world | Joint transitions that differ |
|---|---:|---:|
| Training | 47,978 | 46,532 (96.986%) |
| Calibration | 128 | 122 (95.313%) |
| Gradient probes | 768 | 746 |
| Local-scale bank | 256 | 251 |
| Normalization fixture | 34,085 | 32,927 |

Actions and terminal flags are identical. Source observations, rewards and successors differ on the listed rows. A joint multiset comparison, preserving duplicates and field associations, finds only 1,446 exact shared training transitions; this is not a permutation. Retained float32 rewards differ by as much as 8.90413 return units, and observations by as much as 1.65241 in their recorded coordinates.

Every differing retained state follows an earlier removed above-cutoff action in its original episode; no differing retained state lacks that history. **Deleting a transition did not erase its downstream state consequences.** The original removed tuples are still excluded, and this is not evaluation-label leakage. It is evidence that the realized retained inputs carry world-dependent information. It does not prove that the learner recognized or used it, nor is equality or inequality of one realized sample a general proof about all data-generating laws.

The learner receives float32 `(obs, action, reward, next_obs, done)`. Observation has 13 state coordinates plus time. There is no explicit stall-world identity input: the world flag is metadata used to configure the simulator, not an extra model feature. Current/wave coordinates and time match; differences occur in physical state coordinates 0-9.

Recorded normalization mean/std differ in 10/14 coordinates. All arms and seeds within a world share the same statistics. Applying each world's recorded transform makes all 47,978 normalized observations differ across worlds. A historical receipt string saying "identical all conditions" refers to an inherited within-world normalization rule; it must not be read as identical across worlds. The actual fixture is the within-world CUTOFF/RANDOM retained intersection. RANDOM is not a trained arm in this confirmation.

The analysis independently reconstructed all five streams and matched their hashes against all 30 effective training receipts, including 13 reused from the earlier attempt. Calibration uses one retained row in each of 128 disjoint tuning episodes, with the same 54 within-episode replacements in both worlds. Original successors remain aligned; no deleted-gap adjacency is fabricated. Protected archive roles were decoded only for content authentication, excluded from numerical comparisons and fitting. Full details are in `DATASET_FINDINGS.md` and the dataset JSON.

## 3. What component is U, and what is it being asked to do?

The actual actor objective is native TD3+BC plus `lambda * kappa * mean(U)`, equivalently using score `Q1 - kappa U`. Lambda is computed from unpenalized Q1 and detached. BC strength is unchanged, and the critic retains the native smoothed twin-min TD target and loss form.

U is a positive local scale times a frozen residual unit and a Bayesian/conformal calibration threshold. The local objective fits symmetric soft coverage of a one-step twin-min residual, with a width penalty; it is not a supervised full-return signed-Q-error correction. The calibration uses absolute residual/scale scores. A positive global threshold changes width magnitude, but cancels from within-model ranks and AUROC, so the ordering diagnostics below cannot establish a uniquely Bayesian contribution.

Support novelty, positive Q error, and harmful action choice are different labels. The analysis keeps them separate. Smoothness does not prove that U behaves like Q; an unobserved wider support gap does not automatically imply a larger U. The existing score cannot directly raise a fixed underestimated Q.

## 4. U has local support sensitivity, but weak harm selectivity

Support uses the original strict boundary `abs(float32(a0)) > float32(0.8)`; distance is excess beyond that boundary. It is a constructed one-dimensional action-gap label, not a measured full joint state-action density. This bank also confounds support with action magnitude: there is only one logged candidate and three fixed larger magnitudes.

Across all 24 models, pooled U-versus-support AUROC ranges **0.5175-0.5521**. At the same state, however, a withheld action has higher U than the logged action in **69.10%-94.27%** of comparisons. Both facts are compatible: between-state width variation can obscure a much smaller action-level difference. Calling U universally "support-blind" would be inaccurate.

| Final policy | Stall: pooled / within-state support AUROC | No-stall: pooled / within-state support AUROC |
|---|---:|---:|
| Native | 0.537 / 0.810 | 0.536 / 0.778 |
| BCA 21 | 0.536 / 0.785 | 0.539 / 0.859 |
| BCA 63 | 0.535 / 0.803 | 0.539 / 0.852 |
| BCA 189 | 0.534 / 0.817 | 0.535 / 0.829 |

Entries average the three seed-specific statistics. The known cutoff-distance control has support AUROC 1 by construction; that is not evidence of learned uncertainty quality.

For harmful versus useful withheld actions, the picture is weaker. Among the BCA stall models, within-bank harm AUROC is **0.394-0.544**, based on 59-69 eligible banks per model. Native stall values are 0.415, 0.590 and 0.433. Across-state harm AUROC is also mixed, around chance. In no-stall, apparently perfect conditional values often use only **2-6** banks; at kappa 189 the counts are **1, 0 and 5**. The zero-bank statistic is undefined, not zero or a pass. These sparse comparisons cannot support a general harm-detection claim.

Positive raw error is another distinct event. Its within-bank AUROC is evaluated only where a bank has both error signs: 10-25 eligible stall banks and 3-9 no-stall banks. Higher U does not reliably select the more optimistic action. Bank-centered Pearson correlation between U and signed Q error is negative for every saved model, ranging **-0.361 to -0.022**; absolute-error relationships are weak and inconsistent. Own-policy visited-state correlations vary by seed and are a separate sample. No AUROC direction was flipped and no threshold was fitted; the sign event uses exact error > 0, not a newly claimed practical-error guarantee.

## 5. What the penalty changes, compared with simple bank controls

The diagnostic controls are raw Q1, actual `Q1-kappa U`, a distance penalty with the same **within-bank action-dependent range** as `kappa U`, and allowing only retained candidates. The range match uses saved U and action coordinates, never true returns; it therefore still borrows U's amplitude and is not an independent fixed penalty. Retained-only leaves the logged candidate. None is a newly trained policy arm.

| Final model / world | Raw-Q regret | BCA-score regret | Range-matched distance regret | Retained-only regret |
|---|---:|---:|---:|---:|
| Native / stall | 2.66449 | 2.66449 | 2.66449 | 0.45396 |
| Native / no-stall | 0.02245 | 0.02245 | 0.02245 | 0.31774 |
| BCA 189 / stall | 1.68946 | 1.17188 | 0.74566 | 0.59114 |
| BCA 189 / no-stall | 0.06051 | 0.27764 | 0.32761 | 0.48177 |

Lower regret is better; each row uses that model's own continuation. Native has kappa 0, so its first three controls coincide. All strengths and individual seeds are provided in the CSV and plots.

A simple avoid-all-W rule improves the native stall bank in 197 cases and harms 34; in no-stall it improves only 28 and harms 171. At kappa 189, BCA's score improves 161 stall choices and harms 60, versus 38 helpful and 157 harmful no-stall switches. The distance-shaped control does better on mean stall bank regret but worse on mean no-stall regret at that strength. This supports a conditional avoidance tradeoff, not a conclusion that U or every simple control is always better.

Large downward score offsets are not themselves proof of better or worse ranking: a constant across actions changes neither argmax nor the action gradient. Action-level contrasts matter. At kappa 189, mean U's withheld-minus-logged contrast is about 0.03587 in stall and 0.03444 in no-stall; multiplying by 189 gives about 6.78 and 6.51 score units. Similar downward preferences can have different consequences because useful actions and raw-Q margins differ.

## 6. Preserve the actual policy result and its limits

The trained kappa-189 stall policies improve return by **+11.245, +121.479 and +9.760** across the three seeds. No-stall changes are **-41.421, -98.321 and -20.525**. Every no-stall evaluation start loses at this dose; stall seed 1 supplies about 85% of the summed gain. These are meaningful descriptive observations, not invalidated merely by the report's formal label.

The frozen formal result remains **INCONCLUSIVE**, with C1 **NOT CONFIRMED** and C2 **OVER-CORRECTION** at the prespecified kappa 63. Two Qpi/rollout identity checks exceeded the original 1e-6 relative tolerance; neither tolerance nor result was changed. The 21/189 profile remains exploratory. Three training seeds are the independent training replications; 128 starts per policy are nested evaluations, not 128 independent training runs.

Within a world, candidate state/action inputs match across models. Across worlds, matching row IDs do not imply matching physical states. Across trained models, own-policy Qpi continuations differ; visited-state error panels also change their inputs. These results do not isolate withholding-specific causality, a Bayesian advantage, or a common-policy counterfactual. Saved outputs do not supply full rollout trajectories, true-Q derivatives, conditional OOD calibration coverage, or values for a new region/width. None was invented or regenerated.

## 7. Smallest warranted next decision

Do not redesign the task on the premise that the learner received identical worlds, and do not assume that merely widening the gap or increasing kappa will make U selective for harmful optimism. The sharper hypothesis is: **the current width gives a weak, largely generic preference away from the cutoff, while the value of that preference depends on context.** The saved data support investigating this hypothesis, not declaring it a proved causal mechanism.

For a future knee follow-up, the smallest useful discriminating comparison would hold critic, complete state/action probes and continuation policy fixed, then test whether an independently specified contextual penalty mapping improves harmful-versus-useful choice at matched action-dependent strength. It would need independent evaluation inputs and explicit approval; no mapping, coefficient, dataset or method was changed here. Path A's already approved native-versus-original-BCA comparison stays unchanged. This analysis is not a reason to restart it or strengthen BC.

## Reproduction and evidence

From the existing BCA checkout, run `python -B analysis/knee_support_uncertainty_v1/run_analysis.py --output-dir analysis/knee_support_uncertainty_v1/output/my_new_run` using the existing NumPy/matplotlib environment. The output directory must be new. The script runs only the two local saved-array analyzers and preserves an execution receipt, source hashes, metrics and figures. Original datasets and completed study reports are required; the delivery archive contains analysis code and results, not the original binary datasets or model checkpoints.

Canonical integrated outputs are in `output/verified_run/`: dataset summary and exact source-row membership, per-policy uncertainty metrics, per-bank decisions, three plot pairs and a combined vector PDF, and `RUN_RECEIPT.json`. The detailed field comparison is in `DATASET_FINDINGS.md`; final tests and independent review are in `VERIFICATION.json` and `REVIEW.md`.

The source bindings are the actual `confirmation_v1/cf_worker.py`, `native_screen/screen_worker.py`, `knee_edge_ood_v3/masks.py` and `bindings.py`, pinned UniFloral `candidate.py`, uncertainty `historical/td3_bc_bca.py`/`direct_extension.py`, and `calibration/reference.py`. The final study receipt SHA256 is `b136b39c670e7224c22fe1b8f38062a22d30f608d22c9b4253c406194b53d197`. Reconstructed stream hashes match all 30 completed training receipts. No original input was overwritten, and no training, simulation, recalibration, WSL execution, commit or push occurred.
