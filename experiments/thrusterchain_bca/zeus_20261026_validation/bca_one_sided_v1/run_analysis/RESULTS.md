# One-sided BCA validation — completed, utility screen failed

The single authorized auxiliary-only run completed in 24.866010079 seconds. The prespecified exploratory utility screen **failed**: the new bound improved saved action value relative to Q1 in both development batches, but lost to original BCA in batch 2. Do not promote this variant or start full policy training on this evidence. No rerun, coefficient adjustment, extra fitting or confirmation access was performed during this review.

The run is scientifically complete for its declared scope. `RUN_RESULT.json` records exit 0; its optional `worker_decision` is null because the worker reports the decision as `exploratory_two_batch_utility_screen: false`. Exit 0 means successful execution, not a favorable research result.

**What was tested and preserved.** At the exact transferred UniFloral checkpoint, a separate warm-start copy of BCA's 64×64 local scale network received 128 full-bank auxiliary updates. The copied network was aligned to the one-sided native-target event `y >= Q1 - U`, where `U = q * unit * eta(s,a)`. The current/target actor and twin critics, BC term, fixed normalization, original auxiliary network and original calibration stayed unchanged. The checkpoint actor was originally trained with penalty coefficient zero; these are frozen action-ranking comparisons, not evidence about a trained BCA policy. The score still subtracts uncertainty directly from Q1, with coefficient 1.

The native target used the unchanged discount, finite-horizon mask, target twin-min operator and clipped target-policy smoothing. Each logged row received one frozen native smoothing draw. Fitting used normalized exponential bootstrap weights and the existing coverage-plus-width objective:

```
coverage_i = sigmoid(20 * (y_i/unit - Q1_i/unit + eta_i))
loss = (sum_i w_i * coverage_i - 0.9)^2 + 0.005 * sum_i w_i * eta_i^2
```

Only eta received gradients. Adam used learning rate 0.001 and a fresh optimizer; the residual unit stayed at 2.6822597980499268. Eta retains `softplus(log_eta) + 0.01`, with a public floor of 1e-6. This is a directional uncertainty-bound fit, not supervised return-error correction. Calibration scores were `max(Q1-y,0)/(unit*max(eta,1e-6))`. One Bayesian-bootstrap/conformal calibration used alpha 0.1, credibility 0.95, 1,000 posterior draws and the worst-case test atom; the deployed threshold was the larger of the empirical and posterior thresholds. A nonfinite threshold would refuse action scoring; a zero threshold could not pass the utility screen.

**Data provenance.** The fit used the exact existing 256 training rows from 221 episodes. Calibration reused the original 128 rows from 128 separate tuning-validation episodes. Local validation used one fixed row from each remaining 128 tuning-validation episodes. The three banks are episode-disjoint. The permitted training data were also decoded to reconstruct the original normalization/template; that did not expand the fitting bank. Reserved selection/final-calibration transition rows were skipped before numeric decoding. Candidate evaluator transitions and continuation returns were not used for fitting or calibration. New auxiliary parameters/reference were sealed, then development choices were sealed before saved development values were read.

This is exploratory development evidence: the calibration bank was reused, and the development candidate data had already been inspected. It is not a fresh confirmation experiment.

**Local fitting and validation.** These objectives measure directional coverage plus width, not return-error MSE. The pre-fit measurements below use the copied scale with the new one-sided event and no calibrated multiplier; they are not original calibrated BCA coverage.

| Bank | Objective before | Objective after | Raw directional coverage before | Raw coverage after | Mean raw radius before | Mean raw radius after |
|---|---:|---:|---:|---:|---:|---:|
| Fit, 256 rows | 0.0096966764 | 0.0057538822 | 230/256 = 89.84375% | 229/256 = 89.453125% | 3.71462965 | 2.72331476 |
| Local validation, 128 rows | 0.0101492703 | 0.0118195750 | 113/128 = 88.28125% | 105/128 = 82.03125% | 3.74388623 | 2.77948666 |

The training objective improved while the held-out objective worsened. The local scale contracted and raw validation coverage fell. This reveals a fit/validation gap in this bounded attempt; it does not establish that more training would help, or that all BCA variants are impossible.

**Calibration readouts.** Bayesian threshold q = 1.6866286173447431; ordinary empirical threshold q = 1.2999622747570327. All 1,000 posterior crossings were finite, the reference was nonzero, and action scoring was not refused. Effective calibration sample size was 128.

| Bank | Bound | Covered | Coverage | Mean radius | Maximum radius |
|---|---|---:|---:|---:|---:|
| Fit | Bayesian | 242/256 | 94.53125% | 4.59322077 | 7.45016498 |
| Fit | Ordinary | 232/256 | 90.625% | 3.54020658 | 5.74218492 |
| Fit | Raw scale | 229/256 | 89.453125% | 2.72331486 | 4.41719351 |
| Calibration | Bayesian | 122/128 | 95.3125% | 4.50580810 | 7.42942242 |
| Calibration | Ordinary | 116/128 | 90.625% | 3.47283361 | 5.72619768 |
| Calibration | Raw scale | 107/128 | 83.59375% | 2.67148799 | 4.40489527 |
| Local validation | Bayesian | 116/128 | 90.625% | 4.68796153 | 7.34242147 |
| Local validation | Ordinary | 110/128 | 85.9375% | 3.61322764 | 5.65914204 |
| Local validation | Raw scale | 105/128 | 82.03125% | 2.77948654 | 4.35331252 |

The larger Bayesian radius restored the held-out point estimate to 90.625%. This is a directional native-target coverage observation on logged actions. It is not a true-return bound, a posterior over the Q network, selected-action validity or OOD coverage. The posterior's role remains calibration-risk uncertainty.

**Frozen action utility.** Each of 128 states retained the same nine candidate actions and saved deterministic continuation values. Higher value is better. Regret is the best saved candidate value minus the selected value. Help/harm counts are relative to the Q1 selector. The actor-distance control retained its frozen coefficient; no coefficient sweep occurred.

| Selector | Mean selected value | Mean gain vs Q1 | Mean gain vs original BCA | Mean regret | Switches vs Q1 | Helpful / harmful |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | -54.02758298 | 0 | +0.00098728 | 1.10926702 | 0 | 0 / 0 |
| Original BCA | -54.02857025 | -0.00098728 | 0 | 1.11025429 | 7 | 5 / 2 |
| One-sided Bayesian BCA | -53.96465194 | +0.06293104 | +0.06391832 | 1.04633597 | 11 | 7 / 4 |
| One-sided ordinary | -54.00792540 | +0.01965758 | +0.02064486 | 1.08960944 | 8 | 5 / 3 |
| New raw local scale | -53.98626809 | +0.04131488 | +0.04230216 | 1.06795213 | 5 | 4 / 1 |
| Actor distance | -54.01876800 | +0.00881498 | +0.00980225 | 1.10045204 | 13 | 10 / 3 |
| Frozen actor | -54.18520397 | -0.15762099 | -0.15663372 | 1.26688801 | 54 | 21 / 33 |

The new Bayesian selector's worst single-state gain versus Q1 was -1.2580799487, and its best was +2.6285587690. Its overall mean gain versus actor distance was +0.0541160648. Neither a positive pooled average nor seven helpful switches establishes stable policy benefit.

The prespecified screen required the new Bayesian selector to improve on **both Q1 and original BCA in both batches**, beyond each pair's numerical comparison band. It failed without changing that rule:

| Batch | States | Gain vs Q1 | Gain vs original BCA | Gain vs actor distance | Mean regret | Helpful / harmful vs Q1 |
|---|---:|---:|---:|---:|---:|---:|
| Development 1 | 64 | +0.1031585374 | +0.1442057684 | +0.1579084835 | 1.00844183 | 6 / 3 |
| Development 2 | 64 | +0.0227035460 | **-0.0163691326** | -0.0496763539 | 1.08423012 | 1 / 1 |

Batch 2's gain versus original BCA is -0.01636913260187317, below its positive numerical allowance of 0.000050599485602699126. This is not a numerical tie. Numerical allowances are not statistical confidence intervals.

**Effective-strength controls.** At matched `kappa*q`, raw-scale and ordinary-quantile controls selected exactly the same actions and values as the Bayesian selector in all 128 states. The Bayesian threshold changes global conservatism at this checkpoint; it does not add action-specific information beyond the scale at matched strength. Applying the new threshold to the old local scale also selected exactly the same actions as original BCA in all 128 states. The new local scale changed some choices, but this attempt did not produce a stable advantage over the old BCA control.

**All prespecified strata for the new Bayesian selector.** The original complete table for all controls is [all_prespecified_strata.csv](../run/all_prespecified_strata.csv). Counts below are switches/help/harm versus Q1, not versus original BCA.

| Stratum | n | Gain vs Q1 | Gain vs original BCA | Gain vs actor distance | Regret | Switch / help / harm |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 128 | +0.062931 | +0.063918 | +0.054116 | 1.046336 | 11 / 7 / 4 |
| Development 1 | 64 | +0.103159 | +0.144206 | +0.157908 | 1.008442 | 9 / 6 / 3 |
| Development 2 | 64 | +0.022704 | -0.016369 | -0.049676 | 1.084230 | 2 / 1 / 1 |
| Time 0 | 32 | +0.122099 | +0.122099 | +0.122099 | 0.549701 | 3 / 3 / 0 |
| Time 25 | 32 | 0 | -0.031334 | -0.031334 | 1.458645 | 0 / 0 / 0 |
| Time 50 | 32 | +0.179368 | -0.000351 | -0.006062 | 0.962833 | 3 / 3 / 0 |
| Time 75 | 32 | -0.049743 | +0.165260 | +0.131761 | 1.214166 | 5 / 1 / 4 |
| Development 1 / time 0 | 16 | +0.244198 | +0.244198 | +0.244198 | 0.554332 | 3 / 3 / 0 |
| Development 1 / time 25 | 16 | 0 | 0 | 0 | 1.721490 | 0 / 0 / 0 |
| Development 1 / time 50 | 16 | +0.265815 | 0 | 0 | 0.844718 | 2 / 2 / 0 |
| Development 1 / time 75 | 16 | -0.097379 | +0.332625 | +0.387436 | 0.913227 | 4 / 1 / 3 |
| Development 2 / time 0 | 16 | 0 | 0 | 0 | 0.545071 | 0 / 0 / 0 |
| Development 2 / time 25 | 16 | 0 | -0.062668 | -0.062668 | 1.195799 | 0 / 0 / 0 |
| Development 2 / time 50 | 16 | +0.092920 | -0.000702 | -0.012124 | 1.080947 | 1 / 1 / 0 |
| Development 2 / time 75 | 16 | -0.002106 | -0.002106 | -0.123913 | 1.515104 | 1 / 0 / 1 |

**What this resolves.** The earlier privileged lower-only control established that evaluator-derived candidate target information could improve frozen choices on these development banks. That control used the mean of four candidate target draws. This experiment instead learned a bound from logged transitions with one target draw per row. It therefore tested whether a small BCA-consistent local-scale adaptation could transfer useful directional information while retaining Bayesian/conformal calibration. The observed coverage/width tradeoff and failed batch comparison mean that transfer is not established. The ideal control does not justify fitting privileged candidate or return labels into BCA.

**Decision.** Close this fixed pilot as a failed promotion screen and retain original BCA as the unchanged reference. Full training remains paused. A later change to the local fitting objective or Bayesian role would need its own frozen hypothesis, budget and independent evaluation plan. This result alone does not authorize a longer fit, strength sweep, revised threshold, new simulator work or confirmation retuning. It also does not prove that BCA cannot work: it concerns one checkpoint, one seed, 128 auxiliary steps and reused development data.

**Execution and review evidence.** The worker completed 128/128 auxiliary updates (32,768 gradient rows) and one new calibration. It evaluated 512 logged rows through the current critic, target actor and target critic; three bank-level native-label draws produced one smoothing draw per row. There were 896 auxiliary diagnostic rows, 1,152 development auxiliary rows and 1,152 saved development-value rows read. Actor, critic, target and original auxiliary updates were all zero; original calibration refreshes, simulator transitions, new rollouts and confirmation reads were all zero. These diagnostic ledger counts exclude restoration/template/replay forwards; they are not a GPU-kernel accounting claim. No timeout occurred under the 300-second worker cap.

The run used the isolated Zeus Ubuntu GPU environment, Python 3.12.14, JAX/jaxlib/CUDA plugin 0.6.2, NumPy 2.5.4, and RTX 5070 Ti Laptop GPU. The disclosed JAX/NumPy environment is outside upstream constraints, as on Titan. Frozen UniFloral source, checkpoint and normalization were preserved; no CORL code or checkpoint was substituted.

Independent saved-result review passed 3,643 assertions covering 1,280 selection records, 30 overall/batch/control summaries, 128 calibration scores, the existing 1,000 posterior crossings, post-fit coverage/width and matched-strength identities. Integrity review checked all 15 PREPARED files, all 12 reported run artifacts, source manifests, provenance links, checkpoint/export components, seals and original-state snapshots. The original-state before/after canonical encoding hashes agree. No material execution incompleteness was found.

The review checks immutable bytes and saved arithmetic, not independent attestation of every historical GPU forward. Pre-fit eta and target-critic outputs were not separately saved; posterior exponential draws were not saved; development eta was recovered algebraically from sealed scores for mapping checks. The reviewer did not regenerate models, labels or posterior draws. These limits are recorded in [INDEPENDENT_REVIEW.json](INDEPENDENT_REVIEW.json).

| Evidence | SHA-256 |
|---|---|
| Frozen protocol | `a4a507a67f47c7bdac2a75f279801b33086a452b29679c5ea04eef190f0f299b` |
| Original run report | `320113721d15e54bed4047e9574e94fac6d9ed78eca6d140575f386ae065b011` |
| Native UniFloral td3_bc.py | `6f5ab6f98d706d902a18f11e1c81e99e2691dc4c8a53843bf6521c9f95d00eee` |
| Checkpoint state.msgpack | `874d65b7f4dcb5ec285a0d4360439c73a787b5e8c567ff7c0138e0c1e22948e8` |
| Frozen normalization receipt | `4f6be8817906c2ed99a9e2ef1355c43d9ce5640ec3984e104625b85d25ba5dc3` |
| Independent review receipt | `3db5bbf59b21e33fe8e2464e478282f06b207301e1b09d975439175660bfc5da` |

The raw [report.json](../run/report.json), [protocol.json](../protocol.json), [integrity decision receipt](integrity_and_decision.json), and [independent verifier](verify_saved_results.py) preserve the detailed evidence. To reproduce the saved arithmetic only, from the Windows repository root:

```powershell
.\.venv\Scripts\python.exe -B experiments/thrusterchain_bca/zeus_20261026_validation/bca_one_sided_v1/run_analysis/verify_saved_results.py
```

Do not invoke the scientific launcher again. No execution blocker remains for interpreting this completed run. Assistant WSL execution remains denied; the user manually launched the successful run. No changes were committed or pushed.
