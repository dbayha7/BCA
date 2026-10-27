# Comparable development baselines

All rows below use the same frozen UniFloral checkpoint, 128 development states (64 per batch), nine actions per state, and saved continuation values. State IDs, batches, times, episode seeds and all 1,152 candidate values were checked across the cached-target and one-sided reports. The historical kappa=3 row has a small forward-replay difference in Q/width inputs, described below; it is not a newly computed current-runtime control. Higher gain is better. These are one-action interventions followed by the frozen actor, not trained-policy return comparisons.

Two baselines answer different questions: the **frozen actor** represents what the unchanged trained UniFloral policy does at these states; **raw Q1 selection** represents choosing the highest-Q1 action from the same candidate bank with no penalty. Raw Q1 already improves on the actor by +0.1576209947 pooled (+0.2329282281 / +0.0823137613 by batch). Therefore gains relative to the actor include the candidate-selection intervention itself.

| Selector | Exact score / setting | Pooled mean value | Pooled gain vs raw Q1 | Batch 1 gain vs raw Q1 | Batch 2 gain vs raw Q1 |
|---|---|---:|---:|---:|---:|
| Raw Q1 | argmax Q1 | -54.02758298 | 0 | 0 | 0 |
| Frozen actor | Original candidate index 0 | -54.18520397 | -0.157620995 | -0.232928228 | -0.082313761 |
| Original BCA, earlier development setting | Q1 - 3 U_original | -53.96324663 | +0.064336345 | +0.070355793 | +0.058316897 |
| Original BCA, one-sided pilot control | Q1 - U_original | -54.02857025 | -0.000987276 | -0.041047231 | +0.039072679 |
| Actor distance | Q1 - 0.1 × 19.29619572569095 × mean((a-actor)^2) | -54.01876800 | +0.008814977 | -0.054749946 | +0.072379900 |
| New one-sided Bayesian BCA | Q1 - 1.6866286173447431 × u × eta_new; kappa=1 | -53.96465194 | +0.062931042 | +0.103158537 | +0.022703546 |
| New one-sided ordinary calibration | Q1 - 1.2999622747570327 × u × eta_new; kappa=1 | -54.00792540 | +0.019657579 | +0.016085145 | +0.023230013 |
| New raw local scale | Q1 - u × eta_new; coefficient=1 | -53.98626809 | +0.041314883 | +0.059399754 | +0.023230013 |
| Privileged ideal lower-only control | min(Q1, mean of four native candidate targets); coefficient=1 | -53.83015917 | +0.197423808 | +0.213935361 | +0.180912255 |

Here `u=2.6822597980499268`. U_original is the original frozen calibrated width. Matched-effective-strength raw and ordinary controls reproduce all 128 Bayesian choices exactly; their table values are identical to that Bayesian row.

**Coefficient correction:** the completed one-sided pilot's named `original_bca` control is kappa=1, as explicitly implemented in worker.py. It must not be labeled kappa=3. The kappa=3 row comes from the earlier saved development decision-utility reports; it is included descriptively without replacing the completed pilot's predeclared control or changing its failed decision.

The historical kappa=3 report used the operator archive's `saved_q`/`saved_u`; the one-sided pilot used replayed `q_current`/`u`. The maximum absolute differences were Q=0.0000152587890625 and U=0.00000095367431640625. Candidate actions/observations and all continuation values match exactly. Historical raw-Q1 selection IDs and selected values also exactly match current `no_penalty` in all 128 states, so the paired-gain baseline is identical. This permits a descriptive same-bank outcome comparison but is not exact equality of every scoring input. No new kappa=3 selector was computed to erase that caveat. See [KAPPA3_BASELINE_REVIEW.json](KAPPA3_BASELINE_REVIEW.json).

Original BCA at kappa=3 improved over raw Q1 in both batches, with 18 switches: 15 helpful and 3 harmful. New Bayesian BCA made 11 switches: 7 helpful and 4 harmful. The privileged ideal control made 42 switches: 26 helpful and 16 harmful. Thus even the ideal control does not uniformly help every state.

The new Bayesian scale did not clearly improve over the existing development BCA setting: its pooled gain +0.062931042 is slightly below original kappa=3's +0.064336345, and its batch-2 gain is smaller. Its own fixed screen already failed against original kappa=1 in batch 2. These are descriptive development comparisons, not significance or superiority claims.

The ideal control shows **potential from privileged action-specific target information**, not learned-policy benefit or a formal upper bound. It uses evaluator-derived next-transition information for candidate actions, then a four-draw target mean. That information is unavailable to the deployed offline policy. The learned pilot uses logged-transition labels and a calibrated uncertainty bound, so its information and score are not identical. The unrestricted mean-target selector is intentionally excluded from the main penalty table because it can raise Q1 as well as lower it.

To assess potential fairly, retain this exact ideal lower-only mapping and show pooled and both-batch gains, regret and harmful switches beside raw Q1, frozen actor, original BCA and actor distance. Do not maximize over coefficients, target draws or candidate subsets. A deployable BCA candidate must learn only from permitted logged fitting episodes, calibrate on separate episodes, and keep matched-strength controls; evaluation labels stay out of fitting. Existing development data cannot establish fresh confirmation, selected-action coverage or OOD benefit. The completed locked confirmation result remains unchanged and is not pooled into this table.

Sources: [one-sided report](../run/report.json); [lower-only report](../../operator_analysis/cached_lower_only_v1/report.json); earlier `titan_20261026/payload/unifloral_development_decision_utility_v1/results_20261026T205427_503142Z/checks.json` and its two linked development diagnostics. The kappa=3 checks file SHA-256 is `a95eba3c49739abf4c55ca1a779fc67a7c34e54af40bfbb7dcbc920f186a590c`. Full precision for the first two sources is preserved in their reports; kappa=3 batch gains are +0.070355792667327 and +0.05831689699899277, whose equal-size pooled mean is +0.06433634483315989.

No new selectors, fitting, model forwards, simulation or confirmation reads were performed to assemble this comparison.
