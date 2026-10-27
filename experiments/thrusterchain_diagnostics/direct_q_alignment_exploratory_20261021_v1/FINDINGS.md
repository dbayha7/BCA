# What the saved width signal explains

**Decision: do not advance to broad actor training on this evidence.** The signal is not constant or floored, but its variation across actions has little association with value errors on this bank. First run a separately frozen, small fresh-bank replication of the signal test in `FRESH_BANK_PLAN.md`. If that fails, redesign the signal/target alignment before training. This is an exploratory conclusion for one fixed model, not proof that BCA generally fails.

The 32-state, 384-action bank has already been inspected. All numbers below are descriptive reuse of its completed deterministic Qpi records. No model call, update, new oracle label or rollout was made. Authoritative outputs are `results_final/`; the earlier `results/` is retained preliminary output before adding the explicit equal-state pair denominator and architectural-floor reporting. The preliminary manifest records its earlier source hash, but an exact snapshot of that earlier script was not retained. Its outputs therefore have a source-reproduction gap; they are not claimed to be bound to the final script. Final source, protocol and numeric outputs are fully bound by the verified final manifest.

## Why only 10 of 32 correlations were defined

Let e=Q1−Qpi. There are 323 negative errors and 61 positive errors. At **22 states all 12 errors are negative**, making max(e,0) identically zero. These are exactly the 22 undefined positive-overestimation correlations. None is caused by constant width, insufficient candidates, missing labels or nonfinite data. Signed-error and absolute-error correlations are defined at all 32 states.

| Width association | Mean within-state Spearman | States with defined coefficient |
|---|---:|---:|
| Signed Q1−Qpi error | 0.041302 | 32/32 |
| Absolute Q1−Qpi error | −0.063811 | 32/32 |
| Positive-part overestimation | −0.154882 | 10/32 |

The last row describes a different subset; it is not an estimate over all 32 states with missing correlations set to zero.

## State variation is not action discrimination

Between-state means account for **94.67% of width variance** and **94.69% of signed-error variance**. Pooled width versus absolute-error Spearman is +0.200658, but after subtracting each state's mean it is −0.037951. For signed error, the corresponding values are −0.047106 and +0.028461. This demonstrates that the pooled association does not establish useful discrimination among actions at a fixed state; it does not establish a causal confounding mechanism.

| Time / remaining horizon | States | Mean width | Mean absolute Q1 error |
|---|---:|---:|---:|
| 0 / 100 | 12 | 2.605604 | 31.303531 |
| 50 / 50 | 12 | 2.610614 | 16.623003 |
| 99 / 1 | 8 | 2.660417 | 5.568478 |

These are different physical states at each time, not matched trajectories. Mean widths also differ by context: stall-low 2.0510, stall-high 3.4539, monotone-low 1.9540, monotone-high 3.0259. Complete context/time strata, including the small 2–3-state cells, remain in CSV; no multi-covariate fit or causal time effect is asserted.

Width is more associated with action distance: state-centered Spearman is +0.451547 for mean squared distance from the behavior mode and +0.431464 for mean squared distance from the original actor. These distances are not empirical state support or a density ratio. The associations do not prove that eta is merely a distance estimator.

## Width is varying; a global quantile change cannot repair its ranks

All states have nonconstant widths. Their within-state ranges span **0.136062–0.822985**, median 0.445008. Eta spans **0.714649–2.079980**: zero entries reach the architectural softplus-plus-0.01 floor or the separate 1e−6 width guard. The original actor has no exact ±1 endpoint; its maximum absolute action is 0.979721. Designed grid endpoints account for boundary candidate commands. Saved outputs do not identify network activation saturation or action derivatives.

Here R=1.3715263605117798 and unit=1.4656671285629272 give the same positive factor **R×unit=2.0102011025596624** for every candidate. U=R×unit×max(eta,1e−6), up to the recorded float32 arithmetic difference (maximum 2.88e−7). Width and eta ranks agree at all 32 states.

**For this frozen scalar-reference construction**, changing only the nominal level/reference quantile with eta fixed rescales finite widths, or can produce vacuity. It cannot repair their rank association with value errors. It can change penalty strength and therefore the argmax of Q1−U. This is not a claim about all BCA/IW variants, and no coefficient or coverage sweep was run.

## Relative errors explain the one changed decision

For 2,112 within-state unordered action pairs, compare ΔU with Δe: the width difference versus the difference in signed value errors. Correlation is computed with both orientations of each pair to avoid dependence on arbitrary candidate ordering; these are dependent pairs, not 4,224 independent observations. Pearson is **0.030222** and Spearman **0.027556**. Mean absolute advantage error changes from 4.363834 to 4.364224 after subtracting width.

There are **2,106 comparable pairs**, excluding six true-value near-ties under the fixed 1e−6 rule. Q1−U repairs 18 rankings and breaks 18. The pooled pair loss is unchanged. Equal-state mean concordance changes slightly, **0.661868762748 → 0.661876274767**, because states have different eligible-pair denominators; do not conflate these two aggregations.

Only state 6 (stall-low, time 0) changes its full-bank argmax, from candidate 2 to candidate 6, the original actor. Raw Q favors candidate 2 by 0.055481, while its extra width is 0.164874. The switch improves true Qpi by **4.922153**, although both actions are underestimated: errors are −15.783762 and −20.761396. This is a useful correction of **relative overvaluation**, not evidence of positive absolute overestimation detection.

| Full-bank selector | Mean gain versus raw Q | Mean gain versus original actor | Better / worse / tied versus original actor |
|---|---:|---:|---:|
| Raw Q | 0 | 0.164100 | 5 / 18 / 9 |
| Q1−U | 0.153817 | 0.317917 | 5 / 17 / 10 |
| Frozen matched-distance score | 0.405389 | 0.569489 | 7 / 14 / 11 |
| Original actor | −0.164100 | 0 | 0 / 0 / 32 |

These are one-action deviations followed by the same original actor. They are not new multistep policies. The original-actor baseline is used only for this unrestricted full-bank analysis; it is not silently inserted into common/rare support subsets. Constant-offset rankings match raw Q; detached-width rankings match live-width rankings because this is discrete selection, not an actor-gradient update. Coefficients were not retuned.

## Measurable head differences; unavailable causal explanations

The 95k reference scored a noisy one-step target-network residual around twin-min Q. These records use the current 100k Q1 and actual full-horizon original-actor Qpi. The head, target and 5,000-update reference differences are real definition mismatches, not identified causes of the observed alignment.

Q1 exceeds min(Q1,Q2) on 190/384 candidates, by mean 1.019543 over all candidates (maximum 11.640442). Against the same Qpi, taking the minimum reduces positive errors from 61 to 54 but increases mean absolute error from **19.364570 to 20.076462**, reflecting greater underestimation. It leaves mean within-state width/absolute-error rank association near zero (−0.019012). This arithmetic contrast does not recreate the 95k residual calibration target.

Repeated one-step noise labels, matched saved successor/current-versus-target readouts, old-network predictions on this bank, activation states and Jacobians are absent. Target networks are EMA parameters recorded at 100k, not models with a single known age. Consequently noise variance, a lag-removal counterfactual, and epistemic uncertainty versus representation bias are unavailable. No new model calls were used to manufacture them.

## Reproduction and evidence

Run from this directory, using a Python environment with NumPy and SciPy:

```powershell
& 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' .\analysis.py --input ..\direct_q_diagnostic_20261021_v2 --output .\reproduced_results
```

The script refuses to overwrite an output directory. It checks saved error arithmetic, behavior-mode distances, original-actor slot identity and all original full-bank positive-error correlations. `results_final/summary.json` records versions, input/code/protocol hashes and limitations; `results_final/manifest.json` hashes all numeric outputs. CSVs retain every state, pair and selection, including undefined reasons. No significance or generalization claim is made from this already inspected bank.

Source definition references: v2 `diagnostic.py:87–100,139–165,217–223`; `run_diagnostic.py:182–184`; `protocol.json` and `prepared/engine_identity.json:478–482`. Preserved source closure: `calibration/network.py:30`, `calibration/dose.py:75–76`, `algorithms/td3_bc_bca.py:146–156,244–267`, `experiments/thrusterchain_diagnostics/training.py:491,512–520`. The inherited engine metadata's 64-state action-only law is not the v2 main-bank law; this analysis uses the v2 protocol and saved 32-state arrays.
