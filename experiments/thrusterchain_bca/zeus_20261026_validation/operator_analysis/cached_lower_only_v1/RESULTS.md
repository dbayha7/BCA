# Lower-only ideal-penalty feasibility: completed once

**The lower-only selector passed the prespecified exploratory screen in both batches.** Mean saved-value gain versus Q1 was **+0.2139353609** in development 1 and **+0.1809122553** in development 2. Their mean numerical allowances were 0.0000594685 and 0.0000506929. This shows that some improvement on these saved banks is achievable by lowering scores alone; the successful unrestricted target control did not depend entirely on upward adjustments.

This remains an ideal control using privileged evaluator-derived targets. It does not establish that a learned or calibrated BCA penalty achieves that gain. No fit or method modification was executed.

## Exact check and environment

The [protocol](protocol.json) was frozen at 22:43:50 UTC before this selector was calculated. The one new score was:

```text
lower_only(s,a) = min(Q1(s,a), mean_native_target(s,a))
                = Q1(s,a) - max(Q1(s,a) - mean_native_target(s,a), 0)
```

The coefficient was one. We reused the previously sealed float64 Q1 and four-draw mean-target scores, the same 128-by-nine development bank, original ordering, score ties and return tolerances. The minimum and subtraction forms agreed exactly on all 1,152 candidates, and no score exceeded Q1. Selection records were sealed before reading the saved value file for this check. Previously known development results mean this is not fresh blind confirmation.

This computation ran on **Zeus, native Windows**, using the existing repository `.venv`: Python 3.10.11 and NumPy 2.2.6. It was CPU arithmetic on saved **ThrusterChain / frozen UniFloral** arrays. The earlier 1,152-transition operator audit ran on Zeus's isolated Ubuntu GPU environment; this check did not invoke WSL, JAX, GPU or the simulator.

New work: one selector, 128 selections, 15 stratum summaries. Resource counts were **zero** new evaluator transitions, full rollouts, model forwards, GPU calls, fits, optimizer/calibration updates, candidate actions and confirmation reads. The four prior controls and their 60 summaries were reused unchanged.

## Result, including harms

| Selector | Mean gain vs Q1 | Mean regret | Switches | Helped / harmed |
|---|---:|---:|---:|---:|
| Q1, prior control | 0 | 1.109267 | 0 | 0 / 0 |
| Twin-min, prior control | +0.029658 | 1.079609 | 25 | 13 / 12 |
| Actor, prior control | -0.157621 | 1.266888 | 54 | 21 / 33 |
| Unrestricted mean target, prior control | +0.292847 | 0.816420 | 59 | 38 / 21 |
| **Lower-only ideal penalty** | **+0.197424** | **0.911843** | **42** | **26 / 16** |

The lower-only score selected mean saved value **-53.830159**, versus Q1's -54.027583; higher is better. It left 86 states unchanged. Harmful switches occurred in **12.5% of all states**, or **38.0952% of switches**. Mean loss among harmful switches was **0.4249942463**; worst loss **0.9728271068**. There were no numerical score ties or inconclusive switches. The gain is not uniform and is not a safety guarantee.

| Stratum | States | Gain vs Q1 | Gain vs actor | Regret | Switches | Helped | Harmed |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 128 | +0.197424 | +0.355045 | 0.911843 | 42 | 26 | 16 |
| development_1 | 64 | +0.213935 | +0.446864 | 0.897665 | 21 | 13 | 8 |
| development_2 | 64 | +0.180912 | +0.263226 | 0.926021 | 21 | 13 | 8 |
| time0 | 32 | +0.032003 | +0.221266 | 0.639798 | 9 | 5 | 4 |
| time25 | 32 | +0.177335 | +0.224941 | 1.281310 | 13 | 9 | 4 |
| time50 | 32 | +0.153775 | +0.073523 | 0.988426 | 9 | 7 | 2 |
| time75 | 32 | +0.426583 | +0.900448 | 0.737840 | 11 | 5 | 6 |
| development_1/time0 | 16 | +0.218929 | +0.434750 | 0.579600 | 3 | 3 | 0 |
| development_1/time25 | 16 | +0.185208 | +0.140458 | 1.536283 | 7 | 4 | 3 |
| development_1/time50 | 16 | +0.207311 | -0.066319 | 0.903223 | 6 | 4 | 2 |
| development_1/time75 | 16 | +0.244294 | +1.278565 | 0.571554 | 5 | 2 | 3 |
| development_2/time0 | 16 | -0.154924 | +0.007783 | 0.699995 | 6 | 2 | 4 |
| development_2/time25 | 16 | +0.169462 | +0.309424 | 1.026336 | 6 | 5 | 1 |
| development_2/time50 | 16 | +0.100238 | +0.213366 | 1.073628 | 3 | 3 | 0 |
| development_2/time75 | 16 | +0.608872 | +0.522331 | 0.904126 | 6 | 3 | 3 |

Development 2 / time 0 loses **0.154924** versus Q1; development 1 / time 50 loses **0.066319** versus the actor. Neither cell was hidden or used to revise the stopping rule. Full precision, all prior controls and per-state results are in [report.json](report.json), [the CSV](all_prespecified_strata.csv) and [STATE_OUTCOMES.json](STATE_OUTCOMES.json).

## Decision: keep the next proposal within BCA

The user's scope clarification was recorded in [SCOPE_AMENDMENT.json](SCOPE_AMENDMENT.json) **before execution**. It supersedes the historical protocol/code wording suggesting a generic positive-correction fitting pilot. The effective decision in [EFFECTIVE_DECISION.json](EFFECTIVE_DECISION.json) is to propose one BCA-consistent target/calibration revision only. The earlier supervised correction proposal is withdrawn; no evaluator-derived candidate labels will be fitted and no new method is relabeled as BCA.

The smallest substantive hypothesis to develop is a **versioned BCA variant with a Q1-centered, one-sided residual bound**. Keep the existing positive local scale architecture, Bayesian-bootstrap-weighted coverage-plus-width fitting, separate held-out Bayesian/conformal calibration, frozen references and live-action-gradient Q penalty. Change the uncertainty event coherently to:

```text
native critic target y: unchanged smoothed twin-target-min TD target
directional residual: d = Q1(s,a) - y
scale-fitting coverage event: y >= Q1(s,a) - u * eta(s,a)
held-out calibration score: max(d, 0) / (u * max(eta(s,a), 1e-6))
deployed radius: U = q_Bayes * u * max(eta(s,a), 1e-6)
actor/candidate score: Q1 - kappa * U
```

This is an uncertainty bound, not an MSE predictor of a signed correction. The local scale eta must be **refitted** to the directional coverage event using logged fitting transitions. Merely changing the global threshold q on the old eta would give `Q1 - (kappa*q)*u*eta`, which is coefficient retuning and adds no action-specific information. The existing two-sided soft-coverage objective multiplies lower and upper sigmoids; a coherent one-sided revision retains the lower-bound event and removes the upper-side factor. Simply clipping residuals while leaving that symmetric objective unchanged would not implement the proposed event.

Preserve the native critic/target updates, reward/terminal/discount semantics, normalization, original BC term, and original BCA as a separately versioned comparator. Preserve the local scale network's form and Bayesian/conformal machinery initially; retarget their uncertainty event rather than simultaneously changing architectures, priors and optimizers. No component has been changed yet.

The current Bayesian mechanism uses bootstrap weights in local-scale fitting and a posterior over the calibration-score tail/miscoverage and its crossing threshold. It is not a posterior over Q-network weights or true continuation V. The proposed posterior would address **one-sided native-target miscoverage**. That interpretation is consistent with treating calibration as uncertainty about risk, as in [Snell and Griffiths](https://arxiv.org/html/2502.13228v2); it supplies no automatic guarantee about RL action selection or OOD performance.

One future bounded pilot should refit the existing local scale on **logged training transitions**, recalibrate on episode-disjoint held-out logged transitions, then evaluate once on both saved development batches. Rewards and next states for fitting/calibration must come from the logs, not these privileged candidate evaluator calls; saved V is evaluation-only. Fix the architecture, fitting budget, seeds and kappa before fitting, with no strength search. Retain original BCA, ordinary-quantile and raw-scale controls. At the same local scale and matched effective coefficient kappa*q, different global thresholds give identical scores; those controls must prevent attributing coefficient changes to Bayesian action information. Bayesian benefit must instead be assessed through its calibration-risk/coverage behavior, alongside direct value gain, regret and harmful switches.

Before claiming coverage, revalidate the new one-sided event, frozen score function, episode dependence/exchangeability (or justified shift weights), fitting/calibration separation, smoothing and terminal laws, adaptive reference reuse, and finite posterior-draw behavior. Old fitted scales, residual units and thresholds cannot be declared calibrated for the new event. A mass of zero directional scores may yield threshold/radius zero; its interpretation and coefficient-matching degeneracy must be specified before fitting. Even valid native-target calibration does not remove `G = y - V`, certify true Q error, or establish coverage for adaptively selected/OOD actions.

This is one proposed BCA adaptation, not a validated fix or approval for full training. The ideal lower-only result establishes feasibility on the saved bank; learning the local bound from logged data and retaining useful decisions remains untested. No demand is made that every offline correlation be positive before a separately authorized bounded policy pilot.

## Verification and reproducibility

Four new synthetic mapping fixtures passed; the existing verified selection/tolerance helpers were reused without rerunning their scientific entry point. Independent verification passed **3,817 numeric comparisons**, all 1,152 mapping rows, 128 new selections, 15 strata, 60 unchanged prior control rows, both screens, hashes/seals and scope-amendment chronology. See [INDEPENDENT_REVIEW.json](INDEPENDENT_REVIEW.json).

Protocol SHA-256: `2299b78c55b71cb2e4fabd9ffaebddace06dfbfddc68ba7a85470856638ba5ea`.

Result SHA-256: `1f9a03b56114f7527be43bb10feeddfd7f82e0abfda2a26b49951b5b90222b16`.

Independent review SHA-256: `08becb03b426fd03daf608f9c2b08eea0f0768228f50cfd3c7afba60ce3e6375`.

Verify saved evidence from the repository root:

```powershell
.\.venv\Scripts\python.exe -B experiments/thrusterchain_bca/zeus_20261026_validation/operator_analysis/cached_lower_only_v1/verify_lower_only.py
```

Do not rerun the reserved scientific comparison or delete its attempt evidence. Original results are preserved. No source/model changes, commits, pushes or additional exploratory runs occurred. There is no remaining verification blocker or uncertain execution; full training remains paused.
