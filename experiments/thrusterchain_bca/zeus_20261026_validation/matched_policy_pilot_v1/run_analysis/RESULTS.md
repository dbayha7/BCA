# Completed matched policy pilot

The approved pilot completed successfully and passed its predeclared exploratory utility screen. Original direct-Q BCA at kappa 3 beat continued native UniFloral TD3+BC and the fixed actor-distance control in both continuation streams. This is evidence about learned policies controlling entire episodes, beyond the earlier frozen candidate-selection diagnostics.

The reservation records 2026-10-27 00:28:11 UTC (October 26, 8:28:11 p.m. EDT). The terminal result was saved at approximately 00:42:56 UTC (8:42:56 p.m. EDT). Monotonic total elapsed time was 884.793032605 seconds, or 14 minutes 44.8 seconds, within the 4,200-second limit. All seven workers exited 0, with no timeout, interruption, failed policy, or uncommitted update. The top-level PROGRESS.json is a stale running snapshot; RUN_RESULT.json and all terminal worker reports establish completion. No rerun is needed or authorized by this result.

## Results

Each final actor controlled all 100 actions of each of 32 shared, fresh nominal reset episodes. The primary score is the float64 discounted return with gamma 0.98 and no terminal bootstrap. Higher (less negative) is better. These are task return units, not normalized benchmark percentages.

| Policy | Stream 0 mean return | Stream 1 mean return |
|---|---:|---:|
| Unchanged original 20,000-update checkpoint | -74.231089 | -74.231089 |
| Native continuation, no Q penalty | -69.321135 | -69.830265 |
| Original BCA, kappa 3 | **-66.731700** | **-67.319602** |
| Actor-distance control, coefficient 0.1 | -69.399031 | -70.469137 |

| BCA comparison | Stream 0 mean paired gain [95% episode interval] | Stream 1 mean paired gain [95% episode interval] |
|---|---:|---:|
| Versus native continuation | +2.589436 [2.401441, 2.808203] | +2.510663 [2.314457, 2.708318] |
| Versus distance control | +2.667332 [2.477394, 2.865936] | +3.149535 [2.809563, 3.474955] |
| Versus original checkpoint | +7.499390 [6.003051, 9.079922] | +6.911487 [5.482590, 8.491714] |

BCA improved all 32 paired episodes against each of these controls in each stream, with zero harms or numerical ties under the frozen tolerance. The prespecified screen required positive mean gains against native and distance separately in each stream; all four comparisons passed. These are the same 32 reset states in both streams, not 64 independent starts. The intervals are the frozen 2,000 paired episode bootstrap calculation, conditional on the fixed final actors; they do not quantify independent training-initialization uncertainty.

## Execution and method identity

- Six runs, each with exactly 2,000 additional joint twin-critic optimizer updates, 1,000 actor updates, 1,000 updates of each target network, 2,000 original auxiliary updates, and one accepted halfway calibration. Totals: 12,000 critic updates, 6,000 actor updates, 12,000 auxiliary updates, six recalibrations (6,000 posterior draws). All three arms performed the matched auxiliary work.
- Training used the original logged training data and original calibration episode role. No evaluator-return or candidate-transition labels were fitted. There were zero training-time simulator transitions and zero intermediate checkpoint selection reads.
- Evaluation completed 224 episode trajectories (7 policies x 32 shared starts), exactly 22,400 simulator transitions, 32 shared reset calls, and 896 logged probe action rows. Evaluation performed zero optimizer or calibration updates.
- Exact Titan-transferred standalone UniFloral source revision `f2dc1278eae18ed3e22c92255119c54885414d17`; native source SHA-256 `6f5ab6f98d706d902a18f11e1c81e99e2691dc4c8a53843bf6521c9f95d00eee`; original complete checkpoint SHA-256 `874d65b7f4dcb5ec285a0d4360439c73a787b5e8c567ff7c0138e0c1e22948e8`. No CORL source or model was substituted. Source/checkpoint files remain unchanged.
- The original actor, twin critics, optimizer states, targets, normalization and BC term were restored exactly. The BCA arm retained the original positive local-scale architecture, residual objective and Bayesian/conformal calibration; the failed one-sided revision was excluded. The separate continuation adapter bound the saved reference and performed the approved refresh at absolute critic update 21,000.
- The penalty remains inside the actor's Q objective: Q1 minus kappa times width, with the action gradient live. Native critic predictions/targets were not directly penalized. The actor-distance width is `19.29619572569095 * mean((a_live - a_original_20k)^2)` across the four action coordinates, at coefficient 0.1. Kappa is strength, not a head count; native twin critics and the one scale output were unchanged.
- All runs used the isolated Ubuntu Python environment on the RTX 5070 Ti Laptop GPU. The previously disclosed JAX/NumPy versions outside upstream package constraints were retained, as on Titan. No package changes were made.

## What this establishes and what remains open

The trained BCA policies improved nominal-task returns relative to both specified controls under equal initial states, update budgets and paired replay/noise seeds. Continuing native learning also improved the original checkpoint, which is why native continuation is the relevant primary baseline.

This does not establish full-training, independent-initialization, OOD, or selected-action calibration guarantees. Both streams start from one trained checkpoint; calibration episodes were reused; strengths were development-selected. The previous locked confirmation result remains inconclusive and was not reused for selection here.

The distance control uses its fixed approved coefficient; penalty values or gradient magnitudes were not strength-matched to BCA. At the final actor minibatch, the scalar actor-penalty terms were approximately 0.552/0.525 for BCA and 0.000452/0.000407 for distance. These levels are not gradient norms and cannot alone explain the outcome. This pilot compares the specified methods; it does not isolate the value of Bayesian calibration from ordinary quantiles or generic regularization.

Descriptive displacement also does not imply OOD validity. Relative to the original actor, logged-probe action MSE was 0.006519/0.006095 for BCA, 0.006123/0.005677 for native, and 0.005806/0.005251 for distance. Corresponding standardized physical-trajectory displacement was 0.080730/0.072727, 0.079410/0.067889, and 0.076208/0.064598.

Recommended next decision: retain this original BCA kappa-3 candidate and predeclare a bounded replication across independent trained initializations with fresh evaluation starts before a larger benchmark. Do not retune this successful pilot, promote the failed one-sided variant, or start the deferred head-count sweep. No further execution is authorized or started by this report; full training remains on hold.

## Verification and reproduction of this analysis

Read-only post-run verification passed 105 hash/accounting checks and 179 independent cached arithmetic checks. The latter recomputed all episode returns from saved rewards, shared initial states, paired differences, harms/ties, bootstrap intervals and the fixed screen. No model forward pass, fitting, calibration, simulation, confirmation access or WSL call occurred during this review.

- Frozen protocol SHA-256: `df37c490612e4d489a27a083305284bd4a244708735e482f0164629936465908`
- RUN_RESULT.json SHA-256: `57705c272f5b3affeed5691cfa27f671a2854b1283cebd057a7b54ab3dd6913b`
- Evaluation report SHA-256: `7ff19d75806cd96edde90966f2edc5efaee427ff751845c9236f4981db9be6ae`
- Evidence: `INTEGRITY_REVIEW.json` and `INDEPENDENT_ARITHMETIC_REVIEW.json` in this folder; the original terminal reports and sealed actor/state exports remain under `../run/`.

From the repository root, these Windows commands repeat only saved-file verification:

```powershell
.\.venv\Scripts\python.exe -B experiments/thrusterchain_bca/zeus_20261026_validation/matched_policy_pilot_v1/run_analysis/verify_integrity.py
.\.venv\Scripts\python.exe -B experiments/thrusterchain_bca/zeus_20261026_validation/matched_policy_pilot_v1/run_analysis/verify_saved_arithmetic.py
```

Assistant-side WSL access remains denied and was not retried; the user completed the actual run manually. Uploaded attachment access was denied, so this analysis used independently authorized repository artifacts. Neither prevents review of the completed run. Results have not been committed or pushed.
