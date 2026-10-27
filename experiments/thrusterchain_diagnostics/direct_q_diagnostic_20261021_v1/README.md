# Frozen Q ranking diagnostic: source awaiting review

This folder prepares a bounded diagnostic of the already trained `replicate_01/full/bca` terminal100k checkpoint. No training, preparation, tests, or oracle rollout was executed while writing these files. Execution requires a separate review and freeze. Canonical training artifacts remain unchanged.

The question is whether subtracting the existing frozen95k width from **Q1** helps choose among12 fixed actions at each of32 new evaluator-only states. The oracle takes that candidate action once and follows the unchanged actor through time99. It estimates the finite-horizon value of a one-action deviation, not the return of a new policy that repeatedly maximizes the score. There is no actor optimization or Qstar approximation.

`protocol.json` fixes PCG64 streams,32 states,8 per context, the times0/50/99 with3/3/2 counts per context,12 candidates, coefficient matching, numerical tolerances, tie handling and inference before labels. All ten physical position/velocity coordinates are fresh uniform draws from[-.5,.5]; current is.3/1.1, wave amplitude.2 and phase uniform. These are designed states at the declared times, not policy occupancy samples. Eight separate adaptation states use times0/50 in each context and never receive oracle labels.

Six candidates are distinct grid atoms including the nearest actor command and behavior mode (one slot if identical), followed by the original continuous actor action and five uniform continuous commands. Duplicate commands between grid and continuous slots are preserved. Support is classified from actual float32 coordinate equality to the categorical atoms. The exact product probability is the product of.84 at each mode coordinate and.04 at every other atom. Common means beta>=.01, rare means0<beta<.01 and unsupported means beta0. A continuous slot may happen to be on-grid. This conditional action support does not establish state or calibration support.

The frozen reference has miscoverage.1, credibility.95 and128 pooled historical calibration rows, last refreshed at95k. `U=R*unit*max(eta,1e-6)` is checked at every candidate against the actual native `frozen_level_dose(reference,eta,.5).width`; its native usability and any refusal are recorded. Q1-U uses candidate-dependent eta. An unusable/nonfinite width falls back to rawQ1 with a flag; this proposed ranking fallback is not claimed to reproduce the native actor BC intervention. The constant-distance comparator matches the pooled RMS of within-state centered widths on adaptation rows to that of `D=mean((a-behavior_mode)^2)` before any returns. If matching is unavailable it is refused, not retuned. State-constant width and detached-width controls preserve their expected forward rankings.

The width was calibrated to one-step residuals involving the current twin-min critic and noisy target networks, whereas the proposed score is Q1 and the deterministic oracle follows the unsmoothed terminal actor. The historical adaptive calibration bank and new designed state/time/action law differ. Neither numerical width usability nor this diagnostic establishes Q coverage, state support, or a long-return guarantee.

The main oracle budget is384 continuations and21,696 plant transitions:4*(3*100+3*50+2*1)*12. Four deterministic validation fixtures add608 transitions. The authenticated batch and scalar paths must agree with atol1e-4, rtol1e-6; terminal99 must equal immediate reward within1e-10. Actual discrepancies are saved and a failed check stops the run; there is no outcome-driven tolerance adjustment. Actor-action Q=V is checked on fixtures with those deployment tolerances. Float32 batch kernels can differ by roundoff, so no bitwise or exact-population value claim is made.

Exact score argmax takes the first maximum in fixed candidate order. Pairwise ranking uses absolute near-ties<=1e-6: true near-ties are excluded, predicted near-ties earn half credit. Counts and denominators are retained. Error, positive overestimation, width/error Spearman, rankings, selected true-Q advantage versus the original actor, changes versus rawQ, harmful changes and opportunity loss are saved for every candidate/state and subgroup. Within-action-class comparisons rerank within that class and require two eligible candidates. A class with too few candidates is explicitly unavailable.

Confidence intervals use10,000 paired state bootstrap draws within context×time strata. The full bank preserves the fixed3/3/2 time mixture; support subgroups preserve their observed eligible stratum counts, without inventing missing strata. The inference unit is the state, never an action. Any nonempty eligible stratum with fewer than two states refuses its interval. Even the full bank has only2–3 states per stratum and one trained checkpoint. Intervals describe this conditional designed law and fixed candidate realization; they do not measure variation across independently trained models or Monte Carlo oracle uncertainty. The matched-distance control matches score dispersion, not gradient strength.

## Source map

The external engine is authenticated against `../action_only_20261021/freeze.json` **before import**. `FrozenScoreEngine.load` then authenticates the preserved bundle source closure and checkpoint. Its internal legacy64-row selection is not used for this diagnostic's fresh states or actions.

| Diagnostic operation | Preserved source or actual entry point |
| --- | --- |
| Restore final actor, critics, reference, normalizer | `../action_only_20261021/engine.py`: `FrozenScoreEngine.load`; closure `experiments/thrusterchain_diagnostics/training.py`: `TrainingResult.policy`, `TrainingResult.critic` |
| Behavior mode and categorical atoms | Closure `experiments/thrusterchain_diagnostics/training_data.py`: `behavior_parameters`, `ACTION_ATOMS` |
| Q1 and candidate eta | `diagnostic.py`: `read_width_and_q`, current critic head0 and frozen `posterior.cal_params` |
| Native width and refusal | Closure `calibration/dose.py`: `frozen_level_dose`, `level_critic_dose` |
| Main forced-action rollout and failure metadata | Closure `experiments/thrusterchain_diagnostics/training_eval.py`: `_simulate`, the kernel used by `batch_policy_q` |
| Independent scalar replay fixtures | Closure `experiments/thrusterchain_diagnostics/continuous.py`: `policy_q`, `policy_value`, `dynamics` |
| Inputs, coefficient, native readouts and outcome-free receipt | `run_diagnostic.py`: `prepare` |
| Reviewed receipts, oracle, complete records and immutable output | `run_diagnostic.py`: `authenticate_preparation`, `run` |

## Execution after review only

Use the existing BCA virtualenv Python. Both commands require a previously nonexistent output directory inside this folder. Neither resumes or overwrites partial output.

```powershell
& 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' .\run_diagnostic.py prepare --output .\prepared
```

Preparation loads frozen models but computes no return labels. It saves adaptation/evaluation NPZ inputs, all scorer readouts, coefficient receipt, actual engine identity, version/runtime information, all parameter hashes before/after and `preparation_manifest.json`. Review those artifacts and create a separate `freeze.json` with `approved_for_oracle_execution: true`, `preparation_manifest_sha256` and the exact `source_sha256` object from that manifest. This file is an operator review receipt; the code never approves itself.

```powershell
& 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' .\run_diagnostic.py run --prepared .\prepared --freeze .\freeze.json --output .\results
```

The run authenticates frozen code and preparation bytes before loading the model and records a receipt before any fixture or main return. It saves `raw_results.npz`,384 `candidate_records.csv` rows, `paired_results.json`, `paired_decisions.csv`, `within_state_correlations.csv`, `error_summaries.csv`, fixture evidence and final status/hashes. Physical-position failure events and incomplete/nonfinite returns are kept. Any incomplete/nonfinite return refuses aggregate inference instead of silently dropping a row. Frozen parameters are hashed after success or failure. Failed directories remain as evidence and cannot be resumed ambiguously.

`tests/test_diagnostic.py` contains pure input/arithmetic tests and separately gated authenticated replay fixtures. It has been written without execution. The runtime runner already performs the608-transition replay fixtures; running the optional model tests separately adds another608 transitions and must be accounted for as additional validation work.
