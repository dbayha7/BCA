# Descriptive fresh-bank replication

This source-only implementation repeats the frozen-model question on32 newly sampled states. It does not retrain, tune or claim independent training replication. The checkpoint is the same full/BCA replicate01 at100k, with the same95k calibration reference, Q1 score, kappa1 and fixed distance coefficient **0.519506430786272**. The actual native width is used at full produced precision, including its floors and refusal behavior. No fresh labels, preparation or tests have run while writing this folder.

`protocol.json` fixes the new seed2026102123, four contexts with8 states each, time counts3/3/2 at0/50/99, and12 indexed candidates. Candidate construction is identical to the authenticated v2 implementation; duplicates across grid and actor slots stay in the data. Within each state a fixed independent permutation shuffles width and its usability flag. Exact float32 grid probabilities are retained; grid support is not joint state or calibration support.

The primary alignment statistic is the equal-state mean Pearson correlation between U and the **signed** error Q1−Qpi on U's own defined states. Distance and shuffled-width means use their own defined states. Each paired contrast uses its pair-specific common intersection: U with distance, or U with shuffled width. Paired arm means and denominators are also retained. A zero centered predictor or signed-error norm gives an undefined value with its reason, not zero. With m indexed actions, `sum_{i<j}(x_i-x_j)(y_i-y_j)=m*sum_i(x_i-mean(x))(y_i-mean(y))`. Consequently centered Pearson equals the normalized dot product of all pair differences. Those pair differences are not re-centered. All12 indices, including duplicates, enter Pearson.

The primary decision statistic is mean concordance change versus rawQ. Each arm uses the same pairs whose true-Q difference exceeds1e-6 in absolute value. Score differences within1e-6 receive half credit, and an all-true-tie state is undefined. Decisions use first exact argmax in fixed index order. Secondary outputs retain selected Qpi, changes versus rawQ and the original actor, and restorations of the actor's actual command. Index changes and command changes are counted separately. The original actor is a selected-return control, not a full-ranking arm.

The five ranking scores are rawQ1, Q1−U, Q1−fixed_coefficient*D, Q1−shuffledU, and a state-constant offset from Q1. The constant-offset control must preserve the full raw ranking. Finite native-unusable widths use raw-score fallback. Separately, any nonfinite input, negative width, nonfinite score or missing/nonfinite truth invalidates the entire state across paired comparisons. Every state/candidate record remains, with reasons and denominator; invalid states are never replaced. Prelabel-invalid states do not consume oracle transitions.

Preparation saves one10,000×32 bootstrap plan, drawing whole states within the original context×time cells with their fixed3/3/2 counts. Every endpoint uses this same plan. Its defined-state mask is applied inside each draw; null states are never secretly redrawn. Every replicate's defined denominator is saved. If any replicate is undefined, its interval is unavailable. There is no minimum-sample or significance gate, no p-value, and no winner rule. These intervals are descriptive conditional uncertainty for one trained model and32 states, with only2–3 states per stratum.

The oracle remains one candidate action followed by the unchanged actor toH100 with gamma1. Every native actor inference processes exactly one float32 observation, as already validated in v2. The actual float64 plant is unchanged. The runner reuses the authenticated v2 fixture receipt (608 prior validation transitions; maximum batch/scalar discrepancy8.53e-14) and adds **zero** new validation transitions. Planned fresh work is384 continuations /21,696 plant transitions. Frozen parameters and source identities are checked before and after execution.

The code imports v2 only after authenticating its frozen source, preparation identity, coefficient receipt, passed replay receipt and unchanged parameters. `replication.py` owns fresh inputs and pure statistical functions. Its native width readout follows v2's `read_width_and_q` arithmetic and calls the actual authenticated `calibration.dose.frozen_level_dose`, while preserving invalid critic outputs for explicit state-level handling. The v2 `build_candidates`, `direct_policy` and `truth_rollouts` implement candidate construction, canonical actor deployment and the rollout kernel. No historical folder is modified.

After source review, run from this folder with the BCA virtualenv:

```powershell
& 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' -m unittest discover -s tests -v
& 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' .\run_replication.py prepare --output .\prepared
```

Inspect the outcome-free input artifacts. Root then writes a separate `freeze.json` containing `approved_for_oracle_execution: true`, `preparation_manifest_sha256`, and the exact `source_sha256` dictionary from the manifest. The program never approves itself.

```powershell
& 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' .\run_replication.py run --prepared .\prepared --freeze .\freeze.json --output .\results
```

Both output directories must be new children of this folder. The run saves raw NPZ data,384 candidate records,32 state records, paired results, metric summaries, and bootstrap replicate means/denominators. Failed or invalid outputs remain visible. All six synthetic test cases use only fixed arrays and pure functions, with no model, simulator or fresh-label calls.

The historical width concerns one-step noisy-target twin-min residuals; the new question concerns Q1 and a deterministic actor continuation. Neither finite native width nor improved alignment would establish Q coverage, policy improvement from repeated intervention, or general benefit across training seeds.

Prelabel review corrections: each rho mean uses its own defined mask and each paired rho contrast its own common mask; explicit null-correlation reasons and no-comparable-pair reasons are saved; completed/finite/expected-step flags guard every oracle state even if an accumulated return is finite; physical-position failure alone is retained. Direct paired selected-Qpi contrasts of live width versus distance and shuffled width use the same bootstrap plan. These corrections occurred before any fresh preparation or labels.
