# Controlled action-only calibration evaluation

Completed **200/200** registered independent calibration replications. Missing: []; terminal failed: []; historical failure records: [].

Uniform calibration already exceeds the nominal 90% residual-coverage target at all three mixture levels. Exact weighting increases total coverage under action shift, but much of that increase comes with infinite-band refusal and lower finite-and-covered coverage. This is a coverage/usefulness tradeoff, not evidence of policy benefit.

The model, physical state/time/context law, target networks and label-noise law are fixed. Only the current conditional action law changes. Both methods see the same fresh calibration rows and the same evaluation rows. Importance weighting uses exact categorical target/source probabilities.

## Coverage is not enough: keep infinite bands visible

| Mixture λ | Procedure | Coverage including infinity (95% interval) | Finite and covered / all | Refusal / fallback | Finite conditional coverage |
|---|---|---|---|---|---|
| 0.0 | Uniform calibration | 94.11% [93.79%, 94.43%] | 94.11% [93.79%, 94.43%] | 0.00% [0.00%, 0.00%] | 94.11% [93.79%, 94.43%] |
| 0.0 | Exact importance weighting | 94.11% [93.79%, 94.43%] | 94.11% [93.79%, 94.43%] | 0.00% [0.00%, 0.00%] | 94.11% [93.79%, 94.43%] |
| 0.5 | Uniform calibration | 93.48% [93.19%, 93.77%] | 93.48% [93.19%, 93.77%] | 0.00% [0.00%, 0.00%] | 93.48% [93.19%, 93.77%] |
| 0.5 | Exact importance weighting | 95.48% [95.12%, 95.83%] | 78.82% [78.43%, 79.20%] | 16.66% [16.49%, 16.83%] | 94.57% [94.15%, 95.00%] |
| 1.0 | Uniform calibration | 92.84% [92.57%, 93.11%] | 92.84% [92.57%, 93.11%] | 0.00% [0.00%, 0.00%] | 92.84% [92.57%, 93.11%] |
| 1.0 | Exact importance weighting | 97.61% [97.27%, 97.95%] | 64.79% [64.43%, 65.15%] | 32.82% [32.72%, 32.93%] | 96.44% [95.94%, 96.95%] |

Nominal residual coverage is 90%. An infinite band covers every residual mathematically but provides no finite uncertainty bound. “Finite and covered / all” counts only useful covered queries and keeps refused queries in the denominator. Finite conditional coverage conditions on accepted queries and can describe a selected subset.

## Paired weighting contrasts

| Mixture λ | Δ coverage including infinity | Δ finite-and-covered fraction | Δ refusal | Δ multiplier | Jointly finite width difference |
|---|---|---|---|---|---|
| 0.0 | +0.00 pp [+0.00, +0.00] pp | +0.00 pp [+0.00, +0.00] pp | +0.00 pp [+0.00, +0.00] pp | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 0.5 | +1.99 pp [+1.71, +2.27] pp | -14.67 pp [-14.98, -14.35] pp | +16.66 pp [+16.49, +16.83] pp | -0.0538 [-0.0563, -0.0513] | 0.2646 [0.1575, 0.3718] |
| 1.0 | +4.77 pp [+4.43, +5.11] pp | -28.05 pp [-28.41, -27.69] pp | +32.82 pp [+32.72, +32.93] pp | -0.0997 [-0.1027, -0.0968] | 0.8764 [0.7349, 1.0178] |

Differences are exact importance weighting minus uniform on the same queries. Percentage quantities above are differences in percentage points. Width contrasts condition on queries where both bands are finite; they are not unconditional expected widths. Separate method means of finite width and finite conditional coverage use method-specific accepted subsets; only the jointly-finite contrast compares the same subset.

## Calibration mass and refusal causes

| Mixture λ | Mean effective sample size [95% interval] | Positive-weight rows, mean | Banks without target-action hits | Banks with all weights zero |
|---|---|---|---|---|
| 0.0 | 128.0000 [128.0000, 128.0000] | 128.00 | 0/200 | 0/200 |
| 0.5 | 57.3010 [52.2386, 62.3634] | 128.00 | 0/200 | 0/200 |
| 1.0 | 24.0031 [21.5016, 26.5046] | 43.77 | 0/200 | 0/200 |

At λ=1 only the projected actor command has positive target mass at each state. Observed no-hit/all-zero banks: 0/200. All observed weighted refusals in this completed experiment are posterior-infinite thresholds from nonempty positive-weight banks. The separately tested all-zero-bank refusal remains a negative fixture, not an observed main-run failure. A nonempty bank can still produce an infinite posterior threshold because query mass is too large relative to calibration mass.

## Finite widths and multiplier readout

| Mixture λ | Procedure | Finite half-width mean [95% interval] | Replications with a defined width | Mean multiplier [95% interval] |
|---|---|---|---|---|
| 0.0 | Uniform calibration | 3.1899 [3.1120, 3.2679] | 200/200 (0 undefined; 0 missing) | 1.3336 [1.3310, 1.3361] |
| 0.0 | Exact importance weighting | 3.1899 [3.1120, 3.2679] | 200/200 (0 undefined; 0 missing) | 1.3336 [1.3310, 1.3361] |
| 0.5 | Uniform calibration | 3.1697 [3.0923, 3.2471] | 200/200 (0 undefined; 0 missing) | 1.3325 [1.3300, 1.3351] |
| 0.5 | Exact importance weighting | 3.3416 [3.2037, 3.4794] | 200/200 (0 undefined; 0 missing) | 1.2787 [1.2753, 1.2821] |
| 1.0 | Uniform calibration | 3.1492 [3.0724, 3.2261] | 200/200 (0 undefined; 0 missing) | 1.3315 [1.3289, 1.3340] |
| 1.0 | Exact importance weighting | 3.7888 [3.6235, 3.9542] | 200/200 (0 undefined; 0 missing) | 1.2317 [1.2283, 1.2351] |

The deployed continuous actor is off-grid at **64/64** fixed states. Its actions are structurally unsupported by this discrete source law. The supported grid projection is a different target; rounding cannot certify the continuous actor.

## Scope and interpretation

- The replication unit is the independently drawn calibration bank and its paired evaluation random streams. The 4,096 queries within each replication are not treated as 4,096 independent replications of the calibration procedure.
- Intervals are unadjusted two-sided Student-t intervals across independent calibration replications, conditional on this one trained checkpoint and fixed finite state bank. Subgroup intervals are descriptive; no simultaneous or per-state certificate is claimed.
- Undefined finite-band metrics remain undefined and their counts are shown. All refusals remain in overall coverage, useful-coverage and fallback denominators.
- This is the BCA-specific Bayesian threshold procedure with an exact per-query importance-mass adapter. It is not a weighted-conformal reference and no borrowed frequentist coverage guarantee is asserted.
- Scores measure a one-step Bellman-target residual, not error against true long-horizon policy value. The original trained 95k scale is frozen while both calibration and evaluation residuals use the same 100k predictor and target.
- The multiplier is a diagnostic readout only. No actor update, policy improvement, state-visitation robustness, critical-region certification or general conclusion about every IWBCA version follows.
- The finite-state law contains already observed source episodes and the model choice was availability-based after the original training study. This is a frozen post-hoc control, not retroactive preregistration or new-model generalization.

## Reproducibility and evidence

- Checkpoint SHA256: `4088278230fc413c509e27e189daa342468cfc4fb6d2479e3e8c29e149dac3d8`.
- Protocol freeze SHA256: `1fc66520f2715c1e5c2396c1fcc7db9279bfecb20dedf61d58227fd575d7a18f`.
- State-bank SHA256: `2001f5444bb004063dce61e8f05d65defac70ecd2a05ac43385cb0cacc0bb63e`.
- `paired_replicates.csv` retains each mixture and paired method result with bank/query identities, numerator/denominator counts, width, usability, fallback, effective sample size, no-hit and support-audit identity.
- `method_replicates.csv`, `subgroup_replicates.csv` and `paired_records.json` retain additional detail; `aggregate.json` records all undefined/missing counts and source hashes.
- Local `results/replicate_*.npz` files retain draw-level evidence. Analysis does not run calibration or inference again.

Command: `python analysis/aggregate_results.py` (requires SciPy for Student-t quantiles).
