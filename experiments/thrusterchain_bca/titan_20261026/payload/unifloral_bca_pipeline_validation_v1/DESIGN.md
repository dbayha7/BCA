# Stage two: actual BCA pipeline on a frozen UniFloral snapshot

This is a deterministic composition diagnostic, not a production adapter or an efficacy study. The full pilot remains paused. This stage performs zero primary critic updates, zero actor updates and zero rollouts. It uses at most four auxiliary optimizer attempts and three reference refresh calls. The prior 19-check parity milestone remains preserved in the sibling stage-one folder.

## What is reused

The native base is UniFloral standalone TD3+BC at revision `f2dc1278eae18ed3e22c92255119c54885414d17`, with source SHA-256 `6f5ab6f98d706d902a18f11e1c81e99e2691dc4c8a53843bf6521c9f95d00eee`. The frozen stage-one loader selects unchanged computational definitions from its authenticated 40-file snapshot. The frozen stage-one candidate contributes only its actor-loss function here. The native two-critic-then-actor schedule, Adam epsilon `1e-5`, observation normalization, reductions and denominator epsilon `1e-7` are preserved in those files. This stage does not call their optimizer update loop.

The uncertainty source is the actual calibrated-width code distributed in the Titan portable bundle. Exact copies and source hashes are recorded in `uncertainty_provenance.json`. Only uncertainty definitions are selected from the earlier TD3+BC BCA file; no CORL initialization, actor, critic, optimizer or update routine is imported. The real Calibrator, Bayesian bootstrap, scale fitting, held-out refresh, FrozenReference and live_width implementations are reused without changing their mathematical bodies.

## Explicit bridge for this fixture

UniFloral's `dual_q` and `dual_q_target` are exposed to the auxiliary routines under their expected `critic` names through a read-only state view. Actor application returns the native deterministic distribution's `.sample(seed=None)`. Auxiliary `discount` is bound to native `gamma=0.99`. The task-specific gamma 0.98 and horizon 100 are not introduced into this native parity fixture.

The calibrator receives raw observations and the fitting-bank observation mean and standard deviation, using its existing `(obs - mean)/(std + 0.001)` transform. This explicitly chosen fixture binding matches native UniFloral input normalization. It does not reproduce the earlier portable adapter's physical-only preprocessing followed by an extra `/1.001`. It is not a silently adopted production preprocessing decision.

The auxiliary target retains the existing BCA vectorized noise generation and fold-in tags. Its keys come from a separate recorded auxiliary stream. Native sampling/noise RNG is never consumed or replaced. All native parameters, optimizer states, targets and counters must remain identical throughout this stage.

Refresh is invoked manually at predetermined points for testing. This is not the production warmup or refresh schedule. The earlier portable pipeline fitted its auxiliary network before each primary critic update and refreshed after 10,000 updates, then every 5,000; the actor used the previously frozen reference. Production ordering and the core's minibatch ownership remain unresolved integration work. A production core adapter must not discard alternate core batches to mimic the verified native two-critic reuse schedule.

## Actor objective and scales

The candidate uses

`L_actor = -c * mean(Q1) + mean(BC) + c * kappa_penalty * mean(U)`

where `c = stop_gradient(td3_alpha / (mean(abs(Q1)) + 1e-7))` is logged as `q_scale_coefficient`. It is computed from unpenalized Q1. The external grid is `kappa_penalty = {0, 0.03, 0.1, 0.3, 1, 3, 10}`. No grid point is selected as best. Reference parameters, threshold and residual unit are frozen; live actions retain their gradient.

`U = threshold * max(eta(s,a), 1e-6) * residual_scale` for a numerically usable reference. The unchanged width net has two 64-unit ReLU layers and returns `eta = softplus(log_eta) + 0.01`. The width net output and calibrated threshold are dimensionless; the residual scale carries reward/Q units. At fixed parameters and states, positive reward/Q/residual-unit rescaling should co-scale U. This is an algebraic diagnostic, not a claim that Adam training trajectories or fitting from the default unit 1 are scale invariant. Native denominator epsilon `1e-7`, the residual standard-deviation floor `1e-6`, initialization at unit 1, and finite precision break exact global scale invariance. The suite records exact epsilon-aware predictions and gradient/penalty ratios, rather than recommending Q normalization from synthetic results.

## Two scientific decisions are not made by a passing test

1. **Residual proxy versus Q1 error.** The actual fitter and held-out refresh use a twin-min one-step TD residual. The actor penalty acts on Q1. A finite calibrated residual width is not a Q1 value-error guarantee and does not establish a causal decision benefit. A controlled head-disagreement example exposes this mismatch. The diagnostic retains the existing residual definition; substituting Q1 residuals would be a new scientific method.
2. **Invalid ready thresholds.** Legacy `reference_valid` permits a ready reference with positive infinity as its threshold, and `live_width` returns zero with `usable=False`. This is an observed fail-open fallback, not a safety property. A separate proposed rejection predicate is tested diagnostically. It is not installed into the production candidate, and does not authorize silently skipping the penalty. Intentional unready warmup is reported separately. Review of the production policy is pending.

Even if all checks pass, scientific validation remains `incomplete_design_review_required`. Operational calibration-bank exchangeability, adaptive reference reuse, true value error, out-of-distribution behavior, policy ranking, return improvement, production replay scheduling and multi-seed efficacy are not established here.

## Runtime and evidence boundaries

The user's verified environment is JAX/jaxlib 0.6.2, NumPy 2.5.4, Flax 0.10.7, Optax 0.2.8, Distrax 0.1.5, Chex 0.1.90 and TensorFlow Probability 0.25.0. JAX and NumPy are outside the pinned upstream upper bounds; the command requires explicit provisional compatibility acknowledgement. No packages are installed or changed. Highest matmul precision and the native XLA flag are set in a fresh subprocess before imports and asserted at runtime. Stage-one hashes and the native snapshot, plus every stage-two prepared file, are verified before and after the run.

Preparation uses Windows stdlib parsing and hashing only. The scientific suite has not been executed by the assistant. WSL and process-query denials remain binding; only the user-run command can execute the bounded GPU check.
