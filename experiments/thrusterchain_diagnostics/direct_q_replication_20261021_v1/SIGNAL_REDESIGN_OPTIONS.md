# What eta learns, and two proposed signal variants

**Eta is fitted to aggregate soft coverage of a changing one-step TD target. It is not pointwise regression to absolute TD error, an established conditional quantile, or uncertainty about true policy Q.** Option A is the smallest proposed next test. Option B targets propagated value error but needs an additional fitted recursion and stronger offline assumptions. Neither is implemented or authorized here as a new training run.

This account uses the authenticated preserved source and saved engine identity for full/BCA replicate 01. No model, simulator, training or fresh-label calls were made. The original text is preserved byte-for-byte in `SIGNAL_REDESIGN_OPTIONS.encoding_draft.md`; this rewrite repairs encoding and readability only. The eight linked source files are bundled in `audit_source_snapshot/`, with hashes checked against the engine identity.

## Exact implemented signal

For a logged transition `(x, a, r, x_next, m)`, x contains 13 physical/environment coordinates, observed regime and remaining-time fraction. The bootstrap mask m is 0 at the inherent H = 100 terminal boundary and 1 otherwise. The adapter supplies `done = 1 - m`, raw rewards and actions in [-1, 1]. It normalizes the first 13 observation coordinates with training-only mean and `std + 0.001`, leaving regime/time unchanged at this layer. The calibrator then divides every already-normalized observation coordinate by 1.001 because its own mean/std are 0/1, and concatenates the 4 action coordinates. Two 64-unit ReLU layers and a scalar head produce `eta = softplus(log_eta) + 0.01`. Inputs exclude reward, successor, current Q and residual. [Adapter, lines 219-282](audit_source_snapshot/experiments/thrusterchain_diagnostics/training.py); [initialization, lines 129-141](audit_source_snapshot/algorithms/td3_bc_bca.py); [network, lines 8-37](audit_source_snapshot/calibration/network.py).

At each accepted training step, the scale fit uses the pre-update native networks:

```text
epsilon ~ Normal(0, 0.2**2 * I), clipped coordinatewise to [-0.5, 0.5]
a_next = clip(pi_target(x_next) + epsilon, -1, 1)
y = stopgrad[r + m * min(Q1_target, Q2_target)(x_next, a_next)]  # gamma = 1
q = stopgrad[min(Q1, Q2)(x, a)]
d = (y - q) / u
w_i = Exp(1)_i / sum_j Exp(1)_j
c_i = sigmoid(20 * (d_i + eta_i)) * sigmoid(20 * (eta_i - d_i))
L_eta = (sum_i w_i * c_i - 0.9)**2 + 0.005 * sum_i w_i * eta_i**2
u_new = 0.99 * u + 0.01 * max(std_batch(y - q), 1e-6)  # initially u = 1
```

Only calibrator parameters receive this loss's gradients; targets, q and bootstrap weights are detached. Adam learning rate is 0.001. The unit u tracks residual **standard deviation**, not mean absolute error or RMS. The scale fit uses a separate folded noise key from the native critic update. Normalized exponential weights are Dirichlet/Bayesian-bootstrap fit weights, **not target/source importance ratios**. The loss trades batch-average coverage against width; it does not enforce conditional 90% coverage at every `(x, a)`. [Exact target/loss/unit, lines 146-225](audit_source_snapshot/algorithms/td3_bc_bca.py); [soft coverage/weights, lines 34-46](audit_source_snapshot/calibration/network.py).

Of 1,024 full-data episodes, 896 supply 89,600 transitions for actor, critic and eta fitting, with minibatches of 256 drawn with replacement. All 12,800 transitions from 128 pre-reserved episodes are withheld from network fitting and normalization. One uniformly selected row per reserved episode forms the 128-row calibration bank. Repeated reuse and its effect on actor updates make this an adaptive historical reference, not fresh independent evidence. [Split/normalization, lines 223-282](audit_source_snapshot/experiments/thrusterchain_diagnostics/training.py); [fit/update/sampling, lines 295-343](audit_source_snapshot/algorithms/td3_bc_bca.py).

Eta fits on every accepted critic step, including before the first reference refresh. Final counters are critic 100,000, live calibrator 100,000, actor 50,000 and each target 50,000. Actor/target updates occur every 2 critic steps with Polyak coefficient 0.005. References refresh after updates 10k, 15k, ..., 95k; no 100k refresh occurs. Refresh copies current eta parameters and unit, then recomputes noisy-target/current-twin-min residuals on the 128 bank rows. The diagnostic queries **100k Q1/actor with 95k frozen eta, u and R**, not the live 100k calibrator. [Update ordering, lines 121-163](audit_source_snapshot/algorithms/td3_bc.py); [refresh, lines 234-277](audit_source_snapshot/algorithms/td3_bc_bca.py); [schedule, lines 491-520](audit_source_snapshot/experiments/thrusterchain_diagnostics/training.py).

Refresh scores are `abs(y - q) / (max(eta, 1e-6) * u)`, with unit row weights. WBCP uses miscoverage 0.1, credibility 0.95, 1,000 posterior draws and a test atom at infinity; `R = max(lambda_hat, lambda_hpd)`. Stored values are **R = 1.3715263605117798** and **u = 1.4656671285629272**. Native deployment computes `U = R * (max(eta, 1e-6) * u)` at its actual floating-point precision. R supplies a global reference-score threshold; u supplies residual units. Neither adds missing action-specific information. A fixed positive `R * u` changes width magnitude and the strength of `Q1 - kappa * U`, but cannot change eta/U action ordering or Pearson alignment in exact arithmetic while floor/refusal branches stay fixed. Eta's 0.01 architectural floor already exceeds the width's 1e-6 floor. [Reference selection, lines 148-191](audit_source_snapshot/calibration/reference.py); [WBCP, lines 57-117](audit_source_snapshot/calibration/wbcp.py); [native width/refusal, lines 65-95](audit_source_snapshot/calibration/dose.py).

Native BCA uses a detached multiplier `1 + 0.5 * U / (U + u)` on **logged-action BC error**, with neutral multiplier 1 on unusable widths. It does not implement the diagnostic's live candidate `Q1 - U`. That subtraction is one proposed construction, not the unique or original BCA rule. [BC readout/update, lines 283-320](audit_source_snapshot/algorithms/td3_bc_bca.py); [actor loss, lines 128-139](audit_source_snapshot/algorithms/td3_bc.py); [dose, lines 41-62](audit_source_snapshot/calibration/dose.py).

Residual variation combines fitting error, changing models, injected target-action noise and state/action variation, despite a deterministic plant. Bootstrap weighting and R do not separate these into epistemic Q uncertainty. Local residuals can cancel or accumulate along a policy trajectory; their absolute size alone does not determine signed long-return or advantage error.

## Exactly two proposed variants

Before either option could launch, verify an untouched existing offline episode pool exists. Otherwise, separately authorize logger data with whole-episode fit/calibration/evaluation splits. This note neither establishes availability nor creates data. Unused transitions from the adaptive 128 reserved episodes are not an independent episode holdout. Current diagnostic/evaluator banks remain evaluation evidence and cannot become training labels.

**A. Frozen-policy local overestimation signal - recommended smallest next test.** Freeze 100k Q1 and the unsmoothed canonical actor. On logged transitions define:

```text
delta = Q1(x, a) - (r + m * Q1(x_next, pi(x_next)))
z = max(delta, 0)
L = mean rho_tau(z - h(x, a)), with tau = 0.9 and h nonnegative
```

Here rho_tau is pointwise pinball loss. This is a **new loss and target**, not a refit of aggregate BCA; old R is not its calibration. Required data are `(episode_id, t, x, a, r, x_next, m)`, frozen normalizer/Q1/actor hashes and episode partitions fixed before labels. No simulator, true-Q labels, target smoothing or actor training is needed. A separate calibration pool is needed only for a separately named residual interval. Independence from original model training/history requires previously unconsumed archived episodes; otherwise results must be labeled conditional/in-sample. This local positive-residual predictor need not align with signed Q_pi error or improve decisions.

**B. Signed residual propagation / fitted value-error corrector.** Freeze the same Q1, pi and delta. Fit a signed head g with terminal continuation 0:

```text
L = mean (g_t(x, a) - stopgrad(delta_t + m * g_{t+1}(x_next, pi(x_next))))**2
Q1 - Q_pi = delta + P_pi(Q1 - Q_pi)  # delta denotes conditional expected residual
```

This seeks propagated value error, equivalently a form of fitted policy evaluation. If a nonnegative penalty is later wanted, apply `max(g, 0)` **after** signed propagation. Summing positive local residuals instead discards cancellation and changes the target. Required data are the same complete offline tuples, correct time/boundary masks, episode-disjoint fit/evaluation pools and successor coverage for pi. Pi supplies queried successor actions; logged next actions are not substituted. Backward fitting or a fixed fitted-evaluation schedule is an additional algorithm to declare before execution. Function approximation, coverage and projected-Bellman assumptions remain, as does FQE bias. This is not automatically calibrated uncertainty. Logged behavior returns describe behavior continuation, not Q_pi, and cannot substitute for Q_pi labels.

**Shared hard limitation:** the logger has 625 categorical actions; the frozen deterministic actor is generally off-grid. Off-grid successor and candidate actions have zero logging probability. Network queries supply no outcome evidence there. Arbitrary off-grid Q_pi is not identified from these logs without structural/generalization assumptions. Projecting or randomizing the target policy onto supported actions changes the estimand and requires an explicit choice. Neither option uses evaluator-only oracle labels to bypass this limit. [Logger probabilities, lines 60-80](audit_source_snapshot/experiments/thrusterchain_diagnostics/training_data.py).

**Decision:** choose A to test a directly fitted, policy/head-aligned local signal. Choose B only for propagated policy-value error with its offline extrapolation assumptions explicitly accepted. Keep actor/critic fixed and name the variant. Changing R or its nominal level alone is dose tuning, not signal redesign. No implementation, extra sweep, training or Q guarantee is implied.

## Identity record

Preserved commit: `b007052a2494dc651ac00469fa9e922b287cd1e5`.

- Checkpoint SHA256: `4088278230fc413c509e27e189daa342468cfc4fb6d2479e3e8c29e149dac3d8`.
- Frozen 95k calibrator-parameter SHA256: `8f90c41ae7d4e822dadcadf2be64232bb58db79c985f8bb1314c1058ce5f031d`.
- [Engine identity](prepared/engine_identity.json) SHA256: `199a9a595076e254a5b15bcf8e21d18488a0f8c84b31c62c52166048c5a2a792`. All eight linked source files matched its `source_sha256` entries.
- `td3_bc_bca.py`: `3f8452df67b1ac51c78b67c36a6f6269c849aa9746f5e15df771d553fce0536e`.
- `calibration/network.py`: `cfde14313cc13cfe96225234c81a8546121ff50f3daf789717ac14c7720f354c`.
- `calibration/reference.py`: `f26273784c458f058c9ebd9b4ebe1e3e7e20c1fc05abfc3297aead724e3b6b1b`.
- [Training manifest](/C:/Users/David/Documents/Codex/2026-10-20/task/results_20261021/release/ThrusterChain_results_20261021/evidence/core_20261020T2250Z/jobs/replicate_01/full/bca/manifest.json) SHA256: `dba31146dd689f12c16853f2e87b8ab76b49aa29a474748113b1560e07a6bf68`.
- [Native reference values](prepared/reference.json) SHA256: `5738e9246d4da08023f146ba8050a6dff397417e60961bcbde0cdf5cbe90440b`.
