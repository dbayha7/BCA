# Correcting the IW-BCA design

**Decision, 29 September 2026 UTC:** David clarified that the intended IW method is weighted conformal prediction. Existing fitting-IW variants do not implement that method. The correction must weight the held-out threshold, include query mass, and justify the source/target ratio. It must be a separately identified method, preserving prior experiments.

## What changes, in plain language

Current fitting IW changes which examples teach the error-scale network. The intended correction also changes how the held-out prediction errors determine the final band. Errors receive more mass when they represent situations that are more common in the declared target population than in the calibration population. The queried state/action also contributes mass; if there is insufficient supported calibration evidence, the correct answer may be an infinite band.

This is not a new reward, a known true Q-value, or a guarantee that an unfamiliar action is bad. The response currently targeted by BCA is a frozen host-specific one-step Bellman target. Coverage of that response and avoidance of harmful actions remain separate questions.

## Evidence already available

The September 22 six-paper audit already identified unweighted radius refreshes, non-ratio fitting heuristics, and adaptive calibration reuse. Its observed-support Bayesian tail and finite-Monte-Carlo limitations remain unresolved; this correction does not erase them. The current clean repository confirms the same path:

- TD3+BC, ReBRAC, CQL and IQL all refresh through `calibration/reference.py:fit_posterior` with one group.
- `calibration/posterior.py:partitioned_posterior` supplies a group indicator as calibration mass and query mass 1. With one group these are all equal, even when fitting IW is enabled.
- The lower-level `posterior_radius` accepts unequal masses, but that capability is not connected to a declared source/target ratio in the hosts.
- The active training interface uses one global cached radius. A corrected weighted conformal radius generally depends on the query's ratio; it cannot be represented by that single scalar without additional justification.

The additive reference in `experiments/weighted_conformal/` tests the intended threshold. No old source or result has been relabeled or replaced.

## Mathematical contract

For a fixed checkpoint, let `X=(s,a)`. Let `Y` be the explicitly identified host target computed from the observed `(r,s',d)` and frozen target construction, including any fixed law for target-action randomness. Let `m(X)` be the frozen critic prediction and `sigma(X)>0` the frozen scale including its residual unit. Calibration scores are

`S_i = |Y_i - m(X_i)| / sigma(X_i)`.

For source covariate distribution `P_X` and target covariate distribution `Q_X`, define

`w(x) = dQ_X/dP_X(x)`.

For a query `x`, calibration probabilities are `p_i(x)=w(X_i)/(sum_j w(X_j)+w(x))`, and the query probability is `p_*(x)=w(x)/(sum_j w(X_j)+w(x))`. The conformal radius is the `1-alpha` quantile of

`sum_i p_i(x) delta_(S_i) + p_*(x) delta_(+infinity)`.

Equivalently, choose the first sorted score whose cumulative calibration mass reaches `(1-alpha)*(sum_i w(X_i)+w(x))`. Return infinity if no score reaches it. A common positive normalization cancels; separate calibration/query normalization does not.

This is the finite-sample query correction in Tibshirani et al., *Conformal Prediction Under Covariate Shift* (2019), equations (5)-(7), Corollary 1 and its split-conformal extension. The claim is **marginal coverage over target examples and calibration sampling**, not coverage at every action or a high-probability guarantee on a single realized bank. The contract requires correct ratios, target support within source support, the same conditional response law `Y|X`, and the appropriate independent/frozen-score setup. Estimated ratios require their own error analysis. Source: <https://www.stat.berkeley.edu/~ryantibs/papers/weightedcp.pdf>.

### Keep the Bayesian component separate

For explicitly supplied exponential draws `E_mi`, tilt observed-score masses by `E_mi*w(X_i)`, calculate each weighted `1-alpha` quantile, then select the declared credibility percentile over draws. Keep

`R_full(x)=max(R_conformal(x), R_Bayes)` and `width(x)=sigma(x)*R_full(x)`.

The CPU reference retains this observed-support Bayesian construction so that both BCA components remain visible. Taking the maximum cannot shrink the conformal band. It does not establish that the finite-draw Bayesian radius is a 95%-confidence population-risk bound; the prior audit's missing-tail and Monte Carlo findings still apply. This is not a blanket implementation claim for every equation or theorem in the supplied *Weighted Bayesian Conformal Prediction* paper.

## Arithmetic example

Use sorted scores `[1,2,3,4]`, calibration masses `[1,1,1,7]`, query mass `1`, and `alpha=0.4` **for illustration only**. Total mass including the query is `11`; the cutoff is `0.6*11=6.6`. Cumulative observed masses are `[1,2,3,10]`, so the conformal radius is `4`. With equal masses the cutoff is `0.6*5=3`, giving radius `3`.

If the query mass is `100`, the weighted cutoff is `0.6*110=66`, above all observed mass `10`: the radius is infinity. It must not be replaced by the largest observed score or silently dropped. If the full radius is `4`, frozen scale is `2` and center is `10`, width is `8` and the response interval is `[2,18]`. This arithmetic does not claim the response is true policy value.

## What must be resolved before host training integration

1. **Name the target population.** The immediate validation target should be a frozen score under a declared state/action query distribution. Same-state action shift and changed policy state visitation require different ratios. Full `Q(s,a)/P(s,a)` includes state shift; `pi(a|s)/beta(a|s)` alone covers only a justified common state distribution.
2. **Supply defensible ratios.** Existing CQL policy density omits the behavior denominator. Existing TD3/ReBRAC affinity and IQL advantage weights are not automatically the desired ratio. A heuristic can define a specifically tilted source population, but that population must not be renamed the deployed policy.
3. **Handle deterministic support honestly.** TD3/ReBRAC deterministic actions can form a singular target relative to continuous behavior actions. An estimated smooth behavior density does not remove this mismatch. A stochastic neighborhood is a different declared target, requiring a predeclared law; do not silently add actor noise and claim coverage for the original deterministic policy. CQL's stochastic policy density and IQL's actual action semantics must likewise be separated from deterministic evaluation conventions.
4. **Keep density estimation separate.** For a future estimated-ratio branch, fit source-versus-target classification on separate covariates, freeze the estimator, record class-prior correction, inspect overlap and weight concentration, and evaluate on fresh data. Do not use rewards/harm labels to select weights. Estimated ratios are approximate, not the known-ratio theorem. Capping or ESS tempering changes the ratio and potentially the target; no hidden clipping or tempering is allowed in the conformal floor.
5. **Resolve calibration independence.** The current bank affects later actors/critics through BCA, so later scores are not automatically independent of the bank. Merely freezing a checkpoint afterward does not undo earlier feedback. Use a fresh post-freeze calibration/test split or establish an appropriate adaptive result. Whole-episode separation alone does not make all rows within each trajectory IID. Choose a sampling unit that preserves the declared conditional response law; do not select transitions using future episode length without accounting for that selection.
6. **Carry the query ratio through the host.** Store a sorted weighted score bank, ratio-estimator identity and frozen score snapshot, then compute `R_conformal(s,a)` at the actual input used by each host. Existing one-scalar caches and checkpoint schemas need a separate version. Test state/action alignment, normalization, stochastic target keys, query mass and scalar-to-vector changes before a run.
7. **Declare what happens at infinity.** A statistically unbounded band must remain unbounded in coverage reporting. Existing finite-support masks can fall back to native host weights, so lack of a finite certificate does not imply safe action rejection. Predeclare and log the behavior separately; no silent maximum-score replacement or automatic penalty retuning.

## Host boundaries to preserve

| Host | Existing BCA intervention to hold fixed initially | Additional correction required |
|---|---|---|
| TD3+BC | Detached multiplier on actor behavior cloning | Query-specific radius; justified action/state population and support |
| ReBRAC | Detached actor-BC multiplier; critic BC unchanged | Same query-specific radius issue; preserve the regularized host target definition |
| CQL | Detached multiplier on each transition's conservative gap; original dual gap | Ratio at the actual queried input; do not reinterpret one recorded-action width as a band for every negative action |
| IQL | Shrink capped actor-weight excess above one; shared Q/V and fixed gain 1 | Preserve shared nuisance state; distinguish advantage-based fit weights from calibration ratios and actual actor sampling |

Changing how the host consumes width is a separate scientific change. The first comparison should isolate the corrected radius while keeping the scale-fitting rule fixed, then test fitting-IW separately if justified. Otherwise a result cannot distinguish a better weighted threshold from a differently trained scale network. Keep alpha=0.1 and both components; do not tune coverage to seek better returns.

## Staged execution

**Implemented mathematical stage:** a CPU reference, boundary/support tests, exact finite-population coverage enumeration, and a predeclared independent known-ratio simulation. These establish arithmetic behavior in their stated setting only.

**Next real-data stage:** freeze a checkpoint/score and define an independent source and target sampling design with justified ratios. A controlled simulator study can use known action sampling densities on a declared common state population; this would test the calibration mechanism, not offline deployment coverage by itself. If using existing offline data instead, unknown behavior density and target support must be addressed explicitly. No new simulator stage is launched by this document.

**Later training stage:** integrate vector/query-dependent radii, explicit infinity handling and versioned checkpoints; declare whether the adaptive learning loop has a theorem or is empirically evaluated. Freeze all arms/seeds before training. Retain standard BCA, distinguish old fitting-IW from new weighted-threshold BCA, and measure coverage, width/infinity frequency, ratio concentration, within-state action harm and return separately.

## Current completion claim

The intended method is now explicit. A passing reference test or synthetic result does **not** mean the four-host IW training method is complete. Real-population ratios, sampling/independence, query-aware host bindings and frozen experiment matrices remain integration gates. Current standard-BCA/OOD work and all old outcomes continue under their existing identities.
