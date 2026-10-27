# Diagnostic pseudocode: no production method change

This is a frozen fitting-data diagnostic. The following optional centered-scale expression is a hypothesis to inspect, not an approved replacement objective.

## Fixed data and models

```text
verify original source manifests, successful recovery and exported checkpoint
load the exact frozen native UniFloral actor, twin critic and BCA reference
load only the existing 256-row fitting scale bank (observations, actions, group IDs)
for each fitting observation s:
    A_s = stable_float32_deduplicate([
        frozen_actor(s), recorded_action(s), zeros(4), six seeded uniform actions
    ])
    compute Q1(s, A_s), U(s, A_s), |Q1-Q2| and actor distance
    compute action-centered values X_sa - mean_a(X_sa)
C_X = sqrt(mean_s(mean_a((X_sa - mean_a(X_sa))^2)))
also record uncentered RMS and centered fraction of squared magnitude
```

Equal weights are assigned to states and then to their candidate actions. Deduplication can give ragged action banks. These quantities measure ranking contrasts on this bank. They do not measure actual returns, generalization or statistical coverage.

## Native and hypothetical actor-gradient probes

```text
use the first 32 fitting rows for every actor-gradient probe
a_i(theta) = native_actor(theta, observation_i)
D = mean_i(abs(Q1(observation_i, a_i(theta))))
c = stop_gradient(2.5 / (D + 1e-7))       # unchanged native coefficient
BC = mean_i(mean_action_coordinate((a_i - recorded_action_i)^2))
L_native = -c * mean(Q1) + BC
L_direct = L_native + c * kappa_penalty * mean(U)

if centered Q/U scales are finite, nonzero, sufficiently resolved and float32-usable:
    f = stop_gradient(C_Q / C_U)         # from the separate fixed 256-row bank
    L_hypothetical = L_native + c * kappa_penalty * f * mean(U)
else:
    report candidate factor unavailable; do not replace it with zero or one

measure action-space grad_a Q1 and grad_a U at the frozen actor actions
separately measure parameter-space grad_theta mean(Q1), mean(U) and BC
record norms, cosine alignment and cancellation; report ratios unavailable if denominator is zero
verify actual existing candidate's g_kappa1 - g_kappa0 = c * grad_theta mean(U)
verify frozen reference gradients are zero while action derivatives remain live
verify proposed scale statistics are detached, using a counterexample where omission changes the derivative
do not call any optimizer, fitter, refresh routine or simulator
```

Subtracting the same action-independent constant from every action score cannot change its mathematical ranking. Subtracting a detached state-specific mean from U also leaves its action derivative unchanged. Recomputing a mean or denominator through the live actor can change its parameter derivative; that is why gradient routing is checked separately. Float32 ties/rounding must be reported explicitly instead of silently forcing an index comparison to pass.

## Coherent unit changes and epsilon

```text
for positive scale s in [0.1, 1, 10]:
    Q_s = s * Q
    hypothetical reward_s = s * hypothetical reward
    frozen_residual_unit_s = s * original_frozen_residual_unit
    compute U_s using the actual unchanged width pipeline and this copied reference
    verify U_s agrees with s * U and its action derivative agrees with s * grad_a U
    verify ranking invariance for resolved gaps under Q_s - kappa * U_s
    verify candidate C_Q_s / C_U_s agrees with original ratio when usable

    c_s = stop_gradient(2.5 / (s*D + 1e-7))
    beta = s*(D+1e-7) / (s*D+1e-7)
    expected_total_gradient_s = beta*(Q-term gradient + penalty gradient) + BC gradient
    test this expression without changing native epsilon
    separately show epsilon_s = s*1e-7 would give beta=1; label counterfactual only
```

The fitting bank has no rewards or next observations. TD/reward identities therefore use declared synthetic algebra fixtures, not measured training transitions. At Q=0, the native coefficient is finite but large (25,000,000); a nonzero-slope synthetic Q fixture must expose that amplification. No claim of full training invariance follows: residual-unit initialization, EMA floors, optimizer epsilon and learning dynamics are outside this frozen test.
