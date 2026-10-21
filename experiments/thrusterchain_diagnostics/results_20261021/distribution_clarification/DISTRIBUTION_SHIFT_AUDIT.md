# What distribution changes in the coverage graphic?

The existing graphic is a **joint state/time and conditional-action stress test**, with an additional changing model residual. It is not a state-only shift experiment. Full/rare/zero are retention conditions, not a clean decomposition of shift mechanisms.

Let z=(x,g,t), where x has 13 physical coordinates, g is the observed stall/monotone regime, and t is physical episode time (the observation stores (100−t)/100). The physical dynamics and reward given (z,a) are the same in calibration and evaluation. There is no introduced change in the environment's transition/reward law.

## Source trajectories and calibration law

Each source replicate has 1,024 H=100 trajectories, stratified into exactly 256 episodes per regime/load context, with their episode ordering randomly permuted. Starts have x[0:10] independently uniform on [−0.5,0.5], current x[10] in {0.3,1.1}, and (x[11],x[12])=0.2(cos φ,sin φ), φ uniform on [−π,π]. Regime and current stay fixed; subsequent physical states follow the actual dynamics under the behavior policy. Thus the source state law at time t is behavior occupancy d_t^β, not the reset law at every t. [training_data.py, lines 95–125; continuous.py, lines 57–70 and 120–137.]

For A={−1,−k,0,k,1}, k=float32(0.8), let a*_j(z) be the nearest atom to clip(0.6 u_LQR,j(z),−k,k), taking the lower atom on a tie. Behavior coordinates are independent conditional on z:

`β(a|z) = ∏_{j=1}^4 [0.8 1{a_j=a*_j(z)} + 0.2/5]`, for a∈A⁴, and zero outside A⁴.

The greedy atom has mass 0.84 and every other atom mass 0.04 per coordinate. These are categorical masses, not continuous densities. [training_data.py, lines 60–80 and 110–117.]

The pre-outcome critical predicate and inclusion probability are

`C(x,a) = 1{x[10]≥0.8 AND max_j|a_j|>k}`;

`q_α(x,a) = 1 − (1−α) C(x,a)`, with α=1 (full), 0.05 (rare), 0 (zero).

Retention is **joint in load/current and action**. Low-current high actions are retained; high-current low actions are retained. Only their intersection is thinned. Full performs no deletion. Shared source logs/uniforms produce nested masks, with original successor observations preserved; deletion never changes already-generated state trajectories or triggers top-ups. [continuous_data.py, lines 32–34; training_data.py, lines 117–125 and 171–180.]

128 whole source episode IDs are reserved before retention. In every condition one row is selected uniformly from each reserved episode's retained rows. Conditional on the realized complete trajectories and masks, the marginal law of a uniformly selected calibration row is exactly

`P̂_cal,α = (1/128) Σ_{e∈reserved} (1/N_e) Σ_{t=0}^{99} I_et δ_(x_et,g_e,t,a_et)`,

where I_et is the retained indicator and N_e=Σ_t I_et. All 128 reserved episodes have retained rows in this run. Each contributes one independent seeded within-episode selection; no extra row is invented for an empty episode. This finite conditional law is not automatically β(a|z)q_α(z,a)/E_β[q_α(z,A)]: equal-episode weighting, random denominators and trajectory correlations matter. [training_data.py, lines 100–102; training.py, lines 223–247.]

Under full retention, N_e=100 and the per-episode selection time is uniform on {0,…,99}. The entire trajectory log has temporal dependence. Selecting one row per episode removes within-bank repeated-episode sampling, but does not make this bank an IID sample from the designed query law: contexts were stratified/reserved without replacement, states come from time-dependent behavior occupancy, and the same bank is adaptively reused. Do not describe full as an IID in-distribution coverage control.

The realized calibration context counts (stall-low, stall-high, monotone-low, monotone-high) are the following in all three retention conditions; they are not 32 each:

| Replicate | Stall low | Stall high | Monotone low | Monotone high |
|---|---:|---:|---:|---:|
| 1 | 22 | 26 | 37 | 43 |
| 2 | 29 | 29 | 34 | 36 |
| 3 | 31 | 36 | 34 | 27 |
| 4 | 33 | 30 | 31 | 34 |
| 5 | 25 | 25 | 35 | 43 |

Source: frozen `data/seed_1101…1105/{full,rare,zero}/preparation.json`. Critical calibration counts are full 16,22,21,14,15; rare 1,3,0,0,1; zero 0,0,0,0,0. These are pooled adaptive calibration banks, not critical-region certificates.

## Designed coverage-query law

For coverage/Q diagnostics, the query bank uses four independently drawn reset-law physical states per context (16 base states), repeats each physical state at t∈{0,50,99}, then crosses each of the resulting 48 observations with six commands. Contexts and these three times receive equal weights. At t=0 the physical start law agrees with the source reset law. At t=50 or 99, the same reset-law physical values are inserted at a late time; these are not states obtained by rolling the behavior policy forward to that time. Wave phase remains drawn from the same initial uniform law. [training_eval.py, lines 45–64 and 81–95.]

For each query observation z, candidate indices are equally weighted:

`ν_k(da|z) = (1/6)[δ_0⃗ + δ_(−k1⃗) + δ_(+k1⃗) + δ_(−1⃗) + δ_(+1⃗) + δ_π_k(z)]`.

Repeated commands, if any, have their candidate weights added. The deployed actor command is clipped and float32. The first five commands lie on the behavior grid but have very different conditional weights and cross-actuator dependence from β. The actor candidate can lie off the grid, where β has zero mass; it also varies with method/seed/checkpoint. Therefore action shift is already present even if the query observation were held fixed. It is not isolated from the simultaneous state/time shift. There are 288 designed pairs, not 288 independent draws. [training_calibration.py, lines 52–60; training_eval.py, lines 268–275.]

This 48-state coverage-query bank is separate from the 400-start final return bank. The latter starts at t=0 and follows each policy for 100 steps. It must not be confused with the designed coverage law.

## Model-residual drift, not environmental label shift

The calibrated target at optimizer step k is the native noisy TD3 residual

`δ_k(z,a,ε) = r(z,a) + 1{t<99} min_i Q_target,i,k(z′, clip(π_target,k(z′)+ε,−1,1)) − min_i Q_i,k(z,a)`,

with γ=1 and per-coordinate ε=clip(N(0,0.2²),−0.5,0.5). Refresh scores use |δ_k| divided by the current positive scale. [algorithms/td3_bc_bca.py, lines 146–156 and 234–270.]

The final reported coverage is

`(1/288) Σ_queries 1{|δ_100000(z,a,ε_eval)| ≤ q_95000 · s_95000(z,a)}`.

The threshold, calibration scale-network parameters and residual-scale unit are frozen from the last 95k refresh. The target/online actor-critic parameters used in the residual continue changing to 100k. Evaluation uses an independent, fixed noise key with the same noise law. Thus the residual/score target changes even at a fixed (z,a); this is model-residual nonstationarity, not a change in the physical reward or dynamics. The calibration bank was repeatedly reused during training. [training.py, lines 490–520; training_calibration.py, lines 70–91 and 102–117; algorithms/td3_bc_bca.py, lines 283–285; calibration/dose.py, lines 70–85.]

True finite-horizon Q diagnostics and the deterministic Bellman residual are distinct from this noisy target-network residual. A measured coverage value below 0.9 under this joint designed query shift, adaptive reuse and model-residual drift does not isolate a state-shift effect, establish a long-horizon Q coverage failure, or by itself refute a conformal theorem.

Suggested graphic label: **“Empirical residual coverage under a designed state/time + action shift”**. Subtitle: **“Behavior-trajectory calibration → reset-state/time-grid queries with five fixed commands and the learned actor; final residuals use a 95k frozen reference.”** State-only and action-only effects require separately controlled evaluations; this completed design does not separate them.
