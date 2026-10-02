# Step 4 design: where the Q1-residual signal is applied, at matched realized strength, across behaviour quality

Synthesis of three designs (mechanism, placements, transfer) and the critique. The repo was read-only. The settings
numbers below come from closed-form or outcome-free scripts, and so do the step-3 re-readings. No actor was run and no J was computed for any step-4 arm. These scripts are in the session scratchpad:
- `synth_step3_cells.py`, `synth_settings.py`, `synth_pull.py`;
- the critique's `review_step3.py` and `review_align.py`;
- the transfer design's `lq_common.py`.

They must be copied into the repo before the pre-registration is frozen (section 10).

## 0. What this step decides

**Inputs.** The signal is held fixed: Q1's calibrated band. It is tested in five placements:
- **P1:** BC amplification;
- **P2:** actor-Q trust weighting;
- **P3:** a critic-side pessimistic value read by the actor;
- **P1π, P2π:** the same two weightings read at the actor's action.

**Output, per placement, evidence set and behaviour-quality level.** One verdict from a fixed taxonomy:
- targeting real;
- targeting below uniform;
- optimizer-dependent;
- leverage-weighted regularization;
- nuisance weighting;
- harmful assignment;
- strength only;
- nothing.

It also returns an information diagnosis:
- the placement cannot use perfect information;
- the information class is wrong;
- or the signal is too weak.

**Comparisons.** Every comparison is at matched realized strength, defined as the policy displacement the
intervention causes. The strength itself is reported separately, decomposed into a uniform (α-retune) component and
a redistribution component.

**Behaviour cross.** The user's hypothesis, "a signal can correctly identify critic unreliability while stronger BC still hurts by pulling toward poor demonstrations", is tested as an explicit decomposition. The behaviour cross holds the start policy, the critic and the training states fixed, and varies only the logged actions.

## 1. What steps 1-3 hand over, and what the critique changed

**Fixed by steps 1-3.**
- The signal is Q1's residual band.
- The current hook has no dynamic range: dose about 1.3, row SD about 0.02. It is worse than both controls in 121 of 216 cells and better in 60.
- An oracle dose at the same mean moved J a lot, mostly in independent_errors: +0.144 with good data, −1.008 with poor.

**Post-hoc re-reading of step 3 (B1 of the pre-registration).** Same mean is not same strength. Per replicate,
the oracle/constant BC-term gradient-norm ratio is:

| Case | Oracle/constant ratio |
|---|---|
| independent_errors, good | 2.25-3.06 |
| independent_errors, poor | 1.07-2.82 |
| shared_bias | 1.05-1.30 |
| q1_optimistic | 0.87-1.11 |

BCA/constant is 0.9997-1.034. A strength-only model fits the magnitude of BCA − constant with correlation 0.86 (Spearman 0.89; 0.92 without the 5 largest cells). It is off by 2× in the largest cell (−1.34 predicted against −3.03 observed). Its 173 of 182 sign agreements equal the count of cells where BCA − constant has the sign of constant − none. That only says BCA acts like a slightly larger constant dose, so it is not a test of the model.

| Critique item | Fix in this design |
|---|---|
| Adam behaves almost like sign descent on a 6-parameter actor (≤ ~0.15 per coordinate in 500 steps); response to strength can be step-like; no optimizer-free reference | **Fixed point (FP) regime is primary for targeting verdicts** (closed-form 6×6 solve; BFGS for P3). Adam-500 (A500) must be concordant; A5000 subset for horizon. Matching uses first crossing and flags non-monotone ladders |
| Weight variance damps minibatch steps; constant arms have none | FP is full-batch, so it has no damping. A500 full-batch variant of NONE/P0/SIG/SHUF. Kish n_eff and Adam per-coordinate SNR reported for every run. T_SHUF/T_STRAT share the multiset; R at A500 is read only next to the full-batch check |
| regional_optimism is collinear with uniform BC (centered region on zero-mean states) | Dropped. Positive controls: **cone** (non-centered, anisotropic; closed form cos(DΣ, DΣ_cone) ≈ 0.92) and **block C** (localizable actor: per-context gains) |
| P3 push-permutation has zero expected gradient | Dropped. For P3, *any* row permutation has E[v_p(i) s_iᵀ] = 0 on zero-mean states, so no same-multiset null exists. P3's nulls are the P0 uniform frontier and NUIS (the clean-case σ as penalty) |
| Placement confounded with evaluation point | Two same-point families: **L** (P1L, P2L at logged actions, WBCP's certified point) and **Π** (P1π, P2π, P3 at π(s)). Within-family contrasts are primary, each family has its own Holm correction, and cross-family contrasts are descriptive |
| Trust weighting raises the BC share under Adam (E19 contradiction) | Behaviour cross decomposed at matched S: G = Str (uniform/α component) + R (redistribution); data pull measured by pull share; DR6 uses Str and R separately |
| Evidence sets by case name; few cells mislead | Evidence sets M/W/H come from outcome-free alignment_K at K_0 plus FP displacement, frozen before any J; tilt constructs M at every quality level |
| STRAT over-controls where the error is leverage-shaped | Verdict V4 "leverage-weighted"; "strength only" requires R and T_SHUF immaterial |
| Offset moderator wrong; good/poor differ in K_π, Q^π, λ, states | S-common start K_0 = 0.5K* + 0.5D at every level; λ locked per cell; regime-specific forecasts: A500 follows sign(cos_BC − alignment); FP follows the J at the BC target |
| S-common start leaves state covariance different | **Common-state datasets** (X-CS, primary): identical states, noise and next-state shocks across levels; only K_b differs. X-EP (each level's own episodes) is secondary and measures the state-covariance pathway |
| Transfer's LSTD critic-side arm becomes a global quadratic | Not used in single-context LQ. A penalized-backup critic (C2) runs only in block C, where the per-context quadratic class can represent a context-localized penalty |
| Q-trust at logged actions handicapped; ambiguous constant trust | Π family includes P2π in its own Holm family. Constant trust is defined as ω ≡ 1/(1+β), which equals P0(1+β) exactly at FP and up to eps under Adam |
| DR5 mixed confirmation possible from heteroscedasticity alone | DR6b also requires OR1 < SHUF and SIG < NUIS |
| [derived] checks that would fail without a bug | Tolerance policy (section 9). EXACT uses the same locked λ. Adam equivalence is asserted only at the power-of-2 factor. m_eff is checked only where \|cos(g_Q, g_BC)\| < 0.98. Analytic quantities are computed on the same sample |
| Near-zero strength targets | Floor S* ≥ max(2e-3, 0.02·S_host); cells below it are excluded at that level |
| ρ matching tracks gradient cancellation | S (displacement) is the primary match. ρ is a secondary re-verdict, and disagreements are flagged |
| P2 reachability (β ≤ 1) | Mirror form ω = 1/(1+βM): β is unbounded and ω stays in (0,1]. Fallback level rule fixed in advance |
| Step-3 re-analysis wording | Restated as above. `strength_reanalysis.py` is saved and hashed before freezing |
| Count rules treat cells as independent | The unit is the replicate: pooled per evidence set, Holm-corrected. Cell counts are secondary |
| D4RL frozen-critic pull bias, approximate leverage, Adam state, power | Same-multiset nulls are primary (the pull bias is equal across them). Leverage comes from exact per-row gradient norms. Fresh, identical Adam for every arm. Seeds chosen by MDE. The verdict is scoped to "assignment beats leverage-matched shuffles" |

## 2. The fixed signal and the weight transform

**Signal.** One fit and one threshold per cell, shared by every arm and placement:
- Ŵ(s, a) = λ̂ · max(η(s, a), 1e-6) · u.
- η is fit by `lq_harness.fit_scale_signal(signal=1)` on t − Q1: 5,000 steps, batch 256, step-3 keys and initialization.
- λ̂ is the WBCP threshold from `calibrate_signal` (α 0.1, β 0.95, 1,000 draws).
- u is the residual unit.
- η ≥ 0.01 by construction (softplus + 0.01). The calibrator is a ReLU MLP on (s, a), so ∇ₐη exists almost everywhere.

**NUIS (nuisance) signal.** The clean case's Ŵ_clean, fit on the same data, replicate and quality level, with its own threshold.

**Rank-normal transform (P1, P2; identical multiset for every assignment).**
- **Multiset.** For n training rows: z_(j) = Φ⁻¹((j − ½)/n), clipped to [−2.5, 2.5]; M_(j) = exp(s·z_(j)) / mean_j exp(s·z_(j)).
- **Assignment.** Row i receives M at the rank of its assignment variable x_i. Ranks are global, over all n training rows, and ties are broken by a seeded random permutation, never by row order.
- **Primary, s = 0.83:** CV 0.914, range 0.090-5.714, Kish n_eff/n 0.545, p90/p10 8.39 (the hook's CV is about 0.015).
- **Robustness, s = 0.47:** CV 0.482, range 0.277-2.908, Kish 0.811.
- **The rank transform cancels λ̂.** P1 and P2 test the shape of σ, not its calibrated level. The calibrated level enters only through P3 at c = 1 and the T-lin variant.
- **T-lin (SIG only, robustness):** M_i = Ŵ_i / mean Ŵ, compared against SHUF-lin (the same magnitudes permuted).

**π-evaluated assignments (P1π, P2π).** x_i = Ŵ(s_i, π(s_i)).
- At FP, π is frozen at π_0 = −K_0 s with global ranks.
- In A500, x is recomputed every step at the current actor, under stop-gradient. Batch ranks map to z = Φ⁻¹((r_b − ½)/256) with the same s and clip, normalized to batch mean 1. Every batch then has the exact multiset.

## 3. Placements and exact losses

All arms use the host's λ: λ = α / mean_i |Q1(s_i, π(s_i))| under stop-gradient, computed from the unweighted, unpenalized Q1, with α = 2.5. In td3_bc this is already what the code does for P2; for P3 it is a lock. Here d_a = 2.

**A500 batch losses (batch of 256 rows i).**

| Arm | Loss | Knob ladder |
|---|---|---|
| NONE (weighted path) | −λ·mean_i Q1(s_i, π(s_i)) + mean_i (1·mean_a(π(s_i) − a_i)²) | — |
| P0(m), uniform BC (= α retune) | BC row weight m | m ∈ {0.5, 0.75, 1, 1.15, 1.3, 1.6, 2, 2.5, 3, 4, 6, 10} |
| P1, BC amplification | BC row weight v_i = 1 + β·M_i | β ∈ {0.01, 0.03, 0.1, 0.3, 0.6, 1, 2, 3, 6, 10, 20, 30} |
| P2, trust (mirror of P1) | Q row weight ω_i = 1/(1 + β·M_i), BC weight 1 | same β ladder |
| P3, critic-side LCB read by the actor | −λ·mean_i[Q1(s_i, π(s_i)) − c·Ŵ(s_i, π(s_i))] + BC; gradient through σ's action input; calibrator frozen | c ∈ {1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100}; c = 1 is the calibrated 90% lower bound |
| EXACT (ceiling of any critic correction) | −λ·mean_i Q^π(s_i, π(s_i)) + BC, with λ from Q1 (the same λ) | — |

**Relations between the placements.**
- P2's row loss equals P1's row loss divided by (1 + βM_i). The per-row BC:Q ratio is identical, so **P1 against P2 isolates row importance**.
- Constant trust ω ≡ 1/(1+β) equals P0(1+β). This is the user's same-mean constant for both P1 and P2.
- P3 with a constant penalty has zero K-gradient, so it equals NONE.
- A uniform distance penalty c‖π(s) − β(s)‖² is uniform BC toward the behaviour mean. So **P0 is the common uniform reference for all placements**.

**Continuity arms** (block R: bitwise through `BASE.td3_bc_update` and `lq_harness.actor_loop`):
- HOOK: step 3's dose;
- CONST-old;
- SHUF-old;
- OR-raw: step 3's oracle, v_i = d̄·e_i/ē;
- NONE through the None path.

OR-raw(t) is the knobbed version, with v_i(t) = 1 + t·(d̄·e_i/ē − 1): t = 1 is step 3's arm, and t* is matched to S.

**Secondary arms.**
- **P3 with native λ** (λ = α/mean|Q1 − cŴ|): the size of the λ side channel.
- **C2, penalized-backup critic, block C only.** Q1_pen = Q1 − c·Q_p, where Q_p is the per-context LSTD fixed point of the reward penalty p_i = Ŵ(s_i, a_i) under π_0. Construction:
  - features: 15 quadratic monomials of z plus 1, per context;
  - exact next-feature expectation under TD3's clipped target noise: E[ñ] = 0, E[ññᵀ] = v_clip·I;
  - θ_c = (Φ_cᵀ(Φ_c − γΦ̄'_c))⁻¹ Φ_cᵀ p_c.

  Q_p is quadratic in z, so the FP is closed-form.

## 4. Two regimes

**FP: fixed point (primary for targeting verdicts).** With λ frozen at λ_0 = α / mean_i |Q1(s_i, π_0(s_i))| over all n training rows (shared by every arm in a cell) and fixed weights, the actor loss is quadratic in K for P0/P1/P2/EXACT, the HOOK and the OR arms.

Per row, write Q1(s_i, a) = aᵀP_i a + 2aᵀq_i + r_i, with P_i = ½∇²ₐQ1(s_i, 0) and q_i = ½∇ₐQ1(s_i, 0), obtained by autodiff in float64 from the float64 critic parameters. They are exact for every case, because all heads are quadratic in a: the cone's ς(s) depends on s only. Then the FP solves the 6×6 system, with column-major vec and S_i = s_i s_iᵀ:

    [ mean_i( −λ0·u_i·(S_i ⊗ P_i) + (v_i/d_a)·(S_i ⊗ I_2) ) ] vec(K)
        = mean_i( −λ0·u_i·vec(q_i s_iᵀ) − (v_i/d_a)·vec(a_i s_iᵀ) )

- u_i is the Q weight and v_i the BC weight.
- The matrix is half the loss Hessian. If its minimum eigenvalue is ≤ 0, the FP is "unbounded" (nonfinite); if K_fp's closed loop is unstable, J = −inf. Both are tallied.
- Π-family weights are frozen at π_0.
- **P3, and P3 with native λ:** the FP is found by BFGS on the full loss, using JAX value_and_grad (float32) with warm starts along the c ladder. Convergence requires ‖∇L‖_F ≤ 1e-5·‖∇L(K_0)‖_F within 500 iterations; the success flag is reported.
- **P3-OR with e1 ≥ 0, and C2:** closed form.

Consequences that the checks use, all [derived]:
- (i) Every row weighting acts on the FP only through the 3×3 matrices Σ_u = mean u_i S_i and Σ_v = mean v_i S_i (with action-noise-symmetric weights). Targeting can exploit only state-direction anisotropy of the error relative to the true pull, or block structure (block C).
- (ii) q1_optimistic: the error is exactly an anti-BC term of strength λ0·κ·d_a, so P0(1 + λ0κd_a) = EXACT.
- (iii) Cone: P1 with raw (not rank) weights v_i = 1 + λ0κd_aς(s_i) = EXACT. Uniform BC cannot do this, because Σ_ς is not ∝ Σ.
- (iv) Block C q1opt-local: P1 with raw weights 1 + λ0κd_a·1[c_i = 0] = EXACT.
- (v) Tilt: no BC weighting cancels the error, but trust weighting can down-weight the state directions where the error is large.

**A500 (faithful TD3+BC update).** 500 Adam steps (lr 3e-4, `torch_adam`, eps 1e-8), batch 256, step-3 batches and keys. The critic, target critic and target actor stay frozen. It uses a harness-local actor update that copies `_actor_loss_fn` with hooks (section 12).

**A5000 (horizon).** 5,000 steps, on replicates 0-4, for the X-CS tilt and cone cells and the block-C localized cells. Arms: NONE, the P0 ladder, and SIG/SHUF×4/STRAT×4 for P1L, P2L and P2π, plus P3 SIG/NUIS, each matched again at 5,000 steps.

**Full-batch A500.** Batch = all training rows, on replicates 0-4, for the X-CS tilt and cone cells. Arms: NONE, the P0 ladder, and SIG/SHUF×4 for P1L and P2L at S2.

## 5. Strength: definition, floors, blind matching, measurement

**Primary strength.**
- δ = K(arm) − K(NONE), within the same regime, data, batches and keys.
- S = sqrt(tr(δ Σ̂_ev δᵀ)), with Σ̂_ev the second moment of the evaluation-split states. S is the RMS action-space movement the intervention causes.
- S_host = ‖K(NONE) − K_0‖_Σ.

**Targets.**
- S*_k = S(P0(m_k)) for m_k ∈ {1.3, 2, 4}, giving S1, S2 (primary) and S3.
- **Floor:** S*_k is usable in a cell only if S*_k ≥ max(2e-3, 0.02·S_host). Otherwise the cell is "below floor" at that level and excluded from matched verdicts.

**Matching.** Done per regime, cell and arm (every permutation separately):
1. Evaluate S on the knob ladder.
2. Take the first bracket where S crosses S*.
3. **FP:** bisect in log(knob) to |S/S* − 1| ≤ 1e-3 (at most 40 steps).
4. **A500:** rerun once at the interpolated knob. If the result is outside 5%, run up to 6 bisection steps.
5. A non-monotone ladder is flagged and counted. With no crossing, the arm is unreachable at that level and is never extrapolated.

**Fallback level.** If S2 is unreachable or below floor in more than 50% of M cells for a placement in a regime, that placement's primary level becomes S1. The rule reads the reachability matrix, which contains no J.

**Blindness.**
- The matching code does not import `value()` or any J function.
- K's, knobs, S and reachability are written to `matched_knobs.json` and an `.npz`, and the SHA-256 is recorded before a separate process (`score_values.py`) computes J.

**Measured for every run** (reported, not matched).
- **Level 1** (at K_0, all n training rows, exact):
  - term K-gradients g_Q, g_BC, g_pen, the total g, and Δg = g − g_NONE, with their Frobenius norms;
  - ρ = ‖g_BC + g_pen‖/‖g_Q‖;
  - the projection g = a·g_Q,NONE + b·g_BC,NONE + r, giving m_eff = b/a and targeting share τ = ‖r‖/‖g‖ (flagged "unidentified" when |cos(g_Q, g_BC)| > 0.98);
  - cos(−Δg, ∇J(K_0)), and first-order efficiency e = −⟨∇J, Δg⟩/‖Δg‖;
  - the Adam-preconditioned increment Δg̃ = Δg ⊘ (sqrt(v̄_NONE) + eps), where v̄_NONE is the NONE run's Adam second moment averaged over 500 steps, and cos(−Δg̃, ∇J);
  - error-correction alignment cos(Δg, g_EXACT − g_NONE);
  - behaviour pull cos(−Δg, (K_bc − K_0)Σ̂).
- **Level 2** (A500 trajectory):
  - per step: the batch gradient norm of each term, the BC:Q ratio, the Adam step ‖ΔK_t‖_F, the mean per-coordinate SNR |m̂|/(sqrt(v̂) + eps), and the Kish n_eff of the batch weights;
  - summaries: means over steps, path length Σ‖ΔK_t‖, final BC loss, final mean Q1, displacement from K_0 (Σ metric and Frobenius), and S.
- **FP:**
  - K_fp, the Hessian's minimum eigenvalue and condition number, and S_fp;
  - the realized frontier m* = argmin_m ‖K_fp(arm) − K_fp(P0(m))‖_Σ over 200 log-spaced m in [0.25, 100];
  - T_front = G(arm) − G(P0(m*));
  - targeting share = ‖K(arm) − K(P0(m*))‖_Σ / ‖K(arm) − K(NONE)‖_Σ (outcome-free);
  - regret_α = max_m G(P0(m)) − G(arm).
- **Both regimes:**
  - pull share = cos_Σ(δ, K_bc − K(NONE)), with K_bc the least-squares BC gain on the training rows;
  - the decomposition G_reg = J(K_N + a·δ̂_P0) − J(K_N) and G_⊥ = G − G_reg (δ projected on the P0 displacement at the same level, in the Σ inner product), with the reversed order reported as a nonlinearity check.

## 6. Arms and controls

**Assignments for the weighting placements.** All use the same multiset; FP uses 16 permutations each and A500 uses 4.

| Arm | L family (P1L, P2L) | Π family (P1π, P2π) |
|---|---|---|
| SIG | Ŵ(s_i, a_i) | Ŵ(s_i, π(s_i)) |
| SHUF_k | SIG's row weights permuted globally (seed [base, rep, quality, 10+k]) | A500: SIG's batch weights permuted by a fixed per-step key; FP: global |
| STRAT_k | permuted within deciles of the placement's leverage at K_0: P1 ℓ_i = ‖π_0(s_i) − a_i‖·‖s_i‖; P2 ℓ_i = ‖∇ₐQ1(s_i, π_0(s_i))‖·‖s_i‖ | same deciles (A500: within-batch rows that share a decile) |
| ANTI | reversed ranks of SIG | reversed |
| NUIS | ranks of Ŵ_clean(s_i, a_i) | Ŵ_clean(s_i, π(s_i)) |
| OR1 (same information class as SIG) | \|e1(s_i, a_i)\| | \|e1(s_i, π(s_i))\| |
| OR2 (placement ceiling, at π) | P1: ‖∇ₐe1(s_i, π(s_i))‖; P2: 1 − cos(∇ₐQ1, ∇ₐQ^π) at (s_i, π(s_i)) | same |

**OR\* (diagnostic, not rank-transformed).** The exact-cancellation weights of section 4 (ii)-(iv), used for the derived checks only.

**P3 arms.**
- SIG (Ŵ), at matched c and at the natural c = 1;
- NUIS (Ŵ_clean), at matched c and at c = 1;
- OR (|e1(s, a)| as the penalty function);
- EXACT.

**References in every block.**
- NONE;
- the P0 frontier;
- HOOK and P0 at S(HOOK);
- the same-knob table: SIG, SHUF and the constant at β ∈ {0.3, 1, 3}. This is the user's literal "constant strength (same mean)".

**In negative controls (e1 ≡ 0),** OR arms are not applicable, and NUIS ≡ SIG in clean.

## 7. Blocks, data modes and cases

| Block | Data | Start K_0 | Quality | Cases | Cells | Replicates |
|---|---|---|---|---|---|---|
| R (continuity, natural cases) | step-3 episodic | K_b + offset·D, offsets {0.5, 1, 2} | good, poor | clean, noisy_reward, q1_optimistic κ{0.5, 1, 2}, shared_bias κ{0.5, 1, 2}, independent_errors (+ q2_optimistic on rep 0 as a bitwise check) | 54 | 10 (reps 0-4 = step-3 seed 20261001) |
| X-CS (main) | common states | 0.5K* + 0.5D (J −2.7824, gap 0.4873) | expert, mixed, medium, poor | clean, noisy_reward, tilt a* ∈ {0.9, 0.5, 0, −0.5}, q1_optimistic κ1, shared_bias κ1, independent_errors, cone κ{1, 2} | 44 | 10 (seed base 20261002) |
| X-EP (state-covariance pathway) | each level's own episodes | same | same | same | 44 | 10 |
| C (localizable actor) | common states, 4 contexts | K_c = K_0 for every c | expert, poor | clean, q1opt-local κ1, tilt-local a* −0.5 | 6 | 10 |

The pilot uses replicate index 99 and is excluded from every analysis. Every block runs the q1 signal only.

**Quality levels** (the BC target's J is in brackets):

| Level | Behaviour | J of the BC target |
|---|---|---|
| expert | K_b = K* | −2.2951 |
| mixed | per row (CS) or per episode (EP), 50% K* and 50% 0, mode stored; K̄_b = 0.5K* | CS least-squares target 0.5K*, J −2.4985 |
| medium | 0.1338·K* | −3.3641 |
| poor | 0 | −4.4335 |

In CS mode, mixed tests bimodal demonstrations. Its BC target is better than the start (J −2.7824).

**Common-state (CS) data** (one replicate):
1. Roll out K_0 with behaviour noise σ_b = 0.2 for 200 / 200 / 400 episodes of length 50 (train / calibration / evaluation). This gives states s.
2. Draw n ~ N(0, I) and ε ~ N(0, Σ_ε) once per row.
3. For each level q: a = −K_b,q·s + σ_b·n, r = r(s, a), s' = A s + B a + ε, done = 0.
4. The calibration bank uses `stratified_bank` on the calibration episodes, as in step 3.

States, n and ε are bitwise identical across levels. Within a replicate, data and keys are shared across cases (as in step 3). Why K_0 rollouts:
- the Q-term gradient on the data then approximates the on-policy policy gradient;
- tr Σ is 0.136, between expert (0.105) and poor (0.321).

**Episodic (EP) data.** Step 3's `data()` generalized to the four gains (mixed: mode per episode).

**Error definitions** (Q2 = Q^π unless stated; K̄_b is the population behaviour gain):

| Case | e1 = Q1 − Q^π | Notes |
|---|---|---|
| q1_optimistic | κ‖a + K̄_b s‖² | X only at κ = 1: FP Hessian PD at κ ≤ 1, not PD at κ = 2 |
| shared_bias | b = Q^π of κ‖a + K̄_b s‖², both heads | as step 3 |
| independent_errors | c·zᵀW_k z, scale set so median \|Q1 − Q2\| = median \|t − min\| | as step 3; scale set per level on its own rows |
| tilt(a*) | (a + K̄_b s)ᵀ T s | see below |
| cone(κ) | κ·ς(s)·‖a + K̄_b s‖² | see below |
| q1opt-local / tilt-local (block C) | 1[c = 0] times the corresponding error | for tilt, G_true and alignment are computed on context-0 rows |

**Tilt construction.**
- Ĝ = G_true/‖G_true‖, where G_true = mean_i ∇ₐQ^π(s_i, π_0(s_i))(−s_i)ᵀ on the training rows (the harness's alignment_K convention).
- Ĝ⊥ = the Gram-Schmidt residual of a seeded Gaussian 2×3 matrix.
- G_e = c·‖G_true‖·(−Ĝ + Ĝ⊥)/√2, and T = −G_e Σ̂⁻¹, with Σ̂ = mean s sᵀ.
- c comes from (1 − x)/sqrt((1 − x)² + x²) = a*, with x = c/√2: c = 0.461, 0.897, 1.414 and 3.346.
- The error is representable in the existing W[0] slot: W = [[sym(K̄_bᵀT), ½Tᵀ], [½T, 0]].
- H_aa is unchanged, so the FP is always well-posed.
- In X-CS, T is identical across levels within a replicate.
- e1 can be negative, so the target's min picks it up through the bootstrap.

**Cone construction.**
- ς(s) = sigmoid(10(ŝ·v − c0)), with v = (1, −1, 1)/√3 and ŝ = s/‖s‖.
- c0 is set by bisection so that mean ς over training states is 0.25 (Gaussian approximation c0 ≈ 0.390).
- It needs a new critic field: ς is not polynomial in s.
- e1 ≥ 0, so min(Q1, Q2) = Q^π bitwise and the target is unchanged.

**Block C details.**
- obs = [s, onehot(c)]; the context is drawn uniformly per episode and persists into next_obs.
- The actor is K ∈ R^{4×2×3}, with π(obs) = −Σ_c o_c K_c s.
- ContextQuadraticCritic uses H_c, h_c = `quadratic_q(system, K_c)` plus the per-context error.
- J_total = mean_c J(K_c).
- The calibrator input dimension is ds + 4.
- The FP is per context (6×6 each), with λ_0 global, as TD3+BC computes λ over the whole batch.

## 8. Evidence sets (outcome-free, frozen before any J)

**Stage 0.** For every cell and all 10 replicates, with no σ fit, no actor run and no J:
- alignment_K at K_0, the harness's definition;
- d_crit = ‖K_fp(EXACT) − K_fp(NONE)‖_Σ, computed from K's only;
- cos(BC pull, Q^{π0} ascent) = cos((K_bc − K_0)Σ̂, +∇_K mean Q^π(s, −Ks) at K_0);
- the cone's cos(D_qΣ̂, D_qΣ̂_ς).

The results are written to `stage0.json` and hashed.

| Set | Definition |
|---|---|
| N (negative controls) | clean, noisy_reward |
| M (misleading) | mean alignment_K ≤ 0.5 and mean d_crit ≥ 0.01 |
| W | 0.5 < alignment_K < 0.9 and d_crit ≥ 0.01 |
| H (no headroom) | Q1 error present, not in M or W |
| P (positive controls) | cone (X), block-C localized cells (always P, whatever their alignment) |
| M_nat | M cells from natural cases (q1_optimistic, shared_bias, independent_errors) in blocks R and X |

In step 3, M had 10 of the 72 q1 cells:
- good κ2 q1_optimistic: −0.842, −0.702, 0.291;
- good κ2 shared_bias: −0.861, −0.885, −0.936;
- good independent_errors: 0.055, 0.188, 0.042;
- poor shared_bias κ2 offset 2: 0.205.

Tilt a* ∈ {0.5, 0, −0.5} is in M by construction at every level.

## 9. Contrasts, statistics, gates, verdicts and decision rules

**Normalization.**
- Every J difference is divided by the cell's gap g = J_opt − J(K_0): 0.4873 in X and C, 0.274-18.0 in R.
- G(arm) = J(arm) − J(NONE).

**Contrasts at matched S** (SHUF and STRAT averaged over their permutations):

| Contrast | Definition |
|---|---|
| T_SHUF | G(SIG) − G(SHUF) |
| T_STRAT | G(SIG) − G(STRAT) |
| T_NUIS | G(SIG) − G(NUIS) |
| T_ANTI | G(SIG) − G(ANTI) |
| R | G(SIG) − G(P0 at S*): redistribution beyond uniform strength |
| Str | G(P0 at S*) − G(NONE): the uniform component |
| I | G(OR1) − G(SHUF) |
| C | G(OR2) − G(SHUF) |
| P3 contrasts | T_NUIS, R, and C3 = G(P3-OR) − G(P0 at S*) |

G(SIG) = Str + R holds exactly.

**Unit and tests.**
- The unit is the replicate. For each (placement, contrast, evidence set, quality, regime), the pooled value is the mean over the set's cells within a replicate, giving 10 paired values: mean, SE and t with 9 df.
- **Materiality:** |pooled| ≥ 0.01 (1% of the gap).
- **Holm (familywise 5%):** one family per placement, over {T_SHUF, T_STRAT, T_NUIS, R} × 4 levels in X-CS M at FP: 16 tests each for P1L, P2L, P1π and P2π, and 8 for P3 ({T_NUIS, R} × 4).
- Cell-level "meaningful" (|mean| > 2 paired SE and ≥ 0.01·g) is reported as a secondary count.
- **A500 concordance:** an A500 pooled contrast is "materially opposite" if it has the opposite sign with |value| ≥ 0.01.

**Tolerance policy for [derived] items.**
- A miss beyond the stated tolerance fails G0 (a bug).
- Items marked "[derived up to float32/eps]" are reported. They stop the analysis only when the miss is above 100× the tolerance.

**Gates (read first).**
- **G0:** every derived check passes.
- **G1, negative controls (X-CS, FP):**
  - for each weighting placement, T_STRAT is material and Holm-significant (an 8-test family: 2 cases × 4 levels) in at most 1 of 8;
  - for P3, |R| is material in at most 1 of 8. Otherwise P3 is flagged "nuisance-confounded" and is ineligible for D4RL;
  - T_SHUF may be nonzero (leverage or heteroscedasticity).
- **G2, decisive positive control (block C, FP, expert):** C = G(OR2) − G(SHUF) ≥ 0.01, with one-sided p < 0.05, pooled over the two localized cases, for P1L or P2L. If it fails, the harness cannot show targeting: stop and redesign.
- **G2b (cone, X-CS, FP, expert):** reported. A failure means "the linear actor cannot exploit anisotropic localization at these settings". It is not a stop.
- **Pilot power check** (replicate 99, excluded; J computed only for OR1, OR2, EXACT and P0 in block C and cone cells):
  - if C(OR2) < 0.03 for both P1L and P2L at expert in block C, double κ, at most twice;
  - for the cone, step up the κ set {1, 2} → {2, 3} once, keeping the FP Hessian PD;
  - the outcome and any retune go in the pilot addendum before freezing.

**DR1, verdict** per (placement, M, level). FP is primary. Rules are applied in this order of precedence:
- **V0 not testable:** fewer than 50% of M cells reachable at the primary level.
- **V1 targeting real:** all of
  - T_SHUF, T_STRAT and T_NUIS > 0, material and Holm-significant;
  - R ≥ 0.01 and Holm-significant;
  - pooled G(SIG) > G(SHUF) > G(ANTI);
  - no A500 contrast among T_SHUF and R is materially opposite.
- **V2 targeting below uniform:** the T-conditions of V1 hold, but R < 0.01.
- **V3 optimizer-dependent:** FP meets the T-conditions and A500 is materially opposite, or the reverse. But if A5000 agrees with FP, the FP verdict stands, labelled "short-horizon reversal".
- **V4 leverage-weighted regularization:** T_SHUF > 0, material and significant; T_STRAT immaterial.
- **V5 nuisance weighting:** T_SHUF and T_STRAT > 0, material and significant; T_NUIS immaterial.
- **V6 harmful assignment:** T_SHUF or T_STRAT < 0, material and significant.
- **V7 strength only:** Str material, while R and T_SHUF are immaterial.
- **V8 nothing.**

For P3, V1 needs T_NUIS > 0 and R ≥ 0.01, both material and significant.

**DR2, information diagnosis** (M, FP):

| Pattern | Reading |
|---|---|
| C immaterial | the placement cannot use perfect information here |
| C > 0, I immaterial | logged-action (L) or proposed-action (Π) error magnitude is the wrong information |
| I > 0, T_SHUF immaterial | σ is too weak a proxy |
| T_SHUF > 0 | see DR1 |

**DR3, carry to D4RL Stage 4B-1.** A placement qualifies only if all of these hold:
- V1 at the expert or medium level in X-CS;
- G1 passed;
- on M_nat, the SIG's pooled T_SHUF ≥ 0.01 (one-sided p < 0.05) and pooled R ≥ 0. Use FP where it is finite in every M_nat cell, otherwise A500;
- for P3, not nuisance-confounded;
- for the Π family: step 5's coverage at π(s) on real tasks must come first.

If none qualifies, 4B-1 is not proposed, and 4B-0 answers whether the hook is an α retune.

**DR4, recommendation under poor data.** Among V1 placements, prefer one with R ≥ 0 at the poor level. If every V1 placement has R < 0 there, none is recommended for poor-quality data, however good the signal is.

**DR5, step order.**
- If "information class wrong" holds in the majority of (placement × level) M contrasts, step 5 moves ahead of 4B.
- If "signal too weak" holds in the majority, σ quality and step 6 move ahead.

**DR6, the user's hypothesis** ("correct identification, harmful placement"), per placement in X-CS M. Confirmed if both hold:
- **(a) Identification.** Either C > 0, or I > 0 or T_SHUF > 0 (material) at expert; and Spearman(SIG weight, |e1| at the family's evaluation point) ≥ 0.3 at all four levels.
- **(b) Harm.** At poor, G(SIG or OR2, at S2) < −0.01, with Str < −0.01 and |R| < |Str|.

**Placement interaction:** [R(P2) − R(P1)]_poor − [R(P2) − R(P1)]_expert ≥ 0.01, within the same family, and greater than 2 SE.

**DR6b, mixed-data pathway** (X-EP mixed, P1L). Confirmed if all hold:
- corr(SIG weight, poor-mode indicator) ≥ 0.05;
- T_SHUF < −0.01;
- G(OR1) − G(SHUF) < −0.01;
- G(SIG) − G(NUIS) < −0.01.

**DR7, the hook.** If |G(HOOK) − G(P0 at S(HOOK))| is immaterial in at least 90% of cells in every block, the hook is reported as an α retune.

## 10. Order of operations and compute

1. **Save the post-hoc scripts in the repo and hash them:**
   - `experiments/signal/strength_reanalysis.py`: the per-replicate BC-norm ratios, the strength-only model with its sign-equivalence note, and the alignment_K table;
   - `experiments/signal/step4_settings.py`: the multiset statistics, J values, tilt c, cone c0 and cos, cos(BC pull, ascent) with ascent = +∇_K E Q, and FP definiteness. My scratch `synth_pull.py` printed the descent cosine; fix its sign.
2. Write the pre-registration (`runs/wbcp_signal/step4/expectations.md`) with a UTC timestamp and SHA-256. Mirror it to the advisor page.
3. Stage 0 (minutes): `stage0.json`, hashed.
4. Pilot (replicate 99): derived checks, reachability, timing, outcome-free Level-1 and FP targeting shares, and the power check. Write a timestamped addendum, and freeze and hash `scorecard_step4.py`.
5. Ask the user for the main-run compute.
6. Main run: one process per (block, level, replicate), with `jax.clear_caches()` per process (the vm.max_map_count lesson). Then write and hash `matched_knobs.json`, run `score_values.py` (J), run the scorecard, and get an independent audit of the write-up.

**Compute estimate** (unmeasured; extrapolated from step 3: 1.13 s per fit plus 5 actor runs).
- Main run: about 1,480 cell-replicates. Per cell-replicate:
  - one σ fit, about 1.1 s;
  - FP for P0, P1 and P2: about 1 s;
  - P3 BFGS (X-CS and C only): about 10-30 s;
  - A500: about 650 vmapped actor runs, about 5-15 s.
- A5000 and full-batch subsets: about 1-2 CPU-h.
- Total: about 6-12 CPU-h, about 1-2 h of wall time in 6-8 processes.
- Reduced plan (X-CS and C in full; R and X-EP with FP plus A500 for anchors, P0, and P1L/P2L SIG/SHUF/STRAT/NUIS only): about 3-5 CPU-h.
- CPU only.

## 11. Carrying to D4RL

**4B-0: no training** (CPU, minutes per pool).
- Pools: the 5 healthy TD3+BC pools, with step 2's frozen q1 σ and threshold (`runs/wbcp_signal/frozen/<pool>/q1/`) and the parent checkpoint, on 100k training rows at the frozen actor.
- Weights:
  - the rank-normal multiset;
  - T-lin CV;
  - **exact per-row leverage**: per-example ‖∇_θ‖π_θ(s_i) − a_i‖²‖ via vmap(grad), and the analogous Q-term norm;
  - Spearman(SIG weight, leverage).
- Level-1 Δg for P1L, P2L, P1π, P2π and P3 (c at matched ρ, since there is no FP on an MLP), with SIG, SHUF×8, STRAT×8 and ANTI. Report:
  - cos(Δg_SIG, Δg_P0), m_eff and τ;
  - whether SIG's increment lies outside the SHUF cloud.
- HOOK: cos(Δg_HOOK, Δg_P0).
- Data quality:
  - corr(weight, episode normalized return) on every pool;
  - on halfcheetah-medium-expert, the expert-half share of P1L's extra weight.
- 4B-0 describes only. One exception: if cos(Δg_SIG, Δg_P0) ≥ 0.98 on every pool for a placement, D4RL training cannot separate targeting from strength for it, and 4B-1 is not proposed for it.

**4B-1: actor-only fine-tuning on frozen critics** (needs the user's permission; DR3 must pass).
- Setup:
  - start from the 100k-update checkpoint actor;
  - critic, targets, σ and threshold frozen;
  - a fresh Adam state, identical for every arm, with the same seed;
  - 20k actor steps.
- Arms:
  - NONE;
  - P0 at m ∈ {1.3, 2, 4};
  - each eligible placement × {SIG, SHUF×4, STRAT×4 (deciles of exact per-row gradient norms), ANTI}, matched on S measured on held-out dataset states (RMS ‖π_arm − π_NONE‖) through a 4-point knob ladder plus interpolation.
- Returns: 100 episodes per seed, common evaluation seeds, paired by seed.
- Contrasts:
  - **Primary: SIG − mean(SHUF).** They share the multiset and the matched S, so the frozen-critic bias toward data pull applies equally.
  - Secondary: SIG − P0.
- Seeds: chosen from the pilot's paired SD so that the MDE is ≤ 2 normalized points.
- Pools: halfcheetah-medium-expert and hopper-medium first.
- Scope: there is no NUIS or oracle on D4RL. A win is "assignment beats leverage-matched shuffles", consistent with critic-error targeting but not proof of it.
- The quality cross needs TD3+BC pools on hopper-random and halfcheetah-random (cached; about 12 min of CPU each), which needs permission.

**4B-2: full training** (permission; GPU is approved only for CQL/ReBRAC pools).
- Proceed only if 4B-1 gives SIG − SHUF ≥ 2 normalized points and greater than 2 paired SE on at least 2 pools, with none negative.
- Compare the placement against P0 with α re-set to match the realized BC share, on 5 pools × 5 seeds.
- When critics train, three things change:
  - S* and ρ* must be re-matched at each refresh;
  - π shapes the critic through the target actor;
  - a penalized-backup critic must fit its band on an unpenalized head; otherwise the width certifies a critic shaped by the width.

## 12. Build spec

**Do not edit `lq_harness.py`.** Block-R anchors call it directly, which keeps bitwise continuity. New code goes in `experiments/signal/placement_harness.py`, which imports `lq_harness`.

1. **Data**
   - `data_common(h, quality, rep, contexts=None)` and `data_episodic(h, quality, rep)`;
   - quality gains;
   - the mixed mode vector, and K̄_b and K_bc;
   - tie-break permutations.
2. **Critics**
   - `case_critic_x(...)`: reuses `lq_harness.case_critic` for the old cases and adds "tilt" (T into W[0]) and "cone".
   - `PlacementCritic.apply`: QuadraticCritic plus the optional cone term. Use Python-level branching on parameter keys, so cases without a cone compile to QuadraticCritic's program, and test bitwise against it.
   - `ContextQuadraticCritic` and `ContextLinearActor` (one-hot weighted sums).
   - Matching NumPy `head_error_x`.
   - `tilt_matrix(G_true, Sigma_hat, a_star, rng)`, `cone_c0(states)`.
3. **Weights**
   - `rank_normal(x, s, clip, tie_perm)`, `batch_rank_normal(x_b, s)`;
   - `assignments(cell)`, returning SIG, SHUF_k, STRAT_k, ANTI, NUIS, OR1 and OR2 per family;
   - `leverage_P1` and `leverage_P2`;
   - `or_raw(t)`.
4. **Actor update.** `placement_step(args, models, state, batch, key, bc_w, q_w, pen_fn, pen_c, exact_fn, pi_eval)`, the actor part of `td3_bc_update`:

       pi = actor(params, obs); q = critic(critic_params, obs, pi)[..., 0]
       lam = alpha / sg(mean |q|)                       # host lambda, never weighted or penalized
       q_read = exact_fn(obs, pi) if exact_fn else q
       if q_w is not None: q_read = sg(q_w) * q_read    # P2 (q_w recomputed at pi under sg for P2pi)
       q_term = q_read.mean() - (pen_c * pen_fn(obs, pi).mean() if pen_fn else 0)
       bc = (sg(bc_w) * square(pi - a).mean(-1)).mean()  # weighted path; NONE = bc_w ones, q_w ones, pen_c 0
       loss = -lam * q_term + bc

   - Then Adam; the critic, target critic and target actor are restored, as in `actor_loop`.
   - `placement_loop`: `lax.scan`, vmapped over (knob, assignment).
   - Per-step diagnostics: term gradient norms, ‖ΔK‖, Adam SNR and Kish.
   - Skipping the critic update is bitwise-safe, because the harness critic uses `optax.set_to_zero()`; test it.
5. **FP**
   - `fp_rows(critic_params, obs)` → P_i, q_i (float64);
   - `fixed_point(P, q, S, a, u, v, lam0)` → K, minimum eigenvalue, condition number;
   - `fixed_point_penalty(...)`: BFGS, warm-started along the c ladder;
   - `lstd_penalty(context rows, p)`: block C.
6. **Strength**
   - `strength(K_arm, K_none, Sigma_ev)`;
   - `match_knob(ladder, S_of_knob, target, tol, max_bisect)`, returning the knob, S, a monotone flag and reachability;
   - `frontier(...)`, `regret_alpha(...)`.
7. **Level 1.** `terms(...)`: weights and penalty; the projection onto span(g_Q, g_BC) giving m_eff and τ; the Adam-preconditioned increment.
8. **Stage 0.** `stage0(...)`, which must not import `value`.
9. **Outputs.** K's and metrics in `.npz`/JSON per cell × replicate × regime; `score_values.py` computes J afterwards.
10. **Tests:** `test_placement_harness.py`, covering every [derived] item in the pre-registration:
    - zero-knob identity;
    - bitwise equality with `td3_bc_update` when hooks are off;
    - same multiset;
    - Adam and FP scale equivalence;
    - λ-lock;
    - the EXACT identities;
    - the q1_optimistic, cone and block-C cancellation identities;
    - tilt alignment;
    - common-state identity;
    - LSTD in-class exactness;
    - the context critic against Monte Carlo and the Bellman identity;
    - `PlacementCritic` equal to `QuadraticCritic` bitwise without a cone.

**Later, only if a placement is carried:** hooks in `algorithms/td3_bc.py` (`q_weight`, `actor_penalty`) and fields in `td3_bc_bca.Config` (placement, β, s, evaluation point, S* refresh).

## 13. What cannot be fixed, and why

- **The linear actor in blocks R and X can act only through 3×3 weighted second moments.** A null there means "not separable in a linear actor", not "no targeting". Block C tests a localizable actor, but contexts are observed and discrete. On D4RL, MLP actors can localize, so an LQ negative does not transfer as a negative.
- **The errors are constructed.** Tilt is adversarial by design, and only independent_errors and block R's κ = 2 cells are natural misleading cases.
- **A critic-side penalty propagated by Bellman backups cannot be tested for targeting in single-context LQ.** The quadratic critic class smears it into a global quadratic. It is tested only in block C (C2) and, optionally, in 4B-2.
- **FP freezes π-evaluated weights at π_0, while A500 recomputes them.** The difference is reported.
- **The λ-lock removes a side channel the real method has.** Its size is reported through P3 with native λ.
- **The Π family reads σ at actions with no coverage guarantee.** Carrying it requires step 5 first.
- **D4RL has no NUIS and no oracle.** Exact Σ-metric strength is replaced there by held-out action RMS.
- **The 1% materiality floor is a judgment.**
