# BCA as implemented: a component map for restructuring (14 Oct 2026)

Tags: **[M]** measured, meaning read from code or a saved file. **[I]** inferred. **[S]** speculative.

Path shorthands:
- `S2` = outputs/shift/beyond-the-data/stage2/code
- `R2` = runs/shift_explore/stage2
- `G` = runs/shift/68d9e8779e148b7b
- `X` = runs/shift_explore
- `IMP` = the scratchpad's impact/ folder

Paths without a prefix are relative to the repo.

Scope: only gate seeds 0-4 have artifacts. No seed 5-9 artifact exists, because there is no `R2/fresh/`. **[M]**

Every Stage 2 policy, host and deployment output for seeds 0-4 used the **weighted** version. **[M]** Sources: `R2/policy/seed*/lock.json:version` and `R2/host/*/seed*/lock.json:version`, both "weighted".

## 1. What gets "aggregated"

**[M] No model is aggregated.** Each band uses one critic, and that critic's two heads are combined by a minimum:
- the target: `algorithms/td3_bc_bca.py:151-155`;
- the scale fit: `td3_bc_bca.py:170-174`;
- the refresh: `td3_bc_bca.py:252-256`.

The two Bayesian weightings act on data rows, not on models:
- **Scale-fit masses.** Bayesian-bootstrap masses: Dirichlet(1,…,1) over the 256 batch rows, drawn as normalised Exp(1) values (`calibration/network.py:44-46`, `td3_bc_bca.py:175-177`).
- **Threshold posterior.** The WBCP posterior: for each of 1,000 draws, masses w_j·E_j on the bank scores plus a test atom E_{n+1}·t (`calibration/wbcp.py:43-54, 104-111`).
  - R = max(λ̂, λ_HPD) (`wbcp.py:89-102, 115-117`), where λ̂ is the weighted empirical 90% quantile and λ_HPD is the 95%-credible crossing.
  - As written, the weights are uniform: w_j = 1 and t = 1 (`calibration/reference.py:176-179`).

**[I]** A critic ensemble, or a combination of several bands, would be new machinery.

## 2. Pipeline (TD3+BC host; Stage 2 restates it and imports BCA's functions unchanged)

| # | Component | file:line | What it computes |
|---|---|---|---|
| 1 | Residual target | `td3_bc_bca.py:146-156` | y = r + γ(1−d)·min_h Q̄_h(s′, clip(π̄(s′) + clip(N(0, 0.2²), ±0.5), ±1)). In Stage 2's "fixed" bands, π̄ is the stall-aware controller and Q̄ is the same frozen critic (`S2/band.py:214-222`). "Host" bands use the host's own target networks (`band.py:416-421`). q = min_h Q_h(s, a_recorded). |
| 2 | Conformity score | `reference.py:148-149, 175`; `band.py:307-314` | \|y − q\| / (max(η_f(s, a), 1e-6)·u_f). It exists only at recorded transitions, because it needs r and s′. |
| 3 | Scale network η | `network.py:8-31`; `band.py:114-116` | Dense64-ReLU-Dense64-ReLU-Dense1, output η = softplus(·) + 0.01. Inputs: the normalised state divided by 1.001, concatenated with the 4-dimensional command. |
| 3a | Its training rows and target | `td3_bc_bca.py:159-231` | Recorded batches (s_i, a_i) of 256 rows. Loss: (Σ m_i·softcov_i − 0.9)² + 0.005·Σ m_i·η_i², where softcov = σ(20(y−(q−η)))·σ(20(q+η−y)) on y/u and q/u (`:180-190`, `network.py:34-37`). There is no per-row target: one coverage constraint per batch, plus a shrink penalty. |
| 3b | Unit u | `td3_bc_bca.py:196-198` | Exponential moving average (0.99) of std(y − q) over the batch. |
| 3c | Fit schedules | `band.py:243-269`; `learners.py:1253-1279`; `S2/host.py:143` | Detection band: 20,000 steps on a frozen critic, rows from default_rng([20261015, rep, 41]). Co-trained host: one step per update. |
| 4 | Freeze / refresh | `td3_bc_bca.py:234-280`; `reference.py:152-194` | Freezes η_f, u_f and R together. The bank has 827-851 rows (`R2/detect/uniform/*/fixed/seed*/band.json:uniform.bank_rows`). In the host: at update 10,000, then every 5,000, which gives 18 refreshes (`learners.py:1162-1164`; `R2/host/stall/seed0/lock.json:records.bca_host.refreshes`). Before the first refresh the width is 0 (`dose.py:90`, `band.py:158`). |
| 5 | Uniform threshold R | `reference.py:176-179` | One scalar. In the stalling world with the fixed critic, R is 1.47-1.52 and every seed certifies (`band.json:uniform.R`, `uniform.certified`). Certification needs only n ≥ 36 rows (`reference.py:124-136`). R_c = R when R is finite (`band.py:291-304`). |
| 6 | Width U(s, a) | `dose.py:74-76`; `band.py:153-158` | R·u_f·max(η_f, 1e-6). Because R and u_f are global, U ranks pairs exactly as η_f does. |
| 7 | U_max, uniform | `band.py:301-303` | R_c·u_f·max_j η_f over the bank: 0.41-0.44 in the stalling world with the fixed critic (`band.json:uniform.U_max`). |
| 8 | "Cannot certify", uniform | `band.py:44-48, 155-157` | U = U_max everywhere. The paired penalty is then 0, τ = U_max, and the check never fires. **[M]** This cannot happen with ~850 bank rows. |
| 9a | Weight classifier | `S2/weights.py:45-55, 78-100` | Same shape as the η network; its output logit is log w. Target pairs (s_i, clip(π(s_i) + a_i − c(s_i), ±1)), where c is the cloned actor, are labelled 1; recorded pairs (s_i, a_i) are labelled 0. Training: 2,000 Adam steps of 512 rows, learning rate 1e-3, with a fixed initialisation and stream (`:40-41, 61-71`). It is refitted with: the gapped critic's actor for detection (`band.py:538`); each arm's actor before every 4,000-step chunk in the policy test (`policy.py:135-145`); the current actor at every host refresh (`host.py:222-231`). |
| 9b | Threshold table | `band.py:327-357` | The bank is split into a threshold half and an audit half; the audit half's coverage is 0.86-0.98 (`weighted…/band.json:weighted_fit.audit.covered_share`). wbcp.calibrate runs with w_j = exp(log w_j) at 48 log-spaced test masses t, with common random numbers. The grid runs from the smallest bank log-weight minus 1 to log(α/(1−α)·Σw) + 3. R_c(t): +∞ is replaced by the largest finite value, and the table is read by linear interpolation at the query's log w (`:167`). |
| 9c | l* | `band.py:355-356` | The log of the smallest test mass at which R = +∞: 2.68-4.26 in the stalling world (`weighted_fit.l_star`). |
| 9d | Refusal term | `band.py:168` | U_max·softplus(log w − l*). |
| 9e | U_max, weighted | `band.py:360-371` | max of R_c(w_j)·u_f·η_f over threshold-half rows with log w < l*: 0.32-0.80 in the stalling world. |
| 9f | Ũ(s, a) | `band.py:161-169, 484-519` | R_c(w)·u_f·max(η_f, 1e-6) + U_max·softplus(log w − l*). η_f and u_f are reused from the uniform fit. |
| 10 | Lower edge | `band.py:176-183` | P = [U(s, π) − U(s, a_i)]₊; the actor climbs Q1 − k·P. Gradients pass through the command input of the frozen η_f only (uniform), and also through the classifier in both R_c(w) and the refusal term (weighted). The weighted path breaks `ALGORITHMS.md:127-130` (`band.py:38-42`). |

## 3. Consumers

- **As written: the dose.**
  - Dose = 1 + 0.5·U/(U + u_f), multiplying the per-row cloning term. It lies in [1, 1.5); a width of +∞ gives 1.
  - Code: `dose.py:23-62, 65-95`; `td3_bc_bca.py:283-324`; `algorithms/td3_bc.py:132-138`.
- **Policy test (actor only).**
  - Loss: −λ·mean[Q1(s, π) − kW] + mean(π − a)², with λ = α/sg(mean\|Q1\|) and α = 5.
  - kW = k_p·P + k_u·U + k_s·P, where the last term is read at a permuted state (`policy.py:68-93`).
  - Setup: 20,000 steps from the saved cloned actor, against the frozen fixed-controller critic of the gapped log. Arms: `policy.py:393-397`.
  - U(s_i, a_i) is computed once: `policy.py:379` (uniform), `band.py:511` (weighted, at each refit).
- **Full host.**
  - td3_bc_update, with the actor loss −λ·mean(Q1 − kW) + cloning term; the critic update is unchanged.
  - fit_scale runs every update, refreshes follow the schedule in row 4, and the extras are recomputed at each refresh (`host.py:62-111, 140-162, 205-232, 237-288`).
  - Rival, the range hinge: 100·Σ([π − hi]₊² + [lo − π]₊²) (`host.py:114-115`). Bounds are the 0.5% and 99.5% quantiles of the training commands (`X/room/code/harness.py:530-533`).
- **Deployment check.**
  - Rule: if U(s, π(s)) > τ, or U is not finite, clip the command to the recorded range (`deploy_check.py:108-150`; trigger at `:127-128`, clip at `:135-136`).
  - τ is the split-conformal 95% level of U on held-out calibration-split rows outside the gap, the bank and the T4 rows (`deploy_check.py:56-105`). Seed 0 uses 10,986 such rows (`R2/detect/uniform/stall/fixed/seed0/tau.json:heldout_rows`).
  - Bands used: the weighted fixed-critic band for none and bca_k1; the uniform host band for td3bc (`S2/run_stage2.py:528-560`).
  - τ for none at seed 0 is 0.446 (`R2/deploy/stall/seed0/values.json:taus.none`).

## 4. Facts that constrain a redesign

- **[M]** Every factor of the width can be evaluated at any (s, a). Only y, and so the score, needs r and s′ (`outputs/shift/beyond-the-data/mechanism.md:158-173`).
- **[M]** The room test's perfect band is the exact one-step error through the simulator: δ*(s, π) = r + γ·min Q(s′, c(s′)) − min Q(s, π) (`X/room/code/harness.py:323-336`). BCA's fit never sees this information.
- **[M]** η_f is nearly flat over the bank, and only weakly tracks the residual it is meant to scale:
  - coefficient of variation 0.065-0.111 (`IMP/extra.json:[*].eta_bank_cv`);
  - correlation between η_f and \|y − q\| of 0.04-0.43 (`IMP/extra.json:[*].bank_corr_eta_vs_abs_resid`).
- **[M]** The uniform width does not respond to the command at the region states (T1, stalling world):
  - median U at the command, the 80% cap and the recorded command lies in 0.310-0.356;
  - the medians of the command-minus-cap difference lie between −0.0045 and +0.0020 (`IMP/decompose_exact.json:[0-4].T1.uniform`).
- **[M]** At harmful T1 pairs, the weighted version's extra width comes from:
  - the refusal term: 87%;
  - BCA's own residual network η: 0.2%, pooled over seeds 0-4 in the stalling world;
  - the threshold table (R_c): the rest.

  Source: `IMP/decompose_exact.json:[0-4].T1.diff_split`, as term2_sum ÷ weighted_sum and term1_from_eta ÷ weighted_sum. At seed 1 the table is flat at 2.86 (`weighted_fit.R_c`), so the threshold term adds nothing there.
- **[M]** At seed 0, on 2,000 training states in the full-throttle region, the weighted penalty has the right shape:
  - the mean penalty W rises from 0.11 at a 70% command to 2.21 at 100%;
  - the best command under Q1 − W is past 80% at 5.9% of states, against 57% under Q1 alone (`R2/policy/seed0/lock.json:bca_penalty_shape`).
- **[I]** So at training states the penalty is already signed correctly; the losses come from strong-current states visited in rollouts, outside the classifier's training range.

## 5. Saved artifacts for seeds 0-4 that can be run forward without training

| What | Path | Contents |
|---|---|---|
| Data | `runs/shift/pre/data/base/{stall,no_stall}/seed*` | Loaded through A.load_seed and prepared("gap"): 42,500 training rows, 827-851 bank rows, 5,000 evaluation states (`band.json:data`). |
| Fixed-controller critics (check 1) | `G/critics/base/stall/{stall_aware_full, stall_aware_gap, stall_aware_random, stall_aware_full_placebo, rising_thrust_full, rising_thrust_gap}/seed*/critic.msgpack` and `G/critics/base/no_stall/{stall_aware_full, stall_aware_gap}/seed*/critic.msgpack` | 8 critics per seed, 100,000 updates each, none failed. "Placebo" means another initialisation of the critic (`learners.py:619-631`). Their errors on the pools are in `G/harm_critic/base/seed*/rows.npz`. |
| Plain TD3+BC, co-trained | `G/hosts/base/stall/{gap,full}/alpha{2.5,5}/plain/seed*/snapshot_{050,100}.msgpack` | The whole training state: actor, critic and both target networks. Stalling world only. The "host" band was fitted on gap / α 5 / snapshot_100. |
| Cloned actors | `G/clone/base/{stall,no_stall}/seed*/actor.msgpack` | |
| Check 2 actors | `G/harm_return/base/alpha{2.5,5}/seed*/actors.msgpack` | clone, full, gapped, random, host_swap_{full,gapped}, td3bc_gapped, no_stall_{clone,full,gapped}. |
| Check 3 (the gate's room) | `G/room/base/alpha5/seed*/actors.msgpack` | 13 actors with BCA-shaped doses: harm_placed, cap2, even_x*, strength_matched_even, amount_matched, shuffled_0/1, reversed, none. |
| Room test | `X/room/seed*/lock_actors.msgpack` | 48 actors: the perfect-band variants, range_hinge, state_shuffled_band, the matched ladders. Also `no_stall/` (7) and `replication/`. |
| Stand-ins | `X/standins/seed*/lock_actors.msgpack` | 54 actors: ladders of classifier_ratio, density, distance, two_head (with their _plan forms) and range_hinge. Also `no_stall/` (11). |
| Detection pairs | `X/detect/seed*/{stall,no_stall}/pairs.npz` | 24,500 pairs per seed and world (`X/detect/code/pairs.py:11-30`). |
| Detection labels and scores | `labels.npz` and `scores/*.npz` beside each `pairs.npz` | Labels: harmful, over_pref, true_pref, q1/q2 at the command and the cap, unsupported. Scores: every stand-in, the perfect band and the range hinge. |
| Stage 2 bands | `R2/detect/{uniform,weighted}/{stall,no_stall}/fixed/seed*/` and `R2/detect/uniform/stall/host/seed*/` | Per band: band.msgpack (η_f; the weighted file also holds cls_params, fitted with the gapped critic's actor); band.json; tau.json; scores/bca_width_*.npz (U at all 24,500 pairs). bank_scores.npz (scores and η at the bank) is saved for uniform bands only. |
| Stage 2 policy actors | `R2/policy/seed*/actors.msgpack` (49-53 actors) and `no_stall/` (8) | bca_k1/k2/unpaired/shuffled, w0_check, perfect_band, range_hinge, matched controls, the standin:range_hinge ladder, none, clone, full_none. **Not saved:** the classifiers refitted for each chunk (only their diagnostics, in `lock.json:arms.*.refits`). |
| Stage 2 hosts | `R2/host/{stall,no_stall}/seed*/actors.msgpack` | **Actors only** (`host.py:351, 418`). The BCA host's final critic, η_f, frozen reference and classifier were never saved, so any analysis of the host band needs retraining (approval). |
| Deployment check | `R2/deploy/{stall,no_stall}/seed*/values.json` | Points per rule, fire shares and points per start. |

**Loading.**
- **[M]** `band.load_band` and `WidthReader` both call `recheck(approval)` (`band.py:453-481`).
- **[M]** There is also a read-only route: `flax.serialization.msgpack_restore` on band.msgpack, then Calibrator.apply and ClassifierNet.apply (`IMP/exact.py:44-60`).
- **[M]** Gate artifacts load through `SeedContext` (`X/room/code/harness.py:207-275`).

**What can be run forward, and what cannot.**
- **[I]** Possible without training: new width or score formulas built from the saved η_f, classifiers, the 8 critics per seed and the host critics; evaluated at the pairs, the labels and the bank rows; plus actor readouts and the deployment-check rules on any saved actor.
- **[I]** Needs a fit (user approval): new scale-network targets, classifiers for new actors, any per-chunk policy-test state, and the co-trained host's band.
