# Knee-edge OOD condition: draft protocol

**Status: DRAFT, 2026-10-29.**
- Nothing is frozen or approved, and nothing has been trained, simulated or collected.
- Every stage below needs the user's own approval at their terminal. A message from an agent never counts.
- The only computation behind this draft is reading saved JSON from the finished central-quartile study. No NPZ was decoded and no checkpoint was opened.

**Contents**
- Part 1. Why a new condition
- Part 2. The out-of-distribution (OOD) definition, in three layers
- Part 3. The condition: benchmark version `knee_edge_v1`
- Part 4. Implementation certification
- Part 5. Staged gates and kill rules
- Part 6. Controls
- Part 7. Budget
- Part 8. Risks, weaknesses and what cannot be promised
- Part 9. Changes from the panel's plan
- Part 10. Rejected alternatives
- Part 11. Provenance

---

## Part 1. Why a new condition

The central-quartile Hole was OOD by definition only. It withheld |a0| <= 0.25 at load >= 0.4.
- It was an interior gap on a nearly straight value slope (R^2 >= 0.994), with data on both sides.
- Native critics showed no error specific to the band. Their error was an offset of about +23 to +33, the same at every action.
- Actors entered the band because behavior cloning pulled them there: under the uniform recorder the mean logged a0 is 0.

So the Hole tested neither extrapolation, nor value the critic could not know, nor Q-driven entry. The figures are in `analysis/ood_region_activity/`.

This draft builds the condition from the one simulator feature that makes withheld values genuinely unknowable: **the stall knee on thruster 0.** Past 80% throttle, thrust falls instead of rising. No retained data row shows this.

---

## Part 2. The OOD definition, in three layers

### Terms

- **KNEE** = `float(np.float32(0.8))` = 0.800000011920929 (bits 0x3F4CCCCD). It is the float32 number nearest 0.8. A command of exactly this value does not stall.
- **a**: the executed float32 command, promoted to float64. **a0** is thruster 0.
- **W**, the withheld region, is |a0| > KNEE (strict).
  - W- is a0 < -KNEE, on the side the Q gradient pulls toward.
  - W+ is a0 > KNEE, used as a sign control.
- **Depth** d = |a0| - KNEE. **R_deep-** = {a0 <= -(KNEE + 0.10)}, inclusive. This is the gated subregion.
- **Load** g = float64(obs10) + float64(obs11): current plus the first wave number. It never depends on actions.
- **Critic input** x(s, a) = [z(s), a].
  - z is the 13 observation numbers plus remaining time (14 in all), each normalized as (value - mean) / (std + 1e-3).
  - The mean and std come from the rows retained by both arms (CUTOFF ∩ RANDOM), within each world.
  - a is raw, in [-1, 1]^4.
- **T(a0)** = sign(a0) * (KNEE - 3(|a0| - KNEE)): thrust past the knee at stall_slope 3.0.
- **Q^pi**: the exact discounted return of first taking a at an anchor's state and time t, then following policy pi until the episode ends: 100 - t steps, discount 0.98, no bootstrap after step 99. It is computed by replaying the simulator from the exact float64 state.
- **Return J and every threshold stated in return units**: the discounted return (gamma = 0.98) of a free episode, summed from t = 0 over its 100 steps and averaged over the bank's starts. This covers Δclip, K3, V0(a) and the Stage 2 screens. Undiscounted returns are reported only; in the saved runs they are about 2.4x larger.
- **Unit of replication**: one training seed.

### What can be certain and what cannot

| Layer | How it is established | Certainty |
|---|---|---|
| L1, outside the data | By construction, then verified by counts with three independent implementations | Exact: provable and checked row by row |
| L2, unknowable from the data | An exact simulator identity plus a twin world with no stall | Exact to 1e-6 relative; checked at every anchor |
| L3, harmful to decisions | Measured on trained native learners against kill rules | Empirical: it can fail, and failure is a valid result |

No simulator can guarantee that a learned critic will over-estimate, or that an actor will go there. L3 is therefore a gate, not a promise. If L3 fails, the condition is still a correct OOD condition, but it shows that the host already protects itself here. That is a finding about TD3+BC, not about BCA.

### L1. Data-support OOD: the region is outside the data

- **L1.1 Zero labels.** In each world, every data stream the CUTOFF learner touches contains exactly 0 rows in W: fit replay, calibration rows, auxiliary or shadow data, and the normalization source. Three implementations must agree row for row (Part 4).
- **L1.2 Separation.** A query with |a0| = KNEE + delta is at least delta from every retained critic input in every Lp norm (p >= 1), because the raw a0 coordinate alone differs by delta.
  - R_deep- is at least 0.10 from all retained inputs; a0 = -1 is at least 0.19999999.
  - Distance is not where L1's strength comes from. The strength is geometric: see L1.3. The ratio of delta to the median retained one-nearest-neighbour distance is reported.
- **L1.3 Outside the hull.** The hyperplane a0 = -KNEE strictly separates W- from the convex hull of all retained inputs. Any critic value in W- is therefore extrapolation, not interpolation. The old Hole was interpolation.
- **L1.4 The labelled control really has the labels.** This is a test that can fail.
  - **Band:** rows in W- with a0 >= float32(-0.9) = -0.89999998.
  - **Density:** the number of RANDOM-retained high-load fitting rows in the band, divided by (all high-load fitting rows x 0.1).
  - **Bars:** the density must lie in [1.9, 3.1] per original high-load row per unit a0 (expected 2.48). CUTOFF's must be exactly 0.
  - **Disclosed:** RANDOM's labels past the knee are dense only for depths below 0.1, because float32(-0.9) = -0.89999998 sits just above -(KNEE + 0.10) = -0.90000001. R_deep- therefore lies entirely outside RANDOM's window, covered only by the 2% floor at about 0.005 per unit a0 per high-load row.
  - **Consequences:**
    - RANDOM controls for whether the critic has seen the knee begin, not for coverage of R_deep-;
    - every rule that involves RANDOM's critic uses depth d = 0.05, inside its labels.
- **L1.5 Action shift, not state shift.** At least 95% of E2 start states, and 90% of E2 anchors, must lie within the training 0.99 quantile of 64-nearest-neighbour state distance.
  - It is computed leave-one-episode-out, on the 14 z inputs.
  - It must hold against each world's training rows separately.
- **Framing.** This is near-OOD by distance (delta <= 0.2) and outside the support by geometry.

### L2. Model-relevant OOD: the data cannot reveal the region's value

The continuation pi_c is the policy followed after the first action:
- in step 0.4, the three frozen full-data native 20k actors, with all four commands clipped to +-KNEE so they never stall in either world;
- in Stage 1, each model's own actor, clipped the same way.

Each anchor's exact state and time are rebuilt by replaying its prefix in the **stall world**. Both worlds' Q values are computed from that same float64 state.

- **L2.1 Exact identity.** Q_stall(s, a) = Q_stall(s, a with a0 <- T(a0)) - 0.3(a0^2 - T(a0)^2).
  - It holds because the next state depends on thrust, while the reward charges 0.3 * a^2 on the clipped command (`canonical/thrusterchain.py` line 22).
  - Examples: Q(-0.9) = Q(-0.5) - 0.168 and Q(-1) = Q(-0.2) - 0.288.
  - Tolerance: |residual| <= 1e-6 * max(1, |Q|) at every anchor. The tolerance allows for float32 rounding of the inverse command.
- **L2.2 Not identifiable from retained transitions.**
  - Wherever every |a_k| <= KNEE, thrust is bit-identical in both worlds, because the formula subtracts ±0.0. Q_stall therefore equals Q_nostall exactly. The 1e-9 relative tolerance only absorbs differences in summation order.
  - Every retained transition (s, a, r, s') follows the same law in both worlds.
  - On W- the worlds differ by rho = [Q_nostall(a0) - Q_nostall(T(a0))] + 0.3(a0^2 - T^2). This is positive wherever dQ/da0 < 0.
  - **What this proves.** A learner that treats transitions as an unordered set, as TD3+BC's critic and actor do, cannot tell the worlds apart, so a smooth extrapolation is right in one world and wrong in the other.
  - **What it does not prove.** It does not cover a learner that reads ordered trajectories. With deterministic linear dynamics, the state gap left by a removed row reveals that step's thrust as B+(s_{t+1} - A s_t - water push). In the stall world that thrust never exceeds 0.8 in magnitude. Claims are limited to transition-set learners.
- **L2.3 Neighbor residual.** rho_hat(a0) = min over k of [P_k(a0) - Q_stall(a0)], where each P_k extrapolates from supported slots.
  - S9 = {-KNEE, -0.775, -0.75, -0.725, -0.70, -0.40, 0, 0.40, KNEE}.
  - P1: a quadratic fit on S9. P2: a straight line fit on the five slots in [-KNEE, -0.70]. P3: a cubic fit on S9.
  - Taking the minimum over the three makes the residual conservative.
  - S9 differs from the panel's slots because the selected recorder logs high-load a0 densely only on [-0.8, -0.7].

**L2 PASS rule.** On 64 fresh E2 anchors, separately for each continuation seed, all of the following hold:
1. rho_hat(-0.85) >= 0.4 at >= 70% of anchors, with median >= 0.75;
2. rho_hat(-0.90) >= 0.5 at >= 70%, with median >= 0.75;
3. rho_hat(-1.00) >= 1.0 at >= 70%;
4. the true drop Q_stall(-KNEE) - Q_stall(-0.90) >= 0.25 at >= 70%;
5. the predictors are valid: max over k of |P_k - Q_nostall| <= 0.10 at -0.85, <= 0.15 at -0.90 and <= 0.30 at -1.00, at >= 90% of anchors. This item checks the estimator, not the hypothesis. A miss is INCONCLUSIVE.
6. sign control: rho_hat(+0.90) < 0 at >= 90%.

The positive thresholds encode the optimistic sign: a smooth critic over-values exactly where the actor's Q gradient points.

**Coherence with Stage 1.** For a critic that extrapolates smoothly, KC_d is about rho_hat at the same depth.
- K1a needs KC_0.05(CUTOFF) - KC_0.05(RANDOM) >= +0.5, and V0(b) allows KC_0.05(RANDOM) up to +0.25. Item 1's median bar of 0.75 makes K1a passable.
- K1b needs KC_0.10(CUTOFF, stall) - KC_0.10(CUTOFF, no-stall) >= +0.5, where the no-stall term is expected near 0. Item 2's bar of 0.75 makes K1b passable.

### L3. Decision-relevant OOD: the native learner is harmed there

The scope is the native learner only, in the stall world, at the 20k checkpoint and the E2 context, in every one of 3 fresh training seeds. Part 5, Stage 1 gives the exact rules. In summary:
- **L3a.** The critic over-values the region relative to a supported reference action, with the action-independent offset removed. The over-valuation is larger without the labels (CUTOFF vs RANDOM) and larger in the stall world than in the no-stall world for the same arm.
- **L3b.** The actor goes there: CUTOFF enters R_deep- far more than RANDOM. Because BC pulls inward by construction, the pull must come from Q.
- **L3c.** Going there costs return. Clipping the policy's own a0 at the knee raises its return on the same starts in the stall world, but not in the no-stall world.

---

## Part 3. The condition: benchmark version `knee_edge_v1`

The recorder and reset settings come from the reach screen (`REACH_SCREEN_RULE.md`, written before `screens/reach_grid.py` ran). The rule selected candidate **mix15 / C / d = 0.10** (output in `screens/reach_grid_output.txt`).

1. **Withheld set.** P(row) = |float64(a0_exec_f32)| > KNEE, strict. It is applied at every state, load, time and setting of a1..a3. There is one shared implementation (Part 4).
2. **Reset law (new start family `knee_mix15`):**
   - with probability 0.85, current c ~ U[0.1, 0.7], the recorded law;
   - with probability 0.15, c ~ U[0.9, 1.1], the deep-load component;
   - wave amplitude ~ U[0.1, 0.4]; kick 0.5.
   - The current range lies inside the plant's start set [0, 1.5]. No current in (0.7, 0.9) is ever drawn, and this gap is disclosed.
3. **Recorder (`knee_edge_mixture_v1`).** Each step draws exactly five uniforms (u, v, w1, w2, w3) from a fixed stream:
   - if g >= 0.4 (inclusive) and u < 0.98: a0 = -0.9 + 0.2v, a window on [-0.9, -0.7] that straddles the knee;
   - otherwise a0 = -1 + 2v, so a uniform floor keeps every a0 supported somewhere;
   - a1..a3 = -0.7 + 1.4 w_k. These never stall in any row.
   - The exact density: p(a0 | g >= 0.4) = 0.98/0.2 * 1[a0 in [-0.9, -0.7]] + 0.02/2, and p(a0 | g < 0.4) = 1/2.
   - Each row logs its exact log-density.
4. **Worlds.**
   - Primary: the stalling world at stall_slope 3.0, the pre-registered retune; `generate_dataset` accepts only 1.5 and 3.0.
   - Placebo: the no-stall twin (stall=False), which turns the stall off on all four thrusters. It is exact here because a1..a3 never exceed 0.7.
   - The twins use the same reset seeds and the same draws. Requested actions, loads and both masks are byte-identical; states diverge only at the successor of a row in W.
5. **Collection.** 1,280 episodes per world. `_split_assignment` gives 768 fitting, 256 tuning, 128 selection and 128 calibration episodes.
6. **Expected shares:**
   - P(g >= 0.4) = 0.575 exactly. Under the recorded law, g is symmetric about 0.4, which gives 0.5; the deep component always has g >= 0.5. So 0.85 x 0.5 + 0.15 = 0.575.
   - From Monte Carlo over 400,000 episodes: P(g >= 1.0) = 0.078, and 17.3% of episodes contain a row with g >= 1.0.
   - The first row in W comes almost immediately: the median first W step is 1. So the twin worlds' states diverge early in nearly every episode.
7. **Arms, in each world.**
   - **CUTOFF** drops every row in W. Expected removal: 0.494 of high-load rows, 0.200 of low-load rows, about 0.369 overall (about 28.3k of 76.8k fitting rows).
   - **RANDOM** drops the same count within episode x ten-step bin x load band. The load bands are low (g < 0.4), mid (0.4 <= g < 1.0) and deep (g >= 1.0), with lower edges inclusive. RANDOM keeps about 57% of W labels.
   - **High-load BC anchors** (the mean logged a0 the actor is cloned toward), exact from the recorder density: CUTOFF -0.7263, RANDOM -0.784.
8. **Host.** Unifloral TD3+BC at pinned revision f2dc1278, `td3_alpha` 2.5, `policy_noise` 0.2, `noise_clip` 0.5.
   - Gamma must resolve to 0.98. Unifloral's own default is 0.99, so this is checked.
   - Fresh initialization in every arm.
   - Normalization reuses the common-retained-rows method in `central_quartile_training_v1_evaluation_rerun1/data.py` (lines 145-187). The source is the intersection of retained rows, with a calibration-leakage check.
9. **Contexts.**
   - **E2, the gated context:** fresh deep-load starts with c ~ U[0.9, 1.1], wave U[0.1, 0.4], kick 0.5. This is the collection law's deep component, so it is not a state shift. Every E2 step has g >= 0.5, and half have g >= 1.0.
   - **E1:** fresh starts from the collection law. Reported, not gated.
10. **Scope of claims.** Every claim is confined to the stall_slope 3.0 world of `knee_edge_v1`. Nothing transfers to the uniform-recorder benchmark or to stall_slope 1.5.

---

## Part 4. Implementation certification

All new code goes in a new folder, `knee_edge_ood_v1/`, created only after approval A1. No frozen, hashed or sealed folder is edited. That includes `experiments/shift/` and every central-quartile package.

### 4.1 One shared region module, `knee_region.py`

Its source hash is recorded in every receipt.

- **`past_knee_a0(actions, kind="executed")`** returns (mask, side).
  - It takes float32 input of shape (..., 4). With kind="requested" it first applies the collector's own cast, `requested.astype(np.float32)`.
  - It compares strictly in float64 against KNEE.
  - It raises on NaN, on inf, on |a| > 1, and on float64 input passed as executed. -0.0 counts as 0.
- **Other functions:**
  - `depth(a)` returns d in float64; `in_deep(a)` tests d >= 0.10 (inclusive).
  - `load_g(obs_f32)` = float64(obs[..., 10]) + float64(obs[..., 11]).
  - `load_band` returns low, mid or deep, with lower edges inclusive. The recorder's window rule (g >= 0.4) calls this same function.
- **Import-time assertion.** On a boundary fixture, the mask equals canonical `stalled(a, Physics(stall=True, stall_slope=3.0))[..., 0]`.
- **Exclusive use.** It is the only path used by:
  - the recorder validator;
  - the CUTOFF and RANDOM mask builders;
  - the replay, calibration and auxiliary loaders, which fail closed if a CUTOFF stream contains any W row;
  - the normalization intersection;
  - every evaluation counter: free steps, the clip intervention, probe labels, target-smoothed actions in diagnostics, visited rows;
  - the gradient audit and the analysis.

### 4.2 An independent second implementation

It must come from a different author or session, with no shared code.

- **Region.** An IEEE-754 bit test, `(view(uint32) & 0x7FFFFFFF) > 0x3F4CCCCD`. It must raise on NaN and inf, as the primary does; a bare bit test would return True for them.
- **Depth and load bands.** Exact `Fraction` arithmetic on the float32 values.
- **Third check.** Canonical `stalled()[..., 0]` on every stall-world row.
- **Agreement required, row for row:**
  - every logged row in both worlds (2 x 128,000);
  - every mask decision;
  - every evaluated action in every stage;
  - every anchor candidate and every visited row.
- **One disagreement stops the study.**
- **Already done, read-only, by the panel:** 0 disagreements over all 671,090 float32 values with |x| in [0.79, 0.81], and agreement on 40,320 saved first commands.

### 4.3 A new dataset schema version

The current contract rejects this recorder:
- `thrusterchain/datasets.py` `_validate` requires `start_family == "recorded"`, a state-independent recorder and the old density semantics.
- `rerun1/data.py` `_read_source` raises "Recorder law changed".

The new version needs:
- density semantics that depend on load;
- a lineage replay for the 5-uniform recorder and the `knee_mix15` reset, including the per-episode mixture draw;
- role-safe reader bindings;
- derivation of the data plan.

The existing start laws (`START_LAWS`, `start_law`) take a single current range, so they cannot express the reset mixture. The new version therefore draws a per-episode `options["start"]` choice from a pinned named stream, with a fixed draw order: the mixture uniform first, then current, wave and kick. E2 itself can be written with the existing machinery.

Each part has its own tests and an independent check. The old reader must reject the new data, and the new reader must reject the old data.

### 4.3b Proof that the learner received the masked data

The three predicate implementations all inspect the same arrays, so they cannot catch the wrong arrays being used. Upstream `make_train_step` closes over `dataset` and samples rows with `randint(0, len(dataset.obs))` (`td3_bc.py` lines 174-185). Loader checks alone therefore do not prove what the jitted training step received.

- **In-worker assertions**, on the exact arrays bound into `make_train_step`, before the first update:
  - for CUTOFF, max |a0| <= KNEE, using the shared predicate;
  - the length equals the retained count from 4.5;
  - the sha256 of obs, action, reward, next_obs and done is written into the run receipt and matched against the mask builder's output.
- **Alignment fixtures**, synthetic episodes with known rows in W, which must catch:
  - the mask being applied to requested rather than executed actions;
  - obs and next_obs offset by one row;
  - a mask or dataset from the other world being bound. World tags must match, and a cross-world binding raises.
  - RANDOM strata built on the wrong time index.

### 4.4 Tests, on synthetic arrays only

- **Boundaries:**
  - ±KNEE is False; ±nextafter(KNEE, 1) is True; ±nextafter(KNEE, 0) is False; ±1 is True;
  - +0.0 and -0.0 are False;
  - requested 0.8, 0.80000001 and 0.8000000119 all cast to False;
  - NaN, inf, |a| > 1 and float64-as-executed raise, in both implementations.
- **Properties:** symmetric in sign; monotone in |a0|; unaffected by a1..a3 and by state; works for batch shapes (N, 4) and (B, T, 4).
- **Load-band edges** at g = 0.4 and 1.0, using float32 pairs whose float64 sum lands exactly on the edge or one ulp below it.
- **Recorder:**
  - exactly five uniforms per step, in a fixed order, consumed identically on both branches;
  - one mixture uniform per episode in the reset stream;
  - actions and loads byte-identical across the twins;
  - executed |a1..a3| <= 0.7;
  - logged log-density equals the closed form within 1e-14;
  - window, floor and mixture frequencies within 99.9% binomial bounds.
- **Host config** resolves gamma 0.98, done only at t = 99 (no bootstrap), `td3_alpha` 2.5, noise 0.2/0.5, and pin f2dc1278.
- **Cadence** is labeled from the saved evidence: actor 5,000 and critic 10,000 updates per 10k branch (`num_critic_updates_per_step` = 2).
- **Common normalization:** the native and auxiliary initializers receive identical intersection statistics, the same across arms within a world and seed.

### 4.5 Count reconciliation, per world

- 1,280 episodes split 768 / 256 / 128 / 128.
- High-load share of fitting rows in [0.53, 0.62] (expected 0.575). Deep share in [0.05, 0.11] (expected 0.078).
- CUTOFF's removed count inside the 99.9% interval of Binomial(n_high, 0.494) + Binomial(n_low, 0.200), given the realized band counts.
- Retained high-load mean a0 in [-0.75, -0.70] (expected -0.7263).
- 0 retained rows in W.
- RANDOM's per-stratum counts equal CUTOFF's.
- Retained and removed sets are disjoint, and their union is complete.
- Masks are byte-identical across worlds.
- In every episode, float states are identical across worlds up to and including the first row in W, and differ exactly at its successor. Positions are reported; nearly every episode diverges by step 1 or 2.
- **Reported, not gated: the state shift between worlds.** CUTOFF's retained rows reach different states in each world, because a stall-world command in W- delivers about 0.2 less thrust (about 0.4 less on floor rows). For each world and arm, report:
  - lambda from the training audits;
  - the standardized mean difference of each z input.
  - K1b and K3c compare learners trained on these slightly different state distributions; this is disclosed.
- Every row replays bit for bit. Terminal rows are reported per arm.

### 4.6 Checks on frozen checkpoints

The user's own run reads the checkpoints; agents never open them.

- **The G0.5 fold test** (step 0.3).
- **Known-answer reproduction** by the new evaluator on the frozen central-quartile native 20k cells:
  - saved lambda to 1e-6;
  - saved per-probe Q^pi to 1e-9 relative;
  - saved past-knee counts exactly, recomputed with the shared predicate;
  - identity residual <= 1e-6 relative;
  - stall equals no-stall Q for continuations that stay within the knee;
  - complete-state digest unchanged.
- Every screen over saved results lists both `authorized1` and `continuation1`, because seed2_random's audits live only in `continuation1`. It asserts nine cells.

---

## Part 5. Staged gates and kill rules

These rules hold at every step:
- Each step needs its own approval.
- Every threshold is fixed before the step runs.
- An outcome can only stop the plan or let it continue. No outcome may change the recorder, the world, the arms, the seeds or any threshold.
- One attempt per approval. An interruption means INCOMPLETE and needs a fresh approval, which is disclosed.

**Outcome labels**
- **PASS**: every rule holds.
- **FAIL**: the measurement was valid and a rule failed.
- **INCONCLUSIVE**: a validity precondition failed, so the step says nothing about the hypothesis.
- **INCOMPLETE**: numerical or execution failure.

### Step 0.1. Feasibility screens on saved JSON (done)

These are read-only. Their thresholds were set after earlier previews of the same data, so they are feasibility only and never confirmatory.

- **Reach (`screens/reach_grid.py`, rule written first).** The selected candidate's forecast entry into R_deep- has a minimum of 0.562 at probes and 0.317 on visited rows, against bars of 0.40 and 0.30.
  - The forecasts use lambda corrected for the new reset law (scale 0.751).
  - They are extrapolations: edge slopes come from quadratic fits on a0 in [-0.5, 0.5], from old critics evaluated at currents >= 0.7, which lay outside those critics' training data.
- **Sensitivity, recorded after the rule ran** (`screens/REACH_SENSITIVITY.md`). The visited model's constant 0.87 equals 1/(1 + 4 lambda) at the old lambda, so it is stale under the very correction the screen applies.
  - Correcting it makes the same rule select wide/D.
  - Restricting visited rows to current >= 0.9 makes it select mix25/D.
  - Only mix15/C and mix15/D are feasible under all three readings. The tie-break then keeps mix15/C. The selection does not change, but its basis is now "feasible under every pre-listed reading".
  - The binding cell (hole, seed 2) clears the visited bar by 2 rows: 40 of 126, against 38 needed. The margin is thin.
- **Value at k = 3.0.** From quadratic fits of saved own-actor Q^pi at strong current, the median rho(-0.9) is 2.12 to 2.58 by seed, with 88-94% of anchors >= 0.5. These are estimates; step 0.4 replaces them with exact values.
- **Predicate.** The exhaustive float32 band check found 0 disagreements.
- **Arithmetic** (sealed again by script under A1):
  - removal 0.494 / 0.200;
  - anchors -0.7263 / -0.784;
  - RANDOM high-load W density 2.48 per unit a0, against 0.50 in the old uniform data (about 5x).

### Step 0.2. Code, protocol and synthetic tests (approval A1)

There is no simulator, no training and no checkpoint access.

- **Written in this step:**
  - the sealed protocol, kill rules and `seeds.json`;
  - `knee_region.py` and the independent twin;
  - the recorder, the reset family and the new dataset schema (4.3);
  - the common-random-number draw discipline and the normalization fixture;
  - the evaluator counters and the clip intervention;
  - the 10k-fork reach check (Stage 1, R0).
- **Seeds.** All are fresh, and collision-audited against every earlier ThrusterChain dataset, reset, initialization, evaluation-bank and anchor stream. None of the 128 consumed development anchors is reused.
- **KILL:** any test failure or any disagreement between predicate implementations blocks every later approval until it is fixed. Fixing code before any data exists is not driven by outcomes.

### Step 0.3. Frozen-checkpoint checks (approval A2, user-run)

This step does forward passes and deterministic prefix replay on existing banks only. Cap: 0.5M simulator transitions or 30 minutes; the cap fails closed.

- **G0.5, the fold test.** Use each frozen native 20k critic (full, hole and random, x 3 seeds). At each saved high-load probe in the ordinary and strong-current banks, compute Q1 at a0 in {-0.5, -0.6, -0.7, -KNEE, -0.9, -1}, with the other commands at 0.
  - **Fold error:** F(s) = Q1(s, -1) - Q1(s, -0.5) + 0.22499999.
    - The constant is 0.3(1 - T(-1)^2) with T(-1) = -0.5000000298, from the exact stall_slope 1.5 identity. It holds under any continuation.
    - F > 0 means the critic over-values the fold. |F| is also reported.
  - **Steep probes:** saved R_lin = 2[Q^pi(-0.5) - Q^pi(-0.25)] + 0.225 >= 1.0.
  - **Governing bank: ordinary.** Those states lay inside the old critics' training data; strong-current states did not. The strong-current stratum is reported separately and does not decide.
- **Decision, on the full-data critics only:**
  - median F <= 0.5 among steep ordinary probes in every seed: proceed.
  - 25-50% of steep ordinary probes with F <= 0.5 in some seed: proceed only with this written prediction, which the user acknowledges at A3: "Stage 1 V0(b) may fail and end INCONCLUSIVE."
  - under 25% in any seed: STOP. Uniform labels leave most steep states over-valued, so error at the knee is approximation failure, not withholding. Write the shelving record.
- **Measured edge slopes replace the step 0.1 extrapolations.** The model is fixed now:
  - **Rows:**
    - the strong-current bank's high-load probes (the probe forecast);
    - the saved high-load visited rows with current >= 0.9, about 72 per cell (the visited forecast).
  - **Forward passes:** compute Q1 at a0 in {-0.5, -0.6, -0.7, -KNEE, -0.9, -1} at every visited row, as at every probe, with a1..a3 at the row's saved values.
  - **Local curve:** at each row, a least-squares quadratic Q1 = alpha + b a0 + c a0^2 through those six points.
  - **Equilibrium:**
    - pi0 = clip((m + 2 lambda' b) / (1 - 4 lambda' c), -1, 1), with m = -0.7263 and lambda' = the cell's saved 20k lambda x 0.751;
    - one model for both row sets, so the stale 0.87 constant is gone;
    - a row with 1 - 4 lambda' c <= 0 counts as reaching -1.
  - **Forecasts:** the fraction of rows with pi0 <= -(KNEE + 0.10), in each of the nine cells.
  - **KILL** if the probe forecast is below 0.35 in any cell, or the visited forecast is below 0.25 in any cell.
    - These bars sit below step 0.1's on purpose: this step adds information, and it must not tighten into a forking path.
    - The old critics are still outside their state distribution at current >= 0.9. Only Stage 1's R0 measures the real learner.
- **Evaluator known-answer reproduction** (4.6). KILL on any mismatch.

### Step 0.4. Collection and exact probes (approval A3)

This step uses the simulator but does no training. Cap: 2.5M transitions or 1 hour; the cap fails closed.

- **Collect** `knee_edge_v1` at stall_slope 3.0 and its no-stall twin, 2 x 1,280 episodes, followed by an exact replay check.
- **Exact Q^pi probes:**
  - 64 fresh E2 anchors, plus 32 fresh E1 mid-load anchors (reported only);
  - anchors are recorder prefixes from separate named streams at t in {0, 25, 50, 75}, rebuilt in the stall world, and they never enter replay;
  - continuations: the three frozen full-data native 20k actors, clipped at ±KNEE;
  - worlds: stall_slope 3.0, no-stall, and stall_slope 1.5 (identity check only);
  - forced a0: the nine S9 slots plus ±(KNEE + {0.05, 0.10, 0.15, 0.20}).
- **KILL if any of these holds:**
  1. any identity residual > 1e-6 * max(1, |Q|);
  2. stall and no-stall Q differ at any supported slot by > 1e-9 relative;
  3. any twin mismatch in actions, loads, masks, or states before the first W row;
  4. any count outside the bounds in 4.5;
  5. L2 PASS item 1, 2, 3, 4 or 6 missed for any continuation seed: FAIL. The region is not model-relevant, so shelve;
  6. L2 PASS item 5 (predictor validity) missed: INCONCLUSIVE. The residual estimator cannot be trusted, so stop and audit; nothing is shelved;
  7. L1.4 density outside its interval;
  8. L1.5 typicality fails (state shift is present);
  9. any executed |a1..a3| > 0.7.
- A PASS permits only the request for Stage 1.

### Stage 1. Native damage gate (approval A4)

**Data.** `knee_edge_v1` at stall_slope 3.0 and its no-stall twin, each with the hashed CUTOFF and RANDOM masks from step 0.4.
- The 128 calibration episodes get each arm's own mask rule.
- The 256 tuning and 128 selection episodes stay sealed.

**Arms.** {CUTOFF, RANDOM} x {stall 3.0, no-stall} x 3 fresh seeds = 12 native Unifloral TD3+BC runs.
- Seeds are paired across arms and worlds within a seed index, and differ between indices.
- Each run has a 10k warmup (native, with the shadow BCA auxiliary fit and calibration as in the prior study, so Stage 2 can fork), then a 10k native branch: 20k critic updates in all.
- Sanity check: Unifloral BC-only (`algorithms/bc.py` at the same pin) on stall CUTOFF, 3 seeds, 10k updates.

**R0, early reach check at the 10k fork, before any branch.** Run the CUTOFF-stall actor on 32 fresh E2 starts.
- **STOP as FAIL-early** if its R_deep- rate is below 0.05 in every seed. The reading is that the host contained the extrapolation at 10k; 20k was not observed.
- This stops only on a clear failure (K2 needs 0.25 at 20k), and it saves about 1.7 hours of branch compute.
- Lambda and the reach radius 2 * lambda * |dQ1/da0| are reported. At the actor's equilibrium the reach radius equals |pi0 - m|, so it adds no separate test.

**Evaluation.** Fresh named streams, never used in step 0.4 and never reused in Stage 2.
- **Banks:** E2 with 128 starts (gated) and E1 with 64 starts (reported). Checkpoints 15k and 20k; 20k is gated.
- **Free rollouts:**
  - pi (deterministic), pi_clip (a0 clipped to ±KNEE) and pi_clipall (all four clipped);
  - discounted and undiscounted return J;
  - **Δclip = J(pi_clip) - J(pi)** on the same starts: how much return the policy loses by going past the knee;
  - R_deep- and any-depth rates by load band; W+ rate; a1..a3 past-knee counts.
- **World swap.** J of each 20k policy in the other world on the same E2 starts, giving A(pi) = J_nostall - J_stall. Reported.
- **Q probes.** 64 fresh E2 anchors plus 32 E1 anchors.
  - Forced a0: S9, ±(KNEE + {0.05, 0.10, 0.15, 0.20}) and the actor's own action, with a1..a3 at the actor's values. Probes whose a1..a3 pass the knee are excluded and counted.
  - Exact own-continuation Q^pi.
  - **Knee contrast**, offset-free: KC_d(s) = [Q1(s, -(KNEE+d)) - Q1(s, -KNEE)] - [Q^pi(s, -(KNEE+d)) - Q^pi(s, -KNEE)].
  - **Primary depths:**
    - d = 0.05 for every rule involving RANDOM (V0(b), K1a), because -(KNEE + 0.05) = -0.85000001 lies inside RANDOM's labelled band;
    - d = 0.10 for within-arm rules (K1b).
    - All of d = 0.05, 0.10, 0.15 and 0.20 are reported, for Q1 and for the twin minimum.
  - **Reference continuation.** The primary Q^pi uses the clipped own continuation. KC under the unclipped own continuation is also reported, because Q1 estimates the unclipped policy.
- **Gradient audit** on 768 retained logged probes at 10k, 15k and 20k: lambda, the reach radius, and the signs of the Q and BC action gradients.
- **Diagnostic:** the rate of target-smoothed actions past the knee on replay.

**Decision rules.** These apply in the stall world, at E2 and 20k, in every seed unless stated.

- **V0, validity.** If any item fails: INCONCLUSIVE, stop.
  - (a) J_E1(RANDOM) - J_E1(untrained, same initialization) >= +50, in both worlds.
  - (b) RANDOM represents the knee: median KC_0.05(RANDOM) <= +0.25, and RANDOM's R_deep- rate <= 0.10.
  - (c) At least 16 valid anchors per arm.
  - (d) L2 PASS items 1-4 hold under each stall CUTOFF policy's own clipped continuation.
  - (e) a1..a3 past-knee entries <= 0.5% of E2 steps in every arm and world.
- **V1, numerical.** Any predicate disagreement, identity residual > 1e-6 relative, twin mismatch, missing job or budget exhaustion makes the result INCOMPLETE.
- **K1, critic:**
  - K1a: median KC_0.05(CUTOFF) - median KC_0.05(RANDOM) >= +0.5;
  - K1b: within the same arm, median KC_0.10(CUTOFF, stall) - median KC_0.10(CUTOFF, no-stall) >= +0.5.
- **K2, actor:** CUTOFF's R_deep- rate on E2 free steps >= 0.25, and >= RANDOM's rate + 0.15.
- **K3, return.** Within-policy attribution is the primary test, because it is paired on identical starts:
  - Δclip(CUTOFF, stall) >= +2.0;
  - Δclip(CUTOFF, stall) - Δclip(RANDOM, stall) >= +2.0;
  - Δclip(CUTOFF trained in no-stall, evaluated in no-stall) <= +0.5. Clipping must not help where going past the knee really helps.
- **Pre-registered secondary outcomes.** Their directions are predicted; they are reported and not gated:
  - J(RANDOM) - J(CUTOFF) in stall (predicted > 0);
  - A(pi_CUTOFF) - A(pi_RANDOM) (predicted > 0);
  - the stall-minus-no-stall triple difference of J.
  - Noise reference: the closest saved proxy for E2, native 20k at strong current, gives hole-minus-random spreads of +4.9 / -11.5 / -41.0 with a free-episode SD of about 32. So a J miss alongside a Δclip pass reads as "damage present within the policy, below between-arm seed noise".
- **Sanity checks**, true by construction and not evidence:
  - BC action gradients point inward at CUTOFF rows in R_deep-;
  - the BC-only actor's R_deep- rate is <= 0.005.

**Outcomes.**
- **PASS** = V0, V1, K1, K2 and K3 all hold in every seed. It permits only the request for Stage 2.
- **FAIL-consistent** = V0 and V1 hold, and at least one criterion fails in every seed.
- **FAIL-mixed** = V0 and V1 hold, no criterion fails in every seed, but some criterion fails in some seed. It is never read as equivalence, and it gets no mechanistic reading.
- **The pass rate is low even when the hypothesis is true.** There are seven criteria in each of three seeds. If they were independent at 0.95 each, the pass probability would be about 0.34; correlation between criteria raises it. This is disclosed so that a FAIL-mixed is not over-read.

**What each consistent failure means.** These readings apply only when the criterion fails in every seed:
- K1a fails: withholding did not make this Unifloral critic over-value the knee's onset. It does not extrapolate optimistically past a dense edge, so BCA has nothing to repair here.
- K1 passes and K2 fails: the alpha 2.5 BC anchor contains the over-valuation.
- K2 passes and Δclip(CUTOFF, stall) fails: this contradicts step 0.4, so audit before anything else.
- The no-stall Δclip check fails: the harm does not come from the hidden knee. Stop.

### Stage 2. BCA comparison (approval A5, only after a Stage 1 PASS)

**Frozen rules**, copied with hash lineage from the prior sealed study and never tuned:
- BCA kappa = 3: the score is Q1 - 3U, and the actor penalty comes from the width U;
- the auxiliary architecture and calibration rule are unchanged;
- a distance-penalty arm is matched by logged-action gradient RMS at 10k. That match is off-policy and is disclosed as such.

**Arms.** From each of the 12 Stage 1 10k forks, three branches of 10k updates: BCA, distance, and a **native replicate** with a fresh pre-registered branch seed. That is 36 branches.
- **Gated comparisons use the native replicate alone.** The Stage 1 native branch was selected for showing damage, so using it, or averaging it in, would bias recovery upward through regression to the mean.
- The Stage 1 branch is reported separately.

**Fresh evaluation.** A new bank (E2 with 128 starts, E1 with 64 starts) and 64 new E2 anchors, distinct from every earlier stream.

**Detection precheck, at the 10k fork before any branch, using the shadow auxiliary.** At the 64 new E2 anchors, with a1..a3 at the actor's values, compare paired sets at the same states:
- OOD set: a0 in -(KNEE + {0.10, 0.15, 0.20});
- near-edge in-support set: a0 in {-KNEE, -0.775, -0.75, -0.725, -0.70}, the densely logged band.
- **Require** the AUROC of U to be >= 0.70 in stall CUTOFF in every seed. AUROC is the probability that a random OOD action gets a larger U than a random in-support action.
  - It is computed within each anchor, over the 3 x 5 = 15 OOD vs in-support pairs at the same state, then averaged over anchors. Pooling across states would let differences in U between states masquerade as detection.
- **Report:**
  - FPR@95, the share of in-support actions flagged at the threshold that catches 95% of OOD actions; at or below 0.50 counts as strong detection;
  - AUROC against broad in-support slots;
  - AUROC for W+;
  - AUROC of the score Q1 - 3U;
  - RANDOM specificity at d = 0.05 only, the depth RANDOM has labels for, where an AUROC near 0.5 is expected.
- **If the AUROC is below 0.70 in any seed: STOP.** Record that BCA's U is blind to the region; no BCA branches are trained. This is a detection finding, not a failure of the condition.

**Robustness screens** (20k, E2, every seed; "promising" requires all four):
- J(BCA) - J(native replicate) >= +3.0 in stall CUTOFF.
- Recovery = [J(BCA, C) - J(native rep, C)] / [J(native rep, R) - J(native rep, C)] >= 0.50 on the three-seed mean. Only Stage 2 quantities are used.
  - It is computed only if the denominator is at least +3.0 on the three-seed mean. Below that, recovery is INCONCLUSIVE, because a near-zero or negative denominator makes the ratio meaningless. "Promising" then rests on the other three screens, and this is stated.
- J(BCA) >= J(distance) in stall CUTOFF.
- Cost of caution: J(BCA) - J(native rep) >= -2.0 in no-stall CUTOFF and in stall RANDOM.

**Repair screens** (reported separately; they neither cancel nor stand in for robustness):
- BCA's R_deep- rate minus native's <= -0.15 in every seed;
- Δclip(BCA) <= 0.5 x Δclip(native replicate) on the three-seed mean, in stall CUTOFF;
- median KC_0.10 of the adjusted score (Q1 - 3U) <= 0.5 x median KC_0.10 of the native replicate's Q1, in stall CUTOFF, in every seed.

All results are reported per seed, whatever the outcome. No kappa, mask or threshold changes after viewing.

---

## Part 6. Controls

1. **RANDOM.** Count-matched removal within episode x ten-step bin x load band, with the same named seed in both worlds. It controls data quantity, terminal-row loss, coarse state and time, and replay size. It keeps about 57% of W labels, dense in the band where the knee begins. L1.4 checks this with a test that can fail.
2. **Exact no-stall twin.** Region-matched, because a1..a3 never exceed 0.7 and load never depends on actions. Same draws in both worlds.
3. **Within-arm, cross-world contrasts:** KC(C, stall) vs KC(C, no-stall), and Δclip(C) in both worlds. These replace the panel's arm-vs-arm no-stall placebo, which was biased toward FAIL by the BC-anchor gap (-0.7263 vs -0.784) together with RANDOM's no-stall labels showing that going past the knee helps.
4. **Clip intervention, Δclip.** The same policy and the same starts, with a0 clipped at the knee. Evaluation-only, never a method.
5. **World swap**, A(pi) = J_nostall - J_stall on identical starts (secondary).
6. **W+ sign control.** Withheld, but the identity predicts pessimistic error there: no optimistic error and no Q-driven entry are expected.
7. **Offset-free contrasts.** KC against the supported reference -KNEE, and the difference-in-differences against -0.5. Both cancel the +23 to +33 action-independent critic offset.
8. **Validity controls:** learning adequacy; RANDOM's representation of the knee; L1.4; L1.5; L2 under the policy's own continuation; the bound on entries past 0.7 for a1..a3.
9. **BC-only actor and the BC gradient sign:** sanity checks only.
10. **Stage 2 only:** the native replicate, the distance-penalty arm and the U-detection precheck.
11. **Not used as evidence:** the 2026-10-12 gapped-log study (`bca/EXPERIMENTS.md`, `experiments/shift/`), which used a non-Unifloral learner. It supports plausibility only.

---

## Part 7. Budget

- **Timing basis.** Estimates scale from saved Zeus timings: warmup supervisor elapsed 470-512 s; a 10k branch about 500 s, which is conservative; 3.94M evaluation transitions in about 7 minutes. Per-run time under the new data is not yet verified.
- **Paid compute:** none.

| Step | Work | Estimate | Cap |
|---|---|---|---|
| 0.1 | Read-only screens | Done | — |
| 0.2 (A1) | Code, schema and synthetic tests | Agent and user time | — |
| 0.3 (A2) | Forward passes on 9 frozen critics; known-answer replay | < 30 min | 0.5M transitions, 30 min |
| 0.4 (A3) | 2 x 1,280 episodes plus replay check; 96 anchors x 3 continuations x 3 worlds x 17 forced actions | 256k + 256k + about 1.5M transitions | 2.5M transitions, 1 h |
| Stage 1 (A4) | 12 native runs (240k critic updates) + 3 BC-only + evaluation (about 5M transitions) | 3.3-4.0 h | 5 h training (1,800 s per job) + 1 h evaluation |
| Stage 2 (A5) | Detection precheck + 36 branches (360k critic updates) + evaluation (about 12M transitions) | about 5 h + 25 min | 6 h + 1 h |

- **Totals:** about 4.5 h of Zeus wall time through Stage 1; about 10 h through Stage 2 if every gate passes.
- **Disk:** about 4-6 GB of checkpoints per training stage.
- **Caps:** every cap fails closed. A run that hits its cap is INCOMPLETE and is not extended.

---

## Part 8. Risks, weaknesses and what cannot be promised

1. **Reach is the main risk.** Forecast entry on visited rows is 0.317 against a 0.30 bar (2 rows of margin in the binding cell), with K2 requiring 0.25.
   - Every forecast extrapolates edge slopes from fits on [-0.5, 0.5], using critics trained on the old data and evaluated at currents those critics never saw.
   - The selection holds under all three model readings in `screens/REACH_SENSITIVITY.md`, but the margin depends on the model.
   - **The alternative mix15/D.** Its window is [-0.85, -0.75] and its margins are larger (0.38-0.50). It would push RANDOM's labels past the knee to the very edge: float32(-0.85) = -0.85000002 lies only 1e-8 beyond -(KNEE + 0.05). That leaves no labelled depth for the control. C is kept for that reason.
   - A CUTOFF critic trained on a dense 0.1-wide retained band may flatten at its own edge.
   - Step 0.3 measures real edge slopes, and R0 stops early if the host clearly contains the extrapolation.
   - **The trade-off made for reach.** The narrower window C moves the CUTOFF anchor from -0.646 to -0.726. The price is that RANDOM's labels past the knee stop at -0.9.
2. **The control may fail to represent the knee.** That outcome is INCONCLUSIVE, an approximation failure rather than OOD. G0.5 gives an early warning, and V0(b) decides.
3. **a0 is correlated with load.** The window sits at g >= 0.4, which can bias the critic's a0 slope. The critic's slope is compared with the true flank slope and reported.
4. **Disclosed rather than removed:**
   - trajectory history: retained states after a removed row differ between worlds;
   - target-policy smoothing queries W inside Bellman targets. Its rate is measured; it is the standard bootstrapped extrapolation channel.
5. **Benchmark version.** The new recorder, the reset mixture with its unsampled (0.7, 0.9) current gap, stall_slope 3.0 and the a1..a3 bound all mean no transfer to the old benchmark. Deep rows sit in about 17% of episodes, which raises count variance.
6. **Step 0.1's thresholds were set after previews**, so they are feasibility only. The probe banks over-represent steep states.
7. **Detection may fail** (U flat across a0, as before). Stage 2 then stops with a detection finding.
8. **Still to be verified in code:**
   - per-run wall times under the new data;
   - gamma resolution to 0.98;
   - the new dataset schema;
   - the k = 3.0 residuals, which stay estimates until step 0.4.
9. **What cannot be promised.** L1, L2 and the implementation can be made certain, and Part 4 says how. Whether a learned native critic over-values the region, and whether the actor goes there, is an empirical property of TD3+BC that L3 measures. A FAIL there is a valid answer.

---

## Part 9. Changes from the panel's plan

The panel's synthesized plan (KEC-W v1) was attacked by an independent critic. Each finding was checked against the repository and resolved:

| Severity | Finding (verified) | Resolution here |
|---|---|---|
| High | The reach forecast held lambda at its old value. The wide reset law raises mean\|Q1\| from 72 to about 122 (lambda x 0.59), and the plan's own screen then fails: probe minimum 0.375, visited 0.02-0.25. Reproduced by `screens/reach_grid.py` | A pre-written candidate grid with corrected lambda selected mix15 / window C. Step 0.3 measures real slopes; R0 stops early |
| High | The no-stall placebo compared arms against each other and was biased toward FAIL by the anchor gap and by RANDOM's no-stall labels | The no-stall terms are now within-arm (Δclip(C, no-stall), KC(C, stall) - KC(C, no-stall)). Arm-vs-arm no-stall terms are report-only |
| Medium | Stage 2 compared BCA with the Stage 1 branch selected for damage | A native replicate branch per fork; recovery from Stage 2 quantities only |
| Medium | "322/3,140" left out seed2_random | Restored to 361/3,519 (10.3%) for all nine native 20k cells; screens assert nine cells |
| Medium | Edge slopes and k = 3.0 values are extrapolations | Labeled as such; step 0.3 measures Q1 at -0.6 to -1 |
| Medium | Return noise was cited from the ordinary context | The strong-current proxy is cited (+4.9 / -11.5 / -41.0, SD about 32). J is secondary; Δclip is primary |
| Medium | The dataset contract rejects the new recorder | New schema version with cross-rejection tests (4.3) |
| Medium | "5x" label density, and G0.5's 25% bar inconsistent with V0(b) | For window C the density is 2.48 per unit a0, about 5x (the critic's 2.5x was right for the panel's window A). G0.5 now uses a median bar, with a written prediction for the 25-50% zone |
| Low | L2 anchor rebuilding underspecified; L2 not checked under own policies | Stall-world prefix replay; V0(d) |
| Low | L1.4 could not fail | Replaced by a density interval |
| Low | K2 sub-checks true by construction; the twin returned True on NaN; a1..a3 past 0.7 | Relabeled as sanity checks; the twin raises; V0(e) bound plus clip-all reported |
| Low | Stale statements | Normalization reuses rerun1 `data.py`; cadence 5,000/10,000; timings 470-512 s; shares by Monte Carlo; the stall-knee band's value test passed at its minimum (8/29) while support and exposure failed |

A second, independent review of this draft (2026-10-29) found these further issues. Each was reproduced or checked before being fixed:

| Severity | Finding | Resolution here |
|---|---|---|
| High | Units of J, Δclip and every threshold were undefined; discounted and undiscounted returns differ by about 2.4x. The Q^pi horizon was ambiguous | Every J and threshold is discounted (gamma 0.98, from t = 0 over 100 steps). Q^pi runs from the anchor to the episode end (Part 2, Terms) |
| High | Step 0.3's reach rerun named no model, a forking path at a KILL gate. The visited model's 0.87 is 1/(1 + 4 lambda) at the old lambda; corrected, the rule picks wide/D, and with deep-current rows only it picks mix25/D (reproduced) | Step 0.3's model is fully specified: one equilibrium model, measured six-point local quadratic, no 0.87. Sensitivity is recorded; mix15/C is feasible under all three readings |
| Medium | "No learner can tell the worlds apart" overclaimed, since ordered trajectories reveal a removed step's thrust. The "no-W episodes identical" check was vacuous, because the median first W step is 1. The worlds' state distributions differ | L2.2 is restated as transition-level. The twin check now compares states up to the first W row. Per-world lambda and input shift are reported; L1.5 holds in both worlds |
| Medium | G0.5's median was dominated by strong-current states the old critics never saw | The ordinary bank governs; strong current is reported separately; \|F\| is reported |
| Medium | Predictor-validity misses were labeled "shelve"; L2's median bar was incoherent with K1a at d = 0.10 | Predictor validity is INCONCLUSIVE. L2 gains a -0.85 item, and its bars are matched to K1a and K1b (coherence note under L2) |
| Medium | "d = 0.10 inside RANDOM's labelled band" was false (float32(-0.9) = -0.89999998 > -0.90000001) | Every rule involving RANDOM uses d = 0.05; within-arm rules use d = 0.10 |
| Medium | Seven criteria x three seeds give a low pass rate even when the hypothesis is true; single misses were read mechanistically; recovery could divide by noise; "replicate or mean" was an open choice | Outcomes split into FAIL-consistent and FAIL-mixed, and mechanistic readings apply only to consistent failures. Recovery needs a denominator >= +3.0. The comparator is the replicate alone |
| Low to medium | Certification could not prove the jitted step received the masked arrays | In-worker assertions and array hashes, plus alignment fixtures (4.3b) |
| Low | KC's reference policy differed from the critic's | KC under the unclipped own continuation is also reported |
| Low | L1.4's interval included -KNEE, and its denominator was unstated; the fold constant was wrong in the 8th digit; P(g >= 0.4) is exact; Part 10's drops were ordinary-context; R0's two conditions were redundant; AUROC was pooled; the reset mixture cannot be written in `START_LAWS` | All corrected (L1.4, G0.5, Part 3, Part 10, R0, Stage 2, 4.3) |

---

## Part 10. Rejected alternatives

- **Withholding on the existing uniform data** (globally, conditionally, or the minimal design TLCW). Rejected because of reach: 0 of 1,206 ordinary high-load visited rows go past the knee, and BC anchors near 0 need slopes the critics do not have. Its labelled control has fewer knee labels than the full data, and 70% of rows would be removed.
- **The LoadTrack-Knee recorder as written.** The a0-load correlation is about 0.8, the positive side is never logged, and its main variant fails its own slope test.
- **The KEC-v1 physics design as written.** Its twin is contaminated, because a1..a3 ~ U[-1, 1] puts about 49% of rows past the knee. It also gates a context with marginal reach and counts entry at any depth.
- **The panel's KEC-W v1.** It fails its own reach screen once lambda is corrected (Part 9).
- **Raising `td3_alpha` or changing the host.** That would be tuning the host until it fails. CORL-derived learners are excluded.
- **stall_slope 1.5 as the primary world.** It gives about half the value gap. In the ordinary context the median true drop at d = 0.10 is 0.60 at slope 1.5, against 1.18 at 3.0. The strong-current proxy gives 0.84-1.00 against 1.67-2.06. Kept only as an identity check.
- **Strong current as a gate.** Confounded by state shift.
- **Previously failed or superseded, not revived:**
  - the central-quartile Hole;
  - the stall-knee band |a0+0.8| <= 0.05, which failed support and exposure (its value test passed at its minimum);
  - the zero slab;
  - `conditional_ood_gate_v2`;
  - `support_shift_pilot_v1`;
  - the other candidates in the Stage 0 options memo.

---

## Part 11. Provenance

- **Design panel:** `panel/PANEL_OUTPUT.json` (sha256 bf4efe7e…1257). It holds three independent designs, two judges, a synthesis and an adversarial attack. The panel read saved JSON only.
- **Reach screen:**
  - `REACH_SCREEN_RULE.md`, written before the script, sha256 8d4cae03…db46;
  - `screens/reach_grid.py`, a260741b…84a3;
  - `screens/reach_grid_output.txt`, 9909de46…e81b99. It records the sha256 of every input.
  - `screens/REACH_SENSITIVITY.md` with `screens/reach_variants.py` and its output. These were written after the rule ran and are labeled as sensitivity only.
- **Code facts cited:**
  - `titan_20261026/.../core_v4/thrusterchain/canonical/thrusterchain.py`: reward on the clipped command (line 22); the no-stall world turns off every thruster's stall (lines 125-131); start laws (lines 75-91).
  - `unifloral_td3bc_validation_v1/upstream/algorithms/td3_bc.py`: lambda (243-245), noise (199-200), default gamma 0.99 (40).
  - `datasets.py`: `_split_assignment` (135-140) and `_validate` (255-269).
  - `rerun1/data.py`: normalization (145-187).
