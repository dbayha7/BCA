# Knee-edge OOD condition: draft protocol, revision 6

**Status: DRAFT r6, 2026-10-29.**
- Nothing is frozen or approved. Nothing has been trained, simulated or collected.
- Every stage needs the user's own approval at their terminal. A message from an agent never counts.
- **Computation behind this draft:** reading saved JSON from the finished central-quartile study, plus Monte Carlo over the reset law and recorder only (no simulator dynamics). No NPZ was decoded and no checkpoint was opened.
- **History:**
  - revisions 1-5 are kept as `DRAFT_PROTOCOL_r1.md` through `_r5.md`;
  - the reviews that drove each revision are in `reviews/` and summarized in Part 9.

**Contents**
- Part 0. Terms
- Part 1. Why a new condition
- Part 2. The OOD definition, in three layers
- Part 3. The condition: benchmark version `knee_edge_v1`
- Part 4. Implementation certification
- Part 5. Staged gates, kill rules and outcome labels
- Part 6. Controls
- Part 7. Budget
- Part 8. Risks, weaknesses and what cannot be promised
- Part 9. Review history
- Part 10. Rejected alternatives
- Part 11. Provenance

---

## Part 0. Terms

**Simulator and actions**
- **Thruster commands a = (a0, a1, a2, a3).** The executed float32 command, promoted to float64 for every comparison. **a0** is thruster 0.
- **KNEE** = `float(np.float32(0.8))` = 0.800000011920929 (bits 0x3F4CCCCD). Past it, thrust falls instead of rising (the stall). A command of exactly KNEE does not stall.
- **W**, the withheld region: |a0| > KNEE, strict. **W-** = {a0 < -KNEE}; **W+** = {a0 > KNEE}, used as a sign control.
- **Depth** d = |a0| - KNEE.
- **R_deep-** = {a0 <= -(KNEE + 0.10)}, inclusive. This is the region where actor entry is counted.
- **T(a0)** = sign(a0)(KNEE - 3(|a0| - KNEE)): thrust past the knee at stall_slope 3.0.
- **B+**: the pseudo-inverse of the matrix that maps thrust into the next state.
- **Load** g = float64(obs10) + float64(obs11): the current plus the first wave number. It never depends on actions: the current is constant and the wave turns 0.27 rad per step regardless of the command (`canonical/thrusterchain.py` docstring line 14).

**Design labels**
- **mix15**: the reset mixture of Part 3.2.
- **Window C**: the high-load recorder window [-0.9, -0.7]. **Window D**: [-0.85, -0.75].
- **KEC-v1**: the design panel's physics design. **KEC-W v1**: its synthesized plan.
- **TLCW**: the minimal design. **LoadTrack-Knee**: the dataset design.
- The three designs and the plan are in `panel/PANEL_OUTPUT.json`.

**Arms, worlds and contexts**
- **CUTOFF** and **RANDOM**: the two data arms (Part 3.7). **Stall world**: stall_slope 3.0. **No-stall world**: the stall is off on every thruster.
- **E1**: fresh starts from the collection law. **E2**: fresh deep-load starts with c ~ U[0.9, 1.1].
- **Bank**: a fixed, pre-registered set of starts. **Free episode**: a 100-step rollout of a policy from a bank start.
- **Anchor**: an exact simulator state and time, rebuilt by replaying a recorder prefix. Anchors are taken at t in {0, 25, 50, 75}.
  - The **gated anchors** are those at t in {0, 25, 50}. t = 75 anchors are reported only, because their remaining horizon is about half as long.
  - A **valid anchor** is defined over the *gated* probes only: a1..a3 = 0, clipped continuation. Every gated forced probe executed within the box, and every Q^pi computation completed.
    - Reported probes at the actor's own a1..a3 never invalidate an anchor. Those whose a1..a3 pass the knee are excluded from the reports and counted.
  - A **predictor-valid anchor** is a valid anchor where every L2 predictor passes item 5.

**Learner and penalty**
- **Host**: Unifloral TD3+BC at pinned revision f2dc1278. Q1 and Q2 are its two critics.
  - The **twin minimum** min(Q1', Q2') of the lagged target critics forms the Bellman target (`td3_bc.py` line 207).
  - The actor maximizes Q1 (line 240).
- **lambda** = 2.5 / mean|Q1| over each training batch (lines 243-245). It sets the strength of the Q pull against behavior cloning (BC).
- **BC anchor m**: the mean logged a0 at a state. The actor settles near pi0 = m + 2 lambda dQ1/da0.
- **BCA**: the Bayesian-calibrated actor penalty under test.
  - **U** is its uncertainty width; **kappa** = 3 is its fixed weight, giving the score Q1 - 3U.
  - The **shadow auxiliary** is BCA's width model. It is fit during the native warmup so that Stage 2 can fork from the same checkpoint.

**Values and contrasts**
- **Q^pi(s, a)**: the exact discounted return of taking a at an anchor's state and time t, then following policy pi to the episode's end. That is 100 - t steps, discount 0.98, with no bootstrap after step 99. It is computed by replaying the simulator from the exact float64 state.
- **J (return)**: a free episode's discounted return (gamma 0.98, summed from t = 0 over 100 steps), averaged over a bank. Every return threshold uses these units. Undiscounted returns are reported only; they are about 2.4x larger.
- **pi_clip** is the policy with a0 clipped to ±KNEE. **pi_clipall** clips all four commands.
- **Δclip**(pi, world) = J(pi_clip) - J(pi), on the same starts in the named world: the return a policy loses by going past the knee.
- **Triple difference of J**: [J(RANDOM) - J(CUTOFF)] in the stall world minus the same in the no-stall world.
- **KC_s** (knee contrast at slot s) = [Q1(s_slot) - Q1(-KNEE)] - [Q^pi(s_slot) - Q^pi(-KNEE)], at the same state. It is offset-free: the action-independent critic offset of +23 to +33 seen before cancels.
- **rho** = Q_nostall - Q_stall at the same state and action: the true value gap the stall creates. **rho_hat** is its estimate from supported neighbors (L2.3).
- **A(pi)** = J_nostall(pi) - J_stall(pi), the world swap.
- **AUROC**: the probability that a random OOD action gets a larger U than a random in-support action. **FPR@95**: the share of in-support actions flagged at the threshold that catches 95% of OOD actions.

**Probe slots: float32-executed values and exact depths**

| Name | Executed a0 | Depth d | In W | In R_deep- |
|---|---|---|---|---|
| s05 | -0.85000002 | 0.05000001 | yes | no |
| s10 | -0.90000004 | 0.10000002 | yes | yes |
| s15 | -0.94999999 | 0.14999998 | yes | yes |
| s20 | -1.0 | 0.19999999 | yes | yes |
| p10 | +0.90000004 | 0.10000002 | yes (W+) | no |
| near-edge in-support | -KNEE, -0.77499998, -0.75, -0.72500002, -0.69999999 | <= 0 | no | no |

Note that float32(-0.9) = -0.89999998, which is *not* in R_deep- (depth 0.09999996). Every rule names its slot, never a rounded number.

**Outcome labels** are defined in Part 5.0.

---

## Part 1. Why a new condition

The central-quartile Hole withheld |a0| <= 0.25 at load >= 0.4, and it was OOD by definition only:
- it was an interior gap on a nearly straight value slope (R^2 >= 0.994), with data on both sides;
- native critics showed no error specific to the band, only an action-independent offset of about +23 to +33;
- actors entered the band because behavior cloning pulled them there, since under the uniform recorder the mean logged a0 was 0.

It tested neither extrapolation, nor value the critic could not know, nor Q-driven entry. The figures are in `analysis/ood_region_activity/`.

This draft builds the condition from the one simulator feature that makes withheld values unknowable from retained transitions: the stall knee on thruster 0.

---

## Part 2. The OOD definition, in three layers

### What can be certain and what can only be measured

| Layer | How it is established | Status |
|---|---|---|
| L1, outside the data | By construction; verified by counts with three predicate implementations | Exact: provable and checked row by row |
| L2, unknowable from retained transitions | An exact simulator identity, plus a twin world with no stall | The identity is exact (algebra of the simulator). The size of the value gap is an exact computation *given* a stated continuation policy |
| L3, harmful to decisions | Trained native learners measured against kill rules | Measured: it can fail, and a failure is a valid result |

No simulator can guarantee that a learned critic over-estimates, or that an actor goes there. L3 is a gate, not a promise. If L3 fails, the condition is still a correct OOD condition: it shows that this host protects itself here. That is a finding about TD3+BC, not about BCA.

### L1. Data-support OOD: the region is outside the retained data

- **L1.1 Zero labels.** In each world, every data stream the CUTOFF learner touches contains exactly 0 rows in W:
  - fit replay;
  - calibration rows;
  - shadow-auxiliary data;
  - probe and scale banks;
  - the normalization source.
  Three implementations must agree row for row (Part 4.2). The in-worker assertions (Part 4.4) must pass at every binding.
- **L1.2 Separation along a0.** At every state, every W- query has a0 outside the range of retained a0.
  - A query of depth delta is at least delta from every retained critic input in every Lp norm (p >= 1).
  - s10 is at least 0.10000002 away; a0 = -1 is at least 0.19999999 away.
  - This is a statement about one coordinate. Its strength is that the critic must *extrapolate along a0* beyond the last logged command, not that the region is far away. The ratio of delta to the median retained one-nearest-neighbor distance is reported.
- **L1.3 Censored support.** W lies inside the *recorder's* support (the recorder logs commands past the knee) but outside the *retained dataset's* support. This is dataset-support (censoring) shift.
  - CUTOFF's data follow the recorder density truncated to |a0| <= KNEE. It is renormalized by 0.506 at high load and 0.800 at low load, and is zero on W.
  - The truncated density is documented in the dataset receipt.
- **L1.4 The labelled control really has its labels.** A test that can fail, using values sealed from `screens/random_control_mc.py`:
  - **Band:** rows in W- with a0 >= float32(-0.9) = -0.89999998.
  - **Density:** RANDOM-retained high-load fitting rows in the band, divided by (high-load fitting rows x 0.1).
  - **Bars** (simulated mean 2.08; 6 SD bounds, rounded outward):
    - pooled over high load, the density must lie in [2.00, 2.17];
    - mid band (0.4 <= g < 1.0) in [2.00, 2.18];
    - deep band (g >= 1.0) in [1.82, 2.24];
    - CUTOFF's density must be exactly 0.
  - **Known faults these bars catch:** an unstratified RANDOM (expected 3.10) and a RANDOM blind to load band (expected 2.46).
  - **A known fault they do not catch:** a RANDOM that keeps the prior study's two-level high/low flag. It gives a pooled 2.14 and a deep-band 2.24, neither reliably rejected. The per-stratum count check in 4.7(i), on three-band strata, catches it deterministically. All three faults are synthetic tests in 4.6.
- **What RANDOM controls** (from `screens/random_control_mc_output.txt`):
  - RANDOM keeps 48.2% of W labels: 42.4% at high load and 67.6% at low load. Count matching within small strata removes withheld rows preferentially, which is why this is not the naive 51-57%.
  - At high load, its labels past the knee are dense only for depths below 0.1. R_deep- holds about 19 RANDOM rows at high load, against about 1,105 at low load.
  - RANDOM therefore gives the critic (i) the onset of the knee at high load and (ii) the full fold at low load. CUTOFF gives neither.
  - Because the stall belongs to the thruster and not to the state, a critic can transfer the fold across load, but it is not guaranteed to. V0(b) checks this.
- **L1.5 Action shift, not state shift, at the gated states.** At least 95% of E2 start states, and 90% of gated E2 anchors, must lie within the training 0.99 quantile of 64-nearest-neighbor state distance.
  - It is computed leave-one-episode-out on the 14 normalized inputs (13 observation numbers plus remaining time), against each world's training rows separately.
  - E2 is a sub-population of the collection law's deep component, with overlapping support. That is a sub-population shift in state, not a support shift.
  - Typicality of *policy-visited* E2 states is reported per arm and world. Stall-world commands in W- produce thrusts in (-0.8, -0.2], and those in R_deep- produce [-0.5, -0.2]. The recorder delivered the deep range at high load only rarely, through the floor (about 0.3% of high-load rows), so visited states can drift.

### L2. Model-relevant OOD: retained transitions cannot reveal the region's value

**Anchors and continuations**
- The continuation pi_c is the policy followed after the first action:
  - in step 0.4, the three frozen full-data native 20k actors from the prior study, with all four commands clipped to ±KNEE so they never stall;
  - in Stage 1, each model's own actor, clipped the same way.
- Each anchor's exact state and time are rebuilt by prefix replay in the **stall world**. Both worlds' Q values are computed from that same float64 state.

**The three claims**
- **L2.1 Exact identity.** Q_stall(s, a) = Q_stall(s, a with a0 <- T(a0)) - 0.3(a0^2 - T(a0)^2).
  - **Why it holds:** the next state depends only on thrust, and the reward charges 0.3 * a^2 on the clipped command at the state before the step (`canonical/thrusterchain.py` `dynamics`, lines 207 and 218-221).
  - **Examples:** Q(s10) = Q(T(s10)) - 0.168 and Q(-1) = Q(-0.2) - 0.288.
  - **Tolerance:** |residual| <= 1e-6 * max(1, |Q|) at every anchor, allowing for float32 rounding of T.
  - **Limit:** the identity cancels the state, horizon and continuation, so it cannot catch evaluator errors in those. A second, independent Q^pi implementation does (4.5).
- **L2.2 Not identifiable by a per-tuple learner.**
  - Wherever every |a_k| <= KNEE, thrust is bit-identical in both worlds (the formula subtracts ±0.0). So the conditional law P(s', r | s, a) is identical on the retained support. Q_stall = Q_nostall there under any continuation clipped to ±KNEE. The 1e-9 relative tolerance only absorbs summation order.
  - The worlds differ in (i) the state distribution of retained rows, because a removed W- step delivers about 0.2 less thrust in the stall world, and (ii) gaps that can be linked: remaining time and exact next_obs-to-obs matches let a transition set be reordered, and the gap at a removed step reveals its thrust as B+(s_{t+1} - A s_t - water push).
  - TD3+BC's per-tuple losses (`td3_bc.py` lines 186-245) use neither. So for this learner a smooth extrapolation past the knee is right in one world and wrong in the other.
  - The claim covers per-tuple learners only, not every possible use of the data.
- **L2.3 Neighbor residual.** rho_hat(slot) = min over k of [P_k(slot) - Q_stall(slot)].
  - **Supported slots** S9 = {-KNEE, -0.77499998, -0.75, -0.72500002, -0.69999999, -0.40000001, 0, 0.40000001, KNEE}.
  - **Predictors:**
    - P1: a quadratic least-squares fit on S9;
    - P2: a straight line fit on the five slots in [-KNEE, -0.69999999], the extrapolation driven by the edge;
    - P3: a cubic fit on S9.
  - **Why the minimum.** P1 and P3 use slots that CUTOFF's high-load data cover only through the 2% floor. Taking the minimum can only lower rho_hat, so including them makes the L2 test harder to pass, never easier.
  - **Sign control.** The p10 test uses max over k instead, so it is not made easier either.

**L2 PASS rule.** On the gated E2 anchors, separately for each continuation seed:
- **Item 5, estimator validity, is evaluated first.** max over k of |P_k - Q_nostall| <= 0.10 at s05, <= 0.15 at s10 and <= 0.30 at s20, at >= 90% of anchors. A miss is INCONCLUSIVE.
- Items 1-4 are computed only on predictor-valid anchors:
  1. rho_hat(s05) >= 0.4 at >= 70% of predictor-valid anchors, with median >= 0.75;
  2. rho_hat(s10) >= 0.5 at >= 70%, with median >= 0.75;
  3. rho_hat(s20) >= 1.0 at >= 70%;
  4. true drop Q_stall(-KNEE) - Q_stall(s10) >= 0.25 at >= 70%.
- **Item 6, sign control:** max-over-k rho_hat(p10) < 0 at >= 90%.

The positive thresholds encode the optimistic sign: smooth extrapolation over-values exactly where the actor's Q gradient points.

**The size of rho depends on the continuation.** rho is the one-step value gap under a stated continuation policy, and it scales with the state magnitude that policy produces: about 0.0075 x |Q| across proxies and saved Q^pi.
- The old full-data actors' saved Q^pi gives median rho at s05 of 1.13-1.45 at strong current, but 0.65-0.71 in the ordinary context. A clipped-LQR proxy at E2 gives 0.61.
- A competent learner therefore produces a smaller gap. The absolute L2 bars above apply only in step 0.4, under the fixed reference continuation, and a step 0.4 PASS does not carry over to Stage 1 learners.
- In Stage 1 the protocol instead gates whether each K1 criterion is *reachable* under the learners' own continuations (V0(d)). A criterion that a correct, smoothly extrapolating critic could not pass is declared INCONCLUSIVE, never FAIL.

**Link to Stage 1.** For a critic that extrapolates smoothly, KC at a slot is about rho_hat at that slot, under the same continuation.
- **K1a** needs KC_s05(CUTOFF) - KC_s05(RANDOM) >= +0.5. V0(b) allows |KC_s05(RANDOM)| up to 0.4 x rho_hat_R(s05).
  - The worst allowed case reaches 0.5 exactly when median rho_hat_C(s05) >= 0.5 + 0.4 x median rho_hat_R(s05), where C and R name each arm's own continuation.
  - rho_C >= rho_R (both >= 0.83) is sufficient for this, but not necessary.
  - The anchor sets differ slightly (K1a uses the common gated anchors, V0(b) the subset that is predictor-valid under RANDOM's continuation), so this worst case is approximate. V0(d) gates it directly.
- **K1c** needs KC_s10(CUTOFF) >= +0.5. It is reachable when median rho_hat_C(s10) >= 0.65: the bar plus item 5's 0.15 tolerance at s10. V0(d) gates this too.
- **V0(b)'s 0.4 coefficient** covers a correctly labelled RANDOM critic that rounds the slope-3 kink with a softplus width up to about 0.04-0.05.
  - The ratio KC/rho is 0.25 / 0.31 / 0.35 / 0.38 at widths 0.02 / 0.03 / 0.04 / 0.05, and it does not depend on the slope change or on the scale of rho.
  - rho_hat can sit up to 0.10 below rho at s05 (item 5's tolerance), so the coverage depends on predictor bias.
  - Wider rounding still ends Stage 1 INCONCLUSIVE whatever the hypothesis (Part 8.2).

### L3. Decision-relevant OOD: the native learner is harmed there

The scope is native learners only, in the stall world, at the 20k checkpoint and the E2 context, in every one of 3 fresh seeds. Exact rules are in Part 5, Stage 1.
- **L3a. The critic over-values the region.**
  - In absolute terms: KC_s10(CUTOFF) is positive and large.
  - Because of withholding: KC_s05(CUTOFF) exceeds RANDOM's, where RANDOM has labels.
- **L3b. The actor goes there.** CUTOFF enters R_deep- at a substantial rate. Every logged CUTOFF a0 is >= -KNEE, so BC pulls inward and the pull must come from Q.
- **L3c. Going there costs return, because of the hidden knee.**
  - Clipping the same stall-trained CUTOFF policy at the knee raises its return in the stall world.
  - It does so far more than it does when that same policy is evaluated in the no-stall world, where going past the knee really helps.
  - It also does so far more than for the RANDOM policy.

---

## Part 3. The condition: benchmark version `knee_edge_v1`

The recorder and reset settings come from the reach screen. Its rule (`REACH_SCREEN_RULE.md`) was written before `screens/reach_grid.py` ran, and that rule alone selected candidate **mix15 / C / d = 0.10**. Post-hoc sensitivity readings are in `screens/REACH_SENSITIVITY.md`.

1. **Withheld set.** P(row) = |float64(a0_exec_f32)| > KNEE, strict. It applies at every state, load, time and setting of a1..a3, through one shared implementation (Part 4.1).
2. **Reset law, `knee_mix15`.** Each episode draws one mixture uniform from its own named stream:
   - with probability 0.85, current c ~ U[0.1, 0.7];
   - with probability 0.15, c ~ U[0.9, 1.1].
   - Then `options["start"]` sets that current range, and the environment draws in its canonical order (`draw_starts`, lines 308-315): kick normals, current, wave amplitude U[0.1, 0.4], phase. Kick 0.5.
   - No current in (0.7, 0.9) is ever drawn; this gap is disclosed.
3. **Recorder, `knee_edge_mixture_v1`.** Each step draws exactly five uniforms (u, v, w1, w2, w3) from a fixed stream:
   - if g >= 0.4 (inclusive) and u < 0.98: a0 = -0.9 + 0.2v, a window on [-0.9, -0.7];
   - otherwise a0 = -1 + 2v, a uniform floor;
   - a1..a3 = -0.7 + 1.4 w_k, which never stall.
   - **Density:** p(a0 | g >= 0.4) = 0.98/0.2 * 1[a0 in window] + 0.01, and p(a0 | g < 0.4) = 0.5. Each row logs the recorder's log-density; CUTOFF's truncated density is documented (L1.3).
   - **Disclosed:** a1..a3 in (0.7, KNEE] are never logged in any arm. That is a second unsupported action region, shared by all arms (Part 6.10).
4. **Worlds.**
   - Primary: the stalling world at stall_slope 3.0, the pre-registered retune.
   - Placebo: the no-stall twin.
   - Both use the same reset seeds and draws. Requested actions, loads and masks are byte-identical; states diverge at the successor of the first W row.
5. **Collection.** 1,280 episodes per world. `_split_assignment` gives 768 fitting (code 0), 256 tuning (code 1), 128 selection (code 2) and 128 final-calibration (code 3) episodes.
6. **Expected shares.**
   - P(g >= 0.4) = 0.575 exactly: the recorded law is symmetric about 0.4, and the deep component always has g >= 0.5.
   - From Monte Carlo: P(g >= 1.0) = 0.078, and 17.3% of episodes contain a row with g >= 1.0.
   - First W step: P(<= 1) = 0.58, P(<= 4) = 0.86, P(<= 9) = 0.97, median 1.
7. **Arms, in each world.**
   - **CUTOFF** drops every row in W: 0.494 of high-load rows and 0.200 of low-load rows, about 28,347 ± 284 of 76,800 fitting rows.
   - **RANDOM** keeps, in each stratum (episode x ten-step bin x load band), exactly CUTOFF's count, chosen uniformly from all the stratum's rows.
     - This reimplements the sampling procedure of `conditional_ood_gate_central_quartile_v1/geometry.py` `control_masks`, with the knee predicate and **three** load bands: low (g < 0.4), mid (0.4 <= g < 1.0) and deep (g >= 1.0), with lower edges inclusive.
     - The prior function itself uses a two-level flag and cannot be reused unchanged.
     - The control seed is named in `seeds.json`.
   - **High-load BC anchors:**
     - CUTOFF -0.7263, exact from the recorder density: 0.49 x (-0.75) / 0.506. The Monte Carlo gives -0.7264 ± 0.0010.
     - RANDOM -0.7747, Monte Carlo only.
8. **Host.** Unifloral TD3+BC f2dc1278 with `td3_alpha` 2.5 and noise 0.2/0.5.
   - Gamma 0.98 is bound explicitly in the prior runtime (`runtime.py` line 138); the upstream default is 0.99. The binding is re-checked.
   - Fresh initialization in every arm.
   - Normalization uses the method of `central_quartile_training_v1_evaluation_rerun1/data.py` lines 145-187: the source is the retained intersection CUTOFF ∩ RANDOM within each world, epsilon 1e-3.
   - Per-world normalization statistics are reported.
9. **Calibration rows, common to both arms within a world.** Source: the first 128 tuning (code 1) episodes by group id, one row per episode. The rule is adopted verbatim from the prior study (`conditional_ood_gate_central_quartile_v1/geometry.py` lines 320-346), with W as the ineligible set:
   - an original time per episode is drawn from PCG64(calibration seed);
   - if that row is not in W, it is kept;
   - otherwise a uniform row not in W, from the same episode, is drawn from PCG64(replacement seed);
   - an episode with no eligible row fails closed.
   - Both seeds are named in `seeds.json`. Expected replacements: 47.5 ± 5.0 of 128, with 6 SD bounds [17, 78] (`random_control_mc.py`); the count is reported.
   - The set has 0 rows in W, so it is valid for CUTOFF and identical for RANDOM. BCA's calibration therefore does not differ between arms.
   - The other 128 tuning episodes and codes 2 and 3 stay sealed.
10. **Contexts.**
    - **E2 (gated):** c ~ U[0.9, 1.1], wave U[0.1, 0.4], kick 0.5. Every E2 step has g >= 0.5 (the law's infimum), and half have g >= 1.0.
    - **E1:** gates only the learning-adequacy checks (R0's adequacy check, V0(a), and the Stage 2 adequacy floors); otherwise reported. Each check uses its own named stream: `r0_e1_starts`, the Stage 1 E1 bank and the Stage 2 E1 bank, mutually disjoint.
11. **Scope.** Every claim is confined to the stall_slope 3.0 world of `knee_edge_v1`.

---

## Part 4. Implementation certification

New code goes in `knee_edge_ood_v1/`, created only after approval A1. No frozen, hashed or sealed folder is edited: `experiments/shift/` and every central-quartile package stay untouched.

### 4.1 Shared region module `knee_region.py`

Its source hash is recorded in every receipt.

- **`past_knee_a0(actions, kind="executed")`** returns (mask, side).
  - It takes float32 input of shape (..., 4). With kind="requested" it first applies the collector's cast, `requested.astype(np.float32)`.
  - It compares strictly in float64 against KNEE.
  - It raises on NaN, on inf, on |a| > 1, and on float64 input passed as executed. -0.0 counts as 0.
- **Other functions:** `depth`, `in_deep` (d >= 0.10 inclusive), `load_g`, `load_band` (lower edges inclusive) and the slot table of Part 0. The recorder's window rule calls `load_band`.
- **Import-time assertion:** on a boundary fixture, the mask equals canonical `stalled(a, Physics(stall=True, stall_slope=3.0))[..., 0]`.
- **Exclusive use.** It is the only predicate used by:
  - the recorder validator and the mask builders;
  - every loader;
  - the normalization intersection and calibration selection;
  - the evaluation counters (free steps, the clip intervention, probe labels, visited rows, target-smoothed actions);
  - the gradient audit and the analysis.

### 4.2 Independent second implementation

It comes from a different author or session, with no shared code.

- **Region:** an IEEE-754 bit test, `(view(uint32) & 0x7FFFFFFF) > 0x3F4CCCCD`. It raises on NaN and inf, as the primary does.
- **Depth and bands:** exact `Fraction` arithmetic on the float32 values.
- **Third check:** canonical `stalled()[..., 0]` on stall-world rows.
- **Agreement required, row for row:**
  - every logged row in both worlds;
  - every mask;
  - every evaluated action in every stage;
  - every anchor candidate and visited row.
- **One disagreement stops the study.**
- **Already done, read-only, by the design panel:** 0 disagreements over all 671,090 float32 values with |x| in [0.79, 0.81].

### 4.3 New dataset schema version

The current contract rejects this design, for six reasons:
- `datasets.py` `_validate` requires `start_family == "recorded"` (line 255), state-independent density semantics (lines 33-35 and 264), and an initial current in [0.1, 0.7] (lines 343-346);
- the lineage replay hard-codes `START_LAWS["recorded"]`;
- `start: "recorded"` is hard-coded at line 171;
- `action_region` uses a0 > 0.5;
- the rerun1 reader raises "Recorder law changed" (`data.py` line 70);
- that reader pins a path and a SHA.

The new version adds:
- density semantics that depend on load, with the truncated CUTOFF density documented;
- the `knee_mix15` start family, with the mixture uniform on its own named stream and the canonical draw order inside the environment;
- a lineage replay of the 5-uniform recorder, checked bit for bit;
- role-safe reader bindings, including the calibration source of Part 3.9;
- derivation of the data plan.

Cross-rejection is tested against `datasets.load_dataset` and `_validate`: the old validator must reject new data, and the new one must reject old data.

### 4.4 Proof that every learner received the masked arrays

Stage 1 must reuse the prior pipeline, which has the shadow auxiliary. That pipeline does not call upstream `make_train_step`. Its `runtime.py` `make_adapter` (lines 197-209) binds `context.train_transition` and samples rows from it directly.

- **Assertions at every `make_adapter` call**, including the 12 Stage 1 warmups, 12 Stage 1 branches, 21 Stage 2 branches, 3 BC-only adapters and every check-routine adapter. They run on the exact arrays passed in, after the float32 conversion done by `data.py` `bank()`:
  - for CUTOFF, 0 rows in W, using the shared predicate;
  - the length equals the reconciled retained count;
  - a sha256 of obs, action, reward, next_obs and done, in that fixed order and dtype, matched against the mask builder's receipt.
- **The same assertions on every other stream:**
  - `heldout_transition` (calibration, `runtime.py` line 303);
  - the probe bank;
  - the scale bank;
  - the normalization fixture (lines 131-163).
- **BC-only sanity run.** It gets its own adapter, uses the same CUTOFF ∩ RANDOM normalization (not `bc.py`'s own dataset mean, line 199), and is covered by the same assertions.
- **Forced-probe arrays.** One probe-building function serves every forced-action evaluation: KC, F, the onset contrast, the reach forecast, D1/D2, and the 4.8 reproduction of saved probes. Every gated probe array is asserted, by both region implementations, to hold the Part 0 slot value at index 0 and exactly 0.0 at indices 1-3.
- **Alignment fixtures**: synthetic episodes with known W rows, which must catch:
  - the mask applied to requested instead of executed actions;
  - obs and next_obs offset by one row;
  - a mask or dataset bound from the other world (world tags must match, and a cross-world binding raises);
  - RANDOM strata built on the wrong time index;
  - an unstratified RANDOM or one blind to load band (both must fail L1.4);
  - a RANDOM with the prior two-level load flag (it must fail the three-band per-stratum count check in 4.7).

### 4.5 Independent Q^pi evaluator

A second evaluator writes a raw loop over `dynamics()` with explicit float64 state, discount and horizon. It is checked against the environment-based evaluator at every step 0.4 anchor, at three slots (-KNEE, s10, s20), in all three worlds, to 1e-9 relative. This catches state, horizon, discount and continuation errors that L2.1's identity cancels.

### 4.5b World certification: every run is in the world it claims

The canonical `Physics()`, `world_physics()` and `ThrusterChainEnv` all default to stall_slope 1.5 (`canonical/thrusterchain.py` lines 116, 143, 398). Every check in 4.7 is insensitive to the slope: masks and counts depend only on actions, and `stalled()` ignores the slope. A run at 1.5 by mistake would pass them all and silently test about half the gap. So:

- **Physics assertion in every receipt.** Every collection, replay, anchor rebuild, Q^pi evaluation and free rollout asserts `physics.as_dict()` equal to the named world's dict and records it, tagged with the world:
  - stall: stall True, stall_slope 3.0;
  - no-stall: stall False;
  - step 0.4's slope-1.5 world, tagged "identity check only".
- **Data-level known-answer check.** On stall-world rows in W, the thruster-0 thrust reconstructed from (s, a, s') with the canonical matrices must equal T3(a0) to 1e-9, and must differ from the slope-1.5 thrust. For example, T3(-1) = -0.2, whereas slope 1.5 gives -0.5.
- **Literal identity.** The L2.1 identity test uses the literal Part 0 formula sign(a0)(KNEE - 3(|a0| - KNEE)), never `physics.stall_slope`. A slope mix-up therefore breaks the identity instead of hiding inside it.
- **Synthetic fixture.** A deliberate slope-1.5 dataset must fail both the reconstruction check and the identity test.

### 4.6 Tests, on synthetic arrays only

- **Predicate boundaries, built in float32:**
  - ±KNEE is False;
  - ±`np.nextafter(np.float32(KNEE), np.float32(1))` is True;
  - ±`np.nextafter(np.float32(KNEE), np.float32(0))` is False;
  - ±1 is True; +0.0 and -0.0 are False;
  - requested 0.8, 0.80000001 and 0.8000000119 cast to False;
  - NaN, inf, |a| > 1 and float64-as-executed raise, in both implementations.
- **Properties:** symmetric in sign; monotone in |a0|; unaffected by a1..a3 and by state; works for batch shapes (N, 4) and (B, T, 4).
- **Load-band edges** at g = 0.4 and 1.0, using float32 pairs whose float64 sum lands exactly on the edge or one ulp below it.
- **Recorder:**
  - five uniforms per step, in a fixed order, consumed identically on both branches;
  - one mixture uniform per episode;
  - actions and loads byte-identical across the twins;
  - executed |a1..a3| <= 0.7;
  - log-density matches the closed form within 1e-14;
  - frequencies within 99.9% binomial bounds.
- **Host config:** gamma 0.98; done only at t = 99, with no bootstrap; `td3_alpha` 2.5; noise 0.2/0.5; pin f2dc1278; the world physics of 4.5b.
- **Cadence** is labeled from saved evidence: actor 5,000 and critic 10,000 updates per 10k branch.
- **The RANDOM known-fault tests** of L1.4.

### 4.7 Reconciliation, per world

There are two kinds of check, and they carry different labels.

**(i) Plumbing checks. A failure is INCOMPLETE: the code or data path is wrong.**
- 1,280 episodes split 768 / 256 / 128 / 128.
- 0 retained rows in W for CUTOFF in every stream.
- RANDOM's per-stratum counts equal CUTOFF's, on three-band strata.
- Retained and removed sets are disjoint, and their union is complete.
- Masks are byte-identical across worlds.

**(ii) Expected-value checks. A failure is INCONCLUSIVE: the realized data are not the designed condition.**
- The values are the 6 SD bounds in `screens/random_control_mc_output.txt` (sha256 2dc0c383…f716), sealed at A1:
  - All bounds are rounded outward.
  - **Shares:** high-load share of fitting rows in [0.503, 0.647]; deep share in [0.036, 0.120]; episodes containing a deep row in [0.087, 0.260].
  - **CUTOFF:** removed count in [26,643, 30,052]; removed fraction in [0.479, 0.509] at high load and [0.187, 0.214] at low load; high-load anchor in [-0.733, -0.720].
  - **RANDOM:** W retention in [0.460, 0.504]; high-load anchor in [-0.781, -0.768]; L1.4 bands as stated there.
  - **Calibration replacements** in [17, 78].
- Collection is seeded, so a miss here would recur on a rerun. That is why it is INCONCLUSIVE and stops the step instead of being retried.
**(i, continued) Further plumbing checks. A failure is INCOMPLETE.**
- **Twin check, on the dataset's float64 `states` field:**
  - identical across worlds up to and including each episode's first W row;
  - different exactly at its successor;
  - positions reported.
- **Replay:** every row replays bit for bit, and terminal rows are reported per arm.
- **Seeds.** The collision audit is by (entropy, spawn_key) and by derived seed, not raw integers. Reusing any earlier dataset entropy is forbidden, and none of the 128 consumed development anchors is reused.
- **Leakage protection.** The protection is the zero-W check on every stream (4.4). The rerun1 calibration-leakage check (`data.py` line 155) compares ids from disjoint episodes, so it cannot fire, and it is not relied on.

### 4.8 Checks on frozen checkpoints (user-run; agents never open checkpoints)

- **G0.5 and the measured edge slopes** (step 0.3).
- **Known-answer reproduction** by the new evaluator on the frozen central-quartile native 20k cells:
  - **the forced-probe critic path:** reproduce the saved per-probe `q1` and `twin_min` at the saved per-probe `actions` (shape 32 x 6 x 4; slots 1-5 are a0 in [-0.5 ... 0.5] with a1..a3 = 0) to 1e-6, in all nine cells. This exercises observation normalization, the remaining-time input, a0 at index 0, a1..a3 = 0, and the choice of Q1 or twin minimum, the same path that computes KC, F and the AUROCs;
  - saved lambda to 1e-6;
  - saved per-probe Q^pi to 1e-9 relative, which covers anchors at t > 0 with horizons of 100 - t;
  - past-knee counts recomputed exactly;
  - identity residual <= 1e-6;
  - stall equals no-stall within the knee;
  - state digest unchanged.
- **Nine cells asserted.** Every saved-result screen lists both `authorized1` and `continuation1`.
  - At logged high-load probe rows in the native 20k gradient audits, actor commands go past the knee at:
    - full 117 of 1,242;
    - hole 118 of 1,140;
    - random 126 of 1,137.
  - That is 361 of 3,519 (10.3%) pooled over all nine cells. The known-answer check compares per cell.

---

## Part 5. Staged gates, kill rules and outcome labels

### 5.0 Rules for every step

- Each step needs its own approval, and every threshold is fixed before the step runs.
- An outcome can only stop the plan or let it continue. No outcome may change the recorder, world, arms, seeds or thresholds.
- One attempt per approval.
- **After an INCOMPLETE:**
  - A rerun needs a fresh approval and repeats only the failed unit, with identical seeds and streams. Completed units stay sealed, and the discrepancy from any GPU nondeterminism is reported.
  - **At most one code-only fix per unit** is allowed before its rerun.
    - Seeds, thresholds, data and design stay untouched, and the diff is recorded and approved with the rerun.
    - The diff must leave untouched the code that produced already-sealed units; otherwise those units must reproduce bit for bit under the fixed code.
  - A second failure of the same unit on *any* check, under code unchanged since that unit's previous attempt, makes the step INCONCLUSIVE and stops it, whatever the failing value. This prevents re-rolling GPU nondeterminism.
- **After an INCONCLUSIVE:** the step stops. It cannot be re-attempted under the same design, because a fresh attempt would re-roll GPU nondeterminism until something passed.
- **Recording.** Every hypothesis-criterion outcome is recorded per seed whatever the label, so a clean failure is never hidden behind an INCONCLUSIVE.

**Labels, in order of precedence (the first that applies wins):**

| Label | Meaning |
|---|---|
| INCOMPLETE | Numerical or execution failure: predicate disagreement, identity residual or evaluator mismatch beyond tolerance, a twin mismatch, a per-stratum count or split mismatch (4.7(i)), a missing job, an exhausted cap |
| INCONCLUSIVE | A validity precondition failed, so the step says nothing about the hypothesis |
| FAIL | The measurement was valid and a hypothesis criterion failed. **FAIL-consistent**: the *same named* criterion fails in all three seeds. **FAIL-mixed**: some criterion fails in some seed, but no single criterion fails in all three. **FAIL-early**: R0 stopped the run at 10k, after the 10k adequacy check passed. It carries one fixed reading (Stage 1, R0) |
| PASS | Every rule holds in every seed |

Every FAIL stops the plan at that step. Apart from FAIL-early's fixed reading, a mechanistic reading is attached only to FAIL-consistent. FAIL-mixed is never read as equivalence or as absence of an effect.

**Stage 2 labels, in order of precedence:** INCOMPLETE, INCONCLUSIVE, DETECTION-FAIL, NOT-PROMISING (consistent or mixed), PROMISING. They are defined at the end of Stage 2. A sub-screen that cannot be computed because its guard fails is recorded as "not computed (guard)", never with a reserved label.

### Step 0.1. Feasibility screens on saved JSON (done)

These are read-only. Their thresholds were set after earlier previews of the same data, so they are feasibility only.

- **Reach.**
  - Under the as-written rule, the selected candidate's forecast entry into R_deep- has a minimum of 0.562 at probes and 0.317 on visited rows (bars 0.40 and 0.30), with lambda corrected for the new reset law (scale 0.751).
  - Post-hoc sensitivity readings change which candidate the rule would select. mix15/C stays feasible under all of them, but only the as-written rule made the selection.
  - The binding cell clears its bar by 2 rows: 40 of 126, against 38 needed.
  - The slopes are extrapolated from old critics, evaluated at currents those critics never saw.
- **Value at stall_slope 3.0.** The median rho near s10 is 2.12-2.58 by seed at strong current, an estimate from quadratic fits evaluated at a0 = -0.9. Step 0.4 replaces it with exact values at s10.
- **Predicate:** 0 disagreements.
- **Arms:** the simulated values of Part 3.7 and 4.7.

### Step 0.2. Code, schema and synthetic tests (approval A1)

There is no simulator, no training and no checkpoint access. This step writes Part 4 in full, plus `seeds.json`, the sealed protocol and the sealed expected values.

**Label table**
- Any test failure, any predicate disagreement, or any reconciliation value that cannot be re-derived from the sealed script: **INCOMPLETE**. It blocks every later approval until fixed. Fixing code before any data exists is not driven by outcomes.

### Step 0.3. Frozen-checkpoint checks (approval A2, user-run)

This step does forward passes and deterministic prefix replay on existing banks.
- Cap for the gated items: 0.5M transitions or 30 minutes; the first cap to bind fails closed.
- The warning-only onset contrast runs after every gated item is sealed, under its own sub-caps of 200k transitions and 10 minutes. Any cap that binds during it records "not computed", never INCOMPLETE.

- **G0.5, the fold test.**
  - **Critics and points:** each frozen native 20k critic (full, hole and random, x 3 seeds), at saved high-load probes in the ordinary and strong-current banks. Q1 is computed at the float32-executed values a0 in {-0.5, -0.60000002, -0.69999999, -KNEE, s05, -0.89999998, -1}.
  - **Convention.** The gated F, the steep-probe selector R_lin, and the warning-only onset contrast (including the first action of its Q^pi rollouts) use a1..a3 = 0, matching V0(b). Only the reach forecast below uses the actor's own a1..a3.
  - **Fold error at full throttle:** F(s) = Q1(s, -1) - Q1(s, -0.5) + 0.22499999.
    - The constant is 0.3(1 - T15(-1)^2), with T15(-1) = -0.5000000298 at stall_slope 1.5, which is what these critics were trained on.
    - F > 0 is over-valuation. |F| is reported.
    - G0.5 stays one-sided because the failure it screens for is over-valuation of the fold, the error withholding would also produce.
  - **Onset contrast, the early warning for V0(b).** It is a warning only, run last under its own sub-caps (above). It is computed per seed on the full-data critics, as the ratio of medians, median |KC15_s05| / median rho15(s05), over their steep ordinary probes, where:
    - KC15_s05 = [Q1(s05) - Q1(-KNEE)] - [Q^pi_1.5(s05) - Q^pi_1.5(-KNEE)];
    - rho15(s05) = Q^pi_nostall(s05) - Q^pi_1.5(s05);
    - the Q^pi values are exact, under each critic's own frozen actor, clipped, by deterministic prefix replay within this step's cap.
    - This uses the -KNEE reference as V0(b) does, so kink rounding, which pulls the -KNEE value down, shows up. An earlier F05 that compared two points both away from the kink could not detect it.
  - **Steep probes:** R_lin = 2[Q^pi(-0.5) - Q^pi(-0.25)] + 0.22499999 >= 1.0. Its Q^pi values are the saved per-probe values at the slots -0.5 and -0.25 (recorded convention: a1..a3 = 0, each policy's own continuation), so no new replay is needed.
  - **Governing bank:** ordinary. It is inside the old critics' training data; strong current is reported separately.
- **Measured edge slopes for the reach forecast.** The model is fixed now:
  - **Rows:**
    - strong-current high-load probes (the probe forecast);
    - saved high-load visited rows with current >= 0.9, about 72 per cell (the visited forecast).
  - **Forward passes for the forecast:** at every row (probes and visited), Q1 at the seven a0 values above plus the four slots {-0.25, 0, 0.25, 0.5}: 11 points. All are taken at one convention, the actor's own a1..a3 at that row. For the forecast only, the a1..a3 = 0 values are reported as descriptive.
  - **Gating model.** It uses no point near the knee, because the old critics were trained with labels past it: kink rounding bends Q1 near -KNEE, and fits including -KNEE, -0.9 or -1 were shown to bias the forecast toward a false stop.
    - **Edge slope** s_edge = [Q1(-0.5) - Q1(-0.69999999)] / 0.19999999, the secant across the supported edge, centered at -0.6. A least-squares line through equally spaced points has the same slope. Its noise is about 7 sigma, against about 25 sigma for the four-point fit it replaces.
    - **Curvature** q2: from the quadratic Q1 = q0 + q1 a0 + q2 a0^2 fitted on the five slots {-0.5, -0.25, 0, 0.25, 0.5} at the same a1..a3 convention. It is well conditioned and lies far from the knee.
    - **Combined slope term:** q1' = s_edge + 1.2 q2, the linear coefficient consistent with slope s_edge at -0.6.
  - **Equilibrium:**
    - pi0 = clip((m + 2 lambda' q1') / (1 - 4 lambda' q2), -1, 1), with m = -0.7263 and lambda' = the cell's saved 20k lambda x 0.751;
    - a row with 1 - 4 lambda' q2 <= 0 counts as reaching -1.
    - The four-point and six-point fits are reported as descriptive only.
  - **Forecast:** the fraction of rows with pi0 <= -(KNEE + 0.10), per cell, nine cells.
- **Known-answer reproduction** (4.8).

**Label table**

| Outcome | Label | Consequence |
|---|---|---|
| Known-answer mismatch | INCOMPLETE | Fix the evaluator, then a fresh A2 |
| Median F <= 0.5 among steep ordinary probes of the full-data critics in every seed | (pass) | Continue |
| 25-50% of those probes have F <= 0.5 in some seed, or the onset-contrast ratio exceeds 0.3 in some seed. 0.3 sits below V0(b)'s 0.4 coefficient, because the old critics' calibration is optimistic | (pass, flagged) | Continue only with the written prediction "Stage 1 V0(b) may end INCONCLUSIVE", acknowledged by the user at A3 |
| Under 25% in any seed | INCONCLUSIVE | Labelled critics leave most steep states over-valued. Knee error with this host could then be approximation failure, not withholding, and the design cannot separate the two. Stop; the region is *not* declared non-OOD |
| Probe forecast < 0.35 in any cell, or visited forecast < 0.25 in any cell | FAIL (reach) | Stop before data collection. Old critics forecast no reach, even with measured slopes |

The reach bars sit below step 0.1's on purpose. This step adds information, and it must not tighten into a forking path. A lenient feasibility bar can waste compute, but it cannot create a false positive: R0 and K2 decide.

### Step 0.4. Collection and exact probes (approval A3)

This step uses the simulator but does no training. Cap: 2.5M transitions or 1 hour.

- **Collect** `knee_edge_v1` and its twin, 2 x 1,280 episodes, with a replay check.
- **Exact Q^pi probes (user-run,** because they load the frozen actors and agents never open checkpoints**):**
  - 64 fresh E2 anchors, 16 at each t, from separate named streams; they never enter replay. The 48 at t in {0, 25, 50} are gated.
  - 32 E1 mid-load anchors, reported only.
  - Continuations: the three frozen full-data native 20k actors, clipped.
  - Worlds: stall_slope 3.0, no-stall, and stall_slope 1.5 (identity check only).
  - Forced a0: S9, s05, s10, s15, s20, p10, and the T-image of each W slot in each stall world.
- **Independent evaluator cross-check** (4.5).
- **Reconciliation** (4.7), **L1.4** and **L1.5**.

**Label table**

| Outcome | Label |
|---|---|
| Identity residual > 1e-6 * max(1, \|Q\|); stall and no-stall differ on supported slots by > 1e-9 relative; evaluator cross-check fails; a twin mismatch; a per-stratum count or split mismatch (4.7(i)); any executed \|a1..a3\| > 0.7 | INCOMPLETE |
| L2 item 5 (estimator validity) fails; L1.4 outside its bands; L1.5 fails; any 4.7(ii) expected-value check outside its bounds | INCONCLUSIVE |
| L2 item 1, 2, 3, 4 or 6 fails for any continuation seed | FAIL under the reference continuation: the region is not model-relevant for these continuations, so stop. The continuations are old actors running outside their training states, so this is recorded as "not shown model-relevant", not as "non-OOD" |
| Everything holds | PASS. It permits only the request for Stage 1 |

### Stage 1. Native damage gate (approval A4)

**Data.** `knee_edge_v1` (stall 3.0) and its twin, with the hashed masks from step 0.4 and the common calibration set (Part 3.9).

**Arms.** {CUTOFF, RANDOM} x {stall 3.0, no-stall} x 3 fresh seeds = 12 native runs.
- Seeds are paired across arms and worlds within a seed index.
- Each run is a 10k warmup (native, with the shadow auxiliary fit and calibration so that Stage 2 can fork), then a 10k native branch.
- Sanity check: a BC-only actor on stall CUTOFF, 3 seeds.

**R0, at the 10k fork.**
- **Adequacy first.** For stall CUTOFF and stall RANDOM: J_E1(10k) - J_E1(untrained, same initialization) >= +50 in every seed, on 32 fresh E1 starts from the named stream `r0_e1_starts`. Otherwise INCONCLUSIVE, stop.
- **Reach.** Run the stall CUTOFF actor on 32 fresh E2 starts from the named stream `r0_e2_starts`, which is disjoint from every gated bank.
  - If its R_deep- rate is below 0.05 in every seed: **FAIL-early**. Stop, with the fixed reading "the host contained the extrapolation at 10k; 20k was not observed".
  - This saves about 1.7 h.
- At equilibrium the reach radius 2 lambda |dQ1/da0| equals |pi0 - m|, so it is reported, not tested.

**Evaluation.** Fresh streams, never used in step 0.4 and never reused in Stage 2.
- **Banks:** E2 with 128 starts (gated) and E1 with 64. Checkpoints 15k and 20k; 20k is gated.
- **Free rollouts:**
  - pi, pi_clip and pi_clipall;
  - J and Δclip in the training world;
  - **world swap:** every stall-trained policy is also run in the no-stall world on the same starts, giving Δclip(pi, no-stall) and A(pi);
  - R_deep- and any-depth W rates by band; W+ rate; a1..a3 rates past the knee and in (0.7, KNEE].
- **Q probes:**
  - 64 fresh E2 anchors (48 gated) and 32 E1 anchors, common to all arms within a world;
  - forced slots S9, s05, s10, s15, s20, p10, with a1..a3 = 0 for the gated KC. A common value makes KC comparable across arms at the same state. A second set at the actor's own a1..a3, plus the actor's own action, is reported;
  - exact own-continuation Q^pi, clipped as primary and unclipped as a report;
  - Q_nostall at s05, s10, s20 and the S9 slots under each stall CUTOFF and RANDOM policy's clipped continuation, which V0(d) and V0(b) need;
  - KC for Q1, Q2 and the twin minimum;
  - a1 = ±KNEE against a1 = ±0.69999999 probes, with a0 at the actor's value (Part 6.10);
  - KC also at a sample of 48 policy-visited E2 states per arm (t in {0, 25, 50}), via prefix replay of the free rollouts.
- **Gradient audit** at 10k, 15k and 20k: lambda, reach radius, gradient signs.
- **Diagnostics:**
  - the rate of target-smoothed actions in W on replay (estimated at about a third for CUTOFF);
  - the absolute Q1 error at -KNEE by arm, including the offset, because smoothing into W can contaminate the -KNEE reference;
  - the per-world state shift (standardized mean difference of each input) and lambda.

**Decision rules.** These apply in the stall world, at E2 and 20k, in every seed, on the common gated anchors (defined next).

- **Anchor sets.** The **common gated anchors** are the gated anchors valid in every arm compared. K1a, K1c and V0 all use them.
  - V0(b) further restricts them to anchors that are predictor-valid under RANDOM's own clipped continuation.
  - V0(c) floors both sets.
- **V1 (INCOMPLETE):** any numerical failure from the table in 5.0.
- **V0 (INCONCLUSIVE).** These validity items concern only the runs the gated criteria use: stall CUTOFF, stall RANDOM, and the world-swap evaluation of the stall CUTOFF policy. The no-stall-trained runs' adequacy is reported here; that of no-stall CUTOFF gates the request for A5.
  - (a) J_E1(20k) - J_E1(untrained, same initialization) >= +50 for stall CUTOFF and stall RANDOM. This is a lenient floor: the prior study's gains were 155-235.
  - (b) RANDOM represents the knee, under RANDOM's own clipped continuation:
    - L2 item 5 (estimator validity) passes at >= 90% of the common gated anchors;
    - median rho_hat_R(s05) >= 0.20. This resolvability floor is twice item 5's 0.10 tolerance at s05; it is not a magnitude requirement;
    - |median KC_s05(RANDOM)| <= 0.4 x median rho_hat_R(s05), on the predictor-valid subset;
    - RANDOM's R_deep- rate <= 0.10.
    - The rate clause tests whether RANDOM's labels keep its actor out. It keeps 42.4% of high-load W labels, dense on [-0.9, -0.8], plus the full fold at low load (L1.4).
  - (c) At least 32 common gated anchors, and at least 32 of them predictor-valid under RANDOM's continuation.
  - (d) Under each stall CUTOFF policy's own clipped continuation:
    - L2 item 5 passes at >= 90%, and item 6 (sign control) holds;
    - **K1 is reachable:** median rho_hat_C(s05) >= 0.5 + 0.4 x median rho_hat_R(s05), for K1a, and median rho_hat_C(s10) >= 0.65, for K1c.
    - If K1 is not reachable, a correct, smoothly extrapolating critic could not pass it, so the outcome is INCONCLUSIVE, not FAIL.
    - L2 items 1-4 under these continuations are reported. They are not gated, because their absolute bars depend on how competent the continuation is (Part 2).
  - (e) a1..a3 entries past the knee <= 0.5% of E2 steps, for stall CUTOFF and stall RANDOM in the stall world, and for the stall CUTOFF policy in the no-stall world.
- **Hypothesis criteria (FAIL):**

| Criterion | Rule | What it shows |
|---|---|---|
| K1a | median over common anchors of [KC_s05(CUTOFF) - KC_s05(RANDOM)] >= +0.5 | Over-valuation caused by withholding, at a depth where RANDOM has labels |
| K1c | median KC_s10(CUTOFF) >= +0.5, and >= 0.5 x median rho_hat(s10) under the same policies' own continuations at the same anchors | Absolute over-valuation inside R_deep- |
| K2 | CUTOFF R_deep- rate on E2 free steps >= 0.25 | The actor goes there |
| K3a | Δclip(pi_CUTOFF, stall) >= +2.0 | Going there costs return |
| K3b | Δclip(pi_CUTOFF, stall) - Δclip(pi_RANDOM, stall) >= +2.0 | The cost exceeds the labelled control's |
| K3c | Δclip(pi_CUTOFF, stall) - Δclip(pi_CUTOFF, no-stall world, same policy) >= +2.0 | The cost comes from the hidden knee: the same policy is not hurt where going past the knee helps |

- **Consistency checks**, reported and not gated:
  - K1b, the within-arm KC_s10 contrast across worlds. It is roughly the true gap rho, so it largely repeats L2.
  - CUTOFF minus RANDOM entry, which V0(b) and K2 already imply.
  - The no-stall-trained CUTOFF policy's entry rate and Δclip. Its Δclip is uninformative when its entry rate is below 0.05.
  - The BC gradient sign and the BC-only actor's entry rate, which are true by construction.
  - K3a is close to implied by K2 plus the identity. It stays gated as the harm measurement.
  - K3c nearly duplicates K3a when the policy's no-stall Δclip is near 0 or negative, as expected. It adds the attribution to the knee. The pi_clipall version of every Δclip is reported.
- **Pre-registered secondary outcomes**, with predicted directions, reported:
  - J(RANDOM) - J(CUTOFF) (> 0);
  - A(pi_CUTOFF) - A(pi_RANDOM) (> 0);
  - the triple difference of J.
  - Noise reference, from the closest saved proxy (strong current, native 20k): hole-minus-random spreads of +4.9 / -11.5 / -41.0, with a free-episode SD of 23-44.
  - A J miss alongside a Δclip pass reads as "damage present within the policy, below between-arm seed noise".
- **Power.** No power estimate exists for this design: the seed variance available comes only from old data in which nothing at the knee was withheld. Six criteria and five V0 items must hold in each of three seeds, so a PASS is far from certain even if the hypothesis is true. This is disclosed so that a FAIL-mixed is not over-read.

**Readings for FAIL-consistent only.** They apply in order. The first criterion that fails in every seed governs, and "passes" means passes in every seed. If a reading's precondition (the earlier criteria passing in every seed) is not met, no reading is attached.
1. K1c fails: this Unifloral critic does not over-value past a dense edge, even past a proven kink. BCA has nothing to repair here.
2. K1c passes and K1a fails: over-valuation at the knee's onset is not caused by withholding. This reading is given only if median KC_s05(RANDOM) <= 0.1 x rho_hat(s05) in every seed. Otherwise RANDOM's own kink rounding could explain the miss, and no reading is attached.
3. K1 passes and K2 fails: the alpha 2.5 BC anchor contains the over-valuation. No K3 reading is given, because without entry there is nothing to harm.
4. K2 passes and K3a fails: this contradicts L2. Audit first.
5. K2 and K3a pass and K3b fails: the cost of entry is not larger than under the labelled control.
6. K2, K3a and K3b pass and K3c fails: the harm does not come from the hidden knee.

**PASS** permits only the request for Stage 2.

### Stage 2. BCA comparison (approval A5, only after a Stage 1 PASS)

Inference is conditional on the forks that passed Stage 1.

- **Frozen rules**, copied with hash lineage from the prior sealed study: kappa 3; auxiliary architecture and calibration unchanged; a distance-penalty arm matched by logged-action gradient RMS at 10k (an off-policy match, disclosed).
- **Arms.** Only the branches a gated or reported Stage 2 rule uses are trained: 21 branches of 10k updates.
  - From each stall CUTOFF fork: BCA, distance, and a **native replicate**.
  - From each stall RANDOM and each no-stall CUTOFF fork: BCA and a native replicate.
  - The no-stall RANDOM forks, and distance branches outside stall CUTOFF, feed no Stage 2 rule and are not trained.
  - **Seeds.** Stage 2 has its own `branch_native_seed` and `branch_aux_seed` per fork in `seeds.json`, distinct from Stage 1's and collision-audited.
    - The prior pipeline derives a branch's key as fold_in(fork rng, branch_native_seed), with no arm term, and looks seeds up by seed index (`runtime.py` lines 226-228). Reusing Stage 1's seed would make the replicate a copy of the Stage 1 branch.
    - Stage 2 seeds are keyed by (seed index, data arm CUTOFF/RANDOM, world). They are shared by the BCA, distance and replicate branches of that fork.
    - The worker passes the frozen `branch()` a per-fork protocol view, whose `seeds[seed_index]` holds that fork's Stage 2 seeds.
    - Every Stage 1 and Stage 2 branch records the sha256 of its phase-0 minibatch index arrays, in order. The worker asserts that each replicate's digest differs from its fork's Stage 1 native branch.
  - All three Stage 2 branches from a fork share the Stage 2 seed and minibatch stream, so their comparisons use common random numbers.
  - Gated comparisons use the replicate alone. The Stage 1 native branch was selected for showing damage, so it is reported only, re-evaluated on the Stage 2 bank.
- **Fresh evaluation:** a new bank (E2 128, E1 64) and 64 new E2 anchors (48 gated).

**Detection precheck, at the 10k fork, with the shadow auxiliary, before any branch.**
- **Anchors:** 64 fresh E2 anchors (48 gated) from a dedicated named stream, `d12_e2_anchors`, disjoint from every other bank.
- **Computation:** AUROC is computed within each anchor over all OOD vs in-support pairs at that state, then averaged over anchors.
- **Convention:** a1..a3 = 0, the same as the gated KC. The actor's own a1..a3 are reported.
- **D1:** AUROC_CUTOFF of {s10, s15, s20} against the near-edge in-support slots >= 0.70, in stall CUTOFF, every seed.
- **D2:** in the stall world, AUROC_CUTOFF(s05 against near-edge) - AUROC_RANDOM(s05 against near-edge) >= +0.15, every seed. A U that simply grows with |a0| passes D1 trivially. D2 requires U to respond to *missing data*, at a depth where RANDOM has labels.
- **Ties** in U count as 0.5. With one OOD slot against five in-support slots, a per-anchor AUROC takes values in steps of 0.1 before averaging over anchors.
- **Reported:**
  - FPR@95;
  - AUROC for W+;
  - AUROC of Q1 - 3U;
  - a magnitude control: U at the supported a1 = ±0.7 against a1 = 0.
- **D1 or D2 fails: DETECTION-FAIL.** Stop and record that BCA's U does not detect this region; no BCA branches are trained. This is a finding about detection, not a failure of the condition.

**Decision rules (stall world unless stated, E2, 20k):**
- **Precondition for requesting A5,** checked from Stage 1 data before Stage 2 starts: the no-stall CUTOFF runs pass the Stage 1 adequacy floor (J_E1 gain >= +50 at 20k) in every seed. R3 needs them. If they fail, the record reads "Stage 2 not requested (precondition)", because Stage 2 could not issue PROMISING.
- **V1-S2 (INCOMPLETE):** as in Stage 1, plus the minibatch-digest assertion.
- **V0-S2 (INCONCLUSIVE):** every Stage 2 branch passes J_E1(20k) - J_E1(untrained) >= +50 on the Stage 2 E1 bank, in every seed. That covers every replicate and BCA branch in stall CUTOFF, stall RANDOM and no-stall CUTOFF, and the stall CUTOFF distance branch. A branch that has not learned would distort R1, R2 or R3.
- **Reported, not gated: damage reproduced.** Whether the native replicate meets K2 and K3a in stall CUTOFF, per seed. It is not a gate because gating on the comparator would select it for damage and bias every comparison toward BCA.
- **Screens; PROMISING requires all three:**
  - **R1, region-specific gain.** G = [J(BCA, C) - J(rep, C)] - [J(BCA, R) - J(rep, R)]. G > 0 in every seed and the three-seed mean >= +3.0.
    - Prior BCA-minus-native gains at strong current averaged +3.6 (SD 3.55), including full-data cells where nothing was withheld. A difference-in-differences against RANDOM removes that generic effect.
    - **Operating characteristics of R1 alone, from saved same-fork proxies (SD 3.55):** a null pass rate of about 5-10% and power of about 0.3-0.4 at a true G of +3.
    - A paired t-interval is reported, but with 3 seeds its half-width is about 12, so it carries little information.
  - **R2.** J(BCA, C) - J(distance, C) >= 0 in every seed.
  - **R3, cost of caution where going past the knee helps.**
    - H = [J(BCA) - J(rep)] in no-stall CUTOFF minus [J(BCA) - J(rep)] in **stall RANDOM**.
    - **The generic-gain baseline is stall RANDOM, the same baseline R1 uses.** There, V0(b) keeps the native actor's R_deep- rate at or below 0.10, so caution at depth can barely matter. No-stall RANDOM would be the wrong baseline: at high load it holds almost no labels in R_deep- (about 18.5 rows), its actor is free to go deep, and subtracting it would cancel the very cost R3 is meant to detect.
    - Cross-world assumption, disclosed: BCA's generic gain is taken to be similar in both worlds.
    - **What R3 measures once R1 has passed.** H = G + (D_nsC - D_sC), where D = J(BCA) - J(rep) in a cell. The shared stall RANDOM term makes G and H positively correlated (correlation about 0.5). R3 can change the label only when R1 has passed, so its useful operating characteristics are conditional on R1.
    - **Require the three-seed mean of H >= -1.5**, a number sealed now (`screens/stage2_oc.py`). Given that R1 passed at a true G of +3:
      - P(R3 passes) is 0.86 at zero cost, 0.48 at a true -3 cost and 0.12 at -6;
      - the earlier bar of -3.0 gave 0.95 / 0.70 / 0.27, which made it nearly blind once R1 passed.
      - Unconditionally, P(R3) is 0.70 at zero cost and 0.30 at -3.
    - Also reported in no-stall CUTOFF: BCA's R_deep- rate minus the replicate's, and Δclip(BCA) minus Δclip(rep). They show directly whether BCA held the actor out of a region that helps there.
- **Joint operating characteristics of PROMISING (R1 and R2 and R3)**, from `screens/stage2_oc.py`: a branch-level noise model with every shared term included, same-fork SD 3.55, R3 bar -1.5.
  - **Null** (G = 0, no cost): 2.0% if the distance arm equals BCA, 5.7% if BCA beats distance by 3.
  - **Power at G = +3 with no cost:** 5.3% if distance equals BCA, 17.7% if BCA beats distance by 3, 28.5% if R2 is ignored.
  - **At G = +3 with a true -3 cost:** 2.7% / 9.6% / 15.9%.
  - PROMISING is therefore a screen meaning "worth a larger study". NOT-PROMISING is weak evidence against BCA.
- **Recovery** (reported only, and only when its guard holds):
  - Guard: J(rep, R) - J(rep, C) > 0 in every seed, with mean >= +3.0.
  - Recovery = mean[J(BCA, C) - J(rep, C)] / mean[J(rep, R) - J(rep, C)]. At least 0.50 is reported as strong.
  - Otherwise it is "not computed (guard)".
  - A lower confidence bound was considered and declined. The 9 differences between each replicate and its fork's Stage 1 native branch (stall CUTOFF, stall RANDOM, no-stall CUTOFF) give an empirical branch-noise scale, but Stage 1's selection for damage affects it.
  - Those 9 differences are reported as a noise check *without* common random numbers. The two branches use different seeds and minibatch streams, so their spread is expected to exceed the same-fork figure of 3.55, which came from pairs that shared random numbers.
  - The check is not a like-for-like test of 3.55.
- **Repair screens**, reported in stall CUTOFF, each with a guard:
  - BCA's R_deep- rate minus the replicate's <= -0.15 in every seed;
  - Δclip(BCA) <= 0.5 x Δclip(rep), computed only if Δclip(rep) >= +2.0;
  - median KC_s10 of Q1 - 3U <= 0.5 x the replicate's median KC_s10, computed only if the replicate's KC_s10 >= +0.5.
  - A failed guard is recorded as "not computed (guard)".

**Stage 2 labels:**
- **INCOMPLETE > INCONCLUSIVE > DETECTION-FAIL > NOT-PROMISING > PROMISING.**
- **PROMISING:** R1, R2 and R3 all hold.
  - If the native replicate misses K2 or K3a in some seed i, the label is issued as "PROMISING (damage not reproduced in seed i)", and that seed's G is not read as repair of harm.
- **NOT-PROMISING-consistent:** a per-seed clause (R1's sign, R2) fails in all three seeds; or a pooled clause (R1's mean, R3's mean) fails *and* its per-seed quantity is on the failing side in all three seeds.
- **NOT-PROMISING-mixed:** any other failure, for example one outlier seed driving a pooled failure.
- Every result is reported per seed. No kappa, mask or threshold changes after viewing.

---

## Part 6. Controls

1. **RANDOM.** Count-matched within episode x ten-step bin x load band. It reimplements the prior study's sampling procedure with the knee predicate and three load bands (Part 3.7). It holds high-load onset labels and low-load full-fold labels (L1.4). Its values are simulated and sealed.
2. **Exact no-stall twin.** Same draws; a1..a3 bounded to 0.7; load independent of actions.
3. **World swap of the same policy** (K3c). This is the primary cross-world attribution. It replaces the earlier arm-vs-arm no-stall placebo, which the BC-anchor gap biased.
4. **Clip intervention, Δclip.** The same policy on the same starts. Evaluation only.
5. **W+ sign control** (L2 item 6, the p10 probes and the W+ rate).
6. **Offset-free contrasts.** KC against -KNEE. A report-only difference-in-differences against -0.5: [Q1(slot) - Q1(-0.5)] - [Q^pi(slot) - Q^pi(-0.5)].
7. **Validity controls:** V0 (a)-(e), L1.4, L1.5, L2 estimator validity.
8. **Common calibration and common anchors** across arms within a world.
9. **Stage 2:** the native replicate with shared random numbers, the distance arm, the detection contrast D2, and the cost of caution in the no-stall world.
10. **Shared non-region extrapolation.** a1..a3 in (0.7, KNEE] are unlogged in every arm. Per-arm and per-world rates there, and Q1 contrasts at a1 = ±KNEE against ±0.7, are reported as a check on extrapolation that is not specific to the region.
11. **Not used as evidence:** the 2026-10-12 gapped-log study (non-Unifloral learner).

---

## Part 7. Budget

**Timing basis**, from saved Zeus receipts:
- warmup supervisor elapsed 470-512 s;
- native 10k branches 492-520 s, median about 500;
- evaluation 1,975,304 transitions in 392.6 s, about 5,031 per second;
- per-run times under the new data are not yet verified;
- **paid compute:** none.
- **A "job"** is one training process, a warmup or a branch. Caps fail closed, and whichever binds first (per job or per stage) wins.

| Step | Work | Estimate | Caps |
|---|---|---|---|
| 0.1 | Read-only screens | Done | — |
| 0.2 (A1) | Code, schema, tests | Agent and user time | — |
| 0.3 (A2) | Forward passes on 9 frozen critics (probes plus about 648 visited rows, 11 points each); known-answer replay and saved-probe reproduction; then the onset contrast's prefix replay (about 170k transitions) | < 30 min | Gated items: 0.5M transitions, 30 min. Onset contrast: 200k transitions, 10 min ("not computed" if exceeded) |
| 0.4 (A3) | 2 x 1,280 episodes plus replay; about 96 anchors x 3 continuations x 3 worlds x about 22 forced actions; evaluator cross-check | about 2.1M transitions, under 10 min | 2.5M transitions, 1 h |
| Stage 1 (A4) | 12 runs (warmup plus branch) + 3 BC-only (estimated at under 15 min in total) + R0 + evaluation. Evaluation: free rollouts about 2.1M transitions, Q probes about 2.2-4.3M, visited-state KC about 1.3-2.6M, about 5.6-9M in all | 3.6-3.7 h training + 20-30 min evaluation | 1,800 s per job; 5 h training; 1 h evaluation (18M transitions at 5,031/s) |
| Stage 2 (A5) | Detection precheck + 21 branches + evaluation. The evaluation also covers the re-evaluated Stage 1 natives and the untrained baselines on the new bank; about 7M transitions, a linear rescale of revision 4's estimate | about 2.9 h + about 25 min | 1,800 s per job; 4 h; 1 h (covers about 18M transitions) |

- **Totals:** about 4-4.5 h through Stage 1; about 7.5-8 h through Stage 2.
- **Disk:** about 4-6 GB per training stage.

---

## Part 8. Risks, weaknesses and what cannot be promised

1. **Reach is the main risk.**
   - The forecast margin is thin, every forecast extrapolates, and the old critics never saw deep current.
   - Step 0.3 measures edge slopes on supported points, and R0 measures the real learner at 10k.
   - **The alternative mix15/D** (window [-0.85, -0.75]) has larger forecast margins (0.38-0.50). But its window edge float32(-0.85) = -0.85000002 is exactly s05, which leaves RANDOM no labelled depth for K1a. C is kept for that reason.
   - A false stop is more likely than a false pass. Step 0.3 and R0 can end the line of work early at little compute cost.
2. **The labelled control.**
   - RANDOM's high-load labels stop at the knee's onset, and its full-fold labels sit at low load.
   - V0(b) can end Stage 1 INCONCLUSIVE, whatever the hypothesis, for either of two reasons:
     - (i) its critic does not carry the fold over to high load;
     - (ii) its critic is correctly labelled but rounds the kink with a softplus width wider than about 0.04-0.05. The 0.4 coefficient covers widths up to about that.
   - **Competent learners shrink the value gap.** rho scales with the state magnitude the continuation produces. If the learners are competent enough that median rho_hat_C(s05) < 0.5 + 0.4 x median rho_hat_R(s05), or median rho_hat_C(s10) < 0.65, V0(d) makes Stage 1 INCONCLUSIVE.
     - This is by design: a correct critic could not pass K1 there, so a FAIL would be misleading.
     - The cost is that a good learner can end the line of work without an answer.
   - G0.5's onset contrast gives an early warning, on old critics only, flagged at 0.3.
3. **The BC anchors differ** (-0.7263 vs -0.7747). This is conservative for CUTOFF's entry and does not affect the within-policy contrasts K3a and K3c. It is disclosed.
4. **The state distributions differ between the worlds' training data.** Reported (lambda, input shift); K3c avoids it by evaluating one policy in both worlds.
5. **Target smoothing queries W.** About a third of target actions for CUTOFF (estimated), which can contaminate the -KNEE reference used in KC. KC differencing assumes that contamination cancels; the absolute Q1 error at -KNEE is reported.
6. **Critic error is measured at anchors, while entry and harm are measured at visited states.** KC at visited states is reported to connect the two, but it is not gated.
7. **Power.**
   - Stage 1 has no power estimate (Part 5).
   - Stage 2's PROMISING is a weak screen. Jointly (R1, R2 and R3), it has a null pass rate of about 2-6%, and power of about 5-18% at a true +3 benefit, depending on how good the distance arm is (28.5% if R2 is ignored; `screens/stage2_oc.py`). NOT-PROMISING is weak evidence against BCA.
   - R3 rests on an untested cross-world assumption: BCA's generic gain is about the same in both worlds.
   - The noise level behind these figures (SD about 3.55) comes from earlier same-fork proxies. Stage 2 reports its own check of it.
   - Checkpoint-level contrasts were unstable in the prior study. The hole-vs-random gain went from +2.68 / +1.61 / +10.17 at 15k to -1.02 / +6.22 / -1.94 at 20k. Gating on 20k only is correct, but expect sign instability across seeds.
8. **Benchmark-version limits:** the reset gap (0.7, 0.9); stall_slope 3.0 only; the a1..a3 bound with its shared unsupported band; claims for per-tuple learners only.
9. **Not yet verified:**
   - per-run times;
   - the new schema;
   - the size of the value gap at 3.0 (estimates until step 0.4);
   - RANDOM's realized values, which are sealed from simulation and checked at step 0.4;
   - the step 0.3 reach forecast under the new gating model, which needs checkpoint forward passes. If old critics round their kink even at -0.7, it is biased low.
10. **What cannot be promised.** L1, the L2 identity, the predicate and the data plumbing can be made certain, as Part 4 sets out. Whether a learned native critic over-values the region, whether the actor goes there, and whether BCA's U detects it are empirical properties that L3 and Stage 2 measure. A FAIL there is a valid answer.

---

## Part 9. Review history

**Review 0, the design panel's adversarial critic, against the panel's plan KEC-W v1.** Resolved in revision 1:
- lambda dilution under the wide reset law, which led to the candidate grid;
- the arm-vs-arm placebo bias, which led to within-policy contrasts;
- Stage 2 selection bias, which led to the native replicate;
- the 361/3,519 count;
- the extrapolated edge slopes;
- the return-noise reference;
- the dataset contract;
- label density and G0.5 coherence;
- anchor rebuilding;
- L1.4 being unfalsifiable;
- tautological checks;
- NaN handling in the twin;
- the a1..a3 bound;
- stale statements.

**Review 1, an independent reviewer of revision 1** (summarized; the report was delivered in session). Resolved in revision 1 itself:
- undefined units;
- the step 0.3 model being unspecified;
- the stale 0.87 constant;
- the over-claimed non-identifiability;
- the vacuous twin check;
- G0.5's governing bank;
- predictor validity labelled as a shelving FAIL;
- d = 0.10 sitting outside RANDOM's labels;
- conjunctive over-reading;
- the recovery denominator;
- the comparator choice;
- the in-worker proof;
- the KC reference;
- small numeric errors.

**Review 2, a verification workflow on revision 1**: four lenses (physics, kill rules and statistics, implementation, integration), a skeptic re-check of every high or medium finding, and a completeness critic. The record is in `reviews/VERIFY_WORKFLOW_r1.json`, with a digest in `reviews/VERIFY_CRITIC_DIGEST_r1.txt`. Its 23 must-fix items are resolved here:

| # | Severity | Item | Where resolved |
|---|---|---|---|
| 1 | High | The step 0.3 forecast fit through labelled points, so a critic that holds the fold forced the KILL | Step 0.3: the gating fit used the four supported points only. *Superseded in revision 3 (Review 3 item 2): edge secant plus five-slot curvature* |
| 2 | High | Stage 2 "promising" rested on J screens that a generic BCA effect could pass | R1 is a difference-in-differences against RANDOM; R3 is a difference-in-differences with a sealed bar (revision 3); recovery is guarded. The recovery lower confidence bound was declined: one replicate per fork gives no noise estimate. *Rationale revised in revision 4: the replicate-minus-Stage-1 differences give a selection-affected scale, which is reported* |
| 3 | Medium | No check that the replicate reproduces the damage; ratio screens unguarded; no Stage 2 labels | V0-S2; ratio guards; Stage 2 labels |
| 4 | Medium | Labels neither exhaustive nor mutually exclusive; G0.5 STOP contradicted Part 8; rerun scope undefined | 5.0 precedence table; a label table at every step; G0.5 STOP is INCONCLUSIVE; rerun rule |
| 5 | Medium | No absolute over-valuation requirement; V0(b) one-sided; K1b repeated L2 | K1c added; V0(b) two-sided; K1b reported as consistency |
| 6 | Medium | K3c used a different policy than L3c describes | K3c is the world swap of the same stall-trained policy |
| 7 | Medium | RANDOM's expected values were wrong (48% retention, density 2.08, anchor -0.775); L1.4 too wide; low-load fold labels undisclosed | `random_control_mc.py`; sealed values; L1.4 bands at ±6 SD with known-fault tests; RANDOM's coverage restated |
| 8 | Medium | The in-worker proof targeted a function the pipeline never calls | 4.4 targets every `make_adapter` binding and every stream, with hashes after the float32 conversion |
| 9 | Medium | The calibration split contradicted itself | Part 3.9: a common, W-free set from code 1 |
| 10 | Low | L2.2 wording | Restated for per-tuple learners |
| 11 | Low | K2b implied by other rules; "every rule involving RANDOM" too broad | K2b is a report; wording is per rule |
| 12 | Low | Sensitivity readings called "pre-listed" | Called post-hoc; selection from the as-written rule alone; counts at d = 0.10 are 2 / 5 / 3 (REACH_SENSITIVITY.md) |
| 13 | Low | Typicality and KC not measured at visited states; E2 framing | Visited-state typicality and KC reported; E2 framed as a sub-population |
| 14 | Low | Hull and near-OOD framing; truncated density not logged | Framed as censoring shift; truncated density documented; the claim about the Hole dropped |
| 15 | Low | Anchor-time pooling; no power statement; K3a near-implied; the -0.5 contrast undefined | Gated anchors at t in {0, 25, 50}; K3a noted; the contrast defined. A Stage 1 power statement was declined: no seed variance exists for this design. Stage 2's operating characteristics are stated (revision 3) |
| 16 | Low | Slot naming was inconsistent | Slot table (Part 0) |
| 17 | Low | min_k used for the sign control; S9 rationale; V0(d) incomplete | max_k for the sign control; minimum justified as conservative; V0(d) complete |
| 18 | Low | Reset-mixture draw order wrong; contract blockers incomplete | Parts 3.2 and 4.3 |
| 19 | Low | Seed audit compared raw integers; vacuous leakage check relied on | 4.7 |
| 20 | Low | Budget rates; "job" undefined; no BC adapter | Part 7; 4.4 |
| 21 | Low | Detection AUROC trivially 1 for a U that grows with \|a0\|; shared random numbers unspecified | D2 contrast; one branch seed and minibatch stream per fork. The \|a1\| magnitude control is reported, not gated, because D2 already gates against a magnitude-only U |
| 22 | Low | Factual and citation cleanup; terms undefined | Part 0 terms; citations corrected |
| 23 | Low | a1..a3 in (0.7, KNEE] unmeasured | Part 6.10 |

**Review 3, a verification workflow on revision 2**: three lenses (resolution, new-rule logic, numbers and citations), skeptic re-checks and a critic. The record is in `reviews/VERIFY_WORKFLOW_r2.json`, with a digest in `reviews/VERIFY_CRITIC_DIGEST_r2.txt`. It confirmed the arithmetic, slot table, L1.4 bands, anchors, budget rates, hashes and citations. Its 17 items are resolved in revision 3:

| # | Severity | Item | Where resolved |
|---|---|---|---|
| 1 | High | The Stage 2 replicate could be a copy of the Stage 1 branch, because the seed was not separated | Separate Stage 2 branch seeds; a minibatch-digest assertion; Stage 1 branch re-evaluated on the Stage 2 bank |
| 2 | High | The four-point reach fit still carried the old critics' kink at -KNEE and was ill-conditioned | Step 0.3: an edge slope from the line on {-0.5, -0.6, -0.69999999} plus curvature from the five-slot fit |
| 3 | Medium | R1's null pass rate and power were undisclosed | Stated next to R1 and in Part 8.7 |
| 4 | Medium | R3 had no difference-in-differences, and its bar loosened with noise | R3 is a difference-in-differences against no-stall RANDOM, with a sealed mean bar of -3.0 and stated operating characteristics. *Revised in revisions 4-5: baseline stall RANDOM, bar -1.5* |
| 5 | Medium | V0(b)'s 0.25 bar sat at what a smooth, correctly labelled critic produces; G0.5 could not warn of this | V0(b) scaled to 0.3 x rho_hat(s05); G0.5 adds the onset error F05; the rate-clause rationale restated. *Revised in revision 4 (Review 4 items 2-3): coefficient 0.4 plus a floor; F05 replaced* |
| 6 | Medium | Expected-value bounds were routed to INCOMPLETE with no exit; share bars could not be derived from the script | 4.7 split into plumbing (INCOMPLETE) and expected-value (INCONCLUSIVE) checks; the second identical INCOMPLETE becomes INCONCLUSIVE; shares added to the script; hashes recorded |
| 7 | Medium | The calibration replacement rule was new and could be undefined | The prior rule adopted verbatim (Part 3.9), with expected replacements of 47.5 ± 5.0 |
| 8 | Low | E1 wording; V0(a) checked RANDOM only; R0 had no validity check first | 3.10 wording; V0(a) covers CUTOFF and RANDOM; an adequacy check precedes R0; FAIL-early's fixed reading defined in 5.0 |
| 9 | Low | Re-attempts after INCONCLUSIVE were not ruled out; K outcomes could be lost | 5.0: no re-attempt; every K outcome recorded; V0 scoped to the gated runs |
| 10 | Low | V0-S2 gated on the comparator | Damage reproduction is reported, not gated; V0-S2 now covers the no-stall runs' adequacy |
| 11 | Low | Readings had no order | Readings are hierarchical, and no K3 reading is given without entry |
| 12 | Low | Reserved label words were reused; NOT-PROMISING undefined; D2 named no world; AUROC ties | "not computed (guard)"; NOT-PROMISING defined; D2 in the stall world; ties count 0.5; Stage 2 labels added to 5.0; G0.5's one-sided form justified |
| 13 | Low | "Copies control_masks" was misleading | Reworded as a reimplementation with three bands; binary-strata fault fixture; control seed named |
| 14 | Low | The 361/3,519 population | Per-arm breakdown in 4.8 |
| 15 | Low | Probes missing for V0(d) and Part 6.10; KC paired by state only | No-stall Q and a1 probes added; gated KC uses a1..a3 = 0; K3c's overlap with K3a noted |
| 16 | Low | Undefined terms; L2 numbering; overloaded symbols; raw -0.9 | Part 0 additions; item 5 named; q0/q1/q2; slot values used |
| 17 | Low | Numeric and wording nits; R0 stream unnamed; budget not itemized | Corrected; `r0_e2_starts`; budget itemized |

**Review 4, a focused verification workflow on revision 3**: two lenses, skeptic re-checks and a critic. The record is in `reviews/VERIFY_WORKFLOW_r3.json`, with a digest in `reviews/VERIFY_CRITIC_DIGEST_r3.txt`.
- **Confirmed:** both of revision 2's high items are resolved; the Monte Carlo output reproduces byte for byte; all Part 11 hashes match; the R1 and R3 operating characteristics reproduce.
- **No high item remained.** Its 2 medium and 13 low items are resolved in revision 4:

| # | Severity | Item | Where resolved |
|---|---|---|---|
| 1 | Medium | R3's no-stall RANDOM baseline cancelled the deep-region cost of caution | R3's baseline is stall RANDOM, as in R1; no-stall CUTOFF entry and Δclip differences reported |
| 2 | Medium | V0(b)'s 0.3 coefficient covered kink rounding only to a width of about 0.03 | Coefficient 0.4 (covers about 0.05) with a floor rho_hat(s05) >= 0.85; L2 item 1's median raised to 0.85 for coherence; Part 8.2 risk; reading 2 guarded. *Revised in revision 6 (Review 6 item 1): floors replaced by reachability gates; item 1 back to 0.75* |
| 3 | Low | F05 could not detect kink rounding | Replaced by the onset contrast KC15_s05 / rho15(s05), referenced to -KNEE, flagged at 0.4 |
| 4 | Low | PROMISING's operating characteristics were R1's alone | Joint figures stated (null about 2%, power about 5-28%) |
| 5 | Low | V0-S2 timing and scope; no adequacy floor for Stage 2 branches | No-stall adequacy checked before A5; V0-S2 now covers every Stage 2 replicate and BCA branch used by R1 and R3 |
| 6 | Low | "Valid anchor" had two meanings | Valid anchor over gated probes; predictor-valid anchor for L2; excluded-and-counted rule restored |
| 7 | Low | The rerun rules conflicted with code fixes; "count mismatch" was ambiguous | A code-only fix is allowed; the second-failure rule applies to unchanged code; "identically" defined; 4.7(i) named in the INCOMPLETE rows |
| 8 | Low | No K3b reading; reading preconditions; pooled NOT-PROMISING | Reading 5 added; preconditions stated; pooled "consistent" needs all seeds on the failing side |
| 9 | Low | The step 0.3 a1..a3 convention was mixed | One convention (actor's own a1..a3) for all 11 points; q2 refit at it |
| 10 | Low | The minibatch digest had nothing to compare with; the seed lookup; the recovery-LCB rationale | Stage 1 branches record digests; Stage 2 seeds keyed by (seed index, arm, world); LCB rationale revised and the 12 differences reported. *Refined in revision 5: data-arm key; 9 differences once Stage 2 trains 21 branches* |
| 11 | Low | Part 9 table rows misplaced | Rows 22-23 restored under Review 2; Review 2 row 1 annotated |
| 12 | Low | Stale Monte Carlo figures; bounds rounded inward | 28,347 ± 284; CUTOFF anchor exact -0.7263 with the Monte Carlo cross-check; every bound rounded outward |
| 13 | Low | Part 6.1 "exact rule" | Reworded |
| 14 | Low | R0's E1 starts unnamed; E1's role misstated | `r0_e1_starts`; Part 3.10 restated |
| 15 | Low | AUROC step size; slot labels; s_edge description | Step 0.1; -0.60000002 used; s_edge described as the edge secant |

**Review 5, a final consistency check of revision 4**: a reviewer, a skeptic and a critic. The record is in `reviews/VERIFY_WORKFLOW_r4.json`, with a digest in `reviews/VERIFY_CRITIC_DIGEST_r4.txt`.
- **Confirmed:** all 15 round-3 items are resolved; every outward-rounded bound and every Part 11 hash matches; the V0(b) and edge-secant arithmetic is correct.
- **No high item remained.** Its 1 medium and 9 low items are resolved in revision 5:

| # | Severity | Item | Where resolved |
|---|---|---|---|
| 1 | Medium | R3 shares R1's stall RANDOM term (corr about 0.5), so given R1 it caught a -3 cost only about 30% of the time; the joint figures assumed independence | The identity H = G + (D_nsC - D_sC) stated; R3's bar tightened to -1.5 (sealed); conditional and joint operating characteristics recomputed by `screens/stage2_oc.py` |
| 2 | Low | The A5 precondition still required no-stall RANDOM; 15 of 36 branches fed no rule | Precondition on no-stall CUTOFF only, recorded as "not requested (precondition)"; Stage 2 trains 21 branches; budget and assertion counts updated |
| 3 | Low | No adequacy floor for the distance branch, which made R2 trivially passable | V0-S2 covers the stall CUTOFF distance branch |
| 4 | Low | The rerun rule allowed re-rolls when the failing value changed; code-only fixes unlimited | A second failure of the same unit and check under unchanged code ends the step, whatever the value; one code-only fix per unit; sealed units protected |
| 5 | Low | Anchor sets for V0(b) and K1a unspecified; predictor validity never checked under RANDOM's continuation | Common gated anchors; V0(b) predictor-valid under RANDOM's continuation; V0(c) floors the common set |
| 6 | Low | K1a's worst case mixed two continuations; coverage over-stated | Worst case restated per continuation; coverage about 0.04-0.05; a design-failure risk disclosed in Part 8.2 |
| 7 | Low | Step 0.3 gave two conventions for the same points | The gated F and onset contrast use a1..a3 = 0; only the forecast uses the actor's own |
| 8 | Low | The onset flag sat at the pass bar and could exhaust the gated cap | Flag at 0.3, ratio of medians, run last under its own 200k sub-cap; exhaustion recorded as "not computed" |
| 9 | Low | The Stage 2 seed key was ambiguous | Keyed by (seed index, data arm, world), shared by the three branch arms, through a per-fork protocol view. *Citation alignment, missed in revision 5, completed in revision 6 (lines 226-228)* |
| 10 | Low | Stale cross-references | `r0_e1_starts` named; the Review 2 row 2 rationale annotated |

**Review 6, the final independent verification of revision 5**: three lenses (resolution, a fresh end-to-end read, numbers and citations), skeptic re-checks and a critic. The record is in `reviews/VERIFY_WORKFLOW_r5.json`, with a digest in `reviews/VERIFY_CRITIC_DIGEST_r5.txt`.
- **Confirmed:** the OOD definition (L1, L2 and the host objective) is correct against the simulator and host code; the label precedence is exhaustive; no gate can PASS without the OOD condition; every hash matches; the Stage 2 simulation reproduces byte for byte.
- **Resolved in revision 6:** 3 medium items, 1 medium-or-low item and 12 low items:

| # | Severity | Item | Where resolved |
|---|---|---|---|
| 1 | Medium | V0(b)'s absolute floor on rho under RANDOM's continuation, and V0(d)'s absolute L2 bars, scale with how competent the continuation is. A competent learner could end Stage 1 INCONCLUSIVE even when K1 is testable | Absolute bars kept for step 0.4 only. Stage 1 gates reachability directly in V0(d): rho_hat_C(s05) >= 0.5 + 0.4 rho_hat_R(s05), and rho_hat_C(s10) >= 0.65. V0(b) keeps the ratio clause with a resolvability floor of 0.20. Step 0.4 item 1's median returns to 0.75. Part 2 explains the scaling; Part 8.2 discloses the risk. *This design choice (option a, against keeping absolute floors set by a dynamics screen) was accepted by the user in session on 2026-10-29 at 19:57 EDT. That is a design decision, not a stage approval* |
| 2 | Medium | Nothing certified stall_slope 3.0 | 4.5b: a physics assertion in every receipt; thrust reconstructed on W rows equals T3; a literal-formula identity test; a slope-1.5 fixture that must fail |
| 3 | Medium | The forced-probe critic path had no known-answer check | 4.8 reproduces the saved per-probe q1 and twin_min to 1e-6 in all nine cells; one probe-building function, with slot assertions (4.4) |
| 4 | Medium-low | `stage2_oc.py` drew R2's "d = 3" column independently | Fixed to one draw per row; joint figures updated (null 2.0-5.7%, power 5.3-17.7%, 28.5% ignoring R2); the R3 bar is unaffected |
| 5 | Low | V0 items had been re-parented under "Anchor sets"; item 5 under RANDOM's continuation not stated as gate or filter | Anchor sets moved above V0; item 5 under RANDOM's continuation is a gate at >= 90% (V0(b)) |
| 6 | Low | K1a's worst case was stated as a necessary condition | Exact condition stated, sufficiency noted, and gated in V0(d) |
| 7 | Low | The onset contrast could trigger INCOMPLETE through the step's time cap; its critics were not named | Runs after the gated items under its own 200k / 10 min sub-caps ("not computed" if exceeded); full-data critics per seed |
| 8 | Low | R_lin's convention and source were unstated | a1..a3 = 0, saved per-probe Q^pi, no new replay |
| 9 | Low | The runtime.py citations disagreed | Lines 226-228 in both places |
| 10 | Low | The 9 replicate differences were called a test of 3.55 | Described as a noise check without common random numbers, expected to exceed 3.55 |
| 11 | Low | Rerun loophole across different checks | Any second failure of the same unit under unchanged code, on any check, ends the step |
| 12 | Low | D1/D2 anchors unnamed; convention differed from KC | `d12_e2_anchors`; a1..a3 = 0 |
| 13 | Low | PROMISING could be issued without reproduced damage | Qualifier "damage not reproduced in seed i" |
| 14 | Low | Step 0.4's Q^pi probes were not marked user-run | Marked user-run |
| 15 | Low | Wording slips (L2.2 qualifier, W- thrust range, Stage 2 evaluation estimate, adapter count) | Corrected |
| 16 | Low | Truncated hashes; drafts and REACH_SENSITIVITY.md unhashed; folder untracked | Full sha256 in Part 11; committing the folder before A1 needs the user's approval |

---

## Part 10. Rejected alternatives

- **Withholding on the existing uniform data.** Rejected because of reach. At ordinary high-load visited states (15k and 20k pooled), 0 of 1,206 go past the knee, and BC anchors near 0 need slopes the critics lack. Full-data actors already cross the knee as often as the other arms.
- **The LoadTrack-Knee recorder.** a0 is correlated with load (about 0.8), the positive side is never logged, and it fails its own slope test.
- **The KEC-v1 physics design as written.** Its twin is contaminated by a1..a3 ~ U[-1, 1], it gates a context with marginal reach, and it counts entry at any depth.
- **The panel's KEC-W v1.** It fails its own reach screen once lambda is corrected.
- **Raising `td3_alpha` or changing the host.** That is tuning the host until it fails. CORL-derived learners are excluded.
- **stall_slope 1.5 as the primary world.** It gives about half the gap: ordinary-context median drop at d = 0.10 of 0.60, against 1.18 at slope 3.0, and a strong-current proxy of 0.84-1.00 against 1.67-2.06.
- **Strong current as a gate.** Confounded by state shift.
- **mix15/D.** Covered in Part 8.1.
- **Previously failed or superseded, not revived:**
  - the central-quartile Hole;
  - the stall-knee band (support and exposure failed; its value test passed at its minimum);
  - the zero slab;
  - `conditional_ood_gate_v2`;
  - `support_shift_pilot_v1`;
  - the other Stage 0 options-memo candidates.

---

## Part 11. Provenance

**Full sha256 of every supporting file**, relative to the design folder.
- The design folder is untracked in git. Until it is committed (which needs the user's approval), "written before the script" rests on these hashes and on file times.
- The reach-screen rule was written before the script.
- `REACH_SENSITIVITY.md` and `reach_variants.py` are post-hoc sensitivity.

| File | sha256 |
|---|---|
| `panel/PANEL_OUTPUT.json` | bf4efe7ef4340c142cf676c5f7f621123ef3d30607f3673bb603eb6f681b1257 |
| `REACH_SCREEN_RULE.md` | 8d4cae034c401859b1cbfba3b31d8a029d334033e867904e5a7390989eacdb46 |
| `screens/reach_grid.py` | a260741bec3e1c7923a6c90c9bc10704003095434c5b4d34f9d35caaa93684a3 |
| `screens/reach_grid_output.txt` | 9909de46581c6273100cf82ef06abc6a82558e3d320d41a571a0de5ed1e81b99 |
| `screens/REACH_SENSITIVITY.md` | 93c1355a27e68a769883f775e012c01bec2fb7b75f34717f7627a506aea4ecb6 |
| `screens/reach_variants.py` | e1eff4c5de8047387b8df1922f931638dea53f37442a63a859a86a851f3936e7 |
| `screens/reach_variants_output.txt` | 738c9b5f8064f266bd8c1ad801a843a6e05240aa64b749eb4e974a9aa4d37f33 |
| `screens/random_control_mc.py` | 0f9ab35d4d20519731c0c5c5c4428717b794a3c797eaa57ee2b3df72472129b6 |
| `screens/random_control_mc_output.txt` | 2dc0c3837aec351756eef1a96847c08bf8b179398acb56323c94328b6103f716 |
| `screens/stage2_oc.py` | 5379a8f497f6285823fd31ca3c7e490b6827609b2fa4a51c8f51340806fb7797 |
| `screens/stage2_oc_output.txt` | e521d8a9fa9104b1eed54522affd768a1bf25c7985ecf75468854d870756f524 |
| `DRAFT_PROTOCOL_r1.md` | 35bea738bf9ef3b7ba5db315006487acc50b4c4aebc1d4190f50d87f659b16fc |
| `DRAFT_PROTOCOL_r2.md` | 7473ef0790ce9d1549ab94552f6cfa2ac1fd9320b991882cca1e3196c44a7964 |
| `DRAFT_PROTOCOL_r3.md` | 244bcd335e69177b335439cb0526ba77f7852c491582e75925f95a195a59a59e |
| `DRAFT_PROTOCOL_r4.md` | 15f65be3498e744fdf6791bb233c47593b688e46321521d09fb4cec6807bfc50 |
| `DRAFT_PROTOCOL_r5.md` | cc773ebcbb76b66928ba97250e818d792a7b0b2a6a669668aa38c81485f0d408 |
| `reviews/VERIFY_WORKFLOW_r1.json` | a21692bd02b42013b330e8e35152d4f347f93565dd5e32dd7e9752c2272e8b94 |
| `reviews/VERIFY_WORKFLOW_r2.json` | 503a29e54bc17fd6e3489e5d467d25c6f92e49946271fa2b2c8a6dd2082ed35f |
| `reviews/VERIFY_WORKFLOW_r3.json` | 99f6f49414eb1119444fa599858b0cd482fa64f2d8acb67ca786f4268b7fbf7b |
| `reviews/VERIFY_WORKFLOW_r4.json` | 7bc9a92c3e14077eafe43ac4b5a5d0a2ec7c20d93d49603203c9f344e1627083 |
| `reviews/VERIFY_WORKFLOW_r5.json` | 59e9b4697ddf20c092deb0d504cbe472f045469ada6c2db6a4db31f8ce8b1fe5 |
| `reviews/VERIFY_CRITIC_DIGEST_r1.txt` | 62682f645e98ad1fb83a6c75c379fd7de85c857a7a5eb0aa91629f7c81ed3700 |
| `reviews/VERIFY_CRITIC_DIGEST_r2.txt` | f96cae9658d41ea09aa06ecc4e53d973c6bc4794184c294a7650d8138c844433 |
| `reviews/VERIFY_CRITIC_DIGEST_r3.txt` | bd03dae03bd614f6b0061ea3cce71abd262bf40d047e7cff4e2e3afe73334f3a |
| `reviews/VERIFY_CRITIC_DIGEST_r4.txt` | 4872ba09be16b8c17a607663d2f13c519e6b7a402e094d46ab1c59cf55b551ff |
| `reviews/VERIFY_CRITIC_DIGEST_r5.txt` | 2636dfe3bac54f158e8225b9390958946899808d363216c7682ee2192fa34c3f |

- **Prior rules reused:** `conditional_ood_gate_central_quartile_v1/geometry.py` `control_masks` (RANDOM sampling procedure) and lines 320-346 (calibration replacement). Branch seeding: `rerun1/runtime.py` lines 226-228. Gamma binding: line 138. Physics defaults: `canonical/thrusterchain.py` lines 116, 143, 398.
- **Code facts cited:**
  - `titan_20261026/.../core_v4/thrusterchain/canonical/thrusterchain.py`:
    - `dynamics` (207, 218-221);
    - thrust knee in the no-stall world (124-131);
    - `START_LAWS` (87-92);
    - `draw_starts` (308-315).
  - `.../thrusterchain/datasets.py`:
    - `_split_assignment` (135-140);
    - `_validate` (255, 264, 277, 343-346);
    - hard-coded start (171);
    - observation time feature (338).
  - `unifloral_td3bc_validation_v1/upstream/algorithms/td3_bc.py`:
    - target minimum (207);
    - actor uses Q1 (240);
    - lambda (243-245);
    - noise (199-200);
    - default gamma (40).
  - `central_quartile_training_v1_evaluation_rerun1`:
    - `runtime.py` `make_adapter` (197-209), context (131-163), heldout refresh (303);
    - `data.py` normalization (145-187), reader check (70), leakage check (155).
  - `conditional_ood_gate_central_quartile_v1/geometry.py` `control_masks` (RANDOM rule).
