
---

## 4. Arms and controls

### 4.1 Option A arms

**Decision rules**, all on the same held-out scores and groups, one decision per seed-world (world-level), each at its
own margin (section 3.6):

| Arm | On iff |
|---|---|
| **A-BCA (primary)** | `L_gap > m`: BCA's two-posterior gap, uniform weights (`step1lib.certificate`) |
| A-BCA0 | the same at margin 0 (the margin's share) |
| A-BB | Bayesian bootstrap of the two 90% quantiles without WBCP's test atom, 5% lower bound of the gap > m |
| A-SC | plain split conformal: empirical 90% quantile of the group minus that of the rest > m (step 2's `conformal_k1` switch) |
| A-OS | exact order-statistic bound: the distribution-free 97.5% lower bound of the group's 90% quantile minus the 97.5% upper bound of the rest's > m |
| A-BOOT | percentile bootstrap (2,000 resamples) 5% lower bound of the quantile gap > m |
| A-MEAN | Welch 95% lower bound of the mean gap > m_mean; and the plain difference of means > m_mean (the round-1 verifier's rule) |
| A-HL | Hodges-Lehmann 95% lower bound of the shift > m_HL |
| A-RANK | one-sided Mann-Whitney p < 0.05, no margin (step 1's rank test, `step1lib.py:279-287`) |
| A-EXC | Clopper-Pearson 95% lower bound of the group's share above the rest's 90% quantile plus m > 0.10 |
| A-SHUF | A-BCA with outcomes shuffled within each seed-world (group and rest pooled and re-dealt), 200 permutations (`step1lib.py:290-305`) |
| A-RAND | random switching at A-BCA's own on-rate, pooled over the stage's S and N seed-worlds; scored by its exact expectation (section 7.2), one realized coin reported |
| A-NEVER | the host alone |
| A-INS | A-BCA on the in-sample survivors against the bank (A-K3) |
| A-PERFECT | on iff the true value of caution V > 0 (oracle reference); A-PERFECT_hinge for the hinge |

**World-blind set W** (deployment wrappers or existing actors; no outcome read): clip always (A-ALWAYS); the gated hinge
(k = 1) always; a cap at 0.80 always (every thruster clipped to [-0.80, 0.80]; labeled "knows the knee"); the best plain
cap among 0.75, 0.85, 0.90 and 0.95 always, chosen on the pilot by pooled NET and frozen in the confirmatory slot. In
the record the always-on hinge netted +76.5 and the always-on BCA ramp +77.7, against +44.4 for clipping always (CONN
`restructure/step2_results/results.md:20, 29`; ROUND1.md:13), so clipping always is not the best world-blind rule.

**Caution actions.** When on: (i) **clip** (primary): a deployment wrapper, no training; (ii) **gated hinge**: the host
retrained with stage 1's range hinge `100 sum_j([a_j - hi_j]_+^2 + [lo_j - a_j]_+^2)` at k = 1 in the actor loss
(`harness.py:373-376`), deployed in place of the host when on, switched at `m_hinge`; (iii) **clone fallback**
(reported): the clone's command replaces any command with `t(a) > t_lo`. With the same perfect detector the record has
clip +106.1 and +28.2 against the gated clone fallback's +73.8 and -30.4 in strong current and big kicks, and in the
no-stall world -61.6 for the clip against -109.5 for the clone (HV2
`outputs/shift/beyond-the-data/results-2026-10-13/deploy_tables.md:17, 33, 79, 95`), which is why the clone is reported
only.

**Mixed worlds (A-K4).** Declared groups by state, fixed now and not aligned with the stall cut: wave-amplitude cells
[0.1, 0.2), [0.2, 0.3), [0.3, 0.4] (the middle cell straddles 0.25). An episode's cell is its start's amplitude;
starts above 0.4 lie in no cell and run the host. Expected evidence: on seeds 0-4 each amplitude half held 6-18
calibration-split full-throttle rows per seed-world (`_where/feas_amp_0_1_2_3_4.json`), so about 15-45 per half with
both splits and about 8-30 per cell [inferred]. A cell with fewer than 10 full-throttle rows runs the host and is
reported "no evidence".

| Arm | On iff (per episode, by its start) |
|---|---|
| **A-KERN (primary of A-K4)** | A-BCA with weights from a Gaussian kernel in amplitude, Silverman bandwidth, test mass 1, as `mixed_waves/run_route.py:78-92` |
| A-CELL | A-BCA per cell (equal weights within the cell) |
| A-SC per cell | plain split conformal per cell at m |
| A-KQ | kernel-weighted plain quantile gap: the same kernel weights through WBCP's weighted empirical quantile `lambda_hat` (`wbcp.py:89-102`), no posterior, at m |
| A-KMEAN | kernel-weighted difference of means at its own margin `m_kmean` |
| world-level A-BCA | as above |
| caps | 0.75, 0.80 (labeled "knows the knee"), 0.85, 0.90, 0.95 always; the best non-knee cap chosen on the pilot |
| A-ORACLE-REGION | on iff the start's region stalls (reference; round 1's oracle scored 52.6, ROUND1.md:132) |
| A-KERN-CAP (reported) | A-KERN's own decisions with a cap at the pilot's best cap in place of the clip (the action, not the localizer) |
| A-KPERM | A-KERN with the full-throttle scores permuted among the full-throttle rows, each row keeping its amplitude, 200 permutations (round 1's decisive control, `mixed_waves/KILL_RULE.md:42-46`) |

**Information only (A-I).** Margin sweep by evidence size (subsamples of 5, 8, 12, 20, 30 rows and all rows, 400
replicates each, as round 1's small-evidence test); the online monitor in S, N, M3 and M4 (A-BCA re-read after each of
the first 30 strong-current episodes with margin `m_on` and a latch); the score with Q1 alone and with Unifloral's
noisy target.

### 4.2 Option B arms

**Units of decision.**
- **B-GROUP (primary unit)**: one declared deviation group per seed-world: "let the host's own command through
  unclipped" (that is, never apply the fallback). Tries: offline, the conditioned full-throttle evidence rows (section
  3.2); online, first uses (section 5.3).
- **B-RUNG (per deviation; reported)**: per thruster k and sign, rungs from the recorded range edge: rung 0 = (edge,
  0.80], rungs 1-4 = (0.80, 0.85], (0.85, 0.90], (0.90, 0.95], (0.95, 1.00] (float32 edges). A try of a rung: a row
  whose command for that thruster and sign lies in the rung and whose host command at the try's own state lies beyond
  the rung's lower edge (arm B's definition, CONN `arms_abc_f0/code/arm_b.py:8-12`). Walk rung 0, 1, ... in fixed
  order; stop at the first refusal; the map's ceiling for that thruster and sign is the top of the last certified rung
  (`arm_b.py:131-139`). The deployed map clips each thruster to [lo', hi'] with the certified ceilings in place of the
  recorded edges, when `t(a) > t_lo`.
- On the main log, rungs 1-3 hold no offline tries (the log has no commands between `t_lo` and `t_up`), so offline
  B-RUNG can only cap near 0.80 (declared now; reported as information). Online, probe episodes fill every rung the host
  reaches.

**Outcomes per try.**
- **Residual score s (primary)**, the same score as A, at the try's own command (the placement red team's fix, PLACE
  `red.challenges[4]` and `red.revised_recommendation`, test 3: step 1's same-command score carried the stall).
- Arm B's critic-scored gain `g = r + 0.98 Q_h(x', pi(x')) - Q_h(x, a~)` against one rung down (`arm_b.py:13-14`):
  the **negative control**; with shuffled outcomes it still certified 0.90/0.95/1.00 of rungs (`placement/O8_screen_seed0.json`
  via PLACE `red.challenges[4]`), and arm B stopped on its K1 (CONN `arms_abc_f0/RESULTS.md:44-47`).
- k-step segment outcome (k = 10, 25) against the critic at the fallback command: a matched contrast only, because the
  logged continuation follows the recording controller, not the host.
- Shuffled s (tries and rest pooled and re-dealt at the same sizes, 200 permutations, the same tau): the null of B-K1.
- Online only, simulator reference, not deployable: the **twin outcome** of a probe episode, `J(map with deviation) -
  J(map without)` from the same start, harm iff negative.

**Gates**, each with `alpha_B = 0.1` and level 0.95 unless stated, on the same tries and the same `tau`:

| Gate | Certify iff |
|---|---|
| **Exact tail gate (primary)** | BCA's tail posterior computed exactly (Beta form): the harm count is at most the largest count whose Clopper-Pearson 95% upper bound is <= alpha_B; must equal the exact binomial test on 100% of decisions (stop rule S6) |
| Shipped tail gate (reported) | `wbcp.calibrate(s_tries).threshold <= tau` with 1,000 draws and n >= 36 (BCA's own floor); every disagreement with the exact gate is attributed to the floor or to Monte Carlo noise (the floor exists because Monte Carlo BCA sometimes certified at 26-28 rows; LIVE 10533, 10540) |
| Split-conformal tail | the finite-sample-corrected empirical 90% quantile <= tau (+inf, refuse, when undefined) |
| Plain empirical rate | harm share <= alpha_B (no correction; an anti-conservative reference) |
| Hoeffding, empirical Bernstein | 95% upper bound of the harm rate <= alpha_B |
| BCA mean gate (Gaffke-type, Bayesian bootstrap with a ceiling atom) | 95%-credible upper bound of the mean of s <= mean of the rest + m_mean (lifted from MAIN `bca_mean_gate_toy.ipynb` cells 7, 9, 12-13) |
| t-test, empirical Bernstein and HCOPE-style mean bounds | 95% upper bound of the mean of s <= mean of the rest + m_mean; HCOPE-style: Thomas et al. (2015) bound with truncation at a value fixed from the bank's score range |
| Arm B's rule (negative control) | `wbcp.calibrate(-g)` at miscoverage 0.5 has threshold < 0, and arm B's plain tests (`arm_b.py:76-128`) |
| Rejection sampling, weighted split conformal, self-normalized weighted mean | module W only |

**Controls**: the fallback always (= A-ALWAYS); the host always (= A-NEVER); the rest of W; the range limit, each
thruster capped at its 99.5% percentile (`arm_b.py:152-155`); the matched global ceiling, one ceiling for all thrusters
with the certified map's clipped share (`arm_b.py:142-149`); the oracle ceiling map (per thruster and sign, the ceiling
that maximizes mean J on a separate tuning bank); A's switched clip (same score, same margin, the other default); the
clone fallback (reported).

### 4.3 Option C arms

All co-trained: Unifloral's TD3+BC restated with one change in the actor loss (`td3_bc.py:245`):

`actor_loss = -lambda x mean_i [Q1(x_i, pi(x_i)) - k W_i] + mean_i mean_j (pi(x_i)_j - a_ij)^2`,
`lambda = 2.5 / stopgrad(mean_i |Q1(x_i, pi(x_i))| + 1e-7)` (on Q1 alone, as `td3_bc.py:242-244`), k = 1, only the
actor's parameters differentiated through W.

- **Where the band is read.** At `a_i^q = clip(pi(x_i) + xi_i, -1, 1)`, one fresh `xi_i ~ N(0, 0.2^2)` per thruster per
  row per step from BCA's own key (never Unifloral's). Q1 stays at `pi(x_i)`, as Unifloral's line 238 has it, so that
  with k = 0 the step is exactly the host's. (The placement analysis wrote Q1 at the smoothed command too; that form is
  reported as information only, decision D6.) The training rows are kept rows, so the band is read at the declared
  query law's states.
- **The lower edge, paired form** (MAIN `experiments/shift_iw/lower_edge.py:1-17`):
  `W_i = [U_w(x_i, a_i^q) - U_w(x_i, a_i)]_+` where the band certifies the query;
  `W_i = [Q1(x_i, pi(x_i)) - Q1(x_i, a_i)]_+` where it refuses (no value credit, as CONN `command_le_train.py:105-112`).
- **The band**: `U_w(x, a) = R(w(x, a)) x u x max(eta(x, a), 1e-6)`; scores on the bank `|y - Q1| / (u x max(eta,
  1e-6))`, y = Unifloral's own target (`unifloral_td3_bc_bca.py:316-340`); R from WBCP with the bank rows' weights and
  the query's test mass `w(x, a)`, read from a threshold table on a grid of test masses (`iw.mass_grid`,
  `iw.threshold_table`, `iw.lookup`, HOSTS `experiments/shift_hosts/iw.py:136-192`, ungated, in the training
  environment).
- **Exact weights for the main recorder** (new code, section 1): `w = q_xi(a | x) Z(x) / beta(a | x)` outside the ball.
  The training environment evaluates `beta` from the per-row `mu(x)`, hard-water flag and `Z(x)` handed off by the
  scoring environment (section 3.2) and a JAX restatement of the recorder's density (normal driving by change of
  variables through `0.8 tanh(u / 0.8)`; the full-throttle command's density with thruster 1's atom at -1); a stage-0
  test requires the two environments to agree on the density, and that on a large made-up kept-row sample
  `mean(w) = 1 - P_query(w = +inf)` within Monte Carlo error (the query law's mass outside the recorder's support is
  exactly what the weights cannot carry).

**Arms** (in both S and N unless stated; "matched" = matched in S, coefficient locked in N):

| Arm | W |
|---|---|
| **C-BCA (primary)** | as above, k = 1 |
| C-k0 | C-BCA with k = 0: must reproduce the host bit for bit (stop rule S3; first seed-world of each stage) |
| C-F0_m (matched); C-F0 (k = 1, reported) | no value credit where C-BCA refuses, 0 elsewhere (the same refusal, no band; CONN `arms_abc_f0/code/arm_f0.py`); its strength set by scaling k on that term |
| C-RATIO_m; C-RATIO (k = 1, reported) | eta = 1 (the weighted threshold's scale only) |
| C-SC_m; C-SC (k = 1, reported) | weighted split-conformal threshold (Tibshirani et al. 2019) in place of WBCP's; same eta and weights |
| C-SHUF_m; C-SHUF (k = 1, reported) | the same band built from shuffled outcomes: at each refresh the bank scores are permuted across bank rows, and eta's fit targets are permuted within each batch (stream tag 15); same weights and refusal. With equal weights the bank permutation alone changes nothing, so the eta shuffle carries it there |
| C-SUP (matched) | support only: `c x [log w(x, a^q) - log w(x, a_i)]_+`; refusal as C-BCA |
| C-HINGE (matched) | range hinge (`harness.py:373-376`) |
| C-EVEN (matched) | even caution: the cloning term times (1 + nu) (the plain alpha retune the owner rejected) |
| C-RAND1, C-RAND2 | two fixed random-network penalty surfaces (stage 1's 64-wide, 2-deep); matched on the pilot, locked at the pilot's median coefficient for confirmation |
| C-ORACLE (new form) | co-trained one-step exact error on the host's own Q1 and actor: `[delta*(x_i, a_i) - delta*(x_i, a_i^q)]_+`, `delta*(x, a) = r + 0.98 Q1(x', pi(x')) - Q1(x, a)` from one exact float64 step by `dynamics_train.py`, traced under x64 (as `lower_edge.py:37-41` did); exempt from S3, because k = 0 cannot reproduce the host bit for bit under x64 |
| C-ORACLE-S1 (actor-only, reported) | stage 1's one-step oracle as it was run: `min_h Q_h` and the world's top controller at x', against the frozen host critic (`lower_edge.py:24-41`) |
| C-ORACLE-V (actor-only, reported) | stage 1's exact long-run harm penalty `e(s, a) = Q1(s, a) - Q^c_250(s, a)`, c the world's top controller, paired form, against the frozen host critic (stage 1 `harness.py:559, 791`; it scored 82.2 against the one-step oracle's 79.7, stage 1 `RESULTS.md:44, 48`) |
| C-BCA x A, C-HINGE x A, CLIP x A, HINGE1 x A | composites, no training: the arm where A-BCA is on, the host where it is off (HINGE1 = A's gated hinge, k = 1, switched at m_hinge) |

**Matching.** Target: C-BCA's strength in S. Each matched control: a 3-rung geometric ladder of coefficients, log-linear
interpolation to the target, one matched run; tolerance 1%; at most 5 more rounds; otherwise recorded as unmatched,
which counts against C (step 2's rule, PRESPEC.md:178-186). In N each control runs at its coefficient locked in S. All
strengths and fingerprints are locked before any return is computed (step 2 PRESPEC.md:188-194). Matching needs only
strengths, so it can run before K3 is judged, but in the confirmatory stage it starts only after K3 (section 9.3).

**Information only (C-I).** C-I1 eta's share of the variance of `log U_w` against `log R(w)`, and the partial Spearman
correlation of `U_w` with the oracle's error given w (placement test T1's readout, PLACE rank test_plan T1); it fixes
only the wording of C-K2 (expected: eta carries almost nothing, PLACE `red.challenges[11]`). C-I2 an actor-only variant
of C-BCA, C-F0 and C-HINGE against the host's frozen critic (the protocol of every earlier C result: 20,000 steps, the
`lower_edge.py` protocol), to separate placement from critic feedback. C-I3 the sharpness of certified widths (section
6.4).

### 4.4 Shared controls (summary)

| Control | A | B | C |
|---|---|---|---|
| Host alone | A-NEVER | host always | the host (and C-k0) |
| Plain rivals | A-SC, A-OS, A-BOOT, A-MEAN, A-HL, A-RANK, A-EXC, A-BB, each at its own margin | exact binomial / Clopper-Pearson, split-conformal tail, Hoeffding, empirical Bernstein, t, HCOPE-style | C-SC_m, C-RATIO_m, C-F0_m, C-SUP |
| Shuffled outcomes | A-SHUF (sanity), A-KPERM | shuffled s (B-K1) | C-SHUF_m (C-K6) |
| Matched-strength even caution | A-RAND (placement at its own rate) | matched global ceiling | C-EVEN |
| World-blind rules | W | W | W, through the switched composites |
| Support and range rules | caps (mixed) | range limit, knee clip | C-HINGE, C-SUP |
| Oracle references | A-PERFECT, A-ORACLE-REGION | oracle ceiling map, twin outcome | C-ORACLE, C-ORACLE-S1, C-ORACLE-V |
| Fallback variants | clip (primary), gated hinge, clone | clip (primary), clone | refusal = no value credit |

---

## 5. Deployment simulation

### 5.1 Offline computation, then deployment

1. **Train** the hosts (S, N, MW+, MW-), the clones (S, N), A's gated hinge (S, N) and C's arms (S, N). Lock every
   network's fingerprint. For C, also save Q1, Q2, the target networks, the actor, eta, u, the threshold tables and the
   bank weights at each declared refresh j in {10,000, 25,000, 40,000, 55,000, 70,000, 85,000, 95,000}, and the
   networks at the end of each such band's window (just before the next refresh; at the end of training for 95,000).
2. **Compute offline, read-only**: the gap rule on commands; the scores of evidence, bank and replicate rows; A's
   decisions for every rule; B's offline certificates and maps. Lock all decisions in `decisions.json` (write-once). In
   the confirmatory stage this happens before any return of any policy is read (section 9.3).
3. **Deploy**: online episodes of every deployed policy (host, wrappers, composites, trained arms) on every bank and
   world. Decisions never change after step 2 for the offline arms.

### 5.2 Online evaluation episodes

- **Runner**: one episode = 100 steps through `TC.dynamics` with the command rounded to float32 then float64 and clipped,
  exactly as `env.step` (`thruster_chain.py:443-446`). Spot check: 10 starts per bank also run through gymnasium
  `env.step`, bit for bit (stop rule S8). Mixed worlds: each episode's physics from its start's amplitude.
- **One arithmetic for every truth.** `truth.rollout` and `truth.attribution` do not round (`truth.py:85, 368-376`), and
  the clip wrapper's edges are float64 quantiles, so a clipped command differs between the two paths. Before any truth
  call, every deployed policy is therefore wrapped as `x -> float64(float32(p(float32(x))))`. Then `truth.rollout`,
  `truth.attribution` and the online runner coincide bit for bit, and S8 also checks attribution's `J_new` against the
  online J on the spot-check starts.
- **Counts, per seed-world and deployed policy**: 4 situations x 1,000 starts (fresh bank on confirmatory seeds;
  standard bank too); 100 steps each. Every arm runs on the same starts (common random numbers). Worlds: S and N for all
  options; MW+ and MW- for A; M3 and M4 for A and B (information); the ladder for the margin, A-K2 and the readouts.
- **Per start**: J, points, the same policy capped at 80%, the share of steps past the knee, F1, F2, late peak, and the
  visited states, commands, rewards and next states (for the step-level readouts and B-V2).

### 5.3 Staged deployment of B (the online variant)

Per seed-world, situation strong current (recorded conditions and big kicks reported), deployment horizon **H = 1,000
episodes** from the stream `[20261019, 6, k, d, 1]` (shared by S and N):

1. **Reference.** The first 50 episodes run the fallback F. The residuals of their steps at or below `t_lo` give
   `lambda_R,on`, their empirical 90% quantile; `tau_on = lambda_R,on + m_on`. These episodes are part of the
   deployment (F is the default anyway), so they carry no probing cost, and they share no episode with the tries, so
   tau does not depend on the tries. The first 10 also pick B-RUNG's families.
2. **Probe** decision d (B-GROUP: the host unclipped; B-RUNG: F with rung d opened). Each of the next N episodes runs the
   probe map; from it take **one try**: the first step where the deviation was used (`t(a) > t_lo` for the group; the
   rung used, for a rung), with its residual s from the episode's own reward and next state. One try per episode keeps
   tries independent.
3. **Decide** with the exact tail gate after the N probe episodes (primary: one batch decision; **N = 100**, unless the
   pilot's discriminability step of section 8.5 sets N to 150, 200 or 300 in the confirmatory slot). Reported: the
   shipped gate; three looks at 0.4N, 0.7N and N with Bonferroni level 0.05/3 (at N = 100: 40, 70 and 100 tries,
   allowing 0, 1 and 3 harm events; the zero-failure floor at that level is 39, REV `revise_arith.out` section 3); an
   anytime-valid test, the mixture-likelihood e-process `E_n = integral over q uniform on [0, alpha_B] of L_n(q) /
   L_n(alpha_B)`, rejecting "harm rate >= alpha_B" when `E_n >= 20` (zero-failure floor 43, `floor_arith.out` section
   3); and a plain exact decision at 30 tries (zero harm events only). Each reported variant is a separate
   counterfactual deployment on the same starts: identical to the primary up to its stopping time.
4. Certified: deploy the probe map for the rest of the horizon (B-RUNG: then probe the next rung). Refused: return to F.
5. **B-RUNG (reported)**: at most the 2 thruster-sign families whose host command exceeds the recorded edge most often
   in the first 10 reference episodes (at least 1% of steps). Families are probed in parallel at each rung level, one
   try per family per episode, with the confidence Bonferroni-split across families and a fixed-sequence walk within
   each; 5 levels x N episodes fit in H when N = 100. B-V1's truth is computed for each realized probe map, cached by
   map.
6. **Score**: cumulative return over all H episodes, reference and probes included, per world, in points (anchors from
   `anchors_on` on the deployment stream's own starts); stall steps during probing; decision time; the final map's
   score on the scoring bank.
7. **Twin outcome** (reference only): the same protocol with the twin outcome as the try, to show what a perfect outcome
   would certify. **Realized mean variant** (reported): episodes randomized between F and the probe map, mean gates on
   the return difference (Welch t; HCOPE-style empirical Bernstein with declared truncation); about 580 episodes per arm
   would be needed for a 5-point effect (unpaired per-start SD about 34 points in strong current; LIVE 5104-5112 gives a
   bank standard error of 1.63 return units over 1,000 starts).

### 5.4 Crossing the two environments

Every residual needs the host critic, which lives only in the training environment; every episode needs the simulator,
which lives only in the scoring environment. The protocol, one hand-off round trip each:

- **Offline scores** (A, B, ladder, replicates): the scoring environment writes the rows (state, command, reward, next
  state, and the actor's next command from its float32 restatement, HOSTS `actors_f32.py`, `policies.MLPAdapter`);
  the training environment returns `Q_h(x, a)`, `Q_h(x', pi(x'))` and the residual.
- **Staged B**: the probe map is fixed for a decision, so the scoring environment runs all reference and probe episodes
  of the decision, hands off the reference steps and the tries, receives their residuals and decides: one round trip
  per decision (per rung level for B-RUNG). The looks and the e-process are computed from the same stored sequence.
- **Online monitor (information)**: run the 30 episodes with the host alone, score all their steps in one round trip,
  find the episode after which the latch would close, and rerun the later episodes with the clip (deterministic given
  the starts).
- **C's coverage draws**: the scoring environment draws states and commands from the declared law and steps the
  simulator once; the training environment computes y (target networks and target noise), Q1, eta, u, the weight and
  the band, and returns covered or refused per draw.

---

## 6. Metrics (exact formulas)

### 6.1 Return

- Per start i: `J_i`, points `P_i = 100 (J_i - J_rec) / (J_top - J_rec)` with the world's anchors on that bank.
- Per seed k, world w, situation s, policy p: `P_ksw(p) = mean_i P_i`; paired standard error over starts of
  `P_i(p) - P_i(q)` (HOSTS `evaluate.paired`, `evaluate.py:356-362`).
- `NET_k(p) = [P_k,S(p) - P_k,S(host)] + [P_k,N(p) - P_k,N(host)]`, strong current; `NET = mean_k NET_k`.
- `NET_mixed,k(p) = [P_k,MW+(p) - P_k,MW+(host_MW+)] + [P_k,MW-(p) - P_k,MW-(host_MW-)]`, strong current, every start
  in its own region's physics with that region's anchors on the same bank.
- No-stall cost `NSC_k(p) = P_k,N(host) - P_k,N(p)`, per situation.
- Keeps the improvement: `P_k,S,s(p) - P_k,S,s(host)` for s in recorded, big waves, big kicks.
- Every return is also reported in return units.

### 6.2 Harm events

Stall steps per episode (`TC.past_knee`, stalling world only); the stall loss `J(p capped at 80%) - J(p)`; F1 (return
below the zero-command controller's from the same start); F2 (late peak above the 99th percentile of the world's top
controller's late peak on the same bank) (HOSTS PRESPEC.md:486-490; `evaluate.py:23-35`); worst-10% mean J.

### 6.3 Decision accuracy (A and B)

- Value of caution `V_kw` = `P_kw(clip wrapper) - P_kw(host)`, strong current.
- A's decision is correct iff `on = 1[V_kw > 0]`; if `|V_kw| < 2` points both answers count as correct.
- B's group decision is correct iff `certified = 1[-V_kw >= 0]` with the same indifference band; a rung decision is
  correct iff `certified = 1[V_dev >= 0]`, `V_dev` the exact gain of opening that rung over the map without it.
- Expected NET of a rule over replicates: `E-NET_k = sum_w p_on(k, w) V_kw`, with `p_on` the rule's on-rate over the
  replicate evidence sets (section 6.4); realized NET uses the actual evidence. Regret against A-PERFECT; false-switch
  cost `sum over V < 0 of p_on |V|`; missed-harm cost `sum over V > 0 of (1 - p_on) V`.
- Ladder: `E-NET_lad,k(rule) = sum over the 9 middle ladder points i (all but sigma = -1, 1.5, 3.0) of p_on(k, i)
  V_k(i)`, with `p_on(k, i)` over N's replicate blocks with outcomes recomputed under sigma_i, and `V_k(i)` the value of
  caution of N's host in sigma_i-physics (points with `anchors_on`). Clipping always scores `sum_i V_k(i)`; never
  clipping scores 0.

### 6.4 Validity and coverage against simulator truth

The simulator gives three exact truths: the one-step outcome at any (state, command) (`thruster_chain.py:201-221`); the
value `Q^p` of any deterministic policy, wrappers included, by a 1,000-step float64 rollout (`truth.py:98-108`); and the
exact split of any return difference along the deployed path, `truth.attribution` (`truth.py:29-31, 359-383`), whose
terms are zero wherever two policies agree and sum to the difference within 1e-8.

**Replicates (A and offline B).** Replicate r of a seed is a fresh held-out block: 200 calibration and 300 evaluation
episodes of the recording controller, from the streams `[20261019, 5, k, r, 1..2]`, simulated in S and in N from the
same random numbers (and spliced for the mixed worlds), cut by the seed-world's own ball, with its own bank (5 rows per
calibration episode via `calibration/bank.stratified_bank`), its full-throttle rows and its rest, all scored by the
frozen host critic. A replicate is therefore drawn exactly as the actual evidence was, episode clustering included (the
14-28 full-throttle rows of a seed-world came from 13-25 episodes), and its group sizes vary as the actual ones would.
R = 1,000 blocks per seed (about 500,000 episodes per world); their union is the population that defines the truth
(about 35,000-70,000 full-throttle rows and 850,000 rest rows per seed-world expected).

**A (per seed-world, over the R replicates):**
- True gap `Delta_ev = q_0.9(pool_G) - q_0.9(pool_R)` over the union (inverted-cdf quantiles), with its Monte Carlo
  standard error. Also the mean gap, the Hodges-Lehmann shift, and the n-matched gap `q_{0.9/(1 - 1/(n_G + 1))}(pool_G)
  - q_0.9(pool_R)`: the quantity WBCP's posterior actually targets, because each crossing includes the test atom
  (`wbcp.py:43-54`), which for 35-70 rows moves the group's crossing to about its 91-92% quantile.
- **A-V1, boundary false-switch rate.** For each seed-world, shift every full-throttle score of every replicate (and of
  the union) by `c_kw = m - Delta_ev`, keeping episodes and groups. The true gap is then exactly m. `FSR_kw = (1/R)
  sum_r 1[L_gap,r + c_kw > m]`. Because L_gap moves exactly with the shift, `FSR_kw = 1 - COV_kw`, where `COV_kw = (1/R)
  sum_r 1[L_gap,r <= Delta_ev]` on the unshifted replicates: A-V1 is at once the error control at its hardest point and
  the coverage of the true gap by L_gap. The same posterior draws are used for the shifted and unshifted replicates.
- **Size table (reported, and used to read A-K1).** Every rule of section 4.1 at its own boundary (its own statistic's
  true value shifted to its own margin), the rate it switches on; and an equal-shapes null (group and rest pooled and
  re-dealt at the replicate's sizes, then m added to the re-dealt group scores).
- **Natural false-switch rate (reported).** `(1/R) sum_r 1[L_gap,r > m]` in seed-worlds with `Delta_ev <= m` (in practice
  N); round 1's no-stall lower bounds were -0.01 to +0.02 against a 0.10 margin (ROUND1.md:31), so this rate says little
  about the error control.
- **n-matched coverage (reported)**: the share of replicates with `L_gap,r` at or below the n-matched gap.
- Power: `(1/R) sum_r 1[L_gap,r > m]` where `Delta_ev > m`.
- Transfer (not promised): the same gap at the host's own visited steps in strong current (steps past `t_lo` against the
  rest, each with its episode's own outcome).
- Every rate with its Clopper-Pearson 95% interval.

**B:**
- **Truth of a decision.** `pi(t) = P_pop(s > t)`: the harm rate at threshold t over the truth population of its tries
  (offline: the union of the replicate blocks' conditioned full-throttle rows; staged: the union of the first uses of
  the replicate probe sets of section 5.3, `[20261019, 4, k, d, r]`, 1,000 sets of N).
- **B-V1, error of a replicate.** Replicate r computes its own `tau_r` (offline from its own decoupled rest; staged from
  its own reference episodes) and its own certificate. `error_r = certified_r AND pi(tau_r) > alpha_B`, so the error
  includes tau's randomness and the clustering of tries in episodes, which the exact binomial test ignores.
- **B-V1, boundary null (judged).** Per seed-world and per unit (B-GROUP offline; B-GROUP staged), shift every try's
  score by the constant `c_kw` that makes `pi(tau_pop) = 1.1 alpha_B`, `tau_pop` the population threshold (the 90%
  quantile of the population rest plus m, or m_on). Reported at `pi(tau_pop) = alpha_B`, `1.2 alpha_B` and
  `1.5 alpha_B`; the natural decisions and the ladder decisions with `pi` in `(alpha_B, 2 alpha_B]`; and the
  fixed-threshold variant (tau held at the actual evidence's value over all replicates), which isolates the gate.
- **B-V2, transfer of the certified event.** Run the deployed map p_B on the scoring bank (strong current). At every step
  where `p_B(x_t)` departs from `F(x_t)`, compute the exact residual `s_t` from the step's own reward and next state with
  the same critic and actor as the gate. `HR_k = #{departing steps with s_t > tau} / #{departing steps}` per seed-world
  (tau as the gate used it: offline tau, or tau_on). A world with fewer than 3 seeds having any departing step reads
  "vacuous" for B-V2.
- **B's safety readouts (reported with B-R).** From `truth.attribution(p_B, F)`: the share of departing steps with
  `term_t < 0`; the share with `term_t < -c`, c = 1% of F's mean absolute per-step reward on that bank; the harm mass
  `sum min(term_t, 0)`; per start `D_i = J_i(p_B) - J_i(F)`, `P(D_i < 0)`, its 10% tail mean, stall steps, F1, F2.
- Refusals: share refused, split into "too few tries" (n below the floor) and "too many harm events", priced against
  the oracle ceiling map: `P(oracle map) - P(p_B)`.

**C (per seed and world; refusal counts as covered for the promise):**
- **The place of use.** C's band acts only inside training: band j (refresh j's R, eta, u) shapes the actor for the 5,000
  updates after refresh j, at the training batches' kept-row states. The declared refreshes are j in {10,000, 25,000,
  40,000, 55,000, 70,000, 85,000, 95,000}.
- **C-V1 promise.** For each declared refresh j: M = 50,000 fresh draws from the declared law under actor j (kept-row
  states from fresh recorded-law episodes, `[20261019, 7, k, w, j]`); for each, `y = r + 0.98 min_k Qbar_k(x',
  clip(pibar(x') + eps))` with r and x' from one exact simulator step at the drawn command and one fresh target-noise
  draw (the host's own target rule), all networks those saved at refresh j; `COR_j = (1/M) sum 1[refused or |y - Q1| <=
  U_w]` with band j. `COR` = the mean of `COR_j` over the 7 declared refreshes. Reported beside it: each `COR_j`;
  staleness (band j on the networks at the end of its window, which is what the actor climbed); the band refreshed on
  the final networks (information only: nothing uses it).
- **C-V2 usefulness**, at the same draws, with soft refusals counted as refusals: `RS` = refused share; `CAC` = covered
  among certified. Judged overall and in the **near-knee hard-water zone** (states with `c + w1 > 0.565`, query throttle
  in (0.70, 0.80], query outside the ball), where the recorder has support and a finite band is possible. Past 80% on
  the main recorder RS is about 1 by construction (section 2.3): reported, not judged. Also reported inside and outside
  the ball.
- **C-V3 per-bank promise.** For each seed-world and declared refresh j: 200 replicate banks (the banks of the first 200
  replicate blocks: fresh calibration episodes, the same ball cut, 5 rows per episode), scored by the frozen networks of
  refresh j with its eta and u, weights recomputed, each bank's own threshold table, and each bank's `COR_b,j` on the
  same M draws. Per (seed-world, bank) unit, `f_b` = the share of the 7 refreshes with `COR_b,j < 0.90`; `F3` = the mean
  of `f_b` over all units of the stage. The same is computed for C-SC and C-RATIO; `F3(C-SC) - F3(C-BCA)` is reported as
  the coverage credit of the per-bank construction (shared with any training-conditional tolerance limit; with equal
  weights it is Wilks'). Also reported: whether the actual bank's COR is below 0.90, per seed-world.
- **C-I3 sharpness (reported)**: at certified draws in the near-knee zone, the median of `U_w` divided by the 90% quantile
  of `|y - Q1|` in the same zone.
- Information: on-policy coverage at every visited (state, executed command) of the 1,000 strong-current episodes, and
  at the same states with the smoothed command; value error `P(|Q1 - Q^pi| <= U)` and uncaught over-estimation
  `P(Q1 - k U > Q^pi)` at 2,000 visited pairs per bank (`Q^pi` by `truth.q_value_jax`); the share of hard-water states
  whose best throttle for `Q1 - W` lies past 80% (`lower_edge.py:49-58`).
- **Coverage arithmetic.** `coverage_c.py` computes COR as (covered + refused) / all. It does not reuse
  `experiments/shift_iw/coverage.py`'s coverage, where a refused step never counts as covered (`coverage.py:13-15`); a
  stage-0 unit test pins COR on a made-up table that contains refusals.

### 6.5 Uncertainty and dependence

- Starts are independent given a frozen policy; all arms share starts, so differences are paired by start.
- Steps are clustered in episodes: every step-level rate gets an episode bootstrap (2,000 resamples of episodes,
  `[20261019, 8, k, 1, 1]`) within each seed, and the measured design effect is reported. B-V2 uses a two-stage bootstrap
  (seeds, then episodes within seed). Precision at 90% coverage with 1,000 episodes of 100 steps: +-0.19 to +-1.45
  percentage points for within-episode correlation 0 to 0.6 (`floor_arith.out` section 6).
- Seeds are the unit of replication for every return and credit claim (section 8). Replicate-based rates (A-V1, B-V1,
  C-V3) pool independent replicates over seed-worlds; their marks are in section 7.
