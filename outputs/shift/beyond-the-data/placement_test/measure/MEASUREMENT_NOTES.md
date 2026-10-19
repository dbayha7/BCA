# How to measure return and coverage for options A, B, C under simulated deployment (design only)

Design notes, 19 Oct 2026. Nothing trained, nothing run on ThrusterChain data. The only computation is closed-form
arithmetic (`floor_arith.py` / `floor_arith.out` in this folder: binomial floors, e-process boundaries, t-test power,
coverage precision). All numbers quoted from records are exploratory (seeds 0-4).

Abbreviations: TC = origin/main:experiments/shift/thruster_chain.py; TR = origin/main:experiments/shift/truth.py;
EV = origin/claude/shift-hosts:experiments/shift_hosts/evaluate.py; BR = .../shift_hosts/band_readout.py;
HP = .../shift_hosts/PRESPEC.md; S1 = outputs/shift/beyond-the-data/restructure/step1/PRESPEC.md; R1 =
.../restructure/routes/round1/ROUND1.md; ADD = .../restructure/step2_results/addon_report/report.md (all on
origin/claude/main-bca-connection-x4o46s); LIVE = scratchpad/status/live_c264.html; TH = placement/result.json theory.

## 0. What every option shares

**Truth objects the simulator gives exactly.** Dynamics are deterministic (TC:13-22), one implementation used by
env.step, data and truth (TC:24-27). For any deterministic policy p (host actor, or any wrapper: clip, switch, ceiling
map) and any (x, a):
- one-step target pieces r(x,a), x' = dynamics(x, a) exactly (TC:201-221), so a counterfactual one-step target at the
  actor's own command needs no model;
- Q^p(x,a) by one float64 1,000-step rollout (TR:1-9, 98-108; JAX TR:171-189), TD3's smoothed level by MC (TR:305-325);
- the exact performance-difference terms along p_new's own path, term_t = Q_ref^(100-t)(s_t, p_new(s_t)) -
  V_ref^(100-t)(s_t), summing to J_new - J_ref within 1e-8 (TR:29-31, 359-383). Terms are 0 wherever p_new = p_ref, so
  every return difference is attributed exactly to the steps where a rule changed the command.

**Online episode = one start, 100 steps, through the env's arithmetic.** Starts are the only randomness (TC:48-49,
331-336). Use EV.score (EV:302-353: J, points, capped J, regret, F1, F2, late peak, past-80% share, tracking/energy) and
EV.paired (EV:356-362). Spot-check a subset through gymnasium env.step for bit equality (the hosts' S7 closed-loop
check, HP S7, EV:42-48). Score on the standard banks for comparability and on fresh confirmatory banks (seed registry,
`placement_test/seed_registry.json`); any try-gathering stream must be disjoint from scoring banks.

**Harm events (all options).** Stall steps per episode N_stall = sum_t 1[past_knee(a_t)] in the stalling world
(TC:254-267), the stall loss J(p capped at 80%) - J(p) (EV:25, 333-334), F1 and F2 (HP:486-490). In the no-stall world
past-80% steps are reported but are not harm.

**Worlds needed.** Stalling (sigma 1.5) and its no-stall twin (same streams) are the minimum; the no-stall cost is
undefined without the twin. Add: (i) the stall-strength ladder, world_physics('stall', stall_slope=sigma) (TC:143-149;
sigma = -1 equals no stall), at decision level as R1:66-96 did (fixed learner, evidence outcomes and deployment value
recomputed exactly per sigma); (ii) mixed worlds: stall iff wave amplitude > 0.25 and the mirror (R1:121-146). Wave
amplitude is invariant within an episode (TC:11, 170-171), so a mixed online episode simply uses the start's physics,
and a mixed log is a splice of whole episodes. The Unifloral hosts exist only on the weighting-test generator, which has
no no-stall world (HP:68-71, 611-620; seed_registry.json) - a no-stall twin must be generated or the main-study
generator used.

**Dependence and units.** Hierarchy: data seed k (log + training) > world > bank > start i (iid) > step t (dependent:
current constant, wave periodic, state carried). Claims are per seed (unit of replication, as HP:536-549); starts are
iid given a frozen policy; steps are clustered in episodes. All arms run on the same starts (common random numbers):
paired per-start differences. Rules: (1) returns: per seed, mean over starts with paired SE over starts; across seeds,
mean of per-seed differences with Student t on S-1 df (HP:540-547). (2) Step-level coverage: ratio estimator
sum covered / sum steps per seed, CI by episode (cluster) bootstrap (BR:28-34 already does 2,000 draws) or the
cluster-robust delta method; report the measured design effect deff = 1 + (m-1) rho. (3) Error rates of certificates
(probabilities over the draw of the held-out set): replicate held-out sets from fresh generator streams of the same law
(same size, removal, thinning, frozen critic), Clopper-Pearson bounds on the error frequency. (4) Staged deployment:
tries are iid only conditionally on the decision history; use one try per episode, fresh tries per decision.

**Precision from past results.**
- Seeds dominate. Step 2's NET: SE 4.49 = starts 1.58 + seeds 4.20 (ADD:142-155); conformal vs BCA NET: seeds 0.85,
  starts 0.19 (ADD:185). With 1,000 starts the start part is <= 12-15% of the variance; 1,000 starts per bank is enough,
  2,000 halves it. Per-start SD of a seed-averaged paired difference = SE_starts x sqrt(1000): NET ~50 points, recorded
  G2 ~19, conformal-vs-BCA ~6. Half-width 1.96 sigma/sqrt(n): NET-type differences +-3.1 points at n = 1,000.
- Absolute per-start SD (stall-aware controller, LIVE:5104-5112): 21.9 / 34.3 / 21.9 / 88.8 points (recorded / strong /
  waves / kicks) - the cost of any UNPAIRED (realized, no-twin) comparison.
- Seeds: minimum detectable paired effect (one-sided 5%, power 80%) = 1.36 sd at 5 seeds, 0.85 sd at 10
  (floor_arith.out section 5). Seed sds on record: NET 9.4, BCA-vs-forced 4.9, conformal-vs-BCA 1.9, G2 0.9-2.7 points
  (ADD:142-195). E.g. sd 1.9 needs 24 seeds for a 1-point effect, 8 for 2 points.
- Coverage: SE = sqrt(p(1-p) deff/(E m)). At p = 0.9, 1,000 episodes x 100 steps: +-0.19 pp (rho 0), +-0.61 (0.1),
  +-1.03 (0.3), +-1.45 (0.6). BR's 200 starts give +-1.4 to +-3.2 pp at rho 0.1-0.6 (floor_arith.out section 6).
  Seed-to-seed spread is larger (coverage at the actor 73-79% across seeds on medium-expert,
  RESULTS_coverage_at_actor.md:41-44), so report per seed and the seed mean with t interval.
- Certificate error rates: R = 800-1,000 replicate held-out sets per seed-world show <= 7.3% when the truth is 5%
  (CP upper 6.3-6.5%; floor_arith.out section 7).

## A. Harm switch on a declared group or world (O8b)

**What it promises (TH).** With equal weights each group's WBCP crossing is a Wilks/exact-binomial upper tolerance
bound (TH (b)); group-wise promises are valid for groups declared without outcomes (TH Use 5). The step-1 statistic,
the 5% quantile of the two-posterior gap (S1:146-154), is itself only "stretched" theory (rank O8). Each draw's
crossing includes the test atom (wbcp.py:43-54), which lifts the small group's crossing more than the large group's,
so at n_G = 15-30 the gap's lower bound can be anti-conservative at margin 0 (consistent with 4.5-43% false alarms at
margin 0, R1:111). The measurement must therefore test the actual frequentist behaviour, not assume it.

**Objects.** Group G: rows with max_k|a_k| > t_lo by the gap rule from commands only (S1:51-63), or a declared state x
command cell (Mondrian, e.g. wave-amplitude bins for the mixed world). Score s = Q_h(x,a) - [r + 0.98 Q_h(x', p_ref(x'))]
(S1:137-145; declare h = min(Q1,Q2) or Q1 and p_ref = host actor without noise for Unifloral hosts). Evidence: held-out
group rows (calibration and held-out episodes, outside the ball: 15-30 per seed-world, R1:25) and bank rows. Switch:
ON iff L > m, m = 0.10 fixed in advance. Caution action: clip to the recorded range when t(pi_H(x)) > t_lo (S1:192-199),
or the gated range hinge (retrained actor; step-2 pattern).

**Truth (computed, per seed-world).**
- Delta_ev = q0.9(F_G) - q0.9(F_R), F = law of s at fresh rows generated exactly like the evidence (recorder of that
  world, training start law, same removal, group rule and thinning; frozen critic). s is exact per row; only the
  population is sampled: use >= 1e5 fresh rows per group (quantile MC SE reported). Also the mean gap, and the
  n-matched functional q_{0.9+1/(n_G+1)}(F_G) - q0.9(F_R) to show how much of L is the atom.
- Delta_dep = the same functional on the host's own visited steps in deployment episodes (group = visited steps past
  t_lo; s from the episode's own r_t, x_{t+1}). Not promised (TH Use 6); it measures transfer.
- Delta_V = the same functional on the value error Q_h - Q^{p_ref} (TR q_value_jax): information; one-step coverage is
  not value-error coverage (TH Use 4).
- V_w,b = mean_i [J_caution(x0_i) - J_host(x0_i)] by paired online episodes, per bank b, world w. Correct decision:
  ON iff V_w,strong > 0 (primary bank declared; others reported).

**Coverage / validity metrics.**
- A-V1 lower-bound coverage per seed-world: cov = (1/R) sum_r 1[L^(r) <= Delta_ev] over R = 1,000 replicate evidence
  sets; nominal >= 0.95 (bank-conditional reading). Report per seed-world with CP interval and the minimum.
- A-V2 false certification: P(L > m) for seed-worlds with Delta_ev <= m (no-stall, mild ladder rungs), bar <= 0.05;
  test size under the exact exchangeable null (fresh G and R rows pooled and re-dealt, Delta = 0) at m = 0 and m = 0.10.
- A-V3 power: P(L > m) where Delta_ev > m.
- Margin check: on the ladder, sigma* = break-even of V (R1:78: about 0.21); compare m with Delta_ev(sigma*). If they
  differ, the margin, not the posterior, places the decision (R1:34, 108-119).

**Decision accuracy and return.** p_on(k,w) from replicates plus the actual-evidence decision. Confusion table over
seeds x {stall, no-stall, mixed+, mixed-, ladder}; mixed worlds also per declared cell. Expected NET = mean_k sum_w
p_on V_w (strong current), realized NET, perfect-switch NET = sum max(V,0), regret, false-switch cost FSC = sum_{V<0}
p_on |V| (a no-stall false switch costs about 61.8 points in strong current with deployment clipping, 2.0 / 8.3 / 21.3
in the other situations, fallback.md:167-174; 30.9 for the trained hinge, ADD:244), missed-harm cost sum_{V>0}
(1-p_on) V. Because the switch only selects between two fixed policies, online return per seed-world is exactly
decision x V: the online episodes are needed for V (once per seed-world-bank), the replicates for p_on.

**Rivals on the same evidence and margin.** Plain split conformal (empirical q0.9 gap); Bayesian bootstrap without the
atom; ordinary bootstrap percentile bound; exact order-statistic bound (97.5% lower bound on q0.9(F_G) minus 97.5%
upper bound on q0.9(F_R), distribution-free); Welch mean-gap bound; Hodges-Lehmann shift bound; Mann-Whitney p < 0.05
(no margin; S1:240-251); exceedance count vs Bin(n_G, 0.1) (Clopper-Pearson); no-margin BCA; forced-on; random
switching at the switch's own rate (S1:201-202); shuffled outcomes (S1:225-238). BCA's own credit: score (vs shuffled,
vs support-only group size), eta (an eta-scaled score s/(u eta), matched), the Bayesian step (vs atom-free and plain
tests), weights (exact w(c) vs unweighted vs rejection sampling; only where the evidence law overlaps the target,
i.e. the weighting-test generator; strong current in the main study has no overlap, TC:42).

## B. Safety certificate per deviation (O8a), offline and staged

**What it promises (TH; rank O8).** A bank-conditional certificate on the decision's own held-out outcomes: with
probability >= beta over the tries, P_try(g <= 0) <= alpha (tail gate) or E[g] >= 0 (mean gate; Gaffke, proof
unreviewed). Equal weights: exact binomial / Wilks (TH (b)). With weights: about the deployment mix, with four
conditions; credibility != confidence (TH (b)), toys held <= 4.3% (LIVE:10799-10814). Nothing is promised for an
outcome that does not contain the harm (arm B's one-step critic gain missed the stall; rank O8).

**Objects.** Fallback F = clip to the recorded range (deployment clip +106.0 vs clone +68.9 strong current, +28.0 vs
-51.8 big kicks, fallback.md:85-87). Decisions d = (thruster k, rung j) above the recorded top, rungs 0.05 wide, walked
in a fixed order until the first refusal (arm_b.py:8-19, 131-139). Map policy p_M(x)_k = sign(pi_H(x)_k)
min(|pi_H(x)_k|, t_k). Outcomes: offline g1 = arm B's one-step critic gain (negative control); offline k-step segment
outcome against the critic at the fallback (k = 10, 25); online twin episode outcome Delta_e = J_{M+d}(x0_e) -
J_M(x0_e) on eligible episodes (some step reaches the rung); online twin first-use advantage g_e = Q_M^(100-t*)(x_t*,
a_{M+d}) - Q_M^(100-t*)(x_t*, a_M); online realized (no twin) = randomized episodes, mean gate only.

**Validity metrics.**
- B-V1 promise level: pi_d = true P_try(g <= 0) from >= 1e4 fresh tries of the same law (exact). Error = certified and
  pi_d > alpha; frequency over R replicate try sets per decision; bar <= 5% (+ a declared MC allowance, e.g. 7.3%).
- B-V2 deployment harmful-use rate: on the scoring bank, run the final map p_B; at every step where p_B differs from F,
  the attribution term tau_t (TR:359-383, ref = F) gives the exact effect; HR = #{tau_t < 0} / #{uses}; per decision via
  the single-thruster counterfactual (thruster k alone deviates, then F). Harm mass HM = sum min(tau, 0). Bar: HR_d <=
  alpha. Not a promise (different state law: first-eligible states vs p_B's visits), so it measures transfer.
- B-V3 episode safety: Delta_e = J_{p_B} - J_F per start (sum of tau): P(Delta_e < 0), CVaR10 of Delta_e, worst-10%
  mean J, N_stall, F1/F2; policy level (Seldonian): mean Delta >= 0 per seed and the share of seeds where p_B < F.
- Refusal: share refused, split evidence-limited (n < n_min) vs evidence-based; false refusals against an oracle map
  (rung opened iff it raises mean J on a separate tuning bank); price J(oracle map) - J(p_B). In the no-stall world the
  refusal cost is the point (clip-always costs about 61.8 there).
- Twin sanity check: with equal weights BCA's tail decision must equal the exact binomial decision on 100% of decisions
  (LIVE 28,000/28,000 agreement, rank O8).

**Staged deployment (owner's online variant).** Budget N in {10, 30, 100} episodes per decision from a fresh deployment
start stream; one try per episode (independence); decisions in a fixed sequence per thruster (fixed-sequence testing:
stopping at the first refusal controls the family-wise error per thruster at the per-test level; Bonferroni over the
four thrusters or declare one family per thruster). Valid procedures: (a) fixed-n batch decision at N; (b) looks at
10/30/100 with Bonferroni spending; (c) anytime-valid e-process: mixture over q uniform on [0, alpha] of
L(q)/L(alpha), a test supermartingale under every null p >= alpha (Ville). Evidence floor (beta 0.95, failures
allowed; floor_arith.out sections 1-3):

| alpha | fixed n: n_min | f at N = 10/30/100 | 3 looks: n_min | f at 10/30/100 | anytime: zero-failure n | f at 30/100 |
|---|---|---|---|---|---|---|
| 0.5 | 5 | 1 / 10 / 41 | 6 | 1 / 8 / 38 | 7 | 7 / 35 |
| 0.2 | 14 | - / 2 / 13 | 19 | - / 1 / 11 | 20 | 0 / 9 |
| 0.1 | 29 | - / 0 / 4 | 39 | - / - / 3 | 43 | - / 2 |
| 0.05 | 59 | - / - / 1 | 80 | - / - / 0 | 88 | - / 0 |

What staging does: offline, 15-30 group tries per seed-world and 0-78 per rung (rank tradeoffs) mostly refuse;
online tries drawn from the deployment law need no importance weights (n_eff = n), so the floor is 29, not about 83-86
under the strong shift (LIVE:10987-10995). But with equal weights BCA reduces to the exact binomial test, so B-online
can credit only the placement. Weights matter again only if probing happens under one law (e.g. recorded mix) and the
certificate targets another with overlap (weighting-test generator). N = 10 cannot certify any alpha <= 0.25; N = 30
certifies alpha 0.1 only with zero failures and only as a single fixed-n look; N = 100 allows 4 / 3 / 2 failures.
Twin probing costs no deployment return (the deviation is evaluated in the twin); realized probing exposes probe
episodes and needs hundreds of episodes per decision for a mean comparison (unpaired per-start SD 34 points in strong
current: about 580 per arm for a 5-point effect, 36-146 for 20-10 points, approximate t only; a guaranteed mean bound
needs a return floor, and the provable one is about -6.6e6, rank O9). Report cumulative return over the whole horizon
(probe + post), stall steps during probing, and the decision time.

## C. Value lower edge at the actor's own command (O4)

**What it promises (TH Uses 2-4, 6).** Marginal coverage >= 1 - alpha of |y - Q1| at draws from the DECLARED query law
(test-mix states x smoothed command pi(s) + xi, xi = clip(N(0, 0.2^2), +-0.5)), with exact state x command weights and
refusal counted as covered; bank-conditional only with equal weights; nothing at the deterministic command actually
executed, nothing for Q^pi error, nothing for the slope, and the actor's argmax choosing its own queries is outside
the theorem. Every past run used a min(Q1,Q2) band (TH; rank O4); the test must recalibrate on Q1.

**Coverage metrics (per seed, then across seeds).** Covered: |y - Q1(x,a)| <= U_w(x,a); refused: U_w = +inf; refused
counts as covered for the promise, against coverage for usefulness (shift_iw/coverage.py:7-38, BR:28-34).
- C1 promise test: at the trained actor frozen at the end, M = 50,000 fresh draws from the declared law; y = r + 0.98
  min_k Qbar_k(x', clip(pibar(x') + eps)) with r, x' from one exact simulator step at the counterfactual command and one
  fresh target-noise draw (the host's own target rule). Report covered-or-refused (bar >= 90%), covered-among-certified,
  refused share, all split within 80% / past 80% and inside / outside the ball. Three band states: (i) the band of the
  last training refresh (what training used), (ii) a final refresh rescoring the bank with the final critic and actor
  (what deployment would use), (iii) the refresh-time actor's own law (pure promise). (i) vs (ii) measures staleness and
  feedback.
- C1-bank: bank-conditional check: B = 200-1,000 replicate banks (fresh calibration episodes, same thinning and removal)
  -> R_b per bank -> coverage of each on the M draws; report P(coverage_b < 0.90) with CP upper bound vs 5%
  (DEPENDENCE.md:10-12 definition). Equal-weight arms only carry this promise.
- C2 on-policy deployment coverage: at every visited (x_t, a_t = pi(x_t)) of 1,000 online episodes per bank (not BR's
  first 200), y_t from the episode's own r_t, x_{t+1} plus one target-noise draw; also at x_t with a = pi(x_t) + xi to
  separate state-law shift from the deterministic-command gap. Episode-bootstrap CIs, deff reported, by zone and bank.
- C3 value error (not promised): P(|Q1 - Q^pi| <= U) and the lower-edge validity P(Q1 - k U_w <= Q^pi) at visited
  (x,a), Q^pi by TR q_value_jax (and the smoothed level, TR:305-325). The direction that matters is uncaught
  over-estimation P(Q1 - k U_w > Q^pi).
- C4 shape: over-preference D at the trained actor (TR:270-274) and the share of hard-water states whose argmax of
  Q1 - W over a throttle grid lies past 80% (command band check (e), RESULTS_command_weighted_band.md:42-44, 77-89).

**Return.** Online episodes, every bank, stall and no-stall twin: paired per-seed differences vs host alone (Unifloral
TD3+BC) and vs each rival; no-stall cost = J_host - J_arm in the no-stall world (prior 14.8 actor-only, 16.2
co-trained, results-2026-10-14/README.md:32-36, 76-77); stall steps; F1/F2. Variants: always-on, and switched by A
(ON only where A certifies; then the no-stall cost is A's false-switch rate x cost). Rivals (rank T5): N, Q1 oracle G
(room check: if G - N < 1.8 the test is uninformative), F0 (same refusal, no band), ratio-only (eta = 1), eta shuffled
within state, weighted split conformal (no Bayesian step), plain no-credit past 80%, range hinge matched to the
displacement, clone fallback at refusal, classifier support, even caution matched, BCA's capped pull and the plain
alpha retune {1.25, 1.88, 2.5}. Because F0's refusal set equals BCA's by construction (the +inf verdict ignores the
scores, rank O5), A - F0 isolates the finite width; ratio-only vs A isolates eta; split conformal vs A the Bayesian step.

## Reporting rules (all)
Per seed always; pooled only with all seeds (HP:549-550); return claims one-sided t on S-1 df; coverage claims
one-sided lower bounds (episode bootstrap within seed, t across seeds) plus the count of seeds below the bar; error
rates as CP upper bounds; every rate with its denominator (decisions, uses, steps, episodes). Pass marks, margin m,
alpha, beta, banks, streams, outcomes, groups and the break-even rule frozen before any data are read; seeds 0-4
exploratory, confirmatory seeds per seed_registry.json under a pre-registration and the owner's go.
