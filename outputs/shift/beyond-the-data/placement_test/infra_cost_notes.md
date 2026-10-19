# Placement full test (A harm switch, B per-deviation certificate, C value lower edge): infrastructure and cost

Design only. Nothing here trained on ThrusterChain data or ran an experiment. The only runs were micro-benchmarks on
made-up arrays (bench/ beside this file). Branch abbreviations: MAIN = origin/main, HOSTS = origin/claude/shift-hosts,
CONN = origin/claude/main-bca-connection-x4o46s.

## 0. This container
- 4 CPUs (Intel Xeon @ 2.80GHz, 1 thread per core), 15 GB RAM, no GPU (nvidia-smi absent).
- No runs/ folder (gitignored, .gitignore), no /opt/bca-venv, no /opt/bca-shift-venv, no ~/.cache/bca-shift/py
  (gymnasium 1.0.0 dependency dir). So experiments/shift cannot be imported here as is (it needs numpy 2.2.6 plus
  gymnasium from BCA_SHIFT_DEPS: MAIN experiments/shift/__init__.py:17-43; install_deps.sh). Unifloral's own files need
  Python 3.10, jax<0.6, numpy<2, d4rl, mujoco-py, distrax, tyro, wandb and the Dockerfile variables
  (baselines/unifloral/requirements.txt; Dockerfile; HOSTS experiments/shift_hosts/unifloral_import.py:69-76, 228-234).
- Every saved log, critic and actor of seeds 0-4 lives on the laptop/desktop under runs/ (not here). New work must
  regenerate data (deterministic) and train new Unifloral hosts anyway.

## 1. Data
Main gapped log, both worlds (same generator, own log per world):
- MAIN experiments/shift/dataset.py:176-226 generate(rep, world) — 1,000 episodes x 100 steps through env.step,
  streams default_rng([20261012, rep, 7]) starts+noise, [.., 8] coin (152-158); splits (500, 200, 300) by episode:
  train 0-499, calibration 500-699, evaluation 700-999 (13-15, 58); calibration bank = 5 rows per calibration episode
  via calibration/bank.stratified_bank, default_rng([20261012, rep, 10]) (214-220). world='no_stall' gives the twin.
- Densest-15% removal: MAIN experiments/shift/shift.py:52 (GAP_Q 0.15), 111-123 define_gap (experiments/truth/gap.py
  find_gap on all 50,000 training commands), 164-192 make_removal('gap'): training rows and bank rows in the ball
  leave; evaluation split untouched. Gate pipeline: MAIN experiments/shift/acceptance.py:590-657 data_seed (generate,
  define_gap, removals, leak checks), 705-734 load_seed (regenerate + verify against runs/.../record.json, which is
  NOT in this container), 692-698 SeedData.prepared -> learners.prepare.
- Normalisation/prepared rows: MAIN experiments/shift/learners.py:193-197 normalization (std + 0.001), 251-272 prepare.
  Unifloral uses its own obs_mean/obs_std (raw std, +1e-3 inside the network): HOSTS unifloral_import.py:394-397;
  baselines/unifloral/algorithms/td3_bc.py:105, 310-311.
- Held-out calibration episodes: rows of split 1 outside the ball and outside the bank, about 15,973 per seed-world,
  of which 15-30 lie past t_lo (HV2 numbers in MAIN .../routes/round1/scratch/_feasibility/count_rows_1_2_3_4.json;
  code count_rows.py). Route code that selects them: MAIN .../routes/round1/scratch/heldout_calibration_rows/
  run_route.py:102-109. Stage 2's tau rows (same set): MAIN .../stage2/code/deploy_check.py:56-65.
- Gap rule (outcome-free, commands only): MAIN .../restructure/step1/step1lib.py:174-211 (t = max_k |a_k|; largest
  gap above the 99th percentile; >= 0.05 wide, >= 9 rows). t_lo 0.78-0.79 in all 10 seed-worlds (step1 PRESPEC.md).
- Spliced wave-size mixed world (ROUND1 route 5): MAIN .../routes/round1/scratch/mixed_waves/run_route.py:41 (CUT
  0.25), 68-75 stalls/world_of, 85-97 kernel-weighted WBCP posterior, 100-113 rollout with a fixed physics per call.
  It spliced EVIDENCE rows of the two gate worlds; no spliced training log, gap or critic exists (ROUND1.md: "only
  then the spliced-log critics ... needs a separate go-ahead"). Current-split mixed dynamics as a per-row physics
  switch: MAIN .../routes/round1/scratch/runtime_monitor/prep.py:71-86 make_dyn.
- Weighting-test data (known current-law shift, exact w(c)): MAIN experiments/shift_iw/world.py:7-48 (laws, w(c),
  HoldRecorder, 5 qualities random/medium-replay/medium/medium-expert/expert, ball = expert's densest-15%),
  data.py:1-50 (training-law log 500/200/300, mild and strong test logs 300 episodes each, classifier samples, start
  banks default_rng([20261016, 1000]), anchors random quality = 0, stall-aware = 100). NO no-stall twin:
  world.py:69 PHYSICS = TC.STALL; data.py:80 ENV_ID stall, 336, 454, 646, 652 use W.PHYSICS (world.simulate and
  HoldRecorder do accept physics: world.py:218, 324). Inferred from the recorder: expert-quality logs never command past
  the knee (normal driving < 0.8, hold exactly float32(0.8)), so an expert no-stall twin would be bit-identical to the
  stall log; only the random-command rows (eps > 0 qualities) carry past-80% outcomes.
- Four-learner hosts' prepared files (weighting test only, qualities medium-expert and medium): HOSTS
  experiments/shift_hosts/prepare.py:1-41; runner QUALITIES runner.py:126. No main-log prepare exists for Unifloral.

## 2. Start banks, rollout evaluator, points
- Banks: MAIN thruster_chain.py:83-92 (START_LAWS recorded / strong_current c~U[0.7,1.1] / big_waves amp~U[0.4,0.7] /
  big_kicks SD 1.5), 308-315 draw_starts, 331-336 bank_starts(situation) (1,000 starts, default_rng([20261012,2,9]),
  common random numbers across situations; the same starts serve both worlds: physics is separate).
- Exact evaluator: MAIN truth.py:77-95 rollout (float64, deterministic), 330-331 bank_rollout, 339-349 anchors,
  352-354 points = 100 (J - J_rec)/(J_top - J_rec). Anchors frozen in MAIN outputs/shift/preregistration.json
  anchors 'base/stall' (top = stall-aware controller) and 'base/no_stall' (top = rising-thrust controller).
  learners.evaluate_policy (MAIN learners.py:925-942) does both worlds (+ capped twin, past-80% share).
- Unifloral adapter: HOSTS experiments/shift_hosts/policies.py:79-92 MLPAdapter (truth.MLPPolicy with obs_std + 1e-3,
  nan_to_num). HOSTS evaluate.py:302-353 score() is stalling-world only in practice: PHYSICS = TC.STALL (88), anchors
  'base/stall' (100-102, 179-203), pinned stall-world scoring cache for regret/F1/F2 (103-123, 245-263). No-stall
  scoring must call truth.rollout / learners.evaluate_policy with 'base/no_stall' anchors (new thin wrapper).
- Fallback actions: recorded range = per-thruster 0.5%/99.5% quantiles of the gapped training commands
  (MAIN .../step1/evaluate.py:162-163; .../scripts/fallback/fallback.py:8, 49, 209-226 clip99); knee clip / capped
  controller MAIN recorder.py:174-191; range limit / matched global ceiling CONN
  outputs/shift_iw/explore/arms_abc_f0/code/arm_b.py:142-155.

## 3. Online episodes (deployment simulation)
- API: MAIN thruster_chain.py:389-459 ThrusterChainEnv; reset(seed, options={'state'| 'start'}) 416-436; step casts the
  command to float32 then float64 and clips (443-446); truncation only at t = 100 (454); 8 registered ids 462-481
  (env_id(world, situation)). Dynamics are deterministic (13-22): the only randomness of an online episode is its start
  draw (and any declared policy noise).
- Fresh online starts: TC.draw_starts(n, default_rng(<new tags>), START_LAWS[s]) or env.reset(seed=...) (default draw
  from self.np_random). Must not reuse the bank stream [20261012, 2, 9] or data streams [20261012, rep, 7/8/10].
- Bit-for-bit caveat: truth.rollout does NOT round the command to float32 (truth.py:85) while env.step does
  (thruster_chain.py:443) and dataset.simulate does (dataset.py:169). An online-episode runner should round commands to
  float32 (or call env.step) and say which; dataset.replay (321-347) is the existing env.step equality check.
- Existing online-style code: step 1's gated rollout (MAIN .../step1/evaluate.py:134-156, 'never'/'on'/'random' at
  the check's rate) and the run-time monitor's episode-by-episode looks with a latch (MAIN .../routes/round1/scratch/
  runtime_monitor/prep.py:87-119, looks.py:33-38 compose). 200 starts per situation was the convention.

## 4. Exact truth for coverage
- One-step targets at any (s, a): s', r = TC.dynamics(s, a, physics) exactly (thruster_chain.py:201-221); the host's
  own target y = r + 0.98 min Q_target(s', clip(pi_target(s') + eps)) by HOSTS unifloral_td3_bc_bca.py:316-349
  (_own_target / _band_values, verbatim Unifloral lines). band_parts (352-359) is NOT permission-gated; it returns
  q = min(Q1, Q2) — option C needs a Q1 variant (new).
- Q^pi(s, a) by float64 rollout: MAIN truth.py:98-108 q_value (horizon 1,000), 171-189 q_value_jax (pass
  adapter.mlp, which has .jax; MLPAdapter itself does not), learners.exact_q (learners.py:413) for batches.
- Coverage arithmetic: MAIN experiments/shift_iw/coverage.py:100-106 width, 251-316 refusals/outcomes/summaries,
  153-208 per-test-mass threshold table; HOSTS band_readout.py:1-60 (actor/visited/recorded rows on Unifloral critics,
  the scoring side supplies exact next states by file: evaluate.py:443-512).

## 5. Unifloral TD3+BC host and reuse without the fingerprint gate
- Import machinery (no study gate inside): HOSTS unifloral_import.py:215-285 prepare (Dockerfile variables, SHA-256
  of baselines/unifloral/algorithms/td3_bc.py against configs/unifloral.json pin f2dc127 (57), import under a private
  module name, XLA flag repair, jax_default_matmul_precision 'highest'); 353-356 args_for; 359-377 make_transition;
  380-454 init_like_training_block; 457-461 make_unifloral_step (Unifloral's own step, unchanged).
- Gates: HOSTS gate.py:1-34 require_run_approval (fingerprint = prespec.json + code manifest of the whole
  shift_hosts package and EXTERNAL_CODE 56-65: calibration/*.py, experiments/shift/*.py, experiments/shift_iw/*.py,
  experiments/truth, experiments/signal, Unifloral files); guard.py:127-136 require_bca; runner.py:1192 (command_run)
  and execute 1074; unifloral_td3_bc_bca.py:104/230-255 (BCA step), 392 (refresh), 408 (recompute); evaluate.py:367-390
  (actor files). Ungated: bca_core.py (pure arithmetic, 5-7; fit_scale_core 282, calibrate_bank 399, apply_refresh
  417), band_parts, policies.py, actors_f32.py (write/read actor files 135-180), handoff.py (two-env files),
  evaluate.score on a policy object.
- Reuse path for a new study: a NEW package (e.g. experiments/placement_test/, outside EXTERNAL_CODE so the hosts'
  approval fingerprint is untouched) that imports unifloral_import + bca_core + band_parts + policies/actors_f32/
  handoff read-only and carries its own prespec, code manifest and owner-approval gate. Do not add files under
  experiments/shift, experiments/shift_iw or calibration (they are in the hosts' fingerprint), and do not reuse the
  hosts' Approval for a different study. Two-environment split stays (training env numpy<2 / scoring env numpy 2.2.6).

## 6. Existing decision code to reuse
- (A) harm switch: step1lib.py:216-221 overestimate (score s = min_h Q(s,a) - [r + 0.98 min_h Q(s', c(s'))]),
  244-276 certificate (two WBCP posteriors, 5% quantile of the pairwise gap > margin 0.10; plain gap, e11 form),
  279-287 Mann-Whitney, 290-305 outcome shuffle (common random numbers), 310-322 deployment_net, 339-352 go_no_go.
  Plain split conformal / mean / median gaps: MAIN .../routes/round1/scratch/heldout_calibration_rows/verify/
  verify.py; step 2's conformal_k1 and meangap_k1 switches: MAIN .../step2/step2lib.py:315-367 arm_table.
  Note: step 1's score uses the stall-aware FIXED controller's critic (learners.fixed_critic), not a TD3+BC host; with
  a Unifloral host the next command and heads must be redefined (new).
- (B) per-deviation certificate: CONN .../arms_abc_f0/code/arm_b.py:8-20 (tries, rungs, gain g, walk), 76-92
  bca_decide (wbcp at alpha 0.5, threshold < 0), 95-128 plain_decide (weighted median, weighted mean, exact sign test,
  weighted t-test, weighted empirical Bernstein), 131-139 walk, 142-155 matched global / range limit. Mean gate
  (Gaffke-type, Bayesian-bootstrap weights + floor atom), tail gate (exact binomial allowed_below), Hoeffding,
  empirical Bernstein, t-test, split-conformal tail: MAIN bca_mean_gate_toy.ipynb cells 7, 9, 12-13 and
  bca_weights_toy.ipynb cells 7, 9, 13 (notebooks only; must be lifted into a module). Clopper-Pearson: bca_weights_toy
  cell 9 cp_upper. Rejection-sampled exact test, rejection-sampled BCA mean, weighted split conformal, stratified
  exact: bca_weights_toy cell 13 (TAIL_GATES / LEVEL_GATES). No HCOPE-style (Thomas 2015) mean bound found in any
  .py/.ipynb on MAIN, CONN or HOSTS (new; empirical Bernstein exists in arm_b.py:123-127 and the toys).
- (C) lower edge: MAIN experiments/shift_iw/lower_edge.py:1-64 (objective, paired form, refusal handling), 196-215
  band_width / band_penalty / capped_thresholds, 293-323 actor_step/_loop; Stage 2 v6 MAIN .../stage2/code/band.py
  (176 paired, 181 lower_edge_objective, 465 WidthReader), policy.py:68-104 actor_step_bca; command-weighted band
  CONN outputs/shift_iw/explore/channel/command_band_check.py:6-10, 52-86 (exact recorder, spread and ratio densities
  for the WEIGHTING-TEST HoldRecorder only), command_le_train.py:4-12, 73-123 (cmd_strong, cmd_train, untargeted);
  F0 control CONN .../arms_abc_f0/code/arm_f0.py. All of these run on experiments/shift learners (actor-only against
  a frozen critic), not on Unifloral; the main log's recorder (full-throttle command with a point mass at -1 on
  thruster 1, recorder.py:8-17) has no density code (new).
- Step 2 ramp and matching: MAIN .../step2/step2lib.py:383-404 ramp/outcome_ramp/ramp_penalty, 419 net, 434 go_no_go;
  machinery.py:155 RampRunner; v6 policy.match_controls (stage2/code/policy.py:179).
- WBCP: MAIN calibration/wbcp.py:43-54 crossings, 57-117 calibrate (weights, test mass, alpha, beta, draws).

## 7. Seeds
- Main generator (master 20261012): 0-4 seen (gate); 5-9 version 7 / retune gate; 10-14 version 8 / near80 claim 4;
  15-19 named as near80's fallback confirmation; 100-119 old confirmation (data records in preregistration.json: stall 100-119, no-stall 100-104; retune 5-9);
  900-909 calibration. (outputs/shift/preregistration.json seeds; step2 PRESPEC Words; CONN near80/prereg.json seeds,
  choices #6.) Free-looking: 20-24 or higher, owner to allocate.
- Weighting-test generator (master 20261016): 0-4 seen; 10-14 reserved for the four-learner study (HOSTS PRESPEC.md
  617-621); 15-19 named for arms A/B/C confirmation (CONN arms_abc_f0/README.md:3-4).

## 8. Compute (micro-benchmarks on made-up arrays, bench/)
- Unifloral TD3+BC step restated (2 critic updates + 1 actor update per outer step, batch 256, 2x256 MLPs), scanned in
  2,500-step chunks, JAX 0.6.2 CPU, made-up 42,500 x 13 arrays:
  - 1 process, 1 thread (unifloral_import's CPU flag): 13.1 ms/outer step -> 21.9 min per 100,000 steps.
  - 1 process, 4 threads: 8.9 ms -> 14.9 min (only 1.47x).
  - 4 concurrent 1-thread processes: 13.8-14.5 ms each -> about 23.5 min per run, about 10 runs/hour on this machine.
  - + BCA-like per-step fit (own target, min-head q, 17-64-64-1 scale-net Adam step, dose read): 16.3-16.8 ms
    (+18%) -> about 27.5 min per run. Refreshes (18 per run) not included (seconds each).
  - + value lower edge in the actor loss (eta forward at pi(s)+xi): 14.0-14.1 ms (no measurable extra).
  - Caveat: Unifloral's real env is Python 3.10 / jax<0.6; first chunk compile 24-50 s.
- Laptop GPU (HOSTS PRESPEC.md:601-603): 0.58 ms alone, 1.26 ms with BCA -> 0.97 / 2.1 min per 100k: about 23x / 13x
  faster than one CPU thread here; about 6x / 3.3x this whole 4-CPU container's throughput.
- Old CORL-style learner for scale: 8 min per 100k on the cloud (MAIN experiments/shift/SHIFT_ENV.md:404-409),
  11.7 measured (preregistration.json budget.measured.plain_host) — Unifloral does twice the updates per step.
- Exact evaluator arithmetic (made-up matrices, same per-step work): 1,000 starts x 100 steps = 0.92-1.01 s; with
  the capped twin about 1.7-1.9 s per policy per bank (consistent with HOSTS PRESPEC.md 13: 59.6 s for 5 readouts x 7
  banks); 200 starts 0.12 s; one online episode at batch 1 = 11-17 ms; Q^pi by 1,000-step rollout for 1,000 (s, a)
  pairs about 9 s (1 thread, numpy); exact one-step targets for 1,000 rows < 1 ms.

## 9. What must be newly written
1. A main-log Unifloral prepare (gapped log, both worlds; bank; held-out calibration/evaluation rows; next commands)
   and a two-env hand-off for it (HOSTS prepare.py is weighting-test only).
2. The study package with its own prespec, frozen pass marks, code manifest and owner gate (gate.py pattern).
3. Host critic score for (A) on a Unifloral TD3+BC critic (next command = target actor or actor; heads), Q1-only band
   parts and a Q1 band fit/refresh for (C) (from bca_core + band_parts), exact state x command weights for the MAIN
   recorder (point mass at full throttle) and the smoothing law.
4. Restated TD3+BC actor step with the lower edge (Q1 - k U_w, refusal = no value credit) and matched controls (range
   hinge, even pull, push-less, random surfaces) on Unifloral — existing ones are actor-only on the old learner.
5. (B) deviation definition on a Unifloral host, a staged online-deployment loop (fresh start stream, episodes per
   stage, tries per decision >= 29, Learn-then-Test / multiplicity), the mean/tail/rejection-sampled gates lifted
   from the toy notebooks into a tested module, and an HCOPE-style mean bound.
6. Online episode runner with float32 command rounding, per-start mixed physics (wave cut) and a no-stall scorer with
   'base/no_stall' anchors; a spliced mixed-world training log + its gap/bank if the mixed world is trained on.
7. A no-stall twin of the weighting-test generator if (B)/(C) need cross-world contrast there.
