# BCA code and calibration audit (Unifloral host), 27 October 2026

Scope: the original direct-Q BCA as implemented on the pinned Unifloral standalone TD3+BC, read from
`experiments/thrusterchain_bca/titan_20261026/payload` and `experiments/thrusterchain_bca/zeus_20261026_validation`
in checkout `adf51269` of branch `codex/thrusterchain-bca-handoff-20261026`. Every line reference below was read in
this checkout. Nothing was run, installed or changed. The repository's CORL-derived `algorithms/td3_bc.py` is not part
of this audit and is never imported by the code audited here (the loader refuses `algorithms.*` and `runtime.*`
imports; `bca_bridge.py:175-176`).

Paths are abbreviated: `T/` = `experiments/thrusterchain_bca/titan_20261026/payload/`,
`Z/` = `experiments/thrusterchain_bca/zeus_20261026_validation/`,
`U/` = `T/unifloral_bca_pipeline_validation_v1/uncertainty_source/`.

## 1. Host identity and what BCA changes

| Item | Value | Where |
|---|---|---|
| Host | Unifloral standalone TD3+BC, repository revision `f2dc1278eae18ed3e22c92255119c54885414d17` | `T/unifloral_td3bc_validation_v1/native_loader.py:13` |
| Host source hash | `td3_bc.py` sha256 `6f5ab6f98d706d902a18f11e1c81e99e2691dc4c8a53843bf6521c9f95d00eee` (recomputed in this audit, matches) | `native_loader.py:15`; `Z/MODEL_IDENTITY.md` |
| How the host is loaded | hash-checked AST slice of eight definitions (`Args`, `AgentTrainState`, `Transition`, `SoftQNetwork`, `DualQNetwork`, `DeterministicTanhActor`, `create_train_state`, `make_train_step`); CLI, D4RL, Gym, wandb excluded | `native_loader.py:16-19, 49-94` |
| Terminal checkpoint | `state.msgpack` sha256 `874d65b7...2948e8`; 20,000 critic, 10,000 actor, 20,000 auxiliary updates; last refresh at critic 20,000; `kappa_penalty = 0.0`; gamma 0.98; batch 256; seeds 2026102302 (native) and 2026102361 (auxiliary) | `Z/MODEL_IDENTITY.md`; `T/unifloral_checkpoint_v1/training.py:97-107` |
| Observations | 14 float32 coordinates (13 physical + remaining fraction); normalization inside every network: `(obs - mean) / (std + 1e-3)` with population mean/std of the training observations | `td3_bc.py:71, 105`; `native_loader.py:106-107` |

**Networks** (`T/unifloral_td3bc_validation_v1/upstream/algorithms/td3_bc.py`):

- Critic head (`SoftQNetwork`, lines 65-77): concat(normalized obs, action) -> Dense 256, ReLU -> Dense 256, ReLU -> Dense 1.
- Two heads (`DualQNetwork`, lines 80-95): `nn.vmap` over `axis_size=2` with unshared parameters and separate
  initializations; output shape `(..., 2)`. **These two heads are the only "critic ensemble".**
- Actor (`DeterministicTanhActor`, lines 98-114): normalized obs -> Dense 256, ReLU -> Dense 256, ReLU -> Dense 4 ->
  deterministic value passed through `tanh`, so commands lie in [-1, 1].
- Optimizer: Adam, learning rate 3e-4, `eps=1e-5` (lines 38, 117-122). Polyak step 0.005 (line 41).

**One native outer step** (`td3_bc.py:174-279`), hyperparameters `td3_alpha=2.5`, `policy_noise=0.2`,
`noise_clip=0.5`, `num_critic_updates_per_step=2` (lines 43-46):

```
1. Sample one minibatch B of 256 transitions (lines 181-185).
2. Repeat twice on the SAME minibatch (lax.scan, length num_critic_updates_per_step = 2; lines 226-231):
   a. for each row, a' = clip( pi_target(s') + clip(N(0, 0.2^2), -0.5, 0.5), -1, 1 )          (lines 195-201)
   b. y = r + gamma * (1 - done) * min( Q1_target(s', a'), Q2_target(s', a') )                 (lines 204-208)
   c. critic loss = mean over batch AND over the two heads of ( Q_h(s, a_logged) - y )^2         (lines 216-218)
   d. one Adam step on both heads                                                              (lines 220-222)
3. Actor, once per outer step (lines 234-254):
   a. a_pi = pi(s);  q = Q1(s, a_pi)   (head index 0 only, line 240)
   b. lambda = 2.5 / ( mean_B |q| + 1e-7 ), stop_gradient                                        (lines 243-244)
   c. BC = mean over the 4 action coordinates of ( a_pi - a_logged )^2                            (line 239)
   d. actor loss = - lambda * mean_B q  +  mean_B BC                                             (line 245)
4. Polyak update of actor target and critic targets with step 0.005 (lines 257-268).
```

So the Unifloral host makes **two critic updates per actor update**, and the actor sees head 1 only. The target
uses the twin minimum; the actor objective does not. The ThrusterChain finite horizon enters only through `done`
(1 at the 100th transition, so no bootstrap past the horizon; `core_v4/docs/environment.md`).

**What BCA adds, and only this** (`T/unifloral_td3bc_validation_v1/candidate.py:97-125`, used by both pilots through
`make_microsteps(..., width_fn, kappa_penalty)`):

```
actor loss_BCA = [ - lambda * mean_B Q1(s, a_pi) + mean_B BC ]  +  lambda * kappa * mean_B U(s, a_pi)
               = - lambda * mean_B ( Q1(s, a_pi) - kappa * U(s, a_pi) ) + mean_B BC
```

- `lambda` is computed from the **unpenalized** Q1 and detached (lines 105-106), exactly as the host does.
- The reference (scale-network parameters, residual unit, threshold) is wrapped in `stop_gradient` before the
  width is evaluated (line 108); the live action `a_pi` is **not** detached, so `dU/da` reaches the actor.
- With `kappa_penalty = 0` the native expression is used verbatim and `width_fn` is never called (lines 75-95).
- The critic update in the same file (lines 127-154) is a line-for-line copy of the host's: **no width enters the
  TD target or the critic loss**. The BC term is unchanged. Nothing is added to the critic, the targets or the data.

Receipts: pipeline check 7 shows loss error 0 and actor-gradient error 0 between the kappa-0 path and the native
actor loss; checks 8-14 show the loss and gradient identity `grad(native) + lambda*kappa*grad(mean U)` for kappa in
{0, 0.03, 0.1, 0.3, 1, 3, 10} within rtol 2e-5 / atol 2e-6; check 15 shows the actor gradient through the frozen
reference is exactly 0 while the same gradient with the freeze removed would be 1.76, so the stop-gradient is doing
real work (`T/unifloral_bca_pipeline_validation_v1/results_20261023T171638_093548Z/pipeline_checks_summary.json`).

Kappa by experiment: 0 for the 20k checkpoint; **3** in the matched-policy pilot (`Z/matched_policy_pilot_v1/continuation.py:15`,
`train_worker.py:27`) and in the support-shift pilot (`Z/support_shift_pilot_v1/runtime.py:199`); 1 in the one-sided
auxiliary pilot's frozen ranking (`Z/bca_one_sided_v1/run_analysis/RESULTS.md`). Kappa is a strength, not a count.

## 2. The residual the auxiliary model learns (and what it is not)

`U/historical/td3_bc_bca.py` is the pinned source of `native_target`, `fit_scale` and `refresh`; the bridge compiles
only those definitions and `live_width` from `direct_extension.py`, by AST hash (`bca_bridge.py:21-22, 117-131`).

```
y  = stop_gradient( r + (1 - done) * gamma * min_h Q_h,target( s', clip(pi_target(s') + clipped noise) ) )   (td3_bc_bca.py:146-156)
q  = stop_gradient( min_h Q_h( s, a_logged ) )      -- CURRENT critic, twin minimum, LOGGED action             (170-174)
residual = y - q
```

- One fresh smoothing draw per row per call (`fold_in(rng, 1179210836)`, line 167). Terminal mask `(1 - done)`.
- The fit's coverage event is two-sided (`soft_coverage`, `U/calibration/network.py:34-37`) and the calibration
  scores use `|y - q|` (`reference.py:175`). **The band is a band on the absolute one-step twin-min TD residual at
  logged actions.** It is not the error of Q1 against the simulator's own-policy return, it carries no sign, and it is
  read at logged actions while the penalty is applied at the actor's action. Pipeline check 17 makes the gap concrete
  with a fixture (Q1 = 10, Q2 = 0, reward 0, target 0): the twin-min residual is 0 and the calibrated width is 0, while
  Q1 is off by 10.
- Consequence for the interface: reducing Q1 by kappa*U is a correction of magnitude only. Whether it lands on
  overestimated actions is an empirical question answered by the simulator-reference comparisons, not by the fit.

## 3. Scale network, residual unit and calibration

**Local scale network** `eta(s, a)` (`U/calibration/network.py:8-31`): obs normalized with the native mean/std and
`+0.001` (the bridge passes the actor's `obs_mean`/`obs_std`, `bca_bridge.py:249, 270`), concatenated with the action,
two Dense-64 ReLU layers, Dense 1, `eta = softplus(log_eta) + 0.01`. So `eta > 0.01` always; the public floor
`max(eta, 1e-6)` (`reference.py:148-149`, `direct_extension.py:60`) never binds.

**Residual unit** `u`: initialized at 1.0 (`bca_bridge.py:276`); after every accepted fit
`u <- 0.99 u + 0.01 * max( std_B(y - q), 1e-6 )` (`td3_bc_bca.py:196-198`).

**One auxiliary fit** (`td3_bc_bca.py:159-231`), Adam with learning rate 0.001 (`Config.cal_lr`, line 39; `bca_bridge.py:275`):

```
1. y, q as in section 2 on the current minibatch (same 256 rows the critic just used).
2. p = normalized Bayesian-bootstrap weights: E_i ~ Exp(1), p_i = E_i / sum_j E_j   (network.py:44-46; one draw per fit)
3. coverage_i = sigmoid(20 * (y_i/u - (q_i/u - eta_i))) * sigmoid(20 * ((q_i/u + eta_i) - y_i/u))   (network.py:34-37; beta = 20)
4. loss = ( sum_i p_i * coverage_i - 0.9 )^2 + 0.005 * sum_i p_i * eta_i^2        (lines 180-191; alpha = 0.1, width_penalty = 0.005)
5. one Adam step on the scale network; then the unit update above
6. accept only if every input, the loss, the gradient, eta > 0, p >= 0 summing to 1 and u > 0 are finite (199-227); otherwise keep the old state
```

In the integrated trainer the fit runs **after** the critic update of the same call and on the same minibatch
(`T/unifloral_checkpoint_v1/training.py`, `step`: microstep -> `fit_step(auxiliary, micro.batch, ...)`), one fit per
critic update, so 20,000 fits for the 20k checkpoint. (In the older historical extension the fit preceded the critic
step; that ordering is not what the Unifloral integration does.)

**Calibration refresh** (`td3_bc_bca.py:234-280` -> `reference.py:152-194` -> `wbcp.py:57-117`), at critic updates
10,000, 15,000, 20,000 for the checkpoint (`training.py:97-107`) and at 21,000 in the continuation
(`continuation.py:247-249`); on the **held-out bank** of 128 rows, one row from each of 128 held-out episodes that
never train (`training.py`, `create`: `heldout_transition` of 128 rows with 128 distinct episode ids):

```
1. y_j (fresh smoothing draw), q_j = twin-min current critic at the logged action, eta_j from the current scale net, for the 128 held-out rows
2. freeze cal_params and u                                                    (reference.py:185-188)
3. scores_j = | y_j - q_j | / ( max(eta_j, 1e-6) * u )  in float64             (reference.py:164, 175)
4. wbcp.calibrate(scores, unit weights, test_mass 1, alpha 0.1, beta 0.95, draws 1000)
   (reference.py:176-179; WBCPConfig defaults reference.py:17-20; caller bca_bridge.py:262 uses the defaults;
    the pilots record alpha .1, credibility .95, draws 1000 in train_worker.py:27)
   a. lambda_hat = smallest sorted score whose cumulative weight reaches 0.9 of the total (exact rationals)   (wbcp.py:89-102)
   b. for m = 1..1000: draw E_1..E_128 and one extra E_129 for the unseen test point, all Exp(1);
      crossing_m = smallest score at which cumulative mass >= 0.9 * (total mass including the test atom), else +inf   (wbcp.py:43-54, 104-111)
   c. lambda_hpd = the 950th smallest crossing  (ceil(0.95 * 1000))                                               (wbcp.py:115)
   d. R = max(lambda_hat, lambda_hpd);  n_eff = 128 for unit weights                                              (wbcp.py:116-117)
5. store R, lambda_hat, lambda_hpd, n_eff, u, cal_params as the FrozenReference with ready = True;
   a certified value that does not fit float32 is rejected   (reference.py:180-188)
```

Meanings: `alpha = 0.1` is the target miscoverage (and the fit's coverage target 0.9); `beta = 0.95` is the posterior
credibility of the selected threshold; `draws = 1000` are **Bayesian-bootstrap posterior draws over the
threshold**, each a reweighting of the 128 held-out scores plus one test atom. They are not critic heads and not
model samples. With unit weights this is BQ-CP (the exchangeable case); importance weights are accepted by
`wbcp.calibrate` but **no caller in the current work passes any** (reference.py:176-179 passes none), so the
"importance-weighted BCA" remains a proposal here.

**Refusal.** A draw whose test atom alone exceeds alpha never crosses, so if fewer than 950 draws cross, `lambda_hpd`
and `R` are +inf; `certifiable(n)` shows a uniform bank needs n >= 36 at these settings (reference.py:124-136), so 128
rows always qualify in principle, but a refresh is still refused if any input is nonfinite (reference.py:166-174). The
integrated trainers refuse to commit a refresh that is not ready and finite (`training.py`: "Refresh was rejected,
uncertified, or ready-infinite; checkpoint commit refused"; `continuation.py:242-246`).

Receipt: pipeline check 4 recorded threshold 4.874597, lambda_hat 4.534386, lambda_hpd 4.874597, sigma_post 0.2940,
n_eff 128, clamp_binds false, 128 scores, on the test snapshot. That is a software receipt on a fresh snapshot with a
manual early refresh, not the trained checkpoint's threshold.

## 4. Readiness, freezing, update order, resume

- Before the first refresh the reference is "not ready": thresholds +inf, u = 1, `ready = False`
  (`reference.py:139-145`). `live_width` then returns zero width with `usable = False` (`direct_extension.py:62-67`).
  In the continuation adapter an unusable or invalid width is turned into NaN so the step is **refused**, never
  silently trained with zero penalty (`continuation.py:72-77`). The kappa-0 checkpoint never stored a reference in the
  actor state at all (`training.py`, `validate`: "Kappa-zero checkpoint must keep auxiliary reference out of native
  actor state").
- Order of one committed call (`training.py step`; `continuation.py:204-259`): native microstep (critic update; on
  the second critic of a pair also actor and targets) -> auxiliary fit on the same batch with the post-update critic ->
  refresh if due -> validate the whole proposed state -> commit everything or nothing. The actor therefore consumes the
  reference frozen at the last refresh and the scale parameters frozen inside it, not the live scale network;
  "the boundary actor already ran with the prior reference; the next actor sees the newly frozen posterior"
  (`continuation.py:247-248`).
- Checkpoint and resume: the full `TrainingState` (native and target parameters, both optimizers, scale network and
  its optimizer, unit, posterior, both RNG keys, the held minibatch, phase and counters) must round-trip byte-exactly
  (pipeline check 5; `Z/titan_continuation_20261027/PROPOSAL.md`: 227 leaves). Counters are validated against
  optimizer steps on every call (`training.py validate`; `continuation.py:171-201`).
- Parity and tolerances: the kappa-0 path is bit-identical to the host (check 7, errors 0.0); gradient routing checks
  used rtol 2e-5 / atol 2e-6 with an explicit 1e-8 non-vacuity floor (`pipeline_checks_summary.json: tolerances`); the
  Unifloral parity validation has its own receipt (`T/unifloral_td3bc_validation_v1/results_20261023T162222_095882Z/parity.json`).
  No tolerance was loosened in this audit and none was found loosened in the receipts.

## 5. Draws versus heads; calibration versus guarantees

| Quantity | Count | Source |
|---|---|---|
| Critic heads | 2 | `td3_bc.py:92` |
| Posterior draws per refresh | 1,000 | `reference.py:20`, `wbcp.py:57` |
| Held-out calibration rows | 128 (one per held-out episode) | `training.py create` |
| Bootstrap weights per auxiliary fit | one Exp(1) draw of length 256 | `td3_bc_bca.py:175-177` |

The calibrated statement is: with probability 0.95 under the Bayesian-bootstrap posterior, at least 90% of
exchangeable held-out one-step twin-min residuals at logged actions fall within `R * u * eta`. It is **not** a
statement about (a) Q1's error against the simulator reference, (b) the actor's own actions, which are not logged
actions and are not exchangeable with the bank, (c) states outside the recorded start law, or (d) policy return. The
head-count sweep is deferred and has no local receipt.

## 6. Scales, offsets, floors

- The penalty is added inside the same `lambda`-scaled bracket as Q1, so its weight relative to Q1 is exactly kappa;
  pipeline check 18 shows that scaling Q and width together by 0.1 keeps the penalty ratio at 0.9999978 (the only
  deviation is the native `1e-7` in the denominator).
- At the final actor minibatch of the matched-policy pilot the scalar penalty terms were about 0.552 / 0.525 for BCA
  (kappa 3) and 0.000452 / 0.000407 for the distance control (`Z/matched_policy_pilot_v1/run_analysis/RESULTS.md`).
  These are loss levels, not gradient norms; the support-shift protocol therefore matches **action-gradient RMS** on
  logged probes instead (`Z/support_shift_pilot_v1/design.py:82-98`, `PROTOCOL.md`).
- A width that is constant across actions at a state has zero action gradient and cannot move the actor, and cannot
  change a candidate ranking. Nothing in the current code asserts scale invariance beyond check 18, and no universal
  kappa is justified by evidence here.
- Floors: `eta > 0.01` by construction; `u >= 1e-6` through the EMA floor; `mean|Q1| + 1e-7` in lambda; widths must be
  finite and positive when `R > 0` (`direct_extension.py:68-69`).

## 7. Claim-to-evidence table

| Claim | Status | Evidence |
|---|---|---|
| The host is the pinned Unifloral TD3+BC, not CORL | validated | hash 6f5ab6f9..., AST slice loader, `MODEL_IDENTITY.md`, parity receipt |
| BCA changes only the actor objective; critic loss and TD target untouched | validated by source and checks 7-14 | `candidate.py:97-154`, `pipeline_checks_summary.json` |
| Action gradient through `eta` is live; reference parameters frozen | validated | check 15 (0 vs 1.76), `direct_extension.py:54-60`, `candidate.py:108` |
| kappa = 0 reproduces the host bit for bit | validated | check 7 |
| Auxiliary fit and refresh are deterministic given the RNG stream and resume exactly | validated | check 5; pilot restore receipts |
| Threshold is a 95%-credible BQ-CP quantile from 1,000 posterior draws on 128 held-out rows | validated by source | `wbcp.py`, `reference.py`, check 4 |
| The band covers the absolute twin-min one-step residual on held-out logged rows | this is what it is defined to do; coverage on the trained checkpoint's bank is not reported locally | section 2; check 17 shows it is not Q1 error |
| The band tracks Q1's true error at the actor's actions | **not established** | no local receipt; the brief's Titan-only held-out evaluation (unavailable here) reports mixed decision outcomes |
| The penalty improves action choices against raw Q1 and simple controls | **not established** | operator analysis and one-sided pilot are descriptive/negative; held-out evaluation unavailable here |
| The penalty improves policy return | exploratory positive in one setting only | matched-policy pilot: +2.59 / +2.51 return units over native continuation in two streams from ONE checkpoint, 32 shared starts; not independent seeds, not OOD |
| Independent-initialization replication under a support gap | **not completed locally** | support-shift attempt 1 interrupted, attempt 2 stopped after two seed-0 warmups; Titan continuation proposed |
| One-sided variant | screen failed | `Z/bca_one_sided_v1/run_analysis/RESULTS.md` |
| Adaptive signed signal, importance-weighted BCA, head-count sweep | proposals, not implemented in the current host | `SIGNAL_REDESIGN_OPTIONS.md`; no caller passes weights |

## Open items this audit could not close

1. The held-out evaluation, adaptive-signal recovery, explicit OOD diagnostic and timeout-recovery folders named in the
   brief are not present on this machine; nothing about them is verified here.
2. The trained checkpoint's own calibration diagnostics (threshold, lambda_hat, lambda_hpd at critic 20,000) are inside
   the runtime assets, not in a report read here; the interface must show them as unavailable until extracted.
3. Coverage of the frozen band on the held-out bank after training is not reported in any local report.
