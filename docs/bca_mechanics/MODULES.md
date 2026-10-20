# BCA's components, in the order data flows through them

**What this is.** A map of BCA (Bayesian Conformal Aggregation), the project's method for putting an error band around an
offline critic and turning that band into caution for the actor. The map has eleven modules, M1 to M11, in the order
data flows through them. Each module says what it computes, what goes in and out, the formula, the constants, where the
code is, what "working as intended" means, the pseudocode with each line labeled, every variant, what is recorded, and a
verdict.

**Written on** 20 October 2026 from the code at commit 52914e4 of branch `claude/main-bca-connection-x4o46s` (paths below
are relative to the repository root), plus branch `claude/shift-hosts` (the four-learner study) and branch `main` (the toy
notebooks and the weighting test's records). Nothing was trained for this document. Every number is copied from a record
in the repository, or printed by one of the two scripts in `docs/bca_mechanics/scripts/`; the record or script is named
next to the number.

**The four verdict words.** works / works with a caveat / does not work / untested. After the verdict comes the one
sentence to say to the advisor.

**Three labels for the evidence.** *Pre-registered*: the rule was frozen and approved before the run, and the result was
judged by that rule. *Exploratory*: designed after seeing seeds 0-4, or run outside a pre-registration. *Description
only*: what the code does; no claim about a result.

---

## Words used

- **Recorded row**: one step of a recorded log: the state s, the command a, the reward r and the next state s'. In
  ThrusterChain (the project's simulated boat) the state is 13 numbers and the command is 4 thruster throttles in
  [-1, 1]. The water current c is the 11th number of the state (index 10).
- **Training rows, held-out calibration rows, evaluation rows**: the three parts of a log, split by episode. The
  held-out calibration rows (the code calls them "the bank") are 5 rows from each of 200 held-out episodes, 1,000 rows
  (`experiments/shift/dataset.py:13-14`, drawn by `calibration/bank.py:45-102`). After a removal about 800-845 of them
  remain. In the D4RL hosts the held-out rows are reserved from whole episodes, K rows per episode
  (`calibration/reference.py:49-121`; K in `configs/td3_bc.yaml`).
- **A removal**: rows hidden from training and from the held-out calibration rows. *The ball*: the main test's
  removal, the densest 15% of recorded commands. *The band* (near-80): also every command with a throttle from 70% to
  90%. *Survivors*: full-throttle commands outside the ball that stay in training.
- **The critic pair** Q1, Q2 and their **target copies** Q̄1, Q̄2: two value networks (256-256 ReLU,
  `algorithms/td3_bc.py:62-88`) trained by TD3+BC; the target copies trail them (Polyak factor 0.005). The **critic's
  value** is q = min(Q1, Q2).
- **The actor** π and its target copy π̄: the controller being trained.
- **The target** y: the one-step value the host trains its critic toward (reward plus the discounted target critic's
  value at the next step).
- **The residual**: y − q, in critic value units.
- **The error-size network** η(s, a): a small network that predicts, per row, how large |y − q| should be, as a
  multiple of the unit.
- **The unit** u: a running scale of the residual (an exponential moving average of the standard deviation of y − q
  over training batches), in critic value units.
- **A score** ρ: |y − q| divided by u·η, a dimensionless size of one residual.
- **A weight** w_j: how much a held-out row counts, the ratio of how often its situation occurs at test time to how
  often it occurred in training. **The test mass** t: the weight of the row being certified.
- **The threshold** R: the score below which 90% of scores fall, with 95% credibility. **Refusal**: R = +infinity,
  "cannot certify".
- **The band** q ± U and **the width** U = R·u·η(s, a) (a half-width).
- **The dose** m: the multiplier BCA puts on TD3+BC's **behavior-cloning term**, the part of the actor loss that pulls
  the actor's command toward the recorded one.
- **A refresh**: the moment BCA freezes η and u and recomputes R on the held-out rows.
- **Worlds**: the *stalling world* (a thruster's push falls past 80% throttle) and the *no-stall world*.
  **Situations**: four fixed sets of 1,000 test starts: recorded conditions, strong current, big waves, big kicks.
  **Points**: 0 is the recording controller, 100 the world's best scripted controller.
- **Seeds**: data seeds 0-4 (seen; exploratory) and 10-14 (fresh; confirmatory).
- **The weighting test**: `experiments/shift_iw`, specification version 5, five data qualities (random, medium-replay,
  medium, medium-expert, expert), five seeds each; records in `outputs/shift_iw/v5/` on branch `main`.
- **The four-learner study**: `experiments/shift_hosts` on branch `claude/shift-hosts` (TD3+BC, CQL, IQL, ReBRAC with
  standard BCA and with weighted BCA); record `outputs/shift_hosts_v3/results.json` on that branch.
- **The restructured BCA**: step 1 (the harm switch) and step 2 (the ramp), `outputs/shift/beyond-the-data/restructure/`;
  the near-80 run (version 8), `outputs/shift/beyond-the-data/near80/`, results folder `results-20261020T115246Z`.

---

## The map

| # | Module | What it computes, in one line | Code |
|---|---|---|---|
| M1 | Residual | the critic's one-step error y − q at a recorded row | `algorithms/td3_bc_bca.py:146-156` |
| M2 | Error-size network and unit | η(s, a) > 0 and the running unit u, fitted every update | `algorithms/td3_bc_bca.py:159-231`, `calibration/network.py:8-46` |
| M3 | Scores | ρ_j = \|y_j − q_j\| / (u_f · max(η_f, 1e-6)) on the held-out rows, with η and u frozen | `calibration/reference.py:148-175` |
| M4 | Weights | w_j per held-out row and the test mass t; 1 in standard BCA | `calibration/wbcp.py:66-77`, `experiments/shift_iw/world.py:128-164`, `classifier.py:69-117` |
| M5 | Threshold with refusal | R = max(plain 90% quantile, 95%-credible quantile) over 1,000 draws; +infinity = refusal | `calibration/wbcp.py:43-117`, `calibration/reference.py:124-194` |
| M6 | Band | U = R · u_f · max(η_f(s, a), 1e-6); covered when \|y − q\| ≤ U | `calibration/dose.py:65-95`, `experiments/shift_iw/coverage.py:100-105, 262-303` |
| M7 | Dose | m = 1 + 0.5·U/(U + u_f) on the behavior-cloning term | `calibration/dose.py:23-62`, `algorithms/td3_bc.py:128-139` |
| M8 | Width penalty on critic values (toy) | choose a = argmax[q − U] | `bca_channel_toy.ipynb` cell 12 (branch `main`) |
| M9 | Lower edge | the actor climbs Q1 − k·[U(s, π(s)) − U(s, a_i)]_+ | `experiments/shift_iw/lower_edge.py:196-312` |
| M10 | Harm switch certificate | is the critic more over-optimistic on full-throttle rows than elsewhere? on/off and a height | `outputs/shift/beyond-the-data/restructure/step1/step1lib.py:174-276` |
| M11 | Ramp height | a throttle ramp of that height in the actor loss, when the switch is on | `outputs/shift/beyond-the-data/restructure/step2/step2lib.py:315-401` |

M1 to M7 are BCA as the hosts run it (ALGORITHMS.md, Procedures B to D). M8 to M11 are the four ways the band or its
parts have been *used*, each a module of its own. The harm switch (M10, M11) reuses M5 but not M2, M3 or M6: there is
no error-size network in it.

---

## M1. The residual

**What it computes.** The critic's one-step error at a recorded row: the host's own target minus the critic's value.

**Inputs.** A row (s, a, r, s', d), the critic pair, the target copies, the target actor, one random key.
**Outputs.** y, q and the residual e = y − q, one number each per row.

**Formula.** y = r + γ·(1 − d)·min(Q̄1(s', a'), Q̄2(s', a')) with a' = clip(π̄(s') + ε, −1, 1),
ε = clip(N(0, 0.2²), −0.5, 0.5); q = min(Q1(s, a), Q2(s, a)); e = y − q.

**Constants.** γ = 0.98 in ThrusterChain (0.99 in the D4RL hosts); noise 0.2 clipped at 0.5; d is always 0 in
ThrusterChain. The fit's noise key is fold_in(key, 1179210836); a refresh's is fold_in(key, 1213156420).

**Where it lives.** `algorithms/td3_bc_bca.py:146-156` (native_target), the fit's q at 170-174, the refresh's q and
residual at 252-256 and 263-270; the host's own target `algorithms/td3_bc.py:106-115`. The four-learner study restates
each learner's target in `experiments/shift_hosts/bca_core.py` (branch `claude/shift-hosts`, lines 18-24 of its header).

**Working as intended means.** For the same key, y equals the target the host trains on (the test
`test_critic_parts_are_bcas_own_target_and_q`, `experiments/shift_iw/test_learners_iw.py:415`), and recomputing e from
the saved critic reproduces every downstream number (see M3).

**Pseudocode** (`algorithms/td3_bc_bca.py:146-156`, 170-174):
```
1  [M1.1 target noise]   eps = clip(Normal(0, 0.2^2), -0.5, 0.5), one draw per row from the pinned key
2  [M1.2 next command]   a' = clip(target_actor(s') + eps, -1, 1)
3  [M1.3 target]         y = r + 0.98 * (1 - d) * min(targetQ1(s', a'), targetQ2(s', a')); held constant
4  [M1.4 critic value]   q = min(Q1(s, a), Q2(s, a)) at the RECORDED command a
5  [M1.5 residual]       e = y - q
```

**Variants.**
- *Fixed-controller critic* (stage 2's band, the near-80 version-6 arm): the "actor" is the stall-aware controller, so
  a' = clip(c(s') + ε, −1, 1) (`outputs/shift/beyond-the-data/stage2/code/band.py:214-222`).
- *IQL*: y uses the value network V(s'); no noise (ALGORITHMS.md:279). *ReBRAC*: the target subtracts its critic
  penalty (ALGORITHMS.md:281-282). *CQL*: its sampled backup.
- *The harm switch* uses a different residual: the one-sided overestimate with no noise (M10.4).

**Recorded.** y and q per held-out row are recomputed, never stored; their fingerprints are in every band record
(`scores_sha256`). The near-80 version-6 band records `unit` (u_f): 0.157 (stalling world, seed 0), 0.149 (no-stall,
seed 0) (`.../near80/results-20261020T115246Z/files/learners/near80/{stall,no_stall}/seed0/v6_band.json`).

**Verdict: works.** (Description only.) *The residual is the critic's own one-step training error at the recorded row,
with the host's own target; recomputing it from the saved networks reproduces BCA's stored threshold bit for bit.*

---

## M2. The error-size network η and the unit u

**What it computes.** A positive per-row scale η(s, a) such that about 90% of residuals satisfy |y − q| ≤ u·η, and the
running unit u that puts η in critic value units.

**Inputs.** A training batch of 256 rows, its y and q (M1), the current η parameters and u, one random key.
**Outputs.** New η parameters and a new u (or the old ones if the step is invalid).

**Formula.** η(s, a) = softplus(net(normalize(s), a)) + 0.01, net = two layers of 64 ReLU units and one output.
Loss L = (Σ_i m_i c_i − 0.9)² + 0.005·Σ_i m_i η_i², with bootstrap masses m_i = E_i/Σ_j E_j, E ~ Exponential(1),
and soft coverage c_i = sigmoid(20·(y_i/u − q_i/u + η_i)) · sigmoid(20·(q_i/u + η_i − y_i/u)).
u ← 0.99·u + 0.01·max(SD(y − q), 1e-6).

**Constants.** Hidden 64 × 2; floor 0.01 on η; coverage target 1 − α = 0.9; slope 20; width penalty 0.005; Adam
learning rate 0.001; unit averaging 0.99; unit floor 1e-6; u starts at 1; the fit runs at every update from the first
(the 10,000-update warm-up delays only the *use* of the band, ALGORITHMS.md:262-263). Settings:
`experiments/shift/learners.py:104-105` (BCA_SETTINGS), `configs/td3_bc.yaml`.

**Where it lives.** `calibration/network.py:8-31` (the network), 34-37 (soft coverage), 44-46 (bootstrap masses);
`algorithms/td3_bc_bca.py:132-137` (construction, Adam), 159-231 (fit_scale); restated for the four learners in
`experiments/shift_hosts/bca_core.py:282-317` (TD3+BC, CQL, ReBRAC) and 325-359 (IQL's fitter, which also balances
coverage) on branch `claude/shift-hosts`.

**Working as intended means.** (a) η is finite and positive at every row (checked at every fit, `td3_bc_bca.py:215`,
and refresh, line 261); (b) the fit is accepted on every step; (c) on the held-out rows the share with |y − q| ≤ u·η
is near 0.9, so the threshold R of M5 comes out near 1 or a little above; (d) η is larger where the critic's true
error is larger (the weighting test's mark M2).

**Pseudocode** (`algorithms/td3_bc_bca.py:159-231`; `calibration/network.py:16-31, 34-37, 44-46`):
```
1  [M2.1 inputs]        normalize s by the training rows' mean and SD (+0.001); append the command a
2  [M2.2 network]       h = ReLU(W1 x); h = ReLU(W2 h); eta = softplus(w3 . h) + 0.01     (so eta > 0.01)
3  [M2.3 masses]        E_i ~ Exponential(1) for the 256 rows (fold_in(key, 1128352850)); m_i = E_i / sum_j E_j
4  [M2.4 soft coverage] c_i = sigmoid(20 (y_i/u - (q_i/u - eta_i))) * sigmoid(20 ((q_i/u + eta_i) - y_i/u))
5  [M2.5 loss]          L = (sum_i m_i c_i - 0.9)^2 + 0.005 * sum_i m_i eta_i^2
6  [M2.6 step]          one Adam step (learning rate 0.001) on eta's parameters only (y, q, u held constant)
7  [M2.7 unit]          u_new = 0.99 u + 0.01 max(SD_i(y_i - q_i), 1e-6)
8  [M2.8 accept]        keep the new eta and u only if every number is finite, eta > 0, the masses sum to 1,
                        u > 0 and u_new > 0; otherwise keep the old eta and u
```

**Variants.** *Co-trained* (the host's run): fitted at every update on the live critic. *On a frozen critic* (the
weighting test's band, stage 2, near-80's version-6 arm): 20,000 fit steps on batches of kept training rows, then one
refresh (`experiments/shift/learners.py:1253-1279`, `experiments/shift_iw/learners_iw.py:482-519`). *IQL*: a fitter
with a balance term (`bca_core.py:325-359`). *Scalar η* (`state_dep=False`, `network.py:28-30`): one number, not used
by any ThrusterChain study.

**Recorded.** `accepted_share` (1.0 in the near-80 version-6 bands, seed 0, both worlds), the unit, and η at every
held-out row (`v6_bank_scores.npz` on branch `claude/near80-run-files`). The weighting test's mark M2 (expert): the
width's rank correlation with the exact critic error, inside each tenth of the current range and after removing the
effect of the float radius, is 0.063 on average (one-sided lower bound 0.018, so above 0;
`outputs/shift_iw/v5/expert/report.json`, marks.M2, `given_float_radius_within_deciles`).

**Verdict: works with a caveat.** (Mechanics description only; the M2 mark pre-registered.) *The error-size network is
a learned, always-positive scale, fitted every update and accepted every step, but its link to the critic's true error
is weak: rank correlation about 0.06 within current deciles, just above zero.*

---

## M3. The scores

**What it computes.** One dimensionless number per held-out row: the residual's size in units of the frozen scale.

**Inputs.** The held-out rows' y and q (M1, with the refresh key), the frozen η_f and u_f (M2).
**Outputs.** Scores ρ_1..ρ_n (float64), n = the number of held-out rows.

**Formula.** σ_j = u_f · max(η_f(s_j, a_j), 1e-6); ρ_j = |y_j − q_j| / σ_j.

**Constants.** The floor 1e-6 on η; σ computed in float32 then converted to float64 (`calibration/reference.py:148-150,
164`).

**Where it lives.** `calibration/reference.py:148-150` (positive_scale), 164-175 (inside freeze_reference); the
weighting test recomputes them the same way in `experiments/shift_iw/learners_iw.py:540-552`; stage 2 in
`outputs/shift/beyond-the-data/stage2/code/band.py:307-314`; the four-learner study in `bca_core.py:387-391`.

**Working as intended means.** Every ρ_j is finite and every σ_j positive (else the refresh is invalid and the previous
reference is kept, `reference.py:166-174`); recomputing the scores from the saved critic, saved η_f, u_f and the refresh
key, then calling M5 with the same random stream, reproduces the stored R, λ̂, λ_HPD and n_eff bit for bit.

**Pseudocode** (`calibration/reference.py:152-175`):
```
1  [M3.1 freeze]    eta_f, u_f = the current eta and u (they stay frozen until the next refresh)
2  [M3.2 scale]     sigma_j = max(eta_f(s_j, a_j), 1e-6) * u_f   (float32, then float64)
3  [M3.3 score]     rho_j = |y_j - q_j| / sigma_j                 (float64)
4  [M3.4 validity]  require every rho_j finite and every sigma_j > 0; else: invalid refresh, keep the old reference
```

**Variants.** None in the arithmetic. The group whose scores are taken differs by use: all held-out rows (M5-M7); the
tail and rest groups of the harm switch use a different residual (M10).

**Recorded.** The weighting test's mark W0 (bit-for-bit reproduction of the stored threshold from recomputed scores):
pass in 5 of 5 seeds on all five qualities (`outputs/shift_iw/v5/*/report.json`, marks.W0). The four-learner study's
stop rule S3 (the same check at every weighted refresh): passed (`outputs/shift_hosts_v3/results.json`, integrity.S3).
The near-80 version-6 bands: `threshold_recomputed.equal: true` (seed 0, both worlds).

**Verdict: works.** (Description only; the reproduction checks pre-registered.) *The scores are a deterministic
function of the saved networks and the refresh key, and they reproduce BCA's stored threshold bit for bit in every
recorded refresh.*

---

## M4. The weights

**What it computes.** How much each held-out row counts (w_j) and how much the row being certified counts (the test
mass t). Standard BCA uses w_j = 1 and t = 1.

**Inputs.** For the exact weights: the current c of each held-out row and the known laws. For the estimated weights:
a sample of observations from the training law and one from the test law. For the standard case: nothing.
**Outputs.** w_1..w_n ≥ 0 and t > 0.

**Formulas.**
- Standard: w_j = 1, t = 1 (the default when no weights are passed, `calibration/wbcp.py:66-69`).
- Exact (the weighting test): w(c) = p_test(c)/p_train(c) = (M(−2)/M(λ))·exp((λ + 2)c) with the training law's
  exponent −2 and the test law's λ (`experiments/shift_iw/world.py:128-135`): λ = −2 ("none"): w = 1; λ = 0 ("mild"):
  0.354·e^{2c}, range 0.43-3.19; λ = 2 ("strong"): 0.0907·e^{4c}, range 0.135-7.39 (`experiments/iw/spec.json`,
  laws.tilts). A test step whose command lies inside the ball has ratio +infinity and is refused before any threshold
  (`world.py:160-164`, `coverage.py:251-259`).
- Estimated: a logistic regression of "test sample vs training sample" on the 13-number observation, fitted by
  Newton's method with no penalty (`experiments/shift_iw/classifier.py:69-101`); ŵ(x) = exp(fitted log-odds(x))·n0/n1
  (`classifier.py:42-46`).
- The test mass: *per step* (the method the owner chose): t = the step's own weight w(x); *single* (kept as a
  comparison): one t for every step, the mean of w over test-law steps outside the ball, by quadrature
  (`experiments/shift_iw/run.py:549-607`).
- Effective size (Kish): n_eff = (Σ w)²/Σ w² (`coverage.py:94-97`); as a share of the held-out rows: 1, 0.724, 0.352
  for none, mild, strong (`spec.json` laws.kish_effective_bank_rows).

**Constants.** Grid ratio 1.01 for the per-step table (M5.9). Only ratios matter: weights and t are divided by the
largest weight before use (`wbcp.py:83-85`).

**Where it lives.** `calibration/wbcp.py:66-77` (defaults and validation); `experiments/shift_iw/world.py:128-164`;
`classifier.py:69-117`; `coverage.py:78-87` (unit check), 94-97, 251-259; `run.py:549-607`. The four-learner study:
`experiments/shift_hosts/iw.py:527-557` (the exact table, the estimated table and a shuffled-weights placebo at the
last refresh) and `PRESPEC.md` section 5.4 on branch `claude/shift-hosts`.

**Working as intended means.** (a) Weights of 1 with t = 1 give the unweighted threshold bit for bit (mark W0; the
four-learner stop rule S2); (b) the estimated classifier recovers the known slope of log w in c (0, 2, 4); (c) the
weighted arms refuse every test step inside the ball and none outside (mark W3); (d) estimated-weight coverage is within
0.02 of exact-weight coverage (mark W4).

**Pseudocode** (`wbcp.py:66-85`; `world.py:128-135`; `classifier.py:69-117`; `coverage.py:251-259`):
```
1  [M4.1 standard]   w_j = 1 for every held-out row; t = 1
2  [M4.2 exact]      w_j = p_test(c_j) / p_train(c_j) from the known laws of the current; a step's t = w(c_step)
3  [M4.3 estimated]  fit logistic regression (training sample = 0, test sample = 1) on the observation, Newton, no
                     penalty; w_j = exp(log-odds(x_j)) * n0 / n1; a step's t likewise at its observation
4  [M4.4 support]    a test step whose command lies inside the ball: ratio +infinity -> refused before M5
5  [M4.5 scale]      divide every w_j and t by max_j w_j (only ratios matter)
6  [M4.6 size]       n_eff = (sum_j w_j)^2 / sum_j w_j^2
```

**Variants.** standard (weights 1); exact; estimated; per-step versus single test mass; the four-learner study's
`iw` (exact weights, strong law), `iw_ones` (every weight 1: must equal standard bit for bit), `iw_repeat` (the same
run twice: must be identical) and a shuffled-weights placebo (weights permuted across held-out rows,
`iw.py:527-531`).

**Recorded** (`outputs/shift_iw/v5/expert/report.json`, branch `main`): W0 pass (5/5 seeds; the same on all five
qualities). W4: fitted slope on c 0.099 ± 0.22 (exact 0), 2.25 ± 0.28 (exact 2), 4.22 ± 0.24 (exact 4); estimated
minus exact coverage under the strong shift 0.0079; estimated minus unweighted with no shift 9.1e-5 (W4 pass). W3:
refused inside 1.0, outside 0.0 in every seed (pass on four qualities; fail on medium-expert for a reason recorded in
its report, not re-read here).

**Verdict: works with a caveat.** (W0, W3, W4 pre-registered.) *With weights of 1 BCA is exactly standard BCA, and
with the known shift the estimated weights recover the true slope within about 10%, but every weight used so far comes
from a world whose shift law is known; a real deployment has to supply and defend its own ratio.*

---

## M5. The threshold, with refusal

**What it computes.** The score R below which 90% of the scores fall, with 95% credibility: the larger of a plain
weighted 90% quantile and a credible quantile over 1,000 random reweightings. R = +infinity means "cannot certify".

**Inputs.** Scores ρ_1..ρ_n (M3), weights w_1..w_n and t (M4), a random stream.
**Outputs.** R, λ̂ (the plain quantile), λ_HPD (the credible quantile), the 1,000 crossings, n_eff.

**Formula.** λ̂ = the smallest sorted score whose cumulative weight reaches 0.9 of the total (Eq. 1 of Lou and Luo,
arXiv:2604.06464v3). For each of 1,000 draws of E_1..E_{n+1} ~ Exponential(1): the crossing is the first sorted score
at which Σ_{j≤k} E_j w_j ≥ 0.9·(Σ_j E_j w_j + E_{n+1}·t), or +infinity if none does (the extra atom E_{n+1}·t is the
unseen test point at the worst case). λ_HPD = the 950th smallest crossing. R = max(λ̂, λ_HPD).

**Constants.** α = 0.1 (miscoverage), β = 0.95 (credibility), M = 1,000 draws (`WBCPConfig`,
`calibration/reference.py:17-29`); the posterior key fold_in(refresh key, 1347375956) turned into a NumPy generator
(`reference.py:177`); stored in float32 (`reference.py:180-187`).

**Where it lives.** `calibration/wbcp.py:43-54` (crossings), 57-117 (calibrate); `calibration/reference.py:124-136`
(certifiable: the smallest safe group size), 152-194 (freeze_reference: validity, float32 storage, the invalid case);
the schedule `experiments/shift/learners.py:1162-1164` and the refresh call 1217-1222; the weighting test's single
weighted threshold `learners_iw.py:939-944` and per-row table 945-952; the table itself `coverage.py:153-205`; the
four-learner study `iw.py:461-522`; stage 2's weighted fallback tabulation `stage2/code/band.py:333-357`; the harm
switch's two posteriors `step1lib.py:237-241`.

**Working as intended means.** (a) With weights of 1 the sampler is the BQ-CP posterior of Snell and Griffiths
(`experiments/wbcp/test_wbcp.py:21`); the exact, infinite-draw form is the binomial test: a threshold exists iff
P(Binomial(n, 0.1) ≤ x) ≤ 0.05 for some x; (b) refusal happens exactly when fewer than 950 of the 1,000 draws cross,
and a draw crosses with probability 1 − 0.9^n, so the chance of a finite R is P(Binomial(1000, 1 − 0.9^n) ≥ 950):
0.000 at n ≤ 24, 0.031 at 26, 0.405 at 28, 0.700 at 29, 0.896 at 30, 0.996 at 32, 1 − 1e-6 or better at n ≥ 36
(`certifiable`; `scripts/threshold_checks.py` check 1); (c) R does not fall as t grows (the per-step table's assumption);
(d) a common factor on the weights and t changes nothing; (e) the band U = R·u·η covers about 90% of fresh rows from
the same law.

**Pseudocode** (`calibration/wbcp.py:57-117`; schedule `learners.py:1162-1164`; table `coverage.py:173-205`):
```
1  [M5.1 sort]           sort the scores ascending; carry the weights along; divide weights and t by max w
2  [M5.2 plain quantile] lambda_hat = the smallest sorted score at which the cumulative weight >= 0.9 * total weight
                         (ties count as covered; settled in exact rationals, wbcp.py:89-102)
3  [M5.3 draws]          for m = 1..1000: E_1..E_n, E_{n+1} ~ Exponential(1) from the posterior key
4  [M5.4 crossing]       lambda^(m) = the first sorted score k with sum_{j<=k} E_j w_j >= 0.9 (sum_j E_j w_j + E_{n+1} t);
                         +infinity if no k reaches it (the test atom alone holds more than 10% of the mass)
5  [M5.5 credible]       lambda_HPD = the 950th smallest of the 1,000 crossings   (ceil(0.95 * 1000))
6  [M5.6 threshold]      R = max(lambda_hat, lambda_HPD); R = +infinity means refusal ("cannot certify")
7  [M5.7 store]          store R, lambda_hat, lambda_HPD, n_eff in float32 with eta_f and u_f; a finite R that
                         overflows float32 is refused; an invalid refresh keeps the previous reference unchanged
8  [M5.8 schedule]       co-training: refresh at update 10,000, then every 5,000, last at 95,000 of 100,000
                         (18 refreshes); on a frozen critic: 20,000 fit steps, then one refresh
9  [M5.9 per-step table] with weights: R(t) at every t = 1.01^k from the smallest to the largest mass looked up,
                         the same draws for every t; a row or step reads R at the smallest grid mass >= its own
                         weight; take the running maximum if R ever fell along the grid (recorded per table)
```

**Variants.** Unweighted (the hosts, step 1, stage 2's primary, the near-80 version-6 arm); exact or estimated weights
with a single test mass (reported comparison); per-step table (the weighting test's method, the four-learner `iw`);
stage 2's declared fallback (48 log-spaced masses, linear interpolation in log t, an extra "refusal" term; built,
never used for a claim); the harm switch's two posteriors on two groups (M10). The *refresh schedule* is itself a
variant: every 5,000 updates in co-training versus one refresh on a frozen critic.

**Recorded.** Every refresh's R, λ̂, λ_HPD, n_eff and σ_post (`refresh_diagnostics` in band records; `refreshes` in
BCA run records). Near-80 version-6 bands (seed 0): R = 1.469 (stalling world, 821 held-out rows, λ̂ = 1.363,
λ_HPD = 1.469) and 1.533 (no-stall, 823 rows) (`.../learners/near80/{stall,no_stall}/seed0/v6_band.json`). The
certificate's R_tail per group size follows rule (b) in all 40 recorded seed-worlds: finite for every n_tail ≥ 30,
+infinity for every n_tail ≤ 28, and at n_tail = 29 two infinite and one finite (`scripts/certificate_table.py`;
the table is in M10). Each band record of the weighting test stores whether its per-step table was monotone and
whether the running maximum changed anything (`monotone`, `running_max_applied` in `tables`); those records live in
the run folder `runs/shift_iw_v5`, which is not committed and was not re-read here; synthetic check 4 found R(t)
monotone on a 402-point grid.

**Synthetic checks** (`scripts/threshold_checks.py`, run with the repository's `calibration/wbcp.py`, NumPy 2.5.4):
check 1, the share of random score sets with a finite R matched the binomial prediction at every n (for example 39.0%
observed against 40.5% predicted at n = 28, 67.0% against 70.0% at 29, 100% at n ≥ 33); check 2, weights of 1
reproduce the unweighted R, λ̂, λ_HPD and all 1,000 crossings bit for bit; check 3, decisions ("R is at most a claimed
value") agree with the exact binomial form in 100% of 300 sets at n = 36, 100 and 843 and in 92.7% at n = 45; check 4,
R(t) never fell along a 402-point grid; check 5, a factor of 1,000 on weights and t changed nothing. One more thing
check 3 shows: for 29 to 45 scores the exact credible threshold is the single largest score (46-60: the second
largest; 61-75: the third), so with few scores R is one order statistic from the top and moved by up to 44% between
random streams, while at 843 scores it moved by at most 2.1%.

**Verdict: works.** (Unit-tested; W0 pre-registered; the synthetic checks are new.) *The threshold is the published
weighted Bayesian conformal rule, it reproduces bit for bit, and it refuses exactly as the binomial arithmetic
predicts: almost never below 26 held-out rows, always from 36 up, and by chance in between.*

---

## M6. The band

**What it computes.** The half-width U at any (s, a), and whether a held-out row is covered by q ± U.

**Inputs.** R (M5), u_f and η_f (M2), a row's (s, a), and for coverage its y and q (M1).
**Outputs.** U(s, a) in critic value units; per row: covered (1), missed (0) or refused (−1).

**Formula.** U(s, a) = R · u_f · max(η_f(s, a), 1e-6); covered iff q − U ≤ y ≤ q + U.

**Constants.** The floor 1e-6; float32 arithmetic (`calibration/dose.py:75-76`); coverage intervals by resampling
whole episodes 2,000 times (`coverage.py:275-287`).

**Where it lives.** `calibration/dose.py:65-95` (frozen_level_dose: width 0 before the first refresh, +infinity on
refusal); `calibration/network.py:40-41` (hard_coverage); `experiments/shift_iw/coverage.py:100-105` (width),
262-273 (outcomes), 290-303 (summarize); the readouts `learners_iw.py:632-652` and 686-716; stage 2's substitution of
U_max for a refused band `stage2/code/band.py:153-158`; the near-80 width readout `near80/code/v6bca.py:90-93`.

**Working as intended means.** On fresh held-out rows from the same law as the calibration rows, the share covered is
near 0.90 (the weighting test's bar [0.89, 0.94] pooled over seeds, each seed at least 0.86: mark M1). Under a shift,
the unweighted band's coverage drops (W2: below 0.88) and the weighted band restores it (W1: in [0.89, 0.96]).

**Pseudocode** (`calibration/dose.py:65-95`; `coverage.py:262-273, 290-303`):
```
1  [M6.1 width]        U(s, a) = R * u_f * max(eta_f(s, a), 1e-6)                 (float32)
2  [M6.2 not ready]    before the first refresh: U = 0 everywhere (the host runs unchanged)
3  [M6.3 refusal]      R = +infinity: U = +infinity; the row is "unsupported"
4  [M6.4 covered]      a held-out row is covered when q - U <= y <= q + U (ties covered); a refused row is never covered
5  [M6.5 coverage]     coverage of a group = covered rows / all rows (refused rows count against it);
                       95% interval by resampling whole episodes
```

**Variants.** Read at the recorded (s, a) (M7); read at the actor's own command (M9, differentiable in a); with one
threshold for every row (standard) or each row's own threshold R(w_i) (per-step); U_max in place of +infinity (stage 2,
near-80); the largest finite threshold of the table in place of +infinity (the weighting test's lower edge).

**Recorded** (`outputs/shift_iw/v5/expert/report.json`, branch `main`; the same marks pass on all five qualities,
`outputs/shift_iw/v5/summary.json`): M1, no shift, unweighted: 0.914 (5 seeds, interval 0.896-0.932). W2, strong
shift, unweighted, outside the ball: 0.833 (interval 0.801-0.865; the bar is "below 0.88"). W1, per-step exact
weights: 0.919 under the strong shift, 0.913 under the mild (single test mass: 0.912, 0.912). Inside the ball (the
removed commands) the unweighted band covered only 0.769 under the strong shift and 0.808 under the mild (W3); the
weighted arms refuse every such step. Synthetic check 6: R = 1.413 on 843 synthetic held-out rows gave coverage 0.921
on 20,000 fresh rows.

**Verdict: works with a caveat.** (M1, W1, W2 pre-registered.) *The band covers 90% of fresh rows from the same law
and the weighted band keeps 90% under the known shift, but it is a band on the one-step target, not on the true value,
it holds on average over rows, and where the training data are missing (inside the ball) it is too narrow by a tenth.*

---

## M7. The dose on the behavior-cloning term

**What it computes.** A multiplier m between 1 and 1.5 on each row's pull toward the recorded command, larger where
the band is wide relative to the unit.

**Inputs.** U at the recorded row (M6, read at (s_i, a_i)), u_f.
**Outputs.** m_i per batch row, detached from every gradient.

**Formula.** m_i = 1 + blend · U_i/(U_i + u_f), blend = 0.5. TD3+BC's actor loss every second update:
−λ·mean_i Q1(s_i, π(s_i)) + mean_i m_i · mean_j (π(s_i)_j − a_ij)², λ = α/mean_i |Q1(s_i, π(s_i))| (held constant).

**Constants.** blend 0.5 (so m < 1.5); α = 2.5 (D4RL hosts and the weighting test), 5.0 in the near-80 actor-only
runs; policy delay 2; the ratio is computed as a normalized fraction to avoid overflow (`dose.py:50-60`).

**Where it lives.** `calibration/dose.py:23-62` (level_critic_dose; ratio at 59, dose at 60), 65-95
(frozen_level_dose); `algorithms/td3_bc_bca.py:283-289` (bc_readout), 295-324 (update); the host's actor loss
`algorithms/td3_bc.py:102-105, 128-139`; ALGORITHMS.md:334-340 (Procedure D). The weighting test's per-row dose
`learners_iw.py:655-683` (refusal gives 1.5 at lines 681-682) and its dose-change readout 719-729; the four-learner
study's carried refusal rule `iw.py:279-312` and its table of rules per learner 383-412; IQL's counterpart (a
shrinkage of the actor weight, not a dose) `calibration/advantage.py:32-70`. The placement variants of the main study
`experiments/shift/learners.py:1106-1121`.

**Working as intended means.** m ∈ [1, 1.5); m does not fall as U rises (up to one float32 step, about 1.2e-7); m
equals the closed form; m = 1 before the first refresh and, under the standard rule, on every refused row; with
weights of 1 the per-row dose equals standard BCA's bit for bit.

**Pseudocode** (`calibration/dose.py:23-95`; `algorithms/td3_bc.py:128-139`; `learners_iw.py:655-683`):
```
1  [M7.1 readout point]  read eta_f at the RECORDED row (s_i, a_i), not at the actor's command
2  [M7.2 width]          U_i = R * u_f * max(eta_f(s_i, a_i), 1e-6)
3  [M7.3 dose]           m_i = 1 + 0.5 * U_i / (U_i + u_f)                 (a normalized ratio, float32)
4  [M7.4 refusal]        U_i = +infinity (R = +infinity): standard BCA gives m_i = 1 (the native host);
                         the weighting test's per-row arm and the four-learner study's IW setups give m_i = 1.5
5  [M7.5 not ready]      before the first refresh m_i = 1
6  [M7.6 use]            every 2nd update: loss = -lambda * mean_i Q1(s_i, pi(s_i))
                                                + mean_i m_i * mean_j (pi(s_i)_j - a_ij)^2,
                         lambda = alpha / mean_i |Q1(s_i, pi(s_i))| (held constant); m is detached
7  [M7.7 per-row]        with weights: m_i uses the row's own threshold R(w_i) from the table (M5.9)
```

**Variants.** Standard (one R for every row; refusal → 1); per-row (R(w_i); refusal → 1.5); the four-learner
"most caution" flag (TD3+BC, CQL, ReBRAC: refused → 1.5; IQL: refused → plain cloning weight); placement
diagnostics of the main study ("flattened": every row the batch mean; "shuffled"; "noise_map"); the toy's A2
(the same formula in a two-step world).

**Recorded.**
- Mechanics (`outputs/shift_iw/v5/expert/report.json` marks.M3, pass on all five qualities): over 42,500 kept training
  rows per seed, the largest fall of the dose as the width rises was 1.19e-7 (one float32 step; the bar 1e-6), the
  largest gap to the closed form 1.04e-7, refused rows 0. The bit-for-bit identities are tested in
  `test_learners_iw.py:494, 524` and were the four-learner study's stop rule S2 (passed).
- Effect of weighting on the dose (marks.R2): mean dose change from the unweighted to the exact strong-shift
  threshold +0.039 (expert; interval 0.025-0.054) and +0.053 (medium-expert); predicted return from that change 0.42
  and 0.92 points, below the 1.8 needed, so R2 failed on these two qualities and BCA's training arms were not run
  there (I1 "not run" on every quality). The dose saturates: with U about 10u it is already 1.455 (check 7).
- Four-learner study (`outputs/shift_hosts_v3/results.json`, branch `claude/shift-hosts`; exploratory, seeds 0-4):
  TD3+BC on medium-expert, standard BCA minus alone: +6.54 points (one-sided lower bound +1.40; p = 0.027; Holm-adjusted
  0.107: shelved). CQL on medium-expert: −3.86 (shelved); CQL weighted minus standard: +1.83 (shelved). TD3+BC's
  weighted setup, and IQL and ReBRAC in every setup, were stopped by stop rule S7 (a float32 restatement check) and
  not judged. At the last refresh, CQL's mean dose over kept
  training rows rose from about 1.35 at the standard threshold to 1.49 at the weighted (exact strong) thresholds
  (`qualities.medium.expanded.cql.readouts`: `mean_dose_last` 1.495, pointer `change` 0.144); ReBRAC's stayed at 1.32
  (change about 0). Refused rows at the last refresh: 0 for both.
- Toy (`bca_channel_toy.ipynb`, branch `main`): A2 (this dose in a discrete host) had regret 0.38 against 0.12 for a
  rule that simply refuses thin data; shelved by its pre-written kill rule.

**Verdict: works with a caveat.** (Mechanics M3, W0 pre-registered; effects exploratory.) *The dose is computed
exactly as specified, stays in [1, 1.5) and is monotone in the width, but it saturates, so weighting moves it by only
0.04-0.05 and no study has shown it raises return.*

---

## M8. The width penalty on critic values (toy worlds only)

**What it computes.** A decision rule that subtracts the band's width from the critic's value at every action and
takes the best remaining action.

**Inputs.** q(t, c, a) and U(t, c, a) in a two-step, two-context toy world.
**Outputs.** The chosen action per (t, c).

**Formula.** a* = argmax_a [q(t, c, a) − 1.0 · U(t, c, a)], U = r_t · scale(t, c, a), r_t the exact BCA threshold
on all held-out scores of step t (survivors of the risky action included).

**Constants.** Gain 1.0; α 0.1, β 0.95; the toy's per-cell scale (pooled residual SD, shrunk with 5 pseudo-rows).

**Where it lives.** `bca_channel_toy.ipynb` on branch `main`, cell 12 (the rule cell): `width = stack(r0*scale[0],
r1*scale[1])`, `A1 = (q - GAIN*width).argmax(-1)`, `A2 = bc_policy(q, n, 1 + BLEND*width/(width + unit))` (lines
88-93 of that cell); constants in cell 2; the rules table in cell 9; the kill rules in cell 10.

**Working as intended means.** The rule closes at least a quarter of the room (the regret gap between the best rule
that never reads the risky action's outcomes and the oracle); kill rule test 1, fingerprinted before the run.

**Pseudocode** (`bca_channel_toy.ipynb`, cell 12):
```
1  [M8.1 threshold]  r_t = the exact BCA threshold on all held-out scores of step t
2  [M8.2 width]      U(t, c, a) = r_t * scale(t, c, a)          (no error-size network: a per-cell SD)
3  [M8.3 penalty]    choose a = argmax_a [ q(t, c, a) - 1.0 * U(t, c, a) ]
```

**Variants.** A1 (this module) and A2 (the dose of M7 in the toy's discrete host). The earlier
`bca_discrete_toy.ipynb` reached the same conclusion with no held-out tries of the risky action.

**Recorded** (`bca_channel_toy.ipynb`, "Bottom line", branch `main`): regret 0.43 (A1) and 0.38 (A2) against 0.12 for
refusing thin data; both shelved. The notebook's own words: "The width is measured where the data are, so it cannot
see how the risky action turns out."

**Verdict: does not work.** (Toy worlds, toy seeds; kill rules written before the run.) *Subtracting the width from
the critic's value at every action was tested only in a toy and did worse than refusing thin data, because the width
is measured where the data are and cannot see how an untried action turns out.*

---

## M9. The lower edge

**What it computes.** A penalty in the actor's objective equal to how much wider the band is at the actor's command
than at the recorded command of the same state.

**Inputs.** The frozen η_f, u_f and R (or per-row R_i), the frozen critic, the actor, a batch of training rows.
**Outputs.** One actor step.

**Formula.** W_i = [U(s_i, π(s_i)) − U(s_i, a_i)]_+ with U(s, a) = R_i · u_f · max(η_f(s, a), 1e-6);
loss = −λ·mean_i [Q1(s_i, π(s_i)) − k·W_i] + mean_i mean_j (π(s_i)_j − a_ij)², λ = α/mean_i |Q1(s_i, π(s_i))| on Q1
alone, k = 1.

**Constants.** k = 1 (k = 2 a declared secondary arm; k = 0 the plumbing check); α = 2.5 in the weighting test, 5 in
stage 2 and near-80; 20,000 actor-only steps in chunks of 4,000 from the cloned actor, Adam 3e-4, batch 256.

**Where it lives.** `experiments/shift_iw/lower_edge.py:196-201` (band_width), 204-208 (band_penalty), 211-224
(capped_thresholds: refusal → the largest finite threshold of the table), 293-312 (actor_step), 688-725 (the arms);
stage 2 `outputs/shift/beyond-the-data/stage2/code/band.py:141-183` (width, refusal → U_max, paired penalty),
291-304 (U_max from the held-out rows) and `policy.py:68-93` (the actor step); the near-80 version-6 arm
`near80/code/v6bca.py:36, 96-106` (k = 1, uniform weights); the specification `experiments/iw/spec.json`
(lower_edge).

**Working as intended means.** (a) With k = 0 the step is the plain actor step (tested, `test_lower_edge.py:113`);
(b) the width at the actor's command is BCA's own, differentiable in the command through η_f's input only
(`test_lower_edge.py:250`); (c) the penalty rises past 80% throttle on hard-water states (check (e): it must not fall
from 80% to 81% on more than a set share of states, its mean must rise, its slope at 80.5% must be positive, and
Q1 − W must prefer a throttle past 80% on fewer states than Q1 alone); (d) against the exact one-step oracle at k = 1,
BCA's penalty should have a ratio of means near 1 and a positive correlation (the proxy check).

**Pseudocode** (`lower_edge.py:196-224, 293-312`; `stage2/code/band.py:153-158, 176-183`):
```
1  [M9.1 width at the actor's command]   U_pi = R_i * u_f * max(eta_f(s_i, pi(s_i)), 1e-6);
                                         the gradient reaches pi through eta_f's command input only
2  [M9.2 width at the recorded command]  U_rec = U(s_i, a_i), a constant per row
3  [M9.3 paired penalty]                 W_i = max(U_pi - U_rec, 0)
4  [M9.4 actor loss]                     -lambda * mean_i [Q1(s_i, pi(s_i)) - k W_i] + mean_i mean_j (pi(s_i)_j - a_ij)^2,
                                         lambda = alpha / mean_i |Q1(s_i, pi(s_i))| (held constant), k = 1
5  [M9.5 refusal]                        R_i = +infinity: the largest finite threshold of the table (weighting test),
                                         or U_max, the largest finite width over the held-out rows (stage 2, near-80)
6  [M9.6 weighted]                       R_i = R(w(c_i)) per row: the penalty at each state is the unweighted one
                                         times R(w(c_i)) / R_u
```

**Variants.** Paired (primary) versus unpaired (U(s, π(s)) alone, a declared secondary arm of stage 2); uniform versus
per-row weighted thresholds; stage 2's weighted fallback (classifier-based, with an extra refusal term; built, not
used for a claim); the near-80 "version-6" arm (uniform, k = 1, on the fixed-controller critic's band).

**Recorded.**
- Weighting test (`outputs/shift_iw/v5/summary.json`, branch `main`): the *oracle* lower edge (the same penalty with
  the exact one-step error, no BCA) raised return on expert and medium-expert (L1 pass) and not on random,
  medium-replay or medium (L1 fail). BCA's own lower-edge arms were never trained: L2 (the predicted return from the
  weighted band's scale change) was −0.37 points on expert and +1.15 on medium-expert, below the 1.8 needed, so L3 is
  "not run" everywhere. The high-current rows' threshold ratio R(w)/R_u − 1 was +0.51 (expert) and +0.76
  (medium-expert) under the strong shift (marks.L2). The proxy check found check (e)'s part e4 failing on some seeds
  for BCA's penalty (marks.L2.proxy_check).
- Near-80 (exploratory, seeds 0-4, `.../near80/results-20261020T115246Z/files/looks/exploratory.md`): in the near-80
  setting, strong current, stalling world, version 6's BCA (this module at k = 1) scored −13.0 points; the controller
  trained without caution scored −13.2; BCA's switched ramp (M11) 87.4. Its strength (how far it moved the actor:
  the root-mean-square change of the command from the no-caution actor on 5,000 evaluation states) was 0.020-0.023
  in every seed and both worlds, against 0.146-0.172 for the switched ramp in the stalling world and 0.14-0.16 for the
  hand-set range limit (`looks/exploratory.json`, reported.locks.near80.{stall,no_stall}.<seed>.strengths). In
  words: the lower edge left the actor where it was.
- Stage 2's own run (version 6, the main setting) and its result letters are in
  `outputs/shift/beyond-the-data/stage2_v7/` and `results-2026-10-13/`; not re-read for this document.

**Verdict: does not work.** (Mechanics tested; effects exploratory, with the weighting test's marks pre-registered.)
*The arithmetic is right and the same penalty driven by the exact error helps, but with BCA's band the penalty is too
flat at the actor's command to move the actor: version 6's BCA scored −13.0 where no caution scored −13.2 and the
switched ramp 87.4.*

---

## M10. The harm switch certificate

**What it computes.** Whether a second critic, trained without the full-throttle survivors, is more over-optimistic
on held-out full-throttle rows than on the rest, with 95% credibility and a margin; and a height, the gap between the
two groups' thresholds.

**Inputs.** The kept training commands (for the gap rule), the second critic, the held-out rows, the survivors' rows,
two fixed random streams.
**Outputs.** on/off, the statistic q05, the height h = R_tail − R_rest, the group sizes and both thresholds.

**Formulas.** Throttle index t(a) = max_k |a_k|. The gap rule: among kept training rows with t above its 99th
percentile, the largest difference between consecutive sorted t's; its ends are t_lo and t_up; defined when
t_up − t_lo ≥ 0.05 and at least 9 rows have t ≥ t_up (those rows are the survivors). The one-sided overestimate on a
held-out row: s = min(Q1, Q2)(o, a) − [r + γ·(1 − d)·min(Q1, Q2)(o', c(o'))], c the stall-aware controller, no
noise. Tail = survivors plus held-out rows with t > t_lo; rest = held-out rows with t ≤ t_lo. Two BCA posteriors (M5
with weights 1) on the tail's scores and the rest's. q05 = the 5% quantile of (tail crossing − rest crossing) over all
1,000 × 1,000 pairs of draws. On iff q05 > 0.10. Height = R_tail − R_rest.

**Constants.** Quantile 0.99, min gap 0.05, min tail 9 (`near80_constants.py:109`); α 0.1, β 0.95, 1,000 draws,
statistic quantile 0.05, margin 0.10 value units (`near80_constants.py:110`); streams [20261015, 9700, seed, world,
1] and [..., 2] (`near80_constants.py:111`); γ = 0.98.

**Where it lives.** `outputs/shift/beyond-the-data/restructure/step1/step1lib.py:174-176` (throttle), 179-206 (gap
rule), 216-221 (overestimate), 226-234 (pairwise quantile), 237-241 (posterior), 244-276 (certificate), 290-305
(shuffle control); the near-80 restatement `near80/code/learner_phases.py:98-151` (second critic), 156-173
(overestimates), 186-196 (groups), 199-236 (switch); step 1's `PRESPEC.md` sections "Scores" and "Certificate".

**Working as intended means.** On in the stalling world and off in the no-stall world; shuffling the scores between
the groups switches it off; the height is finite whenever the tail holds at least 36 rows (M5's rule) and may be
infinite below that.

**Pseudocode** (`step1lib.py:174-276`; `learner_phases.py:156-236`):
```
1   [M10.1 throttle index]   t(a) = max over the four thrusters of |a_k|   (no error-size network anywhere in M10)
2   [M10.2 gap rule]         sort the t's above their 99th percentile; the largest gap between neighbors gives
                             t_lo and t_up; defined iff t_up - t_lo >= 0.05 and >= 9 rows have t >= t_up (survivors)
3   [M10.3 second critic]    train the same fixed-controller critic on the kept rows MINUS the survivors (100,000
                             updates), so the survivors are held out from it
4   [M10.4 overestimate]     on each held-out row: s = min(Q1,Q2)(o, a) - [r + 0.98 (1 - d) min(Q1,Q2)(o', c(o'))],
                             c = the stall-aware controller, no noise (positive = the critic promised too much)
5   [M10.5 groups]           tail = survivors + held-out rows with t > t_lo;  rest = held-out rows with t <= t_lo
6   [M10.6 posteriors]       BCA's posterior (M5.3-M5.4, weights 1) on the tail's scores and on the rest's,
                             streams [20261015, 9700, seed, world, 1] and [..., 2]
7   [M10.7 statistic]        q05 = the 5% quantile of (tail crossing - rest crossing) over all 1,000 x 1,000 pairs
                             (+inf - finite = +inf; inf - inf counts as -inf: no evidence of a gap)
8   [M10.8 switch]           on iff q05 > 0.10 value units
9   [M10.9 height]           h = R_tail - R_rest, each R = max(lambda_hat, lambda_HPD) of its group;
                             h = +infinity when the tail cannot be certified
10  [M10.10 shuffle control] permute the pooled scores between the groups and recompute with the same streams
```

**Variants.** Step 1's form (both groups get BCA's posterior; margin 0.10) versus the design's earlier form (the
rest's plain 90% quantile; margin 0; reported only as `e11_form`); a noise-averaged overestimate (reported in step 1
only); the rank test (Mann-Whitney) and the mean-gap switch as rivals (M11.7).

**Recorded** (`scripts/certificate_table.py` prints every row from
`.../near80/results-20261020T115246Z/files/learners/*/*/seed*/switch.json` and
`.../restructure/step1/results/results.json`):

| setting | world | seeds | on | q05 | n_tail | height |
|---|---|---|---|---|---|---|
| step 1, main (seeds 0-4) | stalling | 5/5 | 0.78 to 1.05 | 35-59 | 1.61, 1.80, 2.07, 2.42, 6.02 |
| step 1, main | no-stall | 0/5 | −0.047 to 0.021 | 29-60 | 0.04-0.20, one +inf (n_tail 29) |
| near-80 (seeds 0-4) | stalling | 5/5 | 0.56 to 1.07 | 34-57 | 1.41, 1.92, 2.56, 3.15, 5.94 |
| near-80 | no-stall | 0/5 | −0.044 to 0.053 | 29-60 | 0.05-0.21, one +inf (n_tail 29) |
| random rows (seeds 0-4) | stalling | 5/5 | 0.77 to 1.26 | 29-52 | 1.69-5.61 |
| random rows | no-stall | 0/5 | −0.047 to 0.023 | 28-56 | 0.09-0.26, one +inf (n_tail 28) |
| main, fresh seeds 10-14 | stalling | 5/5 | 0.68 to 1.03 | 19, 53, 22, 18, 30 | +inf, 2.29, +inf, +inf, 4.59 |
| main, fresh seeds 10-14 | no-stall | 0/5 | −0.012 to 0.033 | 18-36 | 0.13-0.78, two +inf |

Step 1's go/no-go: C1 (on 5/5 stalling, off 5/5 no-stall), C2 (net ≥ 90: 106.1) and C3 (shuffled scores keep at
most half of the net) all true; verdict GO (`step1/results/results.json`, go_no_go). The switch itself was right in
all 40 recorded seed-worlds: on in every stalling world, off in every no-stall world.

**The exceptional case.** The switch reads q05, which stays finite when a few of the tail's 1,000 crossings are
infinite; the height reads R_tail, which is infinite as soon as fewer than 950 crossings are finite. So the switch can
be *on* with an *infinite height*. It never happened on seeds 0-4 (every stalling tail had 29-59 rows), and it happened
in 3 of the 5 fresh stalling seeds of the main setting (seeds 10, 12, 13: 19, 22 and 18 tail rows). By M5's
arithmetic a tail of 18-22 rows can never be certified (probability 0.000 at n ≤ 24). The consequence is in M11.

**Verdict: works with a caveat.** (Step 1 exploratory; near-80 seeds 0-4 exploratory, seeds 10-14 pre-registered.)
*The switch turned on in every stalling world and off in every no-stall world across 40 seed-worlds, but its height
is a difference of two thresholds that is undefined whenever the full-throttle group holds fewer than about 29 rows,
which happened in 3 of 5 fresh seeds.*

---

## M11. The ramp height

**What it computes.** A penalty on throttle in the actor loss: zero up to where the recorded throttles stop, rising in
a straight line across the empty stretch to the certificate's height, flat beyond; present only when the switch is
on.

**Inputs.** t_lo, t_up (M10.2), the switch and height (M10.8-M10.9), the frozen critic, the cloned actor.
**Outputs.** One trained actor per arm.

**Formula.** U(a) = h · clip((t(a) − t_lo)/(t_up − t_lo), 0, 1); penalty P_i = k · [U(π(s_i)) − U(a_i)]_+;
loss = −λ·mean_i [Q1(s_i, π(s_i)) − P_i] + mean_i mean_j (π(s_i)_j − a_ij)², λ = 5/mean_i |Q1|.

**Constants.** h = the certificate's height when on, else 0; k = 1 (primary), 2 (dose readout); α = 5; 20,000
actor-only steps in chunks of 4,000 from the cloned actor; the plumbing tolerance 1e-5 (`near80_constants.py:114`).
There is no error-size network, unit or score in M11: the ramp reads the command's throttle only.

**Where it lives.** `outputs/shift/beyond-the-data/restructure/step2/step2lib.py:315-364` (the arm table: how each
arm's switch, height and k are read from step 1's record; a switched-on arm with an infinite height is refused at
lines 355-357), 383-401 (throttle, ramp, penalty); step 2's `PRESPEC.md` sections "The new penalty" and "The arms";
the near-80 arm rows `near80/code/policy_phase.py:99-150` and its `PREREGISTRATION.md` sections 1 and 5.2.

**Working as intended means.** (a) An arm with h = 0 reproduces the no-caution actor within 1e-5 on every parameter
(the plumbing check); (b) the switched ramp beats the best rival that cannot switch itself off by at least 10 points
of net across worlds (points gained in the stalling world minus points lost in the no-stall world, strong current);
(c) it keeps its gain in the other three situations and beats the matched even pull; (d) a *calibrated* height would
beat both a smaller and a larger height (this was not shown; see below).

**Pseudocode** (`step2lib.py:383-401, 315-364`):
```
1  [M11.1 ramp]            ramp(t) = clip((t - t_lo) / (t_up - t_lo), 0, 1)
2  [M11.2 surface]         U(a) = h * ramp(t(a)); h = the certificate's height when the switch is on, else 0
3  [M11.3 paired penalty]  P_i = k * max(U(pi(s_i)) - U(a_i), 0)   (charged only for going further up than the
                           recorded command did), k = 1
4  [M11.4 actor loss]      -lambda * mean_i [Q1(s_i, pi(s_i)) - P_i] + mean_i mean_j (pi(s_i)_j - a_ij)^2,
                           lambda = 5 / mean_i |Q1(s_i, pi(s_i))|; 20,000 actor-only steps against the frozen critic
5  [M11.5 h = 0]           the penalty and its gradient are exactly 0: the arm must equal the no-caution actor within 1e-5
6  [M11.6 refusal]         a switched-on arm with h = +infinity is refused: that seed-world's actor phase stops
7  [M11.7 rivals]          obca_forced: always on at the stalling world's h (world-blind); fixed_gated: h = 1.0;
                           conformal_k1: h = the plain 90%-quantile gap, on when > 0.10; meangap_k1 / meandiff_k1:
                           the mean gap, on by a rank test / when > 0.10; knee_ramp: BCA's h, ramp from 80% to t_up
```

**Variants.** k = 1 and k = 2; the always-on ramp; the fixed, conformal and mean-gap heights; the ramp started at the
stall; the main, near-80 and random-rows settings (each with its own t_lo, t_up, switch and height).

**Recorded.**
- Step 2 (exploratory, seeds 0-4; `restructure/step2_results/results.json`): net across worlds 102.9 for BCA's
  switched ramp against 77.7 for the always-on ramp (the best world-blind rival), 69.9 for the matched range limit,
  89.3 for the fixed height 1.0, 97.3 for the conformal height, 57.2 for the mean gap; G1, G2, G3 true. Attribution
  (`attribution`): every rival with a smaller stalling-world height did worse, so "whether BCA's height did better
  because it is Bayesian or only because it is larger cannot be told here" (the record's own words).
- Near-80, seeds 0-4 (`looks/exploratory.md`): strong current, stalling world, near-80 setting: switched ramp 87.4,
  ramp at twice the height 85.5, fixed height 1.0 82.7, conformal 87.1, ramp from the stall 87.7, hand-set range
  limit 89.8, no caution −13.2, full-log controller 82.3; no-stall world: 93.2 for the switched ramp, equal to no
  caution (h = 0 there). Marks: claim 1 net lead 42.9 (needed 10), even pull 100.2 (needed 15), gains kept in the
  three other situations (6.2, 16.5, 22.8; floor −3); claim 2 unseen-throttle gain 8.6 (needed 5, positive in 5 seeds),
  overshoot loss 1.0 (at most 5); claim 3 invalid: hiding the band cost the hand-set range limit only 4.3 points
  (needed 5), so the near-80 setting stopped before the fresh seeds by its pre-set rule. Plumbing: the recorded height-0 checks
  passed with difference 0.0 (for example `looks/exploratory.json`,
  reported.locks.near80.no_stall.0.checks.h0_reproduces_reference).
- Near-80, fresh seeds 10-14, main setting (claim 4, pre-registered; `looks/confirmatory.md`): the actor phase
  stopped in 3 of 5 stalling seeds because the switch was on with an infinite height (M10's exceptional case;
  `files/stops/policy.main.stall.s{10,12,13}.json`); by the pre-set rule every mark that needs them fails, so claim 4
  did not hold on fresh seeds.

**Verdict: works with a caveat.** (Step 2 exploratory; near-80 claims 1-2 exploratory, claim 4 pre-registered and
failed by rule.) *The ramp placed by the switch earned a large net on the exploratory seeds and left the no-stall
world untouched, but the fresh-seed confirmation stopped in 3 of 5 seeds because the height is undefined for small
survivor groups, and the rivals show the height is "large enough", not that it is calibrated.*

---

## Consistent and exceptional cases (for debugging and for the advisor's question about weighting)

The advisor asked whether the weighting of the penalty is right: with no shift the penalty should be low, and if the
weighted version gives a *higher* penalty with no shift, why. The records answer this directly.

**Consistent (holds in every record).**
1. *Weights of 1 are standard BCA, bit for bit.* W0 passed in 25 of 25 seed-quality pairs; the four-learner stop rule
   S2 passed; the synthetic check 2 reproduces R, λ̂, λ_HPD and all 1,000 crossings. So with no shift and known
   weights, the weighted penalty cannot differ from the standard one at all.
2. *With estimated weights and no real shift, the difference is noise.* The classifier's fitted slope under the "none"
   law was 0.10 ± 0.22 (exact 0), and coverage differed from the unweighted band by 9.1e-5 (W4, expert).
3. *Under the known shift the weighted threshold rises, in the right direction.* The high-current rows' threshold
   ratio R(w)/R_u − 1 was +0.51 (expert) and +0.76 (medium-expert) under the strong shift; coverage went from 0.833
   (unweighted) to 0.919 (weighted). The dose moved up by 0.04-0.05 (weighting test) and by about 0.14 (CQL,
   four-learner, from 1.35 to 1.49).
4. *The dose saturates.* 1 + 0.5·U/(U + u) is 1.455 already at U = 10u, so whenever the band is wide relative to the
   unit the dose is near 1.5 and further widening does almost nothing. This is why weighting barely moves the dose
   (R2 failed) while the lower edge, which is linear in U, moves by 50-76% on the same rows.
5. *Refusal follows the binomial arithmetic.* Across 40 recorded certificates, R_tail was finite for every tail of 30
   or more rows and infinite for every tail of 28 or fewer; the synthetic check reproduces the whole curve.

**Exceptional (seen in a few records; the cases to show the advisor).**
6. *Switch on, height infinite.* Three of the five fresh stalling seeds of the main setting (tails of 18-22 rows).
   The switch (a 5% quantile of pairwise differences) and the height (a difference of thresholds) refuse at different
   group sizes. A rule for that case was never pre-registered, so the run stopped by its own check.
7. *Few scores make the credible threshold one score from the top.* With 29-45 scores the exact credible threshold is
   the single largest score; between random streams it moved by up to 44% in the synthetic check. The certificate's
   heights (tails of 29-60 rows) are differences of such thresholds: 1.4 to 6.0 across seeds.
8. *Refusal means the least caution under the standard rule.* A refused row gets dose 1 (M7.4). It never bit (0
   refused rows in every recorded refresh), but the weighting test and the four-learner study had to declare the
   opposite rule (1.5) for the weighted arms, where refusal is reachable through large test masses.
9. *The lower edge with BCA's band does nothing.* Version 6's BCA: −13.0 against −13.2 for no caution (near-80,
   seeds 0-4), strength 0.02 in every seed and world against 0.15-0.17 for the ramp; the same penalty with the exact
   error helps (L1 pass on expert, medium-expert).
10. *An invalid refresh is silent.* `td3_bc_bca.py:271-272` keeps the previous reference and reports
    `posterior_inputs_valid: false`; the count is `uncertified_refreshes` in BCA run records. None was recorded in the
    weighting test's band records read here.

---

## How to reproduce the numbers in this document

- `docs/bca_mechanics/scripts/certificate_table.py` (standard library): prints every certificate of the near-80 run
  and of step 1 with M5's expectation for its tail size, and the three recorded stops. Run from the repository root:
  `python docs/bca_mechanics/scripts/certificate_table.py`.
- `docs/bca_mechanics/scripts/threshold_checks.py` (NumPy and the repository's `calibration/wbcp.py`): seven synthetic
  checks of M4-M7 (refusal by size against the binomial prediction, the unit check, decisions against the exact
  binomial form, monotone in the test mass, scale invariance, coverage of fresh rows, the dose's range and refusal
  values). Run from the repository root with `PYTHONPATH=.`: all seven print PASS (NumPy 2.5.4, Python 3.14; the
  pinned environment is NumPy 2.2.6, Python 3.10, which the synthetic checks do not need).
- The weighting test's records: `outputs/shift_iw/v5/{expert,medium-expert,medium,medium-replay,random}/report.json`
  and `summary.json` on branch `main` (`git show origin/main:outputs/shift_iw/v5/summary.json`).
- The four-learner study's record: `git show origin/claude/shift-hosts:outputs/shift_hosts_v3/results.json`.
- Step 1 and step 2: `outputs/shift/beyond-the-data/restructure/step1/results/results.json`,
  `.../restructure/step2_results/results.json`.
- Near-80: `outputs/shift/beyond-the-data/near80/results-20261020T115246Z/` (records) and branch
  `claude/near80-run-files` (data, scores, saved critics and bands).
- The toy notebooks: `bca_channel_toy.ipynb`, `bca_mean_gate_toy.ipynb`, `bca_weights_toy.ipynb` on branch `main`;
  each runs in under five minutes with NumPy and Matplotlib and reprints its own numbers.

## The module list

M1 residual; M2 error-size network and unit; M3 scores; M4 weights; M5 threshold with refusal; M6 band; M7 dose on
the behavior-cloning term; M8 width penalty on critic values (toy); M9 lower edge; M10 harm switch certificate;
M11 ramp height.

| Module | Verdict | Evidence label |
|---|---|---|
| M1 residual | works | description only |
| M2 error-size network and unit | works with a caveat | description; the M2 mark pre-registered |
| M3 scores | works | description; W0 and S3 pre-registered |
| M4 weights | works with a caveat | W0, W3, W4 pre-registered |
| M5 threshold with refusal | works | unit tests; W0 pre-registered; synthetic checks new |
| M6 band | works with a caveat | M1, W1, W2 pre-registered |
| M7 dose | works with a caveat | M3 pre-registered; effects exploratory |
| M8 width penalty (toy) | does not work | toy; kill rules written before the run |
| M9 lower edge | does not work | mechanics tested; effects exploratory |
| M10 harm switch certificate | works with a caveat | step 1 exploratory; near-80 seeds 10-14 pre-registered |
| M11 ramp height | works with a caveat | step 2 exploratory; near-80 claim 4 pre-registered, failed by rule |
