# How BCA computes and uses its band, and where exact weights can enter

Scout notes for the importance-weighting experiment (experiments/shift_iw). Read-only survey of the repository at
/root/bca-shift; nothing here ran on ThrusterChain data. The only code run was a check on small synthetic arrays
(section 6). Line numbers refer to /root/bca-shift.

## Terms used below

- **Critic**: the TD3+BC network that predicts Q(s, a), the discounted value of command a in state s. It has two heads;
  BCA uses the smaller of the two, q = min(Q1, Q2).
- **TD3+BC**: the host learner (twin-critic actor-critic with a behaviour-cloning term in the actor loss).
- **Target y**: the critic's own one-step estimate, y = r + 0.98 · min over heads of the target critic at
  (s', target actor(s') + clipped random noise).
- **Residual**: y - q. Its absolute value is what BCA's band covers.
- **Scale network**: a small network eta(s, a) > 0 that sets how wide the band is at each row, times a unit u (a moving
  average of the residual spread).
- **Score**: |y - q| / (u · eta(s, a)).
- **Calibration bank**: held-out rows used only to choose the threshold.
- **Threshold R**: one number chosen on the bank so that the scores fall below it 90% of the time, with 95% posterior
  credibility. Chosen by weighted Bayesian conformal prediction (WBCP, Lou and Luo), which is a conformal rule with a
  Bayesian posterior over the threshold.
- **Width U(s, a)** = R · u · eta(s, a). This is a half-width: the band is [q - U, q + U].
- **Dose**: the per-row multiplier on TD3+BC's behaviour-cloning term, 1 + 0.5 · U / (U + u). It lies in [1, 1.5).
- **Weight w**: the likelihood ratio dP_test / dP_calibration at a bank row.
- **Test mass**: the weight given to the one extra "unseen test point" in WBCP's posterior.
- **Cannot certify**: R = +infinity. This happens when that test point alone can carry more than 10% of the posterior
  mass too often.

## Takeaways

- **(a)** The band covers the critic's own sampled one-step residual: |y - q| <= U(s, a) at the recorded command. It does
  not cover the error against the true value Q^pi. You can measure coverage on held-out rows with BCA's own functions:
  `native_target`, the critic, the width field of `bc_readout`, and `calibration.network.hard_coverage`. With an
  unweighted bank of about 850 rows, expect about 91.7% coverage, not 90%.
- **(b)** Weights are not supported end to end. `wbcp.calibrate` accepts weights and a test mass. But `WBCPConfig`,
  `freeze_reference` and `refresh` never pass them, and the stored threshold is a single number. A per-row test mass is
  not handled anywhere.
  - Minimal faithful route: run BCA's unchanged `refresh`, then recompute the same bank scores and call `calibrate`
    with exact weights and a scalar test mass. Then replace the four threshold fields of the frozen reference.
  - With unit weights this gives bit-identical results (checked on synthetic arrays).
  - The scale fit stays unweighted, so BCA's signal (the shape eta) does not change. Only R changes.
- **(c)** R = +infinity gives width +infinity, which counts as "unsupported", so every row gets dose 1 (no caution).
  `calibrate` also raises an error on an infinite test mass, so the caller must handle refusal in the gap.
  - Report a three-way outcome per test step: covered, not covered, refused. Never count a refused step as covered.
  - Exact-weighted refusal inside the gap holds by construction. It comes from knowing the gap, not from BCA detecting
    it.
- **(e)** `learners.bca_band` already fits the band on a frozen critic without training an actor. It is locked to
  experiments/shift's own approval file, so shift_iw must restate its roughly 20 lines under its own gate. The pieces
  it calls are not gated: `bca_initial_state`, `bca_config`, `make_args`, `P.fit_scale` and `P.refresh`.

## (a) What the band covers, and how to measure its coverage

**The residual.**
- Target: `native_target` (algorithms/td3_bc_bca.py:146-156) computes
  y = r + (1 - done) · gamma · min_k Qbar_k(s', clip(target actor(s') + eps)), where eps = clip(N(0, 0.2²), ±0.5).
  The noise eps is drawn fresh from a key, so y is random given (s, a).
- Prediction: the online critic, min over heads (refresh: td3_bc_bca.py:252-256).
- Score: |y - q| / (max(eta, 1e-6) · u_f) at refresh (calibration/reference.py:164-175). u_f is the unit frozen at the
  refresh.
- In shift data `done` is always False, because terminals are False (learners.py docstring and `_transition`).

**What it is not.** The score is the absolute one-step residual against the critic's own target networks. It is not
|q - Q^pi| (ALGORITHMS.md:693-710). Two consequences for the mechanics check:

- **Coverage** must be checked on the residual BCA actually calibrates (one draw of target noise per row).
- **"Width rises where the exact critic error is larger"** is a separate question: a rank correlation between U and the
  exact error |q(s, a) - Q^pi(s, a)|.
  - Q^pi comes from `learners.exact_q` (learners.py:413-434), with pi the host actor, using the same q = min(Q1, Q2)
    that BCA scores.
  - The scale fit learns the spread of y - q, and that spread includes the spread of the target noise. So also report
    the rank correlation of U with `truth.noise_map` (truth.py:293). That quantity is a control with no information
    about critic error (truth.py docstring). Without it, a width that only tracks target noise could be mistaken for a
    width that tracks critic error.

**Measuring coverage with BCA's own functions.** For a held-out batch H (a normalised `Transition`, same normalisation
as training), after a band fit and refresh, compute:

```
y     = P.native_target(args, models, state.native, H, fresh_key, 1.0)           # td3_bc_bca.py:146
q     = models[1].apply(state.native.critic.params, H.obs, H.action).min(-1)     # as refresh, td3_bc_bca.py:252-256
U     = P.bc_readout(models, state, H, blend).width                               # td3_bc_bca.py:283 -> dose.py:65-93
cover = calibration.network.hard_coverage(y, q, U)                                # network.py:40-41, ties covered
```

- `bc_readout(...).width` is exactly R · u_f · max(eta_f, 1e-6) (dose.py:75-76). It is 0 if the reference is not ready
  (dose.py:90), so check `posterior_inputs_valid` and `posterior_certified` from `refresh` (td3_bc_bca.py:275-278).
- Use a fresh key for the held-out target noise, not the refresh key (fold_in(key, 1213156420), td3_bc_bca.py:249).
- **Which rows count as held out.** Held-out rows must never have been used by the critic, the scale fit or the bank.
  The evaluation split (separate episodes) qualifies. Training rows do not: both the critic and the scale were fitted
  on them, so coverage there is inflated.
- **Rows inside the gap.** With the gap removal, the bank has no gap commands. Held-out rows whose command lies in the
  gap are therefore not exchangeable with the bank. Report them as a separate group even in the no-tilt control.
- **Clustering.** Rows within one episode are dependent. Get standard errors by resampling whole episodes.

**What coverage to expect (for the pass marks).** The threshold is credible at 95% that coverage is at least 90%. So
average coverage is above 90%. With equal weights and continuous scores, conditional coverage across banks is
Beta(k, n + 1 - k), with k the selected rank (computed exactly):

| bank rows n | mean coverage | SD across banks | 5th percentile |
|---|---|---|---|
| 300 | 0.927 | 0.015 | 0.901 |
| 500 | 0.922 | 0.012 | 0.902 |
| 850 (1,000 minus a 15% gap) | 0.917 | 0.0095 | 0.900 |
| 1,000 | 0.915 | 0.0088 | 0.900 |

- With weights, use the Kish effective size n_eff in place of n as an approximation. `calibrate` returns n_eff, and it
  is stored in `FrozenReference.n_eff`.
- When the weights are scaled so their bank mean is 1, the exact test mass is E_test[w] = E_bank[w²], about n / n_eff.
  So a usable effective size and a small test mass are the same condition.

**A forecast that needs no tilt run.** Only the law of c changes. So unweighted coverage under any tilt is
∫ cov(c) p_test(c) dc, where cov(c) is coverage given c measured on no-tilt held-out rows. For steps outside the gap,
also multiply by the share of steps outside the gap at each c.
- The scale network sees c: it is observation index 10 (thruster_chain.py:70), and the calibrator reads the full
  normalised observation (network.py:16-31).
- If the scale already absorbs the c-dependence, cov(c) will be flat and unweighted BCA will not fall short under
  tilt. In that case the step-4 expectation "unweighted falls short" fails for a known reason.
- Plot cov(c) in step 3 before writing that expectation as a pass mark.

## (b) Weights and test mass: not supported end to end

**Where the weights get lost.**
- `wbcp.calibrate(scores, rng, weights=None, test_mass=None, *, alpha, beta, draws)` (calibration/wbcp.py:57-117) fully
  supports per-row weights and a scalar test mass:
  - It validates them (wbcp.py:66-77).
  - It rescales both by max(weights), since only ratios matter (wbcp.py:83-85; a common factor cancels, wbcp.py:13-14).
  - The empirical quantile lambda_hat uses the weights but not the test mass (wbcp.py:89-102).
  - The posterior gives the test atom mass test_mass · E_{n+1} (wbcp.py:43-54).
- `WBCPConfig` has only alpha, credibility and draws (reference.py:16-28).
- `freeze_reference` calls `wbcp.calibrate(scores, rng, alpha=..., beta=..., draws=...)` with no weights
  (reference.py:176-179), and its docstring says "unit weights".
- `td3_bc_bca.refresh` calls `freeze_reference` with no weights argument (td3_bc_bca.py:263-270).
- ALGORITHMS.md:88-89 says so explicitly: "WBCP weights are uniform in the current design; a likelihood-ratio study
  would supply them as a separately declared change."

**A per-row test mass is not handled anywhere.**
- `calibrate` takes a single scalar test mass.
- `FrozenReference.threshold` must be a scalar (reference.py:34, and `reference_valid` raises otherwise,
  reference.py:205-206).
- `frozen_level_dose` refuses a non-scalar threshold (dose.py:72).
- The module defines the test mass as wbar = E_test[w], a single number (wbcp.py:13-14). The earlier D4RL work used it
  the same way: the exact test mass E_test[w*] = mean(a²)/mean(a)², and for the estimated arm wbar = the mean of the
  estimated weights over unlabelled test rows (experiments/wbcp/d4rl_benchmark.py:24-25, 44-52, 459-464).

**Minimal faithful route (recommended).** Re-weight after an unchanged refresh:

1. Run `P.refresh(...)` unchanged. It freezes psi_f and u_f and returns a valid reference.
2. Recompute the bank scores exactly as refresh did: the same key fold_in(key, 1213156420) for the target noise
   (td3_bc_bca.py:249), the same float32-to-float64 path as reference.py:164-175, and `reference.positive_scale`.
3. Call `wbcp.calibrate(scores, np.random.default_rng(np.asarray(fold_in(key, 1347375956), np.uint32).ravel()),
   weights, test_mass, alpha=0.1, beta=0.95, draws=1000)`. This uses the same posterior random stream as
   td3_bc_bca.py:268 and reference.py:177.
4. `reference._replace(threshold=, lambda_hat=, lambda_hpd=, n_eff=)` as float32. Keep the float32 overflow guard of
   reference.py:180-183. `reference_valid` still holds because threshold = max(lambda_hat, lambda_hpd)
   (wbcp.py:117, reference.py:216-217).
5. Identity test: weights = ones and test_mass = 1.0 reproduce the unweighted reference bit for bit. This was confirmed
   at the `calibrate` level (section 6).

This keeps BCA's signal unchanged:
- The scale fit stays unweighted. Its Bayesian-bootstrap masses are a separate measure (ALGORITHMS.md:85-89), so the
  shape eta is the same.
- Weighting changes only R. The dose is still read by `bc_readout` (td3_bc_bca.py:283-288).

**Exact weights for this design.**
- Bank row j has weight w_j = p_test(c_j) / p_train(c_j). Read c from `infos/state[:, 10]` (exact float64,
  dataset.py:21). The removal constant Z cancels because it multiplies every weight and the test mass alike
  (wbcp.py:13-14).
- **No gap (full-data control):** test mass = E_test[w] = ∫ p_test(c)² / p_train(c) dc, by one-dimensional quadrature.
  Episodes all have 100 steps, so the c-law of steps equals the c-law of episodes.
- **With the gap:** the band is about commands outside the gap, so the test law is "test steps outside the gap".
  - Test mass = E_test|outside[p_test(c) / p_train(c)].
  - This needs the c-dependent share of steps in the gap, which has no closed form. Compute it as the mean of
    p_test(c) / p_train(c) over steps outside the gap in a large independent simulated test sample (not the scored
    test steps), and report its standard error. This mirrors the "Ts" sample of d4rl_benchmark.py.
- **Inside the gap:** the ratio is infinite, so those steps are refused by rule (section c).
- **Estimated weights:** a classifier on c (or on the observation) trained on unlabelled draws from both laws; wbar is
  the mean of the estimated weights over unlabelled test rows. The pattern is in d4rl_benchmark.py:459-464 and
  `common_scale` at :320-323. A classifier that does not see the command cannot see an action gap, so the
  estimated-weight arm will never refuse gap steps. State this before the run.

**Choosing the tilt strengths (certification limit).**
- The test atom must stay at or below 10% of the posterior mass in at least 95% of draws. Roughly, a finite threshold
  needs test_mass <= alpha / (1 - alpha) · Σ_j w_j / (-ln(1 - beta)) = 0.037 · Σ_j w_j.
- With weights scaled to a bank mean of 1, that is test_mass <= 0.037 n. For n = 850 the limit is about 31. The
  synthetic check found the first refusal at 32.
- The threshold climbs steeply near the limit. In the synthetic check it rose from 4.11 at test mass 1 to 4.26 at 8,
  4.80 at 16 and 8.30 at 30.
- With a usable effective size (n_eff >= n/3, so a test mass of about 3), the limit is far away and R rises only
  modestly.

**What scalar weighting can and cannot do to the dose (step 5).**
- One weighted R replaces one unweighted R. Every row's width scales by the same factor, so the ranking of rows by dose
  is unchanged (ALGORITHMS.md:95-97, 712-713). Weighted and unweighted BCA differ only in the amount of caution, not in
  where it goes. That belongs in the step-5 expectation.
- Variant, not recommended for a first pass: a per-row test mass w(c_i) for the training row being dosed would give a
  per-row R(c_i), and so a placement that depends on c.
  - It needs no edit to BCA. Compute U_i = R(c_i) · u_f · max(eta_i, 1e-6) outside BCA and pass it to
    `level_critic_dose` (dose.py:23), which accepts a vector of widths. R depends only on c, so a table per training
    episode is enough, refreshed at each refresh.
  - The co-trained loop would then need the batch row ids (`bca_train_step` draws them at learners.py:1151).
  - R is non-decreasing in test mass with the same random draws, so rounding test mass up on a grid errs on the safe
    side.
  - Rows whose w(c_i) passes the certification limit would get R = +infinity, and so dose 1. That is no caution on
    exactly the strong-current rows.
- Context: entry 16 of EXPERIMENTS.md, from the earlier ThrusterChain study, stopped before BCA because "caution of
  BCA's kind cannot reach" the strong-current loss. Step 5's room check may find no room again.

## (c) What "cannot certify" (R = +infinity) does today, and how to report it

**Code path.**
- `crossings` returns +inf for a draw whose test atom alone exceeds alpha (wbcp.py:54). R = +inf when more than 5% of
  draws do so (wbcp.py:115-117), and `certified` is False (wbcp.py:34-36).
- `freeze_reference` stores the +inf reference as valid and ready. It only rejects a finite threshold that overflows
  float32 (reference.py:180-188).
- So `refresh` commits it: `posterior_inputs_valid` is True and `posterior_certified` is False
  (td3_bc_bca.py:271-278).
- `frozen_level_dose`: width = inf · scale = inf (dose.py:76), and validity explicitly allows +inf (dose.py:86).
- `level_critic_dose`: support requires a finite width (dose.py:52). Unsupported rows get width 0 in the ratio, so
  ratio 0 and dose 1 (dose.py:53-60).
- This is documented: dose.py:66-69 and ALGORITHMS.md:338 ("valid but unusable row: m_i = 1").
- **Net effect:** a run whose threshold cannot certify silently becomes plain TD3+BC until the next refresh.
  `bc_support_fraction` = 0 is the only trace in the update metrics (td3_bc_bca.py:316-318). IQL handles the same case
  the other way, giving weight 0 (ALGORITHMS.md:348).
- Synthetic confirmation (section 6): threshold +inf gives widths [inf, inf, inf], doses [1, 1, 1] and support False.

**Where it matters here.**
- Training and bank rows never contain gap commands. So exact weights are finite on every row BCA doses, and refusal in
  the gap never reaches the dose. It is purely a coverage-reporting matter for test steps.
- `calibrate` rejects an infinite test mass with ValueError (wbcp.py:76-77, confirmed). The caller must therefore set
  "refused" for gap steps itself rather than pass w = inf.

**Simplest honest report, per arm and tilt:**
- **Steps outside the gap:** coverage = covered / (covered + not covered), with an episode-resampled interval, plus the
  share refused. That share should be 0.
- **Steps inside the gap:** share refused, plus coverage among the steps that were not refused.
  - Exact-weighted: 100% refused, and say so as a fact of construction. The ratio is infinite there because we removed
    those commands, so this checks the bookkeeping. It is not a discovery by BCA.
  - Unweighted and estimated-weighted never refuse. Their coverage inside the gap is the informative number: it shows
    whether a finite band is falsely confident on the unseen common command.
- Never count a refused step as covered. The earlier D4RL benchmark counted abstentions as risk 0
  (d4rl_benchmark.py:33, 529). Do not copy that convention here.
- If a whole tilt cannot certify (scalar R = +inf), report "the bank could not certify this tilt" for all of its steps.
- For step 5, also count refreshes with `posterior_certified` = False and batches with `bc_support_fraction` < 1. Those
  are where a weighted arm fell back to plain TD3+BC.

## (e) Fitting the band on a frozen critic, without an actor

**`learners.bca_band` does exactly this** (learners.py:1253-1279):
- `make_args` and `bca_config` (learners.py:1083-1088, the configs/td3_bc.yaml BCA settings: miscoverage 0.1,
  credibility 0.95, 1,000 draws, blend 0.5, scale learning rate 0.001, slope 20, width penalty 0.005, unit moving
  average 0.99).
- `bca_initial_state(args, config, native=host_state)` (learners.py:1091-1103). This is td3_bc_bca.initialize without
  its D4RL configuration checks at td3_bc_bca.py:115-124, which reject a non-D4RL dataset. The Calibrator is
  state-dependent on normalised inputs.
- A `jax.lax.scan` of `P.fit_scale` over `band_fit_steps` = 20,000 batches of 256 kept training rows (learners.py:146,
  1262-1271). `fit_scale` only updates the calibrator and the unit (td3_bc_bca.py:222-226).
- One `P.refresh` on `prepared.bank` (learners.py:1273). `refresh` only replaces `posterior` (td3_bc_bca.py:272).
- A check that the host state is unchanged by hash (learners.py:1276-1277).

No actor update runs anywhere in this path.

**What shift_iw needs.**
- `bca_band` calls `require_approval` (learners.py:1073-1080). That check is bound to the one official pre-registration
  under runs/shift (`_bound_to_official_prereg`, learners.py:986-1016, with the runs-root check at :998). It also
  returns only doses, not the state.
- So shift_iw should restate the same steps, roughly 20 lines, behind its own gate (pass marks written and shown), and
  return the fitted `State`.
- The ungated pieces it can import are `learners.make_args`, `bca_config`, `bca_initial_state`, `rng_np`,
  `tree_sha256`, `agent_template` and `load_tree`, plus `P.fit_scale` and `P.refresh`.
- **Needs:**
  - The full TD3+BC agent state, not just the actor and critic parameters. `native_target` reads the target actor and
    target critic (td3_bc_bca.py:149-155), and `refresh` reads the online critic.
  - Normalised training rows for the fit and normalised bank rows for the refresh (`learners.prepare`,
    learners.py:251-274).
  - Held-out rows for coverage (section a).
- **Weighted variant:** the same fitted state, then the re-weighting in section (b). The scale fit is shared, so
  unweighted, exact-weighted and estimated-weighted bands differ only in R. That is a clean paired comparison on one fit
  per seed.
- **Caveat:** a band fitted on a frozen final critic is what a refresh at that critic would produce. It is not the same
  object as the co-trained BCA arm's band, whose calibrator moved with the critic (fitting from update 1, refreshes
  every 5,000 updates after 10,000). That is fine for the mechanics and coverage checks, but say so.

## 6. Synthetic check (small arrays, CPU 4, no ThrusterChain data)

Script: scratchpad scout_synth.py (not kept in the repository). Results:

- `calibrate` with weights = ones and test_mass = 1.0 is bit-identical to unweighted, in both threshold and posterior
  crossings.
- Multiplying the weights and the test mass by 8 leaves the threshold bit-identical.
- `calibrate(..., test_mass=inf)` raises ValueError: "weighted calibration requires a finite positive test_mass".
- n = 850 unit weights: the threshold is non-decreasing in test mass (4.105 at 1, 4.257 at 8, 4.796 at 16, 8.304 at 30).
  It becomes +inf from test mass 32. The rule of thumb gives 31.5.
- `frozen_level_dose` with R = +inf, u = 2 and eta = [0.5, 1, 2] gives widths [inf, inf, inf], doses [1, 1, 1], support
  False, and inputs valid. With R = 1.3 it gives widths [1.3, 2.6, 5.2] and doses [1.197, 1.283, 1.361]. When not ready
  it gives widths 0 and doses 1.
- `level_critic_dose` with per-row widths from thresholds [1.3, 1.3, inf] gives doses [1.197, 1.283, 1.0]. So a per-row
  threshold works without editing BCA.
