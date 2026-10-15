# Kill rule: small-evidence head-to-head (written 2026-10-14, before any result of this test exists)

EXPLORATORY, designed after seeing seeds 0-4. Seeds 0-4 only. No training. Proposed, not yet run.

## Test
Held-out one-sided overestimates (noiseless form, step1lib.overestimate) of each world's gate critic. Full-throttle
group drawn from the removed full-throttle training rows (held out from that critic): n in {3, 5, 8, 12, 20, 30, 45},
400 random draws per n, per seed-world (rng [20261015, 9930, seed, w, n]). Rest: the bank rows at or below t_lo.
Rules: (B) step 1's frozen certificate (step1lib.certificate); (C) plain split-conformal gap
lambda_hat_tail - lambda_hat_rest > 0.10, no posterior; (R) one-sided Mann-Whitney p < 0.05; (B0) BCA at margin 0.
Expected net per rule and n = mean over seeds of P(on | stall) x gain_stall - P(on | no-stall) x loss_no_stall, with
gain and loss per seed from CAL/e12_gated_deploy.json (<k>_stall_strong_current.gated_minus_never) and
CAL/e14_ungated.json (<k>_strong_current.cost).
The realistic operating point (the 14-28 held-out calibration-split rows past t_lo) is the separate route
heldout_calibration_rows; it is read from that route's output when it exists, not recomputed here.

## Shelve "BCA's posterior handles thin evidence better than a plain test" if
at no n in the ladder does rule B's expected net exceed the best of C and R by >= 5 points.

## Also reported
The smallest n at which each rule's expected net reaches 90; if R beats B by >= 5 at n = 12-30 (the range
ThrusterChain's held-out evidence falls in), flag that step 1's decision rule loses to a textbook test there.

## Triage (selected, rank 4 of 5)
Written 2026-10-14 22:17 UTC by the triage step, before any result of this route exists (runs/shift_explore/routes/ does not exist; this folder held only the proposer's rule above, written before step 1's first evaluation output at 22:07). Seen before writing: step 1's finished evaluation (runs/shift_explore/restructure/step1/logs/evaluate.log: verdict GO, net 106.1; per seed-world statistic, height, Mann-Whitney p). This route uses none of step 1's critics or scores.
Why this matters now: in step 1 a plain p < 0.05 rank test would have switched on falsely in no-stall seed 2
(Mann-Whitney p = 0.032) while BCA's certificate stayed off (statistic +0.010). That is one seed-world; this route
measures whether the difference is systematic, and whether it comes from the posterior or only from the 0.10 margin.

Sharpened, fixed now. The proposer's rule ("at no n does B beat the best of C and R by 5 or more") lets the best of 7
ladder points decide; it is replaced by an average over the realistic evidence sizes. The n values are chosen from
row counts only: 14-28 held-out calibration rows past t_lo per seed-world (routes/_feasibility/count_rows_*.json) and
28-59 survivors. For each rival X in {C (plain split-conformal gap lambda_hat_tail - lambda_hat_rest > 0.10, no
posterior), R (one-sided Mann-Whitney p < 0.05)}:
  D_X = mean over n in {12, 20, 30, 45} of [expected net of B at n - expected net of X at n],
  expected net = mean over seeds 0-4 of P(on | stall) x gain_seed - P(on | no-stall) x loss_seed,
  gain and loss per seed from prespec.json deployment.seen_design_values (strong current; the step-1 code checks these
  against CAL/e12 and CAL/e14).
- Shelve the claim "BCA's posterior adds decision value over plain conformal" if D_C < 5.
- Shelve the claim "BCA's certificate beats a plain rank test on thin evidence" if D_R < 5.
- The route is shelved if both claims are shelved.
Evidence: full-throttle group = n rows drawn without replacement from the removed full-throttle training rows (held
out from the gate critic), 400 draws per n per seed-world, rng default_rng([20261015, 9930, seed, w, n]); score =
noiseless overestimate (step1lib.overestimate; = -deltanl_gap in TB/heldout_seed<k>_<world>.npz for seeds 0-2, one
forward pass for seeds 3-4); rest = bank rows with throttle <= t_lo. Rule B = step1lib.certificate with step 1's
frozen settings.
Reported only: every n in {3, 5, 8, 12, 20, 30, 45}; B0 (margin 0); the proposer's original rule; the smallest n at
which each rule's expected net reaches 90; a flag if R beats B by 5 or more at any n in 12-30.
CPU cap 60 minutes (estimate 40).

## Run-time precision (added 2026-10-14 22:44 UTC by the route tester, before any draw, score or certificate of this route was computed)
Stricter or equal only. No threshold, n value, rule or decision above is changed. Exploratory; gate seeds 0-4 only.
- Data: step 1's evaluate.gate_context(prespec) and experiments.shift.acceptance.load_seed(p.runs, p.config, variant,
  world, seed); pg = sd.prepared("gap", learn). t_lo = step1lib.gap_rule(step1lib.throttle(pg.actions)) with
  prespec groups.gap_rule; it must equal prespec groups.expected_from_design t_lo (<= 1e-12) and n_survivors, or
  PLUMBING STOP (no verdict).
- Pool (where the n full-throttle rows are drawn from): tracebed.shared_candidates(sds)[0] = the gate's removed
  training rows whose command is past the 80% knee on any thruster, at the log positions both worlds have, ascending
  (the "cand" group of scratchpad/restructure/testbed/heldout_seed<k>_<world>.npz). Plumbing stops: seeds 0-2, equal
  to the npz rows[pos_cand]; every pool row has throttle > t_lo; no pool row is among the gate critic's training rows
  (pg.rows).
- Score: the noiseless one-sided overestimate of the gate's gapped fixed-controller critic
  (acceptance.get_critic(p, sd, "gap", "stall_aware", train=False)), computed exactly as testbed heldout.py's
  "noiseless" form, score = -deltanl_gap. Seeds 0-2 use the npz values; the forward pass here must reproduce them
  at every pool and bank row (|difference| <= 1e-9) or PLUMBING STOP. Seeds 3-4 use the forward pass; it must
  reproduce the seed-3/4 npz -deltanl_gap at the bank rows (<= 1e-9) or PLUMBING STOP.
- Rest: the npz bank rows (rg.bank_rows) with throttle <= t_lo; must equal pg.bank_rows with throttle <= t_lo and
  step 1's n_rest (runs/shift_explore/restructure/step1/results.json) or PLUMBING STOP.
- Draws: g = numpy default_rng([20261015, 9930, seed, w, n]), w = 1 stalling / 2 no-stall; for j = 0..399 in order,
  idx = g.choice(pool size, n, replace=False); the full-throttle group is the pool scores at idx.
- Rule B (step 1's frozen certificate): tail posterior step1lib.posterior(tail, [20261015, 9931, seed, w, n, j+1, 1],
  prespec certificate cfg); rest posterior step1lib.posterior(rest, [20261015, 9931, seed, w, 2], cfg), computed
  once per seed-world and reused for every n and draw; q05 = step1lib.pairwise_quantile(tail crossings, rest
  crossings, 0.05); on iff q05 > 0.10. Equivalence with step1lib.certificate (same posteriors) is checked on
  synthetic numbers before the run (PLUMBING STOP if q05 or plain_gap differ).
- Rule B0: the same q05 > 0.
- Rule C: lambda_hat_tail - lambda_hat_rest > 0.10, lambda_hat = wbcp.calibrate(...).lambda_hat (the certificate's
  plain_gap: the empirical 90% quantile, the ceil(0.9 n)-th smallest score; no posterior). Reported only beside it,
  not deciding: C with the textbook finite-sample correction (the ceil(0.9 (n+1))-th smallest score, +inf when that
  exceeds n).
- Rule R: step1lib.rank_test(tail, rest)["p"] < 0.05 (scipy mannwhitneyu, alternative "greater", default method).
- Expected net of a rule at n: mean over seeds 0-4 of [P_on(stall, seed, n) x gain_seed - P_on(no_stall, seed, n) x
  loss_seed], P_on = share of the 400 draws on; gain_seed = on - never of prespec deployment.seen_design_values
  "stall/seed<k>/strong_current"; loss_seed = never - on of "no_stall/seed<k>/strong_current".
- Decision: D_C and D_R exactly as above (exact float comparison with 5, no rounding). Both shelved -> route shelved
  ("dead end"). One shelved -> that claim is shelved and the route is reported alive only for the other claim.
- A gate critic that is missing or failed: every rule off for that seed-world (it then cancels in D_X).
- Reported only: Monte Carlo standard errors (draw-level) of the nets and of D_X; per seed-world on-rates; the
  realistic operating point read (not recomputed) from runs/shift_explore/routes/heldout_calibration_rows/results.json
  and step 1's results.json.
- Script: this folder's run_route.py; results: runs/shift_explore/routes/small_evidence/results.json (write-once,
  .sha256) and run.log. taskset -c 2,3, nice 10, PYTHONDONTWRITEBYTECODE=1, JAX_PLATFORMS=cpu. CPU cap 60 minutes.
- Seen before writing this section: step 1's and heldout_calibration_rows' finished logs (verdicts, per seed-world
  q05 and Mann-Whitney p), and prespec certificate.seen_held_out_testbed (20 draws of k = 10, 30, 45 pool rows,
  seeds 0-2: stalling 5% bound 0.115 to 2.63, no-stall -0.13 to +0.067). No draw of this route was computed.
