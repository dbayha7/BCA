# Kill rule: harm-dose ladder (written 2026-10-14, before any result of this test exists)

EXPLORATORY, designed after seeing seeds 0-4. Seeds 0-4 only. No training. Proposed, not yet run.

## Test
Stall strength sigma in {-1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5, 3}: thrust past 80% = 0.8 - sigma (|a| - 0.8)
(thruster_chain.Physics(stall=True, stall_slope=sigma); sigma = -1 is the no-stall world, 1.5 the stalling world,
3 the pre-registered retune).
Evidence: one-step outcomes recomputed exactly under sigma at held-out rows from their logged states
(thruster_chain.dynamics); rewards do not depend on sigma. Critic: the no-stall gate critic (primary; never saw a
stall outcome); the stalling gate critic as sensitivity. Evidence rows, primary: held-out calibration-split rows past
t_lo (14-28 per seed) plus the bank; rich: removed full-throttle training rows.
Decision per seed and sigma: step 1's frozen certificate (margin 0.10). Value per seed and sigma: points of 'clip past
t_lo' minus 'never' at 200 strong-current starts (no-stall world's 'none' actor primary; stalling actor sensitivity),
one common points scale (the stalling world's anchors).
Ladder net of a rule = sum over sigma of the seed-mean realised value of its decisions. Oracle: per sigma, clip iff
the seed-mean value is positive. Constant rules: clip always, never.

## Shelve if EITHER holds (primary evidence)
1. BCA's ladder net is below 80% of the oracle's.
2. BCA's ladder net is less than 10 points above the better constant rule.

## Reported, not deciding
Rich evidence; plain Mann-Whitney; plain split-conformal gap (lambda_hat only) > 0.10; the sigma where BCA switches
on versus the sigma where clipping starts to pay; BCA's height R_tail - R_rest against sigma.

## Triage (selected, rank 3 of 5)
Written 2026-10-14 22:17 UTC by the triage step, before any result of this route exists (runs/shift_explore/routes/ does not exist; this folder held only the proposer's rule above, written before step 1's first evaluation output at 22:07). Seen before writing: step 1's finished evaluation (runs/shift_explore/restructure/step1/logs/evaluate.log: verdict GO, net 106.1; per seed-world statistic, height, Mann-Whitney p). This route uses none of step 1's critics or scores.

Sharpened, fixed now. The proposer's rule 2 is replaced by rule 2' because sigma = -1 and 1.5 are the two worlds
already answered by step 1, and they would dominate a 9-step sum. Shelve if EITHER holds (primary evidence, no-stall
gate critic, seeds 0-4):
1. BCA's ladder net over all 9 sigma values is below 80% of the oracle's ladder net (unchanged).
2'. Over the 7 sigma values other than -1 and 1.5, BCA's ladder net is less than 10 points above the better of the two
   constant rules (always clip, never) over the same 7 values.
Definitions: value v(sigma, seed) = points of 'clip past t_lo' minus 'never' at the first 200 strong-current starts in
world sigma, no-stall 'none' actor, stalling-world anchors. A rule's ladder net = sum over sigma of the seed-mean of
d(sigma, seed) x v(sigma, seed), with d = 1 when the rule switches on. Oracle: per sigma, clip in every seed iff the
seed-mean v is positive.
Plumbing stops (not results): at sigma = -1 and 1.5 the recomputed one-step outcomes must reproduce the matching log's
logged next states to 1e-9, and the rollouts must reproduce prespec.json deployment.seen_design_values to 1e-9.
Reported only: rich evidence; the stalling gate critic and stalling actor; one-sided Mann-Whitney p < 0.05 and the
plain split-conformal gap (lambda_hat only) > 0.10 as rival switches (if the rank test's ladder net is within 5 of
BCA's, the credit goes to reading outcomes); the switch-on sigma against the break-even sigma; the height by sigma.
CPU cap 45 minutes (estimate 20).

## Run-time precision (added 2026-10-14 by the route tester, before any score, outcome or rollout of this route was computed)
Stricter or equal only. No threshold, rule, group definition or pass mark above is changed. Not read before writing
this: the sibling route's results (runs/shift_explore/routes/heldout_calibration_rows/results.json, run.log).
- Data, primary: the no-stall world's gate log, experiments.shift.acceptance.load_seed(p.runs, p.config, variant,
  "no_stall", seed) (regenerated and hash-checked); pg = sd.prepared("gap", learn); t_lo = step1lib.gap_rule on
  pg.actions, which must equal prespec.json groups.expected_from_design["no_stall/seed<k>"] (plumbing stop).
- Full-throttle group (primary): every calibration-split log row (infos/split == 1) whose command is outside the
  removed ball (not sd.gap.contains) and has step1lib.throttle > t_lo, bank rows past t_lo included, ascending log
  row. Rest: pg.bank rows with throttle <= t_lo. None of these rows is a gate-critic training row (checked; stop).
- World sigma: phys = thruster_chain.Physics(stall=True, stall_slope=sigma). Recomputed outcome of a row:
  x' = thruster_chain.dynamics(infos/state, float64(actions), phys)[0]; next observation = (float32(x') - obs_mean) /
  obs_std (pg's normalisation); obs, action, logged float32 reward and terminal unchanged. Rows of the rest group are
  recomputed the same way at every sigma.
- Score: step 1's evaluate.overestimates (imported unchanged) with the no-stall gate critic
  acceptance.get_critic(p, sd, "gap", "stall_aware", train=False) and its fixed controller
  learners.FixedController("stall_aware", pg.physics, pg.obs_mean, pg.obs_std). Gamma 0.98 (prespec).
- Certificate: step1lib.certificate(tail, rest, [20261015, 9970, seed, 2], prespec.json "certificate"), the same
  random tags at every sigma (common random numbers), with step1lib.load_prespec() and verify_manifest() checked
  first. d(sigma, seed) = certificate on. A missing or failed critic: d = 0 at every sigma for that seed.
- Values: rollouts exactly as step 1's evaluate.rollout (restated to return the per-start returns J): the no-stall
  world's saved 'none' actor (evaluate.none_actor("no_stall", seed), SHA-256 checked), policy
  learners.mlp_policy(actor, no-stall pg, "none"); starts thruster_chain.bank_starts("strong_current", 1000)[:200];
  100 steps in phys; mode 'never' and mode 'on' (when max_k |a_k| > the no-stall t_lo, clip each thruster to the
  no-stall pg.actions 0.5% / 99.5% quantiles). Points = truth.points(J, anchors_of(p, variant, "stall")
  ["strong_current"]), mean over the 200 starts. v(sigma, seed) = points(on) - points(never).
- Plumbing stops (any failure stops the run; no verdict is then given):
  (a) no-stall log, sigma = -1: max |x' - infos/next_state| <= 1e-9 over the full-throttle and rest rows;
  (b) stalling log (load_seed(..., "stall", seed)), sigma = 1.5: the same over its own full-throttle and rest rows
      (its own t_lo, which must equal prespec's "stall/seed<k>");
  (c) at every sigma, the rest rows' recomputed next states equal infos/next_state to 1e-9, and every recomputed
      reward rounded to float32 equals the logged reward;
  (d) rollouts: no-stall actor, sigma = -1, no-stall anchors -> never/on equal seen_design_values
      "no_stall/seed<k>/strong_current" to 1e-9; stalling actor (evaluate.none_actor("stall", seed), stalling pg's
      t_lo and range), sigma = 1.5, stalling anchors -> never/on equal "stall/seed<k>/strong_current" to 1e-9.
- Decision arithmetic: m(sigma) = mean over seeds 0-4 of v; BCA(sigma) = mean over seeds of d * v; oracle(sigma) =
  max(m(sigma), 0); always(sigma) = m(sigma); never = 0. Rule 1 shelves if sum_9 BCA < 0.8 * sum_9 oracle. Rule 2'
  shelves if sum_7 BCA - max(sum_7 always, 0) < 10, sum_7 over sigma in {-0.5, -0.25, 0, 0.25, 0.5, 1, 3}. Exact
  float comparisons, no rounding.
- Reported only, not deciding: rich evidence (full-throttle group plus removed training rows past t_lo,
  sd.removals["gap"].removed_train_rows, tags [20261015, 9971, seed, 2]); the stalling gate critic on the stalling log
  (its own groups, tags [20261015, 9972, seed, 1]); the stalling actor's values (stalling pg's t_lo and range,
  stalling anchors); the rank-test switch (one-sided Mann-Whitney p < 0.05 on the primary scores) and the plain
  split-conformal switch (certificate's plain_gap = lambda_hat_tail - lambda_hat_rest > 0.10); per-seed switch-on
  sigma and the break-even sigma (first ladder sigma with m > 0, and the linear zero crossing of m); height and the
  certificate statistic by sigma.
- Script: this folder's run_route.py; results: runs/shift_explore/routes/harm_dose/results.json (write-once) and
  run.log. CPU cap 45 CPU-minutes (kill rule); taskset -c 2,3, nice 10, PYTHONDONTWRITEBYTECODE=1, JAX_PLATFORMS=cpu.
