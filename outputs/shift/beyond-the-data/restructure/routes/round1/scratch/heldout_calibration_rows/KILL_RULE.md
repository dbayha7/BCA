# Kill rule: held-out calibration rows as BCA's full-throttle evidence

Written 2026-10-14, before any outcome (reward, next state, residual) of the rows below was computed. Exploratory,
designed after seeing seeds 0-4. Gate seeds 0-4 only. No training. Outputs only under this folder and
/home/user/bca/runs/shift_explore/routes/heldout_calibration_rows/.

## What is tested
BCA's calibration bank today is 5 rows per calibration episode (827-851 rows) and holds 0-2 full-throttle rows.
The same calibration episodes hold about 15,800-16,070 more rows outside the removed ball that no critic ever trained
on; 14-28 of them per seed-world are full-throttle rows past the gap (counted from commands only:
scratchpad/routes/_feasibility/count_rows_*.json:*.n_heldcal_tail). The route: give BCA's certificate those rows,
scored by the gate's own gapped critic, with no retraining.

## Fixed settings
- Critic: the gate's gapped fixed-controller critic (runs/shift/68d9e8779e148b7b/critics/base/<world>/stall_aware_gap/
  seed<k>/critic.msgpack, SHA-256 checked against record.json).
- Score: step1lib.overestimate: min-head Q(s, a) - [r + 0.98 * min-head Q(s', c(s'))], c = stall-aware controller,
  no noise.
- t_lo: step1lib.gap_rule on the gapped kept training commands (must equal step 1's prespec value).
- Full-throttle group: every calibration-split row outside the removed ball with throttle > t_lo (bank rows included).
- Rest: calibration-bank rows with throttle <= t_lo (identical to step 1's rest group).
- Certificate: step1lib.certificate with step 1's frozen settings (prespec.json: alpha 0.1, credibility 0.95,
  1,000 draws, 5% pairwise quantile, margin 0.10), rng tags [20261015, 9900, seed, w], w = 1 stall / 2 no-stall.
- NET: step1lib.deployment_net with the strong-current never/on points of prespec.json
  deployment.seen_design_values (no rollouts are needed; they do not depend on any critic).
- Shuffle: step1lib.shuffled_certificates, 200 permutations, permutation tags [20261015, 9901, seed, w].

## Kill rule (any one shelves the route)
1. The certificate is not on in all 5 stalling-world seeds and off in all 5 no-stall seeds.
2. NET is below 90.
3. The mean NET over the 200 outcome shuffles is above half of NET.

## Reported, not deciding
Plain one-sided Mann-Whitney test on the same scores; the height R_tail - R_rest; the same certificate with the
detection script's T4 rows left out of the full-throttle group (9-20 rows; Stage 2's tau hygiene); the same with the
rest = every held-out calibration row at or below t_lo; group sizes and means.

## Seen before writing
Only row counts and log positions (commands). Earlier, unrelated to these rows: in-sample survivor residuals
(CAL/e3c_tail.json), 6 bank full-throttle rows per world (CAL/e8_bank_tail.json), held-out residuals of REMOVED rows
in the trace test bed (TB/summary.json:E1). No residual of the 14-28 non-bank calibration rows has been computed.

## Triage (selected, rank 1 of 5)
Written 2026-10-14 22:17 UTC by the triage step, before any result of this route exists (runs/shift_explore/routes/ does not exist; this folder held only the proposer's rule above, written before step 1's first evaluation output at 22:07). Seen before writing: step 1's finished evaluation (runs/shift_explore/restructure/step1/logs/evaluate.log: verdict GO, net 106.1; per seed-world statistic, height, Mann-Whitney p). This route uses none of step 1's critics or scores.

The rule above is kept unchanged: it is already three fixed yes/no checks. Clarifications only, fixed now:
- Plumbing stop (not a result): if step1lib.gap_rule gives a t_lo or survivor count different from step 1's
  prespec.json value in any seed-world, stop and report; do not run the certificate.
- NET is read from prespec.json deployment.seen_design_values (strong current); only the certificate pattern decides it.
- Not a repeat of step 1: different evidence rows (the 14-28 non-bank calibration-split rows past t_lo, never used by
  step 1), the gate's own critic (no retraining), and both groups from calibration episodes (removes step 1's flagged
  different-episodes confound). Its result is not implied by step 1's GO: fewer rows and a different critic.
- CPU cap 30 minutes (estimate 15). Outputs only under this folder and runs/shift_explore/routes/heldout_calibration_rows/.

## Run-time precision (added 2026-10-14 22:22 UTC by the route tester, before any score of these rows was computed)
Stricter or equal only; no threshold, group or setting above is changed.
- Data: experiments.shift.acceptance.load_seed (regenerated log, checked against the gate's data record); pg =
  sd.prepared("gap", learn). Normalisation = pg.obs_mean / pg.obs_std (the gate critic's own).
- Full-throttle group: log rows r with infos/split == calibration, command outside the removed ball
  (not sd.gap.contains), and step1lib.throttle(command) > t_lo; ascending log-row order. Transitions built with
  learners._transition(log, rows, pg.obs_mean, pg.obs_std). Plumbing checks (stop, not a result): the bank rows past
  t_lo are a subset of this group; no row of the group is among the gate critic's training rows pg.rows; the group's
  size equals count_rows_*.json n_heldcal_tail + n_bank_tail.
- Rest group: pg.bank rows with throttle <= t_lo, in pg.bank_rows order (step 1's rest group).
- Gate critic: acceptance.get_critic(p, sd, "gap", "stall_aware", train=False) (SHA-256 checked against record.json
  result.file_sha256). A failed or missing critic: certificate off for that seed-world (counts against rule 1).
- Certificate: step1lib.certificate(tail, rest, [20261015, 9900, seed, w], cfg = prespec.json "certificate"), with
  step1lib.load_prespec() and step1lib.verify_manifest() checked first (the frozen step-1 code and settings).
- Shuffle: step1lib.shuffled_certificates(tail, rest, [20261015, 9900, seed, w], [20261015, 9901, seed, w], 200, cfg);
  shuffled nets and their mean: step1lib.shuffle_summary.
- NET: step1lib.deployment_net with points[world][seed] = prespec.json deployment.seen_design_values
  "<world>/seed<k>/strong_current" never / on.
- Rule 3 is read with NET as computed. If NET <= 0, rule 2 already shelves the route.
- Reported only: T4 variant (T4 log rows = runs/shift_explore/detect/seed<k>/<world>/pairs.npz rows with set == "T4",
  log_row) with certificate tags [20261015, 9902, seed, w]; rest = every held-out calibration row outside the ball
  at or below t_lo (bank included), tags [20261015, 9903, seed, w]; forced-on net; rank-test-switched net (p < 0.05);
  step 1's per seed-world statistic beside this one (runs/shift_explore/restructure/step1/results.json, read only).
- Script: scratchpad/routes/heldout_calibration_rows/run_route.py; results:
  runs/shift_explore/routes/heldout_calibration_rows/results.json (write-once) and run.log.
