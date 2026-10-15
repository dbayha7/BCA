# Kill rule: run-time outcome monitor (written 2026-10-14, before any result of this test exists)

EXPLORATORY, designed after seeing seeds 0-4. Seeds 0-4 only. No training. Proposed, not yet run.

## Test
Four worlds, all built from saved seed 0-4 files (no new log is recorded):
- S: stalling log, stalling physics at deployment, Stage 2 'none' actor of the stalling world.
- N: no-stall log, no-stall physics, 'none' actor of the no-stall world.
- M3 "harm only beyond the log": stall only when current > 0.7. The log (currents 0.1-0.7) is exactly the no-stall
  log; strong-current deployment (currents 0.7-1.1) is exactly stalling physics. Actor: the no-stall 'none' actor.
- M4 "harm only inside the log": stall only when current <= 0.7. Log = the stalling log; strong-current deployment =
  no-stall physics. Actor: the stalling 'none' actor.
200 strong-current starts (bank order), 100 steps each. Before each episode the monitor computes BCA's certificate
(step 1's frozen rule: step1lib.certificate, alpha 0.1, credibility 0.95, 1,000 draws, margin 0.10) on one-sided
overestimates of the matching gate critic (noiseless next command, min of heads) at the steps executed so far:
full-throttle group = executed steps with throttle > t_lo; rest = executed steps at or below t_lo (primary).
Certificate on -> clip commands past t_lo to the recorded range for this and every later episode (latch).
Before episode 1 there is no evidence: off.

## Shelve the route if ANY of these holds (pooled over seeds 0-4)
1. The certificate switches on in any of the 10 no-stall-physics seed-worlds (N, M4) within the 200 episodes.
2. It is not on by the start of episode 6 in any of the 10 stalling-physics seed-worlds (S, M3).
3. Four-world net (points gained over 'never' in S and M3, minus points lost against 'never' in N and M4) is below
   85% of the oracle's (clip past t_lo from episode 1 exactly where the deployment physics stalls).

## Reported, not deciding
Plain one-sided Mann-Whitney (p < 0.05) on the same steps after each episode; rest = the offline bank (secondary);
outcome-permuted control; clip-always; the offline step-1 certificate. If the rank test's net is within 5 points of
BCA's, the credit goes to "reading outcomes", not to the Bayesian posterior.

## Triage (selected, rank 2 of 5; merged with the proposal "online-outcome-monitor")
Written 2026-10-14 22:17 UTC by the triage step, before any result of this route exists (runs/shift_explore/routes/ does not exist; this folder held only the proposer's rule above, written before step 1's first evaluation output at 22:07). Seen before writing: step 1's finished evaluation (runs/shift_explore/restructure/step1/logs/evaluate.log: verdict GO, net 106.1; per seed-world statistic, height, Mann-Whitney p). This route uses none of step 1's critics or scores.

The four-world rule above stays the decisive core. Added or fixed now:
- Looks: the certificate is recomputed after every episode 1..200 (per-episode mode, as written above). Primary
  rest group = executed steps with throttle <= t_lo in the same deployment.
- Decisive item 4 (from online-outcome-monitor): an outcome-permuted control. At every look the executed steps'
  scores are pooled and dealt back to the two groups at random (rng default_rng([20261015, 9940, seed, world_index,
  episode])), with the same latch. Shelve if its four-world net is at least half of the real four-world net.
- Four-world net = sum over the four worlds of the seed-mean (monitor points minus never points), strong current,
  200 starts, each world's deployment-physics anchors. Gains count in S and M3, losses in N and M4. The oracle clips
  from episode 1 exactly in S and M3 and never in N and M4.
- Plumbing stops (not results): the simulator must reproduce logged reward and next state at logged (state, command)
  pairs to 1e-9, and the 'never' and 'on' rollouts in S and N must reproduce prespec.json
  deployment.seen_design_values to 1e-9. If either fails, stop and report.
- Reported only: blocks of 10 episodes (fleet mode); rest = offline bank; one-sided Mann-Whitney p < 0.05 after each
  episode; clip-always; the offline step-1-style certificate; cost in N's recorded conditions. If the rank test's
  four-world net is within 5 points of BCA's, the credit goes to reading outcomes, not to the Bayesian posterior.
- CPU cap 90 minutes (estimate 50: about 2,000 certificate looks in N and M4 at up to 20,000 rows each).

## Precision added by the test runner (2026-10-14 ~22:30 UTC, before any code of this test ran and before any result)
Nothing above is loosened; these lines only fix details the rule left open.
- World index (for every rng tag): S = 1, N = 2, M3 = 3, M4 = 4. Log world (data, gap rule t_lo, recorded range,
  normalisation, 'none' actor, gate critic, stall-aware controller in the score): S stall, N no_stall, M3 no_stall,
  M4 stall. Deployment physics and points anchors: S stall, N no_stall, M3 stall, M4 no_stall. M3/M4 are run with
  an explicit current-dependent dynamics (stall iff current > 0.7, resp. iff current <= 0.7); the run refuses
  unless every log row has current <= 0.7 and every deployment state has current > 0.7 (so they are exact).
- Episodes = thruster_chain.bank_starts('strong_current', 1000)[:200] in bank order, episode e = start e-1.
- Executed step's group: float64 max_k |a_k| of the executed command a = clip(pi(x), -1, 1) compared with t_lo
  (> t_lo: full-throttle group; <= t_lo: rest), the same comparison as the clip rule.
- Score of an executed step: min_h Q(o, a) - [r + 0.98 min_h Q(o', c(o'))] with the log world's gate gapped
  stall-aware critic (A.get_critic(..., 'gap', 'stall_aware', train=False)), o, o' = log normalisation of float32-
  rounded x, x'; a as float32; r rounded to float32 as the log stores it; c = noiseless stall-aware controller of
  the log world; no terminal (as in the log).
- Real certificate at the look after episode e: step1lib.certificate(tail, rest, [20261015, 9930, seed, wi, e],
  prespec certificate settings: alpha 0.1, credibility 0.95, 1,000 draws, 5% posterior gap, margin 0.10).
  Permuted control: same posterior tags (common random numbers); pooled scores (tail then rest, episode order)
  permuted by default_rng([20261015, 9940, seed, wi, e]).permutation(n); first n_tail = full-throttle group.
  One permutation per look; its own latch.
- Latch: if the look after episode e is on, episodes e+1..200 run with the clip; episodes 1..e as 'never'.
  Monitor points of a seed-world = mean over the 200 episodes of the composed per-start points.
- Criterion (1) counts looks after episodes 1..200 (the look after episode 200 changes no episode but still
  counts as a switch-on: stricter). Criterion (2): on at some look after episode e <= 5.
- Criterion (3): four-world net < 0.85 x oracle four-world net. Criterion (4): permuted four-world net >=
  0.5 x real four-world net (if the real net is <= 0, criterion (3) already shelves).
- Plumbing: simulator reproduction on every row of both logs, all seeds (next state |diff| <= 1e-9 against
  infos/next_state; float32-rounded reward equal to the logged float32 reward within 1e-9). Rollout reproduction:
  S and N 'never' and 'on' points, strong current and recorded conditions, against prespec.json
  deployment.seen_design_values within 1e-9.
- Reported-only details: fleet mode = looks only after episodes 10, 20, ..., 200, same tags as the per-episode look
  at that episode; rest = offline bank = the log's calibration-bank rows (gate gapped) with t <= t_lo, gate-critic
  scores, posterior tags [20261015, 9950, seed, wi, e]; Mann-Whitney = step1lib.rank_test, latch at the first look
  with p < 0.05; clip-always = clip from episode 1 in all four worlds; offline certificate = step 1's saved on/off
  (runs/shift_explore/restructure/step1/results.json per_seed_world.*.certificate.on, read only), with M3 taking
  the no_stall value and M4 the stall value; N's recorded-conditions cost = the same monitor on N with the 200
  recorded-condition starts. If the CPU budget runs short, the reported-only items are cut first (fleet mode and
  Mann-Whitney are cheap; rest = offline bank may be run at fleet looks only; this would be stated).
