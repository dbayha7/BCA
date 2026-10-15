# Kill rule: multi-step residuals at full-throttle rows

Written 2026-10-14, before any outcome of this route was computed (only log positions and commands were read:
scratchpad/routes/_feasibility/followers_*.json). Exploratory, designed after seeing seeds 0-4. Gate seeds 0-4 only.
No training. Outputs only under this folder and /home/user/bca/runs/shift_explore/routes/multi_step_residuals/.

## What is tested
Whether scoring a full-throttle row by its n-step overestimate along the recorded episode separates the stalling world
more strongly than the one-step overestimate BCA would use.

## Fixed settings
- Critic: the gate's gapped fixed-controller critic. c = stall-aware controller, no noise, gamma 0.98.
- n-step overestimate of a kept training row at (episode e, step t) whose next n - 1 steps of episode e are also kept
  training rows: min-head Q(s_t, a_t) - [sum_{i<n} 0.98^i r_{t+i} + 0.98^n min-head Q(s'_{t+n-1}, c(s'_{t+n-1}))],
  s' = the stored next observation.
- Groups: survivors (throttle > t_lo) with the n-step window available; rest = 5,000 kept training rows with
  throttle <= t_lo and the window available, drawn with rng [20261015, 9920, seed, w]. Both in-sample for the critic.
- n = 2 primary, n = 1 the reference, n = 3 reported.
- Statistic per seed-world: g_n = (mean_tail - mean_rest) / sd_rest.

## Kill rule (any one shelves)
1. Stalling world: g_2 < 1.5 x g_1 in 2 or more of the 5 seeds.
2. No-stall world: |g_2| > 0.5 in 2 or more of the 5 seeds.
3. Fewer than 9 survivors have the 2-step window in any seed-world.

## Reported, not deciding
n = 3; the held-out calibration-split full-throttle rows with available windows (4-13 rows at n = 2).

## Triage
Triage 2026-10-14 22:17 UTC: NOT SELECTED this round; not run. No result of this route exists. Reason: dead by construction. The one-step target already contains the next state's value, so the stall's cost is
already in the one-step score; extra steps add the recording controller's in-range noise. The log never holds a repeated
stall (0 consecutive survivor pairs in 10 of 10 seed-worlds: routes/_feasibility/count_rows_*.json).
