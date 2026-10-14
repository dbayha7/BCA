# Kill rule: residuals of the actor's own critic (stalling-world screen)

Written 2026-10-14, before any outcome of this route was computed (only the host snapshot's structure was read:
scratchpad/routes/_feasibility/host_load_check.py). Exploratory, designed after seeing seeds 0-4. Gate seeds 0-4
only. No training. Outputs only under this folder and /home/user/bca/runs/shift_explore/routes/actor_value_residuals/.

## What is tested
Whether residuals of the co-trained actor's critic show the stall at full-throttle rows more strongly than the
fixed-controller critic's residuals. Only the stalling-world host is saved, so this is a one-world screen: a cross-world
use would need a no-stall host trained, which is only worth asking for if the signal is clearly stronger.

## Fixed settings
- Host: runs/shift/68d9e8779e148b7b/hosts/base/stall/gap/alpha5/plain/seed<k>/snapshot_100.msgpack, loaded with
  learners.agent_template(), file SHA-256 checked against record.json result.snapshots[-1].file_sha256.
- Host residual: min-head Q(s, a) - [r + 0.98 min-head Q_target(s', clip(actor_target(s'), -1, 1))], no noise.
- Fixed-controller residual: as in heldout_calibration_rows, gate gapped critic.
- Groups: survivors (throttle > t_lo; in-sample for both critics) and bank rows with throttle <= t_lo (held out for
  both critics).
- Statistic per seed: g = (mean_tail - mean_rest) / sd_rest, for each critic.

## Kill rule (any one shelves)
1. g_host < 1.5 x g_fixed in 2 or more of the 5 seeds.
2. g_host <= 0 in 2 or more of the 5 seeds.

## Reported, not deciding
alpha 2.5 host; snapshot_050; AUROC of tail versus rest for both critics.

## Triage
Triage 2026-10-14 22:17 UTC: NOT SELECTED this round; not run. No result of this route exists. Reason: a one-world screen. Only the stalling-world host is saved, so it cannot show a gain across worlds; using it
would need a no-stall host trained first. The proposer expects the actor critic's two errors to partly cancel.
