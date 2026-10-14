# Kill rule: BCA corrects the critic (closed-form weighted last-layer posterior)

Written 2026-10-14, before any outcome of this route was computed. Exploratory, designed after seeing seeds 0-4.
Gate seeds 0-4 only. No gradient training: one weighted ridge solve per head. Outputs only under this folder and
/home/user/bca/runs/shift_explore/routes/last_layer_critic_correction/.

## What is tested
Whether a Bayesian update of the frozen gapped critic's last layer, with the surviving full-throttle training rows
up-weighted, corrects the critic's false preference for past-80% commands where it matters (strong current) without
removing the truly good past-80% preferences of the no-stall world.

## Fixed settings
- Features phi_h as in posterior_scaled_height (gate critic, 42,500 gapped training rows, plus constant 1).
- Targets y_i = r_i + 0.98 min-head Q(s'_i, c(s'_i)) = train_q + train_delta (scratchpad/restructure/calibration/
  delta_seed<k>_<world>.npz).
- Weights: 1 for rows with throttle <= t_lo; W for the survivors (throttle > t_lo), W = (rho/(1 - rho)) x
  n_rest / n_tail. Primary rho = 0.10. Reported: rho in {0.02, 0.05, 0.25}.
- Per head: w_h' = (Phi_h' M Phi_h + lam_h I)^-1 (Phi_h' M y + lam_h w_h0), lam_h = 1e-4 x mean diagonal of
  Phi_h' Phi_h, w_h0 = the trained last layer (with bias). Corrected Q_h' = phi_h . w_h'.
- Control: the same solve with W = 1 (refit without up-weighting); every comparison below is against this control.

## Kill rule (pooled over seeds 0-4, rho = 0.10; any one shelves)
1. Stalling world: fewer than 50% of T3 harmful pairs (labels.npz harmful) have corrected Q1'(command) - Q1'(cap)
   <= 0, OR the T3a share of states whose best cap under Q1' alone is past 80% (e6 grid) is above 0.385.
2. No-stall world: among T3 pairs past 80% with true_pref > 0.05, the share keeping a positive corrected preference
   is below 0.8 x the control's share.
3. Validation on rows the correction never saw: in the stalling world, the mean overestimate of the held-out
   calibration-split full-throttle rows (heldout_calibration_rows groups) does not fall by at least half of its gap to
   the rest's mean, relative to the control; OR in either world the mean overestimate of bank rows at or below t_lo
   moves by more than 0.05 value units relative to the control.

## Reported, not deciding
T1 harmful flip share; the rho ladder; per-seed values; held-out residual of no-stall full-throttle rows.

## Seen before writing
runs/shift_explore/trace/pooled_v3.json (adding 0.5-5% distinct full-throttle rows left 96.5-71.5 of 99.5 harm
points; 10% left 23.4). No corrected critic has been computed.

## Triage
Triage 2026-10-14 22:17 UTC: NOT SELECTED this round; not run. No result of this route exists. Merged with survivor-weighted-critic (both correct the critic with the survivors' outcomes). Reason: existing evidence
predicts failure (adding 2-5% distinct real full-throttle rows still left 85.8-71.5 of 99.5 harm points:
runs/shift_explore/trace/pooled_v3.json:map.rows[*].harm_strong_current_mean), and strong-current states have 9-13x the
training leverage (restructure/action_aware/summary.json), so a fit at recorded currents is not expected to carry there.
BCA's share would be a generic weighted ridge with a hand-set weight.
