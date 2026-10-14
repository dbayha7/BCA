# Kill rule: BCA height scaled by the critic's last-layer posterior

Written 2026-10-14, before any outcome of this route was computed. Exploratory, designed after seeing seeds 0-4.
Gate seeds 0-4 only. No training (a closed-form matrix inverse on frozen features). Outputs only under this folder and
/home/user/bca/runs/shift_explore/routes/posterior_scaled_height/.

## What is tested
Whether scaling BCA's outcome-calibrated full-throttle height by the frozen critic's last-layer posterior standard
deviation sigma(s, a) (large where the critic extrapolates in state, e.g. strong current) makes the training-channel
penalty strong enough at strong current, where a single height calibrated at recorded currents is too small.

## Fixed settings
- sigma(s, a): per head h, features phi_h = the gate critic's penultimate ReLU outputs plus a constant 1, at the 42,500
  gapped training rows (scratchpad/restructure/calibration/data_seed<k>_<world>.npz: train_obs, train_act);
  G_h = Phi_h' Phi_h + lam_h I, lam_h = 1e-4 x mean diagonal of Phi_h' Phi_h; sigma = u_f x mean over h of
  sqrt(phi_h' G_h^-1 phi_h); u_f = Stage 2 uniform band.json unit. (AA/nl_posterior.py, c = 1e-4.)
- Rows and score: exactly the heldout_calibration_rows route's groups and overestimate e. Normalised score z = e / sigma.
- Thresholds: wbcp.calibrate (alpha 0.1, credibility 0.95, 1,000 draws, uniform weights), deployed threshold
  max(lambda_hat, lambda_hpd), tags [20261015, 9910, seed, w, 1] (full-throttle) and [..., 2] (rest).
- Width: U_sigma(s, a) = sigma(s, a) x [R_rest_z + f(t(a)) (R_tail_z - R_rest_z)], f(t) = clip((t - t_lo)/(1 - t_lo),
  0, 1). If R_tail_z is infinite in a seed-world, that seed-world counts as no caution (Q1 alone).
- Comparator (same rows): constant height U_c = R_rest_e + f(t(a)) (R_tail_e - R_rest_e) on the raw scores e.
- Control: shuffled sigma, i.e. sigma read at a state taken from a fixed permutation of the query states
  (rng [20261015, 9911, seed, w]), same command.
- Proxy: e6_bestresponse.py's rule unchanged: throttle-cap grid (0.70 ... 1.00), Q1 = gate critic head 1,
  P = [U(s, a_m) - U(s, a_ref)]_+, refs from pairs.npz, sets T1 and T3a (T3, t3_source 'actor'), k in {1, 2}.

## Kill rule (stalling world, certificate assumed on, pooled mean over seeds 0-4, k = 1; any one shelves)
1. T3a share of states whose best cap is past 80% is above 0.385 (matched range hinge, CAL/e6b_matched_hinge.json:
   pooled.T3a.best_past80).
2. That share is not at least 0.10 below the constant-height comparator's on the same rows.
3. The shuffled-sigma control's share is less than 0.05 above the sigma-scaled share (state dependence not the cause).
4. T1 share above 0.056 (matched hinge, same file, pooled.T1.best_past80).

## Reported, not deciding
k = 2; the three current bins; no-stall shares with the certificate forced on (cost if the gate were wrong); median
sigma ratio T3a/T1; the same computed with in-sample training survivors (comparable to CAL/e6 tail_B).

## Seen before writing
AA/nl_posterior.json (sigma at locked pairs: T3 median 0.026-0.028, T1 0.009-0.011, T4 0.008-0.009),
CAL/e6_bestresponse_pooled.json, CAL/e6b_matched_hinge.json. No normalised score or sigma-scaled width computed.

## Triage
Triage 2026-10-14 22:17 UTC: NOT SELECTED this round; not run. No result of this route exists. Merged with coverage_strong_current (both ask how large the caution must be at strong current). Reason: the scale is
the last-layer posterior from shelved route 1, which measures how novel the state is, not harm; multiplied onto a throttle
ramp it is a range limit scaled by novelty (close to shelved route 3). The test is a proxy that could at most justify an
actor pilot, which is step 2's decision. Step 1's held-out heights (1.6-6.0 in the stalling world, evaluate.log) are
well above the in-sample 0.60-1.27 that motivated the route, so its premise (the height is too small at strong current)
needs re-checking first.
