# BCA leverage readout v1: rule (written before any number below is computed)

Written 2026-10-31, before the script ran. Exploratory analysis of records already on file. No compute, no
checkpoint, no training. It supports no protocol claim.

## Question

Can BCA as built, at its configured strength (kappa = 3), change what the native actor does past the knee? The actor
sees BCA only through the action gradient of its penalty. If that gradient cannot reverse the actor's push into the
past-knee region, BCA cannot reduce the harm there, wherever the critic is wrong.

## Data

The gradient audits saved during the knee-edge exploratory Stage 1 are
`knee_edge_ood_v2/stage1_exploratory_run/attempt_1/{warmup_*/gradient_audit_010000, branch_*/gradient_audit_015000,
branch_*/gradient_audit_020000}.json`.
- They cover 12 runs (3 seeds x {cutoff, random} x {stall3, nostall}), 768 fixed logged probe rows each.
- Per row (`per_row`), the actor's own action and three per-row action-loss gradients, each with the batch's stopped
  lambda: q (from -c Q1), bc and prospective_bca3 (c x 3 x dU/da, BCA's penalty at kappa = 3, measured in shadow).
- These are logged training states, not the evaluation anchors. The actor's actions there are its own.

## Rows scored

High-load rows (`per_row.high_load`) where the actor's own a0 < -KNEE: rows where the native actor is in the
withheld/harmful direction W-. The inward direction is +a0, toward -KNEE.
- The loss-descent direction on a0 is -g0, so "points inward" means g0 < 0.
- "Native pushes outward" means (q + bc)_0 > 0.

## Readouts, per run and checkpoint

1. **Reversal share:** among scored rows where native pushes outward, the share where adding BCA makes the net push
   inward: (q + bc + bca3)_0 < 0.
2. **Slope ratio:** the median over scored rows of |bca3_0| / |q_0|, BCA's pull on a0 relative to the critic's.
3. **Inward share:** the share of scored rows where bca3_0 < 0. This is dose-free: at any kappa, BCA can reverse at
   most the rows whose BCA gradient points inward.
4. **Dose needed:** over scored outward rows whose BCA gradient points inward, the median of
   3 x (q + bc)_0 / (-bca3_0), the kappa at which BCA would just cancel native's push. It is compared with the
   method's grid maximum, kappa = 10.
5. **Placement (reported, never decides):** within each data arm and seed, stall minus no-stall of the inward share
   and of the median signed -bca3_0 / |q_0|. Positive means BCA pushes inward more where past-knee entry is harmful.

## Decision at 20k, the checkpoint where every other knee readout is on record

- **NO LEVERAGE:** in every stall run (3 seeds x 2 arms), the reversal share is < 0.25 AND the slope ratio is < 0.25.
- **LEVERAGE:** in every stall run of at least one data arm, the reversal share is >= 0.5 AND the inward share is >= 0.75.
- **PARTIAL:** anything else.

10k and 15k are reported the same way.

## What each verdict leads to

- **NO LEVERAGE:**
  - BCA as built, at its configured dose, cannot steer this host's actor away from the knee, with or without labels.
  - A BCA Stage 2 on this bed would be expected to equal native.
  - The next decision is about the method, not the benchmark. It is reported to the user, with the dose-free
    inward share (could any kappa work?) and the dose needed (would kappa = 10 work?).
- **LEVERAGE:** the checkpoint readout of U at the evaluation anchors, and the room screen, are worth commissioning.
- **PARTIAL:** commission the checkpoint U readout at the anchors (a user-run WSL job) before any Stage 2 design.
