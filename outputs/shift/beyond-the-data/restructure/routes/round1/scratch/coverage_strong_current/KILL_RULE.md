# Kill rule: coverage where the actor acts (written 2026-10-14, before any result of this test exists)

EXPLORATORY, designed after seeing seeds 0-4. Seeds 0-4 only. No training. Proposed, not yet run.

## Test
Visited steps: 'never' rollouts of the Stage 2 'none' actor at 200 strong-current starts, both worlds. At every
visited step with throttle > t_lo, the TRUE one-step overestimate of the gate critic: s* = Q(s, a) - [r + 0.98
Q(s', c(s'))], s' and r from one exact simulator step (thruster_chain.dynamics), min of heads, noiseless next command;
the same at the command clipped to the recorded range (paired excess = s*(a) - s*(clip a)).
BCA's thresholds from held-out evidence at recorded currents: R_tail and R_rest (wbcp deployed thresholds,
max(lambda_hat, lambda_hpd)), primary evidence = removed full-throttle training rows (rich, held out), secondary =
held-out calibration-split rows past t_lo.

## Shelve the claim "BCA's calibrated threshold holds where the actor acts" if, in the stalling world (pooled seeds
0-4, rich evidence), EITHER
1. R_tail is at or above s* at fewer than 80% of visited past-t_lo steps, or
2. the height R_tail - R_rest is at or above the paired excess at fewer than 80% of them.
Consequence if shelved: at strong current only the certificate's yes/no may be used; any consumer that uses the
size (training penalty k x U, value comparison) has no calibrated basis there.

## Reported, not deciding
No-stall world; coverage by current band (0.7-0.8, 0.8-0.9, 0.9-1.1); plain split conformal (lambda_hat) against
BCA's posterior threshold (the posterior's own share); recorded conditions as the in-distribution reference.

## Triage
Triage 2026-10-14 22:17 UTC: NOT SELECTED this round; not run. No result of this route exists. Merged with posterior_scaled_height. Reason: it checks whether a size-based use of the height is valid at strong
current; it cannot show a gain by itself. Worth running when step 2 picks a size-based consumer.
