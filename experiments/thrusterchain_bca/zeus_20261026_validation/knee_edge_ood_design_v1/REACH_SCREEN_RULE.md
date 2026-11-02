# Reach screen: candidates and selection rule (written before the screen ran)

Written 2026-10-29, before `screens/reach_grid.py` was written or run. This is a feasibility screen on saved JSON from the finished central-quartile study. Those results have already been seen, so this screen can only rule designs out or rank them; it is never confirmatory evidence.

## Why this screen exists

Unifloral TD3+BC (pin f2dc1278, `td3_alpha` 2.5) settles each actor command near

    pi0 = m + 2 * lambda * dQ1/da0

- m is the behavior-cloning (BC) anchor, the mean logged a0 at that state.
- lambda = 2.5 / mean|Q1| over the training batch (`td3_bc.py` lines 243-245).

For the native actor to be drawn into the withheld region by Q rather than by BC, the reach 2 * lambda * |dQ1/da0| must exceed the gap between m and the gated region.

The design panel's adversarial review found that widening the reset law raises mean|Q1| from about 72 to about 122, which scales lambda by about 0.59. With that correction, the recommended design (KEC-W v1) fails its own reach screen.

## Candidates (fixed now)

Every candidate keeps the following unchanged:
- the host;
- `td3_alpha` 2.5;
- the region predicate |a0| > KNEE;
- stall_slope 3.0;
- a1..a3 ~ U[-0.7, 0.7].

| Lever | Values |
|---|---|
| Reset law (current c) | `wide`: U[0.1, 1.1]. `mix25`: U[0.1, 0.7] with probability 0.75, else U[0.9, 1.1]. `mix15`: the same with probability 0.85 and 0.15 |
| High-load a0 window and floor share e | `A`: [-1, -0.6], e = 0.05 (the panel's plan). `B`: [-0.9, -0.7], e = 0.05. `C`: [-0.9, -0.7], e = 0.02. `D`: [-0.85, -0.75], e = 0.02 |
| Gated depth d (R_deep- = a0 <= -(KNEE + d)) | 0.10 (the plan), 0.05 |

## Forecast models

These are the panel's own models; only lambda and m change between candidates.

- **lambda scale.** mean|Q1| = a + b * E[c^2]. The fit runs through the saved ordinary (72.1) and strong-current (196.3) values, and the scale is 72.1 / mean|Q1|(law).
- **Anchor m.** The exact mean of retained high-load a0 under CUTOFF:
  - (1 - e) * U[window] restricted to a0 >= -KNEE,
  - plus e * U[-1, 1] restricted to |a0| <= KNEE.
- **Probe forecast.** At strong-current high-load probes, pi0 = clip((m + 2 * lambda' * b) / (1 - 4 * lambda' * c)), where:
  - b and c come from a quadratic fit of saved Q1 on the five slots in [-0.5, 0.5];
  - lambda' is the cell's saved lambda times the scale.
  - The forecast is the fraction of probes with pi0 in R_deep-.
- **Visited forecast.** On strong-current high-load visited rows, a0_new = scale * a0_old + 0.87 * m. The forecast is the fraction of rows with a0_new in R_deep-.
- All nine native 20k cells are included. seed2_random's gradient audit is read from `central_quartile_training_v1_continuation1`, and the script asserts nine cells.

## Selection rule

1. **Feasibility.** A candidate is feasible only if both forecasts clear their bars in all nine cells:
   - the probe forecast is at least 0.40;
   - the visited forecast is at least 0.30.
2. **Tie-breaks.** Among feasible candidates, choose in this order:
   - (1) d = 0.10 over d = 0.05;
   - (2) floor e = 0.05 over 0.02;
   - (3) reset law closest to the plan, in the order wide, mix25, mix15;
   - (4) window closest to the plan, in the order A, B, C, D.
3. **No feasible candidate.** If no candidate is feasible, record that these models cannot ensure reach by recorder design at `td3_alpha` 2.5, and take the question to the user. Do not invent new candidates after seeing the grid.
4. **Status of the result.** The chosen candidate becomes the proposal only. It is superseded by measured edge slopes from the user-run frozen-critic forward passes (step 0.3, the G0.5 fold test, extended to a0 in {-0.6, -0.7, -0.8, -0.9, -1}). The step 0.3 rule is set in DRAFT_PROTOCOL.md before that step runs.

## Known weaknesses of this screen

- The slopes near the edge are extrapolated from quadratic fits on [-0.5, 0.5].
- The critics were trained on the old uniform data at stall_slope 1.5.
- The visited model is a crude shift.
- No forecast exists for the RANDOM control's own entry.
