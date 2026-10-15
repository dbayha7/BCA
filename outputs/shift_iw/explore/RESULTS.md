# EXPLORATORY: weighted BCA arms on all five qualities

**EXPLORATORY: outside the pre-registered claims; the version-5 gates did not pass.** The pre-registered report allowed
no BCA arm on any quality. The owner asked for the arms anyway ("Run the arms anyway, labelled exploratory"). Nothing
here is a pre-registered result, and no claim is judged.

Made 2026-10-13 from the run of 11:46-14:10 (every job exited 0; the frozen version-5 results were not changed). Numbers
come from `RESULTS.json` (written by `make_results.py`, read only).

## In short

- **Weighting did not raise return through either use of BCA's band, on any quality.** On the strong starts, no
  weighted arm gained 1.8 return units (the size the room and prediction checks require) with a lower bound above 0.
- **Capped pull (BCA's own use):** per-step weighting made no difference at the top two qualities (expert +0.1,
  medium-expert −0.9) and made things worse at medium-replay (−26.4) and medium (−13.8).
- **Lower edge:** every weighted change lies between −1.1 and +2.7 return units (at most 0.2 points). BCA's penalty is
  30-160 times smaller than the perfect one, so these actors barely move away from the no-caution actor.
- **The pre-registered predictions held where they were asked:** about +0.4 for the expert capped pull (observed +0.1),
  and near 0 for the lower edge at the top qualities (observed +0.5 and −0.3). Five seeds cannot tell +0.4 from 0,
  though: for the expert capped pull the 95% range is about ±4 return units.
- **BCA's capped pull itself, unweighted, costs a lot below the top:** 67-112 return units worse than plain TD3+BC
  on random, medium-replay and medium. On medium-expert it is 6.9 better. On expert it is level.

## How to read it

- **Return units:** the exact 100-step return, averaged over 1,000 fixed starting states. All returns are negative;
  closer to 0 is better.
- **Points:** the same return on a 0-100 scale. 0 is uniform random commands, 100 the stall-aware controller. On the
  strong starts, 1.8 return units is about 0.14 points.
- **Strong starts / no-shift starts:** the fixed start bank under the strong test mix (mostly strong currents, the
  shift the weighting targets) and under the training mix (no shift).
- **Capped pull:** BCA's own use of its band. The actor is pulled harder toward the recorded command where the band is
  wide. It is trained together with TD3+BC for 100,000 updates.
- **Lower edge:** a penalty on commands where the band is wider than at the recorded command. Only the actor is
  trained, against the frozen critic, as in the room check.
- **Weighted per step:** each training row uses its own threshold under the test mix. **Single threshold:** one
  threshold for every row (a comparison only). **Mild:** the lower edge weighted for the mild mix (shown only).
- **Plain TD3+BC:** the learner on the gapped data, with no BCA. **No-caution actor:** the room check's actor, trained
  actor-only against the same frozen critic with no extra caution.
- **Lower bound:** one-sided 95% lower bound over the five seeds (t with 4 degrees of freedom). All differences are
  paired by seed. **Range:** the two-sided 95% interval.

## 1. Weighted per step minus unweighted, strong starts

**Capped pull**

| Quality | Return: mean | Return: lower bound | Return per seed (0-4) | Points: mean | Points: lower bound | Points per seed (0-4) |
|---|---|---|---|---|---|---|
| random | -1.2 | -11.7 | -8.1, -16.5, +9.1, +8.6, +1.0 | -0.10 | -0.92 | -0.64, -1.30, +0.71, +0.67, +0.08 |
| medium-replay | -26.4 | -46.6 | -14.4, -10.1, -39.1, -57.7, -10.4 | -2.07 | -3.67 | -1.14, -0.79, -3.08, -4.54, -0.82 |
| medium | -13.8 | -23.6 | -10.6, -14.7, -4.4, -30.9, -8.4 | -1.09 | -1.86 | -0.83, -1.16, -0.35, -2.43, -0.66 |
| medium-expert | -0.9 | -4.0 | -3.1, -4.0, +1.6, +3.3, -2.6 | -0.07 | -0.31 | -0.24, -0.31, +0.13, +0.26, -0.20 |
| expert | +0.1 | -3.1 | +1.9, +2.5, +1.0, -5.8, +0.9 | +0.01 | -0.25 | +0.15, +0.19, +0.08, -0.46, +0.07 |

**Lower edge**

| Quality | Return: mean | Return: lower bound | Return per seed (0-4) | Points: mean | Points: lower bound | Points per seed (0-4) |
|---|---|---|---|---|---|---|
| random | +2.7 | -1.2 | +1.1, +9.1, -1.7, +1.5, +3.3 | +0.21 | -0.09 | +0.08, +0.72, -0.13, +0.12, +0.26 |
| medium-replay | -0.5 | -1.9 | -0.3, -0.2, -2.8, -0.5, +1.4 | -0.04 | -0.15 | -0.02, -0.02, -0.22, -0.04, +0.11 |
| medium | -1.1 | -1.7 | -0.5, -1.9, -1.0, -1.6, -0.2 | -0.08 | -0.14 | -0.04, -0.15, -0.08, -0.13, -0.02 |
| medium-expert | +0.5 | +0.2 | +0.0, +0.6, +0.4, +0.6, +0.9 | +0.04 | +0.02 | +0.00, +0.05, +0.03, +0.05, +0.07 |
| expert | -0.3 | -1.2 | -0.0, -0.1, +0.5, +0.2, -1.9 | -0.02 | -0.09 | -0.00, -0.01, +0.04, +0.02, -0.15 |

- Only medium-expert's lower edge has a lower bound above 0. Every seed agrees on it, but it is tiny: 0.5 return units,
  0.04 points, under a third of 1.8.
- The capped pull's losses on medium-replay and medium hold on every seed.

## 2. The comparison arms, strong starts

**Capped pull, single threshold minus unweighted**

| Quality | Return: mean | Return: lower bound | Return per seed (0-4) | Points: mean | Points: lower bound | Points per seed (0-4) |
|---|---|---|---|---|---|---|
| random | +15.2 | -4.9 | +14.6, -8.1, +45.6, +24.2, -0.1 | +1.20 | -0.39 | +1.15, -0.64, +3.59, +1.91, -0.01 |
| medium-replay | -27.9 | -41.3 | -29.1, -20.9, -11.5, -49.5, -28.6 | -2.20 | -3.25 | -2.29, -1.64, -0.91, -3.90, -2.25 |
| medium | +0.9 | -16.3 | -16.7, +11.9, -8.8, +27.1, -8.9 | +0.07 | -1.28 | -1.32, +0.94, -0.69, +2.13, -0.70 |
| medium-expert | -2.0 | -5.7 | -0.4, -2.9, -8.0, -1.4, +2.8 | -0.15 | -0.45 | -0.03, -0.23, -0.63, -0.11, +0.22 |
| expert | -1.1 | -4.7 | +3.2, -3.1, +2.5, -2.4, -5.6 | -0.09 | -0.37 | +0.25, -0.25, +0.20, -0.19, -0.44 |

- The single threshold behaves like the per-step version except on random (+15.2, lower bound −4.9: two of five seeds
  gain nothing) and medium (+0.9 against −13.8). Per step minus single threshold is −16.5 on random (lower bound
  −30.1), and between −14.7 and +1.6 elsewhere with lower bounds below 0.
- The capped pull retrains TD3+BC as a whole. At the low qualities its paired differences spread over 25-55 return
  units across seeds. These gaps between the two weighted versions are within that spread.

**Lower edge, mild minus unweighted (shown only)**

| Quality | Return: mean | Return: lower bound | Return per seed (0-4) | Points: mean | Points: lower bound | Points per seed (0-4) |
|---|---|---|---|---|---|---|
| random | +1.8 | -2.3 | +2.4, +8.6, -3.1, -0.3, +1.3 | +0.14 | -0.18 | +0.19, +0.67, -0.24, -0.02, +0.11 |
| medium-replay | -1.0 | -1.9 | -0.5, -1.3, -1.3, -2.2, +0.4 | -0.08 | -0.15 | -0.04, -0.10, -0.10, -0.18, +0.03 |
| medium | +0.2 | -1.2 | +2.3, -1.2, -0.5, -0.7, +1.2 | +0.02 | -0.10 | +0.18, -0.10, -0.04, -0.06, +0.09 |
| medium-expert | +0.2 | -0.2 | +0.1, +0.5, -0.4, +0.2, +0.5 | +0.01 | -0.01 | +0.01, +0.04, -0.03, +0.02, +0.04 |
| expert | +0.0 | -0.2 | -0.3, +0.1, +0.2, +0.2, -0.2 | +0.00 | -0.02 | -0.02, +0.01, +0.02, +0.02, -0.01 |

## 3. Each arm against plain TD3+BC and against the no-caution actor, strong starts

Mean return difference (lower bound) [points]. Per-seed values are in `RESULTS.json`.

**Arm minus plain TD3+BC**

| Arm | random | medium-replay | medium | medium-expert | expert |
|---|---|---|---|---|---|
| Capped pull, unweighted | -111.7 (-126.8) [-8.79] | -73.7 (-93.1) [-5.80] | -67.1 (-83.1) [-5.28] | +6.9 (+2.7) [+0.55] | -0.6 (-2.0) [-0.04] |
| Capped pull, weighted per step | -112.9 (-133.6) [-8.88] | -100.0 (-123.6) [-7.87] | -80.9 (-103.5) [-6.36] | +6.0 (-0.3) [+0.47] | -0.5 (-4.0) [-0.04] |
| Capped pull, single threshold | -96.5 (-120.7) [-7.59] | -101.6 (-114.4) [-7.99] | -66.1 (-85.0) [-5.20] | +5.0 (-0.7) [+0.39] | -1.7 (-4.7) [-0.13] |
| Lower edge, unweighted | +11.7 (-29.6) [+0.92] | +11.0 (+0.6) [+0.86] | -22.5 (-50.5) [-1.77] | +13.5 (+10.3) [+1.07] | +4.5 (-0.5) [+0.35] |
| Lower edge, weighted per step (strong) | +14.3 (-24.3) [+1.13] | +10.5 (+0.5) [+0.82] | -23.5 (-51.4) [-1.85] | +14.0 (+10.7) [+1.11] | +4.2 (-1.4) [+0.33] |
| Lower edge, weighted per step (mild) | +13.4 (-25.5) [+1.06] | +10.0 (-0.9) [+0.78] | -22.3 (-50.7) [-1.75] | +13.7 (+10.7) [+1.08] | +4.5 (-0.6) [+0.35] |

**Arm minus the room check's no-caution actor**

| Arm | random | medium-replay | medium | medium-expert | expert |
|---|---|---|---|---|---|
| Capped pull, unweighted | -122.3 (-166.2) [-9.62] | -84.3 (-112.6) [-6.63] | -44.6 (-83.5) [-3.51] | -6.6 (-11.0) [-0.52] | -5.2 (-9.6) [-0.41] |
| Capped pull, weighted per step | -123.5 (-167.4) [-9.72] | -110.6 (-140.0) [-8.71] | -58.4 (-99.6) [-4.59] | -7.6 (-12.6) [-0.60] | -5.1 (-10.3) [-0.40] |
| Capped pull, single threshold | -107.0 (-140.4) [-8.42] | -112.2 (-131.3) [-8.83] | -43.6 (-86.5) [-3.43] | -8.6 (-15.0) [-0.68] | -6.3 (-10.8) [-0.49] |
| Lower edge, unweighted | +1.1 (-4.9) [+0.09] | +0.4 (-0.9) [+0.03] | +0.0 (-2.4) [+0.00] | -0.0 (-0.6) [-0.00] | -0.1 (-0.4) [-0.01] |
| Lower edge, weighted per step (strong) | +3.8 (+0.3) [+0.30] | -0.1 (-2.4) [-0.01] | -1.0 (-3.3) [-0.08] | +0.5 (-0.3) [+0.04] | -0.4 (-1.2) [-0.03] |
| Lower edge, weighted per step (mild) | +2.9 (-0.1) [+0.23] | -0.6 (-2.7) [-0.05] | +0.2 (-2.5) [+0.02] | +0.2 (-0.7) [+0.01] | -0.1 (-0.4) [-0.01] |

- **Capped pull.** The extra pull toward the recorded commands hurts wherever those commands are poor. At random,
  medium-replay and medium, most recorded commands are uniform random. There, all three capped-pull arms fall far below
  both references, weighted or not. Weighting widens the band, which strengthens the pull, so it makes this worse. At
  the top two qualities the capped pull sits 5-9 return units below the no-caution actor.
- **Lower edge.** All three arms stay within 4 return units of the no-caution actor (within 1 at the top qualities).
  BCA's penalty is too small to move the actor. Their gap to plain TD3+BC is therefore the gap between the no-caution
  actor and plain TD3+BC, not an effect of the band. That gap is +13.5 on medium-expert and −22.5 on medium.
- For scale, the lower edge with the perfect penalty beats the no-caution actor by 73.5 return units on medium-expert
  and 107.6 on expert. BCA's lower edge recovers none of it.

## 4. The cost on the no-shift starts (weighted minus unweighted)

| Quality | Capped pull, per step | Capped pull, single threshold | Lower edge, per step (strong) | Lower edge, mild |
|---|---|---|---|---|
| random | +1.3 (-4.4) [+0.17] | +8.8 (+0.4) [+1.15] | +0.7 (-0.3) [+0.09] | +0.6 (-0.3) [+0.07] |
| medium-replay | -9.1 (-16.0) [-1.19] | -10.1 (-14.4) [-1.32] | -0.1 (-0.5) [-0.01] | -0.1 (-0.9) [-0.02] |
| medium | -4.6 (-8.1) [-0.61] | -0.5 (-8.8) [-0.06] | -0.4 (-0.8) [-0.06] | +0.1 (-0.3) [+0.01] |
| medium-expert | -0.5 (-0.9) [-0.07] | -0.7 (-1.9) [-0.09] | +0.2 (+0.1) [+0.02] | +0.0 (-0.1) [+0.00] |
| expert | -0.3 (-0.9) [-0.04] | -0.4 (-1.2) [-0.06] | -0.0 (-0.1) [-0.00] | -0.0 (-0.1) [-0.01] |

Mean return difference (lower bound) [points]. Per-seed values are in `RESULTS.json`.

- Where the capped pull's weighting hurts under the shift, it also hurts without it, by about a third as much
  (medium-replay −9.1, medium −4.6). Elsewhere the cost is under 1 return unit.
- One exception: on random, the single threshold gains 8.8 without a shift (lower bound +0.4). That is a comparison
  arm on the quality with the noisiest training, among many comparisons.
- The lower edge's cost without a shift is at most 0.4 return units.

## 5. Did the observed changes match the pre-registered predictions?

"Can weighting move return" predicted the return change from weighting before any BCA arm was trained, from the frozen
band and the room checks. Its bar was 1.8 return units in size.

| Quality | Capped pull: predicted | Capped pull: observed (95% range) | Lower edge: predicted | Lower edge: observed (95% range) |
|---|---|---|---|---|
| random | -5.1 | -1.2 (-14.9 to +12.5) | -2.9 | +2.7 (-2.3 to +7.7) |
| medium-replay | -3.4 | -26.4 (-52.8 to +0.0) | -3.5 | -0.5 (-2.3 to +1.4) |
| medium | -4.6 | -13.8 (-26.6 to -1.1) | -4.3 | -1.1 (-2.0 to -0.2) |
| medium-expert | +0.9 | -0.9 (-4.9 to +3.0) | +1.2 | +0.5 (+0.1 to +0.9) |
| expert | +0.4 | +0.1 (-4.1 to +4.3) | -0.4 | -0.3 (-1.5 to +0.9) |

- **Expert capped pull: matches.** Predicted +0.4, observed +0.1. Neither is distinguishable from 0. The seeds'
  spread (about ±4) is ten times the predicted effect, so this run can neither confirm nor refute the +0.4 itself.
  Medium-expert is the same story (+0.9 predicted, −0.9 observed).
- **Lower edge at the top qualities: matches.** Predicted +1.2 (medium-expert) and −0.4 (expert); observed +0.5 and
  −0.3. The medium-expert prediction lies just above the observed range, but both are far below 1.8.
- **Below the top: the direction held, the size did not.**
  - The capped pull's predictions there were harmful (−3.4 to −5.1), which is why they cleared the 1.8 bar in size.
    The observed harm was larger: about 8 times on medium-replay and 3 times on medium. Random was within noise.
  - The lower edge's predictions (−2.9 to −4.3) overstated what happened (−1.1 to +2.7). This is what the shown-only
    check before training warned: the prediction treats BCA's penalty as the perfect one, but BCA's penalty is about
    30 times smaller there (58 times on medium-expert, 163 on expert).

## Checks and caveats

- **Training and scoring.** Every arm finished: 25 seeds × 3 capped-pull arms at 100,000 updates, none failed, and no
  band refresh failed to certify. The lower edge had no refused rows. Every arm was scored on all three start banks, at
  1,000 starts each.
- **Shape check on BCA's own penalty.** This check asks whether BCA's own lower-edge penalty rises past 80% throttle
  and pulls the best command back. It failed on some seeds: random seed 2 (unweighted, mild); medium seed 4 (all three);
  medium-expert seed 0 (all three); expert seeds 1 and 3 (all three) and seed 4 (both weighted). It is recorded and does
  not stop the arms, as in the official stage. On medium seed 4 the penalty did not rise past 80%. Everywhere else it
  did not reduce how often a throttle past 80% looks best (for example 0.815 of rows against 0.814 without it).
- **Many comparisons.** `RESULTS.json` holds 63 paired comparisons in return units per quality (21 pairs on three start
  banks), with no correction for multiple testing. A lower bound above 0 here is a pointer, not evidence.
- **Different training for the references.** The capped pull retrains TD3+BC as a whole. The lower edge and the
  no-caution actor train only the actor against a frozen critic. Compare the capped pull with plain TD3+BC, and the
  lower edge with the no-caution actor.
- **Files from another session.** A different session added `plain_twin/` under the exploratory output folder, and
  `plain_twin_*` and `PLAIN_TWIN_*` files here. They are not part of these results.

## Files

- `RESULTS.json`: everything above, plus per-seed values for every comparison on all three start banks, the predictions
  read from each quality's frozen report, the lower edge's per-seed checks, and the SHA-256 of every input record.
- `make_results.py`: builds `RESULTS.json`, read only, with the package's own statistics. To rerun it, from
  `/root/bca-shift` with `PYTHONPATH=/root/bca-shift`, run
  `python /mnt/c/Users/David/dev/bca-shift-work/iw_explore/make_results.py`.
- Run outputs: `/root/bca-shift/runs/shift_iw_v5_explore/<quality>/{bca,bca_lower_edge,evaluate}/...`. The run's
  own table is in `summary/summary.txt`.
