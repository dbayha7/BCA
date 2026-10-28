# Shelved: the stall-knee band as the conditional hole (28 October 2026)

**Sources.**
- **Gate figures:** `../conditional_ood_gate_v1_retry_20261028/run/report.json` and `run/geometry.json`, attempt 2 of the
  conditional OOD design gate.
- **Run time and report hash:** `run/RESULT.json`.
- **Forecasts:** the analytic receipt, `../conditional_ood_design_v2/analytic_receipt_20261027/` (`FIGURE_MAP.md` and
  `outputs/`).
- **Actor range over 128 states:** `EVIDENCE.json` in this folder.

**Run.** Attempt 2 ran on 28 October 2026 at 15:33 UTC.
- **Duration:** 63.7 seconds; report SHA-256 `03e17545…`.
- **Work done:** all 28,104 planned simulator transitions.
- **Training:** zero critic, actor, auxiliary or calibration updates.
- **Data:** zero reads of selection or final-calibration rows.

Attempt 1, in `../conditional_ood_gate_v1/`, stopped before any data on one code defect: it read a file attribute the
in-memory canonical module lacks. A second defect in its crash handler hid that error message. Attempt 1 is kept as a
software-failure record.

## The idea

Withhold the logged training rows where the first thruster was pushed hard upstream, right at its stall point. That is
`a0` within 0.05 of −0.8, at states where the water load is at least 0.4. The same commands stay labeled at lower load.
A critic trained without those labels might misjudge them, and BCA's penalty might correct that.

## What we tested, and the kill rule

The gate was a no-training screen of the design. Its rules were frozen before it decoded any of its arrays. They were not
frozen before all data had been seen:
- its 64 anchor episodes had been used in earlier development;
- the saved actor's development bank, with 0 of 128 states in the band, was already known.

The gate used the saved 20k actor and critic. The design fails if any test below fails, or if any integrity, coverage or
balance check fails (those are listed under "What happened").

| Test | What it measured | Kill rule |
|---|---|---|
| Local support | Training rows near each anchor, in the same ten-step time bin and load side, within standardized RMS distance 0.5 | Each load side needs at least 16 anchors, and at least 80% of them must pass. A passing anchor needs 32 rows from 8 episodes, both before and after removal. It also needs at least 2 original band rows from 2 episodes, and at least 2 retained rows from 2 episodes in each neighboring band. |
| Value gap | The first thruster's command swept across the band and its two neighbors, other three commands held at the actor's own, exact return measured to step 100 under that actor | At least 25% of high-load anchors must show a center-to-neighbor gap of at least 1 return unit |
| Policy exposure | 16 free episodes with the actor | At least 100 high-load steps, entry into the band on at least 1% of them, in at least 2 episodes |

## What happened

| Test | Result | Verdict |
|---|---|---|
| Local support | 0 of 29 high-load and 0 of 35 low-load anchors passed. Same-side rows within distance 0.5 had a median of 0, a mean of 1.7 and a best of 17. The distance to the 64th neighbor ranged from 0.58 to 1.00, with a median of 0.75 (0.79 at high load). No anchor had more than 1 band row in its neighborhood. | **Failed** |
| Value gap | 8 of 29 high-load anchors had a gap of at least 1 unit, exactly the 8 needed. The ninth-largest gap was 0.999. The median gap was 0.70 and the largest 2.46. The median spread inside the band alone was 0.35. | Passed, with no margin |
| Policy exposure | 927 high-load steps across all 16 episodes. The actor entered the band on 0 of them, in 0 episodes. | **Failed** |

These checks passed:
- **Coverage:** the region held 1,967 training rows across 612 episodes. The same commands at low load held 1,926 rows.
- **Exclusion:** no withheld row reached training or shared calibration labels.
- **Random-removal control:** counts matched exactly. The largest standardized state-mean difference from the hole pool
  was 0.003.
- **Calibration:** every calibration episode was eligible.
- **Replay:** the 2,400-step recorded-prefix replay reproduced every recorded observation at all 64 anchors.
- **Parity:** direct dynamics matched the environment exactly at t = 0, 25, 50 and 99, on one 100-step path (104
  transitions).
- **Frozen model:** unchanged.

**Descriptive, not decisive.** The full-data critic overestimated value at both load levels. At high load, its mean
absolute error was 21.7 inside the band and 22.3 outside it, with a mean signed error of about +18. At low load, it was
19.3 inside and 19.7 outside. With labels present, the critic was no worse on the band than off it. Whether withholding
the labels changes that was not tested: no condition-specific model was trained.

At 2 of the 29 high-load anchor states, at loads 0.65 and 0.97, the actor's own command fell inside the band. Those
states came from recorded episodes, not from the actor's own trajectories.

No BCA-adjusted score or penalty was computed or examined. The only BCA-model forward pass was the loader's two-row
consistency check on its uncertainty width, which fed no decision.

## Why

- **Support failed first on a row count that does not depend on the band.** No anchor had 32 same-side rows within
  radius 0.5. The analytic receipt predicted that: it forecast about 1.4 to 2.3 rows within 0.5, and 1.7 were measured
  (1.1 at high load, 2.2 at low). Its forecast for the 64th-neighbor distance was about 0.72 to 0.73. The measured median,
  0.75, was slightly larger, and 0.79 at high load. So neighborhoods were a little sparser than the Gaussian model
  predicted, especially at high load. In 13 standardized coordinates, radius 0.5 is too tight for 768 training episodes,
  whatever the region.
- **Whether the band would have met its own row criteria under an adequate radius was not tested.** Binomial arithmetic
  suggests it would probably have fallen short. Requiring 2 rows in the band and in each neighboring band would pass only
  about 58% of anchors even with 64 neighbors, against the 80% required. That figure is derived by treating the bands as
  independent.
- **Exposure failed because the one saved actor does not go there on its own.** It can command inside the band, as it did
  at 2 recorded high-load states. But in free play it never entered the band, and at its 128 bank states its first
  command stayed between −0.483 and +0.521. How close its free commands came to the band was not reported. One actor was
  evaluated; other trained actors are untested.
- **The value test passed at its minimum.** These are exact returns under this one actor's continuation. So the band is
  not uniformly value-flat for this actor. But its median gap, 0.70, is below the 1-unit practical size.

This is evidence against this design, not evidence that BCA fails.

**What carries forward:**
- **Code to reuse:** the prefix replay, parity check, fixed-other-commands continuation and free-rollout accounting. They
  ran to completion and passed their built-in checks.
- **The four-cell error table.**
- **The lesson:** the support rule and policy exposure must be designed in before a region is proposed, not checked
  after it.

The rules for any replacement are fixed before its data is read and independently of whether BCA wins.
