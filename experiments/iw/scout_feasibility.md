# Feasibility numbers for the importance-weighting experiment (exact simulation, no learning)

Scout run on 2026-10-13, CPU 2 only. Nothing was learned and no BCA code ran. Every number below comes from simulating
ThrusterChain's stalling world with its own `dynamics()` (the package in `experiments/shift` is imported read-only).

## Bottom line

- **The suggested recording controller (textbook gain clipped at 80% per thruster) does not work.** Thruster 1 sits at
  exactly 80% on 19-28% of training rows, but the densest-15% removal rule takes out only 16-30% of those rows, so
  7,700-9,900 of them stay in training. The rule works in four dimensions, and these rows are spread out in thrusters
  2-4. Its noise-free form also scores 74-90 points, which leaves little room between cloning and the stall-aware
  controller.
- **What works: v1's recording controller with its bad full-throttle command swapped for a good 80% hold.** In hard
  water the recorder flips a coin each step. Half the time it sends one fixed command: thruster 1 at exactly 80% and
  thrusters 2-4 in the matching hold pattern, plus a small jitter. The removal rule takes out 99.7-99.8% of these
  commands. The hold beats the recorder's normal command at the same states, and it is the stall-aware move on most
  strong-current test steps.
- **Recommended tilts** (all closed form on [0.1, 1.1]): training p_train(c) ∝ e^(-2c), mild test law uniform, strong
  test law ∝ e^(+2c), the training law mirrored. For a 1,000-row calibration bank the Kish effective size is 1,000 /
  724 / 352 rows. After the removal (bank of about 843 rows) it is 843 / 627 / 277 rows.
- **The exact Z (the share of training rows the removal keeps) is 0.8433** (standard error 0.0002) for the recommended
  recorder. Z is a constant, so it cancels in weighted conformal prediction. Only the infinite weight on removed
  actions matters, and it forces a refusal.
- **Two things the next steps must handle:**
  - Even with no tilt, about 15.7% of test steps have a recorded action inside the removed ball, because the test
    episodes are never thinned. Those steps must be refused by the weighted method and reported apart.
  - A weight estimated by a classifier on c or on the observation cannot see the action. It must be multiplied by the
    known ball indicator, or it will never refuse.

## Terms

- **Recording controller**: the controller that made the log. It is noisy, and it scores 0 points.
- **Stall-aware controller**: the textbook gain K capped at 80% throttle. It scores 100 points.
- **Hold demand**: what K's hold against the water alone asks of thruster 1, (K_e e)_1 = 0.910 c + 0.822 w1 − 0.320 w2.
  Here c is the current and (w1, w2) the wave.
- **Hard water**: hold demand above a trigger.
- **The gap**: the densest-15% ball. This is `experiments/truth/gap.find_gap` with `shift.gap_settings()`, run on the
  float64 of the float32 training commands, exactly as v1 does it. Every training or bank row whose command lies in
  the ball is removed.
- **The 80% hold H**: −0.8 · K[:, current] / K[0, current] = (−0.800, −0.110, −0.468, −0.325). This is the hold
  pattern scaled so that thruster 1 is exactly at the 80% limit, float32(−0.8) = −0.800000011920929. That value is at
  the knee and does not stall.
- **Kish effective size**: (Σw)² / Σw², the number of equally weighted rows a weighted bank is worth.
- **Tilt**: a change in how often each current strength appears. Given c, everything else (waves, float starts,
  recorder, physics) is identical in training and test.

## 1. The recording controller

All variants use v1's normal driving: 0.8 tanh(u / 0.8) with u = −(0.25 K_s s + K_e e) + 0.3 n. This never reaches
exactly 80%. Logs have 500 episodes × 100 steps = 50,000 training rows, with currents drawn from p_train ∝ e^(−2c) on
[0.1, 1.1]. There are three data seeds per variant.

| Recorder | Rows with thruster 1 at exactly 80% | Those the ball removes | Share of removed rows that are the 80% hold | Ordinary commands removed (share of log) | Exact-80% rows left in training | Hard-water rows kept (state coverage) |
|---|---|---|---|---|---|---|
| Clip, hold ×1.0 (the suggested one) | 18.8-21.0% | 16-27% | 21-37% | 9.4-11.9% | 7,700-7,850 | 73-85% |
| Clip, hold ×1.3 | 25.6-28.1% | 27-30% | 47-55% | 6.7-8.0% | 9,270-9,900 | 69-73% |
| **80% hold, trigger 0.65, coin 0.5 (spec 1, recommended)** | **9.1-10.5%** | **99.7-99.8%** | **60-70%** | **4.5-6.0%** | **11-14** | **42-43%** |
| 80% hold, trigger 0.80, coin 0.5 (spec 2) | 4.5-5.3% | 98.7-99.8% | 29-35% | 9.7-10.6% | 4-30 | 40-42% |
| 80% hold, trigger 0.80, coin 0.7 | 6.2-7.4% | 98.8-99.8% | 41-49% | 7.6-8.8% | 9-38 | 25-26% |
| 80% hold, trigger 0.60, coin 0.5 | 11.1-12.5% | 99.7-99.9% | 74-83% | 2.5-3.9% | 7-15 | 44-46% |

**Why the clip recorder fails.** Its 80% rows share thruster 1's value but are spread by noise in thrusters 2-4. The
densest 4-dimensional ball (radius about 0.40) lands in the bulk of ordinary commands instead. Only a command that
repeats in all four thrusters forms a cluster dense enough for the rule to find. That needs a fixed pattern (H) with a
small jitter: 0.03 on thrusters 2-4, enough to stay clear of `gap.py`'s point-mass guard (radius at least 0.05).

**Why a coin and not "always" in hard water.** With a coin, hard-water states stay in the log with normal-driving
commands: 42% of hard-water rows are kept in spec 1. The hole is then in the command, not in the states, as the plan
requires. Without the coin, the removal would delete the hard-water states outright.

### Spec 1 (recommended): 80% hold, trigger 0.65, coin 0.5

- In hard water (hold demand > 0.65), each step, with probability 0.5: the command is H with jitter 0.03 · n on thrusters
  2-4, and thruster 1 exactly −0.8. Otherwise the recorder drives normally.
- **The ball** (data seed 0): centre (−0.58, −0.12, −0.43, −0.27), radius 0.267, mass 15.0% (seeds 1-2: radius 0.250
  and 0.276).
- **What else it removes:** ordinary normal-driving commands near the hold, 4.5-6.0% of the log. Their thruster 1 lies
  between −0.72 and −0.41 (5th to 95th percentile), at currents 0.27-1.03.
- **Thruster 1 left after removal:**
  - range −0.80 to +0.75;
  - only 11-14 exact-80% rows leak, holds whose jitter put them just outside the ball;
  - 1st percentile −0.76, 0.1st percentile −0.785;
  - 1.3-1.5% of kept rows at or beyond −0.75.

  So the gapped log practically never shows thruster 1 past −0.76.
- **Good.** The comparison uses the exact discounted value of each command under the stall-aware controller
  (discount 0.98), at states where the recorder gives the hold. For scale, one step's cost there is about 1-3.

  | | Training law (mean c 0.81) | Strong test law (mean c 0.91) |
  |---|---|---|
  | 80% hold − normal command | +0.097 ± 0.010, better at 61% of states | +0.133 ± 0.010, better at 67% |
  | stall-aware command − normal command | +0.45 | +0.51 |
  | v1's full-throttle command − normal command | −1.03 | −1.40 |

  So the hold is a good command, far from v1's bad one, but not the best one: it trails the stall-aware command by
  0.35-0.38, because thrusters 2-4 are a fixed pattern. That gap is the price of the exact repetition the removal rule
  needs. Where the hold demand is below 0.7 the hold is about neutral (+0.02 to +0.06); above 0.8 it is better
  (+0.12 to +0.19).

### Spec 2 (alternative): the same, with trigger 0.80

- The hold is given only where K's hold alone would stall, which is the cleaner meaning. It is slightly better where it
  is given: +0.17 to +0.18, better at 75% of states.
- But it is only 5% of training rows, so the ball (radius 0.30-0.31, centre near thruster 1 = −0.54) is two-thirds
  ordinary commands.
- Thruster 1 left after removal is about the same: 1st percentile −0.76, 1.7-1.8% at or beyond −0.75.

### Points

The simplest choice is to keep v1's convention on each test law's fixed bank: **0 = this recording controller (with
fixed noise and coin streams), 100 = the stall-aware controller.** The normal-driving-only controller (no noise, never
the hold, about what cloning the gapped log gives) is reported as a reference row, not an anchor.

| Spec 1, mean 100-step return | No tilt | Mild | Strong |
|---|---|---|---|
| Recording controller (0 points) | −84.3 | −109.1 | −140.8 |
| Stall-aware controller (100 points) | −36.0 | −55.8 | −81.3 |
| Normal driving only | −51.7 (67.5 points) | −79.6 (55.5 points) | −114.2 (44.7 points) |

- The spread between the two anchors is 48-59 return units, and cloning-style driving sits at 45-68 points, so there is
  room above it.
- Spec 2 is nearly identical: normal driving only scores 68.8 / 57.7 / 47.6 points.
- These returns integrate exact per-current returns over each law. A fixed 1,000-start bank will differ slightly.

## 2. Tilts on [0.1, 1.1]

p_λ(c) = e^(λc) / M(λ), with M(λ) = (e^(1.1λ) − e^(0.1λ)) / λ and M(0) = 1.

| Law | λ | Mean c | Share with c > 0.9 | Share with c > 0.7 | Exact weight w(c) = p_test / p_train | Weight range |
|---|---|---|---|---|---|---|
| Training, and test under no tilt | −2 | 0.443 | 7.7% | 19.2% | 1 | 1 |
| Mild test | 0 (uniform) | 0.600 | 20.0% | 40.0% | 0.35396 e^(2c) | 0.43 to 3.19 |
| Strong test (training mirrored) | +2 | 0.757 | 38.1% | 63.7% | 0.090718 e^(4c) | 0.135 to 7.39 |

Because log w is linear in c, a logistic regression on c is a correctly specified classifier for the estimated-weight
arm.

**Spec 1, per test law.** Shares are of test steps recorded by this recorder under the test law, with the seed-0 ball.

| | No tilt | Mild | Strong |
|---|---|---|---|
| Kish effective size, 1,000-row bank, no removal (closed form n / E_train[w²]) | 1,000 | 724 | 352 |
| Kish effective size, removed bank (about 843 rows kept) | 843 | 627 | 277 |
| Independent episodes worth (bank = 200 episodes × 5 rows sharing c) | 200 | 145 | 71 |
| **Pertinence:** stall-aware controller at its 80% limit on thruster 1, at the recorded test states | 31.8% | 48.6% | **65.6%** |
| Same, at the stall-aware controller's own states | 15.1% | 31.0% | 50.3% |
| The 80% hold in the test log | 10.2% | 19.3% | 29.3% |
| **Recorded action inside the gap** | **15.7%** | **25.8%** | **36.4%** |
| of which the 80% hold | 10.2 | 19.2 | 29.2 |
| of which ordinary commands | 5.5 | 6.6 | 7.2 |

Spec 2 gives Kish 844 / 623 / 283 (removed bank), pertinence 34.6% / 52.8% / 70.5%, and inside the gap 15.6% / 23.3% /
31.6% (the hold 5.2 / 11.2 / 18.8; ordinary commands 10.4 / 12.1 / 12.8).

**By current (spec 1).** All test-law shares are integrals of these curves against p(c).

| c | 80% hold | Inside the gap | Stall-aware at limit (recorded states) |
|---|---|---|---|
| 0.1 | 0.000 | 0.015 | 0.044 |
| 0.3 | 0.000 | 0.042 | 0.130 |
| 0.5 | 0.068 | 0.143 | 0.327 |
| 0.7 | 0.239 | 0.330 | 0.631 |
| 0.9 | 0.412 | 0.492 | 0.846 |
| 1.1 | 0.499 | 0.551 | 0.945 |

**Precision note for the pass marks.**
- With 277 effective rows under the strong tilt, the realised coverage of a 90% band spreads by about ±1.8 percentage
  points (one standard deviation).
- Counting the 5-rows-per-episode clustering (71 effective episodes), the spread is up to about ±3.6.
- So the strong-tilt marks should allow about ±4 points per seed, or pool the 5 seeds, or use a larger bank.

## 3. The exact Z of the removal

- The kept training distribution is p_train(x, a) · 1[a outside the ball] / Z, with Z = P_train(recorded action
  outside the ball).
- For a kept action, the exact density ratio of a test row to a kept training row is Z · p_test(c) / p_train(c). For an
  action inside the ball it is infinite.
- **Spec 1: Z = 0.8433 (standard error 0.0002)**, with the seed-0 ball. Spec 2: Z = 0.8445 (0.0002). The other
  variants give 0.841-0.850.
- The training log itself keeps exactly 85.0%, because the ball is fitted to hold 15% of that sample. The population
  value differs slightly.
- Z is computed as the integral of p_train(c) · P(command outside the ball | c) over a 41-point current grid, with
  2,000 common starts per point.
- Check by direct sampling: 40,000 episodes drawn straight from the strong law give an in-gap share of 0.3146 ± 0.0008,
  against 0.3157 ± 0.0004 from the curve integral.
- **Z cancels in weighted conformal prediction.** `calibration/wbcp.calibrate` uses only weight ratios (it rescales by
  the largest weight), so w and Z·w give the same threshold.
- `wbcp.calibrate` refuses a non-finite `test_mass`, so in-gap test steps must be refused ("cannot certify") before
  the call. That matches the infinite weight: the weighted quantile becomes +∞.

## 4. Consequences for the plan

- **No tilt is not "no shift".** About 15.7% of test steps are inside the gap at every tilt, because test episodes are
  never thinned.
  - Outside the gap, the exact weights are a constant, so weighted equals unweighted, as the control expects.
  - Inside the gap, weighted refuses while unweighted gives a band.
  - Coverage must be reported separately outside and inside the gap.
- **Estimated weights must keep the ball.** A classifier on c or on the observation estimates p_test(c) / p_train(c)
  only. The estimated-weight arm should use (estimated ratio) × 1[action outside the known ball]. The ball is public
  and exact, so this adds no estimation.
- **The removed command is good but not optimal.** The 80% hold is better than the recorder's normal command and far
  better than v1's full-throttle command. It is worse than the stall-aware controller's own state-dependent command,
  because exact repetition, which the removal rule needs, rules out state feedback on thrusters 2-4.
- **Cost.** One recorder variant takes about 1 minute on one CPU: three 50,000-row logs with their gaps, the current
  curves (82,000 episodes) and the goodness checks.

## Files

- Scripts in `C:\Users\David\dev\bca-shift-work\iw\scout\`:
  - `scout.py`: the recorders, curves, integrals, Z, Kish and goodness;
  - `check_direct.py`: the direct-sampling check;
  - `tilt_table.py`: closed-form tilt numbers;
  - `show.py` and `curve_table.py`: readouts;
  - `k_peek.py`: prints K.
- Outputs in `/root/bca-shift/runs/shift_iw/scout/`: `<variant>.json` for `B_p0.5_t0.65` (spec 1), `B_p0.5` (spec 2),
  `B_p0.7`, `B_p0.5_t0.6`, `A_hold1.0` and `A_hold1.3`, plus `run1.log`.
- Run: `python scout.py <variant> ...` from the repository root with the venv and `PYTHONPATH=/root/bca-shift`.
