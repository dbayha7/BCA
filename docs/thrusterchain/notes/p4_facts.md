# Page pass 4: verified facts and chart data

Made 2026-10-13. Read only: no page edit, nothing written under `/root/bca-shift`. Every number below was recomputed
or read from the named file. The full data, including every histogram, are in `notes/p4_facts.json`.

**How it was made.** Three read-only scripts, run in WSL with `/opt/bca-shift-venv` and `PYTHONPATH=/root/bca-shift`
on one CPU:

- `scripts/p4_facts_compute.py`;
- `scripts/p4_facts_extra.py`;
- `scripts/p4_facts_build.py`, which writes the JSON.

Their intermediate files are in `tmp/p4/facts/`.

**Where the main test's data live.** The design-seed logs are not stored on disk: `runs/shift` holds only
`pre/stage0_base.json` and `build_checks.json`, and the harm-check seeds' per-seed outputs stayed on the cloud machine.
`experiments/shift/dataset.generate(900, "stall")` regenerates design seed 900 bit for bit (the generator is
deterministic and seeded), and `shift.define_gap` and `shift.make_removal` apply the densest-15% removal. The
regenerated log reproduces every seed-900 number on the page: 13.42%, 26.86%, 49.97%, −78.78, 7,500 removed, 42,500
kept, 60 kept rows past 80%, 847 calibration rows, 99.11%, 88.69%, 46.54%, 99.859%, 85.03%, the ball's centre and
radius, and episode 2 (return −149.00, 36 full-throttle steps, 38 deleted rows, steps 67 and 91 normal driving). Bank
rollouts reproduce every reference return to 4 decimals.

## 0. Cautions for the writers

1. **Both removals use the densest-15% rule.** They differ in where the ball lands.
   - **Main test.** Seed 900's ball has centre (−0.7655, −0.1343, −0.5337, −0.3767) and radius 0.2691.
     - It removes 7,500 rows: 6,652 full-throttle commands (88.69% of the removed rows; 99.11% of all full-throttle
       commands) and 848 normal-driving commands that lie near the pattern.
     - The capped pattern also lies inside the ball, 0.069 from its centre. This is the full-throttle pattern with only
       thruster 1 set to −0.8: (−0.8, −0.137, −0.585, −0.406). So the main test's removal also deletes commands near
       the 80% hold (page #shift says so).
   - **Weighting test.** The expert's ball has centre about (−0.58, −0.10 to −0.13, −0.41 to −0.44, −0.28 to −0.32)
     and radius 0.252–0.274 across seeds 0–4. It holds the 80% hold command (−0.8, −0.1096, −0.4679, −0.3250), which
     is 0.8 × the full-throttle pattern.
     - Holds are 60–69% of the rows it removes. The rest are normal-driving commands near its centre.
     - "The 80% hold removal" is a fair short name, but it is the same rule.
2. **"3.6–9.9% past +80%" does not reproduce exactly** (section 8). Use the verified wording given there.
3. **Stage 1's 93.9 of 98.9.**
   - The 98.9 is verified from the stop report.
   - The 93.9 comes only from the page's own Stage 1 table. The raw Stage 1 outputs (`runs/shift_explore/room/`) were
     made in a cloud session and are not on this machine.
   - What the penalty was: [δ*(s, actor's command) − δ*(s, recorded command)]₊. This is how much more the critic's
     exact one-step error at the actor's command exceeds the error at the recorded command, counted only when
     positive (`outputs/shift/beyond-the-data/stage1/README.md`). "Placed by the exact one-step error" is right. The
     reference to the recorded command matters, so say it.
4. **"1/30 to 1/160" compares penalties, not raw band widths.**
   - BCA's own lower-edge penalty is its band width at the actor's command minus its width at the recorded command,
     counted only when positive.
   - Measured at the actor's commands, it is 1/31–1/33 of the perfect penalty on the three lower qualities, 1/58 on
     medium-expert and 1/163 on expert.
   - The perfect penalty is the same quantity built from the critic's exact one-step error.
   - Row by row the two correlate −0.17 to 0.29 (means per quality).
5. **The reward of one step is the same in both worlds** for the same state and throttle: r = −(|s|² + 0.3|a|²) uses
   the throttle, not the push. The stall shows up only in the next states, where the floats drift further, and so in
   the true value (section 4).
6. **Raw difficulty versus harm.**
   - By raw return, big starting pushes is the hardest situation for both reference controllers, the cloned actor
     and the full-log critic's actor.
   - For the actor of the critic trained after the removal, and for plain TD3+BC, strong current has the lowest raw
     return: −275 and −279, against −211 and −215 in big starting pushes (section 6).
7. **Histogram edges at 80%.** The 80% hold is exactly float32(0.8) = 0.800000011920929, which is the knee itself, so
   it does not stall. With plain edges at ±0.8 it would fall in the bin past 80%. The JSON's histograms use
   knee-aware edges (`conventions.histogram_bins`), so the hold counts as at 80%.
8. **Thruster sign.** Every past-80% command in the main test's data is thruster 1 pushing upstream (negative). No
   command of thrusters 2–4 ever goes past 80% in that data (full log or kept), and thruster 1's positive commands
   never exceed +0.73.
9. **Holds removed in the weighting test.**
   - The page's "99.57–99.79% of holds removed" is the design check's quadrature value (`world.json` D1).
   - Counting the actual training rows gives 99.54–99.88%.
   - Both are correct, but they measure different things.

## 1. The quoted numbers: verified or not

| Claim (as quoted) | Verdict | Exact value | Source |
|---|---|---|---|
| Only 60 of the 42,500 kept rows go past 80% | verified | 60 of 42,500 (0.1412%), past 80% on any thruster. All 60 are thruster 1 upstream, and all are full-throttle commands that lie outside the ball (0.89% of the 6,712 survive). | seed 900 regenerated; page #shift |
| 93.7% of strong-current states are in the full-throttle region | verified | 93.696% of the 100,000 visited states have c + w₁ > 0.565. This is the same for every controller, because no thruster changes the water. Recorded conditions 27.86%, big waves 39.61%, big starting pushes 27.86%. | bank rollouts; page #shift table |
| The rising-thrust controller scores −54.1 points | verified | Strong current, stalling world: return −335.22, −54.13 points. It goes past 80% on 84.0% of states and loses 154.1 points by doing so. | bank rollouts; page #reference-scores |
| −79.80 and −26.36 | verified | Recorded conditions, stalling world: the recording controller (0 points) averages −79.7996 and the stall-aware controller (100 points) −26.3600. | `runs/shift/pre/stage0_base.json` record.anchors; recomputed |
| 0.65 at 90%, 0.5 at full throttle | verified | 0.65000003 and 0.50000003. They sit 3 × 10⁻⁸ above the round numbers because the knee is float32(0.8). | `thruster_chain.thrust` |
| The water's push points upstream on 20.3% of big-wave states | verified | 20.343% have c + w₁ < 0. Range −0.595 to 1.389 (the page says −0.6 to 1.4). Recorded conditions: 6.13% upstream, range −0.295 to 1.089. Strong current: never upstream, range 0.304 to 1.492. | bank rollouts |
| A penalty placed by the exact one-step error won back 93.9 of the 98.9 points lost | 98.9 verified; 93.9 from the page only | Harm 98.93 points (seeds 95.2–102.8). Stage 1 recovery +93.9 (seeds 83.4 to 103.9). | `outputs/shift/stop-report/check2_return/alpha5/pooled.json`; page #beyond-the-data |
| Harm 5.8 / 11.8 / 16.6 / 98.9 | verified | recorded conditions 5.81, big starting pushes 11.79, big waves 16.58, strong current 98.93 | same pooled.json, `situations.*.harm` |
| 53.4 / 150.1 / 70.6 / 99.7 (the unlabelled numbers in "Already measured") | verified | The stall-aware controller's 100-step return minus the recording controller's: recorded conditions 53.44, strong current 150.15, big waves 70.62, big starting pushes 99.74 | anchors |
| 1/30 to 1/160 of the true error | verified, if worded as penalties | random 1/33.1, medium-replay 1/32.8, medium 1/31.2, medium-expert 1/58.1, expert 1/162.5. Correlation −0.167, −0.125, −0.033, 0.212, 0.289. | `runs/shift_iw_v5/<quality>/report/report.json` marks.L2.proxy_check |
| No thruster-1 commands past 80% in the expert data; 3.6–9.9% in the others, all random | none and all random verified; the range is off | see section 8 | `runs/shift_iw_v5/<q>/data/seed<k>/train_law.hdf5` |
| The capped pull is a behaviour constraint, not CQL-style value pessimism; the lower edge is the value-pessimism channel | consistent with the page and the sources | The page's Results section already says this, and the weighting test calls the lower edge "where value-pessimism methods act". | page #results, #weighting-test |

Other page numbers re-checked on the way:

- strong current: the stall-aware controller holds thruster 1 at −80% on 71.9% of states (some thruster on 73.2%);
- the stall-aware controller's command lies in the removed ball on 17.2% of recorded-conditions states and 57.0% of
  strong-current states;
- the weighting test's expert data keep exactly 42,500 rows in every seed; the lower qualities keep 44,902–49,938;
- 1 point is 1.5015 return units in strong current (section 5).

## 2. Data for the push-against-throttle chart

**Push and cost.**

- Push: f(a) = a while |a| ≤ 0.8. Past 0.8, in the stalling world only, f(a) = sign(a)(0.8 − 1.5(|a| − 0.8)).
- Throttle cost: 0.3a² per thruster, in both worlds, on the clipped command, even while the thruster stalls.
- The most push a thruster can give in the stalling world is 0.8, at 80%.

| Throttle | 0 | 20% | 40% | 60% | 80% | 85% | 90% | 95% | 100% |
|---|---|---|---|---|---|---|---|---|---|
| Push, stalling world | 0 | 0.2 | 0.4 | 0.6 | 0.8 | 0.725 | 0.65 | 0.575 | 0.5 |
| Push, no-stall world | 0 | 0.2 | 0.4 | 0.6 | 0.8 | 0.85 | 0.9 | 0.95 | 1.0 |
| Throttle cost 0.3a² | 0 | 0.012 | 0.048 | 0.108 | 0.192 | 0.217 | 0.243 | 0.271 | 0.300 |

The JSON's `thrust_and_cost.table` gives every 5%, including "push minus cost". That difference is only a slider
convenience: push and cost are not in the same units. To show a real "net value", use the exact-value change in
section 4.

**Thruster-1 histograms: main test, design seed 900** (`stall_chart.main_test_seed900_thruster1`). There are 50
signed bins on [−1, 1] and also 50 magnitude bins on [0, 1].

- Full training split: 50,000 rows.
  - 6,712 are past 80% (13.42%), all upstream.
  - 3,326 are exactly −1.
  - The full-throttle commands sit at −1.00 to −0.93 (1st to 99th percentile).
  - Normal driving runs from −0.788 to +0.732.
- After the densest-15% removal: 42,500 rows.
  - 60 are past 80%, all in the bin from −1 to −0.96 (49 of them exactly −1).
  - Commands from −0.8 to −0.6 drop from 5,915 to 5,249.
  - Commands from −0.6 to −0.5 lose another 182.
  - Commands above −0.496 are untouched: that is where the ball's shadow on thruster 1 ends. With these, 848
    normal-driving rows are removed in all.
- Removed: 7,500 rows. Thruster 1 runs −1.00 to −0.65 (5th to 95th percentile).

**Thruster-1 histograms: weighting test, expert seed 0** (`stall_chart.weighting_test_expert_seed0_thruster1`).

- 50,000 training rows, with none past 80%.
- 4,950 rows are the hold at exactly −0.800000011920929. In the knee-aware bins they fall in [−0.8, −0.76), together
  with 428 other rows.
- After the removal: 42,500 rows. That bin keeps 414 rows (it had 5,378).

## 3. Observation Space: 13 numbers, values at rest, ranges per situation

**Names.** These follow the environment docstring:

- s (10): the positions and speeds of the five floats (coordinates 2j and 2j + 1);
- c: the strength of the current, which never changes during an episode;
- (w₁, w₂): the wave, a point turning 0.27 rad per step (a 23.3-step cycle).

The page calls the float numbers the "position number" and "speed number" of a float's swing.

**Source.** The table gives the 5th to 95th percentile over the 100,000 states at which a command is given (1,000
evaluation-bank starts × steps 0–99), stalling world, recording controller. This is the controller that made the
data, run with its fixed bank noise, as for the 0-point anchor. The stall-aware controller's ranges are in the JSON:
they are narrower on the floats, but c and the wave are identical. In the recorded conditions the main test's
training split agrees with this column to about 0.02, so this column describes the data.

| # | Number | At rest | Recorded conditions | Strong current | Big waves | Big starting pushes |
|---|---|---|---|---|---|---|
| 0 | position number of float 1 | 0 | −0.63 to 0.16 | −1.70 to −0.06 | −0.76 to 0.18 | −1.15 to 0.56 |
| 1 | speed number of float 1 | 0 | −0.31 to 0.52 | −0.32 to 1.16 | −0.43 to 0.70 | −0.76 to 1.06 |
| 2 | position number of float 2 | 0 | −0.41 to 0.30 | −0.91 to 0.20 | −0.60 to 0.34 | −0.71 to 0.58 |
| 3 | speed number of float 2 | 0 | −0.39 to 0.41 | −0.54 to 0.69 | −0.52 to 0.59 | −0.66 to 0.70 |
| 4 | position number of float 3 | 0 | −0.46 to 0.24 | −0.73 to 0.18 | −0.52 to 0.27 | −0.69 to 0.46 |
| 5 | speed number of float 3 | 0 | −0.34 to 0.41 | −0.35 to 0.53 | −0.36 to 0.43 | −0.55 to 0.63 |
| 6 | position number of float 4 | 0 | −0.37 to 0.36 | −0.41 to 0.35 | −0.38 to 0.35 | −0.58 to 0.56 |
| 7 | speed number of float 4 | 0 | −0.38 to 0.39 | −0.36 to 0.40 | −0.38 to 0.39 | −0.56 to 0.58 |
| 8 | position number of float 5 | 0 | −0.38 to 0.20 | −0.50 to 0.09 | −0.42 to 0.24 | −0.62 to 0.44 |
| 9 | speed number of float 5 | 0 | −0.24 to 0.34 | −0.18 to 0.40 | −0.26 to 0.35 | −0.47 to 0.57 |
| 10 | current c | no rest value; fixed for the episode | 0.13 to 0.67 | 0.72 to 1.08 | 0.13 to 0.67 | 0.13 to 0.67 |
| 11 | wave w₁, the push the wave adds now | no rest value; turns every step | −0.30 to 0.30 | −0.30 to 0.30 | −0.58 to 0.58 | −0.30 to 0.30 |
| 12 | wave w₂, the push a quarter cycle earlier | no rest value; turns every step | −0.30 to 0.30 | −0.30 to 0.30 | −0.58 to 0.58 | −0.30 to 0.30 |

Start laws, for the table's lead:

- recorded conditions: c 0.1–0.7, wave amplitude 0.1–0.4, float start SD 0.5;
- strong current: c 0.7–1.1;
- big waves: amplitude 0.4–0.7;
- big starting pushes: float start SD 1.5.

**Where the floats settle under a steady current with no wave** (`observation_space.settled_under_steady_current_no_wave`):

- c = 0.4, stall-aware controller: float 1 settles at (−0.061, +0.013), float cost 0.023, thruster 1 at −0.379.
- c = 0.4, no throttle: float 1 settles at (−1.14, +0.26), float cost 1.93.
- c = 0.9, stall-aware controller: thruster 1 is pinned at −0.8, and float 1 settles at (−0.286, +0.066), float cost
  0.178. This is the "drifts a little downstream" the page mentions.

This matches the page's statement that a settled float has a negative position number and a positive speed number.

## 4. Rewards: worked example

**Recommended state:** floats at rest, c = 0.6, w₁ = 0.3, w₂ = 0. This state is inside the recorded conditions
(c ≤ 0.7, amplitude 0.3 ≤ 0.4) and in the full-throttle region (push 0.9 > 0.565). Thruster 1 is set to −0.8, −0.9
and −1.0. Thrusters 2–4 are at the stall-aware controller's command (−0.135, −0.519, −0.321). Its own thruster-1
command here is −0.793.

"Exact value" is the discounted (0.98) sum of rewards over 1,000 steps, with the stall-aware controller driving after
the first step: the page's true value.

| World | Throttle 1 | Push | Throttle cost (thruster 1) | Reward this step | Float cost, next state | Exact value | Change from 80% |
|---|---|---|---|---|---|---|---|
| stalling | 80% | 0.80 | 0.192 | −0.3091 | 0.0370 | −12.0395 | — |
| stalling | 90% | 0.65 | 0.243 | −0.3601 | 0.0428 | −12.1706 | −0.131 |
| stalling | 100% | 0.50 | 0.300 | −0.4171 | 0.0567 | −12.3368 | −0.297 |
| no-stall | 80% | 0.80 | 0.192 | −0.3091 | 0.0370 | −12.0395 | — |
| no-stall | 90% | 0.90 | 0.243 | −0.3601 | 0.0376 | −12.0472 | −0.008 |
| no-stall | 100% | 1.00 | 0.300 | −0.4171 | 0.0418 | −12.0684 | −0.029 |

How to read it:

1. The step's own reward falls by the throttle cost alone: −0.051 at 90% and −0.108 at 100%, the same in both worlds.
2. In the stalling world the push also falls (0.65, then 0.5), so the floats drift further. Float 1's next speed
   number is 0.030, 0.075 and 0.120, and the next float cost rises.
3. Added over the future, pushing past 80% loses 0.13 (90%) and 0.30 (100%) of true value in the stalling world, but
   only 0.008 and 0.029 in the no-stall world.
4. The page's median "true price of pushing past 80%" is 0.357 over 3,000 full-throttle-region rows (page #shift). It
   uses the pattern commands, so it is not directly comparable to this single state, but it is the same size.

**Second state, strong current:** floats at rest, c = 0.9, no wave.

- The stall-aware controller's own thruster-1 command is −0.8: the unclipped textbook command would be −0.819.
- Exact values, stalling world: −26.836 (80%), −27.123 (90%, −0.288), −27.526 (100%, −0.690).
- No-stall world: −26.836, −26.824 (+0.012), −26.868 (−0.033).
- The rewards this step are −0.2979, −0.3489 and −0.4059.

**Link to return and points.**

- Return = the sum of the 100 rewards.
- Points = 100 (J − J_rec) / (J_top − J_rec) per situation, where J is the average return over the situation's 1,000
  fixed starts.

| Situation | Recording controller (0 points) | Stall-aware controller (100 points) | Return units per point |
|---|---|---|---|
| recorded conditions | −79.80 | −26.36 | 0.534 |
| strong current | −253.94 | −103.79 | 1.502 |
| big waves | −106.81 | −36.19 | 0.706 |
| big starting pushes | −263.24 | −163.50 | 0.997 |

An example of scale: the rising-thrust controller pushes past 80% on 84.0% of strong-current states and loses 154.1
points by it, which is 231.4 return units.

Ignore `rewards_example.recorded_row_ep2_t1` in `tmp/p4/facts/compute.json`. At that recorded state (seed 900, row
201) thruster 1 should push downstream (+0.18), so the comparison of upstream throttles is not a clean example.

## 5. Difficulty levels

The situations are ordered by **harm**: the actor of the full-log critic minus the actor of the critic trained after
the densest-15% removal, critic weight 5, mean of seeds 0–4. Source:
`outputs/shift/stop-report/check2_return/alpha5/pooled.json`. The actors' points come from the stop report README's
check-2 table. Plain TD3+BC comes from `beyond-the-data/anatomy.md` §6. Raw returns are derived from those points and
the anchors.

| Order | Situation | Standard term | Harm, points (95% interval) | Harm, return units | Outside the recorded conditions | Raw return: recording / stall-aware | Raw return of the actor of the critic trained after the removal (points) |
|---|---|---|---|---|---|---|---|
| 1 | recorded conditions | in-distribution | 5.8 (4.8 to 6.8) | 3.1 | about 1% of states (the float test's 99% mark) | −79.80 / −26.36 | −34.1 (85.5) |
| 2 | big starting pushes | initial-state shift | 11.8 (10.2 to 13.4) | 11.8 | 10.9–17.3% of visited states | −263.24 / −163.50 | −210.9 (52.5) |
| 3 | big waves | disturbance shift | 16.6 (14.5 to 18.7) | 11.7 | 100% | −106.81 / −36.19 | −57.4 (70.0) |
| 4 | strong current | dynamics extrapolation | 98.9 (96.2 to 101.7) | 148.5 | 100% | −253.94 / −103.79 | −275.4 (−14.3) |

Notes on the table:

- **Harm is not raw difficulty.**
  - Raw difficulty for the reference controllers, easiest to hardest: recorded conditions, big waves, strong current,
    big starting pushes. Big starting pushes is the hardest raw for both reference controllers, for the cloned actor
    (−261.2) and for the full-log critic's actor (−199.1).
  - The missing data hurt least there, because those starts begin outside the recorded conditions only on the floats,
    and only 11–18% of the steps stay outside.
- **Big waves and big starting pushes are nearly tied in return units** (11.7 and 11.8). They differ in points
  because each situation's point is a different size.
- **The share of steps the removal critic's actor pushes past 80%** is 4.7% in recorded conditions, 15.8% in big
  starting pushes, 16.5% in big waves and 73.2% in strong current.
- **Plain TD3+BC minus the cloned actor**, in points: +29.8, +46.7, +27.3 and −71.1 (in the table's order).

## 6. Four-throttle widget

**Push and cost.** As in section 2, the same for every thruster. Each thruster adds 0.3 × its push to the speed
numbers of its two floats in the next state:

- thruster 1: floats 1 and 2 (observation items 1 and 3);
- thruster 2: floats 2 and 3 (items 3 and 5);
- thruster 3: floats 3 and 4 (items 5 and 7);
- thruster 4: floats 4 and 5 (items 7 and 9).

The water adds 0.3 (c + w₁) to every float's speed number.

**Where the recorded data sit: main test, seed 900, training split.** The table gives the 5th / 50th / 95th
percentile, with the 1st to 99th in brackets.

| Thruster | Full log (50,000 rows) | After the densest-15% removal (42,500) | Removed rows | Past 80%: full, kept |
|---|---|---|---|---|
| 1 | −1.00 / −0.40 / 0.23 (−1.00 to 0.43) | −0.67 / −0.33 / 0.26 (−0.73 to 0.45) | −1.00 / −0.99 / −0.65 | 13.42%, 0.14% |
| 2 | −0.47 / −0.09 / 0.42 (−0.60 to 0.56) | −0.49 / −0.03 / 0.44 (−0.61 to 0.57) | −0.20 / −0.14 / −0.07 | 0%, 0% |
| 3 | −0.62 / −0.26 / 0.32 (−0.68 to 0.49) | −0.59 / −0.19 / 0.34 (−0.68 to 0.50) | −0.63 / −0.58 / −0.49 | 0%, 0% |
| 4 | −0.53 / −0.18 / 0.35 (−0.63 to 0.51) | −0.54 / −0.12 / 0.37 (−0.64 to 0.52) | −0.46 / −0.41 / −0.33 | 0%, 0% |

Ranges of the full log: thruster 1 runs from −1.00 to 0.73, thruster 2 from −0.77 to 0.72, thruster 3 from −0.78 to
0.74 and thruster 4 from −0.76 to 0.74.

**The removed region, main test.**

- The ball has centre (−0.7655, −0.1343, −0.5337, −0.3767) and radius 0.2691.
- Its shadow on each thruster: [−1.03, −0.50], [−0.40, 0.13], [−0.80, −0.26] and [−0.65, −0.11].
- The shadows are only a drawing aid. A command is removed only when its distance to the centre, over all four
  throttles together, is at most 0.2691.
- The full-throttle pattern (−1, −0.137, −0.585, −0.406) lies 0.242 from the centre. The capped pattern (thruster 1
  at −0.8) lies 0.069 from it.

**The removed region, weighting test.**

- The expert's ball, per seed, is in `weighting_test_ball` in the JSON. For seed 0: centre (−0.583, −0.132, −0.425,
  −0.280), radius 0.268.
- The hold (−0.8, −0.110, −0.468, −0.325) lies 0.216–0.235 from the centre across seeds.
- Expert seed 0, thruster 1: −0.80 / −0.39 / 0.24 for all rows, and −0.70 / −0.32 / 0.27 after the removal. No
  command past 80% either way.

**Where the commands sit in each situation** (`four_throttle_widget.commands_on_evaluation_banks`). Thruster 1's
median command:

| Situation | Recording controller | Stall-aware controller | Rising-thrust controller |
|---|---|---|---|
| recorded conditions | −0.40 | −0.39 | −0.39 |
| strong current | −0.78 | −0.80, held at −0.8 on 71.9% of states | −1.00, past 80% on 84.0% of states |
| big waves | −0.43 | −0.39 | −0.45 |
| big starting pushes | −0.41 | −0.38 | −0.38 |

**Ties to the methods** (from the page's own text, not new numbers):

- TD3+BC's pull acts on the kept training rows, toward each row's recorded command.
- BCA's capped pull reads its band at the recorded commands and strengthens that pull.
- The lower edge reads the band at the actor's own command relative to the recorded one.
- Commands with no data nearby are past 80% (main test) and the 80% hold (weighting test). At those commands the
  critic is never corrected.

## 7. Points

See section 4's table. A loss of 98.9 points in strong current is 148.5 return units, almost the whole 150.15 between
the two reference controllers.

## 8. Weighting test (version 5): commands past 80%

Source: `runs/shift_iw_v5/<quality>/data/seed<k>/train_law.hdf5`, seeds 0–4, training split (`infos/split` == 0). The
random flag is `infos/random`. The rows kept are those outside `truth/in_ball`.

| Quality | Thruster 1 past 80% upstream (a₁ < −0.8), all training rows | The same, rows kept after the removal | Downstream (a₁ > +0.8), all rows | Share that are uniform random commands |
|---|---|---|---|---|
| random | 9.98–10.11% | 9.99–10.12% | 9.93–10.04% | 100% |
| medium-replay | 8.21–8.43% | 8.44–8.64% | 8.14–8.36% | 100% |
| medium | 6.45–6.59% | 6.80–6.94% | 6.43–6.54% | 100% |
| medium-expert | 3.25–3.53% | 3.61–3.90% | 3.25–3.42% | 100% |
| expert | 0 | 0 | 0 | — |

What this shows:

- In the expert data, no command of any thruster goes past 80% in any seed.
- In the other qualities, every row past 80% is a uniform random command.
- The rows kept after the removal (the rows TD3+BC trains on) give 3.6–10.1% upstream. That is the nearest reading of
  the quoted "3.6–9.9%".
- Suggested wording: "none in the expert data; in the other qualities about 3% to 10% of training steps push
  thruster 1 upstream past 80%, and every one of them is a uniform random command."
- The qviewer HANDOFF's "3.5% to 10.0%" matches seed 0 alone, all rows, upstream (medium-expert 3.53%, random 9.98%).

Other facts about this data:

- Holds are 9.07–10.41% of the expert's training rows.
- The removal deletes 7,500 rows in every expert seed. For seed 0 it deletes 81 random-quality rows, 1,302
  medium-replay, 2,561 medium and 4,699 medium-expert.

## 9. Files

- `notes/p4_facts.json` holds:
  - `verified_claims`;
  - `thrust_and_cost`;
  - `stall_chart`: signed and magnitude histograms for the main test (full, kept, removed) and the weighting test's
    expert seed 0 (all, hold removed, removed);
  - `four_throttle_widget`: per-thruster percentiles and histograms, the balls, bank commands;
  - `observation_space`: both controllers, all situations, and the training data;
  - `rewards_example`;
  - `points`;
  - `difficulty`;
  - `weighting_test_data`;
  - `lower_edge_size`;
  - `weighting_test_ball`.
- Scripts: `scripts/p4_facts_compute.py`, `scripts/p4_facts_extra.py`, `scripts/p4_facts_build.py` and
  `scripts/p4_facts_iwball.py` (a ball printout).
- Intermediate files: `tmp/p4/facts/compute.json`, `extra.json` and `page_text.txt` (the current page's visible text,
  made by `scripts/p4_facts_extract.py`).
