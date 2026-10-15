# ThrusterChain: a controlled distribution-shift test for BCA

## Why this exists

BCA is a method for learning a controller only from recorded data, without trying anything new. It sits on top of
TD3+BC, a standard learner with two parts:

- **The critic** estimates how good each command is in each situation.
- **The actor** is the controller. It picks commands the critic rates highly while staying close to the recorded
  commands (the "behaviour-cloning" term).

BCA adds caution row by row. On training rows where a calibrated band on the critic's error says the critic may be
wrong, it pulls the actor harder toward the recorded command, by a multiplier between 1 and 1.5.

The claim under test has two parts:

1. BCA's per-row signal finds the training rows where the critic is wrong in a way that costs return.
2. The extra caution it puts there makes the controller do better in conditions the data did not cover, while it
   still beats the recording.

ThrusterChain checks that claim against exact answers. It is a real Gymnasium environment in which the distribution
shift is made on purpose, by dials fixed in advance, and measured exactly, state by state. Before BCA is ever run, the
test shows (or fails to show) that the shift really makes the critic wrong, really costs return, and that caution
placed by the exact harm could help. If any of that fails, BCA is not run and the report says why.

## Quick facts

| | |
|---|---|
| Import | `import experiments.shift.thruster_chain` (registers the ids below), from the repository root, **before** `import gymnasium` |
| Main id | `BCAShift/ThrusterChain-v0` |
| Action space | `Box(-1.0, 1.0, (4,), float32)`: four throttles |
| Observation space | `Box(low, high, (13,), float32)`: five floats' positions and speeds, the current, the wave |
| Episode length | 100 steps (truncation only; never terminated) |
| Discount used by every learner and by the truth | 0.98 |
| Dynamics | deterministic (no noise anywhere) |
| Gymnasium | 1.0.0, installed in its own directory (see "How to run"); importing `experiments.shift` makes it importable |
| Python | the repository's environment (`requirements.txt`: Python 3.10, numpy exactly 2.2.6, jax, flax, scipy, h5py) |

## The task

Five floats sit in a row in a channel, tied together by springs. A steady current and waves push every float
downstream. Four thrusters sit between neighbouring floats: thruster 1 pushes floats 1 and 2, thruster 2 pushes floats
2 and 3, and so on. The controller's job is to hold the floats at rest against the water while spending little
throttle.

**The physics twist.** A thruster's push grows with throttle up to 80%. Past 80% the propeller stalls and the push
falls: 0.65 at 90% throttle and 0.5 at full throttle, against 0.8 at 80%. The throttle still costs energy. A good
controller therefore pushes up to 80% and no further.

The 80% point is placed on the float32 grid: it is float32(0.8) = 0.800000011920929, the number an action of exactly
0.8 becomes in the float32 action space. So a command of exactly 80% never stalls, whether it is sent as float32 or as
float64; the next float32 above it does. (The push past 80% is therefore 3e-8 above the round numbers: 0.65000003 at
90%, 0.50000003 at full throttle.)

## Action space

| Index | Action | Range |
|---|---|---|
| 0 | throttle of thruster 1 (floats 1 and 2) | -1 to 1 |
| 1 | throttle of thruster 2 (floats 2 and 3) | -1 to 1 |
| 2 | throttle of thruster 3 (floats 3 and 4) | -1 to 1 |
| 3 | throttle of thruster 4 (floats 4 and 5) | -1 to 1 |

`step()` rounds the command to float32, then clips it to [-1, 1], as MuJoCo clamps its control range.

## Observation space

| Index | Observation | Bound |
|---|---|---|
| 0, 2, 4, 6, 8 | positions of floats 1-5 | -300 to 300 |
| 1, 3, 5, 7, 9 | speeds of floats 1-5 | -300 to 300 |
| 10 | strength of the current | 0 to 1.5 |
| 11, 12 | the wave's position in its 23-step cycle, as two numbers like the hands of a clock | -1 to 1 |

The float bound is proven, not guessed: from any allowed start the floats stay within 256.4 of rest (the matrix powers
of the float dynamics sum to 55.48 and the largest input per step is 2.82), and `step()` checks it.

## Dynamics

With s the ten float numbers, c the current, (w1, w2) the wave and a the clipped command:

- s' = A s + 0.3 B thrust(a) + 0.3 v (c + w1)
- c' = c (the current never changes during an episode)
- (w1, w2) turns by 0.27 radians per step (a cycle of 23.3 steps)

A is the five-oscillator plant of the earlier gapped linear-quadratic study in this repository
(`experiments/signal/gapped_lq.py`; radii 0.98-0.96, angles 0.10 / 0.20 / 0.35 / 0.55 / 0.80 radians, coupling seed
20261011), imported read-only. B lets thruster k push the speeds of floats k and k + 1. v
pushes the speed of every float equally: the water's push is c + w1.

**Thrust**, per thruster: thrust(a) = a while |a| <= 0.8; past 80%, thrust(a) = sign(a) (0.8 - 1.5 (|a| - 0.8)),
with 0.8 the float32 knee above. In the no-stall comparison world thrust(a) = a.

There is one dynamics function, `dynamics()`, in numpy and in JAX with the same arithmetic order. `step()`, the data
generator and the truth all call it.

## Reward

r = -(|s|^2 + 0.3 |a|^2): the squared distance of the floats from rest plus 0.3 times the squared throttles, on the
clipped command. Energy follows the throttle even while a thruster stalls. Returns are reported as the undiscounted
100-step sum, with the discounted value beside it.

## Starting state

`reset(seed=...)` draws from the start law of the id: float positions and speeds ~ N(0, 0.5^2), current ~ U[0.1, 0.7],
wave amplitude ~ U[0.1, 0.4], wave phase ~ U[0, 2 pi).

- `options={"state": 13 numbers}` restarts from an exact state (refused outside the start set: float radius at most
  100, current 0 to 1.5, wave amplitude at most 1, with 1e-9 allowed for rounding, because the wave's turn keeps the
  amplitude in exact arithmetic but float64 rounding can lift it a few units in the last place). Replay, the truth and
  the evaluation banks use it.
- `options={"start": {"current": (lo, hi), "wave": (lo, hi), "kick": sd}}` overrides the start law's ranges.

`reset` returns the float32 observation and `info = {"state": float64 copy, "t": 0}`.

## Episode end

Truncation only, at t = 100 (`terminated` is always False). The recorded log stores `terminals = False` and
`timeouts = True` at t = 99, so TD3+BC bootstraps through the time limit.

## Arguments, ids, info and render

`ThrusterChainEnv(stall=True, start="recorded", render_mode=None, stall_slope=1.5)`.

| Id | World | Start law |
|---|---|---|
| `BCAShift/ThrusterChain-v0` | stalling | recorded conditions |
| `BCAShift/ThrusterChain-StrongCurrent-v0` | stalling | current ~ U[0.7, 1.1] |
| `BCAShift/ThrusterChain-BigWaves-v0` | stalling | wave amplitude ~ U[0.4, 0.7] |
| `BCAShift/ThrusterChain-BigKicks-v0` | stalling | float start standard deviation 1.5 |
| `BCAShift/ThrusterChainNoStall-v0` and its three variants | no-stall comparison world | the same four start laws |

Step info: `state` (float64), `t`, `push` (c + w1 where the command was given), `thrust` (the push each thruster
applied), `stalled` (|a_k| past the 80% knee; a command of exactly 80% does not stall; always False in the no-stall
comparison world), `command` (the clipped command).

`render_mode="ansi"` returns one line: time, current, wave, the five float positions and the throttles, with `*` on
stalling thrusters.

## The recording controller

The log comes from one recording controller with two modes. Its numbers come from the textbook gain K: the textbook
optimal linear controller (the discounted linear-quadratic regulator) of the model in which thrust keeps rising with
throttle.

- **Normal driving** (about 87% of steps). It holds fully against the current and waves, damps the floats' swinging
  only a quarter as hard as K would, wobbles (noise with standard deviation 0.3) and never goes past 80%.
- **The full-throttle command.** In the full-throttle region (the water's push c + w1 above 0.565) it gives, half of
  the time, a fixed pattern with thruster 1 at full throttle: (-1, -0.137, -0.585, -0.406) plus a little jitter. This is
  the single most common command in the log, and in the stalling world it is a bad choice, because full throttle stalls.

The trigger depends only on the water, which no thruster changes, so both worlds share the recording controller and its
trigger.

**Reference controllers** (fixed, no data):

| Controller | What it does | Role |
|---|---|---|
| recording controller | both modes, with fixed noise and coin streams per evaluation bank | 0 points |
| stall-aware controller | K, capped at 80% throttle | 100 points in the stalling world |
| rising-thrust controller | K, capped at 100% (it assumes thrust keeps rising) | 100 points in the no-stall comparison world |
| normal driving only | the normal mode without wobble | about what cloning the gapped log gives |

Exact 100-step returns on 1,000 fixed starts per situation, stalling world:

| Situation | recording controller | stall-aware | normal driving only | rising-thrust |
|---|---|---|---|---|
| recorded conditions | -79.8 | -26.4 | -37.9 (78.4 points) | -35.1 (83.6 points) |
| strong current | -253.9 | -103.8 | -146.9 (71.3) | -335.2 (-54.1) |
| big waves | -106.8 | -36.2 | -55.6 (72.5) | -61.8 (63.8) |
| big kicks | -263.2 | -163.5 | -205.4 (58.0) | -219.2 (44.2) |

## The recorded log

Per data seed: 1,000 episodes of 100 steps, generated by stepping registered environments through `env.step` with the
command rounded to float32 first (so the stored action is exactly the applied action). Every row replays bit for bit.

- Arrays, with D4RL field names: `observations`, `actions`, `rewards`, `next_observations`, `terminals`, `timeouts`;
  plus `infos/state`, `infos/next_state` (float64), `infos/episode`, `infos/t`, `infos/split` (0 training,
  1 calibration, 2 evaluation).
- `infos/full_throttle_command` (whether the recording controller gave the full-throttle command) is a diagnostic only:
  ordinary access refuses it, and a test proves that no learner, removal rule or scorer reads it.
- Splits by episode: 500 training episodes (50,000 rows), 200 calibration episodes (a bank of 1,000 rows, 5 per
  episode), 300 evaluation episodes (held out).
- HDF5 with a JSON manifest holding SHA-256 hashes of the file and of every array.

**Which file to train on.** `python -m experiments.shift.dataset --out <dir> --seeds 0` writes, per seed:

| File | Holds | Train on it? |
|---|---|---|
| `thrusterchain_stall_seed0.hdf5` | the full log: all three splits, the infos and the recording controller's flag | **no**: it includes the held-out evaluation episodes |
| `thrusterchain_stall_seed0_train_gapped.hdf5` | the six D4RL arrays of the kept training rows after the densest-15% removal | yes: this is the test's offline dataset |
| `thrusterchain_stall_seed0_train_full.hdf5` | the six D4RL arrays of the whole training split | the zero-shift control |
| `thrusterchain_stall_seed0_train_random.hdf5` | the same after random removal of the same number of rows | the random-removal control |

In the training files, read `next_observations`, never `observations[i + 1]`: each kept row stores its own true next
observation, and after a removal the next stored row may belong to a later step (rows are never spliced, as in D4RL).
In code, the same rows come from `shift.make_removal(log, "gap", gap).kept_train_rows(log)`.

## The shift dials, and how each is measured

Every dial is set before any learner trains and is frozen in the pre-registration.

**Dial 1: which training rows are removed** (from the training split and the calibration bank, never the evaluation
split).

| Setting | What it removes | Role |
|---|---|---|
| none | nothing | the zero-shift control |
| densest 15% | the D4RL rule (`experiments/truth/gap.py`, unchanged): the 15% of recorded commands inside the densest ball | the test |
| random | the same number of rows, drawn uniformly | removes data without removing the information |

Measured on calibration seeds 900-904: the densest-15% ball removes 99.1-99.8% of the full-throttle commands. Afterwards
only 0.04-0.14% of kept training rows show any throttle past 80%, against 12-14% before. 46-48% of the full-throttle
region's rows stay in the log, with normal-driving commands, so the hole is in the commands at covered states, not in
the states.

**Dial 2: whether the removed rows hide anything.** In the stalling world thrust falls past 80%. In the no-stall
comparison world it keeps rising. Below 80% the two worlds are identical, and 99.86-99.97% of kept training rows follow
the identical rule in both. Nothing that sees only the gapped log can tell the worlds apart, so the no-stall comparison
world measures the price of caution when the unseen is harmless.

**Dial 3: where the controller is tested.** Four fixed banks of 1,000 seeded starts, the same for every controller:

| Situation | Start law | Outside the recorded range |
|---|---|---|
| recorded conditions | the training law | none |
| strong current | current ~ U[0.7, 1.1] | every start (recorded maximum 0.7) |
| big waves | wave amplitude ~ U[0.4, 0.7] | every start (recorded maximum 0.4) |
| big kicks | float start standard deviation 1.5 | floats beyond the training 99th percentile on part of the steps |

**The shift card**, measured exactly for every (log, policy, situation) on the states the policy visits:

1. removed share m(s): the probability that the recording controller's command at that state falls in the removed
   ball (Monte Carlo, 4,096 draws);
2. unrecorded-throttle share: steps commanding past 80% on any thruster;
3. full-throttle-region share;
4. outside the recorded conditions: current above 0.7, wave above 0.4, or floats beyond the training 99th percentile;
5. inside the removed ball, split into past 80% and at or below 80%;
6. distance to the data: the 10th nearest training row in standardised (state, command) space;
7. covered share: the share of unrecorded-throttle steps that happen inside the recorded conditions;
8. loss from unrecorded throttle: the return of the same policy capped at 80% at evaluation, minus its own return.

## Controls

| Control | What it rules out |
|---|---|
| zero shift (nothing removed) | harm that is not caused by the removal |
| random removal of the same number of rows | harm from simply having less data |
| no-stall comparison world | harm that does not need the hidden stall |
| placebo critic (another initialisation) | seed noise |
| noise map (the spread of TD3's smoothed target under a perfect critic) | caution that only follows target noise |
| amount-matched, shuffled and strength-matched caution | caution that helps for its amount, not its placement |

## Truth and its precision

The dynamics are deterministic, so for any deterministic policy (learned actors included, with their weights cast to
float64) the true action value is one float64 rollout through the same `dynamics()`:
Q(s, a) = r(s, a) + sum over t = 1 ... 999 of 0.98^t r(s_t, pi(s_t)).

| Check | Measured |
|---|---|
| 1,000 vs 2,000 steps | 1.2e-8 |
| Bellman residual | 5e-14 |
| closed form in the no-stall comparison world (linear policy) | 1.9e-14 relative |
| JAX vs numpy | 7e-15 |
| autodiff slope vs central differences | within 1e-6 |
| attribution terms vs return difference | within 1e-8 |

Per-row labels, at the float32 inputs the critic sees:

- **over-preference** D = [(critic(s, pi(s)) - critic(s, a)) - (true(s, pi(s)) - true(s, a))]+ : how much more the
  critic prefers the policy's command over the recorded one than the truth does. This is exactly what extra cloning at
  that row corrects, and it is the primary harm label.
- **pull past 80%**: [critic(s, c_hi) - critic(s, c_cap)] - [true(s, c_hi) - true(s, c_cap)], with c_hi the
  full-throttle pattern and c_cap the same pattern capped at 80%: how much more the critic prefers pushing past 80% than
  the truth does.
- **advantage error**: the critic's error at a command minus its error at the policy's command (the level cancels).

Points, per situation, on identical starts for every arm: 100 x (the controller's 100-step return minus the recording
controller's) / (the stall-aware controller's 100-step return minus the recording controller's).
Exact per-step attribution splits a return difference into the steps where it is lost.

## The harm checks (no BCA code runs here)

On gate seeds 0-4, with the pass rules fixed in the pre-registration:

- **H1: is the critic wrong where data are missing?** Critics that score one fixed controller (the stall-aware
  controller; the rising-thrust controller is reported) are trained on the full, gapped, randomly thinned and placebo
  logs, and on both logs of the no-stall comparison world. Everything but the data is equal. The gapped critic must be
  at least twice as wrong at the removed commands, no more wrong far from the gap, not explained by random removal or
  by another initialisation, at least 1.5 times worse in the stalling world than in the no-stall comparison world, and
  it must prefer a pull past 80% more than the full critic does.
- **H2: does acting on that error cost return?** From one cloned actor, actor-only TD3+BC runs differ only in their
  frozen critic. In strong current, the gapped critic's actor must lose at least 5 points, more than random removal and
  the no-stall comparison world explain, push past 80% at least 1.5 times as often, and lose its return at those steps:
  the loss rate there must be positive and at least twice the loss rate at its other steps (if the other steps lose
  nothing, any positive loss rate at the past-80% steps counts).
- **The alpha ladder rule** (an amendment to the design, fixed before any outcome). H2 runs first at TD3+BC's
  published alpha 2.5. If it fails there, it reruns at
  alpha 5, then at alpha 10. The first alpha at which H1 and H2 both pass is used for every later arm. The choice never
  looks at any BCA output.
- **H3: could caution help?** On the frozen plain TD3+BC critic, caution placed by the exact over-preference in BCA's
  own dose form (multipliers in [1, 1.5)) must beat, by at least 3 points, even caution of the same strength (matched
  blind, before any return is computed), the same amount of caution spread evenly, and the same doses shuffled. A
  situation that passes is room-certified. The best even caution and a cap-2 diagnostic (doses up to 2) are shown.
- **H4: how many seeds?** The spread of two paired differences of co-trained runs (the larger standard deviation
  across the gate seeds) gives N, the number of confirmation seeds; N above 20 stops the test.

**Stop rule.** If H1 fails, or H2 fails at every alpha, the one pre-registered retune runs (stall slope 3.0, fresh
seeds 5-9). If that fails too, the report says "this world does not produce critic-caused harm from the missing common
command" and BCA is never run. No room-certified situation, or N above 20, also stops the test. A failed seed fails a
pooled mark; seeds are never dropped or replaced.

## The BCA test (only after the user approves)

The gate report is written, and the user approves it by typing the start of its SHA-256 at a terminal. Every BCA arm,
calibrator fit, threshold refresh and dose readout refuses without that approval file. Then, on fresh confirmation
seeds 100 to 99 + N, TD3+BC with BCA (algorithms/td3_bc_bca.py unchanged) is compared with plain TD3+BC, BCA's own
doses spread evenly, shuffled and placed by the noise map, even caution of the same strength, the best even caution
and behaviour cloning.

Tests that are judged together use Holm's correction, a correction for testing several things at once.

- **Detection** (three tests, judged together): does BCA's dose rank rows by the exact over-preference (a row among the
  10% most over-preferred gets a larger dose than another row at least 60% of the time, against 50% by chance), put
  more caution in the full-throttle region on the gapped log than on the full log, and follow the over-preference
  beyond target noise and the region?
- **Policy**: BCA minus plain TD3+BC on each room-certified situation (judged together), then BCA against the same
  doses spread evenly, shuffled, and even caution of the same strength (judged together); no loss in recorded
  conditions; still at least 5 points better than cloning; no harm on the full log.
- **Failed runs count against BCA.** A failed BCA run scores 0 points (the recording controller's level). A failed
  comparison run, the blind placements of BCA's own doses included, is rerun once; if it fails again its difference
  with BCA is set to 0. All arms of a seed must share data, initial parameters and batch-row stream (checked by hash).

The verdict sentences are fixed in advance, from "BCA finds the harm, and its caution, placed where it finds it,
improves returns out of distribution while still improving on the recording" down to "BCA costs performance" and
"Inconclusive". Two in-between sentences cover a gain whose placement matters but whose doses do not follow the exact
critic error, and a placed gain that no longer beats the recording by 5 points; the full order is in
`acceptance.VERDICT_ORDER`.

## How to run

Every command runs **from the repository root** with **the repository's Python environment** (`requirements.txt`:
Python 3.10 with numpy exactly 2.2.6, jax, flax, scipy and h5py); below, `python` means that environment's Python.
`experiments/shift` refuses to import with any other numpy.

Gymnasium is not installed into that environment. It lives in its own directory (`BCA_SHIFT_DEPS`, default
`~/.cache/bca-shift/py`), which importing `experiments.shift` appends to the end of `sys.path`. So import
`experiments.shift` (any of its modules) **before** `gymnasium`. To use gymnasium without the package, put the directory
on the path yourself: `PYTHONPATH=$HOME/.cache/bca-shift/py:$PWD python ...`.

```sh
# once: gymnasium 1.0.0 into its own directory (never into the shared Python environment)
BCA_SHIFT_PYTHON=/path/to/your/venv/bin/python sh experiments/shift/install_deps.sh

# every command of the gate, in order
python -m experiments.shift.acceptance plan

# a single episode
python -c "import experiments.shift.thruster_chain, gymnasium as gym; env = gym.make('BCAShift/ThrusterChain-v0', render_mode='ansi'); env.reset(seed=0); env.step(env.action_space.sample()); print(env.render())"

# tests (CPU)
JAX_PLATFORMS=cpu python -m unittest experiments.shift.test_thruster_chain experiments.shift.test_dataset \
    experiments.shift.test_shift experiments.shift.test_truth experiments.shift.test_learners \
    experiments.shift.test_acceptance
```

**Evaluate a controller as the reported numbers do.** The reported returns and points use the four fixed evaluation
banks, `thruster_chain.bank_starts(situation)` (1,000 starts each, the same for every controller), entered through
`reset(options={"state": ...})`. `reset(seed=i)` draws from the same start law but not from the fixed bank, so its
averages differ slightly from the reported ones.

```python
import numpy as np
import experiments.shift.thruster_chain as TC
import experiments.shift.recorder as RC
import gymnasium as gym

controller = RC.stall_aware()                      # any function from the 13-number state to 4 throttles
env = gym.make("BCAShift/ThrusterChain-StrongCurrent-v0")
returns = []
for start in TC.bank_starts("strong_current"):     # the fixed bank of 1,000 starts
    obs, info = env.reset(options={"state": start})
    total, done = 0.0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(controller(info["state"]).astype(np.float32))
        total, done = total + reward, terminated or truncated
    returns.append(total)
print(f"mean 100-step return: {np.mean(returns):.1f}")   # -103.8 for the stall-aware controller in strong current
```

Run the gate with `JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 nice -n 5`: the pre-registration records these flags (and the
package versions and the dependency directory's hash), and every later stage refuses if they differ. Outputs go to
`runs/shift/pre/` before the pre-registration and to `runs/shift/<first 16 hex of the pre-registration's SHA-256>/`
after it; every JSON has a `.sha256` sidecar. The real preset is pre-registered and run only in `runs/shift`; to check
the code without touching `runs/shift`, use `--runs <temporary directory> --preset smoke`.

## Costs

Budget figures, one process per CPU:

| Run | Minutes |
|---|---|
| plain TD3+BC, 100,000 updates | 8 |
| fixed-controller critic, 100,000 updates | 6 |
| BCA-family run | 12 |
| behaviour cloning | 1.5 |

Before the stop: about 13 process-hours (about 25 more if the retune runs). The BCA stage: about 115 minutes per
confirmation seed (about 21 process-hours at N = 10, 40 at N = 20).

## How to read the results

- **Points**: 0 is the recording controller, 100 the stall-aware controller; more than 100 is possible.
- **Harm-certified**: the missing common command really misleads TD3+BC's critic and costs return.
- **Room-certified**: caution placed by the true harm could help in that situation. BCA is judged only there.
- **What to expect**: TD3+BC at alpha 2.5 stays close to the recorded commands, so the harm lies mostly in conditions
  beyond the recording, and a stop before BCA is the most likely single outcome. That is a real finding for the D4RL
  Hopper run. A positive result would show that BCA can find and reduce this kind of harm; it would not predict
  results on Hopper.

## Files

| File | Holds |
|---|---|
| `thruster_chain.py` | the environment, the one dynamics function, the ids, start laws and evaluation banks |
| `recorder.py` | the textbook gain, the recording controller and the reference controllers |
| `dataset.py` | the recorded log, HDF5 and manifest, splits, calibration bank, replay |
| `shift.py` | the removal rule, random removal, frozen pools, m(s) and the shift card |
| `truth.py` | exact values, slopes, over-preference, returns, points, attribution |
| `stage0.py` | the outcome-free world card (marks T1-T9) and forecasts |
| `learners.py` | plain TD3+BC, fixed-controller critics, cloning, actor-only arms, matching, BCA (switched off) |
| `detection.py` | statistics, detection measures, comparators and state maps |
| `acceptance.py` | every stage, the pre-registration and its hash check, the stop rule and the verdict |
