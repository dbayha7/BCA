# Step 2 of the restructured BCA: the policy pilot (pre-specification)

**Exploratory. Designed after seeing seeds 0-4 and after step 1. Written and frozen before any step-2 actor is
trained.**

This file says in plain words what `prespec.json` fixes in machine-readable form. If the two ever differ,
`prespec.json` is what the code reads. Both are frozen by `freeze.py` (SHA-256 in `prespec.json.sha256` and
`code_manifest.json`). The training, evaluation and pooled scripts refuse to run if any frozen file has changed, and
`freeze.py` refuses to run again once any official step-2 output exists.

It sits outside the gate. It changes no pre-registered rule, and no file of version 6, version 7 or step 1. It
writes only under `runs/shift_explore/restructure/step2/`.

## What was approved

The user approved step 2 training (relayed to this session, 14 Oct 2026). The user fixed the design below and asked
that these rules be written, frozen and committed before any official training. Nothing else is approved: only seeds
0-4, and no retuning on these seeds.

## Words used here

- **Seed**: one of five repeats of the whole experiment (0-4), each with its own recorded log and its own trained
  networks. Seeds 5-9 belong to version 7's confirmatory run and seeds 10-14 to a later confirmation; the code refuses
  them.
- **World**: the simulated thruster system. In the **stalling world**, thrust collapses past 80% throttle, so
  full-throttle commands are harmful. In the **no-stall world**, thrust keeps rising, so full throttle helps. Each
  world has its own recorded log, made by the same controller. That controller reacts to what happens, so once the
  outcomes differ, the states differ, and so do the commands it records: the two worlds' recorded commands are alike
  but not identical (93-99% of the gapped training rows differ, by up to 1.33-1.51). Step 1's gap rule therefore
  gives each world slightly different ends (see The arms).
- **Actor**: the network that picks a command from the state. **Critic**: the network that predicts the value of a
  command. Here the critic is frozen: the gate trained it on the recorded log, and step 2 never changes it.
- **none**: the actor trained against the frozen critic with no caution (the gate's check 2 saved it). **clone**:
  the actor that copies the recorded commands. **full_none**: like none, but with a critic trained on the full log
  (a reference for what the missing data cost).
- **Points**: the score of an actor, 0 for the recording controller and 100 for the best known controller, from exact
  100-step returns. **Situations**: recorded conditions, strong current (where full throttle is tempting; the scored
  situation), big waves, big kicks.
- **Throttle index** of a command: the largest of its four thruster magnitudes, `t = max |a_j|`.
- **The gap, t_lo and t_up**: in the recorded log, almost no commands lie between about 0.79 and 0.98-0.995 throttle.
  Step 1's gap rule found this empty band from the commands alone: `t_lo` is its lower end, `t_up` its upper end.
- **Value unit**: the critic's own unit (one unit of predicted discounted reward).
- **Certificate** (step 1): BCA's yes or no, per seed and world, that the critic is credibly more over-optimistic on
  full-throttle steps than on the rest. **Height** (step 1): how much more, `R_full - R_rest`, in value units, from
  BCA's own thresholds.
- **Penalty**: a term subtracted from what the actor maximises, so that it avoids some commands.
- **Strength** of an actor: how far its commands moved from none's, as the root-mean-square change on 5,000 held-out
  recorded states.
- **Matched**: a control or rival tuned, from strengths alone, to move the actor exactly as far as the primary arm
  does, before any return is computed.
- **Controls** (same strength, no information): the **even pull** (a stronger pull towards the recorded commands
  everywhere), **push less** (a penalty on large commands everywhere) and two **random surfaces** (penalties shaped by
  two fixed random networks).
- **Range hinge**: a penalty on commands outside the recorded range. **Stand-in**: the best data-only rival, the range
  hinge with the stand-ins' own matching (version 6's choice). **Perfect band**: the room test's oracle penalty, which
  knows the true one-step error (a reference, not a method).
- **World-blind rival**: a penalty that never reads an outcome. Its knobs and coefficients are carried over from the
  stalling world; what it reads from data (the ramp's or the range's ends, the recorded commands it pulls towards,
  the states) is each world's own. So it does nearly, not exactly, the same in both worlds.
- **Lock**: the file of every trained actor's fingerprint (SHA-256), written before any return is computed.
- **Side by side**: arms of one penalty kind trained together in one program ("vmapped"), from the same start and on
  the same batches.
- **Check (b)**: version 6's plumbing check: a penalty switched off must give exactly none's actor.
- **Phase**: one step of the work run as one process (for example "train the stalling world of seed 2").

## Why this test

Step 1 showed that BCA, fed held-out residuals, switches caution on in all five stalling-world seeds and off in all
five no-stall seeds, and that the gated deployment check then nets 106 points across worlds. Step 2 asks whether the
same outcome-reading signal also works as a training penalty: does an actor trained with it avoid full throttle where
full throttle harms, and keep it where it helps, better than world-blind rivals?

## What is trained (version 6's protocol, unchanged)

Per seed (0-4) and world, version 6's actor-only policy test, on the room harness:

- the gate's frozen fixed-controller critic of that world, trained on the log with the gap;
- start from the gate's saved cloned actor of that world;
- alpha 5; 20,000 actor steps in 4,000-step chunks; batches of 256 rows from check 2's fixed stream
  (`default_rng([20261015, seed, 35])`), the same for every arm; Adam, learning rate 3e-4;
- the arms of one kind side by side.

Every process runs pinned to one CPU, with `OMP_NUM_THREADS=1`, `JAX_PLATFORMS=cpu` and no bytecode.

## The new penalty: the outcome ramp

Each outcome-ramp arm has a switch, a height `h` and a multiplier `k`, all read from step 1's results (no refit).

- `U(a) = h * ramp(t(a))`, with `ramp(t) = clip((t - t_lo) / (t_up - t_lo), 0, 1)`: zero up to `t_lo`, rising
  linearly to `h` at `t_up`, flat after. `t_lo` and `t_up` are that seed-world's gap.
- The actor's loss is version 6's paired form, exactly:
  `-lambda * mean[ Q1(s, pi(s)) - k * [U(pi(s)) - U(a_rec)]_+ ] + mean (pi(s) - a_rec)^2`,
  with `lambda = 5 / mean |Q1|` (not differentiated) and `a_rec` the row's recorded command. `[x]_+` means
  `max(x, 0)`: the actor is charged only for going further up the ramp than the recorded command did.
- Only the actor learns. With the switch off, `h = 0`: the penalty and its gradient are exactly zero.

## The arms

The six outcome-ramp arms:

| arm | switch | height h | k |
|---|---|---|---|
| `obca_k1` (PRIMARY) | step 1's BCA certificate | BCA height when on, else 0 | 1 |
| `obca_k2` | the same | the same | 2 |
| `obca_forced` (world-blind) | always on | this seed's stalling-world BCA height, in both worlds | 1 |
| `fixed_gated` | BCA certificate | 1.0 value unit (a declared default) when on | 1 |
| `conformal_k1` | plain split conformal: 90% quantile of the full-throttle residuals minus that of the rest, above 0.10 | that difference when on | 1 |
| `meangap_k1` | step 1's rank test, p < 0.05 | mean full-throttle residual minus mean of the rest, floored at 0, when on | 1 |

What this gives on seeds 0-4 (computed from step 1's results; the code refuses if it computes anything else):

Stalling world, every switch on:

| seed | t_lo | t_up | obca h (k1, k2, forced) | fixed_gated | conformal_k1 | meangap_k1 |
|---|---|---|---|---|---|---|
| 0 | 0.7860 | 0.9953 | 6.019 | 1.0 | 1.733 | 0.657 |
| 1 | 0.7868 | 0.9874 | 2.420 | 1.0 | 1.446 | 0.681 |
| 2 | 0.7914 | 0.9766 | 2.065 | 1.0 | 1.096 | 0.533 |
| 3 | 0.7910 | 0.9876 | 1.605 | 1.0 | 1.199 | 0.492 |
| 4 | 0.7871 | 0.9893 | 1.796 | 1.0 | 1.170 | 0.539 |

No-stall world: the BCA certificate, the conformal switch and (except in seed 2) the rank-test switch are off, so
`obca_k1`, `obca_k2`, `fixed_gated`, `conformal_k1` and `meangap_k1` have `h = 0`. In seed 2 the rank test is on
there (p = 0.032), so `meangap_k1` has `h = 0.048` in that one seed-world. `obca_forced` keeps the stalling heights
above, on the no-stall world's own ramp ends. In seed 3 of the no-stall world, step 1's height is infinite, but the
certificate is off, so `h = 0` and nothing is refused. (The code refuses if a switched-on arm ever has an infinite
height.)

The ramp ends are each world's own, from that world's recorded commands only, never from outcomes. Because the two
logs differ slightly (see Words), `t_lo` differs between the worlds in every seed (by 0.0002 to 0.0084: lower in the
no-stall world in seeds 1-4, higher in seed 0), and `t_up` differs in seed 4:

| seed | t_lo, stalling | t_lo, no-stall | t_up, stalling | t_up, no-stall |
|---|---|---|---|---|
| 0 | 0.7860 | 0.7863 | 0.9953 | 0.9953 |
| 1 | 0.7868 | 0.7843 | 0.9874 | 0.9874 |
| 2 | 0.7914 | 0.7829 | 0.9766 | 0.9766 |
| 3 | 0.7910 | 0.7840 | 0.9876 | 0.9876 |
| 4 | 0.7871 | 0.7854 | 0.9893 | 0.9935 |

What is carried over from the stalling world is only `h` and `k` for `obca_forced`, and only the locked knobs and
coefficients for the controls and the stand-in.

In the stalling world, `obca_forced` has exactly `obca_k1`'s switch, height and k (the certificate is on in all five
seeds). Both are trained, as separate arms in the same program.

**Also trained in the stalling world** (version 6's set):

- `w0_check`: the ramp program with `h = 0` and `k = 0` (check (b); not scored);
- `perfect_band` (reference) and the room's `range_hinge` at k = 1 (reported);
- the controls' ladders (even pull, push less, two random surfaces), then each control matched blind to `obca_k1`'s
  strength with version 6's machinery (push less is reported only);
- the stand-in (range hinge) with its own 4-rung ladder and matcher, matched blind to `obca_k1`'s strength;
- scored beside them, already saved: `none`, `clone`, `full_none`.

**In the no-stall world** (version 6's no-stall pattern, after the stalling world's lock of the same seed):

- `none`, `clone`;
- the six outcome-ramp arms with their no-stall switch and height, and `w0_check`, side by side;
- the even pull at the stalling world's matched pull;
- the two random surfaces and push less at the stalling world's matched coefficient (the same penalty function and
  coefficient in both worlds, applied to that world's own rows);
- the stand-in at its stalling-world coefficient;
- the range hinge and the perfect band.

## Built-in checks (each stops its phase on failure)

- **Gap rule.** Step 1's gap rule, re-run on that world's recorded commands, must give the table's `t_lo` and `t_up`
  exactly.
- **Check (b).** `w0_check` must reproduce none's saved actor within 1e-5 on every parameter (in the stalling world
  it runs alone and first, as in version 6).
- **Height 0 means none.** Every outcome-ramp arm whose height is 0 in that seed-world must also reproduce none
  within 1e-5. In the no-stall world that is four or five arms per seed: a strong test of the switch plumbing.

A stopped phase writes `stop.json` (write-once). It is never rerun or replaced. Every condition that needs that
seed-world then fails, so the verdict is NO-GO.

## Matching

- **Target**: the strength of `obca_k1` in the stalling world, as version 6 did for its primary arm.
- **Controls**: version 6's `match_controls`, unchanged (ladders, the blind ladder extension, the gate's matcher,
  tolerance 1%, at most 5 rounds).
- **Stand-in**: the stand-ins' own ladder and matcher (version 6's calls).
- If `obca_k1`'s strength is not positive, every control and the stand-in are recorded unmatched, and every
  condition that needs them fails.
- An error inside the stand-in rival is recorded and counts against BCA, as in version 6.

## Lock before returns

After every arm of a phase is trained and matched, `actors.msgpack` and `lock.json` are written (write-once), with
every actor's fingerprint, the checks, the strengths, the matches, the timings and the peak memory. Then:

- the no-stall phase of a seed starts only after that seed's stalling-world lock, and reads its locked knobs;
- **no evaluation starts until every seed-world (5 seeds x 2 worlds) has a lock or a stop.**

## Evaluation (version 6's)

For every scored actor of a locked seed-world:

- exact 100-step returns on the full 1,000-start banks of all four situations, points against the gate's anchors,
  the same actor capped at 80%, the share of steps past 80%;
- from one more exact rollout per situation: the share of steps past 80% per thruster, and the share of visited
  steps whose throttle index lies above `t_lo` (where the actor actually goes).

Output: `values.json` per seed-world, write-once.

## Pooled measures (strong current; the mean over seeds 0-4 of per-seed differences)

**NET** of an arm x:

`NET(x) = [points of x minus points of none, stalling world] minus [points of none minus points of x, no-stall world]`

It is what the arm gains where full throttle harms, minus what it costs where full throttle helps. Every seed is
needed: a missing, stopped, failed or unmatched value leaves NET undefined, and a condition that needs it fails.

**World-blind rivals W**: the stand-in (matched in the stalling world, locked in the no-stall world), the even pull
(matched / locked), the two random surfaces (matched / locked) and `obca_forced`. None of them reads an outcome; each
reads each world's own commands and states (see Words).

## Go / no-go (fixed before training)

- **G1.** `NET(obca_k1)` is at least the largest NET in W plus 10.
- **G2.** It keeps the improvement in the stalling world: `obca_k1` minus none is at least -3 in recorded
  conditions, in big waves and in big kicks.
- **G3.** In the stalling world, strong current, `obca_k1` minus the matched even pull is at least 15 (version 6's
  M1 margin).

**GO** if all three hold. That licenses drafting version 8 (fresh seeds 10-14). It is still exploratory, not a
claim.

**NO-GO** (kill rule): outcome-reading BCA does not carry over into trained policies across worlds. It is shelved
as a training penalty and kept only as the deployment check. There is no retuning on these seeds.

## Attribution (reported; it decides only the wording of the claim)

- **A1**: if `NET(conformal_k1)` is at least `NET(obca_k1) - 5`, the Bayesian part adds nothing measurable.
- **A2**: if `NET(fixed_gated)` is at least `NET(obca_k1) - 5`, the calibrated height adds nothing beyond the switch.
- **A3**: if `NET(meangap_k1)` is at least `NET(obca_k1) - 5`, calibration adds nothing beyond raw average
  residuals.

What the A comparisons can tell apart (added by the review below, before training). On seeds 0-4 the three rivals
have BCA's switch in every seed-world (`meangap_k1`: all but no_stall/seed2, where `h = 0.048`), and in the stalling
world each rival's height is below BCA's in every seed (meangap 0.49-0.68 < fixed 1.0 < conformal 1.10-1.73 <
BCA 1.61-6.02). So each A compares one penalty shape at two doses. The wording is therefore fixed both ways:

- an A that holds: the meaning above;
- an A that fails: "a larger dose did better", and nothing is attributed to the Bayesian part or to calibration
  (the exact texts are in `prespec.json`, `attribution.wording`);
- **no credit to calibration on these seeds**, whatever the outcome: no arm gives a non-BCA height as large as
  BCA's, so nothing separates a calibrated height from a larger one;
- **dose readout** (reported): `NET(obca_k2) - NET(obca_k1)` above +5 means BCA's height lies on the rising part of
  the dose curve (more penalty helps; BCA's height is not an optimum), within 5 a plateau (BCA's height is large
  enough, which is not the same as calibrated), below -5 that the best dose lies below twice BCA's height;
- every A is also reported per seed (each seed's NET difference, and in how many seeds it is at least -5), beside a
  per-seed dose table (dose `k * h`, stalling gain over none, NET term) for `meangap_k1`, `fixed_gated`,
  `conformal_k1`, `obca_k1` and `obca_k2`. Seed 0 carries the largest BCA-to-conformal ratio (3.47, against
  1.34-1.88 in the other seeds), so it can drive the means.

## Also reported (decides nothing)

- Every arm's NET, with `clone` shown as a baseline; NET in recorded conditions.
- The attribution per seed, the dose readout and the per-seed dose table (above).
- Version 6's marks restated for `obca_k1`: M1 (minus the matched even pull at least 15, positive in at least 4 of 5
  seeds), M2 (each random surface at least 15), M3 (recorded conditions at least none - 3 and clone + 25; big waves
  and big kicks at least none - 3), M4 (minus the matched stand-in at least -5).
- Points of every arm in every world and situation; the share of visited steps past `t_lo`.
- The checks, strengths and matches; the ramp's shape (on the room's 2,000 full-throttle-region states, how often
  the best grid throttle for `Q1 - k [U(a) - U(a_rec)]_+` lies past 80%).
- Time and peak memory (VmHWM) of every phase.

## Departures and judgement calls

1. **Random surfaces and push less in the no-stall world** run at the stalling world's matched coefficient (the knob
   times the stalling world's scale): the same penalty function and coefficient in both worlds, applied to that
   world's own rows, as version 6 did for the stand-in. The knob is recorded beside it.
2. **Push less is added to the no-stall arms** (reported only), so that every matched control has a NET. It is not
   in W and decides nothing.
3. **Evaluations wait for every lock** (version 6 waited only for its own seed's lock).
4. **A height-0 arm that misses none by more than 1e-5 stops its phase**, like version 6's check (b).
5. **The stand-in calls are restated** (the same calls in the same order as version 6's `standin_matched`) only so
   that a scratch run can pass a smaller step count.
6. **The no-stall phase loads only the no-stall world's data** (version 6 also loaded the stalling world's, unused).
7. **Version 6's shift cards are not computed** (not requested). The per-thruster shares come from the same extra
   rollout as the past-`t_lo` share.
8. **Version 6's code is used without its approval checks.** It is executed from verified bytes (the copy in
   `outputs/shift/beyond-the-data/stage2/code`, byte-identical to what version 6's manifest pins): `stage2_common.py`,
   `weights.py` (which `band.py` imports), `band.py`, `fresh.py` and `policy.py`. Step 2 refuses if any module of
   that folder or of step 1's folder was loaded without its hash checked. None of version 6's phase functions or
   writers is called: they all require version 6's approval or write into version 6's folder.
9. **The plain split-conformal switch** uses step 1's 90% quantiles of the two groups, as the design says. BCA's
   own plain gap (`plain_gap`) is recorded beside it; on seeds 0-4 the two agree to four decimals.
10. **The attribution's wording is fixed both ways** (review below): a failed A is worded as dose, nothing is
    credited to calibration on these seeds, and a dose readout and per-seed reports are added. Reported only.
11. **weights.py is pinned and every import from the pinned folders is checked** (review below).

## The machine

This runs on the user's laptop (AMD Ryzen AI MAX+ 395, WSL Ubuntu 24.04), not the cloud machine that ran the gate,
version 6 and step 1. Checked while preparing (scratch folders, nothing kept in the repository):

- the gate's own exact checks pass unchanged here (data regeneration and anchors);
- none's rollouts reproduce step 1's numbers to the last digit (-13.646482434566705 and 96.30729831423376 points);
- the room's switched-off training program reproduces check 2's saved none actors bit for bit (seed 0, both worlds).

So version 7's constants wrapper is not used.

## Verification while preparing (scratch folders outside the repository)

These runs produce no evidence about the outcome. They used the code as first frozen (the smoke run recorded code
manifest `d80af4f7...`). The review below changed the code afterwards and re-ran the smoke run on the final code.

**Reproduction check** (`train_policy.py --w0-repro`, seed 0, the full 20,000 steps). The ramp program with every
height set to 0 reproduced check 2's saved none actor bit for bit: largest parameter difference 0.0, equal SHA-256.

- Stalling world: 1 arm, 36 s.
- No-stall world: 7 arms side by side (the width of the real no-stall program), 264 s.

No scored arm was trained and no return was computed. So check (b) and the height-0 checks can pass on this
machine.

**Smoke run** (`STEP2_SCRATCH=/root/step2_scratch/smoke_final STEP2_STEPS=40 STEP2_CHUNK=20 STEP2_SEEDS=0
STEP2_LANES=1 sh run_step2.sh`). Exit 0 after 8 minutes 15 seconds:

- the 26 unit tests passed;
- both worlds trained and locked, and every check passed: the gap rule exactly, and check (b) and the five no-stall
  height-0 arms against the scratch reference (difference 0.0);
- `obca_k1` had a positive strength, so all four controls and the stand-in were matched, then locked in the no-stall
  world;
- both evaluations and the pooled look ran.

An earlier smoke run gave the same numbers to the last digit. Its verdict means nothing: 40-step actors.

| phase (40 steps) | wall time | peak memory (VmHWM) |
|---|---|---|
| stalling world, training | 153 s | 1,174 MB |
| no-stall world, training | 149 s | 1,085 MB |
| stalling world, evaluation | 108 s | 384 MB |
| no-stall world, evaluation | 112 s | 368 MB |

At 40 steps the training time is almost all fixed cost: loading the gate's data (12-21 s), compiling, and the
stand-in's data side (60-77 s).

**Timing probe** (400 steps per kind, one arm, actors discarded, no return). Full-length cost per arm on one CPU:

| kind | seconds per arm for 20,000 steps |
|---|---|
| outcome ramp | 39 |
| plain (even pull) | 45 |
| push less | 42 |
| random surface | 47 |
| range hinge (room) | 71 |
| range hinge (stand-in) | 91 |
| perfect band | 128 |

**Refusals.** Each of these refused without writing anything:

- official training without taskset, with a reserved seed (5, 12), with `--steps`, with `OMP_NUM_THREADS=4`, or with
  a scratch folder inside the repository;
- an official evaluation before every lock exists;
- an official pooled look with nothing to pool, or with `--seeds`;
- the official runner with a scratch setting.

`runs/shift_explore/restructure/` still does not exist. No new `__pycache__` appeared in the repository.

**Machine load.** Other sessions ran heavy jobs on this laptop during these measurements (one used about 22 cores).
The times above are therefore on the high side. When the laptop was otherwise idle, the plain program took 31 s per
arm instead of 45.

## Budget (full size, extrapolated)

From the arms the smoke run trained per kind and the probe's costs:

- **stalling-world phase**: about 35-50 CPU-minutes. The smoke run's arms give 39 minutes; at full length the
  matching may need a few more rungs (version 6's seed 0 needed 6 more random-surface and 3 more push-less rungs).
- **no-stall phase**: about 13-17 CPU-minutes;
- **each evaluation**: about 2 CPU-minutes;
- **in all**: about 5-5.5 CPU-hours. With 4 lanes, about 2 hours of wall time (four seeds side by side, then seed 4,
  then the evaluations). With `STEP2_LANES=5`, about 1 hour 10 minutes.

Peak memory is about 1.2 GB per phase, so memory is not the limit. By default a lane waits for 4,000 MB free before
starting a phase.

## To run step 2

Detached, so that closing the session cannot stop it half-way:

```
setsid nohup sh /home/user/bca/outputs/shift/beyond-the-data/restructure/step2/run_step2.sh </dev/null >/dev/null 2>&1 &
```

On this laptop, Ubuntu (WSL) shuts down when nothing runs in it, and that would stop the run. Keep one Ubuntu window
open for the whole run, or start it in tmux, as version 7's README does:

```
tmux new -d -s step2 'sh /home/user/bca/outputs/shift/beyond-the-data/restructure/step2/run_step2.sh'
```

- **Progress**: `runs/shift_explore/restructure/step2/logs/run.log`. The verdict is its last line and the heading of
  `results.md`.
- **Settings**: `STEP2_LANES` (default 4), `STEP2_CPUS` (default one CPU per physical core: 0 2 4 ...),
  `STEP2_MIN_FREE_MB` (default 4000), `PY` (default `/root/bcavenv/bin/python`).
- **Other jobs**: avoid heavy jobs at the same time. They slow every phase but do not change any number (each phase
  is pinned to one CPU and deterministic).
- **Interruptions**: rerunning the script skips every finished phase. A phase interrupted before its lock may be
  rerun (it is deterministic). An `actors.msgpack` without a `lock.json` means an interrupted write: inspect it by
  hand. Killed by a signal, the runner keeps `run.lock`; remove it once no phase is running.

## Files

| File | Role |
|---|---|
| `prespec.json`, `prespec.json.sha256` | the frozen specification |
| `code_manifest.json` | SHA-256 of every file below |
| `step2lib.py` | the arm table, the ramp and penalty, NET, the go/no-go, attribution and dose reports, output guards, the verified-import helpers |
| `machinery.py` | loads the gate, the room harness, version 6's code and step 1's gap rule from verified bytes and checks that nothing from those folders bypassed the check; the outcome-ramp runner |
| `train_policy.py` | one world of one seed per call; official or scratch mode; the reproduction check |
| `evaluate_policy.py` | exact returns of one locked seed-world; `values.json`, write-once |
| `pooled.py` | `results.json` and `results.md`, write-once |
| `run_step2.sh` | unit tests, training over 4 lanes, evaluations, pooled look. `STEP2_SCRATCH=<dir>` runs a tiny smoke version outside the repository |
| `test_step2.py` | unit tests on synthetic data |
| `freeze.py` | writes the two hash files |

## Review notes (independent review, 15 Oct 2026, before any official training)

Independent reviewers found three defects. Two skeptical checks confirmed each one. No official step-2 output
existed (`runs/shift_explore/restructure/` does not exist), so the specification could still be changed and frozen
again.

Changes made, and why:

1. **What this file said about the two worlds' commands** (Words, The arms, Pooled measures, departure 1;
   `prespec.json` `measures.world_blind_meaning`). This file said that the two worlds' recorded commands were
   identical and only the outcomes differed. That is false. Each world's log was recorded separately, by a controller that
   reacts to what happens, so the states and the commands differ as soon as the outcomes do. This review checked it
   in scratch on the gapped training rows (42,500 x 4 per seed-world, loaded through step 2's own loaders): in every
   seed the two worlds' commands differ, in 93-99% of the rows, by up to 1.33-1.51. That is why step 1's gap rule
   gives a different `t_lo` in every seed and a different `t_up` in seed 4.
   - The code was already right. Each phase uses its own seed-world's ends, re-derived by the gap rule from that
     world's commands and checked against the table. The range hinge, the stand-in's data side and the pull targets
     come from each world's own data.
   - Only the words were wrong. This file now gives both worlds' ends in a table and defines a world-blind rival as
     one that never reads an outcome (its knobs carried over from the stalling world, its data each world's own),
     not as one that is literally the same penalty in both worlds.
   - The reviewer wrote that the no-stall ramp starts lower in every seed. In seed 0 it starts 0.0002 higher; in
     seeds 1-4 it starts 0.0017 to 0.0084 lower.
   - No computation, threshold or verdict changes.
2. **weights.py now runs from verified bytes** (`machinery.py`, `step2lib.py`, `evaluate_policy.py`;
   `prespec.json` `inputs.v6_code`). `band.py` imports version 6's `weights.py` through the normal import system
   (`import weights as Wt`), and step 2 never checked that file's SHA-256. Version 6 checked it only inside its
   approval checks, which step 2 skips, and the gate's guard does not cover it. So departure 8 ("executed from
   verified bytes") did not hold for that one file. Its bytes matched version 6's manifest (`5ca35cf2...`), so
   nothing had diverged. Now:
   - `weights.py` is pinned in `prespec.json` and executed from its verified bytes before `band.py`, whose
     `import weights` then binds to the verified module;
   - `machinery.check_imports` refuses if any module from version 6's code folder or from step 1's folder was loaded
     without its hash checked, or if `band.py`'s `weights` is not the verified module. It runs when `machinery.py`
     is imported, again just before every lock is written and again just before every values file is written. Its
     list of verified modules is recorded in every lock and every values file;
   - `exec_verified` moved unchanged from `machinery.py` to `step2lib.py`, so that it can be unit-tested.

   Step 2 calls none of `weights.py`'s functions, so no number changes. Checked in scratch, on copies outside the
   repository:
   - with the old load order, the new check refuses (`weights (...) ... without a hash check`);
   - with `weights` imported the normal way first, loading refuses (`a module named 'weights' is already loaded from
     elsewhere`);
   - the real code loads, and `band.py`'s `weights` is the verified module.
3. **The attribution's wording is fixed for both outcomes** (`step2lib.py`, `pooled.py`; `prespec.json`
   `attribution`; the Attribution section above). On seeds 0-4, A1-A3 cannot credit anything to the Bayesian part or
   to calibration as such:
   - the rivals' switches equal BCA's in all 10 seed-worlds (`meangap_k1`: all but no_stall/seed2, where
     `h = 0.048`);
   - every rival's stalling-world height is below BCA's in every seed;
   - the height-0 arms reproduce none in the no-stall world.

   So A1-A3 are points on one dose curve. Before, a meaning was given only when an A held. That left a failed A open
   to being read after the fact as credit to BCA's parts. Now, fixed before training:
   - a failed A is worded "a larger dose did better", and nothing is attributed to the Bayesian part or to
     calibration;
   - nothing is credited to calibration on these seeds in any case;
   - `obca_k2` against `obca_k1` is reported as a dose readout (rising, plateau or falling, margin 5). It says where
     BCA's height lies on the dose curve and never counts as evidence of calibration;
   - every A is also reported per seed, beside a per-seed dose table, so that seed 0's large BCA-to-conformal ratio
     is visible.

   The A1-A3 rules, their margin and their wordings when they hold are unchanged. Attribution still decides nothing:
   G1-G3, the checks, the matching and the lock are untouched.

   The reviewer proposed allowing credit to calibration when `NET(obca_k2) <= NET(obca_k1) + 5`. This review did not
   adopt that rule. A plateau shows that BCA's height is large enough, not that it is calibrated, and no arm of this
   design can tell the two apart.
4. **Tests** (`test_step2.py`): 12 new tests, 38 in all, all pass:
   - each world's own ramp ends in the arm table;
   - the attribution: its wordings both ways, its per-seed differences, the dose readout and its boundaries, the
     dose table, and the facts on the frozen arm table (the switches, the height order, the ratios quoted above);
   - the verified imports: `exec_verified`, `unverified_modules` and `local_imports` on synthetic modules; every
     sibling import of the pinned version-6 files is itself pinned, with the hashes on disk and version 6's manifest;
   - this file's two tables against `prespec.json`, and the wordings in `prespec.json` against the code.

Checked, not changed:

- **The same loose sentence elsewhere.** `analysis/restructure/ANSWER.md` says the same about the commands. It is
  read-only reference material outside step 2, so it is not changed here.
- **Departure 1.** "The same penalty in both worlds" for the random surfaces and push less was right about the penalty
  function and coefficient. It now also says "applied to that world's own rows".
- **No new numbers in results.json.** The per-seed NETs and points were already there. The new tables only put them
  beside the doses.
- **The other imports.** Every other repository module step 2 loads is checked: the gate's guard covers
  `algorithms/`, `calibration/`, `runtime/` and `experiments/`; the room harness and the stand-ins module are checked
  by their own manifest loaders; step 2's own files by its code manifest.
- **The arms.** No arm was added or removed. An arm with a non-BCA height as large as BCA's could separate
  calibration from dose, but adding it would change the approved design.

Verification by this review (scratch folder `/root/step2_review_scratch`, outside the repository):

- **The two worlds' commands.** All five seeds, both worlds, loaded through `machinery.context` and `prepared`:
  42,500 x 4 gapped training rows in each seed-world. The two worlds' commands differ in 92.8%, 95.4%, 98.7%, 96.0%
  and 99.5% of the rows (seeds 0-4), by up to 1.33, 1.42, 1.51, 1.36 and 1.45. Step 1's gap rule on each world's
  commands gives exactly the ends in the table above.
- **Unit tests.** 38 of 38 pass.
- **Smoke run** on the final code (`STEP2_SCRATCH=/root/step2_review_scratch/smoke STEP2_STEPS=40 STEP2_CHUNK=20
  STEP2_SEEDS=0 STEP2_LANES=1 STEP2_CPUS=0 sh run_step2.sh`). Exit 0 after 7 minutes 4 seconds, while another
  session's heavy job was loading the laptop:
  - both worlds trained and locked; every check passed (the gap rule, check (b), the height-0 arms);
  - `check_imports` passed at every lock and values file, which list `band`, `fresh`, `policy`, `stage2_common`,
    `step1lib` and `weights` as loaded from verified bytes;
  - both evaluations and the pooled look ran, and `results.md` shows the new attribution lines, the dose readout
    and the dose table;
  - every actor's SHA-256 (45 in the stalling world, 17 in the no-stall world), the strengths, the matches, every
    value and every pooled number equal the writer's earlier smoke run to the last digit. So the review changed no
    computation. Its verdict means nothing: 40-step actors.

  | phase (40 steps) | wall time | peak memory (VmHWM) |
  |---|---|---|
  | stalling world, training | 142 s | 1,178 MB |
  | no-stall world, training | 119 s | 1,085 MB |
  | stalling world, evaluation | 97 s | 382 MB |
  | no-stall world, evaluation | 91 s | 365 MB |
- **Freeze.** `freeze.py` was re-run before the smoke run and again after this section was written (the last run
  gives the final manifest). No official output existed at any of these runs. `prespec.json` (version 2) has SHA-256 `839ed716...cd8da04` in both. The smoke run
  recorded code manifest `c320e3d7...162fc01`; the final manifest differs from it only by `PRESPEC.md` (this
  section).
- **Repository.** `runs/shift_explore/restructure/` still does not exist. No new `__pycache__` appeared in the
  repository. Only this folder is committed.
