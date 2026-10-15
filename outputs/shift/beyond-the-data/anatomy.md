# Where the strong-current harm comes from

**EXPLORATORY.** Computed outside the gate, on the gate's saved critics and actors (gate seeds 0-4, alpha 5), with
exact truth by simulation. No training, no BCA code, no BCA band or dose. Nothing here changes a pre-registered rule.
The "harm-placed doses" below are check 3's own exact-harm doses, recomputed with check 3's own code, not BCA doses.
Numbers are means over the five gate seeds, with the smallest and largest seed in brackets.

Words used: the **gapped log** is the training data after the densest-15% removal; the **full log** keeps every row.
A **critic** scores commands; an **actor** is the learned controller. The **full-throttle region** is the set of states
where the water's push (current plus wave) is above 0.565; there the recording controller sometimes gave the
full-throttle command that the removal deleted. **Throttle 1** is thruster 1's throttle (it holds against the current).
"Check 1, 2, 3" are the harm checks of the stop report.

## The answer

The harm is caused by one wrong belief, held at states that are in the data: **at training states in the
full-throttle region, the gapped critic sees no cost in pushing throttle 1 past 80%.** Up to 80% it is exactly right.
The actor is trained only at training states, so it learns this there: it pushes past 80% at 42% of those training
states (10% with the full-log critic). As the current grows beyond the recorded range, the actor's commands keep
rising, so the same mistake is made at most steps in strong current. The critic's own errors at strong-current states
do not matter. Every critic, the full-log one included, is wrong there, and the actor never reads them.

To make the harm disappear, one quantity has to be corrected: **the critic's value of commands past 80% relative to
80%, at training states in the full-throttle region, at the actor's own commands.** It must go from about +0.05 to
the true −0.46 (+0.23 against −0.77 where the push is above 0.8). The evidence is that the gapped critic's actor,
with only its past-80% commands cut back to 80% at evaluation, scores **+101.8 points in strong current** (against
−14.3 uncapped, and 84.6 for the full-log critic's actor). Everything else that critic taught is good.

Check 3's caution cannot do this, for two reasons. Its signal and its pull both act at the recorded command, where
the gapped critic is right. Its pull target is the recording controller's normal-driving command (throttle 1 about
0.5), which says "drive like the recorder", not "stop at 80%".

## The chain, step by step

### 1. What the removal leaves (data)

In the gapped log, 14% of training rows lie in the full-throttle region, and 3% have a push above 0.8. In those
region rows the recorded throttle 1 averages **0.51** [0.50, 0.52]. It is at or above 0.75 on only 2.9% of them and
past 80% on 0.7%. The data say almost nothing about throttles from 0.75 upward at exactly the states where
those throttles matter.

### 2. The critic's belief at training states: right up to 80%, wrong past it

The critic's value of throttle 1 at v, minus its value at 80%, averaged over training states in the full-throttle
region. The other thrusters come from the stall-aware controller. Truth is the exact value of the controller these
critics score (`curves.json`).

| throttle 1 | 0.5 | 0.6 | 0.7 | 0.75 | **0.8** | 0.85 | 0.9 | 0.95 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| truth | −0.23 | −0.14 | −0.06 | −0.03 | 0 | −0.10 | −0.21 | −0.33 | **−0.46** |
| gapped-log critic | −0.24 | −0.14 | −0.06 | −0.03 | 0 | +0.02 | +0.04 | +0.05 | **+0.06** |
| full-log critic | −0.28 | −0.12 | −0.00 | +0.02 | 0 | −0.06 | −0.17 | −0.30 | −0.46 |
| random-removal critic | −0.26 | −0.11 | +0.00 | +0.02 | 0 | −0.06 | −0.16 | −0.28 | −0.44 |

- **Error in the preference for full throttle over 80%** (critic minus truth), on these states: gapped **+0.51**
  [0.48, 0.53]. Full log −0.02, random removal −0.00, full log with another initialisation −0.01. Both no-stall-world
  critics +0.01 to +0.02 (the no-stall truth there is +0.06). Above a push of 0.8: gapped +1.00 [0.87, 1.12]
  against a truth of −0.77; full log +0.01; random removal +0.05.
- **The command each critic rates best is past 80%** on 55% of these states for the gapped critic, 3% for the
  full-log critic, and 0% in truth.
- **Inside the recorded range the gapped critic is no worse.** For the recorded command against the stall-aware
  command at training rows, the mean error is −0.01 (gapped), −0.03 (full) and −0.03 (random). The mean absolute
  error is 0.07 for the gapped critic, against 0.12 and 0.12.
- At training states outside the region, all critics are close to the truth (gapped −0.01, full −0.09).
- The plain TD3+BC critic co-trained on the gapped log (the critic of check 3), scored against its own actor, has the
  same blind spot and a larger one: +1.05 [0.75, 1.34] in the region. Its full-log counterpart shows −0.01.

### 3. The actor learns it at training states

Every actor update reads the critic only at (training state, the actor's own command) (`learners.actor_step`). The
gapped and full-log actors share their start, rows and batches. So they can differ only through the critics' values
there.

| at training states | recorded | cloned actor | stall-aware | full-log critic's actor | random-removal critic's actor | gapped critic's actor |
|---|---|---|---|---|---|---|
| past 80%, region rows | 0.7% | 0% | 0% | 10% [6, 14] | 9% | **42% [38, 43]** |
| past 80%, push above 0.8 | 0% | 1% | 0% | 17% | 17% | **74% [67, 77]** |
| mean throttle 1, push above 0.8 | 0.59 | 0.61 | 0.76 | 0.74 | 0.73 | **0.87** |

These are the critics' slopes along throttle 1 (value per unit of throttle) at the commands the actors emit at
training states:

- at the gapped critic's actor's past-80% commands (9% of training rows): truth **−3.9**, gapped critic **+1.1** (wrong
  sign), full-log −3.6, random removal −3.3;
- at the stall-aware controller's commands between 0.75 and 0.8 (26% of rows): truth −0.68, gapped **+0.63**, full-log
  −0.53, random removal −0.51;
- below 0.75, every critic matches the truth to within about 0.1 (for example 0.29 true and 0.33 gapped at the gapped
  actor's commands between 0.6 and 0.75).

The full-log critic bends just before 80% (its curve peaks near 0.75). So its actor settles a little below the cap at
the high-push training states, with a margin. The gapped critic keeps rising through 80%, so its actor settles past
it.

(A minor difference: the actor's weight on the critic, 5 / mean |Q|, is 0.316 with the gapped critic and 0.285 with
the full-log one.)

### 4. Leaving the recorded range: the actor extrapolates what it learned at the data's edge

Throttle 1 as the current rises from 0.1 to 1.2 (recorded maximum 0.7). The floats are at rest, the wave amplitude
is 0.25, and each value averages 8 wave phases (`actor_sweep_seed*.json`).

| actor | current 0.7 | 0.9 | 1.1 | mean passes 80% at current | slope inside / outside the range |
|---|---|---|---|---|---|
| stall-aware controller | 0.63 | 0.74 | 0.80 | never | 0.90 / 0.35 |
| cloned actor | 0.51 | 0.63 | 0.71 | 1.13 (2 seeds; 3 never) | 0.71 / 0.45 |
| full-log critic's actor | 0.59 | 0.71 | 0.78 | 1.05 (3 seeds; 2 never) | 0.85 / 0.42 |
| random-removal critic's actor | 0.60 | 0.73 | 0.80 | 0.95 (3 seeds; 2 never) | 0.88 / 0.44 |
| **gapped critic's actor** | 0.63 | **0.83** | **0.94** | **0.87 [0.83, 0.92], all 5 seeds** | 0.91 / **0.68** |
| plain TD3+BC, gapped log | 0.59 | 0.74 | 0.89 | 0.97, all 5 seeds | 0.82 / 0.72 |
| rising-thrust controller (assumes no stall) | 0.64 | 0.81 | 0.93 | 0.88 | 0.91 / 0.67 |

With 64 float configurations drawn from training rows, the share of commands past 80% at currents 0.5, 0.7, 0.9 and
1.1 is:

- gapped critic's actor: **0.19 / 0.39** / 0.66 / 0.87;
- full-log critic's actor: 0.05 / 0.16 / 0.39 / 0.66;
- cloned actor: 0.00 / 0.00 / 0.03 / 0.17;
- the no-stall world's actors (trained where pushing past 80% helps): 0.21 / 0.41 / 0.66 / 0.86.

The gapped critic's actor behaves like an actor trained in the world without a stall. The gapped log cannot tell the
two worlds apart. The difference starts inside the recorded range (currents 0.5 to 0.7) and grows outside it. That
is because strong current calls for high throttle on most steps: the recording controller would have given the
removed full-throttle command at 57% of strong-current states, against 17% in the recorded conditions.

### 5. The critics' errors in strong current do not drive the harm

All critics are wrong at states they never saw. Here is the error in the preference for full throttle over 80% (truth
in brackets):

| states | gapped | full log | random removal | other initialisation |
|---|---|---|---|---|
| training states in the region (−0.46) | +0.51 | −0.02 | −0.00 | −0.01 |
| region floats and waves with the current set to 0.7 / 0.9 / 1.1 (−0.69 / −1.31 / −2.29) | +0.87 / +1.76 / +3.04 | −0.04 / +0.38 / +1.31 | +0.00 / +0.43 / +1.39 | +0.01 / +0.45 / +1.49 |
| states the stall-aware controller visits in strong current (−1.24) | +1.60 | +0.47 | +0.53 | +0.58 |
| states the gapped critic's actor visits in strong current (−1.90) | +2.51 | +1.14 | +1.18 | +1.18 |

The full-log critic underrates the cost by 0.4 to 1.3 in strong current, yet its actor scores 84.6. In strong current
its preference keeps the right sign (−0.77 against a truth of −1.24), and the command it rates best is past 80% on
11% of states, against 80% for the gapped critic. The decisive point is that the actor never reads the critic at
those states. So the question "is the critic right in strong current?" is the wrong one. The full-log critic is
not right there, and that does no harm.

### 6. Where the return is lost

Points of each check 2 actor, as trained and with its commands capped at 80% at evaluation
(`gate_readout.json`, read from the gate's saved evaluations):

| actor | strong current | capped | recorded | capped | big waves | capped | big kicks | capped |
|---|---|---|---|---|---|---|---|---|
| cloned actor | 54.4 | 61.3 | 56.3 | 56.4 | 43.6 | 44.3 | 2.0 | 13.4 |
| full-log critic's actor | 84.6 | 98.5 | 91.3 | 92.2 | 86.6 | 88.1 | 64.3 | 88.6 |
| **gapped critic's actor** | **−14.3** | **+101.8** | 85.5 | 93.9 | 70.0 | 92.8 | 52.5 | 93.4 |
| plain TD3+BC, gapped log | −16.8 | +100.4 | 86.1 | 94.5 | 70.9 | 93.9 | 48.7 | 93.3 |
| check 3, no caution | −10.1 | +101.1 | 87.6 | 94.5 | 72.7 | 93.1 | 53.7 | 93.4 |
| check 3, even ×3 | +7.1 | +92.2 | 82.4 | 84.7 | 67.3 | 78.1 | 41.0 | 70.5 |

Remove its past-80% choices and the gapped critic's actor is the best policy in every situation (93 to 102 points).
The policy improvement it learned from the gapped log is sound. The harm is entirely in the past-80% choices, which
matches the stop report's attribution (a loss of 2.03 per past-80% step and −0.01 elsewhere). Strong uniform
caution (even ×3) does lower the share past 80%, but it also costs about 10 points of the improvement in every
situation (capped 92.2 against 101.8).

## Why extra pull toward the recorded command on training rows (check 3) misses

Check 3's harm-placed doses were recomputed with its own oracle. At the first refresh they match the saved dose
summaries exactly in all 5 seeds (mean, spread and maximum).

- **At the start the doses ignore the region.** The cloned actor stays inside the recorded range, so the exact
  over-preference at that point measures errors inside the range. The region holds 15% of the extra dose and 14% of
  the rows (mean dose 1.169 in the region, 1.163 elsewhere). Only 14% of the over-preference sits on rows where the
  actor is past 80%.
- **Later the doses do find the rows, but the dose form flattens them.** At the end of training, 64% of the
  over-preference sits on rows where the actor is past 80%. The top 10% of rows by dose are 37% region rows. Yet the
  mean dose is 1.22 in the region against 1.16 elsewhere (1.27 at a push above 0.8). The factor is capped below 1.5,
  and 44% of rows have no over-preference.
- **The pull points the wrong way for this harm.** At the top-dose rows the recorded throttle 1 is 0.42, against the
  actor's 0.62. In the region it is 0.51 against 0.70, and only 3% of recorded commands there are at 0.75 or more. The
  recorded commands do not show the saturation at 80%. The pull says "lower and noisier, like the recorder" in every
  thruster, not "stop at 80%".
- **What each arm changes, against no extra caution**, measured as the root-mean-square change of the command. For
  harm-placed caution it is 0.053 at training rows and 0.093 at strong-current states: the change does carry over to
  the shifted states. But throttle 1 moves by only −0.010 in the region and −0.006 in strong current. The share past
  80% in strong current is 0.68 against 0.68. Strength-matched even caution gives −0.014 / −0.010, cap 2 gives
  −0.019 / −0.013, and even ×3 gives −0.072 / −0.048 (share 0.60). For comparison, the cloned actor differs from the
  no-caution actor by −0.18 / −0.18 (share 0.24).

So the caution reaches the strong-current states, but it changes the wrong coordinates.

## The swap test: what correcting only the commands past 80% at training states would remove (no training)

- The actor reads the critic only at training states (step 3). So the full-log critic's values in strong current
  play no part in the 98.9-point difference between the two check 2 actors.
- At training states the two critics agree inside the recorded range. Their systematic errors are −0.01 and −0.03.
  Their disagreement there (mean absolute 0.12) is the same size as each critic's own error and as the random-removal
  critic's. Past 80% they disagree systematically: by 0.53 [0.48, 0.57] in the region and 0.99 at a push above 0.8.
- The random-removal critic loses as many rows but keeps the past-80% information. It differs from the full-log
  critic inside the range by about as much, and its actor loses only 3.1 points.
- **Estimate:** correcting only the past-80% values at the region's training states would make the gapped critic
  equivalent, in everything the actor reads, to the full-log or random-removal critic. That would remove about 96 of
  the 99 points (99 minus the 3 points that the in-range differences explain). This is an inference, not a run.
- **Upper reference:** cutting the gapped critic's actor's past-80% commands back to 80% at evaluation removes all 99
  points and adds 17 more (+101.8 against 84.6). A corrected critic would still leave its actor past 80% on some steps,
  as it does for the full-log critic's actor (41% of strong-current steps, 14 points lost there).

## What would have to be corrected, and where it must be measured

- **Quantity:** Q(s, throttle 1 past 80%) − Q(s, throttle 1 at 80%), the critic's slope across the 80% point.
- **Where:** at training states in the full-throttle region (14% of training rows; most of all the 3% with a push
  above 0.8), at the commands the actor itself proposes there.
- **Size:** from +0.05 to −0.46 on average (+0.23 to −0.77 above a push of 0.8). In slope, from about +0.6 to +1.1
  per unit of throttle down to −0.7 to −3.9.
- **Not where it must be measured:**
  - at the recorded command (there the gapped critic is right: absolute error 0.07);
  - at strong-current states (every critic is wrong there, and it does no harm).

## What this implies for BCA (interpretation only; nothing was run)

1. **To detect this harm, BCA must read its band at the actor's command, not only at the recorded one.** In the
   TD3+BC version, the residual-scale network is applied at the recorded row's state and command
   (`algorithms/td3_bc_bca.py`, line 284). At those pairs the gapped critic is more accurate than the full-log one.
   The network does take the command as an input, so the open detection question is whether its width rises at
   (training state, a command past 80%) in the region. That can be asked only with the user's approval.
2. **To repair it, the dose must act on the critic's value of unsupported commands at those states, not on the pull
   toward the recorded command.** BCA's CQL version (conservative Q-learning, `algorithms/cql_bca.py`) already
   multiplies the per-row conservative gap. That gap pushes the critic down at sampled and policy commands relative
   to the recorded one, at that training state. That is the lever this anatomy points to. A limit that keeps the
   actor's command inside the recorded range at flagged states is the other one.
3. **The result the user is after has this shape.** The capped gapped critic's actor scores 101.8 in strong current
   and 93 to 94 elsewhere. A signal that finds the unsupported commands in the region, plus a correction confined to
   them, would keep all of the policy improvement and remove the harm. Uniform caution cannot do both: even ×3 still
   scores only +7.1 in strong current and loses 5 to 13 points elsewhere.

## Files

Scripts (in `scratchpad/ood/anatomy/`):

- `common.py`: loaders, which read through the gate's own verified loaders;
- `critic_beliefs.py`: part 1, critic against truth along the throttle sweep;
- `actor_sweep.py`: part 2, actors against current;
- `caution_anatomy.py`: part 3, doses and the actor change per arm;
- `mechanism.py`: part 4, what the actor reads at training states;
- `gate_readout.py`: capped points, read from the saved gate evaluations;
- `summarise.py` and `curves.py`: pooling;
- `run_queue.sh`: the runner, one CPU per process.

Outputs (in `runs/shift_explore/anatomy/`):

- `critic_beliefs_seed{0..4}.json`, `actor_sweep_seed{0..4}.json`, `caution_anatomy_seed{0..4}.json` and
  `mechanism_seed{0..4}.json`: per seed;
- `gate_readout.json`, `curves.json` and `summary.json`: pooled.

Every file carries the EXPLORATORY label. Nothing under `experiments/shift` or `runs/shift` was written.

Checks:

- The recomputed first-refresh doses equal check 3's saved dose summaries in all 5 seeds.
- Check 3's no-caution actor gives the same sweep as check 2's actor trained on the co-trained gapped critic, as it
  should.
- In seed 0 only, the co-trained critics were also scored on every probe set (400 states). Seeds 1 to 4 used 300
  states on 6 sets. The co-trained critics' figures quoted above use those 6 sets.
