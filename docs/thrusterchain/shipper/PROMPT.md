Build **ThrusterChain Offline RL Studio**: an interactive web app for learning, running and inspecting offline reinforcement learning on a small physics benchmark called ThrusterChain. Everything runs in the browser. There is no backend, no login, and all simulation and training happen client-side.

The attached file `thrusterchain_spec.json` is the single source of truth for the environment. Use its numbers exactly. Do not invent, round or "simplify" any matrix, constant or rule.

## 1. The environment (must be exact)

ThrusterChain is a channel of 5 floats in a row, joined by 4 thrusters. Water pushes every float downstream.

**State.** The state is 13 float64 numbers: `p1, v1, …, p5, v5` (position and speed of each float), then `current`, `wave1`, `wave2`.

**Observation.** The agent sees 14 numbers: the 13 state values cast to float32, plus `remaining_fraction = (100 − t)/100`.

**Action.** 4 commands in [−1, 1]. Clip the requested command to [−1, 1], then cast it to float32 with `Math.fround`. That executed command is what dynamics and reward use.

**Dynamics.** `x_next = A·x + B·thrust(command)`, using the exact A (13×13) and B (13×4) from the spec. Compute in float64 (plain JS numbers), and keep the summation order fixed.

**Thrust.**
- **Stall world:** `thrust(a) = clip(a, −k, k) − 1.5·sign(a)·max(|a| − k, 0)`, with k = 0.800000011920929. Pushing past the knee gives *less* thrust.
- **No-stall world:** `thrust(a) = a`.

**Reward.** `r = −(sum of squares of the 10 float coordinates of the state before the step + 0.3·sum of squares of the executed command)`. Rewards are negative, so "higher" means "less negative".

**Episodes.**
- Exactly 100 steps, discount 0.98, terminal reward 0.
- The episode ends as terminated at step 100. There is no bootstrapping past the end.
- Return from step t is `G_t = Σ_{k=0}^{99−t} 0.98^k r_{t+k}`.

**Signs.** Current, wave1 and positive thrust all push floats in the positive (downstream) direction. Negative commands push upstream, against the water.

**Exogenous forcing.** Current is constant within an episode. The waves rotate 0.27 rad per step. Commands cannot change current or waves.

**Start laws.** There are 4 start laws, each set by its current range, wave-amplitude range and kick size:

| Start law | Use |
|---|---|
| `recorded` | the training distribution |
| `strong_current` | evaluation shift |
| `big_waves` | evaluation shift |
| `big_kicks` | evaluation shift |

A start draws the 10 float coordinates as `kick × standard normal`, then current, wave amplitude and phase, in exactly that order. Use the spec's validity rules: an invalid draw is an error, never clipped or resampled.

**Seeded randomness.** Use a seeded random number generator, such as PCG32 or xoshiro128\*\*, with a Box–Muller normal. Show the seed everywhere. The streams will not match NumPy's, and that is fine; say so in the app.

**Acceptance test (build it first and keep it in the app).** A "Simulator tests" page runs all 10 `fixture_cases` and the `over_knee_stall` replay trajectory from the spec. It shows the maximum absolute error, which must be ≤ 1e-9. If any check fails, show a red banner on every page.

**Restoring state for value checks.** A reset always starts at t = 0, so reset cannot restore a mid-episode state. To evaluate from step t, keep the float64 state plus the integer t. Step the dynamics function directly and set the time input to `(100 − t − j)/100`.

## 2. Episode Theater: watch agents play (headline feature)

The Theater is the centerpiece: a video-like player for any agent's episode.

**Visual.**
- **Channel:** a vertical channel with the water flowing *down* the screen, and animated flow lines whose speed scales with current + wave1. The five floats sit in a row across the channel, joined by chain links.
- **Positions:** each float moves down the screen by its position. A dashed line marks rest at position 0, with "downstream is down" labeled.
- **Thrusters:** each thruster sits between its two neighboring floats. An arrow shows its effective thrust, pointing down for positive and up for negative. It turns orange when stalled, past the knee.
- **Readouts:** under each thruster, two small bars show the command and the effective thrust. A header shows the step, the remaining fraction, the world, and the current and wave push.
- **Side panel:** this step's reward split into position cost and energy cost, the running discounted return, and, when available, the agent's predicted Q next to the reference value.

**Playback.** Play, pause, step forward and back, a scrub bar, speed from 0.25× to 4×, loop, and keyboard shortcuts.

**Agents you can watch:**
- **Scripted:** zero command, constant command (choose a value), pulsed command, the random data recorder, and a "clipped LQ controller". The last is a finite-horizon LQ on the exact A and B, with commands clipped to ±k. Label it an approximation, not a certified optimum.
- **In-app trained:** any agent trained here, at any saved checkpoint.
- **Imported:** a trained actor loaded from a JSON file, using the schema below. This lets me watch policies I trained elsewhere.

**Compare mode.**
- Up to 4 agents play the *same* start seed, start law and world, side by side and synchronized.
- A shared timeline marks where their commands diverge.
- A summary row shows each agent's return.

**Overlays you can toggle:** stall flags; a trail of past positions; entry into a user-defined "withheld action region" (see section 3), drawn as a marker on the timeline; per-step Q error.

**Export.**
- Download the current view as a **WebM video**, using `canvas.captureStream()` and `MediaRecorder`, at a chosen speed and resolution.
- Also export an **animated GIF** and the raw trajectory as JSON or CSV.
- A "copy link" button encodes agent, seed, start law and world in the URL, so the same episode replays exactly.

**Episode gallery.** Saved episodes appear as thumbnail filmstrips, stored in IndexedDB. Filter them by agent, start law, world and return. Click one to play it.

**Imported actor schema** (validate it and show clear errors):

```json
{"name": "string", "obs_mean": [14 numbers], "obs_std": [14 numbers], "obs_eps": 0.001,
 "layers": [{"W": [[in x out]], "b": [out], "activation": "relu|tanh|linear"}],
 "output_activation": "tanh", "notes": "string"}
```

The actor computes `normalized = (obs − obs_mean)/(obs_std + obs_eps)`, runs the dense layers, and outputs 4 commands in [−1, 1].

## 3. Dataset Studio

**Generate offline datasets with the recorder:**
- **Recorder law:** with probability ε, a0 ~ Uniform(0.5, 1]; otherwise a0 ~ Uniform[−1, 0.5]. a1–a3 ~ Uniform[−1, 1]. The recorder does not look at the state.
- **ε presets:** broad 0.25 (exactly uniform), rare 0.01, absent 0.
- **Other behavior policies, optionally:** the clipped LQ controller plus Gaussian noise of a chosen size, and mixtures of the two. These give "expert", "medium" and "mixed" style datasets.

**Size and splits.** Set the number of episodes and the seed. Splits are whole episodes: about 60% train, 20% tuning-validation, 10% selection and 10% final-calibration. Show the split counts.

**Stored per step:** float32 observation, executed command, requested command, reward, next observation, terminated, truncated, episode id, t and split.

**Coverage views:**
- histograms of each command coordinate;
- a heatmap of load `g = current + wave1` against a0;
- state-coordinate histograms per time bin.

**Support editing.** Each edit makes a new dataset version and never changes the original:
- **Global cutoff:** remove rows with a0 > c.
- **Conditional hole:** remove rows with `g ≥ threshold` and `|a0 + k| ≤ h`.
- **Matched random-removal control:** inside each (episode, ten-step time bin, load side) cell, remove exactly as many random rows as the hole removed.

For each edit, report rows removed, episodes affected and per-cell counts. Always keep whole tuples (state, action, reward, next state); never rewrite an action.

**Export and import:** datasets as JSON and CSV, with a SHA-256 hash shown.

## 4. Algorithm Lab: train offline agents in the browser

**Execution.** Train in Web Workers with TensorFlow.js: WebGPU if available, then WebGL, then CPU. Training never freezes the page. Show updates per second, an ETA, and pause, resume and cancel controls.

**Algorithms.** Implement each faithfully, with defaults shown and editable:
- **BC:** behavior cloning, mean squared error on the commands.
- **TD3+BC:**
  - networks: two critics, one actor, 256-256 ReLU, tanh output;
  - actor update: maximizes λ·Q1 minus mean squared error to the dataset command, with λ = 2.5/(mean|Q1| + 1e-7) computed without gradient;
  - critic target: twin-min with target policy smoothing (noise 0.2, clip 0.5);
  - schedule: one actor update per 2 critic updates; target networks updated by Polyak averaging at 0.005;
  - optimizer: Adam at 3e-4, γ 0.98, batch 256;
  - inputs: normalized by `(obs − mean)/(std + 1e-3)`, with mean and std from the **training split only**.
- **IQL:** expectile value function at 0.7, advantage-weighted actor with temperature 3.
- **CQL:** conservative penalty with an adjustable weight.
- **ReBRAC:** behavior-regularized actor and critic penalties.
- **Uncertainty-penalized actor (teaching version):** the actor maximizes `Q1 − κ·U`, where U is the spread of a small critic ensemble. Label it clearly as a simplified illustration of uncertainty-penalized actors, not any specific published method.

**Run setup.** Pick a dataset version, algorithm, seed, update budget and checkpoint schedule, for example 0, 10%, 50% and 100%.

**Live charts:** critic loss, actor loss, mean Q, and the evaluation return on fixed held-out start seeds.

**Run queue.** Queue several runs, for example 3 seeds × 2 algorithms, and save everything to IndexedDB.

**Export.** Export any actor in the imported-actor JSON schema, so it round-trips with the Theater.

## 5. Q-Value Inspector: is the critic right?

**Pick a state** from a dataset, from a Theater frame, or from a fresh state bank. A state bank is reached by running the random recorder to t ∈ {0, 25, 50, 75}, so it does not depend on any trained agent.

**Candidate commands:** the actor's own command; commands sweeping a0 across [−1, 1] with a1–a3 held fixed; and user-added commands.

**Reference value Q^π.**
- Restore the exact state and t, as described in section 1, and run the candidate command once.
- Then follow the **same frozen actor** until step 100, summing discounted rewards with no bootstrap.
- Say clearly that this is the value of *that policy*, not the optimal value Q\*.
- The simulator has no noise, so one rollout is exact.

**Show:**
- Q1, Q2, their minimum, and any penalized score, all against the reference.
- Signed error = prediction − reference, where positive means overestimation.
- Over, near-zero and under labels, with a visible tolerance.
- For a penalized score, a 3×3 table of how each probe's label moved from raw to adjusted.
- Regret within the candidate set: the best reference value minus the reference value of the chosen command.
- A line chart of predicted and reference value across the a0 sweep, with the knee at ±k marked and any withheld region shaded.

## 6. Shift & Support Lab

**Evaluate any agent:**
- **across start laws:** recorded, strong_current, big_waves, big_kicks;
- **across worlds:** stall, no-stall.

Use fixed reserved start seeds, the same for every agent.

**Report:**
- the return distribution as a box plot plus individual points;
- how often the agent's commands enter a withheld region and when it first enters;
- how far its states fall from the training data.

**Keep separate:**
- state shift (new start laws);
- missing actions (holes and cutoffs).

**Mark "inconclusive" when:**
- an agent never visits the withheld region, because its effect was then never tested;
- critic values barely differ across the candidate commands.

## 7. Experiment Board

**Compare runs side by side.** Each trained seed is one independent unit. Show every seed's point, not only means. Pair comparisons on shared start seeds.

**No cherry-picking.** Never drop runs: failed and interrupted runs stay visible with their status. Label results "from this app's runs" and never present them as published research numbers.

**Export:** a report as HTML and CSV.

## 8. Learn: step-by-step interactive lessons

Each lesson is a guided sequence of short steps with a "Next" button. Each step has one idea, a live widget and worked arithmetic with real numbers from the simulator. It is not a long page of sections.

**Lessons:**
1. One step of ThrusterChain, by hand: thrust, push, next state and reward.
2. Why stalling past the knee wastes energy.
3. Offline data and coverage: what the recorder sees and what it never tries.
4. The Bellman target and why critics overestimate commands missing from the data.
5. TD3+BC in numbers: one actor update with λ, the Q term and the BC term.
6. Policy value Q^π versus optimal value Q\*, and how the Q-Value Inspector measures error.
7. Distribution shift versus missing actions, using the Shift & Support Lab.

**Links.** Every lesson links to the matching tool with that state preloaded.

## 9. Design and quality

**Layout.**
- **Desktop:** a left sidebar with grouped parts (Start, Environment, Episode Theater, Data, Algorithms, Inspect, Shift, Results, Learn). Each part has clickable section links.
- **Phone:** a 375 px layout with a bottom bar.
- **Themes:** light and dark.

**Writing.** Use plain language and American spelling. Every technical term (Q, TD, BC, OOD, discount, bootstrap, and so on) gets a hover or tap definition.

**Accessibility.** Keyboard access everywhere, ARIA labels, and color-blind-safe palettes. Stall orange is paired with an icon, not color alone.

**Offline use.** All saved state lives in IndexedDB with export and import. The app works with no network after the first load; make it an installable PWA.

**Speed.** Keep the UI responsive. Run heavy simulation in workers and batch many rollouts at once for reference values.

## 10. Build order

Build in this order and keep each stage working:
1. The exact simulator, the Simulator tests page, and the Environment view with manual sliders.
2. Episode Theater with scripted agents, compare mode and video export.
3. Dataset Studio.
4. BC and TD3+BC training, then the Theater showing trained agents and the actor import and export.
5. The Q-Value Inspector.
6. IQL, CQL, ReBRAC and the uncertainty-penalized actor.
7. The Shift & Support Lab, the Experiment Board and the lessons.

**Out of scope:** user accounts, server-side training, and any claim of research results that this app did not compute itself.
