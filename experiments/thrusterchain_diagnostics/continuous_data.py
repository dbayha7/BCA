"""Declared continuous data generators; likelihoods are categorical action masses.

Initial physical coordinates are independent U[-.5,.5], current is the chosen
load atom (.3 or 1.1), and the wave has amplitude .2 and uniform phase. The four
regime/load context atoms have explicit probabilities. Later visitation laws
are not claimed to have tractable densities. Both tracks use the existing
float64 physics through the isolated verified adapter.
"""

from dataclasses import dataclass
import hashlib
import numpy as np

from .continuous import KNEE, Objective, dynamics, observation


@dataclass
class ContinuousDataset:
    learner_rows: dict
    evaluator: dict
    metadata: dict


def context_weights(name="overlap"):
    """Rows are regime (stall, monotone); columns are load (.3,1.1)."""
    if name not in ("overlap", "thin", "zero"):
        raise ValueError("unknown context frequency law")
    loads = {"overlap": (.5, .5), "thin": (.98, .02), "zero": (1., 0.)}[name]
    return np.tile(loads, (2, 1)) / 2


def critical_region(states, actions):
    """Declared before outcomes: high current and any above-knee command."""
    return (np.asarray(states)[..., 10] >= .8) & np.any(np.abs(actions) > KNEE, axis=-1)


def generate_continuous_data(*, track="trajectory", n_episodes=8,
                             objective=Objective(), seed=0, split="train",
                             retention="full", rare_probability=.05,
                             context_mix="overlap", fixed_states=None,
                             fixed_regimes=None, fixed_times=None):
    """Generate actual trajectories or independent fixed-state one-step queries.

    Fixed arrays describe one row's pre-outcome state/regime/time each. When
    omitted, independently drawn start-law states form the fixed query bank;
    they are not passed off as policy visitation. Uniform independent actuator
    commands in {-1,-knee,0,knee,1} give exact joint mass 1/625. There is no
    clipping atom or smooth action-density claim. Full/rare/zero runs sharing
    seed and split have identical complete logs and nested retained subsets.
    """
    if track not in ("trajectory", "fixed_state") or split not in ("train", "calibration", "holdout"):
        raise ValueError("invalid track or split")
    if retention not in ("full", "rare", "zero") or not 0 <= rare_probability <= 1:
        raise ValueError("invalid retention")
    if not isinstance(n_episodes, int) or n_episodes < 1 or not isinstance(seed, int) or seed < 0:
        raise ValueError("positive episode count and nonnegative integer seed required")
    split_id = {"train": 0, "calibration": 1, "holdout": 2}[split]
    names = ("context", "start", "action", "retention")
    streams = {name: np.random.default_rng(np.random.SeedSequence([seed, split_id, i]))
               for i, name in enumerate(names)}
    weights = context_weights(context_mix)
    contexts = streams["context"].choice(4, n_episodes, p=weights.ravel())
    states = streams["start"].uniform(-.5, .5, (n_episodes, 13))
    states[:, 10] = np.where(contexts % 2 == 0, .3, 1.1)
    phases = streams["start"].uniform(-np.pi, np.pi, n_episodes)
    states[:, 11:] = .2 * np.column_stack((np.cos(phases), np.sin(phases)))
    regimes = contexts // 2
    times = np.zeros(n_episodes, dtype=int)
    if track == "fixed_state":
        if fixed_states is not None:
            states = np.asarray(fixed_states, float)
            if states.ndim != 2 or states.shape[1] != 13 or not len(states) or not np.all(np.isfinite(states)):
                raise ValueError("fixed_states must be finite [N,13]")
            if (np.any(np.linalg.norm(states[:, :10], axis=1) > 100)
                    or np.any((states[:, 10] < 0) | (states[:, 10] > 1.5))
                    or np.any(np.linalg.norm(states[:, 11:], axis=1) > 1 + 1e-9)):
                raise ValueError("fixed states must lie in the original start set")
            if fixed_regimes is None:
                raise ValueError("fixed_regimes required with fixed_states")
            regimes = np.asarray(fixed_regimes)
            times = np.zeros(len(states), int) if fixed_times is None else np.asarray(fixed_times)
            if regimes.shape != (len(states),) or times.shape != regimes.shape:
                raise ValueError("fixed regime/time lengths differ")
            if (not np.all(np.isin(regimes, [0, 1])) or not np.all(np.isfinite(times))
                    or np.any(times != np.floor(times)) or np.any((times < 0) | (times >= objective.horizon))):
                raise ValueError("invalid fixed regime/time")
            regimes, times = regimes.astype(int), times.astype(int)
        elif fixed_regimes is not None or fixed_times is not None:
            raise ValueError("fixed_regimes/fixed_times require fixed_states")
    elif any(v is not None for v in (fixed_states, fixed_regimes, fixed_times)):
        raise ValueError("fixed queries require fixed_state track")
    choices = np.array([-1., -KNEE, 0., KNEE, 1.])
    rows = []
    for episode, (start, regime, start_time) in enumerate(zip(states, regimes, times)):
        state = start.copy()
        step_times = range(objective.horizon) if track == "trajectory" else [int(start_time)]
        for t in step_times:
            action = choices[streams["action"].integers(0, 5, 4)]
            critical = bool(critical_region(state, action))
            q = ({"full": 1., "rare": rare_probability, "zero": 0.}[retention]
                 if critical else 1.)
            # Draw and decide inclusion before computing any transition outcome.
            retained = bool(streams["retention"].random() < q)
            nxt, reward = dynamics(state, action, int(regime))
            rows.append(dict(episode_id=episode, time=t, observation=observation(state, int(regime), t, objective),
                             action=action, reward=reward, next_observation=observation(nxt, int(regime), t + 1, objective),
                             action_probability=1 / 625, retention_propensity=q, terminated=False,
                             truncated=t + 1 == objective.horizon, bootstrap_mask=float(t + 1 < objective.horizon),
                             critical=critical, retained=retained,
                             state=state.copy(), next_state=nxt.copy(), regime=int(regime)))
            state = nxt
    complete = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    keep = complete["retained"]
    learner_keys = ("episode_id", "time", "observation", "action", "reward", "next_observation",
                    "action_probability", "retention_propensity", "terminated", "truncated", "bootstrap_mask")
    learner = {key: complete[key][keep].copy() for key in learner_keys}
    learner["split"] = np.full(int(keep.sum()), split)
    learner["seed"] = np.full(int(keep.sum()), seed, dtype=np.int64)
    learner["provenance"] = np.full(int(keep.sum()), "continuous-" + track + "-v1")
    metadata = dict(schema="thrusterchain-continuous-data-v1", track=track, split=split, seed=seed,
                    random_streams={name: [seed, split_id, i] for i, name in enumerate(names)},
                    context_weights=weights.tolist() if fixed_states is None else None,
                    context_mix=context_mix if fixed_states is None else None,
                    objective=dict(horizon=objective.horizon, gamma=objective.gamma),
                    terminal_bootstrap=False, bootstrap_rule="use bootstrap_mask, including zero at horizon truncation",
                    retention=retention, rare_probability=rare_probability,
                    action_measure="categorical product mass, not a continuous density",
                    initial_law="U[-.5,.5]^10; current atom .3/1.1; wave amplitude .2, uniform phase" if fixed_states is None else None,
                    query_law="supplied fixed states" if fixed_states is not None else "declared initial law",
                    population_visitation="not analytically computed", generated_rows=len(rows),
                    retained_rows=int(keep.sum()), budget_topups=0)
    if fixed_states is not None:
        metadata["fixed_bank"] = dict(rows=len(states), sha256={
            key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            for key, value in (("states_float64", states), ("regimes_int64", regimes.astype(np.int64)),
                               ("times_int64", times.astype(np.int64)))})
    return ContinuousDataset(learner, complete, metadata)
