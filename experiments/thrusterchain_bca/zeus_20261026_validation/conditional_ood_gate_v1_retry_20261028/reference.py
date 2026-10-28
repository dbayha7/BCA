"""Bounded, inference-only Q^pi references with explicit state and time.

The supervisor authenticates all inputs and the frozen inference export. This
module imports no learner and performs no work on import. Preparation tests use
synthetic cores only; ``runtime_parity`` checks the real adapter during the gate.
"""
from pathlib import Path
import hashlib
import time

import numpy as np

GAMMA = .98
HORIZON = 100
KNEE = .800000011920929
CANONICAL_SHA256 = "f22f2e214ad9231bee4ce60ef55e64409534c2654c4e184406553ba8ed67dde1"
MATRIX_SHA256 = "289afb581b4fdbc97fd68566c0b42ff6244b20686e9e49ba177e7e6bd57454a1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def new_ledger(max_transitions=28104, max_actor_rows=30000):
    return {"max_transition_steps": int(max_transitions),
            "max_actor_rows": int(max_actor_rows),
            "transition_steps_attempted": 0, "transition_steps_completed": 0,
            "actor_calls_attempted": 0, "actor_calls_completed": 0,
            "actor_rows_attempted": 0, "actor_rows_completed": 0,
            "resets_attempted": 0, "resets_completed": 0,
            "optimizer_updates": 0, "calibration_updates": 0}


def authenticate_core(core):
    physics = {"stall": True, "stall_slope": 1.5, "knee": KNEE,
               "gain": .3, "throttle_cost": .3, "wave_turn": .27,
               "gamma": GAMMA, "length": HORIZON}
    expected = {"physics": physics, "state_dim": 13, "action_dim": 4,
                "observation_dim": 14, "observation_mode": "time_aware",
                "task_semantics": "finite_horizon", "objective_gamma": GAMMA,
                "horizon": HORIZON, "terminal_reward": 0.,
                "terminal_bootstrap": False, "matrix_sha256": MATRIX_SHA256}
    require(core.canonical_config() == expected, "Canonical task contract differs")
    require(core.dynamics is core.canonical.dynamics, "Canonical dynamics binding differs")
    # load_canonical builds a ModuleType from authenticated AST definitions.
    # That dynamic module intentionally has no __file__. Bind the original
    # function's compilation path to the authenticated adapter's source tree.
    path = Path(core.__file__).resolve().parent / "canonical" / "thrusterchain.py"
    require(Path(core.dynamics.__code__.co_filename).resolve() == path.resolve(),
            "Canonical dynamics compilation path differs")
    require(core.dynamics.__globals__ is core.canonical.__dict__,
            "Canonical dynamics globals binding differs")
    require(hashlib.sha256(path.read_bytes()).hexdigest() == CANONICAL_SHA256,
            "Canonical dynamics source hash differs")
    require(core.Physics(stall=True, stall_slope=1.5).as_dict() == physics,
            "Constructed physics differs")
    return {"config": expected, "canonical_source": str(path),
            "source_locator": "authenticated adapter path plus original function compilation path and globals",
            "canonical_sha256": CANONICAL_SHA256}


def _check_time(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Design-gate deadline reached before dispatch")


def _notify(progress, ledger):
    if progress is not None:
        progress(dict(ledger))


def _array(value, shape, dtype, label):
    value = np.asarray(value)
    require(value.shape == shape and value.dtype == np.dtype(dtype)
            and bool(np.all(np.isfinite(value))), "Invalid " + label)
    return value


def observation(states, times):
    states = np.asarray(states)
    require(states.dtype == np.float64 and states.shape[-1] == 13
            and np.all(np.isfinite(states)), "Exact float64 physical state required")
    times = np.asarray(times)
    require(times.shape == states.shape[:-1] and times.dtype.kind in "iu"
            and np.all((times >= 0) & (times <= HORIZON)), "Invalid observation times")
    return np.concatenate((states, ((HORIZON-times)/HORIZON)[..., None]), axis=-1).astype(np.float32)


def deployed_actions(value, rows):
    value = np.asarray(value)
    require(value.shape == (rows, 4) and value.dtype.kind in "fiu"
            and np.all(np.isfinite(value)) and np.all(np.abs(value) <= 1),
            "Invalid actor/candidate action; no hidden clipping")
    return value.astype(np.float32)


def _charge(ledger, stage, rows, deadline):
    _check_time(deadline)
    require(type(rows) is int and rows > 0, "Invalid transition charge")
    require(ledger["transition_steps_attempted"] + rows <= ledger["max_transition_steps"],
            "Prospective transition cap exceeded")
    ledger["transition_steps_attempted"] += rows
    key = stage + "_steps_attempted"
    ledger[key] = ledger.get(key, 0) + rows


def _complete(ledger, stage, rows):
    ledger["transition_steps_completed"] += rows
    key = stage + "_steps_completed"
    ledger[key] = ledger.get(key, 0) + rows


def actor_actions(policy, observations, ledger, *, deadline=None, progress=None):
    observations = np.asarray(observations)
    _array(observations, (len(observations), 14), np.float32, "actor observations")
    _check_time(deadline)
    rows = len(observations)
    require(ledger["actor_rows_attempted"] + rows <= ledger["max_actor_rows"],
            "Prospective actor-row cap exceeded")
    policy.assert_unchanged()
    ledger["actor_calls_attempted"] += 1
    ledger["actor_rows_attempted"] += rows
    _notify(progress, ledger)
    actions = deployed_actions(policy.act(observations), rows)
    ledger["actor_calls_completed"] += 1
    ledger["actor_rows_completed"] += rows
    policy.assert_unchanged()
    return actions


def candidate_banks(policy, observations, ledger, *, deadline=None, progress=None):
    """Slots actor/left/lower-inside/center/upper-inside/right; other commands fixed.

    Identical slots, if present, are retained and charged. No return, critic or
    BCA score is consulted. The center is the actual canonical float32 knee.
    """
    actor = actor_actions(policy, observations, ledger, deadline=deadline, progress=progress)
    center = np.float32(-KNEE)
    lower = np.nextafter(np.float32(-KNEE-.05), center)
    upper = np.nextafter(np.float32(-KNEE+.05), center)
    require(abs(float(lower)+KNEE) <= .05 and abs(float(upper)+KNEE) <= .05,
            "Executed interior band edges differ")
    banks = np.repeat(actor[:, None, :], 6, axis=1)
    banks[:, 1:, 0] = np.asarray([-.9, lower, center, upper, -.7], np.float32)
    return banks


def _env(core):
    return core.ThrusterChainEnv(stall=True, stall_slope=1.5, start="recorded",
                                observation_mode="time_aware", task_semantics="finite_horizon")


def _check_env(env, obs, terminated, truncated, info, t):
    _array(env.state, (13,), np.float64, "environment state")
    require(env.t == t and np.array_equal(obs, observation(env.state, np.asarray(t))),
            "Environment observation/time mismatch")
    require(bool(terminated) == (t == HORIZON) and not bool(truncated),
            "Finite-horizon terminal/truncation mismatch")
    require(info["remaining_steps"] == HORIZON-t
            and info["bootstrap_mask"] == (0. if t == HORIZON else 1.)
            and info["continuation_discount"] == (0. if t == HORIZON else GAMMA),
            "Finite-horizon bootstrap/discount mismatch")


def _reset(env, seed, ledger, deadline, progress, state=None):
    _check_time(deadline)
    require(ledger["resets_attempted"] < 81, "Reset cap exceeded")
    ledger["resets_attempted"] += 1
    _notify(progress, ledger)
    obs, info = env.reset(seed=int(seed), options=None if state is None else {"state": state.copy()})
    ledger["resets_completed"] += 1
    _check_env(env, obs, False, False, info, 0)
    return obs


def _step(env, action, ledger, stage, deadline, progress):
    action = deployed_actions(np.asarray(action).reshape(1, 4), 1)[0]
    next_t = env.t + 1
    _charge(ledger, stage, 1, deadline)
    _notify(progress, ledger)
    obs, reward, ended, truncated, info = env.step(action)
    _complete(ledger, stage, 1)
    _check_env(env, obs, ended, truncated, info, next_t)
    require(np.isfinite(reward), "Nonfinite environment reward")
    return obs, float(reward)


def _dynamics(core, states, actions, ledger, stage, deadline, progress):
    states = np.asarray(states)
    scalar = states.ndim == 1
    rows = 1 if scalar else len(states)
    _array(states, (13,) if scalar else (rows, 13), np.float64, "direct state")
    actions = deployed_actions(np.asarray(actions).reshape(rows, 4), rows).astype(np.float64)
    _charge(ledger, stage, rows, deadline)
    _notify(progress, ledger)
    nxt, reward = core.dynamics(states, actions[0] if scalar else actions,
                                core.Physics(stall=True, stall_slope=1.5))
    _complete(ledger, stage, rows)
    _array(nxt, states.shape, np.float64, "direct successor")
    _array(reward, () if scalar else (rows,), np.float64, "direct reward")
    require(np.all(np.sqrt(np.sum(nxt[..., :10]**2, axis=-1)) <= 300),
            "Direct state left the physical bound")
    return nxt, reward


def reconstruct_states(core, episode_seeds, actions, observations, times, ledger,
                       *, deadline=None, progress=None):
    """Replay only declared recorded prefixes, checking every saved observation.

    Float32 observations are never reinterpreted as exact physical states. This
    reads no files; the caller supplies authorized episode slices and provenance.
    """
    times = np.asarray(times)
    seeds = np.asarray(episode_seeds)
    n = len(times)
    require(times.shape == (n,) and times.dtype.kind in "iu" and np.all((times >= 0) & (times < 100)),
            "Invalid query times")
    require(seeds.shape == (n,) and seeds.dtype.kind in "iu" and len(np.unique(seeds)) == n,
            "Distinct recorded reset seeds required")
    _array(actions, (n, 100, 4), np.float32, "recorded actions")
    _array(observations, (n, 101, 14), np.float32, "recorded observations")
    require(1 <= n <= 64, "Query cap exceeded")
    require(ledger["transition_steps_attempted"] + int(times.sum()) <= ledger["max_transition_steps"],
            "Recorded prefixes cannot fit remaining cap")
    states = []
    for i, t in enumerate(times):
        env = _env(core)
        try:
            obs = _reset(env, seeds[i], ledger, deadline, progress)
            require(np.array_equal(obs, observations[i, 0]), "Recorded reset observation differs")
            for j in range(int(t)):
                obs, _ = _step(env, actions[i, j], ledger, "prefix", deadline, progress)
                require(np.array_equal(obs, observations[i, j+1]), "Recorded prefix observation differs")
            states.append(env.state)
        finally:
            env.close()
    return np.asarray(states, np.float64)


def continuation_values(core, policy, states, times, banks, ledger, *, deadline=None, progress=None):
    """Full remaining return: candidate first, same frozen actor thereafter.

    Accumulation exponent starts at zero at the query. The t=99 reward is
    included; no t=100 dynamics or terminal bootstrap is evaluated.
    """
    states, times, banks = np.asarray(states), np.asarray(times), np.asarray(banks)
    n = len(states)
    _array(states, (n, 13), np.float64, "query states")
    require(times.shape == (n,) and times.dtype.kind in "iu"
            and np.all((times >= 0) & (times < HORIZON)), "Invalid query times")
    _array(banks, (n, 6, 4), np.float32, "fixed candidate banks")
    require(1 <= n <= 64, "Candidate-row cap exceeded")
    remaining = np.repeat(HORIZON-times, 6)
    total = int(remaining.sum())
    require(ledger["transition_steps_attempted"] + total <= ledger["max_transition_steps"],
            "Complete continuations cannot fit remaining cap")
    current = np.repeat(states, 6, axis=0).copy()
    first = deployed_actions(banks.reshape(-1, 4), n*6)
    values = np.zeros(n*6, np.float64)
    policy.assert_unchanged()
    for offset in range(int(remaining.max())):
        active = np.flatnonzero(offset < remaining)
        if offset == 0:
            actions = first[active]
        else:
            obs = observation(current[active], HORIZON-remaining[active]+offset)
            actions = actor_actions(policy, obs, ledger, deadline=deadline, progress=progress)
        nxt, rewards = _dynamics(core, current[active], actions, ledger, "continuation", deadline, progress)
        values[active] += (GAMMA**offset)*rewards
        current[active] = nxt
    policy.assert_unchanged()
    require(np.all(np.isfinite(values)), "Nonfinite full Q^pi reference")
    ledger["continuation_rows_completed"] = ledger.get("continuation_rows_completed", 0) + n*6
    return values.reshape(n, 6)


def runtime_parity(core, x0, ledger, *, deadline=None, progress=None):
    """104 transitions: one full environment path + four direct comparisons.

    Checks t=0,25,50,99 before action, exact reward/state and observed time after.
    It uses public reset/step only and never mutates a private time field.
    """
    _array(x0, (13,), np.float64, "parity initial state")
    require(ledger["transition_steps_attempted"]+104 <= ledger["max_transition_steps"],
            "Parity cannot fit remaining cap")
    env = _env(core)
    checks = []
    action = np.asarray([-KNEE, .3, -.2, .1], np.float32)
    try:
        _reset(env, 2026102803, ledger, deadline, progress, state=x0)
        for t in range(100):
            before = env.state
            if t in (0, 25, 50, 99):
                nxt, reward = _dynamics(core, before, action, ledger, "parity_direct", deadline, progress)
            _, env_reward = _step(env, action, ledger, "parity_env", deadline, progress)
            if t in (0, 25, 50, 99):
                require(np.array_equal(env.state, nxt) and env_reward == float(reward),
                        "Direct dynamics/environment parity failed at t="+str(t))
                checks.append({"t_before": t, "t_after": t+1, "passed": True,
                               "terminal_bootstrap": 0. if t == 99 else None})
        return {"passed": True, "checks": checks, "transition_steps": 104,
                "scope": "Exact direct/env state, reward, observation, time, termination and discount"}
    finally:
        env.close()


def free_policy_rollouts(core, policy, episode_seeds, ledger, *, deadline=None, progress=None):
    """Unforced actor trajectories at already-development recorded start seeds.

    Returns before-action observations/actions, per-episode discounted returns,
    and true state-conditional hole entry. No efficacy comparison is made here.
    """
    seeds = np.asarray(episode_seeds)
    require(seeds.ndim == 1 and seeds.dtype.kind in "iu" and 1 <= len(seeds) <= 16
            and len(np.unique(seeds)) == len(seeds), "Invalid free-policy seeds")
    n = len(seeds)
    require(ledger["transition_steps_attempted"]+n*100 <= ledger["max_transition_steps"],
            "Complete free-policy paths cannot fit remaining cap")
    envs = []
    observations = np.empty((n, 100, 14), np.float32)
    actions = np.empty((n, 100, 4), np.float32)
    returns = np.zeros(n, np.float64)
    try:
        current = []
        for seed in seeds:
            env = _env(core); envs.append(env)
            current.append(_reset(env, seed, ledger, deadline, progress))
        current = np.asarray(current, np.float32)
        for t in range(100):
            command = actor_actions(policy, current, ledger, deadline=deadline, progress=progress)
            observations[:, t] = current
            actions[:, t] = command
            for i, env in enumerate(envs):
                current[i], reward = _step(env, command[i], ledger, "free_policy", deadline, progress)
                returns[i] += GAMMA**t*reward
        load = observations[..., 10].astype(np.float64)+observations[..., 11].astype(np.float64)
        high = load >= .4
        band = np.abs(actions[..., 0].astype(np.float64)+KNEE) <= .05
        hole = high & band
        return {"observations": observations, "actions": actions, "returns": returns,
                "episode_seeds": seeds.copy(), "hole_entries": int(hole.sum()),
                "high_load_action_rows": int(high.sum()), "action_rows": int(n*100),
                "episodes_with_hole_entry": int(np.any(hole, axis=1).sum()),
                "episodes_with_high_load": int(np.any(high, axis=1).sum()),
                "hole_entries_per_episode": hole.sum(axis=1).tolist(),
                "reset_scope": "Previously development-used recorded episode seeds; no fresh-confirmation claim"}
    finally:
        for env in envs:
            env.close()
