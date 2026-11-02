"""Bounded, evaluator-only conditional values for a frozen deterministic actor.

This module does not import a learner, fit a calibrator, load checkpoints, or
choose a penalty. The caller must authenticate the core and frozen inference
export before calling it. Importing this module runs no scientific code.

Queries and candidate banks are list-compatible objects with accounting metadata.
Keep those objects intact until continuation_values returns (returns, ledger).
All state/action arrays stored in them are defensive, read-only copies.
"""
from copy import deepcopy
import hashlib
import json


MATRIX_SHA256 = "289afb581b4fdbc97fd68566c0b42ff6244b20686e9e49ba177e7e6bd57454a1"
CANONICAL_SOURCE_SHA256 = "f22f2e214ad9231bee4ce60ef55e64409534c2654c4e184406553ba8ed67dde1"
VERIFICATION_STEPS = 104
MAX_CONTINUATIONS = 576


class EvaluationError(RuntimeError):
    """A failed evaluation with a conservative record of attempted transitions."""

    def __init__(self, message, ledger):
        super().__init__(message)
        self.ledger = deepcopy(ledger)


class _EvaluatorList(list):
    def __init__(self, values, *, kind, protocol_key, ledger):
        super().__init__(values)
        self.kind = kind
        self.protocol_key = protocol_key
        self.ledger = deepcopy(ledger)


def _protocol(protocol):
    required = {
        "phase": "development", "query_count": 64,
        "query_times": [0, 25, 50, 75], "query_seed": 2026102400,
        "candidate_seed": 2026102342, "gamma": 0.98, "horizon": 100,
        "family": "recorded", "physics": {"stall": True, "stall_slope": 1.5},
        "max_transition_steps": 60000,
    }
    for key, expected in required.items():
        if key not in protocol or protocol[key] != expected:
            raise ValueError(f"Unsupported evaluator protocol {key}: expected {expected!r}")
    for key in ("query_count", "query_seed", "candidate_seed", "horizon", "max_transition_steps"):
        if type(protocol[key]) is not int:
            raise ValueError(f"Protocol {key} must be an integer, not a coerced value")
    if any(type(t) is not int for t in protocol["query_times"]):
        raise ValueError("Query times must be integers")
    if protocol["physics"].get("stall") is not True:
        raise ValueError("This diagnostic requires the stalling recorded task")
    for key, expected in {"max_continuations": 576, "task_semantics": "finite_horizon",
                          "terminal_bootstrap": False,
                          "confirmation_seed_start_reserved_not_generated": 2026102500,
                          "confirmation_count_reserved_not_generated": 64}.items():
        if key in protocol and protocol[key] != expected:
            raise ValueError(f"Conflicting optional protocol field {key}")
    return hashlib.sha256(json.dumps(required, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _core_contract(core, protocol):
    _protocol(protocol)
    config = core.canonical_config()
    expected_physics = {
        "stall": True, "stall_slope": 1.5, "knee": 0.800000011920929,
        "gain": 0.3, "throttle_cost": 0.3, "wave_turn": 0.27,
        "gamma": 0.98, "length": 100,
    }
    expected = {
        "physics": expected_physics, "state_dim": 13, "action_dim": 4,
        "observation_dim": 14, "observation_mode": "time_aware",
        "task_semantics": "finite_horizon", "objective_gamma": 0.98,
        "horizon": 100, "terminal_reward": 0.0, "terminal_bootstrap": False,
        "matrix_sha256": MATRIX_SHA256,
    }
    if config != expected:
        raise ValueError("Core task/configuration differs from the declared deterministic evaluator")
    if core.dynamics is not core.canonical.dynamics:
        raise ValueError("Evaluator requires the authenticated canonical dynamics function")
    if core.Physics(**protocol["physics"]).as_dict() != expected_physics:
        raise ValueError("Constructed physics differs from the authenticated task")
    # canonical_config calls the core provenance check; the outer runner also
    # verifies the entire core manifest. A config check alone is not a source attestation.
    return deepcopy(config)


def _readonly(value, dtype):
    import numpy as np
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _array(value, shape, name, *, dtype=None):
    import numpy as np
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu" or not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must be finite real values of shape {shape}")
    if dtype is not None and raw.dtype != np.dtype(dtype):
        raise ValueError(f"{name} must preserve exact {np.dtype(dtype)} storage")
    return raw


def _actions(value, count, name):
    import numpy as np
    raw = _array(value, (count, 4), name)
    # An invalid actor is not silently repaired by the environment's clipping.
    if np.any(raw < -1.0) or np.any(raw > 1.0):
        raise ValueError(f"{name} contains an out-of-bounds command; refusing hidden clipping")
    deployed = np.clip(raw.astype(np.float32), -1.0, 1.0)
    if not np.all(np.isfinite(deployed)):
        raise ValueError(f"{name} is nonfinite after float32 deployment")
    return deployed


def _observation(states, remaining):
    import numpy as np
    states = np.asarray(states, np.float64)
    remaining = np.asarray(remaining, np.float64)
    return np.concatenate((states, (remaining / 100.0)[..., None]), axis=-1).astype(np.float32)


def _policy_actions(policy, observations, ledger, stage):
    import numpy as np
    observations = np.asarray(observations)
    _array(observations, (len(observations), 14), "policy observations", dtype=np.float32)
    policy.assert_unchanged()
    ledger[stage + "_policy_calls"] = ledger.get(stage + "_policy_calls", 0) + 1
    ledger[stage + "_policy_rows"] = ledger.get(stage + "_policy_rows", 0) + len(observations)
    actions = _actions(policy.act(observations), len(observations), "frozen actor output")
    policy.assert_unchanged()
    return actions


def _new_ledger(protocol, core_config):
    return {
        "schema": "unifloral.frozen_usefulness.simulation_ledger.v1",
        "phase": "development", "status": "in_progress",
        "query_seed": protocol["query_seed"], "candidate_seed": protocol["candidate_seed"],
        "query_count": 64, "query_times": [0, 25, 50, 75],
        "gamma": 0.98, "horizon": 100, "family": "recorded",
        "core_config": core_config,
        "canonical_source_sha256_expected": CANONICAL_SOURCE_SHA256,
        "source_authentication": "Caller authenticates full core; canonical_config authenticates canonical files",
        "max_transition_steps": 60000, "max_continuations": 576,
        "transition_steps_attempted": 0, "transition_steps_completed": 0,
        "prefix_steps_attempted": 0, "prefix_steps_completed": 0,
        "verification_steps_attempted": 0, "verification_steps_completed": 0,
        "continuation_steps_attempted": 0, "continuation_steps_completed": 0,
        "prefix_policy_calls": 0, "prefix_policy_rows": 0,
        "candidate_policy_calls": 0, "candidate_policy_rows": 0,
        "continuation_policy_calls": 0, "continuation_policy_rows": 0,
        "prefix_resets": 0, "verification_resets": 0,
        "verification_env_steps": 0, "verification_direct_dynamics_rows": 0,
        "continuation_dynamics_calls": 0, "continuations_completed": 0,
        "learner_optimizer_updates": 0, "calibrator_updates": 0,
        "confirmation_starts_generated": 0, "monte_carlo_replications": 0,
        "deterministic_continuations_per_candidate": 1,
        "conditional_monte_carlo_standard_error": 0.0,
        "population_uncertainty_assessed_by_rollout": False,
        "value_scope": "Deterministic finite-horizon value of each first action followed by the same frozen actor",
        "numeric_scope": "Float64 state/reward accumulation with float32 actor observations/actions; no rigorous floating-point error bound",
    }


def _charge(ledger, stage, rows):
    if type(rows) is not int or rows <= 0:
        raise ValueError("Transition charge must be a positive integer")
    if ledger["transition_steps_attempted"] + rows > ledger["max_transition_steps"]:
        raise EvaluationError("Prospective transition budget would be exceeded", ledger)
    # Charge before dispatch, including a possibly failing transition. Completed
    # counters are increased only after the call returns successfully.
    ledger["transition_steps_attempted"] += rows
    ledger[stage + "_steps_attempted"] += rows


def _completed(ledger, stage, rows):
    ledger["transition_steps_completed"] += rows
    ledger[stage + "_steps_completed"] += rows


def _env(core, protocol):
    return core.ThrusterChainEnv(**protocol["physics"], start="recorded",
                                 observation_mode="time_aware", task_semantics="finite_horizon")


def _check_env_transition(env, observation, reward, terminated, truncated, info, expected_t):
    import numpy as np
    state = _array(env.state, (13,), "environment state", dtype=np.float64)
    obs = _array(observation, (14,), "environment observation", dtype=np.float32)
    if env.t != expected_t or not np.array_equal(obs, _observation(state, 100 - expected_t)):
        raise AssertionError("Environment state/time-aware observation does not match the deployed boundary")
    terminal = expected_t == 100
    if bool(terminated) != terminal or bool(truncated):
        raise AssertionError("Unexpected early termination or continuing-task truncation")
    if info["remaining_steps"] != 100 - expected_t or info["bootstrap_mask"] != (0.0 if terminal else 1.0):
        raise AssertionError("Finite-horizon remaining-time/bootstrap semantics differ")
    if info["continuation_discount"] != (0.0 if terminal else 0.98):
        raise AssertionError("Continuation discount differs from finite-horizon objective")
    if not np.isfinite(reward):
        raise AssertionError("Environment returned a nonfinite reward")


def build_queries(core, policy, protocol):
    """Return 64 independently seeded, frozen-actor-prefix query states.

    Only this evaluator receives exact simulator state. A query at time t uses
    t prefix transitions and saves all deployed prefix actions for audit.
    """
    import numpy as np
    key = _protocol(protocol)
    ledger = _new_ledger(protocol, _core_contract(core, protocol))
    envs, observations, prefixes = [], [], [[] for _ in range(64)]
    times = [protocol["query_times"][i % 4] for i in range(64)]
    try:
        policy.assert_unchanged()
        for i in range(64):
            env = _env(core, protocol)
            envs.append(env)
            observation, info = env.reset(seed=protocol["query_seed"] + i)
            ledger["prefix_resets"] += 1
            _check_env_transition(env, observation, 0.0, False, False, info, 0)
            observations.append(observation)
        for step in range(max(times)):
            active = [i for i, t in enumerate(times) if step < t]
            actions = _policy_actions(policy, np.stack([observations[i] for i in active]), ledger, "prefix")
            for row, i in enumerate(active):
                _charge(ledger, "prefix", 1)
                obs, reward, ended, truncated, info = envs[i].step(actions[row])
                _completed(ledger, "prefix", 1)
                _check_env_transition(envs[i], obs, reward, ended, truncated, info, step + 1)
                observations[i] = obs
                prefixes[i].append(actions[row].copy())
        queries = [{
            "group_id": i, "episode_seed": protocol["query_seed"] + i, "t": times[i],
            "state": _readonly(envs[i].state, np.float64),
            "observation": _readonly(observations[i], np.float32),
            "prefix_actions": _readonly(np.asarray(prefixes[i], np.float32).reshape(times[i], 4), np.float32),
        } for i in range(64)]
        if ledger["prefix_steps_completed"] != 2400:
            raise AssertionError("Unexpected prefix accounting")
        policy.assert_unchanged()
        ledger["status"] = "queries_complete"
        return _EvaluatorList(queries, kind="queries", protocol_key=key, ledger=ledger)
    except Exception as exc:
        ledger["status"] = "failed_queries"
        raise EvaluationError(f"Query construction failed: {type(exc).__name__}: {exc}", ledger) from exc
    finally:
        for env in envs:
            env.close()


def _query_contract(queries, protocol):
    import numpy as np
    if not isinstance(queries, _EvaluatorList) or queries.kind != "queries" or queries.protocol_key != _protocol(protocol):
        raise ValueError("Keep the original build_queries list with its accounting metadata")
    if len(queries) != 64 or queries.ledger.get("prefix_steps_completed") != 2400:
        raise ValueError("Query count or prefix receipt differs from the prospective protocol")
    for i, query in enumerate(queries):
        t = protocol["query_times"][i % 4]
        if query["group_id"] != i or query["episode_seed"] != protocol["query_seed"] + i or query["t"] != t:
            raise ValueError("Query identity, seed, or time was changed")
        state = _array(query["state"], (13,), "saved query state", dtype=np.float64)
        obs = _array(query["observation"], (14,), "saved query observation", dtype=np.float32)
        if not np.array_equal(obs, _observation(state, 100 - t)):
            raise ValueError("Query observation does not match full state and remaining time")
        prefix = _array(query["prefix_actions"], (t, 4), "saved prefix actions", dtype=np.float32)
        _actions(prefix, t, "saved prefix actions")


def candidate_bank(policy, queries, protocol):
    """Predeclare actor, zero, and seven uniform candidates; deduplicate stably.

    The single PCG64 stream draws all (64,7,4) coordinates in query order. No
    critic, uncertainty, or simulator return is consulted to choose candidates.
    """
    import numpy as np
    _query_contract(queries, protocol)
    ledger = deepcopy(queries.ledger)
    ledger["status"] = "in_progress_candidates"
    try:
        observations = np.stack([q["observation"] for q in queries])
        actor = _policy_actions(policy, observations, ledger, "candidate")
        rng = np.random.Generator(np.random.PCG64(protocol["candidate_seed"]))
        uniform = rng.uniform(-1.0, 1.0, size=(64, 7, 4)).astype(np.float32)
        banks = []
        for i in range(64):
            proposed = _actions(np.concatenate((actor[i:i + 1], np.zeros((1, 4), np.float32), uniform[i])),
                                9, "candidate bank")
            unique = []
            for action in proposed:
                # Numeric equality intentionally treats -0 and +0 as the same action.
                if not any(np.array_equal(action, earlier) for earlier in unique):
                    unique.append(action)
            banks.append(_readonly(np.stack(unique), np.float32))
        policy.assert_unchanged()
        ledger.update(status="candidates_complete", candidate_seed=protocol["candidate_seed"], uniform_candidate_rows_drawn=448,
                      requested_candidate_rows=576, distinct_candidate_rows=sum(map(len, banks)),
                      generator="numpy.random.PCG64", deduplication="Stable after float32 deployment")
        return _EvaluatorList(banks, kind="banks", protocol_key=_protocol(protocol), ledger=ledger)
    except Exception as exc:
        ledger["status"] = "failed_candidates"
        raise EvaluationError(f"Candidate construction failed: {type(exc).__name__}: {exc}", ledger) from exc


def _dynamics(core, states, actions, physics, ledger, stage):
    import numpy as np
    scalar = np.asarray(states).ndim == 1
    rows = 1 if scalar else len(states)
    _array(states, (13,) if scalar else (rows, 13), "rollout state", dtype=np.float64)
    deployed = _actions(np.asarray(actions).reshape(rows, 4), rows, "rollout action")
    _charge(ledger, stage, rows)
    nxt, rewards = core.dynamics(states, deployed[0].astype(np.float64) if scalar else deployed.astype(np.float64), physics)
    _completed(ledger, stage, rows)
    _array(nxt, (13,) if scalar else (rows, 13), "next physical state", dtype=np.float64)
    _array(rewards, () if scalar else (rows,), "physical reward", dtype=np.float64)
    if np.any(np.sqrt(np.sum(np.asarray(nxt)[..., :10] ** 2, axis=-1)) > 300.0):
        raise AssertionError("Direct dynamics left the canonical physical-state bound")
    if stage == "verification":
        ledger["verification_direct_dynamics_rows"] += rows
    else:
        ledger["continuation_dynamics_calls"] += 1
    return nxt, rewards


def _verification(core, policy, protocol, queries, ledger):
    """104 transitions: 101 env.step calls and three direct dynamics rows."""
    import numpy as np
    physics = core.Physics(**protocol["physics"])
    x0 = queries[0]["state"]
    fixed = [np.zeros(4, np.float32), np.asarray([0.8, -0.8, 1.0, -1.0], np.float32)]
    envs = []
    checks = []

    def env_step(env, action, expected_t):
        _charge(ledger, "verification", 1)
        obs, reward, terminal, truncated, info = env.step(action)
        _completed(ledger, "verification", 1)
        ledger["verification_env_steps"] += 1
        _check_env_transition(env, obs, reward, terminal, truncated, info, expected_t)
        return obs, reward, info

    def compare(env, action, expected_t, label):
        before = env.state
        nxt, direct_reward = _dynamics(core, before, action, physics, ledger, "verification")
        obs, reward, info = env_step(env, action, expected_t)
        if not np.array_equal(env.state, nxt) or float(reward) != float(direct_reward):
            raise AssertionError("Direct dynamics/env.step differ at " + label)
        deployed64 = _actions(action[None, :], 1, "verification action")[0].astype(np.float64)
        # Same explicit scalar accumulation order as canonical reward; this
        # independently checks that the cost is based on pre-transition state.
        s2 = before[0] * before[0]
        for j in range(1, 10):
            s2 = s2 + before[j] * before[j]
        a2 = deployed64[0] * deployed64[0]
        for j in range(1, 4):
            a2 = a2 + deployed64[j] * deployed64[j]
        expected_reward = -(s2 + physics.throttle_cost * a2)
        if float(reward) != float(expected_reward):
            raise AssertionError("Pre-transition state/action reward mismatch")
        checks.append({"name": label, "passed": True, "t_before": expected_t - 1,
                       "t_after": expected_t, "reward": float(reward),
                       "remaining_steps": int(info["remaining_steps"]),
                       "bootstrap_mask": float(info["bootstrap_mask"]),
                       "terminal_continuation_value": 0.0 if expected_t == 100 else "not_terminal"})

    try:
        policy.assert_unchanged()
        for action in fixed:
            env = _env(core, protocol)
            envs.append(env)
            obs, info = env.reset(seed=protocol["query_seed"], options={"state": x0.copy()})
            ledger["verification_resets"] += 1
            _check_env_transition(env, obs, 0.0, False, False, info, 0)
            compare(env, action, 1, "t0_fixed_action_" + str(len(envs)))
        env = envs[0]
        for t in range(2, 100):
            env_step(env, fixed[0], t)
        compare(env, fixed[1], 100, "t99_last_reward_zero_bootstrap")
        if ledger["verification_steps_completed"] != VERIFICATION_STEPS:
            raise AssertionError("Verification exceeded its declared 104-transition scope")
        policy.assert_unchanged()
        ledger["verification"] = {
            "passed": True, "checks": checks,
            "fixed_actions": [a.tolist() for a in fixed],
            "terminal_value_algebra": "At t=99: Q=r99 + gamma*0 = r99; no t=100 reward or bootstrap",
            "comparison": "Exact float64 states/rewards; exact float32 observations and finite-task flags",
        }
    finally:
        for env in envs:
            env.close()


def continuation_values(core, policy, queries, banks, protocol):
    """Return (64 aligned float64 return arrays, complete transition ledger).

    Each candidate gets exactly one deterministic full-state continuation.
    Discount starts at exponent zero at the query, not at the episode origin.
    All continuations use the same frozen actor after their candidate first action.
    """
    import numpy as np
    _query_contract(queries, protocol)
    core_config = _core_contract(core, protocol)
    if core_config != queries.ledger["core_config"]:
        raise ValueError("Core configuration changed after query generation")
    if not isinstance(banks, _EvaluatorList) or banks.kind != "banks" or banks.protocol_key != _protocol(protocol):
        raise ValueError("Keep the original candidate_bank list with its accounting metadata")
    if len(banks) != 64:
        raise ValueError("Expected one candidate bank per independent query")
    ledger = deepcopy(queries.ledger)
    ledger.update(deepcopy(banks.ledger))
    ledger["status"] = "in_progress"
    try:
        counts = []
        for bank in banks:
            if not 1 <= len(bank) <= 9:
                raise ValueError("A candidate bank must have between one and nine deployed actions")
            _array(bank, (len(bank), 4), "saved candidate bank", dtype=np.float32)
            _actions(bank, len(bank), "saved candidate bank")
            if any(np.array_equal(bank[i], bank[j]) for i in range(len(bank)) for j in range(i)):
                raise ValueError("Candidate bank contains an undeclared duplicate after deployment")
            counts.append(len(bank))
        total = sum(counts)
        remaining = np.asarray([100 - q["t"] for q, bank in zip(queries, banks) for _ in bank], np.int64)
        continuation_steps = int(remaining.sum())
        if total > MAX_CONTINUATIONS or total != ledger["distinct_candidate_rows"]:
            raise ValueError("Continuation count differs from the predeclared candidate bank")
        if ledger["transition_steps_attempted"] + VERIFICATION_STEPS + continuation_steps > 60000:
            raise EvaluationError("Complete evaluation cannot fit inside the prospective cap", ledger)
        ledger.update(continuations_planned=total, continuation_steps_planned=continuation_steps,
                      total_transition_steps_planned=2400 + VERIFICATION_STEPS + continuation_steps,
                      candidate_counts=counts)
        policy.assert_unchanged()
        _verification(core, policy, protocol, queries, ledger)
        states = np.stack([q["state"] for q, bank in zip(queries, banks) for _ in bank]).astype(np.float64)
        first = np.concatenate(banks, axis=0)
        values = np.zeros(total, np.float64)
        physics = core.Physics(**protocol["physics"])
        for offset in range(int(remaining.max())):
            active = np.flatnonzero(offset < remaining)
            if offset == 0:
                actions = first[active]
            else:
                observations = _observation(states[active], remaining[active] - offset)
                actions = _policy_actions(policy, observations, ledger, "continuation")
            nxt, rewards = _dynamics(core, states[active], actions, physics, ledger, "continuation")
            values[active] += (protocol["gamma"] ** offset) * rewards
            states[active] = nxt
            ledger["continuations_completed"] += int(np.count_nonzero(remaining[active] == offset + 1))
        policy.assert_unchanged()
        if not np.all(np.isfinite(values)):
            raise AssertionError("Conditional continuation return is nonfinite")
        if ledger["continuation_steps_completed"] != continuation_steps or ledger["continuations_completed"] != total:
            raise AssertionError("Continuation accounting does not match the exact remaining horizons")
        if ledger["transition_steps_attempted"] != ledger["transition_steps_completed"]:
            raise AssertionError("Successful ledger contains an unfinished transition")
        split = np.cumsum(counts)[:-1]
        returns = [_readonly(value, np.float64) for value in np.split(values, split)]
        ledger["status"] = "complete"
        ledger["frozen_policy_unchanged"] = True
        ledger["candidate_selection_used_simulator_labels"] = False
        ledger["conditional_value_error_note"] = (
            "No Monte Carlo sampling error conditional on these saved states and deterministic actor. "
            "Finite-precision error remains; population uncertainty and candidate selection uncertainty are separate."
        )
        return returns, ledger
    except Exception as exc:
        ledger["status"] = "failed_continuations"
        raise EvaluationError(f"Continuation evaluation failed: {type(exc).__name__}: {exc}", ledger) from exc


__all__ = ["build_queries", "candidate_bank", "continuation_values", "EvaluationError"]
