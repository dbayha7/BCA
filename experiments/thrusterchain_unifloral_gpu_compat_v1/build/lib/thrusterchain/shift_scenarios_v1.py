"""Additive dynamics shifts and frozen evaluation for the v6 benchmark.

The wrapper deliberately does not alter the canonical plant, its training data,
or historical studies. Scenario metadata are diagnostic, not policy inputs.
"""
import hashlib
import json
import numbers
from pathlib import Path
from types import MappingProxyType

import gymnasium as gym
import numpy as np

from .comparison import validate_partitions
from . import environment as environment_module
from .canonical import provenance as canonical_provenance
from .environment import ThrusterChainEnv, canonical, dynamics
from .inference_guard import InferenceGuard


SCENARIO_SCHEMA = "thrusterchain.shift_scenarios.v1"
SCENARIOS = MappingProxyType({
    "baseline": MappingProxyType({"command_multiplier": (1., 1., 1., 1.), "disturbance_bound": 0.}),
    "attenuated_thruster": MappingProxyType({"command_multiplier": (.55, 1., 1., 1.), "disturbance_bound": 0.}),
    "bounded_disturbance": MappingProxyType({"command_multiplier": (1., 1., 1., 1.), "disturbance_bound": .02}),
    "combined": MappingProxyType({"command_multiplier": (.55, 1., 1., 1.), "disturbance_bound": .02}),
})


def _seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or not 0 <= seed < 2**63:
        raise ValueError("episode seed must be an integer in [0, 2**63)")
    return int(seed)


def _command(action):
    raw = np.asarray(action)
    if raw.dtype.kind not in "fiu" or raw.shape != (4,) or not np.all(np.isfinite(raw)):
        raise ValueError("action must be four finite real numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        value = raw.astype(np.float32)
    if not np.all(np.isfinite(value)):
        raise ValueError("action must remain finite when rounded to float32")
    return np.clip(value, -1, 1)


def _reset_seed(seed):
    return int(np.random.SeedSequence(_seed(seed), spawn_key=(5101,)).generate_state(1, dtype=np.uint64)[0] & np.uint64(2**63 - 1))


class ShiftScenarioEnv(gym.Wrapper):
    """Fixed finite task with a versioned action map and process disturbance.

    Every case has the same observation and action spaces as the finite v5 task.
    The declared shift is not appended to observations: unknown scenario identity
    is latent and policy evaluation is generally partially observed across cases.
    Disturbance is independently uniform on each physical coordinate in the
    stated symmetric bounded interval after the canonical deterministic step.
    The pre-state and policy command determine reward, even under actuator loss.
    """

    def __init__(self, scenario="baseline", *, start="recorded", stall=True, stall_slope=1.5):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown shift scenario: {scenario!r}")
        super().__init__(ThrusterChainEnv(start=start, stall=stall, stall_slope=stall_slope))
        self.scenario = scenario
        self._spec = SCENARIOS[scenario]
        self._noise_rng = None
        self._episode_rng = None

    def reset(self, *, seed=None, options=None):
        if seed is None:
            if self._episode_rng is None:
                self._episode_rng = np.random.default_rng()
            seed = int(self._episode_rng.integers(0, 2**63, dtype=np.int64))
        else:
            seed = _seed(seed)
            self._episode_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(5100,)))
        # Separate directly addressed streams: drawing noise cannot perturb reset.
        reset_seed = _reset_seed(seed)
        self._noise_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(5102,)))
        obs, info = self.env.reset(seed=reset_seed, options=options)
        info = dict(info, scenario=self.scenario, episode_seed=seed, reset_seed=reset_seed,
                    scenario_schema=SCENARIO_SCHEMA)
        return obs.copy(), info

    def step(self, action):
        if self._noise_rng is None:
            raise RuntimeError("call reset() before step()")
        command = _command(action)
        state_before = self.env.state
        if state_before is None:
            raise RuntimeError("call reset() before step()")
        # Cost is computed from the command sent by the policy, not a discounted
        # actuator command. Canonical dynamics gives the exact same reward formula.
        _, reward = dynamics(state_before, command.astype(np.float64), self.env.physics)
        multiplier = np.asarray(self._spec["command_multiplier"], np.float32)
        applied = (command * multiplier).astype(np.float32)
        obs, _, terminated, truncated, info = self.env.step(applied)
        bound = float(self._spec["disturbance_bound"])
        noise = (self._noise_rng.uniform(-bound, bound, size=10) if bound else np.zeros(10))
        next_state = self.env.state
        next_state[:10] += noise
        if not np.isfinite(next_state).all() or canonical.float_radius(next_state) > canonical.FLOAT_BOUND:
            raise RuntimeError("shifted transition left the declared observation bounds")
        # This versioned wrapper binds to the copied v6 adapter's internal state;
        # reset/step parity tests detect upstream changes to that contract.
        self.env._x = next_state.copy()
        obs = self.env._obs()
        if not self.observation_space.contains(obs) or not np.isfinite(reward):
            raise RuntimeError("shifted transition produced an invalid observation or reward")
        info = dict(info)
        info.update(state=next_state.copy(), command=command.astype(np.float64),
                    actuator_command=applied.astype(np.float64), disturbance=noise.copy(),
                    scenario=self.scenario, scenario_schema=SCENARIO_SCHEMA)
        return obs.copy(), float(reward), terminated, truncated, info


def _state_bytes(policy):
    value = policy.state_bytes()
    if not isinstance(value, bytes):
        raise ValueError("frozen policy state_bytes must return bytes")
    return value


def evaluate_frozen_shifts(policy, *, scenarios, partitions, policy_task_semantics, purpose="final", start="recorded",
                           stall=True, stall_slope=1.5):
    """Evaluate an already-frozen policy on supplied disjoint episode-seed roles.

    `partitions` declares dataset, selection and final episode seeds. This checks
    the declarations only: it cannot attest precommitment, provenance, or that
    any final results/task families were unobserved during previous development.
    The same episode seed is reused across scenarios for paired comparisons.
    """
    if policy_task_semantics != "finite_horizon":
        raise ValueError("shift evaluation requires a declared finite_horizon policy; continuing policies are incompatible")
    if not isinstance(partitions, dict) or set(partitions) != {"dataset", "selection", "final"}:
        raise ValueError("partitions must declare dataset, selection and final")
    checked = validate_partitions(partitions["dataset"], partitions["selection"], partitions["final"])
    for role in checked:
        checked[role] = [_seed(value) for value in checked[role]]
    selection_resets = [_reset_seed(value) for value in checked["selection"]]
    final_resets = [_reset_seed(value) for value in checked["final"]]
    if (len(set(selection_resets)) != len(selection_resets) or len(set(final_resets)) != len(final_resets)
            or set(selection_resets) & set(final_resets)
            or (set(selection_resets) | set(final_resets)) & set(checked["dataset"])):
        raise ValueError("derived evaluation reset seeds overlap each other or declared dataset seeds")
    if purpose not in ("selection", "final"):
        raise ValueError("purpose must be selection or final")
    seeds = [_seed(value) for value in checked[purpose]]
    if not seeds:
        raise ValueError("selected evaluation partition must be nonempty")
    if isinstance(scenarios, str):
        raise ValueError("scenarios must be a sequence of names")
    scenarios = tuple(scenarios)
    if not scenarios or len(set(scenarios)) != len(scenarios) or any(x not in SCENARIOS for x in scenarios):
        raise ValueError("scenario names must be nonempty, unique, and registered")
    before = _state_bytes(policy)
    by_scenario = {}
    with InferenceGuard() as guard:
        for scenario in scenarios:
            env = ShiftScenarioEnv(scenario, start=start, stall=stall, stall_slope=stall_slope)
            episodes = []
            try:
                for seed in seeds:
                    obs, reset_info = env.reset(seed=seed)
                    total, discounted = 0., 0.
                    for step in range(100):
                        if _state_bytes(policy) != before:
                            raise RuntimeError("frozen policy state changed before action")
                        action = policy.act(obs.copy())
                        if _state_bytes(policy) != before:
                            raise RuntimeError("frozen policy state changed during action")
                        obs, reward, terminated, truncated, _ = env.step(action)
                        total += reward
                        discounted += (.98 ** step) * reward
                        if terminated != (step == 99) or truncated:
                            raise RuntimeError("finite-task horizon or termination changed")
                    episodes.append({"episode_seed": seed, "reset_seed": reset_info["reset_seed"],
                                     "steps": 100, "return": total, "discounted_return": discounted})
            finally:
                env.close()
            by_scenario[scenario] = {"specification": dict(SCENARIOS[scenario]), "episodes": episodes}
        if _state_bytes(policy) != before:
            raise RuntimeError("frozen policy state changed during evaluation")
        audit = guard.receipt()
        if any(audit[k] for k in ("training_calls", "dataset_collection_calls", "write_attempts", "forbidden_api_calls")):
            raise RuntimeError("instrumented training or persistence attempted during evaluation")
    source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {"schema": "thrusterchain.shift_evaluation.v1", "scenario_schema": SCENARIO_SCHEMA,
            "scenario_source_sha256": source, "purpose": purpose, "partitions": checked,
            "environment_source_sha256": hashlib.sha256(Path(environment_module.__file__).read_bytes()).hexdigest(),
            "canonical_provenance": canonical_provenance(),
            "policy_task_semantics_declared": policy_task_semantics,
            "policy_task_semantics_attested": False,
            "numpy_version": np.__version__,
            "random_streams": {"bit_generator": "PCG64", "seed_sequence": "numpy.random.SeedSequence",
                "subsequent_episode_spawn_key": [5100], "reset_spawn_key": [5101], "noise_spawn_key": [5102],
                "reset_seed_mapping": "first uint64 masked to 63 bits"},
            "start_law": start, "physics": {"stall": bool(stall), "stall_slope": float(stall_slope)},
            "policy_sha256_before": hashlib.sha256(before).hexdigest(),
            "policy_sha256_after": hashlib.sha256(_state_bytes(policy)).hexdigest(),
            "paired_episode_seeds": True, "episodes_total": len(seeds) * len(scenarios),
            "environment_steps": 100 * len(seeds) * len(scenarios), "scenarios": by_scenario,
            "partition_and_holdout_status": "declarations only; source history and prior exposure not attested",
            "uncertainty": "episodes are conditional on one fixed policy; training-seed uncertainty not measured",
            "inference_audit": audit}
