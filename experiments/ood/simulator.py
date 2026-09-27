"""Strict adapters for the installed D4RL Hopper/Walker MuJoCo 2.1 stack.

Only these audited dynamics/wrappers are supported. Capture the complete declared
restore schema, not just qpos/qvel. Unknown sources or missing fields stop tests.
"""

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np

from experiments.ood.adapters import require


SOURCE_SHA256 = {
    "gym.envs.mujoco.hopper": "3cdf42416ab559b9847212e3f0b6233e25c4f9e2021eaa380509cdbf7606e3ee",
    "gym.envs.mujoco.walker2d": "dec05f755efa64c32dfd76bbb278f5e52d307b1af1b976f07eaf6633268af375",
    "gym.envs.mujoco.mujoco_env": "f098927716b13fb13e25e924034347fb147acaca8e8d7122b5d37bc4da6c013d",
    "gym.wrappers.time_limit": "2adc707e54cd568dafcbe6386ac1e7295ef208721c3edc4a5a9d42789352687e",
    "gym.wrappers.order_enforcing": "b4a1cc1faca0508c1d4be105f462a92ec3af1ec3d69c14b0b89c5693cf01f9cd",
    "d4rl.utils.wrappers": "53ff0ce23479bdc56ffd37e10607c405b29666c589c6fe56047372c5a7d9580a",
    "d4rl.gym_mujoco.gym_envs": "9550b9a6ae7c66fe6d7d98bd1ad6e7b1659386b8c343399b3babaaabf37ce77c",
}
DATA_FIELDS = ("qpos", "qvel", "act", "qacc_warmstart", "ctrl", "qfrc_applied",
               "xfrc_applied", "mocap_pos", "mocap_quat", "userdata")
MODEL_FIELDS = ("body_pos", "body_quat", "site_pos")


def _tree_json(value):
    def encode(x):
        if isinstance(x, np.ndarray):
            return dict(dtype=x.dtype.str, shape=x.shape, hex=x.tobytes().hex())
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    return json.dumps(value, default=encode, sort_keys=True, allow_nan=False)


def tree_hash(value):
    return hashlib.sha256(_tree_json(value).encode()).hexdigest()


def _rng_state(rng):
    return deepcopy(rng.bit_generator.state if hasattr(rng, "bit_generator") else rng.get_state())


def _rng_restore(rng, state):
    if hasattr(rng, "bit_generator"):
        rng.bit_generator.state = deepcopy(state)
    else:
        rng.set_state(deepcopy(state))


class LocomotionAdapter:
    def __init__(self, dataset, *, max_transitions, evidence_dir):
        import importlib
        import gym
        import d4rl.gym_mujoco  # Registers the exact offline environments; no data download.
        require(dataset in ("hopper-medium-v2", "walker2d-medium-replay-v2"), "Unsupported environment.")
        require(type(max_transitions) is int and max_transitions > 0, "An explicit positive step ceiling is required.")
        for name, expected in SOURCE_SHA256.items():
            path = Path(inspect.getfile(importlib.import_module(name)))
            require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                    "Installed simulator/wrapper source changed: " + name)
        self.dataset, self.max_transitions = dataset, max_transitions
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=False)
        self.transitions = 0
        self.env = gym.make(dataset)  # Audited constructor executes one native step.
        try:
            self.order = self.env.env
            self.wrapper = self.order.env
            self.base = self.wrapper._wrapped_env
            require(type(self.env).__name__ == "TimeLimit" and type(self.order).__name__ == "OrderEnforcing"
                    and type(self.wrapper).__name__ == "NormalizedBoxEnv"
                    and type(self.base).__name__ == ("OfflineHopperEnv" if dataset.startswith("hopper") else "OfflineWalker2dEnv"),
                    "Unknown wrapper chain; no partial-state adapter fallback.")
            require(not self.wrapper._should_normalize and self.wrapper._reward_scale == 1.,
                    "Unexpected wrapper observation/reward transform.")
            self.sim = self.base.sim
            self.frame_skip = self.base.frame_skip
            self.max_episode_steps = self.env._max_episode_steps
            require(self.frame_skip == 4 and self.max_episode_steps == 1000, "Unexpected simulator horizon/integration budget.")
            self.action_dim = self.base.action_space.shape[0]
            self._model_sha = hashlib.sha256(self.sim.model.get_mjb()).hexdigest()
            self.identity = dict(dataset=dataset, source_sha256=SOURCE_SHA256,
                model_sha256=self._model_sha, frame_skip=self.frame_skip,
                max_episode_steps=self.max_episode_steps,
                state_absolute_tolerance=0.0, repeated_observation_absolute_tolerance=0.0,
                reward_absolute_tolerance=1e-7,
                action_transform="D4RL lb+(a+1)*0.5*(ub-lb), then clip; preserve incoming dtype",
                restore_fields=list(DATA_FIELDS + MODEL_FIELDS), native_reward="forward + 1 - .001*sum(applied_action**2)")
            self.terminated = self.truncated = False
        except Exception:
            self.env.close()
            raise

    def reset(self, seed):
        require(type(seed) is int and 0 <= seed < 2**32, "Explicit reset seed required.")
        self.env.seed(seed)
        observation = self.env.reset()
        self.terminated = self.truncated = False
        return np.asarray(observation).copy()

    def capture(self):
        require(self.env._elapsed_steps is not None, "Reset before capture.")
        # MuJoCo represents zero-sized inactive fields as None in this runtime.
        # Preserve that measured absence, rather than inventing empty arrays.
        result = {k: deepcopy(getattr(self.sim.data, k)) for k in DATA_FIELDS}
        result.update({k: deepcopy(getattr(self.sim.model, k)) for k in MODEL_FIELDS})
        result.update(schema="bca-locomotion-state-v1", identity=deepcopy(self.identity),
            time=float(self.sim.data.time), elapsed_steps=self.env._elapsed_steps,
            has_reset=self.order._has_reset,
            terminated=self.terminated, truncated=self.truncated,
            udd_state=deepcopy(self.sim.get_state().udd_state),
            rng=_rng_state(self.base.np_random),
            action_rng=_rng_state(self.wrapper.action_space.np_random),
            native_action_rng=_rng_state(self.base.action_space.np_random))
        return result

    snapshot_hash = staticmethod(tree_hash)

    def restore(self, snapshot, *, expected_sha256=None):
        expected = self.capture()
        require(set(snapshot) == set(expected), "Incomplete/extra simulator restore fields.")
        require(snapshot["schema"] == expected["schema"] and snapshot["identity"] == self.identity,
                "Wrong simulator/model identity.")
        if expected_sha256 is not None:
            require(tree_hash(snapshot) == expected_sha256, "Saved state hash mismatch.")
        for k in DATA_FIELDS + MODEL_FIELDS:
            b = expected[k]
            if b is None:
                require(snapshot[k] is None, "Inactive model field must remain None: " + k)
                continue
            a = np.asarray(snapshot[k])
            require(a.shape == b.shape and a.dtype == b.dtype and np.isfinite(a).all(),
                    "Invalid state shape/dtype/values: " + k)
        require(type(snapshot["elapsed_steps"]) is int and 0 <= snapshot["elapsed_steps"] <= self.max_episode_steps,
                "Invalid elapsed time-limit state.")
        require(type(snapshot["terminated"]) is bool and type(snapshot["truncated"]) is bool,
                "Episode-end flags must be boolean.")
        require(snapshot["has_reset"] is True, "Cannot restore a state before reset.")
        require(np.isfinite(snapshot["time"]) and snapshot["time"] >= 0, "Invalid simulator time.")
        # Validate RNG structures before mutating the live simulator.
        for rng, key in ((self.base.np_random, "rng"), (self.wrapper.action_space.np_random, "action_rng"),
                         (self.base.action_space.np_random, "native_action_rng")):
            # Gym's legacy Generator subclass has an incompatible deepcopy
            # reducer on this NumPy version. Validate the saved bit-generator
            # state through a temporary native NumPy instance instead.
            clone = (np.random.Generator(type(rng.bit_generator)(0))
                     if hasattr(rng, "bit_generator") else np.random.RandomState(0))
            _rng_restore(clone, snapshot[key])
        for k in MODEL_FIELDS:
            if snapshot[k] is not None:
                getattr(self.sim.model, k)[:] = snapshot[k]
        require(hashlib.sha256(self.sim.model.get_mjb()).hexdigest() == self._model_sha,
                "Uncaptured model parameter changed; restore refused.")
        state_type = type(self.sim.get_state())
        self.sim.set_state(state_type(snapshot["time"], snapshot["qpos"].copy(),
            snapshot["qvel"].copy(), deepcopy(snapshot["act"]), deepcopy(snapshot["udd_state"])))
        for k in DATA_FIELDS:
            if snapshot[k] is not None:
                getattr(self.sim.data, k)[:] = snapshot[k]
        self.sim.forward()
        # forward() refreshes derived quantities; restore the saved solver warm start afterwards.
        self.sim.data.qacc_warmstart[:] = snapshot["qacc_warmstart"]
        self.env._elapsed_steps = snapshot["elapsed_steps"]
        self.order._has_reset = snapshot["has_reset"]
        self.terminated, self.truncated = snapshot["terminated"], snapshot["truncated"]
        for rng, key in ((self.base.np_random, "rng"), (self.wrapper.action_space.np_random, "action_rng"),
                         (self.base.action_space.np_random, "native_action_rng")):
            _rng_restore(rng, snapshot[key])
        restored = self.capture()
        require(tree_hash(restored) == tree_hash(snapshot), "Restore did not reproduce declared state exactly.")

    def step(self, action):
        require(not (self.terminated or self.truncated), "Episode already ended.")
        require(self.transitions < self.max_transitions, "Engineering/collection step budget exhausted.")
        proposed = np.asarray(action)
        require(proposed.shape == (self.action_dim,) and proposed.dtype.kind == "f"
                and np.isfinite(proposed).all() and (np.abs(proposed) <= 1).all(), "Invalid normalized action.")
        position_before = float(self.sim.data.qpos[0])
        captured = []
        original_step = self.base.step
        require("step" not in self.base.__dict__, "Unexpected existing instance-level step override.")
        number = self.transitions + 1
        with (self.evidence_dir / f"transition-{number:05d}-input.npz").open("xb") as stream:
            np.savez(stream, proposed_action=proposed, state_json=np.asarray(_tree_json(self.capture())))

        def observe_applied(applied):
            captured.append(np.asarray(applied).copy())
            with (self.evidence_dir / f"transition-{number:05d}-applied.npz").open("xb") as stream:
                np.savez(stream, applied_action=captured[-1])
            return original_step(applied)

        self.base.step = observe_applied
        self.transitions += 1  # A failed attempt still consumes its allocated transition.
        try:
            observation, reward, done, info = self.env.step(proposed)
        finally:
            del self.base.step  # Restore inherited method; no permanent wrapper modification.
        require(len(captured) == 1, "Could not observe exactly one applied control vector.")
        applied = captured[0]
        ctrl = self.sim.data.ctrl.copy()
        forward = (float(self.sim.data.qpos[0]) - position_before) / self.base.dt
        cost = -.001 * float(np.square(applied.astype(np.float64)).sum())
        reconstructed = forward + 1. + cost
        error = abs(float(reward) - reconstructed)
        qpos, qvel = self.sim.data.qpos, self.sim.data.qvel
        if self.dataset.startswith("hopper"):
            vector = np.concatenate([qpos, qvel])
            healthy = np.isfinite(vector).all() and (np.abs(vector[2:]) < 100).all() and qpos[1] > .7 and abs(qpos[2]) < .2
        else:
            healthy = .8 < qpos[1] < 2 and -1 < qpos[2] < 1
        self.terminated = not bool(healthy)
        self.truncated = bool(info.get("TimeLimit.truncated", False))
        record = dict(proposed_action=proposed.copy(), applied_action=applied, sim_ctrl=ctrl,
            observation=np.asarray(observation).copy(), reward=float(reward), forward=forward,
            alive=1., action_cost=cost, reconstructed_reward=reconstructed, reward_error=error,
            terminated=self.terminated, truncated=self.truncated, elapsed_steps=self.env._elapsed_steps,
            physics_steps=self.frame_skip)
        # The caller must archive last_record even on a gate failure.
        self.last_record = record
        with (self.evidence_dir / f"transition-{self.transitions:05d}.npz").open("xb") as stream:
            np.savez(stream, **record)
        require(np.array_equal(ctrl, applied), "Applied action differs from simulator controls.")
        require(bool(done) == (self.terminated or self.truncated), "Termination/time-limit semantics mismatch.")
        require(np.isfinite(error) and error <= 1e-7, "Independent applied-action reward reconstruction failed.")
        return record

    def close(self):
        self.env.close()
