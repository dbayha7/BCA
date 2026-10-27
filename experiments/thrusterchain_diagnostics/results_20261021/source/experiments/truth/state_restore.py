"""Restore a logged D4RL Hopper/Walker2d row in MuJoCo and check that one step reproduces the log.

This is the first piece of the D4RL-RT truth instrument (experiments/truth/README.md). The critic's true
error at a logged (s, a) is measured by putting MuJoCo back into the logged state and rolling the frozen
policy out (labeller.py). That is only meaningful if the restored state IS the logged state. This module
does the restore and proves it, row by row, with a one-step reproduction gate.

Supported tasks (TASKS): hopper-medium-v2 and walker2d-medium-replay-v2, the two datasets the repo trains
TD3+BC on. Physics builds the audited simulator through gym.make, exactly as runtime/td3_bc.load_and_prepare
and evaluation do: TimeLimit -> OrderEnforcing -> D4RL NormalizedBoxEnv -> OfflineHopperEnv/OfflineWalker2dEnv
(gym 0.23.1 v2-style HopperEnv/Walker2dEnv, MuJoCo 2.1 via mujoco_py 2.1.2.14). It reuses the source pins of
experiments/ood/simulator.py (SOURCE_SHA256) and that adapter's wrapper-chain checks, but not its per-step
evidence journal, which measured 25 calls/s and cannot carry bulk labels.

Step semantics (Physics.step), copied operation for operation from the installed sources so that a batch of
bare MjSim instances reproduces env.step bit for bit (test_truth.py checks this against a direct gym rollout):
  applied = clip(lb + (a + 1.) * 0.5 * (ub - lb), lb, ub)   NormalizedBoxEnv, float32 (lb = -1, ub = 1, so the
                                                             map is the identity up to float32 rounding)
  ctrl[:] = applied; 4 x mj_step                             MujocoEnv.do_simulation, frame_skip 4, dt 0.008 s
  r = (x' - x) / dt; r += 1; r -= 1e-3 * sum(applied^2)       HopperEnv/Walker2dEnv.step, float64
  done: Hopper  not (finite(state) and all(|state[2:]| < 100) and z > 0.7 and |pitch| < 0.2)
        Walker2d not (0.8 < z < 2.0 and -1 < pitch < 1)
  obs = [qpos[1:], clip(qvel, -10, 10)]                       _get_obs (float64; the data store float32)
The time limit is not part of a step: callers stop at their own horizon (labeller: 1,000 steps).

Restore (Physics.restore). mj_resetData (sim.reset(): zeroes all of mjData, time 0, controls 0, the solver
warm start 0), then qpos and qvel through set_state, then mj_forward, then qacc_warmstart = 0 again. The
data do not store the solver warm start, so it is set to zero, the same value for every restore. Hopper and
Walker2d have no actuator activations (na = 0) and no user data. Because the whole of mjData is reset, a
restored sim does not remember what it simulated before.

Row map. D4RL's qlearning_dataset (what the critic trains on) drops every timeout row and the last raw row.
runtime/td3_bc._convert reproduces it and returns the raw row of every converted row; converted_rows calls
it unchanged, so converted row k is raw row rows[k], its next observation is raw observations[rows[k] + 1],
and a terminal row's next observation is the start of another episode (never compared).

Logged state (logged_states). Source 'infos' when the file has infos/qpos and infos/qvel (D4RL v2 locomotion
files are believed to; Stage 0 lists the keys): qpos, qvel = infos[row], cast float32 -> float64. Source
'rebuilt' otherwise: obs = [qpos[1:], clip(qvel, +-10)] and the x position qpos[0] enters neither the
dynamics nor the reward (only x' - x does), so qpos = [x0, obs[:nq-1]] with an arbitrary x0 (0.0) and
qvel = obs[nq-1:]. Rows with any velocity at the clip (|v| >= 10 in the float32 log) cannot be rebuilt
and are dropped and counted. Caveat, stated plainly: x0 is irrelevant in exact arithmetic but not in
floating point (contact geometry uses world positions), so a rebuilt rollout agrees with the infos rollout
only up to rounding; over long contact-rich rollouts chaos can amplify that rounding. The jitter mode of
labeller.py measures exactly this sensitivity, at a far larger scale (1e-4 of the data SD).

One-step gate (one_step_gate), per checked row:
  (a) restore: float32(obs of the restored sim) must equal the logged float32 observation. Exact (0.0 miss)
      whenever the state came from the same float32 log; the tolerance tol_restore is 0 by default.
  (b) step: one Physics.step with the logged action must reproduce the logged next observation (raw
      observations[row + 1] for non-terminal rows; raw next_observations[row] for terminal rows when the
      file has that key, otherwise the terminal row's next observation is not compared), the logged reward
      and the logged terminal flag.
Reported: the per-row max-abs miss of (a), of the next observation and of the reward, each as median,
99th percentile and max, the terminal agreement, and the counts. Pass: max (a) miss <= tol_restore,
p99 next-observation miss <= tol_next_obs, p99 reward miss <= tol_reward, terminal agreement >=
min_terminal_agreement. The defaults (DEFAULT_TOLERANCES) are loose against float32 storage of the state
and far below one step of motion; the real values are set on Stage 0's 1,000-row pilot and hashed into the
pre-registration before any label is used. A step miss on real data also measures drift between the
simulator that made the data and this MuJoCo 2.1 build. Misalignment by one row (offset=1, the negative
control) restores the wrong state and must fail both (a) and (b); test_truth.py checks that it does.

Nothing here trains anything or reads a checkpoint. Synthetic data (synthetic.py) make the module testable
in a container without the D4RL downloads; no synthetic number is a finding.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "bca-truth-state-restore-v1"
DEFAULT_DATA = Path.home() / ".d4rl" / "datasets"  # experiments/wbcp/freeze_scores.DEFAULT_DATA
TD3_CONFIG = ROOT / "configs" / "td3_bc.yaml"
CLIP = 10.0  # velocity clip of _get_obs
FRAME_SKIP = 4
MAX_EPISODE_STEPS = 1000
DEFAULT_TOLERANCES = dict(tol_restore=0.0, tol_next_obs=1e-3, tol_reward=1e-3, min_terminal_agreement=0.99)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    dataset: str
    base_class: str
    nq: int
    nv: int
    obs_dim: int
    action_dim: int


TASKS = {
    "hopper": TaskSpec("hopper", "hopper-medium-v2", "OfflineHopperEnv", 6, 6, 11, 3),
    "walker2d": TaskSpec("walker2d", "walker2d-medium-replay-v2", "OfflineWalker2dEnv", 9, 9, 17, 6),
}


def task_spec(name):
    """TaskSpec by task name ('hopper') or dataset name ('hopper-medium-v2')."""
    for spec in TASKS.values():
        if name in (spec.name, spec.dataset):
            return spec
    raise ValueError(f"unsupported task {name!r}; supported: {sorted(TASKS)}")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def array_sha256(value):
    """sha256 over dtype, shape and bytes (the convention of runtime/td3_bc._array_hash)."""
    a = np.ascontiguousarray(np.asarray(value))
    require(not a.dtype.hasobject, "object arrays have no reproducible identity")
    h = hashlib.sha256(repr((a.dtype.str, a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def observation(task, qpos, qvel):
    """_get_obs from state arrays, float64: [qpos[1:], clip(qvel, -10, 10)] (also batched on the last axis)."""
    task = task_spec(task) if isinstance(task, str) else task
    qpos, qvel = np.asarray(qpos, np.float64), np.asarray(qvel, np.float64)
    require(qpos.shape[-1] == task.nq and qvel.shape[-1] == task.nv, "state shape differs from the task")
    return np.concatenate([qpos[..., 1:], np.clip(qvel, -CLIP, CLIP)], axis=-1)


class Physics:
    """The audited simulator of one task plus its exact step, for any number of bare MjSim instances.

    The gym environment stays open for the lifetime of the object: it owns the MuJoCo model every new_sim()
    shares (each MjSim has its own mjData). identity() names the sources, the model bytes and the versions.
    """

    def __init__(self, dataset):
        from runtime.environment import setup

        setup()
        import gym
        import d4rl.gym_mujoco  # noqa: F401  registers the offline environments; no data download
        import mujoco_py

        from experiments.ood.simulator import SOURCE_SHA256

        self.task = task_spec(dataset)
        sources = {}
        for name, expected in SOURCE_SHA256.items():
            digest = sha256_bytes(Path(inspect.getfile(importlib.import_module(name))).read_bytes())
            require(digest == expected, "installed simulator/wrapper source changed: " + name)
            sources[name] = digest
        self._mujoco_py = mujoco_py
        self.simulator_error = mujoco_py.MujocoException  # raised by mujoco_py on MuJoCo warnings
        self.env = gym.make(self.task.dataset)
        try:
            order = self.env.env
            wrapper = order.env
            base = wrapper._wrapped_env
            require(type(self.env).__name__ == "TimeLimit" and type(order).__name__ == "OrderEnforcing"
                    and type(wrapper).__name__ == "NormalizedBoxEnv" and type(base).__name__ == self.task.base_class,
                    "unknown wrapper chain; no partial-state fallback")
            require(not wrapper._should_normalize and wrapper._reward_scale == 1.0,
                    "unexpected wrapper observation or reward transform")
            require(base.frame_skip == FRAME_SKIP and self.env._max_episode_steps == MAX_EPISODE_STEPS,
                    "unexpected frame skip or time limit")
            model = base.sim.model
            require((model.nq, model.nv, model.nu, model.na) == (self.task.nq, self.task.nv, self.task.action_dim, 0),
                    "model dimensions differ from the task")
            lb, ub = base.action_space.low, base.action_space.high
            require(lb.dtype == np.float32 and np.array_equal(lb, -np.ones_like(lb)) and np.array_equal(ub, -lb),
                    "native action bounds must be float32 +-1")
            self.wrapper, self.base, self.model = wrapper, base, model
            self.lb, self.ub = lb.copy(), ub.copy()
            self._span = self.ub - self.lb  # the same float32 values NormalizedBoxEnv computes per step
            self._hopper = self.task.name == "hopper"
            self.dt = base.dt  # model.opt.timestep * frame_skip, the float64 the env divides by
            with warnings.catch_warnings():  # mujoco_py's get_mjb leaves its temporary file open
                warnings.simplefilter("ignore", ResourceWarning)
                self.model_sha256 = sha256_bytes(model.get_mjb())
            self._identity = dict(schema=SCHEMA, dataset=self.task.dataset, task=self.task.name,
                                  model_sha256=self.model_sha256, source_sha256=sources, frame_skip=FRAME_SKIP,
                                  dt=float(self.dt), mujoco_py=mujoco_py.__version__, gym=gym.__version__,
                                  numpy=np.__version__, restore="mj_resetData; set qpos, qvel; mj_forward; "
                                  "qacc_warmstart = 0", action_map="clip(lb + (a + 1.) * 0.5 * (ub - lb), lb, ub)")
        except Exception:
            self.env.close()
            raise

    def identity(self):
        return dict(self._identity)

    def new_sim(self):
        """A bare MjSim on the shared model; nsubsteps = frame_skip, so one sim.step() is one env step."""
        return self._mujoco_py.MjSim(self.model, nsubsteps=FRAME_SKIP)

    def restore(self, sim, qpos, qvel):
        """Put sim into the state (qpos, qvel): full mjData reset, set_state, forward, zero warm start."""
        qpos, qvel = np.asarray(qpos, np.float64), np.asarray(qvel, np.float64)
        require(qpos.shape == (self.task.nq,) and qvel.shape == (self.task.nv,), "state shape differs from the task")
        require(np.isfinite(qpos).all() and np.isfinite(qvel).all(), "nonfinite state")
        sim.reset()
        old = sim.get_state()
        sim.set_state(self._mujoco_py.MjSimState(0.0, qpos, qvel, old.act, old.udd_state))
        sim.forward()
        sim.data.qacc_warmstart[:] = 0.0

    def observation(self, sim):
        """_get_obs of sim (float64); the same values as the module-level observation()."""
        data = sim.data
        return np.concatenate([data.qpos[1:], np.clip(data.qvel, -CLIP, CLIP)])

    def applied_action(self, action):
        """NormalizedBoxEnv.step's map, on the same dtypes (float32 in, float32 out); clip(x, lb, ub) is
        minimum(maximum(x, lb), ub) for finite x, written as the two ufuncs to skip np.clip's dispatch."""
        scaled = self.lb + (action + 1.) * 0.5 * self._span
        return np.minimum(np.maximum(scaled, self.lb), self.ub)

    def step(self, sim, action):
        """One env step of sim with the normalized action: (reward, done), bitwise as env.step."""
        action = np.asarray(action)
        if action.shape != (self.task.action_dim,) or not np.isfinite(action).all():
            raise ValueError("Action dimension mismatch or nonfinite action")
        applied = self.applied_action(action)
        data = sim.data
        qpos = data.qpos
        before = qpos[0]
        data.ctrl[:] = applied
        sim.step()
        after, height, ang = qpos[0:3]
        reward = (after - before) / self.dt
        reward += 1.0
        reward -= 1e-3 * np.square(applied).sum()
        if self._hopper:
            s = np.concatenate([qpos, data.qvel])
            done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > 0.7) and (abs(ang) < 0.2))
        else:
            done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        return reward, bool(done)

    def close(self):
        self.env.close()


# ---------------------------------------------------------------- rows and logged states

def load_raw(physics, path):
    """The raw D4RL mapping exactly as runtime/td3_bc.load_and_prepare reads it (env.get_dataset)."""
    path = Path(path)
    require(path.is_file() and path.suffix.lower() in (".h5", ".hdf5"), "existing cached HDF5 file required")
    return physics.env.get_dataset(h5path=str(path.resolve()))


def converted_rows(raw, max_episode_steps=MAX_EPISODE_STEPS):
    """(converted, rows, episode_ids) from runtime/td3_bc._convert, unchanged: converted row k is raw row rows[k]."""
    from runtime.td3_bc import _convert

    return _convert(raw, max_episode_steps)


@dataclass(frozen=True)
class LoggedStates:
    """Restorable states for a set of raw rows. kept[i] is False where the state cannot be rebuilt."""
    raw_rows: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    kept: np.ndarray
    source: str
    x0: float | None

    @property
    def dropped_at_clip(self):
        return int((~self.kept).sum())


def has_infos(raw):
    return "infos/qpos" in raw and "infos/qvel" in raw


def logged_states(task, raw, raw_rows, source="auto", x0=0.0):
    """qpos/qvel (float64) of the given raw rows. source: 'infos', 'rebuilt' or 'auto' (infos when present)."""
    task = task_spec(task) if isinstance(task, str) else task
    raw_rows = np.asarray(raw_rows, np.int64)
    n = len(raw["observations"])
    require(raw_rows.ndim == 1 and np.all(raw_rows >= 0) and np.all(raw_rows < n), "raw rows out of range")
    if source == "auto":
        source = "infos" if has_infos(raw) else "rebuilt"
    if source == "infos":
        require(has_infos(raw), "the file has no infos/qpos and infos/qvel")
        qpos = np.asarray(raw["infos/qpos"])[raw_rows].astype(np.float64)
        qvel = np.asarray(raw["infos/qvel"])[raw_rows].astype(np.float64)
        require(qpos.shape == (len(raw_rows), task.nq) and qvel.shape == (len(raw_rows), task.nv),
                "infos/qpos or infos/qvel has the wrong width")
        return LoggedStates(raw_rows, qpos, qvel, np.ones(len(raw_rows), bool), "infos", None)
    require(source == "rebuilt", "source must be 'auto', 'infos' or 'rebuilt'")
    qpos, qvel, kept = rebuild_from_obs(task, np.asarray(raw["observations"])[raw_rows], x0)
    return LoggedStates(raw_rows, qpos, qvel, kept, "rebuilt", float(x0))


def rebuild_from_obs(task, obs, x0=0.0):
    """(qpos, qvel, kept) from logged observations: qpos = [x0, obs[:nq-1]], qvel = obs[nq-1:].

    kept is False for rows with any velocity at the +-10 clip (the true velocity is unknown there); their
    qpos/qvel are still filled from the clipped values but must not be restored."""
    task = task_spec(task) if isinstance(task, str) else task
    obs = np.asarray(obs)
    require(obs.ndim == 2 and obs.shape[1] == task.obs_dim, "observation width differs from the task")
    obs = obs.astype(np.float64)
    vel = obs[:, task.nq - 1:]
    qpos = np.concatenate([np.full((len(obs), 1), float(x0)), obs[:, :task.nq - 1]], axis=1)
    kept = np.all(np.abs(vel) < CLIP, axis=1)
    return qpos, vel.copy(), kept


def state_sd(states):
    """Per-coordinate SD of the kept states (qpos SD, qvel SD), float64, for the jitter mode of labeller.py.

    The x position qpos[0] gets SD 0: it enters neither the dynamics nor the reward, and its data SD (metres
    travelled) says nothing about the state's scale."""
    q, v = states.qpos[states.kept], states.qvel[states.kept]
    require(len(q) >= 2, "need at least two kept states")
    sd_q = q.std(axis=0)
    sd_q[0] = 0.0
    return sd_q, v.std(axis=0)


# ---------------------------------------------------------------- the gate

def gate_inputs(task, raw, rows, converted, converted_ids, *, source="auto", offset=0, x0=0.0):
    """Logged arrays and states for converted rows. offset != 0 is the negative control: the state is taken
    from raw row rows[k] + offset while everything else comes from converted row k."""
    converted_ids = np.asarray(converted_ids, np.int64)
    require(converted_ids.ndim == 1 and len(converted_ids) and np.all(np.diff(converted_ids) > 0),
            "sorted unique converted ids required")
    require(np.all(converted_ids >= 0) and np.all(converted_ids < len(rows)), "converted ids out of range")
    raw_rows = np.asarray(rows, np.int64)[converted_ids]
    state_rows = raw_rows + int(offset)
    require(np.all(state_rows >= 0) and np.all(state_rows < len(raw["observations"])), "offset leaves the file")
    states = logged_states(task, raw, state_rows, source, x0)
    terminal = np.asarray(converted["terminals"])[converted_ids].astype(bool)
    logged = dict(obs=np.asarray(converted["observations"])[converted_ids],
                  action=np.asarray(converted["actions"])[converted_ids],
                  reward=np.asarray(converted["rewards"])[converted_ids],
                  next_obs=np.asarray(converted["next_observations"])[converted_ids],
                  terminal=terminal, raw_rows=raw_rows, converted_ids=converted_ids)
    if "next_observations" in raw:
        logged["terminal_next_obs"] = np.asarray(raw["next_observations"])[raw_rows]
    # The converted pool must be the raw rows (runtime/td3_bc._convert guarantees it; checked again here).
    require(np.array_equal(logged["obs"], np.asarray(raw["observations"])[raw_rows].astype(np.float32))
            and np.array_equal(logged["action"], np.asarray(raw["actions"])[raw_rows].astype(np.float32)),
            "converted rows differ from their raw rows")
    return logged, states


def _summary(values):
    values = np.asarray(values, np.float64)
    if not len(values):
        return dict(n=0, median=None, p99=None, max=None)
    return dict(n=int(len(values)), median=float(np.median(values)), p99=float(np.percentile(values, 99)),
                max=float(values.max()))


def one_step_gate(physics, logged, states, tolerances=None):
    """Restore each kept row, compare the observation, take one step with the logged action and compare the
    next observation, reward and terminal flag. Returns a strict-JSON-ready dict with per-row miss arrays
    under '_arrays' (popped by callers that write JSON)."""
    tol = dict(DEFAULT_TOLERANCES, **(tolerances or {}))
    require(set(tol) == set(DEFAULT_TOLERANCES), "unknown tolerance")
    kept = np.flatnonzero(states.kept)
    require(len(kept), "no restorable rows")
    sim = physics.new_sim()
    restore_miss = np.zeros(len(kept))
    next_miss = np.full(len(kept), np.nan)
    reward_miss = np.zeros(len(kept))
    term_ok = np.zeros(len(kept), bool)
    sim_terminal = np.zeros(len(kept), bool)
    for j, i in enumerate(kept):
        physics.restore(sim, states.qpos[i], states.qvel[i])
        obs32 = physics.observation(sim).astype(np.float32)
        restore_miss[j] = np.max(np.abs(obs32.astype(np.float64) - np.asarray(logged["obs"][i], np.float64)))
        reward, done = physics.step(sim, np.asarray(logged["action"][i], np.float32))
        nxt = physics.observation(sim)
        terminal = bool(logged["terminal"][i])
        if not terminal:
            next_miss[j] = np.max(np.abs(nxt - np.asarray(logged["next_obs"][i], np.float64)))
        elif "terminal_next_obs" in logged:
            next_miss[j] = np.max(np.abs(nxt - np.asarray(logged["terminal_next_obs"][i], np.float64)))
        reward_miss[j] = abs(float(reward) - float(logged["reward"][i]))
        sim_terminal[j] = done
        term_ok[j] = done == terminal
    compared = np.isfinite(next_miss)
    result = dict(
        schema=SCHEMA, source=states.source, x0=states.x0, rows_requested=int(len(states.kept)),
        rows_checked=int(len(kept)), rows_dropped_at_clip=states.dropped_at_clip,
        terminal_rows=int(np.asarray(logged["terminal"])[kept].sum()),
        next_obs_compared=int(compared.sum()),
        terminal_next_obs_source="raw next_observations" if "terminal_next_obs" in logged else "not compared",
        restore_miss=_summary(restore_miss), next_obs_miss=_summary(next_miss[compared]),
        reward_miss=_summary(reward_miss), terminal_agreement=float(term_ok.mean()),
        simulated_terminals=int(sim_terminal.sum()), tolerances=tol)
    passes = dict(
        restore=bool(restore_miss.max() <= tol["tol_restore"]),
        next_obs=bool(compared.any() and np.percentile(next_miss[compared], 99) <= tol["tol_next_obs"]),
        reward=bool(np.percentile(reward_miss, 99) <= tol["tol_reward"]),
        terminal=bool(term_ok.mean() >= tol["min_terminal_agreement"]))
    result["passes"] = passes
    result["passed"] = all(passes.values())
    result["_arrays"] = dict(kept_index=kept, restore_miss=restore_miss, next_obs_miss=next_miss,
                             reward_miss=reward_miss, simulated_terminal=sim_terminal)
    return result


# ---------------------------------------------------------------- cached files and the Stage 0 command line

def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cached_file(task, data_dir=DEFAULT_DATA, config=TD3_CONFIG):
    """The pinned cached HDF5 of a task (configs/td3_bc.yaml datasets.<task>.cache), sha256-verified."""
    from runtime.config import read_config

    task = task_spec(task) if isinstance(task, str) else task
    entry = read_config(config)["datasets"][task.name]
    require(entry["environment"] == task.dataset, "configs/td3_bc.yaml names a different environment")
    path = Path(data_dir).expanduser().resolve() / entry["cache"]["filename"]
    require(path.is_file(), f"cached file {path} is missing (D4RL download or copy needed)")
    digest = file_sha256(path)
    require(digest == entry["cache"]["sha256"], f"{path} sha256 {digest} differs from configs/td3_bc.yaml")
    return path, dict(path=str(path), sha256=digest, pinned=True, size_bytes=path.stat().st_size)


def _resolve_file(args):
    if args.file is not None:
        path = Path(args.file).expanduser().resolve()
        require(path.is_file(), f"{path} is missing")
        return path, dict(path=str(path), sha256=file_sha256(path), pinned=False, size_bytes=path.stat().st_size)
    return cached_file(args.dataset, args.data_dir)


def _synthetic_mark(path):
    import h5py

    with h5py.File(path, "r") as f:
        return f.attrs.get("bca_synthetic")


def describe_file(task, path):
    """Stage 0 key listing: every HDF5 dataset's shape and dtype, the infos fields, the rows at the clip."""
    import h5py

    task = task_spec(task) if isinstance(task, str) else task
    keys = {}
    with h5py.File(path, "r") as f:
        f.visititems(lambda name, item: keys.__setitem__(name, dict(shape=list(item.shape), dtype=str(item.dtype)))
                     if isinstance(item, h5py.Dataset) else None)
        obs = np.asarray(f["observations"][()])
        timeouts = np.asarray(f["timeouts"][()]).astype(bool) if "timeouts" in f else None
    at_clip = np.any(np.abs(obs[:, task.nq - 1:]) >= CLIP, axis=1)
    mark = _synthetic_mark(path)
    return dict(schema=SCHEMA, kind="keys", dataset=task.dataset, keys=keys,
                has_infos_qpos="infos/qpos" in keys, has_infos_qvel="infos/qvel" in keys,
                has_next_observations="next_observations" in keys, raw_rows=int(len(obs)),
                converted_rows=None if timeouts is None else int(np.count_nonzero(~timeouts[:-1])),
                rows_at_velocity_clip=int(at_clip.sum()), share_at_velocity_clip=float(at_clip.mean()),
                synthetic=mark is not None, synthetic_mark=None if mark is None else str(mark))


def run_gate(task, path, *, rows, seed, source="auto", offset=0, tolerances=None, physics=None):
    """One-step gate on a seeded sample of converted rows (all rows if rows >= the converted count)."""
    own = physics is None
    physics = Physics(task) if own else physics
    try:
        raw = load_raw(physics, path)
        converted, raw_rows, _ = converted_rows(raw)
        n = len(raw_rows)
        if offset > 0:  # the negative control needs a next raw row
            n = int(np.searchsorted(raw_rows, len(raw["observations"]) - offset))
        g = np.random.default_rng(seed)
        ids = np.arange(n) if rows >= n else np.sort(g.choice(n, size=rows, replace=False))
        logged, states = gate_inputs(physics.task, raw, raw_rows, converted, ids, source=source, offset=offset)
        result = one_step_gate(physics, logged, states, tolerances)
        result.update(dataset=physics.task.dataset, sample_seed=seed, rows_sampled=int(len(ids)),
                      converted_total=int(len(raw_rows)), offset=int(offset), physics=physics.identity(),
                      has_infos=has_infos(raw), has_next_observations="next_observations" in raw)
        result["_arrays"]["converted_ids"] = ids[result["_arrays"]["kept_index"]]
        return result
    finally:
        if own:
            physics.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage 0 checks of the truth instrument (no training).")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("keys", "gate"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True, help="hopper | walker2d (or the D4RL name)")
        where = p.add_mutually_exclusive_group()
        where.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="pinned cache directory")
        where.add_argument("--file", type=Path, help="an explicit (unpinned) D4RL-format file, e.g. synthetic")
        p.add_argument("--output", required=True, type=Path, help="strict JSON result (must not exist)")
        if name == "gate":
            p.add_argument("--rows", type=int, default=1000)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--source", choices=("auto", "infos", "rebuilt"), default="auto")
            p.add_argument("--offset", type=int, default=0, help="negative control: misalign states by this many rows")
            for tol, default in DEFAULT_TOLERANCES.items():
                p.add_argument("--" + tol.replace("_", "-"), type=float, default=default)
    args = parser.parse_args(argv)
    path, identity = _resolve_file(args)
    if args.command == "keys":
        result = dict(describe_file(args.dataset, path), file=identity)
        arrays = None
    else:
        tolerances = {k: getattr(args, k) for k in DEFAULT_TOLERANCES}
        result = run_gate(args.dataset, path, rows=args.rows, seed=args.seed, source=args.source,
                          offset=args.offset, tolerances=tolerances)
        arrays = result.pop("_arrays")
        result.update(file=identity, synthetic=_synthetic_mark(path) is not None)
    result["utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as handle:
        handle.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    if arrays is not None:
        with args.output.with_suffix(".npz").open("xb") as handle:
            np.savez(handle, **arrays)
    summary = {k: result[k] for k in ("passed", "passes", "restore_miss", "next_obs_miss", "reward_miss",
                                      "terminal_agreement", "rows_checked", "rows_dropped_at_clip") if k in result}
    print(json.dumps(summary or {k: result[k] for k in ("has_infos_qpos", "has_infos_qvel", "raw_rows",
                                                          "rows_at_velocity_clip")}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
