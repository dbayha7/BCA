"""Synthetic D4RL-format HDF5 files made in the real Hopper/Walker2d simulator, for testing only.

The D4RL downloads (rail.eecs.berkeley.edu, huggingface.co) are blocked in the build container, so the truth
instrument is tested on files written here. They are NOT D4RL data, and no number computed on them is a
finding. Every file carries the root attribute bca_synthetic with that statement (is_synthetic reads it).

How a file is made (generate). The real gym environment of the task (gym.make('hopper-medium-v2') or
gym.make('walker2d-medium-replay-v2'): the audited TimeLimit / OrderEnforcing / NormalizedBoxEnv /
Offline*Env chain) is stepped by a simple seeded stochastic policy, a = tanh(W obs + noise * z) with a fixed
random W (entries N(0, scale^2 / obs_dim)) and z ~ N(0, I), cast to float32. The collection loop follows
D4RL's v2 locomotion collector (as recalled; the real files are the authority, Stage 0 lists their keys):
  qpos, qvel = sim.data.qpos, sim.data.qvel        recorded BEFORE env.step, aligned with observations[i]
  next_obs, r, done, _ = env.step(a)
  t += 1; timeout = t == max_path; terminal = done and not timeout
  rows appended per complete episode; reset after a terminal or a timeout.
Stored dtypes: observations, next_observations, actions, rewards, infos/qpos, infos/qvel float32; terminals,
timeouts bool. write_hdf5(..., with_infos=False) writes the same arrays without infos/*, which exercises the
observation-rebuild fallback of state_restore.py.

The stored state is float32, as in D4RL, so a restored state differs from the simulated one by float32
rounding. The one-step gate therefore reproduces the next observation to about 1e-6, not bit for bit.
"""

from pathlib import Path

import numpy as np

from experiments.truth.state_restore import Physics, require, task_spec

MARK = ("NOT D4RL DATA: synthetic rows from experiments/truth/synthetic.py (seeded linear-tanh policy in the "
        "real simulator), for tests only; no number computed on this file is a finding.")


def generate(physics, n_rows, seed, *, max_path=25, scale=0.3, noise=0.2):
    """Raw D4RL-format arrays: complete episodes until at least n_rows rows."""
    require(type(n_rows) is int and n_rows >= 2 and type(seed) is int and seed >= 0, "explicit sizes and seed")
    require(type(max_path) is int and 1 <= max_path <= physics.env._max_episode_steps, "invalid max_path")
    task = physics.task
    g = np.random.default_rng(seed)
    w = g.normal(scale=scale / np.sqrt(task.obs_dim), size=(task.action_dim, task.obs_dim))
    env, sim = physics.env, physics.base.sim
    keys = ("observations", "actions", "next_observations", "rewards", "terminals", "timeouts", "qpos", "qvel")
    data = {k: [] for k in keys}
    episode = 0
    while len(data["rewards"]) < n_rows:
        env.seed(int(seed * 100003 + episode) % (2**31 - 1))
        s = env.reset()
        traj = {k: [] for k in keys}
        t = 0
        while True:
            a = np.tanh(w @ s + noise * g.normal(size=task.action_dim)).astype(np.float32)
            qpos, qvel = sim.data.qpos.ravel().copy(), sim.data.qvel.ravel().copy()
            ns, r, done, _ = env.step(a)
            t += 1
            timeout = t == max_path
            terminal = bool(done) and not timeout
            for k, v in zip(keys, (s, a, ns, r, terminal, timeout, qpos, qvel)):
                traj[k].append(v)
            s = ns
            if terminal or timeout:
                break
            require(not done, "time limit reached before max_path")
        for k in keys:
            data[k].extend(traj[k])
        episode += 1
    raw = dict(observations=np.asarray(data["observations"], np.float32),
               actions=np.asarray(data["actions"], np.float32),
               next_observations=np.asarray(data["next_observations"], np.float32),
               rewards=np.asarray(data["rewards"], np.float32),
               terminals=np.asarray(data["terminals"], bool),
               timeouts=np.asarray(data["timeouts"], bool))
    raw["infos/qpos"] = np.asarray(data["qpos"], np.float32)
    raw["infos/qvel"] = np.asarray(data["qvel"], np.float32)
    return raw, dict(episodes=episode, rows=int(len(raw["rewards"])), seed=seed, max_path=max_path, scale=scale,
                     noise=noise, terminals=int(raw["terminals"].sum()), timeouts=int(raw["timeouts"].sum()))


def write_hdf5(path, raw, *, with_infos=True, description=MARK):
    import h5py

    path = Path(path)
    with h5py.File(path, "x") as f:
        f.attrs["bca_synthetic"] = description
        for k, v in raw.items():
            if k.startswith("infos/") and not with_infos:
                continue
            f.create_dataset(k, data=v)
    return path


def is_synthetic(path):
    import h5py

    with h5py.File(path, "r") as f:
        return "bca_synthetic" in f.attrs


def write_synthetic(directory, dataset, *, n_rows=3000, seed=0, max_path=25, physics=None):
    """Write <task>_synthetic.hdf5 (with infos/*) and <task>_synthetic_noinfos.hdf5 into directory."""
    task = task_spec(dataset)
    own = physics is None
    physics = Physics(task.dataset) if own else physics
    try:
        raw, meta = generate(physics, n_rows, seed, max_path=max_path)
    finally:
        if own:
            physics.close()
    directory = Path(directory)
    with_infos = write_hdf5(directory / f"{task.name}_synthetic.hdf5", raw, with_infos=True)
    without = write_hdf5(directory / f"{task.name}_synthetic_noinfos.hdf5", raw, with_infos=False)
    return dict(meta, dataset=task.dataset, with_infos=str(with_infos), without_infos=str(without))
