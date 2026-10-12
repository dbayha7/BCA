"""The recorded log: generation through env.step, D4RL field names, HDF5 + JSON manifest, splits, calibration bank and
the replay check.

Generation (generate). One data seed `rep` gives 1,000 episodes of 100 steps from the recorded start law:
  starts and wobble noise   np.random.default_rng([20261012, rep, 7]): draw_starts(1000, ...) then N(0, I) noise
                            (1000, 100, 4)
  coin of the full-throttle command   np.random.default_rng([20261012, rep, 8]).uniform((1000, 100))
Each start goes into a registered environment (gym.make, the world's 'recorded' id) through
reset(options={"state": start}); every step the recording controller's command is rounded to float32 first and then
passed to env.step, so the stored action is exactly the applied action. The 1,000 environments are stepped in lockstep
(one batched controller call per time step).

Splits, by episode (sizes (500, 200, 300)): episodes 0-499 train (50,000 rows), 500-699 calibration (the calibration
bank takes 5 rows from each of these 200 episodes through calibration/bank.stratified_bank with
default_rng([20261012, rep, 10]): 1,000 rows), 700-999 evaluation (held out, never trained on).

Arrays (D4RL / Minari names):
  observations (N, 13) float32, actions (N, 4) float32, rewards (N,) float32, next_observations (N, 13) float32,
  terminals (N,) bool (all False: episodes only end by the time limit), timeouts (N,) bool (True at t = 99, so TD3+BC
  bootstraps through the time limit);
  infos/state and infos/next_state (N, 13) float64 (the exact states), infos/episode, infos/t, infos/split
  (0 train, 1 calibration, 2 evaluation);
  infos/full_throttle_command (N,) bool: whether the recording controller gave the full-throttle command. It is a
  diagnostic only. Log hides it from ordinary item access (log[key] refuses it); only log.diagnostic(key) returns it
  and counts the read, so a test can prove that no learner, removal rule or scorer reads it.

HDF5 (write_hdf5 / read_hdf5): every array under its name, written without timestamps so regeneration gives the same
bytes; a JSON manifest beside it holds the SHA-256 of the file and of every array, the generator settings and the
SHA-256 of the experiments/shift code. Reading verifies every hash and refuses on any mismatch.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

import gymnasium as gym

from calibration import bank as CB
from experiments.shift import ROOT
from experiments.shift import recorder as RC
from experiments.shift import thruster_chain as TC

SCHEMA = "bca-shift-log-v1"
SIZES = (500, 200, 300)
SPLITS = ("train", "calibration", "evaluation")
TAG_STARTS, TAG_COIN, TAG_BANK = 7, 8, 10
BANK_ROWS_PER_EPISODE = 5
D4RL_KEYS = ("observations", "actions", "rewards", "next_observations", "terminals", "timeouts")
INFO_KEYS = ("infos/state", "infos/next_state", "infos/episode", "infos/t", "infos/split")
DIAGNOSTIC_KEYS = ("infos/full_throttle_command",)
ALL_KEYS = D4RL_KEYS + INFO_KEYS + DIAGNOSTIC_KEYS
CODE_FILES = ("__init__.py", "thruster_chain.py", "recorder.py", "dataset.py")


def sha256_array(x):
    x = np.ascontiguousarray(np.asarray(x))
    h = hashlib.sha256()
    h.update(str(x.dtype).encode() + b"|" + str(x.shape).encode() + b"|")
    h.update(x.tobytes())
    return h.hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def code_hashes(files=CODE_FILES):
    here = Path(__file__).resolve().parent
    return {name: sha256_file(here / name) for name in files}


class Log(Mapping):
    """A generated log: D4RL arrays, infos and meta. log[key] gives every array except the recording controller's
    mode flag (DIAGNOSTIC_KEYS), which only log.diagnostic(key) returns; diagnostic_reads counts those reads."""

    def __init__(self, arrays, meta):
        missing = set(ALL_KEYS) - set(arrays)
        if missing:
            raise ValueError(f"log arrays missing: {sorted(missing)}")
        self._arrays = {k: np.asarray(arrays[k]) for k in ALL_KEYS}
        for v in self._arrays.values():
            v.setflags(write=False)
        self.meta = dict(meta)
        self.diagnostic_reads = 0

    def __getitem__(self, key):
        if key in DIAGNOSTIC_KEYS:
            raise KeyError(f"{key} is the recording controller's own mode flag, a diagnostic only; no learner, removal "
                           "rule or scorer may read it (use Log.diagnostic for world-card diagnostics)")
        return self._arrays[key]

    def __iter__(self):
        return (k for k in self._arrays if k not in DIAGNOSTIC_KEYS)

    def __len__(self):
        return len(self._arrays) - len(DIAGNOSTIC_KEYS)

    def diagnostic(self, key):
        if key not in DIAGNOSTIC_KEYS:
            raise KeyError(f"{key} is not a diagnostic array")
        self.diagnostic_reads += 1
        return self._arrays[key]

    @property
    def n_rows(self):
        return int(len(self._arrays["actions"]))

    @property
    def physics(self):
        return TC.Physics(**self.meta["physics"])

    @property
    def spec(self):
        return RC.RecorderSpec(**self.meta["recorder"])

    def split_rows(self, split):
        """Row indices of a split ('train', 'calibration', 'evaluation'), ascending."""
        return np.flatnonzero(self._arrays["infos/split"] == SPLITS.index(split))

    @property
    def bank_rows(self):
        """The calibration bank's rows (before any removal), ascending."""
        return np.asarray(self.meta["bank_rows"], np.int64)

    def array_hashes(self):
        return {k: sha256_array(v) for k, v in self._arrays.items()}


def _split_codes(sizes, length):
    codes = np.repeat(np.arange(len(sizes), dtype=np.int8), np.asarray(sizes) * length)
    return codes


def draw_streams(rep, n_episodes, length=TC.STALL.length):
    """The generator's random streams for data seed rep: (starts, noise, coin) (module docstring)."""
    rng = np.random.default_rng([TC.DATA_SEED, int(rep), TAG_STARTS])
    starts = TC.draw_starts(n_episodes, rng, TC.START_LAWS["recorded"])
    noise = rng.standard_normal((n_episodes, length, TC.DA))
    coin = np.random.default_rng([TC.DATA_SEED, int(rep), TAG_COIN]).uniform(size=(n_episodes, length))
    return starts, noise, coin


def simulate(recorder, physics, starts, noise, coin, length=TC.STALL.length):
    """The same episodes as generate() without the environment objects, by dynamics() directly (vectorised; used for
    the trigger bisection). Commands are rounded to float32 like env.step. Returns states (E, L + 1, 13), commands
    (E, L, 4) float32, rewards (E, L), region and full-throttle flags (E, L)."""
    x = np.asarray(starts, np.float64).copy()
    S, C, R, G, P = [x], [], [], [], []
    for t in range(length):
        cmd, region, press = recorder.parts(x, noise[:, t], coin[:, t])
        a32 = cmd.astype(np.float32)
        x, r = TC.dynamics(x, a32.astype(np.float64), physics)
        S.append(x), C.append(a32), R.append(r), G.append(region), P.append(press)
    st = lambda v: np.stack(v, 1)
    return dict(states=st(S), commands=st(C), rewards=st(R), region=st(G), press=st(P))


def generate(rep, world="stall", spec=RC.DEFAULT_SPEC, sizes=SIZES, stall_slope=1.5):
    """The log of data seed rep in a world ('stall' or 'no_stall'), stepped through env.step (module docstring)."""
    physics = TC.world_physics(world, stall_slope)
    E, L = int(sum(sizes)), physics.length
    starts, noise, coin = draw_streams(rep, E, L)
    recorder = RC.Recorder(spec, physics)
    eid = TC.env_id(world, "recorded")
    envs = [gym.make(eid, stall_slope=physics.stall_slope) for _ in range(E)]
    X = np.empty((E, L + 1, TC.DX))
    obs = np.empty((E, L + 1, TC.DX), np.float32)
    act = np.empty((E, L, TC.DA), np.float32)
    rew = np.empty((E, L), np.float32)
    press = np.empty((E, L), bool)
    term = np.empty((E, L), bool)
    trunc = np.empty((E, L), bool)
    for e, env in enumerate(envs):
        obs[e, 0], info = env.reset(options={"state": starts[e]})
        X[e, 0] = info["state"]
    for t in range(L):
        cmd, _, pr = recorder.parts(X[:, t], noise[:, t], coin[:, t])
        act[:, t] = cmd.astype(np.float32)
        press[:, t] = pr
        for e, env in enumerate(envs):
            o, r, te, tr, info = env.step(act[e, t])
            obs[e, t + 1], X[e, t + 1], rew[e, t], term[e, t], trunc[e, t] = o, info["state"], r, te, tr
    for env in envs:
        env.close()
    if term.any() or not trunc[:, -1].all() or trunc[:, :-1].any():
        raise AssertionError("episodes must end by truncation at t = 100 only")
    flat = lambda v: v.reshape(E * L, *v.shape[2:])
    arrays = {
        "observations": flat(obs[:, :-1]), "actions": flat(act), "rewards": flat(rew),
        "next_observations": flat(obs[:, 1:]), "terminals": flat(term), "timeouts": flat(trunc),
        "infos/state": flat(X[:, :-1]), "infos/next_state": flat(X[:, 1:]),
        "infos/episode": np.repeat(np.arange(E, dtype=np.int32), L), "infos/t": np.tile(np.arange(L, dtype=np.int32), E),
        "infos/split": _split_codes(sizes, L), "infos/full_throttle_command": flat(press),
    }
    n_cal = int(sizes[1])
    episodes, offsets = CB.stratified_bank(np.full(n_cal, L), n_cal * BANK_ROWS_PER_EPISODE, BANK_ROWS_PER_EPISODE,
                                           np.random.default_rng([TC.DATA_SEED, int(rep), TAG_BANK]))
    if not np.array_equal(np.sort(episodes), np.arange(n_cal)):
        raise AssertionError("the calibration bank must take rows from every calibration episode")
    first = int(sizes[0])
    bank_rows = np.sort(np.concatenate([(first + e) * L + o for e, o in zip(episodes, offsets)]))
    meta = dict(schema=SCHEMA, rep=int(rep), world=world, env_id=eid, physics=physics.as_dict(),
                recorder=spec.as_dict(), sizes=[int(s) for s in sizes], length=L, data_seed=TC.DATA_SEED,
                stream_tags=dict(starts_and_noise=TAG_STARTS, coin=TAG_COIN, calibration_bank=TAG_BANK),
                bank_rows=bank_rows.tolist(), bank_rows_per_episode=BANK_ROWS_PER_EPISODE,
                gymnasium_version=gym.__version__)
    return Log(arrays, meta)


def d4rl_dict(log, rows):
    """The six D4RL arrays at the given rows (copies, in row order). A removed row is simply absent: every kept row
    keeps its own true next observation, so no transition is ever spliced."""
    rows = np.asarray(rows, np.int64)
    return {k: np.array(log[k][rows]) for k in D4RL_KEYS}


# ------------------------------------------------------------------------------------------------ HDF5

def write_hdf5(log, path):
    """Write the log's arrays to path (HDF5, no timestamps) and the manifest to path + '.manifest.json'. Returns
    the manifest."""
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w", libver="earliest") as f:
        for key in ALL_KEYS:
            arr = log._arrays[key]
            f.create_dataset(key, data=arr, track_times=False)
        f.attrs["schema"] = SCHEMA
        f.attrs["meta"] = json.dumps(log.meta, sort_keys=True)
    manifest = dict(schema=SCHEMA, file=path.name, file_sha256=sha256_file(path),
                    arrays={k: dict(sha256=sha256_array(v), dtype=str(v.dtype), shape=list(v.shape))
                            for k, v in log._arrays.items()},
                    meta=log.meta, code_sha256=code_hashes())
    manifest_path = Path(str(path) + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2))
    return manifest


def read_hdf5(path, manifest_path=None, expected_file_sha256=None):
    """Read a log written by write_hdf5, verifying the file and every array against the manifest (refuses on any
    mismatch)."""
    import h5py

    path = Path(path)
    manifest_path = Path(manifest_path or str(path) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    file_sha = sha256_file(path)
    if file_sha != manifest["file_sha256"] or (expected_file_sha256 and file_sha != expected_file_sha256):
        raise ValueError(f"{path}: file SHA-256 {file_sha} does not match the manifest")
    with h5py.File(path, "r") as f:
        arrays = {k: f[k][()] for k in ALL_KEYS}
        meta = json.loads(f.attrs["meta"])
    for k, v in arrays.items():
        if sha256_array(v) != manifest["arrays"][k]["sha256"]:
            raise ValueError(f"{path}: array {k} does not match its manifest SHA-256")
    if meta != manifest["meta"]:
        raise ValueError(f"{path}: meta differs from the manifest")
    return Log(arrays, meta)


# ------------------------------------------------------------------------------------------------ replay

def replay(log, rows=None, chunk_report=5):
    """Replay rows through a registered environment: reset(options={"state": infos/state[i]}) then
    step(actions[i]); every row must give, bit for bit, next_observations[i], infos/next_state[i] and the float32
    reward. Also checks observations[i] == float32(infos/state[i]). Returns counts and the first few mismatches."""
    physics = log.physics
    env = gym.make(log.meta["env_id"], stall_slope=physics.stall_slope)
    rows = np.arange(log.n_rows) if rows is None else np.asarray(rows, np.int64)
    state, nstate = log["infos/state"], log["infos/next_state"]
    obs, nobs, act, rew = log["observations"], log["next_observations"], log["actions"], log["rewards"]
    bad = dict(observation=0, next_observation=0, next_state=0, reward=0)
    first, mismatched = [], 0
    for i in rows:
        o0, info0 = env.reset(options={"state": state[i]})
        o1, r, _, _, info = env.step(act[i])
        checks = dict(observation=np.array_equal(o0, obs[i]) and np.array_equal(info0["state"], state[i]),
                      next_observation=np.array_equal(o1, nobs[i]), next_state=np.array_equal(info["state"], nstate[i]),
                      reward=np.float32(r) == rew[i])
        failed = [k for k, ok in checks.items() if not ok]
        for k in failed:
            bad[k] += 1
        if failed:
            mismatched += 1
            if len(first) < chunk_report:
                first.append(dict(row=int(i), failed=failed))
    env.close()
    return dict(rows=int(len(rows)), mismatches=int(mismatched), by_field=bad, first=first,
                rule="reset(options={'state'}) + step(action) must reproduce every row bit for bit")


# ------------------------------------------------------------------------------------------------ CLI

def refuse_protected(out):
    """Refuse output directories under runs/glq (the A1 gate's directory) or the repo's experiments tree."""
    out = Path(out).resolve()
    for protected in (ROOT / "runs" / "glq", ROOT / "experiments", ROOT / "runs" / "shift"):
        if out == protected or protected in out.parents:
            raise SystemExit(f"refused: {out} is inside {protected}; write verification output to a temporary "
                             "directory (runs/shift is written only by the pre-registered pipeline)")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate ThrusterChain logs (HDF5 + manifest) into a directory.")
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--world", choices=sorted(TC.WORLDS), default="stall")
    p.add_argument("--replay", action="store_true", help="replay every row through env.step after writing")
    a = p.parse_args(argv)
    out = refuse_protected(a.out)
    for rep in a.seeds:
        log = generate(rep, a.world)
        path = out / f"thrusterchain_{a.world}_seed{rep}.hdf5"
        manifest = write_hdf5(log, path)
        line = f"{path.name}: {log.n_rows} rows, file sha256 {manifest['file_sha256'][:16]}"
        if a.replay:
            rp = replay(log)
            line += f"; replay {rp['rows']} rows, {rp['mismatches']} mismatches"
        print(line, flush=True)


if __name__ == "__main__":
    sys.exit(main())
