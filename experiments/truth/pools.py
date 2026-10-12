"""Pools L, Pi and T of the D4RL-RT truth instrument: the frozen host, its data split, kNN thinness, and the
(state, action) points whose true Q^pi the labeller measures.

Third piece of the instrument (experiments/truth/README.md). state_restore.py puts MuJoCo into a logged state;
labeller.py rolls the frozen policy out from it. This module decides WHICH states and actions are labelled,
binds the frozen TD3+BC host that defines pi and Q1, and computes every model-side quantity the acceptance
tests need (critic values, thinness counts, residuals), so that scoring (acceptance.py) is pure NumPy on files.

Host binding (restore_host). A host is a finished train.py run of td3_bc --method host: a directory with
resolved.json, preparation.json, result.json (optional) and checkpoint_<step>.msgpack. It is restored exactly as
experiments/wbcp/freeze_scores.py and experiments/signal/frozen_signals.restore_pool restore runs:
  1. resolved.json is the run's row. Rows from the archived standard study lack reservation.rows_per_episode;
     None is set, as frozen_signals does (the whole-episode split those runs used).
  2. The cached HDF5 is <data_dir>/<row.cache.filename>, sha256-checked against row.cache.sha256 (the pins of
     configs/td3_bc.yaml); a missing or different file stops with a message naming it.
  3. freeze_scores.prepare (typed(row), then train.prepare -> runtime/td3_bc.prepare) rebuilds the preparation;
     its run_input_hashes must equal preparation.json's for every key this branch computes. Extra recorded keys
     are allowed and listed (the archived stack also hashed an empty 'reference' pool).
  4. The checkpoint's sha256 must equal result.json's record, the --host/prereg pin and, when given, the accepted
     audit (outputs/standard_bca/td3_bc/<task>/host/s<seed>/verified-*/audit.json, DEFAULT_AUDITS). It is
     restored into the algorithms.td3_bc_bca.initialize template and checked leaf for leaf
     (freeze_scores.same_leaves). If the archived state does not fit this branch's template, only the native
     AgentTrainState (actor, actor_target, critic, critic_target with optimizer states) is restored, again leaf
     for leaf; the record says which path ran.
  5. Gate: (float32(obs) - mean) / std of the converted training rows must equal prepared.training.obs bit for bit.
The host's split is its own preparation's: training_converted_ids, withheld_converted_ids and
heldout_converted_ids (the bank: K rows per withheld episode on this branch; for the archived 202609171 hosts the
whole 1,103 / 1,149 withheld rows). Episodes are runtime/td3_bc._convert's terminal/timeout episodes.

Standardization (standardize): float32 (obs - obs_mean) / obs_std, the host's training-row mean and SD + 0.001,
as runtime/common.normalize_states. The policy is labeller.FrozenPolicy.from_host_state (online actor).

Thinness (ThinnessIndex), the design's rule. A point is x = (standardized s, a) in float64 (14-D Hopper, 23-D
Walker2d). A scipy cKDTree holds the host's training rows. c(x) = the number of training rows from OTHER
episodes within radius r (inclusive), d_k(x) = the distance to the k-th nearest other-episode training row
(k = 10). r = the median d_k of the host's withheld rows (whole withheld episodes, none of them in the tree), so
a typical withheld row has c of about k. Thin: c <= 2; dense: c >= 10 (acceptance.py reads the cut-offs from the
pre-registration). Same-episode rows are removed exactly: the ball count minus a brute-force count inside the
query's own episode; k-NN queries widen k until k other-episode rows are found. Host-policy states (pool Pi)
belong to no logged episode, so nothing is excluded for them.

Pools. Every pool state is restorable (state_restore.logged_states: infos, or rebuilt from the observation;
rows at the +-10 velocity clip cannot be rebuilt and are dropped and counted).
  L   the host's bank rows (all of them) plus `training_rows` seeded training rows. Logged (s, a, r, s', done).
  Pi  host-policy states: `episodes` episodes of pi from a fixed reset bank (env.seed(base + i); env.reset()),
      rolled out in lockstep on bare MjSims (state_restore.Physics.step, bitwise env.step); the state before
      every `every`-th step is recorded, then a seeded subsample of `states` states is kept (all when fewer). The
      episodes' returns are the host's return on that reset bank (R6a).
  T   `rows` seeded training rows (the R4 oracle weights and the shared extraction pool).
Actions per state (slots). L and Pi: pi(s); 'data' = the logged action (L) or the action of the nearest training
row in standardized state space (Pi; any training row is from another episode); central differences
clip(pi(s) +- delta e_j, +-1) (labeller.central_difference_actions); 'cg' = clip(pi(s) + step g / |g|, +-1),
g = grad_a Q1(s, pi(s)) (critic-guided; g = 0 leaves pi(s) and is counted). 3 + 2 d_a slots: 9 (Hopper), 15
(Walker2d). T: 'data' and 'pi'. For every point: Q1 and Q2 (twin critic, float32). L and Pi also get c and d_k.

Model-side extras (pools stage): on L, the noiseless one-step residual b = r + gamma (1 - d) Q1(s', pi(s')) -
Q1(s, a) with the online nets, and `draws` TD3 targets t_j = r + gamma (1 - d) min(Q1', Q2')(s', clip(pi'(s')
+ clip(0.2 z_j, +-0.5), +-1)) with the target nets (CriticProbe.target_draws; equal, to float32 rounding, to
algorithms.td3_bc_bca.native_target for the same noise, test_pools.py). For R6: on up to `action_rows` seeded
withheld rows, |pi(s) - a| and |a - a_nn| (a_nn = the action of the nearest training state), and that many
withheld observations for the two-sample classifier.

Files (save_pool / load_pool): <name>.npz and <name>.json (strict JSON: counts, rules, radius, the npz sha256,
the host identity, the pre-registration sha256, and SYNTHETIC marks when the data file is synthetic). Items for
labeller.run_labels come from pool_items: item_id = state * n_slots + slot.

write_synthetic_run writes a train.py-format run directory around a RANDOMLY INITIALISED host of the exact
architecture on a synthetic file (tests only; refuses real data). It goes through the same resolve -> typed ->
train.prepare path, so restore_host is tested end to end here. Nothing in this module trains anything; no BCA
band or arm is involved.
"""

import contextlib
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.truth import labeller as LB  # noqa: E402
from experiments.truth.state_restore import (  # noqa: E402
    DEFAULT_DATA, array_sha256, converted_rows, file_sha256, logged_states, observation, require, task_spec)

POOL_SCHEMA = "bca-truth-pool-v1"
HOST_SCHEMA = "bca-truth-host-v1"
RAW_KEYS = ("observations", "actions", "rewards", "terminals", "timeouts")  # what train.prepare reads
STATE_KEYS = ("infos/qpos", "infos/qvel", "next_observations")  # read too when present (restore, gate)
TD3_CONFIG = ROOT / "configs" / "td3_bc.yaml"
DEFAULT_AUDITS = {  # accepted audits of the existing 1M hosts (their checkpoints live on the user's machine)
    ("hopper", 202609171): ROOT / "outputs/standard_bca/td3_bc/hopper/host/s202609171/verified-v4/audit.json",
    ("walker2d", 202609171): ROOT / "outputs/standard_bca/td3_bc/walker2d/host/s202609171/verified-v1/audit.json",
}
KEY_TARGET_NOISE = 1414416705  # numpy stream keys
KEY_POOL = 1347374156
SYNTHETIC_RESERVATION = dict(target_size=48, max_fraction=0.4)
SYNTHETIC_HOST_NOTE = ("NOT A TRAINED HOST: randomly initialised TD3+BC networks of the host's exact architecture "
                       "around a synthetic file, written by experiments/truth/pools.write_synthetic_run for tests; "
                       "no number computed with it is a finding.")


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def is_synthetic_file(path):
    import h5py

    with h5py.File(path, "r") as f:
        return "bca_synthetic" in f.attrs


def read_raw(path):
    """The raw arrays of a D4RL-format HDF5: RAW_KEYS (as train.prepare reads them) and STATE_KEYS when present."""
    import h5py

    path = Path(path)
    require(path.is_file(), f"HDF5 file {path} is missing")
    with h5py.File(path, "r") as f:
        missing = [k for k in RAW_KEYS if k not in f]
        require(not missing, f"{path} lacks {missing}")
        return {k: f[k][()] for k in RAW_KEYS + STATE_KEYS if k in f}


def cache_path(row, data_dir):
    """<data_dir>/<row cache filename>, sha256-checked against the row's pin, with errors that name the file."""
    data_dir = Path(data_dir).expanduser().resolve()
    entry = row["cache"]
    path = data_dir / entry["filename"]
    if not path.is_file():
        raise FileNotFoundError(
            f"D4RL cache {path} is missing. The run pins {entry['filename']} with sha256 {entry['sha256']} "
            f"(configs/td3_bc.yaml). Download it with d4rl into ~/.d4rl/datasets or pass --data-dir.")
    digest = file_sha256(path)
    require(digest == entry["sha256"], f"{path} has sha256 {digest}, the run pins {entry['sha256']}")
    return path, digest


# ---------------------------------------------------------------- the host

@dataclass
class Host:
    """A frozen TD3+BC host with its data split. native: algorithms.td3_bc.AgentTrainState."""
    task: Any
    seed: int
    args: Any
    actor: Any
    critic: Any
    native: Any
    raw: dict
    converted: dict
    rows: np.ndarray
    episodes: np.ndarray
    training_ids: np.ndarray
    withheld_ids: np.ndarray
    bank_ids: np.ndarray
    obs_mean: np.ndarray
    obs_std: np.ndarray
    max_action: float
    data_file: Path
    identity: dict
    synthetic: bool
    prepared: Any = field(default=None, repr=False)

    def standardized(self, obs):
        return ((np.asarray(obs, np.float32) - self.obs_mean) / self.obs_std).astype(np.float32)

    def policy(self, batch):
        return LB.FrozenPolicy.from_host_state((self.actor, self.critic), self.native, self.obs_mean, self.obs_std,
                                               batch=batch)


def verify_preparation(run_dir, prepared):
    """preparation.json's run_input_hashes must equal the new preparation's for every key this branch computes."""
    path = Path(run_dir) / "preparation.json"
    require(path.is_file(), f"{path} is missing (train.py writes it; the host's preparation cannot be verified)")
    saved = json.loads(path.read_text(encoding="utf8"))["metadata"]["run_input_hashes"]
    current = prepared.metadata["run_input_hashes"]
    missing = sorted(k for k in current if k not in saved)
    differ = sorted(k for k in current if k in saved and saved[k] != current[k])
    require(not missing and not differ,
            f"the rebuilt preparation differs from {path} (missing {missing}, different {differ}): the checkpoint "
            "was trained on other data, another split or another normalization")
    return dict(record=str(path), sha256=file_sha256(path), keys_checked=sorted(current),
                extra_recorded_keys=sorted(set(saved) - set(current)))


def _flat_leaves(tree):
    import jax

    return jax.tree_util.tree_flatten_with_path(tree)[0]


def _same(flat_a, flat_b):
    return len(flat_a) == len(flat_b) and all(
        pa == pb and np.asarray(a).dtype == np.asarray(b).dtype and np.shape(a) == np.shape(b)
        and np.asarray(a).tobytes() == np.asarray(b).tobytes() for (pa, a), (pb, b) in zip(flat_a, flat_b))


def restore_native(payload, template, rng, step):
    """(native AgentTrainState, how): the full template restore when it is leaf-for-leaf exact, else the native
    subtree alone (also leaf for leaf)."""
    import jax.numpy as jnp
    from flax import serialization

    from experiments.wbcp import freeze_scores as FS

    reason = None
    try:
        restored = serialization.from_bytes({"state": template, "training_rng": rng, "step": jnp.int32(0)}, payload)
        if int(restored["step"]) == step and FS.same_leaves(payload, restored):
            return restored["state"].native, "full algorithms.td3_bc_bca.initialize template, leaf-for-leaf identical"
        reason = "the full restore was not leaf-for-leaf identical"
    except (ValueError, KeyError, TypeError) as error:
        reason = f"{type(error).__name__}: {error}"
    decoded = serialization.msgpack_restore(payload)
    require(isinstance(decoded, dict) and isinstance(decoded.get("state"), dict) and "native" in decoded["state"],
            "the checkpoint has no state.native subtree")
    require(int(decoded["step"]) == step, f"the checkpoint records step {int(decoded['step'])}, not {step}")
    native = serialization.from_state_dict(template.native, decoded["state"]["native"])
    require(_same(_flat_leaves(decoded["state"]["native"]), _flat_leaves(serialization.to_state_dict(native))),
            "the native subtree did not restore leaf for leaf")
    return native, f"native subtree only ({reason})"


def adapt_archived_row(row):
    """A resolved row from the archived standard study (bca-bayesmax-archive), made typeable on this branch.

    Those rows differ from this branch's in two inert ways (checked against experiments/standard_bca/manifest.json.gz
    for the TD3+BC hosts): protocol.reference is None (no reference pool; this branch's RunProtocol has no such
    field) and reservation.rows_per_episode is absent (they withheld whole episodes: None, as frozen_signals sets).
    Anything else, including a non-empty reference pool, is left alone and fails loudly in typed()."""
    import copy

    row = copy.deepcopy(row)
    changes = []
    protocol = row["protocol"]
    if "reference" in protocol:
        require(protocol["reference"] is None, "the archived row has a non-empty reference pool, which this branch "
                "cannot rebuild")
        del protocol["reference"]
        changes.append("dropped protocol.reference (None)")
    if "rows_per_episode" not in protocol["reservation"]:
        protocol["reservation"]["rows_per_episode"] = None
        changes.append("set reservation.rows_per_episode = None (whole withheld episodes)")
    return row, changes


def restore_host(run_dir, *, data_dir=DEFAULT_DATA, step=None, expected_sha256=None, audit="default",
                 gap_manifest=None):
    """Restore a finished td3_bc host run (train.py output) with its data split; see the module docstring.

    A gapped run (experiments/truth/gap_train.py; resolved.json carries action_gap) is restored with the same
    projected reservation it trained with (gap.projection), after its gap manifest (the recorded path, or
    gap_manifest) is verified against the recorded sha256 and the frozen sidecar; without it the restore refuses.
    Its episode ids become the ORIGINAL D4RL episode ids (never the flag fragments), for thinness and blocks."""
    import jax.numpy as jnp  # noqa: F401  (JAX must import before flax restores)

    from algorithms import td3_bc_bca as P
    from experiments.wbcp import freeze_scores as FS

    run_dir = Path(run_dir).expanduser().resolve()
    require(run_dir.is_dir(), f"host run directory {run_dir} is missing (a train.py output directory with "
            "resolved.json, preparation.json and checkpoint_<step>.msgpack)")
    resolved = run_dir / "resolved.json"
    require(resolved.is_file(), f"{resolved} is missing")
    row = json.loads(resolved.read_text(encoding="utf8"))
    require((row.get("algorithm"), row.get("host"), row.get("method")) == ("td3_bc", "td3", "host"),
            f"{run_dir} is not a td3_bc host run (algorithm/host/method {row.get('algorithm')}/{row.get('host')}/"
            f"{row.get('method')})")
    row, adapted = adapt_archived_row(row)
    task = task_spec(row["environment"])
    seed = int(row["protocol"]["seed"])
    step, ckpt, recorded = FS.locate_checkpoint(run_dir, step)
    payload = ckpt.read_bytes()
    digest = sha256_hex(payload)
    require(recorded is None or digest == recorded, f"{ckpt} sha256 {digest} differs from result.json {recorded}")
    require(expected_sha256 is None or digest == expected_sha256,
            f"{ckpt} sha256 {digest} differs from the pinned {expected_sha256}")
    audit_record = None
    if audit == "default":
        # The accepted audit is evidence about one exact run: it applies when resolved.json is that run's file.
        audit = DEFAULT_AUDITS.get((task.name, seed))
        if audit is not None:
            pinned = json.loads(Path(audit).read_text(encoding="utf8")).get("evidence_sha256", {}).get("resolved.json")
            if pinned != file_sha256(resolved):
                audit = None
    if audit is not None:
        audit = Path(audit)
        require(audit.is_file(), f"audit {audit} is missing")
        record = json.loads(audit.read_text(encoding="utf8"))
        require(record.get("accepted") is True and record.get("run_id") == row["run_id"],
                f"{audit} is not an accepted audit of {row['run_id']}")
        entries = [c for c in record.get("checkpoints", []) if c["step"] == step]
        require(len(entries) == 1 and entries[0]["sha256"] == digest,
                f"{ckpt} sha256 {digest} is not the audited step-{step} checkpoint")
        for name in ("resolved.json", "preparation.json"):
            pinned = record.get("evidence_sha256", {}).get(name)
            require(pinned is None or file_sha256(run_dir / name) == pinned, f"{run_dir / name} differs from {audit}")
        audit_record = dict(path=str(audit), sha256=file_sha256(audit))
    data_file, data_sha = cache_path(row, data_dir)
    gap_context, gap = contextlib.nullcontext(), None
    if row.get("action_gap") is not None:
        from experiments.truth import gap as GAP

        gap_context, gap = GAP.host_projection(row, gap_manifest)
    with gap_context:
        (_, args, spec, _), prepared = FS.prepare(row, data_file.parent)
    verification = verify_preparation(run_dir, prepared)
    config = spec.config()
    obs_dim, action_dim = prepared.training.obs.shape[1], prepared.training.action.shape[1]
    require((obs_dim, action_dim) == (task.obs_dim, task.action_dim), "prepared dimensions differ from the task")
    rng, template, models = P.initialize(args, config, obs_dim, action_dim, prepared.max_action)
    native, how = restore_native(payload, template, rng, step)
    raw = read_raw(data_file)
    converted, rows, episodes = converted_rows({k: raw[k] for k in RAW_KEYS})
    meta = prepared.metadata
    require(np.array_equal(rows, np.asarray(meta["dependency_maps"]["raw_current"], np.int64)),
            "converted raw rows differ from the preparation's dependency map")
    require(np.array_equal(episodes, np.asarray(meta["original_terminal_timeout_episode_ids"], np.int64)),
            "episode ids differ from the preparation")
    if gap is not None:
        episodes = gap.original_episodes(rows)  # original D4RL episodes, not the fragments the flags create
    mean = np.asarray(prepared.obs_mean, np.float32)
    std = np.asarray(prepared.obs_std, np.float32)
    training = np.asarray(meta["training_converted_ids"], np.int64)
    host = Host(task=task, seed=seed, args=args, actor=models[0], critic=models[1], native=native, raw=raw,
                converted=converted, rows=rows, episodes=episodes, training_ids=training,
                withheld_ids=np.asarray(meta["withheld_converted_ids"], np.int64),
                bank_ids=np.asarray(meta["heldout_converted_ids"], np.int64), obs_mean=mean, obs_std=std,
                max_action=float(prepared.max_action), data_file=data_file, identity={},
                synthetic=is_synthetic_file(data_file), prepared=prepared)
    require(np.array_equal(host.standardized(converted["observations"][training]),
                           np.asarray(prepared.training.obs)),
            "standardized training observations differ from the prepared pool")
    # Truth labels are raw env returns, so the critic must have been trained on raw rewards (no reward
    # normalization or transform; runtime/rewards.py transforms antmaze only): checked on the prepared pool itself.
    require(not args.normalize_reward and np.array_equal(np.asarray(prepared.training.reward, np.float32),
                                                         converted["rewards"][training].astype(np.float32)),
            "the critic's training rewards are not the raw logged rewards; Q1 and rollout returns would be in "
            "different units")
    host.identity = dict(
        schema=HOST_SCHEMA, run_dir=str(run_dir), run_id=row["run_id"], task=task.name, dataset=task.dataset,
        seed=seed, checkpoint=dict(path=str(ckpt), step=step, sha256=digest, recorded_sha256=recorded,
                                   restore=how), audit=audit_record, row_adaptations=adapted,
        data=dict(path=str(data_file), sha256=data_sha, synthetic=host.synthetic), preparation=verification,
        split=dict(training=int(len(training)), withheld=int(len(host.withheld_ids)), bank=int(len(host.bank_ids)),
                   rows_per_episode=row["protocol"]["reservation"]["rows_per_episode"],
                   target_size=row["protocol"]["reservation"]["target_size"]),
        normalization_sha256=array_sha256(np.concatenate([mean, std])),
        actor_params_sha256=_tree_sha(native.actor.params), critic_params_sha256=_tree_sha(native.critic.params),
        synthetic_host=(run_dir / "SYNTHETIC_HOST.txt").is_file(),
        action_gap=None if gap is None else dict(manifest=str(gap.path), manifest_sha256=gap.sha256,
                                                 centre=gap.data["gap"]["centre"], rho=gap.data["gap"]["rho"],
                                                 episodes="original D4RL episode ids (gap manifest)"))
    return host


def _tree_sha(tree):
    from runtime.td3_bc import _tree_hash

    return _tree_hash(tree)


def write_synthetic_run(run_dir, task, data_file, seed, *, reservation=None, step=0):
    """TEST ONLY: a train.py-format host run around randomly initialised networks on a synthetic file."""
    import jax.numpy as jnp
    from flax import serialization

    from algorithms import td3_bc_bca as P
    from experiments.wbcp import freeze_scores as FS
    from runtime.config import resolve
    from runtime.provenance import write

    task = task_spec(task)
    data_file = Path(data_file).resolve()
    require(is_synthetic_file(data_file), "write_synthetic_run refuses non-synthetic data")
    run_dir = Path(run_dir)
    row = resolve(TD3_CONFIG, "host", seed, run_dir.resolve(), task.name)
    row["cache"] = dict(filename=data_file.name, sha256=file_sha256(data_file))
    row["protocol"]["reservation"].update(reservation or SYNTHETIC_RESERVATION)
    row["synthetic_note"] = SYNTHETIC_HOST_NOTE
    (_, args, spec, _), prepared = FS.prepare(row, data_file.parent)
    rng, state, _ = P.initialize(args, spec.config(), prepared.training.obs.shape[1],
                                 prepared.training.action.shape[1], prepared.max_action)
    payload = serialization.to_bytes({"state": state, "training_rng": rng, "step": jnp.int32(step)})
    run_dir.mkdir(parents=True, exist_ok=False)
    write(run_dir / "resolved.json", row)
    write(run_dir / "preparation.json", dict(accepted=True, metadata=prepared.metadata, learner_updates=0))
    (run_dir / f"checkpoint_{step}.msgpack").write_bytes(payload)
    write(run_dir / "result.json", dict(completed=False, synthetic=True, checkpoints=[
        dict(step=step, path=f"checkpoint_{step}.msgpack", sha256=sha256_hex(payload))]))
    (run_dir / "SYNTHETIC_HOST.txt").write_text(SYNTHETIC_HOST_NOTE + "\n", encoding="utf8")
    return dict(run_dir=str(run_dir), checkpoint_sha256=sha256_hex(payload), step=step, seed=seed)


# ---------------------------------------------------------------- thinness

def features(host, obs, actions):
    """x = (standardized s, a) in float64."""
    return np.concatenate([host.standardized(obs).astype(np.float64), np.asarray(actions, np.float64)], axis=1)


class ThinnessIndex:
    """cKDTree over the host's training rows in x = (standardized s, a), with episode-aware counts."""

    def __init__(self, host, k=10, workers=1):
        from scipy.spatial import cKDTree

        ids = host.training_ids
        self.x = features(host, host.converted["observations"][ids], host.converted["actions"][ids])
        self.episode = host.episodes[ids]
        self.k, self.workers = int(k), int(workers)
        self.tree = cKDTree(self.x)
        order = np.argsort(self.episode, kind="stable")
        bounds = np.r_[0, np.flatnonzero(np.diff(self.episode[order])) + 1, len(order)]
        self._members = {int(self.episode[order[a]]): order[a:b] for a, b in zip(bounds[:-1], bounds[1:])}
        self.radius = None

    def knn_other(self, x, episode, k=None):
        """(n, k) distances to the k nearest training rows from other episodes, ascending."""
        k = self.k if k is None else int(k)
        x, episode = np.asarray(x, np.float64), np.asarray(episode, np.int64)
        out = np.full((len(x), k), np.inf)
        need, width = np.arange(len(x)), k
        n = self.tree.n
        while len(need):
            kq = min(width, n)
            d, idx = self.tree.query(x[need], k=kq, workers=self.workers)
            d, idx = d.reshape(len(need), kq), idx.reshape(len(need), kq)
            d = np.where(self.episode[idx] == episode[need][:, None], np.inf, d)
            d.sort(axis=1)
            found = np.isfinite(d[:, :k]).sum(axis=1) if kq >= k else np.isfinite(d).sum(axis=1)
            done = (found >= k) | (kq == n)
            take = min(k, kq)
            out[need[done], :take] = d[done, :take]
            need, width = need[~done], width * 4
        return out

    def fit_radius(self, x, episode):
        """r = the median k-th other-episode neighbour distance of x (the withheld rows)."""
        dk = self.knn_other(x, episode)[:, -1]
        require(np.isfinite(dk).all(), "too few training rows for the k-NN radius")
        self.radius = float(np.median(dk))
        return self.radius, dk

    def counts(self, x, episode, radius=None):
        """c(x): training rows from other episodes within radius (inclusive)."""
        r = self.radius if radius is None else float(radius)
        require(r is not None and r > 0, "radius not set")
        x, episode = np.asarray(x, np.float64), np.asarray(episode, np.int64)
        total = np.asarray(self.tree.query_ball_point(x, r, return_length=True, workers=self.workers), np.int64)
        same = np.zeros(len(x), np.int64)
        for e in np.unique(episode):
            members = self._members.get(int(e))
            if members is None:
                continue
            q = np.flatnonzero(episode == e)
            dist = np.sqrt(((x[q][:, None, :] - self.x[members][None, :, :]) ** 2).sum(-1))
            same[q] = (dist <= r).sum(axis=1)
        return total - same


class StateIndex:
    """cKDTree over standardized training states: the nearest training row's action (any row; callers pass states
    from withheld episodes or the host policy, so every training row is from another episode)."""

    def __init__(self, host, workers=1):
        from scipy.spatial import cKDTree

        ids = host.training_ids
        self.actions = np.asarray(host.converted["actions"][ids], np.float32)
        self.tree = cKDTree(host.standardized(host.converted["observations"][ids]).astype(np.float64))
        self.host, self.workers = host, workers

    def nearest_action(self, obs):
        _, idx = self.tree.query(self.host.standardized(obs).astype(np.float64), k=1, workers=self.workers)
        return self.actions[idx]


# ---------------------------------------------------------------- host-policy episodes

def reset_bank(physics, seeds):
    """(qpos, qvel) after env.seed(seed); env.reset() for each seed: the common reset bank of all arms."""
    qpos, qvel = [], []
    for s in seeds:
        physics.env.seed(int(s))
        physics.env.reset()
        qpos.append(physics.base.sim.data.qpos.copy())
        qvel.append(physics.base.sim.data.qvel.copy())
    return np.asarray(qpos, np.float64), np.asarray(qvel, np.float64)


def policy_episodes(physics, policy, qpos, qvel, *, horizon=1000, every=None, sims=None):
    """Episodes of pi from the given start states, in lockstep (one fixed-shape policy call per step), with the
    time limit at `horizon`: an evaluation episode. Returns (episodes dict, recorded dict) where recorded holds the
    state before every `every`-th step (t = 0, every, 2 every, ...) of every episode."""
    qpos, qvel = np.asarray(qpos, np.float64), np.asarray(qvel, np.float64)
    n, P, task = len(qpos), policy.batch, physics.task
    returns, lengths = np.zeros(n), np.zeros(n, np.int64)
    terminated, failed = np.zeros(n, bool), np.zeros(n, bool)
    rec = dict(episode=[], t=[], qpos=[], qvel=[])
    sims = list(sims or [])
    while len(sims) < min(P, n):
        sims.append(physics.new_sim())
    for start in range(0, n, P):
        members = np.arange(start, min(n, start + P))
        obs = np.zeros((P, task.obs_dim))
        live = np.zeros(P, bool)
        for j, e in enumerate(members):
            physics.restore(sims[j], qpos[e], qvel[e])
            obs[j] = physics.observation(sims[j])
            live[j] = True
        for t in range(horizon):
            slots = np.flatnonzero(live)
            if not len(slots):
                break
            if every and t % every == 0:
                for j in slots:
                    rec["episode"].append(members[j])
                    rec["t"].append(t)
                    rec["qpos"].append(sims[j].data.qpos.copy())
                    rec["qvel"].append(sims[j].data.qvel.copy())
            acts = policy(obs)
            for j in slots:
                e = members[j]
                try:
                    reward, done = physics.step(sims[j], acts[j])
                except physics.simulator_error:
                    failed[e], live[j], obs[j] = True, False, 0.0
                    continue
                returns[e] += reward
                lengths[e] += 1
                if done:
                    terminated[e], live[j], obs[j] = True, False, 0.0
                else:
                    obs[j] = physics.observation(sims[j])
    recorded = dict(episode=np.asarray(rec["episode"], np.int64), t=np.asarray(rec["t"], np.int64),
                    qpos=np.asarray(rec["qpos"], np.float64).reshape(-1, task.nq),
                    qvel=np.asarray(rec["qvel"], np.float64).reshape(-1, task.nv))
    return dict(returns=returns, lengths=lengths, terminated=terminated, failed=failed), recorded


def normalized_scores(dataset, returns):
    """D4RL-normalized score, 100 (R - R_rand) / (R_exp - R_rand) (runtime.common.normalized_scores)."""
    from runtime.common import normalized_scores as score

    return np.asarray(score(dataset, np.asarray(returns, np.float64)), np.float64)


def dataset_episode_scores(host):
    """Normalized return of every D4RL episode in the raw file (d4rl_level1.episode_normalized_returns)."""
    import d4rl

    from experiments.signal.d4rl_level1 import episode_normalized_returns

    raw = host.raw
    _, returns, normalized = episode_normalized_returns(
        raw["rewards"], raw["terminals"], raw["timeouts"], d4rl.infos.REF_MIN_SCORE[host.task.dataset],
        d4rl.infos.REF_MAX_SCORE[host.task.dataset])
    return returns, normalized


def d_thin_mask(raw, f, seed, quantile=0.75):
    """Design layout D-thin(f): every episode in the top quartile of logged return is kept with probability f
    (seeded), every other episode is kept. Returns (keep mask per raw row, record of row and episode counts)."""
    from experiments.signal.d4rl_level1 import raw_episodes

    episode = raw_episodes(raw["terminals"], raw["timeouts"])
    returns = np.bincount(episode, weights=np.asarray(raw["rewards"], np.float64))
    top = returns >= np.quantile(returns, quantile)
    keep_episode = ~top | (np.random.default_rng([seed, 7]).random(len(returns)) < f)
    keep = keep_episode[episode]
    return keep, dict(f=float(f), seed=int(seed), quantile=float(quantile), episodes=int(len(returns)),
                      top_episodes=int(top.sum()), top_episodes_kept=int((top & keep_episode).sum()),
                      raw_rows=int(len(keep)), raw_rows_kept=int(keep.sum()),
                      top_rows=int(top[episode].sum()), top_rows_kept=int((top[episode] & keep).sum()))


# ---------------------------------------------------------------- critic queries

def _chunks(n, size):
    return [(a, min(n, a + size)) for a in range(0, n, size)]


class CriticProbe:
    """Frozen twin critic, policy and target networks of a host, as fixed-shape jitted programs."""

    def __init__(self, host, chunk=4096):
        import jax
        import jax.numpy as jnp

        self.host, self.chunk = host, int(chunk)
        actor, critic, native = host.actor, host.critic, host.native
        args = host.args
        self.gamma = float(args.discount)
        self.policy_noise, self.noise_clip = float(args.policy_noise), float(args.noise_clip)
        m = host.max_action
        self._q = jax.jit(lambda p, s, a: critic.apply(p, s, a))
        self._grad = jax.jit(lambda p, s, a: jax.grad(lambda x: critic.apply(p, s, x)[..., 0].sum())(a))

        def target(nat, reward, next_obs, done, noise):
            action = jnp.clip(actor.apply(nat.actor_target.params, next_obs) + noise, -m, m)
            q = critic.apply(nat.critic_target.params, next_obs, action).min(axis=-1)
            return reward + (1.0 - done) * args.discount * q

        self._target = jax.jit(target)
        self._jnp = jnp

    def _run(self, fn, *arrays):
        n = len(arrays[0])
        outs = []
        for a, b in _chunks(n, self.chunk):
            part = []
            for x in arrays:
                pad = np.zeros((self.chunk,) + x.shape[1:], x.dtype)
                pad[:b - a] = x[a:b]
                part.append(pad)
            outs.append(np.asarray(fn(*part))[:b - a])
        return np.concatenate(outs) if outs else np.zeros((0,))

    def q(self, obs, actions):
        """(n, 2) float32 [Q1, Q2] at raw observations and actions."""
        s = self.host.standardized(obs)
        a = np.asarray(actions, np.float32)
        return self._run(lambda x, y: self._q(self.host.native.critic.params, x, y), s, a)

    def grad_q1(self, obs, actions):
        s = self.host.standardized(obs)
        a = np.asarray(actions, np.float32)
        return self._run(lambda x, y: self._grad(self.host.native.critic.params, x, y), s, a)

    def critic_guided(self, obs, pi, step):
        """(a_cg, |g|): clip(pi + step g / |g|, +-max_action), g = grad_a Q1(s, pi); g = 0 leaves pi."""
        g = self.grad_q1(obs, pi).astype(np.float64)
        norm = np.linalg.norm(g, axis=1)
        unit = np.divide(g, norm[:, None], out=np.zeros_like(g), where=norm[:, None] > 0)
        m = self.host.max_action
        a = np.clip(np.asarray(pi, np.float64) + step * unit, -m, m).astype(np.float32)
        return a, norm

    def noiseless_residual(self, obs, action, reward, next_obs, done, pi_next):
        """b = r + gamma (1 - d) Q1(s', pi(s')) - Q1(s, a), online nets, float64 from float32 values."""
        q_next = self.q(next_obs, pi_next)[:, 0].astype(np.float64)
        q_now = self.q(obs, action)[:, 0].astype(np.float64)
        return (np.asarray(reward, np.float64) + self.gamma * (1.0 - np.asarray(done, np.float64)) * q_next
                - q_now)

    def target_with_noise(self, reward, next_obs, done, noise):
        """TD3+BC's target r + gamma (1 - d) min(Q')(s', clip(pi'(s') + noise, +-1)) for given (clipped) noise."""
        s = self.host.standardized(next_obs)
        r = np.asarray(reward, np.float32)
        d = np.asarray(done, np.float32)
        n = np.asarray(noise, np.float32)
        return self._run(lambda a, b, c, e: self._target(self.host.native, a, b, c, e), r, s, d, n)

    def target_draws(self, reward, next_obs, done, draws, seed):
        """(n, draws) targets with TD3's smoothing noise clip(policy_noise z, +-noise_clip), z from one seeded
        numpy stream (row-major over rows, draws, action coordinates)."""
        g = np.random.default_rng([seed, KEY_TARGET_NOISE])
        n, d = len(reward), self.host.task.action_dim
        z = g.standard_normal((n, draws, d))
        noise = np.clip(self.policy_noise * z, -self.noise_clip, self.noise_clip).astype(np.float32)
        out = np.zeros((n, draws), np.float64)
        for j in range(draws):
            out[:, j] = self.target_with_noise(reward, next_obs, done, noise[:, j])
        return out


# ---------------------------------------------------------------- pools

def state_slots(action_dim):
    return (["pi", "data"] + [f"cd+{j}" for j in range(action_dim)] + [f"cd-{j}" for j in range(action_dim)]
            + ["cg"])


T_SLOTS = ["data", "pi"]


def _seeded_choice(n, size, seed, tag):
    g = np.random.default_rng([seed, KEY_POOL, tag])
    return np.sort(g.choice(n, size=size, replace=False)) if size < n else np.arange(n)


def _restorable(host, converted_ids, source="auto"):
    """Logged states of converted rows; returns (ids kept, LoggedStates of the kept rows, dropped count).
    source: 'auto' (infos/* when present) or 'rebuilt' (from the observation), the pre-registered state source."""
    states = logged_states(host.task, host.raw, host.rows[converted_ids], source)
    keep = states.kept
    if keep.all():
        return converted_ids, states, 0
    sub = logged_states(host.task, host.raw, host.rows[converted_ids[keep]], source)
    return converted_ids[keep], sub, int((~keep).sum())


def _state_actions(host, policy, probe, obs, data_actions, delta, critic_step):
    """(actions (n, 3 + 2d, d) float32 in state_slots order, cd step (n, d), clipped count, |grad|)."""
    pi = policy.actions(obs)
    plus, minus, step, clipped = LB.central_difference_actions(pi, delta)
    cg, norm = probe.critic_guided(obs, pi, critic_step)
    acts = np.concatenate([pi[:, None], np.asarray(data_actions, np.float32)[:, None], plus, minus, cg[:, None]],
                          axis=1)
    return acts.astype(np.float32), step, clipped, norm


def _point_values(probe, thin, obs, actions, episode, compute_thinness=True):
    """Q1/Q2 (n, S, 2) and, optionally, c (n, S) and d_k (n, S) at every (state, slot) point."""
    n, S, d = actions.shape
    flat_obs = np.repeat(obs, S, axis=0)
    flat_act = actions.reshape(n * S, d)
    out = dict(q=probe.q(flat_obs, flat_act).reshape(n, S, 2))
    if compute_thinness:
        x = features(probe.host, flat_obs, flat_act)
        ep = np.repeat(episode, S)
        out["c"] = thin.counts(x, ep).reshape(n, S)
        out["dk"] = thin.knn_other(x, ep)[:, -1].reshape(n, S)
    return out


def build_pool_L(host, policy, probe, thin, *, training_rows, seed, delta=0.1, critic_step=0.1, draws=32,
                 target_seed=0, state_source="auto"):
    """Pool L: every bank row plus `training_rows` seeded training rows, with the logged transitions."""
    task = host.task
    bank = host.bank_ids
    choice = _seeded_choice(len(host.training_ids), training_rows, seed, 1)
    ids_all = np.concatenate([bank, host.training_ids[choice]])
    in_bank_all = np.r_[np.ones(len(bank), bool), np.zeros(len(choice), bool)]
    ids, states, dropped = _restorable(host, ids_all, state_source)
    in_bank = in_bank_all[np.isin(ids_all, ids)]
    conv = host.converted
    obs = observation(task, states.qpos, states.qvel)
    mismatched = int(np.any(obs.astype(np.float32) != conv["observations"][ids], axis=1).sum())
    data = conv["actions"][ids]
    acts, step, clipped, norm = _state_actions(host, policy, probe, obs, data, delta, critic_step)
    episode = host.episodes[ids]
    values = _point_values(probe, thin, obs, acts, episode)
    reward = conv["rewards"][ids].astype(np.float32)
    next_obs = conv["next_observations"][ids].astype(np.float32)
    done = conv["terminals"][ids].astype(np.float32)
    pi_next = policy.actions(next_obs.astype(np.float64))
    b = probe.noiseless_residual(obs, data, reward, next_obs, done, pi_next)
    targets = probe.target_draws(reward, next_obs, done, draws, target_seed)
    arrays = dict(qpos=states.qpos, qvel=states.qvel, obs=obs, episode=episode, converted_id=ids,
                  raw_row=host.rows[ids], in_bank=in_bank, in_training=~in_bank, t=np.full(len(ids), -1, np.int64),
                  actions=acts, cd_step=step, grad_norm=norm.astype(np.float64), reward=reward, next_obs=next_obs,
                  done=done, residual_b=b, targets=targets, **values)
    meta = dict(kind="L", slots=state_slots(task.action_dim), states=int(len(ids)), bank_rows=int(in_bank.sum()),
                training_rows=int((~in_bank).sum()), requested=dict(bank=int(len(bank)), training=int(training_rows)),
                dropped_at_velocity_clip=dropped, state_source=states.source, obs_mismatches_vs_log=mismatched,
                cd_clipped=int(clipped), zero_gradients=int((norm == 0).sum()), delta=delta, critic_step=critic_step,
                target_draws=int(draws), target_seed=int(target_seed))
    return arrays, meta


def build_pool_Pi(host, physics, policy, probe, thin, state_index, *, episodes, every, states, reset_seed_base,
                  seed, horizon=1000, delta=0.1, critic_step=0.1):
    """Pool Pi: host-policy states from `episodes` episodes on the reset bank reset_seed_base + i."""
    task = host.task
    seeds = [reset_seed_base + i for i in range(episodes)]
    q0, v0 = reset_bank(physics, seeds)
    ep, rec = policy_episodes(physics, policy, q0, v0, horizon=horizon, every=every)
    pick = _seeded_choice(len(rec["t"]), states, seed, 2)
    qpos, qvel = rec["qpos"][pick], rec["qvel"][pick]
    obs = observation(task, qpos, qvel)
    data = state_index.nearest_action(obs)
    acts, step, clipped, norm = _state_actions(host, policy, probe, obs, data, delta, critic_step)
    block = -1 - rec["episode"][pick]
    values = _point_values(probe, thin, obs, acts, np.full(len(pick), -1, np.int64))
    scores = normalized_scores(task.dataset, ep["returns"])
    arrays = dict(qpos=qpos, qvel=qvel, obs=obs, episode=block, converted_id=np.full(len(pick), -1, np.int64),
                  raw_row=np.full(len(pick), -1, np.int64), in_bank=np.zeros(len(pick), bool),
                  in_training=np.zeros(len(pick), bool), t=rec["t"][pick], actions=acts, cd_step=step,
                  grad_norm=norm.astype(np.float64), episode_returns=ep["returns"], episode_scores=scores,
                  episode_lengths=ep["lengths"], episode_terminated=ep["terminated"], episode_failed=ep["failed"],
                  reset_seeds=np.asarray(seeds, np.int64), **values)
    meta = dict(kind="Pi", slots=state_slots(task.action_dim), states=int(len(pick)), recorded=int(len(rec["t"])),
                requested=int(states), episodes=int(episodes), every=int(every), reset_seed_base=int(reset_seed_base),
                horizon=int(horizon), host_score_mean=float(scores.mean()),
                host_mean_length=float(ep["lengths"].mean()), simulator_errors=int(ep["failed"].sum()),
                cd_clipped=int(clipped), zero_gradients=int((norm == 0).sum()), delta=delta, critic_step=critic_step,
                data_slot="action of the nearest training row in standardized state space")
    return arrays, meta


def build_pool_T(host, policy, probe, *, rows, seed, state_source="auto"):
    """Pool T: `rows` seeded restorable training rows, slots (data, pi)."""
    task = host.task
    choice = _seeded_choice(len(host.training_ids), rows, seed, 3)
    ids, states, dropped = _restorable(host, host.training_ids[choice], state_source)
    obs = observation(task, states.qpos, states.qvel)
    pi = policy.actions(obs)
    acts = np.stack([host.converted["actions"][ids].astype(np.float32), pi], axis=1)
    values = _point_values(probe, None, obs, acts, None, compute_thinness=False)
    pos = np.searchsorted(host.training_ids, ids)
    arrays = dict(qpos=states.qpos, qvel=states.qvel, obs=obs, episode=host.episodes[ids], converted_id=ids,
                  raw_row=host.rows[ids], training_position=pos, actions=acts, **values)
    meta = dict(kind="T", slots=list(T_SLOTS), states=int(len(ids)), requested=int(rows),
                dropped_at_velocity_clip=dropped, state_source=states.source)
    return arrays, meta


def r6_inputs(host, policy, state_index, *, rows, seed):
    """R6 (c) and (b) inputs on up to `rows` seeded withheld rows: |pi(s) - a|, |a - a_nn| and their observations."""
    w = host.withheld_ids
    ids = w[_seeded_choice(len(w), rows, seed, 4)]
    obs = host.converted["observations"][ids].astype(np.float64)
    a = host.converted["actions"][ids].astype(np.float64)
    pi = policy.actions(obs).astype(np.float64)
    ann = state_index.nearest_action(obs).astype(np.float64)
    return dict(withheld_ids=ids, withheld_obs=obs, pi_minus_a=np.linalg.norm(pi - a, axis=1),
                a_minus_ann=np.linalg.norm(a - ann, axis=1), withheld_episode=host.episodes[ids])


def pool_items(arrays, slots=None, states=None):
    """labeller items: every (state, slot) point, item_id = state * n_slots + slot."""
    acts = arrays["actions"]
    n, S, d = acts.shape
    states = np.arange(n) if states is None else np.asarray(states, np.int64)
    slots = np.arange(S) if slots is None else np.asarray(slots, np.int64)
    ii = np.repeat(states, len(slots))
    jj = np.tile(slots, len(states))
    return dict(qpos=arrays["qpos"][ii], qvel=arrays["qvel"][ii], action=acts[ii, jj].astype(np.float32),
                item_id=(ii * S + jj).astype(np.int64))


# ---------------------------------------------------------------- files

def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_pool(directory, name, arrays, meta):
    """<name>.npz and <name>.json (strict JSON with the npz sha256); refuses to overwrite."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = LB._npz_bytes({k: np.asarray(v) for k, v in arrays.items()})
    npz, js = directory / f"{name}.npz", directory / f"{name}.json"
    require(not npz.exists() and not js.exists(), f"{npz} exists; pools are written once")
    LB._write_atomic(npz, payload)
    record = dict(_json_ready(meta), schema=POOL_SCHEMA, name=name, npz_sha256=sha256_hex(payload),
                  arrays={k: dict(shape=list(np.shape(v)), dtype=str(np.asarray(v).dtype)) for k, v in arrays.items()})
    LB._write_atomic(js, LB.strict_json(record).encode())
    return record


def load_pool(directory, name):
    directory = Path(directory)
    npz, js = directory / f"{name}.npz", directory / f"{name}.json"
    require(npz.is_file() and js.is_file(), f"pool {name} is missing in {directory}; run the pools stage")
    record = json.loads(js.read_text(encoding="utf8"))
    payload = npz.read_bytes()
    require(sha256_hex(payload) == record["npz_sha256"], f"{npz} differs from its record")
    with np.load(npz) as f:
        arrays = {k: f[k] for k in f.files}
    return arrays, record
