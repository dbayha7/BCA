"""Frozen controller-centred trajectory logs for the continuous training core.

Complete source logs are evaluator/audit artifacts. Only ``learner_rows`` enter
training. Whole calibration episodes are reserved before any retention; source
logs are reused for full/rare/zero, and missing rows are never bridged or topped up.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .continuous import KNEE, Objective, dynamics, fixed_cap_policy, provenance
from .continuous_data import ContinuousDataset, critical_region


SOURCE_SCHEMA = "thrusterchain-training-source-v1"
SOURCE_NAMESPACE = 2026102001
ACTION_ATOMS = np.array([-1.0, -KNEE, 0.0, KNEE, 1.0])
LEARNER_KEYS = ("row_id", "episode_id", "time", "observation", "action", "reward",
                "next_observation", "action_probability", "retention_propensity",
                "terminated", "truncated", "bootstrap_mask", "split", "seed", "provenance")


@dataclass
class SourceLog:
    rows: dict
    calibration_episode_ids: np.ndarray
    metadata: dict


def array_hash(arrays):
    """Stable hash includes names, shapes, dtypes and uncompressed array bytes."""
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return int(value)


def _observations(states, regimes, t, objective):
    return np.column_stack((states, regimes, np.full(len(states), (objective.horizon - t) / objective.horizon)))


def behavior_parameters(observations, objective=Objective()):
    """Return greedy atom indices and exact independent-coordinate masses.

    Target = clip(0.6 * unconstrained finite-horizon LQR, -KNEE, KNEE).
    Nearest-atom ties select the lower atom. Each coordinate independently uses
    its nearest atom with probability .8, or a uniform atom with probability .2.
    Hence the nearest atom has mass .84 and each other atom has mass .04.
    """
    observations = np.asarray(observations, np.float64)
    if observations.ndim != 2 or observations.shape[1] != 15 or not np.isfinite(observations).all():
        raise ValueError("observations must be finite [N,15]")
    times = np.rint(objective.horizon * (1 - observations[:, -1])).astype(int)
    if np.any((times < 0) | (times >= objective.horizon)):
        raise ValueError("behavior observations must be nonterminal")
    gains = fixed_cap_policy(KNEE, objective).gains[times]
    unconstrained = -np.einsum("nad,nd->na", gains, observations[:, :13])
    target = np.clip(.6 * unconstrained, -KNEE, KNEE)
    greedy = np.argmin(np.abs(target[..., None] - ACTION_ATOMS), axis=-1)
    probabilities = np.full(greedy.shape + (5,), .04)
    np.put_along_axis(probabilities, greedy[..., None], .84, axis=-1)
    return greedy, probabilities


def generate_source_data(*, seed=0, n_episodes=1024, reservation_size=128,
                         reservation_seed=911, objective=Objective()):
    """Generate balanced complete trajectories and pre-retention reservation.

    Nondefault sizes/horizons support contract tests; the full-run resolved
    configuration must explicitly retain 1024 episodes, 128 reserved and H=100.
    """
    seed = _positive_integer(seed, "seed", 0)
    n_episodes = _positive_integer(n_episodes, "n_episodes")
    reservation_size = _positive_integer(reservation_size, "reservation_size", 0)
    reservation_seed = _positive_integer(reservation_seed, "reservation_seed", 0)
    if n_episodes % 4 or not reservation_size < n_episodes:
        raise ValueError("episode count must be divisible by four; reserve fewer than all episodes")
    streams = {name: [seed, SOURCE_NAMESPACE, index] for index, name in enumerate(
        ("context", "start", "action", "retention"))}
    streams["reservation"] = [seed, SOURCE_NAMESPACE, reservation_seed, 4]
    rng = {name: np.random.default_rng(np.random.SeedSequence(key)) for name, key in streams.items()}
    # This reservation is fixed before retention uniforms or transition outcomes.
    reserved = np.sort(rng["reservation"].permutation(n_episodes)[:reservation_size]).astype(np.int64)
    contexts = rng["context"].permutation(np.tile(np.arange(4), n_episodes // 4))
    regimes = contexts // 2
    states = np.empty((n_episodes, 13))
    states[:, :10] = rng["start"].uniform(-.5, .5, (n_episodes, 10))
    states[:, 10] = np.where(contexts % 2 == 0, .3, 1.1)
    phases = rng["start"].uniform(-np.pi, np.pi, n_episodes)
    states[:, 11:] = .2 * np.column_stack((np.cos(phases), np.sin(phases)))
    columns = {}
    for t in range(objective.horizon):
        obs = _observations(states, regimes, t, objective)
        greedy, probabilities = behavior_parameters(obs, objective)
        exploration = rng["action"].random((n_episodes, 4)) < .2
        uniform = rng["action"].integers(0, 5, (n_episodes, 4))
        indices = np.where(exploration, uniform, greedy)
        actions = ACTION_ATOMS[indices]
        mass = np.prod(np.take_along_axis(probabilities, indices[..., None], axis=-1)[..., 0], axis=-1)
        critical = critical_region(states, actions)
        inclusion_uniform = rng["retention"].random(n_episodes)
        nxt, rewards = np.empty_like(states), np.empty(n_episodes)
        for regime in (0, 1):
            select = regimes == regime
            nxt[select], rewards[select] = dynamics(states[select], actions[select], regime)
        values = dict(observation=obs, action=actions, reward=rewards,
                      next_observation=_observations(nxt, regimes, t + 1, objective),
                      action_probability=mass, state=states.copy(), next_state=nxt.copy(),
                      regime=regimes, context=contexts, critical=critical,
                      retention_uniform=inclusion_uniform, greedy_atom_index=greedy,
                      action_atom_index=indices)
        for key, value in values.items():
            columns.setdefault(key, []).append(value)
        states = nxt
    # Episode-major stable row IDs preserve original linkage after deletion.
    rows = {key: np.stack(value, axis=1).reshape((-1,) + np.asarray(value[0]).shape[1:])
            for key, value in columns.items()}
    total = n_episodes * objective.horizon
    times = np.tile(np.arange(objective.horizon, dtype=np.int64), n_episodes)
    rows.update(row_id=np.arange(total, dtype=np.int64),
                episode_id=np.repeat(np.arange(n_episodes, dtype=np.int64), objective.horizon),
                time=times, terminated=np.zeros(total, bool),
                truncated=times + 1 == objective.horizon,
                bootstrap_mask=(times + 1 < objective.horizon).astype(np.float64),
                split=np.full(total, "train"), seed=np.full(total, seed, np.int64),
                provenance=np.full(total, "continuous-training-trajectory-v1"))
    metadata = dict(schema=SOURCE_SCHEMA, seed=seed, n_episodes=n_episodes,
        generated_rows=total, objective=objective.as_dict(), plant=provenance(),
        random_streams=streams, context_counts=[n_episodes // 4] * 4,
        context_assignment="balanced stratified equal counts, independently seeded permutation of episode order; not iid context draws",
        context_order=["stall_low", "stall_high", "monotone_low", "monotone_high"],
        initial_law="independent U[-.5,.5]^10; current .3/1.1; wave amplitude .2 and U[-pi,pi] phase",
        behavior=dict(controller="clip(0.6 * unconstrained finite-horizon LQR, -KNEE, KNEE), nearest categorical atom",
                      tie_rule="first (lower) atom", action_atoms=ACTION_ATOMS.tolist(),
                      per_coordinate_exploitation=.8, per_coordinate_uniform_exploration=.2,
                      action_measure="exact product categorical mass; not continuous density"),
        calibration_episode_ids=reserved.tolist(), reservation_size=reservation_size,
        reservation_rule="seeded permutation of all source episode IDs before retention",
        calibration_rule="training.prepare selects one retained row per reserved episode with its independent seed",
        raw_reward=True, observation_normalization="none; adapter fits retained training complement only",
        terminal_bootstrap=False, next_observation_semantics="actual original post-step observation",
        continuous_action_support="finite action atoms; no continuous support/occupancy certificate",
        source_rows_sha256=array_hash(rows), budget_topups=0)
    return SourceLog(rows, reserved, metadata)


def _stratum_counts(rows, select):
    return {name: dict(rows=int(np.sum(select & (rows["context"] == context))),
                      critical=int(np.sum(select & (rows["context"] == context) & rows["critical"])))
            for context, name in enumerate(("stall_low", "stall_high", "monotone_low", "monotone_high"))}


def retain_source(source, retention="full", *, rare_probability=.05):
    if retention not in ("full", "rare", "zero") or not np.isfinite(rare_probability) or not 0 <= rare_probability <= 1:
        raise ValueError("invalid retention specification")
    rows = source.rows
    critical_probability = {"full": 1., "rare": float(rare_probability), "zero": 0.}[retention]
    propensity = np.where(rows["critical"], critical_probability, 1.)
    keep = rows["retention_uniform"] < propensity
    audit = dict(rows, retention_propensity=propensity, retained=keep)
    learner = {key: audit[key][keep].copy() for key in LEARNER_KEYS}
    reserved = np.isin(rows["episode_id"], source.calibration_episode_ids)
    metadata = dict(source.metadata, schema="thrusterchain-training-retention-v1", retention=retention,
                    rare_probability=float(rare_probability), retained_rows=int(keep.sum()),
                    retained_training_rows=int(np.sum(keep & ~reserved)),
                    retained_reserved_rows=int(np.sum(keep & reserved)),
                    retained_strata=_stratum_counts(rows, keep),
                    training_strata=_stratum_counts(rows, keep & ~reserved),
                    reserved_strata=_stratum_counts(rows, keep & reserved),
                    learner_rows_sha256=array_hash(learner),
                    critical_rule="state[10]>=0.8 and any(abs(action)>KNEE), fixed before outcomes",
                    ordered_transitions=retention == "full",
                    retention_linkage="row_id/episode_id/time retain original transitions; never shift or bridge")
    return ContinuousDataset(learner, audit, metadata)


def save_source(source, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / "source_audit.npz", **source.rows)
    metadata = dict(source.metadata, file="source_audit.npz",
                    file_sha256=hashlib.sha256((directory / "source_audit.npz").read_bytes()).hexdigest())
    _write_json(directory / "source.json", metadata)
    return metadata


def load_source(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    path = directory / metadata["file"]
    if metadata["schema"] != SOURCE_SCHEMA or hashlib.sha256(path.read_bytes()).hexdigest() != metadata["file_sha256"]:
        raise ValueError("source artifact identity differs")
    with np.load(path, allow_pickle=False) as archive:
        rows = {key: archive[key] for key in archive.files}
    if array_hash(rows) != metadata["source_rows_sha256"]:
        raise ValueError("source array fingerprint differs")
    return SourceLog(rows, np.asarray(metadata["calibration_episode_ids"], np.int64), metadata)


def save_dataset(dataset, directory):
    """Save only the retained learner view; the source audit is stored separately."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / "learner.npz", **dataset.learner_rows)
    metadata = dict(dataset.metadata, file="learner.npz",
                    file_sha256=hashlib.sha256((directory / "learner.npz").read_bytes()).hexdigest())
    _write_json(directory / "dataset.json", metadata)
    return metadata


def load_dataset(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    path = directory / metadata["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["file_sha256"]:
        raise ValueError("dataset artifact identity differs")
    with np.load(path, allow_pickle=False) as archive:
        learner = {key: archive[key] for key in archive.files}
    if set(learner) != set(LEARNER_KEYS) or array_hash(learner) != metadata["learner_rows_sha256"]:
        raise ValueError("learner array schema/fingerprint differs")
    return ContinuousDataset(learner, {}, metadata)


def generate_data_family(directory, *, seeds=range(5), n_episodes=1024, reservation_size=128):
    """One inexpensive data-only pass; never imports or launches a learner."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    records = []
    for seed in seeds:
        source = generate_source_data(seed=int(seed), n_episodes=n_episodes, reservation_size=reservation_size)
        source_path = directory / f"seed_{seed}" / "audit"
        source_record = save_source(source, source_path)
        for retention in ("full", "rare", "zero"):
            path = directory / f"seed_{seed}" / retention
            record = save_dataset(retain_source(source, retention), path)
            records.append(dict(seed=int(seed), retention=retention, path=str(path.resolve()),
                                learner_sha256=record["learner_rows_sha256"],
                                source_sha256=source_record["source_rows_sha256"],
                                calibration_episode_ids=source.calibration_episode_ids.tolist(),
                                retained_rows=record["retained_rows"]))
    manifest = dict(schema="thrusterchain-training-data-family-v1", datasets=records)
    _write_json(directory / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    args = parser.parse_args()
    print(json.dumps(generate_data_family(args.directory, seeds=args.seeds), indent=2))
