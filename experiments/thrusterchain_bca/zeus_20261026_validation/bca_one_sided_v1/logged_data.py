"""Pinned logged transitions with episode-role filtering before numeric decoding.

Only train (0) and tuning_validation (1) episode transition records may be
decoded. Selection/final_calibration transition records are skipped as opaque
ZIP bytes. No JAX, model, calibration, simulator or evaluator imports.
"""
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HANDOFF = HERE.parents[1] / "titan_20261026"
PAYLOAD = HANDOFF / "payload"
DATA = PAYLOAD / "transfer_bundle/research_runs/pilot_20261023T115408Z/regular_1280_2026102301.npz"
BINDINGS = PAYLOAD / "unifloral_checkpoint_v1/train_20261023T183248_552877Z/bindings.json"
EXPECTED_DATA_SHA = "4c857aaa0373bee4c74db33036303b4cf75c703b289a459525f75090741f3226"
EXPECTED_BINDINGS_SHA = "35adffbcfbde2bc06cab8cd36776b959d2e9d430f9dc1b6144eff225dea35754"
EXPECTED_DIGEST = "6b58db68263dff0d7e9b8d6dfa7c1ca5d4127ed9ee6a3b9069add1c565e88331"
VECTOR_NAMES = ("group_ids", "trajectory_ids", "episode_seeds", "split_codes")
TRANSITION_NAMES = ("observations", "actions", "rewards", "terminated", "truncated")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("Unsupported NPY header version")
    require(not fortran and not dtype.hasobject, "Unsupported NPY encoding")
    return shape, dtype, stream.tell()


def provenance(data=DATA, bindings_path=BINDINGS):
    data, bindings_path = Path(data), Path(bindings_path)
    require(sha(data) == EXPECTED_DATA_SHA, "Pinned logged dataset hash mismatch")
    require(sha(bindings_path) == EXPECTED_BINDINGS_SHA, "Checkpoint bindings hash mismatch")
    with np.load(data, allow_pickle=False) as archive:
        envelope = json.loads(str(archive["manifest"]))
        vectors = {name: np.asarray(archive[name]).copy() for name in VECTOR_NAMES}
    metadata = envelope["metadata"]
    require(envelope["sha256"] == EXPECTED_DIGEST, "Dataset digest mismatch")
    require(metadata["episodes"] == 1280 and metadata["gamma"] == .98
            and metadata["horizon"] == 100 and metadata["task_semantics"] == "finite_horizon"
            and metadata["observation_mode"] == "time_aware"
            and metadata["physics"] == {"stall": True, "stall_slope": 1.5}
            and metadata["provenance"] == "research", "Dataset semantics mismatch")
    for name, value in vectors.items():
        require(value.shape == (1280,) and np.all(np.isfinite(value)), f"Invalid provenance vector {name}")
    groups, codes = vectors["group_ids"], vectors["split_codes"]
    require(len(np.unique(groups)) == 1280 and len(np.unique(vectors["trajectory_ids"])) == 1280,
            "Episode/group IDs not unique")
    require(np.array_equal(np.unique(codes, return_counts=True)[0], [0, 1, 2, 3])
            and np.array_equal(np.unique(codes, return_counts=True)[1], [768, 256, 128, 128]),
            "Episode split counts differ")
    train = np.flatnonzero(codes == 0)
    tuning = np.flatnonzero(codes == 1)
    tuning = tuning[np.argsort(groups[tuning], kind="stable")]
    calibration, validation = tuning[:128], tuning[128:]
    times = np.random.Generator(np.random.PCG64(2026102362)).integers(0, 100, 128)
    scale_indices = np.random.Generator(np.random.PCG64(2026102363)).choice(76800, 256, replace=False)
    bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
    split = bindings["datasetidentity"]["split_receipt"]
    row_ids = vectors["trajectory_ids"][calibration] * 100 + times
    require(split["calibration_times"] == times.tolist()
            and split["calibration_row_ids"] == row_ids.tolist()
            and split["scale_bank_training_row_indices"] == scale_indices.tolist()
            and split["selection_and_final_calibration_used"] is False, "Saved split identity mismatch")
    require(not (set(groups[train]) & set(groups[calibration]))
            and not (set(groups[train]) & set(groups[validation]))
            and not (set(groups[calibration]) & set(groups[validation])), "Episode leakage")
    reserved_eval = (set(range(2026102400, 2026102464)) | set(range(2026102500, 2026102564))
                     | set(range(2026102700, 2026102764)))
    require(not reserved_eval.intersection(vectors["episode_seeds"][np.concatenate((train, tuning))].tolist()),
            "Logged episodes overlap known development/confirmation seeds")
    return {"metadata": metadata, "vectors": vectors, "bindings": bindings,
            "train_indices": train, "calibration_indices": calibration,
            "validation_indices": validation, "calibration_times": times,
            "fitting_training_row_indices": scale_indices, "calibration_row_ids": row_ids}


def read_episodes(archive, name, episode_indices, codes):
    """Read selected C-order episode records only; refuse protected roles."""
    ids = np.asarray(episode_indices, dtype=np.int64)
    require(ids.ndim == 1 and len(np.unique(ids)) == len(ids)
            and np.all((ids >= 0) & (ids < 1280)), "Invalid episode selection")
    require(np.all(np.isin(codes[ids], [0, 1])), "Forbidden selection/final-calibration transition access")
    require(name in TRANSITION_NAMES, "Unapproved numeric transition member")
    with archive.open(name + ".npy") as stream:
        shape, dtype, offset = header(stream)
        require(shape[0] == 1280 and dtype.kind in "fbiu", "Unexpected transition array header")
        stride = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        result = np.empty((len(ids),) + shape[1:], dtype=dtype)
        for destination in np.argsort(ids, kind="stable"):
            episode = int(ids[destination])
            stream.seek(offset + episode * stride)
            raw = stream.read(stride)
            require(len(raw) == stride, "Truncated episode payload")
            result[destination] = np.frombuffer(raw, dtype=dtype).reshape(shape[1:])
    require(np.all(np.isfinite(result)), "Nonfinite selected logged data")
    return result


def _load_core(data=DATA, bindings_path=BINDINGS, include_validation=True):
    """Return original train/fitting/calibration and remaining tuning-validation.

    The latter is a full 12,800-row pool; a bounded caller must predeclare its
    own episode-preserving validation row selection before model execution.
    Calibration reuse is explicit: these original reference episodes already
    calibrated the original checkpoint. No fresh-confirmation claim is valid.
    """
    p = provenance(data, bindings_path)
    parts = (p["train_indices"], p["calibration_indices"])
    if include_validation:
        parts += (p["validation_indices"],)
    indices = np.concatenate(parts)
    with zipfile.ZipFile(data) as archive:
        arrays = {name: read_episodes(archive, name, indices, p["vectors"]["split_codes"])
                  for name in TRANSITION_NAMES}
    expected_episodes = 1024 if include_validation else 896
    require(arrays["observations"].shape == (expected_episodes, 101, 14)
            and arrays["actions"].shape == (expected_episodes, 100, 4)
            and arrays["rewards"].shape == (expected_episodes, 100), "Transition shape mismatch")
    require(not np.any(arrays["truncated"]) and np.all(arrays["terminated"][:, -1])
            and not np.any(arrays["terminated"][:, :-1]), "Finite-horizon mask mismatch")
    require(np.all(np.abs(arrays["actions"]) <= 1), "Action bounds changed")
    expected_time = (1 - np.arange(101, dtype=np.float64) / 100).astype(arrays["observations"].dtype)
    require(np.allclose(arrays["observations"][:, :, -1], expected_time, rtol=0, atol=1e-7),
            "Remaining-time coordinate mismatch")

    def transitions(episode_rows, times=None):
        if times is None:
            obs = arrays["observations"][episode_rows, :-1].reshape(-1, 14)
            actions = arrays["actions"][episode_rows].reshape(-1, 4)
            reward = arrays["rewards"][episode_rows].reshape(-1)
            nxt = arrays["observations"][episode_rows, 1:].reshape(-1, 14)
            done = arrays["terminated"][episode_rows].reshape(-1)
        else:
            obs = arrays["observations"][episode_rows, times]
            actions = arrays["actions"][episode_rows, times]
            reward = arrays["rewards"][episode_rows, times]
            nxt = arrays["observations"][episode_rows, times + 1]
            done = arrays["terminated"][episode_rows, times]
        return {k: np.asarray(v, dtype=np.float32) for k, v in zip(
            ("obs", "action", "reward", "next_obs", "done"), (obs, actions, reward, nxt, done))}

    train = transitions(np.arange(768))
    calibration = transitions(np.arange(768, 896), p["calibration_times"])
    validation_times = np.random.Generator(np.random.PCG64(2026102671)).integers(0, 100, 128)
    validation = transitions(np.arange(896, 1024), validation_times) if include_validation else None
    fitting = {key: value[p["fitting_training_row_indices"]].copy() for key, value in train.items()}
    groups = p["vectors"]["group_ids"]
    train_groups = groups[p["train_indices"]]
    fitting_groups = np.repeat(train_groups, 100)[p["fitting_training_row_indices"]]
    for bank in (train, fitting, calibration) + ((validation,) if include_validation else ()):
        for value in bank.values():
            value.setflags(write=False)
    return {"provenance": p, "train": train, "fitting": fitting, "calibration": calibration,
            "validation": validation, "validation_times": validation_times, "train_group_ids": train_groups,
            "fitting_group_ids": fitting_groups,
            "calibration_group_ids": groups[p["calibration_indices"]],
            "validation_group_ids": groups[p["validation_indices"]],
            "selection_and_final_calibration_transition_rows_decoded": 0}


def load_logged_banks(data=DATA, bindings_path=BINDINGS):
    """Return exact fit256/calibration128 plus fixed validation128 logged rows.

    Local validation uses the remaining tuning-validation groups, sorted by
    group ID, one time each from PCG64(2026102671). It is exploratory and was
    not used by the original checkpoint's train/reference loader.
    """
    core = _load_core(data, bindings_path, include_validation=True)
    p = core["provenance"]
    vectors = p["vectors"]
    fit_ep = p["train_indices"][p["fitting_training_row_indices"] // 100]
    fit_time = p["fitting_training_row_indices"] % 100
    roles = (
        ("fit", core["fitting"], fit_ep, fit_time),
        ("calibration", core["calibration"], p["calibration_indices"], p["calibration_times"]),
        ("validation", core["validation"], p["validation_indices"], core["validation_times"]),
    )
    result = {}
    for role, bank, episode, time in roles:
        bank = dict(bank)
        bank["group_ids"] = vectors["group_ids"][episode].astype(np.int64)
        bank["row_ids"] = (vectors["trajectory_ids"][episode] * 100 + time).astype(np.int64)
        for value in bank.values():
            value.setflags(write=False)
        result[role] = bank
    group_sets = {role: set(bank["group_ids"].tolist()) for role, bank in result.items()}
    require(not (group_sets["fit"] & group_sets["calibration"])
            and not (group_sets["fit"] & group_sets["validation"])
            and not (group_sets["calibration"] & group_sets["validation"]), "Bank episode leakage")
    receipt = {
        "schema": "unifloral.one_sided.logged_data.v1",
        "status": "available_exploratory_calibration_reuse_disclosed",
        "dataset_sha256": EXPECTED_DATA_SHA, "dataset_digest": EXPECTED_DIGEST,
        "bindings_sha256": EXPECTED_BINDINGS_SHA,
        "dataset_path": str(Path(data).resolve()),
        "split_counts": {"train": 768, "tuning_validation": 256, "selection": 128, "final_calibration": 128},
        "rows": {role: len(bank["obs"]) for role, bank in result.items()},
        "unique_groups": {role: len(groups) for role, groups in group_sets.items()},
        "fit_training_row_indices": p["fitting_training_row_indices"].tolist(),
        "fit_episode_indices": fit_ep.tolist(), "fit_times": fit_time.tolist(),
        "calibration_episode_indices": p["calibration_indices"].tolist(),
        "calibration_times": p["calibration_times"].tolist(), "calibration_time_seed": 2026102362,
        "validation_episode_indices": p["validation_indices"].tolist(),
        "validation_times": core["validation_times"].tolist(), "validation_time_seed": 2026102671,
        "group_ids": {role: bank["group_ids"].tolist() for role, bank in result.items()},
        "row_ids": {role: bank["row_ids"].tolist() for role, bank in result.items()},
        "pairwise_episode_overlap": False,
        "calibration_role": "Original one-row-per-episode reference: first128 sorted tuning_validation groups; reused after original refreshes, not fresh confirmation.",
        "validation_role": "Remaining128 tuning_validation episodes, disjoint from original train/reference; not consumed by the original checkpoint loader, global prior inspection history not certified.",
        "calibration_reused_from_original_BCA": True,
        "fresh_confirmation_claim_permitted": False,
        "validation_used_by_original_checkpoint_loader": False,
        "selection_and_final_calibration_transition_rows_decoded": 0,
        "completed_confirmation_files_read": 0,
        "privileged_candidate_transition_rows_read": 0,
        "new_fits": 0, "calibration_refreshes": 0, "model_forwards": 0, "simulator_calls": 0,
        "normalization": {"rule": "Preserve saved native training-observation mean/std and epsilon0.001; no refit on these small banks.",
                          "mean": p["bindings"]["normalization"]["mean"], "std": p["bindings"]["normalization"]["std"]},
        "boundary_contract": {"horizon": 100, "gamma": .98, "finite_horizon": True,
                              "terminal_bootstrap": False, "last_step_terminated": True,
                              "earlier_steps_terminated": False, "truncated": False,
                              "remaining_time_feature": "(100-t)/100, coordinate13; next observation from same episode at t+1"},
        "coverage_limit": "One logged row per calibration episode controls dependence better than treating all100rows as independent; representativeness, adaptive reference reuse and selected-action/OOD validity remain assumptions, not proven by this structural audit.",
    }
    result["receipt"] = receipt
    return result


def load_training_template(native, path=DATA):
    """Drop-in original loader output for exact committed template restoration.

    Only train768 episodes and original calibration128 episodes are decoded.
    This function performs array construction only; the caller owns restoration.
    """
    import jax.numpy as jnp
    core = _load_core(path, BINDINGS, include_validation=False)
    p = core["provenance"]
    order = ("obs", "action", "reward", "next_obs", "done")
    train = native.Transition(*(jnp.asarray(core["train"][name], jnp.float32) for name in order))
    heldout = native.Transition(*(jnp.asarray(core["calibration"][name], jnp.float32) for name in order))
    scale_bank = {"observations": core["fitting"]["obs"], "actions": core["fitting"]["action"], "group_ids": core["fitting_group_ids"]}
    heldout_ids = {"row_ids": p["calibration_row_ids"].astype(np.int64), "episode_ids": core["calibration_group_ids"].astype(np.int64)}
    used = np.concatenate((p["train_indices"], p["calibration_indices"]))
    info = {
        "dataset_sha256": EXPECTED_DATA_SHA, "digest": EXPECTED_DIGEST, "synthetic_fixture": False,
        "train_groups": core["train_group_ids"].tolist(), "calibration_groups": core["calibration_group_ids"].tolist(),
        "used_episode_seeds": p["vectors"]["episode_seeds"][used].astype(np.int64).tolist(), "scale_bank": scale_bank,
        "split_receipt": p["bindings"]["datasetidentity"]["split_receipt"],
    }
    return train, heldout, heldout_ids, info
