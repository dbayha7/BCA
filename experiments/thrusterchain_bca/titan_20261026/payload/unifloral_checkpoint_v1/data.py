"""Read only the pinned offline dataset; no simulator imports or generation."""
import hashlib
import io
import json
from pathlib import Path

DATA_SHA = "4c857aaa0373bee4c74db33036303b4cf75c703b289a459525f75090741f3226"
DATA_DIGEST = "6b58db68263dff0d7e9b8d6dfa7c1ca5d4127ed9ee6a3b9069add1c565e88331"


def load_training(native, path):
    import numpy as np
    import jax.numpy as jnp
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != DATA_SHA:
        raise ValueError("Only the exact previously validated offline dataset is supported")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        envelope = json.loads(str(archive["manifest"]))
        keys = ("observations", "actions", "rewards", "terminated", "truncated",
                "group_ids", "trajectory_ids", "episode_seeds", "split_codes")
        arrays = {key: np.array(archive[key], copy=True) for key in keys}
    metadata = envelope["metadata"]
    if (envelope["sha256"] != DATA_DIGEST or metadata["gamma"] != .98
            or metadata["horizon"] != 100 or metadata["task_semantics"] != "finite_horizon"
            or metadata["observation_mode"] != "time_aware"
            or metadata["physics"] != {"stall": True, "stall_slope": 1.5}
            or metadata["provenance"] != "research" or metadata["episodes"] != 1280):
        raise ValueError("Dataset task/provenance differs from the frozen plan")
    if any(not np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("Nonfinite offline data")
    if (arrays["observations"].shape != (1280, 101, 14)
            or arrays["actions"].shape != (1280, 100, 4)
            or arrays["rewards"].shape != (1280, 100)):
        raise ValueError("Unexpected dataset shape")
    if np.any(arrays["truncated"]) or not np.all(arrays["terminated"][:, -1]) or np.any(arrays["terminated"][:, :-1]):
        raise ValueError("Finite-horizon terminal masks differ")
    groups, codes = arrays["group_ids"], arrays["split_codes"]
    if len(np.unique(groups)) != 1280:
        raise ValueError("Dataset requires one independent episode per group")
    train_ids = np.flatnonzero(codes == 0)
    cal_pool = np.flatnonzero(codes == 1)
    cal_ids = cal_pool[np.argsort(groups[cal_pool], kind="stable")][:128]
    if len(train_ids) != 768 or len(cal_pool) != 256 or len(cal_ids) != 128:
        raise ValueError("Training/reference split counts differ")
    rng_cal = np.random.Generator(np.random.PCG64(2026102362))
    cal_times = rng_cal.integers(0, 100, size=128)
    train_groups = groups[train_ids].astype(np.int64)
    cal_groups = groups[cal_ids].astype(np.int64)
    used = np.concatenate((train_ids, cal_ids))
    used_seeds = arrays["episode_seeds"][used].astype(np.int64).tolist()
    reserved = set(range(2026102400, 2026102464)) | set(range(2026102500, 2026102564))
    if reserved.intersection(used_seeds):
        raise ValueError("Offline episode seeds overlap reserved usefulness evaluation")

    def transition(episode, times=None):
        if times is None:
            obs = arrays["observations"][episode, :-1].reshape(-1, 14)
            action = arrays["actions"][episode].reshape(-1, 4)
            reward = arrays["rewards"][episode].reshape(-1)
            nxt = arrays["observations"][episode, 1:].reshape(-1, 14)
            done = arrays["terminated"][episode].reshape(-1)
        else:
            obs = arrays["observations"][episode, times]
            action = arrays["actions"][episode, times]
            reward = arrays["rewards"][episode, times]
            nxt = arrays["observations"][episode, times + 1]
            done = arrays["terminated"][episode, times]
        return native.Transition(*(jnp.asarray(value, jnp.float32) for value in (obs, action, reward, nxt, done)))

    train, heldout = transition(train_ids), transition(cal_ids, cal_times)
    row_groups = np.repeat(train_groups, 100)
    scale_indices = np.random.Generator(np.random.PCG64(2026102363)).choice(len(row_groups), 256, replace=False)
    scale_bank = {"observations": np.asarray(train.obs)[scale_indices].astype(np.float32),
                  "actions": np.asarray(train.action)[scale_indices].astype(np.float32),
                  "group_ids": row_groups[scale_indices]}
    heldout_ids = {"row_ids": (arrays["trajectory_ids"][cal_ids] * 100 + cal_times).astype(np.int64),
                   "episode_ids": cal_groups}
    info = {"dataset_sha256": actual, "digest": DATA_DIGEST, "synthetic_fixture": False,
            "train_groups": train_groups.tolist(), "calibration_groups": cal_groups.tolist(),
            "used_episode_seeds": used_seeds, "scale_bank": scale_bank,
            "split_receipt": {"train_episodes": 768, "train_rows": 76800,
                "calibration_source": "first 128 tuning_validation groups sorted by group ID",
                "calibration_rows": 128, "calibration_row_ids": heldout_ids["row_ids"].tolist(),
                "calibration_times": cal_times.tolist(), "calibration_row_seed": 2026102362,
                "scale_bank_seed": 2026102363, "scale_bank_training_row_indices": scale_indices.tolist(),
                "selection_and_final_calibration_used": False,
                "data_validation": "Exact previously validated file SHA; structure and split checks; no simulator oracle called"}}
    return train, heldout, heldout_ids, info
