"""Role-safe intact transition banks for the fixed support intervention.

Import performs no file reads, model construction, normalization or simulation.
Only original train and the declared 128 calibration episode records are decoded.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
CONDITIONS = ("restricted", "broad_count_matched")
FIELDS = ("obs", "action", "reward", "next_obs", "done")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def _ids_digest(value):
    return hashlib.sha256(np.asarray(value, dtype="<i8").tobytes()).hexdigest()


def _frozen(value, dtype):
    array = np.asarray(value, dtype=dtype)
    # A bytes-backed copy cannot be made writable by a caller.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _read_source():
    logged_path = HERE.parent / "bca_one_sided_v1/logged_data.py"
    logged = _module("_support_role_safe_logged", logged_path)
    p = logged.provenance()  # Hashes and provenance vectors; no protected tuples.
    train_idx, cal_idx = p["train_indices"], p["calibration_indices"]
    codes = p["vectors"]["split_codes"]
    require(len(train_idx) == 768 and len(cal_idx) == 128, "Original episode roles changed")
    require(np.all(codes[train_idx] == 0) and np.all(codes[cal_idx] == 1), "Protected episode role requested")
    indices = np.concatenate((train_idx, cal_idx))
    require(len(np.unique(indices)) == 896, "Training/calibration episode overlap")
    with zipfile.ZipFile(logged.DATA) as archive:
        arrays = {name: logged.read_episodes(archive, name, indices, codes)
                  for name in logged.TRANSITION_NAMES}
    return {"arrays": arrays, "train_count": 768,
            "episode_ids": p["vectors"]["group_ids"][indices],
            "trajectory_ids": p["vectors"]["trajectory_ids"][indices],
            "role_codes": codes[indices], "dataset_sha256": logged.EXPECTED_DATA_SHA,
            "bindings_sha256": logged.EXPECTED_BINDINGS_SHA,
            "logged_reader_sha256": hashlib.sha256(logged_path.read_bytes()).hexdigest()}


def _validate_source(source):
    arrays = source["arrays"]
    n, nt = len(source["episode_ids"]), source["train_count"]
    require(0 < nt < n, "Both training and calibration episodes required")
    require(len(np.unique(source["episode_ids"])) == n and len(np.unique(source["trajectory_ids"])) == n,
            "Episode identity overlap or duplication")
    require(np.all(np.asarray(source["role_codes"])[:nt] == 0)
            and np.all(np.asarray(source["role_codes"])[nt:] == 1), "Protected role or role ordering violation")
    expected = {"observations": (n, 101, 14), "actions": (n, 100, 4),
                "rewards": (n, 100), "terminated": (n, 100), "truncated": (n, 100)}
    require(set(arrays) == set(expected), "Incomplete original episode records")
    for key, shape in expected.items():
        value = np.asarray(arrays[key])
        require(value.shape == shape and value.dtype.kind in "fbiu" and np.all(np.isfinite(value)),
                "Invalid original field: " + key)
    require(not np.any(arrays["truncated"]) and np.all(arrays["terminated"][:, -1])
            and not np.any(arrays["terminated"][:, :-1]), "Original finite-horizon masks changed")
    require(np.all(np.abs(arrays["actions"]) <= 1), "Original action bounds violated")
    expected_time = (1 - np.arange(101, dtype=np.float64) / 100).astype(arrays["observations"].dtype)
    require(np.allclose(arrays["observations"][:, :, -1], expected_time, rtol=0, atol=1e-7),
            "Original remaining-time coordinate changed")


def _assemble(native, condition, source, plan):
    """Pure assembly used by CPU fixtures; load_condition fixes the real source."""
    require(condition in CONDITIONS, "Unknown support condition")
    _validate_source(source)
    design = _module("_support_condition_design", HERE / "design.py")
    require(plan["dataset_sha256"] == source["dataset_sha256"], "DATA_PLAN dataset identity mismatch")
    require(plan["row_selection_seeds"] == design.SEEDS, "DATA_PLAN row selection seeds changed")
    record = plan["conditions"][condition]
    arrays, nt = source["arrays"], source["train_count"]
    episode = np.asarray(source["episode_ids"], dtype=np.int64)
    trajectory = np.asarray(source["trajectory_ids"], dtype=np.int64)
    train_actions, cal_actions = arrays["actions"][:nt], arrays["actions"][nt:]
    rows = design.training_rows(train_actions, episode[:nt])[condition]
    probes = design.one_per_episode(rows, episode[:nt], 100, design.SEEDS["gradient_probes"])
    scale_rows = np.random.Generator(np.random.PCG64(design.SEEDS["scale_bank"])).choice(rows, 256, replace=False)
    eligible_cal = np.flatnonzero(design.support_mask(cal_actions).reshape(-1)) if condition == "restricted" else np.arange(len(cal_actions) * 100)
    cal_rows = design.one_per_episode(eligible_cal, episode[nt:], 100, design.SEEDS["calibration_rows"])

    def source_ids(local, ids):
        return ids[local // 100] * 100 + local % 100

    train_ids = source_ids(rows, trajectory[:nt])
    probe_ids = source_ids(probes, trajectory[:nt])
    scale_ids = source_ids(scale_rows, trajectory[:nt])
    cal_ids = source_ids(cal_rows, trajectory[nt:])
    require(record["training_rows"] == len(rows) and record["training_episodes"] == nt,
            "DATA_PLAN training counts mismatch")
    require(_ids_digest(train_ids) == record["training_source_row_ids_sha256"], "DATA_PLAN train row digest mismatch")
    for field, ids in (("gradient_probe_source_row_ids", probe_ids),
                       ("scale_bank_source_row_ids", scale_ids), ("calibration_source_row_ids", cal_ids)):
        require(np.array_equal(np.asarray(record[field], dtype=np.int64), ids), "DATA_PLAN source row identity mismatch: " + field)
    require(np.all(np.isin(probe_ids, train_ids)) and np.all(np.isin(scale_ids, train_ids)),
            "Derived fitting/probe rows outside condition replay")
    require(not np.intersect1d(train_ids, cal_ids).size, "Calibration tuple leaked into training")

    def bank(local, start, count):
        obs = arrays["observations"][start:start + count]
        a = arrays["actions"][start:start + count]
        r = arrays["rewards"][start:start + count]
        done = arrays["terminated"][start:start + count]
        intact = {"obs": obs[:, :-1].reshape(-1, 14), "action": a.reshape(-1, 4),
                  "reward": r.reshape(-1), "next_obs": obs[:, 1:].reshape(-1, 14),
                  "done": done.reshape(-1)}
        selected = design.select_intact_transitions(intact, local)
        selected = {key: _frozen(value, np.float32) for key, value in selected.items()}
        require(all(np.all(np.isfinite(value)) for value in selected.values()), "Native float32 conversion overflow")
        if condition == "restricted":
            require(np.all(selected["action"][:, 0] <= .5), "Excluded action leaked into restricted bank")
        return native.Transition(**selected)

    train = bank(rows, 0, nt)
    heldout = bank(cal_rows, nt, len(cal_actions))
    probe = bank(probes, 0, nt)
    scale_bank = bank(scale_rows, 0, nt)
    above_train = int(np.sum(train.action[:, 0] > .5))
    above_cal = int(np.sum(heldout.action[:, 0] > .5))
    require(above_train == record["training_rows_above_gap_cutoff"]
            and above_cal == record["calibration_rows_above_gap_cutoff"], "DATA_PLAN support counts changed")
    require(float(train.action[:, 0].min()) == record["a0_min"]
            and float(train.action[:, 0].max()) == record["a0_max"], "DATA_PLAN support extent changed")
    heldout_episode = episode[nt:][cal_rows // 100]
    require(len(np.unique(heldout_episode)) == len(cal_actions), "Calibration must use one row per original episode")
    receipt = {"condition": condition, "dataset_sha256": source["dataset_sha256"],
               "bindings_sha256": source.get("bindings_sha256"),
               "training_rows": len(rows), "training_episodes": nt,
               "training_source_row_ids_sha256": _ids_digest(train_ids),
               "gradient_probe_source_row_ids_sha256": _ids_digest(probe_ids),
               "scale_bank_source_row_ids_sha256": _ids_digest(scale_ids),
               "calibration_source_row_ids_sha256": _ids_digest(cal_ids),
               "gradient_probe_source_row_ids": probe_ids.tolist(),
               "scale_bank_source_row_ids": scale_ids.tolist(),
               "calibration_source_row_ids": cal_ids.tolist(),
               "calibration_episode_ids": heldout_episode.tolist(),
               "gradient_probe_rows": len(probes), "scale_bank_rows": len(scale_rows),
               "calibration_rows": len(cal_rows), "training_rows_above_gap_cutoff": above_train,
               "calibration_rows_above_gap_cutoff": above_cal,
               "training_terminal_rows": int(np.sum(train.done)),
               "calibration_terminal_rows": int(np.sum(heldout.done)),
               "protected_transition_rows_decoded": 0,
               "normalization": "Not performed by loader; native initializer fits condition train observations only",
               "tuple_conversion": "Intact original tuple selection; native float32 cast only; actions/rewards/successors/masks not replaced",
               "logged_reader_sha256": source.get("logged_reader_sha256")}
    return SimpleNamespace(train=train, heldout=heldout,
        heldout_ids={"row_ids": _frozen(cal_ids, np.int64), "episode_ids": _frozen(heldout_episode, np.int64)},
        probe=probe, scale_bank=scale_bank, source_row_ids=_frozen(train_ids, np.int64), receipt=receipt)


def load_condition(native, condition):
    """Read exactly the approved original roles and produce raw native tuples."""
    require(condition in CONDITIONS, "Unknown support condition")
    plan_raw = (HERE / "DATA_PLAN.json").read_bytes()
    plan = json.loads(plan_raw)
    result = _assemble(native, condition, _read_source(), plan)
    require(len(result.train.obs) == 57752 and len(result.probe.obs) == 768
            and len(result.scale_bank.obs) == 256 and len(result.heldout.obs) == 128,
            "Frozen condition bank sizes changed")
    result.receipt["data_plan_sha256"] = hashlib.sha256(plan_raw).hexdigest()
    return result


def audit_condition(condition):
    """Permitted CPU data-only audit; no native model/runtime imports."""
    from collections import namedtuple
    native = SimpleNamespace(Transition=namedtuple("Transition", FIELDS))
    return load_condition(native, condition).receipt
