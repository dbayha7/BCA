"""Original logged tuples and common retained-only normalization for training.

Import is inert. The only production decoder reads the original 768 training
and 128 separately identified calibration episodes. No probe/evaluation or
selection/final-calibration episode transition records are read.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
GATE = HERE.parent / "conditional_ood_gate_central_quartile_v1"
CONDITIONS = ("full", "hole", "random")
FIELDS = ("obs", "action", "reward", "next_obs", "done")
SEEDS = {"gradient_probes": 2026102841, "scale_bank": 2026102842}
SCALE_ROWS = 256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ids_digest(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _frozen(value, dtype):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _geometry():
    return _module("_central_quartile_data_geometry", GATE / "geometry.py")


def _read_source():
    logged_path = HERE.parent / "bca_one_sided_v1/logged_data.py"
    logged = _module("_central_quartile_role_safe_logged", logged_path)
    p = logged.provenance()
    train, calibration = p["train_indices"], p["calibration_indices"]
    codes = p["vectors"]["split_codes"]
    require(len(train) == 768 and len(calibration) == 128,
            "Original 768 training and 128 calibration roles required")
    indices = np.concatenate((train, calibration))
    require(len(np.unique(indices)) == 896 and np.all(codes[train] == 0)
            and np.all(codes[calibration] == 1), "Protected or overlapping role requested")
    behavior = p["metadata"]["behavior"]
    require(behavior["name"] == "continuous_region_mixture_v1" and behavior["epsilon"] == .25
            and behavior["state_independent"] is True, "Recorder law changed")
    with zipfile.ZipFile(logged.DATA) as archive:
        arrays = {name: logged.read_episodes(archive, name, indices, codes)
                  for name in logged.TRANSITION_NAMES}
    return {"arrays": arrays, "train_count": 768,
            "episode_ids": p["vectors"]["group_ids"][indices],
            "trajectory_ids": p["vectors"]["trajectory_ids"][indices],
            "role_codes": codes[indices], "source_episode_indices": indices,
            "dataset_sha256": logged.EXPECTED_DATA_SHA,
            "bindings_sha256": logged.EXPECTED_BINDINGS_SHA,
            "logged_reader_sha256": _sha(logged_path), "behavior": behavior}


def _validate_source(source):
    arrays = source["arrays"]
    n, nt = len(source["episode_ids"]), source["train_count"]
    require(type(nt) is int and 0 < nt < n, "Training and separate calibration episodes required")
    for key in ("episode_ids", "trajectory_ids"):
        value = np.asarray(source[key])
        require(value.shape == (n,) and value.dtype.kind in "iu" and len(np.unique(value)) == n,
                "Episode identity overlap or duplication")
    roles = np.asarray(source["role_codes"])
    require(roles.shape == (n,) and np.all(roles[:nt] == 0) and np.all(roles[nt:] == 1),
            "Protected role or role ordering violation")
    expected = {"observations": (n, 101, 14), "actions": (n, 100, 4),
                "rewards": (n, 100), "terminated": (n, 100), "truncated": (n, 100)}
    require(set(arrays) == set(expected), "Incomplete original episode records")
    for key, shape in expected.items():
        value = np.asarray(arrays[key])
        require(value.shape == shape and value.dtype.kind in "fbiu" and np.all(np.isfinite(value)),
                "Invalid original field: " + key)
    require(arrays["observations"].dtype == np.float32 and arrays["actions"].dtype == np.float32,
            "Original executed observation/action precision changed")
    require(not np.any(arrays["truncated"]) and np.all(arrays["terminated"][:, -1])
            and not np.any(arrays["terminated"][:, :-1]), "Original finite-horizon masks changed")
    require(np.all(np.abs(arrays["actions"]) <= 1), "Original action bounds violated")
    time = (1 - np.arange(101, dtype=np.float64)/100).astype(np.float32)
    require(np.array_equal(arrays["observations"][:, :, -1], np.broadcast_to(time, (n, 101))),
            "Original remaining-time coordinate changed")


def _one_per_episode(eligible, episode_ids, seed):
    eligible = np.asarray(eligible, np.int64)
    ids = np.asarray(episode_ids, np.int64)
    require(len(np.unique(eligible)) == len(eligible) and np.all(eligible >= 0)
            and np.all(eligible < len(ids)*100), "Invalid eligible probe rows")
    rng = np.random.Generator(np.random.PCG64(seed))
    result = []
    for index in np.argsort(ids, kind="stable"):
        rows = np.sort(eligible[eligible//100 == index])
        require(len(rows) > 0, "Training episode has no retained probe row; no backfill")
        result.append(int(rows[rng.integers(len(rows))]))
    return np.asarray(result, np.int64)


def _source_ids(rows, trajectories):
    rows = np.asarray(rows, np.int64)
    return np.asarray(trajectories, np.int64)[rows//100]*100 + rows%100


def _derive(source):
    """Pure planning from supplied original train/calibration tuples only."""
    _validate_source(source)
    geometry = _geometry()
    require((geometry.CENTER, geometry.HALF_WIDTH, geometry.LOAD_CUTOFF,
             geometry.CONTROL_SEED, geometry.CALIBRATION_SEED, geometry.CALIBRATION_REPLACEMENT_SEED)
            == (0., .25, .4, 2026102811, 2026102362, 2026102812), "Passed gate design changed")
    arrays, nt = source["arrays"], source["train_count"]
    episodes, trajectories = np.asarray(source["episode_ids"]), np.asarray(source["trajectory_ids"])
    train_obs = arrays["observations"][:nt, :-1].reshape(-1, 14)
    train_actions = arrays["actions"][:nt].reshape(-1, 4)
    row_episodes = np.repeat(episodes[:nt], 100)
    row_times = np.tile(np.arange(100, dtype=np.int64), nt)
    masks, control = geometry.control_masks(train_obs, train_actions, row_times, episode_ids=row_episodes)
    hole = geometry.region_masks(train_obs, train_actions)["hole"]
    normalization_rows = np.flatnonzero(masks["hole"] & masks["random"])
    require(len(normalization_rows) > 0 and not np.any(hole[normalization_rows]),
            "No valid common retained normalization rows")
    normalization_ids = _source_ids(normalization_rows, trajectories[:nt])
    calplan = geometry.calibration_plan(arrays["observations"][nt:, :-1].reshape(-1, 14),
        arrays["actions"][nt:].reshape(-1, 4), episodes[nt:])
    require(calplan["status"] == "eligible_plan_only" and calplan["calibration_rows_planned"] == len(episodes)-nt
            and calplan["hole_rows_used"] == 0, "Calibration episode has no eligible row; no backfill")
    cal_rows = np.asarray(calplan["selected_rows"], np.int64)
    cal_ids = _source_ids(cal_rows, trajectories[nt:])
    require(not np.intersect1d(normalization_ids, cal_ids).size, "Calibration normalization leakage")
    normalization_obs = train_obs[normalization_rows]
    plan = {"schema": "bca.central_quartile_training_data.v1",
        "dataset_sha256": source["dataset_sha256"],
        "bindings_sha256": source.get("bindings_sha256"),
        "logged_reader_sha256": source.get("logged_reader_sha256"),
        "gate_geometry_sha256": _sha(GATE / "geometry.py"),
        "row_selection_seeds": dict(SEEDS),
        "mask_seeds": {"random_control": geometry.CONTROL_SEED,
                       "calibration_original": geometry.CALIBRATION_SEED,
                       "calibration_replacement": geometry.CALIBRATION_REPLACEMENT_SEED},
        "decoded_roles": {"training_episodes": nt, "calibration_episodes": len(episodes)-nt,
            "episodes": len(episodes), "transition_tuples": len(episodes)*100,
            "development_probe_transition_rows": 0, "selection_or_final_calibration_transition_rows": 0,
            "all_file_bytes_hashed_and_provenance_metadata_read": True,
            "freshness": "Already development-used train/calibration episodes; no fresh confirmation claim"},
        "training_episode_ids": episodes[:nt].tolist(), "calibration_episode_ids": episodes[nt:].tolist(),
        "common_calibration": {"source_row_ids": cal_ids.tolist(),
            "source_row_ids_sha256": _ids_digest(cal_ids),
            "local_rows": cal_rows.tolist(), "selected_row_digest": calplan["selected_row_digest"],
            "original_rows_inside_hole": calplan["original_rows_inside_hole"],
            "replacement_episode_ids": calplan["replacement_episode_ids"], "hole_rows": 0,
            "rule": calplan["selection_rule"],
            "scope": "All conditions, including full, share this hole-excluding calibration restriction; no hole-conditional coverage guarantee"},
        "common_normalization": {"rows": len(normalization_rows),
            "source_row_ids_sha256": _ids_digest(normalization_ids),
            "local_row_ids_sha256": _ids_digest(normalization_rows),
            "observation_bytes_sha256": hashlib.sha256(normalization_obs.astype('<f4').tobytes()).hexdigest(),
            "hole_rows": 0, "source": "Original TRAIN observations at intersection of hole and random retained rows",
            "coordinates": 14, "native_epsilon": .001,
            "statistic_semantics": "Supply same raw Transition fixture to unchanged native and auxiliary initializers; native population statistics, including time coordinate",
            "audit_mean_float64": normalization_obs.mean(axis=0, dtype=np.float64).tolist(),
            "audit_std_float64": normalization_obs.std(axis=0, dtype=np.float64, ddof=0).tolist()},
        "control_retained_local_row_digests": control["retained_row_digests"],
        "control_matching": "Exact retained counts within original episode, ten-step time bin and observed load side",
        "conditions": {}}
    selections = {}
    for name in CONDITIONS:
        rows = np.flatnonzero(masks[name])
        require(len(rows) >= SCALE_ROWS, "Too few retained rows for fixed scale bank")
        probes = _one_per_episode(rows, episodes[:nt], SEEDS["gradient_probes"])
        scale = np.random.Generator(np.random.PCG64(SEEDS["scale_bank"])).choice(rows, SCALE_ROWS, replace=False)
        ids = _source_ids(rows, trajectories[:nt])
        probe_ids, scale_ids = _source_ids(probes, trajectories[:nt]), _source_ids(scale, trajectories[:nt])
        require(np.all(np.isin(normalization_ids, ids)) and np.all(np.isin(probe_ids, ids))
                and np.all(np.isin(scale_ids, ids)), "Normalization/probe/scale row outside condition replay")
        require(not np.intersect1d(ids, cal_ids).size, "Calibration tuple leaked into replay")
        require(len(np.unique(probes//100)) == nt, "Probe must use one retained row per training episode")
        selections[name] = {"train": rows, "probe": probes, "scale_bank": scale}
        plan["conditions"][name] = {"training_rows": len(rows), "training_episodes": nt,
            "training_source_row_ids_sha256": _ids_digest(ids),
            "training_rows_inside_hole": int(hole[rows].sum()),
            "training_terminal_rows": int(arrays["terminated"][:nt].reshape(-1)[rows].sum()),
            "gradient_probe_source_row_ids": probe_ids.tolist(),
            "gradient_probe_source_row_ids_sha256": _ids_digest(probe_ids), "gradient_probe_rows": len(probes),
            "scale_bank_source_row_ids": scale_ids.tolist(),
            "scale_bank_source_row_ids_sha256": _ids_digest(scale_ids), "scale_bank_rows": len(scale),
            "calibration_source_row_ids_sha256": _ids_digest(cal_ids),
            "normalization_source_row_ids_sha256": _ids_digest(normalization_ids)}
    return plan, selections, normalization_rows, cal_rows


def _check_gate_evidence(plan):
    """Compare only saved construction/calibration identities, not value outcomes."""
    report_path = GATE / "run/report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    saved = json.loads((GATE / "run/geometry.json").read_text(encoding="utf-8"))
    require(report["status"] == "completed_development_design_gate"
            and report["decision"] == "passed_development_design_gate", "Required design gate did not pass")
    require(saved["control_receipt"]["retained_row_digests"] == plan["control_retained_local_row_digests"],
            "Training masks differ from passed gate")
    require(saved["calibration_plan"]["selected_rows"] == plan["common_calibration"]["local_rows"]
            and saved["calibration_plan"]["selected_row_digest"] == plan["common_calibration"]["selected_row_digest"],
            "Calibration rows differ from passed gate")
    for name in CONDITIONS:
        require(saved["analysis"]["counts"]["retained_rows"][name] == plan["conditions"][name]["training_rows"],
                "Training count differs from passed gate")
    return {"directory": GATE.name, "report_sha256": _sha(report_path),
            "geometry_receipt_sha256": _sha(GATE / "run/geometry.json"),
            "prepared_sha256": _sha(GATE / "PREPARED.json")}


def _assemble(native, condition, source, plan):
    require(condition in CONDITIONS, "Unknown support condition")
    expected, selections, normalization_rows, cal_rows = _derive(source)
    supplied = dict(plan); supplied.pop("passed_gate_evidence", None)
    require(_canonical(expected) == _canonical(supplied), "DATA_PLAN identity, selection or normalization mismatch")
    arrays, nt = source["arrays"], source["train_count"]
    episodes, trajectories = np.asarray(source["episode_ids"]), np.asarray(source["trajectory_ids"])

    def bank(local, start, count):
        obs = arrays["observations"][start:start+count]
        intact = {"obs": obs[:, :-1].reshape(-1, 14), "action": arrays["actions"][start:start+count].reshape(-1, 4),
                  "reward": arrays["rewards"][start:start+count].reshape(-1), "next_obs": obs[:, 1:].reshape(-1, 14),
                  "done": arrays["terminated"][start:start+count].reshape(-1)}
        chosen = {key: _frozen(value[local], np.float32) for key, value in intact.items()}
        require(all(np.all(np.isfinite(v)) for v in chosen.values()), "Native conversion overflow")
        return native.Transition(**chosen)

    selected = selections[condition]
    train = bank(selected["train"], 0, nt)
    heldout = bank(cal_rows, nt, len(episodes)-nt)
    probe = bank(selected["probe"], 0, nt)
    scale_bank = bank(selected["scale_bank"], 0, nt)
    normalization_fixture = bank(normalization_rows, 0, nt)
    cal_ids = _source_ids(cal_rows, trajectories[nt:])
    heldout_episode = episodes[nt:][cal_rows//100]
    record = dict(plan["conditions"][condition])
    record.update(condition=condition, dataset_sha256=source["dataset_sha256"],
        common_normalization=plan["common_normalization"], common_calibration=plan["common_calibration"],
        decoded_roles=plan["decoded_roles"], original_training_episode_ids=episodes[:nt].tolist(),
        tuple_conversion="Intact source tuple selection; native float32 conversion only",
        normalization_input_use="Same raw retained-only fixture for native and auxiliary initialization: observation statistics plus dimensions/dtypes only; no optimizer uses this fixture",
        calibration_development_status="Reused, adaptively inspected development calibration episodes; no fresh marginal or hole-conditional coverage guarantee",
        passed_gate_evidence=plan.get("passed_gate_evidence"),
        protected_transition_rows_decoded=0, development_probe_transition_rows_decoded=0)
    return SimpleNamespace(train=train, heldout=heldout, probe=probe, scale_bank=scale_bank,
        normalization_fixture=normalization_fixture,
        normalization_source_row_ids=_frozen(_source_ids(normalization_rows, trajectories[:nt]), np.int64),
        source_row_ids=_frozen(_source_ids(selected["train"], trajectories[:nt]), np.int64),
        heldout_ids={"row_ids": _frozen(cal_ids, np.int64), "episode_ids": _frozen(heldout_episode, np.int64)},
        metadata=record, receipt=record)


def load_condition(native, condition):
    require(condition in CONDITIONS, "Unknown support condition")
    raw = (HERE / "DATA_PLAN.json").read_bytes()
    plan = json.loads(raw)
    require(plan["passed_gate_evidence"] == _check_gate_evidence(plan), "Passed gate evidence changed")
    result = _assemble(native, condition, _read_source(), plan)
    require(len(result.probe.obs) == 768 and len(result.heldout.obs) == 128
            and len(result.scale_bank.obs) == 256, "Frozen auxiliary/calibration bank sizes changed")
    result.metadata["data_plan_sha256"] = hashlib.sha256(raw).hexdigest()
    return result


def prepare_plan():
    """Explicit CPU data-only preparation; writes a new plan exactly once."""
    target = HERE / "DATA_PLAN.json"
    require(not target.exists(), "Existing DATA_PLAN is preserved; no automatic re-freeze")
    source = _read_source()
    plan, _, _, _ = _derive(source)
    plan["passed_gate_evidence"] = _check_gate_evidence(plan)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False)+"\n")
    return {"status": "data_plan_prepared_no_model_or_simulator_work", "path": str(target),
            "sha256": _sha(target), "decoded_roles": plan["decoded_roles"],
            "training_rows": {key: value["training_rows"] for key, value in plan["conditions"].items()},
            "normalization_rows": plan["common_normalization"]["rows"],
            "calibration_replacements": len(plan["common_calibration"]["replacement_episode_ids"])}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(prepare_plan(), sort_keys=True))
