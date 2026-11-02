"""Prepare proposal provenance using permitted train/calibration actions only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile
import numpy as np
import design

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("proposal_role_safe_data", HERE.parent / "bca_one_sided_v1/logged_data.py")
logged = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logged)
p = logged.provenance()
with zipfile.ZipFile(logged.DATA) as archive:
    train = logged.read_episodes(archive, "actions", p["train_indices"], p["vectors"]["split_codes"])
    calibration = logged.read_episodes(archive, "actions", p["calibration_indices"], p["vectors"]["split_codes"])
groups = p["vectors"]["group_ids"]
trajectory = p["vectors"]["trajectory_ids"]
row_plans = design.training_rows(train, groups[p["train_indices"]])


def source_rows(local, episodes):
    return trajectory[episodes][local // 100] * 100 + local % 100


def digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


conditions = {}
for name, rows in row_plans.items():
    probes = design.one_per_episode(rows, groups[p["train_indices"]], 100, design.SEEDS["gradient_probes"])
    bank = np.random.Generator(np.random.PCG64(design.SEEDS["scale_bank"])).choice(rows, 256, replace=False)
    eligible_cal = np.flatnonzero(design.support_mask(calibration).reshape(-1)) if name == "restricted" else np.arange(12800)
    cal_rows = design.one_per_episode(eligible_cal, groups[p["calibration_indices"]], 100, design.SEEDS["calibration_rows"])
    actions = train.reshape(-1,4)[rows]
    conditions[name] = {"training_rows": len(rows), "training_episodes": 768,
        "training_source_row_ids_sha256": digest(source_rows(rows, p["train_indices"])),
        "gradient_probe_source_row_ids": source_rows(probes, p["train_indices"]).tolist(),
        "scale_bank_source_row_ids": source_rows(bank, p["train_indices"]).tolist(),
        "calibration_source_row_ids": source_rows(cal_rows, p["calibration_indices"]).tolist(),
        "a0_min": float(actions[:,0].min()), "a0_max": float(actions[:,0].max()),
        "training_rows_above_gap_cutoff": int(np.sum(actions[:,0] > .5)),
        "calibration_rows_above_gap_cutoff": int(np.sum(calibration.reshape(-1,4)[cal_rows,0] > .5))}
assert conditions["restricted"]["training_rows"] == conditions["broad_count_matched"]["training_rows"] == 57752
assert conditions["restricted"]["training_rows_above_gap_cutoff"] == 0
assert conditions["broad_count_matched"]["training_rows_above_gap_cutoff"] > 0
seeds = list(range(2026104000, 2026104032))
prior = set(p["vectors"]["episode_seeds"].tolist())
for start, stop in ((2026102400,2026102464),(2026102500,2026102564),(2026102700,2026102764),(2026102800,2026102832)):
    prior.update(range(start,stop))
assert not set(seeds).intersection(prior)
receipt = {"status": "proposed_row_plan_not_execution_authorization", "dataset_sha256": logged.EXPECTED_DATA_SHA,
    "row_selection_seeds": design.SEEDS, "conditions": conditions, "evaluation_reset_seeds": seeds,
    "known_reset_seed_collisions": [], "seed_scope": "Known pinned logged/development/confirmation/pilot metadata only; no claim about unrecorded experiments",
    "protected_transition_rows_decoded": 0, "models_initialized": 0, "fitting_updates": 0,
    "calibration_calls": 0, "simulator_calls": 0, "attempt_reserved": False,
    "counts": design.maximum_counts()}
(HERE / "DATA_PLAN.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": receipt["status"], "conditions": {key: {k:value[k] for k in ("training_rows", "training_episodes", "a0_min", "a0_max", "training_rows_above_gap_cutoff", "calibration_rows_above_gap_cutoff")} for key,value in conditions.items()}, "counts": receipt["counts"]}))
