"""The thin-data study (PROTOCOL.md Amendments 11 and 12). Builds its datasets and trains its policies.

    python -I -B thin_study.py --stage data                       (no training: datasets, checks, receipts)
    python -B thin_study.py --label thin_pilot --stage baselines  (TD3+BC baselines only, on the GPU)
    python -B thin_study.py --label thin_dev2 --stage baselines   (the same, under the learner of Amendment 12)
    python -B thin_study.py --label thin_dev2 --stage methods     (Version A and the two permission arms)

Reuses run_study.py's training loop, refresh and saving, and learner.py. Evaluation is in thin_evaluate.py and
needs no learner. A stage is written once per label: an existing START file stops the run.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))     # the repository root, for the calibration package the learner imports
import collect  # noqa: E402
import collect_thin as thin  # noqa: E402
import unseen_move_checks as checks  # noqa: E402

DATA_LABEL = "thin_dev"
DATA_SEEDS = (5301, 5302, 5303)
TRAIN_SEEDS = (20, 21, 22)
GROUPS = {"honeypot": ["honeypot"], "bottleneck_r2_one_hot": ["bottleneck_r2_exit35", "bottleneck_r2_exit25"]}
ARMS = ("complete",) + tuple(thin.arm_name(k) for k in thin.DOSES)
# Amendment 12. Version A alone, then the two permission arms together.
METHOD_CHUNKS = (("support_blind_width",), ("permit", "permit_plain"))
# By the stop rule of Amendment 11 no method is trained on thin Honeypot data: dose 0 and complete data only.
METHOD_ARMS = {"honeypot": ("complete", thin.arm_name(0)), "bottleneck_r2_one_hot": ARMS}
SOURCES = ("PROTOCOL.md", "learner.py", "run_study.py", "thin_study.py", "collect_thin.py", "environment.py",
           "collect.py", "unseen_move_checks.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(task, arm, seed):
    """A dataset of any arm of this study, with its acceptance report."""
    if arm == "complete":
        data, facts = collect.generate(task, "complete", seed)
        return data, facts, checks.check_dataset(task, "complete", data)
    dose = thin.dose_of(arm)
    data, facts = thin.generate(task, seed, dose)
    return data, facts, thin.check(task, dose, data, collect.generate(task, "targeted", seed)[0])


def arm_number(arm):
    """The number the calibration's random seed uses for a data arm: 0 for complete, as in every other
    study, and 100 + dose for a thin arm."""
    return 0 if arm == "complete" else 100 + thin.dose_of(arm)


def build_data():
    folder, receipts = HERE / "data" / DATA_LABEL, HERE / "receipts"
    manifest_path = receipts / f"{DATA_LABEL}_manifest.json"
    if folder.exists() or manifest_path.exists():
        sys.exit(f"label '{DATA_LABEL}' already exists")
    folder.mkdir(parents=True)
    manifest = {"label": DATA_LABEL, "purpose": "development: thin data (PROTOCOL.md Amendment 11), never confirmation",
                "training_updates": 0, "python": platform.python_version(), "numpy": np.__version__,
                "source_sha256": {name: sha256(HERE / name) for name in ("environment.py", "collect.py",
                                                                         "collect_thin.py", "unseen_move_checks.py",
                                                                         "thin_study.py")},
                "doses": list(thin.DOSES), "seeds": list(DATA_SEEDS), "datasets": []}
    reports = []
    for names in GROUPS.values():
        for name in names:
            task = checks.TASKS[name]
            for seed in DATA_SEEDS:
                for arm in ARMS:
                    data, facts, report = generate(task, arm, seed)
                    file_name = f"{name}_{arm}_seed{seed}.npz"
                    np.savez_compressed(folder / file_name, **data)
                    manifest["datasets"].append({"file": file_name, "task": name, "arm": arm, "seed": seed,
                                                 "array_sha256": collect.digest(data),
                                                 "checks_passed": report["passed"], **facts})
                    reports.append({"file": file_name, "seed": seed, **report})
                    print("PASS" if report["passed"] else "FAIL", file_name, flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    (receipts / f"{DATA_LABEL}_checks.json").write_text(json.dumps(reports, indent=2, allow_nan=False))
    failed = [r["file"] for r in reports if not r["passed"]]
    print(f"{len(reports)} datasets written; {len(failed)} failed their checks")
    sys.exit(1 if failed else 0)


def train(label, stage, smoke=False):
    import jax
    import jax.numpy as jnp
    import learner as L
    import run_study as RS

    data_seeds, train_seeds = (DATA_SEEDS[:1], TRAIN_SEEDS[:1]) if smoke else (DATA_SEEDS, TRAIN_SEEDS)
    arms = ("complete", thin.arm_name(0), thin.arm_name(32)) if smoke else ARMS
    out = HERE / "results" / label
    out.mkdir(parents=True, exist_ok=True)
    start_file = out / f"START_{stage}.json"
    if start_file.exists():
        sys.exit(f"{start_file} exists: this stage was already started under this label")
    if stage == "methods" and not (out / "COMPLETE_baselines.json").exists():
        sys.exit("the baselines of this label are not complete")
    manifest = json.loads((HERE / "receipts" / f"{DATA_LABEL}_manifest.json").read_text())
    start_file.write_text(json.dumps({
        "stage": stage, "started_unix": time.time(), "python": platform.python_version(), "jax": jax.__version__,
        "devices": [str(d) for d in jax.devices()], "matmul_precision": "float32",
        "source_sha256": {name: sha256(HERE / name) for name in SOURCES},
        "data_label": DATA_LABEL, "data_seeds": data_seeds, "train_seeds": train_seeds, "arms": arms,
        "groups": GROUPS, "method_chunks": METHOD_CHUNKS, "method_arms": METHOD_ARMS,
        "permit_margin": L.PERMIT_MARGIN, "smoke": smoke}, indent=1))

    def load(task, arm, seed):
        name = f"{task.name}_{arm}_seed{seed}.npz"
        entry = next(e for e in manifest["datasets"] if e["file"] == name)
        data = checks.load(HERE / "data" / DATA_LABEL / name)
        assert collect.digest(data) == entry["array_sha256"] and entry["checks_passed"], name
        return data

    def banks_for(task, arm, seed):
        """Fresh rows with the same dose, never used for a gradient step; and the target-move yardstick."""
        banks = {}
        for name, offset in (("reference", RS.REFERENCE_OFFSET), ("audit", RS.AUDIT_OFFSET)):
            fresh, _, report = generate(task, arm, seed + offset)
            assert report["passed"], (task.name, arm, seed, name)
            banks[name] = RS.subsample(fresh, RS.BANK_ROWS, [seed, offset])
        fresh, _ = collect.generate(task, "complete", seed + RS.PROBE_OFFSET)
        rows = []
        for cell, moves in collect.targeted_moves(task).items():
            here = np.flatnonzero(fresh["s"] == cell)
            could = [i for i in here if any(task.could_execute(fresh["a"][i], d) for d in moves)]
            assert len(could) >= RS.PROBE_ROWS_PER_CELL, (task.name, cell, len(could))
            rows.extend(could[:RS.PROBE_ROWS_PER_CELL])
        banks["target_move"] = {key: value[np.array(rows)] for key, value in fresh.items()}
        return banks

    def dump(path, states, group, index, modes, tagged, events, seconds, readouts):
        """The usual policy files, and beside them what the proposal network says at every cell."""
        pick = lambda tree: jax.tree_util.tree_map(lambda x: x[index], tree)  # noqa: E731
        copies = len(tagged) // len(index)
        many = lambda tree: L.repeat(pick(tree), copies)  # noqa: E731
        RS.dump_policies(path, states, many(group["banks"]), many(group["audits"]), many(group["probes"]),
                         many(group["keys"]), modes, readouts, tagged, events, seconds)
        move, q, width, raise_only = (np.asarray(x, np.float64) for x in
                                      readouts["proposal"](states, jnp.arange(100, dtype=jnp.int32)))
        np.savez_compressed(str(path) + "_proposal.npz", move=move, q1=q, width=width, raise_only=raise_only)

    for group_name, task_names in GROUPS.items():
        t0 = time.time()
        runs, datasets, banks, audits, probes = [], [], [], [], []
        for task_name in task_names:
            task = checks.TASKS[task_name]
            for seed in data_seeds:
                for arm in arms:
                    data, bank = RS.to_device(load(task, arm, seed)), banks_for(task, arm, seed)
                    for train_seed in train_seeds:
                        runs.append({"task": task_name, "setting": RS.setting_name(task_name, "one_hot"),
                                     "host_input": "one_hot", "data_seed": seed, "arm": arm, "train_seed": train_seed,
                                     "arm_number": arm_number(arm)})
                        datasets.append(data)
                        banks.append(RS.to_device(bank["reference"]))
                        audits.append(RS.to_device(bank["audit"]))
                        probes.append(RS.to_device(bank["target_move"]))
        keys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(r["train_seed"] + 12345), r["data_seed"]) for r in runs])
        group = {"observation": "one_hot", "runs": runs, "data": L.stack(datasets), "banks": L.stack(banks),
                 "audits": L.stack(audits), "probes": L.stack(probes), "keys": keys}
        n_rows = int(datasets[0]["s"].shape[0])
        segment, readouts = L.make_segment("one_hot", n_rows), L.make_readouts("one_hot")
        print(f"[{group_name}] {len(runs)} runs, {n_rows} rows each, data ready in {time.time() - t0:.0f}s", flush=True)

        if stage == "baselines":
            t0 = time.time()
            native = jnp.full((len(runs),), L.NATIVE)
            states, events, snapshot = RS.train(RS.initial_states(group), group["data"], keys, native, group["banks"],
                                                runs, 0, segment, readouts, snapshot_at=L.REFRESH_STEPS[0])
            seconds = time.time() - t0
            RS.save_tree(out / f"stage1_{group_name}_snapshot.npz", snapshot)
            tagged = [dict(run, method="baseline") for run in runs]
            dump(out / f"stage1_{group_name}", states, group, jnp.arange(len(runs)), native, tagged, events, seconds,
                 readouts)
            print(f"[{group_name}] baselines trained in {seconds:.0f}s", flush=True)
            continue

        wanted = METHOD_ARMS[group_name]
        keep = [i for i, run in enumerate(runs) if run["arm"] in wanted]
        index = jnp.asarray(keep)
        pick = lambda tree: jax.tree_util.tree_map(lambda x: x[index], tree)  # noqa: E731
        snapshot = pick(RS.load_tree(out / f"stage1_{group_name}_snapshot.npz", RS.initial_states(group)))
        kept_runs = [runs[i] for i in keep]
        for chunk in METHOD_CHUNKS:
            many = lambda tree: L.repeat(pick(tree), len(chunk))  # noqa: E731
            modes = jnp.concatenate([jnp.full((len(keep),), L.ARM_NUMBERS[method]) for method in chunk])
            tagged = [dict(run, method=method) for method in chunk for run in kept_runs]
            t0 = time.time()
            states, events, _ = RS.train(L.repeat(snapshot, len(chunk)), many(group["data"]), many(keys), modes,
                                         many(group["banks"]), tagged, L.REFRESH_STEPS[0], segment, readouts)
            seconds = time.time() - t0
            dump(out / f"stage2_{group_name}__{chunk[0]}", states, group, index, modes, tagged, events, seconds, readouts)
            print(f"[{group_name}] {' and '.join(chunk)} trained in {seconds:.0f}s on {len(keep)} runs each", flush=True)
    (out / f"COMPLETE_{stage}.json").write_text(json.dumps({"finished_unix": time.time()}))
    print("done", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("data", "baselines", "methods"), required=True)
    parser.add_argument("--label", default="thin_dev2")
    parser.add_argument("--smoke", action="store_true", help="one dataset seed, one training seed, three arms")
    args = parser.parse_args()
    if args.stage == "data":
        build_data()
    else:
        train(args.label, args.stage, args.smoke)


if __name__ == "__main__":
    main()
