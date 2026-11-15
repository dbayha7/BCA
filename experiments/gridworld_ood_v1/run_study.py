"""Train the study policies. See PROTOCOL.md; this file only trains and saves arrays.

    python -B run_study.py --label smoke_1 --stage smoke
    python -B run_study.py --label dev_v1 --stage stage1
    python -B run_study.py --label dev_v1 --stage stage2      (reads the G0 verdicts written by evaluate_study.py)
    python -B run_study.py --label dev_v3 --study v3 --stage stage1     (Amendment 3: the three method changes)
    python -B run_study.py --label dev_v4 --study v4 --stage stage1     (Amendment 4: the support penalty alone)
    python -B run_study.py --label dev_v5 --study v5 --stage stage1     (Amendment 5: state-blind calibrated width)
    python -B run_study.py --label confirm_v1 --study v6 --stage stage1 (Amendment 6: confirmation on untouched data)
    python -B run_study.py --label confirm_v2 --study v7 --stage stage1 (Amendment 7: second fresh-data block)
    python -B run_study.py --label confirm_v3 --study v8 --stage stage1 (Amendment 8: Version A on nine datasets)
    python -B run_study.py --label confirm_v4 --study v9 --stage stage1 (Amendment 9: competence checked per dataset)
    python -B run_study.py --label confirm_v5 --study v10 --stage stage1 (Amendment 10: both environments)

Stage 1 trains every baseline (10,000 shared updates, snapshot, 15,000 more).
Stage 2 restores the snapshots and trains each method together with its own shuffled control.
A label is written once: an existing START file for a stage stops the run.
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import collect  # noqa: E402
import environment as env  # noqa: E402
import learner as L  # noqa: E402
import unseen_move_checks as checks  # noqa: E402
from calibration.reference import WBCPConfig, freeze_reference  # noqa: E402

BOTTLENECK_TASKS = ["bottleneck_exit45", "bottleneck_exit35", "bottleneck_exit25"]
# group -> (what the networks see, task settings). The Bottleneck is run with both archived
# input encodings; see Amendment 1 in PROTOCOL.md.
GROUPS = {"honeypot": ("one_hot", ["honeypot"]),
          "bottleneck_one_hot": ("one_hot", BOTTLENECK_TASKS),
          "bottleneck_coordinates": ("coordinates", BOTTLENECK_TASKS)}


def setting_name(task_name, observation):
    """The unit the gates are judged on: a task setting, plus the input encoding for the Bottleneck."""
    return task_name if task_name == "honeypot" else f"{task_name}_{observation}"


R2_TASKS = ["bottleneck_r2_exit45", "bottleneck_r2_exit35", "bottleneck_r2_exit25"]
ORIGINAL = (("bca", "shuffled_width"),)
# Amendment 3: the original method again, then each change, every one trained beside its own shuffled control.
CHANGED = (("bca", "shuffled_width"), ("support", "support_shuffled"),
           ("support_onesided", "support_onesided_shuffled"), ("full", "full_shuffled"),
           ("onesided", "onesided_shuffled"))
# Amendment 4: change 1 again, for side-by-side tables, and the support penalty without the calibrated width.
SUPPORT_ONLY = (("support", "support_shuffled"), ("support_only", "support_only_shuffled"))
# Amendment 5: change 1 and its control again, and change 1 with the calibrated width read at shuffled states.
BLIND_WIDTH = (("support", "support_shuffled"), ("support_blind_width",))
# Amendment 6: confirmation. Version A (best in development) and version B (also changes Q values),
# each beside its own shuffled control, on datasets and training seeds never used before.
CONFIRM = (("support_blind_width", "support_shuffled"), ("full", "full_shuffled"))
VALIDATED = {"groups": {"honeypot": ("one_hot", ["honeypot"]), "bottleneck_r2_one_hot": ("one_hot", R2_TASKS)},
             "group_data": {"honeypot": ("dev", (5101, 5102, 5103)),
                            "bottleneck_r2_one_hot": ("dev2", (5201, 5202, 5203))},
             "data_seeds": None, "data_label": None, "train_seeds": (20, 21, 22), "arms": collect.ARMS}
# Study v1: the Honeypot and Bottleneck revision 1. Study v2: Bottleneck revision 2 (Amendment 2), fresh datasets.
# Study v3: the validated environments only, each on the datasets it was validated with (Amendment 3).
# Study v4: the same environments and datasets, for the support-penalty-only arm (Amendment 4).
# Study v5: the same again, for change 1 with a state-blind calibrated width (Amendment 5).
STUDIES = {
    "v1": {"groups": GROUPS, "data_seeds": (5101, 5102, 5103), "train_seeds": (20, 21, 22),
           "arms": collect.ARMS, "data_label": "dev", "group_data": {}, "method_chunks": ORIGINAL},
    "v2": {"groups": {"bottleneck_r2_one_hot": ("one_hot", R2_TASKS),
                      "bottleneck_r2_coordinates": ("coordinates", R2_TASKS)},
           "data_seeds": (5201, 5202, 5203), "train_seeds": (20, 21, 22), "arms": collect.ARMS, "data_label": "dev2",
           "group_data": {}, "method_chunks": ORIGINAL},
    "v3": {**VALIDATED, "method_chunks": CHANGED},
    "v4": {**VALIDATED, "method_chunks": SUPPORT_ONLY},
    "v5": {**VALIDATED, "method_chunks": BLIND_WIDTH},
    "v6": {**VALIDATED, "method_chunks": CONFIRM, "train_seeds": (30, 31, 32),
           "group_data": {"honeypot": ("confirm_h", (7101, 7102, 7103)),
                          "bottleneck_r2_one_hot": ("confirm_b", (7101, 7102, 7103))}},
    # Amendment 7: a second fresh-data block for the same two versions. Only the seeds differ from v6.
    "v7": {**VALIDATED, "method_chunks": CONFIRM, "train_seeds": (40, 41, 42),
           "group_data": {"honeypot": ("confirm2_h", (7201, 7202, 7203)),
                          "bottleneck_r2_one_hot": ("confirm2_b", (7201, 7202, 7203))}},
    # Amendment 8: Version A alone on nine fresh Honeypot datasets.
    "v8": {**VALIDATED, "groups": {"honeypot": ("one_hot", ["honeypot"])},
           "method_chunks": (("support_blind_width", "support_shuffled"),), "train_seeds": (50, 51, 52),
           "group_data": {"honeypot": ("confirm3_h", (7301, 7302, 7303, 7304, 7305, 7306, 7307, 7308, 7309))}},
    # Amendment 9: the same, with baseline competence checked per dataset. Twelve candidates, nine used.
    "v9": {**VALIDATED, "groups": {"honeypot": ("one_hot", ["honeypot"])},
           "method_chunks": (("support_blind_width", "support_shuffled"),), "train_seeds": (60, 61, 62),
           "datasets_wanted": 9,
           "group_data": {"honeypot": ("confirm4_h", tuple(range(7401, 7413)))}},
    # Amendment 10: the procedure of Amendment 9 in both environments.
    "v10": {**VALIDATED, "method_chunks": (("support_blind_width", "support_shuffled"),),
            "train_seeds": (70, 71, 72), "datasets_wanted": 9,
            "group_data": {"honeypot": ("confirm5_h", tuple(range(7501, 7513))),
                           "bottleneck_r2_one_hot": ("confirm5_b", tuple(range(7501, 7513)))}},
}
SMOKE = {"data_seeds": (7,), "train_seeds": (0, 1), "arms": ("complete", "targeted"), "data_label": None,
         "group_data": {}, "smoke": True}
FROZEN_FIELDS = ("threshold", "support_floor", "width_cap", "width_ref", "width_ref_support")
BANK_ROWS = 1280
REFERENCE_OFFSET, AUDIT_OFFSET, PROBE_OFFSET = 100_000, 200_000, 300_000
PROBE_ROWS_PER_CELL = 64
FINAL_STEP = 25_000
SOURCES = ("PROTOCOL.md", "learner.py", "run_study.py", "environment.py", "collect.py", "unseen_move_checks.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def to_device(data):
    return {"s": jnp.asarray(data["s"], jnp.int32), "a": jnp.asarray(data["a"], jnp.float32),
            "r": jnp.asarray(data["r"].reshape(-1), jnp.float32), "s2": jnp.asarray(data["s2"], jnp.int32),
            "done": jnp.asarray(data["done"].reshape(-1), jnp.float32)}


def subsample(data, rows, seed_words):
    keep = np.sort(np.random.default_rng(np.random.SeedSequence(seed_words)).choice(len(data["s"]), rows, replace=False))
    return {key: value[keep] for key, value in data.items()}


def load_training_data(task, arm, seed, label):
    """Study data come from disk and must match the manifest. Smoke data are generated and checked here."""
    if label is None:
        data, _ = collect.generate(task, arm, seed)
        complete = data if arm == "complete" else collect.generate(task, "complete", seed)[0]
        report = checks.check_dataset(task, arm, data, None if arm == "complete" else complete)
        assert report["passed"], (task.name, arm, seed)
        return data, collect.digest(data)
    manifest = json.loads((HERE / "receipts" / f"{label}_manifest.json").read_text())
    name = f"{task.name}_{arm}_seed{seed}.npz"
    entry = next(e for e in manifest["datasets"] if e["file"] == name)
    data = checks.load(HERE / "data" / label / name)
    assert collect.digest(data) == entry["array_sha256"] and entry["checks_passed"], name
    return data, entry["array_sha256"]


def make_banks(task, arm, seed):
    """Fresh rows from the same collector and the same gap. Never used for a gradient step."""
    banks = {}
    for name, offset in (("reference", REFERENCE_OFFSET), ("audit", AUDIT_OFFSET)):
        fresh, _ = collect.generate(task, arm, seed + offset)
        complete = fresh if arm == "complete" else collect.generate(task, "complete", seed + offset)[0]
        assert checks.check_dataset(task, arm, fresh, None if arm == "complete" else complete)["passed"]
        banks[name] = subsample(fresh, BANK_ROWS, [seed, offset])
    # Real transitions of commands that could execute the target move, at the decision cells.
    # They exist only in complete-style collection; here they are a yardstick for the width.
    fresh, _ = collect.generate(task, "complete", seed + PROBE_OFFSET)
    rows = []
    for cell, moves in collect.targeted_moves(task).items():
        here = np.flatnonzero(fresh["s"] == cell)
        could = [i for i in here if any(task.could_execute(fresh["a"][i], d) for d in moves)]
        assert len(could) >= PROBE_ROWS_PER_CELL, (task.name, cell, len(could))
        rows.extend(could[:PROBE_ROWS_PER_CELL])
    banks["target_move"] = {key: value[np.array(rows)] for key, value in fresh.items()}
    return banks


def save_tree(path, tree):
    leaves = jax.tree_util.tree_leaves(tree)
    np.savez_compressed(path, **{f"leaf_{i:03d}": np.asarray(x) for i, x in enumerate(leaves)})


def load_tree(path, template):
    leaves, treedef = jax.tree_util.tree_flatten(template)
    with np.load(path, allow_pickle=False) as saved:
        loaded = [jnp.asarray(saved[f"leaf_{i:03d}"]) for i in range(len(leaves))]
    assert all(a.shape == b.shape and a.dtype == b.dtype for a, b in zip(loaded, leaves))
    return jax.tree_util.tree_unflatten(treedef, loaded)


def raised_width(width, logit, floor, cap):
    """NumPy copy of learner.consumed_width for the support-aware arms."""
    return width + np.maximum(floor - logit, 0.0) / L.LN10 * np.maximum(cap - width, 0.0)


def refresh(states, banks, keys, modes, runs, step, readouts):
    """Set each run's reference from its reference bank. Eager, on the host, as in the archive.

    Besides the calibrated threshold, four numbers are frozen here for the changes of Amendment 3,
    all from the same 1,280 held-out rows:
      support_floor      the 1st percentile of the support score of the recorded commands
      width_cap          half the range of the plain one-step targets
      width_ref          the 90th percentile of the width at the recorded commands
      width_ref_support  the same percentile of the support-aware width
    Every arm gets all four; the switches of an arm decide which it uses."""
    refresh_keys = jax.vmap(lambda k: jax.random.fold_in(k, 1_000_000 + step))(keys)
    eta, residual, plain, logit = (np.asarray(x) for x in readouts["heldout"](states, banks, refresh_keys, modes))
    units = np.asarray(states["unit"])
    host_scale = jax.tree_util.tree_map(np.asarray, states["scale"])
    frozen = {name: np.zeros(len(runs), np.float32) for name in FROZEN_FIELDS}
    frozen["support_floor"][:] = -1e9
    ready, events = np.zeros(len(runs), bool), []
    for i, run in enumerate(runs):
        # A run may carry its own arm number (the thin arms of Amendment 11). The three original arms keep 0, 1, 2.
        arm_number = run["arm_number"] if "arm_number" in run else collect.ARMS.index(run["arm"])
        seed_words = np.array([run["train_seed"], run["data_seed"], arm_number, step], np.uint32)
        reference, valid, diagnostics = freeze_reference(L.take(host_scale, i), units[i], eta[i], residual[i],
                                                        seed_words, WBCPConfig())
        usable = bool(valid and diagnostics.get("certified") and np.isfinite(float(reference.threshold)))
        width = np.zeros_like(eta[i])
        if usable:
            ready[i] = True
            frozen["threshold"][i] = float(reference.threshold)
            width = frozen["threshold"][i] * np.maximum(eta[i], 1e-6) * units[i]
            frozen["support_floor"][i] = np.quantile(logit[i], L.SUPPORT_QUANTILE)
            frozen["width_cap"][i] = 0.5 * (plain[i].max() - plain[i].min())
            frozen["width_ref"][i] = np.quantile(width, L.WIDTH_REFERENCE_QUANTILE)
            raised = raised_width(width, logit[i], frozen["support_floor"][i], frozen["width_cap"][i])
            frozen["width_ref_support"][i] = np.quantile(raised, L.WIDTH_REFERENCE_QUANTILE)
        events.append({"run": i, "step": step, "usable": usable, "unit": float(units[i]),
                       **{name: float(values[i]) for name, values in frozen.items()},
                       "lambda_hat": float(diagnostics.get("lambda_hat", np.nan)) if usable else None,
                       "lambda_hpd": float(diagnostics.get("lambda_hpd", np.nan)) if usable else None,
                       "mean_width_on_reference_bank": float(width.mean()),
                       "mean_abs_residual_on_reference_bank": float(np.abs(residual[i]).mean()),
                       "coverage_on_reference_bank": float((np.abs(residual[i]) <= width).mean()),
                       "median_support_score_on_reference_bank": float(np.median(logit[i]))})
    reference = {"scale": states["scale"], "unit": states["unit"], "ready": jnp.asarray(ready),
                 "support": states["support"], **{name: jnp.asarray(values) for name, values in frozen.items()}}
    return dict(states, reference=reference), events


def train(states, data, keys, modes, banks, runs, start, segment, readouts, snapshot_at=None):
    """Advance from update `start` to 25,000, refreshing after 10,000, 15,000 and 20,000."""
    events, snapshot, at = [], None, start
    for stop in (*L.REFRESH_STEPS, FINAL_STEP):
        if stop <= at:
            continue
        states = segment(states, data, keys, modes, jnp.arange(at + 1, stop + 1, dtype=jnp.int32))
        at = stop
        if stop in L.REFRESH_STEPS:
            states, new_events = refresh(states, banks, keys, modes, runs, stop, readouts)
            events.extend(new_events)
        if stop == snapshot_at:
            snapshot = states
    return states, events, snapshot


def dump_policies(path, states, banks, audit_banks, probe_banks, keys, modes, readouts, runs, events, seconds):
    """Save, per policy: its command at every cell, and at every cell and corner command the two
    critics, the original width, the width its arm consumes, and the frozen support score."""
    cells, corners = jnp.arange(100, dtype=jnp.int32), jnp.asarray(env.CORNERS, jnp.float32)
    commands = np.asarray(readouts["policy"](states, cells), np.float64)
    q1, q2, width, consumed, support = (np.asarray(x, np.float64)
                                        for x in readouts["probe"](states, cells, corners, modes))
    audit_keys = jax.vmap(lambda k: jax.random.fold_in(k, 2_000_000))(keys)
    floors = np.asarray(states["reference"]["support_floor"])
    diagnostics = {}
    for name, bank in (("reference_bank", banks), ("audit_bank", audit_banks), ("target_move_rows", probe_banks)):
        w, res, score = (np.asarray(x) for x in readouts["audit"](states, bank, audit_keys, modes))
        diagnostics[name] = [{"rows": int(w.shape[1]), "mean_width": float(w[i].mean()),
                              "mean_abs_residual": float(np.abs(res[i]).mean()),
                              "mean_signed_residual": float(res[i].mean()),
                              "coverage": float((np.abs(res[i]) <= w[i]).mean()),
                              "share_scored_unsupported": float((score[i] < floors[i]).mean()),
                              "median_support_score": float(np.median(score[i]))} for i in range(len(runs))]
    assert np.all(np.isfinite(commands)) and np.all(np.abs(commands) <= 1.0)
    assert all(np.all(np.isfinite(x)) for x in (q1, q2, width, consumed, support))
    np.savez_compressed(path.with_suffix(".npz"), commands=commands, q1=q1, q2=q2, width=width, consumed=consumed,
                        support=support)
    records = []
    for i, run in enumerate(runs):
        records.append({**run, "index": i, "final_step": int(states["step"][i]),
                        "residual_unit": float(states["unit"][i]),
                        "reference_ready": bool(states["reference"]["ready"][i]),
                        "reference_threshold": float(states["reference"]["threshold"][i]),
                        "reference": {name: float(states["reference"][name][i]) for name in FROZEN_FIELDS},
                        "refresh_events": [e for e in events if e["run"] == i],
                        **{name: rows[i] for name, rows in diagnostics.items()}})
    path.with_suffix(".json").write_text(json.dumps({"seconds": seconds, "runs": records}, indent=1, allow_nan=False))


def build_group(group, config):
    observation, task_names = config["groups"][group]
    data_label, data_seeds = config["group_data"].get(group, (config["data_label"], config["data_seeds"]))
    runs, datasets, banks, audits, probes, digests = [], [], [], [], [], {}
    cache = {}
    # The smoke test uses one task setting per group: the Honeypot, and the Bottleneck's primary exit value.
    for task_name in ([task_names[min(1, len(task_names) - 1)]] if config.get("smoke") else task_names):
        task = checks.TASKS[task_name]
        for data_seed in data_seeds:
            for arm in config["arms"]:
                data, digest = load_training_data(task, arm, data_seed, data_label)
                digests[f"{task_name}_{arm}_seed{data_seed}"] = digest
                cache[(task_name, data_seed, arm)] = (to_device(data), make_banks(task, arm, data_seed))
                for train_seed in config["train_seeds"]:
                    runs.append({"task": task_name, "setting": setting_name(task_name, observation),
                                 "host_input": observation, "data_seed": data_seed, "arm": arm,
                                 "train_seed": train_seed})
    for run in runs:
        data, bank = cache[(run["task"], run["data_seed"], run["arm"])]
        datasets.append(data)
        banks.append(to_device(bank["reference"]))
        audits.append(to_device(bank["audit"]))
        probes.append(to_device(bank["target_move"]))
    keys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(r["train_seed"] + 12345), r["data_seed"]) for r in runs])
    return {"observation": observation, "runs": runs, "data": L.stack(datasets), "banks": L.stack(banks),
            "audits": L.stack(audits), "probes": L.stack(probes), "keys": keys, "digests": digests,
            "n_rows": int(datasets[0]["s"].shape[0])}


def select_datasets(verdict, wanted):
    """Amendment 9. A dataset is valid if every one of its complete-data baseline runs is competent (G0).
    The study uses the first `wanted` valid dataset seeds in increasing order; with fewer, it uses none."""
    by_seed = {}
    for row in verdict["per_run"]:
        by_seed.setdefault(row["data_seed"], []).append(bool(row["passed"]))
    valid = [seed for seed in sorted(by_seed) if all(by_seed[seed])]
    enough = len(valid) >= wanted
    return {"candidates": sorted(by_seed), "dropped_not_competent": [s for s in sorted(by_seed) if s not in valid],
            "selected": valid[:wanted] if enough else [], "valid_not_needed": valid[wanted:] if enough else [],
            "enough_valid_datasets": enough, "wanted": wanted}


def initial_states(group):
    obs_dim = L.OBSERVATIONS[group["observation"]][1]
    return L.stack([L.init_state(run["train_seed"], obs_dim) for run in group["runs"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--stage", choices=("smoke", "stage1", "stage2"), required=True)
    parser.add_argument("--study", choices=tuple(STUDIES), default="v1")
    args = parser.parse_args()
    out = HERE / "results" / args.label
    out.mkdir(parents=True, exist_ok=True)
    start_file = out / f"START_{args.stage}.json"
    if start_file.exists():
        sys.exit(f"{start_file} exists: this stage was already started under this label")
    config = dict(STUDIES[args.study])
    if args.stage == "smoke":
        config.update(SMOKE)
    start_file.write_text(json.dumps({
        "stage": args.stage, "study": args.study, "started_unix": time.time(), "python": platform.python_version(),
        "jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
        "matmul_precision": "float32", "source_sha256": {name: sha256(HERE / name) for name in SOURCES},
        "calibration_sha256": {name: sha256(ROOT / "calibration" / name) for name in ("network.py", "reference.py", "wbcp.py")},
        "data_seeds": config["data_seeds"], "group_data": config["group_data"],
        "train_seeds": config["train_seeds"], "arms": config["arms"], "method_chunks": config["method_chunks"],
        "settings_of_the_changes": {"support_quantile": L.SUPPORT_QUANTILE,
                                    "ramp": "cap reached one decade below the floor, not saturated",
                                    "width_reference_quantile": L.WIDTH_REFERENCE_QUANTILE, "kappa": L.KAPPA,
                                    "support_learning_rate": L.SUPPORT_LR}}, indent=1))

    for group_name in config["groups"]:
        t0 = time.time()
        group = build_group(group_name, config)
        segment = L.make_segment(group["observation"], group["n_rows"])
        readouts = L.make_readouts(group["observation"])
        runs, n = group["runs"], len(group["runs"])
        print(f"[{group_name}] {n} runs, {group['n_rows']} rows each, data ready in {time.time() - t0:.0f}s", flush=True)
        if args.stage in ("smoke", "stage1"):
            t0 = time.time()
            native = jnp.full((n,), L.NATIVE)
            states, events, snapshot = train(initial_states(group), group["data"], group["keys"],
                                             native, group["banks"], runs, 0, segment, readouts,
                                             snapshot_at=L.REFRESH_STEPS[0])
            seconds = time.time() - t0
            save_tree(out / f"stage1_{group_name}_snapshot.npz", snapshot)
            tagged = [dict(run, method="baseline") for run in runs]
            dump_policies(out / f"stage1_{group_name}", states, group["banks"], group["audits"], group["probes"],
                          group["keys"], native, readouts, tagged, events, seconds)
            (out / f"stage1_{group_name}_datasets.json").write_text(json.dumps(group["digests"], indent=1))
            print(f"[{group_name}] baselines trained in {seconds:.0f}s", flush=True)
        if args.stage in ("smoke", "stage2"):
            if args.stage == "stage2":
                verdicts = json.loads((out / "g0.json").read_text())
                wanted = config.get("datasets_wanted")
                if wanted is None:
                    keep = [i for i, run in enumerate(runs) if verdicts[run["setting"]]["passed"]]
                else:   # Amendment 9: baseline competence is judged per dataset, from the baselines alone
                    choice = {setting: select_datasets(verdicts[setting], wanted)
                              for setting in sorted({run["setting"] for run in runs})}
                    (out / f"selected_datasets_{group_name}.json").write_text(json.dumps(choice, indent=1))
                    keep = [i for i, run in enumerate(runs) if run["data_seed"] in choice[run["setting"]]["selected"]]
            else:
                keep = list(range(n))
            if not keep:
                print(f"[{group_name}] no task setting passed G0: nothing trained", flush=True)
                continue
            index = jnp.asarray(keep)
            pick = lambda tree: jax.tree_util.tree_map(lambda x: x[index], tree)  # noqa: E731
            snapshot = pick(load_tree(out / f"stage1_{group_name}_snapshot.npz", initial_states(group)))
            kept_runs = [runs[i] for i in keep]
            for chunk in config["method_chunks"]:
                many = lambda tree: L.repeat(pick(tree), len(chunk))  # noqa: E731
                modes = jnp.concatenate([jnp.full((len(keep),), L.ARM_NUMBERS[method]) for method in chunk])
                tagged = [dict(run, method=method) for method in chunk for run in kept_runs]
                t0 = time.time()
                states, events, _ = train(L.repeat(snapshot, len(chunk)), many(group["data"]), many(group["keys"]),
                                          modes, many(group["banks"]), tagged, L.REFRESH_STEPS[0], segment, readouts)
                seconds = time.time() - t0
                dump_policies(out / f"stage2_{group_name}__{chunk[0]}", states, many(group["banks"]),
                              many(group["audits"]), many(group["probes"]), many(group["keys"]), modes, readouts,
                              tagged, events, seconds)
                print(f"[{group_name}] {' and '.join(chunk)} trained in {seconds:.0f}s", flush=True)
    (out / f"COMPLETE_{args.stage}.json").write_text(json.dumps({"finished_unix": time.time()}))
    print("done", flush=True)


if __name__ == "__main__":
    main()
