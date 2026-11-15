"""Train the plain baselines for the validation in PROTOCOL.md. Baselines only: no BCA, no control.

    python -B run_baselines.py --label smoke_1 --smoke
    python -B run_baselines.py --label dev_validation --data-label dev

The learner is the shared one in experiments/gridworld_ood_v1/learner.py, with five action
dimensions. A label is written once: an existing START file stops the run.
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
SHARED = ROOT / "experiments" / "gridworld_ood_v1"
for folder in (HERE, SHARED, ROOT):
    sys.path.insert(0, str(folder))
sys.dont_write_bytecode = True

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import critical_checks as checks  # noqa: E402
import critical_collect as C  # noqa: E402
import critical_env as E  # noqa: E402
import learner as L  # noqa: E402
from calibration.reference import WBCPConfig, freeze_reference  # noqa: E402

STUDY = {"train_seeds": (20, 21, 22), "arms": C.ARMS}        # dataset seeds and tasks come from the manifest
SMOKE = {"data_seeds": (7,), "train_seeds": (0, 1), "arms": C.ARMS}
# Amendment 1's pilot: complete data only, a dataset seed outside the study, six training seeds.
PILOT = {"data_seeds": (7,), "train_seeds": (0, 1, 2, 3, 4, 5), "arms": ("complete",)}
BANK_ROWS, REFERENCE_OFFSET, AUDIT_OFFSET = 1280, 100_000, 200_000
FINAL_STEP, ACTION_DIM = 25_000, E.N_MOVES
FROZEN_FIELDS = ("threshold", "support_floor", "width_cap", "width_ref", "width_ref_support")
SOURCES = ("PROTOCOL.md", "PUBLISHED_SETUP.md", "critical_env.py", "critical_collect.py", "critical_checks.py",
           "run_baselines.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def to_device(data):
    return {"s": jnp.asarray(data["s"], jnp.int32), "a": jnp.asarray(data["a"], jnp.float32),
            "r": jnp.asarray(data["r"].reshape(-1), jnp.float32), "s2": jnp.asarray(data["s2"], jnp.int32),
            "done": jnp.asarray(data["done"].reshape(-1), jnp.float32)}


def subsample(data, rows, seed_words):
    keep = np.sort(np.random.default_rng(np.random.SeedSequence(seed_words)).choice(len(data["s"]), rows, replace=False))
    return {key: value[keep] for key, value in data.items()}


def checked(task, arm, seed, held_out=False):
    """A freshly generated dataset that passes the acceptance checks. Held-out rows are excused
    from the two support counts, which only protect training data."""
    data = C.generate(task, arm, seed)
    complete = data if arm == "complete" else C.generate(task, "complete", seed)
    report = checks.check_dataset(task, arm, data, None if arm == "complete" else complete)
    assert checks.acceptable_as_held_out(report) if held_out else report["passed"], (task.name, arm, seed, report["checks"])
    return data


def load_training_data(task, arm, seed, label):
    """Study data come from disk and must match the manifest. Smoke data are generated and checked here."""
    if label is None:
        data = checked(task, arm, seed)
        return data, C.digest(data)
    manifest = json.loads((HERE / "receipts" / f"{label}_manifest.json").read_text())
    name = f"{task.name}_{arm}_seed{seed}.npz"
    entry = next(e for e in manifest["datasets"] if e["file"] == name)
    data = checks.load(HERE / "data" / label / name)
    assert C.digest(data) == entry["array_sha256"] and entry["checks_passed"], name
    return data, entry["array_sha256"]


def make_banks(task, arm, seed):
    """Fresh rows from the same collector and the same gap. Never used for a gradient step."""
    return {name: subsample(checked(task, arm, seed + offset, held_out=True), BANK_ROWS, [seed, offset])
            for name, offset in (("reference", REFERENCE_OFFSET), ("audit", AUDIT_OFFSET))}


def save_tree(path, tree):
    leaves = jax.tree_util.tree_leaves(tree)
    np.savez_compressed(path, **{f"leaf_{i:03d}": np.asarray(x) for i, x in enumerate(leaves)})


def refresh(states, banks, keys, modes, runs, step, readouts):
    """Set each run's reference from its reference bank, as in experiments/gridworld_ood_v1/run_study.py."""
    refresh_keys = jax.vmap(lambda k: jax.random.fold_in(k, 1_000_000 + step))(keys)
    eta, residual, plain, logit = (np.asarray(x) for x in readouts["heldout"](states, banks, refresh_keys, modes))
    units = np.asarray(states["unit"])
    host_scale = jax.tree_util.tree_map(np.asarray, states["scale"])
    frozen = {name: np.zeros(len(runs), np.float32) for name in FROZEN_FIELDS}
    frozen["support_floor"][:] = -1e9
    ready, events = np.zeros(len(runs), bool), []
    for i, run in enumerate(runs):
        # A run may carry its own number (run_published.py's recipes are not in C.ARMS).
        arm_number = run["arm_number"] if "arm_number" in run else C.ARMS.index(run["arm"])
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
            raised = width + np.maximum(frozen["support_floor"][i] - logit[i], 0.0) / L.LN10 * np.maximum(
                frozen["width_cap"][i] - width, 0.0)
            frozen["width_ref_support"][i] = np.quantile(raised, L.WIDTH_REFERENCE_QUANTILE)
        events.append({"run": i, "step": step, "usable": usable, "unit": float(units[i]),
                       **{name: float(values[i]) for name, values in frozen.items()},
                       "mean_width_on_reference_bank": float(width.mean()),
                       "mean_abs_residual_on_reference_bank": float(np.abs(residual[i]).mean()),
                       "coverage_on_reference_bank": float((np.abs(residual[i]) <= width).mean())})
    reference = {"scale": states["scale"], "unit": states["unit"], "ready": jnp.asarray(ready),
                 "support": states["support"], **{name: jnp.asarray(values) for name, values in frozen.items()}}
    return dict(states, reference=reference), events


def train(states, data, keys, modes, banks, runs, segment, readouts):
    """25,000 updates, with a refresh after 10,000, 15,000 and 20,000. Returns the snapshot taken
    right after the first refresh, from which any later method would start."""
    events, snapshot, at = [], None, 0
    for stop in (*L.REFRESH_STEPS, FINAL_STEP):
        states = segment(states, data, keys, modes, jnp.arange(at + 1, stop + 1, dtype=jnp.int32))
        at = stop
        if stop in L.REFRESH_STEPS:
            states, new_events = refresh(states, banks, keys, modes, runs, stop, readouts)
            events.extend(new_events)
        if stop == L.REFRESH_STEPS[0]:
            snapshot = states
    return states, events, snapshot


def dump(path, states, audits, keys, modes, readouts, runs, events, seconds):
    """Per policy: its preferences at every cell, and at every cell and corner command the two
    critics, the calibrated width, the width with the support raise, and the support score."""
    cells, corners = jnp.arange(100, dtype=jnp.int32), jnp.asarray(E.CORNERS, jnp.float32)
    commands = np.asarray(readouts["policy"](states, cells), np.float64)
    q1, q2, width, consumed, support = (np.asarray(x, np.float64)
                                        for x in readouts["probe"](states, cells, corners, modes))
    audit_keys = jax.vmap(lambda k: jax.random.fold_in(k, 2_000_000))(keys)
    w, res, _ = (np.asarray(x) for x in readouts["audit"](states, audits, audit_keys, modes))
    assert np.all(np.isfinite(commands)) and np.all(np.abs(commands) <= 1.0)
    assert all(np.all(np.isfinite(x)) for x in (q1, q2, width, consumed, support))
    np.savez_compressed(path.with_suffix(".npz"), commands=commands, q1=q1, q2=q2, width=width, consumed=consumed,
                        support=support)
    records = [{**run, "index": i, "final_step": int(states["step"][i]),
                "reference_ready": bool(states["reference"]["ready"][i]),
                "reference": {name: float(states["reference"][name][i]) for name in FROZEN_FIELDS},
                "refresh_events": [e for e in events if e["run"] == i],
                "audit_bank": {"mean_width": float(w[i].mean()), "mean_abs_residual": float(np.abs(res[i]).mean()),
                               "coverage": float((np.abs(res[i]) <= w[i]).mean())}} for i, run in enumerate(runs)]
    path.with_suffix(".json").write_text(json.dumps({"seconds": seconds, "runs": records}, indent=1, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--data-label", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--tasks", nargs="+", help="task names; for --smoke and --pilot only")
    args = parser.parse_args()
    if sum([args.smoke, args.pilot, args.data_label is not None]) != 1:
        sys.exit("give exactly one of --smoke, --pilot or --data-label")
    if args.data_label is not None:
        manifest = json.loads((HERE / "receipts" / f"{args.data_label}_manifest.json").read_text())
        config = dict(STUDY, data_seeds=tuple(manifest["seeds"]),
                      tasks=sorted({e["task"] for e in manifest["datasets"]}, reverse=True))
    else:
        config = dict(PILOT if args.pilot else SMOKE, tasks=list(args.tasks or checks.ORIGINAL_TASKS))
    out = HERE / "results" / args.label
    out.mkdir(parents=True, exist_ok=True)
    start_file = out / "START.json"
    if start_file.exists():
        sys.exit(f"{start_file} exists: this label was already started")
    start_file.write_text(json.dumps({
        "smoke": args.smoke, "pilot": args.pilot, "data_label": args.data_label, "started_unix": time.time(),
        "python": platform.python_version(), "jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
        "matmul_precision": "float32", "method": "baseline only", **config,
        "source_sha256": {**{name: sha256(HERE / name) for name in SOURCES}, "shared learner.py": sha256(SHARED / "learner.py")},
        "calibration_sha256": {name: sha256(ROOT / "calibration" / name) for name in ("network.py", "reference.py", "wbcp.py")},
    }, indent=1))

    readouts = L.make_readouts("one_hot")
    for task_name in config["tasks"]:
        task, t0 = checks.TASKS[task_name], time.time()
        runs, datasets, banks, audits, digests = [], [], [], [], {}
        for data_seed in config["data_seeds"]:
            for arm in config["arms"]:
                data, digest = load_training_data(task, arm, data_seed, args.data_label)
                digests[f"{task_name}_{arm}_seed{data_seed}"] = digest
                bank = make_banks(task, arm, data_seed)
                for train_seed in config["train_seeds"]:
                    runs.append({"task": task_name, "noise": task.noise, "sharpness": task.beta,
                                 "data_seed": data_seed, "arm": arm,
                                 "train_seed": train_seed, "method": "baseline"})
                    datasets.append(to_device(data))
                    banks.append(to_device(bank["reference"]))
                    audits.append(to_device(bank["audit"]))
        n = len(runs)
        data, banks, audits = L.stack(datasets), L.stack(banks), L.stack(audits)
        keys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(r["train_seed"] + 12345), r["data_seed"]) for r in runs])
        states = L.stack([L.init_state(r["train_seed"], 100, ACTION_DIM) for r in runs])
        modes = jnp.full((n,), L.NATIVE)
        segment = L.make_segment("one_hot", int(datasets[0]["s"].shape[0]))
        print(f"[{task_name}] {n} runs, {int(datasets[0]['s'].shape[0])} rows each, data ready in {time.time() - t0:.0f}s", flush=True)
        t0 = time.time()
        states, events, snapshot = train(states, data, keys, modes, banks, runs, segment, readouts)
        seconds = time.time() - t0
        save_tree(out / f"snapshot_{task_name}.npz", snapshot)
        dump(out / f"baselines_{task_name}", states, audits, keys, modes, readouts, runs, events, seconds)
        (out / f"datasets_{task_name}.json").write_text(json.dumps(digests, indent=1))
        print(f"[{task_name}] baselines trained in {seconds:.0f}s", flush=True)
    (out / "COMPLETE.json").write_text(json.dumps({"finished_unix": time.time()}))
    print("done", flush=True)


if __name__ == "__main__":
    main()
