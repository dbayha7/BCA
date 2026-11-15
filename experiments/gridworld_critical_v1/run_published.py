"""Train the plain baselines on the paper's data recipes (PROTOCOL.md, Amendment 2). Baselines
only: no BCA, no control.

    python -B run_published.py --label smoke_published_1 --smoke
    python -B run_published.py --label published_validation --data-label published

Training, the refresh of the learner's reference and the saved read-outs are run_baselines.py's.
A label is written once: an existing START file stops the run.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import run_baselines as B  # noqa: E402  (also puts the shared learner and the repository root on the path)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import critical_env as E  # noqa: E402
import learner as L  # noqa: E402
import published_checks as checks  # noqa: E402
import published_collect as C  # noqa: E402

STUDY = {"train_seeds": (20, 21, 22)}                 # dataset seeds come from the manifest
SMOKE = {"data_seeds": (7,), "train_seeds": (0, 1)}
RECIPE_NUMBER = {recipe: 10 + i for i, recipe in enumerate(C.RECIPES)}       # for the refresh's random words
SOURCES = ("PROTOCOL.md", "PUBLISHED_SETUP.md", "critical_env.py", "published_collect.py", "published_checks.py",
           "run_published.py", "run_baselines.py")


def checked(task, recipe, seed):
    """A freshly generated dataset that passes every acceptance check."""
    data = C.generate(task, recipe, seed)
    optimal = C.generate(task, "optimal", seed) if recipe == "mixed" else None
    report = checks.check_dataset(task, recipe, data, optimal)
    assert report["passed"], (recipe, seed, [name for name, ok in report["checks"].items() if not ok])
    return data


def load_training_data(task, recipe, seed, label):
    """Study data come from disk and must match the manifest. Smoke data are generated and checked here."""
    if label is None:
        data = checked(task, recipe, seed)
        return data, C.digest(data)
    manifest = json.loads((HERE / "receipts" / f"{label}_manifest.json").read_text())
    name = checks.file_name(task, recipe, seed)
    entry = next(e for e in manifest["datasets"] if e["file"] == name)
    data = checks.load(HERE / "data" / label / name)
    assert C.digest(data) == entry["array_sha256"] and entry["checks_passed"], name
    return data, entry["array_sha256"]


def make_banks(task, recipe, seed):
    """Fresh rows of the same recipe. Never used for a gradient step."""
    return {name: B.subsample(checked(task, recipe, seed + offset), B.BANK_ROWS, [seed, offset])
            for name, offset in (("reference", B.REFERENCE_OFFSET), ("audit", B.AUDIT_OFFSET))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--data-label", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke == (args.data_label is not None):
        sys.exit("give exactly one of --smoke or --data-label")
    if args.smoke:
        config = dict(SMOKE)
    else:
        manifest = json.loads((HERE / "receipts" / f"{args.data_label}_manifest.json").read_text())
        config = dict(STUDY, data_seeds=tuple(manifest["seeds_passing_every_check"]))        # gate W0
    out = HERE / "results" / args.label
    out.mkdir(parents=True, exist_ok=True)
    start_file = out / "START.json"
    if start_file.exists():
        sys.exit(f"{start_file} exists: this label was already started")
    task = E.published()
    start_file.write_text(json.dumps({
        "smoke": args.smoke, "data_label": args.data_label, "started_unix": time.time(),
        "python": platform.python_version(), "jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
        "matmul_precision": "float32", "method": "baseline only", "task": task.name, "recipes": list(C.RECIPES),
        **config,
        "source_sha256": {**{name: B.sha256(HERE / name) for name in SOURCES},
                          "shared learner.py": B.sha256(B.SHARED / "learner.py")},
        "calibration_sha256": {name: B.sha256(B.ROOT / "calibration" / name)
                               for name in ("network.py", "reference.py", "wbcp.py")},
    }, indent=1))

    readouts, t0 = L.make_readouts("one_hot"), time.time()
    runs, datasets, banks, audits, digests = [], [], [], [], {}
    for data_seed in config["data_seeds"]:
        for recipe in C.RECIPES:
            data, digest = load_training_data(task, recipe, data_seed, args.data_label)
            digests[checks.file_name(task, recipe, data_seed)] = digest
            bank = make_banks(task, recipe, data_seed)
            for train_seed in config["train_seeds"]:
                runs.append({"task": task.name, "noise": task.noise, "sharpness": task.beta, "data_seed": data_seed,
                             "recipe": recipe, "arm": recipe, "arm_number": RECIPE_NUMBER[recipe],
                             "train_seed": train_seed, "method": "baseline"})
                datasets.append(B.to_device(data))
                banks.append(B.to_device(bank["reference"]))
                audits.append(B.to_device(bank["audit"]))
    n = len(runs)
    data, banks, audits = L.stack(datasets), L.stack(banks), L.stack(audits)
    keys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(r["train_seed"] + 12345), r["data_seed"]) for r in runs])
    states = L.stack([L.init_state(r["train_seed"], 100, B.ACTION_DIM) for r in runs])
    modes = jnp.full((n,), L.NATIVE)
    segment = L.make_segment("one_hot", int(datasets[0]["s"].shape[0]))
    print(f"[{task.name}] {n} runs, {int(datasets[0]['s'].shape[0])} rows each, data ready in {time.time() - t0:.0f}s",
          flush=True)
    t0 = time.time()
    states, events, snapshot = B.train(states, data, keys, modes, banks, runs, segment, readouts)
    seconds = time.time() - t0
    B.save_tree(out / f"snapshot_{task.name}.npz", snapshot)
    B.dump(out / f"baselines_{task.name}", states, audits, keys, modes, readouts, runs, events, seconds)
    (out / f"datasets_{task.name}.json").write_text(json.dumps(digests, indent=1))
    print(f"[{task.name}] baselines trained in {seconds:.0f}s", flush=True)
    (out / "COMPLETE.json").write_text(json.dumps({"finished_unix": time.time()}))
    print("done", flush=True)


if __name__ == "__main__":
    main()
