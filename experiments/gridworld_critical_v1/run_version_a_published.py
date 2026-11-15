"""Version A and its shuffled control on the published map (PROTOCOL.md, Amendment 4).

    python -B run_version_a_published.py --label smoke_version_a --baselines pilot_alpha_1 --seeds 7
    python -B run_version_a_published.py --label version_a_published --baselines published_validation_2 --seeds 6401 6402 6403

Every run starts from the baseline's saved state after 10,000 updates (the snapshot of the
baselines run named by --baselines) and trains to 25,000 with its method switched on, with the
reference refreshed after 15,000 and 20,000. This is stage 2 of
experiments/gridworld_ood_v1/run_study.py. The baseline is continued the same way as a check that
this runner reproduces it. A label is written once.
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

import run_published_alpha as A  # noqa: E402  (also puts the shared learner and the repository root on the path)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import critical_env as E  # noqa: E402
import learner as L  # noqa: E402
import published_checks as checks  # noqa: E402
import run_baselines as B  # noqa: E402
import run_published as P  # noqa: E402

VERSION_A, CONTROL, BASELINE = "support_blind_width", "support_shuffled", "baseline"
CHUNKS = ((VERSION_A, CONTROL), (BASELINE,))          # a method with its control, as in the earlier testbed
SOURCES = A.SOURCES + ("run_version_a_published.py",)


def load_tree(path, template):
    """Restore a saved stack of states (run_study.load_tree of the earlier testbed)."""
    leaves, treedef = jax.tree_util.tree_flatten(template)
    with np.load(path, allow_pickle=False) as saved:
        loaded = [jnp.asarray(saved[f"leaf_{i:03d}"]) for i in range(len(leaves))]
    assert all(a.shape == b.shape and a.dtype == b.dtype for a, b in zip(loaded, leaves))
    return jax.tree_util.tree_unflatten(treedef, loaded)


def continue_training(states, data, keys, modes, banks, runs, segment, readouts, start):
    """From update `start` to 25,000, refreshing the reference after every refresh step still ahead."""
    events, at = [], start
    for stop in (*L.REFRESH_STEPS, B.FINAL_STEP):
        if stop <= at:
            continue
        states = segment(states, data, keys, modes, jnp.arange(at + 1, stop + 1, dtype=jnp.int32))
        at = stop
        if stop in L.REFRESH_STEPS:
            states, new_events = B.refresh(states, banks, keys, modes, runs, stop, readouts)
            events.extend(new_events)
    return states, events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--baselines", required=True, help="results label of the baselines run to start from")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="dataset seeds to train on")
    args = parser.parse_args()
    source = HERE / "results" / args.baselines
    assert (source / "COMPLETE.json").exists(), f"{args.baselines} is not a finished baselines run"
    base_start = json.loads((source / "START.json").read_text())
    alpha, data_label = float(base_start["alpha"]), base_start["data_label"]
    learner_hash = B.sha256(B.SHARED / "learner.py")
    assert learner_hash == base_start["source_sha256"]["shared learner.py"], "the learner changed since the baselines"
    out = HERE / "results" / args.label
    out.mkdir(parents=True, exist_ok=True)
    start_file = out / "START.json"
    if start_file.exists():
        sys.exit(f"{start_file} exists: this label was already started")
    A.set_alpha(alpha)
    assert A.objective_uses_alpha(alpha), "the policy's objective does not use the baselines' alpha"
    task = E.published()
    base_runs = json.loads((source / f"baselines_{task.name}.json").read_text())["runs"]
    keep = [i for i, run in enumerate(base_runs) if run["data_seed"] in args.seeds]
    assert keep and sorted({base_runs[i]["data_seed"] for i in keep}) == sorted(args.seeds), "a seed has no baselines"
    start_file.write_text(json.dumps({
        "baselines": args.baselines, "data_label": data_label, "data_seeds": args.seeds, "alpha": alpha,
        "chunks": CHUNKS, "started_unix": time.time(), "start_step": L.REFRESH_STEPS[0], "kappa": L.KAPPA,
        "python": platform.python_version(), "jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
        "matmul_precision": "float32", "task": task.name, "runs_per_method": len(keep),
        "source_sha256": {**{name: B.sha256(HERE / name) for name in SOURCES}, "shared learner.py": learner_hash},
        "calibration_sha256": {name: B.sha256(B.ROOT / "calibration" / name)
                               for name in ("network.py", "reference.py", "wbcp.py")},
    }, indent=1))

    t0 = time.time()
    template = L.stack([L.init_state(run["train_seed"], 100, B.ACTION_DIM) for run in base_runs])
    index = jnp.asarray(keep)
    snapshot = jax.tree_util.tree_map(lambda x: x[index], load_tree(source / f"snapshot_{task.name}.npz", template))
    assert np.all(np.asarray(snapshot["step"]) == L.REFRESH_STEPS[0]), "the snapshot is not at the first refresh"
    kept = [{k: base_runs[i][k] for k in ("task", "noise", "sharpness", "alpha", "data_seed", "recipe", "arm",
                                         "arm_number", "train_seed")} for i in keep]
    datasets, banks, audits, cache = [], [], [], {}
    for run in kept:
        pair = (run["recipe"], run["data_seed"])
        if pair not in cache:
            data, _ = P.load_training_data(task, run["recipe"], run["data_seed"], data_label)
            bank = P.make_banks(task, run["recipe"], run["data_seed"])
            cache[pair] = tuple(B.to_device(x) for x in (data, bank["reference"], bank["audit"]))
        for target, value in zip((datasets, banks, audits), cache[pair]):
            target.append(value)
    data, banks, audits = L.stack(datasets), L.stack(banks), L.stack(audits)
    keys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(r["train_seed"] + 12345), r["data_seed"]) for r in kept])
    readouts = L.make_readouts("one_hot")
    segment = L.make_segment("one_hot", int(datasets[0]["s"].shape[0]))
    print(f"[{task.name}] alpha {alpha:g}: {len(kept)} runs per method from {args.baselines}, "
          f"ready in {time.time() - t0:.0f}s", flush=True)
    for chunk in CHUNKS:
        many = lambda tree: L.repeat(tree, len(chunk))  # noqa: E731
        modes = jnp.concatenate([jnp.full((len(kept),), L.ARM_NUMBERS[method]) for method in chunk])
        tagged = [dict(run, method=method) for method in chunk for run in kept]
        t0 = time.time()
        states, events = continue_training(many(snapshot), many(data), many(keys), modes, many(banks), tagged,
                                           segment, readouts, L.REFRESH_STEPS[0])
        seconds = time.time() - t0
        B.dump(out / f"methods_{task.name}__{chunk[0]}", states, many(audits), many(keys), modes, readouts, tagged,
               events, seconds)
        print(f"[{task.name}] {' and '.join(chunk)} trained in {seconds:.0f}s", flush=True)
    assert L.ALPHA == alpha
    (out / "COMPLETE.json").write_text(json.dumps({"finished_unix": time.time()}))
    print("done", flush=True)


if __name__ == "__main__":
    main()
