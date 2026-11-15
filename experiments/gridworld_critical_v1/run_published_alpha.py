"""Baselines on the paper's data recipes with a chosen cloning weight (PROTOCOL.md, Amendment 3).
Baselines only: no BCA, no control.

    python -B run_published_alpha.py --label pilot_alpha_0.5 --alpha 0.5 --pilot
    python -B run_published_alpha.py --label published_validation_2 --alpha 0.5 --data-label published2

TD3+BC's policy minimizes  -(alpha / mean|Q|) * Q(s, policy)  +  (policy - recorded command)^2.
The shared learner reads alpha from its constant ALPHA when the update is first compiled. This
runner sets that constant before anything is compiled, records it, and checks that the objective
uses it. Everything else is run_published.py's. A label is written once.
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

import run_published as P  # noqa: E402  (also puts the shared learner and the repository root on the path)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import critical_env as E  # noqa: E402
import learner as L  # noqa: E402
import published_checks as checks  # noqa: E402
import published_collect as C  # noqa: E402
import run_baselines as B  # noqa: E402

PUBLISHED_ALPHA = 2.5
STUDY = {"train_seeds": (20, 21, 22)}                              # dataset seeds come from the manifest
PILOT = {"data_seeds": (7,), "train_seeds": (0, 1, 2, 3, 4, 5)}    # Amendment 3: a seed outside every study
SOURCES = P.SOURCES + ("run_published_alpha.py",)


def set_alpha(alpha):
    """Set the learner's cloning weight. Must happen before any update is compiled."""
    assert alpha > 0
    L.ALPHA = float(alpha)


def objective_uses_alpha(alpha):
    """True if the policy's objective, with no penalty, is exactly
    -(alpha / mean|Q|) * mean(Q) + mean((policy - recorded)^2) for the learner as it is set now."""
    state = L.init_state(0, 100, B.ACTION_DIM)
    cells = np.arange(0, 100, 7)
    obs = jnp.asarray(np.eye(100, dtype=np.float32)[cells])
    recorded = jnp.asarray(E.CORNERS[cells % E.N_MOVES], jnp.float32)
    got = float(L.actor_objective(state["actor"], state["q1"], L.calibrator(100), state["reference"], obs, obs,
                                  recorded, 0.0))
    action = L.actor_apply(state["actor"], obs)
    q = np.asarray(L.critic_apply(state["q1"], obs, action), np.float64)
    wanted = -(alpha / (np.abs(q).mean() + 1e-7)) * q.mean() + float(np.mean((np.asarray(action) - np.asarray(recorded)) ** 2))
    return bool(abs(got - wanted) < 1e-5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--data-label", default=None)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.pilot == (args.data_label is not None):
        sys.exit("give exactly one of --pilot or --data-label")
    if args.pilot:
        config = dict(PILOT)
    else:
        manifest = json.loads((HERE / "receipts" / f"{args.data_label}_manifest.json").read_text())
        config = dict(STUDY, data_seeds=tuple(manifest["seeds_passing_every_check"]))        # gate W0
    out = HERE / "results" / args.label
    out.mkdir(parents=True, exist_ok=True)
    start_file = out / "START.json"
    if start_file.exists():
        sys.exit(f"{start_file} exists: this label was already started")
    set_alpha(args.alpha)
    assert objective_uses_alpha(args.alpha), "the policy's objective does not use the requested alpha"
    task = E.published()
    start_file.write_text(json.dumps({
        "pilot": args.pilot, "data_label": args.data_label, "started_unix": time.time(),
        "alpha": args.alpha, "published_alpha": PUBLISHED_ALPHA, "objective_uses_alpha": True,
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
            data, digest = P.load_training_data(task, recipe, data_seed, args.data_label)
            digests[checks.file_name(task, recipe, data_seed)] = digest
            bank = P.make_banks(task, recipe, data_seed)
            for train_seed in config["train_seeds"]:
                runs.append({"task": task.name, "noise": task.noise, "sharpness": task.beta, "alpha": args.alpha,
                             "data_seed": data_seed, "recipe": recipe, "arm": recipe,
                             "arm_number": P.RECIPE_NUMBER[recipe], "train_seed": train_seed, "method": "baseline"})
                datasets.append(B.to_device(data))
                banks.append(B.to_device(bank["reference"]))
                audits.append(B.to_device(bank["audit"]))
    n = len(runs)
    data, banks, audits = L.stack(datasets), L.stack(banks), L.stack(audits)
    keys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(r["train_seed"] + 12345), r["data_seed"]) for r in runs])
    states = L.stack([L.init_state(r["train_seed"], 100, B.ACTION_DIM) for r in runs])
    modes = jnp.full((n,), L.NATIVE)
    segment = L.make_segment("one_hot", int(datasets[0]["s"].shape[0]))
    print(f"[{task.name}] alpha {args.alpha:g}: {n} runs, {int(datasets[0]['s'].shape[0])} rows each, "
          f"data ready in {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    states, events, snapshot = B.train(states, data, keys, modes, banks, runs, segment, readouts)
    seconds = time.time() - t0
    assert L.ALPHA == float(args.alpha)
    B.save_tree(out / f"snapshot_{task.name}.npz", snapshot)
    B.dump(out / f"baselines_{task.name}", states, audits, keys, modes, readouts, runs, events, seconds)
    (out / f"datasets_{task.name}.json").write_text(json.dumps(digests, indent=1))
    print(f"[{task.name}] baselines trained in {seconds:.0f}s", flush=True)
    (out / "COMPLETE.json").write_text(json.dumps({"finished_unix": time.time()}))
    print("done", flush=True)


if __name__ == "__main__":
    main()
