"""Version C on the published map (PROTOCOL.md, Amendment 6; BCA_IW_PLAN.md).

    python -B run_version_c_published.py --label smoke_version_c --baselines pilot_alpha_1 --seeds 7
    python -B run_version_c_published.py --label version_c_published --baselines published_validation_2 --seeds 6401 6402 6403

Every run starts from the baseline's saved state after 10,000 updates, like Version A. Then:
  1. the weight classifier is trained 10,000 updates against Version C's comparison rows;
  2. the reference is frozen again at update 10,000, now with the table of importance-weighted thresholds;
  3. the run trains to 25,000 with version_c's update step, refreshing after 15,000 and 20,000.
A label is written once.
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

import run_version_a_published as RA  # noqa: E402  (also puts the shared learner and the repository root on the path)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import critical_env as E  # noqa: E402
import learner as L  # noqa: E402
import run_baselines as B  # noqa: E402
import run_published as P  # noqa: E402
import version_c as VC  # noqa: E402
from calibration.reference import WBCPConfig, freeze_reference  # noqa: E402

METHOD = "version_c"
PREFIT_UPDATES = 10_000
SOURCES = RA.SOURCES + ("version_c.py", "run_version_c_published.py", "BCA_IW_PLAN.md")


def refresh(states, banks, keys, runs, step, heldout):
    """run_baselines.refresh, with the scores taken against Version C's own target and the table of
    importance-weighted thresholds added. The two width levels of the one-sided arms are not used
    by Version C and stay zero."""
    modes = jnp.full((len(runs),), VC.VERSION_C)
    refresh_keys = jax.vmap(lambda k: jax.random.fold_in(k, 1_000_000 + step))(keys)
    eta, residual, plain, logit = (np.asarray(x) for x in heldout(states, banks, refresh_keys, modes))
    units = np.asarray(states["unit"])
    host_scale = jax.tree_util.tree_map(np.asarray, states["scale"])
    frozen = {name: np.zeros(len(runs), np.float32) for name in B.FROZEN_FIELDS}
    frozen["support_floor"][:] = -1e9
    blank = {name: np.asarray(value) for name, value in VC.blank_iw_fields().items()}
    tables = {name: [] for name in VC.IW_FIELDS}
    ready, events = np.zeros(len(runs), bool), []
    for i, run in enumerate(runs):
        words = np.array([run["train_seed"], run["data_seed"], run["arm_number"], step], np.uint32)
        reference, valid, diagnostics = freeze_reference(L.take(host_scale, i), units[i], eta[i], residual[i], words,
                                                        WBCPConfig())
        usable = bool(valid and diagnostics.get("certified") and np.isfinite(float(reference.threshold)))
        fields, record = blank, None
        if usable:
            ready[i] = True
            frozen["threshold"][i] = float(reference.threshold)
            frozen["support_floor"][i] = np.quantile(logit[i], L.SUPPORT_QUANTILE)
            frozen["width_cap"][i] = 0.5 * (plain[i].max() - plain[i].min())
            fields, record = VC.iw_fields(VC.scores_of(eta[i], residual[i], units[i]), logit[i], words,
                                          float(reference.threshold))
        for name in VC.IW_FIELDS:
            tables[name].append(np.asarray(fields[name]))
        events.append({"run": i, "step": step, "usable": usable, "unit": float(units[i]),
                       **{name: float(values[i]) for name, values in frozen.items()}, "table": record})
    reference = {"scale": states["scale"], "unit": states["unit"], "ready": jnp.asarray(ready),
                 "support": states["support"], **{name: jnp.asarray(values) for name, values in frozen.items()},
                 **{name: jnp.asarray(np.stack(values)) for name, values in tables.items()}}
    return dict(states, reference=reference), events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
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
    RA.A.set_alpha(alpha)
    assert RA.A.objective_uses_alpha(alpha), "the policy's objective does not use the baselines' alpha"
    task = E.published()
    base_runs = json.loads((source / f"baselines_{task.name}.json").read_text())["runs"]
    keep = [i for i, run in enumerate(base_runs) if run["data_seed"] in args.seeds]
    assert keep and sorted({base_runs[i]["data_seed"] for i in keep}) == sorted(args.seeds), "a seed has no baselines"
    start_file.write_text(json.dumps({
        "baselines": args.baselines, "data_label": data_label, "data_seeds": args.seeds, "alpha": alpha,
        "method": METHOD, "prefit_updates": PREFIT_UPDATES, "grid_points": VC.GRID_POINTS, "kappa": L.KAPPA,
        "started_unix": time.time(), "start_step": L.REFRESH_STEPS[0],
        "python": platform.python_version(), "jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
        "matmul_precision": "float32", "task": task.name, "runs": len(keep),
        "source_sha256": {**{name: B.sha256(HERE / name) for name in SOURCES}, "shared learner.py": learner_hash},
        "calibration_sha256": {name: B.sha256(B.ROOT / "calibration" / name)
                               for name in ("network.py", "reference.py", "wbcp.py")},
    }, indent=1))

    t0 = time.time()
    template = L.stack([L.init_state(run["train_seed"], 100, B.ACTION_DIM) for run in base_runs])
    index = jnp.asarray(keep)
    states = jax.tree_util.tree_map(lambda x: x[index], RA.load_tree(source / f"snapshot_{task.name}.npz", template))
    first = L.REFRESH_STEPS[0]
    assert np.all(np.asarray(states["step"]) == first), "the snapshot is not at the first refresh"
    runs = [dict({k: base_runs[i][k] for k in ("task", "noise", "sharpness", "alpha", "data_seed", "recipe", "arm",
                                              "arm_number", "train_seed")}, method=METHOD) for i in keep]
    datasets, banks, audits, cache = [], [], [], {}
    for run in runs:
        pair = (run["recipe"], run["data_seed"])
        if pair not in cache:
            data, _ = P.load_training_data(task, run["recipe"], run["data_seed"], data_label)
            bank = P.make_banks(task, run["recipe"], run["data_seed"])
            cache[pair] = tuple(B.to_device(x) for x in (data, bank["reference"], bank["audit"]))
        for target, value in zip((datasets, banks, audits), cache[pair]):
            target.append(value)
    rows = int(datasets[0]["s"].shape[0])
    data, banks, audits = L.stack(datasets), L.stack(banks), L.stack(audits)
    keys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(r["train_seed"] + 12345), r["data_seed"]) for r in runs])
    modes = jnp.full((len(runs),), VC.VERSION_C)
    mine, shared = VC.make_readouts("one_hot"), L.make_readouts("one_hot")
    segment = VC.make_segment("one_hot", rows)

    # 1. the weight classifier, 2. the reference with its table at the starting update
    states = VC.with_blank_iw(states)
    states = VC.make_weight_fit("one_hot", rows)(states, data, keys, jnp.arange(1, PREFIT_UPDATES + 1, dtype=jnp.int32))
    states, events = refresh(states, banks, keys, runs, first, mine["heldout"])
    print(f"[{task.name}] alpha {alpha:g}: {len(runs)} Version C runs from {args.baselines}; classifier and table "
          f"ready in {time.time() - t0:.0f}s; usable tables {int(np.asarray(states['reference']['iw_ready']).sum())}",
          flush=True)

    # 3. train to 25,000
    t0, at = time.time(), first
    for stop in (*L.REFRESH_STEPS, B.FINAL_STEP):
        if stop <= at:
            continue
        states = segment(states, data, keys, modes, jnp.arange(at + 1, stop + 1, dtype=jnp.int32))
        at = stop
        if stop in L.REFRESH_STEPS:
            states, new_events = refresh(states, banks, keys, runs, stop, mine["heldout"])
            events.extend(new_events)
    seconds = time.time() - t0
    assert L.ALPHA == alpha
    native = jnp.full((len(runs),), L.NATIVE)             # the shared read-outs know only the shared modes
    B.dump(out / f"methods_{task.name}__{METHOD}", states, audits, keys, native, shared, runs, events, seconds)
    cells, corners = jnp.arange(100, dtype=jnp.int32), jnp.asarray(E.CORNERS, jnp.float32)
    charge, flagged, logm = (np.asarray(x) for x in mine["probe"](states, cells, corners))
    assert np.all(np.isfinite(charge)) and np.all(np.isfinite(logm))
    np.savez_compressed(out / f"charge_{task.name}__{METHOD}.npz", charge=charge, cannot_certify=flagged, log_mass=logm,
                        iw_ready=np.asarray(states["reference"]["iw_ready"]))
    print(f"[{task.name}] Version C trained in {seconds:.0f}s", flush=True)
    (out / "COMPLETE.json").write_text(json.dumps({"finished_unix": time.time()}))
    print("done", flush=True)


if __name__ == "__main__":
    main()
