"""Run one declared BCA experiment; no queue, automatic retry or score selection."""

from pathlib import Path
import argparse
import contextlib
import gzip
import json
import os
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parent
from runtime.config import read_config, resolve, typed
from runtime.provenance import sha, source_files, source_identity, write


@contextlib.contextmanager
def gpu_lock(path):
    import fcntl

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("GPU lock is held by another worker: " + str(path))
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def prepare(row, objects):
    import h5py

    cache = Path(os.environ["D4RL_DATASET_DIR"]) / row["cache"]["filename"]
    if not cache.is_file():
        raise FileNotFoundError("Required D4RL dataset: " + str(cache))
    if sha(cache) != row["cache"]["sha256"]:
        raise ValueError("Dataset SHA256 mismatch: " + str(cache))
    if row["host"] == "iql":
        driver, m, options = objects
        args = driver.resolved_arguments(options, m)[0]
        return m.H.load_data(args)
    runner, args, spec, protocol = objects
    with h5py.File(cache, "r") as f:
        raw = {
            k: f[k][()]
            for k in ("observations", "actions", "rewards", "terminals", "timeouts")
        }
    bound = row.get("action_bound", row.get("action", {}).get("max_action", 1.0))
    horizon = row.get(
        "max_episode_steps", 200 if row["environment"].startswith("pen-") else 1000
    )
    if row["host"] == "cql":
        return runner.prepare(
            raw, args, spec, protocol, max_action=bound, max_episode_steps=horizon
        )
    return runner.prepare(
        raw,
        args,
        spec,
        protocol,
        max_action=bound,
        max_episode_steps=horizon,
        raw_identity={
            "kind": "cached_hdf5",
            "dataset": row["environment"],
            "path": str(cache),
            "sha256": sha(cache),
            "size_bytes": cache.stat().st_size,
        },
    )


def execute(row, objects, out, prepared):
    if row["host"] == "iql":
        from dataclasses import asdict

        driver, m, options = objects
        from runtime.iql_checkpoint import checkpoint_counts as iql_counts

        base = driver.IWRuntime if row["mode"] == "shared" else driver.GroupRuntime

        class CheckedRuntime(base):

            def checkpoint(self, label):
                entry = super().checkpoint(label)
                payload = (self.output / entry["path"]).read_bytes()
                shared = row["mode"] == "shared"
                step = int(self.carry.step) if shared else int(self.carry[2])
                counts = iql_counts(
                    m.serialization.msgpack_restore(payload),
                    row["mode"],
                    step,
                    [int(x) for x in self.actor_applied],
                    [int(x) for x in self.cal_applied] if shared else [],
                )
                if m.serialization.to_bytes(self.carry) != payload:
                    raise ValueError("IQL checkpoint observer changed source state.")
                write(
                    self.output / (label + "_decoded.json"),
                    dict(
                        accepted=True,
                        counters=counts,
                        checkpoint_sha256=sha(self.output / entry["path"]),
                    ),
                )
                return entry

        write(
            out / "manifest.json",
            {
                "schema": getattr(driver, "SCHEMA", "corl-iql-one-group-v1"),
                "options": asdict(options),
                "declared": (
                    driver.declaration(options, m)
                    if row["mode"] == "shared"
                    else {
                        "args": asdict(driver.resolved_arguments(options, m)[0]),
                        "arms": [
                            asdict(a) for a in driver.resolved_arguments(options, m)[1]
                        ],
                    }
                ),
                "source_sha256": source_files(),
            },
        )
        original = m.H.load_data
        m.H.load_data = lambda args: prepared
        try:
            result = driver.execute_group(
                options, m, out, runtime_factory=CheckedRuntime
            )
        finally:
            m.H.load_data = original
        if result["status"] != "complete":
            raise RuntimeError("IQL group failed; preserved result.json has details.")
        driver.verify_result(out, result, options, m.np)
        return result
    from flax import serialization
    from runtime.validation import checkpoint_counts, verify_events

    runner, args, spec, protocol = objects
    checkpoints = []

    def checkpoint(step, carry):
        payload = serialization.to_bytes(
            {"state": carry[1], "training_rng": carry[0], "step": carry[2]}
        )
        path = out / f"checkpoint_{step}.msgpack"
        with path.open("xb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        counts = checkpoint_counts(
            serialization.msgpack_restore(payload),
            row["host"],
            step,
            getattr(args, "policy_freq", 2),
        )
        if (
            serialization.to_bytes(
                {"state": carry[1], "training_rng": carry[0], "step": carry[2]}
            )
            != payload
        ):
            raise ValueError("Checkpoint observer changed source state.")
        checkpoints.append(
            {"step": step, "path": path.name, "sha256": sha(path), "counters": counts}
        )

    with gzip.open(out / "events.jsonl.gz", "xt", encoding="utf8") as journal:

        def event(record):
            journal.write(json.dumps(record, allow_nan=False) + "\n")
            journal.flush()
            if record["kind"] in ("periodic", "final"):
                print(json.dumps(record, allow_nan=False), flush=True)

        result = runner.run_prepared(
            args, spec, protocol, prepared, on_event=event, on_checkpoint=checkpoint
        )
    if result["steps_completed"] != protocol.num_updates:
        raise ValueError("Incomplete update budget.")
    verify_events(protocol, result["events"], result["evaluations"])
    if [c["step"] for c in checkpoints] != row["expected_counts"]["checkpoint_steps"]:
        raise ValueError("Incomplete checkpoint schedule.")
    record = dict(
        completed=True,
        run_id=row["run_id"],
        steps_completed=result["steps_completed"],
        evaluations=result["evaluations"],
        checkpoints=checkpoints,
        events_sha256=sha(out / "events.jsonl.gz"),
        expected_counts=row["expected_counts"],
    )
    write(out / "result.json", record)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--algorithm", choices=("iql", "cql", "td3_bc", "rebrac"), required=True
    )
    p.add_argument(
        "--dataset", help="Dataset name from the algorithm config; use --list."
    )
    p.add_argument("--method", choices=("host", "bca"))
    p.add_argument("--seed", type=int)
    p.add_argument(
        "--list",
        action="store_true",
        help="List methods and datasets without loading JAX.",
    )
    p.add_argument(
        "--print-config",
        action="store_true",
        help="Print the complete resolved declaration.",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Validate types and schedules; no data, model or simulator.",
    )
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate cached data and preparation; zero learner updates.",
    )
    p.add_argument("--output", type=Path)
    p.add_argument("--data-dir", type=Path, default=Path.home() / ".d4rl/datasets")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument(
        "--lock",
        default=os.environ.get(
            "BCA_GPU_LOCK", str(Path.home() / ".cache/bca/gpu.lock")
        ),
    )
    opt = p.parse_args(argv)
    config_path = ROOT / "configs" / (opt.algorithm + ".yaml")
    config = read_config(config_path)
    if opt.list:
        print(
            json.dumps(
                {
                    "algorithm": opt.algorithm,
                    "methods": ["host", "bca"],
                    "datasets": list(config["datasets"]),
                },
                indent=2,
            )
        )
        return
    if opt.method is None or opt.seed is None or opt.dataset is None:
        p.error("--dataset, --method and --seed are required; use --list.")
    output = (opt.output or ROOT / "runs/config-inspection").resolve()
    row = resolve(config_path, opt.method, opt.seed, output, opt.dataset)
    if opt.print_config:
        print(json.dumps(row, indent=2, allow_nan=False))
        return
    if not opt.check and opt.output is None:
        p.error("--output must name a new directory.")
    if sys.flags.optimize:
        raise RuntimeError(
            "Python -O is unsupported: validation assertions must remain enabled."
        )
    os.environ["JAX_PLATFORMS"] = "cpu" if opt.check or opt.prepare_only else opt.device
    os.environ["D4RL_DATASET_DIR"] = str(opt.data_dir.expanduser().resolve())
    for key, value in {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "WANDB_MODE": "disabled",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MUJOCO_PY_FORCE_CPU": "1",
        "MUJOCO_GL": "egl",
        "D4RL_SUPPRESS_IMPORT_ERROR": "1",
    }.items():
        os.environ.setdefault(key, value)
    mujoco_bin = str(
        Path(
            os.environ.get(
                "MUJOCO_PY_MUJOCO_PATH", str(Path.home() / ".mujoco/mujoco210")
            )
        )
        / "bin"
    )
    if sys.platform == "linux" and mujoco_bin not in os.environ.get(
        "LD_LIBRARY_PATH", ""
    ).split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            filter(None, [mujoco_bin, os.environ.get("LD_LIBRARY_PATH", "")])
        )
        os.execv(sys.executable, [sys.executable, *sys.argv])
    from runtime.environment import setup

    setup()
    objects = typed(row)
    if opt.check:
        print(
            json.dumps({"valid": True, "run_id": row["run_id"], "training_updates": 0})
        )
        return
    guard = (
        gpu_lock(opt.lock)
        if opt.device == "cuda" and (not opt.prepare_only)
        else contextlib.nullcontext()
    )
    with guard:
        output.mkdir(parents=True, exist_ok=False)
        write(output / "resolved.json", row)
        write(output / "source.json", source_identity())
        inventory = source_files()
        for name, digest in inventory.items():
            destination = output / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
            if sha(destination) != digest:
                raise ValueError("Source changed during snapshot: " + name)
        try:
            prepared = prepare(row, objects)
            write(
                output / "preparation.json",
                {"accepted": True, "metadata": prepared.metadata, "learner_updates": 0},
            )
            if opt.prepare_only:
                write(
                    output / "exit.json",
                    dict(
                        completed=False,
                        preparation_complete=True,
                        learner_updates=0,
                        exit_code=0,
                    ),
                )
                return
            previous = Path.cwd()
            try:
                os.chdir(output)
                execute(row, objects, output, prepared)
            finally:
                os.chdir(previous)
            if source_files() != inventory:
                raise ValueError("Source/configuration changed during the run.")
            write(output / "exit.json", {"completed": True, "exit_code": 0})
        except BaseException as error:
            write(
                output / "failure.json",
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise


if __name__ == "__main__":
    main()
