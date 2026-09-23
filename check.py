"""Validate every host/BCA configuration; --runtime constructs typed CPU protocols."""

import argparse
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from runtime.config import ALGORITHMS, METHODS, ROOT, _yaml, read_config, resolve, typed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true")
    opt = parser.parse_args()
    if opt.runtime:
        os.environ.update(
            JAX_PLATFORMS="cpu",
            MUJOCO_PY_FORCE_CPU="1",
            WANDB_MODE="disabled",
            D4RL_SUPPRESS_IMPORT_ERROR="1",
        )
        location = str(
            Path(
                os.environ.get(
                    "MUJOCO_PY_MUJOCO_PATH", str(Path.home() / ".mujoco/mujoco210")
                )
            )
            / "bin"
        )
        if sys.platform == "linux" and location not in os.environ.get(
            "LD_LIBRARY_PATH", ""
        ).split(":"):
            os.environ["LD_LIBRARY_PATH"] = ":".join(
                filter(None, [location, os.environ.get("LD_LIBRARY_PATH", "")])
            )
            os.execv(sys.executable, [sys.executable, *sys.argv])
        from runtime.environment import setup

        setup()
    paths = [
        *ROOT.glob("*.py"),
        *(
            p
            for folder in ("algorithms", "calibration", "runtime")
            for p in (ROOT / folder).glob("*.py")
        ),
    ]
    for path in paths:
        compile(path.read_text(encoding="utf8"), str(path), "exec")
    for name in ALGORITHMS:
        tree = ast.parse((ROOT / "algorithms" / f"{name}.py").read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and (node.module.startswith("calibration") or "_bca" in node.module)
            ):
                raise ValueError("Plain host imports BCA: " + name)
            if isinstance(node, ast.Import) and any(
                a.name.startswith("calibration") or "_bca" in a.name for a in node.names
            ):
                raise ValueError("Plain host imports BCA: " + name)
    seeds = _yaml(str(ROOT / "configs/experiment.yaml"))["seeds"]
    hashes = {}
    actors = 0
    for name in ALGORITHMS:
        path = ROOT / "configs" / f"{name}.yaml"
        for dataset in read_config(path)["datasets"]:
            for seed in seeds:
                pair = []
                for method in METHODS:
                    row = resolve(path, method, seed, ROOT / "runs/example", dataset)
                    if opt.runtime:
                        with contextlib.redirect_stdout(io.StringIO()):
                            typed(row)
                    host = (
                        row["options"]["host_parameters"]
                        if name == "iql"
                        else row["native_args"]
                    )
                    pair.append(host)
                    if "options" in row:
                        row["options"]["output_dir"] = "{output_dir}"
                    if row["run_id"] in hashes:
                        raise ValueError("Duplicate run identity")
                    hashes[row["run_id"]] = hashlib.sha256(
                        json.dumps(row, sort_keys=True).encode()
                    ).hexdigest()
                    actors += row.get("actor_count", 1)
                if pair[0] != pair[1]:
                    raise ValueError(
                        "Host settings differ between methods: " + name + "/" + dataset
                    )
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    record = json.loads((ROOT / "configs/sources.json").read_text())
    if "matrix_sha256" in record and record["matrix_sha256"] != digest:
        raise ValueError(
            "Configuration changed; review before updating the recorded identity."
        )
    for name, wanted in record.get("source_sha256", {}).items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != wanted:
            raise ValueError("Source changed since verification: " + name)
    print(
        json.dumps(
            dict(
                accepted=True,
                methods=METHODS,
                physical_runs=len(hashes),
                actor_trajectories=actors,
                matrix_sha256=digest,
                typed_protocols_checked=opt.runtime,
                training_updates=0,
                simulator_steps=0,
            )
        )
    )


if __name__ == "__main__":
    main()
