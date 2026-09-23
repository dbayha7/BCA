"""Check syntax and every declared configuration; --runtime also validates typed protocols."""

from pathlib import Path
import argparse
import contextlib
import hashlib
import io
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "algorithms"))
from _config import read_config, resolve, typed, _yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true")
    options = parser.parse_args()
    if options.runtime:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
        os.environ["MUJOCO_PY_FORCE_CPU"] = "1"
        os.environ["WANDB_MODE"] = "disabled"
        location = str(
            Path(
                os.environ.get("MUJOCO_PY_MUJOCO_PATH", str(Path.home() / ".mujoco/mujoco210"))
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
        from _runtime import setup

        setup()
    for path in [*ROOT.glob("*.py"), *(ROOT / "algorithms").glob("*.py")]:
        compile(path.read_text(), str(path), "exec")
    experiment = _yaml(str(ROOT / "configs/experiment.yaml"))
    hashes = {}
    actors = 0
    for host in ("td3_bc", "rebrac", "cql", "iql"):
        path = ROOT / "configs" / (host + ".yaml")
        config = read_config(path)
        for dataset, data in config["datasets"].items():
            for arm in data["arms"]:
                for seed in experiment["seeds"]:
                    row = resolve(path, arm, seed, ROOT / "runs/example", dataset)
                    if options.runtime:
                        with contextlib.redirect_stdout(io.StringIO()):
                            typed(row)
                    if "options" in row:
                        row["options"]["output_dir"] = "{output_dir}"
                    if row["run_id"] in hashes:
                        raise ValueError("Duplicate declared run ID.")
                    hashes[row["run_id"]] = hashlib.sha256(
                        json.dumps(row, sort_keys=True).encode()
                    ).hexdigest()
                    actors += row.get("actor_count", 1)
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    provenance = ROOT / "configs/sources.json"
    if provenance.exists():
        record = json.loads(provenance.read_text())
        expected = record["configuration_matrix"]
        if (len(hashes), actors, digest) != (
            expected["physical_groups"],
            expected["actor_trajectories"],
            expected["sha256"],
        ):
            raise ValueError(
                "Configuration matrix differs from the recorded reproduction. Review deliberate changes before updating its provenance."
            )
        pins = {
            "algorithms/" + name: source["new_sha256"]
            for name, source in record["relocated_sources"].items()
        }
        pins.update(record["entrypoint_files"])
        pins.update(
            {
                "configs/" + name: item["new_sha256"]
                for name, item in record["reference_tables"].items()
            }
        )
        for name, wanted in pins.items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != wanted:
                raise ValueError("Source differs from the recorded reproduction: " + name)
    print(
        json.dumps(
            dict(
                accepted=True,
                physical_groups=len(hashes),
                actor_trajectories=actors,
                matrix_sha256=digest,
                typed_protocols_checked=options.runtime,
                training_updates=0,
                simulator_steps=0,
            )
        )
    )


if __name__ == "__main__":
    main()
