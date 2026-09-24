"""Validate every host/BCA configuration; --runtime constructs typed CPU protocols."""

import argparse
import ast
import contextlib
import hashlib
import io
import json
import os
import re
from pathlib import Path
import sys
import textwrap
from runtime.config import ALGORITHMS, METHODS, ROOT, _yaml, read_config, resolve, typed


def verify_code_map():
    """Reject stale excerpts or untraceable Python snippets in the integration map."""
    document = (ROOT / "INTEGRATION.md").read_text(encoding="utf8")
    snippets = re.findall(
        r"<!-- source: ([^:\n]+):(\d+):(\d+) -->\n```python\n(.*?)\n```",
        document,
        re.DOTALL,
    )
    if not snippets or len(snippets) != document.count("```python"):
        raise ValueError("Every integration-map snippet must identify its source.")
    for name, first, last, code in snippets:
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Code-map source must be inside this repository.")
        lines = path.read_text(encoding="utf8").splitlines()
        start, end = int(first), int(last)
        if not 1 <= start <= end <= len(lines):
            raise ValueError("Invalid source range in integration map: " + name)
        expected = textwrap.dedent("\n".join(lines[start - 1 : end])).strip("\n")
        if code != expected:
            raise ValueError("Stale integration-map snippet: " + name + ":" + first)
    return len(snippets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument(
        "--unifloral",
        type=Path,
        help="Verify a separate checkout of the pinned original Unifloral source.",
    )
    parser.add_argument(
        "--reference-config",
        nargs=2,
        metavar=("ALGORITHM", "DATASET"),
        help="With --unifloral, show the exact original default reference settings.",
    )
    opt = parser.parse_args()
    if opt.reference_config and opt.unifloral is None:
        parser.error("--reference-config requires --unifloral CHECKOUT")
    snippet_count = verify_code_map()
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
        tree = ast.parse(
            (ROOT / "algorithms" / f"{name}.py").read_text(encoding="utf8")
        )
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
    record = json.loads((ROOT / "configs/sources.json").read_text(encoding="utf8"))
    if "matrix_sha256" in record and record["matrix_sha256"] != digest:
        raise ValueError(
            "Configuration changed; review before updating the recorded identity."
        )
    for name, wanted in record.get("source_sha256", {}).items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != wanted:
            raise ValueError("Source changed since verification: " + name)
    reference = {}
    if opt.unifloral is not None:
        from runtime.upstream import configuration, verify

        reference["unifloral"] = verify(opt.unifloral)
        if opt.reference_config:
            try:
                reference["reference_configuration"] = configuration(
                    *opt.reference_config
                )
            except KeyError:
                parser.error(
                    "Reference algorithms: iql/cql/td3_bc/rebrac; datasets: hopper/walker/halfcheetah"
                )
    print(
        json.dumps(
            dict(
                accepted=True,
                methods=METHODS,
                physical_runs=len(hashes),
                actor_trajectories=actors,
                matrix_sha256=digest,
                typed_protocols_checked=opt.runtime,
                verified_code_snippets=snippet_count,
                training_updates=0,
                simulator_steps=0,
                **reference,
            )
        )
    )


if __name__ == "__main__":
    main()
