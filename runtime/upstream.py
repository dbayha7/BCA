"""Inspect exact Unifloral references without importing or executing their code."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/unifloral.json"


def defaults(path):
    """Read literal Args defaults; never import an upstream training script."""
    tree = ast.parse(Path(path).read_text(encoding="utf8"))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Args"]
    if len(classes) != 1:
        raise ValueError("Expected one upstream Args class: " + str(path))
    return {
        node.target.id: ast.literal_eval(node.value)
        for node in classes[0].body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }


def verify(checkout=None):
    """Verify the bundled tree, or an explicitly supplied upstream checkout."""
    record = json.loads(MANIFEST.read_text(encoding="utf8"))
    checkout = Path(checkout or ROOT / record["path"]).expanduser().resolve()
    # The bundled snapshot has no nested .git; its parent HEAD belongs to BCA.
    revision_checked = (checkout / ".git").exists()
    if revision_checked:
        revision = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != record["revision"]:
            raise ValueError("Unifloral checkout is not at the pinned revision.")
    if set(record["source_sha256"]) != set(record["git_files"]):
        raise ValueError("Unifloral manifest file sets differ.")
    for name, wanted in record["source_sha256"].items():
        path = (checkout / name).resolve()
        if not path.is_relative_to(checkout):
            raise ValueError("Unifloral source must stay inside the snapshot.")
        raw = path.read_bytes()
        identity = record["git_files"][name]
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if (
            hashlib.sha256(raw).hexdigest() != wanted
            or blob != identity["git_blob"]
            or len(raw) != identity["bytes"]
        ):
            raise ValueError("Unifloral source differs: " + name)
    for algorithm, args in record["defaults"].items():
        if defaults(checkout / "algorithms" / (algorithm + ".py")) != args:
            raise ValueError("Unifloral Args differ: " + algorithm)
    return {
        "revision": record["revision"],
        "tree": record["tree"],
        "checkout_revision_checked": revision_checked,
        "source": str(checkout),
        "verified_files": len(record["source_sha256"]),
        "verified_default_configurations": len(record["defaults"]),
        "reference_cells": len(record["defaults"]) * len(record["datasets"]),
        "runtime_installation_tested": False,
        "data_checked": False,
        "training_updates": 0,
        "simulator_steps": 0,
    }


def configuration(algorithm, dataset):
    record = json.loads(MANIFEST.read_text(encoding="utf8"))
    args = dict(record["defaults"][algorithm])
    data = record["datasets"][dataset]
    args.update(dataset=data["environment"], seed=record["seed"])
    budget = args["num_updates"]
    return {
        "family": "unifloral-standalone-default-reference",
        "revision": record["revision"],
        "working_directory": record["path"],
        "algorithm": algorithm,
        "args": args,
        "cache": data["cache"],
        "expected_counts": {
            "outer_updates": budget,
            "updates_per_critic": budget * args.get("num_critic_updates_per_step", 1),
            "actor_updates": budget,
            "periodic_banks": budget // args["eval_interval"],
            "periodic_episodes": args["eval_workers"],
            "final_episodes": args["eval_final_episodes"],
        },
        "evaluation": record["evaluation"][algorithm],
        "command_from_upstream_checkout": [
            "python",
            "algorithms/" + algorithm + ".py",
            "--dataset",
            args["dataset"],
            "--seed",
            str(args["seed"]),
            "--num-updates",
            str(budget),
        ],
        "qualification": record["qualification"],
    }
