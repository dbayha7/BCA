"""Stdlib-only syntax/hash receipt; never imports or executes scientific code."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
names = ("README.md", "CHECKPOINT_CONTRACT.md", "CHECKPOINT_GENERATION_PROPOSAL.md",
         "checkpoint_inventory.json", "protocol.json", "run_diagnostic.py", "inference.py",
         "simulation.py", "metrics.py", "prepare_receipt.py")
files, syntax = {}, []
for name in names:
    raw = (root / name).read_bytes()
    files[name] = hashlib.sha256(raw).hexdigest()
    if name.endswith(".py"):
        compile(ast.parse(raw.decode("utf-8"), filename=name), name, "exec")
        syntax.append(name)
    elif name.endswith(".json"):
        json.loads(raw.decode("utf-8"))
external = {}
for folder, expected in (
    ("unifloral_td3bc_validation_v1", "1e26f713822020275179845f30c8691467951745d6c7c829ff017b7045904516"),
    ("unifloral_bca_pipeline_validation_v1", "6eb91c5bcc6c2edddc1ab93c19424307e12a3c8363bf9063fa4bbe2664180478")):
    base = root.parent / folder
    raw = (base / "prepared_files.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Previous suite manifest changed: " + folder)
    source_files = json.loads(raw)["files"]
    for name, digest in source_files.items():
        if hashlib.sha256((base / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Previous suite source changed: " + name)
    external[folder] = {"manifest_sha256": expected, "verified_files": len(source_files)}
for name, value in (
    ("prepared_files.json", {"schema": "prepared_files.v1", "files": files}),
    ("PREPARATION_RECEIPT.json", {"utc": datetime.now(timezone.utc).isoformat(),
      "status": "static_checks_passed_runtime_not_executed", "python": sys.version,
      "syntax_checked": syntax, "files": files, "previous_sources": external,
      "scientific_imports": 0, "simulator_transitions": 0, "optimizer_updates": 0,
      "trained_checkpoint_available": False,
      "scope": "AST compile, JSON parse and byte identities; no project execution"})):
    with (root / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
print(json.dumps({"status": "static_checks_passed", "python_files": len(syntax),
                  "files": len(files), "scientific_execution": False}))
