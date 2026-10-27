"""Prepare immutable source/test receipts using the standard library only."""
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    for name in ("prepared_files.json", "PREPARATION_RECEIPT.json", "unit_test_output.txt"):
        if (ROOT / name).exists():
            raise FileExistsError(str(ROOT / name))
    names = ["contract.py", "test_contract.py", "checks.py", "run_checks.py", "prepare.py",
             "protocol.json", "README.md", "PSEUDOCODE.md", "PREREGISTERED_CHECKS.md"]
    for name in names:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(name)
        if name.endswith(".py"):
            tree = ast.parse((ROOT / name).read_bytes(), filename=name)
            if name in ("checks.py", "run_checks.py"):
                for node in tree.body:
                    if isinstance(node, ast.Import):
                        modules = [item.name.split('.')[0] for item in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        modules = [(node.module or '').split('.')[0]]
                    else:
                        modules = []
                    if set(modules) & {"jax", "numpy", "flax", "optax", "distrax", "scipy"}:
                        raise ValueError("Scientific top-level import: " + name)
        elif name.endswith(".json"):
            json.loads((ROOT / name).read_text(encoding="utf-8"))
    import test_contract
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_contract)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful() or result.testsRun != 23 or result.skipped:
        raise AssertionError(stream.getvalue())
    (ROOT / "unit_test_output.txt").write_text(stream.getvalue(), encoding="utf-8")
    receipt = {"schema": "unifloral.scale_gradient.preparation.v1",
        "utc": datetime.now(timezone.utc).isoformat(), "status": "static_and_stdlib_tests_pass",
        "python": sys.version, "executable": sys.executable,
        "unit_tests_run": result.testsRun, "unit_tests_passed": result.testsRun,
        "failures": 0, "errors": 0, "skipped": 0,
        "unit_test_output_sha256": digest(ROOT / "unit_test_output.txt"),
        "python_asts": [name for name in names if name.endswith(".py")],
        "protocol_sha256": digest(ROOT / "protocol.json"),
        "gpu_execution": "not_executed_pending_manual_user_command",
        "source_and_checkpoint_preflight": "Run run_checks.py --preflight after this source manifest is created",
        "scientific_module_imports": False, "optimizer_updates": 0, "simulator_steps": 0,
        "scientific_validation_status": "incomplete_design_review_required"}
    write(ROOT / "PREPARATION_RECEIPT.json", receipt)
    names.extend(("PREPARATION_RECEIPT.json", "unit_test_output.txt"))
    write(ROOT / "prepared_files.json", {"schema": "prepared_files.v1",
          "files": {name: digest(ROOT / name) for name in names}})
    print(json.dumps({"status": receipt["status"], "passed_tests": result.testsRun,
                      "manifest_sha256": digest(ROOT / "prepared_files.json")}))


if __name__ == "__main__":
    main()
