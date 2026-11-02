"""Authenticated thin adapter to the existing immutable-data recovery worker."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "minimal_comparison_recovery_v1"
REPO = HERE.parents[4]
PREVIOUS_MANIFEST_SHA = "fc20c6b89b04be6398be64e674dfa1c425ba832436e788e564431acc34c17fc8"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapted_worker(job_path):
    job = json.loads(Path(job_path).read_text())
    files = job["integrity"]["files"]
    if files.get(str(HERE / "MANIFEST.json")) != sha(HERE / "MANIFEST.json"):
        raise ValueError("Continuation manifest is not bound to job")
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    for relative, digest in manifest["files"].items():
        path = REPO / relative
        if files.get(str(path)) != digest or sha(path) != digest:
            raise ValueError("Continuation source/evidence changed before import: " + relative)
    if sha(PREVIOUS / "MANIFEST.json") != PREVIOUS_MANIFEST_SHA:
        raise ValueError("Previous worker package changed")
    spec = importlib.util.spec_from_file_location("_continuation_previous_worker", PREVIOUS / "worker.py")
    previous = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = previous
    spec.loader.exec_module(previous)
    old = previous.adapted_worker(job_path)
    roots, write = old.source_roots, old.write_new
    old.source_roots = lambda: roots() + (HERE,)
    def recorded(path, value):
        return write(path, dict(value, continuation_manifest_sha256=sha(HERE / "MANIFEST.json"),
                                continuation_worker_sha256=sha(__file__),
                                continuation_scope="seed 2 restart only, then unchanged three-seed evaluation"))
    old.write_new = recorded
    return old


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    report = adapted_worker(args.job).run_job(args.job)
    print(json.dumps({key: report.get(key) for key in ("status", "job", "kind", "transitions", "error")}))
    raise SystemExit(0 if report.get("status") == "completed" else 1)
