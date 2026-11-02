"""Reuse the original worker, replacing only its dataset consumer in this process."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "minimal_comparison"
REPO = HERE.parents[4]
ORIGINAL_MANIFEST_SHA = "400490f6f2c3a5ab702e80733fcf28a34a1308f7d06df6df476bc8235e593c1d"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_before_import(job_path):
    job = json.loads(Path(job_path).read_text())
    record = job["integrity"]
    if Path(record["package_dir"]).resolve() != ORIGINAL:
        raise ValueError("Wrong original worker package")
    files = record["files"]
    required = (HERE / "MANIFEST.json", HERE / "worker.py", HERE / "immutable_data.py", ORIGINAL / "worker.py")
    if not all(str(p) in files for p in required):
        raise ValueError("Recovery worker/source is missing from job integrity")
    if sha(ORIGINAL / "MANIFEST.json") != ORIGINAL_MANIFEST_SHA:
        raise ValueError("Original manifest changed before worker import")
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    for relative, expected in manifest["files"].items():
        if files.get(str(REPO / relative)) != expected:
            raise ValueError("Worker job omits or changes a source pin")
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError("Worker source changed before import: " + path)


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def adapted_worker(job_path):
    verify_before_import(job_path)
    old = load_source("_recovery_original_worker", ORIGINAL / "worker.py")
    original_configure, original_roots, original_write = old.configure, old.source_roots, old.write_new
    loader_receipts = []
    old.source_roots = lambda: original_roots() + (HERE,)
    def configure(job):
        modules = original_configure(job)
        D = modules[3]
        loader = load_source("_recovery_exact_immutable_data", HERE / "immutable_data.py")
        def immutable_load(path):
            dataset, receipt = loader.load_verified(path, D)
            loader_receipts.append(receipt)
            return dataset
        D.load = immutable_load
        return modules
    old.configure = configure
    def write(path, value):
        value = dict(value, recovery_worker_sha256=old.sha(__file__),
                     recovery_manifest_sha256=old.sha(HERE / "MANIFEST.json"),
                     immutable_loader_receipts=loader_receipts,
                     loader_amendment="Producer-validated immutable content; original WSL reset replay failure preserved")
        return original_write(path, value)
    old.write_new = write
    return old


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    report = adapted_worker(args.job).run_job(args.job)
    print(json.dumps({key: report.get(key) for key in ("status", "job", "kind", "transitions", "error")}))
    raise SystemExit(0 if report.get("status") == "completed" else 1)
