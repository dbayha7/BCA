"""Worker-side integrity check (PROTOCOL_V2.md change 6). Standard library only, so a worker can run it before it
imports any package module. The supervisor writes `integrity` into every job.json (`units.integrity_record`: the
sha256 of every package source, manifest, seed, protocol and approval file, and of rerun1's reused sources). The
worker re-hashes the same paths and refuses to run on any difference, so a file changed after the supervisor's check
is never loaded. The one file this cannot vouch for is itself; its hash is in the manifests and in the record."""
import hashlib
import json
from pathlib import Path


def verify(argv, worker_file):
    """Returns the number of files verified, or None when there is no --job (the module was imported, not run)."""
    if "--job" not in argv:
        return None
    job = json.loads(Path(argv[argv.index("--job") + 1]).read_text(encoding="utf-8"))
    record = job.get("integrity")
    if not isinstance(record, dict) or "files" not in record or "package_dir" not in record:
        raise SystemExit("job.json carries no integrity record; refusing to run")
    here = Path(worker_file).resolve().parent
    if Path(record["package_dir"]).resolve() != here:
        raise SystemExit(f"worker runs from {here}, not the recorded package folder {record['package_dir']}")
    changed = [p for p, d in record["files"].items()
               if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest() != d]
    if changed:
        raise SystemExit("files changed since the supervisor checked them: " + ", ".join(changed))
    return len(record["files"])
