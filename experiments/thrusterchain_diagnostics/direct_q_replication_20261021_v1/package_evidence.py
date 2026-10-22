"""Create a compact, exclusive saved-analysis evidence archive; no checkpoints."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
release = ROOT / "release"
release.mkdir(exist_ok=False)
paths = []
for path in sorted(ROOT.rglob("*")):
    if not path.is_file():
        continue
    relative = path.relative_to(ROOT)
    if any(part in ("release", "__pycache__") for part in relative.parts):
        continue
    if path.suffix in (".pyc", ".html"):
        continue
    if path.name.endswith(".encoding_draft.md"):
        continue
    paths.append(path)
manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "kind": "analysis-only evidence, no checkpoint binaries",
    "reproduction": "python -B reproduce_analysis.py --output NEW_DIRECTORY; NumPy required; no model or simulator calls",
    "rollout_reproduction_dependency": "Full local checkpoint and authenticated historical source closure are needed to regenerate rollout labels. Their identities are saved; they are not bundled here.",
    "files": {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths},
}
name = "ThrusterChain_fresh_replication_evidence_20261021"
archive = release / (name + ".zip")
with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
    for path in paths:
        handle.write(path, name + "/" + path.relative_to(ROOT).as_posix())
    handle.writestr(name + "/artifact_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
with zipfile.ZipFile(archive) as handle:
    assert handle.testzip() is None
    for relative, expected in manifest["files"].items():
        payload = handle.read(name + "/" + relative)
        assert len(payload) == expected["bytes"]
        assert hashlib.sha256(payload).hexdigest() == expected["sha256"]
receipt = {"path": str(archive), "bytes": archive.stat().st_size,
           "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
           "members": len(paths) + 1, "zip_integrity_and_member_hashes": "passed"}
(release / "package_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, indent=2))
