"""Preserve the eight authenticated source files used by the signal audit."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
identity = json.loads((root / "prepared/engine_identity.json").read_text(encoding="utf-8"))
source = Path(identity["source_closure"])
names = ["algorithms/td3_bc.py", "algorithms/td3_bc_bca.py", "calibration/network.py",
         "calibration/reference.py", "calibration/wbcp.py", "calibration/dose.py",
         "experiments/thrusterchain_diagnostics/training.py",
         "experiments/thrusterchain_diagnostics/training_data.py"]
payloads = {}
for name in names:
    payload = (source / name).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == identity["source_sha256"][name], name
    payloads[name] = payload
destination = root / "audit_source_snapshot"
destination.mkdir(exist_ok=False)
for name, payload in payloads.items():
    path = destination / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
(destination / "manifest.json").write_text(json.dumps({
    "origin": str(source), "purpose": "source-to-objective audit only; not a runnable full learner closure",
    "files": {name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
              for name, payload in payloads.items()},
}, indent=2) + "\n", encoding="utf-8")
print("Saved eight source files; all hashes match the authenticated engine identity.")
