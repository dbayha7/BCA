"""Pinned read-only research dependencies; never writes to their directories."""
from pathlib import Path
import hashlib
import json
import sys

sys.dont_write_bytecode = True
CORE_ROOT = Path(__file__).resolve().parents[2] / "core_v4"
HISTORICAL_ROOT = Path(__file__).resolve().parents[2] / "historical"
GYM_DEPENDENCY_ROOT = Path(__import__("sysconfig").get_path("purelib"))
CORE_SHA256 = "6f36e2714f6012fbc71a9315b19ab995f0b7d57d96026a1e2d97ef66d66d6596"
_PIN_PATH = Path(__file__).with_name("dependency_pins.json")

def verify_dependencies():
    manifest = json.loads((CORE_ROOT / "CURRENT_SOURCE_MANIFEST.json").read_text())
    if manifest["source_sha256"] != CORE_SHA256:
        raise RuntimeError("core source identity differs from the validated v4")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((CORE_ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("core dependency changed: " + name)
    pins = json.loads(_PIN_PATH.read_text())
    for name, expected in pins.items():
        if hashlib.sha256((HISTORICAL_ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("historical dependency changed: " + name)
    return {"core_source_sha256": CORE_SHA256,
            "historical_dependencies": pins}

DEPENDENCY_IDENTITY = verify_dependencies()
for path in (GYM_DEPENDENCY_ROOT, CORE_ROOT, HISTORICAL_ROOT, HISTORICAL_ROOT / "source_snapshot"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import algorithms.td3_bc as BASE
import algorithms.td3_bc_bca as NATIVE
import direct_extension as DIRECT

def _under(module, root):
    return Path(module.__file__).resolve().is_relative_to(root.resolve())

if not _under(BASE, HISTORICAL_ROOT / "source_snapshot") or not _under(NATIVE, HISTORICAL_ROOT / "source_snapshot") or not _under(DIRECT, HISTORICAL_ROOT):
    raise RuntimeError("module collision: historical imports resolved to another source")

import thrusterchain
if not _under(thrusterchain, CORE_ROOT):
    raise RuntimeError("module collision: thrusterchain resolved outside pinned v4")
