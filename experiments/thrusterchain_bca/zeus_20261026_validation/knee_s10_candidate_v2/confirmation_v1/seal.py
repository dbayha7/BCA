"""Write CF_MANIFEST.json (cf.manifest_body). Run once, after the tests pass; the approval names the manifest's hash.

  PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B knee_s10_candidate_v2/confirmation_v1/seal.py
"""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import cf                                                   # noqa: E402


def main():
    manifest = cf.manifest_body()
    raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    out = HERE / "CF_MANIFEST.json"
    with out.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"written": out.name, "manifest_sha256": hashlib.sha256(raw).hexdigest(), "files": len(manifest["files"]),
                      "overestimation_v1_files": len(manifest["overestimation_v1_files"]),
                      "native_screen_files": len(manifest["native_screen_files"])}, indent=1))


if __name__ == "__main__":
    main()
