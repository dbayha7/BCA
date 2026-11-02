"""Write SCREEN_MANIFEST.json (screen.manifest_body): the sha256 of every file the native screen runs on. Run once,
after the tests pass and collection/collection.json exists; the approval names the manifest's hash.

  PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B knee_s10_candidate_v2/native_screen/seal.py
"""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import screen                                               # noqa: E402


def main():
    manifest = screen.manifest_body()
    raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    out = HERE / "SCREEN_MANIFEST.json"
    with out.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"written": out.name, "manifest_sha256": hashlib.sha256(raw).hexdigest(), "files": len(manifest["files"])}, indent=1))


if __name__ == "__main__":
    main()
