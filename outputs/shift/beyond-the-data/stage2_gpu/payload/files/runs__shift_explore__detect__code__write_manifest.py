"""Write runs/shift_explore/detect/code_manifest.json (+ .sha256): the SHA-256 of every detect script and of the
current spec, with copies of the scripts in runs/shift_explore/detect/code/. Every phase refuses unless the scripts
and their copies still have these SHA-256 (common.check_code). A later change needs a new spec version and a new
manifest with a stated reason; the old manifest and copies are kept in code_manifest_history/.
Usage: python write_manifest.py [--reason "..."]
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason", default=None)
    a = ap.parse_args()
    path = C.OUT / C.MANIFEST_NAME
    history = []
    if path.exists():
        if not a.reason:
            raise SystemExit("refused: a manifest exists; pass --reason to write a new one (the old one is kept)")
        old = C.read_json(path)
        hist = C.OUT / "code_manifest_history" / f"v{old.get('version', 1)}"
        hist.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, hist / C.MANIFEST_NAME)
        shutil.copy2(path.with_name(path.name + ".sha256"), hist / (C.MANIFEST_NAME + ".sha256"))
        if (C.OUT / "code").exists():
            shutil.copytree(C.OUT / "code", hist / "code", dirs_exist_ok=True)
        history = old.get("history", []) + [dict(version=old.get("version", 1), sha256=C.file_sha256(
            hist / C.MANIFEST_NAME), reason_replaced=a.reason)]
    spec, spec_sha = C.load_spec()
    files = C.code_hashes()
    code = C.guard_out(C.OUT / "code")
    code.mkdir(parents=True, exist_ok=True)
    for n in files:
        shutil.copy2(C.HERE / n, code / n)
        if C.file_sha256(code / n) != files[n]:
            raise SystemExit(f"refused: the copy of {n} differs")
    rec = dict(label=C.LABEL, kind="detect-code-manifest", version=len(history) + 1, files=files,
               copies_dir=str(code), source_dir=str(C.HERE), spec_sha256=spec_sha, spec_version=spec.get("version"),
               room_pins=C.room_pins(), history=history, reason=a.reason or "first manifest, written before any run",
               rule=("every phase refuses unless each file here and its copy in copies_dir have these SHA-256 and this "
                     "manifest names the current spec"),
               written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    digest = C.write_json(path, rec)
    print(f"manifest written: {path} sha256 {digest}")


if __name__ == "__main__":
    main()
