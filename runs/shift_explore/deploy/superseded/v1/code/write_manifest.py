"""EXPLORATORY: write runs/shift_explore/deploy/code_manifest.json (+ .sha256) and copy the deploy scripts to
runs/shift_explore/deploy/code/. Every job, smoke check and the summary refuse unless each listed script and its copy
have the SHA-256 the manifest records and the manifest names the current spec (common.check_code).

A new manifest replacing an existing one needs --reason "<why the code changed>"; the old manifest and its copies are
kept in code_manifest_history/v<k>/, and the new manifest lists the history.

Usage: python write_manifest.py [--reason TEXT]
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def write_manifest(reason=None):
    spec, spec_sha = C.load_spec()
    pins = C.check_pins(spec)
    path = C.OUT / C.MANIFEST_NAME
    files = C.code_hashes()
    history = []
    old = C.read_json(path)
    if old is not None:
        if old["files"] == files and old["spec_sha256"] == spec_sha:
            print(f"{path} already current")
            return C.file_sha256(path)
        if not reason:
            raise SystemExit("refused: a manifest exists; a new one needs --reason")
        k = int(old.get("version", 1))
        dest = C.OUT / "code_manifest_history" / f"v{k}"
        dest.mkdir(parents=True, exist_ok=True)
        for suffix in ("", ".sha256"):
            shutil.copy2(str(path) + suffix, dest / (C.MANIFEST_NAME + suffix))
        if (C.OUT / "code").exists():
            shutil.copytree(C.OUT / "code", dest / "code", dirs_exist_ok=True)
        history = old.get("history", []) + [dict(version=k, sha256=C.file_sha256(path), archived=str(dest),
                                                   replaced_because=reason)]
    code = C.guard_out(C.OUT / "code")
    code.mkdir(parents=True, exist_ok=True)
    for name in files:
        shutil.copy2(C.HERE / name, code / name)
    version = 1 if old is None else int(old.get("version", 1)) + 1
    payload = dict(label=C.LABEL, kind="deploy-code-manifest", version=version, spec_sha256=spec_sha, pinned=pins,
                   source_dir=str(C.HERE), copies_dir=str(code), files=files, history=history,
                   reason=reason or "first manifest, written before any deploy run",
                   written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   rule=("every deploy job, smoke check and the summary refuse unless each file here and its copy in "
                         "copies_dir have these SHA-256 and this manifest names the current spec; the pinned room "
                         "harness and fallback.py are re-verified at every import; any later change needs a new "
                         "manifest with a reason"))
    digest = C.write_json(path, payload)
    C.check_code(spec_sha)
    print(f"wrote {path} version {version} sha256 {digest}")
    return digest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason")
    a = ap.parse_args()
    write_manifest(a.reason)
