"""EXPLORATORY: write runs/shift_explore/room/code_manifest.json (+ .sha256) and copy the room-test scripts to
runs/shift_explore/room/code/. Every phase and every seed-0 check refuses unless each listed script and its copy have
the SHA-256 the manifest records and the manifest names the current spec (harness.check_code).

A new manifest replacing an existing one needs --reason "<why the code changed>"; the old manifest and its copies are
kept in code_manifest_history/v<k>/, and the new manifest lists the history.

Usage: python write_manifest.py [--reason TEXT]
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402


def write_manifest(reason=None):
    spec, spec_sha = H.load_spec()
    H.check_gate_files(spec)
    path = H.OUT / H.MANIFEST_NAME
    files = H.code_hashes()
    history = []
    old = H.read_json(path)
    if old is not None:
        if old["files"] == files and old["spec_sha256"] == spec_sha:
            print(f"{path} already current")
            return H.file_sha256(path)
        if not reason:
            raise SystemExit("refused: a manifest exists; a new one needs --reason")
        k = int(old.get("version", 1))
        dest = H.OUT / "code_manifest_history" / f"v{k}"
        dest.mkdir(parents=True, exist_ok=True)
        for suffix in ("", ".sha256"):
            shutil.copy2(str(path) + suffix, dest / (H.MANIFEST_NAME + suffix))
        if (H.OUT / "code").exists():
            shutil.copytree(H.OUT / "code", dest / "code", dirs_exist_ok=True)
        history = old.get("history", []) + [dict(version=k, sha256=H.file_sha256(path), archived=str(dest),
                                                   replaced_because=reason)]
    code = H.OUT / "code"
    code.mkdir(parents=True, exist_ok=True)
    for name in files:
        shutil.copy2(H.HERE / name, code / name)
    version = 1 if old is None else int(old.get("version", 1)) + 1
    payload = dict(label=H.LABEL, kind="room-code-manifest", version=version, spec_sha256=spec_sha,
                   source_dir=str(H.HERE), copies_dir=str(code), files=files, history=history,
                   reason=reason or "first manifest, written before any room-test training",
                   written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   rule=("every phase (train, eval, no_stall_*, replication_*, futility, pooled) and every seed-0 "
                         "check refuses unless each file here and its copy in copies_dir have these SHA-256 and this "
                         "manifest names the current spec; any later change needs a new manifest with a reason"))
    digest = H.write_json(path, payload)
    H.check_code(spec_sha)
    print(f"wrote {path} version {version} sha256 {digest}")
    return digest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason")
    a = ap.parse_args()
    write_manifest(a.reason)
