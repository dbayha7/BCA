"""EXPLORATORY: write runs/shift_explore/standins/code_manifest.json (+ .sha256) and copy the stand-in scripts to
runs/shift_explore/standins/code/. Every phase and every seed-0 check refuses unless each listed script and its copy
have the SHA-256 the manifest records and the manifest names the current spec (standins.check_code).

A new manifest replacing an existing one needs --reason; the old one and its copies are kept in
code_manifest_history/v<k>/ and the new manifest lists the history.

Usage: python write_manifest.py [--reason TEXT]
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import standins as S  # noqa: E402


def write_manifest(reason=None, out=None):
    out = Path(out or S.OUT)
    spec, spec_sha = S.load_spec()
    S.check_gate_files(spec)
    path = out / S.MANIFEST_NAME
    files = S.code_hashes()
    history = []
    old = S.read_json(path)
    if old is not None:
        if old["files"] == files and old["spec_sha256"] == spec_sha:
            print(f"{path} already current")
            return S.file_sha256(path)
        if not reason:
            raise SystemExit("refused: a manifest exists; a new one needs --reason")
        k = int(old.get("version", 1))
        dest = out / "code_manifest_history" / f"v{k}"
        dest.mkdir(parents=True, exist_ok=True)
        for suffix in ("", ".sha256"):
            shutil.copy2(str(path) + suffix, dest / (S.MANIFEST_NAME + suffix))
        if (out / "code").exists():
            shutil.copytree(out / "code", dest / "code", dirs_exist_ok=True)
        history = old.get("history", []) + [dict(version=k, sha256=S.file_sha256(path), archived=str(dest),
                                                   replaced_because=reason)]
    code = out / "code"
    code.mkdir(parents=True, exist_ok=True)
    for name in files:
        shutil.copy2(S.HERE / name, code / name)
    version = 1 if old is None else int(old.get("version", 1)) + 1
    payload = dict(label=S.LABEL, kind="standins-code-manifest", version=version, spec_sha256=spec_sha,
                   source_dir=str(S.HERE), copies_dir=str(code), files=files, history=history,
                   room_import=S.ROOM_IMPORT,
                   reason=reason or "first manifest, written before any stand-in run",
                   written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   rule=("every phase (train, eval, no_stall_train, no_stall_eval, marks) and every seed-0 check "
                         "refuses unless each file here and its copy in copies_dir have these SHA-256 and this "
                         "manifest names the current spec; any later change needs a new manifest with a reason"))
    digest = S.write_json(path, payload)
    S.check_code(spec_sha, out)
    print(f"wrote {path} version {version} sha256 {digest}")
    return digest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason")
    write_manifest(ap.parse_args().reason)
