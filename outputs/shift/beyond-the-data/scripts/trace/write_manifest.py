"""EXPLORATORY: write runs/shift_explore/trace/code_manifest.json (+ .sha256) and copy the trace test bed's scripts to
runs/shift_explore/trace/code/. Every phase and the seed-0 checks refuse unless each listed script and its copy have
the SHA-256 the manifest records and the manifest names the current spec (tracebed.check_code).

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
import tracebed as T  # noqa: E402


def write_manifest(reason=None):
    spec, spec_sha = T.load_spec()
    T.check_gate_files(spec)
    path = T.OUT / T.MANIFEST_NAME
    files = T.code_hashes()
    history = []
    old = T.read_json(path)
    if old is not None:
        if old["files"] == files and old["spec_sha256"] == spec_sha:
            print(f"{path} already current")
            return T.file_sha256(path)
        if not reason:
            raise SystemExit("refused: a manifest exists; a new one needs --reason")
        k = int(old.get("version", 1))
        dest = T.guard_out(T.OUT / "code_manifest_history" / f"v{k}")
        dest.mkdir(parents=True, exist_ok=True)
        for suffix in ("", ".sha256"):
            shutil.copy2(str(path) + suffix, dest / (T.MANIFEST_NAME + suffix))
        if (T.OUT / "code").exists():
            shutil.copytree(T.OUT / "code", dest / "code", dirs_exist_ok=True)
        history = old.get("history", []) + [dict(version=k, sha256=T.file_sha256(path), archived=str(dest),
                                                   replaced_because=reason)]
    code = T.guard_out(T.OUT / "code")
    code.mkdir(parents=True, exist_ok=True)
    for name in files:
        shutil.copy2(T.HERE / name, code / name)
    version = 1 if old is None else int(old.get("version", 1)) + 1
    earlier = sorted(str(p.relative_to(T.OUT)) for p in (T.OUT / "superseded").glob("v*/" + T.MANIFEST_NAME))
    payload = dict(label=T.LABEL, kind="trace-code-manifest", version=version, spec_sha256=spec_sha,
                   spec_version=int(spec.get("version", 1)),
                   source_dir=str(T.HERE), copies_dir=str(code), files=files, history=history,
                   manifests_of_earlier_specs=earlier,
                   reason=reason or (f"first manifest under spec v{int(spec.get('version', 1))}, written before any "
                                     "run of the trace test bed"),
                   written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   rule=("every phase (build, train, eval, signals, reference, pooled) and the seed-0 checks refuse "
                         "unless each file here and its copy in copies_dir have these SHA-256 and this manifest names "
                         "the current spec; any later change needs a new manifest with a reason"))
    digest = T.write_json(path, payload)
    T.check_code(spec_sha)
    print(f"wrote {path} version {version} sha256 {digest}")
    return digest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason")
    write_manifest(ap.parse_args().reason)
