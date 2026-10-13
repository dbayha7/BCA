"""Copy the confirmatory results (fresh seeds 5-9) into outputs/shift/beyond-the-data/stage2_gpu/results/ for the user
to commit and push: every JSON record the phases wrote, each with its .sha256 sidecar, the phase logs, the runner's
log and status, and the self-test report. Model files (.msgpack) and arrays (.npz) stay on this machine. Writes
results/BUNDLE.json (with a .sha256 sidecar) naming every file, its SHA-256, where it came from and the backend it
was computed on, and the environment: the gate's software record as measured now, the processor, the code and
pre-registration hashes, the launcher, git. Called by run_confirmatory.sh when the pooled look exists; with
--partial it bundles whatever exists so far and marks the bundle PARTIAL.

Read only outside results/. Never edits a result.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
LIB = Path(__file__).resolve().parent
PKG = LIB.parent
ROOT = Path("/home/user/bca")
OUT = ROOT / "runs" / "shift_explore" / "stage2"
OFFCLOUD = ROOT / "runs" / "shift_explore" / "stage2_offcloud"
RESULTS = PKG / "results"
SEEDS = (5, 6, 7, 8, 9)
LIMIT = 20 * 1000 * 1000
BACKEND_EVIDENCE = ("JAX_PLATFORMS=cpu and OMP_NUM_THREADS=1 are set by the pinned launch.sh for every phase, and the "
                    "gate's check_software, called at the start of every phase that loads ThrusterChain data, refuses "
                    "any process whose jax backend is not 'cpu' (runs/shift/preregistration.json, software block); "
                    "the software record below was measured again with the same settings when this bundle was made")


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return sha_bytes(Path(p).read_bytes())


def encode(rel):
    return rel.replace("/", "__")


def seed_of(rel):
    for part in rel.split("/"):
        if part.startswith("seed") and part[4:].isdigit():
            return int(part[4:])
    return None


def collect():
    """(repository-relative path, kind) of everything the bundle holds."""
    out = []
    for p in sorted(OUT.rglob("*.json")):
        rel = p.relative_to(ROOT).as_posix()
        sub = p.relative_to(OUT).as_posix()
        seed = seed_of(sub)
        if sub in ("pooled_confirmatory.json",) or sub.startswith("detect/marks_confirmatory_"):
            pass
        elif seed is None or seed not in SEEDS:
            continue                                   # seeds 0-4, the late files, approval, prereg, manifests
        elif not sub.startswith(("fresh/", "detect_fresh/", "detect/", "policy/", "deploy/", "host/")):
            continue
        out.append((rel, "result"))
        side = p.with_name(p.name + ".sha256")
        if side.exists():
            out.append((side.relative_to(ROOT).as_posix(), "sidecar"))
    logs = OUT / "logs"
    if logs.exists():
        for p in sorted(logs.glob("*.log")):
            n = p.name
            if any(f"_seed_{k}" in n for k in SEEDS) or "group_confirmatory" in n:
                out.append((p.relative_to(ROOT).as_posix(), "phase log"))
    for name in ("runner.log", "status.json", "restore_last.json", "launch_stderr.txt"):
        if (OFFCLOUD / name).exists():
            out.append(((OFFCLOUD / name).relative_to(ROOT).as_posix(), "runner record"))
    reports = sorted(OFFCLOUD.glob("selftest_*.txt"))
    if reports:
        out.append((reports[-1].relative_to(ROOT).as_posix(), "self-test report"))
    return out


def git(*args):
    try:
        done = subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, text=True, timeout=60,
                              env=dict(PATH=os.environ.get("PATH", "/usr/bin:/bin"), HOME="/root",
                                       GIT_OPTIONAL_LOCKS="0"))
        return done.stdout.strip() if done.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="bundle what exists before the pooled look (marked PARTIAL)")
    ap.add_argument("--test-bundle", action="store_true", help=argparse.SUPPRESS)   # package tests only
    a = ap.parse_args(argv)
    if a.test_bundle and os.environ.get("BCA_OFFCLOUD_TEST") != "1":
        raise SystemExit("refused: --test-bundle is for the package's own tests only")
    final = (OUT / "pooled_confirmatory.json").exists()
    if not final and not a.partial:
        raise SystemExit("refused: pooled_confirmatory.json does not exist yet (use --partial for an interim bundle)")
    sys.path.insert(0, str(LIB))
    import envfacts as E
    import selftest_checks as ST
    status = json.loads((OFFCLOUD / "status.json").read_text(encoding="utf8")) if (OFFCLOUD / "status.json").exists() \
        else {}
    launcher = status.get("launcher") or {}
    if launcher.get("test") and not a.test_bundle:
        raise SystemExit("refused: these outputs were made by the package's test launcher, not by launch.sh")
    if any(t.get("state") == "running" for t in (status.get("tasks") or {}).values()):
        raise SystemExit("refused: a phase is still running")
    man = json.loads((OUT / "code_manifest.json").read_text(encoding="utf8"))
    code_dir = Path(man["source_dir"])
    software = ST.run_probe("software", 0)
    anchors = ST.run_probe("anchors", 0, {"OPENBLAS_CORETYPE": launcher["openblas_coretype"]}
                           if launcher.get("openblas_coretype") else None)
    files_dir = RESULTS / "files"
    if files_dir.exists():
        shutil.rmtree(files_dir)
    files_dir.mkdir(parents=True)
    rows, total = [], 0
    for rel, kind in collect():
        src = ROOT / rel
        data = src.read_bytes()
        name = encode(rel) + (".txt" if rel.endswith(".log") else "")
        (files_dir / name).write_bytes(data)
        row = dict(name=name, path=rel, kind=kind, sha256=sha_bytes(data), bytes=len(data), seed=seed_of(rel))
        if kind == "result":
            side = src.with_name(src.name + ".sha256")
            row["sidecar_ok"] = side.exists() and side.read_text(encoding="utf8").split()[:1] == [row["sha256"]]
            row["backend"] = "cpu"
            try:
                body = json.loads(data.decode("utf8"))
                row.update(record_kind=body.get("kind"), prereg_sha256=body.get("prereg_sha256"))
            except ValueError:
                row["record_kind"] = None
        rows.append(row)
        total += len(data)
    late = {}
    lt = PKG / "payload" / "late" / "LATE.json"
    if lt.exists():
        late = json.loads(lt.read_text(encoding="utf8")).get("files") or {}
    bundle = dict(
        kind="stage2-offcloud-results", partial=not final,
        status=("FINAL: every confirmatory phase finished or failed on record, and the pooled look exists" if final
                else "PARTIAL: an interim copy; the confirmatory phases are not finished"),
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        prereg_sha256=sha_file(OUT / "prereg.json"), gate_prereg_sha256=sha_file(ROOT / "runs/shift/preregistration.json"),
        approval_json_sha256=sha_file(OUT / "approval.json"), code_manifest_sha256=sha_file(OUT / "code_manifest.json"),
        stage2_code={n: sha_file(code_dir / n) for n in sorted(man["files"])},
        launch_sh_sha256=sha_file(code_dir / "launch.sh"), launcher=launcher, late_files_sha256=late,
        backend="cpu", backend_evidence=BACKEND_EVIDENCE,
        software_now=software.get("software"), dependency_record_now=software.get("dependency_record"),
        software_equal_to_gate=software.get("software_equal"), dependency_equal_to_gate=software.get("dependency_equal"),
        anchors_now=dict(gate_accepts=anchors.get("gate_accepts"), openblas=anchors.get("openblas")),
        machine=dict(cpu=E.cpu_info(), kernel=platform.release(),
                     wsl="microsoft" in Path("/proc/version").read_text(encoding="utf8", errors="replace").lower()),
        git=dict(head=git("rev-parse", "HEAD"), branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                 origin=git("rev-parse", "refs/remotes/origin/claude/main-bca-connection-x4o46s"),
                 user_email_during_run=git("config", "--get", "user.email")),
        phases={k: dict(state=v.get("state"), cpu=v.get("cpu"), started=v.get("started"), finished=v.get("finished"),
                        exit_code=v.get("exit_code"), note=v.get("note"), backend="cpu")
                for k, v in (status.get("tasks") or {}).items()},
        files=rows, total_bytes=total,
        excluded="model files (.msgpack), arrays (.npz) and caches stay on the user's machine")
    text = json.dumps(bundle, indent=1, sort_keys=True)
    (RESULTS / "BUNDLE.json").write_text(text, encoding="utf8")
    (RESULTS / "BUNDLE.json.sha256").write_text(f"{sha_bytes(text.encode('utf8'))}  BUNDLE.json\n", encoding="utf8")
    print(f"results: {len(rows)} files, {total} bytes in {RESULTS} ({'final' if final else 'PARTIAL'})")
    if total > LIMIT:
        print(f"WARN the bundle is over {LIMIT} bytes; ask the cloud before pushing")
    bad = [r["path"] for r in rows if r["kind"] == "result" and not r.get("sidecar_ok")]
    if bad:
        print(f"WARN results without a matching sidecar: {bad[:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
