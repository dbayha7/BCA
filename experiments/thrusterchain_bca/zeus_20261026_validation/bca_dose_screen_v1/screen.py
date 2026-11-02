"""BCA dose screen v1: the supervisor (PROTOCOL.md). One attempt. It never imports JAX: every GPU job is a separate
`screen_worker.py` process under matched_policy_pilot_v1's supervise() (an owned process group, killed at its cap).

- Preflight runs in screen_run/preflights/preflight_<n>, before the attempt exists, so an environment fault at launch
  does not use up the single attempt.
- Then screen_run/attempt_1 runs 12 BCA branches at kappa_A, one per exploratory 10k fork. Each uses the same sealed
  branch seeds as the fork's native branch. A failed branch is recorded as a dose failure and the other forks still run.
- Next come 3 evaluations, one per seed whose four branches all completed.
- Last comes the decision (screen_math.decide) against the exploratory native evaluations, whose report hashes are
  pinned by the approved manifest and re-checked. SCREEN_RESULT.json is always written, INCOMPLETE with the traceback
  if anything raises.
- `--decide` recomputes the decision from attempt_1's saved reports. It is JSON only and needs no GPU.

Run in WSL (README.md has the approval command):
  /home/david/bca-operator-audit-20261026/bin/python -B -u <this file> --execute
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import sys
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V3 = HERE.parent / "knee_edge_ood_v3"
for _p in (str(V3), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import attempts as A                                         # noqa: E402
import screen_math as SM                                     # noqa: E402
import step03                                                # noqa: E402
import units as U                                            # noqa: E402
from core import VALIDATION, module, require, sha_file       # noqa: E402
from stage1_worker import sanitize                           # noqa: E402

RUN = HERE / "screen_run"
EXP = VALIDATION / "knee_edge_ood_v2" / "stage1_exploratory_run" / "attempt_1"
COLLECTION = VALIDATION / "knee_edge_ood_v2" / "step04_run" / "attempt_1"
MANIFEST, APPROVAL, DOSE = HERE / "SCREEN_MANIFEST.json", HERE / "SCREEN_APPROVAL.json", HERE / "DOSE.json"
MANIFEST_FILES = ("PROTOCOL.md", "DOSE.json", "dose.py", "screen_worker.py", "screen.py", "screen_math.py",
                  "test_screen.py", "README.md")
V3_PINS = ("EXPLORATORY_MANIFEST.json", "A4_MANIFEST.json")         # the v3 code this screen reuses, sealed
APPROVAL_WORDS = ("I approve the exploratory BCA dose screen v1 at kappa_A = {kappa} on the knee-edge exploratory forks; "
                  "it supports no protocol claim")
BRANCH_SECONDS, TRAIN_SECONDS = 1_800, 6 * 3_600
EVAL_SECONDS, EVAL_TRANSITIONS, PREFLIGHT_SECONDS = 3_600, 18_000_000, 600
FORKS = [f"s{s}_{a}_{w}" for s in SM.SEEDS for a in SM.ARMS for w in SM.WORLDS]


def comparator_files():
    """Every exploratory record the screen reads: forks, native branches and their evaluation, the attempt record."""
    names = ["ATTEMPT.json", "report.json"] + [f"{k}_{n}/report.json" for n in FORKS for k in ("warmup", "branch")]
    return names + [f"evaluate_s{i}/report.json" for i in SM.SEEDS]


def manifest_body(root=HERE):
    """What sealing writes into SCREEN_MANIFEST.json: the package files, the v3 pins and the comparator records."""
    return {"files": {f: sha_file(root / f) for f in MANIFEST_FILES},
            "v3_pins": {f: sha_file(V3 / f) for f in V3_PINS},
            "comparator": {f: sha_file(EXP / f) for f in comparator_files()}}


def check_upstream(manifest):
    A.verify_upstream(V3, manifest["v3_pins"])                       # v3 manifests, their files and rerun1's sources
    changed = [f for f, d in manifest["comparator"].items() if sha_file(EXP / f) != d]
    require(not changed, f"Exploratory records changed since sealing: {changed}")


def check_approval(approval_path=None, manifest_path=None):
    approval_path, manifest_path = approval_path or APPROVAL, manifest_path or MANIFEST
    require(manifest_path.is_file(), "SCREEN_MANIFEST.json is missing: the screen is not sealed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(set(manifest["files"]) == set(MANIFEST_FILES), "The manifest does not list exactly the screen's files")
    changed = [f for f, d in manifest["files"].items() if sha_file(manifest_path.parent / f) != d]
    require(not changed, f"Files changed since sealing: {changed}")
    check_upstream(manifest)
    kappa = json.loads((manifest_path.parent / "DOSE.json").read_text(encoding="utf-8"))["kappa_A"]
    require(approval_path.is_file(), "SCREEN_APPROVAL.json is missing")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    require(approval.get("approval") == APPROVAL_WORDS.format(kappa=kappa), "Approval words differ")
    require(approval.get("screen_manifest_sha256") == sha_file(manifest_path), "Approval names a different manifest")
    require(approval.get("kappa_A") == kappa, "Approval names a different kappa_A")
    return {"approval": approval, "manifest_sha256": sha_file(manifest_path), "kappa_A": kappa, "manifest": manifest}


def integrity_record():
    """The v3 dependency closure (units.dependency_closure) plus this package's files; package_dir is this folder.
    The worker also scans the v3 folder for added importable files (screen_worker.v3_added_files)."""
    files = {str(p): sha_file(p) for p in U.dependency_closure()}
    files.update({str(p): sha_file(p) for p in sorted(HERE.iterdir()) if p.is_file() and p.suffix in (".py", ".json", ".md")})
    return {"package_dir": str(HERE), "files": files}


def inputs():
    attempt = json.loads((EXP / "ATTEMPT.json").read_text(encoding="utf-8"))
    require(sha_file(COLLECTION / "collection.json") == attempt["a3"]["collection_sha256"], "Collection differs from the exploratory run's")
    report = json.loads((EXP / "report.json").read_text(encoding="utf-8"))
    require(report.get("EXPLORATORY") and report.get("exploratory_label") is not None, "Not the finished exploratory run")
    for f in comparator_files()[2:]:
        unit = json.loads((EXP / f).read_text(encoding="utf-8"))
        require(unit["status"] == "completed", f"{f} is not completed")
    return {"collection_dir": str(COLLECTION)}


def supervisor():
    return module("_dose_screen_supervisor", VALIDATION / "matched_policy_pilot_v1/launch.py").supervise


def default_dispatch(job_dir, cap):
    command = [sys.executable, "-B", "-u", str(HERE / "screen_worker.py"), "--job", str(job_dir / "job.json")]
    env = dict(os.environ)
    env.update(step03.WORKER_ENVIRONMENT)
    return supervisor()(command, job_dir, cap, env=env, job_id=job_dir.name)


def _read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run_unit(directory, spec, cap, options, dispatch, caps, cap_key):
    job_dir = directory / spec["id"]
    job_dir.mkdir(parents=True, exist_ok=False)
    step03._write_new(job_dir / "job.json", {**spec, "timeout_seconds": float(cap), **options})
    outcome = dispatch(job_dir, float(cap))
    report = _read(job_dir / "report.json")
    caps[cap_key] = caps.get(cap_key, 0.0) + float(outcome.get("elapsed_seconds", cap))
    if cap_key == "evaluation_seconds":
        caps["evaluation_transitions"] += int((report or {}).get("transitions", spec.get("transitions", 0)) or 0)
    complete = bool(report and report.get("status") == "completed" and outcome.get("exit_code") == 0
                    and not outcome.get("timed_out") and not outcome.get("interrupted"))
    step03._write_new(job_dir / "SUPERVISOR_RESULT.json", {**outcome, "completed": complete, "cap_seconds": cap})
    return (report if complete else None), job_dir


def preflight(options, dispatch):
    root = RUN / "preflights"
    n = len([p for p in root.iterdir() if p.is_dir()]) + 1 if root.is_dir() else 1
    caps = {}
    report, job_dir = run_unit(root, {"id": f"preflight_{n}", "kind": "preflight"}, PREFLIGHT_SECONDS, options, dispatch,
                               caps, "preflight_seconds")
    return report is not None, str(job_dir), caps["preflight_seconds"]


def decide_from(directory, manifest):
    """The decision from the attempt's saved reports (JSON only)."""
    check_upstream(manifest)
    branch_dirs = {n: directory / f"bca_branch_{n}" for n in FORKS}
    done = {n: bool((_read(d / "SUPERVISOR_RESULT.json") or {}).get("completed")) for n, d in branch_dirs.items()}
    evals = {}
    for i in SM.SEEDS:
        sup = _read(directory / f"evaluate_s{i}" / "SUPERVISOR_RESULT.json") or {}
        if sup.get("completed"):
            evals[i] = _read(directory / f"evaluate_s{i}" / "report.json")
    native = {i: _read(EXP / f"evaluate_s{i}" / "report.json") for i in SM.SEEDS}
    audits = {n: SM.load_audit(branch_dirs[n]) for n in FORKS if done[n]}
    kappa = json.loads((HERE / "DOSE.json").read_text(encoding="utf-8"))["kappa_A"]
    decision = SM.decide(evals, native, audits, kappa)
    return {"verdicts": {arm: decision[arm]["verdict"] for arm in SM.ARMS}, "decision": decision,
            "branches_completed": done, "dose_failures": [n for n, ok in done.items() if not ok],
            "physical": U.physical(directory, None)}


def execute(dispatch=None):
    approval = check_approval()
    directory = RUN / "attempt_1"
    require(not directory.exists(), "The screen has one attempt and it already exists")
    found = inputs()
    options = {"provenance": "research", "collection_dir": found["collection_dir"], "test_seeds": None,
               "training_seeds": None, "visited_seed": None, "integrity": integrity_record()}
    dispatch = dispatch or default_dispatch
    ok, pre_dir, pre_seconds = preflight(options, dispatch)
    if not ok:
        return {"label": "PREFLIGHT FAILED", "preflight_dir": pre_dir,
                "reasons": ["preflight failed before the attempt; no attempt was used and no value was computed"]}
    directory.mkdir(parents=True)
    step03._write_new(directory / "ATTEMPT.json", sanitize({**{k: v for k, v in approval.items() if k != "manifest"}, **found,
                                                            "preflight_dir": pre_dir, "preflight_seconds": pre_seconds,
                                                            "started_utc": datetime.now(timezone.utc).isoformat()}))
    caps = {"training_seconds": 0.0, "evaluation_seconds": 0.0, "evaluation_transitions": 0}
    result = {"screen": "bca_dose_screen_v1", "kappa_A": approval["kappa_A"], "units": {}, "caps": caps}
    try:
        branch_ok = {}
        for name in FORKS:
            i, arm, world = int(name[1]), name.split("_")[1], name.split("_")[2]
            cap = min(BRANCH_SECONDS, TRAIN_SECONDS - caps["training_seconds"])
            if cap <= 0:
                branch_ok[name] = False
                result["units"][f"bca_branch_{name}"] = "not run: training cap exhausted"
                continue
            spec = {"id": f"bca_branch_{name}", "kind": "bca_branch", "seed_index": i, "arm": arm, "world": world,
                    "warmup_dir": str(EXP / f"warmup_{name}"), "native_branch_dir": str(EXP / f"branch_{name}"),
                    "transitions": 0}
            report, _ = run_unit(directory, spec, cap, options, dispatch, caps, "training_seconds")
            branch_ok[name] = report is not None
            result["units"][spec["id"]] = "completed" if report is not None else "failed (dose failure)"
        for i in SM.SEEDS:
            names = [n for n in FORKS if n.startswith(f"s{i}_")]
            if not all(branch_ok[n] for n in names):
                result["units"][f"evaluate_s{i}"] = "not run: a branch of this seed failed"
                continue
            cap = EVAL_SECONDS - caps["evaluation_seconds"]
            transitions = EVAL_TRANSITIONS - caps["evaluation_transitions"]
            if cap <= 0 or transitions <= 0:
                result["units"][f"evaluate_s{i}"] = "not run: evaluation cap exhausted"
                continue
            runs = {}
            for name in names:
                runs[f"warmup_{name}"] = str(EXP / f"warmup_{name}")
                runs[f"branch_{name}"] = str(directory / f"bca_branch_{name}")
            spec = {"id": f"evaluate_s{i}", "kind": "evaluate", "seed_index": i, "runs": runs, "transitions": int(transitions)}
            report, _ = run_unit(directory, spec, cap, options, dispatch, caps, "evaluation_seconds")
            result["units"][spec["id"]] = "completed" if report is not None else "failed"
        result.update(label="COMPLETE", **decide_from(directory, approval["manifest"]))
    except BaseException as exc:
        result.update(label="INCOMPLETE", reasons=[f"{type(exc).__name__}: {exc}"], traceback=traceback.format_exc())
        try:
            result["physical"] = U.physical(directory, None)
        except BaseException:
            pass
        _write_result(directory / "SCREEN_RESULT.json", result)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return result
    _write_result(directory / "SCREEN_RESULT.json", result)
    return result


def _write_result(path, result):
    step03._write_new(path, sanitize({**result, "finished_utc": datetime.now(timezone.utc).isoformat()}))


def decide_only():
    approval = check_approval()
    directory = RUN / "attempt_1"
    require(directory.is_dir(), "No attempt to decide")
    result = {"screen": "bca_dose_screen_v1", "kappa_A": approval["kappa_A"], "recomputed": True, **decide_from(directory, approval["manifest"])}
    _write_result(directory / f"SCREEN_RESULT_RECOMPUTED_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json", result)
    return result


def summary(out):
    s = {"label": out.get("label"), "reasons": out.get("reasons"), "verdicts": out.get("verdicts"),
         "kappa_A": out.get("kappa_A"), "dose_failures": out.get("dose_failures")}
    if out.get("decision"):
        s["by_seed"] = {arm: {i: {"delta_entry_stall": round(r["stall3"]["delta_entry"], 3),
                                  "placement": round(r["placement"], 3), "delta_J_stall": round(r["stall3"]["delta_J"], 1),
                                  "delta_J_nostall": round(r["nostall"]["delta_J"], 1),
                                  "q_rank_gain_s10": round(r["q_rank_gain_s10"], 3)}
                              for i, r in out["decision"][arm]["seeds"].items()} for arm in SM.ARMS}
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--execute", action="store_true")
    group.add_argument("--decide", action="store_true")
    args = parser.parse_args()
    print(json.dumps(summary(execute() if args.execute else decide_only()), indent=2, default=str))
