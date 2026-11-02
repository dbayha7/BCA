"""Confirmation of the OOD-action overestimation comparison: the supervisor (CONFIRMATION_RULE.md). It never
imports JAX: every job is a separate `cf_worker.py` process under matched_policy_pilot_v1's supervise(). Units, per
world (stall3, then no-stall), fail-fast: 3 warmups, 3 native branches, 9 BCA branches (kappa 21, 63, 189), 3
native evaluations, 9 BCA evaluations; then the decision (cf_math.decide). CF_RESULT.json is always written.
Run in WSL from PowerShell (README.md):
  wsl -d Ubuntu -- bash -lc "cd <this folder> && PYTHONDONTWRITEBYTECODE=1 <recorded python> -B -u cf.py --execute"
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
CAND = HERE.parent
NS, OE = CAND / "native_screen", CAND / "overestimation_v1"
V3 = CAND.parent / "knee_edge_ood_v3"
F2 = CAND / "formal_g2"
for _p in (str(V3), str(F2), str(NS), str(OE), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cf_math as CM                                         # noqa: E402
import oe                                                    # noqa: E402  manifest body, integrity record, _read
import screen as S                                           # noqa: E402  supervisor, inputs
import step03                                                # noqa: E402
import units as U                                            # noqa: E402
from core import require, sha_file                           # noqa: E402
from stage1_worker import sanitize                           # noqa: E402

RUN = HERE / "cf_run"
MANIFEST, APPROVAL = HERE / "CF_MANIFEST.json", HERE / "CF_APPROVAL.json"
DOSES = (21.0, 63.0, 189.0)
WORDS = (
    "I approve the confirmation of the OOD-action overestimation comparison on the knee_s10_v2 CUTOFF data, Unifloral TD3+BC host, "
    "doses kappa 21, 63 and 189, both worlds, attempt {attempt}",
    "the BCA arms at kappa 21, 63 and 189 are approved",
    "the confirmation seeds are fresh: no run has used them",
    "the dose being confirmed is 63; the other doses are a profile",
)
INSTRUMENT_CAUSE_REL = 1e-6
MAX_ATTEMPTS = 3                                              # amendment A3: 2 -> 3 (attempt 2 ended from outside)
JOB_SECONDS, TRAIN_SECONDS = 1_800, 13 * 3_600                # amendments A1, A2: 5 h -> 9 h -> 13 h (measured job times; Titan)
EVAL_JOB_SECONDS, EVAL_JOB_TRANSITIONS = 900, 200_000
EVAL_SECONDS, EVAL_TRANSITIONS = 2 * 3_600, 3_000_000
SEEDS, WORLDS, ARM = (0, 1, 2), ("stall3", "nostall"), "cutoff"
TRAINING_KINDS = ("warmup", "branch", "bca_branch")
REUSE_KEYS = ("id", "kind", "seed_index", "arm", "world", "kappa")


def manifest_files(root=None):
    root = root or HERE
    return sorted(p.name for p in root.iterdir() if p.is_file() and p.suffix in (".py", ".md", ".json")
                  and p.name not in ("CF_MANIFEST.json", "CF_APPROVAL.json"))


def manifest_body(root=None):
    root = root or HERE
    exploratory = oe.manifest_body()
    return {"schema": "knee_s10_candidate_v2.confirmation_v1.manifest.v1", "doses": list(DOSES),
            "files": {f: sha_file(root / f) for f in manifest_files(root)},
            "overestimation_v1_files": exploratory["files"],
            "native_screen_files": exploratory["native_screen_files"], "v3_pins": exploratory["v3_pins"]}


def check_approval(approval_path=None, manifest_path=None, run_dir=None, root=None):
    approval_path, manifest_path, run_dir = approval_path or APPROVAL, manifest_path or MANIFEST, run_dir or RUN
    root = root or manifest_path.parent
    require(manifest_path.is_file(), "CF_MANIFEST.json is missing: the package is not sealed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed = [f for f, d in manifest["files"].items() if sha_file(root / f) != d]
    changed += [f"overestimation_v1/{f}" for f, d in manifest["overestimation_v1_files"].items() if sha_file(OE / f) != d]
    changed += [f"native_screen/{f}" for f, d in manifest["native_screen_files"].items() if sha_file(NS / f) != d]
    changed += [f"v3/{f}" for f, d in manifest["v3_pins"].items() if sha_file(V3 / f) != d]
    require(not changed, f"Sealed files changed: {changed}")
    require(approval_path.is_file(), "CF_APPROVAL.json is missing: the user's own approval is required")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    previous = sorted((p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("attempt_")), key=lambda p: int(p.name[8:])) \
        if run_dir.is_dir() else []
    attempt = approval.get("attempt")
    require(type(attempt) is int and attempt == len(previous) + 1, f"Approval must name attempt {len(previous) + 1}")
    require(attempt <= MAX_ATTEMPTS, f"There is no attempt {MAX_ATTEMPTS + 1}")
    text = approval.get("approval", "")
    missing = [w for w in (WORDS[0].format(attempt=attempt),) + WORDS[1:] if w not in text]
    require(not missing, "Approval wording missing: " + " | ".join(missing))
    require(approval.get("package_manifest_sha256") == sha_file(manifest_path), "Approval names a different package")
    history = []
    for p in previous:
        result = json.loads((p / "CF_RESULT.json").read_text(encoding="utf-8"))
        history.append({"attempt": p.name, "reading": result.get("reading")})
    if history:
        last = history[-1]["reading"]
        if last == "INCONCLUSIVE":
            evals = [oe.S._read(p / "report.json") for p in previous[-1].iterdir() if p.is_dir() and p.name.startswith("eval_")]
            require(evals and all(r and r.get("status") == "completed" and float(r["instrument"]["I1_max_rel"]) <= INSTRUMENT_CAUSE_REL
                                  for r in evals),
                    "A re-attempt after INCONCLUSIVE needs every evaluation's I1 identity within 1e-6 relative (the named cause)")
        else:
            require(last == "INCOMPLETE", f"No re-attempt after {last}")
        require(attempt < 3 or last == "INCOMPLETE", "Attempt 3 (amendment A3) follows only an INCOMPLETE attempt 2")
    return {"approval": approval, "attempt": attempt, "manifest_sha256": sha_file(manifest_path),
            "history": history, "previous_attempts": [str(p) for p in previous]}


def integrity_record():
    record = oe.integrity_record()
    files = dict(record["files"])
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.suffix in (".py", ".json", ".md"):
            files[str(p)] = sha_file(p)
    return {"package_dir": str(HERE), "files": files}


def default_dispatch(job_dir, cap):
    command = [sys.executable, "-B", "-u", str(HERE / "cf_worker.py"), "--job", str(job_dir / "job.json")]
    env = dict(os.environ)
    env.update(step03.WORKER_ENVIRONMENT)
    return S.supervisor()(command, job_dir, cap, env=env, job_id=job_dir.name)


class Runner:
    def __init__(self, plan, directory, options, dispatch):
        self.plan, self.directory, self.options, self.dispatch = plan, directory, options, dispatch
        self.caps = {"training_seconds": 0.0, "evaluation_seconds": 0.0, "evaluation_transitions": 0}
        self.dirs, self.reports, self.units = {}, {}, {}

    def reusable(self, spec):
        if spec["kind"] == "cf_evaluate":                    # evaluations always rerun
            return None, None
        for prev in reversed(self.plan["previous_attempts"]):
            unit_dir = Path(prev) / spec["id"]
            sup, job = oe.S._read(unit_dir / "SUPERVISOR_RESULT.json") or {}, oe.S._read(unit_dir / "job.json") or {}
            if sup.get("completed") and all(job.get(k) == spec.get(k) for k in REUSE_KEYS):
                report = oe.S._read(unit_dir / "report.json")
                if report and report.get("status") == "completed":
                    return unit_dir, report
        return None, None

    def run(self, spec):
        unit_dir, report = self.reusable(spec)
        if report is not None:
            self.dirs[spec["id"]], self.reports[spec["id"]] = unit_dir, report
            self.units[spec["id"]] = f"reused from {unit_dir.parent.name}"
            return report
        if spec["kind"] in TRAINING_KINDS:
            cap, transitions, cap_key = min(JOB_SECONDS, TRAIN_SECONDS - self.caps["training_seconds"]), 0, "training_seconds"
        else:
            cap = min(EVAL_JOB_SECONDS, EVAL_SECONDS - self.caps["evaluation_seconds"])
            transitions, cap_key = min(EVAL_JOB_TRANSITIONS, EVAL_TRANSITIONS - self.caps["evaluation_transitions"]), "evaluation_seconds"
            if transitions <= 0:
                raise step03.CapExhausted(f"transition cap exhausted before {spec['id']}")
        if cap <= 0:
            raise step03.CapExhausted(f"cap exhausted before {spec['id']}")
        unit_dir = self.directory / spec["id"]
        unit_dir.mkdir(parents=False, exist_ok=False)
        job = {**spec, "timeout_seconds": float(cap), "transitions": int(transitions), **self.options}
        step03._write_new(unit_dir / "job.json", job)
        outcome = self.dispatch(unit_dir, float(cap))
        report = oe.S._read(unit_dir / "report.json")
        self.caps[cap_key] += float(outcome.get("elapsed_seconds", cap))
        if cap_key == "evaluation_seconds":
            self.caps["evaluation_transitions"] += int((report or {}).get("transitions", transitions) or 0)
        complete = bool(report and report.get("status") == "completed" and outcome.get("exit_code") == 0
                        and not outcome.get("timed_out") and not outcome.get("interrupted"))
        step03._write_new(unit_dir / "SUPERVISOR_RESULT.json", {**outcome, "completed": complete, "cap_seconds": cap})
        self.dirs[spec["id"]] = unit_dir
        self.units[spec["id"]] = "completed" if complete else "failed"
        if not complete:
            raise U.UnitFailed(f"job {spec['id']} did not complete: {outcome}")
        self.reports[spec["id"]] = report
        return report


def execute(dispatch=None, *, run_dir=None, approval_path=None, manifest_path=None, root=None, integrity=None):
    plan = check_approval(approval_path, manifest_path, run_dir, root)
    found = S.inputs()
    directory = (run_dir or RUN) / f"attempt_{plan['attempt']}"
    directory.mkdir(parents=True, exist_ok=False)
    options = {"provenance": "research", "collection_dir": found["collection_dir"], "test_seeds": None,
               "training_seeds": None, "visited_seed": None, "integrity": integrity or integrity_record()}
    step03._write_new(directory / "ATTEMPT.json", sanitize({**plan, **found, "started_utc": datetime.now(timezone.utc).isoformat()}))
    runner = Runner(plan, directory, options, dispatch or default_dispatch)
    result = {"package": "knee_s10_candidate_v2/confirmation_v1", "doses": list(DOSES), "attempt": plan["attempt"],
              "manifest_sha256": plan["manifest_sha256"], "units": runner.units, "caps": runner.caps, "reading": "INCOMPLETE"}
    try:
        results = {}
        for world in WORLDS:
            sfx = f"_{world}"
            for i in SEEDS:
                runner.run({"id": f"warmup_s{i}{sfx}", "kind": "warmup", "seed_index": i, "arm": ARM, "world": world})
            for i in SEEDS:
                runner.run({"id": f"branch_s{i}{sfx}", "kind": "branch", "seed_index": i, "arm": ARM, "world": world,
                            "warmup_dir": str(runner.dirs[f"warmup_s{i}{sfx}"])})
            for kappa in DOSES:
                for i in SEEDS:
                    runner.run({"id": f"bca_branch_s{i}_k{kappa:g}{sfx}", "kind": "bca_branch", "seed_index": i, "arm": ARM, "world": world,
                                "kappa": kappa, "warmup_dir": str(runner.dirs[f"warmup_s{i}{sfx}"]),
                                "native_branch_dir": str(runner.dirs[f"branch_s{i}{sfx}"])})
            native, bca = {}, {k: {} for k in DOSES}
            for i in SEEDS:
                native[i] = runner.run({"id": f"eval_native_s{i}{sfx}", "kind": "cf_evaluate", "seed_index": i, "arm": "native", "world": world,
                                        "runs": {"warmup": str(runner.dirs[f"warmup_s{i}{sfx}"]), "branch": str(runner.dirs[f"branch_s{i}{sfx}"])}})
            for kappa in DOSES:
                for i in SEEDS:
                    bca[kappa][i] = runner.run({"id": f"eval_bca_s{i}_k{kappa:g}{sfx}", "kind": "cf_evaluate", "seed_index": i, "arm": "bca",
                                                "world": world, "kappa": kappa,
                                                "runs": {"warmup": str(runner.dirs[f"warmup_s{i}{sfx}"]),
                                                         "branch": str(runner.dirs[f"bca_branch_s{i}_k{kappa:g}{sfx}"])}})
            results[world] = {"native": native, "bca": bca}
        decision = CM.decide(results)
        step03._write_new(directory / "decision.json", sanitize(decision))
        result.update({k: decision[k] for k in ("reading", "validity", "C1", "C2", "all_labels", "reasons", "profile", "paired")})
    except (U.UnitFailed, step03.CapExhausted) as exc:
        result.update(reading="INCOMPLETE", reasons=[f"{type(exc).__name__}: {exc}"])
    except BaseException as exc:
        result.update(reading="INCOMPLETE", reasons=[f"{type(exc).__name__}: {exc}"], traceback=traceback.format_exc())
        _write_result(directory, result)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return result
    _write_result(directory, result)
    return result


def _write_result(directory, result):
    try:
        result["physical"] = U.physical(directory, None)
    except BaseException:
        pass
    step03._write_new(directory / "CF_RESULT.json", sanitize({**result, "finished_utc": datetime.now(timezone.utc).isoformat()}))


def decide_only(run_dir=None):
    run_dir = run_dir or RUN
    attempts = sorted((p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("attempt_")), key=lambda p: int(p.name[8:]))
    require(attempts, "No attempt to decide")
    reports = {}
    for p in attempts:
        for unit in p.iterdir():
            r = oe.S._read(unit / "report.json")
            if unit.is_dir() and r and r.get("status") == "completed":
                reports[unit.name] = r
    results = {}
    for world in WORLDS:
        results[world] = {"native": {i: reports[f"eval_native_s{i}_{world}"] for i in SEEDS},
                          "bca": {k: {i: reports[f"eval_bca_s{i}_k{k:g}_{world}"] for i in SEEDS} for k in DOSES}}
    return CM.decide(results)


def summary(out):
    return {k: out.get(k) for k in ("reading", "validity", "C1", "C2", "all_labels", "reasons", "profile", "paired", "attempt", "caps")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--execute", action="store_true")
    group.add_argument("--decide", action="store_true")
    args = parser.parse_args()
    print(json.dumps(summary(execute() if args.execute else decide_only()), indent=2, default=str))
