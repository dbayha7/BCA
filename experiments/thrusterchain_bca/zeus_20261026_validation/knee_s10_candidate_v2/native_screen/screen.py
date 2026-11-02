"""Native screen for candidate knee_s10_v2: the supervisor (SCREEN_RULE.md). It never imports JAX: every job is a
separate `screen_worker.py` process under matched_policy_pilot_v1's supervise() (an owned process group, killed at
its cap). Units, in order, fail-fast:

  preflight (exempt), g1 (CPU; NOT REPRESENTABLE stops the screen), 12 warmups, r0 (stop rules), 12 native branches,
  3 BC-only actors, 3 native evaluations, the R1 reach read (fails in every seed: STOP, no BCA unit trains),
  12 BCA branches at kappa, 3 BCA evaluations, the decision (screen_math.decide).

One attempt by default; after INCOMPLETE a second attempt under a new approval reuses every unit an earlier attempt
(amendment A1: a third, after attempt 2 failed on the loader's reset regeneration, the same way)
completed (same id, kind, seed, arm, world) and runs the rest. SCREEN_RESULT.json is always written.

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
CAND = HERE.parent
V3 = CAND.parent / "knee_edge_ood_v3"
F2 = CAND / "formal_g2"
for _p in (str(V3), str(F2), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import screen_math as SM                                     # noqa: E402
import step03                                                # noqa: E402
import units as U                                            # noqa: E402
from core import VALIDATION, module, require, sha_file       # noqa: E402
from stage1_worker import sanitize                           # noqa: E402

RUN, COLLECTION = HERE / "screen_run", HERE / "collection"
MANIFEST, APPROVAL = HERE / "SCREEN_MANIFEST.json", HERE / "SCREEN_APPROVAL.json"
KAPPA = 63.0
LOADER_FAULT = "initial state lineage differs"           # the fault amendment A1 names (attempt 2, g1 unit)
WORDS = (
    "I approve the native screen for candidate knee_s10_v2 under OOD definition v4 r4 at kappa 63, attempt {attempt}",
    "I adopt definition v4 r4 for this candidate",
    "RS1: the scored anchors are retained-supported; I overrule the start-share primary miss (121 of 128 against 122, "
    "a one-draw threshold); the secondary's agreement with D3 is on record",
    "the BCA arm at kappa 63 is approved",
    "the screen is exploratory and supports no protocol claim",
)
JOB_SECONDS, TRAIN_SECONDS = 1_800, 10 * 3_600
EVAL_SECONDS, EVAL_TRANSITIONS = 2 * 3_600, 36_000_000
G1_SECONDS, G1_TRANSITIONS, PREFLIGHT_SECONDS = 3_600, 9_000_000, 600
SEEDS, WORLDS, ARMS = (0, 1, 2), ("stall3", "nostall"), ("cutoff", "random")
TRAINING_KINDS = ("warmup", "branch", "bc_only", "bca_branch")
REUSE_KEYS = ("id", "kind", "seed_index", "arm", "world")


def name(i, arm, world):
    return f"s{i}_{arm}_{world}"


FORKS = [name(i, a, w) for i in SEEDS for w in WORLDS for a in ARMS]


def manifest_files(root=None):
    root = root or HERE
    return sorted(p.name for p in root.iterdir() if p.is_file() and p.suffix in (".py", ".md", ".json")
                  and p.name not in ("SCREEN_MANIFEST.json", "SCREEN_APPROVAL.json"))


def manifest_body(root=None):
    """What sealing writes: this package's files, the collection record, and the files of the candidate folders the
    worker loads by path (the formal G2 package, the prefix rule, v1's pi_ref and recorder, the g1 modules, r4)."""
    root = root or HERE
    v1 = CAND.parent / "knee_s10_candidate_v1"
    outside = {"collection/collection.json": COLLECTION / "collection.json", "../prefix_rule.py": CAND / "prefix_rule.py",
               "../../ood_definition_v4/REVISION_r4.md": CAND.parent / "ood_definition_v4" / "REVISION_r4.md",
               "../../ood_definition_v4/DRAFT_DEFINITION.md": CAND.parent / "ood_definition_v4" / "DRAFT_DEFINITION.md",
               "../../knee_s10_candidate_v1/pi_ref.py": v1 / "pi_ref.py",
               "../../knee_s10_candidate_v1/recorder_wide.py": v1 / "recorder_wide.py",
               "../formal_g2/formal_g2_run/attempt_1/report.json": F2 / "formal_g2_run" / "attempt_1" / "report.json",
               "../formal_g2/formal_g2_run/attempt_1/collection.json": F2 / "formal_g2_run" / "attempt_1" / "collection.json",
               "../retained_support/rs1_result.json": CAND / "retained_support" / "rs1_result.json"}
    for p in sorted(F2.iterdir()):
        if p.is_file() and p.suffix in (".py", ".json", ".md"):
            outside[f"../formal_g2/{p.name}"] = p
    for p in sorted((CAND / "g1").iterdir()):
        if p.is_file() and p.suffix in (".py", ".md"):
            outside[f"../g1/{p.name}"] = p
    body = {f: sha_file(root / f) for f in manifest_files(root)}
    body.update({k: sha_file(v) for k, v in outside.items()})
    return {"schema": "knee_s10_candidate_v2.native_screen.manifest.v1", "kappa": KAPPA, "files": body,
            "v3_pins": {f: sha_file(V3 / f) for f in ("A4_MANIFEST.json", "EXPLORATORY_MANIFEST.json")}}


def check_approval(approval_path=None, manifest_path=None, run_dir=None, root=None):
    approval_path, manifest_path, run_dir = approval_path or APPROVAL, manifest_path or MANIFEST, run_dir or RUN
    root = root or manifest_path.parent
    require(manifest_path.is_file(), "SCREEN_MANIFEST.json is missing: the screen is not sealed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed = [f for f, d in manifest["files"].items() if sha_file(root / f) != d]
    require(not changed, f"Sealed files changed: {changed}")
    require(approval_path.is_file(), "SCREEN_APPROVAL.json is missing: the user's own approval is required")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    previous = sorted((p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("attempt_")), key=lambda p: int(p.name[8:])) \
        if run_dir.is_dir() else []
    attempt = approval.get("attempt")
    require(type(attempt) is int and attempt == len(previous) + 1, f"Approval must name attempt {len(previous) + 1}")
    require(attempt <= 3, "There is no attempt 4")
    text = approval.get("approval", "")
    missing = [w for w in (WORDS[0].format(attempt=attempt),) + WORDS[1:] if w not in text]
    require(not missing, "Approval wording missing: " + " | ".join(missing))
    require(approval.get("package_manifest_sha256") == sha_file(manifest_path), "Approval names a different package")
    history = []
    for p in previous:
        result = json.loads((p / "SCREEN_RESULT.json").read_text(encoding="utf-8"))
        history.append({"attempt": p.name, "reading": result.get("reading")})
    if history:
        require(history[-1]["reading"] == "INCOMPLETE", f"No re-attempt after {history[-1]['reading']}")
    if attempt == 3:                                         # amendment A1: only after the named loader fault
        require(any(LOADER_FAULT in ((_read(p / "report.json") or {}).get("error") or "") for p in previous[-1].iterdir() if p.is_dir()),
                "Attempt 3 is allowed only after an attempt 2 that failed on the dataset loader's reset regeneration")
    return {"approval": approval, "attempt": attempt, "manifest_sha256": sha_file(manifest_path), "manifest": manifest,
            "history": history, "previous_attempts": [str(p) for p in previous]}


def inputs():
    record = json.loads((COLLECTION / "collection.json").read_text(encoding="utf-8"))
    for world, spec in record["datasets"].items():
        require((COLLECTION / spec["file"]).is_file(), f"{world} dataset file missing")
    return {"collection_dir": str(COLLECTION), "collection_sha256": sha_file(COLLECTION / "collection.json")}


def integrity_record():
    """The v3 dependency closure plus every file of this package and of the candidate folders a job loads by path."""
    files = {str(p): sha_file(p) for p in U.dependency_closure()}
    extra = [p for p in sorted(HERE.iterdir()) if p.is_file() and p.suffix in (".py", ".json", ".md")]
    extra += [COLLECTION / "collection.json"]
    extra += sorted(p for p in (CAND / "g1").iterdir() if p.suffix in (".py", ".md"))
    extra += [CAND / "prefix_rule.py", CAND / "G2_DEV_RULE_v2.md"]
    extra += sorted(p for p in F2.iterdir() if p.is_file() and p.suffix in (".py", ".json", ".md"))
    extra += [F2 / "formal_g2_run" / "attempt_1" / n for n in ("report.json", "collection.json")]
    v1 = CAND.parent / "knee_s10_candidate_v1"
    extra += [v1 / "pi_ref.py", v1 / "recorder_wide.py", v1 / "formal_g2" / "g2_formal.py", v1 / "formal_g2" / "g2_seeds.py",
              v1 / "formal_g2" / "g2_seeds.json", v1 / "formal_g2" / "d4_bounds.json"]
    for p in extra:
        require(p.is_file(), f"integrity record: missing {p}")
        files[str(p)] = sha_file(p)
    return {"package_dir": str(HERE), "files": files}


def supervisor():
    return module("_native_screen_supervisor", VALIDATION / "matched_policy_pilot_v1/launch.py").supervise


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


class Stop(Exception):
    pass


class Screen:
    def __init__(self, plan, directory, options, dispatch):
        self.plan, self.directory, self.options, self.dispatch = plan, directory, options, dispatch
        self.caps = {"training_seconds": 0.0, "evaluation_seconds": 0.0, "evaluation_transitions": 0,
                     "g1_seconds": 0.0, "g1_transitions": 0}
        self.dirs, self.reports, self.units = {}, {}, {}

    def reusable(self, spec):
        for prev in reversed(self.plan["previous_attempts"]):
            unit_dir = Path(prev) / spec["id"]
            sup, job = _read(unit_dir / "SUPERVISOR_RESULT.json") or {}, _read(unit_dir / "job.json") or {}
            if sup.get("completed") and all(job.get(k) == spec.get(k) for k in REUSE_KEYS):
                report = _read(unit_dir / "report.json")
                if report and report.get("status") == "completed":
                    return unit_dir, report
        return None, None

    def run(self, spec):
        kind = spec["kind"]
        unit_dir, report = self.reusable(spec)
        if report is not None:
            self.dirs[spec["id"]], self.reports[spec["id"]] = unit_dir, report
            self.units[spec["id"]] = f"reused from {unit_dir.parent.name}"
            return report
        if kind in TRAINING_KINDS:
            cap, transitions, cap_key = min(JOB_SECONDS, TRAIN_SECONDS - self.caps["training_seconds"]), 0, "training_seconds"
        elif kind == "g1":
            cap, transitions, cap_key = G1_SECONDS - self.caps["g1_seconds"], G1_TRANSITIONS - self.caps["g1_transitions"], "g1_seconds"
        elif kind == "preflight":
            cap, transitions, cap_key = PREFLIGHT_SECONDS, 0, "preflight_seconds"
        else:
            cap, transitions, cap_key = EVAL_SECONDS - self.caps["evaluation_seconds"], EVAL_TRANSITIONS - self.caps["evaluation_transitions"], "evaluation_seconds"
        if cap <= 0 or (kind in ("g1", "r0", "evaluate", "evaluate_bca") and transitions <= 0):
            raise step03.CapExhausted(f"cap exhausted before {spec['id']}")
        unit_dir = self.directory / spec["id"]
        unit_dir.mkdir(parents=False, exist_ok=False)
        job = {**spec, "timeout_seconds": float(cap), "transitions": int(transitions), **self.options}
        step03._write_new(unit_dir / "job.json", job)
        outcome = self.dispatch(unit_dir, float(cap))
        report = _read(unit_dir / "report.json")
        elapsed = float(outcome.get("elapsed_seconds", cap))
        self.caps[cap_key] = self.caps.get(cap_key, 0.0) + elapsed
        if cap_key == "evaluation_seconds":
            self.caps["evaluation_transitions"] += int((report or {}).get("transitions", transitions) or 0)
        if cap_key == "g1_seconds":
            self.caps["g1_transitions"] += int((report or {}).get("transitions", transitions) or 0)
        complete = bool(report and report.get("status") == "completed" and outcome.get("exit_code") == 0
                        and not outcome.get("timed_out") and not outcome.get("interrupted"))
        step03._write_new(unit_dir / "SUPERVISOR_RESULT.json", {**outcome, "completed": complete, "cap_seconds": cap})
        self.dirs[spec["id"]] = unit_dir
        self.units[spec["id"]] = "completed" if complete else "failed"
        if not complete:
            raise U.UnitFailed(f"job {spec['id']} did not complete: {outcome}")
        self.reports[spec["id"]] = report
        return report

    def runs_for(self, i, branch_prefix="branch"):
        runs = {}
        for w in WORLDS:
            for a in ARMS:
                n = name(i, a, w)
                runs[f"warmup_{n}"] = str(self.dirs[f"warmup_{n}"])
                runs[f"branch_{n}"] = str(self.dirs[f"{branch_prefix}_{n}"])
        if branch_prefix == "branch":
            runs[f"bc_only_s{i}"] = str(self.dirs[f"bc_only_s{i}"])
        return runs


def execute(dispatch=None, *, run_dir=None, approval_path=None, manifest_path=None, root=None, integrity=None):
    plan = check_approval(approval_path, manifest_path, run_dir, root)
    found = inputs()
    directory = (run_dir or RUN) / f"attempt_{plan['attempt']}"
    directory.mkdir(parents=True, exist_ok=False)
    options = {"provenance": "research", "collection_dir": found["collection_dir"], "test_seeds": None,
               "training_seeds": None, "visited_seed": None, "integrity": integrity or integrity_record()}
    step03._write_new(directory / "ATTEMPT.json", sanitize({**{k: v for k, v in plan.items() if k != "manifest"}, **found,
                                                            "started_utc": datetime.now(timezone.utc).isoformat()}))
    screen = Screen(plan, directory, options, dispatch or default_dispatch)
    result = {"screen": "knee_s10_candidate_v2/native_screen", "kappa": KAPPA, "attempt": plan["attempt"],
              "manifest_sha256": plan["manifest_sha256"], "units": screen.units, "caps": screen.caps, "reading": "INCOMPLETE"}
    try:
        pre = screen.run({"id": "preflight", "kind": "preflight"})
        result["preflight"] = pre.get("kind")
        g1 = screen.run({"id": "g1", "kind": "g1"})
        result["g1"] = g1["decision"]
        if g1["decision"]["headline"] != "PASS":
            raise Stop("INCONCLUSIVE", [f"G1 {g1['decision']['headline']}: " + "; ".join(g1["decision"]["all_labels"])])
        for i in SEEDS:
            for w in WORLDS:
                for a in ARMS:
                    screen.run({"id": f"warmup_{name(i, a, w)}", "kind": "warmup", "seed_index": i, "arm": a, "world": w})
        r0 = screen.run({"id": "r0", "kind": "r0", "runs": {k: str(v) for k, v in screen.dirs.items() if k.startswith("warmup_")}})
        result["r0"] = {k: r0[k] for k in ("seeds", "adequacy_ok", "fail_early")}
        if not r0["adequacy_ok"]:
            raise Stop("INCONCLUSIVE", ["R0 adequacy: J_E1(10k) - J_E1(untrained) < +50 in some seed"])
        if r0["fail_early"]:
            raise Stop("STOP: not reached", ["R0: the stall CUTOFF R_deep- rate is below 0.05 in every seed at 10k; no branch trained"])
        for i in SEEDS:
            for w in WORLDS:
                for a in ARMS:
                    n = name(i, a, w)
                    screen.run({"id": f"branch_{n}", "kind": "branch", "seed_index": i, "arm": a, "world": w,
                                "warmup_dir": str(screen.dirs[f"warmup_{n}"])})
        for i in SEEDS:
            screen.run({"id": f"bc_only_s{i}", "kind": "bc_only", "seed_index": i})
        native = {}
        for i in SEEDS:
            native[i] = screen.run({"id": f"evaluate_s{i}", "kind": "evaluate", "seed_index": i, "runs": screen.runs_for(i)})
        r1, r1_reads = SM.r1_consensus(native)
        result["R1_before_bca"] = {"consensus": r1, "reads": r1_reads}
        if r1 == "fails":
            raise Stop("STOP: not reached", ["R1 fails in every seed: the actor does not enter W on this bed; no BCA unit trained"])
        for i in SEEDS:
            for w in WORLDS:
                for a in ARMS:
                    n = name(i, a, w)
                    screen.run({"id": f"bca_branch_{n}", "kind": "bca_branch", "seed_index": i, "arm": a, "world": w,
                                "warmup_dir": str(screen.dirs[f"warmup_{n}"]), "native_branch_dir": str(screen.dirs[f"branch_{n}"])})
        bca = {}
        for i in SEEDS:
            bca[i] = screen.run({"id": f"evaluate_bca_s{i}", "kind": "evaluate_bca", "seed_index": i,
                                 "runs": screen.runs_for(i, "bca_branch")})
        decision = SM.decide(native, bca, KAPPA, g1["decision"])
        step03._write_new(directory / "decision.json", sanitize(decision))
        result.update(reading=decision["reading"], reasons=decision["reasons"], consensus=decision["consensus"],
                      validity=decision["validity"], placebo=decision["placebo"])
    except Stop as stop:
        result.update(reading=stop.args[0], reasons=list(stop.args[1]))
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
    step03._write_new(directory / "SCREEN_RESULT.json", sanitize({**result, "finished_utc": datetime.now(timezone.utc).isoformat()}))


def decide_only():
    plan = check_approval()
    previous = plan["previous_attempts"]
    require(previous, "No attempt to decide")
    directory = Path(previous[-1])
    native = {i: _read(directory / f"evaluate_s{i}" / "report.json") for i in SEEDS}
    bca = {i: _read(directory / f"evaluate_bca_s{i}" / "report.json") for i in SEEDS}
    g1 = (_read(directory / "g1" / "report.json") or {}).get("decision")
    decision = SM.decide({i: r for i, r in native.items() if r}, {i: r for i, r in bca.items() if r}, KAPPA, g1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    step03._write_new(directory / f"decision_recomputed_{stamp}.json", sanitize(decision))
    return decision


def summary(out):
    return {k: out.get(k) for k in ("reading", "reasons", "consensus", "validity", "attempt", "caps")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--execute", action="store_true")
    group.add_argument("--decide", action="store_true")
    args = parser.parse_args()
    print(json.dumps(summary(execute() if args.execute else decide_only()), indent=2, default=str))
