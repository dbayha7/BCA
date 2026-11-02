"""Native screen for candidate knee_s10_v2 under r5: the supervisor (SCREEN_RULE_r5.md). It never imports JAX: every
job is a separate `screen_worker_r5.py` process under matched_policy_pilot_v1's supervise() (an owned process group,
killed at its cap).

Stage 0 (exempt from the attempt count): preflight, i0 (instrument), g1 (reported). The attempt is counted when the
first warmup starts (TRAINING_STARTED.json). Then: 12 warmups, r0, 12 native branches, 3 BC-only actors, 3 native
evaluations, the pre-BCA readings (validity, R1, R2, O1), 12 BCA branches at kappa, 3 BCA evaluations, the decision
(screen_math_r5.decide). SCREEN_RESULT.json is always written. `--dry-run` runs dry_run.main (precondition 3 of r5),
`--decide` recomputes the decision from the last run's saved reports.

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
NATIVE = CAND / "native_screen"
V3 = CAND.parent / "knee_edge_ood_v3"
F2 = CAND / "formal_g2"
for _p in (str(V3), str(F2), str(NATIVE), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import screen_math_r5 as SM                                  # noqa: E402
import device_kinds as DK                                    # noqa: E402
import step03                                                # noqa: E402
import units as U                                            # noqa: E402
from core import VALIDATION, module, require, sha_file       # noqa: E402
from stage1_worker import sanitize                           # noqa: E402

RUN, COLLECTION = HERE / "screen_run", NATIVE / "collection"
MANIFEST, APPROVAL = HERE / "SCREEN_MANIFEST.json", HERE / "SCREEN_APPROVAL.json"
DRY_RUN_RECEIPT = HERE / "dry_run" / "DRY_RUN_RECEIPT.json"
KAPPA = 63.0
WORDS = (
    "I approve the native screen for candidate knee_s10_v2 under OOD definition v4 r5 at kappa 63, attempt {attempt}",
    "I adopt definition v4 r5 for this candidate",
    "RS1: the scored anchors are retained-supported; I overrule the start-share primary miss (121 of 128 against 122, "
    "a one-draw threshold); the secondary's agreement with D3 is on record",
    "the BCA arm at kappa 63 is approved",
    "the screen is exploratory and supports no protocol claim",
)
JOB_SECONDS, TRAIN_SECONDS = 1_800, 14 * 3_600                 # amendment A4: 10 h -> 14 h (Titan's measured speed)
EVAL_SECONDS, EVAL_TRANSITIONS = 2 * 3_600, 36_000_000
G1_SECONDS, G1_TRANSITIONS = 3_600, 9_000_000
PREFLIGHT_SECONDS, I0_SECONDS, I0_TRANSITIONS = 600, 600, 1_000_000
MAX_ATTEMPTS, MAX_STAGE0_FAILURES = 3, 3                       # amendment A5: 2 -> 3 (attempt 2 killed from outside)
SEEDS, WORLDS, ARMS = (0, 1, 2), ("stall3", "nostall"), ("cutoff", "random")
TRAINING_KINDS = ("warmup", "branch", "bc_only", "bca_branch")
GPU_KINDS = TRAINING_KINDS + ("r0", "evaluate", "evaluate_bca")      # reused only on the same device kind (A2)
STAGE0_KINDS = ("preflight", "i0", "g1")
REUSE_KEYS = ("id", "kind", "seed_index", "arm", "world")
DRY_RUN_HASHED = ("screen_worker_r5.py", "screen_math_r5.py", "i0_unit.py", "g1_unit_r5.py", "device_kinds.py",
                  "../native_screen/screen_bind.py", "../native_screen/screen_data.py")


def name(i, arm, world):
    return f"s{i}_{arm}_{world}"


FORKS = [name(i, a, w) for i in SEEDS for w in WORLDS for a in ARMS]


def manifest_files(root=None):
    root = root or HERE
    return sorted(p.name for p in root.iterdir() if p.is_file() and p.suffix in (".py", ".md", ".json")
                  and p.name not in ("SCREEN_MANIFEST.json", "SCREEN_APPROVAL.json"))


def manifest_body(root=None):
    """What sealing writes: this package's files, the dry-run receipt, the r4 modules it imports, the collection
    record, and the candidate files the worker loads by path."""
    root = root or HERE
    v1 = CAND.parent / "knee_s10_candidate_v1"
    outside = {"../native_screen/collection/collection.json": COLLECTION / "collection.json",
               "../prefix_rule.py": CAND / "prefix_rule.py",
               "../../ood_definition_v4/REVISION_r5.md": CAND.parent / "ood_definition_v4" / "REVISION_r5.md",
               "../../ood_definition_v4/REVISION_r4.md": CAND.parent / "ood_definition_v4" / "REVISION_r4.md",
               "../../ood_definition_v4/DRAFT_DEFINITION.md": CAND.parent / "ood_definition_v4" / "DRAFT_DEFINITION.md",
               "../../ood_definition_v4/DESIGN_AUDIT_r4_20261101.md": CAND.parent / "ood_definition_v4" / "DESIGN_AUDIT_r4_20261101.md",
               "../../knee_s10_candidate_v1/pi_ref.py": v1 / "pi_ref.py",
               "../../knee_s10_candidate_v1/recorder_wide.py": v1 / "recorder_wide.py",
               "../formal_g2/formal_g2_run/attempt_1/report.json": F2 / "formal_g2_run" / "attempt_1" / "report.json",
               "../formal_g2/formal_g2_run/attempt_1/collection.json": F2 / "formal_g2_run" / "attempt_1" / "collection.json",
               "../retained_support/rs1_result.json": CAND / "retained_support" / "rs1_result.json"}
    if (root / "dry_run" / "DRY_RUN_RECEIPT.json").is_file():      # required by seal_r5.check_dry_run; hashed when present
        outside["dry_run/DRY_RUN_RECEIPT.json"] = root / "dry_run" / "DRY_RUN_RECEIPT.json"
    for n in ("screen_bind.py", "screen_data.py", "screen_seeds.py", "screen_seeds.json", "prepare.py"):
        outside[f"../native_screen/{n}"] = NATIVE / n
    for p in sorted(F2.iterdir()):
        if p.is_file() and p.suffix in (".py", ".json", ".md"):
            outside[f"../formal_g2/{p.name}"] = p
    for p in sorted((CAND / "g1").iterdir()):
        if p.is_file() and p.suffix in (".py", ".md"):
            outside[f"../g1/{p.name}"] = p
    body = {f: sha_file(root / f) for f in manifest_files(root)}
    body.update({k: sha_file(v) for k, v in outside.items()})
    return {"schema": "knee_s10_candidate_v2.native_screen_r5.manifest.v1", "kappa": KAPPA, "files": body,
            "v3_pins": {f: sha_file(V3 / f) for f in ("A4_MANIFEST.json", "EXPLORATORY_MANIFEST.json")}}


def dry_run_hashes(root=None):
    root = root or HERE
    return {f: sha_file(root / f) for f in DRY_RUN_HASHED}


def check_dry_run(root=None, receipt_path=None):
    """Precondition 3: a completed dry run whose recorded module hashes equal the files being sealed or run."""
    root = root or HERE
    receipt_path = receipt_path or (root / "dry_run" / "DRY_RUN_RECEIPT.json")
    require(receipt_path.is_file(), "dry_run/DRY_RUN_RECEIPT.json is missing: run `screen_r5.py --dry-run` in WSL first")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("status") == "completed", f"The dry run did not complete: {receipt.get('status')}")
    require(receipt.get("hashes") == dry_run_hashes(root), "The dry run exercised different module files than the ones here")
    require(receipt.get("platform") == "Linux" and receipt.get("real_dispatch") is True and receipt.get("device_kinds_ok") is True,
            "The dry run must be the real one in the recorded Linux environment (real dispatch, an allowed device kind)")
    require(receipt.get("device_kind") in DK.ALLOWED_DEVICE_KINDS, "The dry run's device kind is not an allowed one (A2)")
    return receipt


def _runs(run_dir):
    return sorted((p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("run_")), key=lambda p: p.name) if run_dir.is_dir() else []


def recorded_reading(result, counted):
    """Amendment A5: every ending the supervisor sees writes SCREEN_RESULT.json, so a counted attempt without a
    readable one was killed from outside (attempt 2: a forced Windows restart) and reads INCOMPLETE."""
    if result is None:
        return "INCOMPLETE" if counted else None
    return result.get("reading")


def check_approval(approval_path=None, manifest_path=None, run_dir=None, root=None, require_dry_run=True):
    approval_path, manifest_path, run_dir = approval_path or APPROVAL, manifest_path or MANIFEST, run_dir or RUN
    root = root or manifest_path.parent
    require(manifest_path.is_file(), "SCREEN_MANIFEST.json is missing: the screen is not sealed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed = [f for f, d in manifest["files"].items() if sha_file(root / f) != d]
    require(not changed, f"Sealed files changed: {changed}")
    if require_dry_run:
        check_dry_run(root)
    require(approval_path.is_file(), "SCREEN_APPROVAL.json is missing: the user's own approval is required")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    runs = _runs(run_dir)
    attempts = [p for p in runs if (p / "TRAINING_STARTED.json").is_file()]
    stage0_failures = [p for p in runs if p not in attempts]
    require(len(stage0_failures) < MAX_STAGE0_FAILURES, f"{len(stage0_failures)} stage-0 failures on record: re-seal the package")
    attempt = approval.get("attempt")
    require(type(attempt) is int and attempt == len(attempts) + 1, f"Approval must name attempt {len(attempts) + 1}")
    require(attempt <= MAX_ATTEMPTS, f"There is no attempt {MAX_ATTEMPTS + 1}")
    text = approval.get("approval", "")
    missing = [w for w in (WORDS[0].format(attempt=attempt),) + WORDS[1:] if w not in text]
    require(not missing, "Approval wording missing: " + " | ".join(missing))
    require(approval.get("package_manifest_sha256") == sha_file(manifest_path), "Approval names a different package")
    history = []
    for p in runs:
        history.append({"run": p.name, "attempt_counted": p in attempts,
                        "reading": recorded_reading(_read(p / "SCREEN_RESULT.json"), p in attempts)})
    if attempts:
        last = recorded_reading(_read(attempts[-1] / "SCREEN_RESULT.json"), True)
        require(last == "INCOMPLETE", f"No re-attempt after {last}")
    return {"approval": approval, "attempt": attempt, "manifest_sha256": sha_file(manifest_path), "manifest": manifest,
            "history": history, "previous_runs": [str(p) for p in runs], "stage0_failures": len(stage0_failures)}


def inputs():
    record = json.loads((COLLECTION / "collection.json").read_text(encoding="utf-8"))
    for world, spec in record["datasets"].items():
        require((COLLECTION / spec["file"]).is_file(), f"{world} dataset file missing")
    return {"collection_dir": str(COLLECTION), "collection_sha256": sha_file(COLLECTION / "collection.json")}


def integrity_record():
    """The v3 dependency closure plus every file of this package, the r4 modules it imports and the candidate
    folders a job loads by path."""
    files = {str(p): sha_file(p) for p in U.dependency_closure()}
    extra = [p for p in sorted(HERE.iterdir()) if p.is_file() and p.suffix in (".py", ".json", ".md")]
    extra += [p for p in sorted(NATIVE.iterdir()) if p.is_file() and p.suffix in (".py", ".json", ".md")
              and p.name not in ("SCREEN_APPROVAL.json",)]
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
    return module("_native_screen_r5_supervisor", VALIDATION / "matched_policy_pilot_v1/launch.py").supervise


def default_dispatch(job_dir, cap):
    command = [sys.executable, "-B", "-u", str(HERE / "screen_worker_r5.py"), "--job", str(job_dir / "job.json")]
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
                     "g1_seconds": 0.0, "g1_transitions": 0, "i0_seconds": 0.0, "preflight_seconds": 0.0}
        self.dirs, self.reports, self.units = {}, {}, {}
        self.training_started = False
        self.device_kinds = None                             # set from this run's preflight receipt (A2)

    def reusable(self, spec):
        """A completed unit of an earlier run with the same id, kind, seed, arm and world; a GPU unit only if its
        report carries the device kinds of this run (amendment A2: a unit trained on another machine is never
        continued here, and its checkpoints may not even be in this clone)."""
        for prev in reversed(self.plan["previous_runs"]):
            unit_dir = Path(prev) / spec["id"]
            sup, job = _read(unit_dir / "SUPERVISOR_RESULT.json") or {}, _read(unit_dir / "job.json") or {}
            if sup.get("completed") and all(job.get(k) == spec.get(k) for k in REUSE_KEYS):
                report = _read(unit_dir / "report.json")
                if report and report.get("status") == "completed":
                    if spec["kind"] in GPU_KINDS and (self.device_kinds is None or report.get("device_kinds") != self.device_kinds):
                        continue
                    return unit_dir, report
        return None, None

    def run(self, spec):
        kind = spec["kind"]
        # preflight and i0 are environment receipts: always taken afresh; every other completed unit is reusable
        unit_dir, report = (None, None) if kind in ("preflight", "i0") else self.reusable(spec)
        if report is not None:
            self.dirs[spec["id"]], self.reports[spec["id"]] = unit_dir, report
            self.units[spec["id"]] = f"reused from {unit_dir.parent.name}"
            return report
        if kind in TRAINING_KINDS:
            if not self.training_started:                   # r5 D8: the attempt is counted from the first training job
                step03._write_new(self.directory / "TRAINING_STARTED.json",
                                  {"attempt": self.plan["attempt"], "first_job": spec["id"], "device_kinds": self.device_kinds,
                                   "utc": datetime.now(timezone.utc).isoformat()})
                self.training_started = True
            cap, transitions, cap_key = min(JOB_SECONDS, TRAIN_SECONDS - self.caps["training_seconds"]), 0, "training_seconds"
        elif kind == "g1":
            cap, transitions, cap_key = G1_SECONDS - self.caps["g1_seconds"], G1_TRANSITIONS - self.caps["g1_transitions"], "g1_seconds"
        elif kind == "i0":
            cap, transitions, cap_key = I0_SECONDS, I0_TRANSITIONS, "i0_seconds"
        elif kind == "preflight":
            cap, transitions, cap_key = PREFLIGHT_SECONDS, 0, "preflight_seconds"
        else:
            cap, transitions, cap_key = EVAL_SECONDS - self.caps["evaluation_seconds"], EVAL_TRANSITIONS - self.caps["evaluation_transitions"], "evaluation_seconds"
        if cap <= 0 or (kind in ("g1", "i0", "r0", "evaluate", "evaluate_bca") and transitions <= 0):
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


def execute(dispatch=None, *, run_dir=None, approval_path=None, manifest_path=None, root=None, integrity=None, require_dry_run=True):
    plan = check_approval(approval_path, manifest_path, run_dir, root, require_dry_run)
    found = inputs()
    directory = (run_dir or RUN) / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    directory.mkdir(parents=True, exist_ok=False)
    options = {"provenance": "research", "collection_dir": found["collection_dir"], "test_seeds": None,
               "training_seeds": None, "visited_seed": None, "integrity": integrity or integrity_record()}
    step03._write_new(directory / "ATTEMPT.json", sanitize({**{k: v for k, v in plan.items() if k != "manifest"}, **found,
                                                            "started_utc": datetime.now(timezone.utc).isoformat()}))
    screen = Screen(plan, directory, options, dispatch or default_dispatch)
    result = {"screen": "knee_s10_candidate_v2/native_screen_r5", "kappa": KAPPA, "attempt": plan["attempt"],
              "manifest_sha256": plan["manifest_sha256"], "units": screen.units, "caps": screen.caps, "reading": "INCOMPLETE"}
    try:
        pre = screen.run({"id": "preflight", "kind": "preflight"})
        result["preflight"] = pre.get("kind")
        screen.device_kinds = list((pre.get("receipt") or {}).get("device_kinds") or [])
        require(len(screen.device_kinds) == 1 and screen.device_kinds[0] in DK.ALLOWED_DEVICE_KINDS,
                f"Preflight device kinds {screen.device_kinds} are not one allowed kind (A2)")
        result["device_kinds"] = screen.device_kinds
        i0 = screen.run({"id": "i0", "kind": "i0"})
        result["i0"] = {k: i0.get(k) for k in ("I0", "loader", "data_plan", "ok", "finer_max_rel_diff")}
        if not i0.get("ok"):
            raise Stop("INCONCLUSIVE (instrument)", ["I0: " + json.dumps(result["i0"], default=str)])
        g1 = screen.run({"id": "g1", "kind": "g1"})
        result["g1_reported"] = g1["decision"]
        for i in SEEDS:
            for w in WORLDS:
                for a in ARMS:
                    screen.run({"id": f"warmup_{name(i, a, w)}", "kind": "warmup", "seed_index": i, "arm": a, "world": w})
        result["attempt_counted"] = True
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
        reading, pre_reads = SM.pre_bca(native)
        result["pre_bca"] = {"reading": reading, "bed": pre_reads.get("bed"), "reasons": pre_reads.get("reasons")}
        if reading is not None:
            raise Stop(reading, pre_reads["reasons"])
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
        decision = SM.decide(native, bca, KAPPA, g1["decision"], i0)
        step03._write_new(directory / "decision.json", sanitize(decision))
        result.update(reading=decision["reading"], reasons=decision["reasons"], B1=decision.get("B1"), U1=decision.get("U1"),
                      B2={k: decision["B2"][k] for k in ("gain_ok", "beats_shuffled", "beats_constant")} if decision.get("B2") else None,
                      bed=decision.get("bed"))
    except Stop as stop:
        result.update(reading=stop.args[0], reasons=list(stop.args[1]))
    except (U.UnitFailed, step03.CapExhausted) as exc:
        reading = "INCOMPLETE" if screen.training_started else "INCONCLUSIVE (instrument)"
        result.update(reading=reading, reasons=[f"{type(exc).__name__}: {exc}"])
    except BaseException as exc:
        reading = "INCOMPLETE" if screen.training_started else "INCONCLUSIVE (instrument)"
        result.update(reading=reading, reasons=[f"{type(exc).__name__}: {exc}"], traceback=traceback.format_exc())
        _write_result(directory, result, screen)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return result
    _write_result(directory, result, screen)
    return result


def _write_result(directory, result, screen):
    result["attempt_counted"] = bool(screen.training_started)
    try:
        result["physical"] = U.physical(directory, None)
    except BaseException:
        pass
    step03._write_new(directory / "SCREEN_RESULT.json", sanitize({**result, "finished_utc": datetime.now(timezone.utc).isoformat()}))


def decide_only():
    plan = check_approval()
    runs = plan["previous_runs"]
    require(runs, "No run to decide")
    directory = Path(runs[-1])
    native = {i: _read(directory / f"evaluate_s{i}" / "report.json") for i in SEEDS}
    bca = {i: _read(directory / f"evaluate_bca_s{i}" / "report.json") for i in SEEDS}
    g1 = (_read(directory / "g1" / "report.json") or {}).get("decision")
    i0 = _read(directory / "i0" / "report.json")
    decision = SM.decide({i: r for i, r in native.items() if r}, {i: r for i, r in bca.items() if r}, KAPPA, g1, i0)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    step03._write_new(directory / f"decision_recomputed_{stamp}.json", sanitize(decision))
    return decision


def summary(out):
    return {k: out.get(k) for k in ("reading", "reasons", "attempt", "attempt_counted", "B1", "B2", "U1", "caps")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--execute", action="store_true")
    group.add_argument("--decide", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default=None, help="dry run only: output folder outside the repository")
    args = parser.parse_args()
    if args.dry_run:                                        # a fresh interpreter: dry_run.py must import before any Stage 1 module
        import subprocess
        command = [sys.executable, "-B", "-u", str(HERE / "dry_run.py")] + (["--out", args.out] if args.out else [])
        sys.exit(subprocess.call(command))
    print(json.dumps(summary(execute() if args.execute else decide_only()), indent=2, default=str))
