"""One fixed native/BCA comparison. Preparation/checks are stdlib-only; no approval is created here."""
import argparse
import ast
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
VALIDATION = HERE.parents[1]
REPO = HERE.parents[4]
PAYLOAD = VALIDATION.parent / "titan_20261026/payload"
V3 = VALIDATION / "knee_edge_ood_v3"
EST = timezone(timedelta(hours=-5), "EST")
APPROVAL_TEXT = "I approve the minimal knee_s10_v2 native versus BCA kappa 3 comparison: three paired seeds, nine training jobs, three-hour total cap, exploratory restricted-data claim; RS1 remains failed."

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def literal(path, name):
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Missing literal: " + name)

def named_seed(name):
    raw = hashlib.sha256(("knee_s10_v2/minimal_comparison/20261101/" + name).encode()).digest()
    return 1 + int.from_bytes(raw[:8], "big") % (2**31 - 2)

def configuration():
    names = ("native_seed", "aux_seed", "branch_native_seed", "branch_aux_seed", "bc_only_seed")
    data_names = ("random_control", "calibration_original", "calibration_replacement", "gradient_probes", "scale_bank")
    return {
        "schema": "minimal_knee_comparison.v1", "world": "stall3", "data_arm": "cutoff", "kappa": 3.0,
        "dataset_relative": "experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/formal_g2/formal_g2_run/attempt_1/knee_s10_v2_stall3.npz",
        "dataset_archive_sha256": "125131d77f54600497ce4a5d95abff885c1cefbf8ccbafd2d74fd4d52da4bdb2",
        "dataset_content_sha256": "c2a6d61665a539ad8740d191424f90aeea9973566f3c6325226d928161fee533",
        "dataset_entropy": 6846509946489100059,
        "training": [{k: named_seed(f"seed{i}/{k}") for k in names} for i in range(3)],
        "data_seeds": {k: named_seed("data/" + k) for k in data_names},
        "evaluation_seeds": [{k: named_seed(f"evaluation/seed{i}/{k}") for k in ("returns", "anchors")} for i in range(3)],
        "warmup_updates": 10000, "branch_updates": 10000, "total_critic_updates": 90000,
        "total_actor_updates": 45000, "total_auxiliary_updates": 90000, "calibrations": 9,
        "normalization": "cutoff_retained_fitting_rows_only", "gamma": 0.98, "horizon": 100,
        "evaluation_current": [0.9, 1.1], "return_episodes_per_policy": 128,
        "anchor_times": [0, 25, 50], "anchors_per_time": 16,
        "common_slots": ["edge_knee", "s10", "edge_700"], "include_actor_action": True,
        "qpi_clip_all": False, "error_tolerance": 0.15, "numerical_tie_tolerance": 1e-10,
        "practical_return_gain": 2.0,
        "total_timeout_seconds": 10800, "training_job_seconds": 1800, "evaluation_job_seconds": 600,
        "preflight_seconds": 120, "training_job_transition_cap": 128000,
        "evaluation_job_transition_cap": 184800, "total_transition_cap": 1706400,
        "evaluation_return_episodes": 768, "evaluation_return_steps": 76800,
        "evaluation_qpi_steps": 86400, "evaluation_prefix_steps": 7200,
        "dataset_validation_steps": 1536000, "full_training_hold": True, "automatic_retry": False,
        "rs1_primary": "failed: 121/128 starts, required 122; unchanged",
        "status": "prepared_for_review_not_authorized"}

def dependencies():
    """Pin the reused closure without importing its simulator/model modules."""
    files = set(p for p in V3.iterdir() if p.is_file() and p.suffix in (".py", ".json", ".md"))
    rr = VALIDATION / "central_quartile_training_v1_evaluation_rerun1"
    pins = literal(V3 / "step03.py", "RERUN1_SOURCES")
    for name, digest in pins.items():
        p = rr / name
        if sha(p) != digest:
            raise ValueError("Reused runtime differs: " + name)
        files.add(p)
    for name in literal(V3 / "units.py", "EXTERNAL_DEPENDENCIES"):
        files.add(VALIDATION / name)
    for name in literal(V3 / "units.py", "PAYLOAD_DEPENDENCIES"):
        files.add(PAYLOAD / name)
    for name in literal(V3 / "units.py", "PAYLOAD_TREES"):
        files.update(p for p in (PAYLOAD / name).rglob("*") if p.is_file() and p.suffix in (".py", ".json"))
    files.update((VALIDATION / "bca_dose_screen_v1" / name) for name in ("screen_worker.py", "DOSE.json"))
    for folder in (VALIDATION / "knee_s10_candidate_v1", HERE.parent, VALIDATION / "bca_dose_screen_v1"):
        files.update(p for p in folder.glob("*.py") if p.is_file())
    files.add(HERE.parent / "prefix_rule.py")
    files.add(VALIDATION / "knee_s10_candidate_v1/recorder_wide.py")
    files.update(p for p in (PAYLOAD / "unifloral_td3bc_validation_v1/upstream").rglob("*.py"))
    return files

def seed_audit(config):
    values = [v for row in config["training"] + config["evaluation_seeds"] for v in row.values()] + list(config["data_seeds"].values())
    if len(values) != len(set(values)):
        raise ValueError("Duplicate new seed")
    # Named seed/entropy files only: no evaluation outcomes or held-out transitions read.
    old = set()
    def visit(x):
        if type(x) is int: old.add(x)
        elif isinstance(x, dict):
            for v in x.values(): visit(v)
        elif isinstance(x, list):
            for v in x: visit(v)
    for path in VALIDATION.rglob("*.json"):
        if HERE in path.parents or not any(s in path.name.lower() for s in ("seed", "entropy")):
            continue
        visit(read(path))
    if old.intersection(values):
        raise ValueError("New seeds collide with recorded seed values; no automatic substitution")
    return {"new_seed_count": len(values), "prior_integer_count": len(old), "collisions": []}

def seal():
    if (HERE / "MANIFEST.json").exists():
        raise ValueError("Already sealed; review amendments explicitly")
    cfg = configuration()
    if sha(REPO / cfg["dataset_relative"]) != cfg["dataset_archive_sha256"]:
        raise ValueError("Dataset archive differs")
    audit = seed_audit(cfg)
    save(HERE / "config.json", cfg)
    files = dependencies() | {p for p in HERE.iterdir() if p.is_file() and p.suffix in (".py", ".json", ".md")}
    save(HERE / "MANIFEST.json", {"schema": "minimal_comparison.seal.v1", "files": {p.relative_to(REPO).as_posix(): sha(p) for p in sorted(files)},
        "dataset_archive_sha256": cfg["dataset_archive_sha256"], "seed_audit": audit,
        "scientific_execution_during_preparation": False, "approval_recorded": False})

def check():
    manifest = read(HERE / "MANIFEST.json")
    changed = [p for p, h in manifest["files"].items() if not (REPO / p).is_file() or sha(REPO / p) != h]
    if changed: raise ValueError("Pinned files changed: " + repr(changed))
    cfg = read(HERE / "config.json")
    if cfg != configuration(): raise ValueError("Fixed configuration differs")
    if sha(REPO / cfg["dataset_relative"]) != cfg["dataset_archive_sha256"]: raise ValueError("Dataset changed")
    return cfg, {"package_dir": str(HERE), "files": {str(REPO / p): h for p, h in manifest["files"].items()}}

def jobs(root):
    for i in range(3):
        warm = root / f"warmup_s{i}"
        yield {"id": warm.name, "kind": "warmup", "seed_index": i}
        native = root / f"native_s{i}"
        yield {"id": native.name, "kind": "branch", "seed_index": i, "warmup_dir": str(warm)}
        yield {"id": f"bca_s{i}", "kind": "bca_branch", "seed_index": i, "warmup_dir": str(warm), "native_branch_dir": str(native)}
    for i in range(3):
        yield {"id": f"evaluate_s{i}", "kind": "evaluate", "seed_index": i, "warmup_dir": str(root / f"warmup_s{i}"),
               "native_branch_dir": str(root / f"native_s{i}"), "bca_branch_dir": str(root / f"bca_s{i}")}

def overlap_check():
    """User-run only: fail closed on another Python worker in this experiment tree."""
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid(): continue
        try:
            if proc.stat().st_uid != os.getuid(): continue
            args = (proc / "cmdline").read_bytes().split(b"\0")
            if not args or b"python" not in args[0]: continue
            for raw in args[1:]:
                arg = Path(os.fsdecode(raw))
                if arg.name not in ("worker.py", "stage1_worker.py", "screen_worker.py", "screen.py", "launch.py", "stage1.py"): continue
                path = arg if arg.is_absolute() else (proc / "cwd").resolve() / arg
                if path.resolve().is_relative_to(VALIDATION): raise RuntimeError("Another experiment process exists: " + proc.name)
        except (FileNotFoundError, ProcessLookupError): pass


def job_accounting(kind, report, transition_cap):
    """Conservative physical work bounds, independent of scientific success."""
    training = kind in ("warmup", "branch", "bca_branch")
    limits = {"simulator_steps": transition_cap, "critic_updates": 10000 if training else 0,
              "actor_updates": 5000 if training else 0, "auxiliary_updates": 10000 if training else 0,
              "calibrations": 1 if training else 0}
    bounds = {key: {"lower": 0, "upper": cap, "known": cap == 0} for key, cap in limits.items()}
    report = report if isinstance(report, dict) else {}
    steps = report.get("simulator_steps_exact")
    if report.get("execution_counts_known") is True and type(steps) is int and 0 <= steps <= transition_cap:
        bounds["simulator_steps"] = {"lower": steps, "upper": steps, "known": True}
    # Reused workers expose terminal cumulative counts. A completed branch has
    # 20k cumulative updates; only its new 10k work is charged here. Failed or
    # killed jobs without terminal counters retain their full possible tail.
    counts = report.get("counts", {})
    scale = 1 if kind == "warmup" else 2
    expected = {"critic": 10000 * scale, "actor": 5000 * scale,
                "auxiliary": 10000 * scale, "refresh_accepted": scale}
    if training and isinstance(counts, dict) and all(type(counts.get(k)) is int and counts[k] == v for k, v in expected.items()):
        for key in ("critic_updates", "actor_updates", "auxiliary_updates", "calibrations"):
            bounds[key] = {"lower": limits[key], "upper": limits[key], "known": True}
    return {"counts": bounds, "execution_counts_known": all(v["known"] for v in bounds.values())}


def aggregate_accounting(jobs_record):
    names = ("simulator_steps", "critic_updates", "actor_updates", "auxiliary_updates", "calibrations")
    totals = {}
    for name in names:
        rows = [record["accounting"]["counts"][name] for record in jobs_record.values()]
        totals[name] = {"lower": sum(v["lower"] for v in rows), "upper": sum(v["upper"] for v in rows),
                        "known": all(v["known"] for v in rows)}
    return {"counts": totals, "execution_counts_known": all(v["known"] for v in totals.values())}


def analyze(root, cfg):
    """Summarize only completed authenticated reports; never trigger evaluation."""
    metrics = load_module("_minimal_saved_metrics", HERE / "metrics.py")
    per_seed, scores, sources = {}, {}, {}
    for i in range(3):
        path = root / f"evaluate_s{i}" / "report.json"
        report = read(path)
        if report.get("status") != "completed" or report.get("kind") != "evaluate" or report.get("seed_index") != i:
            raise ValueError("Incomplete or mismatched evaluation report")
        a, b = (report["runs"][name] for name in ("native", "bca"))
        per_seed[i] = metrics.paired_returns(a["returns"], b["returns"], a["return_ids"], b["return_ids"], expected_count=128)
        scores[i] = {name: metrics.score_summary(row["q1"], row["qpi"], row["width"], kappa=3.0,
                         error_tolerance=0.15, tie_tolerance=1e-10, common_action_count=3)
                     for name, row in (("native", a), ("bca", b))}
        sources[i] = sha(path)
    combined = metrics.across_seeds(per_seed, expected_seeds=[0, 1, 2])
    promising = combined["positive_seeds"] == 3 and combined["mean_seed_difference"] >= cfg["practical_return_gain"]
    result = {"status": "completed_saved_data_analysis", "report_sha256_by_seed": sources,
              "paired_returns": per_seed, "across_seeds": combined, "score_diagnostics": scores,
              "decision": "promising_exploratory_screen" if promising else "no_consistent_practical_benefit_at_fixed_dose_and_budget",
              "independent_confirmation": False, "full_training_hold": True}
    save(root / "ANALYSIS.json", result)
    return result

def execute(approval_path):
    cfg, integrity = check()
    approval = read(approval_path)
    if approval.get("approval") != APPROVAL_TEXT or approval.get("manifest_sha256") != sha(HERE / "MANIFEST.json"):
        raise ValueError("Explicit launch approval naming this exact manifest is missing")
    if os.name != "posix": raise ValueError("Production launch requires the approved Linux runtime")
    overlap_check()
    root = HERE / "run"
    root.mkdir(exist_ok=False)
    start = time.monotonic()
    save(root / "ATTEMPT.json", {"started_est": datetime.now(EST).isoformat(), "pid": os.getpid(), "approval_sha256": sha(approval_path),
        "manifest_sha256": sha(HERE / "MANIFEST.json"), "no_automatic_retry": True})
    result = {"status": "incomplete", "jobs": {}, "execution_counts_known": True, "known_simulator_steps": 0,
              "training_updates_completed": 0, "full_training_hold": True}
    try:
        supervisor = load_module("_minimal_owned_supervisor", VALIDATION / "matched_policy_pilot_v1/launch.py")
        env = supervisor.child_environment()
        specs = [{"id": "preflight", "kind": "preflight"}] + list(jobs(root))
        for spec in specs:
            check()
            remaining = cfg["total_timeout_seconds"] - (time.monotonic() - start) - 10
            if remaining <= 0: raise TimeoutError("Global three-hour budget exhausted")
            kind = spec["kind"]
            cap = min(remaining, cfg["preflight_seconds"] if kind == "preflight" else cfg["evaluation_job_seconds"] if kind == "evaluate" else cfg["training_job_seconds"])
            transitions = 0 if kind == "preflight" else cfg["evaluation_job_transition_cap"] if kind == "evaluate" else cfg["training_job_transition_cap"]
            directory = root / spec["id"]
            directory.mkdir(exist_ok=False)
            job = dict(spec, config=cfg, integrity=integrity, world="stall3", arm="cutoff", provenance="research",
                       timeout_seconds=cap, transitions=transitions)
            save(directory / "job.json", job)
            result["jobs"][spec["id"]] = {"dispatch_status": "dispatched", "accounting": job_accounting(kind, None, transitions)}
            outcome = supervisor.supervise([sys.executable, "-B", "-u", str(HERE / "worker.py"), "--job", str(directory / "job.json")], directory, cap, env=env, job_id=spec["id"])
            save(directory / "SUPERVISOR_RESULT.json", outcome)
            report = read(directory / "report.json") if (directory / "report.json").exists() else None
            if report and (report.get("job") != spec["id"] or report.get("job_sha256") != sha(directory / "job.json")):
                raise ValueError("Worker report is not bound to the dispatched job")
            result["jobs"][spec["id"]] = {"supervisor": outcome, "report_sha256": sha(directory / "report.json") if report else None,
                                           "accounting": job_accounting(kind, report, transitions)}
            if outcome.get("exit_code") != 0 or outcome.get("timed_out") or outcome.get("interrupted") or not report or report.get("status") != "completed":
                raise RuntimeError("Incomplete job: " + spec["id"])
            if not result["jobs"][spec["id"]]["accounting"]["execution_counts_known"]:
                raise ValueError("Completed job lacks complete terminal work counters: " + spec["id"])
            if kind in ("warmup", "branch", "bca_branch"): result["training_updates_completed"] += 10000
        check()
        analysis = analyze(root, cfg)
        check()
        result["analysis_sha256"] = sha(root / "ANALYSIS.json")
        result["decision"] = analysis["decision"]
        result["status"] = "completed_exploratory_comparison"
    except BaseException as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        result["accounting"] = aggregate_accounting(result["jobs"])
        result["execution_counts_known"] = result["accounting"]["execution_counts_known"]
        result["known_simulator_steps"] = result["accounting"]["counts"]["simulator_steps"]["lower"]
        result["elapsed_seconds"] = time.monotonic() - start
        result["finished_est"] = datetime.now(EST).isoformat()
        save(root / "RESULT.json", result)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--check", action="store_true")
    group.add_argument("--execute", action="store_true")
    parser.add_argument("--approval", type=Path)
    args = parser.parse_args()
    if args.seal: seal()
    elif args.check:
        cfg, record = check()
        print(json.dumps({"status": "static_identity_check_passed", "pinned_files": len(record["files"]), "training_started": False}))
    elif args.approval is None: parser.error("--execute requires an explicit approval file; preparation does not create one")
    else:
        result = execute(args.approval)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["status"] == "completed_exploratory_comparison" else 1)
