"""The WSL dry run of the r5 native screen (REVISION_r5.md precondition 3; SCREEN_RULE_r5.md "The dry run"). User-run
in the recorded WSL GPU environment before the seal:

  preflight; i0 on the real anchors (CPU); a smoke collection from a test entropy (both worlds, 1,280 episodes);
  then one job of each kind on seed 0, CUTOFF, stall3, with smoke-only reductions: warmup, r0, branch, bc_only,
  evaluate, bca_branch, evaluate_bca.

Everything after i0 is test-only (smoke provenance, test seeds, test streams, four anchors per set), so its numbers
are a one-seed pilot on replicate data: the receipt keeps structure only (completed, errors, timings, exact steps,
key sets), never outcome values. Output goes outside the repository; the receipt is copied into the package
(`dry_run/DRY_RUN_RECEIPT.json`) with the hashes of the modules it exercised, and `seal_r5.py` refuses without it.
About 45 to 60 minutes, mostly the five training jobs at full length (the learner's counts are checked at 10k and
20k, so smoke jobs cannot be shortened).

  /home/david/bca-operator-audit-20261026/bin/python -B -u screen_r5.py --dry-run [--out DIR]
  (screen_r5 starts this file in a fresh interpreter, so the widened recorder is substituted before any Stage 1
  module is imported; `python -B dry_run.py --check-imports` proves the import order without running anything)
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import math
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CAND = HERE.parent
NATIVE = CAND / "native_screen"
V3 = CAND.parent / "knee_edge_ood_v3"
F2 = CAND / "formal_g2"
for _p in (str(V3), str(F2), str(NATIVE), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if "recorder" in sys.modules:
    raise SystemExit("dry_run must be the first importer in its process: run it as `python -B dry_run.py`")

import screen_bind                                           # noqa: E402  FIRST: substitutes the widened recorder
import screen_r5 as S                                        # noqa: E402  (imports the Stage 1 worker, after the substitution)
import fixtures as F                                         # noqa: E402
import stage1_smoke                                          # noqa: E402
import step03                                                # noqa: E402
import step04                                                # noqa: E402
from core import REPO, require, sha_file                     # noqa: E402

SMOKE_ENTROPY = F.TEST_ENTROPY + 2                            # not the v3 smoke's (+1) and never a sealed value
VISITED_SEED, STREAM_SEED = 98, 777_000_778
RUN = "s0_cutoff_stall3"
ANCHORS = {"e2": [0, 16, 32, 48], "e1": [0, 8, 16, 24]}
SMOKE_EVAL = {"runs": [["cutoff", "stall3"]], "anchors": ANCHORS, "stream_seed": STREAM_SEED}
JOBS = [{"id": f"warmup_{RUN}", "kind": "warmup", "seed_index": 0, "arm": "cutoff", "world": "stall3"},
        {"id": "r0", "kind": "r0", "smoke": {"seeds": [0], "arms": ["cutoff"], "stream_seed": STREAM_SEED}},
        {"id": f"branch_{RUN}", "kind": "branch", "seed_index": 0, "arm": "cutoff", "world": "stall3"},
        {"id": "bc_only_s0", "kind": "bc_only", "seed_index": 0},
        {"id": "evaluate_s0", "kind": "evaluate", "seed_index": 0, "smoke": SMOKE_EVAL},
        {"id": f"bca_branch_{RUN}", "kind": "bca_branch", "seed_index": 0, "arm": "cutoff", "world": "stall3"},
        {"id": "evaluate_bca_s0", "kind": "evaluate_bca", "seed_index": 0, "smoke": SMOKE_EVAL}]


def _finite(value):
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite(v) for v in value)
    return value is None or isinstance(value, (bool, str)) or (isinstance(value, (int, float)) and math.isfinite(value))


def collect(directory):
    budget = step04.StepBudget(step04.STEP_TRANSITIONS, step04.STEP_SECONDS)
    record = step04.run_collection(directory, budget, entropy=SMOKE_ENTROPY, episodes=step04.EPISODES, provenance="smoke",
                                   test_seeds=F.TEST_SEEDS)
    require(record["unit_label"] == "COMPLETE", "Smoke collection plumbing failed")
    return record


def structure(report):
    """Structure only: never an outcome value."""
    if not report:
        return None
    return {"status": report.get("status"), "error": report.get("error"), "elapsed_seconds": report.get("elapsed_seconds"),
            "simulator_steps_exact": report.get("simulator_steps_exact"), "transitions": report.get("transitions"),
            "keys": sorted(k for k in report if k not in ("traceback",)),
            "counts_finite": _finite(report.get("counts")) if "counts" in report else None}


def main(out=None, dispatch=None, collect_fn=collect):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = (Path(out) if out else Path.home() / "knee_s10_dry_run" / stamp).resolve()
    require(REPO.resolve() not in directory.parents and directory != REPO.resolve(), "Write dry-run output outside the repository")
    directory.mkdir(parents=True, exist_ok=False)
    collection_dir = directory / "collection"
    collection_dir.mkdir()
    record = S.integrity_record()
    hashes = S.dry_run_hashes()
    import platform
    result = {"schema": "knee_s10_candidate_v2.native_screen_r5.dry_run.v1", "started_utc": datetime.now(timezone.utc).isoformat(),
              "out": str(directory), "jobs": {}, "hashes": hashes, "platform": platform.system(),
              "real_dispatch": dispatch is None and collect_fn is collect, "device_kinds_ok": None, "device_kind": None,
              "blinded": "Outcome values are withheld; the smoke jobs' reports under `out` hold values and are not read",
              "scope": {"stage0": ["preflight", "i0 (real anchors, real archives, real data plans)"],
                        "smoke": [j["id"] for j in JOBS], "entropy": SMOKE_ENTROPY, "anchors": ANCHORS}}
    research = {"provenance": "research", "collection_dir": str(S.COLLECTION), "test_seeds": None, "training_seeds": None,
                "visited_seed": None, "integrity": record}
    smoke = {"provenance": "smoke", "collection_dir": str(collection_dir), "test_seeds": F.TEST_SEEDS,
             "training_seeds": stage1_smoke.smoke_training_seeds(), "visited_seed": VISITED_SEED, "integrity": record}
    dispatch = dispatch or S.default_dispatch
    status, dirs = "completed", {}

    def unit(spec, options, cap, transitions=0):
        unit_dir = directory / spec["id"]
        unit_dir.mkdir()
        step03._write_new(unit_dir / "job.json", {**spec, "timeout_seconds": float(cap), "transitions": int(transitions), **options})
        outcome = dispatch(unit_dir, float(cap))
        report = S._read(unit_dir / "report.json") or {}
        ok = bool(report.get("status") == "completed" and outcome.get("exit_code") == 0 and not outcome.get("timed_out"))
        result["jobs"][spec["id"]] = {"completed": ok, "timed_out": bool(outcome.get("timed_out")), **(structure(report) or {})}
        if spec["kind"] == "i0":
            result["jobs"][spec["id"]]["instrument_ok"] = bool(report.get("ok"))
            result["i0"] = {k: report.get(k) for k in ("I0", "loader", "data_plan", "ok", "finer_max_rel_diff")}
        require(ok, f"{spec['id']} did not complete: {report.get('error')}")
        dirs[spec["id"]] = unit_dir
        return report

    try:
        unit({"id": "preflight", "kind": "preflight"}, research, S.PREFLIGHT_SECONDS)
        i0 = unit({"id": "i0", "kind": "i0"}, research, S.I0_SECONDS, S.I0_TRANSITIONS)
        require(i0.get("ok"), "The instrument check did not hold in this environment")
        started = time.monotonic()
        import units
        with units.SimulatorCounter() as counter:
            collect_fn(collection_dir)
        result["collection"] = {"seconds": time.monotonic() - started, "simulator_steps_exact": counter.steps}
        for spec in JOBS:
            spec = dict(spec)
            if spec["kind"] in ("branch", "bca_branch"):
                spec["warmup_dir"] = str(dirs[f"warmup_{RUN}"])
            if spec["kind"] == "bca_branch":
                spec["native_branch_dir"] = str(dirs[f"branch_{RUN}"])
            if spec["kind"] == "r0":
                spec["runs"] = {f"warmup_{RUN}": str(dirs[f"warmup_{RUN}"])}
            if spec["kind"] == "evaluate":
                spec["runs"] = {f"warmup_{RUN}": str(dirs[f"warmup_{RUN}"]), f"branch_{RUN}": str(dirs[f"branch_{RUN}"]),
                                "bc_only_s0": str(dirs["bc_only_s0"])}
            if spec["kind"] == "evaluate_bca":
                spec["runs"] = {f"warmup_{RUN}": str(dirs[f"warmup_{RUN}"]), f"branch_{RUN}": str(dirs[f"bca_branch_{RUN}"])}
            cap = S.JOB_SECONDS if spec["kind"] in S.TRAINING_KINDS else S.EVAL_SECONDS
            transitions = 0 if spec["kind"] in S.TRAINING_KINDS else S.EVAL_TRANSITIONS
            unit(spec, smoke, cap, transitions)
        warm = S._read(dirs[f"warmup_{RUN}"] / "report.json") or {}
        pre = S._read(dirs["preflight"] / "report.json") or {}
        kinds = (pre.get("receipt") or {}).get("device_kinds")
        result["device_kind"] = kinds[0] if isinstance(kinds, list) and len(kinds) == 1 else None
        result["device_kinds_ok"] = bool(result["device_kind"] in S.DK.ALLOWED_DEVICE_KINDS and warm.get("device_kinds") == kinds)
        if not result["device_kinds_ok"]:
            status = "completed with a failed check (device kind)"
    except BaseException as exc:
        status = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        t = {k: (result["jobs"].get(k) or {}).get("elapsed_seconds") for k in (f"warmup_{RUN}", f"branch_{RUN}", "bc_only_s0", f"bca_branch_{RUN}")}
        if all(v is not None for v in t.values()):
            total = 12 * t[f"warmup_{RUN}"] + 12 * t[f"branch_{RUN}"] + 3 * t["bc_only_s0"] + 12 * t[f"bca_branch_{RUN}"]
            result["training_projection"] = {"measured_seconds": t, "projected_training_seconds": total,
                                             "training_cap_seconds": S.TRAIN_SECONDS, "within_cap": total <= S.TRAIN_SECONDS}
        result.update(status=status, finished_utc=datetime.now(timezone.utc).isoformat())
        step03._write_new(directory / "DRY_RUN_RESULT.json", json.loads(json.dumps(result, default=str)))
        receipt_dir = HERE / "dry_run"
        receipt_dir.mkdir(exist_ok=True)
        receipt = receipt_dir / "DRY_RUN_RECEIPT.json"
        if receipt.exists():
            receipt.rename(receipt_dir / f"DRY_RUN_RECEIPT_superseded_{stamp}.json")
        receipt.write_text(json.dumps(json.loads(json.dumps(result, default=str)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result["receipt"] = str(receipt)
        result["receipt_sha256"] = sha_file(receipt)
    return result


def check_imports():
    """The import order held (the widened recorder is the `recorder` module) and every job kind is wired."""
    import recorder
    import screen_worker_r5 as SW
    ok = (Path(recorder.__file__).resolve() == (CAND.parent / "knee_s10_candidate_v1" / "recorder_wide.py").resolve()
          and set(SW.KINDS) >= {j["kind"] for j in JOBS} | {"preflight", "i0"})
    return {"recorder": recorder.__file__, "kinds": sorted(SW.KINDS), "ok": bool(ok)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None)
    parser.add_argument("--check-imports", action="store_true")
    args = parser.parse_args()
    if args.check_imports:
        out = check_imports()
        print(json.dumps(out, indent=2))
        sys.exit(0 if out["ok"] else 1)
    result = main(args.out)
    print(json.dumps({"status": result["status"], "out": result["out"], "jobs": {k: v.get("completed") for k, v in result["jobs"].items()},
                      "training_projection": result.get("training_projection"), "receipt": result.get("receipt")}, indent=2, default=str))
    sys.exit(0 if result["status"] == "completed" else 1)
