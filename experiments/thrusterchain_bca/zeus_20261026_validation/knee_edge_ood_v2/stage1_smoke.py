"""Optional GPU smoke test of the Stage 1 worker (not a protocol step; user-run in the recorded WSL GPU environment).

Purpose: run the Stage 1 worker's job kinds once on the real Unifloral learner and GPU before the single A4 attempt, so
that a wrong attribute, shape or file name shows up here instead of as a counted INCOMPLETE.

Isolation and blinding. Everything is test-only: a fresh smoke collection from a test entropy, test data seeds, test
training seeds and test-only evaluation streams (stage1_eval.stream_item with a smoke stream seed, never a sealed
bank, start set or anchor set). Output goes only under --out (default ~/knee_edge_smoke/<UTC time>), never inside
the repository. It is a full-length 10k + 10k native CUTOFF run on replicate data, so its numbers would be a one-seed
pilot: SMOKE_RESULT.json therefore keeps only structure (completed jobs, errors, timings, key sets, finiteness and
the numerical V1 check), never outcome values. The worker reports under --out do hold values; do not read them.

What it runs: preflight; one 10k warmup (seed 0, CUTOFF, stall3); R0 on that run; its 10k branch; the seed-0 BC-only
actor; the seed-0 evaluation of that run with 4 E2 and 4 E1 anchors, one per anchor time; then stage1.py's own
decision readers on the output, and a projection of A4's training time from the measured job times.
About 20-25 minutes, mostly the two training jobs.

  /home/david/bca-operator-audit-20261026/bin/python -B -u <this file> [--out DIR]
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import math
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fixtures as F
import seeds as S
import stage1
import step03
import step04
from core import REPO, require

SMOKE_ENTROPY = F.TEST_ENTROPY + 1
VISITED_SEED = 99
STREAM_SEED = 777_000_777
RUN = "s0_cutoff_stall3"
ANCHORS = {"e2": [0, 16, 32, 48], "e1": [0, 8, 16, 24]}          # one anchor per time t in {0, 25, 50, 75}
JOBS = [{"id": f"warmup_{RUN}", "kind": "warmup", "seed_index": 0, "arm": "cutoff", "world": "stall3"},
        {"id": "r0", "kind": "r0", "smoke": {"seeds": [0], "arms": ["cutoff"], "stream_seed": STREAM_SEED}},
        {"id": f"branch_{RUN}", "kind": "branch", "seed_index": 0, "arm": "cutoff", "world": "stall3"},
        {"id": "bc_only_s0", "kind": "bc_only", "seed_index": 0},
        {"id": "evaluate_s0", "kind": "evaluate", "seed_index": 0,
         "smoke": {"runs": [["cutoff", "stall3"]], "anchors": ANCHORS, "stream_seed": STREAM_SEED}}]


def sealed_values():
    sealed = S.load()
    used = set(sealed["numpy"].values()) | {v for r in sealed["training"] for k, v in r.items() if k != "index"}
    return used | {r[k] for r in sealed["stage2_forks"] for k in ("branch_native_seed", "branch_aux_seed")}


def smoke_training_seeds():
    """Test-only training seeds, derived like the sealed ones from different names, checked distinct from every
    sealed value (data, bank, training and Stage 2 seeds)."""
    rows = [{"index": i, **{f: S.training_seed(f"smoke_stage1_seed{i}/{f}") for f in S.STAGE1_FIELDS}} for i in range(3)]
    used = sealed_values()
    require(not used & {v for r in rows for k, v in r.items() if k != "index"}, "Smoke seed equals a sealed seed")
    require(not used & (set(F.TEST_SEEDS.values()) | {SMOKE_ENTROPY, VISITED_SEED, STREAM_SEED}), "Test seed equals a sealed seed")
    return rows


def collect(directory):
    budget = step04.StepBudget(step04.STEP_TRANSITIONS, step04.STEP_SECONDS)
    record = step04.run_collection(directory, budget, entropy=SMOKE_ENTROPY, episodes=step04.EPISODES, provenance="smoke",
                                   test_seeds=F.TEST_SEEDS)
    require(record["unit_label"] == "COMPLETE", "Smoke collection plumbing failed")
    return record


def _finite(value):
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite(v) for v in value)
    return value is None or isinstance(value, (bool, str)) or (isinstance(value, (int, float)) and math.isfinite(value))


def parse_checks(evaluation):
    """stage1.py's own decision readers on the real evaluation output; structure only, no outcome values."""
    entry = evaluation["runs"][RUN]["020000"]
    inputs = stage1.anchor_inputs(dict(entry, anchors_e2=_padded(entry["anchors_e2"])), True)
    ok, worst = stage1.equality_check(entry)
    visited = stage1.visited_kc(entry["visited"])
    return {"anchor_inputs_keys": sorted(k for k in inputs if k != "res"), "v1_stall_equals_nostall": ok,
            "v1_largest_scaled_deviation": worst, "visited_kc_finite": _finite(visited),
            "own_world_keys": sorted(entry["own_world"]), "nostall_world_keys": sorted(entry["nostall_world"]),
            "anchor_times": sorted(set(entry["anchors_e2"]["times"]) | set(entry["anchors_e1"]["times"])),
            "bc_only_present": evaluation.get("bc_only") is not None}


def _padded(anchors):
    """anchor_inputs reads the 48 gated rows; the smoke evaluation has 4, so tile them to 64 rows for parsing only."""
    out = dict(anchors)
    for key in ("qpi_clip", "qpi_unclipped", "q1", "q2", "qmin", "qpi_nostall", "valid"):
        if key in anchors:
            rows = anchors[key]
            out[key] = (rows * (64 // len(rows) + 1))[:64]
    return out


def projection(result):
    """A4's training time projected from the measured jobs: 12 warmups + 12 branches + 3 BC-only against 18,000 s,
    and each job against 1,800 s. Evaluation timing is not representative (1 run and 8 anchors here)."""
    t = {k: (result["jobs"].get(k) or {}).get("elapsed_seconds") for k in (f"warmup_{RUN}", f"branch_{RUN}", "bc_only_s0")}
    if any(v is None for v in t.values()):
        return None
    total = 12 * t[f"warmup_{RUN}"] + 12 * t[f"branch_{RUN}"] + 3 * t["bc_only_s0"]
    return {"measured_seconds": t, "projected_training_seconds": total, "training_cap_seconds": stage1.TRAIN_SECONDS,
            "within_training_cap": total <= stage1.TRAIN_SECONDS,
            "each_job_within_1800_s": all(v <= stage1.JOB_SECONDS for v in t.values()),
            "margin_note": "A4 also reloads the sealed collection from /mnt/c in every job; keep a margin"}


def main(out=None, dispatch=None, collect_fn=collect):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = (Path(out) if out else Path.home() / "knee_edge_smoke" / stamp).resolve()
    require(REPO.resolve() not in directory.parents and directory != REPO.resolve(), "Write smoke output outside the repository")
    directory.mkdir(parents=True, exist_ok=False)
    collection_dir = directory / "collection"
    collection_dir.mkdir()
    result = {"started_utc": datetime.now(timezone.utc).isoformat(), "out": str(directory), "jobs": {},
              "blinded": "Outcome values are withheld from this summary; do not read the worker reports' values"}
    options = {"provenance": "smoke", "collection_dir": str(collection_dir), "test_seeds": F.TEST_SEEDS,
               "training_seeds": smoke_training_seeds(), "visited_seed": VISITED_SEED,
               "integrity": stage1.U.integrity_record()}
    dispatch = dispatch or stage1.default_dispatch
    caps = {"training_seconds": 0.0, "evaluation_seconds": 0.0, "evaluation_transitions": 0}
    dirs, status = {}, "completed"
    try:
        collect_fn(collection_dir)
        pre = directory / "preflight"
        pre.mkdir()
        step03._write_new(pre / "job.json", {"id": "preflight", "kind": "preflight", "timeout_seconds": stage1.PREFLIGHT_SECONDS,
                                             **options})
        outcome = dispatch(pre, stage1.PREFLIGHT_SECONDS)
        pre_report = stage1._read(pre / "report.json") or {}
        result["jobs"]["preflight"] = {"completed": outcome.get("exit_code") == 0, "error": pre_report.get("error")}
        require(outcome.get("exit_code") == 0, "Preflight failed")
        for spec in JOBS:
            report, outcome = stage1.run_unit(spec, directory, dirs, caps, options, dispatch)
            failed = stage1._read(directory / spec["id"] / "report.json") or {}
            result["jobs"][spec["id"]] = {"completed": report is not None, "error": None if report else failed.get("error"),
                                          "timed_out": bool(outcome.get("timed_out")),
                                          "elapsed_seconds": (report or failed).get("elapsed_seconds")}
            if report is None:
                status = f"failed at {spec['id']}"
                break
            dirs[spec["id"]] = directory / spec["id"]
        if status == "completed":
            warm = stage1._read(directory / f"warmup_{RUN}" / "report.json")
            result["device_kinds_ok"] = warm.get("device_kinds") == [step03.K.RECORDED_DEVICE_KIND]
            result["parse_checks"] = parse_checks(stage1._read(directory / "evaluate_s0" / "report.json"))
            r0 = stage1._read(directory / "r0" / "report.json")
            result["r0_structure"] = {"keys": sorted(r0), "finite": _finite(r0.get("seeds"))}
            if not (result["device_kinds_ok"] and result["parse_checks"]["v1_stall_equals_nostall"]):
                status = "completed with a failed check (device kind or V1)"
    except BaseException as exc:
        status = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        result.update(status=status, caps=caps, a4_training_projection=projection(result),
                      finished_utc=datetime.now(timezone.utc).isoformat())
        step03._write_new(directory / "SMOKE_RESULT.json", stage1.sanitize(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None)
    result = main(parser.parse_args().out)
    print(json.dumps({"status": result["status"], "out": result["out"],
                      "jobs": {k: v.get("completed") for k, v in result["jobs"].items()},
                      "a4_training_projection": result.get("a4_training_projection")}, indent=2, default=str))
    sys.exit(0 if result["status"] == "completed" else 1)
