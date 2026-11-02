"""Pin the six saved run arrays of the two completed conditional-hole gates.

For each gate this records every array's size, SHA-256 and git blob. It then ties the
arrays to the gate's run record and checks them in three ways:
  - against one another and against invariants of the code that wrote them;
  - against the identities the report recorded;
  - by rerunning the gate's own frozen value analysis on them and comparing the output with
    report.json's "values" section.
Two controls show that the comparison can fail.

Read-only. It reads the six arrays, each gate's report.json, RESULT.json, PREPARED.json
and contract.json, and hashes every file in both gate folders. It executes each gate's own
geometry.py, reference.py and analysis.py from the exact bytes it hashed; those modules
import numpy only and do no work on import. It imports no simulator, reads no dataset rows,
and writes only RECEIPT.json beside itself.
"""
import sys

sys.dont_write_bytecode = True  # importing a gate's modules must never leave __pycache__ in its folder

import ast
import hashlib
import json
import math
import os
import platform
import subprocess
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ZEUS = HERE.parent
REPO = HERE.parents[3]
GATES = ["conditional_ood_gate_v1_retry_20261028", "conditional_ood_gate_central_quartile_v1"]
ARRAYS = ["prelabel_probes.npz", "candidate_values.npz", "free_policy.npz"]
# The pins. A rerun verifies these instead of only recomputing hashes.
PINNED_SHA256 = {
    "conditional_ood_gate_v1_retry_20261028": {
        "prelabel_probes.npz": "193ae58f492cb40a401a28aadc89e306c26c772ef3f97ea7a178a4fc64745406",
        "candidate_values.npz": "fe5602dfafe44b43618f2ceaabf1cdab178f615804a33fa4e86ffe87b8c2f146",
        "free_policy.npz": "d1bb867cc63c4117602f0e1662a817468ccece198c6a2697e8c2e2a5e1f76107"},
    "conditional_ood_gate_central_quartile_v1": {
        "prelabel_probes.npz": "8d8dcccbacad4ec5b892065653a296eaf56160a9f191ad5b1138177d55933ba3",
        "candidate_values.npz": "1b80f960dd66332c040b83f3c7871dec76822cffaa290acee9fffd6db9869933",
        "free_policy.npz": "f7e3b7df58c3b45b29440bcce2f5fdb7c63302cc73be80abab50b27a66762c1f"},
}
# The commits that first recorded each gate's arrays in git.
RECORDING_COMMIT = {"conditional_ood_gate_v1_retry_20261028": "e0a0777c2f613720a0ee31f6255459b8a9bd0bcd",
                    "conditional_ood_gate_central_quartile_v1": "fe751257f22715b506b4740324c9cd7fc80629c9"}
RANDOM_DRAWS = 10
# Member name: (shape, dtype), exactly as worker.py writes them.
EXPECTED = {
    "prelabel_probes.npz": {"states": ((64, 13), "float64"), "times": ((64,), "int64"),
                            "observations": ((64, 14), "float32"), "actions": ((64, 6, 4), "float32"),
                            "q": ((64, 6, 2), "float32"), "episode_ids": ((64,), "int64"),
                            "episode_seeds": ((64,), "int64")},
    "candidate_values.npz": {"values": ((64, 6), "float64"), "q1": ((64, 6), "float32"),
                             "actions": ((64, 6, 4), "float32"), "states": ((64, 13), "float64"),
                             "times": ((64,), "int64"), "episode_ids": ((64,), "int64")},
    "free_policy.npz": {"observations": ((16, 100, 14), "float32"), "actions": ((16, 100, 4), "float32"),
                        "returns": ((16,), "float64"), "episode_seeds": ((16,), "int64")},
}
# Pass rule. The float tolerance was fixed before the first run:
# |reported - recomputed| <= 1e-12 * max(1, |reported|), because the gates ran numpy 2.5.4 on
# Linux and this check may run another numpy build. After review the rest was made stricter:
# integers, booleans, strings, None, dict keys and list lengths must match exactly, an integer
# paired with a float is a mismatch, NaN never matches, and an infinity matches only itself.
FLOAT_TOL = 1e-12
SIMULATOR_MODULE_HINTS = ("thrusterchain", "core_v4", "gymnasium", "jax", "flax", "optax")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(REPO).as_posix()


def git(*args):
    # No optional locks: git status must not refresh and rewrite the index.
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True, env=env).stdout.strip()


def reset_scope_literal(reference_path):
    """The 'reset_scope' string in the dict that reference.free_policy_rollouts returns, read with ast."""
    tree = ast.parse(Path(reference_path).read_text(encoding="utf-8"))
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "free_policy_rollouts")
    found = [value.value for node in ast.walk(function) if isinstance(node, ast.Dict)
             for key, value in zip(node.keys, node.values)
             if isinstance(key, ast.Constant) and key.value == "reset_scope" and isinstance(value, ast.Constant)]
    if len(found) != 1 or not isinstance(found[0], str):
        raise ValueError("reference.free_policy_rollouts must return exactly one literal reset_scope string")
    return found[0]


def times_of(path):
    stat = Path(path).stat()
    utc = datetime.fromtimestamp(stat.st_mtime_ns / 1e9, tz=timezone.utc)
    return {"modified_utc": utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{stat.st_mtime_ns % 1_000_000_000:09d}Z",
            "modified_local_with_offset": utc.astimezone().isoformat(timespec="microseconds")}


def folder_state(folder):
    """Hash of every file, plus whether the files on disk are exactly the files git tracks."""
    files = {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob("*")) if p.is_file()}
    tracked = {Path(t).relative_to(rel(folder)).as_posix() for t in git("ls-files", "--", rel(folder)).splitlines()}
    tracked_dirs = {parent.as_posix() for t in tracked for parent in Path(t).parents}
    dirs = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_dir()}
    return {"files": files, "only_tracked_files": set(files) == tracked,
            "no_untracked_directories": dirs <= tracked_dirs,
            "pycache_present": any(p.name == "__pycache__" for p in folder.rglob("*")),
            "git_status": git("status", "--porcelain", "--untracked-files=all", "--ignored", "--", rel(folder))}


def load_gate_modules(gate_dir, prepared):
    """Execute the gate's geometry.py, reference.py and analysis.py from the bytes hashed here.

    Running the hashed source directly, rather than through the import system, means a stale
    compiled file in a __pycache__ folder can never stand in for the frozen code.
    """
    modules, digests = {}, {}
    for name in ("geometry", "reference", "analysis"):
        path = gate_dir / f"{name}.py"
        source = path.read_bytes()
        digests[name] = {"sha256": hashlib.sha256(source).hexdigest(), "prepared_sha256": prepared.get(rel(path))}
        module = types.ModuleType(name)
        module.__file__ = str(path)
        sys.modules[name] = module  # analysis and reference import geometry by name
        exec(compile(source, str(path), "exec"), module.__dict__)
        modules[name] = module
    return modules, digests


def unload_gate_modules():
    for name in ("geometry", "reference", "analysis"):
        sys.modules.pop(name, None)


def compare(reported, recomputed, path, out):
    """Recursive comparison under the fixed pass rule; collects every mismatch."""
    def bad(**detail):
        out["mismatches"].append({"path": path, **detail})

    if isinstance(reported, dict) and isinstance(recomputed, dict):
        if set(reported) != set(recomputed):
            bad(reported_keys=sorted(reported), recomputed_keys=sorted(recomputed))
        for key in sorted(set(reported) & set(recomputed)):
            compare(reported[key], recomputed[key], f"{path}.{key}", out)
    elif isinstance(reported, list) and isinstance(recomputed, list):
        if len(reported) != len(recomputed):
            bad(reported_length=len(reported), recomputed_length=len(recomputed))
        for i, (a, b) in enumerate(zip(reported, recomputed)):
            compare(a, b, f"{path}[{i}]", out)
    elif isinstance(reported, float) and isinstance(recomputed, float):
        out["float_values"] += 1
        if math.isnan(reported) or math.isnan(recomputed):
            bad(reported=repr(reported), recomputed=repr(recomputed), reason="NaN never matches")
            return
        if math.isinf(reported) or math.isinf(recomputed):
            if reported != recomputed:
                bad(reported=repr(reported), recomputed=repr(recomputed), reason="an infinity matches only itself")
            return
        diff = abs(reported - recomputed)
        if diff != 0.0:
            out["floats_not_bitwise_equal"] += 1
        out["max_abs_float_difference"] = max(out["max_abs_float_difference"], diff)
        if diff > FLOAT_TOL * max(1.0, abs(reported)):
            bad(reported=reported, recomputed=recomputed, difference=diff)
    else:
        out["exact_values"] += 1
        if type(reported) is not type(recomputed) or reported != recomputed:
            bad(reported=reported, recomputed=recomputed,
                reported_type=type(reported).__name__, recomputed_type=type(recomputed).__name__)


def new_comparison():
    return {"exact_values": 0, "float_values": 0, "floats_not_bitwise_equal": 0,
            "max_abs_float_difference": 0.0, "mismatches": []}


def free_summary(geometry, observations, actions, returns, seeds, reset_scope):
    """The free-play summary that reference.free_policy_rollouts returns, rebuilt from the arrays.

    The central-quartile reference.py derives these counts with geometry.region_masks, as here.
    The retry's reference.py writes the same mask formula inline; this is an equivalent
    re-implementation for that gate, confirmed by the comparison with report.json.
    """
    n = len(seeds)
    region = geometry.region_masks(observations.reshape(-1, 14), actions.reshape(-1, 4))
    high, hole = region["high"].reshape(n, 100), region["hole"].reshape(n, 100)
    return {"observations": observations, "actions": actions, "returns": returns, "episode_seeds": seeds,
            "hole_entries": int(hole.sum()), "high_load_action_rows": int(high.sum()), "action_rows": int(n * 100),
            "episodes_with_hole_entry": int(np.any(hole, axis=1).sum()),
            "episodes_with_high_load": int(np.any(high, axis=1).sum()),
            "hole_entries_per_episode": hole.sum(axis=1).tolist(), "reset_scope": reset_scope}


def fixed_candidate_a0(gate, geometry):
    """Slots 1 to 5 of the first command, built as each gate's reference.candidate_banks builds them."""
    if gate == "conditional_ood_gate_v1_retry_20261028":
        center = np.float32(-geometry.KNEE)
        lower = np.nextafter(np.float32(-geometry.KNEE - .05), center)
        upper = np.nextafter(np.float32(-geometry.KNEE + .05), center)
        return np.asarray([-.9, lower, center, upper, -.7], np.float32)
    center = np.float32(0.)
    lower = np.nextafter(np.float32(-geometry.HALF_WIDTH), center)
    upper = np.nextafter(np.float32(geometry.HALF_WIDTH), center)
    return np.asarray([-.5, lower, center, upper, .5], np.float32)


def check_gate(gate):
    gate_dir = ZEUS / gate
    run = gate_dir / "run"
    checks = []

    def check(group, name, passed, detail=None):
        checks.append({"group": group, "check": name, "passed": bool(passed),
                       **({"detail": detail} if detail is not None else {})})

    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    result = json.loads((run / "RESULT.json").read_text(encoding="utf-8"))
    contract = json.loads((gate_dir / "contract.json").read_text(encoding="utf-8"))
    prepared_doc = json.loads((gate_dir / "PREPARED.json").read_text(encoding="utf-8"))
    prepared = prepared_doc["files"]

    # 1. Provenance: the run record, the prepared manifest and the frozen code form one chain.
    records = {name: {"path": rel(p), "bytes": p.stat().st_size, "sha256": sha(p),
                      "git_blob_at_head": git("rev-parse", f"HEAD:{rel(p)}")}
               for name, p in [("report.json", run / "report.json"), ("RESULT.json", run / "RESULT.json"),
                               ("PREPARED.json", gate_dir / "PREPARED.json"), ("contract.json", gate_dir / "contract.json")]}
    check("provenance", "report.json is the report whose hash the supervisor recorded in RESULT.json",
          records["report.json"]["sha256"] == result["report_sha256"])
    check("provenance", "PREPARED.json is the manifest the worker verified before and after the run",
          records["PREPARED.json"]["sha256"] == report["prepared_before"]["prepared_sha256"]
          == report["prepared_after"]["prepared_sha256"])
    check("provenance", "contract.json is the contract the run recorded",
          records["contract.json"]["sha256"] == report["contract_sha256"] == report["prepared_before"]["contract_sha256"])
    in_folder = {k: v for k, v in prepared.items() if k.startswith(rel(gate_dir) + "/")}
    differing = sorted(k for k, v in in_folder.items() if sha(REPO / k) != v)
    check("provenance", "every file of this gate folder pinned in PREPARED.json still has its pinned hash",
          not differing and len(in_folder) > 0, {"pinned_in_folder": len(in_folder), "differing": differing})

    # 2. The arrays themselves.
    arrays, loaded = {}, {}
    for name in ARRAYS:
        path = run / name
        head_blob = git("rev-parse", f"HEAD:{rel(path)}")
        work_blob = git("hash-object", "--no-filters", "--", rel(path))
        with np.load(path, allow_pickle=False) as data:
            members = {key: np.array(data[key]) for key in data.files}
        loaded[name] = members
        arrays[name] = {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha(path),
                        "git_blob_at_head": head_blob, **times_of(path),
                        "members": {k: {"dtype": str(v.dtype), "shape": list(v.shape)} for k, v in members.items()}}
        check("arrays", f"{name}: the SHA-256 equals the hash pinned in this script and the README",
              arrays[name]["sha256"] == PINNED_SHA256[gate][name])
        check("arrays", f"{name}: the committed blob equals the file on disk", head_blob == work_blob)
        first_blob = git("rev-parse", f"{RECORDING_COMMIT[gate]}:{rel(path)}")
        check("arrays", f"{name}: unchanged since commit {RECORDING_COMMIT[gate][:8]} first recorded it",
              first_blob == head_blob)
        found = {k: (tuple(v.shape), str(v.dtype)) for k, v in members.items()}
        check("arrays", f"{name}: member names, shapes and data types are as worker.py writes them",
              found == EXPECTED[name])
        check("arrays", f"{name}: every floating-point member is finite",
              all(np.all(np.isfinite(v)) for v in members.values() if v.dtype.kind == "f"))
    probes, cands, free = (loaded[n] for n in ARRAYS)

    # 3. The arrays against one another and against the identities the report recorded.
    check("identity", "candidate_values q1 equals the first critic head in prelabel_probes, bit for bit",
          np.array_equal(cands["q1"], probes["q"][:, :, 0]))
    for key in ("actions", "states", "times", "episode_ids"):
        check("identity", f"candidate_values {key} equal prelabel_probes {key}, bit for bit",
              np.array_equal(cands[key], probes[key]))
    identity = report["data_identity"]
    check("identity", "probe times equal report data_identity probe_times",
          np.array_equal(probes["times"], np.asarray(identity["probe_times"])))
    check("identity", "probe times follow the contract's cycle 0, 25, 50, 75 over 64 anchors",
          np.array_equal(probes["times"], np.tile(contract["data"]["probe_times_cycle"], 16)))
    check("identity", "probe episode IDs equal report data_identity probe_group_ids",
          np.array_equal(probes["episode_ids"], np.asarray(identity["probe_group_ids"])))
    check("identity", "probe episode seeds equal report data_identity probe_seeds",
          np.array_equal(probes["episode_seeds"], np.asarray(identity["probe_seeds"])))
    check("identity", "free-play start seeds are the first 16 anchor episode seeds, as the worker draws them",
          np.array_equal(free["episode_seeds"], probes["episode_seeds"][:16]))
    check("identity", "RESULT.json's free-play count agrees with the fixed 16 x 100 array shape",
          result["counts"]["free_policy_steps_completed"] == free["observations"].shape[0] * free["observations"].shape[1],
          result["counts"]["free_policy_steps_completed"])

    modules, code = load_gate_modules(gate_dir, prepared)
    try:
        geometry, reference, analysis = modules["geometry"], modules["reference"], modules["analysis"]
        for name, digest in code.items():
            check("provenance", f"{name}.py executed here is the frozen copy pinned in PREPARED.json",
                  digest["sha256"] == digest["prepared_sha256"])

        # 4. Invariants of the code that wrote the arrays.
        check("invariants", "anchor observations equal reference.observation(states, times), bit for bit",
              np.array_equal(probes["observations"], reference.observation(probes["states"], probes["times"])))
        time_column = ((reference.HORIZON - np.arange(100)) / reference.HORIZON).astype(np.float32)
        check("invariants", "every free-play episode's time column runs 1.00, 0.99, ... 0.01 in order",
              np.array_equal(free["observations"][:, :, 13], np.broadcast_to(time_column, (16, 100))))
        starts = [i for i in range(16) if probes["times"][i] == 0]
        check("invariants", "free-play first observations equal the anchor observations recorded at time 0 "
                            "for the same episode",
              len(starts) > 0 and np.array_equal(free["observations"][starts, 0], probes["observations"][starts]),
              {"anchors_compared": len(starts)})
        banks = probes["actions"]
        check("invariants", "candidate slots 1 to 5 use the gate's fixed first commands",
              np.array_equal(banks[:, 1:, 0], np.broadcast_to(fixed_candidate_a0(gate, geometry), (64, 5))))
        check("invariants", "every candidate slot keeps the actor's other three commands",
              np.array_equal(banks[:, :, 1:], np.repeat(banks[:, :1, 1:], 6, axis=1)))
        # The code's own sentence goes into the reanalysis, so the comparison checks the report's copy too.
        reset_scope = reset_scope_literal(gate_dir / "reference.py")
        check("invariants", "the report's reset_scope is the exact string reference.free_policy_rollouts returns",
              report["values"]["free_actor"]["reset_scope"] == reset_scope)

        # 5. Rerun the gate's frozen value analysis exactly as worker.py calls it.
        def reanalyze(values, q1, bank, anchor_obs, free_obs, free_act, free_ret, seeds):
            summary = free_summary(geometry, free_obs, free_act, free_ret, seeds, reset_scope)
            return json.loads(json.dumps(analysis.analyze_values(values, q1, bank, anchor_obs, summary, contract)))

        recomputed = reanalyze(cands["values"], cands["q1"], banks, probes["observations"],
                               free["observations"], free["actions"], free["returns"], free["episode_seeds"])
        comparison = new_comparison()
        compare(report["values"], recomputed, "values", comparison)
        check("reanalysis", "rerunning the frozen value analysis on the arrays reproduces report.json 'values'",
              not comparison["mismatches"])

        # Controls: the same comparison must fail when the inputs are wrong.
        nudged = cands["values"].copy()
        nudged[0, 0] += 1e-9
        nudge = new_comparison()
        compare(report["values"], reanalyze(nudged, cands["q1"], banks, probes["observations"], free["observations"],
                                            free["actions"], free["returns"], free["episode_seeds"]), "values", nudge)
        check("controls", "a change of 1e-9 to one true return is caught", len(nudge["mismatches"]) > 0,
              {"mismatched_values": len(nudge["mismatches"])})
        # Keep the real arrays but replace every input the analysis reads with random values: the two
        # load terms, the actor's first command, the true returns, Q1, and free play's commands and returns.
        per_draw, changed = [], set()
        for draw in range(RANDOM_DRAWS):
            rng = np.random.default_rng([20261028, draw])
            rand_obs = probes["observations"].copy()
            rand_obs[:, 10:12] = rng.uniform(0, .4, (64, 2)).astype(np.float32)
            rand_free = free["observations"].copy()
            rand_free[:, :, 10:12] = rng.uniform(0, .4, (16, 100, 2)).astype(np.float32)
            rand_bank = banks.copy()
            rand_bank[:, 0, 0] = rng.uniform(-1, 1, 64).astype(np.float32)
            rand_act = rng.uniform(-1, 1, free["actions"].shape).astype(np.float32)
            noise = new_comparison()
            compare(report["values"],
                    reanalyze(rng.normal(-50, 20, (64, 6)), rng.normal(-30, 20, (64, 6)).astype(np.float32),
                              rand_bank, rand_obs, rand_free, rand_act, rng.normal(-50, 20, 16), free["episode_seeds"]),
                    "values", noise)
            per_draw.append(len(noise["mismatches"]))
            changed |= {m["path"] for m in noise["mismatches"]}
        total = comparison["exact_values"] + comparison["float_values"]
        check("controls", f"each of {RANDOM_DRAWS} draws of random analysis inputs fails to reproduce the report",
              min(per_draw) > 0, {"values_compared": total, "mismatches_per_draw": per_draw,
                                  "changed_in_at_least_one_draw": len(changed)})
        gap_passing = int(np.sum(geometry.region_masks(probes["observations"], banks[:, 0])["high"]
                                 & (np.asarray(recomputed["center_flank_gap_per_anchor"])
                                    >= contract["decision"]["practical_center_flank_gap"])))
    finally:
        unload_gate_modules()

    v = recomputed
    headline = {
        "value_decision_passed": v["passed"],
        "high_load_anchors": v["high_load_anchors"],
        "high_load_anchors_with_practical_gap": gap_passing,
        "free_play_high_load_steps": v["free_actor"]["high_load_action_rows"],
        "free_play_hole_entries": v["free_actor"]["hole_entries"],
        "free_play_episodes_with_entry": v["free_actor"]["episodes_with_hole_entry"],
        "free_play_episodes": int(len(free["episode_seeds"])),
        "actor_command_in_hole_at_anchors": v["actor_at_logged_anchor_hole_entries"],
        "error_cells": {k: {"rows": c["rows"], "mae": c["mae"], "mean_signed_error": c["mean_signed_error"]}
                        for k, c in v["error_cells"].items()},
        "raw_q1_choice": {k: v["raw_q_decisions"][k] for k in ("improved", "worsened", "tied", "mean_regret")},
    }
    return {
        "gate": gate, "decision_recorded": report["decision"], "records": records, "frozen_code": code,
        "arrays": arrays, "checks": checks, "reanalysis": comparison,
        "controls": {"one_return_nudged_by_1e-9": {"mismatches": len(nudge["mismatches"])},
                     "random_analysis_inputs": {
                         "draws": RANDOM_DRAWS, "mismatches_per_draw": per_draw,
                         "values_changed_in_at_least_one_draw": len(changed),
                         "meaning": "A lower bound on the values that depend on the data. The rest are constants "
                                    "or agreed by chance in every draw."}},
        "headline_reproduced": headline,
        "pinned_by_hash_only": [
            "The second critic head, Q2.",
            "The float64 states beyond their float32 rounding, which the observation rule ties.",
            "The 11 state and observation coordinates other than the two load terms, beyond the observation and "
            "start-state rules.",
            "The two load terms, at the anchors and in free play, beyond which side of the 0.4 cutoff each falls on.",
            "Every first command (the actor's at the anchors, and every free-play step), beyond which side of the "
            "band edges it falls on.",
            "The actor's other three commands: at the anchors, beyond their being the same in every slot; in free "
            "play, not checked at all.",
            "Which free-play trajectory belongs to which seed and return, beyond the four episodes tied to an "
            "anchor at time 0 (0, 4, 8 and 12) and the per-episode hole counts.",
            "The order of steps within free play, beyond the time column.",
        ],
        "not_checked_here": "The support test needs training rows (its record is geometry.json), so the overall "
                            "gate decision is not recomputed.",
    }


def main():
    folders = {gate: ZEUS / gate for gate in GATES}
    before = {gate: folder_state(folder) for gate, folder in folders.items()}
    gates = [check_gate(gate) for gate in GATES]
    after = {gate: folder_state(folder) for gate, folder in folders.items()}
    untouched = {}
    for gate in GATES:
        b, a = before[gate], after[gate]
        untouched[gate] = {
            "same_files_and_hashes_before_and_after": b["files"] == a["files"],
            "files_on_disk_are_exactly_the_tracked_files": b["only_tracked_files"] and a["only_tracked_files"],
            "no_untracked_directories": b["no_untracked_directories"] and a["no_untracked_directories"],
            "no_pycache": not b["pycache_present"] and not a["pycache_present"],
            "git_status_clean_including_ignored": b["git_status"] == "" and a["git_status"] == "",
            "files": len(a["files"]),
        }
    untouched_ok = all(all(v for k, v in u.items() if k != "files") for u in untouched.values())
    loaded_simulator = sorted(m for m in sys.modules if any(h in m.lower() for h in SIMULATOR_MODULE_HINTS))
    receipt = {
        "schema": "bca.gate_array_receipt.v3",
        "purpose": "Pin the six saved run arrays of the two completed conditional-hole gates and check them "
                   "against each gate's run record, code invariants and frozen value analysis.",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "head_commit": git("rev-parse", "HEAD"),
        "script": {"path": rel(Path(__file__)), "sha256": sha(Path(__file__))},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
                    "isolated": bool(sys.flags.isolated), "dont_write_bytecode": sys.dont_write_bytecode},
        "pass_rule": ("Floats: |reported - recomputed| <= 1e-12 * max(1, |reported|), fixed before the first run. "
                      "Made stricter after review: integers, booleans, strings, None, dict keys and list lengths "
                      "exact; an integer paired with a float is a mismatch; NaN never matches; an infinity "
                      "matches only itself."),
        "read_only": {
            "measured": {"simulator_or_learner_modules_loaded": loaded_simulator,
                         "gate_folders_untouched": untouched_ok},
            "by_construction": "The script opens no dataset file and no protected split, trains or fits nothing, "
                               "and writes only " + rel(HERE / "RECEIPT.json") + ".",
        },
        "gate_folders": untouched,
        "gates": gates,
        "limits": "This fixes the arrays' bytes as they are now. It cannot prove after the fact that "
                  "prelabel_probes.npz was written before the reference values, or that nothing changed in the "
                  "hours between the run and the commits that first recorded the arrays. The modified times "
                  "agree with the write order, but they are descriptive only and can be changed.",
    }
    receipt["passed"] = (untouched_ok and not loaded_simulator
                         and all(c["passed"] for g in gates for c in g["checks"]))
    (HERE / "RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    for g in gates:
        failed = [c["check"] for c in g["checks"] if not c["passed"]]
        r = g["reanalysis"]
        rand = g["controls"]["random_analysis_inputs"]
        print(f"{g['gate']}: {len(g['checks']) - len(failed)}/{len(g['checks'])} checks passed; "
              f"{r['exact_values']} exact and {r['float_values']} float values compared, "
              f"{r['floats_not_bitwise_equal']} floats not bitwise equal, {len(r['mismatches'])} mismatches; "
              f"nudge caught {g['controls']['one_return_nudged_by_1e-9']['mismatches']}; random draws caught "
              f"{min(rand['mismatches_per_draw'])} to {max(rand['mismatches_per_draw'])}, "
              f"{rand['values_changed_in_at_least_one_draw']} changed in at least one")
        for name in failed:
            print("  FAILED:", name)
    print("gate folders:", json.dumps(untouched))
    print("simulator modules loaded:", loaded_simulator, "| overall passed:", receipt["passed"])
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
