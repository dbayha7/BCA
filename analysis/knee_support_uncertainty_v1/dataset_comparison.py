"""Read-only NumPy reconstruction of the completed knee confirmation's input streams.

No experiment modules are imported: some have environment/model import side effects.
The reproduced selectors follow masks.py and bindings.py and must match all 30
completed training receipts before this analysis can report success. Only a new
analysis output directory is written; no source dataset or experiment file changes.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VALIDATION = REPO / "experiments/thrusterchain_bca/zeus_20261026_validation"
CAND = VALIDATION / "knee_s10_candidate_v2"
CF = CAND / "confirmation_v1"
RUN = CF / "cf_run/attempt_3"
ARCHIVE_DIR = CAND / "formal_g2/formal_g2_run/attempt_1"
FIELDS = ("obs", "action", "reward", "next_obs", "done")
WORLDS = ("stall3", "nostall")
KNEE = float(np.float32(0.8))
EXPECTED_RESULT = "b136b39c670e7224c22fe1b8f38062a22d30f608d22c9b4253c406194b53d197"
EXPECTED_MANIFEST = "6a524bf8e63c5b874cfd1ee644604964509e2879d06c01700f10023402343448"
ARCHIVES = {
    "stall3": ("125131d77f54600497ce4a5d95abff885c1cefbf8ccbafd2d74fd4d52da4bdb2",
               "c2a6d61665a539ad8740d191424f90aeea9973566f3c6325226d928161fee533"),
    "nostall": ("5f4933fc5d687fc0b6f60e85c96809e7f0ace7023b91209ab89a9e373111b71e",
                "e3549869af13b367cd726d008935b3eb466e38c005d1d5b7856faae3397e572a"),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(canonical([a.dtype.str, list(a.shape)]).encode() + a.tobytes()).hexdigest()


def content_digest(arrays, metadata):
    """Exact core_v4/thrusterchain/datasets.py content_digest, without imports."""
    h = hashlib.sha256(canonical(metadata).encode())
    for name in sorted(arrays):
        a = np.asarray(arrays[name])
        require(not a.dtype.hasobject, "Object array forbidden")
        header = canonical([name, a.dtype.str, list(a.shape)]).encode()
        h.update(len(header).to_bytes(8, "big") + header + np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def transition_digest(fields):
    """Exact bindings.py transition_digest (different JSON spacing than content)."""
    h = hashlib.sha256()
    for name in FIELDS:
        a = np.ascontiguousarray(fields[name])
        header = json.dumps([name, a.dtype.str, list(a.shape)]).encode()
        h.update(len(header).to_bytes(8, "big") + header + a.tobytes())
    return h.hexdigest()


def compare_array(left, right):
    a, b = np.asarray(left), np.asarray(right)
    result = {"left_shape": list(a.shape), "right_shape": list(b.shape),
              "left_dtype": a.dtype.str, "right_dtype": b.dtype.str,
              "left_sha256": array_hash(a), "right_sha256": array_hash(b),
              "same_shape": a.shape == b.shape, "same_dtype": a.dtype == b.dtype}
    if a.shape != b.shape:
        return {**result, "equal_values": False, "byte_identical": False}
    equal = a == b
    rows = equal.reshape(len(a), -1).all(axis=1)
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {**result, "equal_values": bool(equal.all()),
            "byte_identical": a.dtype == b.dtype and a.tobytes() == b.tobytes(),
            "rows": len(a), "equal_rows": int(rows.sum()), "different_rows": int((~rows).sum()),
            "different_elements": int((~equal).sum()), "max_abs_difference": float(delta.max(initial=0)),
            "mean_abs_difference": float(delta.mean()) if delta.size else None,
            "different_by_coordinate": (~equal).reshape(len(a), -1).sum(axis=0).tolist(),
            "max_abs_by_coordinate": delta.reshape(len(a), -1).max(axis=0).tolist() if len(a) else [],
            "first_different_row_offsets": np.flatnonzero(~rows)[:5].tolist()}


def row_tokens(fields):
    """Byte-sensitive multiset identities; preserve dtype, shape and multiplicity."""
    names = sorted(fields)
    n = len(fields[names[0]])
    require(all(len(fields[k]) == n for k in names), "Field row counts disagree")
    arrays = [np.ascontiguousarray(fields[k]) for k in names]
    header = canonical([(k, a.dtype.str, list(a.shape[1:])) for k, a in zip(names, arrays)]).encode()
    return Counter(hashlib.sha256(header + b"".join(a[i].tobytes() for a in arrays)).hexdigest()
                   for i in range(n))


def compare_fields(left, right):
    require(set(left) == set(right), "Different field schemas")
    fields = {k: compare_array(left[k], right[k]) for k in left}
    a, b = row_tokens(left), row_tokens(right)
    return {"fields": fields, "rowwise_byte_identical": all(v["byte_identical"] for v in fields.values()),
            "multiset_byte_identical": a == b, "multiset_common_rows": sum((a & b).values()),
            "multiset_left_only_rows": sum((a - b).values()), "multiset_right_only_rows": sum((b - a).values()),
            "multiset_semantics": "Joint row bytes including field names/dtypes/shapes; multiplicity preserved; SHA256 tokens"}


def source_rows(episodes, local):
    local = np.asarray(local, np.int64)
    return np.stack((np.asarray(episodes)[local // 100], local % 100), axis=1)


def fields_from_source(arrays, source, learner=True):
    e, t = source.T
    out = {"obs": arrays["observations"][e, t], "action": arrays["actions"][e, t],
           "reward": arrays["rewards"][e, t], "next_obs": arrays["observations"][e, t + 1],
           "done": arrays["terminated"][e, t]}
    if learner:
        return {k: np.asarray(v, np.float32) for k, v in out.items()}
    return {**out, "state": arrays["states"][e, t], "next_state": arrays["states"][e, t + 1],
            "truncated": arrays["truncated"][e, t], "requested_action": arrays["requested_actions"][e, t]}


def reconstruct(arrays, seeds):
    """Literal selectors from masks.py + bindings.py; verify receipt hashes later."""
    fitting = np.flatnonzero(arrays["split_codes"] == 0)
    n = len(fitting) * 100
    actions = arrays["actions"][fitting].reshape(-1, 4)
    obs = arrays["observations"][fitting, :-1].reshape(-1, 14)
    times = np.tile(np.arange(100), len(fitting))
    group = np.repeat(arrays["group_ids"][fitting], 100)
    load = obs[:, 10].astype(float) + obs[:, 11].astype(float)
    bands = np.where(load < .4, 0, np.where(load < 1., 1, 2))
    cutoff = np.abs(actions[:, 0].astype(float)) <= KNEE
    strata = np.stack((group, times // 10, bands), axis=1)
    order = np.lexsort((np.arange(n), strata[:, 2], strata[:, 1], strata[:, 0]))
    bounds = np.r_[0, np.flatnonzero(np.any(np.diff(strata[order], axis=0), axis=1)) + 1, n]
    random = np.zeros(n, bool)
    rng = np.random.Generator(np.random.PCG64(seeds["random_control"]))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        rows = order[lo:hi]
        random[rng.choice(rows, int(cutoff[rows].sum()), replace=False)] = True
    norm = np.flatnonzero(cutoff & random)
    retained = np.flatnonzero(cutoff)
    probe_rng = np.random.Generator(np.random.PCG64(seeds["gradient_probes"]))
    probe = []
    for e in range(len(fitting)):
        candidates = retained[retained // 100 == e]
        require(len(candidates) > 0, "No retained fitting row in episode")
        probe.append(candidates[probe_rng.integers(len(candidates))])
    scale = np.random.Generator(np.random.PCG64(seeds["scale_bank"])).choice(retained, 256, replace=False)
    tuning = np.flatnonzero(arrays["split_codes"] == 1)
    tuning = tuning[np.argsort(arrays["group_ids"][tuning], kind="stable")][:128]
    require(len(tuning) == 128, "Calibration episode count differs")
    eligible = np.abs(arrays["actions"][tuning, :, 0].astype(float)).reshape(-1) <= KNEE
    originals = np.random.Generator(np.random.PCG64(seeds["calibration_original"])).integers(0, 100, 128)
    rr = np.random.Generator(np.random.PCG64(seeds["calibration_replacement"]))
    cal, replaced = [], []
    for e in range(128):
        original = e * 100 + int(originals[e])
        candidates = np.flatnonzero(eligible[e * 100:(e + 1) * 100]) + e * 100
        require(len(candidates) > 0, "No eligible calibration row")
        if eligible[original]:
            cal.append(original)
        else:
            cal.append(candidates[rr.integers(len(candidates))]); replaced.append(int(tuning[e]))
    sources = {"train": source_rows(fitting, retained), "heldout": source_rows(tuning, cal),
               "probe": source_rows(fitting, probe), "scale_bank": source_rows(fitting, scale),
               "normalization_fixture": source_rows(fitting, norm)}
    streams = {k: fields_from_source(arrays, s) for k, s in sources.items()}
    return {"sources": sources, "streams": streams, "fitting": fitting,
            "masks": {"cutoff": cutoff, "random": random, "normalization_intersection": cutoff & random},
            "calibration_replacements": replaced}


def normalize_obs(obs, normalization):
    """Recorded float32 model-input transform; not a claim of GPU arithmetic replay."""
    mean, std = (np.asarray(normalization[k], np.float32) for k in ("mean", "std"))
    require(mean.shape == std.shape == (obs.shape[-1],), "Normalization shape differs")
    epsilon = np.float32(normalization["epsilon"])
    require(np.isfinite(mean).all() and np.isfinite(std).all() and (std >= 0).all()
            and np.isfinite(epsilon) and epsilon > 0, "Invalid recorded normalization")
    return (np.asarray(obs, np.float32) - mean) / (std + epsilon)


def analyze():
    inputs = {}
    def record(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        require(expected is None or expected == digest, f"Identity differs: {path}")
        inputs[str(path.relative_to(REPO))] = digest
        return path
    result = json.loads(record(RUN / "CF_RESULT.json", EXPECTED_RESULT).read_text())
    manifest = json.loads(record(CF / "CF_MANIFEST.json", EXPECTED_MANIFEST).read_text())
    require(result["manifest_sha256"] == EXPECTED_MANIFEST, "Result manifest differs")
    screen_seeds = json.loads(record(CAND / "native_screen/screen_seeds.json").read_text())
    cf_seeds = json.loads(record(CF / "cf_seeds.json").read_text())
    require(cf_seeds["screen_seeds_sha256"] == sha(CAND / "native_screen/screen_seeds.json"), "Data-plan seed binding differs")
    reports, normalizations, arrays, plans, world_summary = [], {}, {}, {}, {}
    for unit, status in result["units"].items():
        if unit.startswith("eval_"):
            continue
        attempt = "attempt_2" if status == "reused from attempt_2" else "attempt_3"
        require(status in ("completed", "reused from attempt_2"), "Uncompleted training job")
        directory = CF / "cf_run" / attempt / unit
        report = json.loads(record(directory / "report.json").read_text())
        job = json.loads(record(directory / "job.json").read_text())
        exit_receipt = json.loads(record(directory / "SUPERVISOR_RESULT.json").read_text())
        require(report["status"] == "completed" and exit_receipt["exit_code"] == 0, "Failed source job")
        reports.append((unit, report, job))
        world = report["binding"]["world"]
        if world in normalizations:
            require(normalizations[world] == report["normalization"], "Within-world normalization changed")
        normalizations[world] = report["normalization"]
    require(len(reports) == 30, "Expected 30 effective training receipts")
    # Authenticate all code files consulted against an executed job's dependency closure.
    source_names = ["knee_edge_ood_v3/" + n for n in ("masks.py", "bindings.py", "dataset_knee.py", "stage1_worker.py", "knee_region.py", "worlds.py")]
    source_paths = [VALIDATION / n for n in source_names] + [CAND / "native_screen" / n for n in ("screen_data.py", "screen_bind.py", "screen_worker.py")] + [CF / "cf_worker.py"]
    payload = REPO / "experiments/thrusterchain_bca/titan_20261026/payload"
    source_paths += [payload / p for p in ("transfer_bundle/core_v4/thrusterchain/datasets.py", "unifloral_td3bc_validation_v1/native_loader.py", "unifloral_checkpoint_v1/training.py", "unifloral_td3bc_validation_v1/upstream/algorithms/td3_bc.py")]
    source_paths += [VALIDATION / "central_quartile_training_v1_evaluation_rerun1/runtime.py"]
    source_paths += [VALIDATION / "knee_s10_candidate_v1/recorder_wide.py"]
    closure = reports[0][2]["integrity"]["files"]
    source_authentication = {}
    for path in source_paths:
        key = "/mnt/c/Users/David/dev/BCA/" + path.relative_to(REPO).as_posix()
        if key in closure:
            record(path, closure[key])
            source_authentication[str(path.relative_to(REPO))] = "executed job dependency closure"
        else:
            # Upstream source is separately authenticated by the closure-pinned
            # native_loader, not enumerated directly in the job's file map.
            require(path == payload / "unifloral_td3bc_validation_v1/upstream/algorithms/td3_bc.py",
                    f"Consulted source absent from job closure: {key}")
            loader = payload / "unifloral_td3bc_validation_v1/native_loader.py"
            pins = [ast.literal_eval(n.value) for n in ast.parse(loader.read_text()).body
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SOURCE_SHA256" for t in n.targets)]
            require(len(pins) == 1, "Native loader source pin missing")
            record(path, pins[0])
            source_authentication[str(path.relative_to(REPO))] = "SOURCE_SHA256 literal in job-closure-authenticated native_loader.py"
    for world in WORLDS:
        path = record(ARCHIVE_DIR / f"knee_s10_v2_{world}.npz", ARCHIVES[world][0])
        with np.load(path, allow_pickle=False) as archive:
            envelope = json.loads(str(archive["manifest"]))
            arrays[world] = a = {k: archive[k] for k in archive.files if k != "manifest"}
        require(content_digest(a, envelope["metadata"]) == envelope["sha256"] == ARCHIVES[world][1], "Canonical dataset content differs")
        require(envelope["metadata"]["world"] == world, "World metadata differs")
        plans[world] = p = reconstruct(a, screen_seeds["numpy"])
        receipts = {}
        for name, fields in p["streams"].items():
            source = p["sources"][name]
            require(not np.isin(a["split_codes"][source[:, 0]], (2, 3)).any(), "Protected transition selected")
            require(not (np.abs(fields["action"][:, 0].astype(float)) > KNEE).any(), "Withheld action in retained stream")
            receipts[name] = {"rows": len(source), "sha256": transition_digest(fields), "w_rows": 0}
        matching = [r for _, r, _ in reports if r["binding"]["world"] == world]
        require(all(r["binding"]["streams"] == receipts and r["binding"]["dataset_sha256"] == ARCHIVES[world][1] for r in matching), "Reconstructed streams do not match actual learner receipts")
        world_summary[world] = {"archive_sha256": ARCHIVES[world][0], "content_sha256": ARCHIVES[world][1],
            "metadata_physics": envelope["metadata"]["physics"], "dataset_entropy": envelope["metadata"]["seed"],
            "split_episode_counts": {str(k): int((a["split_codes"] == k).sum()) for k in range(4)},
            "actual_training_receipts_matched": len(matching), "streams": receipts,
            "calibration_replacement_episode_ids": p["calibration_replacements"],
            "normalization": normalizations[world],
            "normalization_note": "Receipt source label inherited from older conditional study; actual fixture is within-world CUTOFF intersect RANDOM, not identical across worlds.",
            "decoded_for_content_hash_only_protected_episodes": int(np.isin(a["split_codes"], (2, 3)).sum())}
    left, right = (arrays[w] for w in WORLDS)
    p, q = (plans[w] for w in WORLDS)
    require(np.array_equal(p["fitting"], q["fitting"]), "Different fitting episode selections")
    sources_equal = {k: compare_array(p["sources"][k], q["sources"][k]) for k in p["sources"]}
    all_source = source_rows(p["fitting"], np.arange(len(p["fitting"]) * 100))
    comparisons = {"source_rows": sources_equal,
        "masks": {k: compare_array(p["masks"][k], q["masks"][k]) for k in p["masks"]},
        "raw_fitting": compare_fields(fields_from_source(left, all_source, False), fields_from_source(right, all_source, False)),
        "retained_raw": compare_fields(fields_from_source(left, p["sources"]["train"], False), fields_from_source(right, q["sources"]["train"], False)),
        "learner_streams": {k: compare_fields(p["streams"][k], q["streams"][k]) for k in p["streams"]},
        "episode_and_context_arrays": {k: compare_array(left[k][p["fitting"]], right[k][q["fitting"]]) for k in ("episode_seeds", "trajectory_ids", "group_ids", "split_codes", "reset_component", "load_band", "behavior_component", "behavior_log_density")},
        "fitting_initial_states": compare_array(left["states"][p["fitting"], 0], right["states"][q["fitting"], 0]),
        "normalization": {k: compare_array(np.asarray(normalizations[WORLDS[0]][k], np.float32), np.asarray(normalizations[WORLDS[1]][k], np.float32)) for k in ("mean", "std")}}
    normalized = []
    for world in WORLDS:
        fields = dict(plans[world]["streams"]["train"])
        for name in ("obs", "next_obs"):
            fields[name] = normalize_obs(fields[name], normalizations[world])
        normalized.append(fields)
    comparisons["normalized_train"] = compare_fields(*normalized)
    history = {}
    for stream in ("train", "heldout"):
        e, t = p["sources"][stream].T
        prior = np.cumsum(np.abs(left["actions"][:, :, 0].astype(float)) > KNEE, axis=1)
        previous = np.where(t == 0, 0, prior[e, np.maximum(t - 1, 0)])
        state_diff = (left["states"][e, t] != right["states"][e, t]).any(axis=-1)
        next_diff = (left["states"][e, t + 1] != right["states"][e, t + 1]).any(axis=-1)
        history[stream] = {"rows": len(e), "rows_with_prior_withheld_action": int((previous > 0).sum()),
            "source_state_different": int(state_diff.sum()), "next_state_different": int(next_diff.sum()),
            "source_different_without_prior_withheld": int((state_diff & (previous == 0)).sum()),
            "next_different_without_prior_withheld": int((next_diff & (previous == 0)).sum()),
            "source_rows_different_examples": p["sources"][stream][state_diff][:5].tolist(),
            "interpretation": "Observed prefix association, not a counterfactual dynamics replay. Consequences of earlier removed transitions remain in later retained states; removed transition tuples remain excluded. This is not evaluation-label leakage."}
    summary = {"schema": "knee_support_uncertainty.dataset_comparison.v1", "status": "completed_saved_data_analysis",
        "finished_est": datetime.now(timezone(timedelta(hours=-5))).isoformat(), "numpy_version": np.__version__,
        "provenance": {"input_files_sha256": inputs, "source_authentication": source_authentication, "analysis_source_sha256": sha(__file__),
                       "confirmation_training_seeds": cf_seeds["training"], "data_plan_seeds": screen_seeds["numpy"]},
        "worlds": world_summary, "comparisons": comparisons, "history": history,
        "evidence_scope": {"simulator_calls": 0, "model_calls": 0, "optimizer_updates": 0, "calibration_updates": 0,
            "roles_compared": "code0 fitting; exactly128 selected code1 calibration episodes; fitting-only normalization/probe/scale streams",
            "protected_roles": "All archive members decoded for exact content hashing; code2/code3 transition values excluded from numerical comparisons and no evaluation labels used.",
            "normalization": "Recorded float32 mean/std applied algebraically in NumPy; no claim of GPU bitwise arithmetic reproduction.",
            "law_claim": "Observed sample equality/inequality does not prove equality/inequality of every underlying distribution."}}
    require(all(sha(REPO / path) == digest for path, digest in inputs.items()), "An input changed during analysis")
    return summary, {f"{w}_{k}": src for w in WORLDS for k, src in plans[w]["sources"].items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "output/dataset")
    args = parser.parse_args(argv)
    out = args.output_dir.resolve()
    require(out.is_relative_to(HERE.resolve()), "Analysis output must stay within this analysis directory")
    require(not out.exists(), "Output exists; preserve previous analysis")
    summary, sources = analyze()
    out.mkdir(parents=True, exist_ok=False)
    with (out / "source_rows.npz").open("xb") as handle:
        np.savez_compressed(handle, **sources)
    summary["source_row_artifact"] = {"file": "source_rows.npz", "sha256": sha(out / "source_rows.npz"),
                                      "arrays": {k: array_hash(v) for k, v in sources.items()}}
    with (out / "dataset_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": summary["status"], "summary": str(out / "dataset_summary.json"),
                      "sha256": sha(out / "dataset_summary.json"), "training_receipts_matched": 30}))


if __name__ == "__main__":
    main()
