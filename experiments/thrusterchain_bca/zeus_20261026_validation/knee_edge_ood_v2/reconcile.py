"""Reconciliation per world (protocol 4.7).

(i) Plumbing checks: a failure is INCOMPLETE (the code or data path is wrong).
(ii) Expected-value checks against the 6 SD bounds sealed from screens/random_control_mc_output.txt: a failure is
INCONCLUSIVE (the realized data are not the designed condition; collection is seeded, so a rerun would repeat it).
"""
import json

import numpy as np

import bindings as B
import masks as M
from core import HERE, require, sha_file
from knee_region import past_knee_a0

EXPECTED = HERE / "expected_values.json"


def sealed_bounds():
    record = json.loads(EXPECTED.read_text(encoding="utf-8"))
    require(record["source_sha256"] == sha_file(HERE.parent / "knee_edge_ood_design_v1/screens/random_control_mc_output.txt"),
            "Sealed expected values no longer match their Monte Carlo source")
    return record


def twin_check(states_stall, states_nostall, actions):
    """States identical up to and including each episode's first W row, different exactly at its successor."""
    w = past_knee_a0(np.asarray(actions))[0]
    report, ok = [], True
    for e in range(len(actions)):
        hits = np.flatnonzero(w[e])
        first = int(hits[0]) if len(hits) else None
        last_equal = 100 if first is None else first
        same = np.array_equal(states_stall[e, :last_equal + 1], states_nostall[e, :last_equal + 1])
        differs = True if first is None else not np.array_equal(states_stall[e, first + 1], states_nostall[e, first + 1])
        ok &= bool(same and differs)
        report.append({"episode": e, "first_w_step": first, "identical_through_first_w": bool(same),
                       "differs_at_successor": bool(differs)})
    return ok, report


def world_summary(arrays, seeds, extra_streams=None):
    """Per-world checks on a dataset's arrays with the seeds resolved by `reconcile` (sealed for research data)."""
    rows = M.fitting_rows(arrays)
    masks, receipt = M.arm_masks(rows["obs"], rows["actions"], rows["times"], rows["episode_ids"],
                                 seed=seeds["random_control"])
    calibration = M.calibration_rows(arrays, seed=seeds["calibration_original"],
                                     replacement_seed=seeds["calibration_replacement"])
    normalization = M.normalization_rows(masks)
    a0 = rows["actions"][:, 0].astype(np.float64)
    high, w, bands = rows["bands"] >= 1, rows["past_knee"], rows["bands"]
    cutoff, random = masks["cutoff"], masks["random"]
    per_episode_deep = (bands.reshape(-1, 100) == 2).any(axis=1)
    codes = np.asarray(arrays["split_codes"])
    counts = [int(np.sum(codes == k)) for k in range(4)]
    streams = {"fit_replay": rows["actions"][cutoff], "normalization": rows["actions"][normalization],
               "calibration": np.asarray(arrays["actions"])[calibration["episodes"]].reshape(-1, 4)[calibration["local_rows"]]}
    streams.update(extra_streams or {})
    plumbing = {
        "split_counts_768_256_128_128": counts == [768, 256, 128, 128],
        "cutoff_zero_w_every_stream": all(not np.any(past_knee_a0(v)[0]) for v in streams.values()),
        "random_counts_equal_cutoff_per_three_band_stratum": M.three_band_counts_equal(
            masks, rows["obs"], rows["actions"], rows["times"], rows["episode_ids"]),
        "retained_removed_disjoint_and_complete": bool(np.all(cutoff ^ w)),
        "done_only_at_t99": bool(np.array_equal(np.asarray(arrays["terminated"]),
                                                np.broadcast_to(np.arange(100) == 99, np.asarray(arrays["terminated"]).shape))),
    }
    terminal = rows["times"] == 99
    plumbing_report = {"terminal_rows": {arm: int(np.sum(terminal & masks[arm])) for arm in ("cutoff", "random")}}
    expected = {
        "share_high": float(high.mean()), "share_deep": float((bands == 2).mean()),
        "episodes_with_deep": float(per_episode_deep.mean()),
        "cutoff_removed": int(np.sum(~cutoff)), "cutoff_removed_fraction_high": float(np.sum(~cutoff & high) / np.sum(high)),
        "cutoff_removed_fraction_low": float(np.sum(~cutoff & ~high) / np.sum(~high)),
        "cutoff_high_load_anchor": float(a0[cutoff & high].mean()),
        "random_w_retention": float(np.sum(random & w) / np.sum(w)),
        "random_high_load_anchor": float(a0[random & high].mean()),
        "l14": M.l14_density(masks, rows["actions"], bands), "calibration_replacements": calibration["replacements"],
    }
    return {"masks": masks, "mask_receipt": receipt, "calibration": calibration, "normalization_rows": normalization,
            "plumbing": plumbing, "plumbing_report": plumbing_report, "expected": expected,
            "streams_checked": sorted(streams)}


def _inside(value, bounds):
    return bounds[0] <= value <= bounds[1]


def expected_value_checks(expected, bounds=None):
    b = (bounds or sealed_bounds())["bounds"]
    l14 = expected["l14"]
    return {
        "share_high": _inside(expected["share_high"], b["share_high"]),
        "share_deep": _inside(expected["share_deep"], b["share_deep"]),
        "episodes_with_deep": _inside(expected["episodes_with_deep"], b["episodes_with_deep"]),
        "cutoff_removed": _inside(expected["cutoff_removed"], b["cutoff_removed"]),
        "cutoff_removed_fraction_high": _inside(expected["cutoff_removed_fraction_high"], b["cutoff_removed_fraction_high"]),
        "cutoff_removed_fraction_low": _inside(expected["cutoff_removed_fraction_low"], b["cutoff_removed_fraction_low"]),
        "cutoff_high_load_anchor": _inside(expected["cutoff_high_load_anchor"], b["cutoff_high_load_anchor"]),
        "random_w_retention": _inside(expected["random_w_retention"], b["random_w_retention"]),
        "random_high_load_anchor": _inside(expected["random_high_load_anchor"], b["random_high_load_anchor"]),
        "l14_random_pooled_high": _inside(l14["random"]["pooled_high"], b["l14_pooled_high"]),
        "l14_random_mid": _inside(l14["random"]["mid"], b["l14_mid"]),
        "l14_random_deep": _inside(l14["random"]["deep"], b["l14_deep"]),
        "l14_cutoff_zero": all(v == 0.0 for v in l14["cutoff"].values()),
        "calibration_replacements": _inside(expected["calibration_replacements"], b["calibration_replacements"]),
    }


def reconcile(datasets, *, test_seeds=None, bounds=None):
    """Both worlds, from validated datasets: per-world plumbing and expected-value checks, cross-world mask identity
    and the twin check. Seeds are the sealed ones for research data (test seeds only for smoke data)."""
    require(set(datasets) == {"stall3", "nostall"}, "Reconciliation needs both worlds")
    import dataset_knee
    for world, ds in datasets.items():
        require(ds.metadata["world"] == world, f"Dataset of world {ds.metadata['world']!r} given as {world!r}")
        if ds.metadata["provenance"] == "research":
            require(type(ds) is dataset_knee.KneeDataset, "Research data must be a validated KneeDataset")
    require(datasets["stall3"].metadata["seed"] == datasets["nostall"].metadata["seed"], "Twins must share the entropy")
    seeds = B._seeds_for(datasets["stall3"], test_seeds)[0]
    world_arrays = {w: ds.arrays for w, ds in datasets.items()}
    summaries = {w: world_summary(a, seeds) for w, a in world_arrays.items()}
    cross = {
        "masks_byte_identical_across_worlds": all(np.array_equal(summaries["stall3"]["masks"][k], summaries["nostall"]["masks"][k])
                                                  for k in ("cutoff", "random")),
        "requested_actions_byte_identical": np.array_equal(world_arrays["stall3"]["requested_actions"],
                                                           world_arrays["nostall"]["requested_actions"]),
    }
    twin_ok, twin = twin_check(np.asarray(world_arrays["stall3"]["states"]), np.asarray(world_arrays["nostall"]["states"]),
                               np.asarray(world_arrays["stall3"]["actions"]))
    cross["twin_states_diverge_exactly_at_first_w_successor"] = twin_ok
    plumbing_ok = all(cross.values()) and all(all(s["plumbing"].values()) for s in summaries.values())
    checks = {w: expected_value_checks(s["expected"], bounds) for w, s in summaries.items()}
    expected_ok = all(all(c.values()) for c in checks.values())
    label = "INCOMPLETE" if not plumbing_ok else ("INCONCLUSIVE" if not expected_ok else "PASS")
    return {"label": label, "cross_world": cross, "twin": twin,
            "worlds": {w: {"plumbing": s["plumbing"], "plumbing_report": s["plumbing_report"],
                           "expected": s["expected"], "expected_checks": checks[w],
                           "calibration_replacements": s["calibration"]["replacements"],
                           "normalization_rows": len(s["normalization_rows"]),
                           "retained_row_digests": s["mask_receipt"]["retained_row_digests"]}
                       for w, s in summaries.items()}}
