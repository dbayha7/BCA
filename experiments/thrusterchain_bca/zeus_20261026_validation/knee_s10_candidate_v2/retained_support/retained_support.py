"""RS1: state support of the formal G2 queries against CUTOFF's retained rows (rule: RS1_RULE.md, written first).

Runs once on the formal G2 attempt-1 datasets of candidate knee_s10_v2, by digest. No training, no checkpoint.

  PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/retained_support/retained_support.py --execute

The pure functions (standardize, thresholds, shares, labels) take arrays and dicts only, so the tests run on
synthetic data. The driver (`execute`) imports the sealed v2 formal driver lazily, because that driver replaces the
`recorder` module on import and refuses to load after any other recorder was imported.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CAND = HERE.parent
FORMAL = CAND / "formal_g2"
ATTEMPT = FORMAL / "formal_g2_run" / "attempt_1"
V3 = CAND.parent / "knee_edge_ood_v3"
if str(V3) not in sys.path:
    sys.path.insert(0, str(V3))

import numpy as np                                          # noqa: E402

from core import require                                    # noqa: E402
from step04_math import NEIGHBORS, QUANTILE, knn_distance   # noqa: E402

RULE, RESULT = HERE / "RS1_RULE.md", HERE / "rs1_result.json"
EXPECTED_DIGESTS = {"stall3": "c2a6d61665a539ad8740d191424f90aeea9973566f3c6325226d928161fee533",
                    "nostall": "e3549869af13b367cd726d008935b3eb466e38c005d1d5b7856faae3397e572a"}
WORLDS = ("stall3", "nostall")
TIMES = (0, 25, 50)
FLOOR = 32
SECONDARY_WINDOW = 2
BARS = {"start_share": 0.95, "anchor_share": 0.90}
STEP_CAP, SECONDS_CAP = 20_000, 600
ORDER = ("INCOMPLETE", "INCONCLUSIVE", "RETAINED SUPPORT FAILS", "RETAINED SUPPORT HOLDS")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------- pure functions

def standardize(pool):
    """Mean and SD of the pool's 14 coordinates (SD below 1e-12 becomes 1), float64."""
    pool = np.asarray(pool, np.float64)
    require(pool.ndim == 2 and pool.shape[1] == 14 and len(pool) > 0, "Pool must be (n, 14)")
    mean, std = pool.mean(axis=0), pool.std(axis=0)
    return mean, np.where(std < 1e-12, 1.0, std)


def group_rows(pool_times, pool_deep, t, window=0):
    """Rows of the pool that are deep-load and within `window` steps of time t (window 0: exactly t)."""
    times, deep = np.asarray(pool_times, np.int64), np.asarray(pool_deep, bool)
    require(times.shape == deep.shape, "Times and deep flags disagree")
    return np.flatnonzero(deep & (np.abs(times - int(t)) <= int(window)))


def thresholds(z_pool, pool_episodes, pool_times, pool_deep, *, times=TIMES, window=0, floor=FLOOR, k=NEIGHBORS,
               quantile=QUANTILE):
    """Per time: the 0.99 'higher' quantile of the group's leave-one-episode-out k-NN distances to the pool, with the
    group's rows and distinct episodes and whether it is below the floor. An empty group has no threshold."""
    z_pool, episodes = np.asarray(z_pool, np.float64), np.asarray(pool_episodes, np.int64)
    require(len(z_pool) == len(episodes) == len(pool_times) == len(pool_deep), "Pool arrays disagree")
    out = {}
    for t in times:
        rows = group_rows(pool_times, pool_deep, t, window)
        entry = {"rows": int(len(rows)), "episodes": int(len(np.unique(episodes[rows]))), "floor": int(floor),
                 "below_floor": bool(len(rows) < floor), "window": int(window), "threshold": None}
        if len(rows):
            d = knn_distance(z_pool[rows], z_pool, episodes, exclude_episodes=episodes[rows], k=k)
            require(bool(np.all(np.isfinite(d))), f"Non-finite group distance at t = {t}")
            entry["threshold"] = float(np.quantile(d, quantile, method="higher"))
        out[int(t)] = entry
    return out


def shares(z_pool, pool_episodes, thresholds_by_t, start_z, anchor_z, anchor_t, *, k=NEIGHBORS):
    """Starts against the t = 0 threshold, anchors against their own time's threshold. A missing threshold admits
    nothing. Returns shares, counts, by-time anchor shares and the raw distances."""
    z_pool, episodes = np.asarray(z_pool, np.float64), np.asarray(pool_episodes, np.int64)
    start_z, anchor_z = np.asarray(start_z, np.float64), np.asarray(anchor_z, np.float64)
    anchor_t = np.asarray(anchor_t, np.int64)
    require(anchor_z.shape[0] == anchor_t.shape[0], "Anchor observations and times disagree")
    d_start = knn_distance(start_z, z_pool, episodes, k=k)
    d_anchor = knn_distance(anchor_z, z_pool, episodes, k=k)
    thr = {int(t): (np.nan if v["threshold"] is None else v["threshold"]) for t, v in thresholds_by_t.items()}
    require(0 in thr, "No t = 0 threshold for the starts")
    a_thr = np.asarray([thr.get(int(t), np.nan) for t in anchor_t], np.float64)
    start_in, anchor_in = d_start <= thr[0], d_anchor <= a_thr
    by_t = {}
    for t in sorted(set(int(x) for x in anchor_t)):
        sel = anchor_t == t
        by_t[t] = {"anchors": int(sel.sum()), "inside": int(anchor_in[sel].sum()), "share": float(anchor_in[sel].mean())}
    return {"start_share": float(start_in.mean()), "starts_inside": int(start_in.sum()), "starts": int(len(start_in)),
            "anchor_share": float(anchor_in.mean()), "anchors_inside": int(anchor_in.sum()), "anchors": int(len(anchor_in)),
            "by_t": by_t, "start_distances": d_start, "anchor_distances": d_anchor}


def world_labels(primary_thresholds, primary_shares, bars=BARS):
    """(below_floor, misses) for one world: misses names the shares under their bars."""
    below = [t for t, v in primary_thresholds.items() if v["below_floor"]]
    misses = [name for name in ("start_share", "anchor_share") if not primary_shares[name] >= bars[name]]
    return below, misses


def labels(per_world, *, digests_ok, instrument_ok, error=None):
    """The RS1 labels (pure). per_world: world -> {"below_floor": [t...], "misses": [share names]}.
    Returns (headline, all labels)."""
    out = []
    if error is not None:
        out.append("INCOMPLETE (" + str(error) + ")")
    if not digests_ok:
        out.append("INCOMPLETE (a dataset digest differs from the formal collection)")
    if not instrument_ok:
        out.append("INCOMPLETE (the all-rows D3 recomputation differs from the formal report)")
    inconclusive = [f"{w} t={t}" for w, v in per_world.items() for t in v["below_floor"]]
    if inconclusive:
        out.append("INCONCLUSIVE (primary group below the floor: " + ", ".join(inconclusive) + ")")
    fails = [f"{w} {m}" for w, v in per_world.items() for m in v["misses"]]
    if fails:
        out.append("RETAINED SUPPORT FAILS (" + ", ".join(fails) + ")")
    all_labels = out or ["RETAINED SUPPORT HOLDS"]
    headline = min(all_labels, key=lambda s: ORDER.index(s.split(" (")[0]))
    return headline.split(" (")[0], all_labels


LEADS_TO = {
    "RETAINED SUPPORT HOLDS": "The r4 native package may be built on this candidate.",
    "RETAINED SUPPORT FAILS": "The r4 native package does not run on this candidate; the finding goes to the ledger and "
                              "the definition decides what a retained-support miss means.",
    "INCONCLUSIVE": "The retained population is too thin at some time to decide; reported.",
    "INCOMPLETE": "Fix the instrument and run again; no number from this run is read.",
}


# ------------------------------------------------------------------------------------------------------ the driver

def _finite(value):
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def execute():
    started, t0 = datetime.now(timezone.utc).isoformat(), time.monotonic()
    result = {"schema": "knee_s10_candidate_v2.retained_support.rs1.v1", "candidate": "knee_s10_v2", "rule": RULE.name,
              "rule_sha256": sha(RULE), "code_sha256": sha(__file__), "started_utc": started,
              "headline": "INCOMPLETE", "all_labels": ["INCOMPLETE"], "expected_digests": dict(EXPECTED_DIGESTS)}
    per_world, digests_ok, instrument_ok, error, steps = {}, True, True, None, 0
    try:
        if str(FORMAL) not in sys.path:
            sys.path.insert(0, str(FORMAL))
        import g2_formal_v2 as G2                                   # noqa: E402  replaces `recorder` first
        import dataset_knee as D                                    # noqa: E402
        import g2_seeds_v2 as GS2                                   # noqa: E402
        import masks as M                                           # noqa: E402
        import step04_math as SM4                                   # noqa: E402
        from knee_region import past_knee_a0                        # noqa: E402
        G = G2.G
        seeds = GS2.load()
        entropy = seeds["numpy"]["dataset_entropy"]
        D._sealed_dataset_entropy = lambda: entropy
        collection = json.loads((ATTEMPT / "collection.json").read_text(encoding="utf-8"))
        report = json.loads((ATTEMPT / "report.json").read_text(encoding="utf-8"))
        result.update(seeds_sha256=sha(GS2.SEEDS_PATH), formal_manifest_sha256=report["manifest_sha256"],
                      formal_report_headline=report["headline"])
        digests_ok = all(collection[w]["sha256"] == EXPECTED_DIGESTS[w] for w in WORLDS)
        require(digests_ok, "collection.json digests differ from the sealed expectation")
        if time.monotonic() - t0 > SECONDS_CAP:
            raise RuntimeError("time cap before loading")
        data = {}
        for w in WORLDS:
            ds = D.load(ATTEMPT / collection[w]["path"])
            require(ds.digest == EXPECTED_DIGESTS[w], f"{w} dataset digest differs")
            data[w] = ds
        result["dataset_digests"] = {w: data[w].digest for w in WORLDS}
        result["load_seconds"] = time.monotonic() - t0

        items = GS2.gated_items()
        require(len(items) == 96, f"Expected 96 gated anchors, got {len(items)}")
        steps += 2 * sum(GS2.bank_item("g2_e2_anchors", i)["t"] for i in items)
        require(steps <= STEP_CAP, "step cap")
        anchors = [G2.build_anchor(i) for i in items]
        require(all(a["both_worlds_identical"] for a in anchors), "P1 fails on rebuild")
        a_obs = np.stack([a["obs"] for a in anchors]).astype(np.float64)
        a_t = np.asarray([a["t"] for a in anchors], np.int64)
        starts = G.start_obs(G2.start_rows()).astype(np.float64)
        require(len(starts) == 128, "Expected 128 starts")
        result["queries"] = {"starts": int(len(starts)), "anchors": int(len(anchors)), "anchor_times": sorted(set(int(t) for t in a_t))}

        for w in WORLDS:
            arrays = data[w].arrays
            rows = M.fitting_rows(arrays)
            arm, _ = M.arm_masks(rows["obs"], rows["actions"], rows["times"], rows["episode_ids"], seed=seeds["numpy"]["random_control"])
            cutoff = arm["cutoff"]
            past = past_knee_a0(rows["actions"])[0]
            require(int(np.sum(cutoff & past)) == 0 and np.array_equal(cutoff, ~past), "Retained mask is not W's complement")
            deep = np.repeat(np.asarray(arrays["reset_component"])[rows["episodes"]] == 1, 100)
            obs_all = rows["obs"].astype(np.float64)
            # the retained pool
            r = np.flatnonzero(cutoff)
            obs_r, ep_r, t_r, deep_r = obs_all[r], rows["episode_ids"][r], rows["times"][r], deep[r]
            mean_r, std_r = standardize(obs_r)
            z_r = (obs_r - mean_r) / std_r
            s_z, a_z = (starts - mean_r) / std_r, (a_obs - mean_r) / std_r
            primary = thresholds(z_r, ep_r, t_r, deep_r, window=0)
            secondary = thresholds(z_r, ep_r, t_r, deep_r, window=SECONDARY_WINDOW, floor=FLOOR)
            sh_p = shares(z_r, ep_r, primary, s_z, a_z, a_t)
            sh_s = shares(z_r, ep_r, secondary, s_z, a_z, a_t)
            # the instrument check: the sealed all-rows D3 on the same queries
            l15 = SM4.l15(rows["obs"], rows["episode_ids"], rows["times"], deep, starts.astype(np.float32), a_obs.astype(np.float32), a_t)
            d3 = report["D"][w]["D3"]
            counts_ok = (int(round(l15["start_share"] * 128)) == int(round(d3["start_share"] * 128))
                         and int(round(l15["anchor_share"] * 96)) == int(round(d3["anchor_share"] * 96)))
            instrument_ok = instrument_ok and counts_ok
            # all-rows distances for the ratio (the sealed standardization: all fitting rows)
            mean_a, std_a = standardize(obs_all)
            z_a = (obs_all - mean_a) / std_a
            d_all = knn_distance((a_obs - mean_a) / std_a, z_a, rows["episode_ids"])
            ratio = sh_p["anchor_distances"] / d_all
            thr_ratio = {t: (None if primary[t]["threshold"] is None else primary[t]["threshold"] / l15["thresholds_by_t"][t]["threshold"])
                         for t in TIMES}
            below, misses = world_labels(primary, sh_p)
            per_world[w] = {"below_floor": below, "misses": misses}
            result.setdefault("worlds", {})[w] = _finite({
                "fitting_rows": int(len(obs_all)), "retained_rows": int(len(r)), "retained_share": float(len(r) / len(obs_all)),
                "retained_high_load_share": float(np.mean(cutoff[rows["bands"] >= 1])),
                "retained_low_load_share": float(np.mean(cutoff[rows["bands"] == 0])),
                "deep_rows_all": int(deep.sum()), "deep_rows_retained": int(deep_r.sum()),
                "primary": {"thresholds_by_t": primary, "start_share": sh_p["start_share"], "starts_inside": sh_p["starts_inside"],
                            "anchor_share": sh_p["anchor_share"], "anchors_inside": sh_p["anchors_inside"], "by_t": sh_p["by_t"],
                            "bars": BARS, "start_ok": bool(sh_p["start_share"] >= BARS["start_share"]),
                            "anchor_ok": bool(sh_p["anchor_share"] >= BARS["anchor_share"]), "below_floor": below},
                "secondary_window_2": {"thresholds_by_t": secondary, "start_share": sh_s["start_share"], "starts_inside": sh_s["starts_inside"],
                                       "anchor_share": sh_s["anchor_share"], "anchors_inside": sh_s["anchors_inside"], "by_t": sh_s["by_t"]},
                "instrument_check_all_rows_d3": {"recomputed": {"start_share": l15["start_share"], "anchor_share": l15["anchor_share"],
                                                                "thresholds_by_t": l15["thresholds_by_t"]},
                                                 "formal_report": {"start_share": d3["start_share"], "anchor_share": d3["anchor_share"]},
                                                 "counts_equal": bool(counts_ok)},
                "anchor_distance_ratio_retained_over_all": {"median": float(np.median(ratio)), "min": float(ratio.min()), "max": float(ratio.max())},
                "threshold_ratio_retained_over_all_by_t": thr_ratio,
                "retained_pool_mean": mean_r.tolist(), "retained_pool_sd": std_r.tolist(),
            })
        headline, all_labels = labels(per_world, digests_ok=digests_ok, instrument_ok=instrument_ok)
    except Exception:                                        # noqa: BLE001  always leave a result
        error = traceback.format_exc()
        result["error"] = error
        headline, all_labels = labels(per_world, digests_ok=digests_ok, instrument_ok=instrument_ok,
                                      error=error.strip().splitlines()[-1])
    result.update(headline=headline, all_labels=all_labels, leads_to=LEADS_TO[headline],
                  accounting={"simulator_steps_exact": int(steps), "step_cap": STEP_CAP,
                              "seconds": time.monotonic() - t0, "seconds_cap": SECONDS_CAP},
                  finished_utc=datetime.now(timezone.utc).isoformat())
    RESULT.write_text(json.dumps(_finite(result), indent=2) + "\n", encoding="utf-8")
    return result


def summary(result):
    out = {k: result.get(k) for k in ("headline", "all_labels", "leads_to", "accounting")}
    for w, v in result.get("worlds", {}).items():
        p, s = v["primary"], v["secondary_window_2"]
        out[w] = {"primary": {"starts": f"{p['starts_inside']}/128", "anchors": f"{p['anchors_inside']}/96",
                              "groups": {t: (e["rows"], e["episodes"], e["threshold"]) for t, e in p["thresholds_by_t"].items()}},
                  "secondary": {"starts": f"{s['starts_inside']}/128", "anchors": f"{s['anchors_inside']}/96",
                                "groups": {t: (e["rows"], e["episodes"], e["threshold"]) for t, e in s["thresholds_by_t"].items()}},
                  "instrument_counts_equal": v["instrument_check_all_rows_d3"]["counts_equal"],
                  "distance_ratio": v["anchor_distance_ratio_retained_over_all"]}
    if "error" in result:
        out["error"] = result["error"]
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(summary(execute()), indent=1, default=str))
