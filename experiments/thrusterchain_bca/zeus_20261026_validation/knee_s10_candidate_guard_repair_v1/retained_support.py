"""Pure retained-population D3 repair; no collection, simulator, model or file access.

Keep the existing 64-row NN, 0.99 threshold quantile and 95%/90% bars. This
checks state typicality, not equality of distributions or joint action support.
"""
import numpy as np

import step04_math as SM4


def retained_l15(rows, cutoff, deep_episode, starts, anchor_obs, anchor_times):
    obs = np.asarray(rows["obs"], np.float64)
    episodes = np.asarray(rows["episode_ids"])
    times = np.asarray(rows["times"])
    mask, deep = np.asarray(cutoff), np.asarray(deep_episode)
    out = {
        "population": "CUTOFF-retained fitting rows only",
        "normalization": "mean/std of CUTOFF-retained fitting observations; original L1.5 floor",
        "metric": "64th-nearest-row standardized Euclidean distance; own episode excluded for reference queries",
        "neighbors": SM4.NEIGHBORS, "reference_quantile": SM4.QUANTILE,
        "start_bar": 0.95, "anchor_bar": 0.90,
        "start_share": None, "anchor_share": None, "ok": False,
        "status": "INCONCLUSIVE", "fitting_rows": len(obs),
    }
    if (obs.ndim != 2 or obs.shape[1] != 14 or mask.dtype != np.bool_
            or deep.dtype != np.bool_ or any(x.shape != (len(obs),) for x in (episodes, times, mask, deep))):
        return {**out, "reason": "Malformed observations, row metadata or boolean retention mask"}
    if any(x.dtype.kind not in "fiu" or not np.isfinite(x).all()
           or not np.equal(x, np.floor(x)).all() or np.any(x < 0) for x in (episodes, times)):
        return {**out, "reason": "Episode and time identifiers must be finite nonnegative integers"}
    if np.any(times >= 100):
        return {**out, "reason": "Training time outside the frozen horizon"}
    keep = np.flatnonzero(mask)
    tr, ep, tm, dp = obs[keep], episodes[keep], times[keep], deep[keep]
    out.update(retained_rows=len(keep), retained_distinct_episodes=int(len(np.unique(ep))),
               removed_rows=int((~mask).sum()), starts=int(len(starts)), anchors=int(len(anchor_obs)))
    if not len(keep) or not np.isfinite(tr).all():
        return {**out, "reason": "Empty or nonfinite retained observations"}
    if len(set(zip(ep.tolist(), tm.tolist()))) != len(keep):
        return {**out, "reason": "Duplicate episode/time row identifiers"}
    query_times = np.asarray(anchor_times)
    starts, anchor_obs = np.asarray(starts), np.asarray(anchor_obs)
    if (starts.ndim != 2 or anchor_obs.ndim != 2 or starts.shape[1] != 14 or anchor_obs.shape[1] != 14
            or query_times.dtype.kind not in "fiu" or query_times.shape != (len(anchor_obs),) or not np.isfinite(query_times).all()
            or not np.equal(query_times, np.floor(query_times)).all()
            or np.any((query_times < 0) | (query_times >= 100))
            or not len(starts) or not len(anchor_obs)
            or not np.isfinite(starts).all() or not np.isfinite(anchor_obs).all()):
        return {**out, "reason": "Empty, nonfinite or malformed query arrays"}
    if (not np.array_equal(tr[:, 13].astype(np.float32), ((100-tm)/100).astype(np.float32))
            or not np.array_equal(anchor_obs[:, 13].astype(np.float32), ((100-query_times)/100).astype(np.float32))
            or not np.all(starts[:, 13].astype(np.float32) == np.float32(1))):
        return {**out, "reason": "Observation remaining-time coordinate disagrees with row, anchor or start time"}
    groups = {}
    for t in sorted({0} | set(int(t) for t in query_times)):
        group = dp & (tm == t)
        groups[str(t)] = {"rows": int(group.sum()), "distinct_episodes": int(len(np.unique(ep[group])))}
    out["retained_deep_reference_groups"] = groups
    insufficient = [t for t, g in groups.items() if g["rows"] < SM4.NEIGHBORS]
    if insufficient:
        return {**out, "reason": "Fewer than 64 retained deep-episode reference rows at t=" + ",".join(insufficient)}
    # Prevent infinite leave-one-episode-out thresholds from becoming automatic passes.
    if any(int(np.sum(ep != e)) < SM4.NEIGHBORS for e in np.unique(ep[dp & np.isin(tm, [int(t) for t in groups])])):
        return {**out, "reason": "Fewer than 64 eligible neighbor rows after excluding a reference episode"}
    result = SM4.l15(tr, ep, tm, dp, starts, anchor_obs, query_times)
    if not all(np.isfinite(v["threshold"]) for v in result["thresholds_by_t"].values()):
        return {**out, "reason": "Nonfinite retained-only support threshold"}
    return {**out, "status": "PASS" if result["pass"] else "FAIL",
            "start_share": result["start_share"], "anchor_share": result["anchor_share"],
            "ok": bool(result["pass"]), "thresholds_by_t": result["thresholds_by_t"]}
