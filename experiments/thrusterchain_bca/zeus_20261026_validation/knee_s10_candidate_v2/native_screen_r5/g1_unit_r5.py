"""The g1 unit under r5 (REVISION_r5.md section 5): G1_RULE.md's fit (the host's critic class, exact Q^{pi_ref} labels,
Adam 3e-4, batch 256, three seeds, both worlds), run to a loss plateau and read relative to its own in-support
error. A reported calibration unit: no label here stops the screen. Imported by screen_worker_r5 with
JAX_PLATFORMS=cpu already set.
"""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
G1DIR = HERE.parent / "g1"
if str(G1DIR) not in sys.path:
    sys.path.insert(0, str(G1DIR))

import screen_bind as SB                                    # noqa: E402
import screen_seeds as SS                                   # noqa: E402
import masks as M                                           # noqa: E402
import stage1_math as SM                                    # noqa: E402
from core import require, sha_bytes                         # noqa: E402
import g1_math as GM                                        # noqa: E402
import g1_labels as GL                                      # noqa: E402
import g1_fit as GF                                         # noqa: E402

STEP, LAST = 20_000, 200_000
CHECKPOINTS = tuple(range(STEP, LAST + 1, STEP))
PLATEAU = 0.05                                               # converged: the chunk loss fell by < 5% over the last 20k
STEP_CAP = 9_000_000
IN_SUPPORT_SLOTS = ("edge_775", "edge_750", "edge_725", "edge_700")
TOL = GM.TOL


def converged_checkpoint(curve):
    """The first checkpoint whose mean chunk loss fell by less than PLATEAU over the preceding STEP updates; LAST,
    flagged not converged, when none did. curve: {updates: mean loss of the chunk ending there}."""
    for c in CHECKPOINTS[1:]:
        before, now = float(curve[c - STEP]), float(curve[c])
        if before > 0 and (before - now) / before < PLATEAU:
            return c, True
    return LAST, False


def read(kc_by_slot, rho):
    """The r5 reading of one fit at one checkpoint: the in-support band, the departure capture, the relative label."""
    valid = np.isfinite(rho) & (rho > 0)
    in_support = np.concatenate([np.abs(kc_by_slot[s][valid]) for s in IN_SUPPORT_SLOTS])
    band = float(np.quantile(in_support, 0.90)) if in_support.size else float("nan")
    med_kc = float(np.median(kc_by_slot["s10"][valid])) if valid.any() else float("nan")
    med_rho = float(np.median(rho[valid])) if valid.any() else float("nan")
    return {"n_valid": int(valid.sum()), "in_support_band": band, "median_kc_s10": med_kc, "median_rho": med_rho,
            "departure_capture": float(1.0 - med_kc / med_rho) if med_rho else float("nan"),
            "representable": bool(np.isfinite(med_kc) and np.isfinite(band) and med_kc <= band + TOL),
            "l1_absolute_reported": {k: v for k, v in GM.l1(kc_by_slot["s10"], rho).items() if k not in ("valid", "flags")}}


def label(*, i0_ok, f0_ok, representable_by_seed):
    if not i0_ok:
        return "INCONCLUSIVE (I0)"
    if not f0_ok:
        return "INCONCLUSIVE (F0)"
    return "REPRESENTABLE" if all(representable_by_seed.values()) else "NOT REPRESENTABLE"


def run(job, out, deadline):
    require(GF.upstream_match(), "SoftQNetwork source differs from the pinned upstream td3_bc.py")
    budget = SB.G2.G.Budget(STEP_CAP, float(job["timeout_seconds"]))
    seeds = SS.g1_seeds()
    anchors = SB.v2_gated_anchors()
    av = GL.anchor_values(anchors, budget)
    formal = float(SB.FORMAL_REPORT["H"]["H2"]["median_exact_rho_s10"])
    i0 = {"median_rho_s10": av["median_rho_s10"], "formal_report_value": formal, "h1_max_rel": av["h1_max_rel"],
          "ok": bool(abs(av["median_rho_s10"] - formal) <= 1e-9 * max(1.0, abs(formal)) and av["h1_max_rel"] <= 1e-9)}
    probes, slots = av["probes"], av["slots"]
    n, k = probes.shape[:2]
    obs_rep, act_flat = np.repeat(av["anchor_obs"], k, axis=0), probes.reshape(-1, 4)
    rho = av["rho"]["s10"]
    worlds_out, f0 = {}, True
    for world in ("stall3", "nostall"):
        deadline()
        ds = SB.load_dataset(world)
        rows = GL.fitting_rows_with_states(ds.arrays)
        arm, _ = M.arm_masks(rows["obs"], rows["actions"], rows["times"], rows["episode_ids"], seed=SS.data_seeds()["random_control"])
        cutoff = arm["cutoff"]
        mean, std = GF.normalizer(rows["obs"][cutoff])
        labels, lrec = GL.labels(rows, world, budget)
        np.savez_compressed(out / f"labels_{world}.npz", labels=labels, times=rows["times"], episode_ids=rows["episode_ids"])
        qpi_w = av[world]
        wrec = {"dataset_sha256": ds.digest, "normalizer": {"mean": mean.tolist(), "std": std.tolist(), "cutoff_rows": int(cutoff.sum())},
                "rows": int(len(labels)), "w_rows": int(rows["past_knee"].sum()),
                "labels": {key: v for key, v in lrec.items() if key != "receipt"}, "fits": {}}
        for s_index, seed in enumerate(seeds):
            deadline()
            params_by, curve = GF.fit(rows["obs"], rows["actions"], labels, mean, std, seed, CHECKPOINTS)
            conv, converged = converged_checkpoint(curve)
            fit_rec = {"seed": int(seed), "loss_last_chunk": {str(c): float(curve[c]) for c in CHECKPOINTS},
                       "converged_at": conv, "converged": converged, "checkpoints": {}}
            for ckpt in (conv, LAST) if conv != LAST else (LAST,):
                params = params_by[ckpt]
                q1 = GF.q_values(params, mean, std, obs_rep, act_flat).reshape(n, k)
                finite = bool(np.all(np.isfinite(q1))) and all(np.isfinite(v) for v in curve.values())
                f0 = f0 and finite
                kcs = {slot: SM.kc(q1, qpi_w, slots, slot) for slot in slots if slot != "edge_knee"}
                entry = {"anchor_mse": float(np.mean((q1 - qpi_w) ** 2)), "finite": finite,
                         "kc_quantiles": {slot: {str(q): float(np.quantile(v, q)) for q in (0.1, 0.5, 0.9)} for slot, v in kcs.items()}}
                if world == "stall3":
                    entry["reading"] = read(kcs, rho)
                    entry["kc_s10"] = kcs["s10"].tolist()
                    entry["kc_edge_700_abs"] = np.abs(kcs["edge_700"]).tolist()
                else:
                    entry["smooth_resolution_share_s10"] = float(np.mean(np.abs(kcs["s10"]) <= TOL))
                raw = GF.params_bytes(params)
                (out / f"params_{world}_seed{s_index}_{ckpt:06d}.msgpack").write_bytes(raw)
                entry["params_sha256"] = sha_bytes(raw)
                fit_rec["checkpoints"][str(ckpt)] = entry
            wrec["fits"][str(s_index)] = fit_rec
        worlds_out[world] = wrec
    fits = worlds_out["stall3"]["fits"]
    at_conv = {s: f["checkpoints"][str(f["converged_at"])]["reading"] for s, f in fits.items()}
    representable = {s: r["representable"] for s, r in at_conv.items()}
    headline = label(i0_ok=i0["ok"], f0_ok=f0, representable_by_seed=representable)
    shares = [r["l1_absolute_reported"]["share"] for r in at_conv.values()]
    n_valid = next(iter(at_conv.values()))["n_valid"]
    pooled = float(np.mean(shares)) if shares else 0.0
    q = GM.q_star(pooled, n_valid)
    band_700 = GM.resolution_band(np.concatenate([fits[s]["checkpoints"][str(fits[s]["converged_at"])]["kc_edge_700_abs"] for s in fits]))
    decision = {"headline": headline, "reported_only": True, "converged_at_by_seed": {s: f["converged_at"] for s, f in fits.items()},
                "converged_by_seed": {s: f["converged"] for s, f in fits.items()},
                "in_support_band_by_seed": {s: r["in_support_band"] for s, r in at_conv.items()},
                "median_kc_s10_by_seed": {s: r["median_kc_s10"] for s, r in at_conv.items()},
                "departure_capture_by_seed": {s: r["departure_capture"] for s, r in at_conv.items()},
                "representable_by_seed": representable, "rho_s10_median": float(np.median(rho)),
                "nonpositive_rho_anchors": int(np.sum(rho <= 0)),
                "l1_absolute_share_by_seed": {s: r["l1_absolute_reported"]["share"] for s, r in at_conv.items()},
                "q_star": None if q is None else q["q_star"], "q_star_detail": q,
                "resolution_band": None if band_700 is None else band_700["band"], "resolution_band_detail": band_700}
    return {"kind": "g1", "rule": "REVISION_r5.md section 5 on G1_RULE.md's fit", "I0": i0, "F0": {"ok": f0}, "anchors": n,
            "slots": slots, "worlds": worlds_out, "decision": decision, "seeds": [int(s) for s in seeds],
            "upstream_match": True, "transitions": budget.used, "by_unit": budget.ledger}
