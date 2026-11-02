"""The g1 unit of the native screen: G1_RULE.md (../g1/) run as one CPU job inside the screen, reported, and read by
the screen's validity (q*, the resolution band, the PASS / NOT REPRESENTABLE headline). Imported by screen_worker
after screen_bind (recorder substituted, paths set); JAX_PLATFORMS must already be "cpu" in this process.
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

CHECKPOINTS = (5_000, 10_000, 20_000, 40_000, 60_000)
HOST_BUDGET, DIAGNOSTIC = 20_000, 60_000
STEP_CAP = 9_000_000


def _l1_record(l1):
    return {k: v for k, v in l1.items() if k not in ("valid", "flags")}


def _reads(fits, updates):
    """Pooled and per-seed L1 reads of the stall3 fits at one update count."""
    rows = {s: f["checkpoints"][str(updates)]["L1"] for s, f in fits.items()}
    n_valid = {s: r["n_valid"] for s, r in rows.items()}
    n_pass = sum(r["n_pass"] for r in rows.values())
    total = sum(n_valid.values())
    return {"updates": updates, "share_by_seed": {s: r["share"] for s, r in rows.items()},
            "median_ok_by_seed": {s: r["median_ok"] for s, r in rows.items()},
            "median_kc_by_seed": {s: r["median_kc"] for s, r in rows.items()},
            "share_pooled": n_pass / total if total else 0.0, "n_valid_per_seed": n_valid}


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
            fit_rec = {"seed": int(seed), "loss_last_chunk": {str(c): curve[c] for c in CHECKPOINTS}, "checkpoints": {}}
            for ckpt, params in params_by.items():
                q1 = GF.q_values(params, mean, std, obs_rep, act_flat).reshape(n, k)
                finite = bool(np.all(np.isfinite(q1))) and all(np.isfinite(v) for v in curve.values())
                f0 = f0 and finite
                kcs = {slot: SM.kc(q1, qpi_w, slots, slot) for slot in slots if slot != "edge_knee"}
                entry = {"anchor_mse": float(np.mean((q1 - qpi_w) ** 2)), "finite": finite,
                         "kc_quantiles": {slot: {str(q): float(np.quantile(v, q)) for q in (0.1, 0.5, 0.9)} for slot, v in kcs.items()}}
                if world == "stall3":
                    entry["L1"] = _l1_record(GM.l1(kcs["s10"], rho))
                    entry["kc_s10"] = kcs["s10"].tolist()
                    entry["kc_edge_700_abs"] = np.abs(kcs["edge_700"]).tolist()
                else:
                    entry["smooth_resolution_share_s10"] = float(np.mean(np.abs(kcs["s10"]) <= GM.TOL))
                raw = GF.params_bytes(params)
                (out / f"params_{world}_seed{s_index}_{ckpt:06d}.msgpack").write_bytes(raw)
                entry["params_sha256"] = sha_bytes(raw)
                fit_rec["checkpoints"][str(ckpt)] = entry
            wrec["fits"][str(s_index)] = fit_rec
        worlds_out[world] = wrec
    fits = worlds_out["stall3"]["fits"]
    gate, diag = _reads(fits, HOST_BUDGET), _reads(fits, DIAGNOSTIC)
    headline, all_labels = GM.labels(i0_ok=i0["ok"], f0_ok=f0, median_ok_by_seed=gate["median_ok_by_seed"],
                                     share_by_seed=gate["share_by_seed"], share_pooled=gate["share_pooled"])
    n_valid = next(iter(gate["n_valid_per_seed"].values()))
    q = GM.q_star(gate["share_pooled"], n_valid)
    band = GM.resolution_band(np.concatenate([fits[s]["checkpoints"][str(HOST_BUDGET)]["kc_edge_700_abs"] for s in fits]))
    decision = {"headline": headline, "all_labels": all_labels, "at_updates": HOST_BUDGET, **gate,
                "q_star": None if q is None else q["q_star"], "q_star_detail": q,
                "resolution_band": None if band is None else band["band"], "resolution_band_detail": band,
                "nonpositive_rho_anchors": int(np.sum(rho <= 0)), "rho_s10_median": float(np.median(rho))}
    return {"kind": "g1", "rule": "../g1/G1_RULE.md", "I0": i0, "F0": {"ok": f0}, "anchors": n, "slots": slots,
            "worlds": worlds_out, "decision": decision, "diagnostic_60k": diag, "seeds": [int(s) for s in seeds],
            "upstream_match": True, "transitions": budget.used, "by_unit": budget.ledger}
