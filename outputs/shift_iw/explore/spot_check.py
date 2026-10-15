"""Independent spot check of RESULTS.md from the per-seed evaluate records (no make_results.py code): weighted per step
minus unweighted on the strong starts, and the absolute points of plain TD3+BC, unweighted BCA and weighted BCA."""
import json

import numpy as np

ROOT = "/root/bca-shift/runs/shift_iw_v5_explore"
V5 = "/root/bca-shift/runs/shift_iw_v5"
QS = ["random", "medium-replay", "medium", "medium-expert", "expert"]

for q in QS:
    ev = {g: [json.load(open(f"{ROOT}/{q}/evaluate/seed{s}/{g}.json"))["evaluations"] for s in range(5)]
          for g in ("bca", "bca_lower_edge")}
    if q == "random":
        print("arms:", list(ev["bca"][0]), list(ev["bca_lower_edge"][0]))
    cp = np.array([e["bca_exact_strong"]["strong"]["J"] - e["bca_unweighted"]["strong"]["J"] for e in ev["bca"]])
    le_keys = list(ev["bca_lower_edge"][0])
    lw = [k for k in le_keys if "strong" in k and "single" not in k and "mild" not in k]
    lu = [k for k in le_keys if "unweighted" in k]
    le = np.array([e[lw[0]]["strong"]["J"] - e[lu[0]]["strong"]["J"] for e in ev["bca_lower_edge"]])
    pts = {name: np.mean([e[k]["strong"]["points"] for e in ev[g]])
           for name, g, k in (("BCA capped pull", "bca", "bca_unweighted"),
                              ("weighted capped pull", "bca", "bca_exact_strong"),
                              ("BCA lower edge", "bca_lower_edge", lu[0]),
                              ("weighted lower edge", "bca_lower_edge", lw[0]))}
    rep = json.load(open(f"{V5}/{q}/report/report.json"))
    plain = rep["shown"]["plain_td3bc_points"]["gap"]["strong"]["mean"]
    print(f"{q:14s} capped pull w-u {cp.mean():+7.1f} {np.round(cp, 1)} | lower edge w-u {le.mean():+5.1f} "
          f"({lw[0]} vs {lu[0]}) | strong points: plain {plain:.1f}, "
          + ", ".join(f"{k} {v:.1f}" for k, v in pts.items()))
