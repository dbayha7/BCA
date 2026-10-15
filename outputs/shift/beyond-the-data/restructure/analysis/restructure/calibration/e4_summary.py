"""READ-ONLY numpy: pooled summary of e4_tailband.json (means over seeds 0-4; seed 3 no-stall's tail cannot certify
under BCA's rule, so its tail widths are the refusal fallback and it is reported separately)."""
import json
import numpy as np
OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/calibration"
d = json.load(open(f"{OUT}/e4_tailband.json"))
SIG = ("tail_B", "tail_A", "weighted_bca", "uniform_bca", "classifier_ratio", "distance", "range_hinge",
       "residual_evidence", "eta_only")
KEYS = ("T1_direction_share", "T1_size_median", "rank_T12", "rank_T3", "T3a_auc_past80_cur[0.7,0.9)",
        "T3a_auc_past80_cur[0.9,1.0)", "T3a_auc_past80_cur[1.0,1.2)", "T3a_auc_past80_all", "T3a_fire_share",
        "T3a_recall_past80", "T3a_false_flag", "T1_penalty_minus_cap_mean_past80", "T1_fire_share_cmd_past80",
        "T1_fire_share_cap")
summ = {}
for w in ("stall", "no_stall"):
    rs = [r for r in d if r["world"] == w]
    summ[w] = {}
    for s in SIG:
        summ[w][s] = {}
        for k in KEYS:
            vals = [r["metrics"][s][k] for r in rs]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            summ[w][s][k] = dict(mean=float(np.mean(vals)) if vals else None,
                                 per_seed=[None if r["metrics"][s][k] is None else round(r["metrics"][s][k], 3) for r in rs])
# world selectivity of the paired penalty at past-80% T1 commands: stall over no-stall
sel = {}
for s in SIG:
    st = np.array([r["metrics"][s]["T1_penalty_minus_cap_mean_past80"] for r in d if r["world"] == "stall"])
    ns = np.array([r["metrics"][s]["T1_penalty_minus_cap_mean_past80"] for r in d if r["world"] == "no_stall"])
    sel[s] = dict(stall=st.round(3).tolist(), no_stall=ns.round(3).tolist(),
                  ratio_mean=float(np.mean(st) / np.mean(ns)) if np.all(np.isfinite(ns)) and np.mean(ns) != 0 else None,
                  ratio_mean_excl_seed3=float(np.mean(st) / np.mean(np.delete(ns, 3))) if np.mean(np.delete(ns, 3)) != 0 else None)
summ["world_selectivity_T1_penalty_past80"] = sel
summ["ramp_value_units"] = {w: {f: [round(r[f"R_{f}"]["tail_minus_bulk_value"], 3) for r in d if r["world"] == w]
                                for f in ("A", "B")} for w in ("stall", "no_stall")}
summ["ramp_shuffled_value_units"] = {w: {f: [round(r[f"R_{f}"]["shuffled_tail_minus_bulk_value"], 3) for r in d if r["world"] == w]
                                         for f in ("A", "B")} for w in ("stall", "no_stall")}
json.dump(summ, open(f"{OUT}/e4_summary.json", "w"), indent=1)
for w in ("stall", "no_stall"):
    print("==", w)
    for s in SIG:
        print(f"{s:18s}", {k.replace("T3a_auc_past80_", "auc").replace("_share", ""): (None if v["mean"] is None else round(v["mean"], 3)) for k, v in summ[w][s].items()})
print(json.dumps(sel, indent=0))
print(summ["ramp_value_units"], summ["ramp_shuffled_value_units"])
