"""READ-ONLY numpy/scipy. Why the trace test bed's residual_evidence stand-in (mean delta over the 10 nearest kept
training rows in z-scored (state, command) space, detect/code/signals.py KNN) missed the trace: the share of
full-throttle (tail) rows among the 10 nearest neighbours of the actor's past-80% T1 and T3a commands."""
import json
import numpy as np
from scipy.spatial import cKDTree
OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/calibration"
DET = "/home/user/bca/runs/shift_explore/detect"
e4 = {(r["seed"], r["world"]): r for r in json.load(open(f"{OUT}/e4_tailband.json"))}
res = {}
for w in ("stall", "no_stall"):
    for s in range(5):
        d = np.load(f"{OUT}/data_seed{s}_{w}.npz")
        X = np.concatenate([d["train_states"].astype(np.float64), d["train_act"].astype(np.float64)], 1)
        mu, sd = X.mean(0), X.std(0)
        tree = cKDTree((X - mu) / sd)
        tail = np.abs(d["train_act"]).max(1) >= e4[(s, w)]["t_up"]
        pz = np.load(f"{DET}/seed{s}/{w}/pairs.npz")
        lab = np.load(f"{DET}/seed{s}/{w}/labels.npz")
        o = {}
        for st, src in (("T1", None), ("T3", "actor")):
            m = (pz["set"] == st) & (pz["role"] == "command") & lab["unsupported"]
            if src:
                m &= pz["t3_source"] == src
            Q = np.concatenate([pz["states"][m], pz["commands"][m].astype(np.float64)], 1)
            _, idx = tree.query((Q - mu) / sd, k=10)
            o[st] = dict(mean_tail_share_in_10nn=float(tail[idx].mean()), any_tail_share=float(tail[idx].any(1).mean()))
        res[f"{s}_{w}"] = o
        print(s, w, o)
json.dump(res, open(f"{OUT}/e9_knn_dilution.json", "w"), indent=1)
