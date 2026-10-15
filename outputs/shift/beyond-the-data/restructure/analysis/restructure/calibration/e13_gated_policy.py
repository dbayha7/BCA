"""READ-ONLY. The GATED stratified band in the actor's objective and the detection read-outs: width
U(s, a) = h ramp(t(a)) (+ a constant), h = the posterior-median tail-minus-bulk gap of score B (e11) if its 5%
posterior bound is > 0, else 0. Best response along a throttle cap at T1 / T3a (e6's read-out, k = 1, 2, 4) and the
T1 size (paired penalty over the critic's false preference at harmful pairs) per seed. No training, no fitting."""
import copy
import json
import numpy as np
import e6_bestresponse as E6

OUT = E6.OUT
gates = json.load(open(f"{OUT}/e11_credible_gap.json"))
e4 = {(r["seed"], r["world"]): r for r in json.load(open(f"{OUT}/e4_tailband.json"))}
E6.KS = (1, 2, 4)
res = {}
for world in ("stall", "no_stall"):
    for seed in range(5):
        g = gates[f"{seed}_{world}"]["B"]
        h = g["post_q50_gap"] if g["post_q05_gap"] > 0 else 0.0
        E = copy.deepcopy(e4[(seed, world)])
        u = E["unit"]
        E["R_B"] = dict(bulk=0.0, tail=h / u)
        r = E6.run(seed, world, {(seed, world): E})
        # T1 size from the widths: P(cmd) - P(cap) over over_pref at harmful pairs
        pz = np.load(f"{E6.DET}/seed{seed}/{world}/pairs.npz")
        lab = np.load(f"{E6.DET}/seed{seed}/{world}/labels.npz")
        t = np.abs(pz["commands"]).max(1).astype(np.float64)
        f = np.clip((t - E["t_lo"]) / (E["t_hi"] - E["t_lo"]), 0, 1)
        U = h * f
        st, ro, pr = pz["set"], pz["role"], pz["pair"]

        def idx(s_, r_):
            rows = np.flatnonzero((st == s_) & (ro == r_))
            return rows[np.argsort(pr[rows])]
        ic, ip, ir = idx("T1", "command"), idx("T1", "cap"), idx("T1", "recorded")
        H = lab["harmful"][ic].astype(bool)
        Wc, Wp = np.maximum(U[ic] - U[ir], 0), np.maximum(U[ip] - U[ir], 0)
        size = float(np.median(((Wc - Wp) / lab["over_pref"][ic])[H])) if H.any() else None
        res[f"{seed}_{world}"] = dict(h=h, size=size, T1=r["T1"], T3a=r["T3a"])
        print(seed, world, "h", round(h, 3), "size", size and round(size, 3),
              "T1 k1/k2/k4", [round(r["T1"][f"tail_B_k{k}_best_past80"], 3) for k in (1, 2, 4)],
              "T3a k1/k2/k4", [round(r["T3a"][f"tail_B_k{k}_best_past80"], 3) for k in (1, 2, 4)],
              "Q1 alone T3a", round(r["T3a"]["q1_alone_best_past80"], 3), flush=True)
pooled = {}
for world in ("stall", "no_stall"):
    rs = [res[f"{s}_{world}"] for s in range(5)]
    pooled[world] = dict(
        h=[round(r["h"], 3) for r in rs],
        size_mean=float(np.mean([r["size"] for r in rs])) if rs[0]["size"] is not None else None,
        **{f"{sn}_k{k}_best_past80": float(np.mean([r[sn][f"tail_B_k{k}_best_past80"] for r in rs]))
           for sn in ("T1", "T3a") for k in (1, 2, 4)},
        **{f"T3a_k{k}_cur[1.0,1.2)": float(np.mean([r["T3a"][f"tail_B_k{k}_best_past80_cur[1.0,1.2)"] for r in rs])) for k in (1, 2, 4)},
        T3a_q1_alone=float(np.mean([r["T3a"]["q1_alone_best_past80"] for r in rs])),
        T1_q1_alone=float(np.mean([r["T1"]["q1_alone_best_past80"] for r in rs])))
print(json.dumps(pooled, indent=0))
json.dump(dict(per=res, pooled=pooled), open(f"{OUT}/e13_gated_policy.json", "w"), indent=1)
