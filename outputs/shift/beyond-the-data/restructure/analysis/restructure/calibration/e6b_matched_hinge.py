"""READ-ONLY forward passes. e6's best-response read-out for the MATCHED range hinge of the Stage 2 policy test
(coefficient from R2/policy/seed<K>/lock.json:standin.coefficient, times the room's form 100 sum_k [a_k - hi_k]_+^2 +
[lo_k - a_k]_+^2), stalling world, T1 and T3a, unpaired as trained. Its coefficient was matched to the weighted bca_k1
actor's region strength, so this is a yardstick, not M4 itself."""
import json
import numpy as np
import jax.numpy as jnp
import e6_bestresponse as E6

R2 = "/home/user/bca/runs/shift_explore/stage2/policy"
res = {}
for seed in range(5):
    c = float(json.load(open(f"{R2}/seed{seed}/lock.json"))["standin"]["coefficient"][0])
    pz = np.load(f"{E6.DET}/seed{seed}/stall/pairs.npz")
    rec = json.load(open(f"{E6.ST2}/uniform/stall/fixed/seed{seed}/band.json"))
    mean, std = np.asarray(rec["data"]["obs_mean"], np.float32), np.asarray(rec["data"]["obs_std"], np.float32)
    crit = E6.load_critic("stall", seed)
    a_tr = np.load(f"{E6.OUT}/data_seed{seed}_stall.npz")["train_act"].astype(np.float64)
    lo, hi = np.quantile(a_tr, 0.005, axis=0), np.quantile(a_tr, 0.995, axis=0)
    r = dict(coef=c)
    for setname, src in (("T1", None), ("T3a", "actor")):
        st = "T1" if setname == "T1" else "T3"
        m = (pz["set"] == st) & (pz["role"] == "command")
        if src:
            m &= pz["t3_source"] == src
        rows = np.flatnonzero(m)
        s = np.asarray(pz["states"][rows])
        o = jnp.asarray((s.astype(np.float32) - mean) / std)
        base = np.asarray(pz["commands"][rows], np.float32)
        cur = s[:, 10]
        g = np.asarray(E6.GRID)
        Qg, Pg = [], []
        for mm in E6.GRID:
            a = np.clip(base.astype(np.float64), -mm, mm)
            Qg.append(np.asarray(E6.Q(crit, o, jnp.asarray(a, jnp.float32)), np.float64)[:, 0])
            Pg.append(c * 100 * (np.maximum(a - hi, 0) ** 2 + np.maximum(lo - a, 0) ** 2).sum(1))
        Qg, Pg = np.stack(Qg), np.stack(Pg)
        best = g[(Qg - Pg).argmax(0)]
        r[setname] = dict(best_past80=float(np.mean(best > 0.8)), P_at_1=float(Pg[-1].mean()),
                          P_at_08=float(Pg[E6.GRID.index(0.80)].mean()), P_at_07=float(Pg[0].mean()))
        if setname == "T3a":
            for clo, chi in ((0.7, 0.9), (0.9, 1.0), (1.0, 1.2)):
                mm_ = (cur >= clo) & (cur < chi)
                r[setname][f"best_past80_cur[{clo},{chi})"] = float(np.mean(best[mm_] > 0.8))
    res[seed] = r
    print(seed, round(c, 3), {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in r.items() if k != "coef"})
pooled = {sn: {k: float(np.mean([res[s][sn][k] for s in res])) for k in res[0][sn]} for sn in ("T1", "T3a")}
print(json.dumps(pooled, indent=0))
json.dump(dict(per=res, pooled=pooled), open(f"{E6.OUT}/e6b_matched_hinge.json", "w"), indent=1)
