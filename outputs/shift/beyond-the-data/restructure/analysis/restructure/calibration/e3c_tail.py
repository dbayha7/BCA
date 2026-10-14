"""READ-ONLY numpy. The surviving past-80% training rows: how far past 80% they go, and the one-step error delta
against the exceedance e = max_k |a_k| - 0.8, per world (pooled seeds 0-4 and per seed): slope by least squares
through the in-range mean (delta = d_in + beta e), and the share of rows with delta below the in-range 10% quantile."""
import json
import numpy as np
OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/calibration"
K = np.float32(0.8)
res = {}
for world in ("stall", "no_stall"):
    E, D, S, Dq = [], [], [], []
    per = {}
    for seed in range(5):
        d = np.load(f"{OUT}/data_seed{seed}_{world}.npz")
        dl = np.load(f"{OUT}/delta_seed{seed}_{world}.npz")["train_delta"]
        thr = np.abs(d["train_act"]).max(1)
        m = thr > K
        din = dl[(thr < 0.8) & (thr >= 0.7)]
        e = thr[m] - 0.8
        y = dl[m] - din.mean()
        beta = float(np.sum(e * y) / np.sum(e * e))
        q10 = np.quantile(din, 0.1)
        per[seed] = dict(n=int(m.sum()), e_q=[float(np.quantile(e, q)) for q in (.1, .5, .9)], beta=beta,
                         share_below_inrange_q10=float(np.mean(dl[m] < q10)),
                         shift_mean=float(y.mean()))
        E.append(e); D.append(y); S.append(np.full(m.sum(), seed)); Dq.append(dl[m] < q10)
    E, D = np.concatenate(E), np.concatenate(D)
    beta = float(np.sum(E * D) / np.sum(E * E))
    bins = {}
    for lo, hi in ((0, .05), (.05, .1), (.1, .2)):
        mm = (E >= lo) & (E < hi)
        bins[f"e[{lo},{hi})"] = dict(n=int(mm.sum()), mean_shift=float(D[mm].mean()) if mm.any() else None)
    res[world] = dict(per_seed=per, pooled_beta=beta, pooled_n=int(len(E)), bins=bins,
                      pooled_share_below_inrange_q10=float(np.mean(np.concatenate(Dq))))
    print(world, "beta", round(beta, 2), "n", len(E), bins, "share<q10", round(res[world]["pooled_share_below_inrange_q10"], 3))
    for s, v in per.items():
        print("  ", s, v["n"], [round(x, 3) for x in v["e_q"]], "beta", round(v["beta"], 2), "shift", round(v["shift_mean"], 3), "share<q10", round(v["share_below_inrange_q10"], 2))
json.dump(res, open(f"{OUT}/e3c_tail.json", "w"), indent=1)
