"""Adversarial verification (read-only): recompute latches, nets and controls from arrays.npz with own code."""
import sys
sys.dont_write_bytecode = True
import json, time
import numpy as np
from scipy import stats
sys.path.insert(0, "/home/user/bca/outputs/shift/beyond-the-data/restructure/step1")
import step1lib as S1
OUT = "/home/user/bca/runs/shift_explore/routes/runtime_monitor"
spec, _ = S1.load_prespec(); cfg = spec["certificate"]
WI = {"S": 1, "N": 2, "M3": 3, "M4": 4}
st1 = json.load(open("/home/user/bca/runs/shift_explore/restructure/step1/results.json"))["per_seed_world"]
offw = {"S": "stall", "N": "no_stall", "M3": "no_stall", "M4": "stall"}
res = {}
t0 = time.process_time()
for s in range(5):
    z = np.load(f"{OUT}/seed{s}/arrays.npz")
    L = json.load(open(f"{OUT}/seed{s}/looks.json"))
    for k in WI:
        sc, tl = z[f"{k}_strong_current_score"], z[f"{k}_strong_current_tail"]
        pn, po = z[f"{k}_strong_current_pts_never"], z[f"{k}_strong_current_pts_on"]
        rec = L["runs"][f"{k}/strong_current"]
        # real latch: recompute every look until on (max 200)
        lat = None; q_at = None; q_prev = None
        for e in range(1, 201):
            c = S1.certificate(sc[:e][tl[:e]], sc[:e][~tl[:e]], [20261015, 9930, s, WI[k], e], cfg)
            if c["on"]:
                lat = e; q_at = c["q05"]; h_at = c["height"]; break
            q_prev = c["q05"]
        # Mann-Whitney latch, own call
        mw = None
        for e in range(1, 201):
            a, b = sc[:e][tl[:e]], sc[:e][~tl[:e]]
            if len(a) and len(b) and stats.mannwhitneyu(a, b, alternative="greater").pvalue < 0.05:
                mw = e; break
        def comp(l):
            m = pn.copy()
            if l is not None and l < 200:
                m[l:] = po[l:]
            return m.mean()
        perm_l = rec["latch_after_episode"]["permuted"]
        on1 = bool(st1[f"{offw[k]}/seed{s}"]["certificate"]["on"])
        res[(s, k)] = dict(lat=lat, lat_file=rec["latch_after_episode"]["real"], q_at=q_at, q_prev=q_prev, h_at=h_at,
                           mw=mw, mw_file=rec["latch_after_episode"]["mann_whitney"],
                           never=pn.mean(), on=po.mean(), real=comp(lat), mwp=comp(mw), perm=comp(perm_l),
                           fleet=comp(rec["latch_after_episode"]["fleet"]),
                           bank=comp(rec["latch_after_episode"]["offline_bank_rest"]),
                           step1=po.mean() if on1 else pn.mean(),
                           ntail=int(tl.sum()), nrest=int((~tl).sum()),
                           file_pts=rec["points"])
        r = res[(s, k)]
        print(s, k, "latch", lat, "(file", r["lat_file"], ") q05 prev %s at %.3f height %.3f" % (
            None if q_prev is None else round(q_prev, 3), q_at, h_at), "MW", mw, "(file", r["mw_file"], ")",
            "real pts %.3f file %.3f" % (r["real"], rec["points"]["real"]), flush=True)
def net(arm):
    g = {k: np.mean([res[(s, k)][arm] - res[(s, k)]["never"] for s in range(5)]) for k in WI}
    return g["S"] + g["M3"] - (-g["N"]) - (-g["M4"]), g
for arm in ("real", "mwp", "perm", "fleet", "bank", "on", "step1"):
    n, g = net(arm)
    print(arm, "net %.2f" % n, {k: round(v, 2) for k, v in g.items()})
orc = np.mean([res[(s, "S")]["on"] - res[(s, "S")]["never"] for s in range(5)]) + np.mean(
    [res[(s, "M3")]["on"] - res[(s, "M3")]["never"] for s in range(5)])
print("oracle %.2f  85%% %.2f  half real %.2f" % (orc, 0.85 * orc, 0.5 * net("real")[0]))
print("cpu", time.process_time() - t0)
