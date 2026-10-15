"""Read-only: certificate statistic with all 200 episodes of evidence in the no-stall-physics worlds (N, M4),
and empirical 90% quantiles of the two groups; tags of the look after episode e (as the run used)."""
import sys
sys.dont_write_bytecode = True
import numpy as np
sys.path.insert(0, "/home/user/bca/outputs/shift/beyond-the-data/restructure/step1")
import step1lib as S1
OUT = "/home/user/bca/runs/shift_explore/routes/runtime_monitor"
spec, _ = S1.load_prespec(); cfg = spec["certificate"]
WI = {"S": 1, "N": 2, "M3": 3, "M4": 4}
for s in range(5):
    z = np.load(f"{OUT}/seed{s}/arrays.npz")
    for k in ("N", "M4"):
        sc, tl = z[f"{k}_strong_current_score"], z[f"{k}_strong_current_tail"]
        row = []
        for e in (20, 50, 200):
            c = S1.certificate(sc[:e][tl[:e]], sc[:e][~tl[:e]], [20261015, 9930, s, WI[k], e], cfg)
            row.append("e%d q05 %.3f on %s" % (e, c["q05"], c["on"]))
        t, r = sc[tl], sc[~tl]
        print(s, k, "; ".join(row), "| q90 tail %.3f rest %.3f | mean tail %.3f rest %.3f" % (
            np.quantile(t, .9), np.quantile(r, .9), t.mean(), r.mean()), flush=True)
    # bank-based reference (recorded conditions) for the log worlds
for s in range(5):
    z = np.load(f"{OUT}/seed{s}/arrays.npz")
    for w in ("stall", "no_stall"):
        bt, br = z[f"bank_tail_{w}"], z[f"bank_rest_{w}"]
        print(s, w, "bank n tail %d rest %d q90 tail %.3f rest %.3f" % (len(bt), len(br), np.quantile(bt, .9) if len(bt) else np.nan, np.quantile(br, .9)))
