"""READ-ONLY numpy (BCA's wbcp on saved residuals; no fit). Is the full-throttle stratum CREDIBLY worse than the bulk?
For the one-sided score B = -delta / u_f: the WBCP posterior of the tail stratum's 90% overestimate quantile (its
per-draw alpha-crossings, wbcp.Calibration.crossings) against the bulk's plain 90% quantile. Reports, in value units
(x u_f): the plain tail-minus-bulk gap, the 5% / 50% posterior quantiles of the gap, and P(gap <= 0); and the same for
BCA's own score A (|delta| / (u_f eta_f))."""
import json
import sys
import numpy as np
import jax.numpy as jnp
from flax import serialization
sys.path.insert(0, "/home/user/bca")
from calibration import wbcp  # noqa: E402
from calibration.network import Calibrator  # noqa: E402
OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/calibration"
ST2 = "/home/user/bca/runs/shift_explore/stage2/detect"
CAL = Calibrator(jnp.zeros(13), jnp.ones(13), state_dep=True)
e4 = {(r["seed"], r["world"]): r for r in json.load(open(f"{OUT}/e4_tailband.json"))}
res = {}
for w in ("stall", "no_stall"):
    for s in range(5):
        E = e4[(s, w)]
        d = np.load(f"{OUT}/data_seed{s}_{w}.npz")
        dl = np.load(f"{OUT}/delta_seed{s}_{w}.npz")["train_delta"]
        bu = serialization.msgpack_restore(open(f"{ST2}/uniform/{w}/fixed/seed{s}/band.msgpack", "rb").read())
        u = float(bu["unit"])
        eta = np.maximum(np.asarray(CAL.apply(bu["cal_params"], jnp.asarray(d["train_obs"]), jnp.asarray(d["train_act"])), np.float64), 1e-6)
        tail = np.abs(d["train_act"]).max(1) >= E["t_up"]
        o = {}
        for nm, sc, scale in (("B", -dl / u, u), ("A", np.abs(dl) / (u * eta), u * float(np.median(eta)))):
            bulk_q = float(np.quantile(sc[~tail], 0.9, method="inverted_cdf"))
            c = wbcp.calibrate(sc[tail], np.random.default_rng([20261015, 9400, s, ord(nm)]), alpha=0.1, beta=0.95, draws=1000)
            cr = c.crossings
            fin = np.isfinite(cr)
            gap = (cr - bulk_q) * scale
            o[nm] = dict(plain_gap=float((c.lambda_hat - bulk_q) * scale),
                         post_q05_gap=float(np.quantile(gap, 0.05)), post_q50_gap=float(np.quantile(gap, 0.5)),
                         P_gap_le_0=float(np.mean(gap <= 0)), finite_share=float(fin.mean()))
        res[f"{s}_{w}"] = o
        print(s, w, "n_tail", int(tail.sum()), {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in o.items()})
json.dump(res, open(f"{OUT}/e11_credible_gap.json", "w"), indent=1)
