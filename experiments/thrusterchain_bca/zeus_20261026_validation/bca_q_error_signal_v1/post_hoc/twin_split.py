"""Own check of the twin-local split at RANDOM/CUTOFF stall3 anchors (native, 20k), JSON only.
In the stall world s05 = -0.85 executes the same thrust as -0.65 (slope 3), so Q^pi(s05) - Q^pi(-0.65) = -0.09 exactly
(action cost 0.3*(0.85^2-0.65^2)). L = Q1(s05) - Q1(-0.65) + 0.09 is Q1's twin-local (one-step) error at the anchor."""
import json
from pathlib import Path
import numpy as np
EXP = Path(r"C:\Users\David\dev\BCA\experiments\thrusterchain_bca\zeus_20261026_validation\knee_edge_ood_v2\stage1_exploratory_run\attempt_1")
X = {"edge_knee": -0.8, "edge_775": -0.775, "edge_750": -0.75, "edge_725": -0.725, "edge_700": -0.70, "mid_m40": -0.40}
N = {"nan": np.nan}
arr = lambda v: np.asarray([[N.get(x, x) for x in r] for r in v], np.float64)
for i in (0, 1, 2):
    rep = json.loads((EXP / f"evaluate_s{i}" / "report.json").read_text(encoding="utf-8"))
    for arm in ("random", "cutoff"):
        a = rep["runs"][f"s{i}_{arm}_stall3"]["020000"]["anchors_e2"]
        names = a["gated_slots"]
        q1, qpi = arr(a["q1"])[:48], arr(a["qpi_clip"])[:48]
        cols = [names.index(k) for k in X]
        xs = np.array(list(X.values()))
        out = {}
        for deg in (2, 3):
            qpi_t = np.array([np.polyval(np.polyfit(xs, qpi[r, cols], deg), -0.65) for r in range(48)])
            q1_t = np.array([np.polyval(np.polyfit(xs, q1[r, cols], deg), -0.65) for r in range(48)])
            s, k = names.index("s05"), names.index("edge_knee")
            ident = qpi[:, s] - qpi_t + 0.09                          # should be ~0 (twin identity check of the interpolant)
            L = q1[:, s] - q1_t + 0.09
            kc = (q1[:, s] - q1[:, k]) - (qpi[:, s] - qpi[:, k])
            out[deg] = (np.median(np.abs(ident)), np.median(L), np.median(kc), np.median(L) / np.median(kc))
        print(f"s{i} {arm:6s}", "  ".join("deg%d |ident| %.4f  L %.3f  KC_s05 %.3f  L/KC %.2f" % (d, *v) for d, v in out.items()))
