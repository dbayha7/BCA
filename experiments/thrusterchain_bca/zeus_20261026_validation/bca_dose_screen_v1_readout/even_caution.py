"""Post hoc, descriptive: on the saved anchors (fixed networks), compare BCA's U contrast at kappa_A with
(a) the same dU shuffled across anchors (no placement) and (b) a uniform ramp of the same typical size (even caution).
Stall world: correct-ranking gain on harmful anchors and false flips on non-harmful ones.
No-stall world: false caution (non-harmful anchors newly ranked s10-worse) and the correct share on harmful ones."""
import json, sys
from pathlib import Path
import numpy as np

VAL = Path(r"C:\Users\David\dev\BCA\experiments\thrusterchain_bca\zeus_20261026_validation")
ATT = VAL / "bca_dose_screen_v1" / "screen_run" / "attempt_1"
KAPPA, SHUFFLES = 63.0, 4000
rng = np.random.default_rng(20261031)
rd = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
NAMES = {"nan": np.nan, "positive_infinity": np.inf, "negative_infinity": -np.inf}
arr = lambda v: np.asarray([[NAMES.get(x, x) for x in r] for r in v], np.float64)  # noqa: E731


def contrasts(entry, slot="s10"):
    a = entry["anchors_e2"]
    names = a["gated_slots"]
    q1, qpi, u = arr(a["q1"])[:48], arr(a["qpi_clip"])[:48], arr(a["U"])[:48]
    valid = np.asarray(a["valid"], bool)[:48]
    i, k = names.index(slot), names.index("edge_knee")
    ct, cr, du = qpi[:, i] - qpi[:, k], q1[:, i] - q1[:, k], u[:, i] - u[:, k]
    keep = valid & np.isfinite(ct) & np.isfinite(cr) & np.isfinite(du)
    return ct[keep], cr[keep], du[keep]


def score(ct, cr, pen):
    """pen: penalty contrast kappa*dU per anchor. Returns gain on harmful, broke, false flips on non-harmful."""
    cp = cr - pen
    harm, ok = ct < 0, ct >= 0
    raw_share = np.mean(cr[harm] < 0) if harm.any() else np.nan
    pen_share = np.mean(cp[harm] < 0) if harm.any() else np.nan
    return {"gain": float(pen_share - raw_share), "pen_share": float(pen_share), "raw_share": float(raw_share),
            "broke": int((harm & (cr < 0) & (cp >= 0)).sum()),
            "false_flips": int((ok & (cr >= 0) & (cp < 0)).sum()), "non_harmful": int(ok.sum()), "harmful": int(harm.sum())}


out = {}
for i in (0, 1, 2):
    rep = rd(ATT / f"evaluate_s{i}" / "report.json")
    for arm in ("cutoff", "random"):
        for world in ("stall3", "nostall"):
            name = f"s{i}_{arm}_{world}"
            ct, cr, du = contrasts(rep["runs"][name]["020000"])
            m = float(np.median(np.abs(du)))
            bca = score(ct, cr, KAPPA * du)
            ramp = score(ct, cr, np.full_like(du, KAPPA * m))
            flip = score(ct, cr, -KAPPA * du)
            sh = [score(ct, cr, KAPPA * rng.permutation(du)) for _ in range(SHUFFLES)]
            gains = np.array([s["gain"] for s in sh])
            out[name] = {"median_abs_dU": m, "bca": bca, "uniform_ramp": ramp, "sign_flipped": flip,
                         "shuffled_mean_gain": float(gains.mean()),
                         "shuffled_share_at_or_above_bca": float(np.mean(gains >= bca["gain"] - 1e-12)),
                         "shuffled_mean_false_flips": float(np.mean([s["false_flips"] for s in sh]))}
print(json.dumps(out, indent=1))
