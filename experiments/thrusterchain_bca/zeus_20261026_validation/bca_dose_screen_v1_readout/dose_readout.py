"""Read-only breakdown of the dose screen's saved JSON reports (no checkpoints, no datasets).
1) recompute the verdicts with screen_math on Windows and compare with SCREEN_RESULT.json;
2) per-anchor Q-value breakdown at s05/s10 for every BCA run (stall3 and nostall)."""
import json, sys
from pathlib import Path
import numpy as np

VAL = Path(r"C:\Users\David\dev\BCA\experiments\thrusterchain_bca\zeus_20261026_validation")
PKG = VAL / "bca_dose_screen_v1"
ATT = PKG / "screen_run" / "attempt_1"
EXP = VAL / "knee_edge_ood_v2" / "stage1_exploratory_run" / "attempt_1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(PKG))
import screen_math as SM  # noqa: E402

rd = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
bca = {i: rd(ATT / f"evaluate_s{i}" / "report.json") for i in SM.SEEDS}
nat = {i: rd(EXP / f"evaluate_s{i}" / "report.json") for i in SM.SEEDS}
forks = [f"s{i}_{a}_{w}" for i in SM.SEEDS for a in SM.ARMS for w in SM.WORLDS]
audits = {n: SM.load_audit(ATT / f"bca_branch_{n}") for n in forks}
kappa = rd(PKG / "DOSE.json")["kappa_A"]
dec = SM.decide(bca, nat, audits, kappa)
saved = rd(ATT / "SCREEN_RESULT.json")
out = {"recomputed_verdicts": {a: dec[a]["verdict"] for a in SM.ARMS}, "saved_verdicts": saved["verdicts"]}
out["recomputed_equals_saved"] = json.dumps(dec, sort_keys=True, default=str) == json.dumps(saved["decision"], sort_keys=True, default=str)

def anchors(entry, k=None):
    a = entry["anchors_e2"]
    names = a["gated_slots"]
    q1, qpi = SM.arr(a["q1"])[:48], SM.arr(a["qpi_clip"])[:48]
    valid = np.asarray(a["valid"], bool)[:48]
    u = SM.arr(a["U"])[:48] if "U" in a else None
    return names, q1, qpi, valid, u

def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else None

rows = {}
for i in SM.SEEDS:
    for arm in SM.ARMS:
        for world in SM.WORLDS:
            name = f"s{i}_{arm}_{world}"
            names, q1, qpi, valid, u = anchors(bca[i]["runs"][name]["020000"])
            _, q1n, qpin, validn, _ = anchors(nat[i]["runs"][name]["020000"])
            k = names.index("edge_knee")
            r = {}
            for slot in ("s05", "s10"):
                j = names.index(slot)
                ct, cr, du = qpi[:, j] - qpi[:, k], q1[:, j] - q1[:, k], u[:, j] - u[:, k]
                keep = valid & np.isfinite(ct) & np.isfinite(cr) & np.isfinite(du)
                kc = cr - ct
                cp = cr - kappa * du
                harm = keep & (ct < 0)
                wrong = harm & (cr >= 0)                       # raw Q1 ranks the harmful slot as at least as good
                need = np.where(du > 0, cr / np.where(du > 0, du, 1), np.inf)   # kappa that just flips a wrong anchor
                r[slot] = {
                    "valid": int(keep.sum()), "harmful": int(harm.sum()), "raw_wrong": int(wrong.sum()),
                    "share_dU_positive_all": float(np.mean(du[keep] > 0)),
                    "share_dU_positive_harmful": float(np.mean(du[harm] > 0)) if harm.any() else None,
                    "median_dU": float(np.median(du[keep])), "median_kappa_dU": float(np.median(kappa * du[keep])),
                    "median_KC": float(np.median(kc[keep])),
                    "median_penalty_over_KC_harmful": float(np.median(kappa * du[harm] / kc[harm])) if harm.any() else None,
                    "spearman_dU_KC_all": spearman(du[keep], kc[keep]),
                    "fixed_wrong_to_right": int((wrong & (cp < 0)).sum()),
                    "broke_right_to_wrong": int((harm & (cr < 0) & (cp >= 0)).sum()),
                    "nonharmful_now_ranked_worse": int((keep & (ct >= 0) & (cr >= 0) & (cp < 0)).sum()),
                    "kappa_needed_wrong_median": float(np.median(need[wrong])) if wrong.any() else None,
                    "share_wrong_fixable_at": {str(x): float(np.mean(need[wrong] <= x)) if wrong.any() else None
                                               for x in (63, 250, 1000, 5000)},
                    "native_median_KC": float(np.median((q1n[:, j] - q1n[:, k] - (qpin[:, j] - qpin[:, k]))[validn])),
                    "median_C_true": float(np.median(ct[keep])),
                }
            rows[name] = r
out["anchors"] = rows
print(json.dumps(out, indent=1))
