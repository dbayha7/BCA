"""The frozen reads and readings of the overestimation comparison (OVERESTIMATION_RULE.md). Pure numpy on the
saved `oe_evaluate` reports; no simulator, no GPU."""
import numpy as np

D_E_OWN, D_J, REACH = 0.5, 2.0, 0.01
SEEDS = (0, 1, 2)
ORDER = ("INCOMPLETE", "INCONCLUSIVE", "NOT REACHED", "HARMFUL", "PROMISING", "NO EFFECT", "MIXED")


def _a(x):
    return np.asarray(x, np.float64)


def unit_reads(report, kappa):
    own, data = report["own"], report["data"]
    e_own = _a(own["q1"]) - _a(own["qpi"])
    e_d = _a(data["q1"]) - _a(data["qpi"])                  # (n, 4): logged, s05, s10, s20
    u = _a(data["u"])
    pen = (_a(data["q1"]) - kappa * u) - _a(data["qpi"])
    out = {"J": float(report["J"]), "exposure_w": float(report["exposure_w"]),
           "e_own": float(e_own.mean()), "e_own_median": float(np.median(e_own)), "share_over_own": float(np.mean(e_own > 0)),
           "e_data": float(e_d[:, 0].mean()), "e_w": float(e_d[:, 1:].mean()),
           "e_w_by_slot": {s: float(e_d[:, k].mean()) for k, s in enumerate(data["slots"]) if k > 0},
           "share_over_w": float(np.mean(e_d[:, 1:] > 0)), "share_over_data": float(np.mean(e_d[:, 0] > 0)),
           "u_data": float(u[:, 0].mean()), "u_w": float(u[:, 1:].mean()), "u_own": float(_a(own["u"]).mean()),
           "p_data": float(pen[:, 0].mean()), "p_w": float(pen[:, 1:].mean()),
           "I1_ok": bool(report["instrument"]["I1_ok"]), "I2_ok": bool(report["instrument"]["I2_ok"])}
    out["finite"] = all(np.isfinite(v) for k, v in out.items() if isinstance(v, float)) and all(np.isfinite(v) for v in out["e_w_by_slot"].values())
    return out


def decide(native, bca, kappa):
    """native, bca: {seed_index: oe_evaluate report}. Returns the decision record."""
    n = {i: unit_reads(native[i], kappa) for i in SEEDS}
    b = {i: unit_reads(bca[i], kappa) for i in SEEDS}
    paired = {i: {"dJ": b[i]["J"] - n[i]["J"], "de_own": b[i]["e_own"] - n[i]["e_own"],
                  "dEXPO": b[i]["exposure_w"] - n[i]["exposure_w"], "de_w": b[i]["e_w"] - n[i]["e_w"]} for i in SEEDS}
    canonical = {i: {"native_e_w_minus_e_data": n[i]["e_w"] - n[i]["e_data"], "native_u_w_minus_u_data": n[i]["u_w"] - n[i]["u_data"],
                     "bca_e_w_minus_e_data": b[i]["e_w"] - b[i]["e_data"], "bca_p_w_minus_p_data": b[i]["p_w"] - b[i]["p_data"]}
                 for i in SEEDS}
    labels, reasons = [], []
    valid = all(r["I1_ok"] and r["I2_ok"] and r["finite"] for r in list(n.values()) + list(b.values()))
    if not valid:
        labels.append("INCONCLUSIVE")
        reasons.append("an instrument check (I1 identity, I2 own action) failed or a read is non-finite")
    if all(n[i]["exposure_w"] < REACH for i in SEEDS):
        labels.append("NOT REACHED")
        reasons.append(f"native W exposure below {REACH} in every seed: " + ", ".join(f"{n[i]['exposure_w']:.4f}" for i in SEEDS))
    dj, de = [paired[i]["dJ"] for i in SEEDS], [paired[i]["de_own"] for i in SEEDS]
    if sum(d <= -D_J for d in dj) >= 2:
        labels.append("HARMFUL")
        reasons.append("return falls by at least 2.0 in two or more seeds: " + ", ".join(f"{d:+.3f}" for d in dj))
    if all(d <= -D_E_OWN for d in de) and all(d >= -D_J for d in dj):
        labels.append("PROMISING")
        reasons.append("own-action critic error falls by at least 0.5 in every seed with no return loss of 2.0: de_own "
                       + ", ".join(f"{d:+.3f}" for d in de) + "; dJ " + ", ".join(f"{d:+.3f}" for d in dj))
    if all(abs(d) < D_E_OWN for d in de) and all(abs(d) < D_J for d in dj):
        labels.append("NO EFFECT")
        reasons.append("every paired difference inside the bars: de_own " + ", ".join(f"{d:+.3f}" for d in de)
                       + "; dJ " + ", ".join(f"{d:+.3f}" for d in dj))
    if not any(l in labels for l in ("HARMFUL", "PROMISING", "NO EFFECT")):
        labels.append("MIXED")
        reasons.append("de_own " + ", ".join(f"{d:+.3f}" for d in de) + "; dJ " + ", ".join(f"{d:+.3f}" for d in dj))
    headline = min(labels, key=ORDER.index)
    return {"reading": headline, "all_labels": labels, "reasons": reasons, "bars": {"d_e_own": D_E_OWN, "d_J": D_J, "reach": REACH},
            "kappa": kappa, "native": n, "bca": b, "paired": paired, "canonical": canonical}
