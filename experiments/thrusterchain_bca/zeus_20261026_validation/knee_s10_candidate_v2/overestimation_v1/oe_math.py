"""The frozen reads and readings of the overestimation comparison (OVERESTIMATION_RULE.md, amendment A2: the
honeypot world stall3 and the placebo world no-stall). Pure numpy on the saved `oe_evaluate` reports; no simulator,
no GPU."""
import numpy as np

D_E_OWN, D_J, REACH, D_EXPO = 0.5, 2.0, 0.01, 0.05
SEEDS = (0, 1, 2)
HONEYPOT, PLACEBO = "stall3", "nostall"
VALIDITY_ORDER = ("INCOMPLETE", "INCONCLUSIVE", "NOT REACHED")
HONEYPOT_ORDER = ("HARMFUL", "PROMISING", "NO EFFECT", "MIXED")
PLACEBO_ORDER = ("OVER-CORRECTION", "PLACEBO WITHOUT STAKE", "CALIBRATED", "PLACEBO MIXED")


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
           "share_under_own": float(np.mean(e_own < 0)),
           "e_data": float(e_d[:, 0].mean()), "e_w": float(e_d[:, 1:].mean()),
           "e_w_by_slot": {s: float(e_d[:, k].mean()) for k, s in enumerate(data["slots"]) if k > 0},
           "share_over_w": float(np.mean(e_d[:, 1:] > 0)), "share_over_data": float(np.mean(e_d[:, 0] > 0)),
           "u_data": float(u[:, 0].mean()), "u_w": float(u[:, 1:].mean()), "u_own": float(_a(own["u"]).mean()),
           "p_data": float(pen[:, 0].mean()), "p_w": float(pen[:, 1:].mean()),
           "I1_ok": bool(report["instrument"]["I1_ok"]), "I2_ok": bool(report["instrument"]["I2_ok"])}
    out["finite"] = all(np.isfinite(v) for k, v in out.items() if isinstance(v, float)) and all(np.isfinite(v) for v in out["e_w_by_slot"].values())
    return out


def _world(native, bca, kappa):
    n = {i: unit_reads(native[i], kappa) for i in SEEDS}
    b = {i: unit_reads(bca[i], kappa) for i in SEEDS}
    paired = {i: {"dJ": b[i]["J"] - n[i]["J"], "de_own": b[i]["e_own"] - n[i]["e_own"],
                  "dEXPO": b[i]["exposure_w"] - n[i]["exposure_w"], "de_w": b[i]["e_w"] - n[i]["e_w"]} for i in SEEDS}
    canonical = {i: {"native_e_w_minus_e_data": n[i]["e_w"] - n[i]["e_data"], "native_u_w_minus_u_data": n[i]["u_w"] - n[i]["u_data"],
                     "bca_e_w_minus_e_data": b[i]["e_w"] - b[i]["e_data"], "bca_p_w_minus_p_data": b[i]["p_w"] - b[i]["p_data"]}
                 for i in SEEDS}
    valid = all(r["I1_ok"] and r["I2_ok"] and r["finite"] for r in list(n.values()) + list(b.values()))
    return {"native": n, "bca": b, "paired": paired, "canonical": canonical, "valid": valid}


def _fmt(values):
    return ", ".join(f"{v:+.3f}" for v in values)


def honeypot_labels(w):
    dj, de = [w["paired"][i]["dJ"] for i in SEEDS], [w["paired"][i]["de_own"] for i in SEEDS]
    labels, reasons = [], []
    if sum(d <= -D_J for d in dj) >= 2:
        labels.append("HARMFUL")
        reasons.append("stall3: return falls by at least 2.0 in two or more seeds: dJ " + _fmt(dj))
    if all(d <= -D_E_OWN for d in de) and all(d >= -D_J for d in dj):
        labels.append("PROMISING")
        reasons.append("stall3: own-action critic error falls by at least 0.5 in every seed with no return loss of 2.0: de_own "
                       + _fmt(de) + "; dJ " + _fmt(dj))
    if all(abs(d) < D_E_OWN for d in de) and all(abs(d) < D_J for d in dj):
        labels.append("NO EFFECT")
        reasons.append("stall3: every paired difference inside the bars: de_own " + _fmt(de) + "; dJ " + _fmt(dj))
    if not labels:
        labels.append("MIXED")
        reasons.append("stall3: de_own " + _fmt(de) + "; dJ " + _fmt(dj))
    return labels, reasons


def placebo_labels(w):
    """The no-stall world, where W is harmless: a penalty that still keeps the actor out and costs return is
    over-correction (the underestimation failure); no return loss is calibrated."""
    dj, dx = [w["paired"][i]["dJ"] for i in SEEDS], [w["paired"][i]["dEXPO"] for i in SEEDS]
    expo = [w["native"][i]["exposure_w"] for i in SEEDS]
    labels, reasons = [], []
    if all(e < REACH for e in expo):
        labels.append("PLACEBO WITHOUT STAKE")
        reasons.append(f"no-stall: the native actor's W exposure is below {REACH} in every seed ({', '.join(f'{e:.4f}' for e in expo)}): "
                       "avoiding W costs nothing here, so the placebo cannot show over-correction")
    if sum(d <= -D_EXPO for d in dx) >= 2 and sum(d <= -D_J for d in dj) >= 2:
        labels.append("OVER-CORRECTION")
        reasons.append("no-stall: BCA leaves W (dEXPO " + _fmt(dx) + ") and loses return (dJ " + _fmt(dj) + ") in two or more seeds, where W is harmless")
    if all(d >= -D_J for d in dj):
        labels.append("CALIBRATED")
        reasons.append("no-stall: no return loss of 2.0 in any seed (dJ " + _fmt(dj) + "; dEXPO " + _fmt(dx) + ")")
    if not any(l in labels for l in ("OVER-CORRECTION", "CALIBRATED")):
        labels.append("PLACEBO MIXED")
        reasons.append("no-stall: dJ " + _fmt(dj) + "; dEXPO " + _fmt(dx))
    return labels, reasons


def decide(results, kappa):
    """results: {world: {"native": {seed: report}, "bca": {seed: report}}} for stall3 and nostall. Returns the
    decision record: validity labels first, then the honeypot and placebo readings as "honeypot / placebo"."""
    worlds = {w: _world(results[w]["native"], results[w]["bca"], kappa) for w in (HONEYPOT, PLACEBO)}
    labels, reasons = [], []
    if not all(worlds[w]["valid"] for w in worlds):
        labels.append("INCONCLUSIVE")
        reasons.append("an instrument check (I1 identity, I2 own action) failed or a read is non-finite")
    hp = worlds[HONEYPOT]
    if all(hp["native"][i]["exposure_w"] < REACH for i in SEEDS):
        labels.append("NOT REACHED")
        reasons.append(f"stall3: native W exposure below {REACH} in every seed: " + ", ".join(f"{hp['native'][i]['exposure_w']:.4f}" for i in SEEDS))
    h_labels, h_reasons = honeypot_labels(hp)
    p_labels, p_reasons = placebo_labels(worlds[PLACEBO])
    honeypot = min(h_labels, key=HONEYPOT_ORDER.index)
    placebo = min(p_labels, key=PLACEBO_ORDER.index)
    validity = min(labels, key=VALIDITY_ORDER.index) if labels else None
    reading = validity or f"{honeypot} / {placebo}"
    return {"reading": reading, "validity": validity, "honeypot": honeypot, "placebo": placebo,
            "all_labels": labels + h_labels + p_labels, "reasons": reasons + h_reasons + p_reasons,
            "bars": {"d_e_own": D_E_OWN, "d_J": D_J, "reach": REACH, "d_expo": D_EXPO}, "kappa": kappa,
            "worlds": {w: {k: v for k, v in worlds[w].items() if k != "valid"} for w in worlds},
            "paired": {w: worlds[w]["paired"] for w in worlds}, "canonical": {w: worlds[w]["canonical"] for w in worlds}}
