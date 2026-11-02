"""The pre-registered reads and readings of the confirmation (CONFIRMATION_RULE.md). Pure numpy on the saved
`cf_evaluate` reports; no simulator, no GPU."""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent / "overestimation_v1") not in sys.path:
    sys.path.insert(0, str(HERE.parent / "overestimation_v1"))
import oe_math as OM                                         # noqa: E402

SEEDS = (0, 1, 2)
DOSES = (21.0, 63.0, 189.0)
CONFIRMED_DOSE = 63.0
HONEYPOT, PLACEBO = "stall3", "nostall"
D_J, D_EXPO, REACH = 2.0, 0.05, 0.01
COST_MEAN, COST_WORST = -4.0, -8.0
C2_ORDER = ("PLACEBO WITHOUT STAKE", "CALIBRATED", "COST BOUNDED", "OVER-CORRECTION", "PLACEBO MIXED")


def _fmt(values):
    return ", ".join(f"{v:+.3f}" for v in values)


def world_reads(native, bca_by_dose):
    """native: {seed: report}; bca_by_dose: {kappa: {seed: report}}. Reads and paired differences per dose."""
    n = {i: OM.unit_reads(native[i], 0.0) for i in SEEDS}
    out = {"native": n, "bca": {}, "paired": {}, "valid": all(r["I1_ok"] and r["I2_ok"] and r["finite"] for r in n.values())}
    for kappa in DOSES:
        b = {i: OM.unit_reads(bca_by_dose[kappa][i], kappa) for i in SEEDS}
        out["bca"][kappa] = b
        out["paired"][kappa] = {i: {"dJ": b[i]["J"] - n[i]["J"], "de_own": b[i]["e_own"] - n[i]["e_own"],
                                    "dEXPO": b[i]["exposure_w"] - n[i]["exposure_w"], "de_w": b[i]["e_w"] - n[i]["e_w"]} for i in SEEDS}
        out["valid"] = out["valid"] and all(r["I1_ok"] and r["I2_ok"] and r["finite"] for r in b.values())
    return out


def c1(hp):
    dj = [hp["paired"][CONFIRMED_DOSE][i]["dJ"] for i in SEEDS]
    label = "CONFIRMED" if all(d >= D_J for d in dj) else "NOT CONFIRMED"
    return label, f"stall3 at kappa {CONFIRMED_DOSE:g}: dJ " + _fmt(dj) + f" against >= +{D_J} in every seed"


def c2(pl):
    dj = [pl["paired"][CONFIRMED_DOSE][i]["dJ"] for i in SEEDS]
    dx = [pl["paired"][CONFIRMED_DOSE][i]["dEXPO"] for i in SEEDS]
    expo = [pl["native"][i]["exposure_w"] for i in SEEDS]
    labels = []
    if all(e < REACH for e in expo):
        labels.append("PLACEBO WITHOUT STAKE")
    if all(d >= -D_J for d in dj):
        labels.append("CALIBRATED")
    elif float(np.mean(dj)) >= COST_MEAN and all(d >= COST_WORST for d in dj):
        labels.append("COST BOUNDED")
    elif sum(d <= -D_EXPO for d in dx) >= 2 and sum(d <= -D_J for d in dj) >= 2:
        labels.append("OVER-CORRECTION")
    if not labels:
        labels.append("PLACEBO MIXED")
    label = min(labels, key=C2_ORDER.index)
    return label, labels, (f"no-stall at kappa {CONFIRMED_DOSE:g}: dJ " + _fmt(dj) + "; dEXPO " + _fmt(dx)
                           + "; native exposure " + ", ".join(f"{e:.3f}" for e in expo))


def profile(worlds):
    """The dose profile: exposure, dJ, de_own by dose and world, and the two monotonicity flags."""
    table = {w: {f"{k:g}": {"exposure": [worlds[w]["bca"][k][i]["exposure_w"] for i in SEEDS],
                            "dJ": [worlds[w]["paired"][k][i]["dJ"] for i in SEEDS],
                            "de_own": [worlds[w]["paired"][k][i]["de_own"] for i in SEEDS]} for k in DOSES}
             for w in (HONEYPOT, PLACEBO)}
    for w in table:
        table[w]["native"] = {"exposure": [worlds[w]["native"][i]["exposure_w"] for i in SEEDS], "J": [worlds[w]["native"][i]["J"] for i in SEEDS]}
    expo_mono = all(worlds[w]["bca"][DOSES[k]][i]["exposure_w"] >= worlds[w]["bca"][DOSES[k + 1]][i]["exposure_w"]
                    for w in (HONEYPOT, PLACEBO) for i in SEEDS for k in range(len(DOSES) - 1))
    gain_mono = all(worlds[HONEYPOT]["paired"][DOSES[k]][i]["dJ"] <= worlds[HONEYPOT]["paired"][DOSES[k + 1]][i]["dJ"]
                    for i in SEEDS for k in range(len(DOSES) - 1))
    return {"table": table, "exposure_monotone": bool(expo_mono), "honeypot_gain_monotone": bool(gain_mono),
            "note": "a profile of this run; the best-looking dose is not a confirmed dose"}


def decide(results):
    """results: {world: {"native": {seed: report}, "bca": {kappa: {seed: report}}}}."""
    worlds = {w: world_reads(results[w]["native"], results[w]["bca"]) for w in (HONEYPOT, PLACEBO)}
    validity, reasons = [], []
    if not all(worlds[w]["valid"] for w in worlds):
        validity.append("INCONCLUSIVE")
        reasons.append("an instrument check (I1 identity, I2 own action) failed or a read is non-finite")
    if all(worlds[HONEYPOT]["native"][i]["exposure_w"] < REACH for i in SEEDS):
        validity.append("NOT REACHED")
        reasons.append("stall3: native W exposure below 0.01 in every seed")
    c1_label, c1_reason = c1(worlds[HONEYPOT])
    c2_label, c2_all, c2_reason = c2(worlds[PLACEBO])
    prof = profile(worlds)
    reading = validity[0] if validity else f"{c1_label} / {c2_label}"
    return {"reading": reading, "validity": validity[0] if validity else None, "C1": c1_label, "C2": c2_label,
            "all_labels": validity + [c1_label] + c2_all, "reasons": reasons + [c1_reason, c2_reason],
            "bars": {"d_J": D_J, "d_expo": D_EXPO, "reach": REACH, "cost_mean": COST_MEAN, "cost_worst": COST_WORST, "confirmed_dose": CONFIRMED_DOSE},
            "profile": prof, "paired": {w: {f"{k:g}": worlds[w]["paired"][k] for k in DOSES} for w in worlds},
            "worlds": {w: {"native": worlds[w]["native"], "bca": {f"{k:g}": worlds[w]["bca"][k] for k in DOSES}} for w in worlds}}
