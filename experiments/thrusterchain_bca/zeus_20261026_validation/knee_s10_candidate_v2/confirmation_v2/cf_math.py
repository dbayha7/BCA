"""The pre-registered reads and readings of confirmation v2 (CONFIRMATION_RULE.md). Pure numpy on the saved
`cf_evaluate` reports; no simulator, no GPU.

Carried from confirmation v1: the reads (`oe_math.unit_reads`), C1, C2 and their precedence, the validity order.
Changed for v2, all written before any v2 number: one dose (kappa 189, the dose being confirmed), five seeds, the
OVER-CORRECTION count as a majority (3 of 5), the I1 tolerance 1e-5 relative applied to each report's recorded
`I1_max_rel` (the sealed evaluator still writes its own 1e-6 flag), and a description with no bar in place of v1's
dose profile."""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent / "overestimation_v1") not in sys.path:
    sys.path.insert(0, str(HERE.parent / "overestimation_v1"))
import oe_math as OM                                         # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
DOSES = (189.0,)
CONFIRMED_DOSE = 189.0
HONEYPOT, PLACEBO = "stall3", "nostall"
D_J, D_EXPO, REACH = 2.0, 0.05, 0.01
COST_MEAN, COST_WORST = -4.0, -8.0
MAJORITY = 3                                                 # of five seeds (v1: 2 of 3)
I1_TOL = 1e-5                                                # relative (v1: 1e-6)
C2_ORDER = ("PLACEBO WITHOUT STAKE", "CALIBRATED", "COST BOUNDED", "OVER-CORRECTION", "PLACEBO MIXED")


def _fmt(values):
    return ", ".join(f"{v:+.3f}" for v in values)


def unit_reads(report, kappa):
    """oe_math's reads, with the I1 check at this rule's tolerance and two reported reads of v2 (no bar)."""
    r = OM.unit_reads(report, kappa)
    r["I1_max_rel"] = float(report["instrument"]["I1_max_rel"])
    r["I1_ok"] = bool(np.isfinite(r["I1_max_rel"]) and r["I1_max_rel"] <= I1_TOL)
    deep = ((report.get("rates") or {}).get("by_band") or {}).get("deep") or {}
    r["exposure_w_deep"] = None if deep.get("w_any_depth") is None else float(deep["w_any_depth"])
    return r


def world_reads(native, bca_by_dose):
    """native: {seed: report}; bca_by_dose: {kappa: {seed: report}}. Reads and paired differences."""
    n = {i: unit_reads(native[i], 0.0) for i in SEEDS}
    out = {"native": n, "bca": {}, "paired": {}, "valid": all(r["I1_ok"] and r["I2_ok"] and r["finite"] for r in n.values())}
    for kappa in DOSES:
        b = {i: unit_reads(bca_by_dose[kappa][i], kappa) for i in SEEDS}
        out["bca"][kappa] = b
        out["paired"][kappa] = {}
        for i in SEEDS:
            rb = np.asarray(bca_by_dose[kappa][i]["returns"], np.float64)
            rn = np.asarray(native[i]["returns"], np.float64)
            out["paired"][kappa][i] = {"dJ": b[i]["J"] - n[i]["J"], "de_own": b[i]["e_own"] - n[i]["e_own"],
                                       "dEXPO": b[i]["exposure_w"] - n[i]["exposure_w"], "de_w": b[i]["e_w"] - n[i]["e_w"],
                                       "share_starts_lower": float(np.mean(rb < rn)), "share_starts_higher": float(np.mean(rb > rn))}
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
    elif sum(d <= -D_EXPO for d in dx) >= MAJORITY and sum(d <= -D_J for d in dj) >= MAJORITY:
        labels.append("OVER-CORRECTION")
    if not labels:
        labels.append("PLACEBO MIXED")
    label = min(labels, key=C2_ORDER.index)
    return label, labels, (f"no-stall at kappa {CONFIRMED_DOSE:g}: dJ " + _fmt(dj) + "; dEXPO " + _fmt(dx)
                           + "; native exposure " + ", ".join(f"{e:.3f}" for e in expo))


def description(worlds):
    """Reported with no bar: per world and seed, the paired reads, W exposure overall and at deep load, the share of
    the 128 starts on which BCA's return is lower, and the native returns."""
    k = CONFIRMED_DOSE
    table = {}
    for w in (HONEYPOT, PLACEBO):
        nat, bca, par = worlds[w]["native"], worlds[w]["bca"][k], worlds[w]["paired"][k]
        table[w] = {"native_J": [nat[i]["J"] for i in SEEDS], "native_exposure": [nat[i]["exposure_w"] for i in SEEDS],
                    "native_exposure_deep": [nat[i]["exposure_w_deep"] for i in SEEDS],
                    "bca_exposure": [bca[i]["exposure_w"] for i in SEEDS], "bca_exposure_deep": [bca[i]["exposure_w_deep"] for i in SEEDS],
                    **{key: [par[i][key] for i in SEEDS] for key in ("dJ", "dEXPO", "de_own", "share_starts_lower", "share_starts_higher")}}
    return {"table": table, "note": "reported with no bar; the readings are C1 and C2 alone"}


def decide(results):
    """results: {world: {"native": {seed: report}, "bca": {kappa: {seed: report}}}}."""
    worlds = {w: world_reads(results[w]["native"], results[w]["bca"]) for w in (HONEYPOT, PLACEBO)}
    validity, reasons = [], []
    if not all(worlds[w]["valid"] for w in worlds):
        validity.append("INCONCLUSIVE")
        reasons.append(f"an instrument check (I1 identity at {I1_TOL:g} relative, I2 own action) failed or a read is non-finite")
    if all(worlds[HONEYPOT]["native"][i]["exposure_w"] < REACH for i in SEEDS):
        validity.append("NOT REACHED")
        reasons.append("stall3: native W exposure below 0.01 in every seed")
    c1_label, c1_reason = c1(worlds[HONEYPOT])
    c2_label, c2_all, c2_reason = c2(worlds[PLACEBO])
    reading = validity[0] if validity else f"{c1_label} / {c2_label}"
    return {"reading": reading, "validity": validity[0] if validity else None, "C1": c1_label, "C2": c2_label,
            "all_labels": validity + [c1_label] + c2_all, "reasons": reasons + [c1_reason, c2_reason],
            "bars": {"d_J": D_J, "d_expo": D_EXPO, "reach": REACH, "cost_mean": COST_MEAN, "cost_worst": COST_WORST,
                     "majority": MAJORITY, "i1_tol": I1_TOL, "confirmed_dose": CONFIRMED_DOSE},
            "description": description(worlds), "paired": {w: {f"{k:g}": worlds[w]["paired"][k] for k in DOSES} for w in worlds},
            "worlds": {w: {"native": worlds[w]["native"], "bca": {f"{k:g}": worlds[w]["bca"][k] for k in DOSES}} for w in worlds}}
