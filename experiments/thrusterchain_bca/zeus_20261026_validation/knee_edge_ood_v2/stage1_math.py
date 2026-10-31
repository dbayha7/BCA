"""Pure decision rules of Stage 1, the native damage gate (protocol Part 5, Stage 1; approval A4). No model or file
access: every input is a per-anchor array or a per-run number computed by the evaluation worker.

Per seed, in the stall world, at E2 and 20k, on the 48 gated Stage 1 E2 anchors (t in {0, 25, 50}):
- common gated anchors: valid under both stall CUTOFF and stall RANDOM; R-valid / C-valid: common and predictor-valid
  (item 5) under RANDOM's / CUTOFF's own clipped continuation;
- V0(a)-(e) (INCONCLUSIVE), K1a, K1c, K2, K3a-c (FAIL), the companions and the readings for FAIL-consistent;
- medians are over finite lists; an even count averages the two middle values (numpy's median); in V0(d) an anchor
  that is not C-valid enters as minus infinity.
"""
import numpy as np

from core import require
from step04_math import ITEM5_TOL

K1_BAR, K1C_ABS, K1C_REL = 0.5, 0.5, 0.5
K2_BAR, K3_BAR = 0.25, 2.0
GAIN_BAR = 50.0
V0B = {"item5_share": 0.90, "rho_floor": 0.20, "kc_ratio": 0.4, "kc_offset": 0.10, "r_deep_max": 0.10}
V0C_FLOOR = 32
V0D = {"item5_share": 0.90, "item6_share": 0.90, "gap_offset": 0.10, "k1a_bar": 0.5, "k1c_bar": 0.65}
V0E_MAX = 0.005
CRITERIA = ("K1a", "K1c", "K2", "K3a", "K3b", "K3c")
READINGS = {
    1: "K1c fails: this Unifloral critic does not over-value past a dense edge, even past a kink proven by the identity, "
       "with the s10 gap shown reachable (V0(d)). BCA has nothing to repair here. The reading describes the measured critic, "
       "CUTOFF KC noise included.",
    2: "K1c passes and K1a fails: over-valuation at the knee's onset is not caused by withholding.",
    3: "K1 passes and K2 fails: the alpha 2.5 BC anchor contains the over-valuation. No K3 reading is given, because "
       "without entry there is nothing to harm.",
    4: "K2 passes and K3a fails: this contradicts L2. Audit first.",
    5: "K2 and K3a pass and K3b fails: the cost of entry is not larger than under the labelled control.",
    6: "K2, K3a and K3b pass and K3c fails: the harm does not come from the hidden knee.",
}
FAIL_EARLY_READING = "the host contained the extrapolation at 10k; 20k was not observed"


def median(values):
    values = np.asarray(values, np.float64)
    return float(np.median(values)) if values.size else None


def kc(q1, qpi, names, slot):
    """KC_slot = [Q1(slot) - Q1(-KNEE)] - [Q^pi(slot) - Q^pi(-KNEE)] per anchor (Part 0); q1, qpi: (anchors, slots)."""
    i, k = names.index(slot), names.index("edge_knee")
    q1, qpi = np.asarray(q1, np.float64), np.asarray(qpi, np.float64)
    return (q1[:, i] - q1[:, k]) - (qpi[:, i] - qpi[:, k])


def item5_ok(item5_error):
    """Per-anchor L2 item 5 under one continuation: every predictor within tolerance at s05, s10 and s20."""
    flags = None
    for name, tol in ITEM5_TOL.items():
        ok = np.asarray(item5_error[name], np.float64) <= tol
        flags = ok if flags is None else flags & ok
    return flags


def anchor_sets(valid_c, valid_r, item5_c, item5_r):
    common = np.asarray(valid_c, bool) & np.asarray(valid_r, bool)
    return {"common": common, "r_valid": common & np.asarray(item5_r, bool), "c_valid": common & np.asarray(item5_c, bool)}


def _at(values, mask):
    return np.asarray(values, np.float64)[np.asarray(mask, bool)]


def seed_rules(c, r, swap):
    """All Stage 1 rules for one seed.

    c, r: stall CUTOFF / stall RANDOM, each a dict with per-gated-anchor arrays `valid`, `item5_ok`, `rho_s05`,
    `rho_s10`, `rho_max_p10`, `kc_s05`, `kc_s10`, and per-run numbers `r_deep_rate` (E2 free steps, stall world),
    `gain_e1` (J_E1(20k) - J_E1(untrained)), `delta_clip` (stall world, E2) and `a123_past_rate` (E2 steps).
    swap: the stall CUTOFF policy in the no-stall world: `delta_clip`, `a123_past_rate`."""
    sets = anchor_sets(c["valid"], r["valid"], c["item5_ok"], r["item5_ok"])
    common, rv, cv = sets["common"], sets["r_valid"], sets["c_valid"]
    n = {k: int(v.sum()) for k, v in sets.items()}
    out = {"counts": n}

    # V0(a) adequacy at 20k, V0(c) floors, V0(e) a1..a3 entries.
    out["V0a"] = {"cutoff_gain": c["gain_e1"], "random_gain": r["gain_e1"],
                  "ok": bool(c["gain_e1"] >= GAIN_BAR and r["gain_e1"] >= GAIN_BAR)}
    out["V0c"] = {**n, "ok": bool(min(n.values()) >= V0C_FLOOR)}
    out["V0e"] = {"cutoff_stall": c["a123_past_rate"], "random_stall": r["a123_past_rate"],
                  "cutoff_nostall_world": swap["a123_past_rate"],
                  "ok": bool(max(c["a123_past_rate"], r["a123_past_rate"], swap["a123_past_rate"]) <= V0E_MAX)}

    # V0(b): RANDOM represents the knee under its own clipped continuation.
    share_r = float(np.mean(_at(r["item5_ok"], common))) if n["common"] else 0.0
    med_rho_r = median(_at(r["rho_s05"], rv))
    med_kc_r = median(_at(r["kc_s05"], rv))
    b_ok = (share_r >= V0B["item5_share"] and med_rho_r is not None and med_rho_r >= V0B["rho_floor"]
            and abs(med_kc_r) <= V0B["kc_ratio"] * (med_rho_r + V0B["kc_offset"]) and r["r_deep_rate"] <= V0B["r_deep_max"])
    out["V0b"] = {"item5_share_common": share_r, "median_rho_hat_R_s05": med_rho_r, "median_kc_s05_R": med_kc_r,
                  "kc_bound": None if med_rho_r is None else V0B["kc_ratio"] * (med_rho_r + V0B["kc_offset"]),
                  "r_deep_rate": r["r_deep_rate"], "ok": bool(b_ok)}

    # V0(d): CUTOFF's estimator validity, sign control and reachability (non-C-valid enters as minus infinity).
    share_c = float(np.mean(_at(c["item5_ok"], common))) if n["common"] else 0.0
    share6 = float(np.mean(_at(c["rho_max_p10"], common) < 0)) if n["common"] else 0.0
    gap05 = np.where(cv, np.asarray(c["rho_s05"], np.float64), -np.inf)[common]
    gap10 = np.where(cv, np.asarray(c["rho_s10"], np.float64), -np.inf)[common]
    kc_r_common = _at(r["kc_s05"], common)
    k1a_i = median(gap05 - V0D["gap_offset"] - kc_r_common)
    k1a_ii = median(gap05 - V0D["gap_offset"])
    k1c_reach = median(gap10)
    reach_k1a = k1a_i is not None and k1a_i >= V0D["k1a_bar"] and k1a_ii >= V0D["k1a_bar"]
    reach_k1c = k1c_reach is not None and k1c_reach >= V0D["k1c_bar"]
    out["V0d"] = {"item5_share_common": share_c, "item6_share_common": share6,
                  "k1a_reach_i": k1a_i, "k1a_reach_ii": k1a_ii, "k1c_reach_median_rho_C_s10": k1c_reach,
                  "minus_infinity_entries": int((common & ~cv).sum()),
                  "k1a_reachable": bool(reach_k1a), "k1c_reachable": bool(reach_k1c),
                  "ok": bool(share_c >= V0D["item5_share"] and share6 >= V0D["item6_share"] and reach_k1a and reach_k1c)}

    # Hypothesis criteria.
    diff = _at(c["kc_s05"], common) - kc_r_common
    k1a = median(diff)
    k1c_abs = median(_at(c["kc_s10"], common))
    k1c_rel_kc, k1c_rel_rho = median(_at(c["kc_s10"], cv)), median(_at(c["rho_s10"], cv))
    abs_ok = k1c_abs is not None and k1c_abs >= K1C_ABS
    rel_ok = k1c_rel_kc is not None and k1c_rel_rho is not None and k1c_rel_kc >= K1C_REL * k1c_rel_rho
    out["K1a"] = {"median_kc_s05_C_minus_R": k1a, "ok": bool(k1a is not None and k1a >= K1_BAR)}
    out["K1c"] = {"median_kc_s10_C_common": k1c_abs, "median_kc_s10_C_c_valid": k1c_rel_kc,
                  "median_rho_hat_C_s10_c_valid": k1c_rel_rho, "absolute_clause": bool(abs_ok), "relative_clause": bool(rel_ok),
                  "failed_clauses": [x for x, ok in (("absolute", abs_ok), ("relative", rel_ok)) if not ok],
                  "ok": bool(abs_ok and rel_ok)}
    out["K2"] = {"cutoff_r_deep_rate": c["r_deep_rate"], "ok": bool(c["r_deep_rate"] >= K2_BAR)}
    out["K3a"] = {"delta_clip_C_stall": c["delta_clip"], "ok": bool(c["delta_clip"] >= K3_BAR)}
    out["K3b"] = {"difference": c["delta_clip"] - r["delta_clip"], "ok": bool(c["delta_clip"] - r["delta_clip"] >= K3_BAR)}
    out["K3c"] = {"difference": c["delta_clip"] - swap["delta_clip"], "ok": bool(c["delta_clip"] - swap["delta_clip"] >= K3_BAR)}

    # Companions (reported) and reading 2's conditions.
    zeroed = _at(c["kc_s05"], common) - np.minimum(kc_r_common, 0.0)
    k1a_zeroed = median(zeroed)
    high = diff >= k1a if k1a is not None else np.zeros(0, bool)
    low_gap = (gap05 - V0D["gap_offset"] < 0.5)[high] if k1a is not None else np.zeros(0, bool)
    med_kc_r_rv = median(_at(r["kc_s05"], rv))
    out["companions"] = {"low_gap_share": float(np.mean(low_gap)) if low_gap.size else None,
                         "median_kc_s05_C_common": median(_at(c["kc_s05"], common)),
                         "k1a_with_positive_random_kc_zeroed": k1a_zeroed,
                         "which_k1c_clause_failed": out["K1c"]["failed_clauses"]}
    out["reading2"] = {"a": bool(med_kc_r_rv is not None and med_rho_r is not None and med_kc_r_rv <= 0.1 * med_rho_r),
                       "b": bool(k1a_zeroed is not None and k1a_zeroed < K1_BAR)}
    out["validity_ok"] = all(out[k]["ok"] for k in ("V0a", "V0b", "V0c", "V0d", "V0e"))
    return out


def reading(per_seed):
    """The reading for FAIL-consistent: the first criterion that fails in every seed governs, if every earlier one
    passes in every seed (K1a's and K1c's order is reading 1 = K1c, reading 2 = K1a)."""
    seeds = list(per_seed.values())
    passes = {k: all(s[k]["ok"] for s in seeds) for k in CRITERIA}
    fails = {k: all(not s[k]["ok"] for s in seeds) for k in CRITERIA}
    if fails["K1c"]:
        return 1
    if fails["K1a"]:
        if passes["K1c"] and all(s["reading2"]["a"] and s["reading2"]["b"] for s in seeds):
            return 2
        return None
    order = [("K2", 3, ("K1a", "K1c")), ("K3a", 4, ("K1a", "K1c", "K2")), ("K3b", 5, ("K1a", "K1c", "K2", "K3a")),
             ("K3c", 6, ("K1a", "K1c", "K2", "K3a", "K3b"))]
    for criterion, number, before in order:
        if fails[criterion]:
            return number if all(passes[b] for b in before) else None
    return None


def label(*, v1_reasons, r0, per_seed):
    """The Stage 1 label in 5.0 precedence. r0: {"adequacy_ok", "fail_early"}; per_seed: seed -> seed_rules output
    (empty when R0 stopped the run). Returns (label, reasons, reading number or None)."""
    if v1_reasons:
        return "INCOMPLETE", list(v1_reasons), None
    if not r0["adequacy_ok"]:
        return "INCONCLUSIVE", ["R0 adequacy: J_E1(10k) - J_E1(untrained) < +50 in some seed (stall CUTOFF or RANDOM)"], None
    if r0["fail_early"]:
        return "FAIL-early", ["R0 reach: stall CUTOFF R_deep- rate < 0.05 in every seed at 10k"], None
    require(len(per_seed) == 3, "Stage 1 decisions need all three seeds")
    invalid = [f"seed {s}: {k} fails" for s, d in per_seed.items() for k in ("V0a", "V0b", "V0c", "V0d", "V0e") if not d[k]["ok"]]
    if invalid:
        return "INCONCLUSIVE", invalid, None
    failed = [f"seed {s}: {k} fails" for s, d in per_seed.items() for k in CRITERIA if not d[k]["ok"]]
    if not failed:
        return "PASS", [], None
    consistent = [k for k in CRITERIA if all(not d[k]["ok"] for d in per_seed.values())]
    if consistent:
        return "FAIL-consistent", failed, reading(per_seed)
    return "FAIL-mixed", failed, None


def secondary(j):
    """Pre-registered secondary outcomes from E2 20k returns. j: {(arm, world, eval_world): J}; A(pi) = J_nostall - J_stall."""
    a_c = j[("cutoff", "stall3", "nostall")] - j[("cutoff", "stall3", "stall3")]
    a_r = j[("random", "stall3", "nostall")] - j[("random", "stall3", "stall3")]
    stall = j[("random", "stall3", "stall3")] - j[("cutoff", "stall3", "stall3")]
    nostall = j[("random", "nostall", "nostall")] - j[("cutoff", "nostall", "nostall")]
    return {"J_random_minus_cutoff_stall": stall, "A_cutoff_minus_A_random": a_c - a_r, "triple_difference_J": stall - nostall,
            "predicted_directions": {"J_random_minus_cutoff_stall": "> 0", "A_cutoff_minus_A_random": "> 0"}}
