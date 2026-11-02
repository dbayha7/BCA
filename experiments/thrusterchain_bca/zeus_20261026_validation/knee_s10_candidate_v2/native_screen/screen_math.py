"""The frozen reads and readings of the native screen (SCREEN_RULE.md, from REVISION_r4.md section 5). Pure functions
on the evaluation reports; no model, simulator or file access. Every read is reported per seed whether or not it
decides. "every seed" means all three; "mixed" means the seeds disagree on a deciding read.

KC_slot = [Q1(slot) - Q1(-KNEE)] - [Q^pi(slot) - Q^pi(-KNEE)] per anchor, against the exact value under the arm's own
clipped continuation (stage1_math.kc). rho_own = Q^pi_nostall(s10) - Q^pi_stall(s10) under that continuation, exact.
The penalized contrast of the BCA run is dQ1 - kappa dU (the value its actor maximizes).
"""
import numpy as np

import stage1_math as SM

SEEDS, ARMS, WORLDS = (0, 1, 2), ("cutoff", "random"), ("stall3", "nostall")
TOL, C = 0.15, 0.3
GAIN_BAR, VALID_FLOOR, A123_MAX = 50.0, 32, 0.005
V0F = {"rho_floor": 0.30, "kc_ratio": 0.3, "kc_offset": 0.15}
PLACEBO = 0.30
O1_ABS, O1_REL, O2_BAR = 0.5, 0.5, 0.5
R1_BAR, R1_TWIN_MAX, R2_BAR = 0.25, 0.10, 2.0
B1_BAR, B2_BAR, B3_BAR, B3S_BAR = -0.10, 0.25, 2.0, 2.0
U1_J_MIN, U1_RANK_MAX = -2.0, 0.25
NAMES = {"nan": np.nan, "positive_infinity": np.inf, "negative_infinity": -np.inf}
READING_ORDER = ("INCOMPLETE", "INCONCLUSIVE", "STOP: not reached", "NO-GO: nothing to repair",
                 "NO-GO: no effect at this dose and budget", "NO-GO: over-correction", "MIXED", "GO")


def arr(values):
    return np.asarray([[NAMES.get(v, v) for v in row] if isinstance(row, list) else NAMES.get(row, row) for row in values],
                      np.float64)


def num(value):
    return float(NAMES.get(value, value)) if value is not None else float("nan")


def med(values):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def name(i, arm, world):
    return f"s{i}_{arm}_{world}"


def entry(report, i, arm, world):
    return report["runs"][name(i, arm, world)]["020000"]


def anchors(e):
    a = e["anchors_e2"]
    names = list(a["gated_slots"])
    out = {"names": names, "q1": arr(a["q1"]), "qpi": arr(a["qpi_clip"]), "valid": np.asarray(a["valid"], bool)}
    if "qpi_nostall" in a:
        ns = list(a["nostall_slots"])
        q_ns = arr(a["qpi_nostall"])
        out["rho_own"] = q_ns[:, ns.index("s10")] - out["qpi"][:, names.index("s10")]
    if "U" in a:
        out["U"] = arr(a["U"])
    return out


def kc(x, slot):
    return SM.kc(x["q1"], x["qpi"], x["names"], slot)


def l1_two_sided(kc_s10, rho, valid):
    """Per anchor -tol <= KC <= c rho + tol on valid anchors with rho > 0; share and the median check."""
    ok = valid & np.isfinite(kc_s10) & np.isfinite(rho) & (rho > 0)
    flags = ok & (kc_s10 >= -TOL) & (kc_s10 <= C * rho + TOL)
    n = int(ok.sum())
    m_kc, m_rho = med(kc_s10[ok]), med(rho[ok])
    return {"n_valid_positive_rho": n, "share": float(flags.sum() / n) if n else float("nan"),
            "median_kc": m_kc, "median_rho": m_rho,
            "median_ok": bool(n and -TOL <= m_kc <= C * m_rho + TOL)}


def ranking(x, kappa):
    """On anchors where the truth says s10 is worse than the knee (C_true < 0): the share ranked correctly (C < 0) by
    raw Q1 and by the penalized value; on anchors where s10 is truly better (C_true > 0): the share ranked wrongly
    (C < 0) by each (under-ranking). Needs U."""
    names, k, i = x["names"], x["names"].index("edge_knee"), x["names"].index("s10")
    c_true, c_raw = x["qpi"][:, i] - x["qpi"][:, k], x["q1"][:, i] - x["q1"][:, k]
    keep = x["valid"] & np.isfinite(c_true) & np.isfinite(c_raw)
    out = {"harmful_anchors": int((keep & (c_true < 0)).sum()), "benign_anchors": int((keep & (c_true > 0)).sum())}
    harmful, benign = keep & (c_true < 0), keep & (c_true > 0)
    out["raw_correct_share"] = float(np.mean(c_raw[harmful] < 0)) if harmful.any() else float("nan")
    out["raw_under_share"] = float(np.mean(c_raw[benign] < 0)) if benign.any() else float("nan")
    if "U" in x:
        c_pen = c_raw - kappa * (x["U"][:, i] - x["U"][:, k])
        out["penalized_correct_share"] = float(np.mean(c_pen[harmful] < 0)) if harmful.any() else float("nan")
        out["penalized_under_share"] = float(np.mean(c_pen[benign] < 0)) if benign.any() else float("nan")
        out["q_rank_gain_s10"] = out["penalized_correct_share"] - out["raw_correct_share"]
        out["under_rank_change_s10"] = out["penalized_under_share"] - out["raw_under_share"]
        out["median_kc_penalized_s10"] = med((c_pen - c_true)[keep])
    return out


def seed_reads(native, bca, i, kappa, g1):
    """Every read of one seed index from its native evaluation (four runs plus BC-only) and its BCA evaluation."""
    c, r = entry(native, i, "cutoff", "stall3"), entry(native, i, "random", "stall3")
    nc, nr = entry(native, i, "cutoff", "nostall"), entry(native, i, "random", "nostall")
    xc, xr, xnc, xnr = anchors(c), anchors(r), anchors(nc), anchors(nr)
    common = xc["valid"] & xr["valid"]
    kc_c, kc_r = kc(xc, "s10"), kc(xr, "s10")
    kc_nc, kc_nr = kc(xnc, "s10"), kc(xnr, "s10")
    rho = xc["rho_own"]
    runs = native["runs"]
    gains = {arm: entry(native, i, arm, "stall3")["own_world"]["e1/pi"]["J"] - runs[name(i, arm, "stall3")]["J_E1_untrained"]
             for arm in ARMS}
    out = {"seed": i}
    # validity
    out["V0a"] = {"gains": gains, "ok": bool(all(g >= GAIN_BAR for g in gains.values()))}
    out["V0c"] = {"common_valid": int(common.sum()), "ok": bool(common.sum() >= VALID_FLOOR)}
    a123 = {"cutoff": c["own_world"]["e2_rates"]["a123_past_knee"], "random": r["own_world"]["e2_rates"]["a123_past_knee"],
            "cutoff_nostall_world": c["nostall_world"]["e2_rates"]["a123_past_knee"]}
    out["V0e"] = {**a123, "ok": bool(max(a123.values()) <= A123_MAX)}
    m_kc_r, m_rho = med(kc_r[common]), med(rho[common])
    out["V0f"] = {"median_kc_random_s10": m_kc_r, "median_rho_own": m_rho,
                  "bound": V0F["kc_ratio"] * (m_rho + V0F["kc_offset"]),
                  "ok": bool(np.isfinite(m_kc_r) and np.isfinite(m_rho) and m_rho >= V0F["rho_floor"]
                             and m_kc_r <= V0F["kc_ratio"] * (m_rho + V0F["kc_offset"]))}
    l1 = l1_two_sided(kc_r, rho, common)
    q_star = None if g1 is None else g1.get("q_star")
    out["L1n"] = {**l1, "q_star": q_star,
                  "ok": bool(q_star is not None and np.isfinite(l1["share"]) and l1["share"] >= q_star and l1["median_ok"])}
    k700 = np.abs(kc(xc, "edge_700") - kc(xr, "edge_700"))
    band = None if g1 is None else g1.get("resolution_band")
    out["BAL"] = {"median_abs_kc_diff_edge_700": med(k700[common]), "band": band,
                  "ok": bool(band is not None and med(k700[common]) <= band)}
    both_ns = xnc["valid"] & xnr["valid"]
    out["placebo_contrast_nostall"] = med((kc_nc - kc_nr)[both_ns])
    # over-valuation
    m_kc_c = med(kc_c[common])
    out["O1"] = {"median_kc_cutoff_s10": m_kc_c, "median_rho_own": m_rho,
                 "ok": bool(np.isfinite(m_kc_c) and m_kc_c >= O1_ABS and m_kc_c >= O1_REL * m_rho)}
    contrast = med((kc_c - kc_r)[common])
    out["O2"] = {"median_contrast_s10_stall": contrast, "median_contrast_s10_nostall": out["placebo_contrast_nostall"],
                 "ok": bool(np.isfinite(contrast) and contrast >= O2_BAR and contrast - out["placebo_contrast_nostall"] > 0)}
    out["U0"] = {f"{arm}_{w}_share_kc_le_minus_tol": float(np.mean(v[m] <= -TOL)) if m.any() else float("nan")
                 for arm, w, v, m in (("cutoff", "stall3", kc_c, common), ("random", "stall3", kc_r, common),
                                      ("cutoff", "nostall", kc_nc, both_ns), ("random", "nostall", kc_nr, both_ns))}
    out["U0"]["kc_quantiles_cutoff_stall3"] = {str(q): float(np.quantile(kc_c[common], q)) for q in (0.1, 0.5, 0.9)} if common.any() else None
    out["U0"]["twin_under_capture_share"] = out["U0"]["random_stall3_share_kc_le_minus_tol"]
    # reach and harm
    bc = native.get("bc_only") or {}
    entry_c = c["own_world"]["e2_rates"]["r_deep_minus"]
    bc_rate = bc.get("e2_rates", {}).get("r_deep_minus", float("nan"))
    out["R1"] = {"cutoff_entry": entry_c, "bc_only_entry": bc_rate, "excess": entry_c - num(bc_rate),
                 "random_entry": r["own_world"]["e2_rates"]["r_deep_minus"],
                 "random_excess_over_bc_only": r["own_world"]["e2_rates"]["r_deep_minus"] - num(bc_rate),
                 "ok": bool(np.isfinite(num(bc_rate)) and entry_c - num(bc_rate) >= R1_BAR),
                 "twin_ok": bool(np.isfinite(num(bc_rate)) and r["own_world"]["e2_rates"]["r_deep_minus"] - num(bc_rate) <= R1_TWIN_MAX)}
    k3a = c["own_world"]["e2/delta_clip"]
    k3b = k3a - r["own_world"]["e2/delta_clip"]
    k3c = k3a - c["nostall_world"]["e2/delta_clip"]
    out["R2"] = {"K3a": k3a, "K3b": k3b, "K3c": k3c, "ok": bool(min(k3a, k3b, k3c) >= R2_BAR)}
    # BCA, paired with the native branch of the same fork
    if bca is not None:
        bc_c, bc_nc = entry(bca, i, "cutoff", "stall3"), entry(bca, i, "cutoff", "nostall")
        d_entry = bc_c["own_world"]["e2_rates"]["r_deep_minus"] - entry_c
        out["B1"] = {"delta_entry_stall": d_entry, "bca_entry": bc_c["own_world"]["e2_rates"]["r_deep_minus"], "ok": bool(d_entry <= B1_BAR)}
        rk = ranking(anchors(bc_c), kappa)
        out["B2"] = {**rk, "ok": bool(np.isfinite(rk.get("q_rank_gain_s10", np.nan)) and rk["q_rank_gain_s10"] >= B2_BAR)}
        d_j = bc_c["own_world"]["e2/pi"]["J"] - c["own_world"]["e2/pi"]["J"]
        d_j_ns = bc_nc["own_world"]["e2/pi"]["J"] - nc["own_world"]["e2/pi"]["J"]
        out["B3"] = {"delta_J_stall": d_j, "ok": bool(d_j >= B3_BAR)}
        out["B3s"] = {"selectivity": d_j - d_j_ns, "ok": bool(d_j - d_j_ns >= B3S_BAR)}
        out["U1"] = {"delta_J_nostall": d_j_ns, "under_rank_change_s10": rk.get("under_rank_change_s10", float("nan")),
                     "bca_entry_nostall": bc_nc["own_world"]["e2_rates"]["r_deep_minus"],
                     "ok": bool(d_j_ns >= U1_J_MIN and np.isfinite(rk.get("under_rank_change_s10", np.nan))
                                and rk["under_rank_change_s10"] <= U1_RANK_MAX)}
        xr_b = anchors(entry(bca, i, "random", "stall3"))
        out["bca_twin_reported"] = {"median_kc_s10": med(kc(xr_b, "s10")[xr_b["valid"]]),
                                    "entry": entry(bca, i, "random", "stall3")["own_world"]["e2_rates"]["r_deep_minus"]}
    return out


def consensus(reads, key):
    """'holds' when every seed passes the read, 'fails' when none does, else 'mixed'."""
    oks = [bool(r[key]["ok"]) for r in reads.values() if key in r]
    if not oks:
        return "missing"
    return "holds" if all(oks) else ("fails" if not any(oks) else "mixed")


def placebo(reads):
    pooled = med([r["placebo_contrast_nostall"] for r in reads.values()])
    return {"pooled_median_contrast_nostall": pooled, "ok": bool(np.isfinite(pooled) and abs(pooled) <= PLACEBO)}


def decide(native, bca, kappa, g1):
    """native, bca: {seed: evaluation report} (bca may be empty when the screen stopped before the BCA units);
    g1: the g1 unit's decision dict (q_star, resolution_band, headline) or None. Returns the reading and every read."""
    reads = {i: seed_reads(native[i], bca.get(i) if bca else None, i, kappa, g1) for i in SEEDS if i in native}
    plc = placebo(reads)
    validity = {k: consensus(reads, k) for k in ("V0a", "V0c", "V0e", "V0f", "L1n", "BAL")}
    validity["PLC"] = "holds" if plc["ok"] else "fails"
    g1_ok = g1 is not None and g1.get("headline") == "PASS"
    reasons = []
    if not g1_ok:
        reasons.append("G1: " + ("not run" if g1 is None else g1.get("headline", "missing")) + " (representation)")
    reasons += [f"{k}: {v}" for k, v in validity.items() if v != "holds"]
    out = {"kappa": kappa, "seeds_evaluated": sorted(reads), "reads": reads, "validity": validity, "placebo": plc,
           "consensus": {k: consensus(reads, k) for k in ("O1", "O2", "R1", "R2", "B1", "B2", "B3", "B3s", "U1")}}
    if len(reads) < 3:
        return {**out, "reading": "INCOMPLETE", "reasons": ["fewer than three seeds evaluated"]}
    if reasons:
        return {**out, "reading": "INCONCLUSIVE", "reasons": reasons}
    cs = out["consensus"]
    if cs["R1"] == "fails":
        return {**out, "reading": "STOP: not reached", "reasons": ["R1 fails in every seed: the actor does not enter W on this bed"]}
    if cs["R1"] == "mixed":
        return {**out, "reading": "MIXED", "reasons": ["R1 differs between seeds"]}
    if cs["O1"] == "fails":
        return {**out, "reading": "NO-GO: nothing to repair",
                "reasons": ["O1 fails in every seed: the native critic does not over-value the contrast here"]}
    if cs["O1"] == "mixed":
        return {**out, "reading": "MIXED", "reasons": ["O1 differs between seeds"]}
    if cs["B1"] == "missing":
        return {**out, "reading": "INCOMPLETE", "reasons": ["the BCA evaluations are missing"]}
    b_states = {k: cs[k] for k in ("B1", "B2", "B3", "B3s")}
    if any(v == "fails" for v in b_states.values()):
        return {**out, "reading": "NO-GO: no effect at this dose and budget",
                "reasons": [f"{k} fails in every seed" for k, v in b_states.items() if v == "fails"]}
    if any(v == "mixed" for v in b_states.values()):
        return {**out, "reading": "MIXED", "reasons": [f"{k} differs between seeds" for k, v in b_states.items() if v == "mixed"]}
    if cs["U1"] != "holds":
        return {**out, "reading": "NO-GO: over-correction", "reasons": [f"U1 {cs['U1']}: BCA costs return or ranking where caution is unneeded"]}
    return {**out, "reading": "GO", "reasons": []}


def r1_consensus(native):
    """After the native evaluations: R1's consensus alone (the supervisor's fail-fast read before any BCA unit)."""
    reads = {i: seed_reads(native[i], None, i, 0.0, None) for i in SEEDS if i in native}
    return consensus(reads, "R1"), {i: r["R1"] for i, r in reads.items()}
