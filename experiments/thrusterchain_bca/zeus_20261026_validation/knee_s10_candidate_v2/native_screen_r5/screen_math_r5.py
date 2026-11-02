"""The frozen reads and readings of the native screen under r5 (SCREEN_RULE_r5.md; REVISION_r5.md section 4). Pure
functions on the evaluation reports; no model, simulator or file access.

KC_slot = [Q1(slot) - Q1(-KNEE)] - [Q^pi(slot) - Q^pi(-KNEE)] per anchor, against the exact value under the arm's own
clipped continuation (stage1_math.kc). rho_own = Q^pi_nostall(s10) - Q^pi_stall(s10) under that continuation, exact.
The penalized contrast of the BCA run is dQ1 - kappa dU, the value its actor maximizes.

Pooling (r5 D5): anchor-level reads concatenate the common valid anchors of the three seeds (seed as stratum); rate
reads average over seeds. Unanimity is kept for B1 only. There is no MIXED reading.
"""
import numpy as np

import stage1_math as SM

SEEDS, ARMS, WORLDS = (0, 1, 2), ("cutoff", "random"), ("stall3", "nostall")
TOL, C = 0.15, 0.3
GAIN_BAR, VALID_FLOOR, A123_MAX, PLACEBO = 50.0, 32, 0.005, 0.30
V0F = {"rho_floor": 0.30, "kc_ratio": 0.3, "kc_offset": 0.15}
R1_BAR, R2_BAR = 0.25, 2.0
O1_ABS, O1_REL, O2_BAR = 0.5, 0.5, 0.5
B1_BAR, B2_BAR, NULL_MARGIN = -0.10, 0.25, 0.05
U1_RANK_MAX, U1_PLACEMENT_MAX = 0.25, 0.05
NULL_SHUFFLES, NULL_SEED = 2000, 20261101
NAMES = {"nan": np.nan, "positive_infinity": np.inf, "negative_infinity": -np.inf}
READINGS = ("INCOMPLETE", "INCONCLUSIVE (instrument)", "INCONCLUSIVE", "STOP: not reached", "STOP: no harm",
            "NO-GO: nothing to repair", "NO-GO: no consistent effect at this dose and budget",
            "NO-GO: strength without placement", "NO-GO: over-correction", "GO")


def arr(values):
    return np.asarray([[NAMES.get(v, v) for v in row] if isinstance(row, list) else NAMES.get(row, row) for row in values],
                      np.float64)


def num(value):
    return float(NAMES.get(value, value)) if value is not None else float("nan")


def med(values):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def fin(x):
    return bool(np.isfinite(x))


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
        out["rho_own"] = arr(a["qpi_nostall"])[:, ns.index("s10")] - out["qpi"][:, names.index("s10")]
    if "U" in a:
        out["U"] = arr(a["U"])
    return out


def kc(x, slot):
    return SM.kc(x["q1"], x["qpi"], x["names"], slot)


def contrasts(x):
    """Per anchor at s10 against edge_knee: the true contrast, the raw Q1 contrast, dU when U is present, and the
    rows that are valid and finite."""
    k, i = x["names"].index("edge_knee"), x["names"].index("s10")
    c_true, c_raw = x["qpi"][:, i] - x["qpi"][:, k], x["q1"][:, i] - x["q1"][:, k]
    keep = x["valid"] & np.isfinite(c_true) & np.isfinite(c_raw)
    du = (x["U"][:, i] - x["U"][:, k]) if "U" in x else None
    if du is not None:
        keep = keep & np.isfinite(du)
    return {"c_true": c_true, "c_raw": c_raw, "du": du, "keep": keep}


def ranking_gain(c_true, c_raw, pen, keep):
    """On harmful anchors (truth: s10 worse) the share ranked correctly (contrast < 0) by the penalized value minus
    by raw Q1; on benign anchors (truth: s10 better) the share ranked wrongly (contrast < 0) by each; net = gain
    minus the benign under-ranking change."""
    harmful, benign = keep & (c_true < 0), keep & (c_true > 0)
    pen_c = c_raw - pen
    raw_ok = float(np.mean(c_raw[harmful] < 0)) if harmful.any() else float("nan")
    pen_ok = float(np.mean(pen_c[harmful] < 0)) if harmful.any() else float("nan")
    raw_under = float(np.mean(c_raw[benign] < 0)) if benign.any() else 0.0
    pen_under = float(np.mean(pen_c[benign] < 0)) if benign.any() else 0.0
    gain = pen_ok - raw_ok
    return {"harmful": int(harmful.sum()), "benign": int(benign.sum()), "raw_correct_share": raw_ok,
            "penalized_correct_share": pen_ok, "gain": gain, "raw_under_share": raw_under,
            "penalized_under_share": pen_under, "under_change": pen_under - raw_under, "net": gain - (pen_under - raw_under)}


def b2_with_nulls(per_seed, kappa, shuffles=NULL_SHUFFLES, seed=NULL_SEED):
    """per_seed: {i: contrasts(...) with du}. The pooled B2 read and its two matched nulls (r5 D4), plus the
    sign-flipped U (reported)."""
    order = sorted(per_seed)
    c_true = np.concatenate([per_seed[i]["c_true"] for i in order])
    c_raw = np.concatenate([per_seed[i]["c_raw"] for i in order])
    keep = np.concatenate([per_seed[i]["keep"] for i in order])
    du = np.concatenate([per_seed[i]["du"] for i in order])
    real = ranking_gain(c_true, c_raw, kappa * du, keep)
    rng = np.random.default_rng(seed)
    nets, gains = [], []
    for _ in range(shuffles):
        parts = []
        for i in order:
            d, k = per_seed[i]["du"].copy(), per_seed[i]["keep"]
            d[k] = rng.permutation(d[k])                   # within the seed's valid anchors
            parts.append(d)
        g = ranking_gain(c_true, c_raw, kappa * np.concatenate(parts), keep)
        nets.append(g["net"])
        gains.append(g["gain"])
    const = np.concatenate([np.full(len(per_seed[i]["du"]), kappa * float(np.median(np.abs(per_seed[i]["du"][per_seed[i]["keep"]]))))
                            for i in order])
    constant = ranking_gain(c_true, c_raw, const, keep)
    flipped = ranking_gain(c_true, c_raw, -kappa * du, keep)
    shuffled = {"net_median": float(np.median(nets)), "gain_median": float(np.median(gains)),
                "net_quantiles": {str(q): float(np.quantile(nets, q)) for q in (0.1, 0.5, 0.9)}, "shuffles": int(shuffles), "seed": int(seed)}
    beats_shuffled = fin(real["net"]) and real["net"] >= shuffled["net_median"] + NULL_MARGIN
    beats_constant = fin(real["net"]) and fin(constant["net"]) and real["net"] >= constant["net"] + NULL_MARGIN
    return {"real": real, "shuffled_null": shuffled, "constant_null": constant, "sign_flipped": flipped,
            "gain_ok": fin(real["gain"]) and real["gain"] >= B2_BAR, "beats_shuffled": bool(beats_shuffled),
            "beats_constant": bool(beats_constant), "margin": NULL_MARGIN}


def seed_reads(native, bca, i, kappa):
    """Every per-seed read of one seed index, and the per-anchor arrays the pooled reads need."""
    c, r = entry(native, i, "cutoff", "stall3"), entry(native, i, "random", "stall3")
    nc, nr = entry(native, i, "cutoff", "nostall"), entry(native, i, "random", "nostall")
    xc, xr, xnc, xnr = anchors(c), anchors(r), anchors(nc), anchors(nr)
    common = xc["valid"] & xr["valid"]
    both_ns = xnc["valid"] & xnr["valid"]
    kc_c, kc_r, kc_nc, kc_nr = kc(xc, "s10"), kc(xr, "s10"), kc(xnc, "s10"), kc(xnr, "s10")
    rho = xc["rho_own"]
    runs = native["runs"]
    gains = {arm: entry(native, i, arm, "stall3")["own_world"]["e1/pi"]["J"] - runs[name(i, arm, "stall3")]["J_E1_untrained"] for arm in ARMS}
    a123 = {"cutoff": c["own_world"]["e2_rates"]["a123_past_knee"], "random": r["own_world"]["e2_rates"]["a123_past_knee"],
            "cutoff_nostall_world": c["nostall_world"]["e2_rates"]["a123_past_knee"]}
    bc_rate = num((native.get("bc_only") or {}).get("e2_rates", {}).get("r_deep_minus"))
    entry_c, entry_r = c["own_world"]["e2_rates"]["r_deep_minus"], r["own_world"]["e2_rates"]["r_deep_minus"]
    m_kc_r, m_rho = med(kc_r[common]), med(rho[common])
    k700 = np.abs(kc(xc, "edge_700") - kc(xr, "edge_700"))
    out = {"seed": i,
           "V0a": {"gains": gains, "ok": bool(all(fin(g) and g >= GAIN_BAR for g in gains.values()))},
           "V0c": {"common_valid": int(common.sum()), "ok": bool(common.sum() >= VALID_FLOOR)},
           "V0e": {**a123, "ok": bool(max(a123.values()) <= A123_MAX)},
           "V0f_reported": {"median_kc_random_s10": m_kc_r, "median_rho_own": m_rho, "bound": V0F["kc_ratio"] * (m_rho + V0F["kc_offset"]),
                            "holds": bool(fin(m_kc_r) and fin(m_rho) and m_rho >= V0F["rho_floor"] and m_kc_r <= V0F["kc_ratio"] * (m_rho + V0F["kc_offset"]))},
           "L1n_reported": l1_two_sided(kc_r, rho, common),
           "BAL_reported": {"median_abs_kc_diff_edge_700": med(k700[common])},
           "O1_seed": {"median_kc_cutoff_s10": med(kc_c[common]), "median_rho_own": m_rho},
           "O2_seed": {"median_contrast_s10_stall": med((kc_c - kc_r)[common]), "median_contrast_s10_nostall": med((kc_nc - kc_nr)[both_ns])},
           "U0": {"cutoff_stall3_share_kc_le_minus_tol": float(np.mean(kc_c[common] <= -TOL)) if common.any() else float("nan"),
                  "random_stall3_share_kc_le_minus_tol": float(np.mean(kc_r[common] <= -TOL)) if common.any() else float("nan"),
                  "kc_quantiles_cutoff_stall3": {str(q): float(np.quantile(kc_c[common], q)) for q in (0.1, 0.5, 0.9)} if common.any() else None},
           "R1_seed": {"cutoff_entry": entry_c, "bc_only_entry": bc_rate, "excess": entry_c - bc_rate,
                       "random_entry": entry_r, "random_excess_over_bc_only": entry_r - bc_rate},
           "R2_seed": {"K3a": c["own_world"]["e2/delta_clip"], "K3b": c["own_world"]["e2/delta_clip"] - r["own_world"]["e2/delta_clip"],
                       "K3c": c["own_world"]["e2/delta_clip"] - c["nostall_world"]["e2/delta_clip"]}}
    pools = {"kc_c": kc_c[common], "kc_r": kc_r[common], "rho": rho[common], "placebo": (kc_nc - kc_nr)[both_ns],
             "r1_excess": entry_c - bc_rate, "k3a": float(c["own_world"]["e2/delta_clip"])}
    if bca is not None:
        bc_c, bc_nc = entry(bca, i, "cutoff", "stall3"), entry(bca, i, "cutoff", "nostall")
        d_entry = bc_c["own_world"]["e2_rates"]["r_deep_minus"] - entry_c
        d_entry_ns = bc_nc["own_world"]["e2_rates"]["r_deep_minus"] - nc["own_world"]["e2_rates"]["r_deep_minus"]
        out["B1"] = {"delta_entry_stall": d_entry, "bca_entry": bc_c["own_world"]["e2_rates"]["r_deep_minus"], "ok": bool(fin(d_entry) and d_entry <= B1_BAR)}
        out["placement_seed"] = {"delta_entry_nostall": d_entry_ns, "placement": d_entry - d_entry_ns}
        out["returns_reported"] = {"checkpoint": "020000",
                                   "delta_J_stall": bc_c["own_world"]["e2/pi"]["J"] - c["own_world"]["e2/pi"]["J"],
                                   "delta_J_nostall": bc_nc["own_world"]["e2/pi"]["J"] - nc["own_world"]["e2/pi"]["J"]}
        out["returns_reported"]["selectivity"] = out["returns_reported"]["delta_J_stall"] - out["returns_reported"]["delta_J_nostall"]
        xb = anchors(bc_c)
        xr_b = anchors(entry(bca, i, "random", "stall3"))
        out["bca_twin_reported"] = {"median_kc_s10": med(kc(xr_b, "s10")[xr_b["valid"]]),
                                    "entry": entry(bca, i, "random", "stall3")["own_world"]["e2_rates"]["r_deep_minus"]}
        pools["bca"] = contrasts(xb)
        pools["placement"] = d_entry - d_entry_ns
    return out, pools


def l1_two_sided(kc_s10, rho, valid):
    ok = valid & np.isfinite(kc_s10) & np.isfinite(rho) & (rho > 0)
    flags = ok & (kc_s10 >= -TOL) & (kc_s10 <= C * rho + TOL)
    n = int(ok.sum())
    m_kc, m_rho = med(kc_s10[ok]), med(rho[ok])
    return {"n_valid_positive_rho": n, "share": float(flags.sum() / n) if n else float("nan"), "median_kc": m_kc,
            "median_rho": m_rho, "median_ok": bool(n and -TOL <= m_kc <= C * m_rho + TOL)}


def pooled_bed_reads(pools):
    kc_c = np.concatenate([p["kc_c"] for p in pools.values()])
    kc_r = np.concatenate([p["kc_r"] for p in pools.values()])
    rho = np.concatenate([p["rho"] for p in pools.values()])
    placebo = np.concatenate([p["placebo"] for p in pools.values()])
    r1 = float(np.mean([p["r1_excess"] for p in pools.values()]))
    k3a = float(np.mean([p["k3a"] for p in pools.values()]))
    m_kc_c, m_rho, m_plc = med(kc_c), med(rho), med(placebo)
    contrast = med(kc_c - kc_r)
    o2_ns = m_plc
    return {"PLC": {"pooled_median_contrast_nostall": m_plc, "ok": bool(fin(m_plc) and abs(m_plc) <= PLACEBO)},
            "R1": {"pooled_mean_excess": r1, "ok": bool(fin(r1) and r1 >= R1_BAR)},
            "R2": {"pooled_mean_K3a": k3a, "ok": bool(fin(k3a) and k3a >= R2_BAR)},
            "O1": {"pooled_median_kc_cutoff_s10": m_kc_c, "pooled_median_rho_own": m_rho, "anchors": int(kc_c.size),
                   "ok": bool(fin(m_kc_c) and fin(m_rho) and m_kc_c >= O1_ABS and m_kc_c >= O1_REL * m_rho)},
            "O2_reported": {"pooled_median_contrast_s10_stall": contrast, "pooled_median_contrast_s10_nostall": o2_ns,
                            "holds": bool(fin(contrast) and contrast >= O2_BAR and contrast > o2_ns)}}


def pre_bca(native):
    """The stage-1 readings available before any BCA unit: validity, then R1, R2, O1 in order. Returns (reading or
    None when the BCA units may run, the reads)."""
    reads, pools = {}, {}
    for i in SEEDS:
        if i in native:
            reads[i], pools[i] = seed_reads(native[i], None, i, 0.0)
    if len(reads) < 3:
        return "INCOMPLETE", {"reasons": ["fewer than three seeds evaluated"], "reads": reads}
    bed = pooled_bed_reads(pools)
    misses = [k for k in ("V0a", "V0c", "V0e") if not all(reads[i][k]["ok"] for i in SEEDS)]
    if not bed["PLC"]["ok"]:
        misses.append("PLC")
    out = {"reads": reads, "bed": bed, "reasons": []}
    if misses:
        out["reasons"] = [f"{k} misses" for k in misses]
        return "INCONCLUSIVE", out
    if not bed["R1"]["ok"]:
        out["reasons"] = [f"R1: pooled entry excess {bed['R1']['pooled_mean_excess']:.3f} < {R1_BAR}: the actor does not enter W on this bed"]
        return "STOP: not reached", out
    if not bed["R2"]["ok"]:
        out["reasons"] = [f"R2: pooled K3a {bed['R2']['pooled_mean_K3a']:.2f} < {R2_BAR}: entering past the knee costs no return here"]
        return "STOP: no harm", out
    if not bed["O1"]["ok"]:
        out["reasons"] = [f"O1: pooled median KC_C(s10) {bed['O1']['pooled_median_kc_cutoff_s10']:.3f} against {O1_ABS} and half of rho "
                          f"{bed['O1']['pooled_median_rho_own']:.3f}: the native critic does not over-value the contrast here"]
        return "NO-GO: nothing to repair", out
    return None, out


def decide(native, bca, kappa, g1=None, i0=None):
    """native, bca: {seed: evaluation report}; g1: the g1 unit's decision dict or None (reported); i0: the i0 unit's
    report or None. Returns the reading and every read."""
    out = {"kappa": kappa, "g1_reported": g1, "i0": i0}
    if i0 is not None and not i0.get("ok", False):
        return {**out, "reading": "INCONCLUSIVE (instrument)", "reasons": ["I0: the instrument check did not hold"]}
    reading, pre = pre_bca(native)
    out.update(reads=pre.get("reads"), bed=pre.get("bed"))
    if reading is not None:
        return {**out, "reading": reading, "reasons": pre["reasons"]}
    if not bca or any(i not in bca for i in SEEDS):
        return {**out, "reading": "INCOMPLETE", "reasons": ["the BCA evaluations are missing"]}
    reads, pools = {}, {}
    for i in SEEDS:
        reads[i], pools[i] = seed_reads(native[i], bca[i], i, kappa)
    out["reads"] = reads
    b1 = {i: reads[i]["B1"] for i in SEEDS}
    b1_ok = all(v["ok"] for v in b1.values())
    b2 = b2_with_nulls({i: pools[i]["bca"] for i in SEEDS}, kappa)
    placement = float(np.mean([pools[i]["placement"] for i in SEEDS]))
    u1 = {"pooled_under_change": b2["real"]["under_change"], "pooled_placement": placement,
          "ok": bool(fin(b2["real"]["under_change"]) and b2["real"]["under_change"] <= U1_RANK_MAX and fin(placement) and placement <= U1_PLACEMENT_MAX)}
    out.update(B1={"by_seed": b1, "ok": b1_ok}, B2=b2, U1=u1)
    if not b1_ok or not b2["gain_ok"]:
        reasons = ([f"B1 misses in seed {i}" for i in SEEDS if not b1[i]["ok"]]
                   + ([f"B2 pooled gain {b2['real']['gain']:.3f} < {B2_BAR}"] if not b2["gain_ok"] else []))
        return {**out, "reading": "NO-GO: no consistent effect at this dose and budget", "reasons": reasons}
    if not (b2["beats_shuffled"] and b2["beats_constant"]):
        return {**out, "reading": "NO-GO: strength without placement",
                "reasons": [f"B2 net {b2['real']['net']:.3f} against shuffled-U {b2['shuffled_null']['net_median']:.3f} and "
                            f"constant {b2['constant_null']['net']:.3f} (+{NULL_MARGIN}): even caution of the same size does as well"]}
    if not u1["ok"]:
        return {**out, "reading": "NO-GO: over-correction",
                "reasons": [f"U1: under-ranking change {u1['pooled_under_change']:.3f} (bar {U1_RANK_MAX}), placement {placement:.3f} (bar {U1_PLACEMENT_MAX})"]}
    return {**out, "reading": "GO", "reasons": []}
