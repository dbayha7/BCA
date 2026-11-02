"""Pure decision rules of Stage 2, the BCA comparison (protocol Part 5, Stage 2; approval A5). No model or file access.

- Detection precheck (at the 10k fork, shadow auxiliary, before any branch): per-anchor AUROC of BCA's width U over
  all OOD-vs-in-support slot pairs at that anchor (ties count 0.5), averaged over the 48 gated `d12_e2_anchors`.
  D1: AUROC_CUTOFF({s10, s15, s20} vs near-edge) >= 0.70 in stall CUTOFF; D2: AUROC_CUTOFF(s05 vs near-edge) -
  AUROC_RANDOM(s05 vs near-edge) >= +0.15; both in every seed, else DETECTION-FAIL.
- V0-S2: every Stage 2 branch gains J_E1(20k) - J_E1(untrained) >= +50 on the Stage 2 E1 bank, every seed.
- R1 (G > 0 every seed, three-seed mean >= +3.0), R2 (BCA >= distance in stall CUTOFF every seed), R3 (mean H >= -1.5);
  PROMISING needs all three. Labels: INCOMPLETE > INCONCLUSIVE > DETECTION-FAIL > NOT-PROMISING > PROMISING.
- Reported: damage reproduced, recovery (with its guard), the repair screens (with guards), the paired t-interval,
  the nine replicate-minus-Stage-1-native differences.
"""
import math

import numpy as np

from core import require

NEAR_EDGE = ("edge_knee", "edge_775", "edge_750", "edge_725", "edge_700")
D1_OOD, D2_OOD = ("s10", "s15", "s20"), ("s05",)
D1_BAR, D2_BAR = 0.70, 0.15
GAIN_BAR = 50.0
R1_MEAN, R3_MEAN = 3.0, -1.5
RECOVERY_GUARD, RECOVERY_STRONG = 3.0, 0.50
REPAIR = {"entry_delta": -0.15, "delta_clip_ratio": 0.5, "delta_clip_guard": 2.0, "kc_ratio": 0.5, "kc_guard": 0.5}
T_975_DF2 = 0.95 / math.sqrt(2 * 0.975 * 0.025)   # Student t, 97.5% quantile, 2 df: exact closed form (2p - 1)/sqrt(2p(1 - p))


def pairwise_auroc(scores_ood, scores_in):
    """Per anchor: P(OOD score > in-support score) + 0.5 P(tie) over all pairs at that anchor; then the mean over
    anchors. Inputs (anchors, k) and (anchors, m). Returns (mean AUROC, per-anchor values)."""
    ood, ins = np.asarray(scores_ood, np.float64), np.asarray(scores_in, np.float64)
    require(ood.ndim == ins.ndim == 2 and len(ood) == len(ins) and len(ood) > 0, "Per-anchor score arrays required")
    require(bool(np.all(np.isfinite(ood))) and bool(np.all(np.isfinite(ins))), "Scores must be finite")
    greater = (ood[:, :, None] > ins[:, None, :]).mean(axis=(1, 2))
    ties = (ood[:, :, None] == ins[:, None, :]).mean(axis=(1, 2))
    per = greater + 0.5 * ties
    return float(per.mean()), per


def fpr_at_95(scores_ood, scores_in):
    """Share of in-support actions flagged at the largest threshold that still catches >= 95% of OOD actions
    (flagged = score >= threshold), pooled over anchors."""
    ood = np.sort(np.asarray(scores_ood, np.float64).ravel())
    ins = np.asarray(scores_in, np.float64).ravel()
    threshold = ood[int(math.floor(0.05 * len(ood)))]
    require(np.mean(ood >= threshold) >= 0.95, "Threshold does not catch 95% of OOD actions")
    return float(np.mean(ins >= threshold)), float(threshold)


def detection(cutoff, random):
    """D1 and D2 for one seed. cutoff/random: {slot name: U (anchors,)} at the gated anchors (a1..a3 = 0)."""
    stack = lambda d, names: np.stack([np.asarray(d[n], np.float64) for n in names], axis=1)
    d1, _ = pairwise_auroc(stack(cutoff, D1_OOD), stack(cutoff, NEAR_EDGE))
    c05, _ = pairwise_auroc(stack(cutoff, D2_OOD), stack(cutoff, NEAR_EDGE))
    r05, _ = pairwise_auroc(stack(random, D2_OOD), stack(random, NEAR_EDGE))
    return {"D1_auroc_cutoff": d1, "D1": bool(d1 >= D1_BAR), "D2_auroc_cutoff_s05": c05, "D2_auroc_random_s05": r05,
            "D2_difference": c05 - r05, "D2": bool(c05 - r05 >= D2_BAR)}


def detection_reported(cutoff, cutoff_score):
    """Reported detection quantities for stall CUTOFF: FPR@95, AUROC for W+ (p10), AUROC of Q1 - 3U (an OOD action
    counts as detected when its score is lower, so the detector is -(Q1 - 3U))."""
    stack = lambda d, names: np.stack([np.asarray(d[n], np.float64) for n in names], axis=1)
    fpr, thr = fpr_at_95(stack(cutoff, D1_OOD), stack(cutoff, NEAR_EDGE))
    wplus, _ = pairwise_auroc(stack(cutoff, ("p10",)), stack(cutoff, NEAR_EDGE))
    score, _ = pairwise_auroc(-stack(cutoff_score, D1_OOD), -stack(cutoff_score, NEAR_EDGE))
    return {"fpr_at_95": fpr, "fpr_threshold": thr, "auroc_w_plus_p10": wplus, "auroc_minus_score_q1_minus_3u": score}


def t_interval(values):
    v = np.asarray(values, np.float64)
    require(len(v) == 3, "Three seeds")
    half = T_975_DF2 * float(v.std(ddof=1)) / math.sqrt(3)
    return {"mean": float(v.mean()), "half_width_95": half, "low": float(v.mean()) - half, "high": float(v.mean()) + half}


def screens(j):
    """R1, R2, R3 from E2 returns at 20k. j[seed][(cell, branch)] with cells 'stall_cutoff', 'stall_random',
    'nostall_cutoff' (each in its own world) and branches 'bca', 'rep', 'distance'."""
    seeds = sorted(j)
    g = {s: (j[s][("stall_cutoff", "bca")] - j[s][("stall_cutoff", "rep")])
         - (j[s][("stall_random", "bca")] - j[s][("stall_random", "rep")]) for s in seeds}
    r2 = {s: j[s][("stall_cutoff", "bca")] - j[s][("stall_cutoff", "distance")] for s in seeds}
    h = {s: (j[s][("nostall_cutoff", "bca")] - j[s][("nostall_cutoff", "rep")])
         - (j[s][("stall_random", "bca")] - j[s][("stall_random", "rep")]) for s in seeds}
    g_mean, h_mean = float(np.mean(list(g.values()))), float(np.mean(list(h.values())))
    out = {"G": g, "G_mean": g_mean, "R1_sign": all(v > 0 for v in g.values()), "R1_mean": g_mean >= R1_MEAN,
           "R2_difference": r2, "R2": all(v >= 0 for v in r2.values()), "H": h, "H_mean": h_mean, "R3": h_mean >= R3_MEAN,
           "G_t_interval": t_interval(list(g.values()))}
    out["R1"] = out["R1_sign"] and out["R1_mean"]
    return out


def recovery(j):
    """Reported only when its guard holds: J(rep, R) - J(rep, C) > 0 every seed, mean >= +3.0."""
    gap = [j[s][("stall_random", "rep")] - j[s][("stall_cutoff", "rep")] for s in sorted(j)]
    if not (all(x > 0 for x in gap) and np.mean(gap) >= RECOVERY_GUARD):
        return {"computed": False, "reason": "not computed (guard)", "gap": gap}
    gain = [j[s][("stall_cutoff", "bca")] - j[s][("stall_cutoff", "rep")] for s in sorted(j)]
    value = float(np.mean(gain) / np.mean(gap))
    return {"computed": True, "recovery": value, "strong": value >= RECOVERY_STRONG, "gap": gap}


def repair(seed):
    """Repair screens in stall CUTOFF for one seed, each with its guard. seed: entry rates, delta-clips and the KC_s10
    arrays (Q1 - 3U for BCA, Q1 for the replicate) with the anchors valid under both continuations."""
    out = {"entry_delta": seed["bca_entry"] - seed["rep_entry"]}
    out["entry"] = out["entry_delta"] <= REPAIR["entry_delta"]
    if seed["rep_delta_clip"] >= REPAIR["delta_clip_guard"]:
        out["delta_clip"] = seed["bca_delta_clip"] <= REPAIR["delta_clip_ratio"] * seed["rep_delta_clip"]
    else:
        out["delta_clip"] = "not computed (guard)"
    both = np.asarray(seed["valid_both"], bool)
    rep_kc = float(np.median(np.asarray(seed["rep_kc_s10"], np.float64)[both])) if both.any() else None
    bca_kc = float(np.median(np.asarray(seed["bca_kc_s10_score"], np.float64)[both])) if both.any() else None
    out.update(anchors_valid_both=int(both.sum()), rep_median_kc_s10=rep_kc, bca_median_kc_s10_q1_minus_3u=bca_kc)
    if rep_kc is not None and rep_kc >= REPAIR["kc_guard"]:
        out["kc"] = bca_kc <= REPAIR["kc_ratio"] * rep_kc
    else:
        out["kc"] = "not computed (guard)"
    return out


def repair_all_seeds(by_seed):
    """Each repair screen across seeds: True only if it holds in every seed; 'not computed (guard)' if its guard
    failed in any seed."""
    out = {}
    for key in ("entry", "delta_clip", "kc"):
        values = [by_seed[s][key] for s in sorted(by_seed)]
        out[key] = "not computed (guard)" if any(isinstance(v, str) for v in values) else bool(all(values))
    return out


def label(*, v1_reasons, detection_by_seed, gains, screen, damage_by_seed):
    """The Stage 2 label in precedence. gains: {name: [three gains]} for every Stage 2 branch. Returns (label, reasons)."""
    if v1_reasons:
        return "INCOMPLETE", list(v1_reasons)
    low = [f"{name} seed {s}: J_E1 gain {g:.1f} < +50" for name, values in gains.items() for s, g in enumerate(values) if g < GAIN_BAR]
    if low:                                       # only reachable after the branches, i.e. after detection passed
        return "INCONCLUSIVE", low
    fails = [f"seed {s}: {k} fails" for s, d in detection_by_seed.items() for k in ("D1", "D2") if not d[k]]
    if fails:
        return "DETECTION-FAIL", fails + ["BCA's U does not detect this region (a finding about detection)"]
    if screen["R1"] and screen["R2"] and screen["R3"]:
        missing = [s for s, ok in damage_by_seed.items() if not ok]
        return ("PROMISING" + (f" (damage not reproduced in seed {', '.join(map(str, missing))})" if missing else "")), []
    reasons = [r for r, ok in (("R1 sign", screen["R1_sign"]), ("R1 mean", screen["R1_mean"]), ("R2", screen["R2"]),
                               ("R3 mean", screen["R3"])) if not ok]
    consistent = (all(v <= 0 for v in screen["G"].values()) or all(v < 0 for v in screen["R2_difference"].values())
                  or (not screen["R1_mean"] and all(v < R1_MEAN for v in screen["G"].values()))
                  or (not screen["R3"] and all(v < R3_MEAN for v in screen["H"].values())))
    return ("NOT-PROMISING-consistent" if consistent else "NOT-PROMISING-mixed"), reasons
