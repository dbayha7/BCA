"""Decision rules of the BCA dose screen v1 (PROTOCOL.md "Readouts" and "Decision"). Pure functions on reports."""
from pathlib import Path
import json

import numpy as np

KNEE = float(np.float32(0.8))
SEEDS, ARMS, WORLDS = (0, 1, 2), ("cutoff", "random"), ("stall3", "nostall")
ENTRY_DROP, PLACEMENT, J_GAIN, NOSTALL_COST = -0.10, -0.05, 10.0, -5.0      # EFFECT, in every seed
Q_RANK_GAIN = 0.20          # EFFECT through Q values: penalized-Q correct ranking minus raw-Q1 correct ranking, every seed
NO_EFFECT_ENTRY = -0.05                                                     # NO EFFECT: every seed above this
HARM_ENTRY, HARM_NOSTALL_J = 0.05, -20.0                                    # HARM: in at least 2 seeds
NAMES = {"nan": np.nan, "positive_infinity": np.inf, "negative_infinity": -np.inf}


def arr(values):
    return np.asarray([[NAMES.get(v, v) for v in row] if isinstance(row, list) else NAMES.get(row, row) for row in values],
                      np.float64)


def num(value):
    return float(NAMES.get(value, value)) if value is not None else float("nan")


def own(report, name):
    return report["runs"][name]["020000"]["own_world"]


def applied_reversal(audit):
    """Descriptive only (survivorship): among high-load rows where the BCA actor is still past the knee at 20k and
    native's push is outward, the share whose net push with the applied penalty is inward."""
    rows = audit["per_row"]
    a0 = arr(rows["actor_actions"])[:, 0]
    high = np.asarray(rows["high_load"], bool)
    g = {k: arr(v)[:, 0] for k, v in rows["action_loss_gradients"].items()}
    native = g["q"] + g["bc"]
    outward = high & (a0 < -KNEE) & (native > 0)
    inward = outward & (native + g["actual_penalty"] < 0)
    return {"rows": int(outward.sum()), "share": float(inward.sum() / outward.sum()) if outward.any() else None,
            "past_knee_rows": int((high & (a0 < -KNEE)).sum())}


def anchor_readouts(entry, kappa=None):
    """Q-value readouts at the 48 gated E2 anchors against exact Q^pi (the run's own clipped continuation).
    Contrasts are slot minus edge_knee: true C = dQ^pi, raw C = dQ1, penalized C = dQ1 - kappa dU (the value the BCA
    actor maximizes). KC = C - C_true. On the anchors where the truth says the past-knee slot is worse (C_true < 0),
    the share ranked correctly (C < 0) by raw Q1 and by the penalized Q."""
    a = entry["anchors_e2"]
    names = a["gated_slots"]
    q1, qpi = arr(a["q1"])[:48], arr(a["qpi_clip"])[:48]
    valid = np.asarray(a["valid"], bool)[:48]
    u = arr(a["U"])[:48] if "U" in a else None
    k = names.index("edge_knee")
    out = {}
    for slot in ("s05", "s10"):
        i = names.index(slot)
        c_true, c_raw = qpi[:, i] - qpi[:, k], q1[:, i] - q1[:, k]
        keep = valid & np.isfinite(c_true) & np.isfinite(c_raw)
        harmful = keep & (c_true < 0)
        out[f"median_kc_{slot}"] = float(np.median((c_raw - c_true)[keep])) if keep.any() else None
        out[f"harmful_anchors_{slot}"] = int(harmful.sum())
        out[f"raw_correct_share_{slot}"] = float(np.mean(c_raw[harmful] < 0)) if harmful.any() else None
        if u is not None and kappa is not None:
            c_adj = c_raw - float(kappa) * (u[:, i] - u[:, k])
            ok = keep & np.isfinite(c_adj)
            out[f"median_delta_u_{slot}"] = float(np.median((u[:, i] - u[:, k])[ok])) if ok.any() else None
            out[f"median_kc_penalized_{slot}"] = float(np.median((c_adj - c_true)[ok])) if ok.any() else None
            h = harmful & ok
            out[f"penalized_correct_share_{slot}"] = float(np.mean(c_adj[h] < 0)) if h.any() else None
    return out


def seed_readouts(bca, native, audits, i, arm, kappa=None):
    d = {}
    for world in WORLDS:
        name = f"s{i}_{arm}_{world}"
        b, n = own(bca, name), own(native, name)
        d[world] = {"entry_bca": num(b["e2_rates"]["r_deep_minus"]), "entry_native": num(n["e2_rates"]["r_deep_minus"]),
                    "J_bca": num(b["e2/pi"]["J"]), "J_native": num(n["e2/pi"]["J"]),
                    "applied_reversal_descriptive": applied_reversal(audits[name]) if name in audits else None,
                    "anchors_bca": anchor_readouts(bca["runs"][name]["020000"], kappa),
                    "anchors_native": anchor_readouts(native["runs"][name]["020000"])}
        d[world]["delta_entry"] = d[world]["entry_bca"] - d[world]["entry_native"]
        d[world]["delta_J"] = d[world]["J_bca"] - d[world]["J_native"]
    d["sel"] = d["stall3"]["delta_J"] - d["nostall"]["delta_J"]
    d["placement"] = d["stall3"]["delta_entry"] - d["nostall"]["delta_entry"]
    q = d["stall3"]["anchors_bca"]                                 # the penalty corrects the Q ranking, within the BCA run
    pen, raw = q.get("penalized_correct_share_s10"), q.get("raw_correct_share_s10")
    d["q_rank_gain_s10"] = (pen - raw) if pen is not None and raw is not None else float("nan")
    return d


def verdict(rows):
    """rows: {seed: seed_readouts} for the seeds that completed. Precedence: INCOMPLETE ARM, EFFECT, BEHAVIORAL EFFECT
    NOT THROUGH Q, NON-SELECTIVE REDUCTION, HARM, NO EFFECT, MIXED. Non-finite values never satisfy a bar."""
    if set(rows) != set(SEEDS):
        return "INCOMPLETE ARM"
    fin = lambda x: bool(np.isfinite(x))                                          # noqa: E731
    drop = all(fin(r["stall3"]["delta_entry"]) and r["stall3"]["delta_entry"] <= ENTRY_DROP for r in rows.values())
    selective = drop and all(fin(r["placement"]) and r["placement"] <= PLACEMENT
                             and fin(r["stall3"]["delta_J"]) and r["stall3"]["delta_J"] >= J_GAIN
                             and fin(r["nostall"]["delta_J"]) and r["nostall"]["delta_J"] >= NOSTALL_COST for r in rows.values())
    through_q = all(fin(r["q_rank_gain_s10"]) and r["q_rank_gain_s10"] >= Q_RANK_GAIN for r in rows.values())
    if selective and through_q:
        return "EFFECT"
    if selective:
        return "BEHAVIORAL EFFECT NOT THROUGH Q"
    if drop:
        return "NON-SELECTIVE REDUCTION"
    harm = sum((fin(r["stall3"]["delta_entry"]) and r["stall3"]["delta_entry"] >= HARM_ENTRY)
               or (fin(r["nostall"]["delta_J"]) and r["nostall"]["delta_J"] <= HARM_NOSTALL_J) for r in rows.values())
    if harm >= 2:
        return "HARM"
    if all(fin(r["stall3"]["delta_entry"]) and r["stall3"]["delta_entry"] > NO_EFFECT_ENTRY for r in rows.values()):
        return "NO EFFECT"
    return "MIXED"


def decide(bca_evals, native_evals, audits, kappa=None):
    """bca_evals: {seed: evaluate report} for the seeds whose four BCA branches completed; native_evals: {seed: report};
    audits: {run name: 20k gradient audit of its BCA branch}."""
    per = {}
    for arm in ARMS:
        rows = {i: seed_readouts(bca_evals[i], native_evals[i], audits, i, arm, kappa) for i in SEEDS if i in bca_evals}
        per[arm] = {"verdict": verdict(rows), "seeds": rows, "seeds_evaluated": sorted(rows)}
    return per


def load_audit(branch_dir):
    report = json.loads((Path(branch_dir) / "report.json").read_text(encoding="utf-8"))
    entry = report["gradient_audits"]["20000"]
    return json.loads((Path(branch_dir) / Path(entry["path"]).name).read_text(encoding="utf-8"))
