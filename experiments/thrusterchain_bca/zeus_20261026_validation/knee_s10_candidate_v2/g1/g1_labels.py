"""Exact labels and exact anchor values for G1 (G1_RULE.md). Numpy and the raw float64 loop only; no JAX.

Importers must set up the path and the widened recorder first (import g2_formal_v2 before this module), as g1_run
and the tests do.
"""
import hashlib

import numpy as np

import masks as M
import pi_ref
import probes as P
import qpi
import stage1_eval as E
from core import require
from knee_region import S9, SLOTS as SLOT_VALUES

SLOTS = list(E.GATED_SLOTS)
HORIZON = 100


def fitting_rows_with_states(arrays):
    """masks.fitting_rows plus the exact float64 state before every row, in the same row order."""
    rows = M.fitting_rows(arrays)
    states = np.asarray(arrays["states"])[rows["episodes"], :HORIZON].reshape(-1, 13)
    rewards = np.asarray(arrays["rewards"])[rows["episodes"]].reshape(-1)
    require(states.dtype == np.float64 and len(states) == len(rows["obs"]), "States and rows disagree")
    require(np.array_equal(states.astype(np.float32), rows["obs"][:, :13]), "Row observations are not the float32 states")
    rows.update(states=states, rewards=rewards)
    return rows


def labels(rows, world, budget=None):
    """Exact Q^{pi_ref}(s_t, a_t) in `world` for every fitting row: (values (n,), receipt)."""
    times = np.asarray(rows["times"], np.int64)
    first = np.asarray(rows["actions"], np.float32)[:, None, :]
    steps = int(np.sum(HORIZON - times))
    if budget is not None:
        budget.charge(f"labels_{world}", steps)
    values, calls, receipt = qpi.raw_values_from_states(rows["states"], times, first, pi_ref.act, world, clip_all=True)
    values = values[:, 0]
    require(bool(np.all(np.isfinite(values))), "Non-finite label")
    last = times == HORIZON - 1
    known = np.max(np.abs(values[last] - rows["rewards"][last]) / np.maximum(1.0, np.abs(rows["rewards"][last])))
    return values, {"exact_steps": steps, "forward_calls": int(calls), "receipt": receipt,
                    "known_answer_t99": {"rows": int(last.sum()), "max_rel_error": float(known), "ok": bool(known <= 1e-12)},
                    "quantiles": {str(q): float(np.quantile(values, q)) for q in (0.0, 0.1, 0.5, 0.9, 1.0)},
                    "sha256": hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()}


def anchor_values(anchors, budget=None):
    """Exact Q^{pi_ref} at the 14 gated slots of every anchor, in both worlds, as the formal G2 computed them."""
    rows = [{"reset_seed": a["reset_seed"], "current": tuple(a["current"]), "prefix_actions": a["prefix_actions"]} for a in anchors]
    probes = P.build(len(rows), SLOTS)
    steps = int(sum(HORIZON - a["t"] for a in anchors)) * len(SLOTS)
    out = {"slots": SLOTS, "probes": probes}
    for world in ("stall3", "nostall"):
        if budget is not None:
            budget.charge(f"anchor_values_{world}", steps)
        q, calls, receipt = qpi.raw_values(rows, probes, pi_ref.act, world, clip_all=True)
        require(bool(np.all(np.isfinite(q))), "Non-finite anchor value")
        out[world] = q
        out[f"receipt_{world}"] = receipt
    ix = {n: SLOTS.index(n) for n in SLOTS}
    q_st, q_ns = out["stall3"], out["nostall"]
    out["h1_max_rel"] = max(float(np.max(np.abs(q_st[:, ix[n]] - q_ns[:, ix[n]]) / np.maximum(1.0, np.abs(q_st[:, ix[n]])))) for n in S9)
    out["rho"] = {n: q_ns[:, ix[n]] - q_st[:, ix[n]] for n in ("s05", "s10", "s20", "p10")}
    out["median_rho_s10"] = float(np.median(out["rho"]["s10"]))
    out["anchor_obs"] = np.stack([a["obs"] for a in anchors]).astype(np.float32)
    out["s10_value"] = float(SLOT_VALUES["s10"])
    return out
