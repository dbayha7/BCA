"""The i0 unit (stage 0, SCREEN_RULE_r5.md): the instrument checks every later unit depends on, run in the job's own
environment before any attempt is counted. No JAX, no training, no checkpoint.

- I0: the exact median rho(s10) over the 96 gated anchors, recomputed here, equals the formal G2 report's value to
  1e-9 relative, and the H1 identity holds (G1_RULE.md I0);
- both archives load through the amendment A1 loader with the formal digests;
- both worlds' data plans re-derive equal to the plans prepare.py recorded (what stage1_worker.load_world checks).
"""
import json
from pathlib import Path

import numpy as np

import screen_bind as SB
import g1_labels as GL
import bindings as B
from core import canonical_json, require

REL_TOL = 1e-9
STEP_CAP = 1_000_000


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def compare(median_rho, formal, h1_max_rel, tol=REL_TOL):
    return {"median_rho_s10": float(median_rho), "formal_report_value": float(formal), "h1_max_rel": float(h1_max_rel),
            "rel_diff": float(abs(median_rho - formal) / max(1.0, abs(formal))), "tol": tol,
            "ok": bool(abs(median_rho - formal) <= tol * max(1.0, abs(formal)) and h1_max_rel <= tol)}


def run(job, out, deadline):
    budget = SB.G2.G.Budget(min(STEP_CAP, int(job.get("transitions") or STEP_CAP)), float(job["timeout_seconds"]))
    loader = {}
    for world in ("stall3", "nostall"):
        deadline()
        ds = SB.load_dataset(world)
        loader[world] = {"digest": ds.digest, "ok": ds.digest == SB.DIGESTS[world]}
    anchors = SB.v2_gated_anchors()
    av = GL.anchor_values(anchors, budget)
    i0 = compare(av["median_rho_s10"], float(SB.FORMAL_REPORT["H"]["H2"]["median_exact_rho_s10"]), av["h1_max_rel"])
    rep = SB.FORMAL_REPORT["H"]["reported"]
    finer = {n: float(abs(np.median(av["rho"][n]) - rep["median_exact_rho"][n]) / max(1.0, abs(rep["median_exact_rho"][n])))
             for n in ("s05", "s10", "s20", "p10")}
    SB.patch_bindings()
    collection = json.loads((Path(job["collection_dir"]) / "collection.json").read_text(encoding="utf-8"))
    plans = {}
    for world in ("stall3", "nostall"):
        deadline()
        ds = SB.load_dataset(world)
        plan = B.plan_streams(ds, world)
        equal = {arm: canonical_json(json.loads(json.dumps(plan["plans"][arm]["receipt"], default=_plain)))
                 == canonical_json(collection["data_plan"][world]["receipts"][arm]) for arm in B.ARMS}
        plans[world] = {"receipts_equal_by_arm": equal, "ok": all(equal.values())}
    ok = bool(i0["ok"] and all(v["ok"] for v in loader.values()) and all(v["ok"] for v in plans.values()))
    require(len(anchors) == 96, "Expected 96 gated anchors")
    return {"kind": "i0", "I0": i0, "finer_max_rel_diff": max(finer.values()), "loader": loader, "data_plan": plans,
            "anchors": len(anchors), "nonpositive_rho_anchors": int(np.sum(av["rho"]["s10"] <= 0)), "ok": ok,
            "transitions": budget.used, "by_unit": budget.ledger}
