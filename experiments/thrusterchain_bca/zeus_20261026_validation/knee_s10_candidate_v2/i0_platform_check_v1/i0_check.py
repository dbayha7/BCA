"""I0 platform rehearsal for the native screen of candidate knee_s10_v2 (attempt-free, read-only, CPU, no JAX).

Why: the screen's g1 unit opens with the instrument check I0 (G1_RULE.md): the exact median rho(s10) over the 96 gated
anchors, recomputed in the run's own environment, must equal the formal G2 report's Windows-computed value to 1e-9
relative, and the H1 identity must hold. Attempts 1 and 2 already showed that the recorded Linux environment does not
reproduce a Windows reset bit for bit (one ULP in `wave_1`, episode 16). If that difference reaches rho, I0 fails,
the g1 unit reads INCONCLUSIVE, and attempt 3, the last the rule allows, is spent before any training.

This script runs exactly the g1 unit's instrument path (`screen_bind.v2_gated_anchors` and `g1_labels.anchor_values`,
the same modules the sealed screen imports) and the amendment A1 dataset loader for both archives, outside any
attempt, and writes a receipt. It trains nothing, fits nothing, opens no checkpoint and writes nothing into the sealed
package. Run it on Windows (it must agree with the formal report exactly, same platform) and then in the recorded WSL
environment (the measurement that matters). A WSL mismatch is a reason to amend I0 before approving attempt 3, not a
reason to touch the sealed rule from here.

  Windows:  PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B <this file>
  WSL:      see README.md
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import platform
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CAND = HERE.parent
VAL = CAND.parent
for _p in (str(VAL / "knee_edge_ood_v3"), str(CAND / "formal_g2"), str(CAND / "native_screen"), str(CAND / "g1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("JAX_PLATFORMS", "cpu")                # never a GPU; the instrument path imports no JAX anyway

import numpy as np                                            # noqa: E402

REL_TOL = 1e-9                                                # G1_RULE.md I0
STEP_CAP = 9_000_000                                          # the g1 unit's cap; this script uses a small fraction
SECONDS_CAP = 1_200.0


def _plain(value):
    """JSON default for numpy scalars and arrays (the same conversion step03._plain makes for the sealed plan)."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def rel_diff(a, b):
    return float(abs(a - b) / max(1.0, abs(b)))


def compare(median_rho, formal, h1_max_rel, tol=REL_TOL):
    """The I0 decision exactly as g1_unit.run computes it, plus the two distances."""
    return {"median_rho_s10": float(median_rho), "formal_report_value": float(formal), "h1_max_rel": float(h1_max_rel),
            "rel_diff": rel_diff(median_rho, formal), "tol": tol,
            "ok": bool(abs(median_rho - formal) <= tol * max(1.0, abs(formal)) and h1_max_rel <= tol)}


def main():
    started = time.monotonic()
    receipt = {"schema": "knee_s10_candidate_v2.i0_platform_check.v1", "started_utc": datetime.now(timezone.utc).isoformat(),
               "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
                            "python": sys.version.split()[0], "numpy": np.__version__, "executable": sys.executable},
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "attempt_consumed": False, "training_updates": 0, "checkpoints_opened": 0}
    require_jax_absent = "jax" not in sys.modules
    import screen_bind as SB                                  # substitutes the recorder, pins the formal report
    import g1_labels as GL
    receipt["jax_imported"] = "jax" in sys.modules
    receipt["jax_absent_before_imports"] = require_jax_absent
    receipt["formal_manifest_sha256"] = SB.FORMAL_REPORT["manifest_sha256"]
    receipt["screen_manifest_sha256"] = hashlib.sha256((CAND / "native_screen" / "SCREEN_MANIFEST.json").read_bytes()).hexdigest()
    budget = SB.G2.G.Budget(STEP_CAP, SECONDS_CAP)

    # 1. The amendment A1 loader, both archives (what every unit of the screen does first).
    loader = {}
    for world in ("stall3", "nostall"):
        t0 = time.monotonic()
        ds = SB.load_dataset(world)
        loader[world] = {"digest": ds.digest, "digest_matches_formal": ds.digest == SB.DIGESTS[world],
                         "rows": int(np.asarray(ds.arrays["states"]).shape[0]), "seconds": round(time.monotonic() - t0, 2)}
    receipt["loader"] = loader

    # 2. The anchors (P1: identical in both worlds) and the exact values at the 14 gated slots, as g1 computes them.
    t0 = time.monotonic()
    anchors = SB.v2_gated_anchors()
    receipt["anchors"] = {"count": len(anchors), "all_both_worlds_identical": all(a["both_worlds_identical"] for a in anchors),
                          "seconds": round(time.monotonic() - t0, 2)}
    t0 = time.monotonic()
    av = GL.anchor_values(anchors, budget)
    formal_h2 = SB.FORMAL_REPORT["H"]["H2"]
    i0 = compare(av["median_rho_s10"], float(formal_h2["median_exact_rho_s10"]), av["h1_max_rel"])
    receipt["I0"] = i0
    receipt["anchor_values_seconds"] = round(time.monotonic() - t0, 2)

    # 3. Finer than I0 asks: every reported rho statistic against the formal report, so a platform drift that stays
    #    under 1e-9 at the median is still visible if it exists anywhere.
    rep = SB.FORMAL_REPORT["H"]["reported"]
    per_slot = {n: {"recomputed": float(np.median(av["rho"][n])), "formal": float(rep["median_exact_rho"][n]),
                    "rel_diff": rel_diff(float(np.median(av["rho"][n])), float(rep["median_exact_rho"][n]))}
                for n in ("s05", "s10", "s20", "p10")}
    quant = {str(q): {"recomputed": float(np.quantile(av["rho"]["s10"], q)), "formal": float(rep["quantiles_exact_rho_s10"][str(q)]),
                      "rel_diff": rel_diff(float(np.quantile(av["rho"]["s10"], q)), float(rep["quantiles_exact_rho_s10"][str(q)]))}
             for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    rho_s10 = np.asarray(av["rho"]["s10"], np.float64)
    receipt["finer"] = {"median_rho_by_slot": per_slot, "quantiles_rho_s10": quant,
                        "max_rel_diff_any_reported_statistic": max([v["rel_diff"] for v in per_slot.values()] + [v["rel_diff"] for v in quant.values()]),
                        "rho_s10_sha256": hashlib.sha256(np.ascontiguousarray(rho_s10).tobytes()).hexdigest(),
                        "q_stall3_sha256": hashlib.sha256(np.ascontiguousarray(av["stall3"]).tobytes()).hexdigest(),
                        "q_nostall_sha256": hashlib.sha256(np.ascontiguousarray(av["nostall"]).tobytes()).hexdigest()}
    # 4. The data-plan re-derivation every training and evaluation job performs first (stage1_worker.load_world):
    #    the plan rebuilt here must equal, as canonical JSON, the plan prepare.py recorded on Windows.
    from core import canonical_json
    import bindings as B
    SB.patch_bindings()
    collection = json.loads((CAND / "native_screen" / "collection" / "collection.json").read_text(encoding="utf-8"))
    plans = {}
    for world in ("stall3", "nostall"):
        ds = SB.load_dataset(world)
        plan = B.plan_streams(ds, world)
        equal = {}
        for arm in B.ARMS:
            fresh = json.loads(json.dumps(plan["plans"][arm]["receipt"], default=_plain))
            equal[arm] = canonical_json(fresh) == canonical_json(collection["data_plan"][world]["receipts"][arm])
        plans[world] = {"receipts_equal_by_arm": equal, "all_equal": all(equal.values())}
    receipt["data_plan_rederivation"] = plans
    plans_ok = all(v["all_equal"] for v in plans.values())
    receipt["simulator_steps_exact"] = int(budget.used)
    receipt["by_unit"] = dict(budget.ledger)
    receipt["elapsed_seconds"] = round(time.monotonic() - started, 2)
    receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
    if i0["ok"] and plans_ok:
        receipt["verdict"] = "I0 WOULD PASS on this platform, and every job's data-plan check would pass"
    elif not i0["ok"]:
        receipt["verdict"] = "I0 WOULD FAIL on this platform: do not approve attempt 3 before amending I0"
    else:
        receipt["verdict"] = "DATA-PLAN RE-DERIVATION DIFFERS on this platform: every training job would refuse; do not approve attempt 3"
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = HERE / f"receipt_{tag}_{receipt['platform']['system'].lower()}.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": receipt["verdict"], "I0": i0, "max_rel_diff_any_reported_statistic": receipt["finer"]["max_rel_diff_any_reported_statistic"],
                      "loader": {w: v["digest_matches_formal"] for w, v in loader.items()}, "simulator_steps_exact": receipt["simulator_steps_exact"],
                      "platform": receipt["platform"]["system"], "receipt": out.name}, indent=2))
    return 0 if (i0["ok"] and plans_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
