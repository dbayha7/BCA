"""Per-anchor development numbers for the OOD definition v4 draft, from the v2 exploratory Stage 1 JSON reports only
(no checkpoint, no dataset array). These runs are DEVELOPMENT data: every number here may inform a rule's form, and
none may serve as a frozen threshold for a candidate judged on the same anchors.

  .venv/Scripts/python.exe -B ood_definition_v4/development_evidence.py
"""
from pathlib import Path
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "knee_edge_ood_v2" / "stage1_exploratory_run" / "attempt_1"
NAMES = {"nan": np.nan, "positive_infinity": np.inf, "negative_infinity": -np.inf}
GATED = 48
SOFTPLUS_W = 0.05


def arr(values):
    return np.asarray([[NAMES.get(v, v) for v in row] for row in values], np.float64)


def kc(q, ref, names, slot):
    i, k = names.index(slot), names.index("edge_knee")
    return (q[:, i] - q[:, k]) - (ref[:, i] - ref[:, k])


def softplus_ratio(d, w=SOFTPLUS_W):
    """v1 Part 2 rounding rule (PROTOCOL_V2.md change 3): KC/rho of a correct critic rounding a kink with width w."""
    return w * (d / w + np.log(2) - np.log1p(np.exp(d / w))) / d


def main():
    out = {"source": str(RUN), "role": "development data; outcomes of this run are known", "seeds": {}}
    for i in (0, 1, 2):
        rep = json.loads((RUN / f"evaluate_s{i}" / "report.json").read_text(encoding="utf-8"))
        a = {f"{arm}_{w}": rep["runs"][f"s{i}_{arm}_{w}"]["020000"]["anchors_e2"]
             for arm in ("cutoff", "random") for w in ("stall3", "nostall")}
        seed = {}
        for slot, depth in (("s05", 0.05), ("s10", 0.10), ("s20", 0.20)):
            names = a["cutoff_stall3"]["gated_slots"]
            if slot not in names:
                continue
            k = {}
            for key, x in a.items():
                q1 = arr(x["q1"])[:GATED]
                k[key] = {"clip": kc(q1, arr(x["qpi_clip"])[:GATED], x["gated_slots"], slot),
                          "own": kc(q1, arr(x["qpi_unclipped"])[:GATED], x["gated_slots"], slot)}
            ns = a["random_stall3"]
            nsl = ns["nostall_slots"]
            # true one-step gap under RANDOM's own clipped continuation (each arm's own actor, NOT a shared one)
            q_ns, q_st = arr(ns["qpi_nostall"])[:GATED], arr(ns["qpi_clip"])[:GATED]
            rho_r = (q_ns[:, nsl.index(slot)] - q_st[:, names.index("edge_knee")]) - \
                    (q_st[:, names.index(slot)] - q_st[:, names.index("edge_knee")])
            c = round(float(softplus_ratio(depth)) + 0.005, 2)
            entry = {"softplus_c_w005": float(softplus_ratio(depth))}
            for ref in ("clip", "own"):
                did = (k["cutoff_stall3"][ref] - k["random_stall3"][ref]) - (k["cutoff_nostall"][ref] - k["random_nostall"][ref])
                entry[f"theta_{ref}"] = {"median_per_anchor_did": float(np.nanmedian(did)),
                                         "q25": float(np.nanpercentile(did, 25)), "q75": float(np.nanpercentile(did, 75))}
            kr = k["random_stall3"]["clip"]
            tol = {"s05": 0.10, "s10": 0.15, "s20": 0.30}[slot]
            c_l1 = {"s05": 0.4, "s10": 0.3, "s20": 0.2}[slot]                     # draft r2 L1 allowance
            entry["twin"] = {"median_kc": float(np.nanmedian(kr)), "median_rho_true": float(np.nanmedian(rho_r)),
                             "capture_median": float(np.nanmedian(1 - kr / rho_r)),
                             "share_l1_rule": float(np.mean((kr <= c_l1 * rho_r + tol) & (kr >= -tol))),
                             "l1_rule": f"-{tol} <= KC_L,j <= {c_l1} rho_j + {tol}", "softplus_c_rounded": c}
            # draft r2 H2, SD-free part: a twin at the edge of L1 leaves theta <= rho (1 - c) - c tol
            entry["R_req_sd_free"] = (0.5 + c_l1 * tol) / (1 - c_l1)
            seed[slot] = entry
        out["seeds"][i] = seed
    for slot in ("s05", "s10", "s20"):
        for ref in ("clip", "own"):
            vals = [out["seeds"][i][slot][f"theta_{ref}"]["median_per_anchor_did"] for i in (0, 1, 2) if slot in out["seeds"][i]]
            if vals:
                out.setdefault("seed_level", {}).setdefault(slot, {})[f"theta_{ref}"] = {
                    "per_seed": vals, "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))}
    path = HERE / "development_evidence.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    json.dump(out["seed_level"], sys.stdout, indent=1)
    print()
    for i in (0, 1, 2):
        print(i, {s: {"capture": round(v["twin"]["capture_median"], 3), "share_l1": round(v["twin"]["share_l1_rule"], 3),
                      "rho_twin": round(v["twin"]["median_rho_true"], 3), "R_req_sd_free": round(v["R_req_sd_free"], 3)}
                  for s, v in out["seeds"][i].items()})


if __name__ == "__main__":
    main()
