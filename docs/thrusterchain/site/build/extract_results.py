"""Extract the Unifloral-era results that the site displays, and cross-check them against their reports.

Reads saved arrays and reports only (no model, no simulator). Writes docs/thrusterchain/site/data/results.json.
Every aggregate the page shows either is copied verbatim from a report or is recomputed from saved arrays
and then checked against the report's own number; a mismatch stops the build.

Run from the repository root with the repository's own interpreter (NumPy is needed for the NPZ banks):
    .venv/Scripts/python.exe -I docs/thrusterchain/site/build/extract_results.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
T = ROOT / "experiments/thrusterchain_bca/titan_20261026/payload"
Z = ROOT / "experiments/thrusterchain_bca/zeus_20261026_validation"
OUT = ROOT / "docs/thrusterchain/site/data/results.json"

RAW_DIR = T / "unifloral_operator_alignment_v1/results_20261026T221700_762125Z"
DEV1 = T / "unifloral_frozen_usefulness_v1_seedfix/results_20261024T234619_729786Z/diagnostic.json"
DEV2 = T / "unifloral_frozen_scale_comparison_v1/results_20261026T172507_062277Z/diagnostic.json"
PILOT = Z / "matched_policy_pilot_v1"
OPER = Z / "operator_analysis"
KAPPAS = ["0.03", "0.1", "0.3", "1", "3", "10"]
TIE = 1e-6  # numerical band used by the frozen analyses: atol 1e-6, rtol 1e-6


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def near(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def load_banks():
    meta = json.loads((RAW_DIR / "raw_banks.json").read_text("utf8"))
    z = np.load(RAW_DIR / "raw_banks.npz")
    dev = meta["development"]
    arr = lambda name: z[dev[name]["npz_key"]]
    rows = dev["metadata"]
    saved_q, saved_u, saved_v = arr("saved_q"), arr("saved_u"), arr("saved_values")
    actions, states = arr("actions"), arr("states")
    # Group the 1,152 rows into 128 states of 9 candidates, preserving original candidate order.
    order = {}
    for i, r in enumerate(rows):
        order.setdefault((r["batch"], r["group_id"]), []).append(i)
    out = []
    for (batch, gid), idx in order.items():
        idx = sorted(idx, key=lambda i: rows[i]["candidate_id"])
        if [rows[i]["candidate_id"] for i in idx] != list(range(9)):
            raise SystemExit("candidate ids are not 0..8 for " + str((batch, gid)))
        r0 = rows[idx[0]]
        out.append({
            "batch": batch, "group_id": gid, "time": r0["time"], "episode_seed": r0["episode_seed"],
            "state": [round(float(v), 8) for v in states[idx[0]]],
            "q1": [float(saved_q[i, 0]) for i in idx],
            "q2": [float(saved_q[i, 1]) for i in idx],
            "u": [float(saved_u[i]) for i in idx],
            "v": [float(saved_v[i]) for i in idx],
            "actions": [[round(float(x), 7) for x in actions[i]] for i in idx],
        })
    if len(out) != 128:
        raise SystemExit("expected 128 states")
    return out, {"raw_banks.npz": sha(RAW_DIR / "raw_banks.npz"), "raw_banks.json": sha(RAW_DIR / "raw_banks.json")}


def select(scores):
    """First exact maximum within the frozen numerical band; mirrors the analyses' tie rule."""
    m = max(scores)
    for i, s in enumerate(scores):
        if s >= m - (TIE + TIE * abs(m)):
            return i
    return int(np.argmax(scores))


def recompute(states, batch, kappa):
    """Equal-state mean paired gain of (Q1 - kappa U) over raw Q1, plus improved/harmed counts and regret."""
    gains, improved, harmed, regret = [], 0, 0, []
    for s in states:
        if s["batch"] != batch:
            continue
        i_raw = select(s["q1"])
        i_k = select([q - kappa * u for q, u in zip(s["q1"], s["u"])])
        g = s["v"][i_k] - s["v"][i_raw]
        gains.append(g)
        band = TIE + TIE * max(abs(s["v"][i_k]), abs(s["v"][i_raw]))
        if g > band:
            improved += 1
        elif g < -band:
            harmed += 1
        regret.append(max(s["v"]) - s["v"][i_k])
    return {"mean_gain": float(np.mean(gains)), "improved": improved, "harmed": harmed,
            "mean_regret": float(np.mean(regret)), "states": len(gains)}


def official(path):
    a = json.loads(path.read_text("utf8"))["analysis"]
    m = a["methods"]
    table = {}
    for name, e in m.items():
        table[name] = {
            "family": e.get("family"), "kappa": e.get("kappa_penalty"), "penalty_factor": e.get("penalty_factor"),
            "gain_vs_raw_q": e["gain_vs_raw_q"]["mean"], "gain_ci95": e["gain_vs_raw_q"]["pointwise_ci95"],
            "gain_vs_actor": e["gain_vs_actor"]["mean"], "gain_vs_actor_ci95": e["gain_vs_actor"]["pointwise_ci95"],
            "improved": e["improved_vs_raw_q_count"], "harmed": e["harmed_vs_raw_q_count"],
            "regret": e["candidate_bank_regret"]["mean"], "regret_ci95": e["candidate_bank_regret"]["pointwise_ci95"],
        }
    return {"methods": table, "q1_errors": a.get("q1_errors"), "relationships": a.get("relationships"),
            "bootstrap": {k: a["bootstrap"].get(k) for k in ("draws", "independent_episode_groups", "scheme", "intervals", "seed")},
            "limitations": a.get("limitations"), "protocol_sha256": a.get("protocol_sha256")}


def pilot():
    rep = json.loads((PILOT / "run/evaluation/report.json").read_text("utf8"))
    res = json.loads((PILOT / "RUN_RESULT.json").read_text("utf8"))
    proto = json.loads((PILOT / "protocol.json").read_text("utf8"))
    comps = rep["comparisons_by_continuation_stream"]
    out = {
        "returns_by_policy": rep["returns_by_policy"],
        "mean_discounted_returns": rep["mean_discounted_returns"],
        "comparisons": {s: {c: {k: v for k, v in comps[s][c].items() if k != "paired_differences"} | {"paired_differences": comps[s][c]["paired_differences"]} for c in comps[s]} for s in comps},
        "bootstrap": rep["bootstrap"], "interval_scope": rep["interval_scope"], "limitations": rep["limitations"],
        "decision": rep["decision"], "primary_screen_passed": rep["primary_screen_passed"],
        "displacement": {k: {kk: vv for kk, vv in v.items() if not kk.endswith("by_row") and not kk.endswith("by_episode")} for k, v in rep["action_and_state_displacement"].items()},
        "run_result": {"status": res["status"], "elapsed_seconds": res["elapsed_seconds"], "decision": res["decision"],
                       "jobs": [{"job": j["job"], "completed": j["completed"], "exit_code": j["exit_code"], "elapsed_seconds": j["elapsed_seconds"],
                                 "updates": j["ledger"].get("committed_additional_updates")} for j in res["jobs"]]},
        "protocol": {k: proto[k] for k in ("arms", "streams", "critic_updates", "actor_updates", "auxiliary_updates", "refresh_at", "absolute_critic_start", "absolute_critic_end", "absolute_refresh", "evaluation_bootstrap_draws", "gamma", "horizon", "bca_kappa", "actor_distance_factor", "actor_distance_coefficient", "alpha", "credibility", "posterior_draws", "evaluation_episodes", "primary_return", "primary_screen", "initialization_scope", "checkpoint_state_sha256", "native_source_sha256", "dataset_sha256")},
        "sources": {"report": rel(PILOT / "run/evaluation/report.json"), "run_result": rel(PILOT / "RUN_RESULT.json"), "protocol": rel(PILOT / "protocol.json"), "results_md": rel(PILOT / "run_analysis/RESULTS.md")},
    }
    # Cross-check: mean of returns equals reported mean; paired differences equal BCA minus control per stream.
    for p, vals in out["returns_by_policy"].items():
        if not near(np.mean(vals), out["mean_discounted_returns"][p], 1e-9):
            raise SystemExit("pilot mean mismatch for " + p)
    for s in ("0", "1"):
        for ctrl in ("native", "distance", "original"):
            pd = comps[s][ctrl]["paired_differences"]
            bca = out["returns_by_policy"][f"seed{s}_bca"]
            other = out["returns_by_policy"]["original" if ctrl == "original" else f"seed{s}_{ctrl}"]
            if not all(near(a - b, c, 1e-9) for a, b, c in zip(bca, other, pd)):
                raise SystemExit(f"paired differences mismatch stream {s} {ctrl}")
            if not near(np.mean(pd), comps[s][ctrl]["mean_gain"], 1e-9):
                raise SystemExit(f"mean gain mismatch stream {s} {ctrl}")
    return out


def strata_csv(path):
    with open(path, newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def cached():
    rows = strata_csv(OPER / "cached_score_mechanism_v1/all_prespecified_strata.csv")
    lower = strata_csv(OPER / "cached_lower_only_v1/all_prespecified_strata.csv")
    def pick(rs, method):
        return {r["stratum"]: {k: (float(r[k]) if r[k] not in ("", None) and k not in ("stratum", "method") else r[k]) for k in r if k not in ("stratum", "method")} for r in rs if r["method"] == method}
    return {
        "score_mechanism": {m: pick(rows, m) for m in ("q1", "twin_min", "actor", "mean_native_target")},
        "lower_only": {m: pick(lower, m) for m in sorted({r["method"] for r in lower})},
        "sources": {"score_mechanism_csv": rel(OPER / "cached_score_mechanism_v1/all_prespecified_strata.csv"), "score_mechanism_results": rel(OPER / "cached_score_mechanism_v1/RESULTS.md"),
                    "lower_only_csv": rel(OPER / "cached_lower_only_v1/all_prespecified_strata.csv"), "lower_only_results": rel(OPER / "cached_lower_only_v1/RESULTS.md"),
                    "state_outcomes": rel(OPER / "cached_score_mechanism_v1/STATE_OUTCOMES.json")},
    }


def operator():
    rows = strata_csv(OPER / "all_prespecified_strata.csv")
    out = {}
    for r in rows:
        out.setdefault(r["stratum"], {}).setdefault(r["metric"], []).append({"draw": int(r["draw"]), "value": (float(r["value"]) if r["value"] else None), "defined": int(r["defined"]), "undefined": int(r["undefined"])})
    summ = json.loads((OPER / "summary.json").read_text("utf8"))
    return {"strata": out, "development_draw_summary": summ.get("development_draw_summary"), "fitting_draw_summary": summ.get("fitting_draw_summary"),
            "sources": {"csv": rel(OPER / "all_prespecified_strata.csv"), "summary": rel(OPER / "summary.json"), "results_md": rel(OPER / "RESULTS.md")}}


def main():
    states, bank_hashes = load_banks()
    off = {"development_1": official(DEV1), "development_2": official(DEV2)}
    checks = []
    for batch in ("development_1", "development_2"):
        for k in KAPPAS:
            rc = recompute(states, batch, float(k))
            o = off[batch]["methods"]["bca_kappa_" + k]
            ok = near(rc["mean_gain"], o["gain_vs_raw_q"], 1e-9) and rc["improved"] == o["improved"] and rc["harmed"] == o["harmed"] and near(rc["mean_regret"], o["regret"], 1e-9)
            checks.append({"batch": batch, "selector": "bca_kappa_" + k, "recomputed": rc, "official": {"gain": o["gain_vs_raw_q"], "improved": o["improved"], "harmed": o["harmed"], "regret": o["regret"]}, "match": ok})
            if not ok:
                raise SystemExit("recomputed kappa " + k + " on " + batch + " does not match the report: " + json.dumps(checks[-1]))
        # raw Q regret and actor comparisons
        rc0 = recompute(states, batch, 0.0)
        if not near(rc0["mean_regret"], off[batch]["methods"]["raw_q"]["regret"], 1e-9):
            raise SystemExit("raw-Q regret mismatch on " + batch)
        checks.append({"batch": batch, "selector": "raw_q", "recomputed": rc0, "official": {"regret": off[batch]["methods"]["raw_q"]["regret"]}, "match": True})
    # Signed error summaries recomputed from arrays (descriptive; the page labels them as recomputed).
    q1 = np.array([q for s in states for q in s["q1"]]); v = np.array([x for s in states for x in s["v"]]); u = np.array([x for s in states for x in s["u"]])
    err = q1 - v
    signed = {}
    for label, mask in (("all", np.ones_like(err, bool)), ("development_1", np.array([s["batch"] == "development_1" for s in states for _ in range(9)])), ("development_2", np.array([s["batch"] == "development_2" for s in states for _ in range(9)]))):
        e = err[mask]
        d = {"n": int(mask.sum()), "mean_signed_error": float(e.mean()), "median_signed_error": float(np.median(e)), "share_overestimated": float((e > 0).mean()), "mean_abs_error": float(np.abs(e).mean()),
             "mean_u": float(u[mask].mean()), "min_u": float(u[mask].min()), "max_u": float(u[mask].max())}
        for k in KAPPAS:
            e2 = (q1 - float(k) * u - v)[mask]
            d["kappa_" + k] = {"mean_signed_error": float(e2.mean()), "share_overestimated": float((e2 > 0).mean()), "mean_abs_error": float(np.abs(e2).mean()),
                               "positive_error_removed": float((np.clip(e, 0, None) - np.clip(e2, 0, None)).mean()), "underestimation_added": float((np.clip(-e2, 0, None) - np.clip(-e, 0, None)).mean())}
        signed[label] = d
    out = {
        "schema": "thrusterchain-site-results-v1",
        "host": {"learner": "Unifloral standalone TD3+BC", "revision": "f2dc1278eae18ed3e22c92255119c54885414d17", "td3_bc_sha256": "6f5ab6f98d706d902a18f11e1c81e99e2691dc4c8a53843bf6521c9f95d00eee",
                 "checkpoint_sha256": "874d65b7f4dcb5ec285a0d4360439c73a787b5e8c567ff7c0138e0c1e22948e8", "checkpoint_counts": {"critic": 20000, "actor": 10000, "auxiliary": 20000}, "kappa_at_training": 0.0},
        "development_banks": {
            "description": "128 states (two batches of 64 episodes, one query per episode at time 0, 25, 50 or 75), nine candidates each: the frozen actor's action (candidate 0), the zero command (candidate 1) and seven seeded uniform commands. V is the deterministic simulator continuation: the candidate once, then the same frozen actor to the horizon; discount 0.98; remaining horizon 100 - t; no terminal bootstrap. Q1, Q2 and U are the checkpoint's saved predictions at the same state and action.",
            "reference_definition": "own-policy continuation of the 20k checkpoint's frozen actor, not Q-star, not a shared policy",
            "states": states,
            "sources": {"arrays": rel(RAW_DIR / "raw_banks.npz"), "array_index": rel(RAW_DIR / "raw_banks.json"), "development_1": rel(DEV1), "development_2": rel(DEV2), "hashes": bank_hashes},
        },
        "official": off,
        "cross_checks": checks,
        "signed_error_recomputed": signed,
        "matched_policy_pilot": pilot(),
        "cached_comparisons": cached(),
        "operator_alignment": operator(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":"), allow_nan=False), "utf8", newline="\n")
    print(json.dumps({"written": rel(OUT), "bytes": OUT.stat().st_size, "states": len(states), "cross_checks": len(checks), "all_match": all(c["match"] for c in checks),
                      "signed_error_all": signed["all"]}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
