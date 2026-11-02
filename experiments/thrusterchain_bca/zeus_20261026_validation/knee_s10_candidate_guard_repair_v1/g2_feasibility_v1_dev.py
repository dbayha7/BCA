"""Development G2 feasibility screen for candidate knee_s10_v1 (rules: G2_RULE.md, written before this ran).

No training, no checkpoint. A test-entropy collection of 1,280 episodes per world under the widened recorder, a
test-only anchor stream, and exact Q^{pi_ref} by the raw float64 loop. Development data: never confirmatory.

  PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B knee_s10_candidate_v1/g2_feasibility.py
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V3 = HERE.parent / "knee_edge_ood_v3"
for p in (str(V3), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import recorder_wide                                        # noqa: E402  must replace `recorder` before any v3 import
sys.modules["recorder"] = recorder_wide

import numpy as np                                          # noqa: E402

import dataset_knee as D                                    # noqa: E402
import fixtures as F                                        # noqa: E402
import masks as M                                           # noqa: E402
import pi_ref                                               # noqa: E402
import probes as P                                          # noqa: E402
import qpi                                                  # noqa: E402
import stage1_eval as E                                     # noqa: E402
import stage1_math as SM                                    # noqa: E402
import step04_math as SM4                                   # noqa: E402
import worlds                                               # noqa: E402
from core import require                                    # noqa: E402
from knee_region import GATED_PAST_KNEE, KNEE, S9, past_knee_a0  # noqa: E402

ENTROPY = F.TEST_ENTROPY + 21              # test-only collection entropy
STREAM = 707_070_707                       # test-only anchor and start stream (checked against every sealed value)
EPISODES = 1280
SIGMA, C_SIGMA, TOL_SIGMA = "s10", 0.3, 0.15
R_REQ = (0.5 + C_SIGMA * TOL_SIGMA) / (1 - C_SIGMA)
D4_WINDOW, D4_BARS = (-0.95, -0.85), {1: 2.00, 2: 1.82}
SLOTS = list(E.GATED_SLOTS)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def start_obs(name, count):
    out = []
    for i in range(count):
        spec = E.stream_item(name, i, STREAM)
        env = worlds.environment("stall3")
        try:
            obs, _ = env.reset(seed=spec["reset_seed"], options={"start": {"current": spec["current"]}})
        finally:
            env.close()
        out.append(np.asarray(obs, np.float32))
    return np.stack(out)


def d_checks(data, starts, anchors):
    out = {}
    for world, ds in data.items():
        arrays = ds.arrays
        rows = M.fitting_rows(arrays)
        a0 = rows["actions"][:, 0].astype(np.float64)
        past = past_knee_a0(rows["actions"])[0]
        cutoff = ~past                                             # CUTOFF keeps every |a0| <= KNEE fitting row
        high = rows["bands"] >= 1
        deep_ep = np.repeat(np.asarray(arrays["reset_component"])[rows["episodes"]] == 1, 100)
        a_obs, a_t = np.stack([a["obs"] for a in anchors]), np.asarray([a["t"] for a in anchors])
        l15_all = SM4.l15(rows["obs"], rows["episode_ids"], rows["times"], deep_ep, starts, a_obs, a_t)
        keep = cutoff
        try:                                                       # D3b is reported only; it may be uncomputable
            l15_cut = SM4.l15(rows["obs"][keep], rows["episode_ids"][keep], rows["times"][keep], deep_ep[keep], starts, a_obs, a_t)
        except ValueError as exc:
            groups = {int(t): int(np.sum(keep & deep_ep & (rows["times"] == t))) for t in (0, 25, 50)}
            l15_cut = {"start_share": None, "anchor_share": None, "pass": None,
                       "not_computable": f"{exc}; CUTOFF deep-load rows per t: {groups}"}
        lo, hi = D4_WINDOW
        d4 = {}
        for band, name in ((1, "mid"), (2, "deep")):
            in_band = rows["bands"] == band
            count = int(np.sum(in_band & (a0 >= lo) & (a0 <= hi)))
            density = count / (int(in_band.sum()) * 0.1) if in_band.any() else 0.0
            d4[name] = {"band_rows": int(in_band.sum()), "rows_in_window": count, "density": density,
                        "bar": D4_BARS[band], "ok": bool(density >= D4_BARS[band])}
        out[world] = {
            "D1": {"cutoff_rows_past_knee": int(np.sum(cutoff & past)), "ok": bool(np.sum(cutoff & past) == 0)},
            "D2": {"min_distance_s10_to_cutoff_high_load_a0": float(np.min(np.abs(a0[cutoff & high] - (-0.9)))),
                   "ok": bool(np.min(np.abs(a0[cutoff & high] - (-0.9))) >= 0.0999)},
            "D3": {"start_share": l15_all["start_share"], "anchor_share": l15_all["anchor_share"], "ok": l15_all["pass"]},
            "D3b_cutoff_rows_reported": {"start_share": l15_cut["start_share"], "anchor_share": l15_cut["anchor_share"],
                                         "would_pass": l15_cut["pass"], "note": l15_cut.get("not_computable")},
            "D4": {**d4, "ok": bool(all(v["ok"] for v in d4.values()))},
            "reported": {"fitting_rows": int(len(a0)), "high_load_rows": int(high.sum()),
                         "high_load_depth_ge_0p1_rows": int(np.sum(high & (a0 <= -0.9))),
                         "twin_W_rows": int(past.sum()), "cutoff_rows": int(cutoff.sum())},
        }
    return out


def h_checks(anchors):
    rows = [{"reset_seed": a["reset_seed"], "current": tuple(a["current"]), "prefix_actions": a["prefix_actions"]}
            for a in anchors]
    probes = P.build(len(rows), SLOTS)
    q_st, calls, receipt = qpi.raw_values(rows, probes, pi_ref.act, "stall3", clip_all=True)
    q_ns, _, receipt_ns = qpi.raw_values(rows, probes, pi_ref.act, "nostall", clip_all=True)
    idx = {n: SLOTS.index(n) for n in SLOTS}
    on_support = max(float(np.max(np.abs(q_st[:, idx[n]] - q_ns[:, idx[n]]))) for n in S9)
    res = SM4.neighbor_residuals(q_st[:, [idx[n] for n in S9]], {n: q_st[:, idx[n]] for n in GATED_PAST_KNEE},
                                 {n: q_ns[:, idx[n]] for n in ("s05", "s10", "s20")})
    item5 = SM.item5_ok(res["item5_error"])
    valid = item5
    rho_true = {n: q_ns[:, idx[n]] - q_st[:, idx[n]] for n in ("s05", "s10", "s20")}
    med = lambda x: float(np.median(x)) if np.size(x) else None          # noqa: E731
    sat = pi_ref.act(np.stack([a["obs"] for a in anchors]))[:, 0] <= np.float32(-KNEE)
    h2 = med(res["rho_hat"][SIGMA][valid])
    return {
        "H1": {"item5_share": float(np.mean(item5)), "bar": 0.90, "ok": bool(np.mean(item5) >= 0.90),
               "item5_error_median": {k: med(v) for k, v in res["item5_error"].items()}},
        "H2": {"median_rho_hat_s10_predictor_valid": h2, "n_valid": int(valid.sum()), "R_req": R_REQ,
               "ok": bool(h2 is not None and h2 >= R_REQ)},
        "reported": {"median_rho_hat": {n: med(res["rho_hat"][n][valid]) for n in ("s05", "s10", "s20")},
                     "median_true_one_step_gap": {n: med(rho_true[n]) for n in rho_true},
                     "sign_control_p10_share_negative_max_k": float(np.mean(res["rho_hat_max"]["p10"] < 0)),
                     "pi_ref_saturated_at_minus_knee_share": float(np.mean(sat)),
                     "stall_equals_nostall_on_S9_max_abs": on_support,
                     "median_abs_Q_at_edge_knee": med(np.abs(q_st[:, idx["edge_knee"]])),
                     "forward_calls": int(calls), "receipts": [receipt, receipt_ns]},
    }


def main():
    import stage1_smoke
    require(STREAM not in stage1_smoke.sealed_values() and ENTROPY not in stage1_smoke.sealed_values(),
            "Development stream or entropy collides with a sealed value")
    started = time.monotonic()
    data = {w: D.generate(episodes=EPISODES, entropy=ENTROPY, world=w, provenance="smoke") for w in ("stall3", "nostall")}
    anchors = [E.build_anchor("stage1_e2_anchors", i, STREAM) for i in range(64)]
    for a in anchors:
        a["t"] = len(a["prefix_actions"])
    gated = [a for a in anchors if a["t"] in (0, 25, 50)]
    require(len(gated) == 48, f"Expected 48 gated anchors, got {len(gated)}")
    starts = start_obs("stage1_e2_bank", 128)
    d = d_checks(data, starts, gated)
    h = h_checks(gated)
    data_ok = all(d[w][k]["ok"] for w in d for k in ("D1", "D2", "D3"))
    testable = all(d[w]["D4"]["ok"] for w in d) and h["H2"]["ok"]
    labels = []
    if not h["H1"]["ok"]:
        labels.append("INCONCLUSIVE (H1: estimator invalid under pi_ref)")
    if not data_ok:
        labels.append("NOT DATA-OOD (" + ", ".join(f"{w} {k}" for w in d for k in ("D1", "D2", "D3") if not d[w][k]["ok"]) + ")")
    if not testable:
        labels.append("NOT TESTABLE (" + ", ".join([f"{w} D4" for w in d if not d[w]["D4"]["ok"]] + ([] if h["H2"]["ok"] else ["H2"])) + ")")
    out = {"schema": "knee_s10_candidate_v1.g2_development.v1", "written_utc": datetime.now(timezone.utc).isoformat(),
           "status": "DEVELOPMENT estimate of G2 (test entropy, test-only stream); not the formal G2; never confirmatory",
           "rule": "G2_RULE.md", "rule_sha256": sha(HERE / "G2_RULE.md"),
           "code_sha256": {f: sha(HERE / f) for f in ("g2_feasibility.py", "pi_ref.py", "recorder_wide.py")},
           "pi_ref_gains_sha256": pi_ref.gains_sha256(), "entropy": ENTROPY, "stream": STREAM, "episodes": EPISODES,
           "headline": labels[0] if labels else "PASS", "all_labels": labels or ["PASS"],
           "D": d, "H": h, "elapsed_seconds": time.monotonic() - started}
    (HERE / "g2_development_result.json").write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps({"headline": out["headline"], "all_labels": out["all_labels"], "H": {k: h[k] for k in ("H1", "H2")},
                      "D4": {w: d[w]["D4"] for w in d}, "D3": {w: d[w]["D3"] for w in d},
                      "reported": h["reported"]["median_rho_hat"], "true_gap": h["reported"]["median_true_one_step_gap"],
                      "saturation": h["reported"]["pi_ref_saturated_at_minus_knee_share"],
                      "elapsed": out["elapsed_seconds"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
