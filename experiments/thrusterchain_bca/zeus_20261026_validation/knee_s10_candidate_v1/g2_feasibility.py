"""Development G2 feasibility screen for candidate knee_s10_v1, revision 3 (rules: G2_RULE_r3.md, written first).

No training and no checkpoint. Uses a test-entropy collection of 1,280 episodes per world under the widened recorder,
a test-only anchor stream, and exact Q^{pi_ref} from the raw float64 loop, with pi_ref version 2 (inside both arms'
data). Development data: never confirmatory. The first run's code is g2_feasibility_v1_dev.py, its result
g2_development_result.json.

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

if "recorder" in sys.modules:
    raise SystemExit("A recorder module was imported before the widened recorder could replace it")
import recorder_wide                                        # noqa: E402  must replace `recorder` before any v3 import
sys.modules["recorder"] = recorder_wide

import numpy as np                                          # noqa: E402

import dataset_knee as D                                    # noqa: E402
import fixtures as F                                        # noqa: E402
import masks as M                                           # noqa: E402
import pi_ref                                               # noqa: E402
import pi_ref_v1_dev                                        # noqa: E402
import probes as P                                          # noqa: E402
import qpi                                                  # noqa: E402
import region_independent as RI                             # noqa: E402
import seeds as S                                           # noqa: E402
import stage1_eval as E                                     # noqa: E402
import step04_math as SM4                                   # noqa: E402
import worlds                                               # noqa: E402
from core import require                                    # noqa: E402
from knee_region import GATED_PAST_KNEE, KNEE, S9, SLOTS as SLOT_VALUES, load_band, load_g, past_knee_a0  # noqa: E402

for _m in (D, E, S):
    require(_m.recorder is recorder_wide, f"{_m.__name__} does not use the widened recorder")

ENTROPY = F.TEST_ENTROPY + 21              # test-only collection entropy (same as the v1 development run)
STREAM = 707_070_707                       # test-only anchor and start stream (same as the v1 development run)
EPISODES = 1280
SIGMA, C_SIGMA, TOL_SIGMA = "s10", 0.3, 0.15
R_REQ = (0.5 + C_SIGMA * TOL_SIGMA) / (1 - C_SIGMA)
D4_WINDOW, D4_BARS = (-0.95, -0.85), {1: 2.00, 2: 1.82}
B1_BAR, H1_TOL = 0.99, 1e-9
SLOTS = list(E.GATED_SLOTS)
S10 = float(SLOT_VALUES["s10"])


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


def in_cutoff_support(obs, a):
    """Per row: is command a inside CUTOFF's retained data for the load band of obs (G2_RULE_r3.md B1)?"""
    high = load_band(load_g(obs)) >= 1
    a0, side = a[:, 0].astype(np.float64), np.abs(a[:, 1:].astype(np.float64))
    lo = np.where(high, -KNEE, -KNEE)
    hi = np.where(high, float(np.float32(-0.6)), KNEE)
    return (a0 >= lo) & (a0 <= hi) & np.all(side <= 0.7 + 1e-7, axis=1)


def d_checks(data, starts, anchors):
    out = {}
    for world, ds in data.items():
        arrays = ds.arrays
        require(ds.metadata["recorder_sha256"] == sha(HERE / "recorder_wide.py")
                and ds.metadata["behavior"]["name"] == "knee_s10_wide_window_v1", "Dataset was not made by the widened recorder")
        rows = M.fitting_rows(arrays)
        a0 = rows["actions"][:, 0].astype(np.float64)
        past = past_knee_a0(rows["actions"])[0]
        independent = RI.withheld_region(rows["actions"])[0]
        masks, _ = M.arm_masks(rows["obs"], rows["actions"], rows["times"], rows["episode_ids"],
                               seed=F.TEST_SEEDS["random_control"])
        cutoff = masks["cutoff"]
        high = rows["bands"] >= 1
        deep_ep = np.repeat(np.asarray(arrays["reset_component"])[rows["episodes"]] == 1, 100)
        a_obs, a_t = np.stack([a["obs"] for a in anchors]), np.asarray([a["t"] for a in anchors])
        l15_all = SM4.l15(rows["obs"], rows["episode_ids"], rows["times"], deep_ep, starts, a_obs, a_t)
        lo, hi = D4_WINDOW
        d4 = {}
        for band, name in ((1, "mid"), (2, "deep")):
            in_band = rows["bands"] == band
            count = int(np.sum(in_band & (a0 >= lo) & (a0 <= hi)))
            density = count / (int(in_band.sum()) * 0.1) if in_band.any() else 0.0
            d4[name] = {"band_rows": int(in_band.sum()), "rows_in_window": count, "density": density,
                        "bar": D4_BARS[band], "ok": bool(density >= D4_BARS[band])}
        nxt = np.asarray(arrays["observations"])[rows["episodes"], 1:100].reshape(-1, 14)   # next states of t < 99
        b1_v2 = float(np.mean(in_cutoff_support(nxt, pi_ref.act(nxt))))
        b1_v1 = float(np.mean(in_cutoff_support(nxt, pi_ref_v1_dev.act(nxt))))
        retained_high = a0[cutoff & high]
        out[world] = {
            "D1": {"cutoff_W_rows": int(np.sum(cutoff & past)), "predicates_agree": bool(np.array_equal(past, independent)),
                   "cutoff_is_complement_of_W": bool(np.array_equal(cutoff, ~past)),
                   "ok": bool(np.sum(cutoff & past) == 0 and np.array_equal(past, independent) and np.array_equal(cutoff, ~past))},
            "D2": {"cutoff_high_load_a0_at_or_past_s10": int(np.sum(retained_high <= S10)),
                   "min_distance_to_s10": float(np.min(np.abs(retained_high - S10))),
                   "ok": bool(np.sum(retained_high <= S10) == 0 and np.min(np.abs(retained_high - S10)) >= 0.0999)},
            "D3": {"start_share": l15_all["start_share"], "anchor_share": l15_all["anchor_share"], "ok": l15_all["pass"]},
            "D4": {**d4, "ok": bool(all(v["ok"] for v in d4.values()))},
            "B1": {"share_inside_cutoff_support_pi_ref_v2": b1_v2, "share_pi_ref_v1_reported": b1_v1,
                   "next_states": int(len(nxt)), "bar": B1_BAR, "ok": bool(b1_v2 >= B1_BAR)},
            "reported": {"fitting_rows": int(len(a0)), "high_load_rows": int(high.sum()),
                         "high_load_depth_ge_0p1_rows": int(np.sum(high & (a0 <= S10))),
                         "twin_W_rows": int(past.sum()), "cutoff_rows": int(cutoff.sum()),
                         "recorder_sha256": ds.metadata["recorder_sha256"]},
        }
    return out


def h_checks(anchors):
    rows = [{"reset_seed": a["reset_seed"], "current": tuple(a["current"]), "prefix_actions": a["prefix_actions"]}
            for a in anchors]
    probes = P.build(len(rows), SLOTS)
    q_st, calls, receipt = qpi.raw_values(rows, probes, pi_ref.act, "stall3", clip_all=True)
    q_ns, _, receipt_ns = qpi.raw_values(rows, probes, pi_ref.act, "nostall", clip_all=True)
    ix = {n: SLOTS.index(n) for n in SLOTS}
    rel = max(float(np.max(np.abs(q_st[:, ix[n]] - q_ns[:, ix[n]]) / np.maximum(1.0, np.abs(q_st[:, ix[n]])))) for n in S9)
    rho = {n: q_ns[:, ix[n]] - q_st[:, ix[n]] for n in ("s05", "s10", "s20", "p10")}
    med = lambda x: float(np.median(x)) if np.size(x) else None          # noqa: E731
    # projection at the first continuation state after the s10 probe (stall3)
    env = worlds.load_environment()
    ph = worlds.physics("stall3")
    x0 = np.stack([qpi.raw_anchor_state(r) for r in rows])
    t0 = np.asarray([len(r["prefix_actions"]) for r in rows])
    x1, _ = env.dynamics(x0, np.tile(np.float64(S10), (len(rows), 1)) * np.array([1, 0, 0, 0]), ph)
    obs1 = np.concatenate([x1, ((100 - (t0 + 1)) / 100)[:, None]], axis=1).astype(np.float32)
    raw1 = pi_ref.raw(obs1)
    a0_projected = raw1[:, 0] < -KNEE
    any_side = np.any(np.abs(raw1[:, 1:]) > 0.7, axis=1)
    # predictors, reported only
    res = SM4.neighbor_residuals(q_st[:, [ix[n] for n in S9]], {n: q_st[:, ix[n]] for n in GATED_PAST_KNEE},
                                 {n: q_ns[:, ix[n]] for n in ("s05", "s10", "s20")})
    preds = SM4.predictors(q_st[:, [ix[n] for n in S9]])
    pv = {k: {s: f(float(SLOT_VALUES[s])) for s in ("s10", "p10")} for k, f in preds.items()}
    min_form = np.abs(np.minimum.reduce([pv[k]["s10"] for k in pv]) - q_ns[:, ix["s10"]]) <= TOL_SIGMA
    import stage1_math as SM
    h2 = med(rho[SIGMA])
    return {
        "H1": {"max_rel_stall_minus_nostall_on_S9": rel, "tol": H1_TOL, "ok": bool(rel <= H1_TOL)},
        "H2": {"median_exact_rho_s10": h2, "n_anchors": len(rows), "R_req": R_REQ, "ok": bool(h2 >= R_REQ)},
        "reported": {
            "median_exact_rho": {n: med(rho[n]) for n in rho},
            "quantiles_exact_rho_s10": {q: float(np.quantile(rho["s10"], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            "rho_s10_split_by_a0_projection_at_first_continuation_state": {
                "projected_share": float(np.mean(a0_projected)), "median_projected": med(rho["s10"][a0_projected]),
                "median_not_projected": med(rho["s10"][~a0_projected]), "any_side_projected_share": float(np.mean(any_side))},
            "predictors": {"item5_max_form_share": float(np.mean(SM.item5_ok(res["item5_error"]))),
                           "item5_min_form_share_s10": float(np.mean(min_form)),
                           "median_rho_hat": {n: med(res["rho_hat"][n]) for n in ("s05", "s10", "s20")}},
            "item6_sign_control": {
                "max_P1_P3_share_negative": float(np.mean(np.maximum(pv["P1"]["p10"], pv["P3"]["p10"]) - q_st[:, ix["p10"]] < 0)),
                "P2_share_negative_outside_fit_range": float(np.mean(pv["P2"]["p10"] - q_st[:, ix["p10"]] < 0)),
                "exact_share_negative": float(np.mean(rho["p10"] < 0))},
            "median_abs_Q_at_edge_knee": med(np.abs(q_st[:, ix["edge_knee"]])),
            "forward_calls": int(calls), "receipts": [receipt, receipt_ns]},
    }


def mean_return(policy, starts_x):
    env = worlds.load_environment()
    ph = worlds.physics("stall3")
    x, total = starts_x.copy(), np.zeros(len(starts_x))
    for t in range(100):
        obs = np.concatenate([x, np.full((len(x), 1), (100 - t) / 100)], axis=1).astype(np.float32)
        x, r = env.dynamics(x, policy(obs).astype(np.float64), ph)
        total += 0.98 ** t * r
    return float(total.mean())


def main():
    import stage1_smoke
    require(STREAM not in stage1_smoke.sealed_values() and ENTROPY not in stage1_smoke.sealed_values(),
            "Development stream or entropy collides with a sealed value")
    require(pi_ref.gains_sha256() == pi_ref.GAINS_SHA256, "pi_ref gains differ from the pin")
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
    starts_x = E.bank_states("stage1_e2_bank", STREAM)
    h["reported"]["mean_return_e2_starts"] = {"pi_ref_v2": mean_return(pi_ref.act, starts_x),
                                              "pi_ref_v1": mean_return(pi_ref_v1_dev.act, starts_x),
                                              "zero": mean_return(lambda o: np.zeros((len(o), 4), np.float32), starts_x)}
    labels = []
    if not h["H1"]["ok"]:
        labels.append("INCONCLUSIVE (H1: the stall = no-stall identity fails; an instrument fault)")
    bad_d = [f"{w} {k}" for w in d for k in ("D1", "D2", "D3") if not d[w][k]["ok"]]
    if bad_d:
        labels.append("NOT DATA-OOD (" + ", ".join(bad_d) + ")")
    bad_t = [f"{w} {k}" for w in d for k in ("D4", "B1") if not d[w][k]["ok"]] + ([] if h["H2"]["ok"] else ["H2"])
    if bad_t:
        labels.append("NOT TESTABLE (" + ", ".join(bad_t) + ")")
    out = {"schema": "knee_s10_candidate_v1.g2_development.r3", "written_utc": datetime.now(timezone.utc).isoformat(),
           "status": "DEVELOPMENT estimate of G2 under rule r3 (same test entropy and stream as the v1 run; outcomes "
                     "of that run were known); not the formal G2; never confirmatory",
           "rule": "G2_RULE_r3.md", "rule_sha256": sha(HERE / "G2_RULE_r3.md"),
           "code_sha256": {f: sha(HERE / f) for f in ("g2_feasibility.py", "pi_ref.py", "recorder_wide.py")},
           "pi_ref_gains_sha256": pi_ref.gains_sha256(), "entropy": ENTROPY, "stream": STREAM, "episodes": EPISODES,
           "headline": labels[0] if labels else "PASS", "all_labels": labels or ["PASS"],
           "D": d, "H": h, "elapsed_seconds": time.monotonic() - started}
    (HERE / "g2_development_result_r3.json").write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps({"headline": out["headline"], "all_labels": out["all_labels"],
                      "D": {w: {k: d[w][k]["ok"] for k in ("D1", "D2", "D3", "D4", "B1")} for w in d},
                      "B1": {w: d[w]["B1"] for w in d}, "H1": h["H1"], "H2": h["H2"],
                      "reported": {k: h["reported"][k] for k in ("median_exact_rho", "quantiles_exact_rho_s10",
                                                                  "rho_s10_split_by_a0_projection_at_first_continuation_state",
                                                                  "predictors", "item6_sign_control", "mean_return_e2_starts")},
                      "elapsed": out["elapsed_seconds"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
