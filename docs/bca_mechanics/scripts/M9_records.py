"""M9 records: gather every recorded number the lower-edge tests quote into one JSON file, with its source.

Reads (all read only; nothing is trained or recomputed here):
  --runs   the run-file root (default: the owner's WSL clone over \\\\wsl.localhost, read only)
             stage2/policy/seed<K>/{lock,values}.json and .../no_stall/{lock,values}.json   (Stage 2, seeds 0-4)
             stage2/detect/{uniform,weighted}/stall/fixed/seed<K>/band.json                 (Stage 2's bands)
             near80/policy/near80/{stall,no_stall}/seed<K>/lock.json                         (near-80, seeds 0-4)
             near80/learners/near80/{stall,no_stall}/seed<K>/v6_band.json                    (version 6's band)
  --repo   the repository root (default: three levels above this file)
             outputs/shift/beyond-the-data/near80/results-20261020T115246Z/files/looks/exploratory.json
             outputs/shift/beyond-the-data/results-2026-10-14/stage2_pooled_exploratory.json
             git show origin/main:outputs/shift_iw/v5/<quality>/report.json   (the weighting test, marks L1-L3)
Writes --out (default docs/bca_mechanics/data/M9/M9_records.json).

Run (Windows, any Python 3.10+ with the standard library; git on PATH):
    python docs/bca_mechanics/scripts/M9_records.py
The near-80 policy locks are also on git branch claude/near80-run-files under runs/shift_explore/near80/policy/.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
WSL = r"\\wsl.localhost\Ubuntu-24.04\home\user\bca\runs\shift_explore"
QUALITIES = ("expert", "medium-expert", "medium", "medium-replay", "random")
SITS = ("recorded", "strong_current", "big_waves", "big_kicks")


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def git_json(repo, spec):
    out = subprocess.run(["git", "-C", repo, "show", spec], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def stage2(runs):
    out = {"source": "runs/shift_explore/stage2/policy/seed<K>/{lock,values}.json (stall) and .../no_stall/ "
                     "(no-stall world); the arm 'bca_k1' is the paired lower edge at k = 1 on the classifier-weighted "
                     "band (lock.version 'weighted', arm kind 'bca_w'); strengths = root-mean-square command change "
                     "from the no-caution actor on 5,000 held-out states; chunk_metrics = [actor_loss, Q1 mean, "
                     "lambda, cloning term, k*W mean] per 4,000-step chunk; points per situation from values.json",
           "seeds": {}}
    for world, sub in (("stall", ""), ("no_stall", "no_stall")):
        for k in range(5):
            d = os.path.join(runs, "stage2", "policy", f"seed{k}", sub)
            lock, vals = load(os.path.join(d, "lock.json")), load(os.path.join(d, "values.json"))
            arm = lock["arms"]["bca_k1"]
            ev = vals["evaluations"]
            rec = dict(version=lock.get("version"), kind=arm["arm"]["kind"], coef=arm["arm"]["coef"],
                       metric_names=arm["metric_names"], chunk_metrics=arm["chunk_metrics"],
                       check_b=lock["checks"]["b"],
                       strengths={n: lock["strengths"].get(n) for n in
                                  ("bca_k1", "bca_k2", "bca_unpaired", "bca_shuffled", "perfect_band",
                                   "range_hinge", "w0_check")},
                       matched={n: m.get("arm") for n, m in lock.get("matches", {}).items()},
                       points={a: {s: ev[a][s]["points"] for s in SITS if s in ev[a]}
                               for a in ("none", "clone", "full_none", "bca_k1", "bca_k2", "bca_unpaired",
                                         "bca_shuffled", "perfect_band", "range_hinge") if a in ev})
            for n, m in lock.get("matches", {}).items():
                a = m.get("arm")
                if a and a in ev:
                    rec["points"][a] = {s: ev[a][s]["points"] for s in SITS if s in ev[a]}
                    rec["strengths"][a] = lock["strengths"].get(a)
            if "bca_penalty_shape" in lock:
                sh = lock["bca_penalty_shape"]
                rec["penalty_shape"] = {k2: sh[k2] for k2 in ("rows", "mean_W", "fall_share_80_to_81",
                                                             "rise_share_80_to_81", "best_past80_share",
                                                             "q1_alone_best_past80_share", "mean_W_full_minus_80")
                                        if k2 in sh}
            bands = {}
            for ver in ("uniform", "weighted"):
                bj = os.path.join(runs, "stage2", "detect", ver, world, "fixed", f"seed{k}", "band.json")
                if os.path.exists(bj):
                    u = load(bj).get("uniform", {})
                    bands[ver] = {k2: u.get(k2) for k2 in ("R", "lambda_hat", "unit", "U_max", "bank_rows")}
            rec["bands"] = bands
            out["seeds"][f"{world}/{k}"] = rec
    return out


def near80(runs, repo):
    out = {"source": "runs/shift_explore/near80/policy/near80/<world>/seed<K>/lock.json (also on branch "
                     "claude/near80-run-files); the arm 'version6_bca' is the paired lower edge at k = 1 on the "
                     "uniform (standard) band v6_band (kind 'bca_u'); points from "
                     "outputs/shift/beyond-the-data/near80/results-20261020T115246Z/files/looks/exploratory.json "
                     "(reported.measures.points.near80); strengths and chunk_metrics as in stage 2",
           "seeds": {}}
    ex = load(os.path.join(repo, "outputs", "shift", "beyond-the-data", "near80", "results-20261020T115246Z",
                           "files", "looks", "exploratory.json"))
    pts = ex["reported"]["measures"]["points"]["near80"]
    for world in ("stall", "no_stall"):
        for k in range(5):
            lock = load(os.path.join(runs, "near80", "policy", "near80", world, f"seed{k}", "lock.json"))
            arm = lock["arms"]["version6_bca"]
            band = load(os.path.join(runs, "near80", "learners", "near80", world, f"seed{k}", "v6_band.json"))
            u = band.get("uniform", {})
            rec = dict(kind=arm["arm"]["kind"], coef=arm["arm"]["coef"], metric_names=arm["metric_names"],
                       chunk_metrics=arm["chunk_metrics"], check_b_w0=lock["checks"].get("b_w0"),
                       strengths={n: lock["strengths"].get(n) for n in
                                  ("version6_bca", "obca_k1", "obca_forced", "range_hinge", "perfect_band",
                                   "knee_ramp", "w0_check")},
                       band={k2: u.get(k2) for k2 in ("R", "lambda_hat", "unit", "U_max", "bank_rows", "certified")},
                       points={a: {s: pts[world][a][s]["per_seed"][k] for s in SITS if s in pts[world][a]}
                               for a in ("none", "version6_bca", "obca_k1", "range_hinge", "perfect_band", "clone",
                                         "w0_check") if a in pts[world]})
            out["seeds"][f"{world}/{k}"] = rec
    return out


def weighting_test(repo):
    out = {"source": "git show origin/main:outputs/shift_iw/v5/<quality>/report.json, marks L1 (the oracle lower "
                     "edge's room check), L2 (the predicted return of the weighted lower edge and the proxy check of "
                     "BCA's penalty against the oracle's) and L3 (BCA's lower-edge arms; never run)",
           "qualities": {}}
    for q in QUALITIES:
        r = git_json(repo, f"origin/main:outputs/shift_iw/v5/{q}/report.json")
        m = r["marks"]
        L1, L2, L3 = m["L1"], m["L2"], m.get("L3", {})
        pc = L2.get("proxy_check", {})
        out["qualities"][q] = dict(
            L1=dict(status=L1.get("status"), oracle_minus_none_mean=L1["oracle_minus_none"]["mean"],
                    oracle_minus_none_per_seed=L1["oracle_minus_none"]["per_seed"],
                    oracle_minus_none_points=L1.get("oracle_minus_none_points"),
                    check_e_passed=[c.get("passed") for c in L1.get("check_e", [])],
                    le_none_strength=L1.get("le_none_strength")),
            L2=dict(status=L2.get("status"), predicted_return=L2.get("predicted_return"),
                    predicted_points=L2.get("predicted_points"),
                    scale_change_mean=L2["scale_change"]["mean"],
                    high_current_scale_change_mean=L2["high_current_scale_change"]["mean"],
                    high_current_scale_change_per_seed=L2["high_current_scale_change"]["per_seed"],
                    refused_rows=L2.get("refused_rows"),
                    proxy_check=dict(band_mean=pc.get("band_mean"), oracle_mean=pc.get("oracle_mean"),
                                     band_over_oracle_per_seed=pc.get("band_over_oracle", {}).get("per_seed"),
                                     pearson_per_seed=pc.get("pearson", {}).get("per_seed"),
                                     spearman_per_seed=pc.get("spearman", {}).get("per_seed"),
                                     band_check_e_passed=pc.get("band_check_e_passed"),
                                     band_check_e_failed=pc.get("band_check_e_failed"))),
            L3=dict(status=L3.get("status"), why=L3.get("why")),
            thresholds={k: v for k, v in r.get("thresholds", {}).items() if k in ("L1", "L2", "L3")})
    return out


def stage2_pooled(repo):
    p = load(os.path.join(repo, "outputs", "shift", "beyond-the-data", "results-2026-10-14",
                          "stage2_pooled_exploratory.json"))
    pol = p["policy"]
    return {"source": "outputs/shift/beyond-the-data/results-2026-10-14/stage2_pooled_exploratory.json (policy)",
            "M1_even_pull": pol["M1_even_pull"], "M2_random_surfaces": pol["M2_random_surfaces"],
            "M3_keeps_improvement": dict(pol["M3_keeps_improvement"]),
            "M4_best_standin": pol["M4_best_standin"], "M5_deployment_check": pol["M5_deployment_check"],
            "claim_numbers": pol["claim_numbers"], "claim_sentence": pol["claim_sentence"],
            "no_stall_cost": pol["no_stall_cost"], "passed": pol["passed"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=WSL)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "bca_mechanics", "data", "M9", "M9_records.json"))
    a = ap.parse_args()
    rec = dict(about="Recorded inputs and outputs of module M9 (the lower edge), gathered by "
                     "docs/bca_mechanics/scripts/M9_records.py on 20 October 2026; every number is copied from the "
                     "named record, none is computed here",
               stage2=stage2(a.runs), stage2_pooled=stage2_pooled(a.repo), near80=near80(a.runs, a.repo),
               weighting_test=weighting_test(a.repo))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    print("wrote", a.out, os.path.getsize(a.out), "bytes")
    print("stage 2 seed-worlds:", len(rec["stage2"]["seeds"]), "| near-80 seed-worlds:", len(rec["near80"]["seeds"]),
          "| weighting-test qualities:", len(rec["weighting_test"]["qualities"]))


if __name__ == "__main__":
    sys.exit(main())
