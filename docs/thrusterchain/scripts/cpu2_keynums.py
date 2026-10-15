"""CPU 2: key v5 numbers per quality, read only."""
import json
QS = ["random", "medium-replay", "medium", "medium-expert", "expert"]
ROOT = "/root/bca-shift/runs/shift_iw_v5"
def m(x): return x["mean"] if isinstance(x, dict) and "mean" in x else x
def rng(x):
    ps = x.get("per_seed") if isinstance(x, dict) else None
    return (round(min(ps),4), round(max(ps),4)) if ps else None
for q in QS:
    r = json.load(open(f"{ROOT}/{q}/report/report.json"))
    M = r["marks"]; S = r["shown"]
    print("=====", q, "bca_allowed", r.get("bca_allowed"), "le_allowed", r.get("bca_lower_edge_allowed"))
    print(" status", {k: v["status"] for k, v in M.items()})
    print(" M1 cov", round(m(M["M1"]["coverage"]),4), "lo", round(M["M1"]["coverage"]["lower_one_sided"],4), "seeds", rng(M["M1"]["coverage"]))
    print(" M2 within_deciles", round(m(M["M2"]["within_deciles"]),4), "lo", round(M["M2"]["within_deciles"]["lower_one_sided"],4), " given radius", round(m(M["M2"]["given_float_radius_within_deciles"]),4), "lo", round(M["M2"]["given_float_radius_within_deciles"]["lower_one_sided"],4))
    print(" M3 inversions", [p["inversions"] for p in M["M3"]["per_seed"]], "maxdrop", [p["max_drop"] for p in M["M3"]["per_seed"]])
    for t in ["mild","strong"]:
        c = M["W1"]["coverage"][t]; print(" W1", t, round(c["mean"],4), "lo", round(c["lower_one_sided"],4), "seeds", rng(c))
    c = M["W2"]["coverage"]; print(" W2 unweighted strong", round(c["mean"],4), "seeds", rng(c))
    w3 = M["W3"]
    for t in ["none","mild","strong"]:
        tt = w3["tilts"][t]
        u = tt["unweighted"]
        print(" W3", t, "exact refused inside", tt["exact"]["refused_inside"], "outside", tt["exact"]["refused_outside"], "est inside", tt["estimated"]["refused_inside"], "est outside", tt["estimated"]["refused_outside"])
        print("    unweighted inside", None if not isinstance(u.get("inside"), dict) else round(u["inside"]["mean"],4), "outside", None if not isinstance(u.get("outside"), dict) else round(u["outside"]["mean"],4), "judged", u.get("judged"), "lower_inside", u.get("lower_inside"), "in_ball_steps", u.get("in_ball_steps"), "note", u.get("note"))
    print(" W3 status", w3["status"], "judged", w3.get("inside_versus_outside_judged"))
    print(" W4 strong est-exact", M["W4"].get("strong_estimated_minus_exact"), "none est-unw", M["W4"].get("none_estimated_minus_unweighted"))
    R1 = M["R1"]; print(" R1 placed-none", round(m(R1["placed_minus_none"]),3), "lo", round(R1["placed_minus_none"]["lower_one_sided"],3), "pts", round(m(R1["placed_minus_none_points"]),3), "pts seeds", rng(R1["placed_minus_none_points"]), " placed-even", None if R1.get("placed_minus_matched_even") is None else round(m(R1["placed_minus_matched_even"]),3), "even note", R1.get("matched_even_note"))
    R2 = M["R2"]; print(" R2 dose change", round(m(R2["dose_change"]),4), "seeds", rng(R2["dose_change"]), "ret/unit dose", round(m(R2["return_per_unit_dose"]),2), "pred return", R2.get("predicted_return"), "pred points", R2.get("predicted_points"), "refused rows", R2.get("refused_rows"))
    L1 = M["L1"]; print(" L1 oracle-none", round(m(L1["oracle_minus_none"]),3), "lo", round(L1["oracle_minus_none"]["lower_one_sided"],3), "pts", round(m(L1["oracle_minus_none_points"]),3), "pts seeds", rng(L1["oracle_minus_none_points"]), "oracle-even", round(m(L1["oracle_minus_matched_even"]),2), "e4 n/a", L1.get("e4_not_applicable_seeds"), "check_e passed", [c["passed"] for c in L1["check_e"]])
    L2 = M["L2"]; print(" L2 scale change", round(m(L2["scale_change"]),4), "seeds", rng(L2["scale_change"]), "ret/unit k", round(m(L2["return_per_unit_k"]),3), "seeds", rng(L2["return_per_unit_k"]), "pts/unit k", round(m(L2["points_per_unit_k"]),3), "pred return", L2.get("predicted_return"), "pred pts", L2.get("predicted_points"))
    pc = L2["proxy_check"]; print("    proxy band/oracle", round(m(pc["band_over_oracle"]),4), "seeds", rng(pc["band_over_oracle"]), "pearson", round(m(pc["pearson"]),3), "spearman", round(m(pc["spearman"]),3), "band check passed", pc.get("band_check_e_passed"))
    print(" shown keys", list(S.keys()))
    for k in ["plain_td3bc_points"]:
        for v in ["full","gap"]:
            print("  ", k, v, {t: round(m(S[k][v][t]),2) for t in ["none","mild","strong"]})
    print("  full-gap strong pts", round(m(S["full_minus_gapped_plain_td3bc_strong"]),3))
    rp = S["room_points"]; print("  room pts strong", {a: (None if m(rp[a]["strong"]) is None else round(m(rp[a]["strong"]),2)) for a in rp})
    le = S["lower_edge"]["room_points"]; print("  LE pts strong", {a: (None if m(le[a]["strong"]) is None else round(m(le[a]["strong"]),2)) for a in le})
