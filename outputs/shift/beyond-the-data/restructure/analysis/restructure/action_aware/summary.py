"""READ-ONLY summary of ens_analyse.json, nl_posterior.json, shape_sweep.json, pairs_slope.json -> summary.json.
Ranges are min-max over the seeds available (per world)."""
import json

D = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/action_aware"
ens = json.load(open(f"{D}/ens_analyse.json"))
nl = json.load(open(f"{D}/nl_posterior.json"))
sh = json.load(open(f"{D}/shape_sweep.json"))
ps = json.load(open(f"{D}/pairs_slope.json"))


def rng(vals):
    vals = [v for v in vals if v is not None]
    return [round(min(vals), 4), round(max(vals), 4)] if vals else None


out = {}
for world in ("stall", "no_stall"):
    E = [r for r in ens if r["world"] == world]
    o = {"T1_over_pref_q50": rng([r["T1_over_pref_median"] for r in E]),
         "T1_critic_pref_q50": rng([r["T1_critic_pref_median"] for r in E]),
         "T3_over_pref_q50": rng([r["T3_over_pref_median"] for r in E]),
         "T3_critic_pref_q50": rng([r["T3_critic_pref_median"] for r in E]),
         "T1_full_critic_over_pref_q50": rng([r["T1_full_critic_over_pref_median"] for r in E])}
    for en in E[0]:
        if not isinstance(E[0][en], dict) or not en.startswith("E"):
            continue
        e = {}
        for st in ("T1", "T3"):
            e[st] = dict(sel=E[0][en][st]["sel"],
                         ds_q50=rng([r[en][st]["ds_q10_q50_q90"][1] for r in E]),
                         s_cmd_q50=rng([r[en][st]["s_cmd_q10_q50_q90"][1] for r in E]),
                         s_cap_q50=rng([r[en][st]["s_cap_q10_q50_q90"][1] for r in E]),
                         ds_over_over_pref_q50=rng([r[en][st]["ds_over_over_pref_median"] for r in E]),
                         k_need_q50=rng([r[en][st]["k_need_median"] for r in E]),
                         flip_k1=rng([r[en][st]["flip_share_k1"] for r in E]),
                         flip_k5=rng([r[en][st]["flip_share_k5"] for r in E]),
                         good_wrong_flip_k1=rng([r[en][st].get("good_wrong_flip_share_k1") for r in E]),
                         good_wrong_flip_k5=rng([r[en][st].get("good_wrong_flip_share_k5") for r in E]),
                         auroc_harm_among_past80=rng([r[en][st].get("auroc_harm_among_past80_ds") for r in E]))
            if st == "T1":
                e[st]["s_rec_q50"] = rng([r[en][st]["s_rec_q10_q50_q90"][1] for r in E])
                e[st]["paired_dP_q50"] = rng([r[en][st]["paired_dP_q10_q50_q90"][1] for r in E])
        e["s_T4_q50"] = rng([r[en]["s_T4_recorded_q10_q50_q90"][1] for r in E])
        o[en] = e
    N = [r for r in nl if r["world"] == world]
    o["NL_seeds"] = [r["seed"] for r in N]
    for c in ("c1e-06", "c0.0001", "c0.01"):
        o[f"NL_{c}"] = {st: dict(ds_q50=rng([r[c][st]["ds_q10_q50_q90"][1] for r in N]),
                                 s_cmd_q50=rng([r[c][st]["s_cmd_q10_q50_q90"][1] for r in N]),
                                 lev_ratio_cmd=rng([r[c][st]["lev_ratio_cmd_over_T4_median"] for r in N]),
                                 lev_ratio_cap=rng([r[c][st]["lev_ratio_cap_over_T4_median"] for r in N]),
                                 flip_k10=rng([r[c][st]["flip_share_k10"] for r in N]))
                        for st in ("T1", "T3")}
        o[f"NL_{c}"]["s_T4_q50"] = rng([r[c]["s_T4_q10_q50_q90"][1] for r in N])
    S = [r for r in sh if r["world"] == world]
    o["shape"] = {"q1_alone_best_past80": rng([r["q1_alone_best_past80_share"] for r in S])}
    for f in S[0]["forms"]:
        o["shape"][f] = {k: rng([r["forms"][f][k] for r in S]) for k in
                         ("span_0p8_to_1p0", "slope_70_80", "slope_80_85", "slope_85_95", "best_past80_share_k1",
                          "best_past80_share_k2", "best_past80_share_k4", "best_ge95_share_k1")}
    P = [r for r in ps if r["world"] == world]
    o["slope_pairs"] = {st: dict(sel=P[0][st]["sel"], dU_q50=rng([r[st]["dU_q10_q50_q90"][1] for r in P]),
                                 dU_over_critic_pref=rng([r[st]["dU_over_critic_pref_q50"] for r in P]),
                                 flip_k1=rng([r[st]["flip_share_k1"] for r in P]),
                                 flip_k1p5=rng([r[st]["flip_share_k1.5"] for r in P]),
                                 good_n=rng([r[st].get("good_n") for r in P]),
                                 good_wrong_flip_k1=rng([r[st].get("good_wrong_flip_share_k1") for r in P]),
                                 auroc_harm_among_past80=rng([r[st].get("auroc_harm_among_past80_dU") for r in P]))
                        for st in ("T1", "T3")}
    o["E2_conformal_multiplier_estimate"] = rng([r["E2_conformal_multiplier_estimate"] for r in P])
    out[world] = o
json.dump(out, open(f"{D}/summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
