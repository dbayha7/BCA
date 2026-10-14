"""READ-ONLY (numpy on saved files). E2: does BCA's own residual |y - q| at the calibration bank vary with throttle
or current (what a localized / Mondrian threshold could pick up), and does it differ between the stalling and the
no-stall world (whether any calibration on recorded rows can see the harm)? Bank scores and eta come from the
Stage 2 uniform fixed-critic band's bank_scores.npz; residual = score * max(eta, 1e-6) * u_f (band.py:307-314)."""
import json
import numpy as np
from scipy.stats import ks_2samp

ST2 = "/home/user/bca/runs/shift_explore/stage2/detect"
OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/calibration"
CUR = 10
TB = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8 + 1e-6)]
CB = [(0.0, 0.4), (0.4, 0.6), (0.6, 0.7 + 1e-6)]
res = {}
for seed in range(5):
    per = {}
    for world in ("stall", "no_stall"):
        d = np.load(f"{OUT}/data_seed{seed}_{world}.npz")
        bs = np.load(f"{ST2}/uniform/{world}/fixed/seed{seed}/bank_scores.npz")
        unit = json.load(open(f"{ST2}/uniform/{world}/fixed/seed{seed}/band.json"))["uniform"]["unit"]
        r = bs["scores"] * np.maximum(bs["eta"], 1e-6) * unit
        a = d["bank_act"]
        thr = np.abs(a).max(1)
        cur = d["bank_obs"][:, CUR] * d["obs_std"][CUR] + d["obs_mean"][CUR]
        o = dict(n=int(len(r)), q50=float(np.quantile(r, .5)), q90=float(np.quantile(r, .9)),
                 max_throttle_max=float(thr.max()), current_max=float(cur.max()),
                 train_max_throttle_max=float(np.abs(d["train_act"]).max()),
                 train_past80_share=float(np.mean(np.abs(d["train_act"]).max(1) > np.float32(0.8))),
                 train_current_max=float((d["train_obs"][:, CUR] * d["obs_std"][CUR] + d["obs_mean"][CUR]).max()))
        bins = {}
        for lo, hi in TB:
            for clo, chi in CB:
                m = (thr >= lo) & (thr < hi) & (cur >= clo) & (cur < chi)
                bins[f"thr[{lo},{min(hi,0.8):.1f})xcur[{clo},{min(chi,0.7):.1f})"] = dict(
                    n=int(m.sum()), q90=float(np.quantile(r[m], .9)) if m.sum() >= 10 else None,
                    mean=float(r[m].mean()) if m.sum() else None)
        for lo, hi in TB:
            m = (thr >= lo) & (thr < hi)
            bins[f"thr[{lo},{min(hi,0.8):.1f})"] = dict(n=int(m.sum()), q90=float(np.quantile(r[m], .9)),
                                                         mean=float(r[m].mean()))
        for clo, chi in CB:
            m = (cur >= clo) & (cur < chi)
            bins[f"cur[{clo},{min(chi,0.7):.1f})"] = dict(n=int(m.sum()), q90=float(np.quantile(r[m], .9)),
                                                          mean=float(r[m].mean()))
        o["bins"] = bins
        o["spearman_resid_vs_throttle"] = float(np.corrcoef(np.argsort(np.argsort(r)), np.argsort(np.argsort(thr)))[0, 1])
        o["spearman_resid_vs_current"] = float(np.corrcoef(np.argsort(np.argsort(r)), np.argsort(np.argsort(cur)))[0, 1])
        per[world] = (o, r, d)
    a_s, a_n = per["stall"][2]["train_act"], per["no_stall"][2]["train_act"]
    same_rows = a_s.shape == a_n.shape and bool(np.array_equal(a_s, a_n))
    ks = ks_2samp(per["stall"][1], per["no_stall"][1])
    res[seed] = dict(stall=per["stall"][0], no_stall=per["no_stall"][0],
                     train_actions_identical_across_worlds=same_rows,
                     train_obs_identical_across_worlds=bool(np.array_equal(per["stall"][2]["train_obs"],
                                                                           per["no_stall"][2]["train_obs"])),
                     bank_resid_ks_stall_vs_nostall=dict(stat=float(ks.statistic), p=float(ks.pvalue)),
                     bank_q90_ratio_stall_over_nostall=float(per["stall"][0]["q90"] / per["no_stall"][0]["q90"]))
json.dump(res, open(f"{OUT}/e2_bank_resid.json", "w"), indent=1)
for s, v in res.items():
    print(s, "same actions", v["train_actions_identical_across_worlds"], "same obs", v["train_obs_identical_across_worlds"],
          "KS", round(v["bank_resid_ks_stall_vs_nostall"]["stat"], 3), round(v["bank_resid_ks_stall_vs_nostall"]["p"], 3),
          "q90 ratio", round(v["bank_q90_ratio_stall_over_nostall"], 3))
    for w in ("stall", "no_stall"):
        o = v[w]
        print("  ", w, "n", o["n"], "q90", round(o["q90"], 3), "maxthr", round(o["max_throttle_max"], 3),
              "train past80", o["train_past80_share"], "curmax", round(o["current_max"], 3),
              "rho thr", round(o["spearman_resid_vs_throttle"], 3), "rho cur", round(o["spearman_resid_vs_current"], 3))
        print("     ", {k: (b["n"], None if b["q90"] is None else round(b["q90"], 3)) for k, b in o["bins"].items() if "x" not in k})
