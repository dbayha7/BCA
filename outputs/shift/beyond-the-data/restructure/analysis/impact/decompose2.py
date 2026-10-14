"""READ-ONLY decomposition of the Stage 2 weighted BCA width into its two terms, from saved files only.

No training, no fitting, no band evaluation: pure numpy on saved score files.

  weighted U~ = R_c(w) * u_f * eta_f  (term1, "BCA term")  +  U_max * softplus(log w - l*)  (term2, "refusal term")
  uniform  U  = R * u_f * eta_f

eta_f is recovered from the saved uniform scores (eta_f = U / (R u_f)); log w is taken from the rehearsal's
classifier_ratio scores (same estimator, same init seed and rows as weights.py) and checked by reconstruction.
"""
import json
import sys
import numpy as np
from scipy.stats import rankdata, spearmanr

ST2 = "/home/user/bca/runs/shift_explore/stage2/detect"
DET = "/home/user/bca/runs/shift_explore/detect"
PLAN_ROLES = {"T1": ("command", "cap", "recorded", "cloned"), "T2": ("command", "cap", "recorded", "cloned"),
              "T3": ("command", "cap"), "T4": ("recorded",)}


def auroc(score, positive):
    positive = np.asarray(positive, bool)
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(np.asarray(score, np.float64))
    return float((r[positive].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def view(arr):
    out = {}
    for s in ("T1", "T2", "T3", "T4"):
        m = arr["set"] == s
        rows = np.flatnonzero(m)
        n = int(arr["pair"][m].max()) + 1
        idx = {}
        for r in set(arr["role"][rows]):
            ix = np.full(n, -1, np.int64)
            rr = rows[arr["role"][rows] == r]
            ix[arr["pair"][rr]] = rr
            idx[r] = ix
        out[s] = idx
    return out


def softplus(x):
    return np.logaddexp(0.0, x)


def one(seed, world):
    pz = np.load(f"{DET}/seed{seed}/{world}/pairs.npz")
    arr = {k: pz[k] for k in ("set", "pair", "role", "commands", "t3_source")}
    lab = np.load(f"{DET}/seed{seed}/{world}/labels.npz")
    harmful, past, over = lab["harmful"].astype(bool), lab["unsupported"].astype(bool), lab["over_pref"]
    T = np.load(f"/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/impact/terms_seed{seed}_{world}.npz"); Uu = T["Uu"]
    Uw = np.load(f"{ST2}/weighted/{world}/fixed/seed{seed}/scores/bca_width_weighted_fixed.npz")["scores"]
    lw = T["lw"]; lw_cr = np.load(f"{DET}/seed{seed}/{world}/scores/classifier_ratio.npz")["scores"]
    bu = json.load(open(f"{ST2}/uniform/{world}/fixed/seed{seed}/band.json"))["uniform"]
    bw = json.load(open(f"{ST2}/weighted/{world}/fixed/seed{seed}/band.json"))["weighted_fit"]
    tau_w = json.load(open(f"{ST2}/weighted/{world}/fixed/seed{seed}/tau.json"))["tau"]
    tau_u = json.load(open(f"{ST2}/uniform/{world}/fixed/seed{seed}/tau.json"))["tau"]
    R, unit = bu["R"], bu["unit"]
    log_t, R_c, l_star, U_max = np.array(bw["log_t"]), np.array(bw["R_c"]), bw["l_star"], bw["U_max"]
    eta = T["eta"]
    Rw = T["Rw"]
    t1 = T["term1"]
    t2 = T["term2"]
    recon = t1 + t2
    err = np.abs(recon - Uw)
    v = view(arr)
    out = dict(seed=seed, world=world, R=R, unit=unit, U_max_w=U_max, U_max_u=bu["U_max"], l_star=l_star,
               R_c_min=float(R_c.min()), R_c_max=float(R_c.max()), tau_w=tau_w, tau_u=tau_u,
               recon_max_abs_err=float(err.max()), recon_median_rel_err=float(np.median(err / np.maximum(Uw, 1e-9))),
               recon_p99_rel_err=float(np.quantile(err / np.maximum(Uw, 1e-9), 0.99)))
    # ---- T1 harmful pairs: levels and differences
    for st in ("T1", "T3"):
        ix = v[st]
        ic, ip = ix["command"], ix["cap"]
        ha = harmful[ic]
        pa = past[ic]
        sel = ha if ha.any() else pa
        tag = "harmful" if ha.any() else "past80"
        d = {}
        for nm, s in (("uniform", Uu), ("weighted", Uw), ("term1", t1), ("term2", t2), ("Rw", Rw), ("eta", eta),
                      ("lw", lw)):
            d[nm] = dict(cmd_median=float(np.median(s[ic][sel])), cap_median=float(np.median(s[ip][sel])),
                         diff_median=float(np.median(s[ic][sel] - s[ip][sel])),
                         diff_mean=float(np.mean(s[ic][sel] - s[ip][sel])),
                         widening=float(np.mean(s[ic][sel] > s[ip][sel])))
            if "recorded" in ix:
                ir = ix["recorded"]
                d[nm]["rec_median"] = float(np.median(s[ir][sel]))
        # share of the weighted cmd-cap difference from each term
        dw = Uw[ic][sel] - Uw[ip][sel]
        d1 = t1[ic][sel] - t1[ip][sel]
        d2 = t2[ic][sel] - t2[ip][sel]
        # term1's change split: R_c(w) change at fixed eta vs eta change at fixed R_c(w) (midpoint split)
        dR = (Rw[ic][sel] - Rw[ip][sel]) * unit * 0.5 * (eta[ic][sel] + eta[ip][sel])
        dE = (eta[ic][sel] - eta[ip][sel]) * unit * 0.5 * (Rw[ic][sel] + Rw[ip][sel])
        d["diff_split"] = dict(tag=tag, n=int(sel.sum()), weighted_sum=float(dw.sum()), term2_sum=float(d2.sum()),
                               term1_sum=float(d1.sum()), term1_from_Rw=float(dR.sum()),
                               term1_from_eta=float(dE.sum()),
                               term2_share=float(d2.sum() / dw.sum()), term1_share=float(d1.sum() / dw.sum()),
                               eta_share=float(dE.sum() / dw.sum()), Rw_share=float(dR.sum() / dw.sum()),
                               over_pref_median=float(np.median(over[ic][sel])))
        # level split: share of the weighted width at the command from term2
        d["level_share_term2_cmd_median"] = float(np.median(t2[ic][sel] / Uw[ic][sel]))
        d["level_share_term2_cap_median"] = float(np.median(t2[ip][sel] / Uw[ip][sel]))
        # paired size (change A1) by term
        if "recorded" in ix:
            ir = ix["recorded"]
            for nm, s in (("uniform", Uu), ("weighted", Uw), ("term1", t1), ("term2", t2), ("recon", recon)):
                W = lambda i: np.maximum(s[i] - s[ir], 0.0)  # noqa: E731
                d[f"size_paired_{nm}"] = float(np.median((W(ic)[sel] - W(ip)[sel]) / over[ic][sel]))
                d[f"size_raw_{nm}"] = float(np.median((s[ic][sel] - s[ip][sel]) / over[ic][sel]))
            # uniform: how often the recorded command's width exceeds both
            d["uniform_rec_above_both_share"] = float(np.mean((Uu[ir][sel] >= Uu[ic][sel]) & (Uu[ir][sel] >= Uu[ip][sel])))
            d["uniform_abs_diff_cmd_cap_median"] = float(np.median(np.abs(Uu[ic][sel] - Uu[ip][sel])))
            d["uniform_abs_diff_cmd_rec_median"] = float(np.median(np.abs(Uu[ic][sel] - Uu[ir][sel])))
        out[st] = d
    # ---- rankings per term (pools as measures.py)
    def pool(sets):
        return np.concatenate([np.concatenate([v[st][r] for r in PLAN_ROLES[st]]) for st in sets])
    r12, r3 = pool(("T1", "T2")), pool(("T3",))
    rk = {}
    for nm, s in (("uniform", Uu), ("weighted", Uw), ("recon", recon), ("term1", t1), ("term2", t2), ("lw", lw),
                  ("eta", eta), ("Rw", Rw), ("classifier_ratio_rehearsal", lw_cr)):
        rk[nm] = dict(T12=auroc(s[r12], harmful[r12]), T3=auroc(s[r3], harmful[r3]),
                      T12_among_past80=auroc(s[r12][past[r12]], harmful[r12][past[r12]]),
                      T3_among_past80=auroc(s[r3][past[r3]], harmful[r3][past[r3]]))
    # within-T3: subsets by source
    out["ranking"] = rk
    out["spearman_weighted_vs_lw_T12"] = float(spearmanr(Uw[r12], lw[r12]).correlation)
    out["spearman_weighted_vs_lw_T3"] = float(spearmanr(Uw[r3], lw[r3]).correlation)
    out["spearman_weighted_vs_uniform_T3"] = float(spearmanr(Uw[r3], Uu[r3]).correlation)
    # T3: who are the harmless pairs that the weighted version ranks high, vs classifier
    ix = v["T3"]
    # share of pool pairs with lw above l* (refusal term in its linear regime)
    out["share_lw_above_lstar_T3_pool"] = float(np.mean(lw[r3] > l_star))
    out["share_lw_above_lstar_T12_pool"] = float(np.mean(lw[r12] > l_star))
    t4 = arr["set"] == "T4"
    out["T4"] = dict(uniform_median=float(np.median(Uu[t4])), weighted_median=float(np.median(Uw[t4])),
                     term2_share_median=float(np.median(t2[t4] / Uw[t4])),
                     uniform_q05_q95=[float(np.quantile(Uu[t4], .05)), float(np.quantile(Uu[t4], .95))],
                     lw_median=float(np.median(lw[t4])))
    # T3 within-state contribution vs across-state in the weighted score (variance decomposition over T3 command/cap)
    ic, ip = ix["command"], ix["cap"]
    for nm, s in (("uniform", Uu), ("weighted", Uw), ("lw", lw), ("term1", t1), ("term2", t2)):
        pair_mean = 0.5 * (s[ic] + s[ip])
        within = 0.5 * (s[ic] - s[ip])
        out.setdefault("T3_variance", {})[nm] = dict(across_state_var=float(np.var(pair_mean)),
                                                      within_state_var=float(np.mean(within ** 2)))
    ic1, ip1, ir1 = v["T1"]["command"], v["T1"]["cap"], v["T1"]["recorded"]
    for nm, s in (("uniform", Uu), ("weighted", Uw), ("lw", lw)):
        pair_mean = (s[ic1] + s[ip1] + s[ir1]) / 3
        within = np.stack([s[ic1], s[ip1], s[ir1]]) - pair_mean
        out.setdefault("T1_variance", {})[nm] = dict(across_state_var=float(np.var(pair_mean)),
                                                      within_state_var=float(np.mean(within ** 2)))
    return out


if __name__ == "__main__":
    res = []
    for world in ("stall", "no_stall"):
        for seed in range(5):
            res.append(one(seed, world))
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    print("written", sys.argv[1])
