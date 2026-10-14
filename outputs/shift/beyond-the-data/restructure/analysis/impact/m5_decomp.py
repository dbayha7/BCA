"""Read-only: split BCA's weighted width into its two terms using saved scores only:
U_w = R_c(lw)/R * U_uniform + U_max * softplus(lw - l*), lw = the classifier-ratio stand-in's logit (same estimator,
same init seed). Checks the reconstruction, then asks which term carries the flags at the deployment tau."""
import json
import numpy as np

KNEE = float(np.float32(0.8))
DET = "/home/user/bca/runs/shift_explore/detect/seed{k}/stall"
S2D = "/home/user/bca/runs/shift_explore/stage2/detect/{v}/stall/fixed/seed{k}"


def softplus(x):
    return np.logaddexp(0.0, x)


out = {}
for k in range(5):
    P = np.load(DET.format(k=k) + "/pairs.npz")
    sets, role, src = P["set"], P["role"], P["t3_source"]
    p80 = np.any(np.abs(np.asarray(P["commands"], np.float64)) > KNEE, axis=1)
    t3a_cmd = (sets == "T3") & (src == "actor") & (role == "command")
    t4 = sets == "T4"
    uw = np.load(S2D.format(v="weighted", k=k) + "/scores/bca_width_weighted_fixed.npz")["scores"]
    uu = np.load(S2D.format(v="uniform", k=k) + "/scores/bca_width_uniform_fixed.npz")["scores"]
    lw = np.load(DET.format(k=k) + "/scores/classifier_ratio.npz")["scores"]
    b = json.load(open(S2D.format(v="weighted", k=k) + "/band.json"))
    wf = b["weighted_fit"]
    R = b["uniform"]["R"]
    log_t, R_c, U_max, l_star = np.array(wf["log_t"]), np.array(wf["R_c"]), wf["U_max"], wf["l_star"]
    t1 = np.interp(lw, log_t, R_c) / R * uu
    t2 = U_max * softplus(lw - l_star)
    rec_err = np.abs(t1 + t2 - uw)
    tau = json.load(open(S2D.format(v="weighted", k=k) + "/tau.json"))["tau"]
    fl = uw > tau
    m_pos, m_neg = t3a_cmd & p80, t3a_cmd & ~p80
    r = dict(tau=tau, U_max=U_max, l_star=l_star, R=R,
             recon_max_abs_err=float(rec_err.max()), recon_median_abs_err=float(np.median(rec_err)),
             t4_median_t1=float(np.median(t1[t4])), t4_median_t2=float(np.median(t2[t4])),
             t4_q95_t1=float(np.quantile(t1[t4], .95)),
             pos_median_t1=float(np.median(t1[m_pos])), pos_median_t2=float(np.median(t2[m_pos])),
             neg_median_t1=float(np.median(t1[m_neg])), neg_median_t2=float(np.median(t2[m_neg])),
             # would either term alone exceed tau?
             pos_flag=float(fl[m_pos].mean()), pos_flag_t1_alone=float((t1[m_pos] > tau).mean()),
             pos_flag_t2_alone=float((t2[m_pos] > tau).mean()),
             neg_flag=float(fl[m_neg].mean()), neg_flag_t1_alone=float((t1[m_neg] > tau).mean()),
             neg_flag_t2_alone=float((t2[m_neg] > tau).mean()),
             # missed past-80% commands: are they ones the classifier itself ranks low?
             missed_pos_lw_median=float(np.median(lw[m_pos & ~fl])) if (m_pos & ~fl).any() else None,
             caught_pos_lw_median=float(np.median(lw[m_pos & fl])),
             t4_lw_q95=float(np.quantile(lw[t4], .95)), t4_lw_median=float(np.median(lw[t4])),
             share_pos_lw_below_t4_median=float((lw[m_pos] < np.median(lw[t4])).mean()),
             share_pos_uw_below_t4_median=float((uw[m_pos] < np.median(uw[t4])).mean()),
             share_pos_uu_below_t4_median=float((uu[m_pos] < np.median(uu[t4])).mean()),
             spearman_uw_lw_t3a=float(np.corrcoef(np.argsort(np.argsort(uw[t3a_cmd])),
                                                  np.argsort(np.argsort(lw[t3a_cmd])))[0, 1]),
             spearman_uw_uu_t3a=float(np.corrcoef(np.argsort(np.argsort(uw[t3a_cmd])),
                                                  np.argsort(np.argsort(uu[t3a_cmd])))[0, 1]))
    out[k] = r
    print("seed", k, json.dumps({a: (round(v, 3) if isinstance(v, float) else v) for a, v in r.items()}))
json.dump(out, open("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/impact/m5_decomp.json", "w"), indent=1)
keys = [k for k in out[0] if isinstance(out[0][k], float)]
print("pooled:", {a: round(float(np.mean([out[k][a] for k in range(5) if out[k][a] is not None])), 3) for a in keys})
