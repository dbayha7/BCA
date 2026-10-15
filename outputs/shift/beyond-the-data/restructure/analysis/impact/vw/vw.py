"""READ-ONLY independent re-check of the 'weighting' analysis. Forward passes of the SAVED frozen networks in each
band.msgpack (no training, no fitting); everything else is numpy on saved files. Writes only to this folder."""
import json
import sys

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import serialization
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, "/home/user/bca")
from calibration.network import Calibrator  # noqa: E402

ST2 = "/home/user/bca/runs/shift_explore/stage2/detect"
DET = "/home/user/bca/runs/shift_explore/detect"
OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/impact/vw"
ROLES = {"T1": ("command", "cap", "recorded", "cloned"), "T2": ("command", "cap", "recorded", "cloned"),
         "T3": ("command", "cap")}


class Cls(nn.Module):  # stage2/code/weights.py ClassifierNet, restated
    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs / 1.001, action], axis=-1)
        x = nn.relu(nn.Dense(64)(x))
        x = nn.relu(nn.Dense(64)(x))
        return nn.Dense(1)(x).squeeze(-1)


CAL, CLS = Calibrator(jnp.zeros(13), jnp.ones(13), state_dep=True), Cls()


def auroc(s, p):
    p = np.asarray(p, bool)
    n1, n0 = p.sum(), (~p).sum()
    r = rankdata(np.asarray(s, np.float64))
    return float((r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def view(arr):
    out = {}
    for st in ("T1", "T2", "T3"):
        m = arr["set"] == st
        out[st] = {}
        for r in np.unique(arr["role"][m]):
            rows = np.flatnonzero(m & (arr["role"] == r))
            out[st][r] = rows[np.argsort(arr["pair"][rows])]
    return out


def one(seed, world):
    bu_rec = json.load(open(f"{ST2}/uniform/{world}/fixed/seed{seed}/band.json"))
    bw_rec = json.load(open(f"{ST2}/weighted/{world}/fixed/seed{seed}/band.json"))
    mean = np.asarray(bu_rec["data"]["obs_mean"], np.float32)
    std = np.asarray(bu_rec["data"]["obs_std"], np.float32)
    bw = serialization.msgpack_restore(open(f"{ST2}/weighted/{world}/fixed/seed{seed}/band.msgpack", "rb").read())
    bu = serialization.msgpack_restore(open(f"{ST2}/uniform/{world}/fixed/seed{seed}/band.msgpack", "rb").read())
    pz = np.load(f"{DET}/seed{seed}/{world}/pairs.npz")
    arr = {k: pz[k] for k in ("set", "pair", "role")}
    o = jnp.asarray((np.asarray(pz["states"], np.float32) - mean) / std)
    a = jnp.asarray(np.asarray(pz["commands"], np.float32))
    eta_w = np.asarray(CAL.apply(bw["cal_params"], o, a), np.float64)
    eta_u = np.asarray(CAL.apply(bu["cal_params"], o, a), np.float64)
    lw = np.asarray(CLS.apply(bw["cls_params"], o, a), np.float64)
    unit, Umax, ls = float(bw["unit"]), float(bw["U_max"]), float(bw["l_star"])
    Rw = np.interp(lw, np.asarray(bw["log_t"], np.float64), np.asarray(bw["R_tab"], np.float64))
    e = np.maximum(eta_w, 1e-6)
    t1 = Rw * unit * e
    t2 = Umax * np.logaddexp(0.0, lw - ls)
    Uw = np.load(f"{ST2}/weighted/{world}/fixed/seed{seed}/scores/bca_width_weighted_fixed.npz")["scores"]
    Uu = np.load(f"{ST2}/uniform/{world}/fixed/seed{seed}/scores/bca_width_uniform_fixed.npz")["scores"]
    Uu_re = float(bu["R"]) * float(bu["unit"]) * np.maximum(eta_u, 1e-6)
    cr = np.load(f"{DET}/seed{seed}/{world}/scores/classifier_ratio.npz")["scores"]
    lab = np.load(f"{DET}/seed{seed}/{world}/labels.npz")
    H, past, over = lab["harmful"].astype(bool), lab["unsupported"].astype(bool), lab["over_pref"].astype(float)
    v = view(arr)
    t4 = arr["set"] == "T4"
    r = dict(seed=seed, world=world,
             recon_w_maxabs=float(np.abs(t1 + t2 - Uw).max()), recon_u_maxabs=float(np.abs(Uu_re - Uu).max()),
             eta_w_vs_u_maxabs=float(np.abs(eta_w - eta_u).max()),
             lw_vs_cr_spearman=float(spearmanr(lw, cr).correlation), lw_vs_cr_medabs=float(np.median(np.abs(lw - cr))),
             cr_init_seed=json.load(open(f"{DET}/seed{seed}/{world}/scores/classifier_ratio.json"))
             .get("signal_info", {}).get("init_seed"),
             bw_init_seed=bw_rec["weighted_fit"]["init_seed"], l_star=ls, U_max=Umax, unit=unit,
             R_uniform=float(bu["R"]))
    ic, ip, ir = v["T1"]["command"], v["T1"]["cap"], v["T1"]["recorded"]
    sel = H[ic] if H[ic].any() else past[ic]
    r["T1_sel_n"] = int(sel.sum())
    r["T1_sel_is_harmful"] = bool(H[ic].any())
    # uniform levels and gaps
    for nm, idx in (("cmd", ic), ("cap", ip), ("rec", ir)):
        r[f"U_uniform_{nm}_median"] = float(np.median(Uu[idx][sel]))
    r["U_uniform_absdiff_cmd_cap_median"] = float(np.median(np.abs(Uu[ic] - Uu[ip])[sel]))
    r["over_pref_median"] = float(np.median(over[ic][sel]))
    r["rec_ge_both_share"] = float(np.mean((Uu[ir] >= Uu[ic]) & (Uu[ir] >= Uu[ip]) & sel) / np.mean(sel))
    # uniform: share narrower / wider at command than cap
    r["uniform_T1_wider_share"] = float(np.mean((Uu[ic] > Uu[ip])[sel]))
    r["uniform_T1_narrower_share"] = float(np.mean((Uu[ic] < Uu[ip])[sel]))
    # exact midpoint split of the cmd - cap difference of the weighted width (sum over selected pairs)
    for st in ("T1", "T3"):
        c, p = v[st]["command"], v[st]["cap"]
        s2 = H[c] if H[c].any() else past[c]
        dW = (Uw[c] - Uw[p])[s2].sum()
        d2 = (t2[c] - t2[p])[s2].sum()
        dR = ((Rw[c] - Rw[p]) * unit * 0.5 * (e[c] + e[p]))[s2].sum()
        dE = ((e[c] - e[p]) * unit * 0.5 * (Rw[c] + Rw[p]))[s2].sum()
        r[f"{st}_split"] = dict(term2=float(d2 / dW), Rw=float(dR / dW), eta=float(dE / dW),
                                closure=float((d2 + dR + dE) / dW))
        r[f"{st}_refusal_cmd"] = float(np.mean((lw[c] > ls)[s2]))
        r[f"{st}_refusal_cap"] = float(np.mean((lw[p] > ls)[s2]))
        # signed share of pairs where eta moves the "wrong" way (narrower at command)
        r[f"{st}_eta_cmd_lt_cap_share"] = float(np.mean((eta_u[c] < eta_u[p])[s2]))
    r["T1_refusal_rec"] = float(np.mean((lw[ir] > ls)[sel]))
    r["T4_refusal"] = float(np.mean(lw[t4] > ls))
    # paired size by term
    def size(s):
        W = lambda i: np.maximum(s[i] - s[ir], 0.0)  # noqa: E731
        return float(np.median(((W(ic) - W(ip)) / over[ic])[sel]))
    r["size_weighted"], r["size_term2"], r["size_term1"], r["size_uniform"] = size(Uw), size(t2), size(t1), size(Uu)
    # rankings per term
    r12 = np.concatenate([v[st][ro] for st in ("T1", "T2") for ro in ROLES[st]])
    r3 = np.concatenate([v["T3"][ro] for ro in ROLES["T3"]])
    eta_bank = np.load(f"{ST2}/uniform/{world}/fixed/seed{seed}/bank_scores.npz")["eta"]
    cf_T4 = Rw * unit * float(np.mean(e[t4])) + t2
    cf_bank = Rw * unit * float(np.mean(np.maximum(eta_bank, 1e-6))) + t2
    if H[r12].any():
        for nm, s in (("weighted", Uw), ("uniform", Uu), ("term1", t1), ("term2", t2), ("lw", lw), ("cr", cr),
                      ("eta", eta_u), ("cf_T4", cf_T4), ("cf_bank", cf_bank)):
            r[f"rank_{nm}"] = [auroc(s[r12], H[r12]), auroc(s[r3], H[r3]),
                              auroc(s[r12][past[r12]], H[r12][past[r12]]), auroc(s[r3][past[r3]], H[r3][past[r3]])]
    # eta spread
    eb = eta_bank
    r["eta_bank_cv"] = float(np.std(eb) / np.mean(eb))
    r["eta_T4_cv"] = float(np.std(eta_u[t4]) / np.mean(eta_u[t4]))
    r["eta_bank_q05_q95"] = [float(np.quantile(eb, .05)), float(np.quantile(eb, .95))]
    r["eta_T4_q05_q95"] = [float(np.quantile(eta_u[t4], .05)), float(np.quantile(eta_u[t4], .95))]
    bs = np.load(f"{ST2}/uniform/{world}/fixed/seed{seed}/bank_scores.npz")["scores"]
    resid = bs * np.maximum(eb, 1e-6) * float(bu["unit"])
    r["corr_eta_absresid_bank"] = float(np.corrcoef(eb, resid)[0, 1])
    r["spearman_eta_absresid_bank"] = float(spearmanr(eb, resid).correlation)
    # T1 variance across vs within state
    for nm, s in (("uniform", Uu), ("weighted", Uw), ("lw", lw)):
        pm = (s[ic] + s[ip] + s[ir]) / 3
        wi = np.stack([s[ic], s[ip], s[ir]]) - pm
        r[f"T1var_{nm}"] = [float(np.var(pm)), float(np.mean(wi ** 2))]
    # level share of term2 at harmful commands
    r["T1_term2_level_share_cmd_median"] = float(np.median((t2[ic] / Uw[ic])[sel]))
    return r


if __name__ == "__main__":
    res = [one(s, w) for w in ("stall", "no_stall") for s in range(5)]
    json.dump(res, open(f"{OUT}/vw.json", "w"), indent=1)
    print("ok")
