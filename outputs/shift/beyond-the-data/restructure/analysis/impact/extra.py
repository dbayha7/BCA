"""READ-ONLY extras on the saved terms (no fitting): refusal-zone shares, counterfactual rankings with eta frozen to
its mean, and the uniform band's eta spread on the bank."""
import json
import numpy as np
from scipy.stats import rankdata

D = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/impact"
ST2 = "/home/user/bca/runs/shift_explore/stage2/detect"
DET = "/home/user/bca/runs/shift_explore/detect"
PR = {"T1": ("command", "cap", "recorded", "cloned"), "T2": ("command", "cap", "recorded", "cloned"),
      "T3": ("command", "cap")}


def auroc(s, p):
    p = np.asarray(p, bool)
    n1, n0 = p.sum(), (~p).sum()
    r = rankdata(s)
    return float((r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def idx(arr, st, role):
    m = (arr["set"] == st) & (arr["role"] == role)
    rows = np.flatnonzero(m)
    return rows[np.argsort(arr["pair"][rows])]


out = []
for world in ("stall", "no_stall"):
    for seed in range(5):
        T = np.load(f"{D}/terms_seed{seed}_{world}.npz")
        pz = np.load(f"{DET}/seed{seed}/{world}/pairs.npz")
        arr = {k: pz[k] for k in ("set", "pair", "role")}
        lab = np.load(f"{DET}/seed{seed}/{world}/labels.npz")
        H, past = lab["harmful"].astype(bool), lab["unsupported"].astype(bool)
        b = json.load(open(f"{ST2}/weighted/{world}/fixed/seed{seed}/band.json"))["weighted_fit"]
        bu = json.load(open(f"{ST2}/uniform/{world}/fixed/seed{seed}/band.json"))["uniform"]
        l_star = b["l_star"]
        lw, eta, Rw, t1, t2, Uw, Uu = (T[k] for k in ("lw", "eta", "Rw", "term1", "term2", "Uw", "Uu"))
        unit = bu["unit"]
        o = dict(seed=seed, world=world, l_star=l_star)
        for st in ("T1", "T3"):
            ic, ip = idx(arr, st, "command"), idx(arr, st, "cap")
            sel = H[ic] if H[ic].any() else past[ic]
            o[f"{st}_refusal_zone_share_cmd"] = float(np.mean(lw[ic][sel] > l_star))
            o[f"{st}_refusal_zone_share_cap"] = float(np.mean(lw[ip][sel] > l_star))
            if st == "T1":
                ir = idx(arr, st, "recorded")
                o["T1_refusal_zone_share_rec"] = float(np.mean(lw[ir][sel] > l_star))
        t4 = arr["set"] == "T4"
        o["T4_refusal_zone_share"] = float(np.mean(lw[t4] > l_star))
        # counterfactual: eta replaced by its T4 mean (removes the residual-scale network's variation)
        eta_bar = float(np.mean(eta[t4]))
        cf = Rw * unit * eta_bar + t2
        r12 = np.concatenate([idx(arr, st, r) for st in ("T1", "T2") for r in PR[st]])
        r3 = np.concatenate([idx(arr, "T3", r) for r in PR["T3"]])
        if H[r12].any():
            o["rank_T12_weighted"] = auroc(Uw[r12], H[r12])
            o["rank_T12_eta_frozen"] = auroc(cf[r12], H[r12])
            o["rank_T3_weighted"] = auroc(Uw[r3], H[r3])
            o["rank_T3_eta_frozen"] = auroc(cf[r3], H[r3])
        # eta spread
        o["eta_T4_cv"] = float(np.std(eta[t4]) / np.mean(eta[t4]))
        o["eta_T4_q05_q50_q95"] = [float(np.quantile(eta[t4], q)) for q in (0.05, 0.5, 0.95)]
        bs = np.load(f"{ST2}/uniform/{world}/fixed/seed{seed}/bank_scores.npz")
        o["eta_bank_cv"] = float(np.std(bs["eta"]) / np.mean(bs["eta"]))
        o["eta_bank_q05_q50_q95"] = [float(np.quantile(bs["eta"], q)) for q in (0.05, 0.5, 0.95)]
        resid = bs["scores"] * np.maximum(bs["eta"], 1e-6) * unit
        o["bank_abs_resid_q50_q90"] = [float(np.quantile(resid, 0.5)), float(np.quantile(resid, 0.9))]
        o["bank_corr_eta_vs_abs_resid"] = float(np.corrcoef(bs["eta"], resid)[0, 1])
        out.append(o)
json.dump(out, open(f"{D}/extra.json", "w"), indent=1)
for o in out:
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else ([round(x, 3) for x in v] if isinstance(v, list) else v)) for k, v in o.items()}))
