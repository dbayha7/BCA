"""READ-ONLY analysis of q_seed*_*.npz (ens_pairs.py) against the locked labels. Pure numpy; no fitting.

For an ensemble E of critic heads: m_E = mean, s_E = sample std (ddof=1) over heads.
  ds      = s_E(s, command) - s_E(s, cap)                  (the width's response to the command, in critic units)
  k_need  = over_pref / ds                                 (strength k at which k*ds removes the false preference)
  flip_k  = share of pairs where Q1-pref - k*ds <= 0       (the penalised actor would now take the cap; k = 1, 2, 5)
Paired (BCA's lower-edge form, T1 only): P(a) = [U(s, a) - U(s, a_rec)]_+, U = s_E; dP = P(cmd) - P(cap).
"""
import json

import numpy as np
from scipy.stats import rankdata

D = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/action_aware"
DET = "/home/user/bca/runs/shift_explore/detect"
ENS = {"stall": {"E2_gap_two_heads": [("stall_aware_gap", 0), ("stall_aware_gap", 1)],
                 "EI_full_two_inits": [("stall_aware_full", 0), ("stall_aware_full", 1),
                                       ("stall_aware_full_placebo", 0), ("stall_aware_full_placebo", 1)],
                 "EG_gap_plus_full": [("stall_aware_gap", 0), ("stall_aware_gap", 1), ("stall_aware_full", 0),
                                      ("stall_aware_full", 1)],
                 "EL_all_four_LEAKY": [(n, h) for n in ("stall_aware_gap", "stall_aware_full",
                                                        "stall_aware_full_placebo", "stall_aware_random")
                                       for h in (0, 1)]},
       "no_stall": {"E2_gap_two_heads": [("stall_aware_gap", 0), ("stall_aware_gap", 1)],
                    "EG_gap_plus_full": [("stall_aware_gap", 0), ("stall_aware_gap", 1), ("stall_aware_full", 0),
                                         ("stall_aware_full", 1)]}}
KS = (1, 2, 5, 10)


def auroc(s, p):
    p = np.asarray(p, bool)
    n1, n0 = int(p.sum()), int((~p).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(s)
    return float((r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def idx(pz, st, role):
    m = (pz["set"] == st) & (pz["role"] == role)
    rows = np.flatnonzero(m)
    return rows[np.argsort(pz["pair"][rows])]


def q(z, where, members):
    return np.stack([z[f"{where}__{n}"][:, h] for n, h in members], axis=1)


def med(x):
    return float(np.median(x)) if len(x) else None


def qs(x):
    return [float(np.quantile(x, v)) for v in (0.1, 0.5, 0.9)] if len(x) else None


def one(seed, world):
    pz = np.load(f"{DET}/seed{seed}/{world}/pairs.npz")
    lab = np.load(f"{DET}/seed{seed}/{world}/labels.npz")
    z = np.load(f"{D}/q_seed{seed}_{world}.npz")
    harm, past = lab["harmful"].astype(bool), lab["unsupported"].astype(bool)
    over, cpref, tpref = lab["over_pref"], lab["critic_pref"], lab["true_pref"]
    out = dict(seed=seed, world=world)
    # context: the full-log critic's own preference at the same pairs (did data in the region fix the critic?)
    full_pref = z["cmd__stall_aware_full"][:, 0] - z["cap__stall_aware_full"][:, 0]
    for st in ("T1", "T3"):
        ic = idx(pz, st, "command")
        sel = harm[ic]
        out[f"{st}_n_harmful"] = int(sel.sum())
        out[f"{st}_n_past80"] = int(past[ic].sum())
        out[f"{st}_n_past80_truly_better"] = int((past[ic] & (tpref[ic] > 0.05)).sum())
        pick = sel if sel.any() else past[ic]
        out[f"{st}_sel"] = "harmful" if sel.any() else "past80"
        out[f"{st}_over_pref_median"] = med(over[ic][pick])
        out[f"{st}_critic_pref_median"] = med(cpref[ic][pick])
        out[f"{st}_true_pref_median"] = med(tpref[ic][pick])
        out[f"{st}_full_critic_pref_median"] = med(full_pref[ic][pick])
        out[f"{st}_full_critic_over_pref_median"] = med(full_pref[ic][pick] - tpref[ic][pick])
    for en, mem in ENS[world].items():
        Qc, Qp = q(z, "cmd", mem), q(z, "cap", mem)   # rows = every pair (the 'cmd' file holds each pair's own command)
        sc, sp = Qc.std(axis=1, ddof=1), Qp.std(axis=1, ddof=1)
        e = {}
        t4 = idx(pz, "T4", "recorded")
        e["s_T4_recorded_q10_q50_q90"] = qs(sc[t4])
        for st in ("T1", "T3"):
            ic, ip = idx(pz, st, "command"), idx(pz, st, "cap")
            sel = harm[ic]
            pick = sel if sel.any() else past[ic]
            ds = sc[ic] - sp[ic]     # sp at a command row is the s_E at that row's own cap (= the cap pair's state)
            assert np.allclose(sp[ic], sc[ip], atol=1e-5), "cap rows and capped commands must agree"
            d = dict(sel=("harmful" if sel.any() else "past80"), n=int(pick.sum()),
                     s_cmd_q10_q50_q90=qs(sc[ic][pick]), s_cap_q10_q50_q90=qs(sp[ic][pick]),
                     ds_q10_q50_q90=qs(ds[pick]), ds_positive_share=float(np.mean(ds[pick] > 0)),
                     ds_over_over_pref_median=med(ds[pick] / over[ic][pick]),
                     k_need_median=med(over[ic][pick] / np.maximum(ds[pick], 1e-9)),
                     pess_mean_minus_s_pref_median=med((Qc.mean(1) - sc)[ic][pick] - (Qp.mean(1) - sp)[ic][pick]))
            for k in KS:
                d[f"flip_share_k{k}"] = float(np.mean(cpref[ic][pick] - k * ds[pick] <= 0))
                d[f"over_removed_share_k{k}"] = float(np.mean(k * ds[pick] >= over[ic][pick]))
            if st == "T1":
                ir, ik = idx(pz, st, "recorded"), idx(pz, st, "knee")
                d["s_rec_q10_q50_q90"] = qs(sc[ir][pick])
                Pc = np.maximum(sc[ic] - sc[ir], 0)
                Pp = np.maximum(sp[ic] - sc[ir], 0)
                d["paired_dP_q10_q50_q90"] = qs((Pc - Pp)[pick])
                dk = sc[ik] - sp[ic]   # 81% knee vs the 80% cap
                d["ds_knee81_q50"] = med(dk[pick])
                d["knee_share_of_cmd_ds_median"] = med(dk[pick] / np.where(np.abs(ds[pick]) > 1e-9, ds[pick], np.nan))
            # no-stall cost side: past-80% pairs where the command is truly better
            good = past[ic] & (tpref[ic] > 0.05)
            if good.any():
                d["good_n"] = int(good.sum())
                d["good_ds_q50"] = med(ds[good])
                for k in KS:
                    d[f"good_wrong_flip_share_k{k}"] = float(np.mean(cpref[ic][good] - k * ds[good] <= 0))
            # harm vs novelty among past-80% command pairs
            pp = past[ic]
            if harm[ic][pp].any() and (~harm[ic][pp]).any():
                d["auroc_harm_among_past80_ds"] = auroc(ds[pp], harm[ic][pp])
                d["auroc_harm_among_past80_s_cmd"] = auroc(sc[ic][pp], harm[ic][pp])
            d["ds_past80_all_q50"] = med(ds[pp])
            e[st] = d
        # ranking over the plan's pools (T1+T2 roles, T3) by s_E at the pair, as the detection measures rank widths
        r12 = np.concatenate([idx(pz, st, r) for st in ("T1", "T2") for r in ("command", "cap", "recorded", "cloned")])
        r3 = np.concatenate([idx(pz, "T3", r) for r in ("command", "cap")])
        e["auroc_T12_s"] = auroc(sc[r12], harm[r12])
        e["auroc_T3_s"] = auroc(sc[r3], harm[r3])
        e["auroc_T12_past80_s"] = auroc(sc[r12], past[r12])
        out[en] = e
    return out


if __name__ == "__main__":
    res = [one(s, w) for w in ("stall", "no_stall") for s in range(5)]
    json.dump(res, open(f"{D}/ens_analyse.json", "w"), indent=1)
    for r in res:
        print(r["world"], r["seed"], "T1", r["T1_sel"], r["T1_n_harmful"], "over", round(r["T1_over_pref_median"], 3),
              "cpref", round(r["T1_critic_pref_median"], 3), "full_over", round(r["T1_full_critic_over_pref_median"], 3))
        for en in ENS[r["world"]]:
            e = r[en]
            for st in ("T1", "T3"):
                d = e[st]
                print(f"   {en:22s} {st} ds50={d['ds_q10_q50_q90'][1]:+.3f} s_cmd50={d['s_cmd_q10_q50_q90'][1]:.3f} "
                      f"s_cap50={d['s_cap_q10_q50_q90'][1]:.3f} kneed={d['k_need_median']:.1f} "
                      f"flip k1/2/5={d['flip_share_k1']:.2f}/{d['flip_share_k2']:.2f}/{d['flip_share_k5']:.2f} "
                      f"good_wrongflip k1/2/5={d.get('good_wrong_flip_share_k1', float('nan')):.2f}/"
                      f"{d.get('good_wrong_flip_share_k2', float('nan')):.2f}/{d.get('good_wrong_flip_share_k5', float('nan')):.2f}"
                      f" auroc_harm|past={d.get('auroc_harm_among_past80_ds')}")
            print(f"   {en:22s} T4 s q10/50/90 = {[round(v, 3) for v in e['s_T4_recorded_q10_q50_q90']]}")
