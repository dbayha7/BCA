"""READ-ONLY. Where would a support boundary read from the SAVED weighted band sit?

For every seed 0-4 and world, at the locked detection pair states (X/detect/seed<K>/<world>/pairs.npz):
  T1 = full-throttle-region training states (gapped critic's actor's command), T2 = other training states,
  T3a = strong-current states visited by the gapped critic's actor, T3b = visited by the stall-aware controller,
sweep each thruster's command outward from 0 (others held at the actor's command) and record the first magnitude at
which BCA's own weighted threshold refuses to certify (classifier log w >= l*, i.e. R = +inf in the tabulated WBCP
threshold) and the first at which the width U~ exceeds the deployment check's tau. Both read at the raw state and at
the state clamped to the training-state box (per-dimension 0.5% / 99.5% quantiles of the gapped training states).

Also: BCA's l* against the closed-form refusal limit of plain weighted split conformal on the same threshold half
(log(alpha/(1-alpha)) + log sum_j w_j), and the share of the actor's own commands that each rule blocks.
Forward passes of the saved eta_f and classifier only. Usage: python boundary.py [seeds...]
"""
import sys
import time

import numpy as np
from scipy.stats import spearmanr

import common as C

GRID = np.round(np.linspace(0.0, 1.0, 101), 4)
NEVER = 1.01


def first_cross(vals, thr):
    """vals (n, G) along the outward grid; first grid magnitude with vals >= thr (NEVER if none)."""
    hit = vals >= thr
    idx = np.where(hit.any(1), hit.argmax(1), -1)
    return np.where(idx >= 0, GRID[np.maximum(idx, 0)], NEVER)


def sweep(band, X, Acmd, j, d):
    n = len(X)
    G = len(GRID)
    a = np.repeat(Acmd[:, None, :], G, 1).copy()
    a[:, :, j] = d * GRID[None, :]
    o = np.repeat(band.norm(X)[:, None, :], G, 1)
    lw = band.logw(o.reshape(-1, C.TC.DX), a.reshape(-1, C.TC.DA)).reshape(n, G)
    uw = band.width(o.reshape(-1, C.TC.DX), a.reshape(-1, C.TC.DA)).reshape(n, G)
    return lw, uw


def auc(y, s):
    from scipy.stats import rankdata
    r = rankdata(s)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def qs(v):
    v = np.asarray(v, np.float64)
    return {str(q): float(np.quantile(v, q / 100)) for q in (10, 25, 50, 75, 90)}


def run(rep, world):
    t0 = time.perf_counter()
    sd, pg = C.seed_data(world, rep)
    band = C.Band(world, rep)
    lo, hi = C.recorded_range(pg)
    blo, bhi = C.state_box(pg)
    P = np.load(C.DET / f"seed{rep}" / world / "pairs.npz")
    Lb = np.load(C.DET / f"seed{rep}" / world / "labels.npz")
    sets, role, src = P["set"], P["role"], P["t3_source"]
    X_all = np.asarray(P["states"], np.float64)
    A_all = np.asarray(P["commands"], np.float32)
    masks = dict(T1=(sets == "T1") & (role == "command"), T2=(sets == "T2") & (role == "command"),
                 T3a=(sets == "T3") & (src == "actor") & (role == "command"),
                 T3b=(sets == "T3") & (src == "controller") & (role == "command"),
                 T4=(sets == "T4"))
    # closed-form refusal limit of plain weighted split conformal on the band's threshold half
    bank_o = np.asarray(pg.bank.obs, np.float32)
    bank_a = np.asarray(pg.bank.action, np.float32)
    lw_bank = band.logw(bank_o, bank_a)
    perm = np.random.default_rng([20261015, int(rep), 71]).permutation(len(lw_bank))
    A_idx = np.sort(perm[: len(lw_bank) // 2])
    m = lw_bank[A_idx].max()
    l_ref_half = float(np.log(C.ALPHA / (1 - C.ALPHA)) + m + np.log(np.exp(lw_bank[A_idx] - m).sum()))
    m2 = lw_bank.max()
    l_ref_full = float(np.log(C.ALPHA / (1 - C.ALPHA)) + m2 + np.log(np.exp(lw_bank - m2).sum()))
    ess_half = float(np.exp(lw_bank[A_idx]).sum() ** 2 / np.exp(2 * lw_bank[A_idx]).sum())
    out = dict(rep=rep, world=world, l_star=band.l_star, l_ref_closed_form_threshold_half=l_ref_half,
               l_ref_closed_form_full_bank=l_ref_full, bank_ess_threshold_half=ess_half,
               threshold_half_rows=int(len(A_idx)), tau=band.tau, U_max=band.U_max,
               recorded_range=dict(lo=lo.tolist(), hi=hi.tolist()), state_box=dict(lo=blo.tolist(), hi=bhi.tolist()),
               sets={})
    for name, mk in masks.items():
        X = X_all[mk]
        Acmd = A_all[mk]
        Xc = np.clip(X, blo, bhi)
        p80 = np.any(np.abs(Acmd.astype(np.float64)) > C.KNEE, 1)
        harm = Lb["harmful"][mk]
        rec = dict(n=int(mk.sum()), past80=float(p80.mean()), harmful=float(harm.mean()),
                   current_q=qs(X[:, C.TC.CURRENT]), push_q=qs(C.TC.push(X)))
        # the actor's own commands: which rules block them
        for tag, XX in (("raw", X), ("clamped", Xc)):
            o = band.norm(XX)
            lw = band.logw(o, Acmd)
            uw = band.width(o, Acmd)
            for rule, fire in (("refuse", lw >= band.l_star), ("tau", uw > band.tau), ("plainwcp", lw >= l_ref_half)):
                r = dict(fire=float(fire.mean()))
                if name != "T4":
                    r.update(recall_past80=float(fire[p80].mean()) if p80.any() else None,
                             false_flag=float(fire[~p80].mean()) if (~p80).any() else None,
                             recall_harmful=float(fire[harm].mean()) if harm.any() else None,
                             fire_not_harmful=float(fire[~harm].mean()) if (~harm).any() else None)
                rec[f"{rule}_{tag}"] = r
            if p80.any() and (~p80).any():
                rec[f"auc_logw_past80_{tag}"] = auc(p80, lw)
                rec[f"auc_Uw_past80_{tag}"] = auc(p80, uw)
        if name == "T4":
            out["sets"][name] = rec
            continue
        # boundaries: thruster 1 (index 0) negative direction is where 99% of past-80% commands are; all reported
        bd = {}
        for j in range(C.TC.DA):
            for d in (-1, 1):
                key = f"th{j + 1}{'-' if d < 0 else '+'}"
                edge = abs(lo[j]) if d < 0 else abs(hi[j])
                r = dict(recorded_edge=float(edge))
                for tag, XX in (("raw", X), ("clamped", Xc)):
                    lw, uw = sweep(band, XX, Acmd, j, d)
                    b_ref = first_cross(lw, band.l_star)
                    b_tau = first_cross(uw, band.tau)
                    b_plain = first_cross(lw, l_ref_half)
                    r[f"plainwcp_{tag}"] = dict(q=qs(b_plain), never=float((b_plain >= NEVER).mean()),
                                               at_or_inside_knee=float((b_plain <= C.KNEE + 1e-9).mean()),
                                               at_or_inside_recorded=float((b_plain <= edge + 1e-9).mean()))
                    r[f"refuse_{tag}"] = dict(q=qs(b_ref), never=float((b_ref >= NEVER).mean()),
                                             at_or_inside_knee=float((b_ref <= C.KNEE + 1e-9).mean()),
                                             at_or_inside_recorded=float((b_ref <= edge + 1e-9).mean()),
                                             blocks_at_0=float((b_ref == 0).mean()))
                    r[f"tau_{tag}"] = dict(q=qs(b_tau), never=float((b_tau >= NEVER).mean()),
                                          at_or_inside_knee=float((b_tau <= C.KNEE + 1e-9).mean()),
                                          at_or_inside_recorded=float((b_tau <= edge + 1e-9).mean()),
                                          blocks_at_0=float((b_tau == 0).mean()))
                    if j == 0 and d < 0:
                        b_fin = np.minimum(b_ref, 1.0)
                        rec[f"th1_boundary_vs_push_spearman_{tag}"] = float(spearmanr(b_fin, C.TC.push(X)).correlation)
                        rec[f"th1_boundary_vs_current_spearman_{tag}"] = float(
                            spearmanr(b_fin, X[:, C.TC.CURRENT]).correlation)
                        # log w along thruster-1 throttle: median curve (the refusal margin profile)
                        rec[f"th1_logw_minus_lstar_median_curve_{tag}"] = {
                            str(g): float(np.median(lw[:, i] - band.l_star)) for i, g in enumerate(GRID) if
                            i % 10 == 0 or g in (0.75, 0.8, 0.85, 0.95)}
                        rec[f"th1_Uw_median_curve_{tag}"] = {
                            str(g): float(np.median(uw[:, i])) for i, g in enumerate(GRID) if
                            i % 10 == 0 or g in (0.75, 0.8, 0.85, 0.95)}
                bd[key] = r
        rec["boundaries"] = bd
        out["sets"][name] = rec
    out["seconds"] = time.perf_counter() - t0
    return out


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or list(range(5))
    res = []
    for rep in seeds:
        for world in ("stall", "no_stall"):
            r = run(rep, world)
            res.append(r)
            s1 = r["sets"]["T1"]["boundaries"]["th1-"]
            s3 = r["sets"]["T3a"]["boundaries"]["th1-"]
            print(f"seed {rep} {world:8s} l*={r['l_star']:.3f} l_ref_half={r['l_ref_closed_form_threshold_half']:.3f} "
                  f"| T1 th1- refuse median {s1['refuse_raw']['q']['50']:.3f} inside-knee {s1['refuse_raw']['at_or_inside_knee']:.2f} "
                  f"never {s1['refuse_raw']['never']:.2f} | T3a raw median {s3['refuse_raw']['q']['50']:.3f} never "
                  f"{s3['refuse_raw']['never']:.2f} clamped median {s3['refuse_clamped']['q']['50']:.3f} never "
                  f"{s3['refuse_clamped']['never']:.2f} ({r['seconds']:.0f}s)", flush=True)
    C.dump(res, f"boundary{'_' + '_'.join(map(str, seeds)) if len(seeds) < 5 else ''}.json")
