"""Read-only: per-step selectivity of BCA's width (and the stand-ins) on the locked detection pair set T3a (strong-current
states visited by the no-caution actor's own unswitched rollouts from the same first 200 strong-current starts the
deployment check uses), at the deployment operating point. Uses only saved scores; nothing is fitted or simulated.
Writes only into this scratchpad."""
import json
import numpy as np

KNEE = float(np.float32(0.8))
DET = "/home/user/bca/runs/shift_explore/detect/seed{k}/stall"
S2D = "/home/user/bca/runs/shift_explore/stage2/detect/{v}/stall/fixed/seed{k}"
S2DEP = "/home/user/bca/runs/shift_explore/stage2/deploy/stall/seed{k}/values.json"
D1 = "/home/user/bca/runs/shift_explore/deploy/seed{k}_stall_gapped.json"
STAND = ("classifier_ratio", "distance", "density", "two_head", "residual_evidence", "range_hinge", "perfect_band")


def auc(pos, neg):
    x = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(float)
    # average ranks for ties
    from collections import defaultdict
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


out = {}
for k in range(5):
    P = np.load(DET.format(k=k) + "/pairs.npz")
    L = np.load(DET.format(k=k) + "/labels.npz")
    sets, role, src = P["set"], P["role"], P["t3_source"]
    cmd = np.asarray(P["commands"], np.float64)
    p80 = np.any(np.abs(cmd) > KNEE, axis=1)
    t3a = (sets == "T3") & (src == "actor")
    t3a_cmd = t3a & (role == "command")
    t3a_cap = t3a & (role == "cap")
    t4 = sets == "T4"
    step, start = P["t3_step"], P["t3_start"]
    sig = {}
    for v in ("weighted", "uniform"):
        sig["bca_" + v] = np.load(S2D.format(v=v, k=k) + f"/scores/bca_width_{v}_fixed.npz")["scores"]
    for s in STAND:
        try:
            sig[s] = np.load(DET.format(k=k) + f"/scores/{s}.npz")["scores"]
        except Exception as e:  # noqa
            pass
    tau_w = json.load(open(S2D.format(v="weighted", k=k) + "/tau.json"))["tau"]
    tau_u = json.load(open(S2D.format(v="uniform", k=k) + "/tau.json"))["tau"]
    rec = dict(n_t3a_cmd=int(t3a_cmd.sum()), past80_share_t3a=float(p80[t3a_cmd].mean()),
               tau_weighted=tau_w, tau_uniform=tau_u, signals={})
    for name, sc in sig.items():
        sc = np.asarray(sc, np.float64)
        thr_t4 = float(np.quantile(sc[t4], 0.95))
        thrs = {"t4_q95": thr_t4}
        if name == "bca_weighted":
            thrs["deploy_tau"] = tau_w
        if name == "bca_uniform":
            thrs["deploy_tau"] = tau_u
        r = dict(nonfinite_t3a=int((~np.isfinite(sc[t3a])).sum()), nonfinite_all=int((~np.isfinite(sc)).sum()),
                 auc_t3a_past80=auc(sc[t3a_cmd & p80], sc[t3a_cmd & ~p80]))
        # paired: command vs its own cap at the same T3 state (pair index shared?)
        for tn, thr in thrs.items():
            fl = ~(sc <= thr)
            q = dict(thr=thr, t4_rate=float(fl[t4].mean()),
                     recall=float(fl[t3a_cmd & p80].mean()),
                     false_flag=float(fl[t3a_cmd & ~p80].mean()),
                     fire=float(fl[t3a_cmd].mean()),
                     precision=float((fl & t3a_cmd & p80).sum() / max((fl & t3a_cmd).sum(), 1)),
                     flag_at_cap_when_cmd_past80=float(fl[t3a_cap][p80[t3a_cmd]].mean()) if t3a_cap.sum() == t3a_cmd.sum() else None)
            bins = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 100)]
            q["by_step"] = {f"{a}-{b - 1}": dict(n=int((t3a_cmd & (step >= a) & (step < b)).sum()),
                                                 past80=float(p80[t3a_cmd & (step >= a) & (step < b)].mean()),
                                                 recall=float(fl[t3a_cmd & p80 & (step >= a) & (step < b)].mean()),
                                                 false_flag=float(fl[t3a_cmd & ~p80 & (step >= a) & (step < b)].mean())
                                                 if (t3a_cmd & ~p80 & (step >= a) & (step < b)).any() else None)
                            for a, b in bins}
            # per-start clustering of misses among past-80% T3a commands
            m = t3a_cmd & p80
            st = start[m]
            miss = ~fl[m]
            us = np.unique(st)
            per = np.array([miss[st == u].mean() for u in us])
            cnt = np.array([(st == u).sum() for u in us])
            pbar = miss.mean()
            # binomial expectation of variance of per-start miss rates
            exp_var = float(np.mean(pbar * (1 - pbar) / cnt))
            q["miss_cluster"] = dict(starts=int(len(us)), mean_miss=float(pbar), var_per_start=float(per.var()),
                                     binomial_var=exp_var,
                                     starts_all_missed=int(((per == 1) & (cnt >= 3)).sum()),
                                     starts_none_missed=int(((per == 0) & (cnt >= 3)).sum()),
                                     share_of_misses_in_top10pct_starts=float(
                                         np.sort(per * cnt)[::-1][:max(1, len(us) // 10)].sum() / max((per * cnt).sum(), 1)))
            if tn in ("deploy_tau",) or (name != "bca_weighted" and tn == "t4_q95"):
                q["per_start_miss"] = {int(u): float(v) for u, v in zip(us, per)}
            r[tn] = q
        rec["signals"][name] = r
    # link per-start T3 misses (unswitched rollout) to the deployment per-start shortfall vs the perfect detector
    dep = json.load(open(S2DEP.format(k=k)))["results"]["none"]["strong_current"]
    d1 = json.load(open(D1.format(k=k)))["situations"]["strong_current"]["rules"]
    bca = np.asarray(dep["bca"]["per_start_points"])
    rnd = np.asarray(dep["random:bca"]["per_start_points"])
    perf = np.asarray(d1["unrec:clip99"]["per_start_points"])
    pm = rec["signals"]["bca_weighted"]["deploy_tau"]["per_start_miss"]
    idx = np.array(sorted(pm))
    mr = np.array([pm[i] for i in idx])
    short = (perf - bca)[idx]
    rec["link"] = dict(n_starts=int(len(idx)), corr_t3_miss_vs_deploy_shortfall=float(np.corrcoef(mr, short)[0, 1]),
                       spearman=float(np.corrcoef(np.argsort(np.argsort(mr)), np.argsort(np.argsort(short)))[0, 1]),
                       shortfall_mean_when_t3_miss_ge_half=float(short[mr >= 0.5].mean()) if (mr >= 0.5).any() else None,
                       n_t3_miss_ge_half=int((mr >= 0.5).sum()),
                       shortfall_mean_when_t3_miss_zero=float(short[mr == 0].mean()) if (mr == 0).any() else None,
                       n_t3_miss_zero=int((mr == 0).sum()),
                       bca_minus_random_when_t3_miss_ge_half=float((bca - rnd)[idx][mr >= 0.5].mean()) if (mr >= 0.5).any() else None,
                       bca_minus_random_when_t3_miss_zero=float((bca - rnd)[idx][mr == 0].mean()) if (mr == 0).any() else None)
    for s in rec["signals"].values():
        for q in s.values():
            if isinstance(q, dict):
                q.pop("per_start_miss", None)
    out[k] = rec

json.dump(out, open("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/impact/m5_t3.json", "w"),
          indent=1)
names = list(out[0]["signals"])
print("signal  [threshold]  pooled over seeds: recall / false_flag / fire / precision / t4_rate / auc")
for n in names:
    for tn in ("deploy_tau", "t4_q95"):
        vals = [out[k]["signals"][n].get(tn) for k in range(5)]
        if vals[0] is None:
            continue
        f = lambda key: np.mean([v[key] for v in vals])
        print("%-18s %-10s rec %.3f ff %.3f fire %.3f prec %.3f t4 %.3f auc %.3f | per-seed rec %s ff %s" % (
            n, tn, f("recall"), f("false_flag"), f("fire"), f("precision"), f("t4_rate"),
            np.mean([out[k]["signals"][n]["auc_t3a_past80"] for k in range(5)]),
            [round(v["recall"], 2) for v in vals], [round(v["false_flag"], 2) for v in vals]))
print()
for k in range(5):
    w = out[k]["signals"]["bca_weighted"]["deploy_tau"]
    print("seed", k, "past80 share T3a %.3f" % out[k]["past80_share_t3a"], "nonfinite (all pairs) BCA-w:",
          out[k]["signals"]["bca_weighted"]["nonfinite_all"])
    print("   by step:", {b: (round(v["past80"], 2), round(v["recall"], 2),
                           None if v["false_flag"] is None else round(v["false_flag"], 2)) for b, v in w["by_step"].items()})
    print("   miss cluster:", {a: (round(b, 3) if isinstance(b, float) else b) for a, b in w["miss_cluster"].items()})
    print("   cap flagged when command past80: %.3f" % w["flag_at_cap_when_cmd_past80"])
    print("   link:", {a: (round(b, 2) if isinstance(b, float) else b) for a, b in out[k]["link"].items()})
    cr = out[k]["signals"]["classifier_ratio"]["t4_q95"]["miss_cluster"]
    print("   classifier ratio miss cluster:", {a: (round(b, 3) if isinstance(b, float) else b) for a, b in cr.items()})
