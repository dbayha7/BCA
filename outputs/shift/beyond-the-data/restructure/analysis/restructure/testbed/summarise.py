"""Pool E1-E3 (and E4 when present) over seeds 0-2 into summary.json. Read-only; numpy only."""
import json
import os

import numpy as np
from scipy.stats import mannwhitneyu

OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/testbed"
SEEDS = (0, 1, 2)
SHARES = ("0.005", "0.01", "0.02", "0.03", "0.05", "0.1")
WORLDS = ("stall", "no_stall")


def auc(a, b):
    """P(a < b) (+ ties/2): how often a stall residual is below a no-stall residual."""
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(u / (len(a) * len(b)))


def e1():
    out = {}
    for rep in SEEDS:
        z = {w: np.load(f"{OUT}/heldout_seed{rep}_{w}.npz") for w in WORLDS}
        r = {}
        for w in WORLDS:
            Z = z[w]
            P = lambda k: Z["pos_" + k]  # noqa: E731
            r[w] = {c: {k: float(Z["delta_" + c][P(k)].mean()) for k in ("cand", "bankcand", "t4", "bank", "surv", "kept")}
                    for c in ["gap", "full"] + [f"trace_{s}" for s in SHARES]}
        # held-out (gap critic) vs in-sample (trace critic) at the SAME added rows, paired across worlds
        rows = z["stall"]["rows"]
        assert np.array_equal(rows[z["stall"]["pos_cand"]], z["no_stall"]["rows"][z["no_stall"]["pos_cand"]])
        crow = rows[z["stall"]["pos_cand"]]
        sweep = {}
        for s in SHARES:
            m = np.isin(crow, z["stall"][f"added_{s}"])
            d = {w: {c: z[w][f"delta_{c}"][z[w]["pos_cand"]] for c in ("gap", f"trace_{s}")} for w in WORLDS}
            t4 = {w: z[w]["delta_gap"][z[w]["pos_t4"]] for w in WORLDS}
            lvl = {w: np.quantile(-t4[w], 0.95) for w in WORLDS}
            sweep[s] = dict(
                n_added=int(m.sum()),
                heldout_gap=dict(stall=float(d["stall"]["gap"][m].mean()), no_stall=float(d["no_stall"]["gap"][m].mean()),
                                 auc_stall_below=auc(d["stall"]["gap"][m], d["no_stall"]["gap"][m]),
                                 flag_stall=float((-d["stall"]["gap"][m] > lvl["stall"]).mean()),
                                 flag_no_stall=float((-d["no_stall"]["gap"][m] > lvl["no_stall"]).mean())),
                insample_trace=dict(stall=float(d["stall"][f"trace_{s}"][m].mean()),
                                    no_stall=float(d["no_stall"][f"trace_{s}"][m].mean()),
                                    auc_stall_below=auc(d["stall"][f"trace_{s}"][m], d["no_stall"][f"trace_{s}"][m])),
                heldout_trace_rest=dict(stall=float(d["stall"][f"trace_{s}"][~m].mean()),
                                        no_stall=float(d["no_stall"][f"trace_{s}"][~m].mean())))
        # how many held-out rows are needed: one-sided Mann-Whitney of k random held-out past-80% residuals (gap
        # critic) against the gate bank's residuals, p < 0.05; share of 400 draws (stall = power, no-stall = false alarm)
        need = {}
        rng = np.random.default_rng([20261014, rep, 800])
        for k in (1, 3, 5, 10, 33):
            frac = {}
            for w in WORLDS:
                cand = z[w]["delta_gap"][z[w]["pos_cand"]]
                bank = z[w]["delta_gap"][z[w]["pos_bank"]]
                hits = 0
                for _ in range(400):
                    pick = cand[rng.choice(len(cand), k, replace=False)]
                    hits += mannwhitneyu(pick, bank, alternative="less").pvalue < 0.05
                frac[w] = hits / 400
            need[str(k)] = frac
        out[rep] = dict(means=r, sweep=sweep, rows_needed=need)
    return out


def pooled(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def e2():
    res = {rep: json.load(open(f"{OUT}/hlc_seed{rep}.json")) for rep in SEEDS}
    out = {}
    for cfg in ("A", "B", "C", "POP"):
        out[cfg] = {}
        for sh in res[0]["stall"][cfg]:
            out[cfg][sh] = {}
            for wt in ("cls", "kern"):
                out[cfg][sh][wt] = {}
                for S in ("T1c", "T1", "T3a"):
                    row = {}
                    for w in WORLDS:
                        ss = [res[r][w][cfg][sh][wt][S] for r in SEEDS]
                        for f in ("refusal", "flag", "median_U", "widening_share", "median_gap"):
                            vals = [x[f] for x in ss if x]
                            vals = [v for v in vals if v is not None and np.isfinite(v)]
                            row[f"{w}_{f}"] = float(np.mean(vals)) if vals else "inf"
                        row[f"{w}_median_U_per_seed"] = [x["median_U"] if x else None for x in ss]
                    out[cfg][sh][wt][S] = row
    return out


def e3():
    res = {rep: json.load(open(f"{OUT}/truth_cover_seed{rep}.json")) for rep in SEEDS}
    out = {}
    for k in res[0]["stall"]:
        out[k] = {}
        for w in WORLDS:
            items = [res[r][w][k] for r in SEEDS]
            if k == "truth_gap":
                out[k][w] = {f: [x[f] for x in items] for f in items[0]}
            elif k.startswith("bca_"):
                out[k][w] = {f: float(np.mean([x[f] for x in items])) for f in ("coverage", "median_truth", "median_U")}
            else:
                out[k][w] = dict(
                    truth_overrating_cmd=[x["truth_overrating_cmd"] for x in items],
                    truth_paired=[x["truth_paired"] for x in items],
                    cls_coverage=float(np.mean([x["cls"]["coverage"] for x in items])),
                    cls_median_U=[x["cls"]["median_U"] for x in items],
                    cls_refusal=float(np.mean([x["cls"]["refusal"] for x in items])),
                    kern_coverage=float(np.mean([x["kern"]["coverage"] for x in items])),
                    kern_median_U=[x["kern"]["median_U"] for x in items],
                    cls_paired_median_gap=[x["cls_paired"]["median_gap"] for x in items],
                    kern_paired_median_gap=[x["kern_paired"]["median_gap"] for x in items])
    return out


def e4():
    files = [f"{OUT}/deploy_lcb_seed{rep}.json" for rep in SEEDS]
    have = [(rep, f) for rep, f in zip(SEEDS, files) if os.path.exists(f)]
    if not have:
        return None
    res = {rep: json.load(open(f)) for rep, f in have}
    out = dict(seeds=[r for r, _ in have])
    for w in WORLDS:
        out[w] = {}
        sits = res[have[0][0]][w]["situations"]
        for sit in sits:
            out[w][sit] = {}
            for rule in sits[sit]:
                pts = [res[r][w]["situations"][sit][rule]["points"] for r in res]
                out[w][sit][rule] = dict(points=float(np.mean(pts)), per_seed=pts,
                                         fire=float(np.mean([res[r][w]["situations"][sit][rule]["fire"] for r in res])),
                                         recall=float(np.mean([res[r][w]["situations"][sit][rule]["recall"] for r in res])))
        out[w]["check_vs_fallback"] = {r: res[r][w]["check_vs_fallback"] for r in res}
    return out


def deploy_pool(prefix, seeds=SEEDS):
    have = [rep for rep in seeds if os.path.exists(f"{OUT}/{prefix}{rep}.json")]
    if not have:
        return None
    res = {rep: json.load(open(f"{OUT}/{prefix}{rep}.json")) for rep in have}
    out = dict(seeds=have)
    for w in WORLDS:
        out[w] = {}
        if "tau" in res[have[0]][w]:
            out[w]["tau"] = {k: [res[r][w]["tau"][k] for r in have] for k in res[have[0]][w]["tau"]}
        if "bands" in res[have[0]][w]:
            out[w]["bands"] = {k: {f: [res[r][w]["bands"][k][f] for r in have] for f in res[have[0]][w]["bands"][k]}
                               for k in res[have[0]][w]["bands"]}
            out[w]["truth"] = {f: [res[r][w]["truth"][f] for r in have] for f in res[have[0]][w]["truth"]}
        for sit, rules in res[have[0]][w]["situations"].items():
            out[w][sit] = {}
            for rule in rules:
                pts = [res[r][w]["situations"][sit][rule]["points"] for r in have]
                out[w][sit][rule] = dict(points=float(np.mean(pts)), per_seed=pts,
                                         fire=float(np.mean([res[r][w]["situations"][sit][rule]["fire"] for r in have])),
                                         recall=float(np.mean([res[r][w]["situations"][sit][rule]["recall"] for r in have])))
    # net across worlds in strong current: won back in the stalling world minus lost in the no-stall world
    sc = "strong_current"
    if sc in out["stall"] and sc in out["no_stall"] and "none" in out["stall"][sc]:
        out["net_strong_current"] = {}
        for rule in out["stall"][sc]:
            if rule in out["no_stall"][sc]:
                won = [a - b for a, b in zip(out["stall"][sc][rule]["per_seed"], out["stall"][sc]["none"]["per_seed"])]
                lost = [b - a for a, b in zip(out["no_stall"][sc][rule]["per_seed"], out["no_stall"][sc]["none"]["per_seed"])]
                out["net_strong_current"][rule] = dict(won=float(np.mean(won)), lost=float(np.mean(lost)),
                                                       net=float(np.mean(won) - np.mean(lost)),
                                                       net_per_seed=[a - b for a, b in zip(won, lost)])
    return out


def hinge_pool():
    have = [rep for rep in SEEDS if os.path.exists(f"{OUT}/hinge_sweep_seed{rep}.json")]
    if not have:
        return None
    res = {rep: json.load(open(f"{OUT}/hinge_sweep_seed{rep}.json")) for rep in have}
    out = dict(seeds=have)
    for w in WORLDS:
        out[w] = {sh: {sit: float(np.mean([res[r][w][sh][sit] for r in have])) for sit in ("strong_current", "recorded")}
                  for sh in res[have[0]][w]}
        out[w]["lo_thruster1"] = {sh: [res[r][w][sh]["lo"][0] for r in have] for sh in res[have[0]][w]}
    return out


if __name__ == "__main__":
    s = dict(E1=e1(), E2=e2(), E3=e3(), E4=e4(), E4_pooled=deploy_pool("deploy_lcb_seed"),
             E5_akern_lcb=deploy_pool("akern_seed"), E6_akern_threshold=deploy_pool("akern_thr_seed"),
             E6_S0_threshold=deploy_pool("akern_thr_s0_seed", (0, 1, 2, 3, 4)), E5_S0_truth=deploy_pool("akern_s0_seed"),
             E7_hinge_sweep=hinge_pool())
    json.dump(s, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
    print("written")
