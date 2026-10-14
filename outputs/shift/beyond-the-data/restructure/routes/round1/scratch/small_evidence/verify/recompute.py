"""Adversarial verification of the small_evidence verdict (read-only on route outputs)."""
import sys
sys.dont_write_bytecode = True
import json, math
import numpy as np
from scipy import stats
sys.path.insert(0, "/home/user/bca"); sys.path.insert(0, "/home/user/bca/outputs/shift/beyond-the-data/restructure/step1")
import step1lib as S

SC = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/routes/small_evidence/scores.npz"
DR = "/home/user/bca/runs/shift_explore/routes/small_evidence/draws.npz"
RJ = "/home/user/bca/runs/shift_explore/routes/small_evidence/results.json"
z = np.load(SC); d = np.load(DR); r = json.load(open(RJ))
spec, _ = S.load_prespec(); cfg = spec["certificate"]
seen = spec["deployment"]["seen_design_values"]
SEEDS = range(5); LAD = [3, 5, 8, 12, 20, 30, 45]
gain = {s: seen[f"stall/seed{s}/strong_current"]["on"] - seen[f"stall/seed{s}/strong_current"]["never"] for s in SEEDS}
loss = {s: seen[f"no_stall/seed{s}/strong_current"]["never"] - seen[f"no_stall/seed{s}/strong_current"]["on"] for s in SEEDS}
print("gain", gain, "mean", np.mean(list(gain.values())), "always-on", np.mean(list(gain.values())) - np.mean(list(loss.values())))

# 1. independent redraw of idx, independent C and R
mism = 0; on = {}
for w in ("stall", "no_stall"):
    wt = S.WORLD_TAG[w]
    for s in SEEDS:
        pool = np.asarray(z[f"{w}_{s}_pool"], float); rest = np.asarray(z[f"{w}_{s}_rest"], float)
        k = math.ceil(0.9 * len(rest) - 1e-12); lam_rest = np.sort(rest)[k - 1]
        for n in LAD:
            g = np.random.default_rng([20261015, 9930, s, wt, n])
            idx = np.array([g.choice(len(pool), n, replace=False) for _ in range(400)])
            if not np.array_equal(idx, d[f"{w}_{s}_n{n}_idx"]): mism += 1
            tails = pool[idx]
            kt = math.ceil(0.9 * n - 1e-12)
            lam_t = np.sort(tails, axis=1)[:, kt - 1]
            Con = (lam_t - lam_rest) > 0.10
            pv = np.array([stats.mannwhitneyu(t, rest, alternative="greater").pvalue for t in tails])
            Ron = pv < 0.05
            Bon = d[f"{w}_{s}_n{n}_q05"] > 0.10
            B0on = d[f"{w}_{s}_n{n}_q05"] > 0.0
            # compare with stored
            dC = np.max(np.abs((lam_t - lam_rest) - d[f"{w}_{s}_n{n}_plain_gap"]))
            dP = np.max(np.abs(pv - d[f"{w}_{s}_n{n}_p"]))
            if dC > 1e-12 or dP > 1e-12: print("MISMATCH", w, s, n, dC, dP)
            on[(w, s, n)] = dict(B=Bon.mean(), B0=B0on.mean(), C=Con.mean(), R=Ron.mean())
print("idx mismatches", mism)

def net(rule, n):
    return np.mean([on[("stall", s, n)][rule] * gain[s] - on[("no_stall", s, n)][rule] * loss[s] for s in SEEDS])
nets = {ru: {n: net(ru, n) for n in LAD} for ru in ("B", "B0", "C", "R")}
for ru in nets: print(ru, {n: round(v, 2) for n, v in nets[ru].items()})
DC = np.mean([nets["B"][n] - nets["C"][n] for n in (12, 20, 30, 45)])
DR = np.mean([nets["B"][n] - nets["R"][n] for n in (12, 20, 30, 45)])
print("D_C", DC, "stored", r["decision"]["D_C"]["value"]); print("D_R", DR, "stored", r["decision"]["D_R"]["value"])
for ru in nets:
    for n in LAD:
        if abs(nets[ru][n] - r["nets"][ru][str(n)]) > 1e-9: print("net mismatch", ru, n)
# on-rate ranges
for ru in ("B", "C", "R", "B0"):
    for n in LAD:
        st = [on[("stall", s, n)][ru] for s in SEEDS]; ns = [on[("no_stall", s, n)][ru] for s in SEEDS]
        print(f"{ru} n{n}: stall {min(st):.4f}-{max(st):.4f}  no_stall {min(ns):.4f}-{max(ns):.4f}  ns per seed {[round(x,4) for x in ns]}")
np.save("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/routes/small_evidence/verify/on.npy", np.array([0]))
