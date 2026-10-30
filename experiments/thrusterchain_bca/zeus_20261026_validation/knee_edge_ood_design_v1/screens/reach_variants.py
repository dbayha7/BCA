"""Re-run the reach grid under model variants (saved JSON only)."""
import itertools
import json
from pathlib import Path
import numpy as np

V = Path(r"C:\Users\David\dev\BCA\experiments\thrusterchain_bca\zeus_20261026_validation")
EV = V / "central_quartile_training_v1_evaluation_rerun1" / "run" / "evaluation"
RUNS = [V / "central_quartile_training_v1_authorized1" / "run", V / "central_quartile_training_v1_continuation1" / "run"]
KNEE = float(np.float32(0.8))
SLOTS = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
DESIGN = np.vstack([np.ones(5), SLOTS, SLOTS ** 2]).T
CELLS = [(c, s) for c in ("full", "hole", "random") for s in range(3)]
RESETS = {"wide": [(1.0, 0.1, 1.1)], "mix25": [(0.75, 0.1, 0.7), (0.25, 0.9, 1.1)],
          "mix15": [(0.85, 0.1, 0.7), (0.15, 0.9, 1.1)]}
WINDOWS = {"A": (-1.0, -0.6, 0.05), "B": (-0.9, -0.7, 0.05), "C": (-0.9, -0.7, 0.02), "D": (-0.85, -0.75, 0.02)}


def L(p):
    return json.loads(p.read_text(encoding="utf-8"))


def lam(cond, seed):
    for run in RUNS:
        p = run / f"seed{seed}_{cond}" / "native" / "gradient_audit_020000.json"
        if p.exists():
            return float(np.mean([b["lambda_from_unpenalized_Q1"] for b in L(p)["batches"]]))


def sm(law):
    return sum(w * (((lo + hi) / 2) ** 2 + (hi - lo) ** 2 / 12) for w, lo, hi in law)


def anchor(lo, hi, e):
    w = (1 - e) * (hi + KNEE) / (hi - lo)
    f = e * KNEE
    return w * (hi - KNEE) / 2 / (w + f)


def mabs(ctx, key="q1"):
    return float(np.mean([np.abs(np.array(L(EV / f"seed{s}_{c}__native__20000__{ctx}.json")["q"]["per_probe"][key])[:, 0]).mean() for c, s in CELLS]))

o, st = mabs("ordinary"), mabs("strong_current")
sl = (st - o) / (sm([(1, .7, 1.1)]) - sm([(1, .1, .7)])); ic = o - sl * sm([(1, .1, .7)])
cells = []
for c, s in CELLS:
    d = L(EV / f"seed{s}_{c}__native__20000__strong_current.json")
    pp, vr = d["q"]["per_probe"], d["visited"]["per_row"]
    hl = np.array(pp["high_load"], bool)
    coef = np.linalg.lstsq(DESIGN, np.array(pp["q1"])[:, 1:].T, rcond=None)[0].T
    vobs = np.array(vr["observations"]); vhl = np.array(vr["high_load"], bool)
    pobs = np.array(d["q"]["per_probe"].get("observations", [[0] * 14] * 32)) if "observations" in pp else None
    cells.append(dict(lam=lam(c, s), b=coef[hl, 1], c=coef[hl, 2], va=np.array(vr["actions"])[vhl, 0],
                      vcur=vobs[vhl, 10]))


def grid(variant):
    feas = []
    rows = []
    for (rn, law), (wn, (lo, hi, e)), dd in itertools.product(RESETS.items(), WINDOWS.items(), (0.10, 0.05)):
        scale = o / (ic + sl * sm(law)); m = anchor(lo, hi, e); gate = -(KNEE + dd)
        pmin, vmin = 1, 1
        for cell in cells:
            l = cell["lam"] * scale
            pi0 = np.clip((m + 2 * l * cell["b"]) / (1 - 4 * l * cell["c"]), -1, 1)
            va = cell["va"]
            fac = 0.87
            if variant == "factor_corrected":
                fac = 1 / (1 + 4 * l)
            if variant == "e2_rows":
                va = va[cell["vcur"] >= 0.9]
            vis = np.clip(scale * va + fac * m, -1, 1)
            pmin = min(pmin, np.mean(pi0 <= gate)); vmin = min(vmin, np.mean(vis <= gate))
        ok = pmin >= .4 and vmin >= .3
        rows.append((rn, wn, dd, round(scale, 3), round(m, 4), round(pmin, 3), round(vmin, 3), ok))
        if ok:
            feas.append((dd, e, rn, wn))
    order = lambda f: (-f[0], -f[1], list(RESETS).index(f[2]), list(WINDOWS).index(f[3]))
    return rows, (min(feas, key=order) if feas else None), len(feas)

for v in ("as_written", "factor_corrected", "e2_rows"):
    rows, sel, n = grid(v)
    print(f"== {v}: feasible {n}, selected {sel}")
    for r in rows:
        if r[2] == 0.10:
            print("  ", r)
print("visited rows with c>=0.9 per cell:", [int((c['vcur'] >= 0.9).sum()) for c in cells])
