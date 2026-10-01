"""Build and execute results.ipynb: every semi-synthetic benchmark result with its plots and tables.

The notebook reads the same artifacts as results_page.py (runs/, Changes 1-11 of DEPENDENCE.md),
draws every figure with matplotlib and shows every number with pandas. It is executed here, so the
saved file carries its outputs. Needs matplotlib, pandas, ipykernel and jupyter_client.

python experiments/wbcp/results_notebook.py [--output experiments/wbcp/results.ipynb]
"""

import argparse
import json
import os
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))

CELLS = [
("md", r"""# BCA calibration benchmark: semi-synthetic D4RL results

Every result from the semi-synthetic tests of BCA's WBCP calibration banks (Changes 1-11 of
`experiments/wbcp/DEPENDENCE.md`), with the plots and the numbers behind them.

**How the tests work.**
- **Setup.** TD3+BC with BCA's learned scale is trained on half of a D4RL dataset's episodes, frozen,
  and used to score every transition in the other half. That pool is resampled under shifts we impose,
  so the true risk of any threshold is known exactly.
- **Failure.** Target miscoverage is α = 0.1 at credibility β = 0.95. A bank *fails* when the threshold
  it certifies misses more than 10% of test rows; a valid rule fails at most 5% of the time. Each rate
  is the share of 2,000-4,000 simulated banks that fail, with its exact 95% interval.
- **Shifts.** Tilts reweight the pool by exp(γ z), where z is closeness of the logged action to the
  policy's (policy), sparseness of the region (density) or size of the state (state).
- **Rules compared.**
  - Uniform BCA: WBCP with equal weights (BQ-CP), today's BCA calibration.
  - WBCP: with weights estimated by a classifier, or with the exact weights.
  - RCPS: a conservative frequentist baseline.

**Sections.**
1. Setup and data
2. Findings and prediction scorecard
3. Whole-episode banks (Changes 1-3)
4. Thinning and spacing (Changes 4, 6)
5. Distribution shift (Changes 5, 7, 8)
6. BCA's implemented bank (Change 9)
7. Other datasets (Change 10)
8. WBCP under strong shift (Change 11)
9. All results
10. Provenance

To refresh after new runs: `python experiments/wbcp/results_notebook.py`."""),

("md", r"""## 1. Setup and data"""),
("code", r"""import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

%matplotlib inline
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "runs").is_dir() and (p / "calibration").is_dir())
sys.path.insert(0, str(ROOT))
from experiments.wbcp.results_page import build, clean  # the loader behind the HTML results page

D = clean(build())
IDX = {(b["e"], b["t"], b["y"], b["s"], b["n"]): b for b in D["blocks"]}
EXP = {e["id"]: e for e in D["exps"]}
print(f"{len(D['blocks'])} result blocks from {len(D['exps'])} runs, loaded {D['generated']}")

pd.set_option("display.max_rows", 400, "display.max_columns", 40, "display.width", 250, "display.max_colwidth", 60)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True, "legend.frameon": False,
                     "legend.fontsize": 8.5, "hatch.linewidth": 0.8, "axes.titlesize": 10.5})
H, S, R, O, W = "hopper-u100000/", "hopper-u100000-spacing/", "hopper-u100000-reservation/", "other-datasets/", "hopper-u100000-weighting/"
SCORES = ("normalized", "raw")
ARMS = {"BQ-CP": "Uniform BCA", "WBCP": "WBCP, estimated weights", "WBCP (oracle w)": "WBCP, exact weights", "RCPS": "RCPS", "W-CRC": "W-CRC"}
COL = {"BQ-CP": "#b8701a", "WBCP": "#0d8a96", "WBCP (oracle w)": "#0b4f58", "RCPS": "#7d889b", "W-CRC": "#586476",
       "pred": "#8792a3", "target": "#c0392b", "c3": "#4b5fc1", "c4": "#2e9a6b", "c5": "#9b4dca", "c6": "#d0632b", "accent": "#0d7a85"}
TILTS = [("policy", 0.5), ("policy", 1.0), ("density", 0.5), ("density", 1.0), ("state", 0.5), ("state", 1.0)]


def tl(t, y):
    return f"{t} γ={y:g}"


def B(e, t, y, n, s="normalized"):
    return IDX.get((e, t, y, s, n))


def A(e, t, y, n, arm, s="normalized"):
    b = B(e, t, y, n, s)
    return b["A"].get(arm) if b else None


def pct(a):
    return np.nan if a is None else 100 * a["f"] / a["T"]


def fmt(a):
    # failures / trials rounded half-up exactly to one decimal, as in DEPENDENCE.md
    return "–" if a is None else f"{((2000 * a['f'] + a['T']) // (2 * a['T'])) / 10:.1f}"


def fail(a):
    return "–" if a is None else f"{fmt(a)}% [{100 * a['lo']:.1f}, {100 * a['hi']:.1f}]"


def yerr(arms):
    return [[pct(a) - 100 * a["lo"] if a else 0 for a in arms], [100 * a["hi"] - pct(a) if a else 0 for a in arms]]


def target(ax, y=5, label="5% target", color=COL["target"]):
    ax.axhline(y, color=color, ls="--", lw=1)
    ax.annotate(label, xy=(1, y), xycoords=("axes fraction", "data"), ha="right", va="bottom", color=color, fontsize=8)


def grouped(ax, cats, series, ylim=None, ylabel="Banks failing (%)"):
    # series: (label, arms aligned with cats, color, hatch)
    k, x = len(series), np.arange(len(cats))
    w = 0.82 / k
    for i, (label, arms, color, hatch) in enumerate(series):
        ax.bar(x - 0.41 + w * (i + 0.5), [pct(a) for a in arms], w, label=label, color=color, hatch=hatch,
               edgecolor="white" if hatch else color, linewidth=0, yerr=yerr(arms), error_kw=dict(elinewidth=0.8, capsize=1.8, ecolor="#333"))
    ax.set_xticks(x, cats)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)


def line(ax, x, arms, label, color, **kw):
    ax.errorbar(x, [pct(a) for a in arms], yerr=yerr(arms), label=label, color=color, marker=kw.pop("marker", "o"), ms=4, capsize=2.5, lw=1.6, **kw)


def pred(src, design, n, s="normalized"):
    for d in D["pred"][src]["scores"][s]["designs"]:
        if d["design"] == design and d["n"] == n:
            return 100 * d["pmc"]
    return np.nan


def table(rows, columns):
    return pd.DataFrame(rows, columns=columns)


display(table([[k, p["rows"]["population"], p["rows"]["population_episodes"], p["rows"]["training"], p["updates"]]
               for k, p in D["pools"].items()], ["Score pool", "Pool rows", "Pool episodes", "Training rows", "TD3+BC updates"]))"""),

("md", r"""**Terms.**

| Term | Meaning |
|---|---|
| Bank | The calibration set a threshold is computed from. BCA's hopper bank is about 1,100 rows. |
| K | Rows taken from each withheld episode. *Random* puts them anywhere, *spaced* puts one in each K-th of the episode, *BCA sampler* is BCA's own reservation. |
| ρ, design effect | ρ is the correlation between misses in the same episode; the design effect 1 + (K − 1) ρ is how much that inflates a bank's variance. |
| Normalized / raw | The score with and without BCA's learned scale σ(s, a). |
| λ* | The true threshold that exactly 10% of test rows exceed. |"""),

("md", r"""## 2. Findings and prediction scorecard

1. **The original bank broke the guarantee.** Whole-episode banks fail about 28% of the time with no shift,
   against 4.7% for independent rows, because misses cluster within episodes.
2. **A thinned bank fixes it on hopper.** K rows from each of many episodes keeps failure near 5% for K up to 10,
   with no wider threshold; the cost is withheld training data.
3. **WBCP handles shift; uniform BCA does not.** Under density and state shift uniform BCA fails 25-98% while WBCP
   stays at 4-9%. K = 5 holds; K = 10 fails under the density tilt.
4. **BCA now uses the thinned bank.** Its own sampler reproduces the validated behaviour: 5.0% without shift and
   every shift range met.
5. **Hopper's K = 5 does not transfer.** On walker2d-medium-replay and pen-cloned ρ is about 7× hopper's; K = 5 fails
   6-8% and the configured walker2d banks fail 18-30%. Estimated weights also hurt WBCP there even without shift.
6. **WBCP's excess under strong shift is small and finite-sample.** It comes from heavy weights on the misses,
   shrinks as banks grow, and matches what the paper itself reports.

| Change | Question | Verdict | Key result |
|---|---|---|---|
| 1 | Do whole-episode banks keep the guarantee? | not pre-registered | 28.0% fail with no shift (independent rows 4.9%) |
| 2 | Does within-episode dependence explain it? | mostly (retrodiction) | predicted 26.1% against 28.0% observed |
| 3 | Does a bigger whole-episode bank help? | yes; raw score mostly | 26.4-27.8% from 1,103 to 11,500 rows |
| 4 | Does thinning to K rows per episode fix it? | largely | K=5 5.2%, K=10 5.7%, K=25 8.0% |
| 5 | Does the shift picture survive thinning? | partly | K=5 held; K=10 missed under the density tilt |
| 6 | Does spacing rows apart help? | mostly | spaced K=25 6.4% against random 8.0% |
| 7 | Does spaced K=10 hold under shift? | yes on the main part | density 6.7% / 10.1%: spacing did not fix it |
| 8 | Does spaced K=5 hold under shift? | yes | WBCP 5.2-8.5%; uniform BCA 26-98% |
| 9 | Does BCA's implementation behave as validated? | mostly | 5.0% without shift; all shift ranges met |
| 10 | Does it transfer to other datasets? | partly | ρ 0.11-0.12; K=5 fails 6-8%; shift expectation failed |
| 11 | Why does WBCP exceed 5% under strong shift? | mostly | finite-sample; excess 2.6-2.9 → 1.0 points over 8× n |"""),

("md", r"""## 3. Whole-episode banks (Changes 1-3)

**Question.** Does BCA's original bank, whole held-out episodes, give the guarantee WBCP promises?

**Answer.** No. About 28% of banks fail with no shift at all. Rows in one episode are correlated
(ρ = 0.016), and over ~480 rows per episode that inflates the variance about 6.5×, while the posterior treats
the rows as independent. A bigger bank of whole episodes only makes the threshold tighter.

**Verdicts.** Change 1 not pre-registered · Change 2 mostly (a retrodiction: predicted 26.1%, observed 28.0%) ·
Change 3 yes for the normalized score, mostly for the raw score."""),
("code", r"""fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, s in zip(axes, SCORES):
    grouped(ax, ["Independent rows", "Whole episodes"],
            [(ARMS[k], [A(H + "e0_iid", "policy", 0, 1103, k, s), A(H + "e1_blocks", "policy", 0, 1103, k, s)], COL[k], None) for k in ("BQ-CP", "WBCP", "RCPS")],
            ylim=(0, 36))
    target(ax)
    ax.set_title(f"No shift, 1,103-row banks ({s} score, 4,000 trials)")
axes[0].legend(loc="upper left")
plt.tight_layout(); plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
ns = [1103, 2300, 4600, 11500]
for ax, s in zip(axes, SCORES):
    bs = [B(H + "e1_blocks", "policy", 0, n, s) for n in ns]
    x = [b["cs"] for b in bs]
    line(ax, x, [b["A"]["BQ-CP"] for b in bs], "Uniform BCA", COL["BQ-CP"])
    line(ax, x, [b["A"]["WBCP"] for b in bs], "WBCP, estimated weights", COL["WBCP"])
    ax.plot(x, [pred("hopper", "whole episodes", n, s) for n in ns], ls="--", marker="D", mfc="none", color=COL["pred"], label="Predicted (Monte Carlo)")
    ax.set_xscale("log"); ax.set_xticks(x, [f"{v:,.0f}" for v in x]); ax.minorticks_off()
    ax.set_ylim(0, 36); target(ax)
    ax.set_xlabel("Mean bank size (rows, log scale)"); ax.set_ylabel("Banks failing (%)")
    ax.set_title(f"Bigger whole-episode banks ({s} score)")
axes[0].legend(loc="lower left")
plt.tight_layout(); plt.show()"""),
("code", r"""fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
bs = [B(H + "e1_blocks", "policy", 0, n) for n in ns]
x = [b["cs"] for b in bs]
ax = axes[0]
ax.plot(x, [b["A"]["BQ-CP"]["th"] for b in bs], marker="o", color=COL["BQ-CP"], label="Whole episodes, mean threshold")
target(ax, bs[0]["L"], f"true threshold λ* = {bs[0]['L']:.3f}")
iid = B(H + "e0_iid", "policy", 0, 1103)
ax.axhline(iid["A"]["BQ-CP"]["th"], color=COL["pred"], ls=":", lw=1.2, label=f"independent rows ({iid['A']['BQ-CP']['th']:.3f})")
ax.set_xscale("log"); ax.set_xticks(x, [f"{v:,.0f}" for v in x]); ax.minorticks_off()
ax.set_xlabel("Mean bank size (rows, log scale)"); ax.set_ylabel("Certified threshold (normalized)")
ax.set_title("The threshold tightens without becoming reliable"); ax.legend(loc="upper right")
ax = axes[1]
for s, ls in zip(SCORES, ("-", "--")):
    lags = D["pred"]["hopper"]["scores"][s]["lags"]
    ks = sorted(int(k) for k in lags)
    ax.plot(ks, [lags[str(k)] for k in ks], marker="o", ls=ls, color=COL["accent"], label=f"{s} score")
ax.axhline(0, color="#999", lw=0.8)
ax.set_xscale("log"); ax.set_xticks(ks, [str(k) for k in ks]); ax.minorticks_off()
ax.set_xlabel("Steps apart (log scale)"); ax.set_ylabel("Correlation of misses")
ax.set_title("Why: misses cluster within an episode (hopper)"); ax.legend()
plt.tight_layout(); plt.show()"""),
("code", r"""fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
cats = [tl(t, y) for t, y in TILTS]
for ax, (arm, name) in zip(axes, (("WBCP", "WBCP, estimated weights"), ("BQ-CP", "Uniform BCA"))):
    grouped(ax, cats, [("whole episodes", [A("wbcp_bench/blocks", t, y, 1103, arm) for t, y in TILTS], COL[arm], None),
                       ("independent rows", [A("wbcp_bench/main", t, y, 1103, arm) for t, y in TILTS], COL[arm], "///")], ylim=(0, 100))
    target(ax); ax.set_title(f"{name} under shift (2,000 trials)"); ax.tick_params(axis="x", rotation=25)
axes[0].legend(loc="upper left")
plt.tight_layout(); plt.show()

rows = []
for s in SCORES:
    for n in ns:
        b = B(H + "e1_blocks", "policy", 0, n, s)
        rows.append([s, n, round(b["cs"]), fail(b["A"]["BQ-CP"]), fail(b["A"]["WBCP"]), fail(b["A"]["RCPS"]), round(b["A"]["BQ-CP"]["th"], 3)])
display(Markdown("**Whole-episode banks by size, no shift (e1_blocks, 4,000 trials)**"))
display(table(rows, ["Score", "Target rows", "Mean bank", "Uniform BCA", "WBCP est. w", "RCPS", "Mean threshold"]))"""),

("md", r"""## 4. Thinning and spacing (Changes 4, 6)

**Question.** Does taking only K rows from each of many episodes restore validity, and does spacing them apart help?

**Answer.** Yes for small K. K ≤ 10 stays near 5% without shift, the threshold does not widen, and spacing
(one row per K-th of the episode) helps at larger K. The cost is training data: every contributing episode
leaves training (K = 5 withholds about 10-11% of hopper, K = 10 about 5%).

**Verdicts.** Change 4 largely: predictions ran 0.1-1.5 points high. Change 6 mostly: spaced K = 25 fails
6.4% against random 8.0%."""),
("code", r"""ks, sk = [1, 2, 5, 10, 25, 50, 100], [2, 5, 10, 25, 50]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, s in zip(axes, SCORES):
    line(ax, ks, [A(H + f"e2_k{k}", "policy", 0, 1103, "BQ-CP", s) for k in ks], "Random rows, observed", COL["c3"])
    line(ax, sk, [A(S + f"e6_strat_k{k}", "policy", 0, 1103, "BQ-CP", s) for k in sk], "Spaced rows, observed", COL["c4"])
    ax.plot(ks, [pred("hopper", f"{k} per episode", 1103, s) for k in ks], ls="--", marker="D", ms=3.5, mfc="none", color=COL["c3"], label="Random rows, predicted")
    ax.plot(sk, [pred("hopper-spacing", f"{k} per episode, stratified", 1103, s) for k in sk], ls="--", marker="D", ms=3.5, mfc="none", color=COL["c4"], label="Spaced rows, predicted")
    iid = A(H + "e0_iid", "policy", 0, 1103, "BQ-CP", s)
    ax.axhline(pct(iid), color=COL["pred"], ls=":", lw=1.2, label=f"independent rows ({fmt(iid)}%)")
    target(ax)
    ax.set_xscale("log"); ax.set_xticks(ks, [str(k) for k in ks]); ax.minorticks_off(); ax.set_ylim(0, 19)
    ax.set_xlabel("Rows per episode K (log scale)"); ax.set_ylabel("Banks failing (%)")
    ax.set_title(f"Uniform BCA, 1,103-row banks, no shift ({s} score)")
axes[0].legend(loc="upper left")
plt.tight_layout(); plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
p = D["pred"]["hopper"]
share = [100 * math.ceil(1103 / k) * p["pool"]["size_biased_length"] / p["dataset"]["rows"] for k in ks]
axes[0].bar([f"K={k}" for k in ks], share, color="#8a94a6")
for i, v in enumerate(share):
    axes[0].text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=8)
axes[0].set_ylabel("Share of hopper withheld (%)"); axes[0].set_title("What it costs: data withheld from training"); axes[0].set_ylim(0, 62)
lam = B(H + "e0_iid", "policy", 0, 1103)["L"]
axes[1].plot(ks, [A(H + f"e2_k{k}", "policy", 0, 1103, "BQ-CP")["th"] for k in ks], marker="o", color=COL["c3"], label="Random rows")
axes[1].plot(sk, [A(S + f"e6_strat_k{k}", "policy", 0, 1103, "BQ-CP")["th"] for k in sk], marker="o", color=COL["c4"], label="Spaced rows")
target(axes[1], lam, f"true threshold λ* = {lam:.3f}")
axes[1].set_xscale("log"); axes[1].set_xticks(ks, [str(k) for k in ks]); axes[1].minorticks_off()
axes[1].set_xlabel("Rows per episode K (log scale)"); axes[1].set_ylabel("Mean certified threshold"); axes[1].set_title("The threshold does not widen"); axes[1].legend()
plt.tight_layout(); plt.show()

rows = []
for k in ks:
    for s in SCORES:
        r, st = A(H + f"e2_k{k}", "policy", 0, 1103, "BQ-CP", s), A(S + f"e6_strat_k{k}", "policy", 0, 1103, "BQ-CP", s)
        rows.append([k, s, f"{pred('hopper', f'{k} per episode', 1103, s):.1f}%", fail(r), fail(A(H + f"e2_k{k}", "policy", 0, 1103, "WBCP", s)),
                     f"{pred('hopper-spacing', f'{k} per episode, stratified', 1103, s):.1f}%" if k in sk else "–", fail(st), round(r["th"], 3)])
display(Markdown("**K sweep, no shift (uniform BCA unless noted; 4,000 trials)**"))
display(table(rows, ["K", "Score", "Random: predicted", "Random: uniform BCA", "Random: WBCP est. w", "Spaced: predicted", "Spaced: uniform BCA", "Mean threshold"]))"""),

("md", r"""## 5. Distribution shift (Changes 5, 7, 8)

**Question.** Does the WBCP-versus-uniform picture survive thinned banks, and which design holds under shift?

**Answer.**
- Uniform BCA fails 25-98% under density and state shift, while WBCP stays at 4-9%.
- K = 5 holds.
- K = 10 fails 1.3-2.5 points more than K = 5 under the density tilt, spaced or not: it draws from too few
  distinct episodes.

**Verdicts.**
- Change 5 partly: K = 10 missed under the density tilt, and the raw score missed more widely.
- Change 7 yes on the main part: spacing did not fix the density case, as predicted.
- Change 8 yes.

At state γ = 1 WBCP abstains in about a third of banks, which counts as a pass."""),
("code", r"""designs = [("Independent rows", "wbcp_bench/main", COL["pred"], None), ("Whole episodes", "wbcp_bench/blocks", COL["target"], None),
           ("K=10 random", H + "e3_k10", COL["c5"], None), ("K=10 random, rerun", S + "e7_random_k10", COL["c5"], "///"),
           ("K=10 spaced", S + "e7_stratified_k10", COL["c6"], None), ("K=5 random", H + "e3_k5", COL["c3"], None), ("K=5 spaced", S + "e8_stratified_k5", COL["c4"], None)]
cats = [tl(t, y) for t, y in TILTS]
fig, axes = plt.subplots(2, 2, figsize=(13, 7.6))
for ax, arm in zip(axes.ravel(), ("WBCP", "WBCP (oracle w)", "BQ-CP", "RCPS")):
    grouped(ax, cats, [(name, [A(e, t, y, 1103, arm) for t, y in TILTS], color, hatch) for name, e, color, hatch in designs],
            ylim=(0, 100 if arm in ("BQ-CP", "RCPS") else 40))
    target(ax); ax.set_title(f"{ARMS[arm]} (normalized score, 1,103-row banks)"); ax.tick_params(axis="x", rotation=20)
axes[0, 0].legend(loc="upper left", ncol=2)
plt.tight_layout(); plt.show()

for s in SCORES:
    rows = [[tl(t, y)] + [fail(A(e, t, y, 1103, "WBCP", s)) for _, e, _, _ in designs] for t, y in TILTS]
    display(Markdown(f"**WBCP (estimated weights) by tilt and design, {s} score**"))
    display(table(rows, ["Tilt"] + [d[0] for d in designs]))"""),
("code", r"""def pseudo(p, n):
    z = 1.96; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"f": round(p * n), "T": n, "lo": c - h, "hi": c + h}

cats, k5, k10 = [], [], []
for g in (0.5, 1.0):
    key = "1.0" if g == 1 else "0.5"
    cats += [f"γ={g:g}\nChange 5", f"γ={g:g}\nrerun (exact w)", f"γ={g:g}\nChanges 7-8"]
    k5 += [A(H + "e3_k5", "density", g, 1103, "WBCP"), pseudo(D["dominant"][f"gamma={key} 5"]["fail"], 2000), A(S + "e8_stratified_k5", "density", g, 1103, "WBCP")]
    k10 += [A(H + "e3_k10", "density", g, 1103, "WBCP"), pseudo(D["dominant"][f"gamma={key} 10"]["fail"], 2000), A(S + "e7_stratified_k10", "density", g, 1103, "WBCP")]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
x = np.arange(len(cats))
axes[0].errorbar(x - 0.08, [pct(a) for a in k5], yerr=yerr(k5), fmt="o", color=COL["c3"], capsize=2.5, label="K = 5")
axes[0].errorbar(x + 0.08, [pct(a) for a in k10], yerr=yerr(k10), fmt="s", color=COL["c5"], capsize=2.5, label="K = 10")
axes[0].set_xticks(x, cats, fontsize=8); axes[0].set_ylim(0, 14); target(axes[0]); axes[0].legend(loc="upper left")
axes[0].set_ylabel("Banks failing (%)"); axes[0].set_title("K = 10 fails more than K = 5 under the density tilt (WBCP)")
q = ["Q1 smallest", "Q2", "Q3", "Q4 largest"]
for key, name, color in (("iid", "Independent rows", COL["pred"]), ("5", "K = 5 random", COL["c3"]), ("10", "K = 10 random", COL["c5"])):
    axes[1].plot(q, [100 * v for v in D["dominant"][f"gamma=1.0 {key}"]["fail_by_share_quartile"]], marker="o", color=color, label=f"{name}, γ=1")
    axes[1].plot(q, [100 * v for v in D["dominant"][f"gamma=0.5 {key}"]["fail_by_share_quartile"]], ls=":", color=color, label=f"{name}, γ=0.5")
target(axes[1]); axes[1].set_ylim(0, 19); axes[1].legend(fontsize=7.5, ncol=2)
axes[1].set_xlabel("Quartile of the largest episode's weight share"); axes[1].set_ylabel("Banks failing (%)")
axes[1].set_title("Post hoc: failures are not driven by one dominant episode")
plt.tight_layout(); plt.show()

rows = [[k.replace(" gamma=", " γ=").replace(".0", ""), round(v["D_K5"], 3), round(v["D_K10"], 3), f"{100 * v['iid_oracle_fail']:.2f}%",
         f"{100 * v['predicted_K10']:.2f}%", f"{100 * v['observed_K10']:.2f}%", round(v["rho_feature"], 4), round(v["rho_weighted_miss"], 4)] for k, v in D["tilted_hopper"].items()]
display(Markdown("**Post hoc: design effect recomputed under each tilt (exact weights; K = 5 entries are noisy)**"))
display(table(rows, ["Tilt", "D at K=5", "D at K=10", "iid failure", "K=10 predicted", "K=10 observed", "ρ of tilt feature", "ρ of weighted misses"]))"""),

("md", r"""## 6. BCA's implemented bank (Change 9)

**Question.** Does BCA's own implementation of the thinned bank behave like the validated design, and what K does
each config get?

**Answer.**
- On hopper, BCA's sampler (distinct episodes chosen in proportion to length, K spaced rows each) fails 5.0%
  without shift and meets every shift range.
- Under the rule "K = 5, raised until it fits the withholding cap", most TD3+BC / ReBRAC / CQL banks stay inside
  the validated range (K ≤ 10 from at least 100 episodes). Most IQL banks do not.
- IQL on hopper (K = 18, 8,192 rows) lies between 5.0% and 6.7%.

**Verdict.** Mostly: the four expectations came out yes, mostly, yes, yes."""),
("code", r"""cats = ["no shift"] + [tl(t, y) for t, y in TILTS]
row = lambda e0, e1, arm: [A(e0, "policy", 0, 1103, arm)] + [A(e1, t, y, 1103, arm) for t, y in TILTS]
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
for ax, arm in zip(axes, ("WBCP", "BQ-CP")):
    grouped(ax, cats, [("BCA's sampler, K=5", row(R + "r1_resv_k5", R + "r2_resv_k5_shift", arm), COL["accent"], None),
                       ("Spaced, K=5", row(S + "e6_strat_k5", S + "e8_stratified_k5", arm), COL["c4"], None),
                       ("Random, K=5", row(H + "e2_k5", H + "e3_k5", arm), COL["c3"], None)], ylim=(0, 12 if arm == "WBCP" else 100))
    target(ax); ax.set_title(f"{ARMS[arm]}: BCA's sampler against the benchmark designs"); ax.tick_params(axis="x", rotation=25)
axes[0].legend(loc="upper left")
plt.tight_layout(); plt.show()

fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
src = [(H + "e0_iid", 1103), (R + "r3_resv_k18_n8192", 8192), (R + "r4_strat_k18_n8192", 8192)]
grouped(axes[0], ["Independent rows\nn = 1,103", "BCA's sampler\nK = 18, n = 8,192", "Spaced, with replacement\nK = 18, n = 8,192"],
        [(ARMS[k], [A(e, "policy", 0, n, k) for e, n in src], COL[k], None) for k in ("BQ-CP", "WBCP")], ylim=(0, 10))
target(axes[0]); axes[0].set_title("IQL's raised K on hopper: a 5.0-6.7% bracket"); axes[0].legend(loc="upper left")
DS = ["halfcheetah", "hopper", "maze2d", "pen-cloned", "pen-expert", "pen-human", "walker2d"]
x = np.arange(len(DS))
for i, (host, name, color) in enumerate((("td3_bc", "TD3+BC / ReBRAC / CQL", COL["accent"]), ("iql", "IQL", COL["c5"]))):
    rows_ = [D["configs"][host][d] for d in DS]
    for j, r in enumerate(rows_):
        axes[1].bar(x[j] - 0.2 + 0.4 * i, 100 * r["withheld_share_at_seed"], 0.4, color=color, hatch=None if r["dependence_validated"] else "///",
                    edgecolor="white", linewidth=0, label=name if j == 0 else None)
axes[1].axhline(10, color=COL["accent"], ls="--", lw=1); axes[1].axhline(25, color=COL["c5"], ls="--", lw=1)
axes[1].set_xticks(x, DS, rotation=20); axes[1].set_ylabel("Share of dataset withheld (%)"); axes[1].set_ylim(0, 30)
axes[1].set_title("Withheld share per config (hatched: outside the validated range; dashed: caps)"); axes[1].legend(loc="upper left")
plt.tight_layout(); plt.show()

rows = [[d, name, r["target"], f"{100 * r['cap']:.0f}%", r["rows_per_episode"], r["episodes"], r["calibration_rows"], f"{100 * r['withheld_share_at_seed']:.1f}%",
         "inside" if r["dependence_validated"] else "outside"] for d in DS for host, name in (("td3_bc", "TD3+BC / ReBRAC / CQL"), ("iql", "IQL")) for r in [D["configs"][host][d]]]
display(Markdown("**Rows per episode by host and dataset (rule: smallest K ≥ 5 that fits the cap for 1,000 simulated seeds)**"))
display(table(rows, ["Dataset", "Hosts", "Bank target", "Cap", "K", "Episodes", "Bank rows", "Withheld", "Validated range"]))"""),

("md", r"""## 7. Other datasets (Change 10)

**Question.** Does hopper's answer (K = 5) carry over to walker2d-medium-replay and pen-cloned?

**Answer.**
- No. ρ is 0.120 on walker2d and 0.108 on pen-cloned, about 7× hopper's.
- K = 5 fails 6-8%, and the configured walker2d banks fail 18-30%.
- On walker2d the correlation never fades within an episode, so spacing cannot help.
- Estimated weights also hurt WBCP there even without shift, unless the weight model gets far more data: 10×
  more brought it from 14.9% to 5.7%.

**Verdicts.**
- Stage A mostly: ρ landed just above the pre-registered 0.02-0.10.
- Stage B without shift mostly.
- Stage B under shift no on walker2d, mixed on pen-cloned. Failure fell where it was expected to rise; a
  post-hoc explanation is in the last figure.
- Stage D yes."""),
("code", r"""DSETS = [("hopper", "hopper-medium", COL["accent"]), ("walker2d-n1024", "walker2d-medium-replay", COL["c5"]), ("pen-n1024", "pen-cloned", COL["c6"])]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
x = np.arange(len(DSETS))
for i, s in enumerate(SCORES):
    vals = [D["pred"][src]["scores"][s]["rho"] for src, _, _ in DSETS]
    axes[0].bar(x - 0.2 + 0.4 * i, vals, 0.4, color=[c for _, _, c in DSETS], hatch=None if i == 0 else "///", edgecolor="white", linewidth=0, label=f"{s} score")
    for j, v in enumerate(vals):
        axes[0].text(x[j] - 0.2 + 0.4 * i, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)
axes[0].set_xticks(x, [name for _, name, _ in DSETS]); axes[0].set_ylabel("ρ of misses within an episode"); axes[0].set_ylim(0, 0.16)
axes[0].set_title("Within-episode correlation (solid: normalized, hatched: raw)")
for src, name, color in DSETS:
    lags = D["pred"][src]["scores"]["normalized"]["lags"]
    kk = sorted(int(k) for k in lags)
    axes[1].plot(kk, [lags[str(k)] for k in kk], marker="o", color=color, label=name)
axes[1].axhline(0, color="#999", lw=0.8); axes[1].set_xscale("log"); axes[1].set_xticks(kk, [str(k) for k in kk]); axes[1].minorticks_off()
axes[1].set_xlabel("Steps apart (log scale)"); axes[1].set_ylabel("Correlation of misses"); axes[1].set_title("How far apart misses stay correlated (normalized)"); axes[1].legend()
plt.tight_layout(); plt.show()

DESIGNS = [("hopper · K=5 BCA sampler", R + "r1_resv_k5", 1103, "hopper-k5", "5 per episode, BCA reservation"),
           ("walker2d · whole episodes", O + "w_blocks_n1024", 1024, "walker2d-n1024", "whole episodes"),
           ("walker2d · K=5 spaced", O + "w_strat5_n1024", 1024, "walker2d-n1024", "5 per episode, stratified"),
           ("walker2d · K=23 BCA sampler", O + "w_resv23_n1024", 1024, "walker2d-n1024", "23 per episode, BCA reservation"),
           ("walker2d · K=23 spaced", O + "w_strat23_n1024", 1024, "walker2d-n1024", "23 per episode, stratified"),
           ("walker2d · K=67 BCA sampler, n=8,192", O + "w_resv67_n8192", 8192, "walker2d-n8192", "67 per episode, BCA reservation"),
           ("walker2d · K=67 spaced, n=8,192", O + "w_strat67_n8192", 8192, "walker2d-n8192", "67 per episode, stratified"),
           ("pen-cloned · whole episodes", O + "p_blocks_n1024", 1024, "pen-n1024", "whole episodes"),
           ("pen-cloned · K=5 BCA sampler", O + "p_resv5_n1024", 1024, "pen-n1024", "5 per episode, BCA reservation"),
           ("pen-cloned · K=5 spaced", O + "p_strat5_n1024", 1024, "pen-n1024", "5 per episode, stratified"),
           ("pen-cloned · K=11 BCA sampler, n=8,192", O + "p_resv11_n8192", 8192, "pen-n8192", "11 per episode, BCA reservation"),
           ("pen-cloned · K=11 spaced, n=8,192", O + "p_strat11_n8192", 8192, "pen-n8192", "11 per episode, stratified")]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
y = np.arange(len(DESIGNS))[::-1]
for ax, s in zip(axes, SCORES):
    arms = [A(e, "policy", 0, n, "BQ-CP", s) for _, e, n, _, _ in DESIGNS]
    ax.errorbar([pct(a) for a in arms], y, xerr=yerr(arms), fmt="o", color=COL["BQ-CP"], capsize=2.5, label="Observed (uniform BCA)")
    ax.plot([pred(src, dz, n, s) for _, _, n, src, dz in DESIGNS], y, "D", mfc="none", color="#222", label="Predicted before the run")
    ax.axvline(5, color=COL["target"], ls="--", lw=1)
    ax.set_yticks(y, [d[0] for d in DESIGNS]); ax.set_xlim(0, 55); ax.set_xlabel("Banks failing (%)"); ax.set_title(f"Predicted against observed, no shift ({s} score)")
axes[0].legend(loc="lower right")
plt.tight_layout(); plt.show()

rows = [[label, s, f"{pred(src, dz, n, s):.1f}%", fail(A(e, "policy", 0, n, "BQ-CP", s)), fail(A(e, "policy", 0, n, "WBCP (oracle w)", s)), fail(A(e, "policy", 0, n, "WBCP", s))]
        for label, e, n, src, dz in DESIGNS for s in SCORES]
display(Markdown("**Designs without shift (4,000 trials)**"))
display(table(rows, ["Design", "Score", "Predicted", "Uniform BCA", "WBCP exact w", "WBCP est. w"]))"""),
("code", r"""cats = ["no shift"] + [tl(t, y) for t, y in TILTS]
row = lambda e0, e1, arm: [A(e0, "policy", 0, 1024, arm)] + [A(e1, t, y, 1024, arm) for t, y in TILTS]
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
for ax, arm in zip(axes, ("WBCP (oracle w)", "BQ-CP")):
    grouped(ax, cats, [("walker2d, K = 23", row(O + "w_resv23_n1024", O + "w_resv23_shift", arm), COL["c5"], None),
                       ("pen-cloned, K = 5", row(O + "p_resv5_n1024", O + "p_resv5_shift", arm), COL["c6"], None)], ylim=(0, 25 if arm != "BQ-CP" else 100))
    target(ax); ax.set_title(f"{ARMS[arm]}: configured TD3+BC-family bank under shift"); ax.tick_params(axis="x", rotation=25)
axes[0].legend(loc="upper right")
plt.tight_layout(); plt.show()

src = [("walker2d\nn = 1,024", O + "w_iid_n1024", 1024), ("walker2d\nn = 8,192", O + "w_iid_n8192", 8192), ("walker2d, n = 8,192\n10× weight fit", O + "d_w_iid_n8192_fit10000", 8192),
       ("pen-cloned\nn = 1,024", O + "p_iid_n1024", 1024), ("pen-cloned\nn = 8,192", O + "p_iid_n8192", 8192), ("hopper\nn = 1,103", H + "e0_iid", 1103)]
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
grouped(axes[0], [s_[0] for s_ in src], [(ARMS[k], [A(e, "policy", 0, n, k) for _, e, n in src], COL[k], None) for k in ("WBCP", "WBCP (oracle w)", "BQ-CP")], ylim=(0, 20))
target(axes[0]); axes[0].set_title("Estimated against exact weights, no shift, independent rows"); axes[0].legend(loc="upper left"); axes[0].tick_params(axis="x", labelsize=8)
x = np.arange(6); labels = None
for i, (key, name, color) in enumerate((("walker2d-medium-replay", "walker2d", COL["c5"]), ("pen-cloned", "pen-cloned", COL["c6"]))):
    rows_ = D["tilted_rho"][key]
    labels = ["no shift" if r["gamma"] == 0 else tl(r["tilt"], r["gamma"]) for r in rows_]
    axes[1].bar(x - 0.3 + 0.3 * i, [r["rho_miss"] for r in rows_], 0.15, color=color, hatch="///", edgecolor="white", linewidth=0, label=f"{name}: ρ of misses")
    axes[1].bar(x - 0.15 + 0.3 * i, [r["rho_weighted"] for r in rows_], 0.15, color=color, label=f"{name}: ρ of weighted term")
axes[1].set_xticks(x, labels, rotation=20); axes[1].set_ylabel("Within-episode correlation"); axes[1].legend(fontsize=7.5)
axes[1].set_title("Post hoc: under a tilt the weighted term is less clustered")
plt.tight_layout(); plt.show()

rows = []
for label, e0, e1 in (("walker2d K=23", O + "w_resv23_n1024", O + "w_resv23_shift"), ("pen-cloned K=5", O + "p_resv5_n1024", O + "p_resv5_shift")):
    for c_, e, t, y in [("no shift", e0, "policy", 0)] + [(tl(t, y), e1, t, y) for t, y in TILTS]:
        a = A(e, t, y, 1024, "WBCP (oracle w)")
        rows.append([label, c_, fail(a), fail(A(e, t, y, 1024, "WBCP")), fail(A(e, t, y, 1024, "BQ-CP")), "–" if a is None or a.get("x") is None else f"{100 * a['x']:.2f}"])
display(Markdown("**Configured bank under shift (normalized score, 4,000 trials)**"))
display(table(rows, ["Bank", "Tilt", "WBCP exact w", "WBCP est. w", "Uniform BCA", "Mean excess | fail (points, exact w)"]))
rows = [[key, r["tilt"], r["gamma"], round(r["lambda_star"], 3), round(r["rho_miss"], 4), round(r["rho_weighted"], 4), round(r["design_effect"], 2), f"{100 * r['predicted_failure']:.1f}%"]
        for key, rr in D["tilted_rho"].items() for r in rr]
display(Markdown("**Post hoc: tilted correlation and the failure it implies**"))
display(table(rows, ["Dataset", "Tilt", "γ", "λ*", "ρ of misses", "ρ of weighted term", "Design effect", "Implied failure"]))"""),

("md", r"""## 8. WBCP under strong shift (Change 11)

**Question.** Why does WBCP fail 7-9% under strong shift on hopper, even with independent rows and exact weights?

**Answer.** It is finite-sample behaviour that the paper allows and also reports.
- **What drives it.** Heavy weights (33-39× the mean) that sit on the misses. A bank that misses those few rows
  reports a posterior spread about a third too small.
- **It shrinks.** The excess falls as banks grow, and failing banks exceed α by under a point.
- **What the paper promises.** Corollary 3 is posterior credibility. The frequency statement is Theorem 4, whose
  slack η_n is larger than α at these sizes. The paper's own §4.2 experiment reports 7.5-7.9% failure,
  attributed to η_n.

**Verdicts.**
- Theory yes.
- A mostly: all twelve failure rates were inside their ranges, and two magnitude predictions were slightly low.
- B yes.
- C partly: the failing-bank spread ratio was as predicted, but the correlation did not separate the cases.

The runs in this section use the normalized score only."""),
("code", r"""NS = [1103, 2206, 4412, 8824]
SWEEP = [("density γ=1", W + "a_density1", "density", 1.0, COL["c5"], [7.4, 6.7, 6.2, 5.9]),
         ("state γ=0.5", W + "a_state05", "state", 0.5, COL["c6"], [8.0, 7.1, 6.5, 6.1]),
         ("policy γ=1 (control)", W + "a_policy1", "policy", 1.0, COL["c3"], None)]
fig, ax = plt.subplots(figsize=(11, 4))
for name, e, t, y, color, path in SWEEP:
    if path:
        ax.fill_between(NS, [v - 0.8 for v in path], [v + 0.8 for v in path], color=color, alpha=0.12, lw=0)
    line(ax, NS, [A(e, t, y, n, "WBCP (oracle w)") for n in NS], f"{name}, exact weights", color)
    ax.plot(NS, [pct(A(e, t, y, n, "WBCP")) for n in NS], ls=":", marker="o", mfc="none", ms=4, color=color, label=f"{name}, estimated weights")
target(ax); ax.set_xscale("log"); ax.set_xticks(NS, [f"{n:,}" for n in NS]); ax.minorticks_off(); ax.set_ylim(0, 10)
ax.set_xlabel("Bank size n (log scale)"); ax.set_ylabel("Banks failing (%)")
ax.set_title("The excess shrinks as the bank grows (shaded: pre-registered paths ±0.8)"); ax.legend(ncol=2, loc="lower left")
plt.tight_layout(); plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for name, e, t, y, color, _ in SWEEP:
    arms = [A(e, t, y, n, "WBCP (oracle w)") for n in NS]
    axes[0].plot(NS, [100 * a["x"] for a in arms], marker="o", color=color, label=name)
    axes[1].plot(NS, [100 * a["q99"] for a in arms], marker="o", color=color, label=name)
for ax in axes:
    ax.set_xscale("log"); ax.set_xticks(NS, [f"{n:,}" for n in NS]); ax.minorticks_off(); ax.set_xlabel("Bank size n (log scale)"); ax.legend()
axes[0].set_ylim(0, None); axes[0].set_ylabel("Mean excess over α (points)"); axes[0].set_title("How far failing banks exceed α (exact weights)")
target(axes[1], 10, "α = 10%"); axes[1].set_ylabel("99th percentile of realized risk (%)"); axes[1].set_title("The worst 1% of banks (exact weights)")
plt.tight_layout(); plt.show()

rows = [[name, n, fail(a), fail(A(e, t, y, n, "WBCP")), f"{100 * a['r']:.2f}%", f"{100 * a['q95']:.2f}%", f"{100 * a['q99']:.2f}%", f"{100 * a['x']:.2f}", round(B(e, t, y, n)["ne"])]
        for name, e, t, y, _, _ in SWEEP for n in NS for a in [A(e, t, y, n, "WBCP (oracle w)")]]
display(Markdown("**Bank-size sweep (iid banks, normalized score, 4,000 trials per row)**"))
display(table(rows, ["Tilt", "n", "WBCP exact w", "WBCP est. w", "Mean risk", "q95", "q99", "Mean excess | fail (points)", "Calibration n_eff"]))"""),
("code", r"""fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw={"width_ratios": [1, 1.2, 1.1]})
src = [(W + "a_density1", "density", 1.0), (W + "b_shuffled_density1", "density", 1.0), (W + "a_policy1", "policy", 1.0)]
grouped(axes[0], ["density γ=1\naligned", "density γ=1\nshuffled", "policy γ=1\nunaligned"], [(ARMS[k], [A(e, t, y, 1103, k) for e, t, y in src], COL[k], None) for k in ("WBCP (oracle w)", "WBCP")], ylim=(0, 10))
target(axes[0]); axes[0].set_title("Alignment with the score drives it"); axes[0].legend(loc="upper right", fontsize=7.5)
cases = [("aligned", "density γ=1\naligned"), ("shuffled", "weights\nshuffled"), ("policy", "policy γ=1")]
metrics = [("failing_median_posterior_sd_over_across", "Failing banks: posterior ÷ true spread", COL["target"]),
           ("passing_median_posterior_sd_over_across", "Passing banks: posterior ÷ true spread", COL["pred"]),
           ("sd_ratio_across_over_posterior", "All banks: true ÷ mean posterior spread", COL["accent"])]
x = np.arange(3)
for i, (key, name, color) in enumerate(metrics):
    axes[1].bar(x - 0.27 + 0.27 * i, [D["mech"][c][key] for c, _ in cases], 0.27, color=color, label=name)
axes[1].axhline(1, color="#666", ls="--", lw=1); axes[1].set_xticks(x, [c[1] for c in cases]); axes[1].set_ylim(0, 1.3)
axes[1].set_title("Failing banks under-report their spread"); axes[1].legend(fontsize=7, loc="lower right")
tcases = [("policy", 0.0, "no shift"), ("policy", 1.0, "policy γ=1"), ("density", 0.5, "density γ=0.5"), ("density", 1.0, "density γ=1"), ("state", 0.5, "state γ=0.5"), ("state", 1.0, "state γ=1")]
x = np.arange(len(tcases))
for i, n in enumerate((1103, 8824)):
    vals = [next((r["eta_bernstein"] for r in D["slack"] if r["tilt"] == t and r["gamma"] == y and r["n"] == n), None) for t, y, _ in tcases]
    axes[2].bar(x - 0.2 + 0.4 * i, [np.nan if v is None else v for v in vals], 0.4, color=COL["c5"] if i == 0 else COL["accent"], label=f"n = {n:,}")
axes[2].axhline(0.1, color=COL["target"], ls="--", lw=1); axes[2].set_yscale("log"); axes[2].set_xticks(x, [c[2] for c in tcases], rotation=25, fontsize=8)
axes[2].set_ylabel("η_n (log scale; missing = infinite)"); axes[2].set_title("Theorem 4's slack exceeds α = 0.1 everywhere"); axes[2].legend(fontsize=7.5)
plt.tight_layout(); plt.show()

rows = [[c, round(D["mech"][k]["fail"] * 100, 2), round(D["mech"][k]["fail_from_posterior_at_lambda_star"] * 100, 2), round(D["mech"][k]["agreement"], 4), round(D["mech"][k]["sd_ratio_across_over_posterior"], 3),
         round(D["mech"][k]["failing_median_posterior_sd_over_across"], 3), round(D["mech"][k]["passing_median_posterior_sd_over_across"], 3), round(D["mech"][k]["corr_r_hat_posterior_sd"], 3), round(D["mech"][k]["r_hat_skewness"], 3)]
        for k, c in (("aligned", "density γ=1 aligned"), ("shuffled", "density γ=1 shuffled"), ("policy", "policy γ=1"))]
display(Markdown("**Per-bank mechanism (exact weights, 4,000 banks × 4,000 posterior draws)**"))
display(table(rows, ["Case", "Failure (%)", "From posterior at λ* (%)", "Agreement", "True ÷ mean posterior SD", "Failing: posterior ÷ true SD", "Passing: posterior ÷ true SD", "corr(R̂, SD)", "Skewness of R̂"]))
rows = [[r["tilt"], r["gamma"], r["n"], round(r["B"], 2), round(r["E_w2"], 3), round(r["n_eff_pop"]), "∞" if r["eta_hoeffding"] is None else round(r["eta_hoeffding"], 3), "∞" if r["eta_bernstein"] is None else round(r["eta_bernstein"], 3)] for r in D["slack"]]
display(Markdown("**Theorem 4 slack for the benchmark's weights (δ = 0.05, m = ⌈√n⌉)**"))
display(table(rows, ["Tilt", "γ", "n", "B = max w ÷ mean w", "E[w²]", "n_eff (population)", "η_n Hoeffding", "η_n Bernstein"]))"""),
("md", r"""**What the paper says** (Lou and Luo, *Weighted Bayesian Conformal Prediction*, arXiv:2604.06464v3):

> Credibility no longer coincides with confidence. (§1)

> WBCP certifies every trial at 7.5–7.9%. The oracle row attributes the residual to η_n, not weight estimation.
> (§4.2; Table 2 at n = 250: 7.9% with estimated weights, 5.6% with oracle weights)"""),

("md", r"""## 9. All results

Every result block from every run: one row per experiment, tilt, bank size and score, with each rule's failure
rate and 95% interval. The last columns (mean risk, percentiles, excess, threshold, abstention) are for WBCP with
estimated weights. Percentiles and excess exist only for runs from Change 10 on."""),
("code", r"""rows = []
for b in D["blocks"]:
    e, d = EXP[b["e"]], b["A"].get("WBCP", {})
    rows.append([e["change"], b["e"], e["dataset"], e["design"], b["n"], "none" if b["y"] == 0 else b["t"], b["y"], b["s"], round(b["L"], 3), round(b["ne"]),
                 fail(b["A"].get("BQ-CP")), fail(b["A"].get("WBCP")), fail(b["A"].get("WBCP (oracle w)")), fail(b["A"].get("RCPS")), fail(b["A"].get("W-CRC")),
                 None if d.get("r") is None else round(100 * d["r"], 2), None if d.get("q95") is None else round(100 * d["q95"], 2),
                 None if d.get("q99") is None else round(100 * d["q99"], 2), None if d.get("x") is None else round(100 * d["x"], 2),
                 None if d.get("th") is None else round(d["th"], 3), None if d.get("ab") is None else round(100 * d["ab"], 1)])
ALL = table(rows, ["Change", "Experiment", "Dataset", "Bank design", "n", "Tilt", "γ", "Score", "λ*", "n_eff", "Uniform BCA", "WBCP est. w", "WBCP exact w",
                   "RCPS", "W-CRC", "WBCP mean risk (%)", "WBCP q95 (%)", "WBCP q99 (%)", "WBCP excess | fail", "WBCP threshold", "WBCP abstain (%)"])
print(f"{len(ALL)} result blocks")
ALL"""),

("md", r"""## 10. Provenance

Expectations were written to timestamped files before each run. They are not under version control, so their
SHA-256 values are listed to be sent to the advisor or committed. The full write-up, with every verdict and its
reasoning, is `experiments/wbcp/DEPENDENCE.md`."""),
("code", r"""display(table([[x["change"], x["label"], x["header"], x["file"], x["sha"]] for x in D["expectations"]], ["Change", "Expectations", "Header", "File", "SHA-256"]))
display(table([[e["change"], e["label"], e["file"], e["dataset"], e["design"], e["trials"], e["seed"]] for e in D["exps"]], ["Change", "Experiment", "File", "Dataset", "Bank design", "Trials", "Seed"]))"""),
]


def notebook_cells():
    cells = []
    for kind, source in CELLS:
        lines = source.strip("\n").split("\n")
        src = [line + "\n" for line in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": "markdown" if kind == "md" else "code", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": src}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return cells


def execute(cells, cwd):
    from jupyter_client.manager import start_new_kernel
    manager, client = start_new_kernel(kernel_name="python3", cwd=cwd)
    failures = []
    try:
        for cell in cells:
            if cell["cell_type"] != "code":
                continue
            msg_id = client.execute("".join(cell["source"]))
            outputs = []
            while True:
                msg = client.get_iopub_msg(timeout=900)
                if msg["parent_header"].get("msg_id") != msg_id:
                    continue
                kind, content = msg["msg_type"], msg["content"]
                if kind == "status" and content["execution_state"] == "idle":
                    break
                if kind == "stream":
                    if outputs and outputs[-1]["output_type"] == "stream" and outputs[-1]["name"] == content["name"]:
                        outputs[-1]["text"] += content["text"]
                    else:
                        outputs.append({"output_type": "stream", "name": content["name"], "text": content["text"]})
                elif kind in ("display_data", "execute_result"):
                    out = {"output_type": kind, "data": content["data"], "metadata": content.get("metadata", {})}
                    if kind == "execute_result":
                        out["execution_count"] = content["execution_count"]
                    outputs.append(out)
                elif kind == "error":
                    outputs.append({"output_type": "error", "ename": content["ename"], "evalue": content["evalue"], "traceback": content["traceback"]})
                    failures.append(f"{content['ename']}: {content['evalue']}")
            reply = client.get_shell_msg(timeout=900)
            while reply["parent_header"].get("msg_id") != msg_id:
                reply = client.get_shell_msg(timeout=900)
            cell["execution_count"] = reply["content"].get("execution_count")
            for stream in outputs:
                if stream["output_type"] == "stream":
                    stream["text"] = stream["text"].splitlines(keepends=True)
            cell["outputs"] = outputs
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", default=os.path.join(HERE, "results.ipynb"))
    parser.add_argument("--no-execute", action="store_true", help="write the notebook without outputs")
    args = parser.parse_args(argv)
    cells = notebook_cells()
    failures = [] if args.no_execute else execute(cells, os.path.dirname(os.path.abspath(args.output)))
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                             "language_info": {"name": "python", "pygments_lexer": "ipython3"}},
                "nbformat": 4, "nbformat_minor": 5}
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(notebook, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    print(f"wrote {args.output}: {len(cells)} cells" + (f"; {len(failures)} cell errors: {failures}" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
