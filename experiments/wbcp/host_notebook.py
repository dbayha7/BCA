"""Build and execute one results notebook per host: its calibration experiments on all seven datasets.

Each notebook reads the host's matrix (host_matrix.py: score pools, dependence predictions and
benchmark runs) and shows every number with its plot: within-episode dependence per dataset, failure
by bank design against the pre-registered predictions, the configured bank, distribution shift,
estimated against exact weights, threshold cost and every raw result. A written interpretation, when
experiments/wbcp/host_findings/<host>.md exists, is inserted as the findings section. The notebook is
executed here, so the saved file carries its outputs (needs matplotlib, pandas, pyyaml, ipykernel and
jupyter_client).

python experiments/wbcp/host_notebook.py --host rebrac [--output experiments/wbcp/results_rebrac.ipynb]
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from experiments.wbcp.host_matrix import DATASETS, HOSTS, run_root  # noqa: E402
from experiments.wbcp.results_notebook import execute, notebook_cells  # noqa: E402

INTRO = r"""# BCA calibration benchmark: __NAME__ host, all seven datasets

Semi-synthetic tests of BCA's WBCP calibration banks with __NAME__ as the host algorithm, on every
dataset BCA is configured for. The same notebook exists for each host (`results_td3_bc.ipynb`,
`results_rebrac.ipynb`, `results_cql.ipynb`, `results_iql.ipynb`), built by
`experiments/wbcp/host_notebook.py` from the matrix in `experiments/wbcp/host_matrix.py`.

**How the tests work.**
- **Score pool.** __NAME__ with BCA's learned scale trains for 100,000 updates on half of a dataset's
  episodes (CPU, no refresh, so the host runs natively while σ(s, a) is fit). The frozen model then scores
  every transition of the other half exactly as __NAME__'s own BCA refresh scores a held-out bank:
  residual = target − Q, score = |residual| / σ(s, a). That half is the pool.
- **Banks.** A simulated calibration bank is drawn from the pool with one of the bank designs below, and
  the test distribution is the pool itself or a tilted copy of it, so the true miscoverage of any threshold
  is known exactly.
- **Failure.** Target miscoverage α = 0.1 at credibility β = 0.95: a bank *fails* when the threshold it
  certifies misses more than 10% of test rows. A valid rule fails at most 5% of the time. Each rate is the
  share of 4,000 simulated banks that fail, with its exact 95% interval.
- **Rules.** Uniform BCA is WBCP with equal weights (BQ-CP), today's BCA calibration. WBCP uses weights
  estimated by a classifier, or the exact weights. RCPS is a conservative frequentist baseline.
- **Bank size.** __NAME__'s configured bank: n and K per dataset from `configs/__HOST__.yaml`.

**Bank designs** (no shift unless stated):

| Design | What it is | Why it is here |
|---|---|---|
| Independent rows | n rows drawn independently | the reference: should fail about 5% |
| K = 2, 5, 10 spaced | K rows from each episode, one per K-th of it | how failure grows with rows per episode |
| Whole episodes | BCA's original bank | the dependence stress test |
| Configured, BCA sampler | K rows per episode with BCA's own reservation (`calibration/bank.py`) | what __NAME__ + BCA deploys |
| Configured, spaced | the same K with the spaced sampler | the other side of the finite-pool bracket |
| Independent rows, large bank | 8,192 rows (1,194 on pen-human) | estimated against exact weights |
| K = 5 spaced, six tilts | policy, density and state tilts at γ = 0.5 and 1 | distribution shift |

**Expectations** were written before the runs (Stage A before the pools, Stage B with the Monte Carlo
predictions before any benchmark run) and are shown in full in section 2 with their SHA-256."""

SETUP = r"""import math
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

%matplotlib inline
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "runs").is_dir() and (p / "calibration").is_dir())
sys.path.insert(0, str(ROOT))
from experiments.wbcp.host_matrix import DATASETS, HOSTS, TILTS, load_results

HOST = "__HOST__"
R = load_results(HOST)
pd.set_option("display.max_rows", 500, "display.max_columns", 40, "display.width", 250, "display.max_colwidth", 70)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True, "legend.frameon": False,
                     "legend.fontsize": 8.5, "axes.titlesize": 10.5})
ARMS = {"BQ-CP": "Uniform BCA", "WBCP": "WBCP, estimated weights", "WBCP (oracle w)": "WBCP, exact weights", "RCPS": "RCPS"}
COL = {"BQ-CP": "#b8701a", "WBCP": "#0d8a96", "WBCP (oracle w)": "#0b4f58", "RCPS": "#7d889b", "pred": "#4b5fc1", "target": "#c0392b"}
SHORT = {"hopper": "hopper-medium", "walker2d": "walker2d-medium-replay", "halfcheetah": "halfcheetah-medium-expert",
         "maze2d": "maze2d-large", "pen-cloned": "pen-cloned", "pen-expert": "pen-expert", "pen-human": "pen-human"}
SCORES = ("normalized", "raw")


def P(d):
    return d.replace("-", "")


# the no-shift designs, as (column label, run-name suffix for a dataset entry, dependence.py design label)
def designs(e):
    k = e["k"]
    rows = [("Independent rows", "iid", None), ("K=2 spaced", "strat2", "2 per episode, stratified"),
            ("K=5 spaced", "strat5", "5 per episode, stratified"), ("K=10 spaced", "strat10", "10 per episode, stratified"),
            ("Configured K, spaced", f"strat{k}", f"{k} per episode, stratified"),
            ("Configured K, BCA sampler", f"resv{k}", f"{k} per episode, BCA reservation"),
            ("Whole episodes", "blocks", "whole episodes")]
    if e["n"] < e["big"]:
        rows.append(("Independent rows, large bank", "iid_big", None))
    return rows


COLUMNS = ["Independent rows", "K=2 spaced", "K=5 spaced", "K=10 spaced", "Configured K, spaced", "Configured K, BCA sampler",
           "Whole episodes", "Independent rows, large bank"]


def block(d, suffix, t="policy", y=0.0, s="normalized"):
    run = R["datasets"][d]["runs"].get(f"{P(d)}_{suffix}")
    if run is None:
        return None
    return next((b for b in run["blocks"] if b["t"] == t and b["y"] == y and b["s"] == s), None)


def arm(d, suffix, a="BQ-CP", t="policy", y=0.0, s="normalized"):
    b = block(d, suffix, t, y, s)
    return None if b is None else b["A"].get(a)


def pct(a):
    return np.nan if a is None else 100 * a["f"] / a["T"]


def fmt(a):
    # failures / trials rounded half-up to one decimal, as in DEPENDENCE.md
    return "–" if a is None else f"{((2000 * a['f'] + a['T']) // (2 * a['T'])) / 10:.1f}"


def fail(a):
    return "–" if a is None else f"{fmt(a)}% [{100 * a['lo']:.1f}, {100 * a['hi']:.1f}]"


def yerr(arms):
    return [[pct(a) - 100 * a["lo"] if a else 0 for a in arms], [100 * a["hi"] - pct(a) if a else 0 for a in arms]]


def prediction(d, label, s="normalized"):
    # (predicted failure in %, Monte Carlo design effect) from dependence.py; independent rows: 5% and D = 1
    e = R["datasets"][d]
    if label is None:
        return 5.0, 1.0
    if e["pred"] is None:
        return np.nan, np.nan
    for x in e["pred"]["scores"][s]["designs"]:
        if x["design"] == label and x["n"] == e["n"]:
            return (np.nan if x["pmc"] is None else 100 * x["pmc"]), x["Dmc"]
    return np.nan, np.nan


def verdict(obs, pred, D):
    # Stage B tolerance: within about 2 points of the prediction, 5 where the design effect exceeds 3
    if np.isnan(obs) or np.isnan(pred):
        return "–"
    tol = 5 if (D is not None and D > 3) else 2
    return "met" if abs(obs - pred) <= tol else ("missed (higher)" if obs > pred else "missed (lower)")


def target(ax, y=5, label="5% target"):
    ax.axhline(y, color=COL["target"], ls="--", lw=1)
    ax.annotate(label, xy=(1, y), xycoords=("axes fraction", "data"), ha="right", va="bottom", color=COL["target"], fontsize=8)


def table(rows, columns):
    return pd.DataFrame(rows, columns=columns)


# a pool whose critic diverged or was still growing is flagged everywhere with a warning sign
FLAG = {d: R["datasets"][d]["pool"]["health"]["status"] for d in DATASETS
        if R["datasets"][d]["pool"] and R["datasets"][d]["pool"]["health"] and R["datasets"][d]["pool"]["health"]["status"] != "ok"}
for d in FLAG:
    SHORT[d] += " ⚠"
if FLAG:
    display(Markdown("**⚠ Critic not converged:** " + "; ".join(f"{SHORT[d]} ({s})" for d, s in FLAG.items())
                     + ". Their calibration results are exact for those scores, but a diverging critic's residuals do not "
                       "represent a trained " + R["name"] + " critic, so they are left out of conclusions about dependence."))
have = [d for d in DATASETS if R["datasets"][d]["pool"]]
runs = sum(len(R["datasets"][d]["runs"]) for d in DATASETS)
missing = sum(len(R["datasets"][d]["missing"]) for d in DATASETS)
print(f"{R['name']}: {len(have)}/7 score pools, {runs} benchmark runs loaded, {missing} not run yet (results root {R['root']})")"""

POOLS_MD = r"""## 1. Score pools and within-episode dependence

One pool per dataset: the withheld half of its episodes, scored by the frozen __NAME__ model. ρ is the
intra-episode correlation of misses at the true threshold λ\* (the indicator score > λ\*), for the
normalized score with the raw score in brackets. It drives everything below: a bank with m rows per episode
has its variance inflated by the design effect D ≈ 1 + (m − 1)ρ, and its failure rate rises to about
1 − Φ(1.645 / √D). The lag correlations show how fast dependence decays along an episode."""

POOLS = r"""rows = []
for d in DATASETS:
    e = R["datasets"][d]
    p, q = e["pool"], e["pred"]
    if p is None:
        rows.append([SHORT[d], "pool not built yet"] + [None] * 13)
        continue
    sn, sr = (q["scores"]["normalized"], q["scores"]["raw"]) if q else (None, None)
    lag = lambda k: None if sn is None or sn["lags"].get(k) is None else round(sn["lags"][k], 3)
    h = p["health"] or {}
    rows.append([SHORT[d], p["rows"]["population"], p["rows"]["population_episodes"], p["rows"]["training"], p["updates"],
                 e["n"], e["k"], math.ceil(e["n"] / e["k"]), None if sn is None else round(sn["lambda_star"], 3),
                 None if sn is None else f"{sn['rho']:.3f} ({sr['rho']:.3f})", lag("1"), lag("50"),
                 h.get("status", "–"), None if h.get("q_growth_last_fifth") is None else f"{100 * h['q_growth_last_fifth']:+.0f}%",
                 None if p["frac_score_le_1"] is None else round(p["frac_score_le_1"], 3)])
display(table(rows, ["Dataset", "Pool rows", "Pool episodes", "Training rows", "Updates", "Bank n", "K", "Episodes in bank",
                     "λ* (normalized)", "ρ normalized (raw)", "Lag-1 corr.", "Lag-50 corr.", "Critic", "Q growth, last 20%",
                     "Scores ≤ 1"]))

fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
for d in DATASETS:
    q = R["datasets"][d]["pred"]
    if q is None:
        continue
    lags = {int(k): v for k, v in q["scores"]["normalized"]["lags"].items() if v is not None}
    axes[0].plot(list(lags), list(lags.values()), marker="o", ms=3, label=SHORT[d])
axes[0].set_xscale("log"); axes[0].set_xlabel("Lag (steps, log scale)"); axes[0].set_ylabel("Correlation of misses")
axes[0].set_title("Dependence along an episode (normalized score)"); axes[0].legend(fontsize=7.5)
x = np.arange(len(DATASETS))
for i, s in enumerate(SCORES):
    vals = [np.nan if R["datasets"][d]["pred"] is None else R["datasets"][d]["pred"]["scores"][s]["rho"] for d in DATASETS]
    axes[1].bar(x - 0.2 + 0.4 * i, vals, 0.4, label=f"{s} score", color=["#0d8a96", "#b8701a"][i])
axes[1].set_xticks(x, [SHORT[d] for d in DATASETS], rotation=25, ha="right", fontsize=8)
axes[1].set_ylabel("ρ (intra-episode correlation of misses)"); axes[1].set_title("ρ by dataset"); axes[1].legend()
plt.tight_layout(); plt.show()"""

EXPECT_MD = r"""## 2. Pre-registered expectations

The expectation file is shown verbatim. It was written before the runs it concerns: Stage A before the score
pools existed, Stage B (the Monte Carlo predictions) after the pools and before any benchmark run. Its
SHA-256 identifies this exact text."""

EXPECT = r"""for x in R["expectations"]:
    display(Markdown(f"`{x['file']}`, SHA-256 `{x['sha']}`"))
    display(Markdown("> " + x["text"].replace("\n", "\n> ")))
if not R["expectations"]:
    print("no expectation file yet")"""

FINDINGS_MD = r"""## 3. Findings

__FINDINGS__"""

NOSHIFT_MD = r"""## 4. Failure by bank design, no shift

Uniform BCA's failure rate for every design on every dataset (normalized score). Without shift every
weight is 1, so uniform BCA and WBCP with exact weights are the same rule; WBCP with estimated weights
differs only by the noise of fitting the weights. Cells read failure (%); the 5% target is the line between
blue and red."""

NOSHIFT = r"""cols = COLUMNS
mat = np.full((len(DATASETS), len(cols)), np.nan)
ann = [["" for _ in cols] for _ in DATASETS]
for i, d in enumerate(DATASETS):
    e = R["datasets"][d]
    for label, suffix, _ in designs(e):
        j = cols.index(label)
        a = arm(d, suffix)
        mat[i, j] = pct(a)
        k = e["k"] if "Configured" in label else None
        ann[i][j] = ("–" if a is None else fmt(a)) + (f"\nK={k}" if k else "")
fig, ax = plt.subplots(figsize=(13, 4.6))
from matplotlib.colors import TwoSlopeNorm
im = ax.imshow(np.clip(mat, 0, 40), cmap="RdBu_r", norm=TwoSlopeNorm(vmin=0, vcenter=5, vmax=40), aspect="auto")
for i in range(len(DATASETS)):
    for j in range(len(cols)):
        ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=7.5, color="white" if (not np.isnan(mat[i, j]) and mat[i, j] > 25) else "black")
ax.set_xticks(range(len(cols)), [textwrap.fill(c, 14) for c in cols], fontsize=8); ax.set_yticks(range(len(DATASETS)), [SHORT[d] for d in DATASETS])
ax.grid(False); plt.colorbar(im, ax=ax, label="Banks failing (%), clipped at 40")
ax.set_title(f"Uniform BCA failure without shift, {R['name']} pools (normalized score; – = not run)")
plt.tight_layout(); plt.show()

for s in SCORES:
    rows = []
    for d in DATASETS:
        e = R["datasets"][d]
        cells = {label: fail(arm(d, suffix, s=s)) + (f" (K={e['k']})" if "Configured" in label else "") for label, suffix, _ in designs(e)}
        rows.append([SHORT[d], e["n"]] + [cells.get(c, "n/a") for c in cols])
    display(Markdown(f"**Uniform BCA, {s} score: failure % [95% interval]**"))
    display(table(rows, ["Dataset", "Bank n"] + cols))"""

SCORE_MD = r"""## 5. Predictions against results

Each no-shift design's observed failure against its Stage B Monte Carlo prediction (dependence.py, the
same sampler on the same pool). Independent rows are predicted at 5%. The pre-registered tolerance is about
±2 points, or ±5 where the design effect exceeds 3."""

SCORE = r"""fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
board = []
for ax, s in zip(axes, SCORES):
    xs, ys = [], []
    for d in DATASETS:
        e = R["datasets"][d]
        for label, suffix, plabel in designs(e):
            a = arm(d, suffix, s=s)
            if a is None:
                continue
            p, D = prediction(d, plabel, s)
            if np.isnan(p):
                continue
            xs.append(p); ys.append(pct(a))
            ax.errorbar(p, pct(a), yerr=[[pct(a) - 100 * a["lo"]], [100 * a["hi"] - pct(a)]], fmt="o", ms=4, capsize=2,
                        color=plt.cm.tab10(DATASETS.index(d)), label=SHORT[d] if label == "Independent rows" else None)
            board.append([SHORT[d], label + (f" (K={e['k']})" if "Configured" in label else ""), s, round(p, 1),
                          None if D is None else round(D, 2), fail(a), verdict(pct(a), p, D)])
    top = max(xs + ys + [10]) * 1.08
    ax.plot([0, top], [0, top], color="#666", lw=1); ax.fill_between([0, top], [-2, top - 2], [2, top + 2], color="#999", alpha=0.15, lw=0)
    ticks = [t for t in (0, 2, 4, 6, 8, 10, 15, 20, 30, 40, 60, 100) if t <= top]
    ax.set_xscale("symlog", linthresh=10, linscale=2); ax.set_yscale("symlog", linthresh=10, linscale=2); ax.set_xlim(0, top); ax.set_ylim(0, top)
    ax.set_xticks(ticks, ticks); ax.set_yticks(ticks, ticks); ax.minorticks_off()
    ax.set_xlabel("Predicted failure (%)"); ax.set_ylabel("Observed failure (%)"); ax.set_title(f"{s} score (band: ±2 points)")
    ax.legend(fontsize=7, loc="upper left")
plt.tight_layout(); plt.show()
B = table(board, ["Dataset", "Design", "Score", "Predicted (%)", "D (Monte Carlo)", "Observed", "Verdict"])
counts = B["Verdict"].value_counts().to_dict()
display(Markdown("**Scorecard:** " + ", ".join(f"{v} {k}" for k, v in counts.items())))
B"""

BARS = r"""fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=False)
for ax, d in zip(axes.flat, DATASETS):
    e = R["datasets"][d]
    ds = [x for i, x in enumerate(designs(e)) if x[1] not in [y[1] for y in designs(e)[:i]]]  # configured K may repeat K=2/5/10
    x = np.arange(len(ds))
    arms = [arm(d, suffix) for _, suffix, _ in ds]
    ax.bar(x, [pct(a) for a in arms], 0.6, color=COL["BQ-CP"], yerr=yerr(arms), error_kw=dict(elinewidth=0.8, capsize=2))
    ax.scatter(x, [prediction(d, p)[0] for _, _, p in ds], marker="_", s=260, color=COL["pred"], zorder=3, label="Stage B prediction")
    target(ax)
    labels = [l.replace("Configured K", f"K={e['k']}").replace("Independent rows, large bank", f"iid n={e['big']}").replace("Independent rows", "iid") for l, _, _ in ds]
    ax.set_xticks(x, labels, fontsize=7, rotation=40, ha="right")
    ax.set_title(f"{SHORT[d]} (n={e['n']})", fontsize=9.5)
    top = np.nanmax([pct(a) for a in arms] + [prediction(d, p)[0] for _, _, p in ds] + [8])
    ax.set_ylim(0, min(100, top * 1.15))
axes[0, 0].set_ylabel("Banks failing (%)"); axes[1, 0].set_ylabel("Banks failing (%)"); axes[0, 0].legend(fontsize=7.5)
axes.flat[-1].axis("off")
fig.suptitle(f"Uniform BCA without shift by design ({R['name']} pools, normalized score; bars: observed with 95% interval)", y=1.0)
plt.tight_layout(); plt.show()"""

CONFIG_MD = r"""## 6. The configured bank

What __NAME__ + BCA actually reserves on each dataset: n rows, K per withheld episode, drawn with BCA's own
sampler. BCA's sampler draws distinct episodes from a finite pool, so it is slightly optimistic when the bank
covers a large share of the pool; the spaced sampler draws with replacement and is slightly pessimistic.
Together they bracket deployment. `dependence_validated` is BCA's own flag (K ≤ 10 and at least 100
episodes)."""

CONFIG = r"""rows = []
for d in DATASETS:
    e = R["datasets"][d]
    k, n = e["k"], e["n"]
    episodes = math.ceil(n / k)
    pool_eps = e["pool"]["rows"]["population_episodes"] if e["pool"] else None
    for s in SCORES:
        resv, spaced = arm(d, f"resv{k}", s=s), arm(d, f"strat{k}", s=s)
        p, D = prediction(d, f"{k} per episode, BCA reservation", s)
        ps, _ = prediction(d, f"{k} per episode, stratified", s)
        rows.append([SHORT[d], s, n, k, episodes, pool_eps, "yes" if (k <= 10 and episodes >= 100) else "no",
                     None if D is None or np.isnan(D) else round(D, 2), fail(resv), round(p, 1), fail(spaced), round(ps, 1),
                     fail(arm(d, f"resv{k}", "WBCP", s=s))])
display(table(rows, ["Dataset", "Score", "n", "K", "Episodes in bank", "Pool episodes", "dependence_validated", "D (MC, BCA sampler)",
                     "Uniform BCA, BCA sampler", "Predicted (%)", "Uniform BCA, spaced", "Predicted spaced (%)", "WBCP est. w, BCA sampler"]))"""

SHIFT_MD = r"""## 7. Distribution shift (K = 5 spaced)

Banks of K = 5 spaced rows per episode, with the test distribution tilted by exp(γ z): z is the closeness
of the logged action to the policy's (policy), the sparseness of the region (density) or the size of the
state (state). Uniform BCA ignores the shift; WBCP reweights the bank by estimated or exact density ratios.
The pre-registered expectation: uniform BCA fails badly under density and state tilts, WBCP fails more than
5% at the strongest tilts but far less than uniform BCA."""

SHIFT = r"""fig, axes = plt.subplots(2, 4, figsize=(16, 7.4))
k = len(ARMS)
w = 0.82 / k
x = np.arange(len(TILTS))
for ax, d in zip(axes.flat, DATASETS):
    for i, (a, name) in enumerate(ARMS.items()):
        arms = [arm(d, "strat5_shift", a, t, y) for t, y in TILTS]
        ax.bar(x - 0.41 + w * (i + 0.5), [pct(v) for v in arms], w, color=COL[a], label=name, yerr=yerr(arms),
               error_kw=dict(elinewidth=0.7, capsize=1.5, ecolor="#333"))
    target(ax)
    ax.set_yscale("symlog", linthresh=10); ax.set_ylim(0, 100)
    ax.set_xticks(x, [f"{t}\nγ={y:g}" for t, y in TILTS], fontsize=7)
    ax.set_title(f"{SHORT[d]} (n={R['datasets'][d]['n']})", fontsize=9.5)
axes[0, 0].set_ylabel("Banks failing (%, symlog)"); axes[1, 0].set_ylabel("Banks failing (%, symlog)")
axes.flat[-1].axis("off"); axes[0, 0].legend(fontsize=7, loc="upper left")
fig.suptitle(f"K = 5 spaced banks under shift ({R['name']} pools, normalized score)", y=1.0)
plt.tight_layout(); plt.show()

for s in SCORES:
    rows = []
    for d in DATASETS:
        for t, y in TILTS:
            b = block(d, "strat5_shift", t, y, s)
            if b is None:
                continue
            rows.append([SHORT[d], f"{t} γ={y:g}", round(b["ne"]), *[fail(b["A"].get(a)) for a in ARMS],
                         None if b["A"]["WBCP (oracle w)"].get("x") is None else round(100 * b["A"]["WBCP (oracle w)"]["x"], 2)])
    display(Markdown(f"**K = 5 spaced under shift, {s} score**"))
    display(table(rows, ["Dataset", "Tilt", "Calibration n_eff (exact w)", *ARMS.values(), "WBCP exact w: mean excess when failing (points)"]))"""

WEIGHTS_MD = r"""## 8. Estimated against exact weights

Without shift the exact weights are all 1, so any difference between WBCP with estimated weights and the
uniform rule is the cost of estimating weights that should be flat (from 1,000 + 1,000 rows, the benchmark's
default). On walker2d and pen-cloned this cost reached 15% failure at 8,192 rows (DEPENDENCE.md, Change 10),
because the score is not locally adaptive while the estimated weights vary with (s, a)."""

WEIGHTS = r"""rows = []
for d in DATASETS:
    e = R["datasets"][d]
    for suffix, n in (("iid", e["n"]), ("iid_big", e["big"])):
        if suffix == "iid_big" and e["n"] >= e["big"]:
            continue
        for s in SCORES:
            b = block(d, suffix, s=s)
            if b is None:
                rows.append([SHORT[d], n, s, "–", "–", "–", None])
                continue
            rows.append([SHORT[d], n, s, fail(b["A"]["BQ-CP"]), fail(b["A"]["WBCP"]), fail(b["A"]["WBCP (oracle w)"]), round(b["nh"])])
W = table(rows, ["Dataset", "n", "Score", "Uniform BCA", "WBCP, estimated weights", "WBCP, exact weights (= uniform)", "Calibration n_eff (estimated w)"])
fig, ax = plt.subplots(figsize=(13, 3.6))
sub = [(d, s) for d in DATASETS for s in ("normalized",)]
x = np.arange(len(DATASETS))
for i, (suffix, name, color) in enumerate((("iid", "configured n", "#0d8a96"), ("iid_big", "large bank", "#0b4f58"))):
    vals = [pct(arm(d, suffix if not (suffix == "iid_big" and R["datasets"][d]["n"] >= R["datasets"][d]["big"]) else "iid", "WBCP")) for d in DATASETS]
    ax.bar(x - 0.2 + 0.4 * i, vals, 0.4, color=color, label=f"WBCP estimated weights, {name}")
target(ax); ax.set_xticks(x, [SHORT[d] for d in DATASETS], rotation=20, ha="right", fontsize=8); ax.set_ylabel("Banks failing (%)")
ax.set_title("The cost of estimating flat weights (no shift, independent rows, normalized score)"); ax.legend(fontsize=7.5)
plt.tight_layout(); plt.show()
W"""

COST_MD = r"""## 9. What a valid bank costs: threshold width

The mean certified threshold relative to λ\* (uniform BCA, no shift). A wider threshold is the price of
certifying with fewer effective rows; designs that fail are too narrow, not too wide."""

COST = r"""rows = []
for d in DATASETS:
    e = R["datasets"][d]
    row = [SHORT[d]]
    for label in COLUMNS:
        match = [suffix for l, suffix, _ in designs(e) if l == label]
        b = block(d, match[0]) if match else None
        row.append(None if b is None or b["A"]["BQ-CP"]["th"] is None else round(100 * (b["A"]["BQ-CP"]["th"] / b["L"] - 1), 1))
    rows.append(row)
display(Markdown("**Mean threshold above λ\\* (%), uniform BCA, normalized score**"))
table(rows, ["Dataset"] + COLUMNS)"""

HOSTS_MD = r"""## 10. The same datasets under the other hosts

ρ and the configured bank's failure for every host whose pools exist so far (each host has its own
notebook). The dataset sets the episode structure; the host sets the score, through its critic and its
learned scale."""

HOSTS_CELL = r"""rows = []
for h, name in HOSTS.items():
    other = R if h == HOST else load_results(h)
    for d in DATASETS:
        e = other["datasets"][d]
        if e["pred"] is None:
            continue
        k = e["k"]
        rows.append([name, SHORT[d], round(e["pred"]["scores"]["normalized"]["rho"], 3), round(e["pred"]["scores"]["raw"]["rho"], 3),
                     e["n"], k, fail(other_arm) if (other_arm := next((b["A"]["BQ-CP"] for b in (e["runs"].get(f"{P(d)}_resv{k}") or {"blocks": []})["blocks"]
                                                                       if b["s"] == "normalized" and b["y"] == 0), None)) else "–"])
X = table(rows, ["Host", "Dataset", "ρ normalized", "ρ raw", "Bank n", "K", "Configured bank, uniform BCA"])
if len(X):
    piv = X.pivot(index="Dataset", columns="Host", values="ρ normalized").reindex([SHORT[d] for d in DATASETS])
    fig, ax = plt.subplots(figsize=(11, 3.4))
    piv.plot.bar(ax=ax, rot=20, width=0.8)
    ax.set_ylabel("ρ (normalized score)"); ax.set_title("Within-episode dependence by host"); ax.legend(fontsize=7.5)
    plt.tight_layout(); plt.show()
X"""

ALL_MD = r"""## 11. All results

Every result block of this host's matrix (and any other run on its pools): one row per run, tilt and
score, with each rule's failure and 95% interval, and WBCP's (estimated weights) mean risk, percentiles,
excess when failing, threshold and abstention."""

ALL = r"""rows = []
for d in DATASETS:
    e = R["datasets"][d]
    for name, run in list(e["runs"].items()) + list(e["extra"].items()):
        for b in run["blocks"]:
            w = b["A"].get("WBCP", {})
            rows.append([SHORT[d], name, run.get("label", "outside the matrix"), b["n"], "none" if b["y"] == 0 else b["t"], b["y"], b["s"],
                         round(b["L"], 3), round(b["ne"]), round(b["cs"]), *[fail(b["A"].get(a)) for a in ARMS],
                         None if w.get("r") is None else round(100 * w["r"], 2), None if w.get("q95") is None else round(100 * w["q95"], 2),
                         None if w.get("q99") is None else round(100 * w["q99"], 2), None if w.get("x") is None else round(100 * w["x"], 2),
                         None if w.get("th") is None else round(w["th"], 3), None if w.get("ab") is None else round(100 * w["ab"], 1)])
ALLR = table(rows, ["Dataset", "Run", "Design", "n", "Tilt", "γ", "Score", "λ*", "n_eff", "Mean bank size", *ARMS.values(),
                    "WBCP mean risk (%)", "WBCP q95 (%)", "WBCP q99 (%)", "WBCP excess | fail", "WBCP threshold", "WBCP abstain (%)"])
print(f"{len(ALLR)} result blocks")
ALLR"""

PROV_MD = r"""## 12. Provenance

Score pools (with the SHA-256 of their arrays), the run files and the commands that made them. Every run
is reproducible from `python experiments/wbcp/host_matrix.py --host __HOST__ --stage {freeze,predict,bench}`."""

PROV = r"""display(table([[SHORT[d], e["pool"]["dir"] if e["pool"] else "–", e["pool"]["algorithm"] if e["pool"] else "–",
                 e["pool"]["device"] if e["pool"] else "–", e["pool"]["npz_sha256"] if e["pool"] else "–",
                 e["pred"]["created"] if e["pred"] else "–"]
                for d, e in R["datasets"].items()], ["Dataset", "Score pool", "Host", "Trained on", "frozen.npz SHA-256", "Predictions created"]))
rows = []
for d, e in R["datasets"].items():
    for name, run in e["runs"].items():
        s = run["settings"]
        rows.append([SHORT[d], name, run["file"], s["trials"], s["seed"], s["n"], s.get("per_episode"), s.get("spacing"), s.get("blocks")])
    for name in e["missing"]:
        rows.append([SHORT[d], name, "not run yet", None, None, None, None, None, None])
table(rows, ["Dataset", "Run", "File", "Trials", "Seed", "n", "K", "Sampler", "Whole episodes"])"""


def cells(host):
    name = HOSTS[host]
    findings_path = os.path.join(HERE, "host_findings", f"{host}.md")
    findings = (open(findings_path, encoding="utf-8").read().strip() if os.path.exists(findings_path) else
                "No written interpretation yet: the sections below show every result against its pre-registered "
                "expectation. The interpretation is added to `experiments/wbcp/host_findings/" + host + ".md` "
                "once the matrix is complete.")
    parts = [("md", INTRO), ("code", SETUP), ("md", POOLS_MD), ("code", POOLS), ("md", EXPECT_MD), ("code", EXPECT),
             ("md", FINDINGS_MD.replace("__FINDINGS__", findings)), ("md", NOSHIFT_MD), ("code", NOSHIFT),
             ("md", SCORE_MD), ("code", SCORE), ("code", BARS), ("md", CONFIG_MD), ("code", CONFIG),
             ("md", SHIFT_MD), ("code", SHIFT), ("md", WEIGHTS_MD), ("code", WEIGHTS), ("md", COST_MD), ("code", COST),
             ("md", HOSTS_MD), ("code", HOSTS_CELL), ("md", ALL_MD), ("code", ALL), ("md", PROV_MD), ("code", PROV)]
    return [(kind, text.replace("__HOST__", host).replace("__NAME__", name)) for kind, text in parts]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", choices=HOSTS, required=True)
    parser.add_argument("--output", help="default experiments/wbcp/results_<host>.ipynb")
    parser.add_argument("--no-execute", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or os.path.join(HERE, f"results_{args.host}.ipynb")
    from experiments.wbcp import results_notebook
    results_notebook.CELLS = cells(args.host)
    nb_cells = notebook_cells()
    failures = [] if args.no_execute else execute(nb_cells, os.path.abspath(os.path.join(HERE, "..", "..")))
    notebook = {"cells": nb_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    with open(output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(notebook, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    print(f"wrote {output}: {len(nb_cells)} cells" + (f", {len(failures)} failing cells" if failures else ""))
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
