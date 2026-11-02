"""Render authenticated, training-only gradient diagnostics. No simulation or model changes.

Usage: .venv/Scripts/python.exe -B analysis/central_quartile_review/gradient_figures.py
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

BASE = Path(__file__).resolve().parents[2] / "experiments/thrusterchain_bca/zeus_20261026_validation"
RERUN = BASE / "central_quartile_training_v1_evaluation_rerun1"
OUT = Path(__file__).resolve().parent / "figures_gradient"
COLORS = {"native": "#334155", "bca": "#087e8b", "distance": "#c36b3b"}
COND = {"full": "Full data", "hole": "Conditional hole", "random": "Random removal"}
ARM = {"native": "Native", "bca": "BCA", "distance": "Distance"}
SHA = lambda b: hashlib.sha256(b).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read_authenticated():
    terminal = json.loads((RERUN / "RUN_RESULT.json").read_bytes())
    require(terminal["status"] == "completed_evaluation_rerun" and terminal["error"] is None
            and terminal["prepared_identity_unchanged"] is True and terminal["all_declared_jobs_completed"] is True,
            "No successful, authenticated study terminal")
    raw1 = (RERUN / "RECOVERY.json").read_bytes()
    raw2 = (RERUN / "RERUN_INPUTS.json").read_bytes()
    require(SHA(raw1) == terminal["recovery_sha256"] and SHA(raw2) == terminal["rerun_inputs_sha256"],
            "Input pin manifest changed")
    prior, continuation = json.loads(raw1), json.loads(raw2)
    rows = []
    for condition in COND:
        for arm in ARM:
            for seed in range(3):
                cell = f"seed{seed}_{condition}"
                job = f"{cell}/{arm}"
                use_continuation = job in continuation["continuation_jobs"]
                root = BASE / (continuation["continuation_package"] if use_continuation else prior["prior_package"])
                pins = continuation["continuation_files_sha256"] if use_continuation else prior["prior_files_sha256"]
                rel = f"run/{job}/gradient_audit_020000.json"
                raw = (root / rel).read_bytes()
                require(SHA(raw) == pins[rel], f"Hash mismatch: {rel}")
                v = json.loads(raw)
                require((v["seed_index"], v["condition"], v["arm"], v["absolute_critic_step"]) ==
                        (seed, condition, arm, 20000), f"Wrong diagnostic identity: {rel}")
                require(v["status"] == "completed_fixed_logged_gradient_audit" and v["probe_rows"] == 768
                        and v["protocol_sha256"] == terminal["protocol_sha256"], f"Invalid diagnostic: {rel}")
                direction = v["inward_action_direction"]
                total = direction["high_load_actor_outside_band_rows"]
                require(total > 0, f"Empty directional panel: {rel}")
                for term in ("q", "bc", "actual_penalty"):
                    counts = direction[term]
                    require(sum(counts[k] for k in ("inward_rows", "outward_rows", "zero_rows")) == total,
                            f"Directional count mismatch: {rel}, {term}")
                row = {"condition": condition, "arm": arm, "seed": seed, "checkpoint": 20000,
                       "source": str((root / rel).relative_to(BASE)), "source_sha256": SHA(raw),
                       "probe_rows": v["probe_rows"], "outside_high_load_rows": total,
                       "q_norm": v["mean_parameter_gradient"]["l2_norm"]["q"],
                       "bc_norm": v["mean_parameter_gradient"]["l2_norm"]["bc"],
                       "penalty_norm": v["mean_parameter_gradient"]["l2_norm"]["actual_penalty"],
                       "cos_q_bc": v["mean_parameter_gradient"]["cosine"]["q/bc"]}
                for term in ("q", "bc", "actual_penalty"):
                    row[term+"_inward"] = direction[term]["inward_rows"]
                    row[term+"_outward"] = direction[term]["outward_rows"]
                    row[term+"_zero"] = direction[term]["zero_rows"]
                    row[term+"_inward_fraction"] = direction[term]["inward_rows"] / total
                rows.append(row)
    require(len(rows) == 27, "Wrong figure inventory")
    return rows, terminal


def setup():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#b3bdc8", "text.color": "#172334",
                         "axes.labelcolor": "#36465b", "xtick.color": "#36465b", "ytick.color": "#36465b",
                         "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white"})


def decorate(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#e6ebef", linewidth=.75)
    ax.set_xlim(-.5, 2.5)
    ax.set_xticks([0,1,2], ["0", "1", "2"])
    ax.tick_params(length=0, pad=5)


def points(ax, rows, condition, key, include_native=True):
    offsets = {"native": -.17, "bca": 0, "distance": .17}
    for arm in ARM:
        if arm == "native" and not include_native:
            continue
        vals = [v for v in rows if v["condition"] == condition and v["arm"] == arm]
        for v in vals:
            ax.scatter(v["seed"]+offsets[arm], v[key], s=50, color=COLORS[arm],
                       edgecolors="white", linewidths=.65, zorder=3)


def header(fig, title, subtitle):
    fig.text(.065, .97, title, fontsize=21, fontweight="bold", ha="left", va="top")
    fig.text(.065, .925, subtitle, fontsize=10.5, color="#526175", ha="left", va="top")


def legend(fig, y):
    handles = [Line2D([],[],marker="o",linestyle="",markerfacecolor=COLORS[a], markeredgecolor="white",
                      markersize=8,label=ARM[a]) for a in ARM]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,y), ncol=3, frameon=False,
               fontsize=10, handletextpad=.4, columnspacing=2)


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT/f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


def plot_norms(rows):
    fig, axes = plt.subplots(4,3,figsize=(13.6,12.2),sharex=True)
    header(fig,"How do actor-loss gradients compare?",
           "Final 20k checkpoint · each point is one training seed · fixed retained logged probes")
    legend(fig,.89)
    terms = [("q_norm", "Q term · L2 norm", (-.02,.65)), ("bc_norm", "BC term · L2 norm", (-.02,.65)),
             ("penalty_norm", "Applied penalty · L2 norm", (-.005,.13)), ("cos_q_bc", "Q vs BC · cosine", (-1.05,1.05))]
    for ri,(key,label,limits) in enumerate(terms):
        for ci,(condition,title) in enumerate(COND.items()):
            ax=axes[ri,ci];decorate(ax);points(ax,rows,condition,key)
            ax.set_ylim(*limits)
            if key == "cos_q_bc": ax.axhline(0,color="#aab6c2",linewidth=.8)
            if ri == 0: ax.set_title(title, pad=12,fontweight="bold")
            if ci == 0: ax.set_ylabel(label)
            if ri == 3: ax.set_xlabel("Training seed")
    fig.text(.065,.045,"Norms and cosine are computed from each mean actor-parameter loss gradient across three fixed batches.\n"
             "These are local gradients at one checkpoint, not executed updates or causal explanations of policy returns. "
             "Native has no applied penalty. Different panels use different y scales.",
             fontsize=9, color="#526175", ha="left",va="bottom",linespacing=1.4)
    fig.subplots_adjust(left=.10,right=.97,top=.815,bottom=.12,hspace=.38,wspace=.23)
    save(fig,"01_parameter_gradient_components")


def plot_direction(rows):
    fig,axes=plt.subplots(3,3,figsize=(13.6,10.6),sharex=True,sharey=True)
    header(fig,"Which local action gradients point into the hole band?",
           "Final 20k checkpoint · high-load logged probes whose actor action is outside |a₀| ≤ 0.25")
    legend(fig,.885)
    for ri,(key,title) in enumerate([("q_inward_fraction","Q term"),("bc_inward_fraction","BC term"),
                                      ("actual_penalty_inward_fraction","Applied penalty")]):
        for ci,(condition,ctitle) in enumerate(COND.items()):
            ax=axes[ri,ci];decorate(ax)
            points(ax,rows,condition,key,include_native=(ri != 2))
            ax.set_ylim(-.03,1.03);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.set_yticks([0,.25,.5,.75,1])
            if ri==0: ax.set_title(ctitle,pad=12,fontweight="bold")
            if ci==0: ax.set_ylabel(title+"\nInward fraction")
            if ri==2: ax.set_xlabel("Training seed")
    fig.text(.065,.045,"Each dot uses the high-load, outside-band subset of 768 retained logged probes (200–232 rows here).\n"
             "Inward means the negative first-action loss gradient points toward the band; it is not an executed actor movement.\n"
             "Native penalty is zero and omitted in the bottom row. The mask is a construction label, not proof of local or joint support.",
             fontsize=9,color="#526175",ha="left",va="bottom",linespacing=1.4)
    fig.subplots_adjust(left=.10,right=.97,top=.80,bottom=.14,hspace=.37,wspace=.22)
    save(fig,"02_action_gradient_direction")


def main():
    rows, terminal = read_authenticated()
    OUT.mkdir(exist_ok=True,parents=True)
    setup();plot_norms(rows);plot_direction(rows)
    with (OUT/"gradient_20k_values.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    manifest={"source": "27 SHA-verified training-only gradient diagnostics; 3 conditions × 3 arms × 3 training seeds",
              "checkpoint": 20000, "probe_rows_per_snapshot": 768,
              "source_manifest_sha256": {"RECOVERY.json":terminal["recovery_sha256"],"RERUN_INPUTS.json":terminal["rerun_inputs_sha256"]},
              "signed_direction": "Negative first-action loss gradient points toward band, conditional high-load and actor action outside band; not actual actor movement",
              "parameter_gradient": "Norms and cosine of mean actor parameter gradients across three fixed batches; local diagnostic, not causal attribution",
              "figures": ["01_parameter_gradient_components", "02_action_gradient_direction"],
              "values": "gradient_20k_values.csv", "input_files": [{"path":r["source"],"sha256":r["source_sha256"]} for r in rows]}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"Rendered 2 figures in PNG/PDF/SVG from {len(rows)} authenticated snapshots: {OUT}")

if __name__ == "__main__":
    main()
