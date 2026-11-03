"""Reproduce the five knee/stall figures using only the included CSV extracts.

No project modules, model evaluation, simulator, training or calibration are used.
Optional --source-repo extracts the exact pinned saved JSON reports first.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages

RESULT_SHA = "b136b39c670e7224c22fe1b8f38062a22d30f608d22c9b4253c406194b53d197"
STUDY = Path("experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/confirmation_v1")
WORLDS = ("stall3", "nostall")
NAMES = {"stall3": "Stall world", "nostall": "No-stall world"}
COLORS = {"stall3": "#0072B2", "nostall": "#D55E00"}
DOSES = (0, 21, 63, 189)
MARKERS = ("o", "s", "^")
GRAY = "#535B64"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract(repo, out):
    study = repo / STUDY
    run = study / "cf_run/attempt_3"
    result_path = run / "CF_RESULT.json"
    if sha(result_path) != RESULT_SHA:
        raise ValueError("Final result differs from the verified completed study")
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    rows, own, candidates, policies, sources = [], [], [], [], []
    for path in sorted(run.glob("eval_*/report.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if d["status"] != "completed":
            raise ValueError(f"Incomplete report: {path}")
        world, seed = d["world"], int(d["seed_index"])
        dose = 0 if d["arm"] == "native" else int(d["kappa"])
        base = dict(world=world, dose=dose, seed=seed)
        assert len(d["returns"]) == 128 and len(d["own"]["qpi"]) == 48 and len(d["data"]["qpi"]) == 192
        policies.append({**base, "arm": d["arm"], "J_reported": d["J"], "exposure": d["exposure_w"],
                         "I1_max_rel": d["instrument"]["I1_max_rel"], "I1_ok": int(d["instrument"]["I1_ok"]),
                         "I2_max_abs": d["instrument"]["I2_max_abs"], "I2_ok": int(d["instrument"]["I2_ok"]),
                         "simulator_steps": d["transitions"], "report_sha256": sha(path)})
        for i, value in enumerate(d["returns"]):
            rows.append({**base, "start_id": i, "return": value})
        for i, (episode, t) in enumerate(d["own"]["rows"]):
            own.append({**base, "probe_id": i, "start_id": episode, "time": t,
                        **{key: d["own"][key][i] for key in ("q1", "qpi", "u")}})
        for i, (episode, t) in enumerate(d["data"]["rows"]):
            for slot in range(4):
                candidates.append({**base, "bank_id": i, "episode_id": episode, "time": t,
                                   "slot": slot, "slot_name": d["data"]["slots"][slot],
                                   **{f"a{j}": d["data"]["probes"][i][slot][j] for j in range(4)},
                                   **{key: d["data"][key][i][slot] for key in ("q1", "qpi", "u")}})
        sources.append({"path": path.relative_to(repo).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size})
    assert len(policies) == 24
    for name, content in (("policies.csv", policies), ("returns.csv", rows), ("own_action_probes.csv", own), ("candidate_probes.csv", candidates)):
        write_csv(data / name, content)
    for path in (result_path, study / "cf_math.py", study / "CONFIRMATION_RULE.md", study.parent / "overestimation_v1/oe_eval.py", study.parent / "overestimation_v1/oe_math.py"):
        sources.append({"path": path.relative_to(repo).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size})
    manifest = {"study": STUDY.as_posix(), "attempt": 3, "final_result_sha256": RESULT_SHA,
                "extracted_est": datetime.now(timezone(timedelta(hours=-5))).isoformat(),
                "sources": sources, "extracts": {p.name: sha(p) for p in sorted(data.glob("*.csv"))},
                "scientific_execution": False, "formal_reading": "INCONCLUSIVE", "C1": "NOT CONFIRMED", "C2": "OVER-CORRECTION",
                "limitations": ["Three training seeds; 128 starts nested within each policy.",
                                "The 192 candidate states are reused across models; 576 bank evaluations are not independent states.",
                                "Own-action probes follow each policy's own visited states and continuation.",
                                "Candidate rankings are retrospective saved-data diagnostics, not deployed policies."]}
    (out / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load(out):
    manifest = json.loads((out / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["final_result_sha256"] == RESULT_SHA
    for name, digest in manifest["extracts"].items():
        if sha(out / "data" / name) != digest:
            raise ValueError(f"CSV identity changed: {name}")
    blocks = {name: read_csv(out / "data" / name) for name in manifest["extracts"]}
    result = {}
    for policy in blocks["policies.csv"]:
        key = policy["world"], int(policy["dose"]), int(policy["seed"])
        def selection(name):
            return [r for r in blocks[name] if (r["world"], int(r["dose"]), int(r["seed"])) == key]
        ret = selection("returns.csv")
        op = selection("own_action_probes.csv")
        cp = selection("candidate_probes.csv")
        assert len(ret) == 128 and len(op) == 48 and len(cp) == 768
        d = {"policy": policy, "returns": np.array([float(r["return"]) for r in ret]),
             "exposure": float(policy["exposure"]), "own": {}, "data": {}}
        for col in ("q1", "qpi", "u"):
            d["own"][col] = np.array([float(r[col]) for r in op])
            d["data"][col] = np.array([float(r[col]) for r in cp]).reshape(192, 4)
        d["candidate_identity"] = [(r["episode_id"], r["time"], r["slot"], r["a0"], r["a1"], r["a2"], r["a3"]) for r in cp]
        assert abs(d["returns"].mean() - float(policy["J_reported"])) < 1e-10
        assert all(np.isfinite(a).all() for a in (d["returns"], *d["own"].values(), *d["data"].values()))
        result[key] = d
    assert len(result) == 24
    for w in WORLDS:
        identity = result[w, 0, 0]["candidate_identity"]
        assert all(result[w, k, s]["candidate_identity"] == identity for k in DOSES for s in range(3))
    return result


def calculate(reports):
    metrics = []
    for (w, k, s), d in sorted(reports.items()):
        native = reports[w, 0, s]
        delta = d["returns"] - native["returns"]
        raw = d["own"]["q1"] - d["own"]["qpi"]
        adjusted = raw - k * d["own"]["u"]
        q, value, u = (d["data"][i] for i in ("q1", "qpi", "u"))
        score = q - k * u
        raw_choice, adj_choice = q.argmax(-1), score.argmax(-1)
        raw_value = np.take_along_axis(value, raw_choice[:, None], -1)[:, 0]
        adj_value = np.take_along_axis(value, adj_choice[:, None], -1)[:, 0]
        gain = adj_value - raw_value
        metrics.append({"world": w, "dose": k, "seed": s, "return_mean": float(d["returns"].mean()),
                        "return_sd_across_starts": float(d["returns"].std(ddof=1)), "paired_return_delta": float(delta.mean()),
                        "paired_delta_sd_across_starts": float(delta.std(ddof=1)), "paired_start_wins": int((delta > 0).sum()),
                        "exposure_percent": float(100 * d["exposure"]), "delta_exposure_pp": float(100 * (d["exposure"] - native["exposure"])),
                        "raw_bias": float(raw.mean()), "adjusted_bias": float(adjusted.mean()),
                        "raw_mae": float(abs(raw).mean()), "adjusted_mae": float(abs(adjusted).mean()),
                        "raw_over_count": int((raw > 0).sum()), "raw_under_count": int((raw < 0).sum()),
                        "adjusted_under_count": int((adjusted < 0).sum()), "own_probe_count": 48,
                        "raw_regret": float((value.max(-1) - raw_value).mean()),
                        "adjusted_regret": float((value.max(-1) - adj_value).mean()),
                        "helpful_switches": int((gain > 0).sum()), "harmful_switches": int((gain < 0).sum()),
                        "unchanged_choices": int((raw_choice == adj_choice).sum()), "bank_count": 192,
                        "mean_u_logged": float(u[:, 0].mean()), "mean_u_withheld": float(u[:, 1:].mean()),
                        "raw_ties": int(((q == q.max(-1, keepdims=True)).sum(-1) > 1).sum()),
                        "adjusted_ties": int(((score == score.max(-1, keepdims=True)).sum(-1) > 1).sum())})
    return metrics


def configure():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 13,
                         "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#BAC1C8", "axes.labelcolor": "#263238", "text.color": "#18242C",
                         "xtick.color": "#4E5963", "ytick.color": "#4E5963", "grid.color": "#DDE2E7",
                         "grid.linewidth": .65, "axes.axisbelow": True, "savefig.facecolor": "white",
                         "pdf.fonttype": 42, "ps.fonttype": 42})


def canvas(title, subtitle, rows=1, figsize=(11.5, 7.1)):
    fig, axes = plt.subplots(rows, 2, figsize=figsize, squeeze=False)
    fig.subplots_adjust(left=.085, right=.97, top=.755 if rows == 1 else .775, bottom=.20 if rows == 1 else .16,
                        wspace=.24, hspace=.42)
    fig.text(.065, .962, title, fontsize=18 if len(title)>54 else 20, fontweight="bold", va="top")
    fig.text(.065, .902, subtitle, fontsize=10.5, va="top", color=GRAY)
    for ax in axes.ravel():
        ax.grid(axis="y", alpha=.8)
    return fig, axes


def seed_handles():
    return [Line2D([], [], color=GRAY, marker=MARKERS[s], linestyle="none", markersize=6, label=f"Seed {s}") for s in range(3)]


def footer(fig, text):
    fig.text(.065, .075, text, va="top", fontsize=8.5, color=GRAY, linespacing=1.45)
    fig.text(.965, .023, "KNEE / STALL  |  NOV 2, 2026  |  SAVED RESULTS", ha="right", fontsize=7.4, color="#73808B")


def sample(metrics, w, k, field):
    return np.array([next(r[field] for r in metrics if r["world"] == w and r["dose"] == k and r["seed"] == s) for s in range(3)])


def panel_title(ax, w):
    ax.set_title(NAMES[w], loc="left", fontweight="bold", color=COLORS[w], y=1.02, pad=8)


def dose_axis(ax, native=False):
    ax.set_xticks(range(4 if native else 3), ["Native", "21\nprofile", "63\nprespecified", "189\nprofile"] if native else ["21\nprofile", "63\nprespecified", "189\nprofile"])
    ax.set_xlabel("Actor penalty strength, κ")
    ax.set_xlim(-.35, 3.35 if native else 2.35)
    ax.axvspan(1.7 if native else .7, 2.3 if native else 1.3, color="#EEF1F4", zorder=0)


def returns_delta(metrics):
    fig, aa = canvas("BCA's benefit depends on the world", "Paired change in discounted policy return relative to native TD3+BC; higher is better.")
    for ax, w in zip(aa[0], WORLDS):
        panel_title(ax, w); dose_axis(ax)
        ax.axhline(0, color=GRAY, lw=1.1)
        vals = np.array([sample(metrics, w, k, "paired_return_delta") for k in DOSES[1:]])
        for s in range(3):
            ax.plot(np.arange(3) + (s-1)*.075, vals[:, s], color=COLORS[w], alpha=.62, lw=1.1,
                    marker=MARKERS[s], ms=7, markeredgecolor="white", markeredgewidth=.6)
        ax.plot(range(3), vals.mean(1), color="#172733", marker="D", ms=6, lw=1.8, zorder=5)
        for x, y in enumerate(vals.mean(1)):
            offset=(-15,26) if w=='stall3' and x==0 else (0,12 if y>=0 else -18)
            ax.annotate(f"mean {y:+.1f}", (x,y), xytext=offset, textcoords="offset points", ha="center", fontsize=9)
        ax.set_ylim(-120, 145); ax.set_ylabel("BCA − native return")
    fig.legend(handles=seed_handles()+[Line2D([],[],color="#172733",marker="D",lw=1.8,label="Three-seed mean")], loc="upper center", bbox_to_anchor=(.52,.855), ncol=4, frameon=False)
    footer(fig, "3 training seeds × 128 common starts per policy. Starts are nested evaluations, not independent training replications.\nAll strengths shown; κ189 gains in all stall seeds and loses in all no-stall seeds. Formal result remains INCONCLUSIVE (two Qπ checks).")
    return fig


def returns_absolute(metrics):
    fig, aa = canvas("The return scale behind the gains and losses", "Each point is one final policy's mean across the same 128 starts; a common vertical scale preserves context.")
    for ax,w in zip(aa[0],WORLDS):
        panel_title(ax,w);dose_axis(ax,True)
        vals=np.array([sample(metrics,w,k,"return_mean") for k in DOSES])
        for s in range(3):
            ax.plot(np.arange(4)+(s-1)*.065,vals[:,s],color=COLORS[w],alpha=.65,lw=1.1,marker=MARKERS[s],ms=7,mec="white",mew=.6)
        ax.plot(range(4),vals.mean(1),color="#172733",marker="D",lw=1.8,ms=6)
        for x,y in enumerate(vals.mean(1)):
            ax.annotate(f"{y:.1f}",(x,y),xytext=(0,10),textcoords="offset points",ha="center",fontsize=9)
        ax.set_ylim(-430,0);ax.set_ylabel("Discounted return, J (higher is better)")
    fig.legend(handles=seed_handles()+[Line2D([],[],color="#172733",marker="D",lw=1.8,label="Three-seed mean")],loc="upper center",bbox_to_anchor=(.52,.855),ncol=4,frameon=False)
    footer(fig,"Horizon 100, discount 0.98. BC coefficient unchanged; native and each BCA dose share the same 10k warmup within each seed/world.\nAll final policies reach 20k cumulative critic updates. Three independent training seeds; no uncertainty interval is inferred from episode counts.")
    return fig


def exposure_tradeoff(metrics):
    fig,aa=canvas("Avoiding unsupported actions can help or hurt", "Change in withheld-region exposure versus paired policy-return change. Labels identify κ; shapes identify seeds.")
    for ax,w in zip(aa[0],WORLDS):
        panel_title(ax,w)
        ax.axhline(0,color=GRAY,lw=1);ax.axvline(0,color=GRAY,lw=1)
        for s in range(3):
            xs=np.array([sample(metrics,w,k,"delta_exposure_pp")[s] for k in DOSES])
            ys=np.array([sample(metrics,w,k,"paired_return_delta")[s] for k in DOSES])
            ax.plot(xs,ys,color=COLORS[w],lw=.9,alpha=.35)
            for j,k in enumerate(DOSES[1:],1):
                ax.scatter(xs[j],ys[j],s=50+j*12,marker=MARKERS[s],c=COLORS[w],edgecolors="white",linewidth=.6,zorder=3)
                offset=(5,4)
                if w=='nostall' and s==0: offset=(5,6 if k!=63 else -12)
                if w=='nostall' and s==1 and k==21: offset=(-24,10)
                if w=='stall3' and s==0 and k==63: offset=(7,-15)
                if w=='stall3' and s==0 and k==189: offset=(-28,6)
                if w=='stall3' and s==1 and k==21: offset=(7,-15)
                ax.annotate(str(k),(xs[j],ys[j]),xytext=offset,textcoords="offset points",fontsize=8.7,color=COLORS[w],
                            arrowprops={"arrowstyle":"-", "lw":.55, "color":COLORS[w], "alpha":.7} if offset!=(5,4) else None)
        ax.scatter([0],[0],marker="*",s=120,c="#172733",zorder=5)
        ax.set_xlim(-81,6);ax.set_ylim(-120,145)
        ax.set_xlabel("Exposure change (percentage points)\n← less use of withheld region")
        ax.set_ylabel("BCA − native return")
    fig.legend(handles=seed_handles()+[Line2D([],[],marker="*",color="#172733",linestyle="none",markersize=10,label="Native reference")],loc="upper center",bbox_to_anchor=(.52,.855),ncol=4,frameon=False)
    footer(fig,"Exposure is the fraction of 128 × 100 executed steps with |a₀| beyond the withheld knee. Each seed/dose is a separate trained policy.\nLines connect the dose profile for readability; they do not establish a causal exposure–return relationship. κ63 was prespecified.")
    return fig


def q_errors(metrics):
    fig,aa=canvas("Raw critic optimism and actor-score pessimism", "Own-action probes: Q₁ − Qπ versus Q₁ − κU − Qπ. The actor score is not the learned critic prediction.",rows=2,figsize=(11.5,9.4))
    for col,w in enumerate(WORLDS):
        for row,field in enumerate(("bias","mae")):
            ax=aa[row,col];panel_title(ax,w if row==0 else w);dose_axis(ax,True)
            raw=np.array([sample(metrics,w,k,"raw_"+field) for k in DOSES])
            adj=np.array([sample(metrics,w,k,"adjusted_"+field) for k in DOSES])
            for s in range(3):
                ax.scatter(np.arange(4)-.12+(s-1)*.035,raw[:,s],marker=MARKERS[s],s=36,facecolors="white",edgecolors=GRAY,lw=1,zorder=4)
                ax.scatter(np.arange(4)+.12+(s-1)*.035,adj[:,s],marker=MARKERS[s],s=36,c=COLORS[w],edgecolors="white",lw=.4,zorder=4)
            ax.plot(np.arange(4)-.12,raw.mean(1),color=GRAY,ls="--",lw=1.3)
            ax.plot(np.arange(4)+.12,adj.mean(1),color=COLORS[w],lw=1.6)
            if row==0:
                ax.axhline(0,color=GRAY,lw=1);ax.set_ylim(-735,250);ax.set_ylabel("Mean signed error\npositive: over / negative: under")
            else:
                ax.set_ylim(0,720);ax.set_ylabel("Mean absolute error")
    handles=[Line2D([],[],color=GRAY,ls="--",marker="o",mfc="white",label="Raw Q₁ error"),Line2D([],[],color="#172733",marker="o",label="Adjusted-score error")]+seed_handles()
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.52,.865),ncol=5,frameon=False,fontsize=9)
    footer(fig,"48 own-action probes per policy (16 at each of t = 0, 25, 50). Points: training seeds; lines: three-seed means.\nVisited states and continuation policies differ across trained policies. These are not common-state critic-accuracy comparisons or true-Q coverage guarantees.")
    return fig


def candidate_ranks(metrics):
    fig,aa=canvas("How the penalty changes candidate choices", "Retrospective four-candidate banks: logged action plus three withheld a₀ commands; lower regret is better.",rows=2,figsize=(11.5,9.4))
    for col,w in enumerate(WORLDS):
        ax=aa[0,col];panel_title(ax,w);dose_axis(ax)
        raw=np.array([sample(metrics,w,k,"raw_regret") for k in DOSES[1:]])
        adj=np.array([sample(metrics,w,k,"adjusted_regret") for k in DOSES[1:]])
        for s in range(3):
            ax.scatter(np.arange(3)-.10+(s-1)*.035,raw[:,s],marker=MARKERS[s],s=40,facecolors="white",edgecolors=GRAY,lw=1,zorder=4)
            ax.scatter(np.arange(3)+.10+(s-1)*.035,adj[:,s],marker=MARKERS[s],s=40,c=COLORS[w],edgecolors="white",lw=.4,zorder=4)
        ax.plot(np.arange(3)-.10,raw.mean(1),color=GRAY,ls="--",lw=1.3)
        ax.plot(np.arange(3)+.10,adj.mean(1),color=COLORS[w],lw=1.6)
        ax.set_ylim(0,4.2 if w=='stall3' else .46);ax.set_ylabel("Mean candidate regret (return units)")
        ax=aa[1,col];dose_axis(ax)
        for j,k in enumerate(DOSES[1:]):
            for off,field,filled in [(-.17,'helpful_switches',True),(.17,'harmful_switches',False)]:
                counts=sample(metrics,w,k,field); vals=100*counts/192
                ax.bar(j+off,vals.mean(),width=.29,facecolor=COLORS[w] if filled else 'white',edgecolor=COLORS[w],hatch=None if filled else '///',alpha=.85,zorder=2)
                for s in range(3):ax.scatter(j+off+(s-1)*.045,vals[s],marker=MARKERS[s],s=25,facecolors='#172733',edgecolors='white',lw=.45,zorder=5)
                ax.text(j+off,max(vals.max(),vals.mean())+2,f'{int(counts.sum())}/576',ha='center',fontsize=8.5)
        ax.set_ylim(0,51);ax.set_ylabel("Banks with improved / worsened choice (%)")
    handles=[Line2D([],[],color=GRAY,ls='--',marker='o',mfc='white',label='Raw Q₁ choice'),Line2D([],[],color='#172733',marker='o',label='Q₁ − κU choice'),Patch(facecolor='#728A9A',label='Helpful switch'),Patch(facecolor='white',edgecolor='#728A9A',hatch='///',label='Harmful switch')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.865),ncol=4,frameon=False,fontsize=9)
    footer(fig,"192 shared state banks × 3 final policies = 576 evaluations per dose/world, not independent states. Seeds: circle 0, square 1, triangle 2.\nRegret uses each policy's own Qπ continuation. This is not deployed actor behavior or proof of the return mechanism. Top-panel scales differ.")
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-repo',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.source_repo:extract(args.source_repo.resolve(),out)
    reports=load(out);metrics=calculate(reports)
    write_csv(out/'policy_metrics.csv',metrics)
    (out/'policy_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    configure()
    makers=[('01_paired_return_change',returns_delta),('02_policy_return_context',returns_absolute),
            ('03_exposure_return_tradeoff',exposure_tradeoff),('04_raw_and_adjusted_q_errors',q_errors),('05_candidate_ranking',candidate_ranks)]
    outputs=[]
    with PdfPages(out/'Knee_Study_Figures.pdf',metadata={'Title':'Knee/stall BCA study - all strengths and seeds','Author':'BCA research analysis','Subject':'Saved-data descriptive results; original formal verdict preserved'}) as pdf:
        for name,maker in makers:
            fig=maker(metrics)
            fig.savefig(out/(name+'.png'),dpi=300)
            fig.savefig(out/(name+'.pdf'))
            pdf.savefig(fig)
            outputs.extend([name+'.png',name+'.pdf']);plt.close(fig)
    validation={'report_count':len(reports),'return_count':sum(len(r['returns']) for r in reports.values()),
                'own_probe_count':sum(len(r['own']['qpi']) for r in reports.values()),
                'candidate_values':sum(r['data']['qpi'].size for r in reports.values()),
                'same_candidate_banks_across_models_verified':True,'all_means_match_saved_J':True,
                'no_raw_or_adjusted_candidate_ties':all(m['raw_ties']==m['adjusted_ties']==0 for m in metrics),
                'formal_I1_failures':sum(1-int(r['policy']['I1_ok']) for r in reports.values()),
                'formal_I2_failures':sum(1-int(r['policy']['I2_ok']) for r in reports.values()),
                'scientific_execution':False,'matplotlib':matplotlib.__version__,'numpy':np.__version__,
                'outputs':{name:sha(out/name) for name in outputs+['Knee_Study_Figures.pdf','policy_metrics.csv','policy_metrics.json']}}
    (out/'VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'figures_generated_from_saved_arrays','output':str(out),'figures':len(makers),'reports':len(reports),'scientific_execution':False}))


if __name__=='__main__':main()
