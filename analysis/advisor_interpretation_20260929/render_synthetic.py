"""Plot previously verified synthetic results without generating new samples."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
base=args.output
base.mkdir(parents=True,exist_ok=True)
source=Path(__file__).resolve().parents[2]/"outputs/weighted_conformal/2026-09-29-v2/results.json"
d=json.loads(source.read_text())
keys=["ordinary_source","ordinary_shifted","weighted_conformal_shifted","weighted_bca_shifted"]
labels=["Ordinary CP\nno shift","Ordinary CP\nshifted","Weighted CP\nshifted","Weighted + Bayesian\nshifted"]
colors=["#6B7C93","#C36B55","#147F92","#7057A4"]
for kind in ["coverage","halfwidth"]:
    fig,ax=plt.subplots(figsize=(12,12*1.184275/3.6068),dpi=200)
    fig.subplots_adjust(left=.09,right=.99,bottom=.22,top=.97)
    values=[d["arms"][k]["coverage"]*100 if kind=="coverage" else d["arms"][k]["mean_finite_radius"] for k in keys]
    bars=ax.bar(np.arange(4),values,width=.59,color=colors)
    if kind=="coverage":
        cis=np.array([d["arms"][k]["coverage_binomial_interval"] for k in keys])*100
        err=np.array([np.array(values)-cis[:,0],cis[:,1]-np.array(values)])
        ax.errorbar(np.arange(4),values,yerr=err,fmt="none",ecolor="#20365C",capsize=6,lw=1.6)
        ax.axhline(90,color="#20365C",ls="--",lw=1.3)
        ax.set_ylim(0,105);ax.set_yticks([0,30,60,90]);ax.set_ylabel("Response coverage (%)",fontsize=18)
    else:
        ax.set_ylim(0,6.6);ax.set_yticks([0,2,4,6]);ax.set_ylabel("Mean finite half-width",fontsize=18)
    for i,(v,bar) in enumerate(zip(values,bars)):
        offset=3.2 if kind=="coverage" else .16
        ax.text(bar.get_x()+bar.get_width()/2,v+offset,f"{v:.2f}%" if kind=="coverage" else f"{v:.3f}",ha="center",va="bottom",fontsize=20,fontweight="bold")
    ax.set_xticks(range(4),labels,fontsize=17);ax.tick_params(axis="y",labelsize=16)
    ax.grid(axis="y",color="#DDE2E8",lw=.6);ax.set_axisbelow(True)
    for side in ["top","right"]:ax.spines[side].set_visible(False)
    fig.savefig(base/f"synthetic_{kind}_slide.png",dpi=200)
    plt.close(fig)
receipt={"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"new_trials":0,"values":{k:d["arms"][k] for k in keys},"half_width_explanation":"sigma=1, so radius equals half-width. Full interval length is twice this value."}
(base/"synthetic_figure_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print("Two plots created from the accepted result JSON. No new trials.")
