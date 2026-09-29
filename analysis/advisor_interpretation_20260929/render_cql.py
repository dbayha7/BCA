"""Render the unchanged exported CQL series with readable tick density."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
base=args.output
base.mkdir(parents=True,exist_ok=True)
source=Path(__file__).resolve().parents[2]/"outputs/advisor_metrics/2026-09-29/METRIC_EVIDENCE.json"
data=json.loads(source.read_text())["metrics"]
host=data["cql-walker2d-host-s202609171"]
bca=data["cql-walker2d-bca-noiw-s202609171"]
receipt={"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"series_changed":False,"figures":[]}
for name,key in [("q","average_qf1"),("loss","qf1_loss"),("gap","cql_min_qf1_loss"),("dose","critic_dose_mean")]:
    fig,ax=plt.subplots(figsize=(6,6*0.57785/1.8161),dpi=200)
    fig.subplots_adjust(left=.12,right=.94,bottom=.31,top=.96)
    colors={"Host":"#20365C","BCA":"#087F8C"}
    for label,values in [("Host",host),("BCA",bca)]:
        if key=="critic_dose_mean" and label=="Host": continue
        ys=np.asarray(values[key],float);xs=np.asarray(values["steps"],float)/1000
        assert ys.shape==xs.shape and np.isfinite(ys).all()
        ax.plot(xs,ys,lw=1,color=colors[label],label=label)
    ax.set_xlim(0,1000);ax.set_xticks([0,250,500,750,1000])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    if name=="dose": ax.set_ylim(.97,1.50);ax.set_yticks([1,1.25,1.5])
    ax.set_xlabel("Host updates (thousands)",fontsize=12,labelpad=4)
    ax.tick_params(axis="both",labelsize=12,length=0,pad=4)
    ax.grid(axis="y",color="#DDE2E8",lw=.6)
    for side in ["top","right"]:ax.spines[side].set_visible(False)
    for side in ["left","bottom"]:ax.spines[side].set_color("#BAC4CF")
    if name=="q":
        ax.legend(loc="lower right",frameon=False,fontsize=11,ncol=2,
                  handlelength=1.2,columnspacing=.8,borderpad=0)
    fig.savefig(base/f"cql_{name}_readable.png",dpi=200)
    receipt["figures"].append({"file":f"cql_{name}_readable.png","key":key,"samples":len(bca["steps"])})
    plt.close(fig)
(base/"cql_figure_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
