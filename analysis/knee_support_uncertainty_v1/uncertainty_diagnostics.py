"""Pure saved-array diagnostics of support novelty, value error and action harm.

No model, simulator, learner or calibration module is imported or executed.
All analyses are retrospective; event directions and controls are declared below.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

KNEE = float(np.float32(0.8))
RESULT_SHA = "b136b39c670e7224c22fe1b8f38062a22d30f608d22c9b4253c406194b53d197"
WORLDS = ("stall3", "nostall")
DOSES = (0, 21, 63, 189)
DEFINITIONS = {
    "status": "retrospective saved-data diagnostics, not a new confirmation test",
    "support": "strict abs(executed float32 a0)>float32(0.8), compared in float64; only this constructed marginal action label",
    "distance": "max(abs(a0)-0.800000011920929,0), not full joint state-action support distance",
    "signed_error": "raw Q1 minus the same model's own-policy Qpi; positive means overestimation",
    "overestimation_event": "signed_error>0 exactly; no fitted or practical-error tolerance",
    "lower_than_logged_event": "withheld candidate Qpi < logged-action Qpi in the same bank and continuation policy",
    "harmful_optimism_event": "withheld candidate both lower-valued than logged and raw signed_error>0; this does not require the actor or raw-Q argmax to select it",
    "auc_orientation": "higher score always predicts event=true; no direction flip; exact ties receive 0.5; one-class outcomes are null",
    "conditional_auc": "unweighted mean of within-bank positive-negative pair concordance, over banks containing both labels; not pooled demeaning",
    "local_correlation": "Pearson after subtracting each bank mean, plus mean/median per-bank Spearman with average tied ranks; constants are undefined",
    "controls": {
        "raw": "argmax raw Q1",
        "bca": "argmax(Q1-kappa U), using the final policy's actual kappa; native=0",
        "distance_range_matched": "argmax(Q1-kappa*range_bank(U)*(d-min_bank(d))/range_bank(d)); zero range gives zero penalty; matches BCA's action-specific penalty range without Qpi or return labels",
        "cutoff": "argmax raw Q1 among candidates outside the withheld set; this bank has only the logged action supported"
    },
    "control_limit": "post hoc frozen-bank diagnostic, no trained arms; distance matching borrows its amplitude from learned U and adapts per bank, so it isolates penalty shape rather than providing an independent non-BCA or fixed-coefficient control",
    "ties": "argmax selects first stored slot; report exact ties",
    "independence": "192 shared banks and 128 common starts reused across 3 models per world; training seed is replication unit",
    "comparability": "inputs match within each world, not necessarily across worlds; own-policy continuations differ across trained models",
    "missing": ["full rollout state trajectories", "true-Q derivatives", "conditional calibration coverage on withheld actions", "new model/inference values"],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ranks(x):
    x=np.asarray(x,np.float64).reshape(-1)
    if not np.isfinite(x).all(): raise ValueError("Non-finite rank input")
    order=np.argsort(x,kind="mergesort");out=np.empty(len(x),np.float64);start=0
    while start<len(x):
        end=start+1
        while end<len(x) and x[order[end]]==x[order[start]]:end+=1
        out[order[start:end]]=(start+end-1)/2+1;start=end
    return out


def pearson(x,y):
    x=np.asarray(x,np.float64).reshape(-1);y=np.asarray(y,np.float64).reshape(-1)
    if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError("Invalid correlation input")
    x=x-x.mean();y=y-y.mean();den=float(np.linalg.norm(x)*np.linalg.norm(y))
    return None if den==0 else float(np.clip(x@y/den,-1,1))


def auc(score,event):
    score=np.asarray(score,np.float64).reshape(-1);event=np.asarray(event,bool).reshape(-1)
    if score.shape!=event.shape or not np.isfinite(score).all():raise ValueError("Invalid AUC input")
    n1=int(event.sum());n0=len(event)-n1
    if not n1 or not n0:return None
    return float((ranks(score)[event].sum()-n1*(n1+1)/2)/(n1*n0))


def conditional_auc(score,event):
    score=np.asarray(score,np.float64);event=np.asarray(event,bool)
    if score.ndim!=2 or score.shape!=event.shape:raise ValueError("AUC banks must align")
    vals=[auc(s,e) for s,e in zip(score,event)];valid=[v for v in vals if v is not None]
    return {"mean":None if not valid else float(np.mean(valid)),"median":None if not valid else float(np.median(valid)),
            "eligible_banks":len(valid),"total_banks":len(vals),"one_class_banks":len(vals)-len(valid)}


def local_correlations(x,y):
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64)
    if x.ndim!=2 or x.shape!=y.shape:raise ValueError("Correlation banks must align")
    vals=[pearson(ranks(a),ranks(b)) for a,b in zip(x,y)];valid=[v for v in vals if v is not None]
    return {"pooled_spearman":pearson(ranks(x),ranks(y)),
            "bank_centered_pearson":pearson(x-x.mean(1,keepdims=True),y-y.mean(1,keepdims=True)),
            "mean_bank_spearman":None if not valid else float(np.mean(valid)),
            "median_bank_spearman":None if not valid else float(np.median(valid)),
            "eligible_banks":len(valid),"constant_banks":len(vals)-len(valid)}


def validate_bank_pairing(records):
    baseline={}
    for (world,dose,seed),d in records.items():
        ident=d["identity"]
        if world not in baseline:baseline[world]=ident
        if ident!=baseline[world]:raise ValueError(f"Candidate-bank identity mismatch within {world}")


def load_saved(repo):
    prior=repo/"analysis/knee_confirmation_figures_v1"
    manifest=json.loads((prior/"SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["final_result_sha256"]!=RESULT_SHA:raise ValueError("Unexpected completed study")
    for item in manifest["sources"]:
        p=repo/item["path"]
        if sha(p)!=item["sha256"]:raise ValueError(f"Changed source: {item['path']}")
    tables={}
    for name,h in manifest["extracts"].items():
        p=prior/"data"/name
        if sha(p)!=h:raise ValueError(f"Changed source extract: {name}")
        with p.open(encoding="utf-8",newline="") as f:tables[name]=list(csv.DictReader(f))
    records={}
    for row in tables["policies.csv"]:
        key=(row["world"],int(row["dose"]),int(row["seed"]))
        def pick(name):return [r for r in tables[name] if (r["world"],int(r["dose"]),int(r["seed"]))==key]
        own=pick("own_action_probes.csv");data=pick("candidate_probes.csv");ret=pick("returns.csv")
        if (len(own),len(data),len(ret))!=(48,768,128):raise ValueError("Incomplete saved arrays")
        identity=[tuple(r[k] for k in ("bank_id","episode_id","time","slot","a0","a1","a2","a3")) for r in data]
        d={"policy":row,"identity":identity,"own":{},"candidate":{},"returns":np.array([float(r["return"]) for r in ret])}
        for k in ("q1","qpi","u"):
            d["own"][k]=np.array([float(r[k]) for r in own]);d["candidate"][k]=np.array([float(r[k]) for r in data]).reshape(192,4)
        d["a0"]=np.array([float(r["a0"]) for r in data]).reshape(192,4)
        d["rows"]=[(int(r["episode_id"]),int(r["time"])) for r in data[::4]]
        records[key]=d
    if len(records)!=24:raise ValueError("Expected 24 saved models")
    validate_bank_pairing(records)
    return records,manifest


def sign_summary(error):
    e=np.asarray(error,np.float64)
    return {"n":int(e.size),"signed_mean":float(e.mean()),"mae":float(abs(e).mean()),
            "median":float(np.median(e)),"over":int((e>0).sum()),"under":int((e<0).sum()),"exact_zero":int((e==0).sum())}


def choice_controls(q,u,distance,support,value,kappa):
    q,u,distance,value=[np.asarray(x,np.float64) for x in (q,u,distance,value)];support=np.asarray(support,bool)
    if q.ndim!=2 or any(x.shape!=q.shape for x in (u,distance,support,value)):raise ValueError("Control banks must align")
    if np.any(support.all(1)):raise ValueError("Cutoff has no supported candidate")
    du=np.ptp(u,axis=1,keepdims=True);dd=np.ptp(distance,axis=1,keepdims=True)
    normalized=np.divide(distance-distance.min(1,keepdims=True),dd,out=np.zeros_like(distance),where=dd!=0)
    scores={"raw":q,"bca":q-kappa*u,"distance_range_matched":q-kappa*du*normalized,
            "cutoff":np.where(~support,q,-np.inf)}
    choices={k:v.argmax(1) for k,v in scores.items()};idx=np.arange(q.shape[0]);raw_value=value[idx,choices["raw"]]
    output={}
    for name,choice in choices.items():
        chosen=value[idx,choice];gain=chosen-raw_value;adv=chosen-value[:,0];isw=support[idx,choice]
        output[name]={"regret_mean":float((value.max(1)-chosen).mean()),"value_gain_vs_raw":float(gain.mean()),
                      "helpful_switches":int((gain>0).sum()),"harmful_switches":int((gain<0).sum()),
                      "unchanged_choices":int((choice==choices["raw"]).sum()),"selected_withheld":int(isw.sum()),
                      "changed_equal_value_choices":int(((choice!=choices["raw"])&(gain==0)).sum()),
                      "selected_withheld_useful_vs_logged":int((isw&(adv>0)).sum()),
                      "selected_withheld_harmful_vs_logged":int((isw&(adv<0)).sum()),
                      "selected_withheld_equal_logged":int((isw&(adv==0)).sum()),
                      "exact_score_tie_banks":int(((scores[name]==scores[name].max(1,keepdims=True)).sum(1)>1).sum()),
                      "bank_count":int(len(choice))}
    return output,choices


def analyze(records):
    results=[];bank_rows=[]
    for (world,dose,seed),d in sorted(records.items()):
        q,v,u=(d["candidate"][k] for k in ("q1","qpi","u"));error=q-v
        a0=d["a0"];support=abs(a0)>KNEE;distance=np.maximum(abs(a0)-KNEE,0)
        if support[:,0].any() or not support[:,1:].all():raise ValueError("Expected logged+three withheld actions")
        advantage=v-v[:,[0]];harm=advantage[:,1:]<0;optimism=error[:,1:]>0
        events={"support":(support,u,distance),"positive_raw_error":(error>0,u,distance),
                "withheld_lower_than_logged":(harm,u[:,1:],distance[:,1:]),
                "withheld_harmful_optimism":(harm&optimism,u[:,1:],distance[:,1:])}
        discrimination={}
        for name,(event,width,dist) in events.items():
            discrimination[name]={"positive":int(event.sum()),"negative":int(event.size-event.sum()),
                                  "u_pooled_auc":auc(width,event),"distance_pooled_auc":auc(dist,event),
                                  "u_conditional_auc":conditional_auc(width,event),"distance_conditional_auc":conditional_auc(dist,event)}
        corrs={"u_distance":local_correlations(u,distance),"u_signed_error":local_correlations(u,error),
               "u_absolute_error":local_correlations(u,abs(error)),"distance_signed_error":local_correlations(distance,error),
               "distance_absolute_error":local_correlations(distance,abs(error)),
               "u_withheld_advantage":local_correlations(u[:,1:],advantage[:,1:]),
               "u_minus_logged_error_minus_logged":local_correlations(u[:,1:]-u[:,[0]],error[:,1:]-error[:,[0]])}
        controls,choices=choice_controls(q,u,distance,support,v,dose)
        native=records[world,0,seed];ret=d["returns"]-native["returns"]
        o=d["own"];oe=o["q1"]-o["qpi"]
        result={"world":world,"dose":dose,"seed":seed,"return_mean":float(d["returns"].mean()),
                "paired_return_delta":float(ret.mean()),"paired_start_wins":int((ret>0).sum()),
                "exposure":float(d["policy"]["exposure"]),"errors":{"own_raw":sign_summary(oe),
                "own_adjusted":sign_summary(oe-dose*o["u"]),"logged_raw":sign_summary(error[:,0]),
                "withheld_raw":sign_summary(error[:,1:]),"all_candidates_adjusted":sign_summary(error-dose*u)},
                "u_logged_mean":float(u[:,0].mean()),"u_withheld_mean":float(u[:,1:].mean()),
                "mean_u_withheld_minus_logged":float((u[:,1:]-u[:,[0]]).mean()),
                "fraction_u_withheld_above_logged":float((u[:,1:]>u[:,[0]]).mean()),
                "discrimination":discrimination,"correlations":corrs,"controls":controls,
                "best_candidate_withheld":int((v.argmax(1)!=0).sum()),
                "withheld_useful_count":int((advantage[:,1:]>0).sum()),"withheld_harmful_count":int(harm.sum()),
                "own_u_signed_error_spearman":pearson(ranks(o['u']),ranks(oe)),
                "own_u_absolute_error_spearman":pearson(ranks(o['u']),ranks(abs(oe)))}
        results.append(result)
        for i,(episode,t) in enumerate(d['rows']):
            row={"world":world,"dose":dose,"seed":seed,"bank":i,"episode":episode,"time":t,
                 "u_logged":u[i,0],"u_withheld_mean":u[i,1:].mean(),"u_action_range":np.ptp(u[i]),
                 "true_action_range":np.ptp(v[i]),"signed_error_logged":error[i,0],
                 "signed_error_withheld_mean":error[i,1:].mean(),"harmful_withheld_count":harm[i].sum()}
            for name,c in choices.items():
                row[name+'_choice_slot']=int(c[i]);row[name+'_chosen_qpi']=v[i,c[i]]
            bank_rows.append(row)
    return results,bank_rows


def flatten(d,prefix=''):
    out={}
    for k,v in d.items():
        key=prefix+k
        if isinstance(v,dict):out.update(flatten(v,key+'.'))
        else:out[key]=v
    return out


def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def plot(results,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.edgecolor':'#B5BEC7','axes.axisbelow':True,'grid.color':'#E1E5E9','pdf.fonttype':42})
    colors={'stall3':'#0072B2','nostall':'#D55E00'};names={'stall3':'Stall world','nostall':'No-stall world'};markers=['o','s','^']
    def get(w,k):return sorted([r for r in results if r['world']==w and r['dose']==k],key=lambda x:x['seed'])
    def canvas(title,sub,rows=2):
        f,a=plt.subplots(rows,2,figsize=(11.5,8.8 if rows==2 else 11),squeeze=False)
        f.subplots_adjust(left=.09,right=.97,top=.79,bottom=.16,wspace=.27,hspace=.45)
        f.text(.06,.967,title,fontsize=18,fontweight='bold',va='top');f.text(.06,.915,sub,fontsize=10,color='#475560',va='top')
        for ax in a.ravel():ax.grid(axis='y',lw=.6)
        return f,a
    def head(ax,w):ax.set_title(names[w],loc='left',color=colors[w],fontweight='bold',y=1.02,pad=8)
    def foot(f,text):f.text(.06,.065,text,fontsize=8.2,color='#475560',va='top',linespacing=1.5)
    seedlegend=[Line2D([],[],color='#40515D',marker=m,ls='',label=f'Seed {i}') for i,m in enumerate(markers)]
    figures=[]
    f,a=canvas('Native baseline: optimism and unsupported choices','Raw critic errors, true candidate values and executed exposure are distinct measurements.')
    for col,w in enumerate(WORLDS):
        rs=get(w,0);ax=a[0,col];head(ax,w)
        for s,r in enumerate(rs):
            y=[r['errors'][k]['signed_mean'] for k in ['own_raw','logged_raw','withheld_raw']]
            ax.plot(np.arange(3)+(s-1)*.04,y,marker=markers[s],color=colors[w],lw=.9,alpha=.7)
        ax.axhline(0,color='#53616D',lw=.8);ax.set_xticks(range(3),['Own visited\nactions','Logged\nactions','Withheld\ncandidates']);ax.set_ylabel('Mean raw Q1 − own-policy Qπ')
        ax=a[1,col]
        fields=['selected_withheld_useful_vs_logged','selected_withheld_harmful_vs_logged','best_candidate_withheld']
        vals=np.array([[r['controls']['raw'][k] if k!='best_candidate_withheld' else r[k] for k in fields] for r in rs])/192*100
        ax.bar(range(3),vals.mean(0),color=colors[w],alpha=.23,width=.62)
        for s in range(3):ax.scatter(np.arange(3)+(s-1)*.09,vals[s],marker=markers[s],color=colors[w],s=42,zorder=3)
        ax.set_ylim(0,60);ax.set_xticks(range(3),['Q-selected W:\nbetter than logged','Q-selected W:\nworse than logged','Bank-best\naction is W']);ax.set_ylabel('Share of 192 candidate banks (%)')
    f.legend(handles=seedlegend,loc='upper center',bbox_to_anchor=(.5,.864),ncol=3,frameon=False)
    foot(f,'Own probes: 48 per model on its visited states. Candidate probes: 192 shared states × 4 actions per model. Top-panel scales differ.\nW is the constructed a0 cutoff gap, not full joint support. Lower-valued than logged is not the same event as positive Q error.')
    figures.append(('01_native_baseline',f))
    f,a=canvas('Does U identify support gaps or harmful value errors?','Fixed positive orientation: higher U or distance predicts the named event. No fitted threshold or AUROC direction flip.',rows=3)
    events=[('support','Known cutoff gap: W label'),('positive_raw_error','Raw overestimation: Q1 − Qπ > 0'),('withheld_lower_than_logged','Among W: lower value than logged')]
    for col,w in enumerate(WORLDS):
        for row,(event,label) in enumerate(events):
            ax=a[row,col]
            ax.set_title((names[w]+'\n' if row==0 else '')+label,loc='left',color=colors[w],fontweight='bold',fontsize=10,pad=8)
            for typ,dx,c,style in [('u',-.10,colors[w],'-'),('distance',.10,'#6A7680','--')]:
                arr=np.array([[np.nan if r['discrimination'][event][typ+'_conditional_auc']['mean'] is None else r['discrimination'][event][typ+'_conditional_auc']['mean'] for r in get(w,k)] for k in DOSES])
                for s in range(3):ax.scatter(np.arange(4)+dx+(s-1)*.025,arr[:,s],marker=markers[s],s=27,c=c,alpha=.8)
                mean=np.array([np.nan if np.isnan(x).all() else np.nanmean(x) for x in arr]);ax.plot(np.arange(4)+dx,mean,color=c,ls=style,lw=1.5)
            ticklabels=[]
            for k,short in zip(DOSES,['Native','21','63*','189']):
                counts=[r['discrimination'][event]['u_conditional_auc']['eligible_banks'] for r in get(w,k)]
                suffix=' / '.join(str(n) if n else 'NA' for n in counts)
                ticklabels.append(short+'\nn='+suffix)
            ax.axhline(.5,color='#9EA7AE',lw=.7);ax.set_ylim(0,1.045);ax.set_xticks(range(4),ticklabels,fontsize=7.8);ax.set_ylabel('Within-bank AUROC')
            ax.set_xlabel('Final policy / kappa',fontsize=8)
    f.legend(handles=[Line2D([],[],color='#172B3A',lw=1.5,label='U'),Line2D([],[],color='#6A7680',ls='--',label='Cutoff distance')]+seedlegend,loc='upper center',bbox_to_anchor=(.5,.867),ncol=5,frameon=False,fontsize=9)
    foot(f,'n lists eligible banks for seeds 0 / 1 / 2 out of 192. NA means no mixed-label banks: AUROC is undefined and no point is plotted.\nOne-class banks excluded; no-stall lower-value results often use only 0-6 banks. Counts matter more than a high sparse-bank AUROC.\nSupport and optimism: 4 actions; lower-than-logged: 3 withheld actions. *63 was prespecified; these diagnostics are retrospective.')
    figures.append(('02_U_discrimination',f))
    f,a=canvas('What do simple bank-level controls change?','Frozen candidate choices only: no additional policies trained and no claim of deployed-control performance.')
    controls=[('raw','Raw Q1','#59656F','--'),('bca','Q1 - kappa U',None,'-'),('distance_range_matched','U-range-matched distance','#8B6BB1',':'),('cutoff','Retained-only (logged)','#008F7A','-.')]
    for col,w in enumerate(WORLDS):
        for row in range(2):
            ax=a[row,col];head(ax,w) if row==0 else None
            for j,(key,label,c,ls) in enumerate(controls):
                if row==1 and key=='raw':continue
                field='regret_mean' if row==0 else 'value_gain_vs_raw';arr=np.array([[r['controls'][key][field] for r in get(w,k)] for k in DOSES]);dx=(j-1.5)*.065
                for s in range(3):ax.scatter(np.arange(4)+dx+(s-1)*.015,arr[:,s],s=23,marker=markers[s],color=c or colors[w],alpha=.8)
                ax.plot(np.arange(4)+dx,arr.mean(1),color=c or colors[w],ls=ls,lw=1.3)
            ax.axhline(0,color='#9AA4AD',lw=.7);ax.set_xticks(range(4),['Native','21','63*','189']);ax.set_xlabel('Final policy / actual κ')
            ax.set_ylabel('Mean candidate regret' if row==0 else 'True chosen-value gain vs raw Q1')
    for col,w in enumerate(WORLDS):
        a[0,col].set_title(names[w],loc='left',color=colors[w],fontweight='bold',fontsize=10,pad=10)
    f.legend(handles=[Line2D([],[],color=c or '#172B3A',ls=ls,lw=1.4,label=label) for key,label,c,ls in controls],loc='upper center',bbox_to_anchor=(.5,.865),ncol=4,frameon=False,fontsize=9)
    foot(f,'Distance penalty matches the within-bank kappa U range using U and actions only. It adapts per bank: not a fixed independent control.\nThe retained-only choice is the logged action. 192 shared banks x 3 models; own-policy Qpi. Axes differ across worlds.\nShapes: seed 0 circle, 1 square, 2 triangle. Native kappa=0: raw, BCA and distance controls coincide. *63 was prespecified.')
    figures.append(('03_candidate_controls',f))
    out.mkdir(parents=True,exist_ok=True)
    with PdfPages(out/'Uncertainty_Diagnostics.pdf') as pdf:
        for name,f in figures:
            f.savefig(out/(name+'.png'),dpi=300);f.savefig(out/(name+'.pdf'));pdf.savefig(f);plt.close(f)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output-dir',type=Path,help='New output directory; must not already exist. Default: beside this script.')
    parser.add_argument('--no-plots',action='store_true');args=parser.parse_args()
    out=Path(__file__).resolve().parent if args.output_dir is None else args.output_dir.resolve()
    if args.output_dir is not None:out.mkdir(parents=True,exist_ok=False)
    repo=args.repo_root.resolve();records,manifest=load_saved(repo);results,banks=analyze(records)
    payload={'definitions':DEFINITIONS,'knee':KNEE,'policy_results':results}
    (out/'uncertainty_metrics.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write_csv(out/'uncertainty_metrics.csv',[flatten(r) for r in results]);write_csv(out/'bank_diagnostics.csv',banks)
    if not args.no_plots:plot(results,out/'plots')
    validation={'finished_est':datetime.now(timezone(timedelta(hours=-5))).isoformat(),'source_result_sha256':RESULT_SHA,
                'figure_source_manifest_sha256':sha(repo/'analysis/knee_confirmation_figures_v1/SOURCE_MANIFEST.json'),
                'consumed_sources':manifest['sources'],
                'consumed_extracts':{('analysis/knee_confirmation_figures_v1/data/'+name):h for name,h in manifest['extracts'].items()},
                'verified_original_sources':len(manifest['sources']),'verified_csv_extracts':len(manifest['extracts']),
                'policy_count':len(records),'bank_evaluations':len(banks),'candidate_values':len(banks)*4,
                'pairing':'all candidate inputs identical across models within each world; no crossworld identity claim',
                'knee_source_path':'experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v3/knee_region.py',
                'knee_source_sha256':sha(repo/'experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v3/knee_region.py'),
                'scientific_execution':False,'no_threshold_fitting':True,'script_sha256':sha(Path(__file__)),
                'outputs':{p.relative_to(out).as_posix():sha(p) for p in [out/'uncertainty_metrics.json',out/'uncertainty_metrics.csv',out/'bank_diagnostics.csv',*sorted((out/'plots').glob('*'))] if p.is_file()}}
    (out/'DIAGNOSTIC_VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'saved_array_diagnostics_complete','policies':len(records),'banks':len(banks),'scientific_execution':False}))


if __name__=='__main__':main()
