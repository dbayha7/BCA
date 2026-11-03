"""Saved-array pairwise decision diagnostics. No learner or simulator imports.

All labels are retrospective descriptions; no classifier, coefficient or threshold
is fitted. Own-policy Qpi remains evaluator-only. Output directories are exclusive.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import time
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULT_SHA = 'b136b39c670e7224c22fe1b8f38062a22d30f608d22c9b4253c406194b53d197'
PREVIOUS_RUN_SHA = '8705604e8547c0ec836412336aebfca2281a2803a38ac9ac3523568786004a86'
KNEE = float(np.float32(.8))
DEFINITIONS = {
    'scope': 'retrospective CPU arithmetic on authenticated saved reports only',
    'pair': 'one withheld complete action versus logged complete action at identical state/time, followed by the same frozen policy',
    'g': 'Qpi_withheld - Qpi_logged; evaluator-only true advantage within this finite candidate pair',
    'm': 'Q1_withheld - Q1_logged; decision-time predicted advantage',
    'e': 'm-g; relative Q-ranking error; cancels a common Q offset',
    'du': 'U_withheld - U_logged; decision-time width contrast',
    'p': 'actual kappa * du; adjusted margin m-p',
    'head': 'h=Q1-min(Q1,Q2), dh=h_withheld-h_logged; min-Q margin=m-dh',
    'head_identity': 'Q1 relative error = min-Q relative error + dh; arithmetic identity, not proof that one head causes error',
    'signs': 'harmful g<-tau; useful g>tau; abs(g)<=tau near/equal and excluded from binary discrimination',
    'wrong_preference': 'harmful and m>0, or useful and m<=0; pair tie chooses logged slot0',
    'signal_orientation': {'negative_raw_margin':'-m: lower predicted advantage predicts harm','delta_u':'du: larger extra width predicts harm','cutoff_distance':'max(abs(a0)-knee,0): deeper action gap predicts harm','head_excess_contrast':'dh: greater Q1 excess over twin-min predicts harm'},
    'discrimination': 'pooled across all eligible pairs; within-bank unweighted mean AUROC among mixed-label banks; between-bank Spearman of mean signal versus harmful fraction, no fitted model',
    'sensitivity': 'exact labels plus observed maximum t0 identity discrepancy and twice that discrepancy; empirical numerical stress scales, not a validated error bound or changed original threshold',
    'dependence': '192 shared banks and 3 pairs per bank per model; 3 independent training seeds; no iid-pair confidence interval',
    'comparability': 'same candidate inputs within world; own-policy continuations differ across models and actual states differ across worlds',
    'fixed_strata': 'candidate slots 1/2/3 are magnitudes .85/.9/1.0; time thirds 0-32/33-65/66-99, descriptive only',
    'prohibited_claims': ['not a deployed action-selection intervention','not Bayesian superiority','not a new original-test pass','no signed residual or local residual supervision reconstructed'],
}

def require(ok, message):
    if not ok: raise ValueError(message)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def effective_kappa(report):
    require(report['arm'] in ['native','bca'],'Unexpected method arm')
    return 0 if report['arm']=='native' else int(report['kappa'])

def ranks(a):
    a=np.asarray(a,float).ravel(); require(np.isfinite(a).all(),'Nonfinite ranks')
    order=np.argsort(a,kind='mergesort'); out=np.empty(a.size); i=0
    while i<a.size:
        j=i+1
        while j<a.size and a[order[j]]==a[order[i]]:j+=1
        out[order[i:j]]=(i+j-1)/2+1;i=j
    return out

def corr(x,y):
    x=np.asarray(x,float).ravel();y=np.asarray(y,float).ravel()
    require(x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all(),'Invalid correlation')
    if len(x)<2:return None
    x=x-x.mean();y=y-y.mean();den=np.linalg.norm(x)*np.linalg.norm(y)
    return None if den==0 else float(np.clip(x@y/den,-1,1))

def auc(x,event):
    x=np.asarray(x,float).ravel();e=np.asarray(event,bool).ravel()
    require(x.shape==e.shape,'AUC shapes differ');n1=int(e.sum());n0=len(e)-n1
    if not n1 or not n0:return None
    return float((ranks(x)[e].sum()-n1*(n1+1)/2)/(n1*n0))

def describe(a):
    a=np.asarray(a,float).ravel()
    if not len(a):return {'n':0,'mean':None,'median':None,'p10':None,'p90':None,'mae':None}
    return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'p10':float(np.quantile(a,.1)),'p90':float(np.quantile(a,.9)),'mae':float(abs(a).mean())}

def pair_arrays(q1,q2,value,u,kappa):
    q1,q2,value,u=[np.asarray(x,float) for x in (q1,q2,value,u)]
    require(q1.ndim==2 and q1.shape[1]==4 and all(x.shape==q1.shape for x in (q2,value,u)),'Expected aligned banks x4')
    require(all(np.isfinite(x).all() for x in (q1,q2,value,u)),'Nonfinite saved candidate')
    delta=lambda x:x[:,1:]-x[:,[0]]
    g=delta(value);m=delta(q1);du=delta(u);dh=delta(q1-np.minimum(q1,q2));p=float(kappa)*du
    return {'g':g,'m':m,'e':m-g,'du':du,'p':p,'adjusted_margin':m-p,'adjusted_error':m-p-g,'dh':dh,'min_margin':m-dh,'min_error':m-dh-g,'withheld_raw_error':q1[:,1:]-value[:,1:],'logged_raw_error':np.broadcast_to(q1[:,[0]]-value[:,[0]],g.shape)}

def events(a,tau):
    g,m,z=a['g'],a['m'],a['adjusted_margin'];harm=g < -tau;use=g > tau;near=~(harm|use)
    return {'harmful':harm,'useful':use,'near':near,'raw_false_preference':harm&(m>0),'raw_missed_useful':use&(m<=0),'corrected_harmful':harm&(m>0)&(z<=0),'unfixed_harmful':harm&(m>0)&(z>0),'new_harmful_preference':harm&(m<=0)&(z>0),'suppressed_useful':use&(m>0)&(z<=0),'preserved_useful_preference':use&(m>0)&(z>0),'rescued_useful':use&(m<=0)&(z>0),'still_missed_useful':use&(m<=0)&(z<=0)}

def discrimination(signal,g,tau):
    signal=np.asarray(signal,float);require(signal.shape==g.shape and signal.ndim==2,'Discrimination banks misaligned')
    selected=abs(g)>tau;harm=g < -tau
    vals=[auc(s[keep],e[keep]) for s,e,keep in zip(signal,harm,selected)]
    good=[v for v in vals if v is not None]
    keepbanks=selected.any(1); means=[]; rates=[]
    for s,e,keep in zip(signal[keepbanks],harm[keepbanks],selected[keepbanks]):means.append(s[keep].mean());rates.append(e[keep].mean())
    return {'pooled_auc':auc(signal[selected],harm[selected]),'within_bank_auc':None if not good else float(np.mean(good)),'eligible_mixed_banks':len(good),'total_banks':len(g),'eligible_pairs':int(selected.sum()),'harmful_pairs':int(harm.sum()),'between_bank_spearman':corr(ranks(means),ranks(rates))}

def load_saved(repo):
    prior=repo/'analysis/knee_support_uncertainty_v1/output/verified_run'; receipt=prior/'RUN_RECEIPT.json'
    require(sha(receipt)==PREVIOUS_RUN_SHA,'Previous analysis receipt changed')
    previous=json.loads(receipt.read_text());inputs={str(receipt.relative_to(repo)).replace('\\','/'):sha(receipt)}
    for rel,digest in previous['outputs'].items():
        p=prior/rel;require(sha(p)==digest,'Changed prior output: '+rel);inputs[p.relative_to(repo).as_posix()]=digest
    mp=repo/'analysis/knee_confirmation_figures_v1/SOURCE_MANIFEST.json';manifest=json.loads(mp.read_text());inputs[mp.relative_to(repo).as_posix()]=sha(mp)
    require(manifest['final_result_sha256']==RESULT_SHA,'Wrong study result')
    records={};identities={}; checks=[]
    for src in manifest['sources']:
        p=repo/src['path'];require(sha(p)==src['sha256'],'Changed original source: '+src['path']);inputs[src['path']]=src['sha256']
        if p.name!='report.json' or not p.parent.name.startswith('eval_'):continue
        o=json.loads(p.read_text());require(o['status']=='completed','Incomplete evaluation')
        key=(o['world'],effective_kappa(o),int(o['seed_index']));require(key not in records,'Duplicate policy')
        d=o['data'];require(np.asarray(d['probes']).shape==(192,4,4),'Incomplete candidate bank')
        identity=json.dumps([d['rows'],d['times'],d['probes']],sort_keys=True)
        if key[0] in identities:require(identity==identities[key[0]],'Within-world candidate identity differs')
        identities[key[0]]=identity;records[key]=o
        ds=[]
        for i,(episode,t) in enumerate(o['own']['rows']):
            if t==0:
                qpi=o['own']['qpi'][i];ret=o['returns'][episode];ds.append({'episode':episode,'qpi':qpi,'return':ret,'absolute':abs(qpi-ret),'relative':abs(qpi-ret)/max(1,abs(ret))})
        require(len(ds)==16,'Unexpected t0 identity count')
        best=max(ds,key=lambda x:x['relative']);require(abs(best['relative']-o['instrument']['I1_max_rel'])<1e-14,'I1 reconstruction mismatch')
        checks.append({'world':key[0],'dose':key[1],'seed':key[2],'reported_ok':o['instrument']['I1_ok'],'reported_tolerance':o['instrument']['I1_tol'],'max_absolute':max(x['absolute'] for x in ds),'worst_relative_case':best,'source':src['path']})
    require(set(records)=={(w,k,s) for w in ['stall3','nostall'] for k in [0,21,63,189] for s in range(3)},'Expected complete 24 models')
    return records,inputs,checks

def analyze(records,thresholds):
    policies=[];pairs=[]
    for (world,kappa,seed),o in sorted(records.items()):
        d=o['data'];a=pair_arrays(d['q1'],d['q2'],d['qpi'],d['u'],kappa)
        actions=np.asarray(d['probes']);distance=np.maximum(abs(actions[:,1:,0])-KNEE,0)
        require((abs(actions[:,0,0])<=KNEE).all() and (distance>0).all(),'Unexpected support labels')
        signals={'negative_raw_margin':-a['m'],'delta_u':a['du'],'cutoff_distance':distance,'head_excess_contrast':a['dh']}
        sensitivity=[]
        for tau in thresholds:
            ev=events(a,tau);groups={}
            for group in ['harmful','useful','raw_false_preference','raw_missed_useful']:
                mask=ev[group];groups[group]={name:describe(v[mask]) for name,v in a.items()}
                groups[group]['positive_du']=int((mask&(a['du']>0)).sum());groups[group]['positive_dh']=int((mask&(a['dh']>0)).sum())
                groups[group]['withheld_error_positive']=int((mask&(a['withheld_raw_error']>0)).sum())
            sensitivity.append({'tau':tau,'counts':{name:int(mask.sum()) for name,mask in ev.items()},'groups':groups,'signals':{name:discrimination(s,a['g'],tau) for name,s in signals.items()},'relative_error':{'q1':describe(a['e'][~ev['near']]),'twin_min':describe(a['min_error'][~ev['near']]),'adjusted':describe(a['adjusted_error'][~ev['near']])},'relative_mae_improved_by_adjustment':int(((abs(a['adjusted_error'])<abs(a['e']))&~ev['near']).sum()),'relative_mae_worsened_by_adjustment':int(((abs(a['adjusted_error'])>abs(a['e']))&~ev['near']).sum())})
        native=records[world,0,seed];dJ=np.asarray(o['returns'])-np.asarray(native['returns'])
        corr_summary={}
        for name,s in signals.items():
            corr_summary[name]={'pooled_gap_spearman':corr(ranks(s),ranks(a['g'])),'within_bank_gap_pearson':corr(s-s.mean(1,keepdims=True),a['g']-a['g'].mean(1,keepdims=True)),'between_bank_gap_spearman':corr(ranks(s.mean(1)),ranks(a['g'].mean(1)))}
        strata={}
        times=np.asarray(d['times'])
        for name,mask in {**{'magnitude_'+str(slot):np.broadcast_to(np.arange(3)[None,:]==slot-1,(192,3)) for slot in [1,2,3]},**{'time_'+label:np.broadcast_to(((times>=low)&(times<=high))[:,None],(192,3)) for label,low,high in [('0_32',0,32),('33_65',33,65),('66_99',66,99)]}}.items():
            strata[name]={'n':int(mask.sum()),'harmful':int((mask&(a['g']<0)).sum()),'useful':int((mask&(a['g']>0)).sum()),'near_exact':int((mask&(a['g']==0)).sum()),'g':describe(a['g'][mask]),'m':describe(a['m'][mask]),'e':describe(a['e'][mask]),'du':describe(a['du'][mask])}
        policies.append({'world':world,'dose':kappa,'reported_config_kappa':o['kappa'],'seed':seed,'banks':192,'pairs':576,'policy_return':o['J'],'paired_return_delta':float(dJ.mean()),'sensitivity':sensitivity,'gap_associations':corr_summary,'pair_gap':describe(a['g']),'raw_margin':describe(a['m']),'head_contrast':describe(a['dh']),'fixed_strata':strata})
        for i,(episode,t) in enumerate(d['rows']):
            for j in range(3):
                row={'world':world,'dose':kappa,'seed':seed,'bank':i,'episode':episode,'time':t,'slot':j+1,'a0':actions[i,j+1,0],'cutoff_distance':distance[i,j]}
                for name,v in a.items():row[name]=v[i,0] if name=='logged_raw_error' else v[i,j]
                pairs.append(row)
    return policies,pairs

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo-root',type=Path,default=REPO);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args(argv)
    out=args.output_dir.resolve();require(out.is_relative_to(HERE),'Output outside isolated analysis');require(not out.exists(),'Output already exists; preserve it')
    started=time.monotonic();repo=args.repo_root.resolve();records,inputs,checks=load_saved(repo)
    eps=max(c['max_absolute'] for c in checks);thresholds=[0.,eps,2*eps]
    policies,pairs=analyze(records,thresholds)
    require(all(sha(repo/p)==v for p,v in inputs.items()),'Input changed during analysis')
    out.mkdir(parents=True,exist_ok=False)
    payload={'definitions':DEFINITIONS,'thresholds':thresholds,'identity_checks':checks,'policies':policies,'saved_component_inventory':{'present':['Q1','Q2','U','Qpi','complete probe actions','source row/time','own-policy return','aggregate instrument checks'],'not_in_candidate_reports':['candidate one-step target','signed TD residual','eta per candidate','local-scale loss/coverage per candidate','per-candidate actor gradient'],'warning':'Additional quantities in training receipts do not automatically provide same-state candidate-aligned supervision.'}}
    (out/'metrics.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (out/'pairs.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    receipt={'status':'completed_saved_pair_analysis','finished_est':datetime.now(timezone(timedelta(hours=-5))).isoformat(),'elapsed_seconds':time.monotonic()-started,'source_result_sha256':RESULT_SHA,'inputs':inputs,'analysis_source_sha256':sha(Path(__file__)),'models':24,'banks':4608,'pairs':len(pairs),'scientific_calls':{'model':0,'simulator':0,'optimizer':0,'calibration':0},'outputs':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    (out/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':receipt['status'],'pairs':len(pairs),'thresholds':thresholds,'receipt':str(out/'RECEIPT.json')}))

if __name__=='__main__':main()
