"""Independent saved-result and execution-amendment review; zero model/physics calls."""
from pathlib import Path
import collections, gzip, hashlib, importlib.util, json, math, os, sys
os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')
from flax import serialization
import numpy as np

root=Path('/home/dbayha/bca-work/standard-noiw-v1');rec=root/'cql-saved-recovery-v2';dest=root/'recovery-validation-v2'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def req(v,m):
    if not v: raise ValueError(m)
audit=read(rec/'audit.json');result=read(rec/'recovered_result.json');run=Path(result['recovery']['original_run'])
req(sha(rec/'recovered_result.json')==audit['recovered_result_sha256'],'Result binding')
for p,h in read(rec/'original_hashes.json').items():req(sha(p)==h,'Original mutation: '+p)
req(result['recovery']['original_worker_exit']['actual_returncode']==1,'Original exit must remain 1')
req(not (run/'result.json').exists() and not (run/'exit.json').exists(),'Original failed path changed')
row=read(run/'resolved.json'); banks=[];scan_steps=[];kinds=collections.Counter()
with gzip.open(run/'events.jsonl.gz','rt') as f:
    for line in f:
        event=json.loads(line);kinds[event['kind']]+=1
        if event['kind'] in ('periodic','final'):banks.append(event)
        if event['kind']=='accepted_scan':
            scan_steps.append(event['step']);req(all(math.isfinite(x) for x in event['metrics_last'].values()),'Nonfinite metric')
req(banks==result['evaluations'] and scan_steps==list(range(1000,1000001,1000)),'Saved journal join')
req(kinds=={'prepared':1,'accepted_scan':1000,'periodic':200,'final':1,'completed':1} and event['kind']=='completed','Closed event sequence')
for bank,wanted in zip(banks,row['protocol']['evaluation_events']):
    req(all(bank[k]==wanted[k] for k in ['kind','step','episode_seeds']),'Bank identity')
    req(len(bank['episodes'])==bank['episode_count']==len(wanted['episode_seeds']),'Bank size')
    for i,e in enumerate(bank['episodes']):
        req(e['completed'] and e['seed']==wanted['episode_seeds'][i] and 0<e['length']<=1000,'Episode completion')
        t=bank['score_transform'];s=100*(e['return']-t['reference_min'])/(t['reference_max']-t['reference_min'])
        req(math.isfinite(s) and abs(s-e['score'])<1e-10,'Score arithmetic')
def scalar_counts(v,path=''):
    out={}
    if isinstance(v,dict):
        for k,x in v.items():
            if k in ('count','step') and np.asarray(x).shape==():out[path+'/'+k]=int(x)
            elif isinstance(x,dict):out.update(scalar_counts(x,path+'/'+k))
    return out
for entry in result['checkpoints']:
    p=run/entry['path'];req(sha(p)==entry['sha256'],'Checkpoint binding');t=serialization.msgpack_restore(p.read_bytes());step=entry['step']
    req(int(t['step'])==step,'Host counter')
    n=t['state']['native']
    for k in ['actor','critic1','critic2','log_alpha','log_alpha_prime']:
        v=scalar_counts(n[k]);req(set(v)=={'/step','/opt_state/0/count'} and all(x==step for x in v.values()),'Native counter '+k)
    for k in ['critic1_target','critic2_target']:
        req(scalar_counts(n[k])=={'/step':step,'/opt_state/0/count':0},'Target count')
    req(all(t['state'][k] is None for k in ['calibrator','posterior','residual_scale']),'Native calibration state')
old=read(root/'source/manifest.json');new=read(dest/'source/manifest.json');pre=read(dest/'preflight.json')
req(new['runs']==[x for x in old['runs'] if x['lane']=='local'][3:],'Scientific rows altered')
req(len(new['runs'])==137 and len({x['row']['run_id'] for x in new['runs']})==137,'Row count/duplicates')
diff=[n for n,h in old['source_sha256'].items() if new['source_sha256'][n]!=h]
req(set(diff)=={'train.py','runtime/validation.py'},'Unexpected source changes')
req(all(sha(dest/'source'/n)==h for n,h in new['source_sha256'].items()),'Frozen new source mismatch')
req(sha(dest/'source/manifest.json')==pre['manifest_sha256'],'Execution manifest mismatch')
req(sha(dest/'supervise.py')==pre['supervisor_sha256'],'Supervisor mismatch')
v=dict(accepted=True,saved_result_accepted=True,original_process_exit=1,original_files_unchanged=True,
       recovered_result_sha256=sha(rec/'recovered_result.json'),audit_sha256=sha(rec/'audit.json'),
       review_source_sha256=sha(__file__),manifest_sha256=pre['manifest_sha256'],unstarted_rows=137,
       changed_sources=diff,final_mean=float(np.mean(banks[-1]['scores'])),
       periodic_curve_mean=float(np.mean([np.mean(b['scores']) for b in banks[:-1]])),
       model_queries=0,simulator_steps=0,training_updates=0)
with (dest/'independent-review.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n')
print(json.dumps(v,indent=2))
