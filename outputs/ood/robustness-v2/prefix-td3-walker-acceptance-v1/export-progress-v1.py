from pathlib import Path
import ast,base64,datetime,gzip,hashlib,json
W=Path(__file__).resolve().parent;R=Path('C:/Users/David Bayha/Documents/GitHub/BCA');P=R/'outputs/ood/robustness-v2/prefix-td3-walker-acceptance-v1'
sha=lambda b:hashlib.sha256(b).hexdigest();j=lambda p:json.loads(p.read_bytes());enc=lambda v:(json.dumps(v,indent=2,sort_keys=True)+'\n').encode()
assert j(W/'independent_training_review_actual_exit.json')['actual_exit']==0
review=j(W/'independent_export_review.json');assert review['accepted_for_recorded_training_comparison'] and not review['ood_execution_accepted']
prefix=j(W/'prefix_summary.json');assert prefix['active_scientific_sqlite_opens']==0
fetch=max(W.glob('review-fetch-*'),key=lambda p:p.name);training=fetch/'training'
assert j(training/'pair_audit_actual_exit.json')['actual_exit']==0
cql=j(W/'cql_closures.json');assert len(cql['completed_receipts'])==5
new=cql['completed_receipts'][-1];assert new['run_id']=='cql-walker2d-bca-noiw-s202609171' and new['actual_exit']['actual_returncode']==0
closure={k:new[k] for k in ('run_id','actual_exit','training_exit','result_sha256')};closure.update(steps_completed=new['result']['steps_completed'],evaluation_banks=len(new['result']['evaluations']),independent_saved_training_acceptance=False)
with (W/'new_cql_closure.json').open('xb') as f:f.write(enc(dict(new_closure=closure,current_worker=cql['live_worker'],queue_counts=dict(completed=5,active=1,unstarted=131,total=137))))
items={}
for p in sorted(W.rglob('*')):
 if not p.is_file() or p.suffix not in ('.py','.json','.stdout','.stderr','.sbatch') or '__pycache__' in p.parts:continue
 n=p.relative_to(W).as_posix()
 if n.startswith('current_workers/') or p.name.startswith(('monitor_','continuation_observation_','observe_','publication_','export_','publish_','prepared_monitor','automation_')):continue
 if n.startswith('review-fetch-') and (fetch.name not in n or '/training/host-verified/' in n or '/training/bca-verified/' in n or p.name=='transport.stdout'):continue
 if p.name in ('cql_closures.json','cql_closure_observation.stdout','cql_closure_observation.stderr','check_cql_closures.py','cql_closure_source.py'):continue
 b=p.read_bytes();assert len(b)<2000000,n
 if p.suffix=='.py':ast.parse(b)
 items[n]=dict(bytes=len(b),sha256=sha(b),base64=base64.b64encode(b).decode())
files={}
def put(path,b):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(b)
 files[path.relative_to(R).as_posix()]=sha(b)
assert not P.exists();P.mkdir(parents=True)
readme=(W.parents[1]/'ood_live_v2/TD3_WALKER_ACCEPTANCE_PREFIXES_20260929.md').read_bytes()
put(P/'README.md',readme)
validation=dict(schema='ood-v2-prefix-td3-walker-training-acceptance-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prefixes=prefix,training_review=review,training_root='/users/dbayha/bca-standard-noiw-v1/cql-validation-recovery-v2/analysis/td3-walker171-saved-121840Z-v1',new_cql_closure=closure,science_pair_complete=False,new_ood_pair_eligible=False,paired_training_inventory=dict(pairs=8,successful_runs=16,recovered_cql_host_separate=True),readout_source_sha256='646e0af0cc396162896a3e31d65f3938985d16f240e13075c50145346053cfc6')
put(P/'validation.json',enc(validation));put(R/'docs/validation/ood-v2-prefix-td3-walker-acceptance.json',enc(validation))
raw=gzip.compress(enc(dict(schema='exact-small-bytes-v1',items=items)),mtime=0);put(P/'evidence.json.gz',raw)
put(P/'package.json',enc(dict(items=len(items),compressed_bytes=len(raw),sha256=sha(raw),files={k:{n:v for n,v in x.items() if n!='base64'} for k,x in items.items()})))
put(P/'.gitattributes',b'* -text\n')
for method,folder in [('host','host'),('bca','bca_noiw')]:
 dest=R/'outputs/standard_bca/td3_bc/walker2d'/folder/'s202609171/verified-v1';assert not dest.exists()
 for p in sorted((training/(method+'-verified')).glob('*.json')):put(dest/p.name,p.read_bytes())
 put(dest/'actual_audit_exit.json',(training/(method+'_audit_actual_exit.json')).read_bytes());put(dest/'.gitattributes',b'* -text\n')
put(R/'docs/validation/standard-td3-walker-first-pair.json',(W/'independent_export_review.json').read_bytes())
status=R/'docs/STANDARD_BCA_STATUS.md';before=status.read_bytes()
with (W/'standard_status_before.md').open('xb') as f:f.write(before)
heading=b'# Standard BCA: execution status\r\n';assert before.startswith(heading)
section='''
## September 29, 12:35 UTC: first TD3+BC Walker training pair accepted

Seed202609171 host/BCA passed saved-data audits and independent arithmetic review,
all actualexit0. The paired training inventory now has8 accepted pairs and16
successful runs; the recovered CQL Hopper host with original exit1 remains separate.
Final normalized means are77.246103 host and82.896940 BCA (+5.650837); periodic
curve means are73.795470 host and71.579274 BCA (-2.216197). BCA wins8/20 paired
final resets. One paired training seed supports a descriptive contrast, not a
replicated benefit or training-seed uncertainty estimate.

The raw data, dependency split, training-only normalization,108 execution-source
pins, three checkpoints per method,402 evaluation banks/4040 episodes and198 BCA
refreshes passed. No learner/model/simulator was run by the review. OOD checkpoint
queries and native simulator gates remain pending for this new pair. All five
existing OOD executions continue; no v2 OOD comparison is complete.

The recovery training queue has5 completed/1 active/131 unstarted rows at12:34.
CQL Walker BCA171 completed actualexit0 at12:18:51; its independent training review
is pending. TD3 HalfCheetah host171 is the current worker. The local queue stays
delegated, the original ReBRAC/IQL queue is unchanged, and the two-GPU cap holds.

[Independent training review](validation/standard-td3-walker-first-pair.json),
[training and OOD evidence readout](../outputs/ood/robustness-v2/prefix-td3-walker-acceptance-v1/README.md).
'''.replace('\n','\r\n').encode()
assert status.read_bytes()==before
status.write_bytes(heading+section+before[len(heading):]);files[status.relative_to(R).as_posix()]=sha(status.read_bytes())
receipt=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_exit=0,files=files,package_items=len(items),compressed_bytes=len(raw),scientific_acceptance=False,new_training_pair_accepted=True,prior_status_sha256=sha(before))
with (W/'export_actual_exit.json').open('xb') as f:f.write(enc(receipt))
print(json.dumps(dict(actual_exit=0,items=len(items),compressed_bytes=len(raw),files=len(files))))
