"""Stage and submit one separate, hash-bound real OOD CPU worker, once."""
from pathlib import Path
import ast,base64,datetime,gzip,hashlib,json,subprocess
W=Path(__file__).resolve().parent
R=Path('C:/Users/David Bayha/Documents/GitHub/BCA')
CORE=W.parent/'ood_robustness_v2'
AMENDED=R/'experiments/ood_robustness_v2'
KEYROOT=W.parent/'standard_bca_noiw_campaign_v1/monitor_20260928T230801Z'
REMOTE='/users/dbayha/bca-ood-v2/rebrac-hopper171-ood-v1'
sha=lambda b:hashlib.sha256(b).hexdigest()
def write(p,b):
    with p.open('xb') as f:f.write(b)
def encoded(v):return (json.dumps(v,indent=2,allow_nan=False)+'\n').encode()

# Inspect the actual immutable local worker and old ledger immediately before dispatch.
local='''from pathlib import Path
import datetime,hashlib,json,sys
w=Path('/mnt/c/Users/David Bayha/Documents/Codex/2026-09-09/this-is-my-phd-research-im/work')
sys.path.insert(0,str(w/'ood_robustness_v2'))
from ancestor_guard import AncestorGuard
b=json.loads((w/'standard_bca_noiw_campaign_v1/monitor_20260928T182657Z/real_ancestor_binding_v2.json').read_bytes())
g=AncestorGuard(b);g.unchanged()
root=Path('/home/dbayha/bca-work/ood-live-v2')
attempt=root/'td3-hopper-s202609171-v1'
decl=json.loads((attempt/'declaration.json').read_bytes())
assert decl['pair']==['td3_bc','hopper',202609171]
source=w/'ood_live_v2/run.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()==decl['sources'][str(source)]
identity=json.loads((root/'process-v1/identity.json').read_bytes())
status=json.loads((attempt/'status.json').read_bytes())
pid=status['pid'];cmd=Path('/proc/'+str(pid)+'/cmdline').read_bytes().replace(b'\\0',b' ').decode()
assert str(source) in cmd and str(attempt) in cmd
import ast
tree=ast.parse(source.read_text())
pair,=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAIR' for t in n.targets)]
assert pair==('td3_bc','hopper',202609171)
assert not (root/'process-v1/actual_exit.json').exists()
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),ancestor=b,
 ancestor_checked_read_only=True,local_worker_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 local_pid=pid,local_command=cmd,local_status=status,local_identity=identity,
 no_local_worker_modified=True,cluster_live_ancestor_access=False)))
'''
p=subprocess.run(['wsl.exe','--exec','/usr/bin/python3','-'],input=local.encode(),capture_output=True,timeout=30)
write(W/'cluster-ood-v2-local-preflight.stdout',p.stdout);write(W/'cluster-ood-v2-local-preflight.stderr',p.stderr)
write(W/'cluster-ood-v2-local-preflight-exit.json',encoded(dict(actual_exit=p.returncode)))
assert p.returncode==0,p.stderr.decode()
attest=json.loads(p.stdout)
partition=dict(schema='ood-v2-disjoint-host-allocation-v1',local_owner='td3_bc',cluster_owner='rebrac',
 cluster_current_pair=['rebrac','hopper',202609171],local_max_environment=18395680,cluster_max_environment=18395680,
 original_reserved_environment=attest['ancestor']['reserved']['global'][0],combined_max_environment=38089713,
 original_environment_cap=39998400,cluster_current_pair_cap=1847568,
 local_immutable_worker_sha256=attest['local_worker_sha256'],no_duplicate_pair_allowed=True,
 remote_ancestor_attestation=True,future_dispatches_debit_existing_allocations=True)
assert partition['original_reserved_environment']+partition['local_max_environment']+partition['cluster_max_environment']==partition['combined_max_environment']<partition['original_environment_cap']
files={n:(W/n).read_bytes() for n in ('cluster_run.py','cluster_loader.py','cluster_review_engineering.py','RESOURCE_PARTITION.md','CLUSTER.md')}
for n in ('ancestor_guard','extension_ledger','recorded_step','native_constructor','precommit_bank','precommit_file','candidate_design'):
    files['core/'+n+'.py']=(CORE/(n+'.py')).read_bytes()
for n in ('collector_driver_amended_v1','outcome_driver_amended_v1','precommit_bank_amended_v1','candidate_design_amended_v1'):
    files['core/'+n+'.py']=(AMENDED/(n+'.py')).read_bytes()
# Exact accepted modules, no source rewrites and no training weights transferred.
for n in ('adapters','collect','oracle','production','real_connection','resources','streaming','simulator','protocol'):
    files['experiments/ood/'+n+'.py']=(R/('experiments/ood/'+n+'.py')).read_bytes()
files['experiments/__init__.py']=b'';files['experiments/ood/__init__.py']=b''
files['ancestor-binding.json']=encoded(attest['ancestor'])
files['local-ancestor-attestation.json']=encoded(attest);files['resource-partition.json']=encoded(partition)
files['runtime-versions.json']=encoded(json.loads((W/'cluster-runtime.stdout').read_bytes())['packages'])
files['full-declaration.json']=(R/'runs/ood/td3-hopper-s202609171-connection-v2/full-declaration.json').read_bytes()
files['scientific-plan.md']=(R/'docs/superpowers/plans/2026-09-28-ood-robustness-v2.md').read_bytes()
assert sha(files['scientific-plan.md'])=='f11ebe8f8e3ccc4af10511ed1a241e8f7c5f0aa9f84770103dd6928e59b60756'
for n in ('saved_key_review.json','stream_input_review.json'):files[n]=(KEYROOT/n).read_bytes()
kr=json.loads(files['saved_key_review.json'])
record,=[x for x in kr['records'] if x['kind']=='v2' and x['pair']==partition['cluster_current_pair']]
files['continuation.u32']=(KEYROOT/record['file']).read_bytes();assert sha(files['continuation.u32'])==record['sha256']
for n in ('query-acceptance.json',):files[n]=(W/'cluster-query-accepted-v1'/n).read_bytes()
files['query-independent-review.json']=(W/'cluster-query-accepted-v1/independent-review.json').read_bytes()
for method,arm,ver in [('host','host','verified-v2'),('bca','bca_noiw','verified-v1')]:
    files[method+'-audit.json']=(R/f'outputs/standard_bca/rebrac/hopper/{arm}/s202609171/{ver}/audit.json').read_bytes()
for n,r in [('host-acceptance','standard-first-rebrac-host'),('pair-acceptance','standard-first-rebrac-pair')]:
    files[n+'.json']=(R/f'docs/validation/{r}.json').read_bytes()
# Reuse the already successful supervision/environment contract exactly, changing
# only the executed file and job time/name/output path in this separate attempt.
original=ast.parse((W/'dispatch_cluster_query.py').read_text())
supervisor,=[node.value.value for node in original.body if isinstance(node,ast.Assign)
    and isinstance(node.targets[0],ast.Subscript) and isinstance(node.targets[0].slice,ast.Constant)
    and node.targets[0].slice.value=='supervisor.py']
files['supervisor.py']=supervisor.replace(b"'cluster_query_gate.py'",b"'cluster_run.py'")
files['job.sbatch']=f'''#!/bin/bash
#SBATCH --job-name=bca-ood-rebrac
#SBATCH --partition=Orion
#SBATCH --account=staruser
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output={REMOTE}/slurm-%j.log
set -eu
exec /users/dbayha/bca-standard-noiw-v1/venv-v2/bin/python {REMOTE}/supervisor.py
'''.encode()
for n,b in files.items():
    if n.endswith('.py'):ast.parse(b,filename=n)
pins={n:sha(b) for n,b in files.items()};files['input-pins.json']=encoded(pins)
write(W/'cluster-ood-v2-resource-partition.json',encoded(partition))
write(W/'cluster-ood-v2-dispatch-pins.json',encoded(pins))
payload=base64.b64encode(gzip.compress(json.dumps({n:base64.b64encode(b).decode() for n,b in files.items()}).encode(),mtime=0)).decode()
bootstrap=f'''from pathlib import Path
import base64,gzip,json,subprocess,hashlib,os
root=Path({REMOTE!r})
assert not root.exists(),'Existing attempt preserved; no duplicate submission'
assert not (root.parent/'rebrac-hopper171-extension.sqlite').exists(),'Prior pair ledger exists'
data=json.loads(gzip.decompress(base64.b64decode({payload!r})))
root.mkdir(parents=True)
for n,b in data.items():
 p=root/n;assert '..' not in p.parts and root in p.parents
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(base64.b64decode(b))
pins=json.loads((root/'input-pins.json').read_bytes())
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in pins.items())
lock=root.parent/'cluster-ood.lock'
if not lock.exists():
 fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
p=subprocess.run(['sbatch','--parsable',str(root/'job.sbatch')],capture_output=True,text=True)
receipt=dict(actual_exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
with (root/'submission.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt));raise SystemExit(p.returncode)
'''
write(W/'cluster-ood-v2-bootstrap.py',bootstrap.encode())
cmd=['wsl.exe','--exec','ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-S','/tmp/uncc.sock','dbayha@hpc.charlotte.edu','/usr/bin/python3 -']
p=subprocess.run(cmd,input=bootstrap.encode(),capture_output=True,timeout=55)
write(W/'cluster-ood-v2-dispatch.stdout',p.stdout);write(W/'cluster-ood-v2-dispatch.stderr',p.stderr)
write(W/'cluster-ood-v2-dispatch-exit.json',encoded(dict(actual_exit=p.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),remote=REMOTE,checkpoint_files_transferred=False)))
print(p.stdout.decode());print(p.stderr.decode());raise SystemExit(p.returncode)
