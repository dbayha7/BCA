"""Freeze and submit the CQL-only preflight-wrapper correction, no science replay."""
from pathlib import Path
import base64,datetime,hashlib,json,subprocess
here=Path(__file__).resolve().parent
old='/users/dbayha/bca-standard-noiw-v1/cql-validation-recovery-v1'
new=old[:-1]+'2'
ssh=['wsl.exe','--exec','ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-S','/tmp/uncc.sock','dbayha@hpc.charlotte.edu','/usr/bin/python3','-']
def call(script):
 p=subprocess.run(ssh,input=script,text=True,capture_output=True,timeout=90)
 if p.returncode:raise RuntimeError(p.stdout+'\n'+p.stderr)
 return json.loads(p.stdout)
receipt=call(f'''
from pathlib import Path
import base64,hashlib,json
r=Path({old!r})
names=['cluster_data_progress.json','data-check-process-v1/actual_exit.json','data-check-process-v1/worker.log','training_slurm_actual_exit.json','execution_freeze.json','source/manifest.json']
assert json.loads((r/'training_slurm_actual_exit.json').read_text())['actual_exit_code']==1
assert not (r/'queue-v1').exists() and not (r/'cql-host-gpu-fixture-v1').exists()
print(json.dumps(dict(files={{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names}},bytes={{n:base64.b64encode((r/n).read_bytes()).decode() for n in names if n!='source/manifest.json'}})))
''')
archive=here/'cluster_attempt1';archive.mkdir(exist_ok=True)
for n,b in receipt['bytes'].items():
 p=archive/n;p.parent.mkdir(parents=True,exist_ok=True);data=base64.b64decode(b);assert not p.exists() or p.read_bytes()==data;p.write_bytes(data)
prior={'original_root':old,'original_job':'27068516','actual_exit':1,'training_started':False,'models_started':False,
 'files':receipt['files'],'correction':'CQL has its own prepared-data hash/schema; use its exact pre-initialization contracts and original CQL fingerprint; reuse seven passed TD3 cells.'}
payload={'cluster_data_check.py':(here/'cluster_data_check_v2.py').read_bytes(),
 'prior_attempt.json':(json.dumps(prior,indent=2)+'\n').encode()}
for n in ('supervise.py','probe_gpu.py','training.sbatch'):
 payload[n]=(here/'cluster_execution'/n).read_text().replace(old,new).encode()
encoded={n:base64.b64encode(b).decode() for n,b in payload.items()}
result=call(f'''
from pathlib import Path
import base64,datetime,hashlib,json,os,shutil,subprocess
old=Path({old!r});r=Path({new!r})
f=json.loads((old/'execution_freeze.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(old/n)==h for n,h in f['files'].items())
m=json.loads((old/'source/manifest.json').read_text())
assert all(sha(old/'source'/n)==h for n,h in m['source_sha256'].items())
gpu=subprocess.run(['squeue','-h','-u','dbayha','-o','%i|%j|%T|%b'],capture_output=True,text=True,check=True).stdout
rows=[x for x in gpu.splitlines() if 'gres/gpu:' in x]
assert len(rows)==1 and rows[0].startswith('27045057|'),rows
r.mkdir(parents=True,exist_ok=False)
shutil.copytree(old/'source',r/'source',ignore=shutil.ignore_patterns('__pycache__'))
for n in f['files']:
 if not n.startswith('source/'):
  (r/n).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old/n,r/n)
for n,b in json.loads({json.dumps(encoded)!r}).items():
 (r/n).write_bytes(base64.b64decode(b))
ownership=json.loads((r/'ownership.json').read_text());ownership.update(destination=str(r),preflight_only_predecessor=str(old),predecessor_actual_exit=1)
(r/'ownership.json').write_text(json.dumps(ownership,indent=2))
names=list(f['files'])+['prior_attempt.json']
freeze={{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{{n:sha(r/n) for n in names}},'manifest_sha256':f['manifest_sha256'],'scientific_rows_unchanged':137,'changed_science':False}}
(r/'execution_freeze.json').write_text(json.dumps(freeze,indent=2))
for n in ['cluster_data_check.py','probe_gpu.py','supervise.py']:compile((r/n).read_bytes(),n,'exec')
subprocess.run(['bash','-n',str(r/'training.sbatch')],check=True)
with (r/'submission_claim.json').open('x') as out:json.dump(dict(previous_gpu_jobs=rows,manifest_sha256=f['manifest_sha256']),out);out.flush();os.fsync(out.fileno())
p=subprocess.run(['sbatch','--parsable',str(r/'training.sbatch')],capture_output=True,text=True)
result=dict(utc=freeze['utc'],returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,job_id=p.stdout.strip().split(';')[0] if p.returncode==0 else None,root=str(r),manifest_sha256=f['manifest_sha256'])
(r/'submission.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
''')
(here/'cluster_v2_submission.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
assert result['returncode']==0
