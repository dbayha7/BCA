"""Freeze and submit one zero-transition query gate to a separate CPU allocation."""
from pathlib import Path
import ast,base64,datetime,gzip,hashlib,json,subprocess
W=Path(__file__).resolve().parent
R=Path('C:/Users/David Bayha/Documents/GitHub/BCA')
REMOTE='/users/dbayha/bca-ood-v2/rebrac-hopper171-query-v1'
files={
 'cluster_query_gate.py':(W/'cluster_query_gate.py').read_bytes(),
 'execution-plan.md':(W/'CLUSTER.md').read_bytes(),
 'experiments/__init__.py':b'', 'experiments/ood/__init__.py':b'',
 'experiments/ood/adapters.py':(R/'experiments/ood/adapters.py').read_bytes(),
 'host-audit.json':(R/'outputs/standard_bca/rebrac/hopper/host/s202609171/verified-v2/audit.json').read_bytes(),
 'bca-audit.json':(R/'outputs/standard_bca/rebrac/hopper/bca_noiw/s202609171/verified-v1/audit.json').read_bytes(),
 'host-acceptance.json':(R/'docs/validation/standard-first-rebrac-host.json').read_bytes(),
 'pair-acceptance.json':(R/'docs/validation/standard-first-rebrac-pair.json').read_bytes()}
ast.parse(files['cluster_query_gate.py'])
files['supervisor.py']=b'''from pathlib import Path
import datetime,json,os,subprocess,sys
r=Path(__file__).resolve().parent
env=dict(os.environ)
env.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONNOUSERSITE='1',
 MUJOCO_PY_FORCE_CPU='1',MUJOCO_PY_MUJOCO_PATH='/users/dbayha/.mujoco/mujoco210',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 MKL_NUM_THREADS='1',D4RL_SUPPRESS_IMPORT_ERROR='1',TF_CPP_MIN_LOG_LEVEL='1',TPU_SKIP_MDS_QUERY='1')
env.pop('PYTHONPATH',None)
env['LD_LIBRARY_PATH']='/users/dbayha/bca-standard-noiw-v1/graphics/lib:/users/dbayha/.mujoco/mujoco210/bin'
env['CPATH']='/users/dbayha/bca-standard-noiw-v1/graphics/include'
env['LIBRARY_PATH']='/users/dbayha/bca-standard-noiw-v1/graphics/lib'
cmd=[sys.executable,'-u',str(r/'cluster_query_gate.py')]
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
with (r/'worker.stdout').open('xb') as out,(r/'worker.stderr').open('xb') as err:
 p=subprocess.Popen(cmd,stdout=out,stderr=err,env=env,cwd=r)
 with (r/'worker-identity.json').open('x') as f:json.dump(dict(pid=p.pid,command=cmd,utc=now(),job_id=os.environ.get('SLURM_JOB_ID'),proc_stat=Path('/proc/'+str(p.pid)+'/stat').read_text()),f,indent=2)
 code=p.wait()
 with (r/'actual_exit.json').open('x') as f:json.dump(dict(actual_exit=code,utc=now(),command=cmd,pid=p.pid,no_retry=True),f,indent=2)
sys.exit(code)
'''
files['job.sbatch']=f'''#!/bin/bash
#SBATCH --job-name=bca-ood-query
#SBATCH --partition=Orion
#SBATCH --account=staruser
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output={REMOTE}/slurm-%j.log
set -eu
exec /users/dbayha/bca-standard-noiw-v1/venv-v2/bin/python {REMOTE}/supervisor.py
'''.encode()
pins={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
files['input-pins.json']=(json.dumps(pins,indent=2)+'\n').encode()
payload=base64.b64encode(gzip.compress(json.dumps({n:base64.b64encode(b).decode() for n,b in files.items()}).encode(),mtime=0)).decode()
bootstrap=f'''from pathlib import Path
import base64,gzip,json,subprocess,hashlib
root=Path({REMOTE!r})
assert not root.exists(),'Preserve existing attempt; duplicate refused'
data=json.loads(gzip.decompress(base64.b64decode({payload!r})))
root.mkdir(parents=True)
for n,b in data.items():
 p=root/n;assert '..' not in p.parts and root in p.parents
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(base64.b64decode(b))
pins=json.loads((root/'input-pins.json').read_bytes())
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in pins.items())
p=subprocess.run(['sbatch','--parsable',str(root/'job.sbatch')],capture_output=True,text=True)
receipt=dict(actual_exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
with (root/'submission.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
raise SystemExit(p.returncode)
'''
with (W/'cluster-query-dispatch-pins.json').open('x') as f:json.dump(pins,f,indent=2)
command=['wsl.exe','--exec','ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-S','/tmp/uncc.sock','dbayha@hpc.charlotte.edu','/usr/bin/python3 -']
p=subprocess.run(command,input=bootstrap.encode(),capture_output=True,timeout=50)
for n,b in [('cluster-query-dispatch.stdout',p.stdout),('cluster-query-dispatch.stderr',p.stderr),('cluster-query-bootstrap.py',bootstrap.encode())]:
    with (W/n).open('xb') as f:f.write(b)
with (W/'cluster-query-dispatch-actual-exit.json').open('x') as f:json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 actual_exit=p.returncode,remote=REMOTE,command=command,checkpoint_files_transferred=False,simulator_steps=0),f,indent=2)
print(p.stdout.decode());print(p.stderr.decode());raise SystemExit(p.returncode)
