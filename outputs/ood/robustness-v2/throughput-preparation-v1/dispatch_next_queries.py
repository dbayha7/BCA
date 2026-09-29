"""Submit five fresh CPU query gates; no simulator, OOD keys or physics ledger."""
from pathlib import Path
import ast,base64,datetime,gzip,hashlib,json,subprocess
W=Path(__file__).resolve().parent
L=W.parents[1]/'ood_live_v2'
R=Path('C:/Users/David Bayha/Documents/GitHub/BCA')
sha=lambda b:hashlib.sha256(b).hexdigest()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
manifest=json.loads((W/'next_pair_manifest.json').read_bytes())
source=(W/'query_next_pair.py').read_bytes()
assert sha(source)==manifest['source_sha256']
original=ast.parse((L/'dispatch_cluster_query.py').read_bytes())
supervisor,=[n.value.value for n in original.body if isinstance(n,ast.Assign)
    and isinstance(n.targets[0],ast.Subscript) and isinstance(n.targets[0].slice,ast.Constant)
    and n.targets[0].slice.value=='supervisor.py']
command=['wsl.exe','--exec','ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12',
         '-S','/tmp/uncc.sock','dbayha@hpc.charlotte.edu','/usr/bin/python3 -']
receipts=[]
for pair in manifest['pairs']:
    env,seed=pair['pair'][1:]
    dest=W/f'query-{env}-{seed}'
    dest.mkdir(exist_ok=False)
    remote=pair['query_root']
    files={'cluster_query_gate.py':source,'supervisor.py':supervisor,
        'pair-intent.json':json.dumps(pair,indent=2).encode(),
        'pair-acceptance.json':Path(pair['acceptance_path']).read_bytes(),
        'experiments/__init__.py':b'','experiments/ood/__init__.py':b'',
        'experiments/ood/adapters.py':(R/'experiments/ood/adapters.py').read_bytes()}
    assert sha(files['pair-acceptance.json'])==pair['acceptance_sha256']
    for method in ('host','bca'):
        files[method+'-audit.json']=Path(pair['methods'][method]['audit_path']).read_bytes()
        assert sha(files[method+'-audit.json'])==pair['methods'][method]['audit_sha256']
    files['job.sbatch']=f'''#!/bin/bash
#SBATCH --job-name=bca-query-{env}-{str(seed)[-3:]}
#SBATCH --partition=Orion
#SBATCH --account=staruser
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output={remote}/slurm-%j.log
set -eu
exec /users/dbayha/bca-standard-noiw-v1/venv-v2/bin/python {remote}/supervisor.py
'''.encode()
    pins={n:sha(b) for n,b in files.items()}
    files['input-pins.json']=(json.dumps(pins,indent=2)+'\n').encode()
    for n,b in files.items():
        p=dest/'staged'/n;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(b)
    payload=base64.b64encode(gzip.compress(json.dumps({n:base64.b64encode(b).decode() for n,b in files.items()}).encode(),mtime=0)).decode()
    bootstrap=f'''from pathlib import Path
import base64,gzip,hashlib,json,os,subprocess
root=Path({remote!r})
assert not root.exists(),'Existing attempt retained; duplicate refused'
data=json.loads(gzip.decompress(base64.b64decode({payload!r})))
root.mkdir()
for n,b in data.items():
 p=root/n;assert '..' not in p.parts and root in p.parents
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(base64.b64decode(b));f.flush();os.fsync(f.fileno())
pins=json.loads((root/'input-pins.json').read_bytes())
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in pins.items())
fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
p=subprocess.run(['sbatch','--parsable',str(root/'job.sbatch')],capture_output=True,text=True)
receipt=dict(actual_exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
with (root/'submission.json').open('x') as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(receipt));raise SystemExit(p.returncode)
'''
    ast.parse(bootstrap)
    (dest/'bootstrap.py').write_bytes(bootstrap.encode())
    p=subprocess.run(command,input=bootstrap.encode(),capture_output=True,timeout=60)
    (dest/'dispatch.stdout').write_bytes(p.stdout);(dest/'dispatch.stderr').write_bytes(p.stderr)
    receipt=dict(utc=now(),actual_exit=p.returncode,pair=pair['pair'],remote=remote,
                 command=command,source_sha256=sha(source),simulator_steps=0,physics_dispatched=False)
    (dest/'dispatch_actual_exit.json').write_bytes(json.dumps(receipt,indent=2).encode())
    assert p.returncode==0,(pair,p.stderr.decode())
    submitted=json.loads(p.stdout);assert submitted['actual_exit']==0,submitted
    job=submitted['stdout'].strip().split(';')[0];assert job.isdecimal(),job
    receipt['job_id']=job;receipts.append(receipt)
    print(json.dumps(receipt),flush=True)
(W/'next_query_submissions.json').write_bytes(json.dumps(receipts,indent=2).encode())
