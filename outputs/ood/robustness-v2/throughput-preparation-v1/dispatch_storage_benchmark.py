from pathlib import Path
import base64,datetime,hashlib,json,subprocess
W=Path(__file__).resolve().parent
root='/users/dbayha/bca-ood-v2/storage-benchmark-20260929T075005Z'
raw=(W/'storage_benchmark.py').read_bytes();compile(raw,'storage_benchmark.py','exec')
supervisor=b'''from pathlib import Path
import datetime,json,os,subprocess,sys
r=Path(__file__).resolve().parent
cmd=['/usr/bin/python3','-B',str(r/'storage_benchmark.py'),'--root',str(r/'shared'),'--node','/tmp/bca-storage-benchmark-'+os.environ['SLURM_JOB_ID']]
with (r/'stdout.json').open('xb') as out,(r/'stderr.txt').open('xb') as err:
 p=subprocess.Popen(cmd,stdout=out,stderr=err)
 with (r/'identity.json').open('x') as f:json.dump(dict(pid=p.pid,command=cmd,job_id=os.environ['SLURM_JOB_ID'],proc_stat=Path('/proc/'+str(p.pid)+'/stat').read_text()),f)
 code=p.wait()
 with (r/'actual_exit.json').open('x') as f:json.dump(dict(actual_exit=code,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd,pid=p.pid),f)
sys.exit(code)
'''
batch=f'''#!/bin/bash
#SBATCH --job-name=bca-storage-only
#SBATCH --partition=Orion
#SBATCH --account=staruser
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output={root}/slurm-%j.log
set -eu
exec /usr/bin/python3 {root}/supervisor.py
'''.encode()
files={'storage_benchmark.py':raw,'supervisor.py':supervisor,'job.sbatch':batch}
payload=base64.b64encode(json.dumps({n:base64.b64encode(b).decode() for n,b in files.items()}).encode()).decode()
code=f'''from pathlib import Path
import base64,json,hashlib,subprocess
r=Path({root!r});r.mkdir(exist_ok=False)
v=json.loads(base64.b64decode({payload!r}))
for n,b in v.items():
 with (r/n).open('xb') as f:f.write(base64.b64decode(b))
p=subprocess.run(['sbatch','--parsable',str(r/'job.sbatch')],capture_output=True,text=True)
v=dict(actual_exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
with (r/'submission.json').open('x') as f:json.dump(v,f)
print(json.dumps(v));raise SystemExit(p.returncode)
'''
command=['wsl.exe','--exec','ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-S','/tmp/uncc.sock','dbayha@hpc.charlotte.edu','/usr/bin/python3 -']
p=subprocess.run(command,input=code.encode(),capture_output=True,timeout=45)
for n,b in [('benchmark_dispatch.stdout',p.stdout),('benchmark_dispatch.stderr',p.stderr),('benchmark_bootstrap.py',code.encode())]:
 with (W/n).open('xb') as f:f.write(b)
receipt=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_exit=p.returncode,remote=root,
             source_pins={n:hashlib.sha256(b).hexdigest() for n,b in files.items()},new_physics=False,gpu_requested=False)
with (W/'benchmark_dispatch_actual_exit.json').open('x') as f:json.dump(receipt,f,indent=2)
print(p.stdout.decode());print(p.stderr.decode());raise SystemExit(p.returncode)
