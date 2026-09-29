"""Small immutable receipts and live process identity, without replay or weights."""
from pathlib import Path
import datetime,hashlib,json,subprocess
w=Path(__file__).resolve().parent
code='''from pathlib import Path
import datetime,hashlib,json,subprocess
r=Path('/users/dbayha/bca-ood-v2/rebrac-hopper171-ood-v2')
names=['submission.json','worker-identity.json','lock-check.json','lock-check-process.json','results/status.json','results/declaration.json','results/extension-declaration.json','results/engineering.json','results/engineering-independent-review.json','results/support-thresholds.json','results/runtime.json','input-pins.json']
out={};pins={}
for n in names:
 b=(r/n).read_bytes();out[n]=json.loads(b);pins[n]=hashlib.sha256(b).hexdigest()
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in out['input-pins.json'].items())
assert out['results/engineering-independent-review.json']['accepted_live_engineering']
identity=out['worker-identity.json'];pid=identity['pid']
probe="from pathlib import Path;import json;p=Path('/proc/%d');print(json.dumps(dict(stat=p.joinpath('stat').read_text(),command=p.joinpath('cmdline').read_bytes().decode().strip(chr(0)).split(chr(0)))))" % pid
p=subprocess.run(['srun','--jobid=27067675','--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1','--gres=none','--nodelist=str-c130','/usr/bin/python3','-c',probe],capture_output=True,text=True,timeout=25)
assert p.returncode==0,p.stderr
actual=json.loads(p.stdout)
assert actual['command']==identity['command']
assert actual['stat'].rsplit(')',1)[1].split()[19]==identity['proc_stat'].rsplit(')',1)[1].split()[19]
out['live-process-check']=dict(command_and_start_match=True,observed=actual,actual_exit=p.returncode)
old=r.parent/'rebrac-hopper171-ood-v1'
out['preserved-worker-v1-failure']=dict(actual_exit=json.loads((old/'actual_exit.json').read_bytes()),stderr=(old/'worker.stderr').read_text(),no_extension_declaration=not(old/'results/extension-declaration.json').exists())
out['job_actual_exit']=json.loads((r/'actual_exit.json').read_bytes()) if (r/'actual_exit.json').exists() else None
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=out,sha256=pins,all_staged_inputs_unchanged=True)))
'''
p=subprocess.run(['wsl.exe','--exec','ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-S','/tmp/uncc.sock','dbayha@hpc.charlotte.edu','/usr/bin/python3 -'],input=code.encode(),capture_output=True,timeout=45)
d=w/'cluster-start-accepted-v1';d.mkdir()
(d/'stdout.json').write_bytes(p.stdout);(d/'stderr.txt').write_bytes(p.stderr)
(d/'actual_exit.json').write_text(json.dumps(dict(actual_exit=p.returncode)))
assert p.returncode==0,p.stderr.decode()
v=json.loads(p.stdout)
print(json.dumps(dict(utc=v['utc'],status=v['files']['results/status.json'],live_identity=v['files']['live-process-check']['command_and_start_match'],engineering=v['files']['results/engineering-independent-review.json']['accepted_live_engineering'],job_actual_exit=v['files']['job_actual_exit'],inputs_unchanged=v['all_staged_inputs_unchanged'])))
