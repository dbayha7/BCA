from pathlib import Path
import datetime,json,subprocess
w=Path(__file__).resolve().parent
code='''from pathlib import Path
import json,subprocess
r=Path('/users/dbayha/bca-ood-v2/rebrac-hopper171-ood-v1')
out={}
for n in ('submission.json','worker-identity.json','actual_exit.json','results/status.json','results/failure.json','results/engineering.json','results/engineering-independent-review.json'):
 p=r/n
 if p.exists():out[n]=json.loads(p.read_bytes())
for n in ('worker.stdout','worker.stderr'):
 p=r/n
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4000));out[n]=f.read().decode(errors='replace')
p=subprocess.run(['sacct','-j','27067667','--format=JobID,State,ExitCode,NodeList','-n','-P'],capture_output=True,text=True)
out['sacct']=p.stdout
print(json.dumps(out))
'''
p=subprocess.run(['wsl.exe','--exec','ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-S','/tmp/uncc.sock','dbayha@hpc.charlotte.edu','/usr/bin/python3 -'],input=code.encode(),capture_output=True,timeout=40)
root=w/('cluster-observation-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
root.mkdir()
(root/'stdout.json').write_bytes(p.stdout);(root/'stderr.txt').write_bytes(p.stderr)
(root/'actual_exit.json').write_text(json.dumps(dict(actual_exit=p.returncode)))
print(p.stdout.decode());print(p.stderr.decode());raise SystemExit(p.returncode)
