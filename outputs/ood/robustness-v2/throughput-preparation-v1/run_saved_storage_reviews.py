from pathlib import Path
import base64,datetime,hashlib,json,subprocess
W=Path(__file__).resolve().parent
source=W/'review_storage_saved.py';raw=source.read_bytes();compile(raw,str(source),'exec')
target='/mnt/c/'+str(source).replace('\\','/')[3:]
local=['wsl.exe','--exec','/usr/bin/python3','-B',target,'/home/dbayha/bca-work/storage-benchmark-20260929T075005Z']
remote_root='/users/dbayha/bca-ood-v2/storage-benchmark-20260929T075005Z'
code=f'''from pathlib import Path
import base64,subprocess,sys
r=Path({remote_root!r});p=r/'review_storage_saved.py'
with p.open('xb') as f:f.write(base64.b64decode({base64.b64encode(raw).decode()!r}))
q=subprocess.run(['/usr/bin/python3','-B',str(p),str(r/'shared')],capture_output=True)
sys.stdout.buffer.write(q.stdout);sys.stderr.buffer.write(q.stderr);raise SystemExit(q.returncode)
'''
remote=['wsl.exe','--exec','ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-S','/tmp/uncc.sock','dbayha@hpc.charlotte.edu','/usr/bin/python3 -']
for name,command,stdin in [('local',local,None),('cluster',remote,code.encode())]:
    p=subprocess.run(command,input=stdin,capture_output=True,timeout=45)
    for suffix,b in [('stdout',p.stdout),('stderr',p.stderr)]:
        with (W/(name+'_saved_review.'+suffix)).open('xb') as f:f.write(b)
    receipt=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_exit=p.returncode,source_sha256=hashlib.sha256(raw).hexdigest(),command=command)
    with (W/(name+'_saved_review_actual_exit.json')).open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt));print(p.stdout.decode());print(p.stderr.decode())
    assert p.returncode==0
