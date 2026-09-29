from pathlib import Path
import base64,datetime,hashlib,io,json,subprocess
import numpy as np
W=Path(__file__).resolve().parent
dest=W/'cluster-query-accepted-v1'
code='''from pathlib import Path
import base64,hashlib,json,subprocess
r=Path('/users/dbayha/bca-ood-v2/rebrac-hopper171-query-v1')
names=['actual_exit.json','query-acceptance.json','host-action-parity.npz','bca-action-parity.npz','host-target-parity.npz','bca-target-parity.npz','bca-frozen-score.npz','worker-identity.json','input-pins.json']
assert json.loads((r/'actual_exit.json').read_bytes())['actual_exit']==0
out={n:dict(sha256=hashlib.sha256((r/n).read_bytes()).hexdigest(),data=base64.b64encode((r/n).read_bytes()).decode()) for n in names}
p=subprocess.run(['sacct','-j','27067658','--format=JobID,State,ExitCode','-n','-P'],capture_output=True,text=True)
print(json.dumps(dict(files=out,sacct=p.stdout,sacct_exit=p.returncode)))
'''
p=subprocess.run(['wsl.exe','--exec','ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-S','/tmp/uncc.sock','dbayha@hpc.charlotte.edu','/usr/bin/python3 -'],input=code.encode(),capture_output=True,timeout=45)
assert p.returncode==0,p.stderr.decode()
v=json.loads(p.stdout);dest.mkdir(exist_ok=False)
checks=[]
for n,r in v['files'].items():
    b=base64.b64decode(r['data']);assert hashlib.sha256(b).hexdigest()==r['sha256']
    with (dest/n).open('xb') as f:f.write(b)
    if n.endswith('.npz'):
        a=np.load(io.BytesIO(b),allow_pickle=False)
        if 'action-parity' in n:
            err=np.abs(a['actual'].astype(np.float64)-a['reference'].astype(np.float64))
            assert np.array_equal(err,a['error']) and (err<=1e-6).all()
            checks.append(dict(file=n,max_error=float(err.max())))
        if 'target-parity' in n:
            assert np.array_equal(a['target'],a['independent'])
            checks.append(dict(file=n,exact=True))
        if 'frozen-score' in n:
            assert np.isfinite(a['width']).all() and (a['width']>=0).all() and a['usable'].all()
            checks.append(dict(file=n,finite_usable=True,width_min=float(a['width'].min()),width_max=float(a['width'].max())))
assert v['sacct_exit']==0 and '27067658|COMPLETED|0:0' in v['sacct']
receipt=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_saved_query_arrays=True,
 actual_exit=0,sacct=v['sacct'],files={n:r['sha256'] for n,r in v['files'].items()},checks=checks,
 simulator_steps=0,ood_results=False)
with (dest/'independent-review.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
