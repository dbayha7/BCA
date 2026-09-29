"""Fetch once per completed CPU query; independently check only saved small arrays."""
from pathlib import Path
import base64,datetime,hashlib,io,json,subprocess
import numpy as np
W=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
submissions=json.loads((W/'next_query_submissions.json').read_bytes())
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
code='''from pathlib import Path
import base64,hashlib,json,subprocess
submissions=SUBMISSIONS
rows=[]
for s in submissions:
 r=Path(s['remote']);out=dict(pair=s['pair'],job_id=s['job_id'],files={},current_staged={})
 names=['submission.json','worker-identity.json','actual_exit.json','failure.json','query-acceptance.json','input-pins.json','pair-intent.json']
 complete=(r/'actual_exit.json').exists()
 if complete:
  names+=['host-action-parity.npz','bca-action-parity.npz','host-target-parity.npz','bca-target-parity.npz','bca-frozen-score.npz','worker.stdout','worker.stderr']
 for n in names:
  p=r/n
  if p.exists():
   assert p.stat().st_size<2000000,(n,p.stat().st_size)
   b=p.read_bytes();out['files'][n]=dict(sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode())
 pins=json.loads((r/'input-pins.json').read_bytes())
 for n in pins:
  out['current_staged'][n]=hashlib.sha256((r/n).read_bytes()).hexdigest()
 p=subprocess.run(['sacct','-j',s['job_id'],'--format=JobID,State,ExitCode,NodeList','-n','-P'],capture_output=True,text=True,timeout=15)
 out['scheduler']=dict(actual_exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
 rows.append(out)
print(json.dumps(rows))
'''.replace('SUBMISSIONS',repr(submissions))
command=['wsl.exe','--exec','ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-S','/tmp/uncc.sock','dbayha@hpc.charlotte.edu','/usr/bin/python3 -']
p=subprocess.run(command,input=code.encode(),capture_output=True,timeout=90)
(W/f'query_fetch_{stamp}.stdout').write_bytes(p.stdout)
(W/f'query_fetch_{stamp}.stderr').write_bytes(p.stderr)
(W/f'query_fetch_{stamp}_actual_exit.json').write_bytes(json.dumps(dict(utc=now(),actual_exit=p.returncode,
    source_sha256=sha(Path(__file__).read_bytes()),remote_code_sha256=sha(code.encode()),command=command),indent=2).encode())
assert p.returncode==0,p.stderr.decode()
rows=json.loads(p.stdout);summary=[]
for s,row in zip(submissions,rows):
    assert row['pair']==s['pair'] and row['job_id']==s['job_id']
    root=W/f"query-{s['pair'][1]}-{s['pair'][2]}"
    raw={n:base64.b64decode(v['data']) for n,v in row['files'].items()}
    assert all(sha(raw[n])==v['sha256'] for n,v in row['files'].items())
    pins=json.loads(raw['input-pins.json'])
    assert pins==json.loads((root/'staged/input-pins.json').read_bytes())==row['current_staged']
    assert raw['pair-intent.json']==(root/'staged/pair-intent.json').read_bytes()
    if 'actual_exit.json' not in raw:
        summary.append(dict(pair=s['pair'],job_id=s['job_id'],actual_exit=None,scheduler=row['scheduler']));continue
    exit_record=json.loads(raw['actual_exit.json'])
    dest=root/'saved-query'
    if dest.exists():
        # Immutable evidence already fetched is compared, never overwritten/reviewed again.
        assert all((dest/n).read_bytes()==b for n,b in raw.items())
        summary.append(json.loads((dest/'independent-review.json').read_bytes()));continue
    dest.mkdir()
    for n,b in raw.items():(dest/n).write_bytes(b)
    assert exit_record['actual_exit']==0,exit_record
    assert 'failure.json' not in raw
    identity=json.loads(raw['worker-identity.json'])
    assert identity['job_id']==s['job_id'] and identity['pid']==exit_record['pid']
    assert identity['command']==exit_record['command'] and identity['command'][-1]==s['remote']+'/cluster_query_gate.py'
    q=json.loads(raw['query-acceptance.json']);intent=json.loads(raw['pair-intent.json'])
    assert q['query_gate_passed'] and q['pair']==s['pair'] and q['backend']=='cpu'
    assert q['simulator_steps']==q['training_updates']==0 and q['simulator_gate_pending'] and not q['ood_outcomes_complete']
    checks=[]
    for method in ('host','bca'):
        r,=[x for x in q['receipts'] if x['method']==method]
        assert r['run_id']==intent['methods'][method]['run_id'] and r['checkpoint_sha256']==intent['methods'][method]['checkpoint_sha256']
        with np.load(io.BytesIO(raw[method+'-action-parity.npz']),allow_pickle=False) as a:
            size=3 if s['pair'][1]=='hopper' else 6
            assert a['actual'].shape==a['reference'].shape==(16,size)
            err=np.abs(a['actual'].astype(np.float64)-a['reference'].astype(np.float64))
            assert np.isfinite(err).all() and np.array_equal(err,a['error']) and (err<=1e-6).all()
            maxerr=float(err.max());assert maxerr==r['action_max_error']
        with np.load(io.BytesIO(raw[method+'-target-parity.npz']),allow_pickle=False) as a:
            assert a['target'].shape==a['independent'].shape and a['target'].size==16
            assert np.isfinite(a['target']).all() and np.array_equal(a['target'],a['independent'])
            assert a['key'].dtype==np.uint32 and a['key'].shape==(2,)
        checks.append(dict(method=method,action_max_error=maxerr,target_exact=True,checkpoint_sha256=r['checkpoint_sha256']))
    with np.load(io.BytesIO(raw['bca-frozen-score.npz']),allow_pickle=False) as a:
        assert np.isfinite(a['width']).all() and (a['width']>=0).all() and a['usable'].all()
        assert np.array_equal(a['width'],a['direct_width']) and np.array_equal(a['dose'],a['direct_dose'])
    scheduler=row['scheduler'];assert scheduler['actual_exit']==0
    assert s['job_id']+'|COMPLETED|0:0|' in scheduler['stdout'],scheduler
    result=dict(utc=now(),pair=s['pair'],job_id=s['job_id'],actual_exit=0,accepted_saved_query_arrays=True,
        checks=checks,width_dose_saved_direct_exact=True,staged_sources_unchanged=True,
        files={n:v['sha256'] for n,v in row['files'].items()},scheduler=scheduler,
        simulator_steps=0,new_ood_keys=0,physics_dispatch_accepted=False,ood_results=False,
        scope='CPU checkpoint query only; live simulator/support/precommit/storage/allocation gates remain')
    (dest/'independent-review.json').write_bytes(json.dumps(result,indent=2).encode())
    summary.append(result)
(W/f'query_summary_{stamp}.json').write_bytes(json.dumps(summary,indent=2).encode())
print(json.dumps([dict(pair=r['pair'],job_id=r['job_id'],actual_exit=r['actual_exit'],accepted=r.get('accepted_saved_query_arrays',False)) for r in summary]))
