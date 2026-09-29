"""Actual cross-process exclusion/release test; zero scientific imports/calls."""
from pathlib import Path
import json,os,subprocess,sys
r=Path(__file__).resolve().parent
sys.path[:0]=[str(r/'core'),str(r)]
from cluster_lease import ClusterLease
from ancestor_guard import identity
lock=r.parent/'cluster-ood.lock'
if len(sys.argv)>1:
    try:
        with ClusterLease(lock):print('acquired')
    except ValueError as e:
        if str(e)=='Shared resource lock already held.':print('blocked');raise SystemExit(3)
        raise
else:
    before=identity(lock)
    with ClusterLease(lock) as lease:
        p=subprocess.run([sys.executable,str(Path(__file__)),'child'],capture_output=True,text=True)
        assert p.returncode==3 and p.stdout.strip()=='blocked',(p.returncode,p.stdout,p.stderr)
        lease.assert_held()
    q=subprocess.run([sys.executable,str(Path(__file__)),'child'],capture_output=True,text=True)
    assert q.returncode==0 and q.stdout.strip()=='acquired',(q.returncode,q.stdout,q.stderr)
    assert identity(lock)==before
    with (r/'lock-check.json').open('x') as f:json.dump(dict(exclusive_child_refused=True,post_release_child_acquired=True,
        lock_identity_unchanged=True,model_queries=0,simulator_steps=0),f,indent=2)
