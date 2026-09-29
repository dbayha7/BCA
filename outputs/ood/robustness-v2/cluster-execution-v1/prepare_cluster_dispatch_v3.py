from pathlib import Path
import ast
w=Path(__file__).resolve().parent
s=(w/'cluster_run.py').read_text()
assert s.count('from ancestor_guard import ExclusiveLease')==1
s=s.replace('from ancestor_guard import ExclusiveLease','from cluster_lease import ClusterLease as ExclusiveLease')
with (w/'cluster_run_v2.py').open('x') as f:f.write(s)
d=(w/'dispatch_cluster_ood_v2.py').read_text()
d=d.replace('rebrac-hopper171-ood-v1','rebrac-hopper171-ood-v2')
d=d.replace("'cluster-ood-v2-","'cluster-ood-v3-")
needle="for n in ('ancestor_guard','extension_ledger'"
assert d.count(needle)==1
d=d.replace(needle,"files['cluster_run.py']=(W/'cluster_run_v2.py').read_bytes()\nfor extra in ('cluster_lease.py','cluster_lock_check.py','CLUSTER_EXECUTION_CORRECTION.md'):files[extra]=(W/extra).read_bytes()\n"+needle)
needle="data=json.loads(gzip.decompress(base64.b64decode({payload!r})))"
assert d.count(needle)==1
d=d.replace(needle,"prior=root.parent/'rebrac-hopper171-ood-v1'\nassert json.loads((prior/'actual_exit.json').read_bytes())['actual_exit']==1\nassert not (prior/'results/extension-declaration.json').exists(), 'Unexpected prior resource reservation'\n"+needle)
needle="p=subprocess.run(['sbatch','--parsable',str(root/'job.sbatch')],capture_output=True,text=True)"
assert d.count(needle)==1
d=d.replace(needle,"check=subprocess.run(['/usr/bin/python3',str(root/'cluster_lock_check.py')],capture_output=True,text=True)\nwith (root/'lock-check-process.json').open('x') as f:json.dump(dict(actual_exit=check.returncode,stdout=check.stdout,stderr=check.stderr),f,indent=2)\nassert check.returncode==0,check.stderr\n"+needle)
ast.parse(d)
with (w/'dispatch_cluster_ood_v3.py').open('x') as f:f.write(d)
