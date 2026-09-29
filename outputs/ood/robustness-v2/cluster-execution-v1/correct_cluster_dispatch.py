"""Preserve v1's pre-dispatch string-quoting failure; no worker was submitted."""
from pathlib import Path
import ast
w=Path(__file__).resolve().parent
s=(w/'dispatch_cluster_ood.py').read_text()
old="assert 'PAIR = (\\'td3_bc\\', \\'hopper\\', 202609171)' in source.read_text()"
assert old in s
s=s.replace(old,"import ast\ntree=ast.parse(source.read_text())\npair,=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAIR' for t in n.targets)]\nassert pair==('td3_bc','hopper',202609171)")
s=s.replace("'cluster-ood-","'cluster-ood-v2-")
ast.parse(s)
with (w/'dispatch_cluster_ood_v2.py').open('x') as f:f.write(s)
