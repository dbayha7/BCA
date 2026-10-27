from pathlib import Path
import sys, json
root=Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
import study
m=study.read_manifest(root/'validation/smoke_20261026_v1')
new=study.source_identity()
print('Benchmark and upstream:',[(k,m['sources'][k],new[k]) for k in ('benchmark_source_sha256','unified_upstream_sha256') if m['sources'][k]!=new[k]])
old=m['sources']['external_sources']; now=new['external_sources']
for k in sorted(set(old)|set(now)):
 if old.get(k)!=now.get(k): print('Changed:',k,'old=',old.get(k),'new=',now.get(k))
