from pathlib import Path
import datetime,hashlib,json,subprocess,sys
W=Path(__file__).resolve().parent
source=W/'storage_benchmark.py';raw=source.read_bytes();compile(raw,str(source),'exec')
target='/mnt/c/'+str(source).replace('\\','/')[3:]
command=['wsl.exe','--cd','/tmp','--exec','/usr/bin/python3','-B',target,
         '--root','/home/dbayha/bca-work/storage-benchmark-20260929T075005Z',
         '--node','/tmp/bca-storage-benchmark-20260929T075005Z-local']
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (W/'storage_benchmark.executed.py').open('xb') as f:f.write(raw)
p=subprocess.run(command,capture_output=True,timeout=300)
for n,b in [('local_benchmark.stdout',p.stdout),('local_benchmark.stderr',p.stderr)]:
    with (W/n).open('xb') as f:f.write(b)
receipt=dict(started=started,ended=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_exit=p.returncode,
             command=command,source_sha256=hashlib.sha256(raw).hexdigest(),new_physics=False)
with (W/'local_benchmark_actual_exit.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
if p.returncode==0:
    v=json.loads(p.stdout);print(json.dumps({k:v[k] for k in ('median_calls_per_second','ratio_to_sqlite','retained_shared_bytes')}))
else:print(p.stderr.decode(errors='replace'))
sys.exit(p.returncode)
