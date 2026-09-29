"""Launch the reviewed unstarted lane exactly once, under its original GPU lock."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,subprocess,sys
r=Path('/home/dbayha/bca-work/standard-noiw-v1/recovery-validation-v2')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pre=read(r/'preflight.json');review=read(r/'independent-review.json');checks=read(r/'checks_v2.json')
assert pre['accepted'] and review['accepted'] and checks['accepted']
assert sha(r/'source/manifest.json')==pre['manifest_sha256']==review['manifest_sha256']
assert sha(r/'supervise.py')==pre['supervisor_sha256']
manifest=read(r/'source/manifest.json')
assert all(sha(r/'source'/n)==h for n,h in manifest['source_sha256'].items())
assert not (r/'controller-process-local-v2').exists() and not (r/'queue-local-v2').exists()
lock=Path('/home/dbayha/bca-work/resource-locks/local-rtx5070ti.lock')
with lock.open('a') as f:
    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    fcntl.flock(f,fcntl.LOCK_UN)
# Exclusive launch claim precedes Popen. A failed dispatch remains an explicit attempt.
with (r/'launch_claim.json').open('x') as f:
    json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_sha256':pre['manifest_sha256'],
               'review_sha256':sha(r/'independent-review.json'),'checks_sha256':sha(r/'checks_v2.json')},f)
    f.flush();os.fsync(f.fileno())
with (r/'supervisor.log').open('x') as log:
    p=subprocess.Popen([sys.executable,str(r/'supervise.py')],cwd=r,stdin=subprocess.DEVNULL,
                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':p.pid,
      'command':[sys.executable,str(r/'supervise.py')],'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
      'proc_stat':Path(f'/proc/{p.pid}/stat').read_text(),'manifest_sha256':pre['manifest_sha256'],
      'actual_controller_exit':None,'new_training_rows':137,'first_run':manifest['runs'][0]['row']['run_id'],
      'prior_runs_retrained':False}
    with (r/'dispatch.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(receipt,indent=2))
