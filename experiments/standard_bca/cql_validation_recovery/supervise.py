from pathlib import Path
import hashlib,json,os,subprocess,sys
root=Path('/users/dbayha/bca-standard-noiw-v1/cql-validation-recovery-v1');source=root/'source';py='/users/dbayha/bca-standard-noiw-v1/venv-v2/bin/python'
sys.path.insert(0,str(source))
from experiments.standard_runner import run_child,save
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
frozen=json.loads((root/'execution_freeze.json').read_text())
assert all(sha(root/n)==h for n,h in frozen['files'].items())
manifest=json.loads((source/'manifest.json').read_text())
assert all(sha(source/n)==h for n,h in manifest['source_sha256'].items())
assert json.loads((root/'ownership.json').read_text())['local_launch_disabled']
assert os.environ['SLURM_JOB_ID']
cpu=dict(os.environ,JAX_PLATFORMS='cpu');gpu=dict(os.environ,JAX_PLATFORMS='cuda')
def check(name,cmd,env,timeout=7200):
    code=run_child(cmd,root/name,timeout,env=env)
    if code:raise SystemExit(code)
check('data-check-process-v1',[py,str(root/'cluster_data_check.py')],cpu)
assert json.loads((root/'cluster_data_acceptance.json').read_text())['accepted']
for host in ('cql','td3_bc'):
    for method in ('host','bca'):
        check(host+'-'+method+'-gpu-fixture-v1',[py,str(root/'probe_gpu.py'),host,'new',method],gpu)
        result=json.loads((root/(host+'_new_'+method+'.json')).read_text())
        assert result['decoded_checkpoint_checked'] and result['simulator_steps']==0
save(root/'compute_acceptance.json',dict(accepted=True,science_retries=0,fixture_updates=12,fixture_simulator_steps=0,data_cells=14))
p=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv'],capture_output=True,text=True,check=True)
(root/'training_gpu_identity.txt').write_text(p.stdout)
cmd=[py,str(source/'experiments/standard_runner.py'),'--manifest',str(source/'manifest.json'),'--sha256','9ecf480c560871832ec6384045e5646de3238de8697bb56d6ee9dec85172b3cf','--lane','local','--output',str(root/'queue-v1'),'--gpu-lock',str(root/'resource-locks/allocated-gpu.lock'),'--data-dir','/users/dbayha/.d4rl/datasets']
raise SystemExit(run_child(cmd,root/'controller-process-v1',30*86400,env=gpu))
