"""Build a separate cluster attempt by reusing the running path's unchanged loop."""
from pathlib import Path
import ast,textwrap
W=Path(__file__).resolve().parent
query=(W/'cluster_query_gate.py').read_text()
head=query[:query.index('def main():')]
body=query[query.index('    assert os.environ'):query.index('        # Fixed first 16 training rows:')]
body=body.replace('    receipts=[]','    adapters={};preparations={}')
body+='        adapter.assert_unchanged()\n        adapters[method]=adapter;preparations[method]=prepared\n'
body+='    return adapters,preparations\n'
loader=head+'def load_pair():\n'+body
worker='''"""One accepted ReBRAC pair, separate CPU allocation; no replay or training."""
from pathlib import Path
import datetime,hashlib,json,os,sys,time,traceback
HERE=Path(__file__).resolve().parent
SOURCE=Path('/users/dbayha/bca-standard-noiw-v1/campaign/source')
ROOT=HERE/'results'
sys.path[:0]=[str(HERE/'core'),str(HERE),str(SOURCE)]
PAIR=('rebrac','hopper',202609171)
LOCK=HERE.parent/'cluster-ood.lock'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def put(p,v):
    raw=(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\\n').encode()
    with Path(p).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return hashlib.sha256(raw).hexdigest()
def status(attempt,stage,**values):
    v=dict(utc=now(),pid=os.getpid(),pair=list(PAIR),stage=stage,scientific_complete=False,**values)
    tmp=attempt/'status.tmp'
    with tmp.open('w') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
    tmp.replace(attempt/'status.json');print(json.dumps(v),flush=True)

class ClosedAncestor:
    """Attested closed local ledger; no claim of live remote filesystem access.

    The active local worker checks the original ledger on every physical call.
    Remote work is additionally confined to this single disjoint ReBRAC pair.
    Both hosts' full allocations sum to the original 20-pair v2 ceiling.
    """
    def __init__(self):
        self.path=Path('/home/dbayha/bca-work/ood-resources-v2/ledger.sqlite')
        self.binding=json.loads((HERE/'ancestor-binding.json').read_bytes())
        self.global_cap=[39998400,159993600]
        self.pins=json.loads((HERE/'input-pins.json').read_bytes())
        self.unchanged()
    def unchanged(self):
        for n in ('ancestor-binding.json','resource-partition.json','local-ancestor-attestation.json'):
            assert sha(HERE/n)==self.pins[n],('Changed resource attestation',n)

def run():
    import numpy as np
    from ancestor_guard import ExclusiveLease
    from extension_ledger import ExtensionLedger,declaration
    from recorded_step import EncodedStore,RecordedStep,content_hash,require
    from native_constructor import construct_recorded
    from collector_driver_amended_v1 import collect_states
    from outcome_driver_amended_v1 import run_panel
    from precommit_bank_amended_v1 import capture_schedule,build_precommit,PIN_NAMES
    from precommit_file import canonical
    from experiments.ood.collect import read_artifact,write_artifact,support_bank
    from experiments.ood.production import paired_reset
    from experiments.ood.streaming import ArtifactArchive,ArchiveDirectory,FrozenPolicy
    from cluster_loader import load_pair
    attempt=ROOT;attempt.mkdir(exist_ok=False)
    status(attempt,'startup')
    pins0=json.loads((HERE/'input-pins.json').read_bytes())
    sources={str(HERE/n):h for n,h in pins0.items()}
    require(all(sha(p)==h for p,h in sources.items()),'Changed staged inputs')
    query=json.loads((HERE/'query-acceptance.json').read_bytes())
    review0=json.loads((HERE/'query-independent-review.json').read_bytes())
    require(query['query_gate_passed'] and query['pair']==list(PAIR)
        and review0['accepted_saved_query_arrays'] and review0['actual_exit']==0,'Unaccepted cluster query gate')
    full=read_artifact(HERE/'full-declaration.json',pins0['full-declaration.json'])
    old_pair,=[p for p in full['pairs'] if p['pair_id']=='rebrac/hopper/s202609171']
    partition=json.loads((HERE/'resource-partition.json').read_bytes())
    require(partition['cluster_current_pair']==list(PAIR) and partition['local_owner']=='td3_bc'
        and partition['cluster_owner']=='rebrac','Overlapping pair allocation')
    review=json.loads((HERE/'saved_key_review.json').read_bytes())
    record,=[r for r in review['records'] if r['kind']=='v2' and r['pair']==list(PAIR)]
    keypath=HERE/'continuation.u32'
    require(sha(keypath)==record['sha256'] and review['all_key_values_arithmetic_equal']
        and review['old_v2_overlaps']==0 and review['producer_actual_exit']==1,'Unaccepted numeric key bytes')
    keys=np.frombuffer(keypath.read_bytes(),dtype='<u4').reshape(256,250,2).copy()
    key_receipt=review['amended_precommit_bindings'][record['index']]
    execution=dict(schema='ood-cluster-execution-v2',pair=list(PAIR),sources=sources,
        scientific_plan_sha256=pins0['scientific-plan.md'],resource_partition=partition,
        query_gate_sha256=pins0['query-acceptance.json'],action_absolute_tolerance=1e-6,
        reward_absolute_tolerance=1e-7,full_state_tolerance=0,no_training=True,no_automatic_retry=True,
        remote_ancestor_is_attestation=True,scientific_acceptance=False,
        captures=[100,300],reset_blocks=64,horizon=250,max_slots=14)
    declaration_hash=put(attempt/'declaration.json',execution)
    sim=None
    with ExclusiveLease(LOCK) as lease:
        guard=ClosedAncestor()
        ledgerpath=HERE.parent/'rebrac-hopper171-extension.sqlite'
        spec=declaration(ledgerpath,LOCK,guard.binding,declaration_hash)
        for name in spec['caps']:
            keep=name=='global' or name=='engineering/rebrac/hopper' or name.endswith('/rebrac/hopper/s202609171')
            if not keep:spec['caps'][name]=[0,0]
        spec['caps']['global']=[1847568,7390272]
        require(guard.binding['reserved']['global'][0]+2*18395680<=guard.global_cap[0], 'Combined allocation expanded')
        put(attempt/'extension-declaration.json',spec)
        with ExtensionLedger.create(ledgerpath,guard,lease,spec) as ledger:
            try:
                models,prepared=load_pair()
                import jax,jax.numpy as jnp,importlib.metadata
                from gym.envs.mujoco.hopper import HopperEnv
                from gym.envs.mujoco.mujoco_env import MujocoEnv
                from experiments.ood.simulator import LocomotionAdapter
                require(jax.default_backend()=='cpu','CPU required')
                packages={p:importlib.metadata.version(p) for p in ('jax','jaxlib','flax','numpy','h5py','gym','mujoco-py','d4rl')}
                require(packages==json.loads((HERE/'runtime-versions.json').read_bytes()),'Numerical runtime changed')
                runtime_hash=put(attempt/'runtime.json',dict(packages=packages,backend=jax.default_backend()))
                for k in ('training','heldout','training_ids','heldout_ids','obs_mean','obs_std','max_action','max_episode_steps'):
                    require(prepared['host'].metadata['run_input_hashes'][k]==prepared['bca'].metadata['run_input_hashes'][k], 'Unpaired input '+k)
                data=prepared['host'];ids=np.asarray(data.training_ids)
                require(np.array_equal(ids,data.metadata['training_converted_ids']), 'Training ID mismatch')
                episodes=np.asarray(data.metadata['original_terminal_timeout_episode_ids'],np.int64)[ids]
                obs=np.asarray(data.training.obs);actions=np.asarray(data.training.action)
                support=support_bank(obs,actions,ids,episodes,seed=int(old_pair['streams']['preparation'][0]))
                vi=np.asarray(support['validation_indices'],np.int64)
                support.update(validation_observations=obs[vi].copy(),validation_actions=actions[vi].copy(),
                    accepted_input_hashes=data.metadata['run_input_hashes'],normalization='Exact ReBRAC raw observations; fixed calibrator statistics remain separate')
                support_path=attempt/'support-bank.json';write_artifact(support_path,support)
                sd=[float(np.sqrt(np.square(support['observations'].astype(np.float64)-x).mean(axis=1).min())) for x in obs[vi]]
                q95,q99=np.quantile(support['validation_distances'],[.95,.99],method='linear')
                feasibility=dict(action_q95=float(q95),action_q99=float(q99),state_q95=float(np.quantile(sd,.95,method='linear')),
                    training_only=True,reference_rows=len(support['observations']),calibration_rows=len(vi),support_sha256=sha(support_path))
                require(0<q95<q99 and feasibility['state_q95']>0 and not set(support['reference_episodes'])&set(support['validation_episodes']),'Degenerate or leaked support split')
                put(attempt/'support-thresholds.json',feasibility)
'''
local=(W/'run.py').read_text()
tail=local[local.index('                policies={k:FrozenPolicy'):local.index("\nif __name__=='__main__':")]
tail=tail.replace("full['engineering_streams']['td3_bc/hopper']","full['engineering_streams']['rebrac/hopper']")
tail=tail.replace("training_acceptance=old_precommit['binding_sha256']","training_acceptance=pins0['pair-acceptance.json']")
tail=tail.replace("binding['checkpoints']['host']['checkpoint_sha256']","models['host']._file_sha")
tail=tail.replace("binding['checkpoints']['bca']['checkpoint_sha256']","models['bca']._file_sha")
tail=tail.replace("binding['checkpoints']['host']['preparation_input_hashes']","data.metadata['run_input_hashes']")
tail=tail.replace("sha(KEYROOT/'stream_input_review.json')","pins0['stream_input_review.json']")
# Run the independently written saved-array engineering reviewer before collection.
needle="                    status(attempt,'collection',transitions=sim.transitions)"
assert tail.count(needle)==1
tail=tail.replace(needle,"                    import cluster_review_engineering\n                    cluster_review_engineering.main()\n"+needle)
worker+=tail+"\nif __name__=='__main__':run()\n"
review=(W/'review_engineering.py').read_text()
review=review.replace("ROOT=Path('/home/dbayha/bca-work/ood-live-v2')\nATTEMPT=ROOT/'td3-hopper-s202609171-v1'","ROOT=Path(__file__).resolve().parent\nATTEMPT=ROOT/'results'")
for name,code in [('cluster_loader.py',loader),('cluster_run.py',worker),('cluster_review_engineering.py',review)]:
    ast.parse(code)
    with (W/name).open('x') as f:f.write(code)
print('Created separate loader, worker and engineering reviewer; no execution or science.')
