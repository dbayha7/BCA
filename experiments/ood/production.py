"""First-pair CPU state collection, separately gated from action outcomes/coverage.

No learner or recovery entrypoint. Uses the closed training/query/state gates and
requires a NEW accepted storage/query-path engineering receipt before collection.
Runs once with the shared local worker lock, original cumulative resource ledger,
unchanged simulator.step, and immutable output directories. No resume/retry path.
"""
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.ood.collect import file_hash,read_artifact,write_artifact,sha,require
from experiments.ood.real_connection import _load_pair,linux_path
from experiments.ood.resources import ResourceLedger,scope_caps
from experiments.ood.streaming import ArtifactArchive,ArchiveDirectory,FrozenPolicy,collect_bank

LEDGER=Path('/home/dbayha/bca-work/ood-resources-v2/ledger.sqlite')
LOCK=Path('/home/dbayha/bca-work/resource-locks/local-rtx5070ti.lock')
SOURCES=['experiments/ood/production.py','experiments/ood/streaming.py',
         'experiments/ood/simulator.py','experiments/ood/adapters.py','experiments/ood/resources.py',
         'experiments/ood/collect.py','experiments/ood/real_connection.py','runtime/environment.py']
PAIR='td3_bc/hopper/s202609171'


def inputs(repo):
    accepted=repo/'runs/ood/td3-hopper-s202609171-connection-v2'
    closed=json.loads((repo/'docs/validation/ood-trained-td3-hopper-connection.json').read_text())
    relative='outputs/ood/td3_bc/hopper/s202609171/trained-connection-v2/precommit.json'
    pre=read_artifact(accepted/'trained-connection-v1/precommit.json',closed['artifacts_sha256'][relative])
    require(closed['trained_cpu_query_gate_accepted'] is True,'Trained gate is not accepted.')
    binding=read_artifact(accepted/'training-binding.json',pre['binding_sha256'])
    full=read_artifact(accepted/'full-declaration.json',pre['declaration_sha256'])
    require(binding['pair_id']==PAIR and binding['training_accepted'] is True,'Wrong pair.')
    from experiments.ood.protocol import digest,load_config
    require(digest({k:v for k,v in full.items() if k!='manifest_sha256'})==full['manifest_sha256'],
            'Changed original full declaration.')
    require(full['config']==load_config(),'Changed scientific settings.')
    for rel,expected in binding['frozen_source_sha256'].items():
        require(file_hash(repo/rel)==expected,'Frozen training source changed: '+rel)
    for item in binding['inputs'].values():
        require(file_hash(linux_path(item['path']))==item['sha256'],'Changed training acceptance input.')
    for rel,expected in pre['source_sha256'].items():
        require(file_hash(repo/rel)==expected,'Closed query implementation changed.')
    real=json.loads((repo/'docs/validation/ood-real-state-connection.json').read_text())
    require(real['accepted_engineering'] is True and real['actual_worker_exit']['actual_returncode']==0
            and real['supervisor_actual_exit']==0,'Real state gate is not accepted.')
    for rel,expected in real['source_sha256'].items():
        require(file_hash(repo/rel)==expected,'Closed real-state implementation changed.')
    require(LEDGER.is_file(),'Missing original cumulative resource ledger; do not recreate.')
    pair,=[p for p in full['pairs'] if p['pair_id']==PAIR]
    return binding,full,pair,pre


@contextmanager
def worker_lease(path=LOCK):
    import fcntl
    require(path==LOCK,'Wrong shared local worker lock.')
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def construct(ledger,attempt,maximum):
    from experiments.ood.simulator import LocomotionAdapter
    from gym.envs.mujoco.mujoco_env import MujocoEnv
    observed=[];original=MujocoEnv.do_simulation
    def observe(env,control,frames):
        require(not observed and frames==4,'Unexpected constructor physics; stop before extra step.')
        record=dict(action=np.asarray(control).copy(),frames=int(frames))
        write_artifact(attempt/'constructor-before-physics.json',record);observed.append(record)
        return original(env,control,frames)
    def call():
        with patch.object(MujocoEnv,'do_simulation',observe):
            return LocomotionAdapter('hopper-medium-v2',max_transitions=maximum,evidence_dir=attempt/'reference-npz')
    sim=ledger.call(str(attempt)+'/constructor',['engineering/td3_bc/hopper'],1,4,call)
    try:
        require(len(observed)==1,'Unobserved constructor call.')
        bounds=dict(normalized_low=sim.wrapper.action_space.low.copy(),normalized_high=sim.wrapper.action_space.high.copy(),
                    native_low=sim.base.action_space.low.copy(),native_high=sim.base.action_space.high.copy(),
                    actuator_ctrlrange=sim.sim.model.actuator_ctrlrange.copy(),dt=float(sim.base.dt),identity=sim.identity)
        write_artifact(attempt/'live-native-bounds.json',bounds)
        require(np.array_equal(bounds['native_low'],-np.ones(sim.action_dim))
                and np.array_equal(bounds['native_high'],np.ones(sim.action_dim)),'Unexpected live action bounds.')
        write_artifact(attempt/'constructor-completed.json',dict(environment=1,physics=4,observed=observed))
        return sim,bounds
    except BaseException:
        sim.close();raise


def transform(action,bounds):
    lo,hi=bounds['native_low'],bounds['native_high']
    return np.clip(lo+(np.asarray(action)+1)*.5*(hi-lo),lo,hi)


def paired_reset(sim,seed):
    # These action RNGs are captured but never sampled after reset. Both are
    # initialized from the paired reset seed, not fresh undeclared randomness.
    sim.wrapper.action_space.seed(seed);sim.base.action_space.seed(seed)
    return sim.reset(seed)


def reserved_step(sim,ledger,attempt,scope,owner,action,bounds):
    rec=ledger.call(str(attempt)+'/'+owner,[scope],1,4,lambda:sim.step(action))
    expected=transform(action,bounds)
    # Original step has already durably archived all actual inputs/applied/outputs.
    require(expected.dtype==rec['applied_action'].dtype and np.array_equal(expected,rec['applied_action']),
            'Independent live-bound wrapper transform differs.')
    return rec


def check_gate(repo,gate,worker_exit):
    exit_record=json.loads(worker_exit.read_text())
    require(exit_record['actual_returncode']==0 and exit_record['timeout'] is False
            and exit_record['interruption_signal'] is None,'Storage/query gate has no successful actual exit.')
    result=read_artifact(gate/'acceptance.json',file_hash(gate/'acceptance.json'))
    independent=json.loads((gate/'independent-check.json').read_text())
    require(independent['accepted'] is True
            and independent['gate_acceptance_sha256']==file_hash(gate/'acceptance.json')
            and independent['worker_actual_exit_sha256']==file_hash(worker_exit)
            and independent['archive_sha256']==file_hash(gate/'transitions.sqlite'),
            'Missing or changed independent storage/query validation.')
    declaration=read_artifact(gate/'declaration.json',result['declaration_sha256'])
    require(result['accepted_engineering'] is True and result['pair_id']==PAIR
            and result['environment_transitions']==7 and result['all_action_errors_zero'] is True
            and result['all_transition_records_exact'] is True,'Incomplete storage/query gate.')
    for rel,expected in declaration['source_sha256'].items():
        require(file_hash(repo/rel)==expected,'Changed accepted production execution source.')
    # Bind the successful worker command to this exact gate, not an unrelated exit0.
    dispatch=json.loads((worker_exit.parent/'dispatch.json').read_text())
    command=dispatch['command']
    require('--phase' in command and command[command.index('--phase')+1]=='gate'
            and command[command.index('--attempt')+1]==str(gate),'Wrong gate worker receipt.')
    return dict(gate_acceptance_sha256=file_hash(gate/'acceptance.json'),
                gate_actual_exit_sha256=file_hash(worker_exit),gate_dispatch_sha256=file_hash(worker_exit.parent/'dispatch.json'),
                independent_check_sha256=file_hash(gate/'independent-check.json'))


def run(repo,attempt,lane,phase,gate=None,gate_exit=None):
    require(phase in ('gate','states'),'No outcome/coverage dispatch implemented.')
    require(os.environ.get('JAX_PLATFORMS')=='cpu' and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
            'CPU and hidden GPUs must be declared before startup.')
    require(not str(attempt).startswith('/mnt/'),'Use local Linux storage for streaming evidence.')
    binding,full,pair,pre=inputs(repo)
    gate_binding=check_gate(repo,gate,gate_exit) if phase=='states' else None
    attempt.mkdir(parents=True,exist_ok=False)
    declaration=dict(schema='ood-cpu-state-execution-v1',phase=phase,pair_id=PAIR,source_sha256={p:file_hash(repo/p) for p in SOURCES},
        original_full_declaration_sha256=pre['declaration_sha256'],training_binding_sha256=pre['binding_sha256'],
        checkpoint_sha256=pre['checkpoint_sha256'],gate_binding=gate_binding,device='cpu',gpu_allocation=0,
        shared_lock=str(LOCK),ledger=str(LEDGER),maximum_environment_transitions=7 if phase=='gate' else 12801,
        maximum_physics_steps=28 if phase=='gate' else 51204,engineering_seed_indices=[4,5] if phase=='gate' else None,
        paired_reset_seeds=full['engineering_streams']['td3_bc/hopper'][4:6] if phase=='gate' else pair['streams']['collection'],
        captures=[0,100] if phase=='states' else ['reset'],episodes_per_collector=64 if phase=='states' else 1,
        action_absolute_tolerance=1e-6,reward_absolute_tolerance=1e-7,state_absolute_tolerance=0.,
        random_action_space_state='Both unused wrapper/native action RNGs initialized from reset seed; no sampling after reset.',
        continuation_streams=pair['streams']['continuation'] if phase=='states' else None,
        scientific_action_outcomes=False,coverage=False,global_readiness=False)
    declaration_sha=write_artifact(attempt/'declaration.json',declaration)
    caps=scope_caps([p['pair_id'] for p in full['pairs']],list(full['engineering_streams']))
    sim=None
    with worker_lease(), ResourceLedger(LEDGER,caps,pre['declaration_sha256']) as ledger:
        before=ledger.audit();require(before['reserved']['global'][0]>=81 and before['entries']>=21,'Missing cumulative history.')
        write_artifact(attempt/'resources-before.json',before)
        try:
            from runtime.environment import setup
            setup()
            import jax
            require(jax.default_backend()=='cpu','Wrong backend.')
            prior_runtime=read_artifact(repo/'outputs/ood/td3_bc/hopper/s202609171/trained-connection-v2/runtime.json',
                json.loads((repo/'docs/validation/ood-trained-td3-hopper-connection.json').read_text())['artifacts_sha256'][
                    'outputs/ood/td3_bc/hopper/s202609171/trained-connection-v2/runtime.json'])
            packages={p:importlib.metadata.version(p) for p in prior_runtime['packages']}
            require(packages==prior_runtime['packages'],'Changed accepted CPU query runtime.')
            write_artifact(attempt/'runtime.json',dict(devices=[str(d) for d in jax.devices()],backend=jax.default_backend(),packages=packages))
            models=_load_pair(repo,lane,binding)
            policies={k:FrozenPolicy(v) for k,v in models.items()}
            if phase=='states':
                # Literal JAX PRNGKey/fold_in construction, vectorized only across
                # independent integer keys. Persist all 256x250 keys before collection.
                import jax.numpy as jnp
                keys=np.asarray(jax.vmap(lambda seed:jax.vmap(lambda t:jax.random.fold_in(jax.random.PRNGKey(seed),t))(
                    jnp.arange(250,dtype=jnp.uint32)))(jnp.asarray(pair['streams']['continuation'],dtype=jnp.uint32)))
                from experiments.ood.adapters import continuation_keys
                for i in (0,127,255):
                    require(np.array_equal(keys[i],continuation_keys(pair['streams']['continuation'][i],250)),
                            'Vectorized continuation key contract differs.')
                key_sha=write_artifact(attempt/'continuation-keys.json',dict(keys=keys,seeds=pair['streams']['continuation'],
                    algorithm='PRNGKey(seed), fold_in(time)',horizon=250,used_in_state_collection=False))
            sim,bounds=construct(ledger,attempt,6 if phase=='gate' else 12800)
            with ArtifactArchive(attempt/'transitions.sqlite') as archive:
                reference_dir=sim.evidence_dir
                if phase=='gate':
                    comparisons=[]
                    for method,seed in zip(('host','bca'),declaration['paired_reset_seeds']):
                        obs=paired_reset(sim,int(seed));state=sim.capture()
                        direct=models[method].actions(obs[None,:])[0]
                        write_artifact(attempt/(method+'-reference-action.json'),dict(observation=obs,action=direct))
                        fast=policies[method].actions(obs[None,:])[0]
                        error=np.abs(direct.astype(np.float64)-fast.astype(np.float64))
                        write_artifact(attempt/(method+'-query-comparison.json'),dict(direct=direct,fast=fast,error=error))
                        require(np.isfinite(error).all() and (error<=1e-6).all(),'Changed eager query action.')
                        sim.evidence_dir=reference_dir
                        ref=reserved_step(sim,ledger,attempt,'engineering/td3_bc/hopper',method+'/reference',direct,bounds)
                        ref_state=sim.capture();write_artifact(attempt/(method+'-reference-transition.json'),dict(record=ref,after=ref_state))
                        sim.evidence_dir=ArchiveDirectory(archive)
                        for repeat in range(2):
                            sim.restore(state,expected_sha256=sim.snapshot_hash(state))
                            rec=reserved_step(sim,ledger,attempt,'engineering/td3_bc/hopper',f'{method}/stream/{repeat}',fast,bounds)
                            end=sim.capture()
                            write_artifact(attempt/f'{method}-stream-{repeat}.json',dict(record=rec,after=end))
                            require(sha(rec)==sha(ref) and sha(end)==sha(ref_state),'Streaming path changed real transition/state.')
                        policies[method].seal()
                        comparisons.append(dict(method=method,max_action_error=float(error.max()),exact_records=True))
                    result=dict(accepted_engineering=True,pair_id=PAIR,environment_transitions=7,physics_steps=28,
                        constructor_transitions=1,explicit_transitions=6,all_action_errors_zero=all(c['max_action_error']==0 for c in comparisons),
                        all_transition_records_exact=True,comparisons=comparisons,scientific_state_bank_accepted=False)
                else:
                    sim.evidence_dir=ArchiveDirectory(archive)
                    class PairedSim:
                        def reset(self,seed):return paired_reset(sim,seed)
                        def capture(self):return sim.capture()
                    states=attempt/'states';states.mkdir()
                    result=collect_bank(PairedSim(),policies,pair['streams']['collection'],states,
                        lambda owner,action:reserved_step(sim,ledger,attempt,'collection/'+PAIR,owner,action,bounds))
                    write_artifact(attempt/'states-index.json',result)
                    result.update(state_bank_completed=True,scientific_action_outcomes_accepted=False,
                                  explicit_transitions=sim.transitions,constructor_transitions=1,
                                  environment_transitions=sim.transitions+1,physics_steps=(sim.transitions+1)*4,
                                  continuation_keys_sha256=key_sha)
                result['archive_audit']=archive.audit()
            for policy in policies.values():policy.seal()
            require(all(file_hash(repo/p)==h for p,h in declaration['source_sha256'].items()),'Execution source changed.')
            result.update(declaration_sha256=declaration_sha,archive_sha256=file_hash(attempt/'transitions.sqlite'),
                          resources_after=ledger.audit(),actual_exit_required_separately=True,coverage_accepted=False)
            write_artifact(attempt/'acceptance.json',result)
            print(json.dumps({k:result[k] for k in ('environment_transitions','physics_steps','actual_exit_required_separately')}))
        except BaseException as exc:
            write_artifact(attempt/'failure.json',dict(error=repr(exc),resources=ledger.audit(),no_retry=True))
            raise
        finally:
            if sim is not None:sim.close()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    for name in ('repo','attempt','lane'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--phase',required=True,choices=['gate','states'])
    p.add_argument('--gate',type=Path);p.add_argument('--gate-exit',type=Path)
    args=p.parse_args();run(args.repo,args.attempt,args.lane,args.phase,args.gate,args.gate_exit)
