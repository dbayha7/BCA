"""Separately accepted, once-only action-harm execution for the first CPU pair.

Preparation cannot step a simulator. Execution requires an independent preflight
receipt binding the exact declaration. All original training files stay frozen.
"""
import datetime,importlib.metadata,json,os,shutil
from pathlib import Path
import numpy as np
from experiments.ood.collect import require,file_hash,read_artifact,write_artifact,sha
from experiments.ood.production import inputs,worker_lease,construct,reserved_step,paired_reset,SOURCES,PAIR,LEDGER,LOCK
from experiments.ood.candidate_stage import accepted_states
from experiments.ood.real_connection import _load_pair
from experiments.ood.streaming import ArtifactArchive,ArchiveDirectory,FrozenPolicy
from experiments.ood.resources import ResourceLedger,scope_caps
from experiments.ood.rollouts import panel


def accepted_candidates(repo,candidates,actual_exit):
    ex=json.loads(actual_exit.read_text());cmd=json.loads((actual_exit.parent/'dispatch.json').read_text())['command']
    require(ex['actual_returncode']==0 and ex['timeout'] is False and ex['interruption_signal'] is None,
            'Candidate worker must have actual exit0.')
    require('experiments.ood.candidate_stage' in cmd and cmd[cmd.index('--attempt')+1]==str(candidates),'Wrong candidate worker.')
    independent=json.loads((candidates/'independent-check.json').read_text())
    require(independent['accepted'] is True and independent['acceptance_sha256']==file_hash(candidates/'acceptance.json')
            and independent['worker_actual_exit_sha256']==file_hash(actual_exit)
            and independent['support_bank_sha256']==file_hash(candidates/'support-bank.json'),'Candidate audit binding changed.')
    result=read_artifact(candidates/'acceptance.json',independent['acceptance_sha256'])
    decl=read_artifact(candidates/'declaration.json',result['declaration_sha256'])
    require(result['precommit_completed'] is True and result['pair_id']==PAIR and len(result['rows'])==256,'Incomplete candidate bank.')
    require(all(file_hash(repo/f)==h for f,h in decl['source_sha256'].items()),'Changed candidate implementation.')
    hashes={}
    for item in result['rows']:
        path=candidates/'rows'/f"candidate-{item['index']:03d}.json";row=read_artifact(path,item['sha256']);hashes[str(path)]=item['sha256']
        if row['status']=='captured':
            action=candidates/'rows'/f"actions-{item['index']:03d}.json"
            require(file_hash(action)==row['action_file_sha256'],'Changed precommitted actions.');hashes[str(action)]=row['action_file_sha256']
    return result,dict(acceptance_sha256=file_hash(candidates/'acceptance.json'),independent_sha256=file_hash(candidates/'independent-check.json'),
        actual_exit_sha256=file_hash(actual_exit),declaration_sha256=file_hash(candidates/'declaration.json'),
        support_bank_sha256=file_hash(candidates/'support-bank.json'),row_file_sha256=hashes)


def prepare(repo,attempt,lane,states,states_exit,candidates,candidates_exit):
    binding,full,pair,pre=inputs(repo)
    sr,sb=accepted_states(repo,states,states_exit);cr,cb=accepted_candidates(repo,candidates,candidates_exit)
    require([(r['index'],r['status']) for r in sr['rows']]==[(r['index'],r['status']) for r in cr['rows']],
            'State/candidate layout differs.')
    paths=list(dict.fromkeys(SOURCES+['experiments/ood/candidates.py','experiments/ood/candidate_stage.py',
                                    'experiments/ood/rollouts.py','experiments/ood/outcome_stage.py']))
    declaration=dict(schema='ood-behavioral-execution-v1',pair_id=PAIR,repo=str(repo),lane=str(lane),
        states=str(states),states_actual_exit=str(states_exit),candidates=str(candidates),candidates_actual_exit=str(candidates_exit),
        state_binding=sb,candidate_binding=cb,checkpoint_sha256=pre['checkpoint_sha256'],
        original_full_declaration_sha256=pre['declaration_sha256'],training_binding_sha256=pre['binding_sha256'],
        source_sha256={f:file_hash(repo/f) for f in paths},device='cpu',gpu_allocation=0,shared_lock=str(LOCK),ledger=str(LEDGER),
        continuations=['host','bca'],primary_continuation='bca',state_rows=256,candidate_slots_per_state=10,horizon=250,
        maximum_outcome_transitions=1280000,maximum_repeat_transitions=5120,maximum_constructor_transitions=1,
        maximum_environment_transitions=1285121,maximum_physics_steps=5140484,maximum_worker_seconds=86400,
        minimum_free_disk_bytes=34359738368,action_absolute_tolerance=1e-6,reward_absolute_tolerance=1e-7,
        first_transition_and_full_state_repeat_tolerance=0.,raw_undiscounted_no_tail=True,
        initialization_reset_seed=full['engineering_streams']['td3_bc/hopper'][6],initialization_seed_index=6,
        missing='Retain all missing states; no replacements.',duplicates='Applied-action aliases reuse one physical trajectory and one ranking vote.',
        keys='Use frozen actual 256x250 JAX key bank; deterministic continuation actors do not draw random numbers.',
        resources='Reserve each outcome/repeat/constructor call durably before physics, in original cumulative ledger.',
        coverage_accepted=False,global_ready=False,exploratory_training_seed_count=1,required_training_seeds=5,
        no_retry=True,no_learning=True)
    attempt.mkdir(parents=True,exist_ok=False)
    write_artifact(attempt/'declaration.json',declaration)
    print(json.dumps(dict(prepared=True,actual_execution_not_started=True,declaration_sha256=file_hash(attempt/'declaration.json'))))


def execute(repo,attempt):
    approval=json.loads((attempt/'execution-acceptance.json').read_text())
    require(approval['accepted'] is True and approval['declaration_sha256']==file_hash(attempt/'declaration.json'),
            'Missing independent behavioral execution acceptance.')
    declaration=read_artifact(attempt/'declaration.json',approval['declaration_sha256'])
    require(declaration['pair_id']==PAIR and declaration['repo']==str(repo),'Wrong execution pair/repository.')
    require(all(file_hash(repo/f)==h for f,h in declaration['source_sha256'].items()),'Changed accepted execution source.')
    require(os.environ.get('JAX_PLATFORMS')=='cpu' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU/hidden GPU startup required.')
    binding,full,pair,pre=inputs(repo)
    states=Path(declaration['states']);candidates=Path(declaration['candidates'])
    sr,sb=accepted_states(repo,states,Path(declaration['states_actual_exit']))
    cr,cb=accepted_candidates(repo,candidates,Path(declaration['candidates_actual_exit']))
    require(sb==declaration['state_binding'] and cb==declaration['candidate_binding'],'Predecessor binding changed.')
    require(shutil.disk_usage(attempt).free>=declaration['minimum_free_disk_bytes'],'Insufficient declared evidence storage.')
    keys=read_artifact(states/'continuation-keys.json',sb['continuation_keys_sha256'])
    require(keys['seeds']==pair['streams']['continuation'] and keys['keys'].shape==(256,250,2)
            and keys['keys'].dtype==np.uint32,'Wrong continuation bank.')
    caps=scope_caps([p['pair_id'] for p in full['pairs']],list(full['engineering_streams']))
    sim=None
    with worker_lease(),ResourceLedger(LEDGER,caps,pre['declaration_sha256']) as ledger:
        write_artifact(attempt/'started.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            declaration_sha256=approval['declaration_sha256'],execution_acceptance_sha256=file_hash(attempt/'execution-acceptance.json')))
        before=ledger.audit();write_artifact(attempt/'resources-before.json',before)
        try:
            from runtime.environment import setup
            setup()
            import jax
            require(jax.default_backend()=='cpu','Wrong runtime backend.')
            prior=read_artifact(states/'runtime.json',file_hash(states/'runtime.json'))
            packages={p:importlib.metadata.version(p) for p in prior['packages']}
            require(packages==prior['packages'],'Changed accepted CPU runtime.')
            write_artifact(attempt/'runtime.json',dict(packages=packages,backend=jax.default_backend(),devices=[str(d) for d in jax.devices()]))
            models=_load_pair(repo,Path(declaration['lane']),binding);policies={k:FrozenPolicy(v) for k,v in models.items()}
            sim,bounds=construct(ledger,attempt,declaration['maximum_environment_transitions']-1)
            state_bounds=read_artifact(states/'live-native-bounds.json',file_hash(states/'live-native-bounds.json'))
            require(sha(bounds)==sha(state_bounds),'Actual native bounds/model changed.')
            # restore() validates against a fully initialized live wrapper/RNG
            # schema. This disjoint engineering reset takes no environment step;
            # every actual outcome starts from its saved scientific snapshot.
            paired_reset(sim,declaration['initialization_reset_seed'])
            write_artifact(attempt/'initialization-state.json',sim.capture())
            index=[];counts=dict(outcomes=0,repeat_checks=0)
            def step(scope,owner,action):
                require(scope in counts,'Invalid physical call scope.')
                maximum=declaration['maximum_outcome_transitions' if scope=='outcomes' else 'maximum_repeat_transitions']
                require(counts[scope]<maximum,'Declared physical scope exhausted.')
                counts[scope]+=1
                return reserved_step(sim,ledger,attempt,scope+'/'+PAIR,scope+'/'+owner,action,bounds)
            def save(name,value):
                path=attempt/'panels'/(name+'.json');path.parent.mkdir(parents=True,exist_ok=True)
                write_artifact(path,value)
            with ArtifactArchive(attempt/'transitions.sqlite') as archive:
                sim.evidence_dir=ArchiveDirectory(archive)
                for continuation in declaration['continuations']:
                    for item in sr['rows']:
                        i=item['index'];owner=f'{continuation}/state{i:03d}'
                        state=read_artifact(states/'states'/f'state-{i:03d}.json',item['sha256'])
                        row=read_artifact(candidates/'rows'/f'candidate-{i:03d}.json',cr['rows'][i]['sha256'])
                        require(state['state_id']==row['state_id'] and state['status']==row['status'],'Changed row pairing.')
                        if state['status']=='missing':result=dict(owner=owner,status='missing',state_id=state['state_id'])
                        else:
                            actions=read_artifact(candidates/'rows'/f'actions-{i:03d}.json',row['action_file_sha256'])
                            require(state['snapshot_sha256']==row['snapshot_sha256']==sha(state['snapshot']),'Changed full state.')
                            result=panel(sim,policies[continuation],state['snapshot'],actions,keys['keys'][i],owner,step,save)
                            result.update(status='completed',state_id=state['state_id'],continuation=continuation)
                        saved=attempt/'panels'/(owner+'/panel.json');saved.parent.mkdir(parents=True,exist_ok=True)
                        digest=write_artifact(saved,result);index.append(dict(owner=owner,status=result['status'],sha256=digest,path=str(saved)))
                        progress=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),panels_closed=len(index),total_panels=512,
                            physical_counts=counts.copy(),last_panel=owner,scientific_acceptance_pending=True)
                        tmp=attempt/'progress.tmp';tmp.write_text(json.dumps(progress)+'\n');tmp.replace(attempt/'progress.json')
                archive_audit=archive.audit()
            for policy in policies.values():policy.seal()
            require(all(file_hash(repo/f)==h for f,h in declaration['source_sha256'].items()),'Execution source changed.')
            result=dict(schema='ood-behavioral-worker-completion-v1',pair_id=PAIR,panels=index,physical_counts=counts,
                constructor_transitions=1,environment_transitions=1+sum(counts.values()),physics_steps=4*(1+sum(counts.values())),
                archive_audit=archive_audit,archive_sha256=file_hash(attempt/'transitions.sqlite'),resources_after=ledger.audit(),
                actor_queries={k:p.queries for k,p in policies.items()},declaration_sha256=approval['declaration_sha256'],
                actual_exit_required_separately=True,independent_scientific_audit_required=True,coverage_accepted=False)
            write_artifact(attempt/'completion.json',result);print(json.dumps(dict(worker_panels_completed=len(index),actual_exit_required=True)))
        except BaseException as exc:
            write_artifact(attempt/'failure.json',dict(error=repr(exc),resources_after=ledger.audit(),no_retry=True));raise
        finally:
            if sim is not None:sim.close()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','execute'])
    for name in ('repo','attempt','lane','states','states-exit','candidates','candidates-exit'):
        p.add_argument('--'+name,type=Path,required=name in ('repo','attempt'))
    a=p.parse_args()
    if a.phase=='prepare':prepare(a.repo,a.attempt,a.lane,a.states,a.states_exit,a.candidates,a.candidates_exit)
    else:execute(a.repo,a.attempt)
