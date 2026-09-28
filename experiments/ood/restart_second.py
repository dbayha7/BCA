"""Explicit second desktop recovery; both ancestors and all numerical code fixed.

The single input-only reserved call retains unknown historical output. Completed
actions are reused exactly once. There is no automatic retry or training path.
"""
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import zlib
import numpy as np
from experiments.ood.collect import require, sha, file_hash, read_artifact, write_artifact
from experiments.ood.restart import finish_panel, check_saved_row, record_from_archive, replay_scope
from experiments.ood import restart as prior

ROOT_NAME='td3-hopper-s202609171-outcomes-v1'
PREVIOUS_NAME='td3-hopper-s202609171-outcomes-restart-v1'
ATTEMPT_NAME='td3-hopper-s202609171-outcomes-restart-v2'
PRIOR_PLAN_SHA='595adb8f485b7ba278ae6756ed427fe611ee4808eb6af4630a9ca958615a49ba'
PARTIAL='bca/state163/slot9'


def merge_unique(first, second):
    require(not set(first).intersection(second),'Duplicate completed-action/panel provenance.')
    return {**first,**second}


def expected_layout():
    panels=[f'{c}/state{i:03d}' for c in ('host','bca') for i in range(256)]
    return [f'{p}/slot{j}' for p in panels for j in range(10)][:4199],panels[:419]


def recovery_counts():
    remaining=5120-4199
    return dict(maximum_new_physical_transitions=remaining*251+1,
        maximum_new_physics_steps=(remaining*251+1)*4,
        maximum_recovery_engineering_transitions=103+1+1,
        new_scientific_outcomes=remaining*250-103,new_scientific_repeats=remaining-1)


def compare_replay(inp, record, evidence):
    require(sha(inp)==evidence['input_sha256'],'Saved replay input/state differs.')
    if evidence['record_sha256'] is None:
        return 'input_only_unknown_prior_output'
    require(sha(record)==evidence['record_sha256'],'Saved replay output differs.')
    return 'full_record'


def boot_check(old):
    root=old.with_name(ROOT_NAME)
    original=prior.boot_check(root)
    process=old.with_name('td3-hopper-s202609171-outcomes-restart-process-v1')
    dispatch=json.loads((process/'dispatch.json').read_text())
    require(not any((old/x).exists() for x in ('failure.json','completion.json'))
        and not (process/'actual_exit.json').exists(),'Previous recovery has a recorded closure/failure.')
    require(datetime.datetime.fromisoformat(dispatch['started']).timestamp()<original['boot_unix'],
        'Previous recovery did not precede current boot.')
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
        except (OSError,UnicodeError):continue
        require(not ('experiments.ood.restart' in argv and str(old) in argv),'Previous recovery still live.')
    return dict(original=original,previous_recovery_dispatch=dispatch,
        original_actual_exit=None,previous_recovery_actual_exit=None,
        reason='User reported another desktop restart; standing restart direction applied to separate checked recovery.')


def check_original(plan):
    prior_plan=read_artifact(Path(plan['first_recovery'])/'recovery-plan.json',PRIOR_PLAN_SHA)
    prior.check_original(prior_plan)
    old=Path(plan['old_attempt'])
    for name,h in plan['original_files_sha256'].items():
        require(file_hash(old/name)==h,'First recovery evidence changed: '+name)
    require(file_hash(old/'transitions.sqlite')==plan['original_archive_sha256'],'First recovery archive changed.')


def scan_archive(archive, count_expected):
    require(archive.db.execute('PRAGMA quick_check').fetchall()==[('ok',)],'Damaged archive.')
    previous='0'*64;count=0;total=0
    for index,name,metadata,payload,digest in archive.db.execute('SELECT id,name,metadata,payload,digest FROM artifacts ORDER BY id'):
        m=json.loads(metadata);d=zlib.decompressobj();raw=d.decompress(payload,archive.maximum+1)
        require(index==count and m['index']==index and m['name']==name and m['previous']==previous
            and sha(m)==digest and m['complete'] is True and m['bytes']==len(raw)
            and len(raw)<=archive.maximum and d.eof and not d.unused_data
            and hashlib.sha256(raw).hexdigest()==m['raw_sha256'],'Changed archive record/chain/payload.')
        count+=1;total+=len(raw);previous=digest
    require(count==count_expected,'Archive count differs from reserved-call boundary.')
    return dict(records=count,uncompressed_bytes=total,last_sha256=previous)


def prepare(repo, old, attempt):
    from experiments.ood.production import inputs,worker_lease,LEDGER,PAIR
    from experiments.ood.resources import ResourceLedger,scope_caps
    from experiments.ood.streaming import ArtifactArchive
    from experiments.ood.candidate_stage import accepted_states
    from experiments.ood.outcome_stage import accepted_candidates
    require(old.name==PREVIOUS_NAME and attempt.name==ATTEMPT_NAME and old.parent==attempt.parent,'Wrong exact roots.')
    attempt.mkdir(parents=True,exist_ok=False)
    with worker_lease():
        boot=boot_check(old);write_artifact(attempt/'interruption.json',boot)
        previous=read_artifact(old/'recovery-plan.json',PRIOR_PLAN_SHA);prior.check_original(previous)
        approval=json.loads((old/'execution-acceptance.json').read_text())
        require(approval['accepted'] is True and approval['plan_sha256']==PRIOR_PLAN_SHA,'Prior acceptance changed.')
        gate=json.loads((old/'replay-gate.json').read_text())
        require(gate==dict(accepted=True,changed_tolerances=False,exact_record_matches=237,
            outcome_prefix_steps=236,repeat_prefix_steps=1),'Prior replay gate changed.')
        require(file_hash(repo/'experiments/ood/restart.py')==previous['recovery_source_sha256']
            and file_hash(repo/'experiments/ood/test_restart.py')==previous['tests_source_sha256'],'Prior recovery code changed.')
        original=previous['original_declaration'];binding,full,pair,pre=inputs(repo)
        require(pre==previous['predecessor_binding'],'Training provenance changed.')
        require(all(file_hash(repo/f)==h for f,h in original['source_sha256'].items()),'Original outcome source changed.')
        states=Path(original['states']);candidates=Path(original['candidates'])
        sr,sb=accepted_states(repo,states,Path(original['states_actual_exit']))
        cr,cb=accepted_candidates(repo,candidates,Path(original['candidates_actual_exit']))
        require(sb==original['state_binding'] and cb==original['candidate_binding'],'Frozen banks changed.')
        keys=read_artifact(states/'continuation-keys.json',sb['continuation_keys_sha256'])['keys']
        files={str(p.relative_to(old)):file_hash(p) for p in old.rglob('*') if p.is_file() and p.name!='transitions.sqlite'}
        new_actions={};new_panels={}
        for c in original['continuations']:
            for i in range(256):
                owner=f'{c}/state{i:03d}'
                for slot in range(10):
                    name=f'panels/{owner}/slot{slot}/completed.json'
                    if name in files:new_actions[f'{owner}/slot{slot}']=dict(path=str(old/name),sha256=files[name])
                name=f'panels/{owner}/panel.json'
                if name in files:new_panels[owner]=dict(path=str(old/name),sha256=files[name])
        complete=merge_unique(previous['reused_actions'],new_actions)
        panels=merge_unique(previous['reused_panels'],new_panels)
        expected_actions,expected_panels=expected_layout()
        require(set(complete)==set(expected_actions) and set(panels)==set(expected_panels),'Unexpected durable prefix.')
        complete={k:complete[k] for k in expected_actions};panels={k:panels[k] for k in expected_panels}
        require(len(new_actions)==2761 and len(new_panels)==276,'Unexpected first-recovery contributions.')
        starts={str(p.parent.relative_to(old/'panels')) for p in old.glob('panels/*/state*/slot*/started.json')}
        require(starts==set(new_actions)|{PARTIAL},'Unexpected incomplete action/start.')
        for c in original['continuations']:
            for i in range(256):
                owner=f'{c}/state{i:03d}'
                if f'{owner}/slot0' not in complete:continue
                state=read_artifact(states/'states'/f'state-{i:03d}.json',sr['rows'][i]['sha256'])
                candidate=read_artifact(candidates/'rows'/f'candidate-{i:03d}.json',cr['rows'][i]['sha256'])
                actions=read_artifact(candidates/'rows'/f'actions-{i:03d}.json',candidate['action_file_sha256'])
                require(actions['alias']==list(range(10)),'Unexpected aliases.')
                rows=[]
                for slot in range(10):
                    e=complete.get(f'{owner}/slot{slot}')
                    if not e:continue
                    p=Path(e['path']);row=read_artifact(p,e['sha256']);check_saved_row(row,slot,keys[i])
                    require(row['length']==250,'Unexpected completed-action length.')
                    def saved(name):
                        q=p.with_name(name+'.json');ancestor=old if old in q.parents else Path(previous['old_attempt'])
                        hashes=files if ancestor==old else previous['original_files_sha256']
                        return read_artifact(q,hashes[str(q.relative_to(ancestor))])
                    start=saved('started');first=saved('first');repeat=saved('repeat')
                    require(start['snapshot_sha256']==sha(state['snapshot']) and start['step_keys_sha256']==sha(keys[i])
                        and np.array_equal(start['sent'],actions['sent'][slot])
                        and np.array_equal(start['expected_applied'],actions['applied'][slot]),'Saved action/state/key changed.')
                    require(sha(first)==sha(repeat) and sha(first['record'])==row['first_transition_sha256']
                        and sha(first['after'])==row['first_end_sha256'],'Saved first/repeat evidence differs.')
                    rows.append(row)
                if owner in panels:
                    e=panels[owner];p=read_artifact(Path(e['path']),e['sha256'])
                    require(p['owner']==owner and p['status']=='completed' and p['continuation']==c
                        and sha(p['slots'])==sha(rows),'Mixed ancestor panel/slot disagreement.')
        with ResourceLedger(LEDGER,scope_caps([p['pair_id'] for p in full['pairs']],list(full['engineering_streams'])),pre['declaration_sha256']) as ledger:
            before=ledger.audit();n=0;mapping={};counts=dict(outcomes=0,repeat_checks=0)
            for token, in ledger.db.execute('SELECT token FROM entries ORDER BY id'):
                if not token.startswith(str(old)+'/'):continue
                rel=token[len(str(old))+1:]
                if rel=='constructor':continue
                scope,owner=rel.split('/',1);require(scope in counts,'Unexpected prior scope.')
                n+=1;counts[scope]+=1
                if owner.startswith(PARTIAL+'/'):mapping[rel]=f'transition-{n:05d}'
            require(counts==dict(outcomes=690353,repeat_checks=2762) and n==693115,'Unexpected first recovery reservations.')
            expected={*(f'outcomes/{PARTIAL}/{i}' for i in range(103)),f'repeat_checks/{PARTIAL}/repeat'}
            require(set(mapping)==expected,'Unexpected partial reservation layout.')
            require(before['reserved']['outcomes/'+PAIR]==[1049853,4199412]
                and before['reserved']['repeat_checks/'+PAIR]==[4200,16800],'Unexpected cumulative science consumption.')
        print(json.dumps(dict(stage='both_prefixes_bound',panels=419,actions=4199,input_only_reservations=1)),flush=True)
        prefix={}
        with ArtifactArchive(old/'transitions.sqlite',read_only=True) as archive:
            audit=scan_archive(archive,n*3-2)
            for rel,name in mapping.items():
                inp=record_from_archive(archive,name+'-input.npz')
                exists=archive.db.execute('SELECT 1 FROM artifacts WHERE name=?',(name+'.npz',)).fetchone() is not None
                input_only=rel==f'outcomes/{PARTIAL}/102'
                require(exists != input_only,'Unexpected missing/present historical output.')
                applied=archive.db.execute('SELECT 1 FROM artifacts WHERE name=?',(name+'-applied.npz',)).fetchone() is not None
                require(applied==exists,'Unexpected applied-vector record at partial boundary.')
                record=record_from_archive(archive,name+'.npz') if exists else None
                prefix[rel]=dict(archive_name=name,input_sha256=sha(inp),record_sha256=sha(record) if exists else None)
            for suffix,rel in [('first',f'outcomes/{PARTIAL}/0'),('repeat',f'repeat_checks/{PARTIAL}/repeat')]:
                name=f'panels/{PARTIAL}/{suffix}.json'
                require(prefix[rel]['record_sha256']==sha(read_artifact(old/name,files[name])['record']),'Partial first/repeat differs.')
        sources={**original['source_sha256'],'experiments/ood/restart.py':previous['recovery_source_sha256'],
                 'experiments/ood/test_restart.py':previous['tests_source_sha256']}
        plan=dict(schema='ood-second-desktop-restart-v1',user_direction='It restarted again. How are things looking?',
            standing_restart_direction='had to restart my desktop so it needs starting again.',
            old_attempt=str(old),first_recovery=str(old),root_original=previous['old_attempt'],
            first_recovery_plan_sha256=PRIOR_PLAN_SHA,ancestor_actual_exits=[None,None],interruption=boot,
            original_files_sha256=files,original_archive_sha256=file_hash(old/'transitions.sqlite'),original_archive_audit=audit,
            original_declaration=original,predecessor_binding=pre,reused_panels=panels,reused_actions=complete,
            reused_scientific_counts=dict(outcomes=4199*250,repeat_checks=4199),partial_owner=PARTIAL,
            replay_prefix=prefix,replay_outcomes=103,replay_repeats=1,replay_full_records=103,input_only_records=1,
            original_reserved_counts=dict(outcomes=1049853,repeat_checks=4200),prior_physical_counts=counts,
            resources_before=before,original_sources=sources,
            recovery_source_sha256=file_hash(repo/'experiments/ood/restart_second.py'),
            tests_source_sha256=file_hash(repo/'experiments/ood/test_restart_second.py'),
            maximum_worker_seconds=72000,device='cpu',training_updates=0,gpu_allocation=0,no_automatic_retry=True,
            historical_input_only_output_unknown=True,coverage_accepted=False,scientific_acceptance_pending=True,**recovery_counts())
        write_artifact(attempt/'recovery-plan.json',plan)
        print(json.dumps(dict(prepared=True,plan_sha256=file_hash(attempt/'recovery-plan.json'),panels=419,actions=4199,
            full_record_checks=103,input_only_checks=1,scientific_acceptance=False)),flush=True)


def execute(repo, attempt):
    from experiments.ood.production import inputs,worker_lease,construct,reserved_step,paired_reset,LEDGER,PAIR
    from experiments.ood.candidate_stage import accepted_states
    from experiments.ood.outcome_stage import accepted_candidates
    from experiments.ood.real_connection import _load_pair
    from experiments.ood.streaming import ArtifactArchive,ArchiveDirectory,FrozenPolicy
    from experiments.ood.resources import ResourceLedger,scope_caps
    acceptance=json.loads((attempt/'execution-acceptance.json').read_text())
    require(acceptance['accepted'] is True,'Second recovery execution not accepted.')
    plan=read_artifact(attempt/'recovery-plan.json',acceptance['plan_sha256']);old=Path(plan['old_attempt']);d=plan['original_declaration']
    require(attempt.name==ATTEMPT_NAME and old.name==PREVIOUS_NAME and old.parent==attempt.parent,'Wrong execution roots.')
    require(os.environ.get('JAX_PLATFORMS')=='cpu' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Require original CPU runtime.')
    require(file_hash(repo/'experiments/ood/restart_second.py')==plan['recovery_source_sha256']
        and file_hash(repo/'experiments/ood/test_restart_second.py')==plan['tests_source_sha256'],'Second recovery code changed.')
    require(all(file_hash(repo/f)==h for f,h in plan['original_sources'].items()),'Ancestor execution source changed.')
    check_original(plan);boot_check(old)
    binding,full,pair,pre=inputs(repo);require(pre==plan['predecessor_binding'],'Changed training binding.')
    states=Path(d['states']);candidates=Path(d['candidates'])
    sr,sb=accepted_states(repo,states,Path(d['states_actual_exit']));cr,cb=accepted_candidates(repo,candidates,Path(d['candidates_actual_exit']))
    require(sb==d['state_binding'] and cb==d['candidate_binding'],'Changed banks.')
    keys=read_artifact(states/'continuation-keys.json',sb['continuation_keys_sha256'])['keys']
    require(shutil.disk_usage(attempt).free>=d['minimum_free_disk_bytes'],'Insufficient disk.')
    sim=None
    with worker_lease(),ResourceLedger(LEDGER,scope_caps([p['pair_id'] for p in full['pairs']],list(full['engineering_streams'])),pre['declaration_sha256']) as ledger:
        require(ledger.audit()==plan['resources_before'],'Ledger changed after preparation.')
        write_artifact(attempt/'started.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            plan_sha256=acceptance['plan_sha256'],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
        try:
            from runtime.environment import setup
            setup()
            import jax
            require(jax.default_backend()=='cpu','Wrong backend.')
            runtime=read_artifact(old/'runtime.json',plan['original_files_sha256']['runtime.json'])
            packages={p:importlib.metadata.version(p) for p in runtime['packages']}
            require(packages==runtime['packages'],'Changed CPU packages.')
            write_artifact(attempt/'runtime.json',dict(packages=packages,backend=jax.default_backend(),devices=[str(x) for x in jax.devices()]))
            policies={k:FrozenPolicy(v) for k,v in _load_pair(repo,Path(d['lane']),binding).items()}
            sim,bounds=construct(ledger,attempt,plan['maximum_new_physical_transitions']-1)
            require(sha(bounds)==sha(read_artifact(old/'live-native-bounds.json',plan['original_files_sha256']['live-native-bounds.json'])),'Changed simulator/bounds.')
            paired_reset(sim,d['initialization_reset_seed']);write_artifact(attempt/'initialization-state.json',sim.capture())
            counts=dict(outcomes=0,repeat_checks=0);charged=dict(outcomes=0,repeat_checks=0,engineering=1)
            replayed={};index=[];pending_reuse=dict(plan['reused_actions'])
            with ArtifactArchive(attempt/'transitions.sqlite') as archive:
                sim.evidence_dir=ArchiveDirectory(archive)
                def step(scope,owner,action):
                    require(sum(counts.values())+1<plan['maximum_new_physical_transitions'],'Physical ceiling exhausted.')
                    actual_scope=replay_scope(scope,owner,PARTIAL,103,1)
                    if actual_scope=='engineering':
                        require(charged['engineering']<105,'Replay overhead ceiling exhausted.')
                        charge='engineering/td3_bc/hopper'
                    else:
                        require(len(replayed)==104,'Science requested before saved-prefix checks.')
                        charge=scope+'/'+PAIR
                    counts[scope]+=1;charged[actual_scope]+=1
                    rec=reserved_step(sim,ledger,attempt,charge,scope+'/'+owner,action,bounds)
                    if actual_scope=='engineering':
                        rel=scope+'/'+owner
                        require(rel not in replayed,'Duplicate prefix call.')
                        inp=record_from_archive(archive,f'transition-{sim.transitions:05d}-input.npz')
                        replayed[rel]=compare_replay(inp,rec,plan['replay_prefix'][rel])
                        if len(replayed)==104:
                            require(list(replayed.values()).count('full_record')==103,'Wrong full-record count.')
                            require(list(replayed.values()).count('input_only_unknown_prior_output')==1,'Wrong unknown-output count.')
                            write_artifact(attempt/'replay-gate.json',dict(accepted=True,full_record_matches=103,
                                input_record_matches=104,input_only_unknown_prior_outputs=1,
                                outcome_prefix_reservations=103,repeat_prefix_reservations=1,changed_tolerances=False))
                    return rec
                def save(name,value):
                    path=attempt/'panels'/(name+'.json');path.parent.mkdir(parents=True,exist_ok=True);write_artifact(path,value)
                for cont in d['continuations']:
                    for item in sr['rows']:
                        i=item['index'];owner=f'{cont}/state{i:03d}'
                        if owner in plan['reused_panels']:
                            e=plan['reused_panels'][owner];read_artifact(Path(e['path']),e['sha256'])
                            index.append(dict(owner=owner,status='reused_completed',**e))
                            for slot in range(10):pending_reuse.pop(f'{owner}/slot{slot}')
                        else:
                            state=read_artifact(states/'states'/f'state-{i:03d}.json',item['sha256'])
                            row=read_artifact(candidates/'rows'/f'candidate-{i:03d}.json',cr['rows'][i]['sha256'])
                            require(state['state_id']==row['state_id'] and state['status']==row['status'],'Changed state pairing.')
                            reused={}
                            for slot in range(10):
                                e=pending_reuse.pop(f'{owner}/slot{slot}',None)
                                if e:reused[slot]=read_artifact(Path(e['path']),e['sha256'])
                            if state['status']=='missing':result=dict(owner=owner,status='missing',state_id=state['state_id'])
                            else:
                                actions=read_artifact(candidates/'rows'/f'actions-{i:03d}.json',row['action_file_sha256'])
                                require(state['snapshot_sha256']==row['snapshot_sha256']==sha(state['snapshot']),'Changed snapshot.')
                                result=finish_panel(sim,policies[cont],state['snapshot'],actions,keys[i],owner,step,save,reused=reused)
                                result.update(status='completed',state_id=state['state_id'],continuation=cont)
                            path=attempt/'panels'/(owner+'/panel.json');path.parent.mkdir(parents=True,exist_ok=True)
                            h=write_artifact(path,result);index.append(dict(owner=owner,status=result['status'],path=str(path),sha256=h))
                        progress=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),panels_closed=len(index),total_panels=512,
                            reused_panels=419,reused_actions=4199,new_physical_counts=counts.copy(),charged_counts=charged.copy(),
                            last_panel=owner,scientific_acceptance_pending=True)
                        temp=attempt/'progress.tmp';temp.write_text(json.dumps(progress)+'\n');temp.replace(attempt/'progress.json')
                require(not pending_reuse and len(replayed)==104,'Incomplete reuse/prefix check.')
                archive_audit=archive.audit()
            for p in policies.values():p.seal()
            check_original(plan)
            require(all(file_hash(repo/f)==h for f,h in plan['original_sources'].items()),'Ancestor source changed during execution.')
            write_artifact(attempt/'completion.json',dict(schema='ood-second-recovery-worker-completion-v1',pair_id=PAIR,
                panels=index,reused_actions=plan['reused_actions'],new_physical_counts=counts,charged_counts=charged,
                original_reserved_counts=plan['original_reserved_counts'],replayed_reserved_calls=104,
                replayed_full_records=103,historical_input_only_output_unknown=True,ancestor_actual_exits=[None,None],
                both_partial_traces_retained_excluded=True,environment_transitions=1+sum(counts.values()),
                physics_steps=4*(1+sum(counts.values())),resources_after=ledger.audit(),archive_audit=archive_audit,
                archive_sha256=file_hash(attempt/'transitions.sqlite'),recovery_plan_sha256=acceptance['plan_sha256'],
                actual_exit_required_separately=True,independent_scientific_audit_required=True,coverage_accepted=False))
            print(json.dumps(dict(worker_panels_completed=len(index),reused_panels=419,actual_exit_required=True)),flush=True)
        except BaseException as exc:
            write_artifact(attempt/'failure.json',dict(error=repr(exc),resources_after=ledger.audit(),no_automatic_retry=True));raise
        finally:
            if sim is not None:sim.close()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','execute'])
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True);p.add_argument('--old',type=Path)
    a=p.parse_args()
    if a.phase=='prepare':prepare(a.repo,a.old,a.attempt)
    else:execute(a.repo,a.attempt)
