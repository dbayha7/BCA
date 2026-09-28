"""One explicitly authorized desktop-interruption recovery; original attempt immutable.

This is not a general retry interface. Preparation requires absent old processes,
a newer boot, no old scientific failure, and the exact inspected durable prefix.
Completed actions are reused. Only the interrupted action is replayed, with all
saved prefix records checked exactly and repeat calls charged as engineering.
"""
import copy
import datetime
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import zlib
import hashlib
import numpy as np
from experiments.ood.collect import require, sha, file_hash, read_artifact, write_artifact

OLD_NAME = 'td3-hopper-s202609171-outcomes-v1'
PARTIAL = 'host/state143/slot8'
REPLAY_OUTCOMES = 236
REPLAY_REPEATS = 1
COMPLETED_ACTIONS = 1438


def check_saved_row(row, slot, keys):
    rewards = np.asarray(row['rewards'])
    require(row['slot'] == row['alias'] == row['source_slot'] == slot,
            'Wrong reused action identity.')
    require(rewards.dtype == np.float64 and rewards.ndim == 1 and 1 <= len(rewards) <= 250
            and np.isfinite(rewards).all() and row['length'] == len(rewards)
            and row['raw_return'] == sum(float(x) for x in rewards), 'Changed reused rewards/counts.')
    require(row['continuation_keys_sha256'] == sha(keys) and row['repeat_exact'] is True
            and row['deterministic_actor_keys_unused'] is True, 'Changed reused keys/repeat evidence.')
    require(row['horizon_exhausted'] == (len(rewards) == 250)
            and (len(rewards) == 250 or row['terminated'] or row['truncated']), 'Incomplete reused action.')


def replay_scope(scope, owner, partial, outcome_count, repeat_count):
    require(scope in ('outcomes', 'repeat_checks'), 'Invalid physical scope.')
    if owner.startswith(partial + '/'):
        suffix = owner[len(partial)+1:]
        if scope == 'outcomes':
            require(suffix.isdigit(), 'Invalid outcome step.')
            if int(suffix) < outcome_count: return 'engineering'
        elif suffix == 'repeat' and repeat_count == 1:
            return 'engineering'
    return scope


def finish_panel(sim, policy, snapshot, actions, step_keys, owner, step, save, *, reused):
    """Same original numerical loop, with validated completed actions bypassed."""
    sent, applied = np.asarray(actions['sent']), np.asarray(actions['applied'])
    aliases = actions['alias']; keys = np.asarray(step_keys)
    require(sent.shape == applied.shape and sent.ndim == 2 and len(sent) == 10
            and sent.dtype == applied.dtype == np.float32 and np.isfinite(sent).all()
            and (abs(sent) <= 1).all(), 'Invalid frozen first-action bank.')
    require(keys.shape == (250,2) and keys.dtype == np.uint32, 'Require frozen 250-step JAX keys.')
    expected = [next((j for j in range(i) if np.array_equal(applied[i],applied[j])),i) for i in range(10)]
    require(aliases == expected and all(type(k) is int and 0 <= k < 10 and aliases[k] == k for k in reused),
            'Changed aliases or invalid reused slot.')
    for slot, row in reused.items(): check_saved_row(row, slot, keys)
    state_sha = sha(snapshot); result = []
    for slot, action in enumerate(sent):
        if aliases[slot] != slot:
            result.append(dict(slot=slot,alias=aliases[slot],source_slot=aliases[slot])); continue
        if slot in reused:
            result.append(copy.deepcopy(reused[slot])); continue
        policy.seal()
        name = f'{owner}/slot{slot}'
        save(name+'/started',dict(snapshot_sha256=state_sha,sent=action,expected_applied=applied[slot],step_keys_sha256=sha(keys)))
        sim.restore(snapshot,expected_sha256=sim.snapshot_hash(snapshot))
        require(sha(sim.capture()) == state_sha, 'First-action restore differs.')
        first = step('outcomes',name+'/0',action); first_end = sim.capture()
        save(name+'/first',dict(record=first,after=first_end))
        require(np.array_equal(first['applied_action'],applied[slot]), 'Frozen first action differs.')
        sim.restore(snapshot,expected_sha256=sim.snapshot_hash(snapshot))
        repeat = step('repeat_checks',name+'/repeat',action); repeat_end = sim.capture()
        save(name+'/repeat',dict(record=repeat,after=repeat_end))
        require(sha(first) == sha(repeat) and sha(first_end) == sha(repeat_end), 'Repeated first transition/state differs.')
        sim.restore(first_end,expected_sha256=sim.snapshot_hash(first_end))
        rewards = [float(first['reward'])]; record = first
        for t in range(1,250):
            if record['terminated'] or record['truncated']: break
            next_action = policy.actions(np.asarray(record['observation'])[None,:])[0]
            record = step('outcomes',name+f'/{t}',next_action)
            rewards.append(float(record['reward']))
        policy.seal()
        row = dict(slot=slot,alias=slot,source_slot=slot,rewards=np.asarray(rewards,np.float64),
            raw_return=float(sum(rewards)),length=len(rewards),terminated=bool(record['terminated']),
            truncated=bool(record['truncated']),horizon_exhausted=len(rewards)==250,
            first_transition_sha256=sha(first),first_end_sha256=sha(first_end),last_record_sha256=sha(record),
            complete_end_state_sha256=sha(sim.capture()),repeat_exact=True,
            continuation_keys_sha256=sha(keys),deterministic_actor_keys_unused=True)
        save(name+'/completed',row); result.append(row)
    require(sha(snapshot) == state_sha, 'Source snapshot mutated.')
    return dict(owner=owner,slots=result,unique_applied_actions=sum(i==j for i,j in enumerate(aliases)),
                reference_slot=0,raw_undiscounted_no_tail=True)


def boot_check(old):
    process = old.with_name(OLD_NAME.replace('outcomes-v1','outcomes-process-v1'))
    dispatch = json.loads((process/'dispatch.json').read_text())
    require(not (process/'actual_exit.json').exists() and not (old/'failure.json').exists()
            and not (old/'completion.json').exists(), 'Not an unclosed desktop interruption.')
    btime = int(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime ')))
    require(datetime.datetime.fromisoformat(dispatch['started']).timestamp() < btime,
            'Old attempt did not precede this boot.')
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try: argv = [x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
        except (OSError,UnicodeError): continue
        require(not ('experiments.ood.outcome_stage' in argv and str(old) in argv), 'Original worker still live.')
    return dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),boot_unix=btime,
                original_dispatch=dispatch,original_actual_returncode=None,
                reason='User reported desktop restart and explicitly requested restart; original exit unavailable.')


def check_original(plan):
    old = Path(plan['old_attempt'])
    for name, h in plan['original_files_sha256'].items():
        require(file_hash(old/name) == h, 'Original interrupted evidence changed: '+name)
    require(file_hash(old/'transitions.sqlite') == plan['original_archive_sha256'], 'Original archive changed.')


def record_from_archive(archive, name):
    with np.load(io.BytesIO(archive.read(name)),allow_pickle=False) as data:
        return {k:(data[k].item() if data[k].ndim == 0 else data[k].copy()) for k in data.files}


def prepare(repo, old, attempt):
    from experiments.ood.production import inputs, worker_lease, LEDGER, PAIR
    from experiments.ood.resources import ResourceLedger, scope_caps
    from experiments.ood.streaming import ArtifactArchive
    from experiments.ood.candidate_stage import accepted_states
    from experiments.ood.outcome_stage import accepted_candidates
    require(old.name == OLD_NAME and old.parent == attempt.parent and old != attempt, 'Wrong recovery roots.')
    attempt.mkdir(parents=True,exist_ok=False)
    with worker_lease():
        boot = boot_check(old)
        write_artifact(attempt/'interruption.json',boot)
        original = read_artifact(old/'declaration.json',file_hash(old/'declaration.json'))
        approval = json.loads((old/'execution-acceptance.json').read_text())
        require(approval['accepted'] is True and approval['declaration_sha256'] == file_hash(old/'declaration.json'), 'Original acceptance changed.')
        require(all(file_hash(repo/f) == h for f,h in original['source_sha256'].items()), 'Original scientific/execution source changed.')
        binding,full,pair,pre = inputs(repo)
        states = Path(original['states']); candidates = Path(original['candidates'])
        sr,sb = accepted_states(repo,states,Path(original['states_actual_exit']))
        cr,cb = accepted_candidates(repo,candidates,Path(original['candidates_actual_exit']))
        require(sb == original['state_binding'] and cb == original['candidate_binding'], 'Changed banks.')
        keys = read_artifact(states/'continuation-keys.json',sb['continuation_keys_sha256'])['keys']
        files = {str(p.relative_to(old)):file_hash(p) for p in old.rglob('*') if p.is_file() and p.name != 'transitions.sqlite'}
        complete = {}; panels = {}; accepted_counts = dict(outcomes=0,repeat_checks=0)
        for cont in original['continuations']:
            for i in range(256):
                owner = f'{cont}/state{i:03d}'
                for slot in range(10):
                    name = f'panels/{owner}/slot{slot}/completed.json'
                    if name not in files: continue
                    row = read_artifact(old/name,files[name]);check_saved_row(row,slot,keys[i])
                    require(row['length'] == 250, 'Unexpected prefix length; requires a new reviewed plan.')
                    base = f'panels/{owner}/slot{slot}/'
                    start = read_artifact(old/(base+'started.json'),files[base+'started.json'])
                    first = read_artifact(old/(base+'first.json'),files[base+'first.json'])
                    repeat = read_artifact(old/(base+'repeat.json'),files[base+'repeat.json'])
                    state = read_artifact(states/'states'/f'state-{i:03d}.json',sr['rows'][i]['sha256'])
                    actions = read_artifact(candidates/'rows'/f'actions-{i:03d}.json',
                        read_artifact(candidates/'rows'/f'candidate-{i:03d}.json',cr['rows'][i]['sha256'])['action_file_sha256'])
                    require(actions['alias'] == list(range(10)), 'Unexpected aliases in inspected prefix.')
                    require(start['snapshot_sha256'] == sha(state['snapshot']) and start['step_keys_sha256'] == sha(keys[i])
                        and np.array_equal(start['sent'],actions['sent'][slot]) and np.array_equal(start['expected_applied'],actions['applied'][slot]), 'Changed saved first action.')
                    require(sha(first) == sha(repeat) and row['first_transition_sha256'] == sha(first['record'])
                        and row['first_end_sha256'] == sha(first['after']), 'Changed saved repeat/first state.')
                    complete[f'{owner}/slot{slot}'] = dict(path=str(old/name),sha256=files[name])
                    accepted_counts['outcomes'] += row['length'];accepted_counts['repeat_checks'] += 1
                name = f'panels/{owner}/panel.json'
                if name in files:
                    p = read_artifact(old/name,files[name])
                    require(p['owner'] == owner and p['status'] == 'completed' and p['continuation'] == cont, 'Wrong saved panel.')
                    for slot,row in enumerate(p['slots']):
                        item = complete[f'{owner}/slot{slot}']
                        require(sha(row) == sha(read_artifact(Path(item['path']),item['sha256'])), 'Panel/slot differ.')
                    panels[owner] = dict(path=str(old/name),sha256=files[name])
        expected = [f'host/state{i:03d}/slot{j}' for i in range(144) for j in range(10)][:COMPLETED_ACTIONS]
        require(list(complete) == expected and list(panels) == [f'host/state{i:03d}' for i in range(143)], 'Durable prefix differs from restart plan.')
        require(set(str(p.relative_to(old)) for p in old.glob('panels/*/state*/slot*/started.json'))
            == {f'panels/{x}/started.json' for x in [*expected,PARTIAL]}, 'More than one partial action or changed start layout.')
        with ResourceLedger(LEDGER,scope_caps([p['pair_id'] for p in full['pairs']],list(full['engineering_streams'])),pre['declaration_sha256']) as ledger:
            before = ledger.audit();mapping = {};n = 0;scope_counts = dict(outcomes=0,repeat_checks=0)
            for token, in ledger.db.execute('SELECT token FROM entries ORDER BY id'):
                if not token.startswith(str(old)+'/'): continue
                rel = token[len(str(old))+1:]
                if rel == 'constructor': continue
                scope,owner = rel.split('/',1);require(scope in scope_counts,'Unexpected old scope.')
                n += 1;scope_counts[scope] += 1
                if owner.startswith(PARTIAL+'/'): mapping[rel] = f'transition-{n:05d}.npz'
            require(scope_counts == dict(outcomes=359736,repeat_checks=1439)
                and set(mapping) == {*(f'outcomes/{PARTIAL}/{i}' for i in range(236)),
                                     f'repeat_checks/{PARTIAL}/repeat'}, 'Old reservations differ.')
            require(before['reserved']['outcomes/'+PAIR] == [359736,1438944]
                    and before['reserved']['repeat_checks/'+PAIR] == [1439,5756], 'Other science already consumed these scopes.')
        print(json.dumps(dict(stage='old_prefix_pinned',complete_panels=len(panels),complete_actions=len(complete))),flush=True)
        with ArtifactArchive(old/'transitions.sqlite',read_only=True) as archive:
            # Sequential payload scan avoids 1M repeated indexed lookups; validates
            # the same chain, raw bytes and zlib framing as ArtifactArchive.audit.
            previous='0'*64;count=0;raw_bytes=0
            for index,name,metadata,payload,digest in archive.db.execute('SELECT id,name,metadata,payload,digest FROM artifacts ORDER BY id'):
                m=json.loads(metadata);decoder=zlib.decompressobj();raw=decoder.decompress(payload,archive.maximum+1)
                require(index==count and m['index']==index and m['name']==name and m['previous']==previous
                        and sha(m)==digest and m['complete'] is True and m['bytes']==len(raw)
                        and len(raw)<=archive.maximum and decoder.eof and not decoder.unused_data
                        and hashlib.sha256(raw).hexdigest()==m['raw_sha256'],'Original archive corrupted.')
                count+=1;raw_bytes+=len(raw);previous=digest
            require(count == n*3, 'Missing old transition artifacts.')
            archive_audit=dict(records=count,uncompressed_bytes=raw_bytes,last_sha256=previous)
            prefix={rel:dict(archive_name=name,record_sha256=sha(record_from_archive(archive,name))) for rel,name in mapping.items()}
            for suffix, rel in [('first',f'outcomes/{PARTIAL}/0'),('repeat',f'repeat_checks/{PARTIAL}/repeat')]:
                name=f'panels/{PARTIAL}/{suffix}.json'
                require(prefix[rel]['record_sha256']==sha(read_artifact(old/name,files[name])['record']),
                        'Archive scalar/array decoding differs from saved first/repeat records.')
        plan=dict(schema='ood-desktop-restart-v1',user_direction='had to restart my desktop so it needs starting again.',
            old_attempt=str(old),old_declaration_sha256=file_hash(old/'declaration.json'),old_process_exit=None,
            interruption=boot,original_files_sha256=files,original_archive_sha256=file_hash(old/'transitions.sqlite'),
            original_archive_audit=archive_audit,original_declaration=original,predecessor_binding=pre,
            reused_panels=panels,reused_actions=complete,reused_scientific_counts=accepted_counts,
            partial_owner=PARTIAL,replay_prefix=prefix,replay_outcomes=236,replay_repeats=1,
            original_reserved_counts=scope_counts,resources_before=before,
            original_sources=original['source_sha256'],recovery_source_sha256=file_hash(repo/'experiments/ood/restart.py'),
            tests_source_sha256=file_hash(repo/'experiments/ood/test_restart.py'),
            maximum_new_physical_transitions=924183,maximum_new_physics_steps=3696732,
            maximum_recovery_engineering_transitions=238,maximum_worker_seconds=72000,
            training_updates=0,gpu_allocation=0,device='cpu',no_automatic_retry=True,
            coverage_accepted=False,scientific_acceptance_pending=True)
        write_artifact(attempt/'recovery-plan.json',plan)
        print(json.dumps(dict(prepared=True,recovery_plan_sha256=file_hash(attempt/'recovery-plan.json'),
                             reused_panels=143,reused_actions=1438,old_prefix_records_checked=237)),flush=True)


def execute(repo, attempt):
    from experiments.ood.production import inputs,worker_lease,construct,reserved_step,paired_reset,LEDGER,PAIR
    from experiments.ood.candidate_stage import accepted_states
    from experiments.ood.outcome_stage import accepted_candidates
    from experiments.ood.real_connection import _load_pair
    from experiments.ood.streaming import ArtifactArchive,ArchiveDirectory,FrozenPolicy
    from experiments.ood.resources import ResourceLedger,scope_caps
    acceptance=json.loads((attempt/'execution-acceptance.json').read_text())
    require(acceptance['accepted'] is True, 'Recovery execution not accepted.')
    plan=read_artifact(attempt/'recovery-plan.json',acceptance['plan_sha256']);old=Path(plan['old_attempt']);d=plan['original_declaration']
    require(os.environ.get('JAX_PLATFORMS')=='cpu' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Require original CPU runtime.')
    require(file_hash(repo/'experiments/ood/restart.py')==plan['recovery_source_sha256']
            and file_hash(repo/'experiments/ood/test_restart.py')==plan['tests_source_sha256'],'Recovery sources changed.')
    require(all(file_hash(repo/f)==h for f,h in plan['original_sources'].items()),'Original execution source changed.')
    check_original(plan);boot_check(old)
    binding,full,pair,pre=inputs(repo);require(pre==plan['predecessor_binding'],'Changed training binding.')
    states=Path(d['states']);candidates=Path(d['candidates'])
    sr,sb=accepted_states(repo,states,Path(d['states_actual_exit']));cr,cb=accepted_candidates(repo,candidates,Path(d['candidates_actual_exit']))
    require(sb==d['state_binding'] and cb==d['candidate_binding'],'Changed state/candidate banks.')
    keys=read_artifact(states/'continuation-keys.json',sb['continuation_keys_sha256'])['keys']
    require(shutil.disk_usage(attempt).free>=d['minimum_free_disk_bytes'],'Insufficient disk.')
    sim=None
    with worker_lease(),ResourceLedger(LEDGER,scope_caps([p['pair_id'] for p in full['pairs']],list(full['engineering_streams'])),pre['declaration_sha256']) as ledger:
        require(ledger.audit()==plan['resources_before'],'Original ledger changed since recovery preparation.')
        write_artifact(attempt/'started.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            plan_sha256=acceptance['plan_sha256'],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
        try:
            from runtime.environment import setup
            setup()
            import jax
            require(jax.default_backend()=='cpu','Wrong runtime backend.')
            prior=read_artifact(old/'runtime.json',plan['original_files_sha256']['runtime.json'])
            packages={p:importlib.metadata.version(p) for p in prior['packages']}
            require(packages==prior['packages'],'Changed accepted CPU packages.')
            write_artifact(attempt/'runtime.json',dict(packages=packages,backend=jax.default_backend(),devices=[str(x) for x in jax.devices()]))
            models=_load_pair(repo,Path(d['lane']),binding);policies={k:FrozenPolicy(v) for k,v in models.items()}
            sim,bounds=construct(ledger,attempt,plan['maximum_new_physical_transitions']-1)
            require(sha(bounds)==sha(read_artifact(old/'live-native-bounds.json',plan['original_files_sha256']['live-native-bounds.json'])),'Changed simulator/bounds.')
            paired_reset(sim,d['initialization_reset_seed'])
            write_artifact(attempt/'initialization-state.json',sim.capture())
            counts=dict(outcomes=0,repeat_checks=0);charged=dict(outcomes=0,repeat_checks=0,engineering=1);replayed=set()
            index=[];pending_reuse=dict(plan['reused_actions'])
            with ArtifactArchive(old/'transitions.sqlite',read_only=True) as original_archive,ArtifactArchive(attempt/'transitions.sqlite') as archive:
                sim.evidence_dir=ArchiveDirectory(archive)
                def step(scope,owner,action):
                    require(sum(counts.values())+1<plan['maximum_new_physical_transitions'],'Recovery physical ceiling exhausted.')
                    actual_scope=replay_scope(scope,owner,PARTIAL,236,1)
                    if actual_scope=='engineering':
                        require(charged['engineering']<238,'Recovery overhead ceiling exhausted.')
                        charge='engineering/td3_bc/hopper'
                    else:
                        require(len(replayed)==237,'New scientific outcome requested before exact replay gate.')
                        charge=scope+'/'+PAIR
                    counts[scope]+=1;charged[actual_scope]+=1
                    rec=reserved_step(sim,ledger,attempt,charge,scope+'/'+owner,action,bounds)
                    rel=scope+'/'+owner
                    if actual_scope=='engineering':
                        evidence=plan['replay_prefix'][rel]
                        prior_record=record_from_archive(original_archive,evidence['archive_name'])
                        require(sha(rec)==evidence['record_sha256']==sha(prior_record),'Restart prefix differs from saved record; stop.')
                        replayed.add(rel)
                        if len(replayed)==237:
                            write_artifact(attempt/'replay-gate.json',dict(accepted=True,exact_record_matches=237,
                                outcome_prefix_steps=236,repeat_prefix_steps=1,changed_tolerances=False))
                    return rec
                def save(name,value):
                    path=attempt/'panels'/(name+'.json');path.parent.mkdir(parents=True,exist_ok=True);write_artifact(path,value)
                for cont in d['continuations']:
                    for item in sr['rows']:
                        i=item['index'];owner=f'{cont}/state{i:03d}'
                        if owner in plan['reused_panels']:
                            e=plan['reused_panels'][owner];result=read_artifact(Path(e['path']),e['sha256'])
                            index.append(dict(owner=owner,status='reused_completed',**e))
                            for slot in range(10):pending_reuse.pop(f'{owner}/slot{slot}')
                        else:
                            state=read_artifact(states/'states'/f'state-{i:03d}.json',item['sha256'])
                            row=read_artifact(candidates/'rows'/f'candidate-{i:03d}.json',cr['rows'][i]['sha256'])
                            require(state['state_id']==row['state_id'] and state['status']==row['status'],'Changed pairing.')
                            reused={}
                            for slot in range(10):
                                e=pending_reuse.pop(f'{owner}/slot{slot}',None)
                                if e:reused[slot]=read_artifact(Path(e['path']),e['sha256'])
                            if state['status']=='missing':result=dict(owner=owner,status='missing',state_id=state['state_id'])
                            else:
                                actions=read_artifact(candidates/'rows'/f'actions-{i:03d}.json',row['action_file_sha256'])
                                require(state['snapshot_sha256']==row['snapshot_sha256']==sha(state['snapshot']),'Changed state.')
                                result=finish_panel(sim,policies[cont],state['snapshot'],actions,keys[i],owner,step,save,reused=reused)
                                result.update(status='completed',state_id=state['state_id'],continuation=cont)
                            path=attempt/'panels'/(owner+'/panel.json');path.parent.mkdir(parents=True,exist_ok=True)
                            digest=write_artifact(path,result);index.append(dict(owner=owner,status=result['status'],path=str(path),sha256=digest))
                        progress=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),panels_closed=len(index),total_panels=512,
                            reused_panels=143,reused_actions=1438,new_physical_counts=counts.copy(),charged_counts=charged.copy(),
                            last_panel=owner,scientific_acceptance_pending=True)
                        tmp=attempt/'progress.tmp';tmp.write_text(json.dumps(progress)+'\n');tmp.replace(attempt/'progress.json')
                require(not pending_reuse and len(replayed)==237,'Reuse or replay gate incomplete.')
                archive_audit=archive.audit()
            for p in policies.values():p.seal()
            check_original(plan)
            require(all(file_hash(repo/f)==h for f,h in plan['original_sources'].items()),'Original source changed during recovery.')
            write_artifact(attempt/'completion.json',dict(schema='ood-recovery-worker-completion-v1',pair_id=PAIR,panels=index,
                reused_actions=plan['reused_actions'],new_physical_counts=counts,charged_counts=charged,
                original_reserved_counts=plan['original_reserved_counts'],replayed_prefix_records=237,
                original_partial_retained_excluded=True,environment_transitions=1+sum(counts.values()),
                physics_steps=4*(1+sum(counts.values())),resources_after=ledger.audit(),archive_audit=archive_audit,
                archive_sha256=file_hash(attempt/'transitions.sqlite'),recovery_plan_sha256=acceptance['plan_sha256'],
                actual_exit_required_separately=True,independent_scientific_audit_required=True,coverage_accepted=False))
            print(json.dumps(dict(worker_panels_completed=len(index),reused_panels=143,actual_exit_required=True)),flush=True)
        except BaseException as exc:
            write_artifact(attempt/'failure.json',dict(error=repr(exc),resources_after=ledger.audit(),no_automatic_retry=True));raise
        finally:
            if sim is not None:sim.close()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','execute'])
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--old',type=Path)
    a=p.parse_args()
    if a.phase=='prepare':prepare(a.repo,a.old,a.attempt)
    else:execute(a.repo,a.attempt)
