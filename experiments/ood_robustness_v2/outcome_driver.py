"""Injected v2 finite-horizon panel driver, preserving all14 nominal slots."""
import copy
import hashlib
import numpy as np
from precommit_file import canonical
from precommit_bank import PLAN_SHA256,capture_schedule
from candidate_design import applied_unit
from recorded_step import require,content_hash


def _array(v,dtype,shape):
    require(type(v) is dict and set(v)=={'dtype','shape','hex'} and v['dtype']==np.dtype(dtype).str and v['shape']==list(shape),'Invalid precommitted array schema.')
    require(type(v['hex']) is str and len(v['hex'])==int(np.prod(shape))*np.dtype(dtype).itemsize*2,'Invalid precommitted array bytes.')
    a=np.frombuffer(bytes.fromhex(v['hex']),dtype=dtype).reshape(shape).copy()
    require(np.isfinite(a).all(),'Nonfinite precommitted array.')
    return a


def run_panel(*,sim,policy,snapshot,bank,bank_sha256,snapshot_sha256,keys,continuation,step,store):
    require(continuation in ('host','bca'),'Unknown continuation.')
    require(hashlib.sha256(canonical(bank)).hexdigest()==bank_sha256,'External precommit content hash mismatch.')
    require(content_hash(snapshot)==snapshot_sha256,'External snapshot content hash mismatch.')
    require(not snapshot['terminated'] and not snapshot['truncated'],'Terminal initial state cannot receive a first action.')
    require(bank['schema']=='ood-v2-state-precommit-v1' and bank['protocol_sha256']==PLAN_SHA256,'Wrong protocol precommit.')
    index=bank['capture']['state_index'];schedule=capture_schedule(*bank['pair'])
    require(type(index) is int and 0<=index<256 and bank['capture']==schedule[index],'Wrong nominal capture.')
    require(type(snapshot['elapsed_steps']) is int and snapshot['elapsed_steps']==bank['capture']['capture_step'],'Snapshot is not at the precommitted capture step.')
    require(bank['anchor_slot']==0 and bank['host_reference_slot']==12 and bank['bca_policy_slot']==13,'Changed anchor/reference slots.')
    size=3 if bank['pair'][1]=='hopper' else 6
    keys=np.asarray(keys).copy()
    require(keys.dtype==np.uint32 and keys.shape==(250,2) and np.array_equal(keys,_array(bank['keys'],np.uint32,(250,2))),'Wrong continuation keys.')
    slots=bank['slots'];require(len(slots)==14 and [r['slot'] for r in slots]==list(range(14)),'Wrong nominal action slots.')
    actions={};owners={}
    for i,row in enumerate(slots):
        role='near' if i<4 else 'moderate' if i<8 else 'strong' if i<12 else 'host_policy' if i==12 else 'bca_policy'
        require(row['role']==role,'Wrong fixed action role.')
        if row['status']=='missing':
            require(i not in (0,12,13) and row['alias_of'] is None and row['reason']=='fixed_pool_band_quota_unfilled','Invalid missing slot.')
            continue
        require(row['status']=='present','Unrecognized action status.')
        sent=_array(row['sent'],np.float32,(size,));applied=_array(row['applied'],np.float32,(size,))
        require((abs(sent)<=1).all() and np.array_equal(applied,applied_unit(sent)),'Changed action transform.')
        owner=next((j for j in actions if np.array_equal(applied,actions[j][1])),i)
        require(row['alias_of']==owner,'Changed exact alias ownership.')
        actions[i]=(sent,applied);owners[i]=owner
    require(bank['unique_trajectory_slots']==[i for i in actions if owners[i]==i],'Changed trajectory index.')
    original_bank=bank_sha256;rows=[];owner_results={};owner_hashes={}
    owner=f'{continuation}/state{index:03d}'
    def restore(state):
        sim.restore(state,expected_sha256=sim.snapshot_hash(state))
        require(content_hash(sim.capture())==content_hash(state),'Full restored state differs.')
    for slot,row in enumerate(slots):
        name=f'panels/{owner}/slot{slot}'
        if slot not in actions:
            result=dict(slot=slot,status='missing',reason=row['reason'])
            store.put(name+'/missing',result);rows.append(result);continue
        source=owners[slot]
        if source!=slot:
            require(source in owner_results,'Alias owner not completed.')
            result=dict(slot=slot,status='alias',source_slot=source,source_result_sha256=owner_hashes[source])
            store.put(name+'/alias',result);rows.append(result);continue
        policy.seal();sent,applied=actions[slot]
        store.put(name+'/started',dict(slot=slot,precommit_content_sha256=original_bank,snapshot_content_sha256=snapshot_sha256,
                  continuation=continuation,keys=keys.copy(),deterministic_native_actor_keys_unused=True))
        restore(snapshot)
        first=step('outcomes',f'{owner}/slot{slot}/0',sent);first_record=first['record'];first_end=copy.deepcopy(first['after'])
        store.put(name+'/first',first)
        require(np.array_equal(first_record['applied_action'],applied),'First applied action differs from bank.')
        restore(snapshot)
        repeat=step('repeat_checks',f'{owner}/slot{slot}/repeat',sent);store.put(name+'/repeat',repeat)
        require(content_hash(first_record)==content_hash(repeat['record']) and content_hash(first_end)==content_hash(repeat['after']),'Repeated first record/full state mismatch.')
        restore(first_end);current=first;rewards=[float(first_record['reward'])]
        for t in range(1,250):
            rec=current['record']
            if rec['terminated'] or rec['truncated']: break
            action=policy.actions(np.asarray(rec['observation'])[None,:])[0]
            current=step('outcomes',f'{owner}/slot{slot}/{t}',action)
            rewards.append(float(current['record']['reward']))
        final=copy.deepcopy(sim.capture())
        require(content_hash(final)==content_hash(current['after']),'Final state changed after archived physical output.')
        policy.seal();record=current['record']
        result=dict(slot=slot,status='completed',source_slot=slot,rewards=np.asarray(rewards,np.float64),raw_return=float(sum(rewards)),
            length=len(rewards),terminated=record['terminated'],truncated=record['truncated'],horizon_exhausted=len(rewards)==250,
            first_record_sha256=content_hash(first_record),first_end_sha256=content_hash(first_end),last_record_sha256=content_hash(record),
            final_state=final,final_state_sha256=content_hash(final),last_call_evidence_sha256=current['evidence_sha256'],
            repeat_exact=True,raw_undiscounted_no_tail=True,keys_sha256=content_hash(keys),deterministic_actor_keys_unused=True)
        owner_hashes[slot]=store.put(name+'/completed',result);owner_results[slot]=result;rows.append(result)
        require(content_hash(snapshot)==snapshot_sha256 and hashlib.sha256(canonical(bank)).hexdigest()==original_bank,'Source state or action bank mutated.')
    return dict(owner=owner,slots=rows,unique_applied_actions=len(owner_results),reference_slot=12,anchor_slot=0,
                complete_final_state_contents_saved=True,scientific_acceptance=False)
