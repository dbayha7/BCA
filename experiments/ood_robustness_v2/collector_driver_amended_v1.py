"""Injected fixed capture schedule. Missing/terminal captures are never replaced."""
import copy
import numpy as np
from precommit_bank_amended_v1 import capture_schedule
from recorded_step import require,content_hash


def collect_states(sim,policies,pair,step,store,paired_reset):
    require(set(policies)=={'host','bca'},'Both declared collectors required.')
    schedule=capture_schedule(*pair);rows=[]
    for collector in ('host','bca'):
        policy=policies[collector]
        for block in range(64):
            expected=[r for r in schedule if r['collector']==collector and r['reset_block']==block]
            policy.seal();obs=np.asarray(paired_reset(sim,expected[0]['reset_seed'])).copy()
            initial=sim.capture()
            require(initial['elapsed_steps']==0 and not initial['terminated'] and not initial['truncated'],'Fresh paired reset failed.')
            terminal=None;captures={}
            for t in range(1,301):
                action=policy.actions(obs[None,:])[0]
                result=step('collection',f'collection/{collector}/block{block:02d}/step{t:03d}',action)
                rec=result['record'];obs=np.asarray(rec['observation']).copy()
                require(rec['elapsed_steps']==t,'Collection elapsed steps differ from nominal schedule.')
                if rec['terminated'] or rec['truncated']:
                    terminal=dict(elapsed_steps=t,terminated=rec['terminated'],truncated=rec['truncated'],
                                  full_state_sha256=content_hash(result['after']),call_evidence_sha256=result['evidence_sha256'])
                    break
                if t in (100,300):
                    snapshot=copy.deepcopy(sim.capture())
                    require(content_hash(snapshot)==content_hash(result['after']),'Capture changed after recorded step.')
                    captures[t]=dict(snapshot=snapshot,observation=obs.copy(),snapshot_content_sha256=content_hash(snapshot),
                                     call_evidence_sha256=result['evidence_sha256'])
            policy.seal()
            for nominal in expected:
                row=dict(nominal)
                if nominal['capture_step'] in captures:
                    row.update(status='captured',**captures[nominal['capture_step']])
                else:
                    require(terminal is not None,'Unexplained missing scheduled capture.')
                    row.update(status='missing',reason='native_episode_ended_before_or_at_capture',terminal=terminal)
                row['artifact_sha256']=store.put(f"states/state{row['state_index']:03d}",row)
                rows.append(row)
    require([r['state_index'] for r in rows]==list(range(256)),'Capture rows reordered/lost.')
    return dict(rows=rows,captured=sum(r['status']=='captured' for r in rows),missing=sum(r['status']=='missing' for r in rows),
                reset_blocks=64,collectors=2,captures=[100,300],replacement_states=0,scientific_acceptance=False)
