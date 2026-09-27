"""Fixed action consequences from complete saved states; no learning/scoring."""
import numpy as np
from experiments.ood.collect import require,sha


def panel(sim,policy,snapshot,actions,step_keys,owner,step,save):
    """One state/continuation panel. Callbacks reserve before every physical step.

Keys are the saved real JAX bank. Deterministic TD3/ReBRAC actors do not consume
randomness. This driver cannot accept a stochastic actor or grant execution gates.
One extra first transition per unique action is charged to repeat_checks. Every
first record AND complete end state must agree exactly before continuation.
"""
    sent,applied=np.asarray(actions['sent']),np.asarray(actions['applied'])
    aliases=actions['alias'];keys=np.asarray(step_keys)
    require(sent.shape==applied.shape and sent.ndim==2 and len(sent)==10
            and sent.dtype==applied.dtype==np.float32 and np.isfinite(sent).all()
            and (abs(sent)<=1).all(),'Invalid frozen first-action bank.')
    require(keys.shape==(250,2) and keys.dtype==np.uint32,'Require the frozen 250-step JAX key bank.')
    expected=[next((j for j in range(i) if np.array_equal(applied[i],applied[j])),i) for i in range(10)]
    require(aliases==expected,'Changed aliases.')
    state_sha=sha(snapshot);result=[]
    for slot,action in enumerate(sent):
        if aliases[slot]!=slot:
            result.append(dict(slot=slot,alias=aliases[slot],source_slot=aliases[slot]))
            continue
        policy.seal()
        name=f'{owner}/slot{slot}'
        save(name+'/started',dict(snapshot_sha256=state_sha,sent=action,expected_applied=applied[slot],step_keys_sha256=sha(keys)))
        sim.restore(snapshot,expected_sha256=sim.snapshot_hash(snapshot))
        require(sha(sim.capture())==state_sha,'First-action restore differs.')
        first=step('outcomes',name+'/0',action);first_end=sim.capture()
        save(name+'/first',dict(record=first,after=first_end))
        require(np.array_equal(first['applied_action'],applied[slot]),'Frozen first action differs from actual applied action.')
        sim.restore(snapshot,expected_sha256=sim.snapshot_hash(snapshot))
        repeat=step('repeat_checks',name+'/repeat',action);repeat_end=sim.capture()
        save(name+'/repeat',dict(record=repeat,after=repeat_end))
        require(sha(first)==sha(repeat) and sha(first_end)==sha(repeat_end),'Repeated first transition/state differs; stop.')
        sim.restore(first_end,expected_sha256=sim.snapshot_hash(first_end))
        rewards=[float(first['reward'])];record=first
        for t in range(1,250):
            if record['terminated'] or record['truncated']:break
            next_action=policy.actions(np.asarray(record['observation'])[None,:])[0]
            record=step('outcomes',name+f'/{t}',next_action)
            rewards.append(float(record['reward']))
        policy.seal()
        row=dict(slot=slot,alias=slot,source_slot=slot,rewards=np.asarray(rewards,np.float64),
            raw_return=float(sum(rewards)),length=len(rewards),terminated=bool(record['terminated']),
            truncated=bool(record['truncated']),horizon_exhausted=len(rewards)==250,
            first_transition_sha256=sha(first),first_end_sha256=sha(first_end),last_record_sha256=sha(record),
            complete_end_state_sha256=sha(sim.capture()),repeat_exact=True,
            continuation_keys_sha256=sha(keys),deterministic_actor_keys_unused=True)
        save(name+'/completed',row);result.append(row)
    require(sha(snapshot)==state_sha,'Source snapshot mutated.')
    return dict(owner=owner,slots=result,unique_applied_actions=sum(i==j for i,j in enumerate(aliases)),
                reference_slot=0,raw_undiscounted_no_tail=True)
