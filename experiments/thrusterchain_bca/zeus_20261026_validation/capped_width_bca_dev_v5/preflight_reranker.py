"""No-simulation preflight of all six real frozen models and full score shape."""
import json
import time
import numpy as np
import reranker_runner as r


def main():
    started=time.monotonic()
    pins,manifest=r.input_pins(); c=r.initialize(reconstruct=True)
    _,reference,_=r.frozen.load_simulator()
    ledger=reference.new_ledger(1,10000)
    ledger.update(max_critic_rows=5760,candidate_scoring_rows_attempted=0,candidate_scoring_rows_completed=0)
    states=np.repeat(c['obs'][:16,:13].astype(np.float64)[None,:,:],5,axis=0)
    cases=[]
    for seed in r.old.SEEDS:
        for arm in r.old.ARMS:
            item=c['items'][(seed,arm)]
            cap=next(x for x in manifest['cases'] if x['seed']==seed and x['arm']==arm)
            actions,info=r.select_step(c,item,reference,states,25,np.asarray(cap['caps']),ledger,
                                       time.monotonic()+60)
            assert actions.shape==(5,16,4) and all(np.isfinite(a).all() for a in info.values())
            assert r.runtime.tree_digest(item['state'])==item['before']==cap['before']
            cases.append({'seed':seed,'arm':arm,'candidate_rows':960,'checkpoint_digest':item['before'],
                          'guard_on':int(info['guards'].sum()),
                          'selected_cap_saturated':int(info['clip_saturated'].sum())})
            print('preflight',seed,arm,flush=True)
    assert ledger['critic_rows_attempted']==ledger['critic_rows_completed']==5760
    assert ledger['candidate_scoring_rows_attempted']==ledger['candidate_scoring_rows_completed']==5760
    assert ledger['transition_steps_attempted']==0 and ledger['optimizer_updates']==0
    result={'status':'passed_no_simulator_steps_or_optimizer_updates','cases':cases,
            'inputs':pins,'source_sha256':r.sha(r.HERE/'reranker_runner.py'),
            'ledger':ledger,'elapsed_seconds':time.monotonic()-started,
            'scope':'retained-training observations, altered to common time 25, plumbing only'}
    r.json_write(r.HERE/'RERANKER_PREFLIGHT.json',result)
    print(json.dumps({'status':result['status']}),flush=True)


if __name__=='__main__':main()
