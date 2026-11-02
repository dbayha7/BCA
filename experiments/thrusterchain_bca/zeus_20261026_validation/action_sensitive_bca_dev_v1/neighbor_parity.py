"""Bounded real-input host/JAX neighbor parity check; no updates or simulation.

The input archive and all saved anchor/probe descriptors are authenticated by
the original source-data reader, frozen data plan, and completed evaluation.
This is an implementation check on development-consumed data, not a new
scientific evaluation. It must run on CPU while other GPU work is active.
"""
from pathlib import Path
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
RERUN = HERE.parent/'central_quartile_training_v1_evaluation_rerun1'
sys.path.insert(0, str(RERUN))
import data
sys.path.insert(0, str(HERE))
import compatibility as compatibility


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(data_bytes):
    return hashlib.sha256(data_bytes).hexdigest()


def artifact(base, desc):
    require(set(desc) == {'path','bytes','sha256'}, 'bad artifact descriptor')
    rel = Path(desc['path'])
    require(not rel.is_absolute() and len(rel.parts) == 1 and ':' not in str(rel), 'unsafe artifact path')
    path=base/rel
    require(path.is_file() and not path.is_symlink(), 'missing or linked artifact')
    raw=path.read_bytes()
    require(len(raw)==desc['bytes'] and sha(raw)==desc['sha256'], 'artifact hash mismatch')
    return raw


def main():
    require(os.environ.get('JAX_PLATFORMS')=='cpu', 'explicit CPU backend required')
    import jax
    import jax.numpy as jnp
    require(jax.default_backend()=='cpu', 'CPU backend required')
    started=time.monotonic()
    terminal=json.loads((RERUN/'RUN_RESULT.json').read_bytes())
    require(terminal['status']=='completed_evaluation_rerun' and terminal['prepared_identity_unchanged'],
            'completed source evaluation required')
    raw=(RERUN/'run/evaluation/report.json').read_bytes()
    require(sha(raw)==terminal['jobs'][0]['report_sha256'], 'evaluation report hash mismatch')
    report=json.loads(raw)
    require(report['reproduction']['tier1_exact']==143 and report['reproduction']==terminal['reproduction'],
            'source reproduction differs')
    source=data._read_source()
    plan,selections,_,_=data._derive(source)
    expected=json.loads((RERUN/'DATA_PLAN.json').read_bytes()); expected.pop('passed_gate_evidence',None)
    require(data._canonical(plan)==data._canonical(expected),'frozen data plan mismatch')
    nt=source['train_count']; arrays=source['arrays']; selection=selections['hole']
    obs_all=arrays['observations'][:nt,:-1].reshape(-1,14)
    actions_all=arrays['actions'][:nt].reshape(-1,4)
    train_ix=selection['train']; obs=obs_all[train_ix]; actions=actions_all[train_ix]
    ids=data._source_ids(train_ix,source['trajectory_ids'][:nt]); eps=ids//100; times=ids%100
    require(len(ids)==67108 and len(np.unique(eps))==768, 'unexpected train identities')
    ref=compatibility.fit_reference(obs,actions,eps,times,row_ids=ids,k=64)
    jref=compatibility.jax_reference(ref)
    fn=jax.jit(lambda o,a:compatibility.compatibility_from_observations_jax(
        jref,o,a,k=64,return_neighbors=True))
    output=RERUN/'run/evaluation'; cases=[]
    for context in ('ordinary','strong_current'):
        meta=json.loads(artifact(output,report['snapshot_results'][context+'_bank']))
        bank_bytes=artifact(output,meta['arrays'])
        with np.load(io.BytesIO(bank_bytes),allow_pickle=False) as z:
            qobs=z['observations'].copy(); qt=z['times'].copy()
        require(qobs.shape==(32,14) and qt.shape==(32,), 'bad anchor shapes')
        host=compatibility.query_neighbors(ref,qobs,qt)
        _,jids,_,valid=fn(jnp.asarray(qobs),jnp.zeros((32,4),jnp.float32))
        jids=np.asarray(jids,dtype=np.int64)
        require(np.asarray(valid).all(), 'JAX neighbor lookup invalid')
        row_to_index={int(r):i for i,r in enumerate(ref.row_ids)}
        lookup=np.array([[row_to_index[int(r)] for r in rr] for rr in jids],np.int64)
        jacts=ref.actions[lookup]
        overlap=[len(set(h)&set(j)) for h,j in zip(host['row_ids'],jids)]
        mismatch=sum(not np.array_equal(h,j) for h,j in zip(host['row_ids'],jids))
        membership=sum(set(h)!=set(j) for h,j in zip(host['row_ids'],jids))
        # Score a fixed, outcome-independent bank: one retained training action
        # and five declared fixed vectors. No cached Q or return labels are read.
        act=np.zeros((32,6,4),np.float32)
        act[:,0,:]=actions[:32]
        act[:,1:,0]=np.asarray([-.5,-.25,0.,.25,.5],np.float32)
        flat=jnp.asarray(act.reshape(-1,4)); bw=jnp.asarray(ref.action_bandwidth,np.float32)
        hn=jnp.asarray(np.repeat(host['actions'],6,axis=0)); jn=jnp.asarray(np.repeat(jacts,6,axis=0))
        hcost=np.asarray(compatibility.compatibility_cost_jax(flat,hn,bw))
        jcost=np.asarray(compatibility.compatibility_cost_jax(flat,jn,bw))
        score_diffs=np.abs(hcost-jcost).tolist()
        cases.append({'context':context,'anchor_rows':32,'bank_sha256':sha(bank_bytes),
                      'ordered_neighbor_mismatches':mismatch,'membership_mismatches':membership,
                      'minimum_neighbor_overlap':min(overlap),
                      'candidate_action_scores_compared':len(score_diffs),
                      'max_abs_compatibility_difference':float(np.max(score_diffs)),
                      'mean_abs_compatibility_difference':float(np.mean(score_diffs))})
    # First 64 outcome-independent retained training probes; strict fold-heldout
    # statistics and neighbors, not merely leave-one-row-out selection.
    lookup={int(v):i for i,v in enumerate(ids)}
    indices=np.asarray([lookup[int(x)] for x in plan['conditions']['hole']['gradient_probe_source_row_ids'][:64]])
    host=compatibility.crossfit_query(obs,actions,eps,times,indices,row_ids=ids,k=64)
    fmap=compatibility.episode_folds(eps,folds=5)
    row_fold=np.array([fmap[int(e)] for e in eps]); jids=np.empty((len(indices),64),np.int64)
    for fold in np.unique(row_fold[indices]):
        train=row_fold!=fold; pos=np.flatnonzero(row_fold[indices]==fold)
        fold_ref=compatibility.fit_reference(obs[train],actions[train],eps[train],times[train],row_ids=ids[train],k=64)
        require(fold_ref.fingerprint_sha256==host['reference_sha256_by_fold'][str(int(fold))], 'fold digest differs')
        jfold=compatibility.jax_reference(fold_ref)
        f=jax.jit(lambda o,a:compatibility.compatibility_from_observations_jax(jfold,o,a,k=64,return_neighbors=True))
        _,rows,_,valid=f(jnp.asarray(obs[indices[pos]]),jnp.asarray(actions[indices[pos]]))
        require(np.asarray(valid).all(), 'crossfit JAX lookup invalid')
        jids[pos]=np.asarray(rows,np.int64)
    cross={'query_rows':len(indices),'ordered_neighbor_mismatches':sum(not np.array_equal(h,j) for h,j in zip(host['row_ids'],jids)),
           'membership_mismatches':sum(set(h)!=set(j) for h,j in zip(host['row_ids'],jids)),
           'minimum_neighbor_overlap':min(len(set(h)&set(j)) for h,j in zip(host['row_ids'],jids)),
           'fold_reference_sha256':host['reference_sha256_by_fold']}
    result={'status':'completed_cpu_neighbor_parity','jax':jax.__version__,'backend':jax.default_backend(),
            'reference_sha256':ref.fingerprint_sha256,'training_dataset_sha256':source['dataset_sha256'],
            'evaluation_report_sha256':sha(raw),'anchor_cases':cases,'strict_crossfit':cross,
            'simulator_transitions':0,'optimizer_updates':0,'elapsed_seconds':time.monotonic()-started}
    path=HERE/'neighbor_parity_result.json'
    with path.open('x',encoding='utf8',newline='\n') as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__': main()
