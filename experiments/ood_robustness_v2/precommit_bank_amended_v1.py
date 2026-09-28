"""V2 outcome-blind precommit inputs; no model/simulator/resource writer imports.

External inputs must come from independently bound accepted files at integration.
This module does not certify receipt provenance, completeness of old stream
inventories, JAX key derivation, native runtime bounds or checkpoint acceptance.
"""
import hashlib
AMENDMENT_SHA256 = '188395c83b3558c99a8671146128c1e58ee189fa0e19f1715cca08ee7fe4359b'
import numpy as np
from candidate_design_amended_v1 import stream_seed, proposal_pool, select_bands, applied_unit, distances, band
from precommit_file import canonical

PLAN_SHA256='f11ebe8f8e3ccc4af10511ed1a241e8f7c5f0aa9f84770103dd6928e59b60756'
PIN_NAMES=('state_snapshot','training_acceptance','host_checkpoint','bca_checkpoint',
           'prepared_data','support_bank','source_manifest','runtime',
           'execution_declaration','stream_inventory')


def _hash(value): return hashlib.sha256(canonical(value)).hexdigest()


def _sha(value):
    if type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('External SHA256 pin required.')
    return value


def _array(value):
    a=np.asarray(value)
    return dict(dtype=a.dtype.str,shape=list(a.shape),hex=a.tobytes(order='C').hex())


def _array_hash(value):
    # Bound large key tables without the small JSON artifact byte limit.
    a=np.asarray(value)
    header=canonical(dict(dtype=a.dtype.str,shape=list(a.shape)))
    return hashlib.sha256(header+b'\0'+a.tobytes(order='C')).hexdigest()


def capture_schedule(host,environment,training_seed):
    stream_seed(host,environment,training_seed,'reset',0)  # strict protocol identity
    rows=[]
    for collector in ('host','bca'):
        for reset_block in range(64):
            for capture_step in (100,300):
                i=len(rows)
                rows.append(dict(state_index=i,collector=collector,reset_block=reset_block,
                    capture_step=capture_step,reset_seed=stream_seed(host,environment,training_seed,'reset',reset_block),
                    **{p+'_seed':stream_seed(host,environment,training_seed,p,i) for p in ('candidate','continuation','random_score')}))
    return rows


def planned_streams(old_seeds):
    if (not isinstance(old_seeds,(list,tuple)) or not 0<len(old_seeds)<=200_000
            or any(type(v) is not int or not 0<=v<2**32 for v in old_seeds)):
        raise ValueError('Nonempty externally inventoried predecessor uint32 seeds required.')
    values=[stream_seed(h,e,s,p,i) for h in ('td3_bc','rebrac') for e in ('hopper','walker2d')
            for s in range(202609171,202609176) for p in ('reset','candidate','continuation','random_score')
            for i in range(64 if p=='reset' else 256)]
    if len(set(values))!=len(values) or set(values)&set(old_seeds):
        raise ValueError('Fresh seed collision; no resampling permitted.')
    return dict(seeds=values,count=len(values),old_inventory_sha256=_hash(list(old_seeds)),
                predecessor_inventory_completeness_accepted=False,execution_accepted=False)


def _keys(value,minimum,maximum):
    a=np.asarray(value)
    if a.dtype!=np.uint32 or a.ndim!=2 or a.shape[1]!=2 or not minimum<=len(a)<=maximum:
        raise ValueError('Bounded uint32 key pairs required.')
    return a


def bind_keys(pair,keys,old_keys,prior_v2_keys):
    capture_schedule(*pair)
    k=np.asarray(keys)
    if k.dtype!=np.uint32 or k.shape!=(256,250,2): raise ValueError('Exact256x250x2 uint32 key table required.')
    old=_keys(old_keys,1,2_000_000);prior=_keys(prior_v2_keys,0,2_000_000)
    keyset=lambda a:{(int(x),int(y)) for x,y in a.reshape(-1,2)}
    fresh=keyset(k)
    if len(fresh)!=256*250 or fresh&keyset(old) or fresh&keyset(prior):
        raise ValueError('Continuation key collision; no resampling permitted.')
    return dict(schema='ood-v2-key-table-binding-v1',pair=list(pair),shape=list(k.shape),
                table_sha256=_array_hash(k),row_sha256=[_array_hash(row) for row in k],
                old_keys_sha256=_array_hash(old),prior_v2_keys_sha256=_array_hash(prior),
                key_derivation_accepted=False,predecessor_inventory_completeness_accepted=False,
                execution_accepted=False)


def _scalar(value,positive=False):
    if (isinstance(value,(bool,str,complex,np.complexfloating)) or not np.isscalar(value)
            or not np.isfinite(value) or value<0 or (positive and value==0)):
        raise ValueError('Finite nonnegative support scalar required.')
    return float(value)


def _action(value,size):
    a=np.asarray(value)
    if a.dtype!=np.float32 or a.shape!=(size,) or not np.isfinite(a).all() or (abs(a)>1).any():
        raise ValueError('Exact bounded native-dimensional float32 sent action required.')
    return a


def build_precommit(*,pair,state_index,pins,key_receipt,keys,anchor_sent,neighboring_actions,
                    host_sent,bca_sent,q95,q99,state_distance,state_q95):
    schedule=capture_schedule(*pair)
    if type(state_index) is not int or not 0<=state_index<256: raise ValueError('Declared nominal state index required.')
    if type(pins) is not dict or set(pins)!=set(PIN_NAMES): raise ValueError('Exact external input pin set required.')
    for value in pins.values(): _sha(value)
    key_fields={'schema','pair','shape','table_sha256','row_sha256','old_keys_sha256','prior_v2_keys_sha256',
                'key_derivation_accepted','predecessor_inventory_completeness_accepted','execution_accepted'}
    if (type(key_receipt) is not dict or set(key_receipt)!=key_fields
            or any(key_receipt[n] is not False for n in ('key_derivation_accepted','predecessor_inventory_completeness_accepted','execution_accepted'))
            or key_receipt.get('schema')!='ood-v2-key-table-binding-v1' or key_receipt.get('pair')!=list(pair)
            or key_receipt.get('shape')!=[256,250,2] or len(key_receipt.get('row_sha256',[]))!=256):
        raise ValueError('Externally bound pair key receipt required.')
    for n in ('table_sha256','old_keys_sha256','prior_v2_keys_sha256'):
        _sha(key_receipt.get(n))
    for h in key_receipt['row_sha256']: _sha(h)
    k=_keys(keys,250,250)
    if _array_hash(k)!=key_receipt['row_sha256'][state_index]: raise ValueError('State key row disagrees with pair precommit.')
    q95,q99=_scalar(q95,True),_scalar(q99,True)
    band(0.,q95,q99)
    sd,sq=_scalar(state_distance),_scalar(state_q95,True)
    size=3 if pair[1]=='hopper' else 6
    anchor,host,bca=(_action(a,size) for a in (anchor_sent,host_sent,bca_sent))
    neighbors=np.asarray(neighboring_actions)
    if neighbors.shape!=(32,size) or neighbors.dtype!=np.float32 or not np.isfinite(neighbors).all() or (abs(neighbors)>1).any():
        raise ValueError('Exactly32 bound native neighbor actions required.')
    if not np.array_equal(anchor,neighbors[0]):
        raise ValueError('Anchor must be the first distance-ordered nearest recorded action.')
    pool=proposal_pool(anchor,schedule[state_index]['candidate_seed'])
    selection=select_bands(anchor,pool,neighbors,q95,q99)
    rows=[];actual=[]
    def add(role,proposed,sent,proposal_index):
        applied=applied_unit(sent);d=float(distances(applied[None],neighbors)[0]);slot=len(rows)
        owner=next((j for j,a in actual if np.array_equal(a,applied)),slot)
        rows.append(dict(slot=slot,role=role,status='present',alias_of=owner,
            proposed=_array(proposed),sent=_array(sent),applied=_array(applied),
            clipped=[bool(x) for x in (abs(proposed)>1)],distance=d,band=band(d,q95,q99),
            proposal_index=proposal_index))
        actual.append((slot,applied.copy()))
    for label in ('near','moderate','strong'):
        for j in range(4):
            selected=selection['selected'][label]
            if j>=len(selected):
                rows.append(dict(slot=len(rows),role=label,status='missing',alias_of=None,
                                 reason='fixed_pool_band_quota_unfilled'))
                continue
            index=selected[j]['proposal_index']
            add(label,anchor if index==-1 else pool['proposed'][index],
                anchor if index==-1 else pool['sent'][index],index)
    add('host_policy',host,host,None);add('bca_policy',bca,bca,None)
    return dict(schema='ood-v2-state-precommit-v1',protocol_sha256=PLAN_SHA256,amendment_sha256=AMENDMENT_SHA256,pair=list(pair),
        capture=schedule[state_index],pins=dict(pins),key_receipt_sha256=_hash(key_receipt),keys=_array(k),
        neighbor_actions_sha256=_array_hash(neighbors),state_distance=sd,state_q95=sq,
        action_q95=q95,action_q99=q99,state_support='near' if sd<=sq else 'distant',
        slots=rows,anchor_slot=0,host_reference_slot=12,bca_policy_slot=13,
        unique_trajectory_slots=[r['slot'] for r in rows if r['status']=='present' and r['alias_of']==r['slot']],
        missing_support_slots=dict(selection['missing']),support_quotas_complete=selection['complete'],
        pool_size=8192,duplicate_proposals=selection['duplicate_proposals'],
        empirical_support_only=True,score_or_outcome_access=False,execution_accepted=False)


def check_precommit(value,**externally_bound_inputs):
    expected=build_precommit(**externally_bound_inputs)
    if canonical(value)!=canonical(expected): raise ValueError('Precommit differs from independently supplied bound inputs.')
    return _hash(value)
