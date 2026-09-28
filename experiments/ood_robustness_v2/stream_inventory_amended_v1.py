"""Externally pinned declared-stream inventory. No seed repair or JAX imports."""
import hashlib
import json
from candidate_design_amended_v1 import stream_seed

PAIRS=[(h,e,s) for h in ('td3_bc','rebrac') for e in ('hopper','walker2d') for s in range(202609171,202609176)]
STREAM_SIZES=dict(preparation=2,collection=64,candidates=256,random_score=256,continuation=256,
    fresh_calibration=200,test=500,calibration_transition_selection=200,test_transition_selection=500,
    calibration_target_noise=200,test_target_noise=500)

def require(ok,message):
    if not ok:raise ValueError(message)

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def parse_bound(raw,pin):
    require(type(raw) is bytes and 0<len(raw)<=25_000_000,'Bounded declaration bytes required.')
    require(type(pin) is str and len(pin)==64 and hashlib.sha256(raw).hexdigest()==pin,'External declaration hash differs.')
    def pairs(items):
        result={}
        for k,v in items:
            require(k not in result,'Duplicate declaration key.');result[k]=v
        return result
    value=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite declaration.')))
    require(type(value) is dict,'Declaration object required.')
    return value

def seed_fields(value):
    """Match the frozen protocol's inherited seed-field rule, with provenance."""
    stack=[(value,[],False)];out=[];nodes=0
    while stack:
        v,path,seeded=stack.pop();nodes+=1
        require(nodes<=3_000_000 and len(path)<=32,'Declaration tree bounds exceeded.')
        if type(v) is dict:
            for k,x in reversed(list(v.items())):
                require(type(k) is str,'Non-string declaration key.')
                stack.append((x,path+[k],seeded or 'seed' in k))
        elif type(v) is list:
            for i in range(len(v)-1,-1,-1):stack.append((v[i],path+[i],seeded))
        elif seeded and type(v) is int:
            require(0<=v<2**32,'Declared seed outside uint32.')
            out.append(dict(seed=v,path=path))
    return out

def inventory(training_raw,training_pin,v1_raw,v1_pin):
    training=parse_bound(training_raw,training_pin);v1=parse_bound(v1_raw,v1_pin)
    require(type(training.get('runs')) is list and len(training['runs'])==280,'Complete280-run standard declaration required.')
    require(v1.get('schema')=='bca-ood-declaration-v1' and v1.get('manifest_sha256')==digest({k:v for k,v in v1.items() if k!='manifest_sha256'}),'Changed original OOD declaration content.')
    require(type(v1.get('pairs')) is list and len(v1['pairs'])==20 and type(v1.get('checkpoints')) is list and len(v1['checkpoints'])==40,'Complete original pair/checkpoint declaration required.')
    expected={f'{h}/{e}/s{s}' for h,e,s in PAIRS};ids=[r['pair_id'] for r in v1['pairs']]
    require(len(set(ids))==20 and set(ids)==expected,'Missing/duplicate/wrong original pair.')
    records=[dict(source='standard_training',**r) for r in seed_fields(training['runs'])]
    records += [dict(source='v1_resolved_training',**r) for r in seed_fields(v1['checkpoints'])]
    def add(values,path,n):
        require(type(values) is list and len(values)==n and all(type(x) is int and 0<=x<2**32 for x in values),'Incomplete/invalid original stream.')
        records.extend(dict(source='v1_ood',seed=x,path=path+[i]) for i,x in enumerate(values))
    for i,pair in enumerate(v1['pairs']):
        require(set(pair['streams'])==set(STREAM_SIZES),'Wrong original stream purposes.')
        for purpose,n in STREAM_SIZES.items():add(pair['streams'][purpose],['pairs',i,'streams',purpose],n)
    cells={f'{h}/{e}' for h,e,_ in PAIRS}
    require(set(v1['engineering_streams'])==cells,'Incomplete original engineering streams.')
    for cell in sorted(cells):add(v1['engineering_streams'][cell],['engineering_streams',cell],1024)
    add(v1['bootstrap_seeds'],['bootstrap_seeds'],10000);add(v1['plotting_fixture_seeds'],['plotting_fixture_seeds'],1)
    require(records,'Empty predecessor inventory.')
    return dict(records=records,unique_seeds=sorted({r['seed'] for r in records}),source_pins=dict(standard_training=training_pin,v1_declaration=v1_pin),
                scope='All numeric seed fields in the complete standard280-run and original OOD40-checkpoint declarations, plus every declared V1 OOD stream; no unrelated campaign search.')

def check_v2(existing):
    require(type(existing) is dict and type(existing.get('unique_seeds')) is list and existing['unique_seeds']==sorted({r['seed'] for r in existing['records']}),'Changed predecessor inventory.')
    owners={};duplicates=[];collisions=[];prior=set(existing['unique_seeds'])
    for h,e,s in PAIRS:
        for purpose in ('reset','candidate','continuation','random_score'):
            for index in range(64 if purpose=='reset' else 256):
                value=stream_seed(h,e,s,purpose,index);owner=dict(host=h,environment=e,training_seed=s,purpose=purpose,index=index)
                if value in owners:duplicates.append(dict(seed=value,first=owners[value],second=owner))
                owners[value]=owner
                if value in prior:
                    matches=[r for r in existing['records'] if r['seed']==value]
                    collisions.append(dict(seed=value,v2=owner,predecessors=matches))
    return dict(schema='ood-v2-stream-inventory-gate-v1',v2_streams=16640,v2_unique_seeds=len(owners),
        predecessor_unique_seeds=len(prior),predecessor_occurrences=len(existing['records']),within_v2_duplicates=duplicates,
        predecessor_collisions=collisions,passed=not duplicates and not collisions,stream_execution_accepted=False,
        automatic_repair=False,jax_keys_generated=False,scientific_acceptance=False)
