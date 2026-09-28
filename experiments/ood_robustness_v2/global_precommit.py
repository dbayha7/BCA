"""Pair-wide byte/structure barrier, not native semantic or execution acceptance.

The caller independently pins index/context/readers. No key/pool generation,
model, physics, ledger, archive construction or stream amendment occurs here.
Raw readers/writers must be separately accepted, exclusive and durable.
"""
import copy
import hashlib
import json
import math
import re
import struct
from precommit_file import canonical

PLAN = 'f11ebe8f8e3ccc4af10511ed1a241e8f7c5f0aa9f84770103dd6928e59b60756'
PINS = {'training_acceptance','host_checkpoint','bca_checkpoint','prepared_data',
        'support_bank','source_manifest','runtime','execution_declaration','stream_inventory'}
GATES = {'stream_inventory_and_jax','captured_state_schema','precommit_reconstruction',
         'warning_recomputation','runtime_sources_paths','native_action_reward_repeat',
         'resources_storage_ownership','supervisor_actual_path'}


def require(ok, message):
    if not ok: raise ValueError(message)


def digest(raw): return hashlib.sha256(raw).hexdigest()


def sha(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None,'Invalid SHA256.')
    return value


def ref(value):
    require(type(value) is dict and set(value)=={'name','sha256'},'Exact artifact reference required.')
    require(type(value['name']) is str and re.fullmatch('[A-Za-z0-9_/-]{1,200}',value['name']) is not None
            and '..' not in value['name'] and not value['name'].startswith('/'),'Unsafe artifact name.')
    sha(value['sha256']);return value


def pairs(items):
    result={}
    for k,v in items:
        require(k not in result,'Duplicate JSON key.');result[k]=v
    return result


def load(read, reference):
    ref(reference);raw=read(reference['name'])
    require(type(raw) is bytes and 0<len(raw)<=2_000_000,'Bounded raw bytes required.')
    require(digest(raw)==reference['sha256'],'External artifact hash mismatch.')
    value=json.loads(raw,object_pairs_hook=pairs)
    require(canonical(value)==raw,'Canonical finite artifact required.')
    return value


def append(write,read,name,value):
    raw=canonical(value);write(name,raw)
    require(read(name)==raw,'Durable barrier readback failed.')
    return dict(name=name,sha256=digest(raw))


def array_hash(raw,shape,dtype='<u4'):
    return digest(canonical(dict(dtype=dtype,shape=shape))+b'\0'+raw)


def key_bytes(value,shape):
    require(type(value) is dict and set(value)=={'dtype','shape','hex'}
            and value['dtype']=='<u4' and value['shape']==shape,'Little-endian uint32 key table required.')
    require(type(value['hex']) is str and len(value['hex'])==math.prod(shape)*8,'Wrong key byte count.')
    raw=bytes.fromhex(value['hex']);require(len(raw)==math.prod(shape)*4,'Invalid key bytes.')
    return raw


def validate_context(context):
    require(type(context) is dict and set(context)=={'pair','protocol_sha256','pins'},'Exact external context required.')
    p=context['pair']
    require(type(p) is list and len(p)==3 and p[0] in ('td3_bc','rebrac') and p[1] in ('hopper','walker2d')
            and type(p[2]) is int and 202609171<=p[2]<=202609175,'Undeclared pair.')
    require(context['protocol_sha256']==PLAN,'Changed original protocol.')
    require(type(context['pins']) is dict and set(context['pins'])==PINS,'Missing external context pins.')
    for h in context['pins'].values(): sha(h)


def nominal(capture,i):
    expected=dict(state_index=i,collector='host' if i<128 else 'bca',reset_block=(i%128)//2,
                  capture_step=100 if i%2==0 else 300)
    require(all(type(capture.get(k)) is type(v) and capture[k]==v for k,v in expected.items()),'Nominal capture mismatch.')
    for purpose in ('reset','candidate','continuation','random_score'):
        v=capture.get(purpose+'_seed')
        require(type(v) is int and 0<=v<2**32,'Missing declared stream integer.')
    # No seed formula is recomputed here: independent stream acceptance is required.
    return {**expected,**{p+'_seed':capture[p+'_seed'] for p in ('reset','candidate','continuation','random_score')}}


def finite(value,minimum=0,maximum=None):
    require(type(value) in (int,float) and math.isfinite(value) and value>=minimum
            and (maximum is None or value<=maximum),'Invalid warning score.')


def inspect_index(read,index_ref,context):
    """Read every dependency before a seal; return immutable structural summary."""
    validate_context(context);index=load(read,index_ref)
    require(set(index)=={'schema','context','key_table','key_binding','rows'}
            and index['schema']=='ood-v2-global-precommit-index-v1'
            and canonical(index['context'])==canonical(context),'Index/context mismatch.')
    rows=index['rows'];require(type(rows) is list and len(rows)==256,'All256 nominal captures required.')
    kr=load(read,index['key_binding']);kt=load(read,index['key_table'])
    require(set(kr)=={'schema','pair','shape','table_sha256','row_sha256','old_keys_sha256','prior_v2_keys_sha256',
                      'key_derivation_accepted','predecessor_inventory_completeness_accepted','execution_accepted'}
            and all(kr[k] is False for k in ('key_derivation_accepted','predecessor_inventory_completeness_accepted','execution_accepted')),
            'Component key binding cannot certify its own derivation.')
    sha(kr['old_keys_sha256']);sha(kr['prior_v2_keys_sha256'])
    raw=key_bytes(kt,[256,250,2]);keys=list(struct.iter_unpack('<II',raw))
    require(len(set(keys))==64000,'Duplicate saved continuation key.')
    require(kr.get('schema')=='ood-v2-key-table-binding-v1' and kr.get('pair')==context['pair']
            and kr.get('shape')==[256,250,2] and kr.get('table_sha256')==array_hash(raw,[256,250,2]),'Key binding mismatch.')
    row_hashes=[array_hash(raw[i*2000:(i+1)*2000],[250,2]) for i in range(256)]
    require(kr.get('row_sha256')==row_hashes,'Key row binding mismatch.')
    refs=[copy.deepcopy(index_ref),index['key_table'],index['key_binding']];summary=[];captures=[]
    for i,row in enumerate(rows):
        require(type(row) is dict and set(row)=={'state_index','capture','bank','warnings'}
                and type(row['state_index']) is int and row['state_index']==i,'Index omission/reordering.')
        capture=load(read,row['capture']);refs.append(row['capture']);n=nominal(capture,i);captures.append(capture)
        status=capture.get('status');require(status in ('captured','missing'),'Invalid capture status.')
        if status=='missing':
            require(row['bank'] is None and row['warnings'] is None,'Missing state cannot have candidates/scores.')
            require(capture.get('reason')=='native_episode_ended_before_or_at_capture','Unexplained missing state.')
            t=capture.get('terminal',{})
            require(type(t.get('elapsed_steps')) is int and 1<=t['elapsed_steps']<=n['capture_step']
                    and type(t.get('terminated')) is bool and type(t.get('truncated')) is bool
                    and (t['terminated'] or t['truncated']),'Missing-state native flag/counter mismatch.')
            sha(t.get('full_state_sha256'));sha(t.get('call_evidence_sha256'))
            summary.append(dict(state_index=i,status=status));continue
        snapshot=capture.get('snapshot');require(type(snapshot) is dict,'Full supplied snapshot required.')
        sh=digest(canonical(snapshot));require(sh==capture.get('snapshot_content_sha256'),'Snapshot content mismatch.')
        require(type(snapshot.get('elapsed_steps')) is int and snapshot['elapsed_steps']==n['capture_step']
                and snapshot.get('terminated') is False and snapshot.get('truncated') is False,'Terminal/wrong-step capture.')
        sha(capture.get('call_evidence_sha256'))
        bank=load(read,row['bank']);warnings=load(read,row['warnings']);refs.extend([row['bank'],row['warnings']])
        require(bank.get('schema')=='ood-v2-state-precommit-v1' and bank.get('pair')==context['pair']
                and bank.get('protocol_sha256')==PLAN and bank.get('capture')==n,'Wrong bank identity.')
        require(bank.get('pins')==dict(context['pins'],state_snapshot=sh),'Bank external provenance mismatch.')
        require(bank.get('key_receipt_sha256')==index['key_binding']['sha256']
                and key_bytes(bank.get('keys'),[250,2])==raw[i*2000:(i+1)*2000],'Bank keys changed.')
        require(bank.get('score_or_outcome_access') is False and bank.get('execution_accepted') is False,'Producer bank cannot accept itself.')
        slots=bank.get('slots');require(type(slots) is list and len(slots)==14,'All14 slots required.')
        require(set(warnings)=={'schema','pair','state_index','bank_sha256','slots'}
                and warnings['schema']=='ood-v2-warning-precommit-v1' and warnings['pair']==context['pair']
                and type(warnings['state_index']) is int and warnings['state_index']==i
                and warnings['bank_sha256']==row['bank']['sha256'],'Warning bank/identity mismatch.')
        ws=warnings['slots'];require(type(ws) is list and len(ws)==14,'All14 warning slots required.')
        for j,(s,w) in enumerate(zip(slots,ws)):
            role='near' if j<4 else 'moderate' if j<8 else 'strong' if j<12 else 'host_policy' if j==12 else 'bca_policy'
            require(type(s.get('slot')) is int and s['slot']==j and s.get('role')==role
                    and type(w.get('slot')) is int and w['slot']==j and w.get('status')==s.get('status'),'Warning/slot mismatch.')
            if s['status']=='missing':
                require(j not in (0,12,13) and set(w)=={'slot','status'},'Missing native/anchor or invented score.');continue
            require(s['status']=='present' and set(w)=={'slot','status','width','support','random','constant','native_width'},'Wrong score fields.')
            owner=s.get('alias_of');require(type(owner) is int and 0<=owner<=j
                and slots[owner].get('status')=='present' and slots[owner].get('alias_of')==owner,'Invalid alias owner.')
            finite(w['width']);finite(w['support']);finite(w['random'],0,1)
            require(w['support']==s.get('distance') and type(w['constant']) in (int,float)
                    and w['constant']==.5 and w['native_width'] is None,'Support/constant/native-width changed.')
            if owner<j: require(w['width']==ws[owner]['width'] and w['support']==ws[owner]['support'],'Alias warning differs.')
        summary.append(dict(state_index=i,status=status,snapshot_sha256=sh))
    for i in range(0,256,2):
        early,late=captures[i:i+2]
        require(early['reset_seed']==late['reset_seed'],'Within-block reset seed changed.')
        if early['status']=='missing':
            require(late['status']=='missing' and early['terminal']==late['terminal'],'Missing early capture replaced later.')
        elif late['status']=='missing': require(late['terminal']['elapsed_steps']>100,'Terminal before captured state.')
        if i<128: require(early['reset_seed']==captures[i+128]['reset_seed'],'Collectors did not share reset seed.')
    names=[ref(r)['name'] for r in refs]
    require(len(set(names))==len(names),'Artifact-name alias in global inventory.')
    return dict(schema='ood-v2-global-precommit-seal-v1',context=copy.deepcopy(context),index=copy.deepcopy(index_ref),
                dependencies=copy.deepcopy(refs[1:]),rows=summary,captured=sum(r['status']=='captured' for r in summary),
                nominal_captures=256,nominal_panels=512,scope='all256_nominal_rows_of_one_pair',
                semantic_acceptance=False,execution_accepted=False,scientific_acceptance=False)


def seal_precommit(read,write,index_ref,context):
    seal=inspect_index(read,index_ref,context)
    # Refuse a source changing during the pair-wide scan before acknowledging seal.
    for r in [seal['index'],*seal['dependencies']]: load(read,r)
    return append(write,read,'barrier/precommit',seal)


def verify_seal(read,seal_ref,context):
    seal=load(read,seal_ref)
    expected=inspect_index(read,seal.get('index'),context)
    require(canonical(seal)==canonical(expected),'Seal differs from externally bound inventory.')
    for r in [seal_ref,seal['index'],*seal['dependencies']]: load(read,r)
    return seal
