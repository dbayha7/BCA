"""Injected reserve/evidence/step bridge. Never constructs scientific objects.

The caller must independently accept the actual adapter, archive and declaration.
Its native step must archive input and intercepted control BEFORE physics, and
output afterwards. We bind that raw triplet plus full capture contents to the
reservation before acknowledging completion. Synthetic callbacks are not science.
"""
import copy
import hashlib
import math
import re
import numpy as np
from precommit_file import canonical
from precommit_bank import capture_schedule
from candidate_design import applied_unit


def require(ok,message):
    if not ok: raise ValueError(message)


def encode(value,depth=0):
    require(depth<=30,'Typed evidence depth exceeded.')
    if isinstance(value,np.ndarray):
        require(value.dtype.kind in 'biuf' and value.nbytes<=500_000 and np.isfinite(value).all(),'Invalid evidence array.')
        return dict(__array__=value.tobytes(order='C').hex(),dtype=value.dtype.str,shape=list(value.shape))
    if isinstance(value,np.generic): return encode(np.asarray(value),depth+1)
    if type(value) is tuple: return {'__tuple__':[encode(v,depth+1) for v in value]}
    if type(value) is list: return [encode(v,depth+1) for v in value]
    if type(value) is dict:
        require(all(type(k) is str for k in value),'Non-string evidence key.')
        require(not ({'__array__','__tuple__'}&set(value)),'Reserved typed evidence key.')
        return {k:encode(v,depth+1) for k,v in value.items()}
    require(value is None or type(value) in (str,int,float,bool),'Unsupported evidence value.')
    require(type(value) is not float or math.isfinite(value),'Nonfinite evidence scalar.')
    return value


def packed(value): return canonical(encode(value))
def content_hash(value): return hashlib.sha256(packed(value)).hexdigest()


class EncodedStore:
    """Adapter for an independently accepted durable append/read archive."""
    def __init__(self,archive): self.archive=archive
    def put(self,name,value):
        require(type(name) is str and re.fullmatch(r'[A-Za-z0-9_/-]+',name) is not None and '..' not in name,'Unsafe evidence name.')
        raw=packed(value);self.archive.append(name,raw)
        require(self.archive.read(name)==raw,'Durable archive readback differs.')
        return hashlib.sha256(raw).hexdigest()


def _action(value,size):
    a=np.asarray(value)
    require(a.dtype==np.float32 and a.shape==(size,) and np.isfinite(a).all() and (abs(a)<=1).all(),'Wrong sent action.')
    return a.copy()


class RecordedStep:
    def __init__(self,sim,ledger,store,raw_read,pair,attempt_token):
        capture_schedule(*pair)
        require(type(attempt_token) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,100}',attempt_token) is not None,'Invalid attempt token.')
        self.sim,self.ledger,self.store,self.raw_read=sim,ledger,store,raw_read
        self.pair=tuple(pair);self.attempt=attempt_token
        self.size=3 if pair[1]=='hopper' else 6

    def __call__(self,phase,owner,action):
        require(phase in ('collection','outcomes','repeat_checks','engineering'),'Unknown physical scope.')
        require(type(owner) is str and len(owner)<=200 and re.fullmatch(r'[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*',owner) is not None,'Invalid call owner.')
        a=_action(action,self.size);host,env,seed=self.pair
        scope=f'engineering/{host}/{env}' if phase=='engineering' else f'{phase}/{host}/{env}/s{seed}'
        token=f'{self.attempt}/{phase}/{owner}';prefix='calls/'+owner
        def callback():
            number=self.sim.transitions+1
            before=copy.deepcopy(self.sim.capture())
            require(not before['terminated'] and not before['truncated'],'Cannot step a terminal state.')
            before_sha=self.store.put(prefix+'/before',dict(token=token,scope=scope,sim_transition=number,
                sent=a.copy(),snapshot=before))
            rec=self.sim.step(a.copy())
            after=copy.deepcopy(self.sim.capture());rec=copy.deepcopy(rec)
            # Persist all available evidence before checking semantic gates. A
            # mismatch leaves the reservation pending, never refunded/retried.
            output_sha=self.store.put(prefix+'/output',rec)
            after_sha=self.store.put(prefix+'/after',after)
            raw_hashes={}
            for suffix in ('-input','-applied',''):
                name=f'transition-{number:05d}{suffix}.npz';raw=self.raw_read(name)
                require(type(raw) is bytes and 0<len(raw)<=2_000_000,'Missing/unbounded native call evidence.')
                raw_hashes[name]=hashlib.sha256(raw).hexdigest()
            require(self.sim.transitions==number,'Physical call counter mismatch.')
            expected=applied_unit(a)
            for name,value,dtype in (('proposed_action',a,np.float32),('applied_action',expected,np.float32),('sim_ctrl',expected,np.float64)):
                x=np.asarray(rec[name]);require(x.dtype==dtype and x.shape==a.shape and np.array_equal(x,value),'Applied/control/input mismatch: '+name)
            require(type(rec['reward']) is float and math.isfinite(rec['reward']),'Invalid raw reward.')
            for name in ('forward','alive','action_cost','reconstructed_reward','reward_error'):
                require(type(rec[name]) is float and math.isfinite(rec[name]),'Invalid reward field: '+name)
            cost=-.001*float(np.square(expected.astype(np.float64)).sum())
            reconstructed=rec['forward']+1.+cost
            require(rec['alive']==1. and rec['action_cost']==cost and rec['reconstructed_reward']==reconstructed,'Native saved reward arithmetic differs.')
            require(0<=rec['reward_error']<=1e-7 and abs(rec['reward']-reconstructed)<=1e-7,'Unchanged reward1e-7 gate failed.')
            require(type(rec['physics_steps']) is int and rec['physics_steps']==4,'Wrong physics charge.')
            require(type(rec['elapsed_steps']) is int and rec['elapsed_steps']==before['elapsed_steps']+1==after['elapsed_steps'],'Elapsed-step mismatch.')
            for name in ('terminated','truncated'):
                require(type(rec[name]) is bool and type(after[name]) is bool and rec[name]==after[name],'Episode flags mismatch.')
            observation=np.asarray(rec['observation'])
            require(observation.dtype==np.float64 and observation.shape==((11,) if env=='hopper' else (17,)) and np.isfinite(observation).all(),'Invalid native observation.')
            evidence=dict(schema='ood-v2-physical-call-evidence-v1',token=token,scope=scope,sim_transition=number,
                before_sha256=before_sha,output_sha256=output_sha,full_after_sha256=after_sha,native_raw_sha256=raw_hashes,
                environment_reserved=1,physics_reserved=4,full_after_contents_saved=True,independent_scientific_acceptance=False)
            completion=self.store.put(prefix+'/completed',evidence)
            return dict(record=rec,after=after,evidence_sha256=completion),completion
        return self.ledger.call(token,scope,callback)
