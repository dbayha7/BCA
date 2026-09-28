"""Bounded projection of a hash-bound support artifact; no pickle/science imports.

The accepted bank is larger than the closed generic artifact decoder input cap.
This new projection uses explicit numeric array schemas; it does not change that
decoder's limits or claim to re-audit the original training preparation.
"""
import hashlib,json,math,re
import numpy as np


def _pairs(items):
    out={}
    for k,v in items:
        if k in out: raise ValueError('Duplicate support JSON key.')
        out[k]=v
    return out


def _invalid(value): raise ValueError('Nonfinite JSON value.')


def _float(value):
    x=float(value)
    if not math.isfinite(x): raise ValueError('Overflowing support JSON value.')
    return x


def decode_projection(raw, external_sha256):
    if (type(raw) is not bytes or not 0<len(raw)<=32_000_000
            or type(external_sha256) is not str or not re.fullmatch('[0-9a-f]{64}',external_sha256)
            or hashlib.sha256(raw).hexdigest()!=external_sha256):
        raise ValueError('Unbounded or unbound support artifact.')
    v=json.loads(raw.decode('utf-8'),object_pairs_hook=_pairs,parse_constant=_invalid,parse_float=_float)
    if type(v) is not dict: raise ValueError('Expected support object.')
    result={}
    for name in ('observations','actions','validation_observations','validation_actions'):
        a=v[name]
        if (type(a) is not dict or set(a)!={'__array__','dtype','shape'} or a['dtype'] not in ('<f4','<f8')
                or type(a['shape']) is not list or len(a['shape'])!=2
                or not all(type(n) is int for n in a['shape']) or not 0<a['shape'][0]<=32768 or not 0<a['shape'][1]<=64):
            raise ValueError('Invalid explicit support array schema.')
        dtype=np.dtype(a['dtype']);size=math.prod(a['shape'])*dtype.itemsize
        if (size>16_000_000 or type(a['__array__']) is not str or len(a['__array__'])!=size*2
                or re.fullmatch('[0-9a-f]*',a['__array__']) is None):
            raise ValueError('Invalid support array size/hex.')
        arr=np.frombuffer(bytes.fromhex(a['__array__']),dtype=dtype).reshape(a['shape']).copy()
        if not np.isfinite(arr).all(): raise ValueError('Nonfinite support array.')
        result[name]=arr
    if (len(result['observations'])!=len(result['actions'])
            or len(result['validation_observations'])!=len(result['validation_actions'])
            or result['observations'].shape[1]!=result['validation_observations'].shape[1]
            or result['actions'].shape[1]!=result['validation_actions'].shape[1]
            or (abs(result['actions'])>1).any() or (abs(result['validation_actions'])>1).any()
            or type(v['neighbors']) is not int or not 1<=v['neighbors']<=min(32,len(result['observations']))):
        raise ValueError('Inconsistent support geometry.')
    for key in ('reference_episodes','validation_episodes'):
        if type(v[key]) is not list or not v[key] or not all(type(i) is int for i in v[key]) or len(v[key])!=len(set(v[key])):
            raise ValueError('Invalid support episode split.')
    if set(v['reference_episodes'])&set(v['validation_episodes']): raise ValueError('Support episode leakage.')
    d=np.asarray(v['validation_distances'],dtype=np.float64)
    if d.shape!=(len(result['validation_actions']),) or not np.isfinite(d).all() or (d<0).any():
        raise ValueError('Invalid calibration distances.')
    result.update(neighbors=v['neighbors'],validation_distances=d,
                  reference_episodes=v['reference_episodes'],validation_episodes=v['validation_episodes'])
    return result
