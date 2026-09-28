"""Bounded, exclusive precommit artifacts; no scientific or resource imports."""
import hashlib
import json
import math
import os
from pathlib import Path
import stat

LIMIT=2_000_000


def _values(value):
    todo=[(value,0)];nodes=0
    while todo:
        v,d=todo.pop();nodes+=1
        if d>32 or nodes>100_000: raise ValueError('Artifact nesting/node bound exceeded.')
        if type(v) is dict:
            if any(type(k) is not str for k in v): raise ValueError('String keys required.')
            todo.extend((item,d+1) for item in v.values())
        elif type(v) is list: todo.extend((item,d+1) for item in v)
        elif v is None or type(v) in (str,bool,int): pass
        elif type(v) is float and math.isfinite(v): pass
        else: raise ValueError('Explicit finite JSON values required.')


def canonical(value):
    _values(value)
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
    if len(raw)+1>LIMIT: raise ValueError('Artifact byte bound exceeded.')
    return raw


def _path(path):
    p=Path(path)
    if not p.is_absolute() or '..' in p.parts: raise ValueError('Absolute non-redirected artifact path required.')
    if any(q.is_symlink() for q in [p,*p.parents]): raise ValueError('Symlink artifact path refused.')
    if not p.parent.is_dir(): raise ValueError('Existing artifact parent required.')
    return p


def _identity(st):
    return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)


def write_once(path,value):
    p=_path(path);raw=canonical(value)+b'\n'
    with p.open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno());saved=_identity(os.fstat(f.fileno()))
    # A failure after creation deliberately leaves the partial/unacknowledged file.
    # No overwrite, cleanup-and-retry, rename replacement or silent repair exists.
    if _identity(p.stat())!=saved or _path(p)!=p: raise ValueError('Artifact identity changed during publication.')
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return hashlib.sha256(raw).hexdigest()


def _object(pairs):
    result={}
    for k,v in pairs:
        if k in result: raise ValueError('Duplicate JSON key.')
        result[k]=v
    return result


def read_bound(path,expected_sha256):
    if (type(expected_sha256) is not str or len(expected_sha256)!=64
            or any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise ValueError('External SHA256 required.')
    p=_path(path);before=p.stat()
    if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=LIMIT:
        raise ValueError('Bounded regular artifact required.')
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:
        if _identity(os.fstat(f.fileno()))!=_identity(before): raise ValueError('Artifact changed before read.')
        raw=f.read(LIMIT+1)
        if _identity(os.fstat(f.fileno()))!=_identity(before): raise ValueError('Artifact changed during read.')
    if _identity(p.stat())!=_identity(before) or _path(p)!=p: raise ValueError('Artifact path changed during read.')
    if len(raw)!=before.st_size or hashlib.sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('Artifact external hash mismatch.')
    try:
        value=json.loads(raw,object_pairs_hook=_object,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON.')))
        canonical(value)
    except (RecursionError,UnicodeError) as e: raise ValueError('Invalid bounded JSON.') from e
    return value
