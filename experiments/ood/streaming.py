"""Execution-only storage/query helpers; original simulator and learner stay intact.

ArchiveDirectory implements the exact file interface used by the accepted
LocomotionAdapter.step. Its NPZ byte streams commit before the unchanged method
continues, including the applied vector before physics. No numerical step code is
replaced. FrozenPolicy uses the same eager actor/normalization operations, with
full checkpoint immutability checks at declared block boundaries instead of disk
reads/serialization at every query. Each changed path requires its own gate.
"""
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
import zlib

import numpy as np

from experiments.ood.collect import packed, sha, require, write_artifact


class ArtifactArchive:
    def __init__(self, path, maximum_record_bytes=2000000, *, read_only=False):
        self.path, self.maximum, self.read_only = Path(path), maximum_record_bytes, read_only
        require(type(maximum_record_bytes) is int and maximum_record_bytes>0, 'Invalid record ceiling.')
        self.previous, self.index, self.failed = '0'*64, 0, False
        if read_only:
            self.db=sqlite3.connect(self.path.resolve().as_uri()+'?mode=ro',uri=True)
            return
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('xb'):pass
        self.db=sqlite3.connect(self.path,isolation_level=None)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.execute('CREATE TABLE artifacts (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, metadata TEXT NOT NULL, payload BLOB NOT NULL, digest TEXT NOT NULL)')
        for action in ('UPDATE','DELETE'):
            self.db.execute(f"CREATE TRIGGER artifacts_{action} BEFORE {action} ON artifacts BEGIN SELECT RAISE(ABORT,'immutable evidence'); END")

    def __enter__(self):return self
    def __exit__(self,*_):self.db.close()

    def append(self,name,raw,complete=True):
        require(not self.read_only and not self.failed,'Read-only or failed evidence archive.')
        require(isinstance(raw,bytes) and len(raw)<=self.maximum,'Artifact record exceeds bound.')
        require(self.db.execute('SELECT id FROM artifacts WHERE name=?',(name,)).fetchone() is None,'Duplicate artifact name.')
        meta=dict(index=self.index,name=name,bytes=len(raw),raw_sha256=hashlib.sha256(raw).hexdigest(),
                  complete=complete,previous=self.previous)
        digest=sha(meta)
        self.db.execute('INSERT INTO artifacts VALUES (?,?,?,?,?)',
                        (self.index,name,packed(meta).decode(),zlib.compress(raw,1),digest))
        self.index+=1;self.previous=digest
        if not complete:self.failed=True

    def read(self,name,allow_partial=False):
        row=self.db.execute('SELECT metadata,payload,digest FROM artifacts WHERE name=?',(name,)).fetchone()
        require(row is not None,'Missing artifact.')
        meta=json.loads(row[0]);require(sha(meta)==row[2] and meta['name']==name,'Changed artifact metadata.')
        require(meta['complete'] or allow_partial,'Incomplete artifact.')
        require(0<=meta['bytes']<=self.maximum,'Invalid saved size.')
        obj=zlib.decompressobj(); raw=obj.decompress(row[1],self.maximum+1)
        require(obj.eof and not obj.unused_data and len(raw)==meta['bytes']
                and hashlib.sha256(raw).hexdigest()==meta['raw_sha256'],'Corrupt artifact payload.')
        return raw

    def audit(self):
        require(self.db.execute('PRAGMA quick_check').fetchall()==[('ok',)],'Damaged artifact database.')
        previous='0'*64;count=0;total=0
        for index,name,metadata,digest in self.db.execute('SELECT id,name,metadata,digest FROM artifacts ORDER BY id'):
            meta=json.loads(metadata)
            require(index==count and meta['index']==index and meta['previous']==previous and sha(meta)==digest,
                    'Broken artifact chain.')
            raw=self.read(name);total+=len(raw);previous=digest;count+=1
        return dict(schema='ood-artifact-archive-v1',records=count,uncompressed_bytes=total,last_sha256=previous)


class ArchiveDirectory:
    def __init__(self,archive):self.archive=archive
    def __truediv__(self,name):
        require(re.fullmatch(r'transition-[0-9]{5,}(?:-input|-applied)?\.npz',name) is not None,'Unexpected simulator artifact name.')
        return ArchiveFile(self.archive,name)


class ArchiveFile:
    def __init__(self,archive,name):self.archive,self.name=archive,name
    @contextmanager
    def open(self,mode):
        require(mode=='xb','Only exclusive binary writes are supported.')
        require(not self.archive.failed and not self.archive.read_only,'Failed/read-only artifact archive.')
        require(self.archive.db.execute('SELECT id FROM artifacts WHERE name=?',(self.name,)).fetchone() is None,
                'Duplicate simulator artifact; no retry.')
        stream=io.BytesIO()
        try:
            yield stream
        except BaseException:
            self.archive.append(self.name,stream.getvalue(),complete=False)
            raise
        else:
            self.archive.append(self.name,stream.getvalue())
        finally:stream.close()


class FrozenPolicy:
    def __init__(self,adapter):
        adapter.assert_unchanged()
        self.adapter=adapter
        self.params=adapter.state.native.actor.params
        self.model=adapter.models[0]
        self.mean,self.std=adapter.mean,adapter.std
        self.obs_dim,self.action_dim,self.bound=adapter.obs_dim,adapter.action_dim,adapter.bound
        self.closed=False;self.queries=0

    def actions(self,observations):
        import jax.numpy as jnp
        require(not self.closed,'Closed policy block.')
        x=jnp.asarray(observations)
        require(x.ndim==2 and x.shape[1]==self.obs_dim and np.isfinite(x).all(),'Invalid policy observations.')
        result=np.asarray(self.model.apply(self.params,(x-self.mean)/self.std))
        require(result.shape==(len(x),self.action_dim) and np.isfinite(result).all()
                and (np.abs(result)<=self.bound).all(),'Invalid frozen actor output.')
        self.queries+=1
        return result

    def seal(self):
        # Full model/calibrator state and checkpoint bytes are still checked,
        # before and after each 100-step collection episode and outcome block.
        self.adapter.assert_unchanged()
        require(self.params is self.adapter.state.native.actor.params
                and self.model is self.adapter.models[0]
                and self.mean is self.adapter.mean and self.std is self.adapter.std,'Changed policy binding.')


def collect_bank(sim,policies,seeds,output,step):
    """The fixed 64x2x2 scientific layout, also tested with exact mock dynamics.

The caller must supply a source/receipt-bound simulator and reserved step function.
This helper cannot grant execution acceptance. Save each row immediately, including
missing captures. No replacement or silent restart; source outputs are exclusive.
"""
    require(set(policies)=={'host','bca'} and isinstance(seeds,list) and len(seeds)==64
            and all(type(s) is int and 0<=s<2**32 for s in seeds) and len(set(seeds))==64,
            'Require the full distinct 64 paired reset seeds.')
    rows=[];reset_states={}
    for collector in ('host','bca'):
        policy=policies[collector]
        for episode,seed in enumerate(seeds):
            policy.seal()
            obs=sim.reset(seed)
            initial=sha(sim.capture())
            if episode in reset_states:require(initial==reset_states[episode],'Paired full reset states differ.')
            else:reset_states[episode]=initial
            ended=False
            for t in range(101):
                if t in (0,100):
                    index=len(rows)
                    row=dict(index=index,state_id=f'{collector}/{episode}/{t}',collector=collector,
                        episode=episode,reset_seed=seed,capture_step=t,status='missing' if ended else 'captured',
                        missing_reason='episode_ended_before_capture' if ended else None,
                        reset_shared_block=episode if t==0 else None)
                    if not ended:
                        snapshot=sim.capture()
                        row.update(observation=obs.copy(),snapshot=snapshot,snapshot_sha256=sha(snapshot))
                    row_hash=write_artifact(Path(output)/f'state-{index:03d}.json',row)
                    rows.append(dict(index=index,state_id=row['state_id'],status=row['status'],sha256=row_hash))
                    # The complete row remains on disk; the index stays small.
                    rows[-1]['capture_step']=t
                if ended or t==100:continue
                action=policy.actions(np.asarray(obs)[None,:])[0]
                rec=step(f'{collector}/{episode}/{t}',action)
                obs=rec['observation'];ended=bool(rec['terminated'] or rec['truncated'])
            policy.seal()
    return dict(schema='ood-state-bank-index-v1',rows=rows,captured=sum(r['status']=='captured' for r in rows),
                missing=sum(r['status']=='missing' for r in rows),paired_reset_blocks=len(reset_states),
                scientific_outcomes_collected=False,coverage_accepted=False)
