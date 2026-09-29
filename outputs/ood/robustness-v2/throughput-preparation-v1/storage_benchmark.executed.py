"""Synthetic storage comparison only. No scientific imports, actions or keys.

The SQLite baseline reproduces the seven evidence inserts plus two FULL/DELETE
accounting commits; it excludes model/physics and the real guard overhead.
Sequential modes keep identical logical bytes and four durability boundaries.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import shutil
import sqlite3
import statistics
import struct
import subprocess
import sys
import time
import zlib

NOW = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
SHA = lambda b: hashlib.sha256(b).hexdigest()
JSON = lambda x: json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
ROLES = [('before',4608),('native_input',2048),('native_applied',512),
         ('native_output',4096),('output',2048),('after',4608),('completed',768)]
PAYLOADS = {name: b''.join(hashlib.sha256((name+str(i)).encode()).digest()+bytes(32)
                         for i in range((size+63)//64))[:size] for name,size in ROLES}
MODES = ('sqlite_shared','append_shared','append_node_only','append_node_replicated')


def durable_json(p, obj):
    with p.open('xb') as f:
        f.write(JSON(obj)+b'\n');f.flush();os.fsync(f.fileno())


def fsync_dir(p):
    fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def pack(index, call, kind, raw, previous):
    meta=dict(index=index,call=call,kind=kind,bytes=len(raw),sha256=SHA(raw),previous=previous)
    body=JSON(meta)+b'\n'+zlib.compress(raw,1)
    digest=hashlib.sha256(body).digest()
    return struct.pack('!I',len(body))+digest+body,digest.hex()


def decode_frame(frame):
    assert len(frame)>=36
    n=struct.unpack('!I',frame[:4])[0]
    assert n<=100000 and len(frame)==n+36 and hashlib.sha256(frame[36:]).digest()==frame[4:36]
    metadata,compressed=frame[36:].split(b'\n',1)
    meta=json.loads(metadata);raw=zlib.decompress(compressed)
    assert len(raw)==meta['bytes'] and SHA(raw)==meta['sha256']
    return meta,raw,frame[4:36].hex()


class Store:
    def __init__(self, mode, root, node):
        self.mode=mode;self.index=0;self.previous='0'*64;self.fds=[];self.paths=[];self.pending_reads=[]
        self.db=None;self.ledger=None;self.flushes=0
        if mode=='sqlite_shared':
            self.paths=[root/'archive.sqlite',root/'ledger.sqlite']
            for p in self.paths:
                fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600);os.close(fd)
            self.db=sqlite3.connect(self.paths[0],isolation_level=None)
            self.ledger=sqlite3.connect(self.paths[1],isolation_level=None)
            for db in (self.db,self.ledger):
                db.execute('PRAGMA synchronous=FULL')
                assert db.execute('PRAGMA journal_mode=DELETE').fetchone()==('delete',)
            self.db.execute('CREATE TABLE evidence(id INTEGER PRIMARY KEY,call INTEGER,kind TEXT,frame BLOB)')
            self.ledger.execute('CREATE TABLE accounting(id INTEGER PRIMARY KEY,call INTEGER,kind TEXT,frame BLOB)')
            self.ledger.execute('CREATE TABLE totals(id INTEGER PRIMARY KEY,n INTEGER)')
            self.ledger.execute('INSERT INTO totals VALUES(1,0)')
            fsync_dir(root)
        else:
            self.paths=[root/'journal.bin'] if mode=='append_shared' else [node/'journal.bin']
            if mode=='append_node_replicated':self.paths.append(root/'replica.bin')
            for p in self.paths:
                self.fds.append(os.open(p,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600))
                fsync_dir(p.parent)

    def append(self, call, kind, raw):
        frame,digest=pack(self.index,call,kind,raw,self.previous)
        if self.db is not None:
            if kind in ('reserve','ack'):
                self.ledger.execute('BEGIN IMMEDIATE')
                self.ledger.execute('INSERT INTO accounting VALUES(?,?,?,?)',(self.index,call,kind,frame))
                if kind=='reserve':self.ledger.execute('UPDATE totals SET n=n+1 WHERE id=1')
                self.ledger.execute('COMMIT')
                saved=self.ledger.execute('SELECT frame FROM accounting WHERE id=?',(self.index,)).fetchone()[0]
            else:
                self.db.execute('INSERT INTO evidence VALUES(?,?,?,?)',(self.index,call,kind,frame))
                saved=self.db.execute('SELECT frame FROM evidence WHERE id=?',(self.index,)).fetchone()[0]
            assert saved==frame
        else:
            for fd in self.fds:
                offset=os.lseek(fd,0,os.SEEK_END);view=memoryview(frame)
                while view:
                    n=os.write(fd,view);assert n>0;view=view[n:]
                self.pending_reads.append((fd,offset,frame))
        self.previous=digest;self.index+=1

    def boundary(self):
        if self.db is None:
            for fd in self.fds:os.fsync(fd);self.flushes+=1
            for fd,offset,frame in self.pending_reads:assert os.pread(fd,len(frame),offset)==frame
            self.pending_reads.clear()

    def close(self):
        for fd in self.fds:os.close(fd)
        if self.db is not None:self.db.close();self.ledger.close()


def call(store, number, crash=None):
    store.append(number,'reserve',JSON(dict(token=number,environment=1,physics=4)))
    store.boundary()
    if crash=='reserve':os._exit(23)
    for role,_ in ROLES[:3]:store.append(number,role,PAYLOADS[role])
    store.boundary()
    if crash=='pre_action':os._exit(23)
    # No callback or physics: only a synthetic boundary after durable input/control.
    for role,_ in ROLES[3:]:store.append(number,role,PAYLOADS[role])
    store.boundary()
    if crash=='post_output':os._exit(23)
    store.append(number,'ack',JSON(dict(token=number,evidence_head=store.previous)))
    store.boundary()
    if crash=='complete':os._exit(23)


def frames(path):
    if path.suffix=='.sqlite':
        db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True);db.execute('PRAGMA query_only=ON')
        name='evidence' if path.name=='archive.sqlite' else 'accounting'
        rows=[r[0] for r in db.execute('SELECT frame FROM '+name+' ORDER BY id')];db.close()
        return rows,0
    raw=path.read_bytes();rows=[];offset=0
    while offset<len(raw):
        if len(raw)-offset<36:return rows,len(raw)-offset
        n=struct.unpack('!I',raw[offset:offset+4])[0]
        if n>100000 or len(raw)-offset<36+n:return rows,len(raw)-offset
        rows.append(raw[offset:offset+36+n]);offset+=36+n
    return rows,0


def audit(paths):
    rows=[];tail=0
    if len(paths)==2 and paths[0].suffix!='.sqlite':
        assert paths[0].read_bytes()==paths[1].read_bytes()
        paths=paths[:1]
    for p in paths:
        part,t=frames(p);rows.extend(part);tail+=t
    decoded=[decode_frame(r) for r in rows];decoded.sort(key=lambda v:v[0]['index'])
    previous='0'*64;calls={}
    for i,(meta,raw,digest) in enumerate(decoded):
        assert meta['index']==i and meta['previous']==previous
        previous=digest;calls.setdefault(meta['call'],[]).append(meta['kind'])
        if meta['kind'] in PAYLOADS:assert raw==PAYLOADS[meta['kind']]
    expected=['reserve']+[n for n,_ in ROLES]+['ack']
    for kinds in calls.values():assert kinds==expected[:len(kinds)]
    return dict(records=len(decoded),calls=len(calls),reserved_environment=len(calls),reserved_physics=4*len(calls),
                completed=sum(k[-1]=='ack' for k in calls.values()),pending=[n for n,k in calls.items() if k[-1]!='ack'],
                head=previous,incomplete_tail_bytes=tail,restart_allowed=False)


def child(args):
    root=Path(args.root);node=Path(args.node)
    root.mkdir();node.mkdir()
    store=Store(args.mode,root,node)
    if args.crash=='partial_frame':
        assert args.mode!='sqlite_shared'
        store.append(0,'reserve',JSON(dict(token=0,environment=1,physics=4)));store.boundary()
        for fd in store.fds:os.write(fd,b'\x00\x00\x00');os.fsync(fd)
        os._exit(23)
    call(store,0,args.crash)
    raise AssertionError('Child crash did not occur')


def main(args):
    root=Path(args.root);node=Path(args.node)
    assert root.is_absolute() and node.is_absolute() and root!=node
    root.mkdir();node.mkdir()
    start=NOW();runs=[];deaths=[]
    for rep in range(3):
        for mode in MODES:
            name=f'r{rep}-{mode}';d=root/name;local=node/name;d.mkdir();local.mkdir()
            store=Store(mode,d,local);start_time=time.perf_counter()
            for i in range(32):call(store,i)
            elapsed=time.perf_counter()-start_time
            store.close();checked=audit(store.paths)
            assert checked['completed']==32 and not checked['pending'] and not checked['incomplete_tail_bytes']
            pins=[]
            for p in store.paths:
                raw=p.read_bytes();pins.append(dict(path=str(p),bytes=len(raw),sha256=SHA(raw)))
                if node in p.parents:
                    copied=d/('retained-node-'+p.name)
                    with copied.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
                    assert copied.read_bytes()==raw
            runs.append(dict(name=name,mode=mode,seconds=elapsed,calls=32,calls_per_second=32/elapsed,
                             python_fsync_calls=store.flushes,configured_sqlite_transactions=288 if mode=='sqlite_shared' else 0,
                             audit=checked,files=pins))
    for mode in MODES:
        for cut in ('reserve','pre_action','post_output','complete'):
            name=f'death-{mode}-{cut}';d=root/name;local=node/name
            command=[sys.executable,'-B',str(Path(__file__).resolve()),'--root',str(d),'--node',str(local),'--mode',mode,'--crash',cut]
            p=subprocess.run(command,capture_output=True,timeout=20)
            assert p.returncode==23,(command,p.returncode,p.stderr)
            paths=[d/'archive.sqlite',d/'ledger.sqlite'] if mode=='sqlite_shared' else [d/'journal.bin'] if mode=='append_shared' else [local/'journal.bin']
            if mode=='append_node_replicated':paths.append(d/'replica.bin')
            checked=audit(paths)
            assert checked['reserved_environment']==1 and checked['completed']==int(cut=='complete')
            assert checked['pending']==([] if cut=='complete' else [0])
            for path in paths:
                if node in path.parents:shutil.copyfile(path,d/('retained-node-'+path.name))
            deaths.append(dict(mode=mode,cut=cut,actual_exit=p.returncode,stdout=p.stdout.decode(),stderr=p.stderr.decode(),audit=checked))
    d=root/'death-partial';local=node/'death-partial'
    p=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--root',str(d),'--node',str(local),
                      '--mode','append_node_replicated','--crash','partial_frame'],capture_output=True,timeout=20)
    assert p.returncode==23
    checked=audit([local/'journal.bin',d/'replica.bin'])
    assert checked['reserved_environment']==1 and checked['pending']==[0] and checked['incomplete_tail_bytes']==3
    shutil.copyfile(local/'journal.bin',d/'retained-node-journal.bin')
    deaths.append(dict(mode='append_node_replicated',cut='partial_frame',actual_exit=23,audit=checked))
    medians={m:statistics.median(r['calls_per_second'] for r in runs if r['mode']==m) for m in MODES}
    retained_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    assert retained_bytes<64*1024*1024
    report=dict(schema='ood-synthetic-storage-benchmark-v1',started=start,ended=NOW(),hostname=os.uname().nodename,
                source_sha256=SHA(Path(__file__).read_bytes()),shared_root=str(root),node_root=str(node),
                samples=runs,crash_receipts=deaths,median_calls_per_second=medians,
                ratio_to_sqlite={m:medians[m]/medians['sqlite_shared'] for m in MODES},retained_shared_bytes=retained_bytes,
                synthetic_only=True,model_calls=0,simulator_calls=0,scientific_resources_opened=False,
                production_storage_accepted=False,power_loss_tested=False,node_loss_tested=False,
                semantics='Reserve sync; three before-action records sync; four output/full-state records sync; completion sync. Replica sync participates in every boundary.',
                limits=['SQLite transaction shape, not full runtime or guard overhead.',
                        'Node-only mode is not durable against node/job loss; it is never an accepted substitute.',
                        'Synthetic payloads and process-death tests do not prove real full-schema integration or power-loss durability.',
                        'Setup, recovery scan and copying retained node files are outside measured call loop.'])
    durable_json(root/'report.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--node',required=True)
    parser.add_argument('--mode',choices=MODES);parser.add_argument('--crash')
    args=parser.parse_args()
    if args.crash:child(args)
    else:main(args)
