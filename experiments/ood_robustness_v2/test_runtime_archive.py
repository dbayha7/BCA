import ast
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from contextlib import contextmanager
import zlib
import numpy as np
from runtime_archive import RuntimeArchive
from recorded_step import EncodedStore,RecordedStep,packed,require
from test_drivers import Sim,Ledger,PAIR

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('/mnt/c/Users/David Bayha/Documents/GitHub/BCA/experiments/ood/streaming.py')
FROZEN=json.loads((ROOT/'work/standard_bca_noiw_campaign_v1/monitor_20260928T164055Z/join_external_bindings.json').read_bytes())['source_sha256']['experiments/ood/streaming.py']
raw=SOURCE.read_bytes();assert hashlib.sha256(raw).hexdigest()==FROZEN
# Exact frozen class bodies; no import of scientific modules or construction.
tree=ast.parse(raw);selected=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('ArtifactArchive','ArchiveDirectory','ArchiveFile')]
assert len(selected)==3
scope=dict(Path=Path,sqlite3=sqlite3,hashlib=hashlib,json=json,zlib=zlib,re=re,io=io,contextmanager=contextmanager,
           require=require,packed=packed,sha=lambda v:hashlib.sha256(packed(v)).hexdigest())
exec(compile(ast.Module(body=selected,type_ignores=[]),str(SOURCE),'exec'),scope)
Factory,Directory=scope['ArtifactArchive'],scope['ArchiveDirectory']

class Lease:
    def __init__(self): self.valid=True;self.calls=0
    def assert_held(self):
        self.calls+=1
        if not self.valid: raise ValueError('synthetic lease lost')

class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.path=self.root/'archive.sqlite'
        self.lease=Lease();self.archive=None
    def tearDown(self):
        if self.archive is not None:self.archive.close()
        self.tmp.cleanup()
    def create(self,**kw):
        args=dict(path=self.path,lease=self.lease,archive_factory=Factory,source_path=SOURCE,source_sha256=FROZEN,
                  maximum_file_bytes=10_000_000,minimum_free_bytes=0)
        args.update(kw);self.archive=RuntimeArchive(**args);return self.archive
    def test_durable_append_read_and_closed_source_unchanged(self):
        a=self.create();a.append('one',b'payload');self.assertEqual(a.read('one'),b'payload')
        self.assertEqual(a.db.execute('PRAGMA synchronous').fetchone(),(2,));self.assertEqual(a.db.execute('PRAGMA journal_mode').fetchone(),('delete',))
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(),FROZEN)
    def test_fsync_file_directory_before_return(self):
        a=self.create();events=[];original=os.fsync
        with patch('runtime_archive.os.fsync',side_effect=lambda fd:(events.append(os.fstat(fd).st_mode),original(fd))[1]): a.append('x',b'x')
        import stat
        self.assertTrue(any(stat.S_ISREG(m) for m in events));self.assertTrue(any(stat.S_ISDIR(m) for m in events))
    def test_native_archive_directory_and_full_typed_store_share_wrapper(self):
        a=self.create();directory=Directory(a)
        with (directory/'transition-00001-applied.npz').open('xb') as f:f.write(b'synthetic prephysics payload')
        self.assertEqual(a.read('transition-00001-applied.npz'),b'synthetic prephysics payload')
        store=EncodedStore(a);value=dict(x=np.array([-0.,1.]),rng=('state',np.array([2],np.uint32)),inactive=None)
        self.assertEqual(store.put('full/after',value),hashlib.sha256(packed(value)).hexdigest())
        self.assertEqual(a.read('full/after'),packed(value))
    def test_fsync_failure_preserves_committed_bytes_and_poison(self):
        a=self.create()
        with patch('runtime_archive.os.fsync',side_effect=OSError('synthetic durability failure')):
            with self.assertRaises(OSError):a.append('committed',b'x')
        self.assertTrue(a.failed);self.assertEqual(a.db.execute('SELECT name FROM artifacts').fetchone(),('committed',))
        with self.assertRaises(ValueError):a.append('retry',b'x')
    def test_duplicate_failure_stops_archive(self):
        a=self.create();a.append('x',b'a')
        with self.assertRaises(ValueError):a.append('x',b'b')
        self.assertTrue(a.failed)
    def test_partial_raw_stream_retained_no_next(self):
        a=self.create();directory=Directory(a)
        with self.assertRaises(RuntimeError):
            with (directory/'transition-00001-input.npz').open('xb') as f:
                f.write(b'partial');raise RuntimeError('synthetic stream failure')
        self.assertTrue(a.failed)
        self.assertEqual(a.db.execute('SELECT name FROM artifacts').fetchone(),('transition-00001-input.npz',))
        with self.assertRaises(ValueError):a.append('next',b'x')
    def test_lease_lost_before_write(self):
        a=self.create();self.lease.valid=False
        with self.assertRaises(ValueError):a.append('x',b'x')
        self.assertEqual(a.db.execute('SELECT count(*) FROM artifacts').fetchone()[0],0)
    def test_changed_archive_file_refused(self):
        a=self.create();self.path.rename(self.root/'retained.sqlite');self.path.write_bytes(b'replacement')
        with self.assertRaises(ValueError):a.append('x',b'x')
    def test_external_write_refused(self):
        a=self.create();os.utime(self.path,ns=(1,1))
        with self.assertRaises(ValueError):a.append('x',b'x')
    def test_changed_or_wrong_source_before_creation(self):
        with self.assertRaises(ValueError):self.create(source_sha256='0'*64)
        self.assertFalse(self.path.exists())
    def test_existing_path_symlink_sidecar_hardlink_refused(self):
        self.path.write_bytes(b'keep')
        with self.assertRaises(ValueError):self.create()
        self.assertEqual(self.path.read_bytes(),b'keep');self.path.unlink()
        self.path.symlink_to(self.root/'absent')
        with self.assertRaises(ValueError):self.create()
        self.path.unlink();Path(str(self.path)+'-journal').write_bytes(b'uncertain')
        with self.assertRaises(ValueError):self.create()
        self.assertEqual(Path(str(self.path)+'-journal').read_bytes(),b'uncertain')
    def test_added_hardlink_refused(self):
        a=self.create();os.link(self.path,self.root/'alias')
        with self.assertRaises(ValueError):a.append('x',b'x')
    def test_relative_path_refused(self):
        with self.assertRaises(ValueError):self.create(path=Path('relative.sqlite'))
    def test_fork_ownership_refused(self):
        a=self.create()
        with patch('runtime_archive.os.getpid',return_value=os.getpid()+1):
            with self.assertRaises(ValueError):a.append('x',b'x')
    def test_space_and_file_ceiling_before_new_artifact(self):
        a=self.create();a.maximum_file_bytes=self.path.stat().st_size+1
        with self.assertRaises(ValueError):a.append('x',b'x')
        self.assertEqual(a.db.execute('SELECT count(*) FROM artifacts').fetchone()[0],0)
    def test_low_disk_space_before_archive_creation(self):
        from collections import namedtuple
        Disk=namedtuple('Disk','total used free')
        with patch('runtime_archive.shutil.disk_usage',return_value=Disk(10,10,0)):
            with self.assertRaises(ValueError):self.create(minimum_free_bytes=100)
        self.assertFalse(self.path.exists())
    def test_readback_corruption_poison(self):
        a=self.create()
        with patch.object(a.inner,'read',return_value=b'wrong'):
            with self.assertRaises(ValueError):a.append('x',b'x')
        self.assertTrue(a.failed)
    def test_recorded_step_storage_failure_keeps_reservation(self):
        a=self.create();sim=Sim();ledger=Ledger();sim.ledger=ledger
        step=RecordedStep(sim,ledger,EncodedStore(a),sim.records.__getitem__,PAIR,'fixture')
        with patch('runtime_archive.os.fsync',side_effect=OSError('synthetic fsync')):
            with self.assertRaises(OSError):step('engineering','one',np.zeros(3,np.float32))
        self.assertIsNotNone(ledger.pending);self.assertEqual(sim.transitions,0);self.assertTrue(a.failed)
    def test_changed_sqlite_durability_mode_refused(self):
        a=self.create();a.db.execute('PRAGMA synchronous=OFF')
        with self.assertRaises(ValueError):a.append('x',b'x')
        self.assertTrue(a.failed)
    def test_process_exit_after_acknowledged_append_keeps_exact_bytes(self):
        import subprocess,sys
        script='''import os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from test_runtime_archive import RuntimeArchive,Factory,SOURCE,FROZEN,Lease
a=RuntimeArchive(path=Path(sys.argv[2]),lease=Lease(),archive_factory=Factory,source_path=SOURCE,source_sha256=FROZEN,maximum_file_bytes=10000000,minimum_free_bytes=0)
a.append('before-process-death',b'committed synthetic bytes')
os._exit(23)
'''
        p=subprocess.run([sys.executable,'-c',script,str(Path(__file__).resolve().parent),str(self.path)],capture_output=True)
        self.assertEqual(p.returncode,23,p.stderr.decode())
        db=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True)
        try:
            db.execute('PRAGMA query_only=ON');name,payload=db.execute('SELECT name,payload FROM artifacts').fetchone()
            self.assertEqual(name,'before-process-death');self.assertEqual(zlib.decompress(payload),b'committed synthetic bytes')
        finally:db.close()
        with self.assertRaises(ValueError):self.create()
        print(json.dumps(dict(synthetic_process_crash_actual_exit=p.returncode,acknowledged_bytes_preserved=True,reopen_refused=True)))
    def test_new_sidecar_refused_without_deletion(self):
        a=self.create();side=Path(str(self.path)+'-journal');side.write_bytes(b'unknown')
        with self.assertRaises(ValueError):a.append('x',b'x')
        self.assertEqual(side.read_bytes(),b'unknown');self.assertTrue(a.failed)
    def test_source_replacement_refused_after_creation(self):
        local=self.root/'source.py';local.write_bytes(SOURCE.read_bytes())
        # Compile exact bodies under the separately pinned temporary source path.
        module=dict(scope);exec(compile(ast.Module(body=selected,type_ignores=[]),str(local),'exec'),module)
        a=self.create(source_path=local,archive_factory=module['ArtifactArchive'])
        local.write_bytes(local.read_bytes()+b'\n# modified\n')
        with self.assertRaises(ValueError):a.append('x',b'x')
    def test_constructor_committed_prephysics_records_visible_to_readonly_reader(self):
        from types import SimpleNamespace
        from native_constructor import construct_recorded
        from test_native_constructor import Native,Physics
        a=self.create();ledger=Ledger();observed=[]
        def probe():
            self.assertIsNotNone(ledger.pending)
            db=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True)
            try:
                db.execute('PRAGMA query_only=ON')
                names={r[0] for r in db.execute('SELECT name FROM artifacts')}
                self.assertEqual(names,{'constructor/input','constructor/applied-before-physics'})
                observed.append(names)
            finally:db.close()
        def factory():
            n=Native(probe=probe);n.step(np.array([.3,-.2,.1],np.float32))
            return SimpleNamespace(base=n,transitions=0,frame_skip=4,close=n.close)
        adapter,result=construct_recorded(factory,Native,Physics,ledger,EncodedStore(a),PAIR,'constructor-integration')
        self.assertEqual(len(observed),1);self.assertEqual(adapter.base.real_calls,1);self.assertIsNone(ledger.pending)
        self.assertIn('native_after_sha256',result);self.assertEqual(a.db.execute('SELECT count(*) FROM artifacts').fetchone()[0],5)
        self.assertEqual(a.inner.audit()['records'],5)
    def test_recorded_step_with_real_temporary_extension_and_archive(self):
        from ancestor_guard import ExclusiveLease
        from extension_ledger import ExtensionLedger,declaration
        from test_extension_ledger import fixture
        old,lock,binding,guard=fixture(self.root);ancestor=old.read_bytes()
        with ExclusiveLease(lock) as lease:
            a=self.create(lease=lease);directory=Directory(a)
            path=self.root/'extension.sqlite';spec=declaration(path,lock,guard.binding,'a'*64)
            with ExtensionLedger.create(path,guard,lease,spec) as ledger:
                sim=Sim();original=sim.step
                def native(action):
                    number=sim.transitions+1
                    for suffix in ('-input','-applied'):
                        with (directory/f'transition-{number:05d}{suffix}.npz').open('xb') as f:f.write(b'synthetic raw '+suffix.encode())
                    # Stand-in dynamics only, after both raw writes completed.
                    record=original(action)
                    with (directory/f'transition-{number:05d}.npz').open('xb') as f:f.write(b'synthetic output')
                    return record
                sim.step=native
                step=RecordedStep(sim,ledger,EncodedStore(a),a.read,PAIR,'integration')
                result=step('engineering','one',np.zeros(3,np.float32))
                self.assertEqual(result['after']['elapsed_steps'],1)
                report=ledger.audit();self.assertFalse(report['pending']);self.assertEqual(report['reserved']['global'],[1,4])
                self.assertEqual(a.inner.audit()['records'],7)
            a.close()
        self.assertEqual(old.read_bytes(),ancestor)

if __name__=='__main__':unittest.main(verbosity=2)
