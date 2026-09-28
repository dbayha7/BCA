"""Synthetic files only. Never opens the real ledger/lock or scientific runtime."""
from pathlib import Path
import hashlib,json,os,sqlite3,subprocess,sys,tempfile,unittest
from ancestor_guard import AncestorGuard, ExclusiveLease, canonical, digest, identity
from extension_ledger import ExtensionLedger, declaration, maximum_caps, cap_check

SCOPE='outcomes/td3_bc/hopper/s202609171'


def fixture(root):
    old=root/'ancestor.sqlite';lock=root/'shared.lock';lock.write_bytes(b'')
    header=dict(schema='ood-resource-ledger-v2',caps={'global':[39998400,159993600]},protocol_sha256='1'*64,
                counts='reservations, not proof of completed physical transitions')
    db=sqlite3.connect(old)
    db.executescript('CREATE TABLE header(id INTEGER PRIMARY KEY,payload TEXT NOT NULL);CREATE TABLE entries(id INTEGER PRIMARY KEY,token TEXT UNIQUE,payload TEXT,digest TEXT);CREATE TABLE totals(scope TEXT PRIMARY KEY,environment INTEGER,physics INTEGER);')
    db.execute('INSERT INTO header VALUES(1,?)',(canonical(header).decode(),))
    db.execute('INSERT INTO entries VALUES(0,?,?,?)',('old-call','{}','2'*64))
    db.execute('INSERT INTO totals VALUES(?,?,?)',('global',1298353,5193412));db.commit();db.close()
    binding=dict(path=str(old),identity=identity(old),header_sha256=digest(header),entries=1,last_sha256='2'*64,
                 reserved={'global':[1298353,5193412]},audit_receipt_sha256='3'*64)
    guard=AncestorGuard(binding)
    return old,lock,binding,guard


class Extension(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.old,self.lock,self.binding,self.guard=fixture(self.root)
        self.old_bytes=self.old.read_bytes();self.path=self.root/'new.sqlite'
        self.spec=declaration(self.path,self.lock,self.binding,'4'*64)

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_budget(self):
        caps=maximum_caps();self.assertEqual(len(caps),65)
        self.assertEqual(caps['global'],[36791360,147165440])
        self.assertEqual(sum(v[0] for k,v in caps.items() if k!='global'),36791360)

    def test_durable_before_callback_and_old_unchanged(self):
        with ExclusiveLease(self.lock) as lease:
            with ExtensionLedger.create(self.path,self.guard,lease,self.spec) as ledger:
                def callback():
                    db=sqlite3.connect(self.path);self.assertEqual(db.execute('SELECT count(*) FROM reservations').fetchone()[0],1)
                    self.assertEqual(db.execute('SELECT environment FROM totals WHERE scope="global"').fetchone()[0],1);db.close()
                    return 'result','a'*64
                self.assertEqual(ledger.call('call-0',SCOPE,callback),'result')
                self.assertEqual(ledger.audit()['reserved']['global'],[1,4])
                self.assertEqual(ledger.audit()['pending'],[])
        self.assertEqual(self.old.read_bytes(),self.old_bytes)

    def test_duplicate_never_calls_again(self):
        called=[]
        with ExclusiveLease(self.lock) as lease, ExtensionLedger.create(self.path,self.guard,lease,self.spec) as ledger:
            ledger.call('one',SCOPE,lambda: (called.append(1),'a'*64))
            with self.assertRaises(ValueError): ledger.call('one',SCOPE,lambda: called.append(2))
        self.assertEqual(called,[1])

    def test_callback_failure_keeps_charge_and_blocks_new_call(self):
        with ExclusiveLease(self.lock) as lease, ExtensionLedger.create(self.path,self.guard,lease,self.spec) as ledger:
            def fail(): raise RuntimeError('synthetic callback failure')
            with self.assertRaises(RuntimeError): ledger.call('one',SCOPE,fail)
            self.assertEqual(ledger.audit()['reserved']['global'],[1,4])
            with self.assertRaises(ValueError): ledger.call('two',SCOPE,lambda: (None,'a'*64))
        with ExclusiveLease(self.lock) as lease:
            with self.assertRaises(ValueError): ExtensionLedger.open(self.path,self.guard,lease,self.spec)

    def test_invalid_completion_leaves_pending(self):
        with ExclusiveLease(self.lock) as lease, ExtensionLedger.create(self.path,self.guard,lease,self.spec) as ledger:
            with self.assertRaises(ValueError): ledger.call('one',SCOPE,lambda: (None,'not-a-hash'))
            self.assertEqual(ledger.audit()['pending'],['one'])

    def test_process_death_after_reservation_persists(self):
        spec=self.root/'spec.json';spec.write_text(json.dumps(self.spec))
        code='''import json,os,sys
from pathlib import Path
from ancestor_guard import AncestorGuard,ExclusiveLease
from extension_ledger import ExtensionLedger
s=json.loads(Path(sys.argv[1]).read_text());g=AncestorGuard(s['ancestor'])
with ExclusiveLease(Path(s['lock_path'])) as lease:
    with ExtensionLedger.create(Path(s['path']),g,lease,s) as ledger:
        ledger.reserve('crash','outcomes/td3_bc/hopper/s202609171')
        os._exit(23)
'''
        env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parent))
        r=subprocess.run([sys.executable,'-c',code,str(spec)],env=env,capture_output=True)
        self.assertEqual(r.returncode,23,r.stderr.decode())
        with ExclusiveLease(self.lock) as lease:
            with self.assertRaises(ValueError): ExtensionLedger.open(self.path,self.guard,lease,self.spec)
        db=sqlite3.connect(self.path);self.assertEqual(db.execute('SELECT count(*) FROM reservations').fetchone()[0],1);db.close()

    def test_scope_cap_blocks_without_reservation(self):
        self.spec['caps'][SCOPE]=[0,0]
        with ExclusiveLease(self.lock) as lease, ExtensionLedger.create(self.path,self.guard,lease,self.spec) as ledger:
            with self.assertRaises(ValueError): ledger.call('one',SCOPE,lambda: self.fail('callback ran'))
            self.assertEqual(ledger.audit()['entries'],0)

    def test_global_and_combined_cap_boundaries(self):
        caps=maximum_caps();totals={k:[0,0] for k in caps}
        cap_check(totals,caps,[1298353,5193412],[39998400,159993600],SCOPE)
        totals['global']=caps['global'][:]
        with self.assertRaises(ValueError): cap_check(totals,caps,[1298353,5193412],[39998400,159993600],SCOPE)
        totals['global']=[0,0]
        with self.assertRaises(ValueError): cap_check(totals,caps,[39998400,159993600],[39998400,159993600],SCOPE)

    def test_declared_ceiling_cannot_expand(self):
        self.spec['caps'][SCOPE][0]+=1
        with ExclusiveLease(self.lock) as lease:
            with self.assertRaises(ValueError): ExtensionLedger.create(self.path,self.guard,lease,self.spec)
        self.assertFalse(self.path.exists())

    def test_unknown_scope_no_callback(self):
        with ExclusiveLease(self.lock) as lease, ExtensionLedger.create(self.path,self.guard,lease,self.spec) as ledger:
            with self.assertRaises(ValueError): ledger.call('one','outcomes/unknown',lambda: self.fail())

    def test_changed_ancestor_blocks_before_callback(self):
        with ExclusiveLease(self.lock) as lease, ExtensionLedger.create(self.path,self.guard,lease,self.spec) as ledger:
            self.old.touch()
            with self.assertRaises(ValueError): ledger.call('one',SCOPE,lambda: self.fail())

    def test_mutated_binding_refused(self):
        self.guard.binding['reserved']['global'][0]=0
        with self.assertRaises(ValueError): self.guard.unchanged()

    def test_mutated_declaration_refused_before_callback(self):
        with ExclusiveLease(self.lock) as lease, ExtensionLedger.create(self.path,self.guard,lease,self.spec) as ledger:
            ledger.header['caps'][SCOPE]=[999999999,3999999996]
            with self.assertRaises(ValueError): ledger.call('one',SCOPE,lambda: self.fail())

    def test_replaced_extension_during_callback_stays_uncertain(self):
        moved=self.root/'moved.sqlite'
        with ExclusiveLease(self.lock) as lease, ExtensionLedger.create(self.path,self.guard,lease,self.spec) as ledger:
            def callback():
                self.path.rename(moved);self.path.write_bytes(b'')
                return None,'a'*64
            with self.assertRaises(ValueError): ledger.call('one',SCOPE,callback)
        db=sqlite3.connect(moved)
        self.assertEqual(db.execute('SELECT count(*) FROM reservations').fetchone()[0],1)
        self.assertEqual(db.execute('SELECT count(*) FROM completions').fetchone()[0],0);db.close()

    def test_ancestor_sidecar_refused(self):
        Path(str(self.old)+'-journal').write_bytes(b'')
        with self.assertRaises(ValueError): AncestorGuard(self.binding)

    def test_wrong_external_ancestor_head(self):
        self.binding['last_sha256']='b'*64
        with self.assertRaises(ValueError): AncestorGuard(self.binding)

    def test_ancestor_symlink_refused(self):
        link=self.root/'link.sqlite';link.symlink_to(self.old)
        self.binding['path']=str(link)
        with self.assertRaises(ValueError): AncestorGuard(self.binding)

    def test_concurrent_lease_refused(self):
        with ExclusiveLease(self.lock):
            with self.assertRaises(ValueError):
                with ExclusiveLease(self.lock): pass

    def test_replaced_lock_refused(self):
        with ExclusiveLease(self.lock) as lease, ExtensionLedger.create(self.path,self.guard,lease,self.spec) as ledger:
            self.lock.rename(self.root/'old.lock');self.lock.write_bytes(b'')
            with self.assertRaises(ValueError): ledger.call('one',SCOPE,lambda: self.fail())

    def test_forked_lease_refused(self):
        with ExclusiveLease(self.lock) as lease:
            lease.pid-=1
            with self.assertRaises(ValueError): ExtensionLedger.create(self.path,self.guard,lease,self.spec)

    def test_exclusive_create_preserves_existing(self):
        with ExclusiveLease(self.lock) as lease:
            with ExtensionLedger.create(self.path,self.guard,lease,self.spec): pass
            before=self.path.read_bytes()
            with self.assertRaises(ValueError): ExtensionLedger.create(self.path,self.guard,lease,self.spec)
            self.assertEqual(self.path.read_bytes(),before)

    def test_closed_reopen_and_append(self):
        with ExclusiveLease(self.lock) as lease:
            with ExtensionLedger.create(self.path,self.guard,lease,self.spec) as ledger:
                ledger.call('one',SCOPE,lambda: (1,'a'*64))
        with ExclusiveLease(self.lock) as lease, ExtensionLedger.open(self.path,self.guard,lease,self.spec) as ledger:
            ledger.call('two',SCOPE,lambda: (2,'b'*64))
            self.assertEqual(ledger.audit()['entries'],2)

    def test_runtime_pending_query_uses_indexed_tail(self):
        with ExclusiveLease(self.lock) as lease, ExtensionLedger.create(self.path,self.guard,lease,self.spec) as ledger:
            plan=ledger.db.execute('EXPLAIN QUERY PLAN SELECT r.token FROM reservations r WHERE r.id=(SELECT max(id) FROM reservations) AND NOT EXISTS(SELECT 1 FROM completions c WHERE c.token=r.token)').fetchall()
            self.assertFalse(any('SCAN ' in r[3] for r in plan),plan)
            self.assertTrue(any('INTEGER PRIMARY KEY' in r[3] for r in plan),plan)

    def test_nonterminal_pending_cannot_hide_behind_closed_tail(self):
        with ExclusiveLease(self.lock) as lease:
            with ExtensionLedger.create(self.path,self.guard,lease,self.spec) as ledger:
                ledger.call('one',SCOPE,lambda: (1,'a'*64));ledger.call('two',SCOPE,lambda: (2,'b'*64))
        db=sqlite3.connect(self.path)
        sql=db.execute('SELECT sql FROM sqlite_master WHERE name="completions_delete"').fetchone()[0]
        db.execute('DROP TRIGGER completions_delete');db.execute('DELETE FROM completions WHERE token="one"');db.execute(sql);db.commit();db.close()
        with ExclusiveLease(self.lock) as lease:
            with self.assertRaises(ValueError): ExtensionLedger.open(self.path,self.guard,lease,self.spec)

    def test_no_refund_update_trigger(self):
        with ExclusiveLease(self.lock) as lease, ExtensionLedger.create(self.path,self.guard,lease,self.spec) as ledger:
            ledger.call('one',SCOPE,lambda: (1,'a'*64))
        db=sqlite3.connect(self.path)
        with self.assertRaises(sqlite3.DatabaseError): db.execute('DELETE FROM reservations')
        db.close()

    def test_corrupt_totals_refused_on_reopen(self):
        with ExclusiveLease(self.lock) as lease:
            with ExtensionLedger.create(self.path,self.guard,lease,self.spec) as ledger:
                ledger.call('one',SCOPE,lambda: (1,'a'*64))
        db=sqlite3.connect(self.path);db.execute('UPDATE totals SET environment=0,physics=0');db.commit();db.close()
        with ExclusiveLease(self.lock) as lease:
            with self.assertRaises(ValueError): ExtensionLedger.open(self.path,self.guard,lease,self.spec)

    def test_corrupt_chain_refused_on_reopen(self):
        with ExclusiveLease(self.lock) as lease:
            with ExtensionLedger.create(self.path,self.guard,lease,self.spec) as ledger:
                ledger.call('one',SCOPE,lambda: (1,'a'*64))
        db=sqlite3.connect(self.path);db.execute('DROP TRIGGER reservations_update');db.execute('UPDATE reservations SET digest=?',('0'*64,));db.commit();db.close()
        with ExclusiveLease(self.lock) as lease:
            with self.assertRaises(ValueError): ExtensionLedger.open(self.path,self.guard,lease,self.spec)

    def test_missing_trigger_refused(self):
        with ExclusiveLease(self.lock) as lease:
            with ExtensionLedger.create(self.path,self.guard,lease,self.spec): pass
        db=sqlite3.connect(self.path);db.execute('DROP TRIGGER header_delete');db.commit();db.close()
        with ExclusiveLease(self.lock) as lease:
            with self.assertRaises(ValueError): ExtensionLedger.open(self.path,self.guard,lease,self.spec)

    def test_new_journal_not_repaired(self):
        with ExclusiveLease(self.lock) as lease:
            with ExtensionLedger.create(self.path,self.guard,lease,self.spec): pass
        side=Path(str(self.path)+'-journal');side.write_bytes(b'uncertain')
        with ExclusiveLease(self.lock) as lease:
            with self.assertRaises(ValueError): ExtensionLedger.open(self.path,self.guard,lease,self.spec)
        self.assertEqual(side.read_bytes(),b'uncertain')


if __name__=='__main__': unittest.main(verbosity=2)
