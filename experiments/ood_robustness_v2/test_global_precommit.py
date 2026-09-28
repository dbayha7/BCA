import copy
import hashlib
import unittest
from unittest.mock import patch
import numpy as np
from precommit_file import canonical
from precommit_bank import bind_keys,capture_schedule
from recorded_step import encode,content_hash
from global_precommit import PLAN,PINS,GATES,digest,load,seal_precommit,verify_seal
from phase_barrier import OutcomePhase

PAIR=['td3_bc','hopper',202609171]
H=lambda text:hashlib.sha256(text.encode()).hexdigest()


class Store:
    def __init__(self): self.raw={};self.fail=None
    def read(self,name): return self.raw[name]
    def write(self,name,raw):
        if name==self.fail: raise OSError('synthetic durability refusal')
        if name in self.raw: raise ValueError('exclusive artifact exists')
        self.raw[name]=raw
    def add(self,name,value):
        raw=canonical(value);self.write(name,raw);return dict(name=name,sha256=digest(raw))


class Fixture:
    def __init__(self,captured=(0,128)):
        self.store=Store();self.context=dict(pair=PAIR.copy(),protocol_sha256=PLAN,pins={n:H(n) for n in PINS})
        self.schedule=capture_schedule(*PAIR);self.keys=np.arange(128000,dtype=np.uint32).reshape(256,250,2)
        self.kr=bind_keys(PAIR,self.keys,np.asarray([[999999,999998]],np.uint32),np.empty((0,2),np.uint32))
        kt=dict(dtype='<u4',shape=[256,250,2],hex=self.keys.tobytes().hex())
        self.index=dict(schema='ood-v2-global-precommit-index-v1',context=copy.deepcopy(self.context),
                        key_table=self.store.add('keys/table',kt),key_binding=self.store.add('keys/binding',self.kr),rows=[])
        for i,n in enumerate(self.schedule):
            c=copy.deepcopy(n);bank=warnings=None
            if i in captured:
                snapshot=dict(elapsed_steps=n['capture_step'],terminated=False,truncated=False,
                              synthetic_hidden=encode(np.asarray([-0.,1.],np.float64)))
                sh=digest(canonical(snapshot));c.update(status='captured',snapshot=snapshot,
                    snapshot_content_sha256=sh,observation=encode(np.zeros(11)),call_evidence_sha256=H('call'+str(i)))
                slots=[dict(slot=j,role='near' if j<4 else 'moderate' if j<8 else 'strong' if j<12 else 'host_policy' if j==12 else 'bca_policy',
                            status='present',alias_of=0,distance=0.) for j in range(14)]
                b=dict(schema='ood-v2-state-precommit-v1',pair=PAIR.copy(),protocol_sha256=PLAN,capture=n,
                       pins=dict(self.context['pins'],state_snapshot=sh),key_receipt_sha256=self.index['key_binding']['sha256'],
                       keys=dict(dtype='<u4',shape=[250,2],hex=self.keys[i].tobytes().hex()),slots=slots,
                       score_or_outcome_access=False,execution_accepted=False)
                bank=self.store.add(f'banks/state{i:03d}',b)
                w=dict(schema='ood-v2-warning-precommit-v1',pair=PAIR.copy(),state_index=i,bank_sha256=bank['sha256'],
                       slots=[dict(slot=j,status='present',width=1.,support=0.,random=j/14,constant=.5,native_width=None) for j in range(14)])
                warnings=self.store.add(f'warnings/state{i:03d}',w)
            else:
                t=101 if i%2 and i-1 in captured else 99
                c.update(status='missing',reason='native_episode_ended_before_or_at_capture',
                         terminal=dict(elapsed_steps=t,terminated=True,truncated=False,full_state_sha256=H('terminal'),call_evidence_sha256=H('call')))
            self.index['rows'].append(dict(state_index=i,capture=self.store.add(f'states/state{i:03d}',c),bank=bank,warnings=warnings))
    def pin(self): return self.store.add('index/all',self.index)
    def seal(self):
        self.index_ref=self.pin();self.seal_ref=seal_precommit(self.store.read,self.store.write,self.index_ref,self.context)
        return self.seal_ref
    def repin(self,ref,change):
        value=load(self.store.read,ref);change(value);raw=canonical(value);self.store.raw[ref['name']]=raw;ref['sha256']=digest(raw)
    def phase(self,review_change=None,owner=lambda:None):
        if not hasattr(self,'seal_ref'): self.seal()
        review=dict(schema='ood-v2-precommit-independent-review-v1',pair=PAIR.copy(),seal_sha256=self.seal_ref['sha256'],
                    execution_declaration_sha256=self.context['pins']['execution_declaration'],reviewer_source_sha256=H('synthetic reviewer'),
                    actual_exit=0,gates={n:True for n in GATES})
        if review_change: review_change(review)
        self.review_ref=self.store.add('review/independent',review)
        return OutcomePhase(read=self.store.read,write=self.store.write,assert_owner=owner,
                            seal_ref=self.seal_ref,review_ref=self.review_ref,context=self.context)


class BarrierTests(unittest.TestCase):
    def test_complete_global_seal_preserves_signed_zero(self):
        f=Fixture();s=f.seal();v=verify_seal(f.store.read,s,f.context)
        self.assertEqual(v['captured'],2);self.assertEqual(len(v['rows']),256)
        self.assertFalse(v['execution_accepted']);self.assertFalse(v['semantic_acceptance'])
        self.assertIn('0000000000000080',f.store.raw['states/state000'].decode())
    def test_missing_final_row_refuses_before_seal(self):
        f=Fixture();f.index['rows'].pop()
        with self.assertRaises(ValueError): f.seal()
        self.assertNotIn('barrier/precommit',f.store.raw)
    def test_reordered_nominal_refused(self):
        f=Fixture();f.index['rows'][2:4]=reversed(f.index['rows'][2:4])
        with self.assertRaises(ValueError): f.seal()
    def test_missing_unexplained_state(self):
        f=Fixture();f.repin(f.index['rows'][2]['capture'],lambda c:c.update(reason='replace me'))
        with self.assertRaises(ValueError): f.seal()
    def test_missing_early_cannot_have_late_capture(self):
        f=Fixture(captured=(1,))
        with self.assertRaises(ValueError): f.seal()
    def test_cross_collector_reset_must_match(self):
        f=Fixture(captured=());f.repin(f.index['rows'][128]['capture'],lambda c:c.update(reset_seed=1))
        with self.assertRaises(ValueError): f.seal()
    def test_full_snapshot_content_hash(self):
        f=Fixture();f.repin(f.index['rows'][0]['capture'],lambda c:c['snapshot'].update(extra=123))
        with self.assertRaises(ValueError): f.seal()
    def test_wrong_key_row_refused(self):
        f=Fixture();f.repin(f.index['rows'][0]['bank'],lambda b:b['keys'].update(hex=f.keys[1].tobytes().hex()))
        with self.assertRaises(ValueError): f.seal()
    def test_key_component_cannot_claim_derivation(self):
        f=Fixture();f.repin(f.index['key_binding'],lambda k:k.update(key_derivation_accepted=True))
        with self.assertRaises(ValueError): f.seal()
    def test_wrong_external_training_pin(self):
        f=Fixture();f.context['pins']['training_acceptance']=H('different')
        with self.assertRaises(ValueError): f.seal()
    def test_last_warning_missing_blocks_every_outcome(self):
        f=Fixture(captured=(255,));del f.store.raw['warnings/state255']
        with self.assertRaises(KeyError): f.seal()
        self.assertNotIn('phase/claimed',f.store.raw)
    def test_support_warning_must_equal_bank(self):
        f=Fixture();f.repin(f.index['rows'][0]['warnings'],lambda w:w['slots'][0].update(support=1.))
        with self.assertRaises(ValueError): f.seal()
    def test_alias_width_warning_must_equal_owner(self):
        f=Fixture();f.repin(f.index['rows'][0]['warnings'],lambda w:w['slots'][13].update(width=2.))
        with self.assertRaises(ValueError): f.seal()
    def test_nonfinite_warning_refused(self):
        f=Fixture();r=f.index['rows'][0]['warnings'];raw=f.store.raw[r['name']].replace(b'"width":1.0',b'"width":NaN')
        f.store.raw[r['name']]=raw;r['sha256']=digest(raw)
        with self.assertRaises(ValueError): f.seal()
    def test_duplicate_json_key_refused(self):
        f=Fixture();r=f.index['rows'][0]['warnings'];raw=f.store.raw[r['name']].replace(b'"state_index":0',b'"state_index":0,"state_index":0')
        f.store.raw[r['name']]=raw;r['sha256']=digest(raw)
        with self.assertRaises(ValueError): f.seal()
    def test_no_overwrite_or_second_seal(self):
        f=Fixture();f.seal()
        with self.assertRaises(ValueError): seal_precommit(f.store.read,f.store.write,f.index_ref,f.context)
    def test_changed_dependency_at_phase_entry(self):
        f=Fixture();f.seal();f.store.raw['states/state255']+=b' '
        with self.assertRaises(ValueError): f.phase()
        self.assertNotIn('phase/claimed',f.store.raw)
    def test_pending_stream_gate_cannot_dispatch(self):
        f=Fixture()
        with self.assertRaises(ValueError): f.phase(lambda r:r['gates'].update(stream_inventory_and_jax=False))
        self.assertNotIn('phase/claimed',f.store.raw)
    def test_independent_review_wrong_seal(self):
        f=Fixture()
        with self.assertRaises(ValueError): f.phase(lambda r:r.update(seal_sha256=H('other')))
    def test_started_durable_before_callback(self):
        f=Fixture();p=f.phase();calls=[]
        def callback(bundle,continuation):
            self.assertIn('phase/panel000/started',f.store.raw);self.assertEqual(continuation,'host')
            self.assertEqual(bundle['capture']['state_index'],0);calls.append(1)
            bundle['bank']['pair'][0]='mutated private copy';return dict(synthetic=True)
        r=p.run_next(callback);self.assertEqual(calls,[1]);self.assertIn(r['name'],f.store.raw)
        self.assertEqual(load(f.store.read,f.index['rows'][0]['bank'])['pair'],PAIR)
    def test_all512_nominal_panels_missing_no_callbacks(self):
        f=Fixture(captured=());p=f.phase();calls=[]
        for i in range(512): p.run_next(lambda *_:calls.append(1))
        self.assertEqual(calls,[]);self.assertIn('phase/panel511/completed',f.store.raw)
        import json
        self.assertEqual(json.loads(f.store.raw['phase/panel255/started'])['continuation'],'host')
        self.assertEqual(json.loads(f.store.raw['phase/panel256/started'])['continuation'],'bca')
        with self.assertRaises(ValueError): p.run_next(lambda *_:{})
    def test_callback_failure_keeps_start_no_retry(self):
        f=Fixture();p=f.phase();calls=[]
        def callback(*_): calls.append(1);raise RuntimeError('synthetic callback failure')
        with self.assertRaises(RuntimeError): p.run_next(callback)
        with self.assertRaises(ValueError): p.run_next(callback)
        self.assertEqual(calls,[1]);self.assertIn('phase/panel000/started',f.store.raw)
        self.assertNotIn('phase/panel000/completed',f.store.raw)
    def test_durability_failure_prevents_callback(self):
        f=Fixture();p=f.phase();f.store.fail='phase/panel000/started';calls=[]
        with self.assertRaises(OSError): p.run_next(lambda *_:calls.append(1))
        self.assertEqual(calls,[])
        with self.assertRaises(ValueError): p.run_next(lambda *_:{})
    def test_completion_failure_no_replay(self):
        f=Fixture();p=f.phase();f.store.fail='phase/panel000/completed';calls=[]
        def callback(*_): calls.append(1);return {}
        with self.assertRaises(OSError): p.run_next(callback)
        with self.assertRaises(ValueError): p.run_next(callback)
        self.assertEqual(calls,[1])
    def test_swallowed_reentrancy_still_poisons(self):
        f=Fixture();p=f.phase()
        def callback(*_):
            try: p.run_next(lambda *_:{})
            except ValueError: pass
            return {}
        with self.assertRaises(ValueError): p.run_next(callback)
        self.assertNotIn('phase/panel000/completed',f.store.raw)
    def test_dependency_mutation_in_callback_leaves_pending(self):
        f=Fixture();p=f.phase()
        def callback(*_): f.store.raw['banks/state000']+=b' ';return {}
        with self.assertRaises(ValueError): p.run_next(callback)
        self.assertNotIn('phase/panel000/completed',f.store.raw)
    def test_fork_refusal_before_callback(self):
        f=Fixture();p=f.phase()
        with patch('phase_barrier.os.getpid',return_value=-1):
            with self.assertRaises(ValueError): p.run_next(lambda *_:{})
        self.assertNotIn('phase/panel000/started',f.store.raw)
    def test_ownership_loss_before_callback(self):
        f=Fixture();state=[True]
        def owner():
            if not state[0]: raise ValueError('lease lost')
        p=f.phase(owner=owner);state[0]=False
        with self.assertRaises(ValueError): p.run_next(lambda *_:{})
        self.assertNotIn('phase/panel000/started',f.store.raw)
    def test_second_phase_cannot_reclaim(self):
        f=Fixture();f.phase()
        with self.assertRaises(ValueError):
            OutcomePhase(read=f.store.read,write=f.store.write,assert_owner=lambda:None,seal_ref=f.seal_ref,review_ref=f.review_ref,context=f.context)
    def test_review_changes_during_callback_no_acknowledgment(self):
        f=Fixture();p=f.phase()
        def callback(*_): f.store.raw['review/independent']+=b' ';return {}
        with self.assertRaises(ValueError): p.run_next(callback)
        self.assertNotIn('phase/panel000/completed',f.store.raw)
    def test_missing_quota_has_no_fabricated_warning(self):
        f=Fixture();r=f.index['rows'][0]
        f.repin(r['bank'],lambda b:b['slots'][5].update(status='missing',alias_of=None))
        f.repin(r['warnings'],lambda w:w.update(bank_sha256=r['bank']['sha256']))
        with self.assertRaises(ValueError): f.seal()
    def test_raw_readback_failure_preserves_unacknowledged_seal(self):
        f=Fixture();index=f.pin()
        def read(name): return f.store.read(name)+(b' ' if name=='barrier/precommit' else b'')
        with self.assertRaises(ValueError): seal_precommit(read,f.store.write,index,f.context)
        self.assertIn('barrier/precommit',f.store.raw)
    def test_closed_driver_injected_with_only_synthetic_objects(self):
        # Import fixture helpers and the closed driver; no earlier suite is run.
        from test_drivers import Drivers,Sim,Policy,Ledger,Store as DriverStore
        from recorded_step import RecordedStep,EncodedStore
        from outcome_driver import run_panel
        sim=Sim();sim.elapsed=100;snapshot=sim.capture();bank,keys=Drivers.bank(snapshot)
        f=Fixture();r=f.index['rows'][0]
        f.repin(r['capture'],lambda c:c.update(snapshot=encode(snapshot),snapshot_content_sha256=content_hash(snapshot)))
        f.repin(r['bank'],lambda b:(b.clear(),b.update(bank)))
        def update_w(w):
            w['bank_sha256']=r['bank']['sha256'];w['slots']=[]
            for s in bank['slots']:
                w['slots'].append(dict(slot=s['slot'],status='missing') if s['status']=='missing' else
                    dict(slot=s['slot'],status='present',width=1.,support=s['distance'],random=.25,constant=.5,native_width=None))
        f.repin(r['warnings'],update_w);p=f.phase();ledger=Ledger();sim.ledger=ledger
        archive=DriverStore();store=EncodedStore(archive)
        step=RecordedStep(sim,ledger,store,sim.records.__getitem__,PAIR,'synthetic-global-barrier')
        def callback(bundle,continuation):
            self.assertIn('phase/panel000/started',f.store.raw)
            self.assertEqual(bundle['capture']['snapshot'],encode(snapshot));self.assertEqual(bundle['bank'],bank)
            return encode(run_panel(sim=sim,policy=Policy(),snapshot=snapshot,bank=bank,
                bank_sha256=r['bank']['sha256'],snapshot_sha256=content_hash(snapshot),keys=keys,
                continuation=continuation,step=step,store=store))
        completed=p.run_next(callback)
        self.assertEqual(sim.transitions,4*251);self.assertEqual(len(ledger.completed),4*251)
        self.assertIsNone(ledger.pending)
        result=load(f.store.read,completed)['result'];self.assertEqual(result['unique_applied_actions'],4)
        self.assertTrue(result['complete_final_state_contents_saved']);self.assertFalse(result['scientific_acceptance'])


if __name__=='__main__': unittest.main(verbosity=2)
