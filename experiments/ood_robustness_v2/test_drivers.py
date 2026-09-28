import copy
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest
import numpy as np
from recorded_step import RecordedStep, EncodedStore, content_hash, encode
from collector_driver import collect_states
from outcome_driver import run_panel
from precommit_bank import PIN_NAMES,bind_keys,build_precommit
from precommit_file import canonical
from candidate_design import applied_unit
from ancestor_guard import ExclusiveLease
from extension_ledger import ExtensionLedger,declaration
from test_extension_ledger import fixture  # fixture helper only; no closed tests rerun

PAIR=('td3_bc','hopper',202609171)


class Store:
    def __init__(self): self.raw={};self.fail=None
    def append(self,name,raw):
        if self.fail and self.fail in name: raise OSError('synthetic archive failure')
        if name in self.raw: raise ValueError('duplicate evidence')
        self.raw[name]=raw
    def read(self,name): return self.raw[name]


class Ledger:
    def __init__(self): self.calls=[];self.tokens=set();self.pending=None;self.completed=[]
    def call(self,token,scope,callback):
        if self.pending is not None or token in self.tokens: raise ValueError('pending/duplicate')
        self.calls.append((token,scope));self.tokens.add(token);self.pending=token
        result,h=callback();self.completed.append((token,h));self.pending=None;return result


class Sim:
    def __init__(self,stop=1000):
        self.transitions=0;self.stop=stop;self.elapsed=0;self.seed=0;self.terminated=False;self.truncated=False
        self.records={};self.fault=None;self.ledger=None
    def capture(self):
        return dict(schema='synthetic-full-state',elapsed_steps=self.elapsed,terminated=self.terminated,
                    truncated=self.truncated,hidden=np.asarray([self.seed,self.elapsed+(1 if self.fault=='repeat_state' and self.transitions==2 else 0)],np.float64),inactive=None,
                    rng=('fixture',np.asarray([self.seed],np.uint32)))
    snapshot_hash=staticmethod(content_hash)
    def restore(self,snapshot,expected_sha256):
        if content_hash(snapshot)!=expected_sha256: raise ValueError('wrong restore hash')
        self.elapsed=snapshot['elapsed_steps'];self.terminated=snapshot['terminated'];self.truncated=snapshot['truncated'];self.seed=int(snapshot['hidden'][0])
    def reset(self,seed):
        self.seed=seed;self.elapsed=0;self.terminated=self.truncated=False
        return np.zeros(11,np.float64)
    def step(self,action):
        if self.ledger is not None and self.ledger.pending is None: raise AssertionError('unreserved callback')
        self.transitions+=1;self.elapsed+=1
        if self.fault=='step': raise RuntimeError('synthetic physical uncertainty')
        applied=applied_unit(action);self.terminated=self.elapsed>=self.stop;self.truncated=False
        if self.fault=='truncation' and self.terminated: self.terminated=False;self.truncated=True
        cost=-.001*float(np.square(applied.astype(np.float64)).sum())
        r=dict(proposed_action=action.copy(),applied_action=applied,sim_ctrl=applied.astype(np.float64),
            observation=np.full(11,float(self.elapsed)),reward=2.+cost,forward=1.,alive=1.,action_cost=cost,
            reconstructed_reward=2.+cost,reward_error=0.,terminated=self.terminated,truncated=self.truncated,
            elapsed_steps=self.elapsed,physics_steps=4)
        if self.fault=='control': r['applied_action']=np.ones_like(action)
        if self.fault=='reward': r['reward_error']=1e-6
        if self.fault=='repeat' and self.transitions%2==0: r['reward']+=.1;r['forward']+=.1;r['reconstructed_reward']+=.1
        for suffix in ('-input','-applied',''):
            self.records[f'transition-{self.transitions:05d}{suffix}.npz']=('synthetic raw '+suffix+str(self.transitions)).encode()
        return r


class Policy:
    def __init__(self): self.seals=0;self.queries=0
    def seal(self): self.seals+=1
    def actions(self,obs): self.queries+=1;return np.zeros((len(obs),3),np.float32)


class Drivers(unittest.TestCase):
    def setUp(self):
        self.sim=Sim();self.ledger=Ledger();self.sim.ledger=self.ledger
        self.archive=Store();self.store=EncodedStore(self.archive)
        self.step=RecordedStep(self.sim,self.ledger,self.store,self.sim.records.__getitem__,PAIR,'synthetic-attempt')
        self.policy=Policy()
    @staticmethod
    def bank(snapshot):
        keys=np.arange(128000,dtype=np.uint32).reshape(256,250,2)
        receipt=bind_keys(PAIR,keys,np.asarray([[999999,999998]],np.uint32),np.empty((0,2),np.uint32))
        pins={n:hashlib.sha256(n.encode()).hexdigest() for n in PIN_NAMES};pins['state_snapshot']=content_hash(snapshot)
        b=build_precommit(pair=PAIR,state_index=0,pins=pins,key_receipt=receipt,keys=keys[0],
            anchor_sent=np.zeros(3,np.float32),neighboring_actions=np.zeros((32,3),np.float32),
            host_sent=np.zeros(3,np.float32),bca_sent=np.zeros(3,np.float32),q95=1.1,q99=1.2,state_distance=0.,state_q95=.3)
        return b,keys[0]
    def panel(self,**updates):
        self.sim.elapsed=100
        snap=self.sim.capture();b,k=self.bank(snap)
        args=dict(sim=self.sim,policy=self.policy,snapshot=snap,bank=b,bank_sha256=hashlib.sha256(canonical(b)).hexdigest(),
                  snapshot_sha256=content_hash(snap),keys=k,continuation='bca',step=self.step,store=self.store)
        args.update(updates);return run_panel(**args)
    def test_reserve_and_full_state_before_completion(self):
        out=self.step('outcomes','bca/state000/slot0/0',np.zeros(3,np.float32))
        self.assertEqual(self.sim.transitions,1);self.assertIsNone(self.ledger.pending)
        self.assertEqual(len(self.ledger.completed),1)
        self.assertEqual(out['after']['elapsed_steps'],1)
        self.assertIn('calls/bca/state000/slot0/0/completed',self.archive.raw)
        self.assertIn('calls/bca/state000/slot0/0/after',self.archive.raw)
        self.assertEqual(self.ledger.calls[0][1],'outcomes/td3_bc/hopper/s202609171')
    def test_before_write_failure_charged_no_callback(self):
        self.archive.fail='/before'
        with self.assertRaises(OSError): self.step('outcomes','one',np.zeros(3,np.float32))
        self.assertEqual(self.sim.transitions,0);self.assertIsNotNone(self.ledger.pending)
    def test_after_write_failure_charged_no_next_callback(self):
        self.archive.fail='/after'
        with self.assertRaises(OSError): self.step('outcomes','one',np.zeros(3,np.float32))
        with self.assertRaises(ValueError): self.step('outcomes','two',np.zeros(3,np.float32))
        self.assertEqual(self.sim.transitions,1);self.assertEqual(len(self.ledger.completed),0)
    def test_physical_failure_not_refunded(self):
        self.sim.fault='step'
        with self.assertRaises(RuntimeError): self.step('outcomes','one',np.zeros(3,np.float32))
        self.assertIsNotNone(self.ledger.pending);self.assertEqual(len(self.ledger.calls),1)
    def test_invalid_phase_owner_action_before_reserve(self):
        for phase,owner,action in [('wrong','x',np.zeros(3,np.float32)),('outcomes','../x',np.zeros(3,np.float32)),('outcomes','x',np.zeros(3,np.float64))]:
            with self.assertRaises(ValueError): self.step(phase,owner,action)
        self.assertEqual(self.ledger.calls,[])
    def test_control_gate_preserves_pending_output(self):
        self.sim.fault='control'
        with self.assertRaises(ValueError): self.step('outcomes','one',np.zeros(3,np.float32))
        self.assertIn('calls/one/after',self.archive.raw);self.assertIsNotNone(self.ledger.pending)
    def test_reward_gate_unchanged(self):
        self.sim.fault='reward'
        with self.assertRaises(ValueError): self.step('outcomes','one',np.zeros(3,np.float32))
        self.assertIsNotNone(self.ledger.pending)
    def test_missing_raw_output_leaves_pending(self):
        self.step.raw_read=lambda name: (_ for _ in ()).throw(KeyError(name))
        with self.assertRaises(KeyError): self.step('outcomes','one',np.zeros(3,np.float32))
        self.assertIsNotNone(self.ledger.pending)
    def test_horizon_aliases_missing_and_final_contents(self):
        p=self.panel();self.assertEqual(len(p['slots']),14)
        self.assertEqual(self.sim.transitions,4*251)
        self.assertEqual([r['status'] for r in p['slots'][4:12]],['missing']*8)
        self.assertEqual(p['slots'][12]['source_slot'],0)
        self.assertEqual(p['slots'][13]['source_slot'],0)
        for r in p['slots'][:4]:
            self.assertEqual(r['length'],250);self.assertTrue(r['horizon_exhausted'])
            self.assertEqual(r['raw_return'],sum(float(x) for x in r['rewards']))
            self.assertEqual(r['final_state']['elapsed_steps'],350)
        self.assertEqual(p['reference_slot'],12);self.assertEqual(p['anchor_slot'],0)
    def test_first_transition_termination_repeat_still_checked(self):
        self.sim.stop=101;p=self.panel()
        self.assertEqual(self.sim.transitions,8);self.assertEqual(self.policy.queries,0)
        self.assertTrue(all(r['length']==1 for r in p['slots'][:4]))
    def test_early_termination_no_extra_step(self):
        self.sim.stop=107;p=self.panel()
        self.assertEqual(self.sim.transitions,4*8)
        self.assertTrue(all(r['length']==7 for r in p['slots'][:4]))
    def test_repeat_mismatch_stops_after_two_calls(self):
        self.sim.fault='repeat'
        with self.assertRaises(ValueError): self.panel()
        self.assertEqual(self.sim.transitions,2)
        self.assertNotIn('panels/bca/state000/slot0/completed',self.archive.raw)
    def test_precommit_hash_refusal_before_callback(self):
        with self.assertRaises(ValueError): self.panel(bank_sha256='0'*64)
        self.assertEqual(self.sim.transitions,0)
    def test_source_snapshot_refusal_before_callback(self):
        with self.assertRaises(ValueError): self.panel(snapshot_sha256='0'*64)
        self.assertEqual(self.sim.transitions,0)
    def test_wrong_capture_step_refused(self):
        snap=self.sim.capture()
        with self.assertRaises(ValueError): self.panel(snapshot=snap,snapshot_sha256=content_hash(snap))
        self.assertEqual(self.sim.transitions,0)
    def test_wrong_keys_refused(self):
        with self.assertRaises(ValueError): self.panel(keys=np.zeros((250,2),np.uint32))
        self.assertEqual(self.sim.transitions,0)
    def test_restore_failure_refused_before_physics(self):
        self.sim.restore=lambda state,expected_sha256: None
        self.sim.seed=2
        snap=self.sim.capture();snap['elapsed_steps']=100;snap['hidden'][0]=3
        with self.assertRaises(ValueError): self.panel(snapshot=snap,snapshot_sha256=content_hash(snap))
        self.assertEqual(self.sim.transitions,0)
    def test_repeat_hidden_state_mismatch(self):
        self.sim.fault='repeat_state'
        with self.assertRaises(ValueError): self.panel()
        self.assertEqual(self.sim.transitions,2)
    def test_native_truncation_stops_panel(self):
        self.sim.stop=107;self.sim.fault='truncation';p=self.panel()
        self.assertEqual(self.sim.transitions,32)
        self.assertTrue(all(r['truncated'] and not r['terminated'] for r in p['slots'][:4]))
    def test_completion_evidence_failure_stays_pending(self):
        self.archive.fail='/completed'
        with self.assertRaises(OSError): self.step('outcomes','one',np.zeros(3,np.float32))
        self.assertEqual(self.sim.transitions,1);self.assertEqual(len(self.ledger.completed),0)
        self.assertIn('calls/one/after',self.archive.raw)
    def test_collection_early_terminal_all_missing_no_replacement(self):
        self.sim.stop=3
        result=collect_states(self.sim,{'host':Policy(),'bca':Policy()},PAIR,self.step,self.store,lambda sim,seed:sim.reset(seed))
        self.assertEqual(result['captured'],0);self.assertEqual(result['missing'],256)
        self.assertEqual(self.sim.transitions,128*3)
        self.assertTrue(all(r['status']=='missing' for r in result['rows']))
    def test_collection_exact_schedule(self):
        result=collect_states(self.sim,{'host':Policy(),'bca':Policy()},PAIR,self.step,self.store,lambda sim,seed:sim.reset(seed))
        self.assertEqual(result['captured'],256);self.assertEqual(self.sim.transitions,128*300)
        self.assertEqual([r['snapshot']['elapsed_steps'] for r in result['rows'][:4]],[100,300,100,300])
        self.assertEqual(result['rows'][0]['reset_seed'],result['rows'][128]['reset_seed'])
    def test_collection_terminal_at_capture_is_missing(self):
        self.sim.stop=100
        r=collect_states(self.sim,{'host':Policy(),'bca':Policy()},PAIR,self.step,self.store,lambda sim,seed:sim.reset(seed))
        self.assertEqual(r['captured'],0);self.assertEqual(r['missing'],256)
    def test_encoded_values_preserve_full_schema_bytes(self):
        x={'array':np.asarray([-0.,1.],np.float32),'rng':('x',np.arange(3,dtype=np.uint32)),'none':None}
        self.assertIn('__tuple__',encode(x)['rng'])
        self.assertNotEqual(content_hash(x),content_hash(dict(x,array=np.asarray([0.,1.],np.float32))))
        with self.assertRaises(ValueError): encode(np.asarray([np.nan]))
    def test_real_component_temp_journal_integration(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);old,lock,binding,guard=fixture(root);original=old.read_bytes();path=root/'extension.sqlite'
            with ExclusiveLease(lock) as lease, ExtensionLedger.create(path,guard,lease,declaration(path,lock,binding,'4'*64)) as ledger:
                self.sim.ledger=None
                step=RecordedStep(self.sim,ledger,self.store,self.sim.records.__getitem__,PAIR,'temporary')
                step('outcomes','one',np.zeros(3,np.float32))
                self.assertEqual(ledger.audit()['reserved']['global'],[1,4])
                self.archive.fail='/after'
                with self.assertRaises(OSError): step('outcomes','two',np.zeros(3,np.float32))
                self.assertEqual(ledger.audit()['reserved']['global'],[2,8]);self.assertEqual(len(ledger.audit()['pending']),1)
            self.assertEqual(old.read_bytes(),original)


if __name__=='__main__': unittest.main(verbosity=2)
