import copy
import hashlib
from types import SimpleNamespace
import unittest
import numpy as np
from native_constructor import construct_recorded,capture_native,validate_constructor
from recorded_step import EncodedStore,content_hash
from test_drivers import Ledger,Store  # fixtures only, no closed test reruns


class Physics:
    def do_simulation(self,control,frames):
        self.real_calls+=1
        if self.probe: self.probe()
        self.sim.data.ctrl[:]=control
        self.sim.data.qpos[0]+=self.dt
        self.sim.data.time+=self.dt


class Native(Physics):
    def __init__(self,dim=3,fault=None,probe=None):
        self.fault=fault;self.probe=probe;self.real_calls=0;self.closed=False;self.frame_skip=4;self.dt=.008
        self.action_space=SimpleNamespace(low=-np.ones(dim,np.float32),high=np.ones(dim,np.float32),_np_random=np.random.RandomState(17))
        n=6 if dim==3 else 9
        data=SimpleNamespace(qpos=np.r_[0.,1.,0.,np.zeros(n-3)],qvel=np.zeros(n),act=None,qacc_warmstart=np.zeros(n),
              ctrl=np.zeros(dim),qfrc_applied=np.zeros(n),xfrc_applied=np.zeros((5,6)),mocap_pos=None,mocap_quat=None,userdata=None,time=0.)
        model=SimpleNamespace(body_pos=np.zeros((5,3)),body_quat=np.zeros((5,4)),site_pos=np.zeros((1,3)),
              actuator_ctrlrange=np.tile([-1.,1.],(dim,1)),get_mjb=lambda:b'synthetic-model')
        self.sim=SimpleNamespace(data=data,model=model,get_state=lambda:SimpleNamespace(udd_state={}))
    @property
    def np_random(self): raise AssertionError('lazy RNG property accessed')
    def step(self,action):
        if self.fault=='step': raise RuntimeError('synthetic native failure')
        before=self.sim.data.qpos[0]
        frames=5 if self.fault=='frames' else 4
        self.do_simulation(action,frames)
        if self.fault=='after_physics': raise RuntimeError('synthetic missing native return after physics')
        if self.fault=='twice_physics': self.do_simulation(action,4)
        reward=(self.sim.data.qpos[0]-before)/self.dt+1.-.001*np.square(action).sum()
        obs=np.r_[self.sim.data.qpos[1:],np.clip(self.sim.data.qvel,-10,10)]
        if self.fault=='reward': reward+=1e-4
        if self.fault=='observation': obs[0]+=.1
        if self.fault=='model': self.sim.model.body_pos[0,0]=1.
        if self.fault=='ctrl': self.sim.data.ctrl[0]+=.1
        if self.fault=='done': return obs,reward,True,{}
        return obs,reward,False,{}
    def close(self): self.closed=True


class Constructor(unittest.TestCase):
    def setUp(self):
        self.archive=Store();self.store=EncodedStore(self.archive);self.ledger=Ledger();self.created=[]
        self.original_step=Native.step;self.original_physics=Physics.do_simulation
    def tearDown(self):
        self.assertIs(Native.step,self.original_step);self.assertIs(Physics.do_simulation,self.original_physics)
    def run_case(self,dim=3,fault=None,factory_mode=None):
        def probe():
            self.assertIsNotNone(self.ledger.pending)
            self.assertIn('constructor/input',self.archive.raw)
            self.assertIn('constructor/applied-before-physics',self.archive.raw)
        def factory():
            self.assertIsNotNone(self.ledger.pending)
            env=Native(dim,fault,probe);self.created.append(env)
            action=np.linspace(-.31,.42,dim,dtype=np.float32)
            if factory_mode!='no_step': env.step(action)
            if factory_mode=='twice_step': env.step(action)
            if factory_mode=='raise_after': raise RuntimeError('synthetic factory failure after physics')
            return SimpleNamespace(base=env,transitions=0,frame_skip=4,action_dim=dim,close=env.close)
        return construct_recorded(factory,Native,Physics,self.ledger,self.store,
                                  ('td3_bc','hopper' if dim==3 else 'walker2d',202609171),'fixture-constructor')
    def test_hopper_nonzero_native_action_and_durable_order(self):
        adapter,r=self.run_case()
        self.assertEqual(adapter.base.real_calls,1);self.assertFalse(adapter.base.closed)
        self.assertEqual(r['environment_reserved'],1);self.assertEqual(r['physics_reserved'],4)
        self.assertEqual(self.ledger.calls[0][1],'engineering/td3_bc/hopper')
        self.assertIsNone(self.ledger.pending)
        self.assertTrue(r['full_native_states_saved']);self.assertFalse(r['wrapper_restore_state_claimed'])
    def test_walker_same_gates(self):
        adapter,r=self.run_case(dim=6)
        self.assertEqual(adapter.action_dim,6);self.assertEqual(adapter.base.real_calls,1)
        self.assertEqual(r['observation_size'],17)
    def test_rng_absence_does_not_initialize(self):
        n=Native();s=capture_native(n)
        self.assertEqual(s['environment_rng'],dict(status='not_initialized',state=None))
        self.assertEqual(s['action_rng']['status'],'present')
        self.assertNotIn('_np_random',n.__dict__)
    def test_existing_rng_copied(self):
        n=Native();n._np_random=np.random.default_rng(4);s=capture_native(n)
        self.assertEqual(s['environment_rng']['status'],'present')
        old=content_hash(s);n._np_random.random();self.assertEqual(content_hash(s),old)
    def test_native_fields_copied_not_aliased(self):
        n=Native();s=capture_native(n);n.sim.data.qpos[0]=99
        self.assertEqual(s['data']['qpos'][0],0.);self.assertIsNone(s['data']['act'])
        self.assertFalse(s['wrapper_state_available'])
    def test_missing_native_field_refused(self):
        n=Native();del n.sim.data.qacc_warmstart
        with self.assertRaises(ValueError): capture_native(n)
    def test_no_step_charge_retained(self):
        with self.assertRaises(ValueError): self.run_case(factory_mode='no_step')
        self.assertEqual(self.created[0].real_calls,0);self.assertIsNotNone(self.ledger.pending)
    def test_second_step_refused_before_second_physics(self):
        with self.assertRaises(ValueError): self.run_case(factory_mode='twice_step')
        self.assertEqual(self.created[0].real_calls,1);self.assertIsNotNone(self.ledger.pending)
    def test_second_physics_refused_before_call(self):
        with self.assertRaises(ValueError): self.run_case(fault='twice_physics')
        self.assertEqual(self.created[0].real_calls,1);self.assertIsNotNone(self.ledger.pending)
    def test_wrong_frames_refused_before_physics(self):
        with self.assertRaises(ValueError): self.run_case(fault='frames')
        self.assertEqual(self.created[0].real_calls,0)
    def test_before_archive_failure_no_physics(self):
        self.archive.fail='applied-before'
        with self.assertRaises(OSError): self.run_case()
        self.assertEqual(self.created[0].real_calls,0);self.assertIsNotNone(self.ledger.pending)
    def test_post_archive_failure_preserves_charge(self):
        self.archive.fail='/output'
        with self.assertRaises(OSError): self.run_case()
        self.assertEqual(self.created[0].real_calls,1);self.assertIsNotNone(self.ledger.pending)
    def test_completion_archive_failure_preserves_charge(self):
        self.archive.fail='/completed'
        with self.assertRaises(OSError): self.run_case()
        self.assertEqual(len(self.ledger.completed),0);self.assertEqual(self.created[0].real_calls,1)
    def test_reward_mismatch_preserves_native_output(self):
        with self.assertRaises(ValueError): self.run_case(fault='reward')
        self.assertIn('constructor/output',self.archive.raw);self.assertIn('constructor/after',self.archive.raw)
        self.assertIsNotNone(self.ledger.pending)
    def test_observation_mismatch(self):
        with self.assertRaises(ValueError): self.run_case(fault='observation')
        self.assertIsNotNone(self.ledger.pending)
    def test_model_mutation(self):
        with self.assertRaises(ValueError): self.run_case(fault='model')
        self.assertIsNotNone(self.ledger.pending)
    def test_actual_control_mismatch(self):
        with self.assertRaises(ValueError): self.run_case(fault='ctrl')
        self.assertIsNotNone(self.ledger.pending)
    def test_native_done_mismatch(self):
        with self.assertRaises(ValueError): self.run_case(fault='done')
        self.assertIsNotNone(self.ledger.pending)
    def test_factory_failure_cleanup_and_no_retry(self):
        with self.assertRaises(RuntimeError): self.run_case(factory_mode='raise_after')
        self.assertTrue(self.created[0].closed);self.assertIsNotNone(self.ledger.pending)
        with self.assertRaises(ValueError): self.run_case()
        self.assertEqual(len(self.created),1)
    def test_native_failure_cleanup_and_method_restore(self):
        with self.assertRaises(RuntimeError): self.run_case(fault='step')
        self.assertTrue(self.created[0].closed);self.assertEqual(self.created[0].real_calls,0)
    def test_after_physics_exception_saves_available_poststate(self):
        with self.assertRaises(RuntimeError): self.run_case(fault='after_physics')
        self.assertEqual(self.created[0].real_calls,1)
        self.assertIn('constructor/failed-native-after',self.archive.raw)
        self.assertNotIn('constructor/output',self.archive.raw)
        self.assertIsNotNone(self.ledger.pending)


if __name__=='__main__': unittest.main(verbosity=2)
