"""Synthetic CPU execution-boundary tests; no checkpoint, JAX or core imports."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
import gpu_execution as gpu


class ExecutionTests(unittest.TestCase):
    def fixture(self,n=2):
        states=np.zeros((n,13),np.float64);states[:,0]=np.arange(n)+1
        actions=np.zeros((n,4),np.float32);actions[:,0]=.3
        return states,actions,np.zeros(n,np.int64)

    @staticmethod
    def dynamics(states,actions,physics):
        nxt=states.copy();nxt[:,0]+=.1
        s2=states[:,0]*states[:,0]
        for j in range(1,10):s2=s2+states[:,j]*states[:,j]
        a2=actions[:,0]*actions[:,0]
        for j in range(1,4):a2=a2+actions[:,j]*actions[:,j]
        return nxt,-(s2+physics.throttle_cost*a2)

    def test_one_step_exact_reward_time_and_charge(self):
        states,actions,times=self.fixture();times[:]=[25,99]
        ledger=gpu.new_ledger()
        result=gpu.one_step(states,actions,times,self.dynamics,SimpleNamespace(throttle_cost=.3),ledger)
        self.assertEqual(ledger['transition_rows_attempted'],2)
        self.assertEqual(ledger['transition_rows_completed'],2)
        np.testing.assert_array_equal(result['done'],[0,1])
        np.testing.assert_array_equal(result['next_obs'][:,-1],np.array([.74,0],np.float32))
        self.assertEqual(result['physics_rewards'].dtype,np.float64)
        self.assertEqual(result['rewards'].dtype,np.float32)
        self.assertEqual(ledger['full_continuations'],0)

    def test_budget_rejects_before_dynamics(self):
        states,actions,times=self.fixture();ledger=gpu.new_ledger();ledger['transition_rows_attempted']=1151
        called=[]
        with self.assertRaisesRegex(ValueError,'budget'):
            gpu.one_step(states,actions,times,lambda *args:called.append(True),SimpleNamespace(throttle_cost=.3),ledger)
        self.assertEqual(called,[]);self.assertEqual(ledger['transition_rows_attempted'],1151)

    def test_post_transition_reward_rejected(self):
        states,actions,times=self.fixture();ledger=gpu.new_ledger()
        def wrong(s,a,p):
            nxt,rewards=self.dynamics(s,a,p)
            return nxt,rewards-1
        with self.assertRaisesRegex(ValueError,'pre-transition reward'):
            gpu.one_step(states,actions,times,wrong,SimpleNamespace(throttle_cost=.3),ledger)
        self.assertEqual(ledger['transition_rows_completed'],2)

    def test_actions_and_times_not_silently_repaired(self):
        states,actions,times=self.fixture();actions[0,0]=1.01
        with self.assertRaisesRegex(ValueError,'clipping'):
            gpu.one_step(states,actions,times,self.dynamics,SimpleNamespace(throttle_cost=.3),gpu.new_ledger())
        actions[0,0]=.2;times[0]=100
        with self.assertRaisesRegex(ValueError,'times'):
            gpu.one_step(states,actions,times,self.dynamics,SimpleNamespace(throttle_cost=.3),gpu.new_ledger())

    def test_noise_and_action_clipping_float32(self):
        actor=np.array([[.9,-.9,.2,-.2]],np.float32)
        raw=np.array([[.8,-.8,.1,-.1]],np.float32)
        clipped,actions=gpu.assemble_smoothed_actions(actor,raw)
        np.testing.assert_array_equal(clipped,np.array([[.5,-.5,.1,-.1]],np.float32))
        np.testing.assert_array_equal(actions,np.array([[1,-1,.3,-.3]],np.float32))

    def test_replay_checks_finite_and_original_tolerance(self):
        expected=np.array([[1.,2.]],np.float32)
        self.assertTrue(gpu.assert_replay(expected+1e-6,expected,'fixture')['passed'])
        with self.assertRaisesRegex(ValueError,'replay failed'):
            gpu.assert_replay(expected+.01,expected,'fixture')
        with self.assertRaisesRegex(ValueError,'Invalid replay'):
            gpu.assert_replay(np.array([[np.nan,2.]]),expected,'fixture')

    def test_current_width_invalid_fails_without_zero_fallback(self):
        ledger=gpu.new_ledger();calls=[]
        policy=SimpleNamespace(q=lambda o,a:np.ones((2,2),np.float32),
                               width=lambda o,a:(np.ones(2,np.float32),False,True))
        binding=SimpleNamespace(policy=policy,assert_unchanged=lambda:calls.append(True))
        bank={'observations':np.zeros((2,14),np.float32),'actions':np.zeros((2,4),np.float32)}
        with self.assertRaisesRegex(ValueError,'invalid or unusable'):
            gpu._current(binding,bank,ledger)
        self.assertEqual(ledger['current_critic_rows'],2)
        self.assertNotIn('u',bank)

    def test_partial_ledger_preserved(self):
        ledger=gpu.new_ledger();ledger['transition_rows_attempted']=9
        error=gpu.OperatorExecutionError('fixture',ledger);ledger['transition_rows_attempted']=10
        self.assertEqual(error.ledger['transition_rows_attempted'],9)

    def test_interrupt_preserves_execution_ledger(self):
        with mock.patch.object(gpu,'_validate_protocol',side_effect=KeyboardInterrupt):
            with self.assertRaises(gpu.OperatorExecutionError) as caught:
                gpu.run({})
        self.assertEqual(caught.exception.ledger['status'],'interrupted')
        self.assertEqual(caught.exception.ledger['transition_rows_attempted'],0)
        self.assertEqual(caught.exception.ledger['policy_optimizer_updates'],0)

    def test_no_top_level_scientific_import_or_work(self):
        tree=ast.parse((Path(__file__).parent/'gpu_execution.py').read_text())
        modules=[]
        for node in tree.body:
            if isinstance(node,ast.Import):modules.extend(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node,ast.ImportFrom):modules.append(node.module.split('.')[0])
        self.assertFalse(set(modules)&{'numpy','jax','flax','optax','thrusterchain','checkpoint_io'})
        source=(Path(__file__).parent/'gpu_execution.py').read_text()
        for prohibited in ('training.step(','.fit(','.refresh(','.continuation_values(','.build_queries(','.candidate_bank('):
            self.assertNotIn(prohibited,source)
        self.assertIn("ledger['transition_rows_attempted'] += n",source)
        self.assertIn('jax.random.PRNGKey(seed)',source)

    def test_preflight_is_metadata_only_and_protocol_fails_early(self):
        with mock.patch.object(gpu.Path,'read_bytes',side_effect=AssertionError('should not read')):
            with self.assertRaisesRegex(ValueError,'budget changed'):
                gpu.verify({})
        tree=ast.parse((Path(__file__).parent/'gpu_execution.py').read_text())
        verify=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='verify')
        names={node.id for node in ast.walk(verify) if isinstance(node,ast.Name)}
        self.assertFalse(names&{'jax','np','core','simulation','checkpoint_io'})
        attributes={node.attr for node in ast.walk(verify) if isinstance(node,ast.Attribute)}
        self.assertFalse(attributes&{'load_batches','dynamics','act','q','width','_core_contract'})


if __name__=='__main__':
    unittest.main()
