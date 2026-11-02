"""State-collection contract checks with fake dynamics; no simulator calls."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import fresh_starts as f


class Env:
    def __init__(self, seed):
        self.state=np.zeros(13,np.float64)
        self.state[12]=(int(seed) % 97)/100.
        self.t=0
    def close(self):
        pass


class Ref:
    @staticmethod
    def observation(states,times):
        return np.concatenate((states,((100-times)/100.)[:,None]),axis=1).astype(np.float32)
    @staticmethod
    def _step(env,action,ledger,stage,deadline,progress):
        assert stage in ('random_prefix','actor_prefix')
        env.state[:4] += np.asarray(action,np.float32).astype(np.float64)
        env.t+=1
        ledger['transition_steps_attempted']+=1;ledger['transition_steps_completed']+=1
        return Ref.observation(env.state[None,:],np.asarray([env.t],np.int64))[0],0.
    @staticmethod
    def actor_actions(policy,obs,ledger,deadline=None,progress=None):
        ledger['actor_rows']+=len(obs)
        return policy.act(obs)


class Eval:
    @staticmethod
    def _reset(core,ref,seed,law,ledger,deadline,progress):
        ledger['resets_attempted']+=1;ledger['resets_completed']+=1
        return Env(seed)


class Policy:
    def __init__(self,value): self.value=value; self.checks=0
    def act(self,obs): return np.full((len(obs),4),self.value,np.float32)
    def assert_unchanged(self): self.checks+=1


class StartsTest(unittest.TestCase):
    def setUp(self):
        self.core=SimpleNamespace(canonical=SimpleNamespace(START_LAWS={'recorded':{},'strong_current':{}}))
        self.ledger={'transition_steps_completed':0,'transition_steps_attempted':0,
                     'resets_completed':0,'resets_attempted':0,'actor_rows':0}

    def test_random_state_times_seed_and_budget(self):
        x=f.random_pool(self.core,Ref,Eval,'ordinary',self.ledger)
        self.assertEqual(self.ledger['transition_steps_completed'],600)
        self.assertEqual(self.ledger['resets_completed'],16)
        np.testing.assert_array_equal(x['times'],np.repeat([0,25,50,75],4))
        self.assertEqual(len(np.unique(x['seeds'])),16)
        np.testing.assert_allclose(x['observations'][:,-1],(100-x['times'])/100.,rtol=0,atol=1e-7)

    def test_visited_state_uses_frozen_actor_and_all_episode_times(self):
        actor=Policy(.25)
        x=f.visited_pool(self.core,Ref,Eval,actor,'strong_current',1,'bca',self.ledger)
        self.assertEqual(self.ledger['transition_steps_completed'],300)
        self.assertEqual(self.ledger['actor_rows'],300)
        self.assertEqual(self.ledger['resets_completed'],8)
        self.assertEqual(actor.checks,2)
        np.testing.assert_array_equal(x['times'],np.repeat([0,25,50,75],2))
        np.testing.assert_allclose(x['states'][:,0],x['times']*.25)
        self.assertEqual((x['seed'],x['arm']),(1,'bca'))

    def test_actual_seed_audit_rejects_previous_evaluation_seed(self):
        self.assertEqual(f.seed_audit()['new_reset_seeds'],128)
        with patch.dict(f.RANDOM_SEEDS,{'ordinary':2026122000}):
            with self.assertRaisesRegex(ValueError,'overlap'):
                f.seed_audit()


if __name__=='__main__': unittest.main()
