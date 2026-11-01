"""Scientific contract tests for the cap and score, with no simulation."""
import unittest
from unittest.mock import patch
import numpy as np
import reranker_runner as r


class CappedWidthTests(unittest.TestCase):
    def test_score_clips_action_contrast_not_absolute_width(self):
        q=np.zeros((1,15),np.float64);q[0,1]=.2
        u=np.ones((1,15),np.float64)*10;u[0,1]=10.2
        cost=np.zeros((1,15),np.float64);cap=np.asarray([.1]);guard=np.asarray([False])
        scores,contrast,clipped=r.score_banks(q,u,cost,guard,1.,cap)
        self.assertAlmostEqual(contrast[0,1],.6)
        self.assertAlmostEqual(clipped[0,1],.1)
        self.assertEqual(np.argmax(scores['symmetric'][0]),0)
        self.assertEqual(np.argmax(scores['capped'][0]),1)
        self.assertEqual(np.argmax(scores['raw_q1'][0]),1)
        self.assertEqual(clipped[0,0],0.)
        scores2,_,_=r.score_banks(q,u+100,cost,guard,1.,cap)
        np.testing.assert_allclose(scores['capped'],scores2['capped'],atol=1e-12)
        np.testing.assert_allclose(scores['symmetric']-scores2['symmetric'],300,atol=1e-12)

    def test_guard_matches_original_hybrid(self):
        q=np.ones((2,15));u=np.ones((2,15))*2;d=np.ones((2,15))*3
        scores,_,_=r.score_banks(q,u,d,np.array([False,True]),.5,np.ones(2))
        np.testing.assert_allclose(scores['guarded'][0],-5.)
        np.testing.assert_allclose(scores['guarded'][1],-6.5)

    def test_top_two_deduplicate_full_action_vectors(self):
        a=np.zeros((15,4),np.float32);a[1]=a[0];a[2,1]=.2
        q=np.full(15,-5.);q[:3]=[10,10,8]
        self.assertEqual(r.distinct_top_gap(a,q),2.)

    def test_cap_requires_50_distinct_positive_episodes_per_bin(self):
        ids=np.arange(768,dtype=np.int64)*100+np.tile(np.array([0,25,50,75]),192)
        actions=np.zeros((768,15,4),np.float32);actions[:,1,0]=.2
        q=np.zeros((768,15),np.float64);q[:,1]=np.tile([.1,.2,.3,.4],192)
        caps,counts,gaps=r.caps_from_probes(actions,q,ids)
        np.testing.assert_allclose(caps,[.1,.2,.3,.4]);np.testing.assert_array_equal(counts,[192]*4)
        q[ids%100==0,1]=0
        with self.assertRaisesRegex(ValueError,'fewer than 50'):r.caps_from_probes(actions,q,ids)

    def test_screen_uses_paired_means_and_all_ordinary_cells(self):
        cases=[]
        for arm in ('native','bca'):
            for seed in range(3):
                strong=np.zeros((5,16));strong[3]=2.
                ordinary=np.zeros((5,16));ordinary[3]=-.9
                cases.append({'seed':seed,'arm':arm,'contexts':{
                    'ordinary':{'returns':ordinary.tolist()},'strong_current':{'returns':strong.tolist()}}})
        self.assertEqual(r.screen(cases)['status'],'passed')
        cases[-1]['contexts']['ordinary']['returns'][3]=[-1.1]*16
        self.assertEqual(r.screen(cases)['status'],'failed')

    def test_full_rollout_first_action_divergence_discount_and_budget(self):
        class Reference:
            @staticmethod
            def observation(states,times):
                assert states.dtype==np.float64 and states.shape[:-1]==times.shape
                return np.concatenate((states,(1-times/100)[...,None]),axis=-1).astype(np.float32)
            @staticmethod
            def _dynamics(core,states,actions,ledger,stage,deadline,progress):
                assert stage=='controller' and states.shape==(80,13) and actions.shape==(80,4)
                ledger['rows']+=len(states)
                out=states.copy();out[:,0]+=actions[:,0]
                return out,out[:,0].copy()
        initial=np.zeros((16,13),np.float64)
        first=np.zeros((5,16,4),np.float32)
        first[:,:,0]=np.arange(5,dtype=np.float32)[:,None]*.1
        info={'slots':np.zeros((4,16),np.int64),'guards':np.zeros((4,16),bool)}
        ledger={'rows':0}
        def fixed_selector(c,item,reference,states,t,caps,ledger,deadline):
            assert t > 0
            return first.copy(),{k:v.copy() for k,v in info.items()}
        with patch.object(r,'select_step',fixed_selector):
            roll=r.controller_rollouts(None,Reference,None,None,initial,first,info,np.ones(4),
                                       ledger,None,lambda *args,**kwargs:None)
        self.assertEqual(ledger['rows'],8000)
        np.testing.assert_array_equal(roll['actions'][:,:,0],first)
        np.testing.assert_allclose(roll['observations'][:,0,1,0],first[:,0,0],atol=1e-7)
        discounted=sum((.98**t)*(t+1) for t in range(100))
        np.testing.assert_allclose(roll['returns'][:,0],first[:,0,0]*discounted,rtol=1e-7)


if __name__=='__main__':unittest.main()
