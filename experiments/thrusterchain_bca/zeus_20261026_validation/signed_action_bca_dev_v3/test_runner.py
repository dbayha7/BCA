"""Contract tests; fake dynamics only, no simulator interactions."""
import unittest
from types import SimpleNamespace
import numpy as np
import runner


class FakeCore:
    @staticmethod
    def Physics(**kwargs):
        return None

    @staticmethod
    def dynamics(states, actions, physics):
        x=np.asarray(states,np.float64)
        a=np.asarray(actions,np.float64)
        nxt=x.copy()
        nxt[..., :4] += .1*a
        return nxt, x[...,0]*.05 - np.square(a).sum(axis=-1)


class Policy:
    def act(self, obs):
        return np.tanh(np.asarray(obs)[:,:4]*.1).astype(np.float32)
    def assert_unchanged(self):
        pass


class RunnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This authenticates source/adapter only; all dynamics below are fakes.
        _,cls.reference,_=runner.frozen.load_simulator()

    def fixtures(self):
        state=np.zeros((4,13),np.float64)
        state[:,0]=[.1,.2,.3,.4]
        times=np.asarray([0,25,75,99],np.int64)
        acts=np.random.Generator(np.random.PCG64(4)).uniform(-1,1,(4,15,4)).astype(np.float32)
        return state,times,acts

    def test_general_matches_authenticated_six_action_continuation(self):
        state,times,acts=self.fixtures()
        l1=self.reference.new_ledger(100000,100000);l2=self.reference.new_ledger(100000,100000)
        old=self.reference.continuation_values(FakeCore,Policy(),state,times,acts[:,:6],l1)
        new=runner.continuation_general(FakeCore,self.reference,Policy(),state,times,acts[:,:6],l2)
        np.testing.assert_array_equal(old,new)
        self.assertEqual(l1['transition_steps_completed'],l2['transition_steps_completed'])
        self.assertEqual(l2['continuation_rows_completed'],24)
        # At t=99 there is exactly the candidate reward and no continuation.
        self.assertAlmostEqual(new[3,0],state[3,0]*.05-float(np.square(acts[3,0].astype(np.float64)).sum()))

    def test_mixed_15_action_path_matches_independent_general_and_budget(self):
        state,times,acts=self.fixtures()
        l1=self.reference.new_ledger(100000,100000);l2=self.reference.new_ledger(100000,100000)
        mixed=runner.continuation15(FakeCore,self.reference,Policy(),state,times,acts,l1)
        entire=runner.continuation_general(FakeCore,self.reference,Policy(),state,times,acts,l2)
        np.testing.assert_array_equal(mixed,entire)
        self.assertEqual(l1['continuation_rows_completed'],60)
        self.assertEqual(l1['transition_steps_completed'],int((100-times).sum())*15)
        with self.assertRaisesRegex(ValueError,'15 candidate'):
            runner.continuation15(FakeCore,self.reference,Policy(),state,times,acts[:,:6],l1)

    def test_ranking_ties_excluded_and_fixed_band_indices(self):
        acts=np.zeros((32,15,4),np.float32)
        for j in range(1,15): acts[:,j,0]=j/15
        values=np.ones((32,15),np.float64);values[:,0]=0
        raw=np.full((32,15),1e-7);raw[:,0]=0
        obs=np.zeros((32,14),np.float32);obs[:,10]=.5
        strata=['natural']*16+['ambiguous_q1']*16
        result,_=runner.summarize(acts,values,{'raw_q1':raw},obs,strata)
        s=result['natural']['scores']['raw_q1']
        self.assertEqual(s['ranking_informative_pairs'],128)
        self.assertEqual(s['ranking_tied_score'],128)
        self.assertEqual(s['ranking_correct'],0)
        self.assertEqual(s['ranking_accuracy'],0.)
        self.assertEqual(s['high_load_fixed_band_error']['rows'],48)

    def test_insufficient_opportunity_inconclusive(self):
        cases=[]
        names=('raw_q1','sym_compat_guarded','sym_hybrid_strength_matched','q1_aligned')
        for arm in ('native','bca'):
            for seed in range(3):
                s={n:{'mean_regret':.2,'harmed':0,'improved':0} for n in names}
                section={'informative_families':7,'raw_mistake_families':1,'scores':s}
                cases.append({'seed':seed,'arm':arm,'contexts':{c:{'summary':{
                    'natural':section,'ambiguous_q1':section}} for c in runner.old.CONTEXTS}})
        r=runner.screen(cases)
        self.assertEqual(r['ordinary']['natural']['status'],'inconclusive')
        self.assertEqual(r['strong_current']['ambiguous_q1']['status'],'inconclusive')

    def test_screen_requires_prespecified_improvement_over_both_controls(self):
        cases=[]
        for arm in ('native','bca'):
            for seed in range(3):
                scores={name:{'mean_regret':regret,'harmed':0,'improved':1} for name,regret in [
                    ('raw_q1',.2),('sym_compat_guarded',.1),
                    ('sym_hybrid_strength_matched',.2),('q1_aligned',.15)]}
                z={'informative_families':8,'raw_mistake_families':1,'scores':scores}
                cases.append({'seed':seed,'arm':arm,'contexts':{c:{'summary':{
                    'natural':z,'ambiguous_q1':z}} for c in runner.old.CONTEXTS}})
        self.assertEqual(runner.screen(cases)['ordinary']['natural']['status'],'passed')
        for c in cases:
            c['contexts']['ordinary']['summary']['natural']['scores']['sym_hybrid_strength_matched']['mean_regret']=.149
        self.assertEqual(runner.screen(cases)['ordinary']['natural']['status'],'failed')


if __name__=='__main__': unittest.main()
