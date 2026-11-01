"""Common-reference contract: fake dynamics and hand-computed return fixtures."""
import unittest
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
        return x.copy(), x[...,0] - np.square(a).sum(axis=-1)


class ZeroPolicy:
    def act(self,obs):
        return np.zeros((len(obs),4),np.float32)
    def assert_unchanged(self):
        pass


class RunnerV4Tests(unittest.TestCase):
    def test_distinct_critics_and_continuations_keep_cross_cells(self):
        # In each state native continuation prefers slot 0, BCA prefers slot 1.
        # The two critics swap raw vs symmetric picks; slot 2 is worst for both.
        n=32
        native_truth=np.full((n,15),-10.,np.float64)
        bca_truth=native_truth.copy()
        native_truth[:,:3]=[1.,0.,-1.]
        bca_truth[:,:3]=[0.,1.,-1.]
        source=np.asarray(['random_prefix']*16+['native_visited']*8+['bca_visited']*8)
        guard=np.asarray([False,True]*16)
        def scores(raw_slot,sym_slot):
            raw=np.full((n,15),-100.,np.float64); raw[:,raw_slot]=2.
            symmetric=np.full((n,15),-100.,np.float64); symmetric[:,sym_slot]=2.
            guarded=symmetric.copy(); guarded[guard,sym_slot]=0.;guarded[guard,2]=2.
            return {'raw_q1':raw,'q1_aligned':symmetric,'sym_compat_guarded':guarded,
                    'sym_compat_unguarded':guarded}
        native_score=scores(0,1);bca_score=scores(1,0)
        results={
            'native':{c:runner.summarize(v,native_score,source,guard)
                      for c,v in [('native',native_truth),('bca',bca_truth)]},
            'bca':{c:runner.summarize(v,bca_score,source,guard)
                   for c,v in [('native',native_truth),('bca',bca_truth)]}}
        def gain(critic,cont):
            return results[critic][cont]['random_prefix']['all']['comparisons']['raw_to_symmetric']['mean_gain']
        self.assertEqual(gain('native','native'),-1.)
        self.assertEqual(gain('native','bca'),1.)
        self.assertEqual(gain('bca','native'),1.)
        self.assertEqual(gain('bca','bca'),-1.)
        guarded=results['native']['native']['random_prefix']
        self.assertEqual(guarded['guard_on']['comparisons']['symmetric_to_guarded']['changed'],8)
        self.assertEqual(guarded['guard_off']['comparisons']['symmetric_to_guarded']['changed'],0)
        self.assertEqual(guarded['guard_on']['comparisons']['symmetric_to_guarded']['harmed'],8)

    def test_authenticated_continuation_matches_analytic_horizon_and_budget(self):
        # Fake deterministic dynamics make the own-policy continuation analytic:
        # first reward state[0]-||a||², then state[0] under the zero actor.
        _,ref,_=runner.frozen.load_simulator()
        state=np.zeros((4,13),np.float64); state[:,0]=[.1,.2,.3,.4]
        times=np.asarray([0,25,75,99],np.int64)
        actions=np.random.Generator(np.random.PCG64(5)).uniform(-1,1,(4,15,4)).astype(np.float32)
        ledger=ref.new_ledger(100000,100000)
        result=runner.v3.continuation15(FakeCore,ref,ZeroPolicy(),state,times,actions,ledger)
        expected=np.empty((4,15),np.float64)
        for i,t in enumerate(times):
            tail=sum(.98**j for j in range(1,100-int(t)))
            expected[i]=state[i,0]*(1.+tail)-np.square(actions[i].astype(np.float64)).sum(axis=1)
        np.testing.assert_allclose(result,expected,rtol=0,atol=1e-11)
        self.assertEqual(ledger['transition_steps_attempted'],int((100-times).sum())*15)
        self.assertEqual(ledger['transition_steps_completed'],ledger['transition_steps_attempted'])
        self.assertEqual(ledger['continuation_rows_completed'],60)
        np.testing.assert_allclose(result[-1],state[-1,0]-np.square(actions[-1].astype(np.float64)).sum(axis=1),rtol=0,atol=1e-14)

    def test_primary_screen_uses_own_actor_source_and_own_continuation(self):
        # Off-diagonal and wrong-source cells are made huge but must not affect
        # the screen. Every own-source cell is genuinely informative and passes.
        cases=[]
        for seed in range(3):
            z={'improved':4,'harmed':0,'near_tied':0,'changed':4,
               'mean_gain':.2,'per_state_gain':[.2]*8}
            bad={**z,'mean_gain':-100.,'harmed':8,'improved':0}
            summaries={critic:{cont:{src:{'all':{'comparisons':{'symmetric_to_guarded':
                z if critic==cont and src==critic+'_visited' else bad}}}
                for src in runner.SOURCES_ORDER} for cont in ('native','bca')}
                for critic in ('native','bca')}
            cases.append({'seed':seed,'contexts':{ctx:{'summaries':summaries}
                          for ctx in ('ordinary','strong_current')}})
        out=runner.screen(cases)
        self.assertEqual(out['ordinary']['status'],'passed')
        self.assertEqual(out['strong_current']['qualifying_seeds'],{'native':3,'bca':3})


if __name__=='__main__':unittest.main()
