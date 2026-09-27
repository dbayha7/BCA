"""Exact toy rollout mechanics, zero real simulator calls."""
import copy,unittest,tempfile,json
from pathlib import Path
import numpy as np
from experiments.ood.collect import sha
from experiments.ood.rollouts import panel

class Policy:
    def __init__(self,action):self.action=action;self.calls=0
    def actions(self,obs):self.calls+=1;return np.array([[self.action]],np.float32)
    def seal(self):pass
class Sim:
    snapshot_hash=staticmethod(sha)
    def __init__(self,limit=1000):self.state=dict(x=0.,t=0);self.limit=limit;self.steps=0
    def restore(self,state,expected_sha256):assert sha(state)==expected_sha256;self.state=copy.deepcopy(state)
    def capture(self):return copy.deepcopy(self.state)
    def step(self,action):
        self.steps+=1;self.state['x']+=float(action[0]);self.state['t']+=1
        return dict(applied_action=action.copy(),observation=np.array([self.state['x']]),reward=self.state['x'],
            terminated=self.state['t']>=self.limit,truncated=False)
def bank():
    sent=np.array([[0.],[1.]]+[[0.]]*8,np.float32)
    return dict(sent=sent,applied=sent.copy(),alias=[0,1]+[0]*8)
class RolloutTest(unittest.TestCase):
    def run_panel(self,sim,policy):
        self.records={};self.counts=dict(outcomes=0,repeat_checks=0)
        def step(scope,owner,action):self.counts[scope]+=1;return sim.step(action)
        def save(name,value):assert name not in self.records;self.records[name]=copy.deepcopy(value)
        return panel(sim,policy,sim.capture(),bank(),np.zeros((250,2),np.uint32),'state/host',step,save)
    def test_horizon_includes_first_step_and_uses_named_continuation(self):
        sim=Sim();policy=Policy(0.);r=self.run_panel(sim,policy)
        self.assertEqual(r['slots'][0]['raw_return'],0.);self.assertEqual(r['slots'][1]['raw_return'],250.)
        self.assertEqual(self.counts,dict(outcomes=500,repeat_checks=2));self.assertEqual(policy.calls,498)
        self.assertEqual(sim.steps,502);self.assertTrue(r['slots'][1]['horizon_exhausted'])
        self.assertEqual(r['slots'][9],dict(slot=9,alias=0,source_slot=0))
    def test_early_end_preserved(self):
        sim=Sim(3);r=self.run_panel(sim,Policy(-1.))
        self.assertEqual(r['slots'][0]['raw_return'],-3.);self.assertEqual(r['slots'][1]['raw_return'],0.)
        self.assertEqual(self.counts,dict(outcomes=6,repeat_checks=2));self.assertEqual(r['slots'][0]['length'],3)
    def test_terminal_first_step_has_no_continuation_query(self):
        policy=Policy(1.);r=self.run_panel(Sim(1),policy)
        self.assertEqual(policy.calls,0);self.assertEqual(self.counts,dict(outcomes=2,repeat_checks=2))
        self.assertEqual(r['slots'][1]['raw_return'],1.)
    def test_corrupt_keys_fail_before_any_step(self):
        sim=Sim()
        with self.assertRaises(ValueError):panel(sim,Policy(0.),sim.capture(),bank(),np.zeros((249,2),np.uint32),'x',None,None)
        self.assertEqual(sim.steps,0)
    def test_changed_repeat_stops_without_continuation(self):
        sim=Sim();old=sim.step
        def wrong(action):
            r=old(action)
            if sim.steps==2:r['reward']+=1
            return r
        sim.step=wrong;policy=Policy(0.)
        with self.assertRaises(ValueError):self.run_panel(sim,policy)
        self.assertEqual(sim.steps,2);self.assertEqual(policy.calls,0)
        self.assertIn('state/host/slot0/repeat',self.records)
    def test_failed_step_is_not_retried(self):
        sim=Sim()
        def fail(action):sim.steps+=1;raise RuntimeError('physical failure')
        sim.step=fail
        with self.assertRaises(RuntimeError):self.run_panel(sim,Policy(0.))
        self.assertEqual(sim.steps,1);self.assertIn('state/host/slot0/started',self.records)
    def test_bad_alias_refused(self):
        sim=Sim();a=bank();a['alias'][1]=0
        with self.assertRaises(ValueError):panel(sim,Policy(0.),sim.capture(),a,np.zeros((250,2),np.uint32),'x',None,None)
        self.assertEqual(sim.steps,0)
    def test_missing_execution_acceptance_never_starts(self):
        from experiments.ood.outcome_stage import execute
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            with self.assertRaises(FileNotFoundError):execute(p,p)
            self.assertFalse((p/'started.json').exists())
    def test_failed_candidate_process_never_accepted(self):
        from experiments.ood.outcome_stage import accepted_candidates
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);ex=p/'actual_exit.json'
            ex.write_text(json.dumps(dict(actual_returncode=1,timeout=False,interruption_signal=None)))
            (p/'dispatch.json').write_text(json.dumps(dict(command=['python'])))
            with self.assertRaises(ValueError):accepted_candidates(p,p,ex)

if __name__=='__main__':unittest.main()
