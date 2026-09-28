"""Execution recovery tests: exact toy outcomes, no physical simulator calls."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiments.ood.test_rollouts import Sim, Policy, bank
from experiments.ood.rollouts import panel
from experiments.ood.restart import finish_panel, replay_scope, check_saved_row, execute, check_original
from experiments.ood.collect import sha


class RestartTest(unittest.TestCase):
    def run_driver(self, driver, reused=None):
        sim, policy = Sim(), Policy(0.)
        state = sim.capture(); records = {}; calls = []
        def step(scope, owner, action):
            calls.append((scope, owner)); return sim.step(action)
        def save(name, value):
            self.assertNotIn(name, records); records[name] = copy.deepcopy(value)
        args = (sim, policy, state, bank(), np.zeros((250,2),np.uint32), 'host/state000', step, save)
        result = driver(*args, reused=reused or {}) if driver is finish_panel else driver(*args)
        return result, records, calls

    def test_new_panel_is_exact_original(self):
        a, ar, ac = self.run_driver(panel)
        b, br, bc = self.run_driver(finish_panel)
        self.assertEqual(sha(a), sha(b)); self.assertEqual(sha(ar), sha(br)); self.assertEqual(ac, bc)

    def test_completed_action_never_stepped_or_resaved(self):
        a, _, _ = self.run_driver(panel)
        b, records, calls = self.run_driver(finish_panel, {0:a['slots'][0]})
        self.assertEqual(sha(a),sha(b)); self.assertEqual(len(calls),251)
        self.assertTrue(all('/slot1/' in owner for _,owner in calls))
        self.assertTrue(all('/slot0/' not in name for name in records))

    def test_all_complete_panel_needs_no_physics(self):
        a,_,_=self.run_driver(panel)
        b,records,calls=self.run_driver(finish_panel,{i:a['slots'][i] for i in (0,1)})
        self.assertEqual(sha(a),sha(b)); self.assertEqual(calls,[]); self.assertEqual(records,{})

    def test_changed_return_refused(self):
        a,_,_=self.run_driver(panel); row=copy.deepcopy(a['slots'][0]);row['raw_return']=1.
        with self.assertRaises(ValueError): self.run_driver(finish_panel,{0:row})

    def test_invalid_counter_or_keys_refused(self):
        a,_,_=self.run_driver(panel);row=copy.deepcopy(a['slots'][0]);row['length']=249
        with self.assertRaises(ValueError):check_saved_row(row,0,np.zeros((250,2),np.uint32))
        row=copy.deepcopy(a['slots'][0]);row['continuation_keys_sha256']='wrong'
        with self.assertRaises(ValueError):check_saved_row(row,0,np.zeros((250,2),np.uint32))

    def test_existing_prefix_is_engineering_only(self):
        self.assertEqual(replay_scope('outcomes','host/state143/slot8/235','host/state143/slot8',236,1),'engineering')
        self.assertEqual(replay_scope('outcomes','host/state143/slot8/236','host/state143/slot8',236,1),'outcomes')
        self.assertEqual(replay_scope('repeat_checks','host/state143/slot8/repeat','host/state143/slot8',236,1),'engineering')
        self.assertEqual(replay_scope('outcomes','host/state143/slot9/0','host/state143/slot8',236,1),'outcomes')

    def test_original_limits_remain_exact(self):
        self.assertEqual(359736 + (5120-1438)*250 - 236,1280000)
        self.assertEqual(1439 + (5120-1438) - 1,5120)
        self.assertEqual((5120-1438)*251+1,924183)

    def test_invalid_replay_scope_refused(self):
        with self.assertRaises(ValueError):replay_scope('invented','x','x',236,1)

    def test_no_acceptance_refuses_execution_before_any_started_file(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            with self.assertRaises(FileNotFoundError):execute(path,path)
            self.assertFalse((path/'started.json').exists())

    def test_modified_original_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);(path/'saved.json').write_text('changed')
            plan=dict(old_attempt=d,original_files_sha256={'saved.json':'incorrect'})
            with self.assertRaises(ValueError):check_original(plan)

    def test_repeated_first_transition_failure_stops_new_work(self):
        sim=Sim();policy=Policy(0.);calls=[]
        def step(scope,owner,action):
            calls.append(owner);r=sim.step(action)
            if len(calls)==2:r['reward']+=1
            return r
        with self.assertRaises(ValueError):finish_panel(sim,policy,sim.capture(),bank(),np.zeros((250,2),np.uint32),
            'host/state000',step,lambda *_:None,reused={})
        self.assertEqual(len(calls),2);self.assertEqual(policy.calls,0)

if __name__=='__main__':unittest.main()
