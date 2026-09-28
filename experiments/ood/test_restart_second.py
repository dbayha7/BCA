"""Second-interruption metadata tests; no model or physical simulator calls."""
import copy
import unittest
import numpy as np
from experiments.ood.restart_second import merge_unique, recovery_counts, compare_replay, expected_layout
from experiments.ood.restart import replay_scope
from experiments.ood.collect import sha


class SecondRestartTests(unittest.TestCase):
    def test_provenance_union_does_not_rewrite_paths(self):
        a={'a':dict(path='/original/a',sha256='1')}; b={'b':dict(path='/recovery/b',sha256='2')}
        before=copy.deepcopy((a,b))
        self.assertEqual(merge_unique(a,b),{**a,**b});self.assertEqual((a,b),before)

    def test_duplicate_even_identical_is_refused(self):
        a={'a':dict(path='x',sha256='1')}
        with self.assertRaises(ValueError):merge_unique(a,a)

    def test_layout_preserves_first_recovery_mixed_panel(self):
        actions,panels=expected_layout()
        self.assertEqual(len(actions),4199);self.assertEqual(len(panels),419)
        self.assertIn('host/state143/slot0',actions)
        self.assertIn('host/state143/slot8',actions)
        self.assertEqual(actions[-1],'bca/state163/slot8')
        self.assertEqual(panels[-1],'bca/state162')

    def test_counts_keep_uncertain_reservation_and_old_caps(self):
        c=recovery_counts()
        self.assertEqual(c,dict(maximum_new_physical_transitions=231172,
            maximum_new_physics_steps=924688,maximum_recovery_engineering_transitions=105,
            new_scientific_outcomes=230147,new_scientific_repeats=920))
        self.assertEqual(1049853+c['new_scientific_outcomes'],1280000)
        self.assertEqual(4200+c['new_scientific_repeats'],5120)

    def test_replay_scope_includes_input_only_reserved_call(self):
        p='bca/state163/slot9'
        self.assertEqual(replay_scope('outcomes',p+'/102',p,103,1),'engineering')
        self.assertEqual(replay_scope('outcomes',p+'/103',p,103,1),'outcomes')

    def sample(self):
        inp=dict(proposed_action=np.array([.1],np.float32),state_json='saved state')
        rec=dict(reward=1.,observation=np.array([2.],np.float64))
        e=dict(input_sha256=sha(inp),record_sha256=sha(rec))
        return inp,rec,e

    def test_full_record_and_input_exact(self):
        inp,rec,e=self.sample()
        self.assertEqual(compare_replay(inp,rec,e),'full_record')

    def test_changed_input_state_or_dtype_refused(self):
        inp,rec,e=self.sample();inp['state_json']='other'
        with self.assertRaises(ValueError):compare_replay(inp,rec,e)
        inp,rec,e=self.sample();inp['proposed_action']=inp['proposed_action'].astype(np.float64)
        with self.assertRaises(ValueError):compare_replay(inp,rec,e)

    def test_changed_reward_refused(self):
        inp,rec,e=self.sample();rec['reward']=2.
        with self.assertRaises(ValueError):compare_replay(inp,rec,e)

    def test_input_only_is_explicitly_not_output_verification(self):
        inp,rec,e=self.sample();e['record_sha256']=None
        self.assertEqual(compare_replay(inp,rec,e),'input_only_unknown_prior_output')

    def test_missing_input_hash_refused(self):
        inp,rec,e=self.sample();e['input_sha256']=None
        with self.assertRaises(ValueError):compare_replay(inp,rec,e)


if __name__=='__main__':unittest.main()
