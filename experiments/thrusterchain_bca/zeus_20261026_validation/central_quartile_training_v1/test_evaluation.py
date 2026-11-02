"""Synthetic fixtures only: no real simulator/model/data execution."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import evaluation as e


def load_reference():
    root = Path(__file__).resolve().parent.parent / 'conditional_ood_gate_central_quartile_v1'
    gs = importlib.util.spec_from_file_location('_eval_geometry_fixture', root / 'geometry.py')
    g = importlib.util.module_from_spec(gs); gs.loader.exec_module(g)
    rs = importlib.util.spec_from_file_location('_eval_reference_fixture', root / 'reference.py')
    r = importlib.util.module_from_spec(rs)
    with patch.dict(sys.modules, {'geometry': g}):
        rs.loader.exec_module(r)
    return r


R = load_reference()


class FakeEnv:
    def __init__(self, **kwargs):
        self.state = np.zeros(13, np.float64); self.t = 0

    def reset(self, seed=None, options=None):
        self.state = np.zeros(13, np.float64)
        if options and 'start' in options:
            self.state[10] = options['start']['current'][0]
        self.t = 0
        return R.observation(self.state, np.asarray(0)), self.info()

    def info(self):
        return {'remaining_steps': 100-self.t,
                'bootstrap_mask': 0. if self.t == 100 else 1.,
                'continuation_discount': 0. if self.t == 100 else .98}

    def step(self, action):
        self.t += 1
        return R.observation(self.state, np.asarray(self.t)), float(action[0]), self.t == 100, False, self.info()

    def close(self):
        pass


class FakeCore:
    ThrusterChainEnv = FakeEnv
    Physics = staticmethod(lambda **kwargs: None)

    @staticmethod
    def dynamics(states, actions, physics):
        return np.asarray(states).copy(), np.asarray(actions[..., 0], np.float64)


class ZeroPolicy:
    def __init__(self):
        self.q_calls = 0

    def assert_unchanged(self):
        pass

    def act(self, observations):
        return np.zeros((len(observations), 4), np.float32)

    def q(self, observations, actions):
        self.q_calls += 1
        return np.zeros((len(observations), 2), np.float32)


def ledger():
    out = R.new_ledger(2000000, 2000000)
    out.update(max_resets=129, max_critic_rows=24192)
    return out


def bank():
    states = np.zeros((32, 13), np.float64); states[:, 10] = .5
    times = np.tile(np.asarray([0, 25, 50, 75], np.int64), 8)
    return {'context': 'ordinary', 'bank_id': 'synthetic', 'states': states, 'times': times,
            'observations': R.observation(states, times), 'free_starts': states.copy()}


class EvaluationTests(unittest.TestCase):
    def test_exact_budget(self):
        b = e.transition_budget()
        self.assertEqual((b['untrained_snapshots'], b['trained_snapshots'], b['total_snapshots']), (9, 63, 72))
        self.assertEqual((b['free_episodes'], b['q_values']), (4608, 24192))
        self.assertEqual((b['prefix_steps'], b['q_continuation_steps'], b['free_steps']), (2400, 1512000, 460800))
        self.assertEqual(b['total_steps'], 1975304)
        self.assertEqual(b['actor_rows'], 1952640)
        self.assertEqual(b['actor_rows']+b['distance_anchor_actor_rows'],b['actor_rows_total_including_distance'])
        self.assertEqual((b['visited_rows'],b['critic_rows'],b['width_rows_max']),(16128,40320,40320))

    def test_fixed_candidates_do_not_inherit_other_actor_commands(self):
        own = np.ones((32, 4), np.float32) * .123
        a = e.fixed_candidate_bank(own)
        np.testing.assert_array_equal(a[:, 0], own)
        np.testing.assert_array_equal(a[:, 1:, 1:], 0)
        np.testing.assert_array_equal(a[0, 1:, 0], [-.5, -.25, 0, .25, .5])
        self.assertTrue(np.all(np.abs(a[:, 2:5, 0]) <= .25))

    def test_signed_classes_and_magnitudes(self):
        m = e.error_metrics([-.2, -.1, 0, .1, .3], np.zeros(5))
        self.assertEqual((m['under'], m['near_zero'], m['over']), (1, 3, 1))
        self.assertAlmostEqual(m['mae'], .14)
        self.assertAlmostEqual(m['rmse'], np.sqrt(.15 / 5))
        self.assertAlmostEqual(m['mean_under_magnitude'], .2)
        with self.assertRaisesRegex(ValueError, 'finite'):
            e.error_metrics([np.inf], [0])

    def test_same_critic_changes_and_crossings(self):
        m = e.paired_error_change([1., 1., -.5, 0], [-1., 0., -1., .05], np.zeros(4))
        self.assertEqual((m['improved'], m['worsened'], m['near_tied']), (1, 1, 2))
        self.assertEqual(m['class_transitions']['over_to_under'], 1)
        self.assertEqual(m['class_transitions']['over_to_near_zero'], 1)
        self.assertEqual(sum(m['class_transitions'].values()), 4)

    def test_numerical_ties_and_decision_harm(self):
        scores = np.zeros((32, 6)); scores[:, 1] = .5e-6
        values = np.zeros_like(scores); values[:, 1] = 1.
        r = e.decision_metrics(scores, values)
        self.assertEqual(r['selected_slot'], [0] * 32)
        self.assertEqual(r['mean_regret'], 1.)
        scores[:, 1] = 1.
        values[:, 1] = -2.
        r = e.decision_metrics(scores, values)
        self.assertEqual(r['harmed_vs_actor'], 32)

    def test_own_actor_separate_from_fixed_primary(self):
        b = bank(); own = np.ones((32, 4), np.float32) * .2
        a = e.fixed_candidate_bank(own); raw = np.zeros((32, 6)); v = np.ones((32, 6))
        p = e.q_panels(b['observations'], a, raw, v, raw-.5, raw-.25)
        self.assertEqual(p['panels']['high_band_fixed_primary']['raw']['rows'], 96)
        self.assertEqual(p['panels']['high_band_fixed_primary']['state_count'], 32)
        self.assertEqual(p['panels']['own_actor_high_hole']['raw']['rows'], 32)
        self.assertEqual(p['panels']['low_band_fixed_control']['raw']['rows'], 0)
        self.assertIsNone(p['panels']['low_band_fixed_control']['raw']['mae'])
        self.assertEqual(p['panels']['high_band_fixed_primary']['twin_min']['mae'], 1.25)

    def test_seed_pairing_and_missingness(self):
        rows = []
        for seed in [1, 2, 3]:
            for condition in e.CONDITIONS:
                for arm in ['native', 'bca']:
                    rows.append({'seed': seed, 'condition': condition, 'arm': arm, 'context': 'ordinary',
                                 'checkpoint': 20000, 'bank_id': 'common',
                                 'mae': 5. if arm == 'native' else (3. if condition == 'hole' else 4.)})
        r = e.paired_seed_interaction(rows, seeds=[1, 2, 3], context='ordinary', checkpoint=20000, metric='mae')
        self.assertEqual(r['mean_interaction'], 1.)
        self.assertEqual(r['seed_count'], 3)
        self.assertTrue(r['all_three_strictly_positive'])
        for row in rows:
            if row['arm'] == 'bca' and row['condition'] == 'hole':
                row['mae'] = 3.5
        self.assertTrue(e.paired_seed_interaction(rows, seeds=[1, 2, 3], context='ordinary',
                        checkpoint=20000, metric='mae')['all_three_strictly_positive'])
        for row in rows:
            if row['arm'] == 'bca' and row['condition'] == 'hole':
                row['mae'] = 4.
        self.assertFalse(e.paired_seed_interaction(rows, seeds=[1, 2, 3], context='ordinary',
                         checkpoint=20000, metric='mae')['all_three_strictly_positive'])
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            e.paired_seed_interaction(rows[:-1], seeds=[1, 2, 3], context='ordinary', checkpoint=20000, metric='mae')
        rows[0]['bank_id'] = 'different'
        with self.assertRaisesRegex(ValueError, 'identical'):
            e.paired_seed_interaction(rows, seeds=[1, 2, 3], context='ordinary', checkpoint=20000, metric='mae')

    def test_real_reference_interface_toy_snapshot_exact_accounting(self):
        l = ledger(); p = ZeroPolicy()
        result = e.evaluate_snapshot(FakeCore, R, p, bank(), l)
        self.assertEqual(l['transition_steps_completed'], 15200)
        self.assertEqual(l['actor_rows_completed'], 15040)
        self.assertEqual(l['critic_rows_completed'], 320)
        self.assertEqual(result['free']['hole_steps'], 3200)
        self.assertEqual(result['free']['undiscounted_returns'], [0.] * 32)
        self.assertEqual(result['q']['panels']['all_fixed']['raw']['rmse'],
                         result['q']['panels']['all_fixed']['twin_min']['rmse'])
        np.testing.assert_allclose(np.asarray(result['q']['per_probe']['qpi'])[0], [0, -.5, -.25, 0, .25, .5])
        self.assertEqual(l['optimizer_updates'], 0)
        self.assertEqual(result['visited']['panels']['high_hole']['rows'],128)
        self.assertEqual(result['visited']['additional_simulator_transitions'],0)

    def test_signed_margins_allow_good_bad_and_tied_hole_actions(self):
        raw=np.zeros((3,6)); values=raw.copy()
        values[0,3]=2.; values[1,1]=3.; values[2,1]=values[2,3]=1.
        adjusted=raw.copy(); adjusted[:,3]=-1.
        m=e.signed_candidate_margins(raw,values,adjusted)
        self.assertEqual(m['center_minus_best_retained_qpi'],[2.,-3.,0.])
        self.assertEqual((m['best_fixed_action_strictly_in_band'],m['best_fixed_action_strictly_outside_band'],
                          m['best_fixed_action_tied_across_band']),(1,1,1))
        self.assertEqual(m['adjustment_choice_change']['near_tied_states'],3)

    def test_visited_suffix_is_time_relative_and_needs_no_new_transitions(self):
        b=bank(); times=np.asarray(e.VISITED_TIMES)
        obs=R.observation(np.repeat(b['states'][:,None,:],4,axis=1),np.tile(times,(32,1)))
        free={'cached_visited':{'times':list(e.VISITED_TIMES),'observations':obs.tolist(),
              'actions':np.zeros((32,4,4)).tolist(),'rewards':np.ones((32,100)).tolist()}}
        l=ledger();p=ZeroPolicy()
        panel=e.visited_policy_panel(p,free,l,R,lambda o,a,q:q-.2,None,None)
        expected=(1-.98**(100-times))/(1-.98)
        np.testing.assert_allclose(np.asarray(panel['per_row']['qpi_suffix']).reshape(32,4),np.tile(expected,(32,1)),atol=1e-12)
        self.assertEqual(l['transition_steps_attempted'],0)
        self.assertEqual(l['actor_rows_attempted'],0)
        self.assertEqual(l['critic_rows_completed'],128)
        self.assertEqual(panel['panels']['high_hole']['episode_families'],32)
        self.assertEqual(panel['per_row']['time'][:8],[0,25,50,75,0,25,50,75])

    def test_untrained_is_free_only(self):
        l = ledger(); p = ZeroPolicy()
        result = e.evaluate_snapshot(FakeCore, R, p, bank(), l, trained=False)
        self.assertIsNone(result['q']); self.assertEqual(p.q_calls, 0)
        self.assertEqual(l['transition_steps_completed'], 3200)

    def test_bank_prefix_time_and_reset_accounting(self):
        l = ledger()
        b = e.build_context_bank(FakeCore, R, context='ordinary', q_seeds=np.arange(32),
                                 free_seeds=np.arange(32, 64), prefix_seed=100,
                                 start_law={'current': (.5, .6), 'wave': (0, .2), 'kick': 1.}, ledger=l)
        self.assertEqual(l['transition_steps_completed'], 1200)
        self.assertEqual(l['resets_completed'], 64)
        np.testing.assert_array_equal(b['times'], np.tile([0, 25, 50, 75], 8))
        self.assertEqual(b['states'].dtype, np.float64)
        self.assertEqual(b['observations'].dtype, np.float32)
        self.assertEqual(len(b['bank_id']), 64)

    def test_cap_refuses_before_real_or_toy_dispatch(self):
        l = ledger(); l['max_transition_steps'] = 0
        with self.assertRaisesRegex(ValueError, 'cap'):
            e.evaluate_snapshot(FakeCore, R, ZeroPolicy(), bank(), l, trained=False)
        self.assertEqual(l['transition_steps_completed'], 0)


if __name__ == '__main__':
    unittest.main()
