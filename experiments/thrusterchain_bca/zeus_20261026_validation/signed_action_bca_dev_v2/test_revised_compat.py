import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'action_sensitive_bca_dev_v1'))
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None
import revised_compat as rc


def fixture():
    rng = np.random.default_rng(871)
    episodes = 12
    t = np.tile(np.arange(100, dtype=np.int64), episodes)
    ep = np.repeat(np.arange(episodes, dtype=np.int64), 100)
    obs = rng.normal(size=(len(ep), 14)).astype(np.float32)
    obs[:, -1] = 1.0 - t / 100
    actions = rng.uniform(-.8, .8, (len(ep), 4)).astype(np.float32)
    rows = 10000 + ep * 100 + t
    probes = np.arange(episodes) * 100 + 25
    return obs, actions, ep, t, rows, probes


class RevisedCompatibilityTests(unittest.TestCase):
    def fit(self):
        x, a, e, t, rows, ix = fixture()
        ref, cross, receipt = rc.fit_crossvalidated(x, a, e, t, ix, row_ids=rows,
                                                     folds=3, k=4)
        return x, a, e, t, rows, ix, ref, cross, receipt

    def test_episode_fold_exclusion_and_train_only_statistics(self):
        x, a, e, t, rows, ix, ref, cross, r = self.fit()
        self.assertEqual(len(r['mean_negative_log_likelihood']), 4)
        self.assertEqual(r['selected_multiplier'], r['grid_multipliers'][
            int(np.argmin(r['mean_negative_log_likelihood']))])
        for i in range(len(ix)):
            self.assertNotIn(e[ix[i]], cross['episode_ids'][i])
        np.testing.assert_allclose(ref.base.state_mean, x[:, :13].mean(0, dtype=np.float64))
        np.testing.assert_allclose(ref.base.action_bandwidth,
                                   np.maximum(np.maximum(a.astype(np.float64).std(0), .05)
                                              * ref.multiplier, .05))
        self.assertEqual(ref.base.fingerprint_sha256, r['reference_sha256'])
        self.assertFalse(ref.base.action_bandwidth.flags.writeable)

    def test_normalized_density_selection_and_bad_probes(self):
        x, a, e, t, rows, ix, ref, cross, r = self.fit()
        ix2 = np.r_[ix, ix[0] + 1]
        with self.assertRaisesRegex(ValueError, 'one query per episode'):
            rc.fit_crossvalidated(x, a, e, t, ix2, row_ids=rows, folds=3, k=4)
        widths = np.maximum(cross['bandwidths'] / ref.multiplier * .5, .05)
        nll = rc._cost_numpy(a[ix], cross['actions'], widths) + np.log(widths).sum(1) + 2*np.log(2*np.pi)
        self.assertTrue(np.isfinite(nll).all())

    def test_state_distance_gate_is_explicit(self):
        x, a, e, t, rows, ix, ref, cross, r = self.fit()
        q = x[ix[:1]].copy()
        q[0, :13] += 100.0
        found = rc.query_neighbors(ref, q, t[ix[:1]])
        self.assertFalse(found['reliable'][0])
        self.assertGreater(found['distances'][0, -1], ref.radius_p95)
        self.assertEqual(r['state_radius_p95'], ref.radius_p95)
        thresholds = np.array([r['outer_fold_inner_radius_p95'][str(int(f))] for f in cross['query_folds']])
        self.assertTrue(np.array_equal(cross['reliable'], cross['distances'][:, -1] <= thresholds))

    def test_matching_bandwidth_does_not_use_outer_fold_actions(self):
        x, a, e, t, rows, ix, ref, cross, receipt = self.fit()
        fold = int(cross['query_folds'][0])
        mapping = rc.original.episode_folds(e, folds=3, seed=20261029)
        selected_episodes = {episode for episode, value in mapping.items() if value == fold}
        changed = a.copy()
        changed[np.isin(e, list(selected_episodes))] *= -.3
        _, cross2, receipt2 = rc.fit_crossvalidated(x, changed, e, t, ix, row_ids=rows,
                                                     folds=3, k=4)
        positions = cross['query_folds'] == fold
        np.testing.assert_array_equal(cross['row_ids'][positions], cross2['row_ids'][positions])
        np.testing.assert_allclose(cross['bandwidths'][positions], cross2['bandwidths'][positions])
        self.assertEqual(receipt['outer_fold_selected_multipliers'][str(fold)],
                         receipt2['outer_fold_selected_multipliers'][str(fold)])

    @unittest.skipIf(jax is None, 'JAX is not installed on this host')
    def test_live_action_gradient_matches_finite_difference(self):
        x, a, e, t, rows, ix, ref, cross, r = self.fit()
        neighbors = jnp.asarray(cross['actions'][:1])
        width = jnp.asarray(cross['bandwidths'][:1], dtype=jnp.float32)
        act = jnp.asarray(a[ix[:1]])
        f = lambda v: rc.compatibility_cost_jax(v, neighbors, width).sum()
        gradient = np.asarray(jax.grad(f)(act))
        for d in range(4):
            delta = np.zeros((1, 4), np.float32); delta[0, d] = 1e-3
            fd = float((f(act + delta) - f(act - delta)) / .002)
            self.assertAlmostEqual(float(gradient[0, d]), fd, delta=.01)


if __name__ == '__main__':
    unittest.main()
