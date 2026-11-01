import unittest

import numpy as np

import jax
import jax.numpy as jnp

import compatibility as c


def fixture(episodes=8):
    """Finite-horizon, distinct episode/time identities, with 13D raw state."""
    rng = np.random.Generator(np.random.PCG64(402))
    n = episodes * 100
    t = np.tile(np.arange(100, dtype=np.int64), episodes)
    ep = np.repeat(np.arange(episodes, dtype=np.int64), 100)
    obs = rng.normal(size=(n, 14)).astype(np.float32)
    obs[:, -1] = (1 - t / 100).astype(np.float32)
    acts = rng.uniform(-.9, .9, (n, 4)).astype(np.float32)
    rows = ep * 100 + t + 10000
    return obs, acts, ep, t, rows


class ReferenceTests(unittest.TestCase):
    def test_fit_training_only_and_time_alignment(self):
        obs, act, ep, t, rows = fixture()
        ref = c.fit_reference(obs, act, ep, t, row_ids=rows, k=3)
        np.testing.assert_allclose(ref.state_mean, obs[:, :13].mean(axis=0, dtype=np.float64))
        np.testing.assert_allclose(ref.action_bandwidth, np.maximum(act.astype(np.float64).std(axis=0), .05))
        self.assertFalse(ref.actions.flags.writeable)
        self.assertFalse(ref.state_mean.flags.writeable)
        with self.assertRaises(ValueError):
            c.fit_reference(obs, act, ep, t + 1, row_ids=rows, k=3)
        bad = t.copy(); bad[0] = 1
        with self.assertRaises(ValueError):
            c.fit_reference(obs, act, ep, bad, row_ids=rows, k=3)

    def test_same_episode_is_excluded_and_knn_order_is_stable(self):
        obs, act, ep, t, rows = fixture()
        # All query states are equidistant; source row IDs break exact ties.
        obs[:, :13] = 0
        ref = c.fit_reference(obs, act, ep, t, row_ids=rows, k=3)
        q = c.query_neighbors(ref, obs[[0, 121]], t[[0, 121]], exclude_episode_ids=ep[[0, 121]])
        self.assertTrue(np.all(q['episode_ids'][0] != ep[0]))
        self.assertTrue(np.all(q['episode_ids'][1] != ep[121]))
        for i, query in enumerate([0, 121]):
            pool = rows[(t // 10 == t[query] // 10) & (ep != ep[query])]
            np.testing.assert_array_equal(q['row_ids'][i], np.sort(pool)[:3])
        order = np.random.default_rng(3).permutation(len(obs))
        ref2 = c.fit_reference(obs[order], act[order], ep[order], t[order], row_ids=rows[order], k=3)
        q2 = c.query_neighbors(ref2, obs[[0, 121]], t[[0, 121]], exclude_episode_ids=ep[[0, 121]])
        np.testing.assert_array_equal(q['row_ids'], q2['row_ids'])

    def test_crossfit_excludes_whole_fold_from_stats_and_neighbors(self):
        obs, act, ep, t, rows = fixture()
        fold_map = c.episode_folds(ep, folds=4)
        qidx = np.flatnonzero(np.array([fold_map[int(x)] == 0 for x in ep]))[:3]
        output = c.crossfit_query(obs, act, ep, t, qidx, row_ids=rows, folds=4, k=3)
        self.assertTrue(all(fold_map[int(e)] != 0 for e in output['episode_ids'].flat))
        # Altering *all* held-out fold rows cannot change fitted bandwidth or digest.
        alter = np.array([fold_map[int(e)] == 0 for e in ep])
        obs2 = obs.copy(); obs2[alter, :13] += 20
        act2 = act.copy(); act2[alter] *= .1
        output2 = c.crossfit_query(obs2, act2, ep, t, qidx, row_ids=rows, folds=4, k=3)
        self.assertEqual(output['reference_sha256_by_fold'], output2['reference_sha256_by_fold'])
        np.testing.assert_array_equal(output['bandwidths'], output2['bandwidths'])

    def test_insufficient_neighbors_refuses_and_no_episode_leak(self):
        obs, act, ep, t, rows = fixture(episodes=2)
        ref = c.fit_reference(obs, act, ep, t, row_ids=rows, k=11)
        with self.assertRaisesRegex(ValueError, 'fewer than K'):
            c.query_neighbors(ref, obs[:1], t[:1], exclude_episode_ids=ep[:1])


class JaxTests(unittest.TestCase):
    def test_cost_value_finite_difference_and_frozen_reference_gradients(self):
        neighbors = jnp.array([[[0.,0.,0.,0.], [.5,0.,0.,0.], [-.5,0.,0.,0.]]], jnp.float32)
        bw = jnp.array([.4,.5,.6,.7], jnp.float32)
        actions = jnp.array([[.3,.1,-.2,.2]], jnp.float32)
        f = lambda a, n, h: c.compatibility_cost_jax(a, n, h).sum()
        val = float(jax.jit(f)(actions, neighbors, bw))
        self.assertTrue(np.isfinite(val) and val >= 0)
        grad = np.asarray(jax.grad(f, argnums=0)(actions, neighbors, bw))
        eps=1e-3
        delta=np.zeros((1,4),np.float32); delta[0,0]=eps
        finite=(float(f(actions+delta,neighbors,bw))-float(f(actions-delta,neighbors,bw)))/(2*eps)
        self.assertAlmostEqual(float(grad[0,0]), finite, delta=2e-4)
        self.assertGreater(np.linalg.norm(grad), 0)
        np.testing.assert_array_equal(np.asarray(jax.grad(f,argnums=1)(actions,neighbors,bw)), 0)
        np.testing.assert_array_equal(np.asarray(jax.grad(f,argnums=2)(actions,neighbors,bw)), 0)
        zero=jnp.zeros((1,3,4),jnp.float32)
        self.assertAlmostEqual(float(f(jnp.zeros((1,4)),zero,bw)),0.,places=6)
        self.assertAlmostEqual(float(f(jnp.array([[.4,0,0,0]]),zero,bw)),.5,places=5)

    def test_runtime_lookup_jit_matches_host_cost_and_action_gradient(self):
        obs, act, ep, t, rows = fixture()
        ref=c.fit_reference(obs,act,ep,t,row_ids=rows,k=3)
        query=np.array([17,145,333])
        found=c.query_neighbors(ref,obs[query],t[query],exclude_episode_ids=ep[query])
        action=jnp.asarray(act[query]); jref=c.jax_reference(ref)
        host=np.asarray(c.compatibility_cost_jax(action,jnp.asarray(found['actions']),jnp.asarray(ref.action_bandwidth)))
        def live(a):
            return c.compatibility_from_observations_jax(jref,jnp.asarray(obs[query]),a,k=3,
                                                         exclude_episode_ids=jnp.asarray(ep[query],jnp.int32))
        actual=np.asarray(jax.jit(live)(action))
        np.testing.assert_allclose(actual,host,rtol=2e-5,atol=2e-5)
        direct=jax.grad(lambda a: live(a).sum())(action)
        expected=jax.grad(lambda a: c.compatibility_cost_jax(a,jnp.asarray(found['actions']),
                                                           jnp.asarray(ref.action_bandwidth)).sum())(action)
        np.testing.assert_allclose(direct,expected,rtol=2e-5,atol=2e-5)
        # Actor-facing lookup must not accidentally update the state bank,
        # state-normalization tensors, or the input observations.
        def frozen_sum(states, neighbor_actions, mean, scale, widths, query_obs):
            changed = dict(jref, states=states, actions=neighbor_actions, mean=mean,
                           scale=scale, bandwidth=widths)
            return c.compatibility_from_observations_jax(changed, query_obs, action, k=3,
                          exclude_episode_ids=jnp.asarray(ep[query],jnp.int32)).sum()
        gradients = jax.grad(frozen_sum, argnums=(0,1,2,3,4,5))(
            jref['states'], jref['actions'], jref['mean'], jref['scale'], jref['bandwidth'],
            jnp.asarray(obs[query]))
        for gradient in gradients:
            np.testing.assert_array_equal(np.asarray(gradient), 0)

    def test_invalid_time_or_not_enough_neighbors_is_nan(self):
        obs, act, ep, t, rows=fixture(episodes=2)
        ref=c.jax_reference(c.fit_reference(obs,act,ep,t,row_ids=rows,k=11))
        f=jax.jit(lambda x,a,e: c.compatibility_from_observations_jax(ref,x,a,k=11,exclude_episode_ids=e))
        v=np.asarray(f(jnp.asarray(obs[:1]),jnp.asarray(act[:1]),jnp.asarray(ep[:1],jnp.int32)))
        self.assertTrue(np.isnan(v[0]))
        broken=obs[:1].copy();broken[0,-1]=.34567
        v=np.asarray(f(jnp.asarray(broken),jnp.asarray(act[:1]),jnp.asarray([-1],jnp.int32)))
        self.assertTrue(np.isnan(v[0]))


if __name__ == '__main__':
    unittest.main()
