import unittest
import numpy as np
import jax
import jax.numpy as jnp
import score_controls as sc


class ControlsTest(unittest.TestCase):
    def setUp(self):
        self.obs = jnp.zeros((3, 14), jnp.float32)
        self.a = jnp.asarray([[.2, -.3, .5, .1], [-.4, .2, -.2, .6], [.3, .4, -.5, -.1]], jnp.float32)
        self.neighbors = jnp.zeros((3, 2, 4), jnp.float32)
        self.bandwidths = jnp.ones((3, 4), jnp.float32)
        self.guard = np.array([True, False, True])
        self.q = lambda o, a: jnp.sum(2 * a, axis=1)
        self.width = lambda o, a: (2 + .1*jnp.sum(a*a, axis=1), jnp.ones(3, bool), jnp.ones(3, bool))
        self.signed = lambda o, a: (1 + .05*jnp.sum((a-.3)**2, axis=1), None, None)
        self.cost = lambda a, n, h: jnp.mean(jnp.sum(((a[:, None, :]-n)/h[:, None, :])**2,axis=2),axis=1)

    def match(self, **kw):
        return sc.match_controls(self.obs,self.a,self.q,self.width,self.signed,self.cost,
                                 self.neighbors,self.bandwidths,self.guard,**kw)

    def test_two_scale_targets_are_matched_independently(self):
        m = self.match()
        signed = np.asarray(self.signed(self.obs,self.a)[0])
        width = np.asarray(self.width(self.obs,self.a)[0])
        self.assertAlmostEqual(m['lambda_mean']*float(np.mean(signed)),3*float(np.mean(width)),places=5)
        self.assertAlmostEqual(m['lambda_gradient']*m['signed_action_gradient_rms'],
                               3*m['width_action_gradient_rms'],places=5)
        self.assertAlmostEqual(m['eta']*m['guarded_cost_action_gradient_rms'],
                               .25*m['q_action_gradient_rms'],places=5)
        self.assertNotAlmostEqual(m['lambda_mean'],m['lambda_gradient'],places=2)
        self.assertEqual(m['guarded_probe_count'],2)

    def test_score_algebra_and_guard(self):
        m=self.match(); q=np.array([-2., -1., 2.]); w=np.array([1.,1.2,1.]); b=np.array([.4,.5,.6]); d=np.array([.3,.8,.2])
        out=sc.apply_scores(q,w,b,d,self.guard,m)
        np.testing.assert_allclose(out['signed_mean_matched'],q-m['lambda_mean']*b)
        np.testing.assert_allclose(out['signed_gradient_matched'],q-m['lambda_gradient']*b)
        np.testing.assert_allclose(out['sym_compat_guarded'],q-3*w-m['eta']*d*self.guard)
        np.testing.assert_allclose(out['sym_compat_unguarded'],q-3*w-m['eta']*d)
        self.assertEqual(out['sym_compat_guarded'][1],out['q1_aligned'][1])
        self.assertLess(out['sym_compat_unguarded'][1],out['q1_aligned'][1])
        np.testing.assert_allclose(out['sym_hybrid_strength_matched'],q-m['hybrid_matched_kappa']*w)

    def test_live_gradient_matches_finite_difference(self):
        m=self.match(); eta=m['eta']; idx=(0,1)
        def f(a):
            return jnp.sum(3*self.width(self.obs,a)[0]+eta*self.cost(a,self.neighbors,self.bandwidths)*jnp.asarray(self.guard))
        grad=np.asarray(jax.grad(f)(self.a)); eps=.001
        plus=np.asarray(self.a).copy(); minus=plus.copy(); plus[idx]+=eps; minus[idx]-=eps
        numeric=(float(f(jnp.asarray(plus)))-float(f(jnp.asarray(minus))))/(2*eps)
        self.assertAlmostEqual(float(grad[idx]),numeric,delta=.002)
        self.assertAlmostEqual(m['hybrid_matched_kappa']*m['width_action_gradient_rms'],
                               sc._rms(jax.grad(f)(self.a)),places=5)

    def test_zero_or_invalid_denominators_and_flags_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'signals'):
            sc.match_controls(self.obs,self.a,self.q,self.width,self.signed,self.cost,self.neighbors,self.bandwidths,np.zeros(3,bool))
        with self.assertRaisesRegex(ValueError,'guard'):
            sc.match_controls(self.obs,self.a,self.q,self.width,self.signed,self.cost,self.neighbors,self.bandwidths,np.ones(3))
        invalid=lambda o,a:(jnp.ones(3),jnp.ones(3,bool),jnp.zeros(3,bool))
        with self.assertRaisesRegex(ValueError,'unusable'):
            sc.match_controls(self.obs,self.a,self.q,invalid,self.signed,self.cost,self.neighbors,self.bandwidths,self.guard)
        with self.assertRaisesRegex(ValueError,'cap'):
            self.match(max_coefficient=.001)


if __name__=='__main__': unittest.main()
