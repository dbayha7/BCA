"""Deterministic mechanical checks. No checkpoint loading or learner training."""
import unittest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from objective import (actor_objective, candidate_scores, freeze_parameters,
                       raw_q_scale, validate_inputs)

def scalar_loss(theta, mode="live", strength=1.0, fixed_scale=None):
    action = jnp.asarray([[theta]])
    q = jnp.asarray([2.0 + theta])
    width = jnp.asarray([2.0 * (1.0 + theta)])
    return actor_objective(action, jnp.zeros_like(action), q, width,
        mode=mode, strength=strength, constant=7.0,
        fixed_host_scale=fixed_scale).loss

class ObjectiveTests(unittest.TestCase):
    def test_zero_penalty_forward_and_gradient(self):
        for theta in (-.3, .4, .8):
            host = scalar_loss(theta, "none")
            zero = scalar_loss(theta, "live", 0.0)
            np.testing.assert_array_equal(host, zero)
            np.testing.assert_array_equal(jax.grad(lambda t: scalar_loss(t,"none"))(theta),
                jax.grad(lambda t: scalar_loss(t,"live",0.0))(theta))

    def test_live_and_detached_analytic_gradient(self):
        theta = .4
        c = 2.5 / 2.4
        live = float(jax.grad(scalar_loss)(theta))
        host = float(jax.grad(lambda t: scalar_loss(t,"none"))(theta))
        detached = float(jax.grad(lambda t: scalar_loss(t,"detached"))(theta))
        self.assertAlmostEqual(live, c + .8, places=12)
        self.assertAlmostEqual(host, -c + .8, places=12)
        self.assertEqual(host, detached)
        self.assertAlmostEqual(live-host, 2*c, places=12)

    def test_constant_additive_placebo(self):
        theta = .4
        self.assertNotEqual(float(scalar_loss(theta,"constant")),
                            float(scalar_loss(theta,"none")))
        np.testing.assert_array_equal(
            jax.grad(lambda t: scalar_loss(t,"constant"))(theta),
            jax.grad(lambda t: scalar_loss(t,"none"))(theta))

    def test_finite_difference_with_frozen_scale(self):
        theta, h = .4, 1e-5
        c = float(raw_q_scale(jnp.array([2.4])))
        f = lambda t: scalar_loss(t, fixed_scale=c)
        finite = float((f(theta+h)-f(theta-h))/(2*h))
        analytic = float(jax.grad(f)(theta))
        self.assertAlmostEqual(finite, analytic, places=9)
        # A naive numerical recomputation of a stopped scale is not this derivative.
        wrong_fd = float((scalar_loss(theta+h)-scalar_loss(theta-h))/(2*h))
        self.assertGreater(abs(wrong_fd-analytic), .1)

    def test_frozen_model_parameters_keep_action_gradient(self):
        def loss(theta, critic, calibrator):
            cp = freeze_parameters(critic)
            up = freeze_parameters(calibrator)
            a = jnp.array([[theta]])
            q = cp["intercept"] + cp["slope"] * a[:,0]
            u = up["unit"] * (1 + up["slope"] * a[:,0])
            return actor_objective(a, jnp.zeros_like(a), q, u).loss
        cp = dict(intercept=jnp.array(2.), slope=jnp.array(1.))
        up = dict(unit=jnp.array(2.), slope=jnp.array(1.))
        ga, gc, gu = jax.grad(loss, argnums=(0,1,2))(.4,cp,up)
        self.assertAlmostEqual(float(ga), 2.5/2.4+.8, places=12)
        for tree in (gc,gu):
            for leaf in jax.tree_util.tree_leaves(tree):
                np.testing.assert_array_equal(leaf,0.)

    def test_rank_harmful_action_and_beneficial_action(self):
        q = jnp.array([3.,2.])
        harmful_score,_ = candidate_scores(q,jnp.array([2.,.1]))
        beneficial_score,_ = candidate_scores(q,jnp.array([.1,.2]))
        self.assertEqual(int(jnp.argmax(q)),0)
        self.assertEqual(int(jnp.argmax(harmful_score)),1)
        self.assertEqual(int(jnp.argmax(beneficial_score)),0)
        # The same width can be misleading; do not suppress the contradictory case.
        true_beneficial_values = np.array([3.,2.])
        regret = true_beneficial_values.max()-true_beneficial_values[int(jnp.argmax(harmful_score))]
        self.assertEqual(regret,1.)

    def test_detached_width_still_changes_discrete_ranking(self):
        q,u = jnp.array([3.,2.]),jnp.array([2.,.1])
        live,_ = candidate_scores(q,u,mode="live")
        detached,_ = candidate_scores(q,u,mode="detached")
        np.testing.assert_array_equal(live,detached)
        # Gradient no-op is specific to this continuous actor update, not argmax selection.

    def test_batch_reduction_and_shapes(self):
        a=jnp.array([[.2,.4],[-.3,.1]])
        b=jnp.zeros_like(a)
        q=jnp.array([2.,3.]); u=jnp.array([.4,.6])
        validate_inputs(a,b,q,u)
        terms=actor_objective(a,b,q,u)
        expected=-(2.5/2.5)*np.mean(np.array(q-u))+np.mean(np.array(a)**2)
        self.assertAlmostEqual(float(terms.loss),float(expected),places=12)
        with self.assertRaises(ValueError):
            actor_objective(a,b,q[:,None],u)
        with self.assertRaises(ValueError):
            validate_inputs(a,b,q,u[:,None])

    def test_invalid_width_refuses_before_arithmetic(self):
        for width in (np.inf,np.nan,-1.):
            with self.assertRaises(ValueError):
                validate_inputs(np.array([[.4]]),np.array([[0.]]),
                                np.array([2.4]),np.array([width]))

    def test_zero_q_normalizer_refuses(self):
        with self.assertRaises(ValueError):
            validate_inputs(np.array([[.4]]),np.array([[0.]]),
                            np.array([0.]),np.array([1.]))

    def test_float32_gradient(self):
        theta=jnp.float32(.4)
        grad=float(jax.grad(scalar_loss)(theta))
        self.assertAlmostEqual(grad,2.5/2.4+.8,places=6)

    def test_zero_strength_bypasses_unusable_width(self):
        a=jnp.array([[.4]]); q=jnp.array([2.4]); u=jnp.array([np.inf])
        validate_inputs(a,jnp.zeros_like(a),q,u,strength=0.)
        validate_inputs(a,jnp.zeros_like(a),q,u,mode="none")
        terms=actor_objective(a,jnp.zeros_like(a),q,u,strength=0.)
        self.assertTrue(np.isfinite(float(terms.loss)))
        self.assertEqual(float(terms.penalty_mean),0.)

    def test_derived_overflow_refuses(self):
        for q in (np.array([3e38,3e38],dtype=np.float32),
                  np.array([1e-45,1e-45],dtype=np.float32)):
            with self.assertRaises(ValueError):
                validate_inputs(np.zeros((2,1),np.float32),np.zeros((2,1),np.float32),
                                q,np.ones(2,np.float32))

    def test_mixed_sign_normalizer(self):
        def loss(t,mode):
            q=jnp.array([-1.+t,3.+2*t])
            u=jnp.array([1.+2*t,2.-t])
            a=jnp.zeros((2,1))
            return actor_objective(a,a,q,u,mode=mode).loss
        self.assertAlmostEqual(float(raw_q_scale(jnp.array([-1.,3.]))),1.25)
        self.assertAlmostEqual(float(jax.grad(lambda t:loss(t,"none"))(0.)),-1.875)
        self.assertAlmostEqual(float(jax.grad(lambda t:loss(t,"live"))(0.)),-1.25)

    def test_same_state_scalar_sgd_identity(self):
        theta,lr=.4,.001
        host=theta-lr*jax.grad(lambda t:scalar_loss(t,"none"))(theta)
        detached=theta-lr*jax.grad(lambda t:scalar_loss(t,"detached"))(theta)
        zero=theta-lr*jax.grad(lambda t:scalar_loss(t,"live",0.))(theta)
        np.testing.assert_array_equal(host,detached)
        np.testing.assert_array_equal(host,zero)

    def test_width_floor_derivatives(self):
        width=lambda x:2*jnp.maximum(x,1e-6)
        self.assertEqual(float(jax.grad(width)(-1.)),0.)
        self.assertEqual(float(jax.grad(width)(1.)),2.)
        # At the exact tie, current JAX maximum splits the tangent equally.
        self.assertEqual(float(jax.grad(width)(1e-6)),1.)

    def test_same_score_can_help_or_harm(self):
        q=jnp.array([3.,2.]); u=jnp.array([3.,1.])
        score,_=candidate_scores(q,u)
        chosen=int(jnp.argmax(score)); host=int(jnp.argmax(q))
        truthful=np.array([0.,1.]); misleading=np.array([3.,2.])
        self.assertEqual(truthful[chosen]-truthful[host],1.)
        self.assertEqual(misleading[chosen]-misleading[host],-1.)

    def test_residual_not_q_error_counterexample(self):
        # Two time steps; reward is zero at both, true finite-horizon Q is zero.
        # Approximate next Q=5 makes the current one-step residual zero at Q=5.
        predicted_current, predicted_next, reward = 5.,5.,0.
        residual=abs(reward+predicted_next-predicted_current)
        true_q_error=abs(predicted_current-0.)
        self.assertEqual(residual,0.)
        self.assertEqual(true_q_error,5.)

if __name__ == "__main__":
    unittest.main(verbosity=2)
