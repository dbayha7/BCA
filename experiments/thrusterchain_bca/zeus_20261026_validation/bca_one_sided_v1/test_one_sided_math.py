"""Synthetic CPU fixtures only: no logged banks, learned checkpoints or rollouts."""

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import unittest
from datetime import datetime, timezone

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import jax
import jax.numpy as jnp
import numpy as np

import one_sided_math as m


HERE = Path(__file__).resolve().parent
WBCP = HERE.parents[1] / "titan_20261026/payload/unifloral_bca_pipeline_validation_v1/uncertainty_source/calibration/wbcp.py"
WBCP_SHA = "fdd36b42a1d2c440cfa5e4df4b59c421c8fca477f8b3ffc5e7d9f095e889c023"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sha(WBCP) != WBCP_SHA:
            raise ValueError("Pinned pure WBCP calibration source changed")
        spec = importlib.util.spec_from_file_location("synthetic_pinned_wbcp", WBCP)
        cls.wbcp = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.wbcp
        spec.loader.exec_module(cls.wbcp)
        if jax.default_backend() != "cpu":
            raise ValueError("Synthetic fixtures require CPU")

    def setUp(self):
        self.eta = jnp.array([.25, .5, 1.], dtype=jnp.float32)
        self.q = jnp.array([1., 0., -2.], dtype=jnp.float32)
        self.y = jnp.array([0., 1., -1.], dtype=jnp.float32)
        self.unit = jnp.float32(2.)
        self.weights = jnp.array([.2, .3, .5], dtype=jnp.float32)

    def test_original_defaults_and_eta_transform(self):
        self.assertEqual(m.DEFAULT_WIDTH_PENALTY, .005)
        self.assertEqual(m.DEFAULT_BETA, 20.)
        np.testing.assert_allclose(m.positive_eta(jnp.array([-100., 0., 2.])), np.logaddexp(0., [-100., 0., 2.]) + .01, rtol=2e-7)

    def test_directional_signs_and_public_scale_floor(self):
        np.testing.assert_array_equal(m.directional_scores([2., 0., 1.], [0., 3., 1.], [1., 1., 1.], 2.), [1., 0., 0.])
        np.testing.assert_allclose(m.directional_scores([1e-6], [0.], [1e-9], 2.), [.5])
        np.testing.assert_allclose(m.radius([1e-9], 2., 3.), [6e-6])

    def test_one_sided_underestimation_event(self):
        coverage = m.soft_coverage(jnp.array([.1]), jnp.array([0.]), jnp.array([10.]), 1.)
        self.assertGreater(float(coverage[0]), .999999)
        self.assertEqual(m.directional_scores([0.], [10.], [.1], 1.)[0], 0.)
        self.assertTrue(m.hard_coverage([0.], [10.], [.1], 1.)[0])

    def test_tie_boundary_soft_and_hard_are_distinct(self):
        coverage = m.soft_coverage(jnp.array([.25]), jnp.array([1.]), jnp.array([.5]), 2.)
        self.assertEqual(float(coverage[0]), .5)
        self.assertTrue(m.hard_coverage([1.], [.5], [.25], 2.)[0])
        self.assertFalse(m.hard_coverage([1.], [.49], [.25], 2.)[0])

    def test_weighted_objective_exact_formula(self):
        coverage = 1 / (1 + np.exp(-20 * (np.asarray(self.y) / 2 - np.asarray(self.q) / 2 + np.asarray(self.eta))))
        expected = (np.sum(np.asarray(self.weights) * coverage) - .9) ** 2 + .005 * np.sum(np.asarray(self.weights) * np.asarray(self.eta) ** 2)
        self.assertAlmostEqual(float(m.scale_loss(self.eta, self.q, self.y, self.unit, self.weights)), float(expected), places=6)

    def test_only_eta_has_loss_gradient(self):
        gradients = jax.grad(m.scale_loss, argnums=(0, 1, 2, 3, 4))(self.eta, self.q, self.y, self.unit, self.weights)
        self.assertGreater(float(jnp.linalg.norm(gradients[0])), 0.)
        for gradient in gradients[1:]:
            np.testing.assert_array_equal(gradient, np.zeros_like(gradient))

    def test_eta_gradient_matches_finite_difference(self):
        eta = jnp.array([.2, .3], dtype=jnp.float32)
        q = jnp.array([.25, .3], dtype=jnp.float32)
        y = jnp.zeros(2)
        weights = jnp.array([.4, .6])
        gradient = np.asarray(jax.grad(m.scale_loss)(eta, q, y, 1., weights))
        finite = []
        for index in range(2):
            offset = np.zeros(2, dtype=np.float32); offset[index] = 1e-3
            finite.append((float(m.scale_loss(eta + offset, q, y, 1., weights)) - float(m.scale_loss(eta - offset, q, y, 1., weights))) / .002)
        np.testing.assert_allclose(gradient, finite, rtol=5e-4, atol=5e-4)

    def test_jit_vmap_scan_and_no_tracer_leak(self):
        eager = m.scale_loss(self.eta, self.q, self.y, self.unit, self.weights)
        with jax.checking_leaks():
            compiled = jax.jit(lambda eta: m.scale_loss(eta, self.q, self.y, self.unit, self.weights))
            np.testing.assert_allclose(compiled(self.eta), eager, rtol=1e-6)
        batch = jnp.stack([self.eta, self.eta * 1.1])
        mapped = jax.jit(jax.vmap(lambda eta: m.scale_loss(eta, self.q, self.y, self.unit, self.weights)))(batch)
        self.assertEqual(mapped.shape, (2,))
        _, scanned = jax.lax.scan(lambda carry, eta: (carry, m.scale_loss(eta, self.q, self.y, self.unit, self.weights)), jnp.float32(0), batch)
        np.testing.assert_allclose(mapped, scanned, rtol=1e-6)

    def test_scale_equivariance_with_frozen_unit(self):
        c = 7.
        np.testing.assert_allclose(m.directional_scores(self.q, self.y, self.eta, self.unit), m.directional_scores(self.q*c, self.y*c, self.eta, self.unit*c), rtol=1e-6)
        np.testing.assert_allclose(m.radius(self.eta, self.unit*c, 2.), c*m.radius(self.eta, self.unit, 2.), rtol=1e-6)
        np.testing.assert_allclose(m.scale_loss(self.eta, self.q*c, self.y*c, self.unit*c, self.weights), m.scale_loss(self.eta, self.q, self.y, self.unit, self.weights), rtol=1e-6)

    def test_numpy_shape_and_domain_rejections(self):
        cases = [(1., [0.], [1.], 1.), ([1.], 0., [1.], 1.), ([1.], [0.], 1., 1.), ([1.], [0.], [1.], [1.]), ([1.,2.], [0.], [1.], 1.), ([1.], [0.], [0.], 1.), ([1.], [0.], [1.], 0.), ([np.inf], [0.], [1.], 1.)]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                m.directional_scores(*args)

    def test_jax_shape_and_domain_rejections(self):
        for eta, q, y, unit, weights in [(1.,self.q,self.y,2.,self.weights), (self.eta,1.,self.y,2.,self.weights), (self.eta,self.q,self.y,[2.],self.weights), (self.eta,self.q,self.y,2.,1.)]:
            with self.assertRaises(ValueError): m.scale_loss(eta,q,y,unit,weights)
        for weights in [jnp.zeros(3), jnp.array([-.1,.1,1.]), jnp.ones(3)]:
            with self.assertRaises(ValueError): m.scale_loss(self.eta,self.q,self.y,2.,weights)
        with self.assertRaises(ValueError): m.scale_loss(self.eta,self.q,self.y,0.,self.weights)
        invalid = jax.jit(m.scale_loss)(self.eta,self.q,self.y,jnp.float32(0),self.weights)
        self.assertFalse(bool(jnp.isfinite(invalid)))

    @staticmethod
    def toy_apply(params, obs, actions):
        return m.positive_eta(actions @ params["weight"] + params["bias"] + obs[:,0]*0.)

    def test_frozen_parameters_and_live_action_gradient(self):
        params = {"weight":jnp.array([.2,-.3]), "bias":jnp.float32(.1)}
        obs, actions = jnp.zeros((2,1)), jnp.array([[.2,.3],[-.1,.4]])
        def total(p,a,unit,threshold):
            result=m.frozen_radius(self.toy_apply,p,obs,a,unit,threshold)
            return result.radius.sum()
        gp,ga,gu,gt=jax.grad(total,argnums=(0,1,2,3))(params,actions,jnp.float32(2),jnp.float32(3))
        for value in jax.tree_util.tree_leaves(gp): np.testing.assert_array_equal(value,np.zeros_like(value))
        self.assertGreater(float(jnp.linalg.norm(ga)),0.)
        self.assertEqual(float(gu),0.);self.assertEqual(float(gt),0.)
        result=jax.jit(lambda a:m.frozen_radius(self.toy_apply,params,obs,a,2.,3.))(actions)
        self.assertTrue(bool(result.valid));self.assertTrue(bool(result.usable))

    def test_zero_valid_infinity_refused_without_nan_or_product(self):
        params={"weight":jnp.array([.2,-.3]),"bias":jnp.float32(.1)}
        obs,actions=jnp.zeros((2,1)),jnp.zeros((2,2))
        for threshold,usable in [(0.,True),(np.inf,False),(-1.,False),(np.nan,False)]:
            result=jax.jit(lambda a:m.frozen_radius(self.toy_apply,params,obs,a,2.,threshold))(actions)
            self.assertEqual(bool(result.usable),usable);self.assertEqual(bool(result.valid),usable)
            np.testing.assert_array_equal(result.radius,[0.,0.])
            gradient=jax.grad(lambda a:m.frozen_radius(self.toy_apply,params,obs,a,2.,threshold).radius.sum())(actions)
            self.assertTrue(np.all(np.isfinite(gradient)));np.testing.assert_array_equal(gradient,np.zeros_like(gradient))
        np.testing.assert_array_equal(m.radius([1.,2.],2.,0.),[0.,0.])
        for threshold in [np.inf,np.nan,-1.]:
            with self.assertRaises(m.CalibrationRefusal):m.radius([1.],2.,threshold)

    def test_synthetic_posterior_alpha_beta_and_threshold_definition(self):
        scores=np.arange(1,129,dtype=float)/128
        base=self.wbcp.calibrate(scores,np.random.default_rng(17),alpha=.1,beta=.95,draws=1000)
        self.assertEqual(base.lambda_hat,116/128)
        self.assertEqual(base.lambda_hpd,float(base.crossings[949]))
        self.assertEqual(base.threshold,max(base.lambda_hat,base.lambda_hpd))
        relaxed=self.wbcp.calibrate(scores,np.random.default_rng(17),alpha=.2,beta=.95,draws=1000)
        lower_beta=self.wbcp.calibrate(scores,np.random.default_rng(17),alpha=.1,beta=.5,draws=1000)
        self.assertGreaterEqual(base.threshold,relaxed.threshold)
        self.assertGreaterEqual(base.threshold,lower_beta.threshold)
        self.assertEqual(base.n_eff,128.)

    def test_synthetic_posterior_zero_and_infinite_threshold(self):
        zero=self.wbcp.calibrate(np.zeros(128),np.random.default_rng(19),draws=1000)
        tiny=self.wbcp.calibrate(np.array([1.]),np.random.default_rng(19),draws=1000)
        self.assertEqual(zero.threshold,0.);self.assertTrue(np.isinf(tiny.threshold))
        np.testing.assert_array_equal(m.radius([1.,2.],1.,zero.threshold),[0.,0.])
        with self.assertRaises(m.CalibrationRefusal):m.radius([1.],1.,tiny.threshold)

    def test_one_synthetic_auxiliary_update_only(self):
        log_eta=jnp.array([-2.,-2.],dtype=jnp.float32)
        q,y,weights=jnp.array([.1,.1]),jnp.zeros(2),jnp.array([.5,.5])
        q_before,y_before=np.array(q),np.array(y)
        objective=lambda p:m.scale_loss(m.positive_eta(p),q,y,1.,weights)
        before=float(objective(log_eta))
        updated=log_eta-.001*jax.grad(objective)(log_eta)
        self.assertLess(float(objective(updated)),before)
        np.testing.assert_array_equal(q,q_before)
        np.testing.assert_array_equal(y,y_before)


if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(MathTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    review={
        "schema":"unifloral.bca_one_sided.synthetic_math_review.v1",
        "status":"pass" if result.wasSuccessful() else "failed",
        "utc":datetime.now(timezone.utc).isoformat(),
        "tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
        "source_sha256":{"one_sided_math.py":sha(HERE/'one_sided_math.py'),"test_one_sided_math.py":sha(Path(__file__)),"pinned_wbcp.py":sha(WBCP)},
        "runtime":{"python":sys.version,"jax":jax.__version__,"numpy":np.__version__,"backend":jax.default_backend(),"devices":[str(d) for d in jax.devices()],"jax_enable_x64":jax.config.jax_enable_x64},
        "retained_defaults":{"alpha":.1,"posterior_credibility":.95,"posterior_draws":1000,"cal_beta":20.,"width_penalty":.005,"cal_lr":.001,"eta_output":"softplus(log_eta)+.01","public_eta_floor":1e-6},
        "gradient_contract":"scale_loss trains eta only; Q1, target, unit and bootstrap weights stopped. frozen_radius stops parameter leaves/unit/threshold and preserves action derivatives.",
        "refusal_contract":"Infinite/invalid JAX references return finite zero placeholders and usable=False/valid=False; callers MUST reject actor updates. NumPy radius raises CalibrationRefusal. Threshold zero is usable.",
        "unit_contract":"No EMA update in primitives; caller supplies fixed original residual unit.",
        "limitations":["Synthetic CPU fixtures only; no logged data, real checkpoint/model or operational calibration bank loaded.","Only pinned pure wbcp.py was imported for synthetic posterior fixtures; no historical host/learner code imported.","One synthetic gradient update checks auxiliary arithmetic only; no real auxiliary, actor or critic training occurred.","Under jit invalid scale_loss numeric inputs produce nonfinite loss; the runner must reject nonfinite loss or gradient before optimizer commit.","No BCA efficacy, logged-data calibration, exchangeability, shift robustness or action-selection guarantee is established."],
        "real_bank_reads":0,"real_fits":0,"real_calibrations":0,"model_forwards_on_real_data":0,"actor_updates":0,"critic_updates":0,
    }
    (HERE/'MATH_REVIEW.json').write_text(json.dumps(review,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({"status":review['status'],"tests_run":result.testsRun,"receipt":str(HERE/'MATH_REVIEW.json')}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
