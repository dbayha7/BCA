"""Checks of the LQ signal harness, including built-in checks 1-5 of runs/wbcp_signal/expectations.md (Step 3).

Exact identities where the construction makes them exact (bitwise equality across signals and cases), Monte
Carlo with explicit standard errors for Q^pi and J, and equality with the repository's own fit_scale, refresh
and td3_bc actor update. CPU only, small sizes.
JAX_PLATFORMS=cpu python -m unittest experiments.signal.test_lq_harness
"""
import inspect
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from scipy import linalg

import algorithms.td3_bc as BASE
import algorithms.td3_bc_bca as P
from calibration.reference import initial_reference
from experiments.signal import lq_harness as H

SMALL = H.Settings(train_episodes=30, cal_episodes=40, cal_rows_per_episode=2, eval_episodes=30,
                   episode_length=20, fit_steps=40, actor_steps=15, batch_size=64)
EQUAL_HEAD_CASES = ("clean", "noisy_reward", "shared_bias")


def rollout_returns(system, K, s, a, rng, horizon=300):
    """Discounted returns of: action a at state s, then a = -K s (one row per rollout)."""
    total, discount = H.reward(system, s, a), 1.0
    for _ in range(horizon):
        s = H.step(system, s, a, rng)
        a = -s @ K.T
        discount *= system.gamma
        total = total + discount * H.reward(system, s, a)
    return total


def spec_q(system, K):
    """The specification's literal formulas (independent of quadratic_q): returns Q(s, a), V(s) and J."""
    A, B, Qc, Rc, g, S = system.A, system.B, system.Qc, system.Rc, system.gamma, system.noise
    M = A - B @ K
    Ps = linalg.solve_discrete_lyapunov(math.sqrt(g) * M.T, Qc + K.T @ Rc @ K)
    c = g * np.trace(Ps @ S) / (1 - g)

    def q(s, a):
        m = s @ A.T + a @ B.T
        return (-(np.einsum("ni,ij,nj->n", s, Qc, s) + np.einsum("ni,ij,nj->n", a, Rc, a))
                - g * (np.einsum("ni,ij,nj->n", m, Ps, m) + np.trace(Ps @ S)) - g * c)

    return q, lambda s: -np.einsum("ni,ij,nj->n", s, Ps, s) - c, -np.trace(Ps @ system.init) - c


class ExactQTests(unittest.TestCase):
    """Built-in check 1: Q^pi and J against Monte Carlo, the Bellman identity, and the literal formulas."""

    @classmethod
    def setUpClass(cls):
        cls.system = H.default_system()
        cls.K = H.lqr_gain(cls.system) + H.DIRECTION
        cls.Hq, cls.h, cls.P = H.exact_q(cls.system, cls.K)
        rng = np.random.default_rng(0)
        cls.s, cls.a = rng.normal(scale=0.5, size=(5, 3)), rng.normal(scale=0.3, size=(5, 2))

    def test_matches_the_specification_formulas(self):
        q, v, j = spec_q(self.system, self.K)
        np.testing.assert_allclose(H.qform(self.Hq, self.h, self.s, self.a), q(self.s, self.a), rtol=1e-10)
        np.testing.assert_allclose(H.qform(self.Hq, self.h, self.s, -self.s @ self.K.T), v(self.s), rtol=1e-10)
        self.assertAlmostEqual(H.value(self.system, self.K), j, places=10)

    def test_q_matches_monte_carlo(self):
        n = 20000
        rng = np.random.default_rng(1)
        for s, a in zip(self.s, self.a):
            g = rollout_returns(self.system, self.K, np.tile(s, (n, 1)), np.tile(a, (n, 1)), rng)
            se = g.std() / math.sqrt(n)
            exact = float(H.qform(self.Hq, self.h, s, a))
            self.assertLess(abs(g.mean() - exact), 4 * se + 1e-6, (g.mean(), exact, se))

    def test_value_matches_monte_carlo(self):
        n, rng = 50000, np.random.default_rng(2)
        s0 = rng.standard_normal((n, 3)) @ np.linalg.cholesky(self.system.init).T
        g = rollout_returns(self.system, self.K, s0, -s0 @ self.K.T, rng)
        self.assertLess(abs(g.mean() - H.value(self.system, self.K)), 4 * g.std() / math.sqrt(n))

    def test_bellman_identity(self):
        """Q(s,a) = E[r + gamma Q(s', pi(s'))]: exactly with E[s'^T X s'] = m^T X m + tr(X Sigma), and by Monte Carlo."""
        sys, K, g = self.system, self.K, self.system.gamma
        Pi = np.vstack([np.eye(3), -K])
        X = Pi.T @ self.Hq @ Pi  # Q(s', -K s') = s'^T X s' + h
        m = self.s @ sys.A.T + self.a @ sys.B.T
        rhs = H.reward(sys, self.s, self.a) + g * (np.einsum("ni,ij,nj->n", m, X, m) + np.trace(X @ sys.noise) + self.h)
        np.testing.assert_allclose(H.qform(self.Hq, self.h, self.s, self.a), rhs, rtol=1e-10)
        rng, n = np.random.default_rng(3), 400000
        for s, a in zip(self.s, self.a):
            s2 = H.step(sys, np.tile(s, (n, 1)), np.tile(a, (n, 1)), rng)
            y = H.reward(sys, s, a) + g * H.qform(self.Hq, self.h, s2, -s2 @ K.T)
            self.assertLess(abs(y.mean() - H.qform(self.Hq, self.h, s, a)), 4 * y.std() / math.sqrt(n))

    def test_shared_bias_is_the_value_of_a_shifted_reward(self):
        """b = dr + gamma E b(s', pi(s')): a TD residual sees only -dr, not the bootstrapped rest of b."""
        sys, K, kappa = self.system, self.K, 0.7
        K_b = H.lqr_gain(sys)
        params = H.case_critic(sys, "shared_bias", K, K_b, kappa)
        b = lambda s, a: H.head_error(params, s, a)
        np.testing.assert_array_equal(b(self.s, self.a)[:, 0], b(self.s, self.a)[:, 1])
        dr = kappa * np.sum(np.square(self.a + self.s @ K_b.T), axis=1)
        Hb, hb, _ = H.quadratic_q(sys, K, kappa * H.deviation_form(K_b))
        np.testing.assert_allclose(b(self.s, self.a)[:, 0], H.qform(Hb, hb, self.s, self.a), rtol=1e-10)
        Pi = np.vstack([np.eye(3), -K])
        X = Pi.T @ Hb @ Pi
        m = self.s @ sys.A.T + self.a @ sys.B.T
        bootstrap = sys.gamma * (np.einsum("ni,ij,nj->n", m, X, m) + np.trace(X @ sys.noise) + hb)
        np.testing.assert_allclose(b(self.s, self.a)[:, 0], dr + bootstrap, rtol=1e-10)
        self.assertTrue(np.all(bootstrap > 0))  # the part no TD residual sees

    def test_lqr_gain_is_the_optimum(self):
        K = H.lqr_gain(self.system)
        self.assertLess(np.abs(H.policy_gradient(self.system, K)).max(), 1e-6)
        rng = np.random.default_rng(4)
        for _ in range(20):
            self.assertLess(H.value(self.system, K + 0.05 * rng.standard_normal(K.shape)), H.value(self.system, K))

    def test_exact_policy_gradient_ascent_improves_J(self):
        K, values = self.K.copy(), [H.value(self.system, self.K)]
        for _ in range(10):
            K = K + 0.01 * H.policy_gradient(self.system, K)
            values.append(H.value(self.system, K))
        self.assertTrue(np.all(np.diff(values) > 0), values)


class SignalAndCriticTests(unittest.TestCase):
    def test_signals_equal_their_definitions(self):
        rng = np.random.default_rng(5)
        heads = jnp.asarray(rng.normal(size=(500, 2)), jnp.float32)
        t = jnp.asarray(rng.normal(size=500), jnp.float32)
        q1, q2 = np.asarray(heads[:, 0]), np.asarray(heads[:, 1])
        r = {sig: np.asarray(t - H.signal_q(jnp.int32(i), heads, t)) for i, sig in enumerate(H.SIGNALS)}
        np.testing.assert_array_equal(r["min"], np.asarray(t) - np.minimum(q1, q2))
        np.testing.assert_array_equal(r["q1"], np.asarray(t) - q1)
        np.testing.assert_array_equal(np.abs(r["maxabs"]), np.maximum(np.abs(np.asarray(t) - q1), np.abs(np.asarray(t) - q2)))
        equal = jnp.stack([heads[:, 0], heads[:, 0]], axis=1)
        same = [np.asarray(t - H.signal_q(jnp.int32(i), equal, t)) for i in range(3)]
        np.testing.assert_array_equal(same[0], same[1])
        np.testing.assert_array_equal(same[0], same[2])

    def test_critic_heads_are_exact_q_plus_error(self):
        sys = H.default_system()
        K_b = H.lqr_gain(sys)
        K = K_b + H.DIRECTION
        rng = np.random.default_rng(6)
        s, a = rng.normal(scale=0.4, size=(300, 3)), rng.normal(scale=0.3, size=(300, 2))
        forms = H.random_forms(5, rng)
        Hq, h, _ = H.exact_q(sys, K)
        f32 = lambda x: jnp.asarray(x, jnp.float32)
        heads = {}
        for case in H.CASES:
            p = H.case_critic(sys, case, K, K_b, 1.0, forms, 0.5)
            heads[case] = np.asarray(H.QuadraticCritic.apply(jax.tree_util.tree_map(f32, p), f32(s), f32(a)))
            expected = H.qform(Hq, h, s, a)[:, None] + H.head_error(p, s, a)
            np.testing.assert_allclose(heads[case], expected, rtol=2e-5, atol=2e-5)
        # The min identities behind built-in checks 3 and 4 hold bitwise in float32.
        np.testing.assert_array_equal(heads["q2_optimistic"].min(1), heads["q2_optimistic"][:, 0])
        np.testing.assert_array_equal(heads["q1_optimistic"].min(1), heads["clean"][:, 0])
        np.testing.assert_array_equal(heads["q2_optimistic"][:, 0], heads["clean"][:, 0])
        for case in EQUAL_HEAD_CASES:
            np.testing.assert_array_equal(heads[case][:, 0], heads[case][:, 1])

    def test_independent_errors_reach_the_head_gap_ratio(self):
        harness = H.Harness(SMALL)
        rows = harness.data("good", 0)["train"]
        K = harness.behavior_gain("good") + H.DIRECTION
        forms, c, achieved, attempt = H.independent_errors(harness.system, rows, K, harness.args, (1, 2, 3), 1.0)
        self.assertAlmostEqual(achieved, 1.0, delta=1e-3)
        np.testing.assert_allclose([np.linalg.norm(f) for f in forms], [1.0, 1.0])
        self.assertEqual(H.independent_errors(harness.system, rows, K, harness.args, (1, 2, 3), 1.0)[3], attempt)

    def test_controls_by_construction(self):
        rng = np.random.default_rng(7)
        dose = 1 + 0.5 * rng.random(1000)
        error = rng.random(1000) ** 2
        perm = rng.permutation(1000)
        arms, degenerate = H.control_doses(dose, error, perm)
        self.assertFalse(degenerate)
        self.assertEqual(np.ptp(arms["constant"]), 0.0)  # zero variance: every row the same value
        np.testing.assert_array_equal(np.sort(arms["shuffled"]), np.sort(arms["bca"]))
        np.testing.assert_array_equal(arms["bca"], dose.astype(np.float32))
        for name in ("constant", "shuffled", "oracle"):
            self.assertAlmostEqual(float(np.mean(arms[name], dtype=np.float64)), dose.mean(), places=5)
        np.testing.assert_allclose(arms["oracle"] / arms["oracle"].mean(), error / error.mean(), rtol=1e-5)
        arms, degenerate = H.control_doses(dose, np.zeros(1000), perm)
        self.assertTrue(degenerate)
        np.testing.assert_array_equal(arms["oracle"], arms["constant"])

    def test_distance_terciles_and_binned_miss(self):
        rng = np.random.default_rng(10)
        for n in (600, 601, 602):
            d = rng.random(n)
            bins = H.distance_terciles(d)
            sizes = np.bincount(bins, minlength=3)
            self.assertEqual(sizes.sum(), n)
            self.assertLessEqual(np.ptp(sizes), 1)
            self.assertLessEqual(d[bins == 0].max(), d[bins == 1].min())  # t1 nearest
            self.assertLessEqual(d[bins == 1].max(), d[bins == 2].min())
            values, width = rng.random(n), 0.3 + rng.random(n) * 0.4
            out = H.binned_miss(values, width, bins, "m")
            self.assertEqual(list(out), ["m_t1", "m_t2", "m_t3"])
            pooled = sum(out[f"m_t{k + 1}"] * sizes[k] for k in range(3)) / n
            self.assertAlmostEqual(pooled, H.miss(values, width), places=12)
            self.assertEqual(H.binned_miss(values, 0.5, bins, "m")["m_t2"], H.miss(values[bins == 1], 0.5))
        self.assertLessEqual(np.ptp(np.bincount(H.distance_terciles(np.zeros(9)))), 0)  # ties: equal sizes

    def test_summaries_flag_excluded_values(self):
        stat = H.summarize([1.0, -math.inf, 3.0])
        self.assertEqual((stat["mean"], stat["n"], stat["nonfinite"], stat["values"]), (2.0, 2, 1, [1.0, None, 3.0]))
        self.assertEqual(H.fmt(stat), "2 ± 1 (n=2, 1 nonfinite)")
        self.assertEqual(H.fmt(H.summarize([None, None])), "n/a")  # undefined or not applicable throughout
        self.assertEqual(H.fmt(H.summarize([None, 4.0])), "4 (n=1, 1 nonfinite)")
        self.assertEqual(H.fmt(H.summarize([-math.inf, None])), "n/a (n=0, 2 nonfinite)")
        self.assertEqual(H.summarize([None, math.nan, 1.0])["undefined"], 1)
        self.assertEqual(H.fmt(H.summarize([1.0, 3.0])), "2 ± 1")
        self.assertEqual(H.fmt(H.summarize([2.0])), "2")


class RepositoryFidelityTests(unittest.TestCase):
    """The harness runs BCA's own code: fit_scale, refresh and td3_bc's actor update."""

    @classmethod
    def setUpClass(cls):
        cls.h = h = H.Harness(SMALL)
        data = h.data("good", 0)
        K_b = h.behavior_gain("good")
        cls.K = K_b + H.DIRECTION
        cls.params = H.case_critic(h.system, "q1_optimistic", cls.K, K_b, 1.0)  # heads differ
        cls.native = h.native(cls.K, cls.params)
        cls.train, cls.cal = h.transitions(data["train"], False), h.transitions(data["cal"], False)
        key = jax.random.PRNGKey(11)
        cal = h.models[2]
        state = TrainState.create(apply_fn=cal.apply, params=cal.init(key, jnp.zeros((1, 3)), jnp.zeros((1, 2))),
                                  tx=optax.adam(h.config.cal_lr))
        cls.state = P.State(cls.native, state, jnp.asarray(1.0), initial_reference(state.params))
        cls.keys = [jax.random.fold_in(key, i) for i in range(6)]
        cls.batches = [jax.tree_util.tree_map(lambda x: x[i * 64:(i + 1) * 64], cls.train) for i in range(6)]

    def fit(self, fn, state):
        step = jax.jit(fn)
        for batch, key in zip(self.batches, self.keys):
            state, _ = step(state, batch, key)
        return state

    def assert_same_fit(self, a, b):
        for x, y in zip(jax.tree_util.tree_leaves((a.calibrator.params, a.residual_scale)),
                        jax.tree_util.tree_leaves((b.calibrator.params, b.residual_scale))):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))

    def test_fit_scale_signal_min_is_the_repository_fit_scale(self):
        h = self.h
        repo = self.fit(lambda st, b, k: P.fit_scale(h.args, h.config, h.models, st, b, k, H.MAX_ACTION), self.state)
        mine = self.fit(lambda st, b, k: H.fit_scale_signal(h.args, h.config, h.models, st, b, k, jnp.int32(0),
                                                            H.MAX_ACTION), self.state)
        self.assert_same_fit(repo, mine)
        self.assertFalse(np.array_equal(np.asarray(repo.residual_scale), 1.0))  # it moved

    def test_fit_scale_signal_q1_is_fit_scale_on_a_q1_only_critic(self):
        h = self.h

        class Q1Only:  # the current critic exposes only head 0; the target critic keeps both heads
            @staticmethod
            def apply(params, obs, action):
                heads = H.QuadraticCritic.apply(params, obs, action)
                return heads[..., :1] if "q1_only" in params else heads

        native = self.native._replace(critic=self.native.critic.replace(
            params=dict(self.native.critic.params, q1_only=jnp.zeros(()))))
        models = (h.models[0], Q1Only(), h.models[2])
        repo = self.fit(lambda st, b, k: P.fit_scale(h.args, h.config, models, st, b, k, H.MAX_ACTION),
                        self.state._replace(native=native))
        mine = self.fit(lambda st, b, k: H.fit_scale_signal(h.args, h.config, h.models, st, b, k, jnp.int32(1),
                                                            H.MAX_ACTION), self.state)
        self.assert_same_fit(repo, mine)
        other = self.fit(lambda st, b, k: H.fit_scale_signal(h.args, h.config, h.models, st, b, k, jnp.int32(0),
                                                             H.MAX_ACTION), self.state)
        self.assertFalse(np.array_equal(np.asarray(other.residual_scale), np.asarray(mine.residual_scale)))

    def test_calibration_is_the_repository_refresh(self):
        h = self.h
        state = self.fit(lambda st, b, k: H.fit_scale_signal(h.args, h.config, h.models, st, b, k, jnp.int32(0),
                                                             H.MAX_ACTION), self.state)
        key = jax.random.PRNGKey(12)
        repo, metrics, _ = P.refresh(h.args, h.config, h.models, state, self.cal, key, H.MAX_ACTION,
                                     heldout_ids=np.arange(len(self.cal.obs)))
        self.assertTrue(metrics["posterior_certified"])
        mine, _, _, _ = H.calibrate_signal(h.args, h.config, h.models, state, self.cal, key, 0, H.MAX_ACTION)
        for field in ("threshold", "lambda_hat", "lambda_hpd", "residual_scale", "n_eff", "ready"):
            np.testing.assert_array_equal(np.asarray(getattr(repo.posterior, field)), np.asarray(getattr(mine, field)))

    def test_actor_terms_are_the_td3_bc_actor_gradient(self):
        h = self.h
        actor = TrainState.create(apply_fn=H.LinearActor.apply, params={"K": jnp.asarray(self.K, jnp.float32)},
                                  tx=optax.sgd(1.0))
        native = self.native._replace(actor=actor, actor_target=actor)
        dose = jnp.asarray(1 + 0.5 * np.random.default_rng(8).random(len(self.train.obs)), jnp.float32)
        for mult in (dose, None):
            new, _ = BASE.td3_bc_update(h.args, h.models[0].apply, h.models[1].apply, native, self.train, 0,
                                        jax.random.PRNGKey(0), H.MAX_ACTION, bc_multiplier=mult)
            qg, bg = H.actor_terms(h.args, h.models, native.critic.params, actor.params["K"], self.train.obs,
                                   self.train.action, mult)
            np.testing.assert_allclose(np.asarray(new.actor.params["K"]), np.asarray(actor.params["K"] - qg - bg),
                                       rtol=1e-5, atol=1e-6)

    def test_q_term_ascent_improves_J_on_the_clean_case(self):
        """With exact Q^pi and no BC, the harness's own actor loop climbs the true value."""
        h = self.h
        native = h.native(self.K, H.case_critic(h.system, "clean", self.K, h.behavior_gain("good")))
        n = len(self.train.obs)
        rows = jnp.asarray(np.random.default_rng(9).integers(n, size=(300, 64)), jnp.int32)
        keys = jax.vmap(partial_fold(jax.random.PRNGKey(5)))(jnp.arange(300))
        out, _ = h.actor[True](native, self.train, rows, keys, jnp.zeros(n, jnp.float32))
        K1 = np.asarray(out.actor.params["K"], np.float64)
        self.assertGreater(H.value(h.system, K1), H.value(h.system, self.K))
        self.assertGreater(H.cosine(K1 - self.K, H.policy_gradient(h.system, self.K)), 0.0)


def partial_fold(key):
    return lambda i: jax.random.fold_in(key, i)


class HarnessTests(unittest.TestCase):
    """Built-in checks 2-5 and the shared data, seeds and targets, on one small replicate of every case."""

    @classmethod
    def setUpClass(cls):
        cls.h = H.Harness(SMALL)
        grid = cls.h.settings_grid(H.CASES, [1.0], [1.0])
        cls.records, cls.diagnostics, cls.prints = cls.h.replicate("poor", 0, grid)
        cls.checks = H.builtin_checks(cls.prints)

    def by(self, case, signal, table=None):
        table = self.prints if table is None else table
        (row,) = [r for r in table if r["case"] == case and r["signal"] == signal]
        return row

    def test_every_check_passes(self):
        for name, results in self.checks.items():
            self.assertTrue(results, name)
            for tag, ok in results:
                self.assertTrue(ok, f"{name}: {tag}")

    def test_check2_equal_heads_give_identical_results_for_every_signal(self):
        for case in EQUAL_HEAD_CASES:
            metrics = [self.by(case, s, self.records)["metrics"] for s in H.SIGNALS]
            self.assertEqual(json.dumps(metrics[0]), json.dumps(metrics[1]), case)
            self.assertEqual(json.dumps(metrics[0]), json.dumps(metrics[2]), case)

    def test_check3_q2_optimistic_min_is_q1(self):
        a, b = (self.by("q2_optimistic", s, self.records)["metrics"] for s in ("min", "q1"))
        self.assertEqual(json.dumps(a), json.dumps(b))
        c = self.by("q2_optimistic", "maxabs")
        self.assertNotEqual(c["residual_eval"], self.by("q2_optimistic", "min")["residual_eval"])

    def test_check4_q1_optimistic_min_residual_is_the_clean_one(self):
        a, b = self.by("q1_optimistic", "min"), self.by("clean", "min")
        for field in ("residual_cal", "residual_eval", "fit_targets", "sigma", "threshold", "doses"):
            self.assertEqual(a[field], b[field], field)
        self.assertNotEqual(a["actor"], b["actor"])  # the actor climbs Q1, which differs

    def test_check5_q_term_gradient_is_shared_by_the_signals(self):
        for case in H.CASES:
            self.assertEqual(len({self.by(case, s)["qgrad"] for s in H.SIGNALS}), 1, case)
            norms = {self.by(case, s, self.records)["metrics"]["q_term_grad_norm"] for s in H.SIGNALS}
            self.assertEqual(len(norms), 1, case)

    def test_same_data_seeds_and_targets_across_signals_and_cases(self):
        for field in ("fit_batches", "cal_target", "eval_target", "fit_targets"):
            for case in H.CASES:
                self.assertEqual(len({self.by(case, s)[field] for s in H.SIGNALS}), 1, (case, field))
        self.assertEqual(len({p["fit_batches"] for p in self.prints}), 1)
        # noisy_reward changes only the observed reward, so its targets differ from clean's
        self.assertNotEqual(self.by("noisy_reward", "min")["cal_target"], self.by("clean", "min")["cal_target"])

    def test_independent_errors_separate_all_three_signals(self):
        digests = {self.by("independent_errors", s)["residual_eval"] for s in H.SIGNALS}
        self.assertEqual(len(digests), 3)
        (d,) = [d for d in self.diagnostics if d["case"] == "independent_errors"]
        self.assertAlmostEqual(d["independent_ratio"], SMALL.independent_ratio, delta=1e-3)

    def test_tercile_metrics_partition_the_pooled_misses(self):
        n_eval = SMALL.eval_episodes * SMALL.episode_length
        self.assertEqual(n_eval % 3, 0)  # equal terciles, so the pooled miss is their plain mean
        for r in self.records:
            m = r["metrics"]
            for name in ("miss_q1err_pi", "miss_q1err_logged"):
                self.assertAlmostEqual(np.mean([m[f"{name}_t{k}"] for k in (1, 2, 3)]), m[name], places=12)
            if r["case"] in ("clean", "noisy_reward", "q2_optimistic"):  # Q1 = Q^pi exactly
                self.assertEqual([m[f"miss_q1err_pi_t{k}"] for k in (1, 2, 3)], [0.0, 0.0, 0.0])
        for d in self.diagnostics:
            t = [d[f"pi_beta_distance_t{k}"] for k in (1, 2, 3)]
            self.assertTrue(0 < t[0] < t[1] < t[2], t)
            self.assertAlmostEqual(np.mean(t), d["pi_beta_distance"], places=12)

    def test_controls_and_exact_quantities_in_records(self):
        for r in self.records:
            m = r["metrics"]
            self.assertTrue(1.0 <= m["dose_min"] <= m["dose_p05"] <= m["dose_mean"] <= m["dose_p95"] <= m["dose_max"] < 1.5)
            self.assertEqual(m["fit_accepted"], 1.0)
            if r["case"] in ("clean", "noisy_reward", "q2_optimistic"):  # Q1 = Q^pi exactly
                self.assertEqual(m["miss_q1err_logged"], 0.0)
                self.assertEqual(m["alignment_K"], 1.0)
                self.assertIsNone(m["dose_corr_q1err_logged"])
                self.assertEqual(m["oracle_degenerate"], 1.0)


class RunTests(unittest.TestCase):
    def test_run_writes_every_preregistered_quantity(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "lq"
            payload = H.run(SMALL, out, behaviors=("good",), replicates=2, kappas=(1.0,), offsets=(0.5, 1.0),
                            log=lambda *_: None)
            text = (out / "results.json").read_text()
            json.loads(text, parse_constant=lambda c: self.fail("non-JSON constant " + c))
            self.assertTrue((out / "results.md").read_text().startswith("# LQ signal harness"))
            with self.assertRaises(FileExistsError):
                H.run(SMALL, out, behaviors=("good",), replicates=1, log=lambda *_: None)
        self.assertTrue(all(payload["builtin_checks_summary"].values()), payload["builtin_checks_summary"])
        self.assertEqual(len(payload["results"]), len(H.CASES) * len(H.SIGNALS) * 2)
        self.assertEqual({r["pi_offset"] for r in payload["results"]}, {0.5, 1.0})
        needed = ["dose_mean", "dose_sd", "dose_p05", "dose_p95", "dose_max", "miss_own_logged", "miss_q1err_logged",
                  "miss_q1err_pi", "miss_resid_q1_logged", "miss_resid_q2_logged", "width_logged",
                  "dose_corr_q1err_logged", "dose_corr_q1err_pi", "dose_corr_deviation", "alignment_K",
                  "alignment_action", "alignment_rowwise", "q_term_grad_norm"]
        needed += [f"miss_q1err_{at}_t{k}" for at in ("pi", "logged") for k in (1, 2, 3)]
        needed += [f"{k}_{arm}" for arm in H.ARMS for k in ("J_change", "bc_term_grad_norm", "update_pg_cosine")]
        needed += [f"J_gain_bca_minus_{arm}" for arm in H.ARMS[1:]]
        for r in payload["results"]:
            self.assertFalse(set(needed) - set(r["metrics"]))
            self.assertEqual(r["metrics"]["J_change_bca"]["n"], 2)
            self.assertIsNotNone(r["metrics"]["J_change_bca"]["se"])
        diag = payload["settings"][0]["diagnostics"]
        for name in ("J_pi", "pi_beta_distance", "logged_deviation", "action_outside_pi", "head_gap_over_residual",
                     "pi_beta_distance_t1", "pi_beta_distance_t2", "pi_beta_distance_t3"):
            self.assertIn(name, diag)

    def test_default_grid_varies_the_distance_within_a_behaviour(self):
        """Step 3 expectation 3's distance clause: >= 2 offsets by default in run() and the CLI, all stable."""
        self.assertGreaterEqual(len(set(H.DEFAULT_OFFSETS)), 2)
        self.assertEqual(inspect.signature(H.run).parameters["offsets"].default, H.DEFAULT_OFFSETS)
        self.assertEqual(inspect.signature(H.run).parameters["kappas"].default, H.DEFAULT_KAPPAS)
        ns = H.build_parser().parse_args(["--output", "x"])
        self.assertEqual((ns.pi_offset, ns.kappa), (list(H.DEFAULT_OFFSETS), list(H.DEFAULT_KAPPAS)))
        H.Harness(H.Settings()).check_stable(H.BEHAVIORS, H.DEFAULT_OFFSETS)

    def test_unstable_settings_are_refused_before_any_compute(self):
        with tempfile.TemporaryDirectory() as tmp:
            for settings, offsets in ((replace(SMALL, poor_gain=-1.0), (1.0,)), (SMALL, (4.0,))):
                out = Path(tmp) / "lq"
                with self.assertRaisesRegex(ValueError, "not discounted-stable"):
                    H.run(settings, out, behaviors=("poor",), replicates=1, offsets=offsets, log=lambda *_: None)
                self.assertFalse(out.exists())

    def test_results_json_is_strict(self):
        """A non-finite number anywhere in the payload fails the write instead of emitting 'Infinity'."""
        real = H.aggregate

        def aggregate(records, diagnostics):
            results, rows = real(records, diagnostics)
            return results, rows + [dict(stray=math.inf)]

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(H, "aggregate", aggregate):
            out = Path(tmp) / "lq"
            with self.assertRaisesRegex(ValueError, "JSON"):
                H.run(SMALL, out, cases=("clean",), signals=("min",), behaviors=("good",), replicates=1,
                      offsets=(1.0,), log=lambda *_: None)
            self.assertFalse((out / "results.json").exists())


if __name__ == "__main__":
    unittest.main()
