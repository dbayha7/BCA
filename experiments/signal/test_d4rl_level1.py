"""CPU checks for experiments/signal/d4rl_level1.py: step 4's stage 4B-0 Level-1 diagnostics on frozen TD3+BC pools.

Weights: the rank-normal multiset reproduces STEP4_DESIGN section 2 / expectations B6 (n = 10,000: s = 0.83 CV 0.914,
range 0.090-5.714, Kish 0.545, p90/p10 8.39; s = 0.47 CV 0.482, range 0.277-2.908, Kish 0.811); ranks break ties by
the seeded permutation, never by row order; SIG, SHUF, STRAT and ANTI share the multiset bitwise (expectation 2.3),
STRAT moves weights only within leverage deciles and ANTI reverses SIG's ranks. Gradients, on a synthetic TD3+BC actor,
twin critic and scale network (random parameters, 50 rows in padded chunks): the vjp operator G assembles the gradient
of the build spec's loss for P0, P1, P2 and P3 (jax.grad of explicit_loss), and the repository td3_bc_update's actor
gradient with and without a BC multiplier (captured exactly in the optimizer state); the vmap(grad) leverage equals a
per-row loop and N times G of a one-row cotangent; P0(m) projects to m_eff = m, tau = 0, a constant trust u = 1/(1 +
beta) to m_eff = 1 + beta (expectation 2.4), knob 0 returns NONE (2.1) and a penalty whose width ignores the action
returns NONE (2.5). Matching: first crossing, tolerance, non-monotone flag and unreachability on closed forms. Data
quality: the raw episode segmentation is runtime.td3_bc._convert's, the returns count the timeout row, the
medium-expert share is beta-free. End to end: the analysis on the synthetic engine, and the whole run on a synthetic
short-train pool (freeze_scores, then frozen_signals fit and evaluate), including the reproduction gate and the
data-quality reads. No J is computed anywhere: the module has no value function to call.
JAX_PLATFORMS=cpu python -m unittest experiments.signal.test_d4rl_level1
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.signal import d4rl_level1 as D

OBS, ACT, N = 4, 2, 50


def synthetic_engine(n=N, seed=0, action_free_width=False, **kwargs):
    """A TD3+BC actor, twin critic and scale network with random parameters on random rows."""
    import jax
    import jax.numpy as jnp

    import algorithms.td3_bc as BASE
    from calibration.network import Calibrator

    g = np.random.default_rng(seed)
    actor, critic = BASE.Actor(OBS, ACT, 1.0), BASE.DualCritic(OBS, ACT)
    cal = Calibrator(jnp.zeros(OBS), jnp.ones(OBS), state_dep=True)
    k = jax.random.split(jax.random.PRNGKey(seed), 3)
    zeros = jnp.zeros((1, OBS)), jnp.zeros((1, ACT))
    actor_params = actor.init(k[0], zeros[0])
    critic_params = critic.init(k[1], *zeros)
    cal_params = cal.init(k[2], *zeros)
    # larger scale-network weights, so that eta varies across rows and actions
    cal_params = jax.tree_util.tree_map(lambda x: 3.0 * x, cal_params)
    if action_free_width:  # the first layer ignores the action inputs: grad_a W = 0 exactly
        kernel = cal_params["params"]["Dense_0"]["kernel"]
        cal_params = jax.tree_util.tree_map(lambda x: x, cal_params)
        cal_params["params"]["Dense_0"]["kernel"] = kernel.at[OBS:].set(0.0)
    obs = g.normal(size=(n, OBS)).astype(np.float32)
    action = g.uniform(-0.9, 0.9, size=(n, ACT)).astype(np.float32)
    settings = dict(row_chunk=16, k_block=8, leverage_chunk=8)
    settings.update(kwargs)
    engine = D.Level1((actor, critic, cal), actor_params, critic_params, cal_params, 0.7, 1.3, obs, action, 2.5,
                      **settings)
    return engine


def flat(tree):
    from jax.flatten_util import ravel_pytree

    return np.asarray(ravel_pytree(tree)[0], np.float64)


class WeightTests(unittest.TestCase):
    def test_multiset_reproduces_the_design_statistics(self):
        primary = D.multiset_stats(D.multiset(10_000, 0.83))
        self.assertEqual(round(primary["cv"], 3), 0.914)
        self.assertEqual((round(primary["min"], 3), round(primary["max"], 3)), (0.090, 5.714))
        self.assertEqual(round(primary["kish"], 3), 0.545)
        self.assertEqual(round(primary["p90_p10"], 2), 8.39)
        robust = D.multiset_stats(D.multiset(10_000, 0.47))
        self.assertEqual((round(robust["cv"], 3), round(robust["min"], 3), round(robust["max"], 3),
                          round(robust["kish"], 3)), (0.482, 0.277, 2.908, 0.811))
        for n in (1, 7, 256, 10_001):
            m = D.multiset(n)
            self.assertAlmostEqual(float(m.mean()), 1.0, places=12)
            self.assertTrue(np.all(np.diff(m) >= 0))

    def test_identical_to_placement_harness(self):
        """The task's reuse clause: placement_harness.rank_normal appeared after this module; the two agree bitwise."""
        if not Path(D.__file__).with_name("placement_harness.py").exists():
            self.skipTest("experiments/signal/placement_harness.py is not written yet")
        from experiments.signal import placement_harness as PH

        g = np.random.default_rng(9)
        for n in (1, 10, 257, 5000):
            x = np.round(g.normal(size=n), 1)  # many ties
            tie = g.permutation(n)
            for s in (D.S_PRIMARY, D.S_ROBUST):
                np.testing.assert_array_equal(D.multiset(n, s, D.CLIP), PH.multiset(n, s, D.CLIP))
                weights, rank = PH.rank_normal(x, s, D.CLIP, tie)
                np.testing.assert_array_equal(D.rank_normal(x, s, D.CLIP, tie), weights)
                np.testing.assert_array_equal(D.ranks(x, tie), rank)

    def test_ranks_break_ties_by_the_permutation_not_by_row_order(self):
        x = np.array([0.3, 0.1, 0.3, 0.3, -1.0, 0.1])
        tie = np.array([5, 0, 1, 4, 3, 2])
        r = D.ranks(x, tie)
        np.testing.assert_array_equal(np.sort(r), np.arange(6))
        self.assertEqual(r[4], 0)
        self.assertLess(r[1], r[5])  # tie at 0.1: tie_perm 0 before 2
        self.assertEqual([r[i] for i in (2, 3, 0)], [3, 4, 5])  # tie at 0.3: tie_perm 1, 4, 5
        w = D.rank_normal(x, 0.83, 2.5, tie)
        np.testing.assert_array_equal(w, D.multiset(6)[r])
        other = D.ranks(x, np.array([0, 5, 4, 1, 3, 2]))
        self.assertFalse(np.array_equal(r, other))  # another permutation reorders the ties only
        np.testing.assert_array_equal(np.sort(r[[1, 5]]), np.sort(other[[1, 5]]))
        with self.assertRaises(ValueError):
            D.ranks(x, np.arange(5))
        with self.assertRaises(ValueError):
            D.ranks(np.array([0.0, np.nan]), np.arange(2))

    def test_assignments_share_the_multiset(self):
        n, seed = 503, 17
        g = np.random.default_rng(3)
        x, leverage = g.lognormal(size=n), g.lognormal(size=n)
        tie = np.random.default_rng([seed, 1]).permutation(n)
        deciles = D.decile_index(leverage, tie)
        self.assertLessEqual(np.ptp(np.bincount(deciles)), 1)
        shuf, strat = D.shuffle_permutations(n, seed), D.strat_permutations(deciles, seed, "bc")
        weights, m = D.assignments(x, tie, shuf, strat)
        arms = D.arm_list(weights)
        self.assertEqual([name for name, _ in arms][:2] + [arms[-1][0]], ["SIG", "SHUF_0", "ANTI"])
        self.assertEqual(len(arms), 1 + D.N_SHUF + D.N_STRAT + 1)
        self.assertTrue(D.same_multiset(arms, m))
        sig = weights["SIG"]
        self.assertTrue(np.all(np.diff(sig[np.argsort(x)]) >= 0))  # nondecreasing in x (the clip ties the ends)
        np.testing.assert_array_equal(weights["ANTI"], m[n - 1 - D.ranks(x, tie)])
        for w in weights["STRAT"]:  # each decile keeps its own weights
            for d in range(D.DECILES):
                np.testing.assert_array_equal(np.sort(w[deciles == d]), np.sort(sig[deciles == d]))
        self.assertEqual(len({w.tobytes() for w in weights["SHUF"]}), D.N_SHUF)
        again, _ = D.assignments(x, tie, D.shuffle_permutations(n, seed), D.strat_permutations(deciles, seed, "bc"))
        for (_, a), (_, b) in zip(arms, D.arm_list(again)):
            np.testing.assert_array_equal(a, b)
        q_strat = D.strat_permutations(deciles, seed, "q")
        self.assertFalse(np.array_equal(q_strat[0], strat[0]))  # leverage kinds draw their own permutations
        self.assertFalse(D.same_multiset([("x", sig * 1.0000001)], m))


class GradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = synthetic_engine()

    def explicit_grad(self, **kwargs):
        import jax
        import jax.numpy as jnp

        e = self.engine
        obs, action = e.obs[:e.n], e.action[:e.n]
        kwargs = {k: (jnp.asarray(v, jnp.float32) if isinstance(v, np.ndarray) else v) for k, v in kwargs.items()}
        grad = jax.grad(D.explicit_loss, argnums=1)(e.models, e.params, e.critic_params, e.cal_params, e.unit,
                                                     e.threshold, obs, action, e.alpha, **kwargs)
        return flat(grad)

    def assert_close(self, got, expected, rtol=2e-5):
        scale = np.max(np.abs(expected))
        self.assertGreater(scale, 0)
        self.assertLessEqual(np.max(np.abs(got - expected)) / scale, rtol)

    def test_operator_assembles_the_loss_gradient_of_every_placement(self):
        e = self.engine
        g = np.random.default_rng(5)
        v, u = 1.0 + g.lognormal(size=e.n), 1.0 / (1.0 + 2.0 * g.lognormal(size=e.n))
        gq = e.weighted(e.cq, np.ones((1, e.n)))[0]
        gb, gp = e.stacked([e.cb, e.cp])
        self.assert_close(gq + gb, self.explicit_grad())  # NONE
        self.assert_close(gq + 3.0 * gb, self.explicit_grad(bc_w=np.full(e.n, 3.0)))  # P0(3)
        self.assert_close(gq + e.weighted(e.cb, v[None])[0], self.explicit_grad(bc_w=v))  # P1
        self.assert_close(e.weighted(e.cq, u[None])[0] + gb, self.explicit_grad(q_w=u))  # P2 (lambda unweighted)
        self.assert_close(gq + gb + 0.7 * gp, self.explicit_grad(pen_c=0.7))  # P3 through W's action input
        self.assertGreater(np.linalg.norm(gp), 0)

    def test_operator_matches_the_repository_actor_update(self):
        import jax
        import jax.numpy as jnp
        import optax
        from flax.training.train_state import TrainState

        import algorithms.td3_bc as BASE
        from runtime.networks import Transition

        e = self.engine
        capture = optax.GradientTransformation(  # the actor's gradient lands, exactly, in the optimizer state
            lambda p: jax.tree_util.tree_map(jnp.zeros_like, p),
            lambda grads, state, params=None: (jax.tree_util.tree_map(jnp.zeros_like, grads), grads))
        actor = TrainState.create(apply_fn=e.models[0].apply, params=e.params, tx=capture)
        critic = TrainState.create(apply_fn=e.models[1].apply, params=e.critic_params, tx=optax.set_to_zero())
        state = BASE.AgentTrainState(actor, actor, critic, critic)
        g = np.random.default_rng(1)
        batch = Transition(e.obs[:e.n], e.action[:e.n], jnp.asarray(g.normal(size=e.n), jnp.float32),
                           jnp.asarray(g.normal(size=(e.n, OBS)), jnp.float32), jnp.zeros(e.n, jnp.float32))
        args = BASE.Args()
        gq = e.weighted(e.cq, np.ones((1, e.n)))[0]
        gb = e.stacked([e.cb])[0]
        for dose in (None, e.rows["dose"]):
            new, _ = BASE.td3_bc_update(args, e.models[0].apply, e.models[1].apply, state, batch, 0,
                                        jax.random.PRNGKey(0), 1.0,
                                        bc_multiplier=None if dose is None else jnp.asarray(dose, jnp.float32))
            expected = flat(new.actor.opt_state)
            got = gq + (gb if dose is None else e.weighted(e.cb, dose[None])[0])
            self.assert_close(got, expected)

    def test_leverage_is_the_per_example_gradient_norm(self):
        import jax

        e = self.engine
        lev = e.leverage()
        models, lam = e.models, np.float32(e.lam)
        from calibration.reference import positive_scale

        def bc(p, s, a):
            return ((models[0].apply(p, s[None])[0] - a) ** 2).mean()

        def q(p, s):
            return lam * models[1].apply(e.critic_params, s[None], models[0].apply(p, s[None]))[0, 0]

        def pen(p, s):
            pi = models[0].apply(p, s[None])
            return lam * e.threshold * positive_scale(models[2].apply(e.cal_params, s[None], pi), e.unit)[0]

        for i in (0, 17, e.n - 1):
            s, a = e.obs[i], e.action[i]
            loop = dict(bc=np.linalg.norm(flat(jax.grad(bc)(e.params, s, a))),
                        q=np.linalg.norm(flat(jax.grad(q)(e.params, s))),
                        pen=np.linalg.norm(flat(jax.grad(pen)(e.params, s))))
            for kind in D.LEVERAGE_KINDS:
                self.assertAlmostEqual(lev[kind][i] / loop[kind], 1.0, delta=1e-5)
            one = np.zeros((e.n, 1))
            one[i] = 1.0
            via_g = dict(bc=e.n * np.linalg.norm(e.stacked([one * e.cb])[0]),
                         q=e.n * np.linalg.norm(e.stacked([one * e.cq])[0]),
                         pen=e.n * np.linalg.norm(e.stacked([one * e.cp])[0]))
            for kind in D.LEVERAGE_KINDS:
                self.assertAlmostEqual(lev[kind][i] / via_g[kind], 1.0, delta=1e-4)

    def test_projection_identities(self):
        e = self.engine
        gq = e.weighted(e.cq, np.ones((1, e.n)))[0]
        gb = e.stacked([e.cb])[0]
        for m in (1.3, 2.0, 4.0):
            p = D.projection(gq + m * gb, gq, gb)
            self.assertAlmostEqual(p["m_eff"], m, delta=1e-9 * m)
            self.assertLessEqual(p["tau"], 1e-9)
        for beta in (0.5, 1.0, 3.0):  # constant trust is P0(1 + beta) up to the loss scale
            trust = e.weighted(e.cq, np.full((1, e.n), 1.0 / (1.0 + beta)))[0]
            p = D.projection(trust + gb, gq, gb)
            self.assertAlmostEqual(p["m_eff"], 1.0 + beta, delta=1e-4 * (1.0 + beta))
            self.assertLessEqual(p["tau"], 1e-4)
            self.assertAlmostEqual(np.linalg.norm(gb) / np.linalg.norm(trust),
                                   (1.0 + beta) * np.linalg.norm(gb) / np.linalg.norm(gq), delta=1e-4)

    def test_knob_zero_and_an_action_free_width_return_none(self):
        e = self.engine
        gq = e.weighted(e.cq, np.ones((1, e.n)))[0]
        M = D.multiset(e.n)[np.random.default_rng(0).permutation(e.n)]
        zero = e.weighted(e.cq, 1.0 / (1.0 + 0.0 * M[None]))[0]  # P2's evaluator at beta = 0
        np.testing.assert_array_equal(zero, gq)
        np.testing.assert_array_equal(e.weighted(e.cb, (1.0 + 0.0 * M)[None])[0], e.stacked([e.cb])[0])  # P1
        flat_width = synthetic_engine(action_free_width=True)
        np.testing.assert_array_equal(flat_width.rows["wa"], 0.0)
        np.testing.assert_array_equal(flat_width.stacked([flat_width.cp])[0], 0.0)  # P3: Delta g = 0 for every c
        self.assertFalse(np.all(e.rows["wa"] == 0.0))


class MatchingTests(unittest.TestCase):
    def test_first_crossing_and_tolerance(self):
        ladder = np.array(D.BETA_LADDER)
        slopes = np.array([0.5, 2.0])  # rho(k) = 1 + slope k: linear in the knob

        def evaluate(a, k):
            return 1.0 + slopes[a] * k, [("payload", float(x)) for x in k]

        ladder_rho = evaluate(np.repeat([0, 1], len(ladder)), np.tile(ladder, 2))[0].reshape(2, -1)
        out = D.match_knobs(ladder, 1.0, ladder_rho, [1.3, 2.0, 4.0, 100.0], evaluate)
        for (a, t), match in out.items():
            target = [1.3, 2.0, 4.0, 100.0][t]
            if t == 3:  # 1 + 2 x 30 < 100: no crossing, never extrapolated
                self.assertFalse(match["reachable"])
                self.assertIsNone(match["knob"])
                continue
            self.assertTrue(match["reachable"] and match["converged"] and match["monotone"])
            self.assertLessEqual(abs(match["rho"] / target - 1), D.TOL)
            self.assertAlmostEqual(match["knob"], (target - 1) / slopes[a], delta=D.TOL * target / slopes[a])
            self.assertEqual(match["payload"], ("payload", match["knob"]))

    def test_non_monotone_ladder_takes_the_first_crossing(self):
        ladder = np.array([1.0, 2.0, 3.0, 4.0])
        values = {1.0: 3.0, 2.0: 1.5, 3.0: 5.0, 4.0: 6.0}  # crosses 2 between 0 and 1, again between 2 and 3

        def evaluate(a, k):
            return np.interp(k, [0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 1.5, 5.0, 6.0]), None

        out = D.match_knobs(ladder, 1.0, np.array([[values[k] for k in ladder]]), [2.0], evaluate)[0, 0]
        self.assertFalse(out["monotone"])
        self.assertTrue(out["converged"])
        self.assertLess(out["knob"], 1.0)  # the bracket [0, 1], searched in the knob itself
        self.assertAlmostEqual(out["knob"], 0.5, delta=1e-3)

    def test_targets_must_exceed_the_host(self):
        with self.assertRaises(ValueError):
            D.match_knobs([1.0], 2.0, np.array([[3.0]]), [1.5], lambda a, k: (k, None))

    def test_cloud_band(self):
        nulls = [dict(reachable=True, x=v) for v in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]
        lo, hi = np.percentile([r["x"] for r in nulls], [5, 95])
        inside = D.cloud(dict(reachable=True, x=0.45), nulls, "x")
        self.assertEqual((inside["p05"], inside["p95"], inside["outside"]), (lo, hi, False))
        above = D.cloud(dict(reachable=True, x=0.79), nulls, "x")
        self.assertEqual((above["outside"], above["side"]), (True, "above"))
        few = D.cloud(dict(reachable=True, x=0.5), nulls[:1] + [dict(reachable=False)], "x")
        self.assertIsNone(few["outside"])


class DataQualityTests(unittest.TestCase):
    def test_raw_episodes_follow_convert(self):
        from types import SimpleNamespace  # noqa: F401  (runtime.td3_bc builds its own namespace)

        import runtime.td3_bc as T

        g = np.random.default_rng(2)
        n = 230
        terminals, timeouts = np.zeros(n, bool), np.zeros(n, bool)
        timeouts[49::50] = True
        terminals[[12, 77, 99, 160]] = True  # 99: both flags on one row
        raw = dict(observations=g.normal(size=(n, 3)).astype(np.float32),
                   actions=g.uniform(-1, 1, (n, 2)).astype(np.float32),
                   rewards=g.normal(size=n).astype(np.float32), terminals=terminals, timeouts=timeouts)
        _, rows, ids = T._convert(raw, 1000)
        episode = D.raw_episodes(terminals, timeouts)
        np.testing.assert_array_equal(episode[rows], ids)
        _, returns, normalized = D.episode_normalized_returns(raw["rewards"], terminals, timeouts, -2.0, 8.0)
        brute, start = [], 0
        for i in range(n):
            if terminals[i] or timeouts[i] or i == n - 1:
                brute.append(float(np.sum(raw["rewards"][start:i + 1], dtype=np.float64)))
                start = i + 1
        np.testing.assert_allclose(returns, brute, rtol=0, atol=1e-9)  # the timeout row's reward counts
        np.testing.assert_allclose(normalized, 100 * (np.array(brute) + 2.0) / 10.0, rtol=0, atol=1e-9)

    def test_expert_half_and_shares(self):
        terminals, timeouts = np.zeros(40, bool), np.zeros(40, bool)
        timeouts[9::10] = True
        rewards = np.r_[np.zeros(20), np.ones(20)]
        episode, _, normalized = D.episode_normalized_returns(rewards, terminals, timeouts, 0.0, 10.0)
        rows_raw = np.array([0, 5, 15, 21, 30, 39])
        is_expert, record = D.expert_half(40, episode, normalized, rows_raw)
        np.testing.assert_array_equal(is_expert, [False, False, False, True, True, True])
        self.assertEqual((record["expert_half"], record["cut_is_episode_boundary"]), ("second", True))
        M = np.array([0.5, 1.0, 2.0, 0.25, 1.5, 0.75])
        shares = [D.share(beta * M, is_expert) for beta in (0.1, 1.0, 30.0)]  # P1's extra weight beta M
        for s in shares:
            self.assertAlmostEqual(s, M[3:].sum() / M.sum(), places=12)
        flipped, record = D.expert_half(40, episode, normalized[::-1].copy(), rows_raw)
        self.assertEqual(record["expert_half"], "first")
        np.testing.assert_array_equal(flipped, ~is_expert)


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = synthetic_engine(n=60, seed=4)
        cls.payload, cls.arrays = D.analyse(cls.engine, log=lambda *_: None)

    def test_every_quantity_is_reported_and_checks_pass(self):
        p = self.payload
        self.assertTrue(all(v is True for k, v in p["checks"].items() if not k.endswith("difference")), p["checks"])
        self.assertEqual(set(p["placements"]), set(D.PLACEMENTS))
        for name, placement in p["placements"].items():
            self.assertEqual(set(placement["levels"]), {"1.3", "2", "4"})
            for level in placement["levels"].values():
                arms = level["arms"]
                expected = 1 + D.N_SHUF + D.N_STRAT + (0 if name == "P3" else 1)
                self.assertEqual(len(arms), expected)
                for record in arms.values():
                    if record["reachable"]:
                        self.assertLessEqual(abs(record["rho"] / level["rho_target"] - 1), D.TOL)
                        self.assertAlmostEqual(record["rho_check"] / record["rho"], 1.0, delta=1e-9)
                        for key in ("cos_dg_p0", "m_eff", "tau", "dg_norm"):
                            self.assertIsNotNone(record[key])
                        self.assertEqual("cos_dg_const_trust" in record, name.startswith("P2"))
                summary = level["summary"]
                self.assertEqual(summary["shuf_cos"]["metric"], "cos_dg_p0")
                self.assertIn("outside", summary["strat_cos"])
        self.assertIn("natural_c1", p["placements"]["P3"])
        for key in ("cos_dg_p0", "m_eff", "tau", "m_equivalent", "dose"):
            self.assertIn(key, p["hook"])
        self.assertEqual(set(p["leverage"]["spearman"]), {"SIG_L", "SIG_pi", "dose"})
        self.assertEqual(set(p["t_lin"]), {"L", "pi"})
        json.dumps(D._json(p), allow_nan=False)  # strict JSON

    def test_ladders_are_validated_and_counted(self):
        with self.assertRaises(ValueError):
            D.analyse(self.engine, c_ladder=(1.0, 0.5), log=lambda *_: None)
        for placement in self.payload["placements"].values():
            for level in placement["levels"].values():
                count = sum(not r["monotone"] for r in level["arms"].values())
                self.assertEqual(level["summary"]["nonmonotone_ladders"], count)

    def test_p1_direction_does_not_depend_on_the_level(self):
        levels = self.payload["placements"]["P1L"]["levels"]
        cosines = [levels[k]["arms"]["SIG"]["cos_dg_p0"] for k in levels if levels[k]["arms"]["SIG"]["reachable"]]
        self.assertGreater(len(cosines), 1)
        self.assertLess(np.ptp(cosines), 1e-9)  # Delta g_P1 = beta g_BC[M]
        self.assertAlmostEqual(float(np.mean(self.arrays["sig_L"])), 1.0, places=12)  # extra weight has mean beta

    def test_p0_and_hook(self):
        p = self.payload
        for key, record in p["p0"].items():
            self.assertAlmostEqual(record["m_eff"], record["m"], delta=1e-9 * record["m"])
            self.assertAlmostEqual(record["rho"] / p["base"]["rho_host"], record["m"], delta=1e-9 * record["m"])
            self.assertAlmostEqual(record["cos_dg_p0"], 1.0, places=12)
        dose = self.arrays["dose"]
        eta = self.engine.rows["eta_logged"]
        lam_eta = np.float64(np.float32(1.3)) * np.maximum(eta, 1e-6)
        np.testing.assert_allclose(dose, 1.0 + 0.5 * lam_eta / (1.0 + lam_eta), rtol=1e-6)  # BCA's closed form
        self.assertAlmostEqual(p["hook"]["dose"]["mean"], float(dose.mean()), places=12)


class PoolRunTests(unittest.TestCase):
    """The whole run on a synthetic short-train pool: restore, gate, Level-1, data quality, outputs."""

    @classmethod
    def setUpClass(cls):
        from experiments.signal import frozen_signals as S
        from experiments.wbcp import freeze_scores as F
        from experiments.wbcp.test_freeze_scores import synthetic_row, write_raw

        cls.tmp = Path(tempfile.mkdtemp(prefix="d4rl-level1-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, ignore_errors=True)
        write_raw(cls.tmp / "synthetic.hdf5")
        F.freeze_short_train(synthetic_row(cls.tmp), cls.tmp / "pool", updates=20, train_fraction=0.5, block=10,
                             score_batch=10**6, data_dir=cls.tmp)
        S.fit(cls.tmp / "pool", cls.tmp / "fit", signals=("min", "q1"), updates=30, block=30, data_dir=cls.tmp)
        designs = [dict(name="iid", label="iid", n=40, per_episode=None, spacing="random"),
                   dict(name="reservation", label="resv", n=12, per_episode=3, spacing="reservation")]
        S.evaluate(cls.tmp / "fit", banks=3, seed=11, designs=designs)
        cls.result = D.run(cls.tmp / "fit", cls.tmp / "out", rows=10**6, data_dir=cls.tmp, log=lambda *_: None)

    def test_outputs_and_gate(self):
        r = self.result
        self.assertEqual(r["schema"], D.SCHEMA)
        self.assertTrue(r["gate"]["passed"])
        self.assertEqual(r["settings"]["rows"], r["settings"]["training_rows"])
        saved = json.loads((self.tmp / "out" / "level1.json").read_text(encoding="utf8"))
        self.assertEqual(saved["level1"]["checks"], D._json(r["level1"]["checks"]))
        self.assertTrue((self.tmp / "out" / "level1.md").read_text(encoding="utf8").startswith("# Step 4B-0"))
        with np.load(self.tmp / "out" / "rows.npz") as rows:
            self.assertEqual(len(rows["training_row"]), r["settings"]["rows"])
            for name in ("sig_L", "sig_pi", "leverage_bc", "leverage_q", "leverage_pen", "dose",
                         "episode_normalized_return"):
                self.assertEqual(rows[name].shape, rows["training_row"].shape)
        evaluation = json.loads(next((self.tmp / "fit").glob("evaluation_*.json")).read_text(encoding="utf8"))
        self.assertEqual(r["settings"]["threshold_value"], evaluation["population"]["lambda_star"]["q1"])
        with self.assertRaises(FileExistsError):
            D.run(self.tmp / "fit", self.tmp / "out", rows=10, data_dir=self.tmp)

    def test_episode_returns_on_the_rows(self):
        import h5py

        with h5py.File(self.tmp / "synthetic.hdf5", "r") as f:
            rewards, terminals, timeouts = (np.asarray(f[k][()]) for k in ("rewards", "terminals", "timeouts"))
        q = self.result["data_quality"]
        lo, hi = q["reference"]["reference_min"], q["reference"]["reference_max"]
        with np.load(self.tmp / "out" / "rows.npz") as rows:
            for raw_row, value in zip(rows["raw_row"], rows["episode_normalized_return"]):
                start = raw_row
                while start > 0 and not (terminals[start - 1] or timeouts[start - 1]):
                    start -= 1
                end = raw_row
                while not (terminals[end] or timeouts[end]) and end < len(rewards) - 1:
                    end += 1
                expected = 100 * (np.sum(rewards[start:end + 1], dtype=np.float64) - lo) / (hi - lo)
                self.assertAlmostEqual(value, expected, places=9)
        self.assertIsNone(q["medium_expert"])  # a hopper-medium row
        self.assertIn("pearson", q["corr_with_episode_normalized_return"]["SIG_L"])


if __name__ == "__main__":
    unittest.main()
