"""Checks of the Gapped LQ training harness (experiments/signal/glq_train.py) and its blind matcher (glq_match.py).

What is checked, on a small seed of the world (40 / 20 / 20 episodes) and tiny runs (hundreds of updates):
- the harness's critic and actor steps are td3_bc_update bit for bit with every hook off (harness_update at an actor
  update and at a critic-only update; actor_step against td3_bc_update's actor with and without a bc_multiplier; the
  HOST chunk program against a scan of td3_bc.make_train_step; the multiplier hook against td3_bc_update with
  bc_multiplier = table[rows]);
- the frozen critic stays frozen in actor-only runs (Tier A's loop equals td3_bc_update followed by restoring the
  critic, its target and the target actor, as lq_harness.actor_loop; the critic's outputs are bitwise unchanged after a
  vmapped Tier A run); FQE's actor stays at the probe while its critic follows td3_bc_update;
- EXACT's Q^pi: the float32 tables against gapped_lq's closed form (World.q, World.grad_a_q), exact_rows against the
  same to 1e-12 and NaN in an unstable context, and EXACT's actor step equals td3_bc's actor step on a critic whose
  heads are Q^pi;
- the exact-error rows: grad_a Q1 against jax.grad, e_A and OR-D identities, all errors zero (up to a constant level)
  for a critic equal to Q^pi + 3;
- oracle-weight identities (OR-wrong, OR-thin, ANTI, SHUF multisets, EVEN, rank-normal OR-err, OR-D, the capped forms,
  knob 0 = NONE) and the two oracle code paths agreeing;
- glq_match: strength and second moments equal placement_harness's, its matcher reproduces placement_harness.match_knob,
  the knob-0 bracket, unreachable targets, and that it imports only the standard library and NumPy;
- Tier A's blind record holds no return, its hashes, and the scores' paired differences; the Level-1 screen's rho
  matching and first-order gains; grad J against a directional difference;
- a tiny co-training run decreases the critic loss and writes loadable snapshots and strict JSON; BCA refuses to run
  (its program is traced, never executed).
JAX_PLATFORMS=cpu python -m unittest experiments.signal.test_glq_train
"""
import ast
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

import jax
import jax.numpy as jnp
import optax

import algorithms.td3_bc as BASE
from experiments.signal import gapped_lq as G
from experiments.signal import glq_match as M
from experiments.signal import glq_train as T
from experiments.signal import placement_harness as PH

WORLD = G.World()
DATA = G.generate(WORLD, 0, sizes=(40, 20, 20))
SET = T.TrainSettings(updates=400, chunk=100, exact_refresh=100)
TRAIN = G.transitions(DATA["train"])
N = len(TRAIN.obs)


def leaves_equal(test, x, y, msg=""):
    lx, ly = jax.tree_util.tree_leaves(x), jax.tree_util.tree_leaves(y)
    test.assertEqual(len(lx), len(ly))
    for a, b in zip(lx, ly):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), msg)


def start_state(K=None, rep=0):
    """An AgentTrainState at the behaviour-clone gains (or K), td3_bc.initialize's critic."""
    args = T.make_args(SET, rep)
    kind = T.actor_kind("context_linear")
    K = T.bc_gains(DATA["train"])[0] if K is None else K
    rng, state, _ = T.initial_state(args, kind, kind.params(K), SET.lr)
    return args, kind, (kind, T._CRITIC), rng, state


def batch_rows(k, size=256):
    return np.random.default_rng([5, k]).integers(N, size=size)


def batch(rows):
    return jax.tree_util.tree_map(lambda x: x[rows], TRAIN)


class UpdateTests(unittest.TestCase):
    def test_harness_update_hooks_off_is_td3_bc_update_bitwise(self):
        """Critic step, actor step and both soft target updates equal td3_bc_update at it even (actor update) and odd
        (critic only), compiled on both sides."""
        args, kind, models, rng, state = start_state()
        ours = jax.jit(lambda s, b, it, k: T.harness_update(args, models, s, b, it, k, T.MAX_ACTION))
        ref = jax.jit(lambda s, b, it, k: BASE.td3_bc_update(args, kind.apply, T._CRITIC.apply, s, b, it, k,
                                                            T.MAX_ACTION))
        s1 = s2 = state
        for t in range(1, 5):
            b, key = batch(batch_rows(t)), jax.random.PRNGKey(t)
            s1, m1 = ours(s1, b, jnp.int32(t), key)
            s2, m2 = ref(s2, b, jnp.int32(t), key)
            leaves_equal(self, s1, s2, f"update {t}")
            for name in ("critic_loss", "actor_loss", "q_mean", "lambda", "bc_loss"):
                np.testing.assert_array_equal(np.asarray(m1[name]), np.asarray(m2[name]), name)
        self.assertFalse(np.array_equal(np.asarray(s1.actor.params["K"]), np.asarray(state.actor.params["K"])))

    def test_actor_step_is_td3_bc_actor_bitwise(self):
        """actor_step (Tier A's step) equals td3_bc_update's new actor, unweighted and with a bc_multiplier."""
        args, kind, models, rng, state = start_state()
        w = jnp.asarray(1.0 + np.random.default_rng(1).random(256), jnp.float32)
        b, key = batch(batch_rows(9)), jax.random.PRNGKey(3)
        for mult in (None, w):
            ours = jax.jit(lambda a, c, bt: T.actor_step(args, models, a, c, bt, mult))(state.actor,
                                                                                      state.critic.params, b)[0]
            # td3_bc_update's actor step reads the critic after its own critic step: give it a frozen critic
            zero = optax.set_to_zero()
            frozen = state._replace(critic=state.critic.replace(tx=zero, opt_state=zero.init(state.critic.params)))
            ref = jax.jit(lambda s, bt: BASE.td3_bc_update(args, kind.apply, T._CRITIC.apply, s, bt, 0, key,
                                                           T.MAX_ACTION, bc_multiplier=mult))(frozen, b)[0]
            leaves_equal(self, ours, ref.actor, f"multiplier {mult is not None}")

    def test_host_chunk_is_make_train_step(self):
        """The HOST program (no table) equals a scan of td3_bc.make_train_step from the same carry, bit for bit."""
        args, kind, models, rng, state = start_state()
        runner = (rng, state, jnp.int32(0))
        ours = T.chunk_program(args, models, "HOST", 30, False, False, "exact")(runner, TRAIN, None, None)[0]
        step = BASE.make_train_step(args, kind.apply, T._CRITIC.apply, TRAIN, T.MAX_ACTION)
        ref = jax.jit(lambda r: jax.lax.scan(step, r, None, length=30))(runner)[0]
        leaves_equal(self, ours, ref)

    def test_multiplier_hook_is_bc_multiplier(self):
        """HOST with a table passes table[rows] as td3_bc_update's bc_multiplier."""
        args, kind, models, rng, state = start_state()
        table = jnp.asarray(1.0 + (np.asarray(DATA["train"]["context"]) == G.T_WRONG) * 2.0, jnp.float32)
        runner = (rng, state, jnp.int32(0))
        ours = T.chunk_program(args, models, "HOST", 20, True, False, "exact")(runner, TRAIN, table, None)[0]

        def step(r, _):
            key, st, it = r
            it = it + 1
            key, kb, kn = jax.random.split(key, 3)
            idx = jax.random.randint(kb, (args.batch_size,), 0, N)
            st, _ = BASE.td3_bc_update(args, kind.apply, T._CRITIC.apply, st, batch(idx), it, kn, T.MAX_ACTION,
                                       bc_multiplier=table[idx])
            return (key, st, it), None

        ref = jax.jit(lambda r: jax.lax.scan(step, r, None, length=20))(runner)[0]
        leaves_equal(self, ours, ref)

    def test_fqe_freezes_the_actor_and_follows_td3_bc_critic(self):
        args, kind, models, rng, state = start_state(K=T.probe_gains(WORLD, SET))
        runner = (rng, state, jnp.int32(0))
        ours = T.chunk_program(args, models, "FQE", 20, False, False, "exact")(runner, TRAIN, None, None)[0]
        np.testing.assert_array_equal(np.asarray(ours[1].actor.params["K"]), np.asarray(state.actor.params["K"]))
        leaves_equal(self, ours[1].actor_target, state.actor_target)

        def step(r, _):
            key, st, it = r
            it = it + 1
            key, kb, kn = jax.random.split(key, 3)
            idx = jax.random.randint(kb, (args.batch_size,), 0, N)
            new, _ = BASE.td3_bc_update(args, kind.apply, T._CRITIC.apply, st, batch(idx), it, kn, T.MAX_ACTION)
            return (key, new._replace(actor=st.actor, actor_target=st.actor_target), it), None

        ref = jax.jit(lambda r: jax.lax.scan(step, r, None, length=20))(runner)[0]
        leaves_equal(self, ours[1].critic, ref[1].critic)
        leaves_equal(self, ours[1].critic_target, ref[1].critic_target)
        self.assertFalse(np.array_equal(np.asarray(jax.tree_util.tree_leaves(ours[1].critic.params)[0]),
                                        np.asarray(jax.tree_util.tree_leaves(state.critic.params)[0])))


class FrozenCriticTests(unittest.TestCase):
    def test_actor_only_loop_is_td3_bc_update_with_restored_critic(self):
        """Tier A's loop equals lq_harness.actor_loop's pattern: td3_bc_update, then the critic, its target and the
        target actor restored (bitwise for the actor), while td3_bc_update alone moves the critic."""
        args, kind, models, rng, state = start_state()
        rows = jnp.asarray(np.stack([batch_rows(100 + t) for t in range(12)]), jnp.int32)
        table = jnp.ones(N, jnp.float32) * 1.5
        ours = jax.jit(lambda a, c: T._tier_a_loop(args, models, c, a, TRAIN, rows, table))(state.actor,
                                                                                             state.critic.params)[0]

        zero = optax.set_to_zero()  # lq_harness.Harness.native's frozen critic optimizer
        frozen = state._replace(critic=state.critic.replace(tx=zero, opt_state=zero.init(state.critic.params)))

        def body(st, xs):
            t, r = xs
            new, _ = BASE.td3_bc_update(args, kind.apply, T._CRITIC.apply, st, batch(r), args.policy_freq * t,
                                        jax.random.PRNGKey(0), T.MAX_ACTION, bc_multiplier=table[r])
            return new._replace(actor_target=frozen.actor_target, critic=frozen.critic,
                                critic_target=frozen.critic_target), None

        ref, _ = jax.jit(lambda s: jax.lax.scan(body, s, (jnp.arange(12), rows)))(frozen)
        leaves_equal(self, ours.params, ref.actor.params)
        # with the critic's own Adam, td3_bc_update moves the critic, which Tier A never does
        moved, _ = BASE.td3_bc_update(args, kind.apply, T._CRITIC.apply, state, batch(rows[0]), 0,
                                      jax.random.PRNGKey(0), T.MAX_ACTION)
        first = lambda tree: np.asarray(jax.tree_util.tree_leaves(tree)[1])
        self.assertFalse(np.array_equal(first(moved.critic.params), first(state.critic.params)))

    def test_vmapped_tier_a_leaves_the_critic_unchanged(self):
        args, kind, models, rng, state = start_state()
        TA = T.TierA(WORLD, DATA, state.critic.params, SET, 0, steps=40, refresh=20, n_shuf=1)
        obs, act = TRAIN.obs[:512], TRAIN.action[:512]
        before, digest = np.asarray(T._CRITIC.apply(TA.critic_params, obs, act)), T.tree_sha256(TA.critic_params)
        gains, _ = TA.run([("NONE", 0.0), ("OR-err", 3.0)])
        self.assertEqual(T.tree_sha256(TA.critic_params), digest)
        np.testing.assert_array_equal(np.asarray(T._CRITIC.apply(TA.critic_params, obs, act)), before)
        self.assertFalse(np.array_equal(gains[0], TA.K_start))


class ExactTests(unittest.TestCase):
    def test_tables_match_the_closed_form(self):
        """EXACT's float32 Q^pi and its action gradient against World.q and World.grad_a_q."""
        K = T.bc_gains(DATA["train"])[0]
        (H, h), stable = T.exact_tables(WORLD, K)
        self.assertTrue(stable.all())
        ev = DATA["eval"]
        rng = np.random.default_rng(2)
        s, c = ev["state"][:2000], ev["context"][:2000]
        a = WORLD.policy(K, s, c) + 0.3 * rng.standard_normal((len(s), G.DA))
        obs = np.concatenate([s, np.eye(G.NC)[c]], 1)
        q32 = np.asarray(T.exact_q_jax((H, h), jnp.asarray(obs, jnp.float32), jnp.asarray(a, jnp.float32)))
        q64 = WORLD.q(K, s, a, c)
        np.testing.assert_allclose(q32, q64, rtol=2e-5, atol=1e-3)
        g32 = np.asarray(jax.grad(lambda x: T.exact_q_jax((H, h), jnp.asarray(obs, jnp.float32), x).sum())(
            jnp.asarray(a, jnp.float32)))
        np.testing.assert_allclose(g32, WORLD.grad_a_q(K, s, a, c), rtol=1e-4, atol=1e-3)
        q, g, st = T.exact_rows(WORLD, K, s, a, c)
        np.testing.assert_allclose(q, q64, rtol=1e-12)
        np.testing.assert_allclose(g, WORLD.grad_a_q(K, s, a, c), rtol=1e-12, atol=1e-12)

    def test_unstable_context_is_nan_and_keeps_previous_tables(self):
        K = WORLD.k_star.copy()
        K[G.T_WRONG] = -20 * WORLD.k_star[G.T_RIGHT]  # far outside the stable set
        self.assertGreaterEqual(WORLD.radius(K)[G.T_WRONG], 1)
        ev = DATA["eval"]
        q, g, stable = T.exact_rows(WORLD, K, ev["state"], ev["action"], ev["context"])
        self.assertFalse(stable[G.T_WRONG])
        self.assertTrue(np.all(np.isnan(q[ev["context"] == G.T_WRONG])))
        self.assertTrue(np.all(np.isfinite(q[ev["context"] != G.T_WRONG])))
        prev, _ = T.exact_tables(WORLD, WORLD.k_star)
        tables, st = T.exact_tables(WORLD, K, prev)
        np.testing.assert_array_equal(np.asarray(tables[0][G.T_WRONG]), np.asarray(prev[0][G.T_WRONG]))
        self.assertEqual(int((~st).sum()), 1)

    def test_exact_actor_step_is_td3_bc_on_a_q_pi_critic(self):
        """With lam 'exact', EXACT's actor step is td3_bc's actor step on a critic whose heads are both Q^pi."""
        args, kind, models, rng, state = start_state()
        tables, _ = T.exact_tables(WORLD, kind.gains(state.actor.params))

        class QpiCritic:
            @staticmethod
            def apply(params, obs, action):
                q = T.exact_q_jax(tables, obs, action)
                return jnp.stack([q, q], -1)

        b = batch(batch_rows(4))
        ours = jax.jit(lambda a: T.actor_step(args, models, a, state.critic.params, b, None, tables, "exact"))(
            state.actor)[0]
        ref = jax.jit(lambda a: T.actor_step(args, (kind, QpiCritic), a, None, b))(state.actor)[0]
        leaves_equal(self, ours.params, ref.params)

    def test_exact_tables_carry_the_td3_level(self):
        """A-review regression: with smoothing, EXACT's tables hold Q^pi + c_c (H unchanged), so EXACT's lambda equals
        the lambda of a perfect TD3 critic (one equal to Q^pi + c_c), not the 9-10% larger one of Q^pi; cotrain uses
        them unless exact_level='deterministic'."""
        args, kind, models, rng, state = start_state()
        K = kind.gains(state.actor.params)
        (H0, h0), _ = T.exact_tables(WORLD, K)
        (H1, h1), _ = T.exact_tables(WORLD, K, smoothing=(0.2, 0.5))
        np.testing.assert_array_equal(np.asarray(H0), np.asarray(H1))
        np.testing.assert_allclose(np.asarray(h1, np.float64) - np.asarray(h0, np.float64), WORLD.td3_level(K),
                                   rtol=1e-5)
        b = batch(batch_rows(5))
        lam = lambda tables: float(jax.jit(lambda a: T.actor_step(args, models, a, state.critic.params, b, None,
                                                                  tables, "exact"))(state.actor)[1][2])
        pi = np.asarray(kind.apply(state.actor.params, b.obs), np.float64)
        s, c = np.asarray(b.obs, np.float64)[:, :G.DS], np.argmax(np.asarray(b.obs)[:, G.DS:], 1)
        q_td3 = WORLD.q(K, s, pi, c) + WORLD.td3_level(K)[c]
        self.assertAlmostEqual(lam((H1, h1)) * np.abs(q_td3).mean() / args.alpha, 1.0, places=4)
        self.assertGreater(lam((H0, h0)) / lam((H1, h1)), 1.05)
        calls = []
        real = T.exact_tables

        def spy(world, K, previous=None, smoothing=None):
            calls.append(smoothing)
            return real(world, K, previous, smoothing)

        with mock.patch.object(T, "exact_tables", spy):
            T.cotrain(WORLD, DATA, replace(SET, updates=100, snapshots=(1.0,)), "EXACT", 0, log=None,
                      diagnostics=False)
            T.cotrain(WORLD, DATA, replace(SET, updates=100, snapshots=(1.0,), exact_level="deterministic"), "EXACT",
                      0, log=None, diagnostics=False)
        self.assertEqual(calls, [(0.2, 0.5), None])
        with self.assertRaises(ValueError):
            replace(SET, exact_level="other")


class DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args, cls.kind, cls.models, cls.rng, cls.state = start_state()
        cls.K = T.bc_gains(DATA["train"])[0]

    def test_critic_rows_gradient_is_jax_grad_of_q1(self):
        ev = DATA["eval"]
        obs, a = np.asarray(ev["obs"][:300], np.float32), np.asarray(ev["action"][:300], np.float32)
        q, g = T.critic_rows(self.state.critic.params, obs, a, chunk=128)
        ref = jax.grad(lambda x: T._CRITIC.apply(self.state.critic.params, jnp.asarray(obs), x)[..., 0].sum())(
            jnp.asarray(a))
        np.testing.assert_allclose(g, np.asarray(ref), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(q[:128], np.asarray(T._CRITIC.apply(self.state.critic.params, obs[:128], a[:128])),
                                   rtol=1e-6, atol=1e-6)

    def test_error_identities(self):
        err = T.error_rows(WORLD, self.state.critic.params, self.K, DATA["eval"])
        np.testing.assert_allclose(err["e_A"], err["e1_logged"] - err["e1_pi"], rtol=0, atol=0)
        np.testing.assert_allclose(err["or_d"], np.maximum(0.0, -err["e_A"]), rtol=0, atol=1e-9)
        np.testing.assert_allclose(err["or_d"], np.maximum(0.0, (err["q1_pi"] - err["q1_logged"]) -
                                                            (err["qpi_pi"] - err["qpi_logged"])), rtol=0, atol=1e-9)
        np.testing.assert_allclose(err["E_pi"], np.linalg.norm(err["grad_q1_pi"] - err["grad_qpi_pi"], axis=1))
        summary = T.summarize_errors(err)
        shares = [summary["per_context"][c]["share_sum_E2"] for c in G.CONTEXTS]
        self.assertAlmostEqual(sum(shares), 1.0, places=12)
        self.assertAlmostEqual(summary["pooled"]["thin_share_sum_E2"],
                               shares[G.T_WRONG] + shares[G.T_RIGHT], places=12)
        json.dumps(G._clean(summary), allow_nan=False)

    def test_errors_vanish_for_a_critic_equal_to_q_pi_plus_a_constant(self):
        K = self.K

        def fake(params, obs, action, chunk=8192):
            s, c = obs[:, :G.DS].astype(np.float64), np.argmax(obs[:, G.DS:], 1)
            q = WORLD.q(K, s, action.astype(np.float64), c) + 3.0
            return np.stack([q, q + 1.0], 1), WORLD.grad_a_q(K, s, action.astype(np.float64), c)

        with mock.patch.object(T, "critic_rows", fake):
            err = T.error_rows(WORLD, None, K, DATA["eval"])
        np.testing.assert_allclose(err["e1_pi"], 3.0, atol=1e-6)
        np.testing.assert_allclose(err["E_pi"], 0.0, atol=1e-9)
        np.testing.assert_allclose(err["or_d"], 0.0, atol=1e-6)
        np.testing.assert_allclose(err["e_A"], 0.0, atol=1e-6)

    def test_a_perfect_td3_critic_passes_the_level_error_reference(self):
        """A-review regression: a critic equal to TD3's fixed point Q^pi + c_c has e1_td3 = 0 and a rich median
        |e1_td3| / |Q^pi| of 0, while the deterministic reference reads > 0.05 (R2's mark) for the same perfect
        critic."""
        K = self.K
        level = WORLD.td3_level(K)

        def fake(params, obs, action, chunk=8192):
            s, c = obs[:, :G.DS].astype(np.float64), np.argmax(obs[:, G.DS:], 1)
            q = WORLD.q(K, s, action.astype(np.float64), c) + level[c]
            return np.stack([q, q], 1), WORLD.grad_a_q(K, s, action.astype(np.float64), c)

        with mock.patch.object(T, "critic_rows", fake):
            err = T.error_rows(WORLD, None, K, DATA["eval"])
        np.testing.assert_allclose(err["e1_td3_pi"], 0.0, atol=1e-5)
        np.testing.assert_allclose(err["e1_pi"], level[err["context"]], atol=1e-5)
        np.testing.assert_array_equal(err["td3_level"], level)
        pooled = T.summarize_errors(err)["pooled"]
        self.assertLess(pooled["rich_main_median_rel_e1_td3_pi"], 1e-6)
        self.assertGreater(pooled["rich_main_median_rel_e1_pi"], 0.05)
        with mock.patch.object(T, "critic_rows", fake):  # another smoothing gives another reference
            other = T.error_rows(WORLD, None, K, DATA["eval"], smoothing=(0.1, 0.5))
        self.assertGreater(np.abs(other["e1_td3_pi"]).min(), 1.0)

    def test_noisy_code_is_refused(self):
        noisy = G.generate(WORLD, 0, sizes=(20, 10, 10), code_sd=0.1)
        with self.assertRaises(NotImplementedError):
            T.error_rows(WORLD, self.state.critic.params, self.K, noisy["eval"])


class OracleWeightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args, kind, models, rng, state = start_state()
        cls.TA = T.TierA(WORLD, DATA, state.critic.params, SET, 0, steps=40, refresh=20)
        cls.oracle = cls.TA.oracle_variables(cls.TA.K_start)
        cls.ctx = np.asarray(DATA["train"]["context"])

    def test_context_weights(self):
        TA, ctx, b = self.TA, self.ctx, 3.0
        wrong, thin, anti = (TA.table(f, b) for f in ("OR-wrong", "OR-thin", "ANTI"))
        np.testing.assert_array_equal(wrong, np.where(ctx == G.T_WRONG, 1 + b, 1.0))
        np.testing.assert_array_equal(thin, np.where(np.isin(ctx, G.THIN), 1 + b, 1.0))
        np.testing.assert_array_equal(thin + anti, np.full(len(ctx), 2 + b))
        for k in range(T.N_SHUF):
            shuf = TA.table(f"SHUF{k}", b)
            np.testing.assert_array_equal(np.sort(shuf), np.sort(thin))
            self.assertFalse(np.array_equal(shuf, thin))
        np.testing.assert_array_equal(TA.table("EVEN", 4.0), np.full(len(ctx), 4.0))
        np.testing.assert_array_equal(TA.table("NONE", 0.0), np.ones(len(ctx)))
        cap = TA.table("CAP-wrong", 0.49)
        self.assertTrue(np.all((cap >= 1) & (cap < 1.5)))

    def test_oracle_weights(self):
        TA, oracle = self.TA, self.oracle
        x = TA.variable("err", oracle)
        np.testing.assert_array_equal(np.sort(x), PH.multiset(len(x)))
        self.assertAlmostEqual(float(x.mean()), 1.0, places=12)
        order = np.argsort(oracle["E"], kind="stable")
        self.assertTrue(np.all(np.diff(x[order]) >= 0))
        self.assertAlmostEqual(float(TA.variable("ord", oracle).mean()), 1.0, places=12)
        self.assertTrue(np.all(TA.variable("ord", oracle) >= 0))
        cap = TA.table("CAP-err", 10.0, oracle)
        self.assertTrue(np.all((cap >= 1) & (cap < 1.5)))
        for fam in ("OR-wrong", "OR-err", "OR-D", "CAP-err", "SHUF0"):
            np.testing.assert_array_equal(TA.table(fam, 0.0, oracle), np.ones(len(x)))

    def test_oracle_paths_agree(self):
        """Tier A's E and OR-D on training rows equal error_rows' on the same rows."""
        err = T.error_rows(WORLD, self.TA.critic_params, self.TA.K_start, DATA["train"])
        np.testing.assert_allclose(self.oracle["E"], err["E_pi"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(self.oracle["ord"], err["or_d"], rtol=1e-12, atol=1e-9)


class MatchTests(unittest.TestCase):
    def test_strength_and_moments_equal_placement_harness(self):
        rng = np.random.default_rng(3)
        s, c = rng.standard_normal((500, G.DS)), rng.integers(G.NC, size=500)
        np.testing.assert_array_equal(M.second_moments(s, c, G.NC), PH.second_moment(s, c, G.NC))
        Sig = M.second_moments(s, c, G.NC)
        K1, K0 = rng.standard_normal((7, G.NC, G.DA, G.DS)), rng.standard_normal((G.NC, G.DA, G.DS))
        np.testing.assert_array_equal(M.strength(K1, K0, Sig), PH.strength(K1, K0, Sig))

    def test_matcher_reproduces_match_knob(self):
        ladder = (0.3, 1.0, 3.0, 10.0, 30.0)
        for f, target in ((lambda k: math.log1p(k), 1.7), (lambda k: 0.2 * k ** 0.5, 0.55), (lambda k: k, 3.0)):
            ref = PH.match_knob(ladder, f, target, tol=0.01, max_bisect=5, interpolate=True)
            states, _ = M.match_parallel({"F": (ladder, [f(k) for k in ladder])}, {"t": target},
                                         lambda props: [f(k) for _, k in props], tol=0.01, max_rounds=6)
            st = states[("F", "t")]
            self.assertAlmostEqual(st["knob"], ref["knob"], places=12)
            self.assertEqual(st["converged"], ref["converged"])

    def test_zero_knob_bracket_unreachable_and_monotone(self):
        f = dict(F=lambda k: 2.0 * k, G=lambda k: 0.5 * k)
        states, _ = M.match_parallel({"F": ((0.0, 0.3, 1.0), (0.0, 0.6, 2.0)), "G": ((0.0, 1.0), (0.0, 0.5))},
                                     {"t": 0.3}, lambda props: [f[name](k) for name, k in props])
        self.assertTrue(states[("F", "t")]["converged"])
        self.assertAlmostEqual(states[("F", "t")]["knob"], 0.15, places=12)  # linear interpolation from knob 0
        self.assertAlmostEqual(states[("G", "t")]["knob"], 0.6, places=12)
        st = M.start((0.0, 1.0, 2.0), (0.0, 0.1, 0.05), 0.5)
        self.assertFalse(st["reachable"])
        self.assertFalse(st["monotone"])
        self.assertIsNone(M.public(st)["relative_error"])

    def test_glq_match_imports_no_evaluation_code(self):
        tree = ast.parse(Path(M.__file__).read_text())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or "").split(".")[0])
        self.assertEqual(names, {"hashlib", "math", "pathlib", "numpy"})


class TierATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args, kind, models, rng, state = start_state()
        cls.critic = state.critic.params
        fams = T.family_ladders(2)
        fams = {k: fams[k] for k in ("EVEN", "OR-wrong", "OR-err", "SHUF0", "SHUF1", "CAP-err")}
        cls.blind, cls.TA = T.tier_a_train(WORLD, DATA, cls.critic, SET, 0, steps=60, refresh=30, n_shuf=2,
                                           families=fams, max_rounds=2, log=None)
        cls.score = T.tier_a_score(WORLD, cls.blind)

    def test_blind_record_holds_no_returns_and_its_hashes(self):
        text = json.dumps(G._clean(self.blind), default=G._json_default)
        for word in ('"NS', '"J"', '"J_mix"', '"J_inf"', '"score"'):
            self.assertNotIn(word, text)
        self.assertEqual(self.blind["glq_match_sha256"], M.source_sha256())
        rec = {k: v for k, v in self.blind.items() if k != "blind_sha256"}
        digest = __import__("hashlib").sha256(json.dumps(G._clean(rec), sort_keys=True,
                                                         default=G._json_default).encode()).hexdigest()
        self.assertEqual(digest, self.blind["blind_sha256"])
        self.assertEqual(self.score["blind_sha256"], self.blind["blind_sha256"])

    def test_targets_matching_and_scores(self):
        ladder = self.blind["ladder"]
        none = next(r for r in ladder if r["family"] == "NONE")
        self.assertEqual(none["S"], 0.0)
        for m in T.TARGET_M:
            even = next(r for r in ladder if r["family"] == "EVEN" and r["knob"] == m)
            self.assertEqual(self.blind["targets"][f"m={m}"], even["S"])
        for name, rec in self.blind["matches"].items():
            if rec["K"] is not None and rec["knob"]:
                S = float(M.strength(np.asarray(rec["K"]), np.asarray(none["K"]), self.TA.Sigma_ev))
                self.assertAlmostEqual(S, rec["S"], places=10)
            sc = self.score["matched"][name]
            if sc.get("NS_mix") is not None:
                ref = self.score["even_at_target"][rec["target_tag"]]
                self.assertAlmostEqual(sc["NS_minus_EVEN"], sc["NS_mix"] - ref, places=12)
        self.assertEqual(len(self.score["null_shuf_pairs"]["m=2.0"]), 1)
        json.dumps(G._clean(self.score), allow_nan=False)

    def test_vmapped_arms_match_single_runs(self):
        arms = [("NONE", 0.0), ("OR-wrong", 3.0), ("EVEN", 2.0)]
        together, _ = self.TA.run(arms)
        for i, arm in enumerate(arms):
            alone, _ = self.TA.run([arm])
            np.testing.assert_allclose(together[i], alone[0], rtol=0, atol=1e-6)
        self.assertLess(np.abs(self.TA.none_path() - together[0]).max(), 1e-4)


class Level1Tests(unittest.TestCase):
    def test_screen_matches_rho_and_first_order_gains(self):
        args, kind, models, rng, state = start_state()
        K = T.bc_gains(DATA["train"])[0] + 0.05 * WORLD.k_star
        out = T.level1_screen(WORLD, DATA, state.critic.params, K, SET, n_shuf=1)
        for m in T.TARGET_M:
            level = out["levels"][f"m={m}"]
            for fam, rec in level.items():
                if rec.get("knob") is not None:
                    self.assertAlmostEqual(rec["rho"] / (m * out["rho_host"]), 1.0, places=5)
            self.assertAlmostEqual(level["EVEN"]["minus_EVEN"], 0.0, places=12)

    def test_grad_j_is_the_directional_derivative(self):
        K = WORLD.k_behaviour
        D = np.random.default_rng(4).standard_normal(K.shape)
        h = 1e-5
        J = lambda X: float(WORLD.mix @ WORLD.j_finite(X))
        fd = (J(K + h * D) - J(K - h * D)) / (2 * h)
        self.assertAlmostEqual(float(np.sum(T.grad_j(WORLD, K) * D)) / fd, 1.0, places=5)


class ActorClassTests(unittest.TestCase):
    def test_actor_classes_are_linear_in_s_and_match_their_gains(self):
        """Every class's pi(obs) = -K(code) s with K from gains() at the one-hot codes (g7: K_theta(code))."""
        rng = np.random.default_rng(6)
        s = rng.standard_normal((50, G.DS))
        c = rng.integers(G.NC, size=50)
        obs = jnp.asarray(np.concatenate([s, np.eye(G.NC)[c]], 1), jnp.float32)
        obs2 = jnp.asarray(np.concatenate([2 * s, np.eye(G.NC)[c]], 1), jnp.float32)
        for name in T.ACTORS:
            kind = T.actor_kind(name)
            params = kind.init(jax.random.PRNGKey(1)) if name == "gain_scheduled" else kind.params(
                rng.standard_normal((G.NC, G.DA, G.DS)))
            pi = np.asarray(kind.apply(params, obs), np.float64)
            K = kind.gains(params)
            np.testing.assert_allclose(pi, -np.einsum("nij,nj->ni", K[c], s), rtol=1e-5, atol=1e-5, err_msg=name)
            np.testing.assert_allclose(np.asarray(kind.apply(params, obs2)), 2 * pi, rtol=1e-5, atol=1e-5)

    def test_gain_scheduled_fit_and_a_short_run(self):
        kind = T.actor_kind("gain_scheduled")
        K_bc = T.bc_gains(DATA["train"])[0]
        params, loss = kind.fit(kind.init(jax.random.PRNGKey(2)), K_bc, steps=1500)
        self.assertLess(loss, 1e-4 * float(np.mean(K_bc ** 2)) + 1e-6)
        rec = T.cotrain(WORLD, DATA, replace(SET, actor="gain_scheduled", actor_init="bc", updates=100,
                                             snapshots=(1.0,)), "HOST", 0, log=None)
        self.assertEqual(rec["actor_class"], "gain_scheduled")
        self.assertTrue(np.isfinite(rec["chunks"][-1]["critic_loss"]))
        self.assertIn("100", rec["diagnostics"])

    def test_global_linear_starts_at_the_pooled_clone(self):
        TA = T.TierA(WORLD, DATA, start_state()[4].critic.params, replace(SET, actor="global_linear"), 0, steps=20,
                     refresh=20, n_shuf=1)
        np.testing.assert_allclose(TA.K_start, np.repeat(TA.K_bc_pooled[None], G.NC, 0), rtol=1e-6, atol=1e-7)


class CotrainTests(unittest.TestCase):
    def test_tiny_host_run(self):
        """The critic loss falls, snapshots reload bit for bit, and run.json is strict JSON with its hashes."""
        with tempfile.TemporaryDirectory() as tmp:
            rec = T.cotrain(WORLD, DATA, replace(SET, updates=600), "HOST", 0, out_dir=tmp, log=None)
            losses = [c["critic_loss"] for c in rec["chunks"]]
            self.assertLess(losses[-1], 0.5 * losses[0])
            self.assertEqual([s["update"] for s in rec["snapshots"]], [150, 300, 600])
            last = rec["snapshots"][-1]
            loaded = T.load_snapshot(last["path"], rec["state"], last["sha256"])
            leaves_equal(self, loaded, rec["state"])
            payload = json.loads(Path(tmp, "run.json").read_text())
            self.assertEqual(payload["mode"], "HOST")
            self.assertIn("600", payload["diagnostics"])
            self.assertTrue(Path(tmp, "errors_100.npz").exists())

    def test_exact_and_fqe_runs(self):
        rec = T.cotrain(WORLD, DATA, replace(SET, updates=200, snapshots=(1.0,)), "EXACT", 0, log=None)
        self.assertEqual(rec["exact_unstable_context_refreshes"], 0)
        self.assertTrue(all(np.isfinite(c["critic_loss"]) for c in rec["chunks"]))
        host = T.cotrain(WORLD, DATA, replace(SET, updates=200, snapshots=(1.0,)), "HOST", 0, log=None,
                         diagnostics=False)
        # same critic initialisation and batches, different actor objective
        self.assertFalse(np.array_equal(np.asarray(rec["state"].actor.params["K"]),
                                        np.asarray(host["state"].actor.params["K"])))
        fqe = T.cotrain(WORLD, DATA, replace(SET, updates=200, snapshots=(1.0,)), "FQE", 0, log=None)
        np.testing.assert_array_equal(np.asarray(fqe["state"].actor.params["K"], np.float64).astype(np.float32),
                                      T.probe_gains(WORLD, SET).astype(np.float32))

    def test_bca_is_switched_off(self):
        with self.assertRaises(RuntimeError):
            T.cotrain(WORLD, DATA, SET, "BCA", 0)
        lowered = T._bca_cotrain(WORLD, DATA, SET, 0, refresh_every=2, execute=False)  # traced, never run
        self.assertIsNotNone(lowered)


if __name__ == "__main__":
    unittest.main()
