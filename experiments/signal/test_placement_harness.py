"""Checks of the step-4 placement harness: every [derived] item of runs/wbcp_signal/step4/expectations_DRAFT.md that the
core can check, and STEP4_DESIGN.md section 12 item 10.

Covered: Part 0 items 1 (tilt alignment, T identical across levels), 4 (fixed-point cancellation identities) and the
closed-form part of 6 (step4_settings); Part 2 items 1-7 (zero knob, bitwise td3_bc_update and step-3 bca path, same
multiset, Adam and fixed-point scale equivalence, constant penalty, P3-OR, common states and NUIS = SIG in clean) and
the matching procedure of item 8; Part 3 item 1 (J_total, the context critic against Monte Carlo and the Bellman
identity, context-local fixed points, LSTD); plus the lambda-lock, skipping the critic update, and PlacementCritic =
QuadraticCritic bitwise without a cone.

Blindness (design section 5): small settings with a seed that is not the real configuration's, and no J of a step-4
arm. J appears only where a check needs it, on the clean case (NONE weighted against NONE through the None path) or on
toy context gains (J_total = mean_c J(K_c)); each such test says so.
JAX_PLATFORMS=cpu python -m unittest experiments.signal.test_placement_harness
"""
import functools
import math
import unittest
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

import algorithms.td3_bc as BASE
import algorithms.td3_bc_bca as P
from experiments.signal import lq_harness as LQ
from experiments.signal import placement_harness as PH
from experiments.signal import step4_settings as SETTINGS

SMALL = PH.Settings(seed=7, train_episodes=20, cal_episodes=40, cal_rows_per_episode=2, eval_episodes=20,
                    episode_length=20, fit_steps=40, actor_steps=15, batch_size=64)


@functools.lru_cache(maxsize=None)
def harness():
    return PH.PlacementHarness(SMALL)


@functools.lru_cache(maxsize=None)
def common(quality, contexts=None):
    return PH.data_common(harness(), quality, 0, contexts)


@functools.lru_cache(maxsize=None)
def fitted(case, kappa=None, a_star=None, quality="expert"):
    """A fitted X-CS cell (signal and NUIS), shared by the tests of this process."""
    h = harness()
    data = common(quality)
    params, info = PH.critic_params_x(h, data, case, kappa, a_star)
    cell = PH.make_cell(h, data, params, info=info)
    stream = PH.streams(h, 0, 0, len(data["train"]["state"]))
    train, cal = PH.transitions(data["train"], case == "noisy_reward"), PH.transitions(data["cal"], False)
    cell.sig, _ = PH.fit_signal(h, cell.native, train, cal, stream)
    cell.nuis = cell.sig if case == "clean" else fitted("clean", quality=quality)[0].sig
    return cell, stream, train


def run_one(spec, cell, stream, train, run, ctx=None, ties=None):
    """placement_loop for one run, jitted without vmap."""
    h = harness()
    ctx = PH.loop_context(cell) if ctx is None else ctx
    if ties is None:
        ties = jnp.asarray(np.tile(np.arange(SMALL.batch_size), (SMALL.actor_steps, 1)), jnp.int32)
    loop = jax.jit(functools.partial(PH.placement_loop, h.args, cell.models, spec))
    run = {k: jnp.asarray(v, jnp.int32 if k == "assign" else jnp.float32) for k, v in run.items()}
    return loop(cell.native, train, stream.actor_rows, stream.actor_keys, ties, ctx, run)


RUN = dict(a=1.0, b=0.0, beta=0.0, c=0.0, assign=0)


def K_of(out):
    return np.asarray(out[0].actor.params["K"])


def rel(x, y):
    return float(np.max(np.abs(np.asarray(x) - np.asarray(y))) / np.max(np.abs(np.asarray(y))))


def fp_matrix(terms, u, v, lam0, rows=None, n=None):
    sel = slice(None) if rows is None else rows
    count = terms["Tq"][sel].shape[0] if n is None else n
    u, v = np.broadcast_to(u, (terms["Tq"].shape[0],))[sel], np.broadcast_to(v, (terms["Tq"].shape[0],))[sel]
    m = terms["ds"] * terms["da"]
    return ((-lam0 * (u @ terms["Tq"][sel]) + (v @ terms["Tb"][sel]) / terms["da"]) / count).reshape(m, m)


def vec(K):
    return np.asarray(K).T.reshape(-1)  # column-major


# ---------------------------------------------------------------------------------------------------------
# 1. Data


class DataTests(unittest.TestCase):
    def test_common_states_noise_and_shocks_are_identical_across_levels(self):
        """Part 2 item 7: in X-CS only K_b differs between levels."""
        h = harness()
        ref = common("expert")
        for q in PH.QUALITIES:
            d = common(q)
            for split in ("train", "cal", "eval"):
                for name in ("state", "noise", "shock", "reward_noise", "episode"):
                    np.testing.assert_array_equal(d[split][name], ref[split][name], f"{q} {split} {name}")
            tr = d["train"]
            mean = -tr["state"] @ h.k_opt.T * tr["gain"][:, None]
            np.testing.assert_allclose(tr["action"], mean + SMALL.behavior_noise * tr["noise"], rtol=0, atol=1e-15)
            np.testing.assert_allclose(tr["next_state"], tr["state"] @ h.system.A.T + tr["action"] @ h.system.B.T
                                       + tr["shock"], rtol=0, atol=1e-15)
            np.testing.assert_array_equal(tr["reward"], LQ.reward(h.system, tr["state"], tr["action"]))
            np.testing.assert_array_equal(d["meta"]["K_bar"], PH.QUALITY_GAIN[q] * h.k_opt)
            np.testing.assert_allclose(d["meta"]["K_bc"], PH.ls_gain(tr["state"], tr["action"]))
        mixed = common("mixed")["train"]["gain"]
        self.assertEqual(set(np.unique(mixed)), {0.0, 1.0})
        np.testing.assert_array_equal(common("poor")["train"]["action"],
                                      SMALL.behavior_noise * common("poor")["train"]["noise"])

    def test_states_come_from_K0_rollouts_and_the_bank_takes_every_calibration_episode(self):
        h = harness()
        d = common("medium")
        L = SMALL.episode_length
        rows = LQ.simulate(h.system, h.K0, SMALL.behavior_noise, 80, np.random.default_rng([7, 0, PH.TAG_CS, 0]))
        np.testing.assert_array_equal(d["train"]["state"], rows["obs"][:SMALL.train_episodes * L])
        self.assertEqual(len(d["cal"]["state"]), SMALL.cal_episodes * SMALL.cal_rows_per_episode)
        self.assertEqual(len(np.unique(d["cal"]["episode"])), SMALL.cal_episodes)

    def test_contexts_add_a_persistent_one_hot_to_the_same_states(self):
        d, ref = common("expert", 4), common("expert")
        for split in ("train", "eval"):
            np.testing.assert_array_equal(d[split]["state"], ref[split]["state"])
            np.testing.assert_array_equal(d[split]["obs"][:, :3], ref[split]["state"])
            np.testing.assert_array_equal(d[split]["obs"][:, 3:], np.eye(4)[d[split]["context"]])
            np.testing.assert_array_equal(d[split]["next_obs"][:, 3:], d[split]["obs"][:, 3:])
            for e in np.unique(d[split]["episode"]):
                self.assertEqual(len(np.unique(d[split]["context"][d[split]["episode"] == e])), 1)
        self.assertEqual(d["meta"]["K_bc_context"].shape, (4, 2, 3))

    def test_episodic_levels(self):
        h = harness()
        L = SMALL.episode_length
        for q in ("expert", "poor"):
            d = PH.data_episodic(h, q, 0)
            rows = LQ.simulate(h.system, h.behavior_gain(q), SMALL.behavior_noise, 80,
                               np.random.default_rng([7, 0, PH.TAG_EP, PH.QUALITIES.index(q), 0]))
            np.testing.assert_array_equal(d["train"]["obs"], rows["obs"][:SMALL.train_episodes * L])
        mixed = PH.data_episodic(h, "mixed", 0)["train"]
        for e in np.unique(mixed["episode"]):
            self.assertEqual(len(np.unique(mixed["gain"][mixed["episode"] == e])), 1)
        gains = np.tile(h.k_opt, (5, 1, 1))
        a = PH.simulate_gains(h.system, gains, 0.2, np.random.default_rng(3))
        b = LQ.simulate(h.system, h.k_opt, 0.2, 5, np.random.default_rng(3))
        for name in ("obs", "action", "next_obs"):
            np.testing.assert_allclose(a[name], b[name], rtol=1e-12, atol=1e-12)

    def test_streams_are_step3s_batches_and_keys(self):
        """Block R: tag = BEHAVIORS.index(behavior) under step 3's seed gives step 3's rows and keys."""
        h = PH.PlacementHarness(replace(SMALL, seed=20261001))
        cfg, rep, b, n = h.cfg, 1, 1, 400
        st = PH.streams(h, rep, b, n)
        key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(cfg.seed), rep), b)
        fold = functools.partial(jax.random.fold_in, jax.random.fold_in(key, 1))
        fit_keys = jax.vmap(fold)(jnp.arange(cfg.fit_steps))
        actor_rows = np.random.default_rng([cfg.seed, rep, b, 3]).integers(n, size=(cfg.actor_steps, cfg.batch_size))
        np.testing.assert_array_equal(np.asarray(st.fit_keys), np.asarray(fit_keys))
        np.testing.assert_array_equal(np.asarray(st.actor_rows), actor_rows)
        np.testing.assert_array_equal(np.asarray(st.init_key), np.asarray(jax.random.fold_in(key, LQ.KEY_INIT)))


# ---------------------------------------------------------------------------------------------------------
# 2. Critics


class CriticTests(unittest.TestCase):
    def test_placement_critic_is_quadratic_critic_bitwise_without_a_cone(self):
        """Section 12 item 10: eager, jitted and inside td3_bc_update."""
        h = harness()
        d = common("expert")
        s, a = (jnp.asarray(d["train"][k], jnp.float32) for k in ("state", "action"))
        forms = LQ.random_forms(5, np.random.default_rng(1))
        cases = [(c, LQ.case_critic(h.system, c, h.K0, h.k_opt, 1.0, forms, 0.5)) for c in LQ.CASES]
        cases.append(("tilt", PH.critic_params_x(h, d, "tilt", a_star=0.0)[0]))
        for case, params in cases:
            p32 = PH.to_jax32(params)
            np.testing.assert_array_equal(np.asarray(PH.PlacementCritic.apply(p32, s, a)),
                                          np.asarray(LQ.QuadraticCritic.apply(p32, s, a)), case)
            np.testing.assert_array_equal(np.asarray(jax.jit(PH.PlacementCritic.apply)(p32, s, a)),
                                          np.asarray(jax.jit(LQ.QuadraticCritic.apply)(p32, s, a)), case)
        native = h.lq.native(h.K0, cases[-1][1])
        batch = PH.transitions(d["train"], False)
        outs = [BASE.td3_bc_update(h.args, LQ.LinearActor.apply, critic, native, batch, 0, jax.random.PRNGKey(2),
                                   LQ.MAX_ACTION) for critic in (PH.PlacementCritic.apply, LQ.QuadraticCritic.apply)]
        for x, y in zip(jax.tree_util.tree_leaves(outs[0]), jax.tree_util.tree_leaves(outs[1])):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))

    def test_new_heads_are_exact_q_plus_error(self):
        h = harness()
        d = common("medium")
        s = d["train"]["state"]
        a = d["train"]["action"] + 0.3 * np.random.default_rng(2).standard_normal(d["train"]["action"].shape)
        clean = LQ.case_critic(h.system, "clean", h.K0, h.k_opt)
        for case, kw in (("tilt", dict(a_star=-0.5)), ("cone", dict(kappa=2.0)), ("q1_optimistic", dict(kappa=1.0))):
            params = PH.critic_params_x(h, d, case, **kw)[0]
            p32, f32 = PH.to_jax32(params), (lambda x: jnp.asarray(x, jnp.float32))
            heads = np.asarray(PH.PlacementCritic.apply(p32, f32(s), f32(a)))
            expected = LQ.qform(params["H"], params["h"], s, a)[:, None] + PH.head_error_x(params, s, a)
            np.testing.assert_allclose(heads, expected, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(np.asarray(PH.PlacementCritic.error(p32, f32(s), f32(a))),
                                       PH.head_error_x(params, s, a)[:, 0], rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(np.asarray(PH.PlacementCritic.exact(p32, f32(s), f32(a))),
                                       LQ.qform(clean["H"], clean["h"], s, a), rtol=1e-5, atol=1e-5)
            if case == "cone":  # e1 >= 0: the target's min is Q^pi bitwise
                self.assertTrue(np.all(PH.head_error_x(params, s, a)[:, 0] >= 0))
                clean_heads = np.asarray(PH.PlacementCritic.apply(PH.to_jax32(clean), f32(s), f32(a)))
                np.testing.assert_array_equal(heads.min(1), clean_heads[:, 0])
                np.testing.assert_array_equal(heads[:, 1], clean_heads[:, 1])
        cone = PH.critic_params_x(h, d, "cone", kappa=1.0)
        self.assertAlmostEqual(PH.cone_weight(s, PH.CONE_V, cone[1]["c0"]).mean(), PH.CONE_MEAN, places=9)

    def test_tilt_reaches_its_alignment_and_T_is_identical_across_levels(self):
        """Part 0 item 1: alignment_K = a* within 1e-4 at every level; T bitwise identical across X-CS levels."""
        h = harness()
        for a_star in PH.TILT_ALIGNMENTS:
            Ts = []
            for q in PH.QUALITIES:
                d = common(q)
                params, info = PH.critic_params_x(h, d, "tilt", a_star=a_star)
                s = d["train"]["state"]
                align = PH.alignment_K(PH.critic_grads(PH.PlacementCritic, params, s, PH.policy_np(h.K0, s)), s)
                self.assertLess(abs(align - a_star), 1e-4, (q, a_star, align))
                np.testing.assert_allclose(np.linalg.eigvalsh(params["H"][3:, 3:] + params["W"][0][3:, 3:]),
                                           np.linalg.eigvalsh(params["H"][3:, 3:]))  # H_aa unchanged
                Ts.append(info["T"])
            for T in Ts[1:]:
                np.testing.assert_array_equal(T, Ts[0])
            x = PH.tilt_c(a_star) / math.sqrt(2)
            self.assertAlmostEqual((1 - x) / math.sqrt((1 - x) ** 2 + x ** 2), a_star, places=12)
        self.assertAlmostEqual(PH.tilt_c(0.0), math.sqrt(2), places=12)

    def test_independent_error_forms_are_shared_across_levels(self):
        """X-CS holds the critic fixed across levels: the same forms, with the scale set per level on its own rows."""
        h = harness()
        (pe, ie), (pp, ip) = (PH.critic_params_x(h, common(q), "independent_errors") for q in ("expert", "poor"))
        self.assertEqual(ie["independent_redraws"], ip["independent_redraws"])
        np.testing.assert_allclose(pe["W"] / ie["independent_scale"], pp["W"] / ip["independent_scale"], rtol=1e-12)
        self.assertNotEqual(ie["independent_scale"], ip["independent_scale"])
        for info in (ie, ip):
            self.assertAlmostEqual(info["independent_ratio"], SMALL.independent_ratio, delta=1e-3)

    def test_tilt_local_aligns_context_zero(self):
        h = harness()
        d = common("expert", 4)
        params, _ = PH.critic_params_c(h, d, "tilt-local", a_star=-0.5)
        tr = d["train"]
        rows = tr["context"] == 0
        K0 = np.tile(h.K0, (4, 1, 1))
        grads = PH.critic_grads(PH.ContextQuadraticCritic, params, tr["obs"][rows], PH.policy_np(K0, tr["obs"][rows]))
        self.assertLess(abs(PH.alignment_K(grads, tr["state"][rows]) + 0.5), 1e-4)
        other = PH.critic_grads(PH.ContextQuadraticCritic, params, tr["obs"][~rows], PH.policy_np(K0, tr["obs"][~rows]))
        np.testing.assert_array_equal(other["error"], 0.0)

    def test_context_critic_heads_and_min_identity(self):
        h = harness()
        d = common("poor", 4)
        tr = d["train"]
        K = np.tile(h.K0, (4, 1, 1)) + 0.05 * np.random.default_rng(4).standard_normal((4, 2, 3))
        params = PH.context_case_critic(h.system, "q1opt-local", K, h.behavior_gain("poor"), kappa=1.0)
        f32 = lambda x: jnp.asarray(x, jnp.float32)
        heads = np.asarray(PH.ContextQuadraticCritic.apply(PH.to_jax32(params), f32(tr["obs"]), f32(tr["action"])))
        for c in range(4):
            rows = tr["context"] == c
            Hc, hc, _ = LQ.exact_q(h.system, K[c])
            expect = LQ.qform(Hc, hc, tr["state"][rows], tr["action"][rows])
            np.testing.assert_allclose(heads[rows, 1], expect, rtol=1e-5, atol=1e-5)
            err = PH.context_head_error(params, tr["obs"][rows], tr["action"][rows])
            np.testing.assert_allclose(heads[rows, 0], expect + err[:, 0], rtol=1e-5, atol=1e-5)
            if c:
                np.testing.assert_array_equal(heads[rows, 0], heads[rows, 1])
        np.testing.assert_array_equal(heads.min(1), heads[:, 1])  # e1 >= 0
        pi = np.asarray(PH.ContextLinearActor.apply({"K": f32(K)}, f32(tr["obs"])))
        np.testing.assert_allclose(pi, PH.policy_np(K, tr["obs"]), rtol=1e-5, atol=1e-6)


def rollout_returns(system, K, s, a, rng, horizon=300):
    total, discount = LQ.reward(system, s, a), 1.0
    for _ in range(horizon):
        s = LQ.step(system, s, a, rng)
        a = -s @ K.T
        discount *= system.gamma
        total = total + discount * LQ.reward(system, s, a)
    return total


class ContextValueTests(unittest.TestCase):
    """Part 3 item 1: J_total = mean_c J(K_c); the context critic against Monte Carlo and the Bellman identity.

    J is evaluated here on toy per-context gains (K_0 plus seeded perturbations), not on any step-4 arm.
    """

    @classmethod
    def setUpClass(cls):
        cls.h = harness()
        cls.K = np.tile(cls.h.K0, (4, 1, 1)) + 0.1 * np.random.default_rng(5).standard_normal((4, 2, 3))
        cls.params = PH.context_case_critic(cls.h.system, "clean", cls.K, cls.h.k_opt)

    def test_context_critic_matches_monte_carlo(self):
        system, rng = self.h.system, np.random.default_rng(6)
        n = 20000
        for c in (0, 3):
            for _ in range(2):
                s, a = rng.normal(scale=0.4, size=3), rng.normal(scale=0.3, size=2)
                obs = np.concatenate([s, np.eye(4)[c]])[None]
                q = float(PH.exact_np(self.params, obs, a[None])[0])
                g = rollout_returns(system, self.K[c], np.tile(s, (n, 1)), np.tile(a, (n, 1)), rng)
                self.assertLess(abs(g.mean() - q), 4 * g.std() / math.sqrt(n) + 1e-6, (c, g.mean(), q))

    def test_context_critic_satisfies_the_bellman_identity(self):
        """Q((s, c), a) = r + gamma E Q((s', c), pi_c(s')) exactly (E[s'^T X s'] = m^T X m + tr(X Sigma_eps))."""
        system, rng = self.h.system, np.random.default_rng(7)
        s, a = rng.normal(scale=0.4, size=(6, 3)), rng.normal(scale=0.3, size=(6, 2))
        for c in range(4):
            Hc, hc = self.params["H"][c], self.params["h"][c]
            Pi = np.vstack([np.eye(3), -self.K[c]])
            X = Pi.T @ Hc @ Pi
            m = s @ system.A.T + a @ system.B.T
            bootstrap = np.einsum("ni,ij,nj->n", m, X, m) + np.trace(X @ system.noise) + hc
            rhs = LQ.reward(system, s, a) + system.gamma * bootstrap
            obs = np.concatenate([s, np.tile(np.eye(4)[c], (6, 1))], 1)
            np.testing.assert_allclose(PH.exact_np(self.params, obs, a), rhs, rtol=1e-10)

    def test_J_total_is_the_mean_over_contexts(self):
        """The context MDP (context uniform per episode) has J = mean_c J(K_c) = E_{s0, c} Q((s0, c), pi(s0))."""
        system, rng, n = self.h.system, np.random.default_rng(8), 60000
        J_total = np.mean([LQ.value(system, self.K[c]) for c in range(4)])
        s0 = rng.standard_normal((n, 3)) @ np.linalg.cholesky(system.init).T
        c = rng.integers(4, size=n)
        g = np.zeros(n)
        for k in range(4):
            rows = c == k
            g[rows] = rollout_returns(system, self.K[k], s0[rows], -s0[rows] @ self.K[k].T, rng)
        self.assertLess(abs(g.mean() - J_total), 4 * g.std() / math.sqrt(n))
        closed = np.mean([np.trace(LQ.exact_q(system, self.K[k])[2] @ system.init) + LQ.exact_q(system, self.K[k])[1]
                          for k in range(4)])
        self.assertAlmostEqual(closed, J_total, places=10)


# ---------------------------------------------------------------------------------------------------------
# 3. Weights


class WeightTests(unittest.TestCase):
    def test_multiset_matches_the_design(self):
        """B6 (also step4_settings): s = 0.83 at n = 10,000: CV 0.914, range 0.090-5.714, Kish 0.545."""
        m = PH.multiset(10000)
        self.assertAlmostEqual(m.mean(), 1.0, places=12)
        self.assertTrue(np.all(np.diff(m) >= 0))
        self.assertAlmostEqual(m.std() / m.mean(), 0.914, places=3)
        self.assertAlmostEqual(m.sum() ** 2 / (len(m) * np.sum(m ** 2)), 0.545, places=3)
        self.assertAlmostEqual(m.min(), 0.090, places=3)
        self.assertAlmostEqual(m.max(), 5.714, places=3)

    def test_global_ranks_break_ties_by_the_seeded_permutation(self):
        x = np.array([3.0, 1.0, 1.0, 1.0, 2.0])
        tie = np.array([0, 2, 0, 1, 4])
        w, rank = PH.rank_normal(x, tie_perm=tie)
        np.testing.assert_array_equal(rank, [4, 2, 0, 1, 3])
        np.testing.assert_array_equal(w, PH.multiset(5)[rank])

    def test_batch_ranks(self):
        rng = np.random.default_rng(9)
        x = jnp.asarray(np.round(rng.random(64), 1), jnp.float32)  # many ties
        tie = jnp.asarray(rng.permutation(64), jnp.int32)
        M = np.asarray(PH.multiset(64), np.float32)
        w = np.asarray(PH.batch_rank_normal(x, tie=tie))
        np.testing.assert_array_equal(np.sort(w), np.sort(M))
        order = np.lexsort((np.asarray(tie), np.asarray(x)))
        np.testing.assert_array_equal(w[order], M)  # ascending in x, ties by the tie keys, not by row order
        anti = np.asarray(PH.batch_rank_normal(x, tie=tie, reverse=True))
        np.testing.assert_array_equal(anti[order], M[::-1])

    def test_or_raw_reproduces_step3s_oracle(self):
        rng = np.random.default_rng(10)
        dose, error, perm = 1 + 0.5 * rng.random(300), rng.random(300) ** 2, rng.permutation(300)
        arms, _ = LQ.control_doses(dose, error, perm)
        np.testing.assert_array_equal(PH.or_raw(dose, error, perm, 1.0)[0], arms["oracle"])
        np.testing.assert_array_equal(PH.or_raw(dose, error, perm, 0.0)[0], np.ones(300, np.float32))
        half = PH.or_raw(dose, error, perm, 0.5)[0].astype(np.float64)
        np.testing.assert_allclose(half, 1 + 0.5 * (arms["oracle"].astype(np.float64) - 1), rtol=1e-6)


class AssignmentTests(unittest.TestCase):
    """Part 2 item 3 (same multiset) and item 7's NUIS = SIG in clean, on fitted cells."""

    @classmethod
    def setUpClass(cls):
        cls.cell, cls.stream, cls.train = fitted("tilt", a_star=-0.5)
        cls.assign = PH.assignments(cls.cell)

    def test_every_assignment_has_bitwise_the_same_multiset(self):
        for fam in PH.FAMILIES:
            for p in PH.PLACEMENTS:
                arms = self.assign["weights"][fam][p]
                self.assertEqual(len(arms), 2 + 16 + 16 + 3)  # SIG, ANTI, SHUF x16, STRAT x16, NUIS, OR1, OR2
                ref = np.sort(arms["SIG"])
                for name, w in arms.items():
                    np.testing.assert_array_equal(np.sort(w), ref, f"{fam} {p} {name}")
                    self.assertEqual(w.dtype, np.float64)
                self.assertFalse(np.array_equal(arms["SIG"], arms["SHUF0"]))

    def test_strat_permutes_within_leverage_deciles(self):
        for p in PH.PLACEMENTS:
            dec, arms = self.assign["deciles"][p], self.assign["weights"]["L"][p]
            for k in range(3):
                for g in range(10):
                    np.testing.assert_array_equal(np.sort(arms[f"STRAT{k}"][dec == g]), np.sort(arms["SIG"][dec == g]))
        lev = self.assign["leverage"]
        tr = self.cell.data["train"]
        np.testing.assert_allclose(lev["P1"], np.linalg.norm(-tr["state"] @ harness().K0.T - tr["action"], axis=1)
                                   * np.linalg.norm(tr["state"], axis=1))

    def test_anti_reverses_and_sig_follows_the_width(self):
        w, x = self.assign["weights"]["L"]["P1"], self.assign["x"]["L"]["SIG"]
        order = np.argsort(x, kind="stable")
        self.assertTrue(np.all(np.diff(w["SIG"][order]) >= 0))
        self.assertTrue(np.all(np.diff(w["ANTI"][order]) <= 0))

    def test_nuis_is_sig_in_clean(self):
        clean, stream, train = fitted("clean")
        again = PH.fit_signal(harness(), clean.native, train, PH.transitions(clean.data["cal"], False), stream)[0]
        for a, b in zip(jax.tree_util.tree_leaves(again), jax.tree_util.tree_leaves(clean.sig)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        A = PH.assignments(clean)
        self.assertFalse(A["has_error"])  # OR arms not applicable in a negative control
        for fam in PH.FAMILIES:
            np.testing.assert_array_equal(A["weights"][fam]["P1"]["NUIS"], A["weights"][fam]["P1"]["SIG"])
            self.assertNotIn("OR1", A["weights"][fam]["P1"])

    def test_pi_family_batches_carry_the_exact_multiset_at_every_step(self):
        """Part 2 item 3, family pi at A500: per step, every arm's batch weights are the batch multiset."""
        h, cell = harness(), self.cell
        M = np.sort(np.asarray(PH.multiset(SMALL.batch_size), np.float32))
        ctx = PH.loop_context(cell)
        cp = cell.native.critic.params
        for p in PH.PLACEMENTS:
            table, tie = PH.pi_tables(cell, self.assign, p, self.stream.actor_rows)
            self.assertEqual(len(table), 13)
            for t in (0, 7, 14):
                rows = self.stream.actor_rows[t]
                obs = self.train.obs[rows]
                pi = cell.models[0].apply(cell.native.actor.params, obs)
                for arm, (kind, reverse, perm) in table.items():
                    x = PH.assignment_at(kind, cell.models, cp, ctx, obs, pi)
                    w = np.asarray(PH.batch_rank_normal(x, tie=jnp.asarray(tie[t]), reverse=reverse)[perm[t]])
                    np.testing.assert_array_equal(np.sort(w), M, f"{p} {arm} step {t}")
        # the pi-family variables at pi_0 agree with the global (float64) ones
        obs = self.train.obs
        pi0 = cell.models[0].apply(cell.native.actor.params, obs)
        np.testing.assert_allclose(np.asarray(PH.assignment_at("or1", cell.models, cp, ctx, obs, pi0)),
                                   self.assign["x"]["pi"]["OR1"], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(np.asarray(PH.assignment_at("or2_p1", cell.models, cp, ctx, obs, pi0)),
                                   self.assign["x"]["OR2"]["P1"], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(np.asarray(PH.assignment_at("or2_p2", cell.models, cp, ctx, obs, pi0)),
                                   self.assign["x"]["OR2"]["P2"], rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(np.asarray(PH.assignment_at("sig", cell.models, cp, ctx, obs, pi0)),
                                   self.assign["x"]["pi"]["SIG"], rtol=1e-5)


# ---------------------------------------------------------------------------------------------------------
# 4. Actor update


class ActorUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell, cls.stream, cls.train = fitted("q1_optimistic", kappa=1.0)

    def test_hooks_off_is_td3_bc_update_bitwise(self):
        """Part 2 item 2: one compiled step against td3_bc_update (eager and jitted), and the whole loop against
        lq_harness.actor_loop. td3_bc_update always compiles its actor update (inside lax.cond), so the comparison is
        between compiled programs; an eager, op-by-op placement_step rounds differently (about 1e-7 relative)."""
        h, cell, st = harness(), self.cell, self.stream
        step = jax.jit(lambda s, b, k: PH.placement_step(h.args, cell.models, s, b, k))
        update = jax.jit(lambda s, b, k: BASE.td3_bc_update(h.args, LQ.LinearActor.apply, LQ.QuadraticCritic.apply,
                                                            s, b, 0, k, LQ.MAX_ACTION))
        for t in range(3):
            batch = jax.tree_util.tree_map(lambda x: x[st.actor_rows[t]], self.train)
            new, m = step(cell.native, batch, st.actor_keys[t])
            for ref, m_ref in (BASE.td3_bc_update(h.args, LQ.LinearActor.apply, LQ.QuadraticCritic.apply, cell.native,
                                                  batch, 0, st.actor_keys[t], LQ.MAX_ACTION),
                               update(cell.native, batch, st.actor_keys[t])):
                for x, y in zip(jax.tree_util.tree_leaves(new.actor), jax.tree_util.tree_leaves(ref.actor)):
                    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
                for name in ("actor_loss", "q_mean", "lambda", "bc_loss"):
                    np.testing.assert_array_equal(np.asarray(m[name]), np.asarray(m_ref[name]), name)
        n = len(self.train.obs)
        k_step3 = np.asarray(h.lq.actor[False](cell.native, self.train, st.actor_rows, st.actor_keys,
                                               jnp.ones(n, jnp.float32))[0].actor.params["K"])
        np.testing.assert_array_equal(K_of(run_one(PH.Spec(bc="none"), cell, st, self.train, RUN)), k_step3)

    def test_hook_dose_is_step3s_bca_path_bitwise(self):
        """Part 2 item 2: P1 with v = the hook's dose (a = 0, b = 1) equals step 3's bca arm bit for bit."""
        h, cell, st = harness(), self.cell, self.stream
        dose = P.bc_readout(cell.models, P.State(cell.native, None, None, cell.sig), self.train, SMALL.blend).dose
        k_step3 = np.asarray(h.lq.actor[True](cell.native, self.train, st.actor_rows, st.actor_keys,
                                              dose)[0].actor.params["K"])
        out = run_one(PH.Spec(diagnostics=False), cell, st, self.train, dict(RUN, a=0.0, b=1.0),
                      ctx=PH.loop_context(cell, w_table=np.asarray(dose)[None]))
        np.testing.assert_array_equal(K_of(out), k_step3)

    def test_skipping_the_critic_update_is_bitwise_safe(self):
        h, cell, st = harness(), self.cell, self.stream
        batch = jax.tree_util.tree_map(lambda x: x[st.actor_rows[0]], self.train)
        ref, _ = BASE.td3_bc_update(h.args, LQ.LinearActor.apply, LQ.QuadraticCritic.apply, cell.native, batch, 0,
                                    st.actor_keys[0], LQ.MAX_ACTION)
        for x, y in zip(jax.tree_util.tree_leaves(ref.critic.params),
                        jax.tree_util.tree_leaves(cell.native.critic.params)):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))

    def test_zero_knob_returns_none_bitwise(self):
        """Part 2 item 1 (A500 part): P1 at beta 0, P0 at m = 1, P2 at beta 0, P3 at c 0 (SIG and OR), OR-raw at t 0
        and the pi family at beta 0 all return NONE through the weighted path bit for bit."""
        cell, st, tr = self.cell, self.stream, self.train
        A = PH.assignments(cell, 2, 2)
        n = len(tr.obs)
        ctx = PH.loop_context(cell, w_table=np.stack([A["weights"]["L"]["P1"]["SIG"], np.ones(n)]))
        none = K_of(run_one(PH.Spec(), cell, st, tr, dict(RUN, assign=1)))
        arms = [(PH.Spec(), dict(RUN, b=0.0)), (PH.Spec(trust=True), dict(RUN, beta=0.0)),
                (PH.Spec(pen="sig"), dict(RUN, c=0.0)), (PH.Spec(pen="or"), dict(RUN, c=0.0))]
        for spec, run in arms:
            np.testing.assert_array_equal(K_of(run_one(spec, cell, st, tr, run, ctx=ctx)), none, str(spec))
        oracle0 = PH.or_raw(np.ones(n), np.abs(PH.head_error_x(cell.params, cell.data["train"]["state"],
                                                               cell.data["train"]["action"])[:, 0]),
                            np.arange(n), 0.0)[0]
        out = run_one(PH.Spec(), cell, st, tr, dict(RUN, a=0.0, b=1.0),
                      ctx=PH.loop_context(cell, w_table=oracle0[None]))
        np.testing.assert_array_equal(K_of(out), none)
        table, tie = PH.pi_tables(cell, A, "P2", st.actor_rows, arms=["SIG"])
        kind, reverse, perm = table["SIG"]
        pi_ctx = PH.loop_context(cell, perm_table=perm[None], reverse=np.array([reverse]))
        for spec in (PH.Spec(family="pi", trust=True), PH.Spec(family="pi")):
            out = run_one(spec, cell, st, tr, RUN, ctx=pi_ctx, ties=jnp.asarray(tie, jnp.int32))
            np.testing.assert_array_equal(K_of(out), none, str(spec))

    def test_none_weighted_matches_the_none_path_in_J(self):
        """Part 2 item 1, second clause: |dJ| <= 1e-5 (up to float32). J on the clean case only."""
        cell, st, tr = fitted("clean")
        weighted = K_of(run_one(PH.Spec(), cell, st, tr, RUN)).astype(np.float64)
        none_path = K_of(run_one(PH.Spec(bc="none"), cell, st, tr, RUN)).astype(np.float64)
        system = harness().system
        self.assertLessEqual(abs(LQ.value(system, weighted) - LQ.value(system, none_path)), 1e-5)

    def test_vmapped_program_matches_single_runs(self):
        h, cell, st, tr = harness(), self.cell, self.stream, self.train
        A = PH.assignments(cell, 2, 2)
        arms = A["weights"]["L"]["P1"]
        ctx = PH.loop_context(cell, w_table=np.stack([arms["SIG"], arms["SHUF0"]]))
        ties = jnp.asarray(np.tile(np.arange(SMALL.batch_size), (SMALL.actor_steps, 1)), jnp.int32)
        runs = PH.runs(4, a=[1, 1, 2, 1], b=[0, 1, 0, 3], assign=[0, 0, 0, 1])
        out = h.program(PH.Spec())(cell.native, tr, st.actor_rows, st.actor_keys, ties, ctx, runs)
        K = np.asarray(out[0].actor.params["K"])
        for i in range(4):
            single = K_of(run_one(PH.Spec(), cell, st, tr, {k: np.asarray(v[i]) for k, v in runs.items()}, ctx=ctx))
            np.testing.assert_allclose(K[i], single, rtol=0, atol=1e-6)
        self.assertEqual(out[2]["step_norm"].shape, (4, SMALL.actor_steps))
        self.assertFalse(np.array_equal(K[0], K[2]))

    def test_lambda_lock(self):
        """lambda = alpha / mean |Q1(s, pi(s))| from the unweighted, unpenalised Q1, whatever the hooks read."""
        h, cell = harness(), self.cell
        batch = jax.tree_util.tree_map(lambda x: x[self.stream.actor_rows[0]], self.train)
        cp = cell.native.critic.params
        w = jnp.asarray(1 + np.random.default_rng(11).random(SMALL.batch_size), jnp.float32)
        ctx = PH.loop_context(cell)
        hooks = [dict(), dict(bc_w=w), dict(q_w=1 / (1 + w)), dict(pen_fn=PH.penalty_fn("sig", cell.models, cp, ctx),
                                                                     pen_c=jnp.float32(3.0)),
                 dict(exact_fn=lambda o, a: cell.models[1].exact(cp, o, a))]
        lams = [np.asarray(PH.placement_step(h.args, cell.models, cell.native, batch, None, **hk)[1]["lambda"])
                for hk in hooks]
        for lam in lams[1:]:
            np.testing.assert_array_equal(lam, lams[0])
        pi = cell.models[0].apply(cell.native.actor.params, batch.obs)
        q1 = cell.models[1].apply(cp, batch.obs, pi)[..., 0]
        np.testing.assert_array_equal(lams[0], np.asarray(h.args.alpha / jnp.abs(q1).mean()))

    def test_adam_scale_equivalence(self):
        """Part 2 item 4 (A500): constant trust 0.5 against P0(2): the gradient doubles exactly, and max|dK| / max|K|
        <= 1e-4 after the loop (Adam is scale-invariant up to eps)."""
        h, cell, st, tr = harness(), self.cell, self.stream, self.train
        batch = jax.tree_util.tree_map(lambda x: x[st.actor_rows[0]], tr)
        B = SMALL.batch_size
        grad = lambda **hk: jax.grad(lambda p: PH._actor_terms(h.args, cell.models, cell.native.critic.params, p,
                                                               batch, pen_fn=None, pen_c=None, exact_fn=None,
                                                               **hk)[0])(cell.native.actor.params)["K"]
        g_trust = grad(bc_w=jnp.ones(B), q_w=jnp.full(B, 0.5))
        g_p0 = grad(bc_w=jnp.full(B, 2.0), q_w=None)
        np.testing.assert_array_equal(np.asarray(g_p0), 2 * np.asarray(g_trust))
        trust = K_of(run_one(PH.Spec(trust=True), cell, st, tr, dict(RUN, beta=1.0)))
        p0 = K_of(run_one(PH.Spec(), cell, st, tr, dict(RUN, a=2.0)))
        self.assertLessEqual(np.max(np.abs(trust - p0)) / np.max(np.abs(p0)), 1e-4)
        self.assertFalse(np.array_equal(p0, K_of(run_one(PH.Spec(), cell, st, tr, RUN))))

    def test_constant_penalty_is_none(self):
        """Part 2 item 5: P3 with W replaced by a constant equals NONE (bitwise at A500, identical at the fixed
        point)."""
        cell, st, tr = self.cell, self.stream, self.train
        none = K_of(run_one(PH.Spec(), cell, st, tr, RUN))
        ctx = PH.loop_context(cell, const=0.7)
        for c in (1.0, 30.0):
            np.testing.assert_array_equal(K_of(run_one(PH.Spec(pen="const"), cell, st, tr, dict(RUN, c=c), ctx=ctx)),
                                          none)
        d = cell.data["train"]
        lam0 = PH.lambda0(cell.params, d["obs"], harness().K0)
        T1 = PH.fp_terms(*PH.fp_rows(cell.params, d["obs"]), d["state"], d["action"])
        k_none = PH.solve_fp(T1, 1.0, 1.0, lam0)["K"]
        res = PH.fixed_point_penalty(cell.models, cell.params, d["obs"], d["action"], "const", ctx, lam0, (1.0, 30.0),
                                     k_none, harness().K0)
        for r in res:
            self.assertTrue(r["success"])
            np.testing.assert_array_equal(r["K"], k_none)

    def test_p3_oracle_reads_the_exact_action_gradient(self):
        """Part 2 item 6, first clause: at K_0, per row grad_a (Q1 - |e1|) = grad_a Q^pi within 1e-4 relative
        (q1_optimistic and the cone, e1 >= 0)."""
        for case, kappa in (("q1_optimistic", 1.0), ("cone", 2.0)):
            h = harness()
            d = common("poor")
            params = PH.critic_params_x(h, d, case, kappa)[0]
            cp = PH.to_jax32(params)
            obs = jnp.asarray(d["train"]["obs"], jnp.float32)
            pi = jnp.asarray(PH.policy_np(h.K0, d["train"]["obs"]), jnp.float32)
            crit = PH.PlacementCritic
            g = jax.grad(lambda a: (crit.apply(cp, obs, a)[..., 0] - jnp.abs(crit.error(cp, obs, a))).sum())(pi)
            ge = jax.grad(lambda a: crit.exact(cp, obs, a).sum())(pi)
            err = np.linalg.norm(np.asarray(g) - np.asarray(ge), axis=1) / np.linalg.norm(np.asarray(ge), axis=1)
            self.assertLess(err.max(), 1e-4, case)

    def test_diagnostics(self):
        cell, st, tr = self.cell, self.stream, self.train
        A = PH.assignments(cell, 1, 1)
        table, tie = PH.pi_tables(cell, A, "P1", st.actor_rows, arms=["SIG"])
        _, reverse, perm = table["SIG"]
        out = run_one(PH.Spec(family="pi"), cell, st, tr, dict(RUN, a=0.0, b=1.0),
                      ctx=PH.loop_context(cell, perm_table=perm[None], reverse=np.array([reverse])),
                      ties=jnp.asarray(tie, jnp.int32))
        diag = out[2]
        M = PH.multiset(SMALL.batch_size)
        kish_M = M.sum() ** 2 / (len(M) * np.sum(M ** 2))  # pi family: every batch carries the exact multiset
        np.testing.assert_allclose(np.asarray(diag["kish_w"]), kish_M, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(diag["kish_bc"]), kish_M, rtol=1e-5)  # a = 0, b = 1: bc_w = w
        np.testing.assert_array_equal(np.asarray(diag["kish_q"]), 1.0)
        np.testing.assert_allclose(np.asarray(diag["bc_q_ratio"]), np.asarray(diag["g_bc_norm"] / diag["g_q_norm"]))
        self.assertTrue(np.all(np.asarray(diag["step_norm"]) > 0))
        self.assertTrue(np.all(np.asarray(diag["adam_snr"]) > 0))
        np.testing.assert_array_equal(np.asarray(diag["g_pen_norm"]), 0.0)
        v_bar = np.asarray(out[1]["K"])
        self.assertEqual(v_bar.shape, (2, 3))
        self.assertTrue(np.all(v_bar > 0))

    def test_pi_eval_hook_is_the_precomputed_weights(self):
        """placement_step's pi_eval (weights re-ranked at the current actor under stop-gradient) equals passing the
        same weights directly, bit for bit; placement_loop computes them in its body."""
        h, cell, st, tr = harness(), self.cell, self.stream, self.train
        batch = jax.tree_util.tree_map(lambda x: x[st.actor_rows[0]], tr)
        ctx = PH.loop_context(cell)
        cp = cell.native.critic.params
        tie = jnp.asarray(np.random.default_rng(20).permutation(SMALL.batch_size), jnp.int32)
        weigh = lambda obs, pi: (None, 1.0 / (1.0 + 2.0 * PH.batch_rank_normal(
            PH.assignment_at("sig", cell.models, cp, ctx, obs, pi), tie=tie)))
        hooked = jax.jit(lambda s: PH.placement_step(h.args, cell.models, s, batch, None, pi_eval=weigh))(cell.native)
        pi = jax.lax.stop_gradient(cell.models[0].apply(cell.native.actor.params, batch.obs))
        direct = jax.jit(lambda s: PH.placement_step(h.args, cell.models, s, batch, None,
                                                     q_w=weigh(batch.obs, pi)[1]))(cell.native)
        np.testing.assert_array_equal(np.asarray(hooked[0].actor.params["K"]), np.asarray(direct[0].actor.params["K"]))


# ---------------------------------------------------------------------------------------------------------
# 5. Fixed point


class FixedPointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = harness()
        cls.d = common("expert")
        cls.tr = cls.d["train"]
        cls.n = len(cls.tr["state"])

    def rows(self, params, critic=None, part="q1"):
        P_, q_ = PH.fp_rows(params, self.tr["obs"] if critic is None else self.ctx_tr["obs"], critic, part)
        tr = self.tr if critic is None else self.ctx_tr
        return PH.fp_terms(P_, q_, tr["state"], tr["action"])

    def test_fp_rows_reproduce_the_heads(self):
        h, tr = self.h, self.tr
        rng = np.random.default_rng(12)
        a = rng.normal(scale=0.5, size=tr["action"].shape)
        for case, kw in (("tilt", dict(a_star=0.0)), ("cone", dict(kappa=2.0)), ("independent_errors", {}),
                         ("shared_bias", dict(kappa=1.0))):
            params = PH.critic_params_x(h, self.d, case, **kw)[0]
            P_, q_ = PH.fp_rows(params, tr["obs"])
            q1 = PH.q1_np(params, tr["obs"], a)
            q0 = PH.q1_np(params, tr["obs"], 0 * a)
            np.testing.assert_allclose(np.einsum("ni,nij,nj->n", a, P_, a) + 2 * np.sum(a * q_, 1) + q0, q1,
                                       rtol=1e-10, atol=1e-10)

    def test_fixed_point_minimises_the_frozen_lambda_loss(self):
        h, tr = self.h, self.tr
        params = PH.critic_params_x(h, self.d, "tilt", a_star=-0.5)[0]
        lam0 = PH.lambda0(params, tr["obs"], h.K0)
        u, v = 1 + 0.3 * np.cos(np.arange(self.n)), 1 + 0.5 * np.sin(np.arange(self.n)) ** 2
        res = PH.fixed_point(*PH.fp_rows(params, tr["obs"]), tr["state"], tr["action"], u, v, lam0)
        self.assertTrue(res["finite"])

        def loss(K):
            pi = PH.policy_np(K, tr["obs"])
            bc = np.mean(v * np.mean((pi - tr["action"]) ** 2, 1))
            return -lam0 * np.mean(u * PH.q1_np(params, tr["obs"], pi)) + bc

        base = loss(res["K"])
        rng = np.random.default_rng(13)
        for _ in range(10):
            dK = 1e-3 * rng.standard_normal((2, 3))
            self.assertGreater(loss(res["K"] + dK), base)
        grad = np.zeros((2, 3))
        for idx in np.ndindex(2, 3):
            e = np.zeros((2, 3))
            e[idx] = 1e-5
            grad[idx] = (loss(res["K"] + e) - loss(res["K"] - e)) / 2e-5
        self.assertLess(np.abs(grad).max(), 1e-8)
        # half the loss Hessian: second difference along a direction equals 2 dK^T A dK
        T = PH.fp_terms(*PH.fp_rows(params, tr["obs"]), tr["state"], tr["action"])
        A = fp_matrix(T, u, v, lam0)
        dK = rng.standard_normal((2, 3))
        second = (loss(res["K"] + 1e-3 * dK) - 2 * base + loss(res["K"] - 1e-3 * dK)) / 1e-6
        self.assertAlmostEqual(second, 2 * vec(dK) @ A @ vec(dK), places=5)
        self.assertAlmostEqual(res["min_eig"], np.linalg.eigvalsh(A).min(), places=12)

    def test_fixed_point_scale_equivalence(self):
        """Part 2 item 4 (fixed point): P2 with omega = 1/(1 + beta) equals P0(1 + beta) within 1e-6, every beta.

        The P2 matrix is omega times P0's, so the two are also nonfinite together (independent_errors on these 400 rows
        is not positive definite at small beta; a tilt never is, since it leaves H_aa unchanged)."""
        h, tr = self.h, self.tr
        betas = np.array(PH.BETA_LADDER)
        for case, kw in (("tilt", dict(a_star=0.0)), ("independent_errors", {})):
            params = PH.critic_params_x(h, self.d, case, **kw)[0]
            lam0 = PH.lambda0(params, tr["obs"], h.K0)
            T = self.rows(params)
            p2 = PH.solve_fp(T, (1 / (1 + betas))[:, None] * np.ones(self.n), np.ones((len(betas), self.n)), lam0)
            p0 = PH.solve_fp(T, np.ones((len(betas), self.n)), (1 + betas)[:, None] * np.ones(self.n), lam0)
            np.testing.assert_array_equal(p2["finite"], p0["finite"])
            self.assertTrue(p0["finite"].all() if case == "tilt" else p0["finite"].any(), case)
            for i in np.flatnonzero(p0["finite"]):
                self.assertLess(rel(p2["K"][i], p0["K"][i]), 1e-6, (case, betas[i]))
            zero = PH.solve_fp(T, 1 / (1 + 0 * np.ones(self.n)), 1 + 0 * np.ones(self.n), lam0)  # knob 0: NONE
            np.testing.assert_array_equal(zero["K"], PH.solve_fp(T, np.ones(self.n), np.ones(self.n), lam0)["K"])

    def test_q1_optimistic_and_cone_cancellation_identities(self):
        """Part 0 item 4 / design section 4 (ii)-(iii): P0(1 + lambda_0 kappa d_a) = EXACT (q1_optimistic) and P1
        with raw weights 1 + lambda_0 kappa d_a c(s_i) = EXACT (cone), within 1e-6, when the error is centred at the BC
        target the rows carry; centred at K-bar_b the gap is exactly lambda_0 kappa A^-1 vec((K_bc,w - K-bar_b)
        Sigma_w)."""
        h, tr = self.h, self.tr
        s = tr["state"]
        c0 = PH.cone_c0(s)
        sig_c = PH.cone_weight(s, PH.CONE_V, c0)
        for case, kappa, w in (("q1_optimistic", 1.0, np.ones(self.n)), ("cone", 2.0, sig_c)):
            centred = PH.ls_gain(s, tr["action"], w)
            for centre, exact_ok in ((centred, True), (None, False)):
                params = PH.critic_params_x(h, self.d, case, kappa, centre=centre)[0]
                lam0 = PH.lambda0(params, tr["obs"], h.K0)
                T1, Te = self.rows(params), self.rows(params, part="exact")
                arm = PH.solve_fp(T1, 1.0, 1 + lam0 * kappa * 2 * w, lam0)
                ex = PH.solve_fp(Te, 1.0, 1.0, lam0)
                self.assertTrue(arm["finite"] and ex["finite"])
                if exact_ok:
                    self.assertLess(rel(arm["K"], ex["K"]), 1e-6, case)
                else:
                    K_bar = self.d["meta"]["K_bar"]
                    Sw = (s * w[:, None]).T @ s / self.n
                    pred = lam0 * kappa * np.linalg.solve(fp_matrix(Te, 1.0, 1.0, lam0), vec((centred - K_bar) @ Sw))
                    np.testing.assert_allclose(vec(arm["K"] - ex["K"]), pred, rtol=1e-8, atol=1e-12)
                    self.assertGreater(rel(arm["K"], ex["K"]), 1e-6)  # the population-centred identity is not exact

    def test_block_c_cancellation_identity_and_context_locality(self):
        """Part 0 item 4 (block C) and Part 3 item 1: P1 with raw weights 1 + lambda_0 kappa d_a 1[c = 0] = EXACT per
        context; with lambda_0 set to the clean value, contexts 1-3 under a context-0 error solve as in clean
        (1e-10)."""
        h = self.h
        d = common("expert", 4)
        self.ctx_tr = tr = d["train"]
        crit, C, n = PH.ContextQuadraticCritic, 4, len(tr["state"])
        ctx0 = (tr["context"] == 0).astype(float)
        centre = PH.ls_gain(tr["state"], tr["action"], ctx0)
        K0 = np.tile(h.K0, (C, 1, 1))
        params = PH.critic_params_c(h, d, "q1opt-local", kappa=1.0, centre=centre)[0]
        lam0 = PH.lambda0(params, tr["obs"], K0)
        T1, Te = self.rows(params, crit), self.rows(params, crit, "exact")
        arm = PH.solve_fp_context(T1, tr["context"], 1.0, 1 + lam0 * 1.0 * 2 * ctx0, lam0, C)
        ex = PH.solve_fp_context(Te, tr["context"], 1.0, 1.0, lam0, C)
        self.assertTrue(arm["finite"] and ex["finite"])
        self.assertLess(rel(arm["K"], ex["K"]), 1e-6)
        clean = PH.critic_params_c(h, d, "clean")[0]
        lam_clean = PH.lambda0(clean, tr["obs"], K0)
        for case, kw in (("q1opt-local", dict(kappa=1.0)), ("tilt-local", dict(a_star=-0.5))):
            err = PH.critic_params_c(h, d, case, **kw)[0]
            k_err = PH.solve_fp_context(self.rows(err, crit), tr["context"], 1.0, 1.0, lam_clean, C)["K"]
            k_clean = PH.solve_fp_context(self.rows(clean, crit), tr["context"], 1.0, 1.0, lam_clean, C)["K"]
            np.testing.assert_allclose(k_err[1:], k_clean[1:], rtol=0, atol=1e-10)
            self.assertGreater(np.abs(k_err[0] - k_clean[0]).max(), 1e-3)
        K_rows = PH.policy_np(arm["K"], tr["obs"])  # per-context gains act on their own rows only
        np.testing.assert_allclose(K_rows[tr["context"] == 2], -tr["state"][tr["context"] == 2] @ arm["K"][2].T)
        self.assertEqual(n, len(tr["context"]))

    def test_unbounded_fixed_point_is_nonfinite(self):
        """q1_optimistic at kappa = 2 is not PD at K_0 (closed form): the arm is nonfinite, never a number."""
        h, tr = self.h, self.tr
        params = PH.critic_params_x(h, self.d, "q1_optimistic", 2.0)[0]
        lam0 = PH.lambda0(params, tr["obs"], h.K0)
        res = PH.solve_fp(self.rows(params), 1.0, 1.0, lam0)
        self.assertFalse(res["finite"])
        self.assertLessEqual(res["min_eig"], 0)
        self.assertTrue(np.all(np.isnan(res["K"])))

    def test_p3_oracle_fixed_point_is_exact(self):
        """Part 2 item 6, second clause: with e1 >= 0, P3-OR at c = 1 (closed form: P1 - c Pe) equals EXACT within 1e-6
        (same locked lambda_0); BFGS on the float32 loss agrees with the closed form along the c ladder."""
        h, tr = self.h, self.tr
        for case, kappa in (("q1_optimistic", 1.0), ("cone", 2.0)):
            params = PH.critic_params_x(h, self.d, case, kappa)[0]
            lam0 = PH.lambda0(params, tr["obs"], h.K0)
            T1, Te, Terr = self.rows(params), self.rows(params, part="exact"), self.rows(params, part="error")
            ex = PH.solve_fp(Te, 1.0, 1.0, lam0)["K"]
            combo = lambda c: dict(T1, Tq=T1["Tq"] - c * Terr["Tq"], bq=T1["bq"] - c * Terr["bq"])
            self.assertLess(rel(PH.solve_fp(combo(1.0), 1.0, 1.0, lam0)["K"], ex), 1e-6, case)
            cell = PH.make_cell(h, self.d, params)
            k_none = PH.solve_fp(T1, 1.0, 1.0, lam0)["K"]
            ladder = (0.1, 0.5, 1.0)
            res = PH.fixed_point_penalty(cell.models, params, tr["obs"], tr["action"], "or", PH.loop_context(cell),
                                         lam0, ladder, k_none, h.K0)
            for r, c in zip(res, ladder):
                self.assertTrue(r["success"], (case, c, r["message"]))
                self.assertLess(rel(r["K"], PH.solve_fp(combo(c), 1.0, 1.0, lam0)["K"]), 1e-4, (case, c))

    def test_lstd_in_class_exactness_and_constant_reward(self):
        """Part 3 item 1: LSTD on an in-class reward equals the closed form within 1e-8 (deterministic transitions; with
        shocks the sampled next state makes it differ by O(n^-1/2)), and a constant reward gives 1/(1 - gamma) = 20."""
        h = self.h
        system = h.system
        rng = np.random.default_rng(14)
        s = rng.normal(scale=0.4, size=(3000, 3))
        a = -s @ h.K0.T + 0.2 * rng.standard_normal((3000, 2))
        G = rng.standard_normal((5, 5))
        G = (G + G.T) / 2
        z = np.concatenate([s, a], 1)
        p = np.einsum("ni,ij,nj->n", z, G, z)
        v_clip = PH.clipped_noise_variance(h.args.policy_noise, h.args.noise_clip)
        det = replace(system, noise=np.zeros((3, 3)))
        H_, _, _ = LQ.quadratic_q(det, h.K0, G)
        h_ = system.gamma * v_clip * np.trace(H_[3:, 3:]) / (1 - system.gamma)  # target noise adds a constant
        s2 = s @ system.A.T + a @ system.B.T
        (M, m0, _), = PH.lstd_penalty(s, a, s2, p, h.K0, system.gamma, v_clip)
        np.testing.assert_allclose(M, H_, rtol=0, atol=1e-8 * np.abs(H_).max())
        self.assertAlmostEqual(m0, h_, delta=1e-8 * max(1.0, abs(h_)))
        s2_noisy = s2 + rng.standard_normal((3000, 3)) @ np.linalg.cholesky(system.noise).T
        (M1, m1, _), = PH.lstd_penalty(s, a, s2_noisy, np.ones(3000), h.K0, system.gamma, v_clip)
        np.testing.assert_allclose(M1, 0.0, atol=1e-8)
        self.assertAlmostEqual(m1, 20.0, delta=1e-8)
        (M2, _, _), = PH.lstd_penalty(s, a, s2_noisy, p, h.K0, system.gamma, v_clip)
        self.assertGreater(np.abs(M2 - H_).max(), 1e-6)  # sampled shocks: not exact
        draws = np.clip(np.random.default_rng(15).normal(scale=0.2, size=2_000_000), -0.5, 0.5)
        self.assertAlmostEqual(v_clip, float(np.mean(draws ** 2)), delta=4 * np.std(draws ** 2) / math.sqrt(len(draws)))

    def test_lstd_per_context_and_c2_rows(self):
        h = self.h
        d = common("poor", 4)
        tr = d["train"]
        K = np.tile(h.K0, (4, 1, 1))
        v_clip = PH.clipped_noise_variance()
        p = 1.0 + tr["context"]  # context-constant penalty: Q_p = p / (1 - gamma) in each context
        out = PH.lstd_penalty(tr["state"], tr["action"], tr["next_state"], p, K, h.system.gamma, v_clip, tr["context"])
        for c, (M, m0, _) in enumerate(out):
            self.assertAlmostEqual(m0, (1.0 + c) / (1 - h.system.gamma), delta=1e-7)
            np.testing.assert_allclose(M, 0.0, atol=1e-8)
        M = np.random.default_rng(16).standard_normal((5, 5))
        M = (M + M.T) / 2
        P_, q_ = PH.quadratic_rows(M, tr["state"])
        a = tr["action"]
        z = np.concatenate([tr["state"], a], 1)
        z0 = np.concatenate([tr["state"], 0 * a], 1)
        np.testing.assert_allclose(np.einsum("ni,nij,nj->n", a, P_, a) + 2 * np.sum(a * q_, 1),
                                   np.einsum("ni,ij,nj->n", z, M, z) - np.einsum("ni,ij,nj->n", z0, M, z0), atol=1e-12)


# ---------------------------------------------------------------------------------------------------------
# 6. Strength


class StrengthTests(unittest.TestCase):
    def test_strength_is_the_rms_action_change(self):
        rng = np.random.default_rng(17)
        s = rng.normal(size=(500, 3))
        K1, K2 = rng.normal(size=(2, 3)), rng.normal(size=(2, 3))
        S = PH.strength(K1, K2, PH.second_moment(s))
        self.assertAlmostEqual(S, math.sqrt(np.mean(np.sum(((K1 - K2) @ s.T) ** 2, 0))), places=12)
        batch = PH.strength(np.stack([K1, K2]), K2, PH.second_moment(s))
        np.testing.assert_allclose(batch, [S, 0.0], atol=1e-12)
        c = rng.integers(4, size=500)
        Kc1, Kc2 = rng.normal(size=(4, 2, 3)), rng.normal(size=(4, 2, 3))
        obs = np.concatenate([s, np.eye(4)[c]], 1)
        Sc = PH.strength(Kc1, Kc2, PH.second_moment(s, c, 4))
        diff = PH.policy_np(Kc1, obs) - PH.policy_np(Kc2, obs)
        self.assertAlmostEqual(Sc, math.sqrt(np.mean(np.sum(diff ** 2, 1))), places=12)

    def test_match_knob(self):
        f = lambda k: 0.1 * math.log1p(k)
        ladder = PH.BETA_LADDER
        for target in (0.01, 0.05, 0.2):
            m = PH.match_knob(ladder, f, target)
            self.assertTrue(m["reachable"] and m["converged"] and m["monotone"])
            self.assertLessEqual(abs(m["S"] / target - 1), 1e-3)
            self.assertLessEqual(m["evaluations"], 40)
        a500 = PH.match_knob(ladder, f, 0.05, tol=0.05, max_bisect=6, interpolate=True)
        self.assertTrue(a500["converged"])
        self.assertLessEqual(a500["evaluations"], 7)
        self.assertFalse(PH.match_knob(ladder, f, 1.0)["reachable"])  # above the ladder: never extrapolated
        self.assertFalse(PH.match_knob(ladder, f, 1e-4)["reachable"])  # below the first knob
        bumpy = [0.0, 0.2, 0.1, 0.3, 0.5]  # first crossing of 0.15 is between knobs 2 and 3
        g = lambda k: float(np.interp(math.log(k), np.log([1, 2, 3, 4, 5]), bumpy))
        m = PH.match_knob([1, 2, 3, 4, 5], g, 0.15)
        self.assertFalse(m["monotone"])
        self.assertEqual(m["bracket"], (0, 1))
        self.assertTrue(m["converged"])
        nan = PH.match_knob([1, 2, 3], lambda k: 0.2, 0.2, S_ladder=[0.1, float("nan"), 0.3])
        self.assertFalse(nan["reachable"])  # a nonfinite (unbounded) point never forms a bracket

    def test_matching_on_fixed_points(self):
        """Part 2 item 8 (fixed-point part): P1-SIG matched to S2 = S(P0(2)) within 1e-3."""
        cell, _, _ = fitted("tilt", a_star=-0.5)
        h, tr = harness(), cell.data["train"]
        A = PH.assignments(cell, 1, 1)
        lam0 = PH.lambda0(cell.params, tr["obs"], h.K0)
        T = PH.fp_terms(*PH.fp_rows(cell.params, tr["obs"]), tr["state"], tr["action"])
        Sig = PH.second_moment(cell.data["eval"]["state"])
        k_none = PH.solve_fp(T, 1.0, 1.0, lam0)["K"]
        target = PH.strength(PH.solve_fp(T, 1.0, 2.0, lam0)["K"], k_none, Sig)
        S_of = lambda beta: PH.strength(PH.solve_fp(T, 1.0, 1 + beta * A["weights"]["L"]["P1"]["SIG"], lam0)["K"],
                                        k_none, Sig)
        m = PH.match_knob(PH.BETA_LADDER, S_of, target)
        self.assertTrue(m["reachable"] and m["converged"], m)
        self.assertLessEqual(abs(m["S"] / target - 1), 1e-3)
        frontier_K = PH.solve_fp(T, np.ones((len(PH.FRONTIER_M), len(tr["state"]))),
                                 PH.FRONTIER_M[:, None] * np.ones(len(tr["state"])), lam0)["K"]
        fr = PH.frontier(PH.solve_fp(T, 1.0, 2.0, lam0)["K"], k_none, frontier_K, Sig)
        self.assertLess(abs(fr["m_star"] - 2.0) / 2.0, 0.04)  # P0(2) is its own nearest uniform point (grid spacing)
        self.assertLess(fr["share"], 0.05)

    def test_frontier_and_regret(self):
        rng = np.random.default_rng(18)
        Sig = PH.second_moment(rng.normal(size=(200, 3)))
        ms = np.geomspace(0.25, 100, 200)
        base, direction = rng.normal(size=(2, 3)), rng.normal(size=(2, 3))
        K_P0 = base + np.log(ms)[:, None, None] * direction
        fr = PH.frontier(K_P0[57], base, K_P0, Sig, ms)
        self.assertEqual(fr["index"], 57)
        self.assertAlmostEqual(fr["share"], 0.0, places=12)
        off = K_P0[57] + rng.normal(size=(2, 3))
        self.assertGreater(PH.frontier(off, base, K_P0, Sig, ms)["share"], 0)
        self.assertEqual(PH.regret_alpha(-3.0, [-3.5, -2.5, float("-inf")]), 0.5)
        self.assertIsNone(PH.regret_alpha(float("-inf"), [-1.0]))


# ---------------------------------------------------------------------------------------------------------
# 7. Level 1


class Level1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell, _, _ = fitted("cone", kappa=2.0, quality="poor")  # expert: BC pull ~ collinear with the ascent
        cls.tr = cls.cell.data["train"]

    def terms(self, **hooks):
        h = harness()
        return PH.terms(h.args, self.cell.models, self.cell.params, h.K0, self.tr["obs"], self.tr["action"], **hooks)

    def test_uniform_bc_and_constant_trust_project_onto_the_frontier(self):
        """P0(m): m_eff = m and tau = 0; constant trust 1/(1 + beta): m_eff = 1 + beta (identified cells only)."""
        n = len(self.tr["obs"])
        none = self.terms(bc_w=np.ones(n))
        for m in (0.5, 2.0, 6.0):
            l1 = PH.level1(self.terms(bc_w=np.full(n, m)), none)
            self.assertTrue(l1["identified"], l1["cos_gq_gbc"])
            self.assertAlmostEqual(l1["m_eff"], m, delta=1e-4 * m)
            self.assertLess(l1["tau"], 1e-5)
            self.assertAlmostEqual(l1["rho"], m * PH.level1(none, none)["rho"], delta=1e-4 * m)
        l1 = PH.level1(self.terms(bc_w=np.ones(n), q_w=np.full(n, 1 / 3)), none)
        self.assertAlmostEqual(l1["m_eff"], 3.0, delta=3e-4)
        self.assertLess(l1["tau"], 1e-5)

    def test_terms_add_up_and_match_step3s_actor_terms(self):
        h, n = harness(), len(self.tr["obs"])
        cp = self.cell.native.critic.params
        pen = PH.penalty_fn("sig", self.cell.models, cp, PH.loop_context(self.cell))
        t = self.terms(bc_w=1 + np.arange(n) % 3, pen_fn=pen, pen_c=0.5)
        np.testing.assert_allclose(t["g_Q"] + t["g_pen"] + t["g_BC"], t["g"], rtol=1e-5, atol=1e-6)
        self.assertGreater(np.linalg.norm(t["g_pen"]), 0)
        none = self.terms()
        qg, bg = LQ.actor_terms(h.args, self.cell.models, cp, jnp.asarray(h.K0, jnp.float32),
                                jnp.asarray(self.tr["obs"], jnp.float32), jnp.asarray(self.tr["action"], jnp.float32),
                                None)
        np.testing.assert_allclose(none["g_Q"], np.asarray(qg, np.float64), rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(none["g_BC"], np.asarray(bg, np.float64), rtol=1e-6, atol=1e-8)

    def test_level1_cosines(self):
        n = len(self.tr["obs"])
        cp = self.cell.native.critic.params
        none = self.terms(bc_w=np.ones(n))
        exact = self.terms(bc_w=np.ones(n), exact_fn=lambda o, a: self.cell.models[1].exact(cp, o, a))
        arm = self.terms(bc_w=1 + 2 * PH.cone_weight(self.tr["state"], PH.CONE_V, self.cell.info["c0"]))
        gJ = np.random.default_rng(19).normal(size=(2, 3))
        l1 = PH.level1(arm, none, exact, grad_J=gJ, v_bar=np.ones((2, 3)), pull=np.ones((2, 3)))
        dg = arm["g"] - none["g"]
        self.assertAlmostEqual(l1["cos_dg_gradJ"], LQ.cosine(-dg, gJ), places=12)
        self.assertAlmostEqual(l1["cos_dg_adam_gradJ"], l1["cos_dg_gradJ"], places=6)  # v-bar = 1: same direction
        self.assertAlmostEqual(l1["efficiency"], -np.sum(gJ * dg) / np.linalg.norm(dg), places=10)
        self.assertAlmostEqual(l1["cos_correction"], LQ.cosine(dg, exact["g"] - none["g"]), places=12)
        self.assertGreater(l1["cos_correction"], 0.5)  # cone-weighted BC moves toward the exact critic's gradient
        scaled = PH.level1(arm, none, grad_J=gJ, v_bar=np.full((2, 3), 4.0))
        self.assertAlmostEqual(scaled["cos_dg_adam_gradJ"], l1["cos_dg_gradJ"], places=6)


# ---------------------------------------------------------------------------------------------------------
# Closed-form settings (Part 0 item 6, B5, B6)


class SettingsTests(unittest.TestCase):
    def test_closed_form_settings_match_the_draft(self):
        """Part 0 item 6 (closed form), B5 and B6: step4_settings reproduces every number the draft quotes."""
        out = SETTINGS.compute()
        table = SETTINGS.comparison(out)
        for key, row in table.items():
            self.assertTrue(row["match"], (key, row))
        self.assertLess(out["fp_min_eig_q1opt_k2_clean_lambda"], 0)  # kappa = 2 not PD, hence kappa = 1 in X
        self.assertGreater(out["fp_min_eig_q1opt_k1_clean_lambda"], 0)
        self.assertGreater(min(out[f"fp_min_eig_cone_k2_{q}"] for q in PH.QUALITIES), 0)
        for q in ("expert", "mixed", "medium", "poor"):  # B5 uses the ascent; synth_pull printed the descent
            self.assertEqual(out[f"cos_pull_common_{q}"], -out[f"synth_pull_descent_common_{q}"])


class GuardTests(unittest.TestCase):
    """Guards added after review: no row-order tie-breaking, and no silent NUIS fallback to SIG."""

    def test_rank_normal_refuses_row_order_ties(self):
        x = np.array([1.0, 1.0, 2.0])
        with self.assertRaises(ValueError):
            PH.rank_normal(x)
        with self.assertRaises(ValueError):
            PH.batch_rank_normal(jnp.asarray(x))

    def test_missing_nuis_is_nan_not_sig(self):
        class Cell:  # the minimal fields loop_context reads
            data = {"train": {"state": np.zeros((4, 3))}}
            sig = {"w": jnp.ones(2)}
            nuis = None
        ctx = PH.loop_context(Cell())
        self.assertTrue(bool(jnp.all(jnp.isnan(ctx["nuis"]["w"]))))
        self.assertTrue(bool(jnp.all(ctx["sig"]["w"] == 1.0)))


if __name__ == "__main__":
    unittest.main()
