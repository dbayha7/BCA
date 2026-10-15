"""Checks of the Gapped LQ world (experiments/signal/gapped_lq.py): exact truth per context against Monte Carlo and
against independent algebra, Stage 0's machinery and stability checks, the behaviour data, the R0/R1 log metrics and
the float32 boundary. CPU only, small sizes, fixed seeds (every Monte Carlo check is a |z| <= 3 test at a fixed seed).
JAX_PLATFORMS=cpu python -m unittest experiments.signal.test_gapped_lq
"""
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.signal import gapped_lq as G
from experiments.signal import lq_harness as LQ

WORLD = G.World()
SMALL = (100, 40, 60)  # train / calibration / evaluation episodes: exact mix counts (30, 30, 15, 15, 10) etc.


def random_stable_gain(world, rng, c, scale=0.05, tries=100):
    """K*_c plus a small random perturbation that keeps context c discounted-stable."""
    for _ in range(tries):
        K = world.k_star[c] + scale * rng.standard_normal((G.DA, G.DS))
        if G.discounted_radius(world.systems[c], K) < 0.999:
            return K
    raise AssertionError("no stable perturbation found")


def policies(world, c, rng):
    """Gains whose exact truth the tests check: K*, K2, the behaviour clone, K = 0 and a random stable gain."""
    return dict(k_star=world.k_star[c], k_two=world.k_two[c], behaviour=world.k_behaviour[c],
                zero=np.zeros((G.DA, G.DS)), random=random_stable_gain(world, rng, c))


def bellman_residual(system, K, P, h, s):
    """Relative residual of V(s) = r(s, -K s) + gamma E V(s'), with E[s'^T P s'] = m^T P m + tr(P Sigma_eps), m = (A -
    B K) s. Zero only when (P, h) is the value of this K in this system: P = Pi^T G Pi + gamma M^T P M fixes P."""
    a = -s @ K.T
    m = s @ (system.A - system.B @ K).T
    lhs = np.einsum("ni,ij,nj->n", s, P, s) + h
    rhs = LQ.reward(system, s, a) + system.gamma * (np.einsum("ni,ij,nj->n", m, P, m) + np.trace(P @ system.noise) + h)
    return float(np.max(np.abs(lhs - rhs) / np.maximum(1.0, np.abs(lhs))))


class PlantTests(unittest.TestCase):
    def test_transition_matrix(self):
        w = WORLD
        self.assertLessEqual(np.max(np.abs(np.linalg.eigvals(w.A))), w.spec.max_radius)
        C = w.A - G.oscillators(w.spec)
        self.assertAlmostEqual(np.linalg.norm(C, 2), w.spec.coupling_norm, places=12)
        for j, (r, t) in enumerate(zip(w.rho, w.spec.theta)):
            np.testing.assert_allclose(G.oscillators(w.spec)[2 * j:2 * j + 2, 2 * j:2 * j + 2], r * G.rotation(t),
                                       rtol=0, atol=0)
        self.assertEqual(w.rho, tuple(w.spec.rho_top - w.spec.rho_step * j for j in range(5)))

    def test_input_matrices(self):
        w, gain = WORLD, WORLD.spec.gain
        B = w.B[G.R0]
        for k in range(G.DA):
            expect = np.zeros(G.DS)
            expect[[2 * k + 1, 2 * k + 3]] = gain
            np.testing.assert_array_equal(B[:, k], expect)
        self.assertTrue(np.all(B[:5, 2:] == 0))  # actuators 3-4 reach oscillators 3-5 only
        self.assertTrue(np.all(B[7:, :2] == 0))  # actuators 1-2 never reach oscillators 4-5 directly
        np.testing.assert_array_equal(w.B[G.R1][:, :2], w.spec.r1_scale * B[:, :2])
        np.testing.assert_array_equal(w.B[G.R1][:, 2:], B[:, 2:])
        np.testing.assert_array_equal(w.B[G.T_WRONG][:, :2], B[:, :2])
        np.testing.assert_array_equal(w.B[G.T_WRONG][:, 2:], -B[:, 2:])  # exact negation
        np.testing.assert_array_equal(w.B[G.T_RIGHT], B)
        np.testing.assert_array_equal(w.B[G.DECOY], B)
        np.testing.assert_array_equal(w.noise_sd, [0.1, 0.1, 0.1, 0.1, 0.3])
        rot = G.World(replace(G.STAGE0_SPEC, wrong_map="rotate135"))
        W = G.rotation(3 * math.pi / 4)
        np.testing.assert_allclose(W @ W.T, np.eye(2), atol=1e-15)
        np.testing.assert_allclose(rot.B[G.T_WRONG][:, 2:], B[:, 2:] @ W, atol=1e-15)

    def test_behaviour_gains(self):
        w = WORLD
        for c in G.RICH:
            np.testing.assert_array_equal(w.k_behaviour[c], w.spec.behaviour_fraction * w.k_star[c])
            np.testing.assert_array_equal(w.behaviour_sd[c], [w.spec.rich_noise_sd] * 4)
        for c in G.THIN:
            np.testing.assert_array_equal(w.k_behaviour[c], w.k_two[c])
            np.testing.assert_array_equal(w.k_two[c][2:], 0)
            np.testing.assert_array_equal(w.behaviour_sd[c], [0.1, 0.1, w.spec.thin_rung, w.spec.thin_rung])
        np.testing.assert_array_equal(w.k_two[G.T_WRONG], w.k_two[G.T_RIGHT])  # same first two columns
        np.testing.assert_allclose(w.k_star[G.DECOY], w.k_star[G.R0], rtol=1e-10)  # certainty equivalence


class ExactTruthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.default_rng(0)
        cls.s = cls.rng.standard_normal((16, G.DS))

    def test_bellman_identity_per_context_depends_on_the_policy(self):
        """V = r + gamma E V to 1e-12 for each policy's own (P, h); a wrong P, a 1e-9 perturbation, another context's
        h, or another context's P (for a policy that uses actuators 3-4) all fail it."""
        rng = np.random.default_rng(1)
        for c, system in enumerate(WORLD.systems):
            gains = policies(WORLD, c, rng)
            for name, K in gains.items():
                _, h, P = LQ.exact_q(system, K)
                self.assertLessEqual(bellman_residual(system, K, P, h, self.s), 1e-12, (G.CONTEXTS[c], name))
                # negative controls
                other = gains["zero" if name != "zero" else "k_star"]
                _, h2, P2 = LQ.exact_q(system, other)
                self.assertGreater(bellman_residual(system, K, P2, h2, self.s), 1e-3, (G.CONTEXTS[c], name))
                X = rng.standard_normal(P.shape)
                bump = 1e-9 * np.abs(P).max() * (X + X.T)
                self.assertGreater(bellman_residual(system, K, P + bump, h, self.s), 1e-11, (G.CONTEXTS[c], name))
        K = WORLD.k_star[G.R0]
        _, h_decoy, P_decoy = LQ.exact_q(WORLD.systems[G.DECOY], K)
        np.testing.assert_allclose(P_decoy, LQ.exact_q(WORLD.systems[G.R0], K)[2], rtol=1e-10)  # P ignores noise
        self.assertGreater(bellman_residual(WORLD.systems[G.R0], K, P_decoy, h_decoy, self.s), 1e-3)
        K = WORLD.k_star[G.T_RIGHT]  # uses actuators 3-4
        _, h_r, P_r = LQ.exact_q(WORLD.systems[G.T_RIGHT], K)
        self.assertLessEqual(bellman_residual(WORLD.systems[G.T_RIGHT], K, P_r, h_r, self.s), 1e-12)
        self.assertGreater(bellman_residual(WORLD.systems[G.T_WRONG], K, P_r, h_r, self.s), 1e-3)

    def test_q_bellman_identity_per_context(self):
        """Q(s, a) = r(s, a) + gamma E Q(s', -K s'), with Q(s', -K s') read from H: Pi^T H Pi = P only for this K."""
        rng = np.random.default_rng(2)
        a = rng.standard_normal((16, G.DA))
        for c, system in enumerate(WORLD.systems):
            K = WORLD.k_behaviour[c]
            H, h, _ = LQ.exact_q(system, K)
            for gain, ok in ((K, True), (WORLD.k_star[c] + 0.1, False)):
                Pi = np.vstack([np.eye(G.DS), -gain])
                X = Pi.T @ H @ Pi
                m = self.s @ system.A.T + a @ system.B.T
                rhs = LQ.reward(system, self.s, a) + system.gamma * (
                    np.einsum("ni,ij,nj->n", m, X, m) + np.trace(X @ system.noise) + h)
                err = np.max(np.abs(LQ.qform(H, h, self.s, a) - rhs) / np.abs(rhs))
                if ok:
                    self.assertLessEqual(err, 1e-12, G.CONTEXTS[c])
                else:
                    self.assertGreater(err, 1e-4, G.CONTEXTS[c])

    def test_finite_recursions_agree_and_converge_to_lyapunov(self):
        rng = np.random.default_rng(3)
        for c, system in enumerate(WORLD.systems):
            for name, K in policies(WORLD, c, rng).items():
                fwd, bwd = G.finite_return(system, K), G.finite_return_backward(system, K)
                self.assertLess(abs(fwd - bwd) / abs(fwd), 1e-10, (G.CONTEXTS[c], name))
                long = G.finite_return(system, K, length=4000)
                self.assertLess(abs(long - LQ.value(system, K)) / abs(long), 1e-9, (G.CONTEXTS[c], name))

    def test_finite_return_matches_monte_carlo_per_context(self):
        """The 100-step J of the behaviour clone (deterministic) and of the noisy behaviour itself (the data
        generator's own rollouts), against the covariance recursion, |z| <= 3 per context."""
        n, L = 3000, WORLD.spec.length
        for c in range(G.NC):
            rng = np.random.default_rng([4, c])
            ctx = np.full(n, c)
            s0, shock = rng.standard_normal((n, G.DS)), rng.standard_normal((n, L, G.DS))
            gains = np.repeat(WORLD.k_behaviour[c][None], n, 0)
            ret, _ = G.rollout_returns(WORLD, gains, ctx, s0, shock)
            z = (ret.mean() - WORLD.j_bc[c]) / (ret.std(ddof=1) / math.sqrt(n))
            self.assertLessEqual(abs(z), 3, (G.CONTEXTS[c], z))
            noise = rng.standard_normal((n, L, G.DA))
            ep = G.simulate_episodes(WORLD, ctx, gains, WORLD.behaviour_sd[ctx], s0, noise, shock)
            ret = ep["reward"] @ (WORLD.spec.gamma ** np.arange(L))
            exact = WORLD.j_finite(WORLD.k_behaviour, behaviour_noise=True)[c]
            z = (ret.mean() - exact) / (ret.std(ddof=1) / math.sqrt(n))
            self.assertLessEqual(abs(z), 3, (G.CONTEXTS[c], "noisy", z))

    def test_q_and_value_match_monte_carlo_per_context(self):
        """Q^pi(s, a) by rollouts of 300 steps plus the exact gamma^300 V(s_300) tail, |z| <= 3; J_inf likewise."""
        n, T = 2000, 300
        rng = np.random.default_rng(5)
        for c in range(G.NC):
            K = WORLD.k_star[c]
            H, h, P = WORLD.quadratic(WORLD.k_star)[c]
            s, a = rng.standard_normal(G.DS), 0.3 * rng.standard_normal(G.DA)
            system = WORLD.systems[c]
            first = LQ.reward(system, s, a)
            s1 = s @ WORLD.A.T + a @ WORLD.B[c].T + WORLD.noise_sd[c] * rng.standard_normal((n, G.DS))
            ret, sT = G.rollout_returns(WORLD, np.repeat(K[None], n, 0), np.full(n, c), s1,
                                        rng.standard_normal((n, T, G.DS)))
            tail = WORLD.spec.gamma ** T * (np.einsum("ni,ij,nj->n", sT, P, sT) + h)
            y = first + WORLD.spec.gamma * (ret + tail)
            z = (y.mean() - LQ.qform(H, h, s, a)) / (y.std(ddof=1) / math.sqrt(n))
            self.assertLessEqual(abs(z), 3, (G.CONTEXTS[c], z))
            s0 = rng.standard_normal((n, G.DS))
            ret, sT = G.rollout_returns(WORLD, np.repeat(K[None], n, 0), np.full(n, c), s0,
                                        rng.standard_normal((n, T, G.DS)))
            y = ret + WORLD.spec.gamma ** T * (np.einsum("ni,ij,nj->n", sT, P, sT) + h)
            z = (y.mean() - WORLD.j_inf(WORLD.k_star)[c]) / (y.std(ddof=1) / math.sqrt(n))
            self.assertLessEqual(abs(z), 3, (G.CONTEXTS[c], "J_inf", z))

    def test_riccati_optima(self):
        """grad J_inf = 0 at K*_c (and K2_c within the 2-actuator class); random perturbations lose; g > 0."""
        rng = np.random.default_rng(6)
        for c, system in enumerate(WORLD.systems):
            K = WORLD.k_star[c]
            grad, ref = LQ.policy_gradient(system, K), LQ.policy_gradient(system, 0.5 * K)
            self.assertLess(np.abs(grad).max(), 1e-5 * np.abs(ref).max(), G.CONTEXTS[c])
            for _ in range(5):
                self.assertLess(LQ.value(system, K + 0.02 * rng.standard_normal(K.shape)), LQ.value(system, K))
            K2 = WORLD.k_two[c]
            g2 = LQ.policy_gradient(system, K2)[:2]
            ref2 = LQ.policy_gradient(system, 0.5 * K2)[:2]
            self.assertLess(np.abs(g2).max(), 1e-5 * np.abs(ref2).max(), G.CONTEXTS[c])
            self.assertGreater(LQ.value(system, K), LQ.value(system, K2))

    def test_t_wrong_t_right_identity(self):
        """J_Tw(K12, D) = J_Tr(K12, -D) (J_inf and the 100-step J), Q_Tw(s, a) = Q_Tr(s, flip a); W D for rotate135."""
        rng = np.random.default_rng(7)
        a = rng.standard_normal((8, G.DA))
        flip = np.array([1.0, 1.0, -1.0, -1.0])
        for world, W in ((WORLD, -np.eye(2)), (G.World(replace(G.STAGE0_SPEC, wrong_map="rotate135")),
                                               G.rotation(3 * math.pi / 4))):
            sw, sr = world.systems[G.T_WRONG], world.systems[G.T_RIGHT]
            for _ in range(5):
                K = random_stable_gain(world, rng, G.T_RIGHT, 0.03)
                Kw = K.copy()
                Kw[2:] = np.linalg.solve(W, K[2:])  # T_wrong block D with W D = T_right's block
                self.assertAlmostEqual(G.finite_return(sw, Kw) / G.finite_return(sr, K), 1.0, places=12)
                self.assertAlmostEqual(LQ.value(sw, Kw) / LQ.value(sr, K), 1.0, places=12)
                if world is WORLD:
                    Hw, hw, _ = LQ.exact_q(sw, Kw)
                    Hr, hr, _ = LQ.exact_q(sr, K)
                    np.testing.assert_allclose(LQ.qform(Hw, hw, self.s[:8], a), LQ.qform(Hr, hr, self.s[:8], a * flip),
                                               rtol=1e-11)
        D = WORLD.k_star[G.T_RIGHT][2:]
        Kw = np.vstack([WORLD.k_two[G.T_WRONG][:2], D])
        Kr = np.vstack([WORLD.k_two[G.T_RIGHT][:2], -D])
        self.assertAlmostEqual(G.finite_return(WORLD.systems[G.T_WRONG], Kw)
                               / G.finite_return(WORLD.systems[G.T_RIGHT], Kr), 1.0, places=12)

    def test_row_truth(self):
        """World.q / v / grad_a_q equal lq_harness's quadratic forms per row's context; grad_a matches differences."""
        rng = np.random.default_rng(8)
        n = 40
        ctx = np.arange(n) % G.NC
        s, a = rng.standard_normal((n, G.DS)), rng.standard_normal((n, G.DA))
        K = WORLD.k_behaviour
        q = WORLD.q(K, s, a, ctx)
        for c in range(G.NC):
            H, h, _ = LQ.exact_q(WORLD.systems[c], K[c])
            np.testing.assert_array_equal(q[ctx == c], LQ.qform(H, h, s[ctx == c], a[ctx == c]))
        pi = WORLD.policy(K, s, ctx)
        np.testing.assert_array_equal(WORLD.v(K, s, ctx), WORLD.q(K, s, pi, ctx))
        g = WORLD.grad_a_q(K, s, a, ctx)
        eps = 1e-5
        for k in range(G.DA):
            d = np.zeros(G.DA)
            d[k] = eps
            fd = (WORLD.q(K, s, a + d, ctx) - WORLD.q(K, s, a - d, ctx)) / (2 * eps)
            np.testing.assert_allclose(g[:, k], fd, rtol=1e-6, atol=1e-6)
        global_K = WORLD.k_star[G.R0]
        np.testing.assert_array_equal(WORLD.per_context(global_K)[3], global_K)
        per_episode = WORLD.j_episodes(WORLD.k_star[[0, 2, 4]], [0, 2, 4])
        np.testing.assert_array_equal(per_episode, WORLD.j_opt[[0, 2, 4]])

    def test_aleatoric_sd_matches_monte_carlo(self):
        rng = np.random.default_rng(9)
        K = WORLD.k_behaviour
        for c in (G.R0, G.DECOY):
            s, a = rng.standard_normal((1, G.DS)), 0.3 * rng.standard_normal((1, G.DA))
            exact = WORLD.aleatoric_sd(K, s, a, np.array([c]))[0]
            _, h, P = WORLD.quadratic(K)[c]
            s2 = s @ WORLD.A.T + a @ WORLD.B[c].T + WORLD.noise_sd[c] * rng.standard_normal((200000, G.DS))
            y = WORLD.spec.gamma * (np.einsum("ni,ij,nj->n", s2, P, s2) + h)
            self.assertLess(abs(y.std() / exact - 1), 0.01, G.CONTEXTS[c])

    def test_score_and_density(self):
        sc = WORLD.score(WORLD.k_behaviour)
        np.testing.assert_allclose(sc["NS"], 0.0, atol=1e-12)
        self.assertAlmostEqual(sc["NS_mix"], 0.0, places=12)
        sc = WORLD.score(WORLD.k_star)
        np.testing.assert_allclose(sc["NS"], 1.0, atol=1e-12)
        self.assertAlmostEqual(sc["NS_mix"], 1.0, places=12)
        null = G.World(replace(G.STAGE0_SPEC, behaviour_fraction=1.0))
        per, mixed = null.normalised(null.j_finite(null.k_star))
        self.assertTrue(all(per[c] is None for c in G.RICH))
        self.assertAlmostEqual(mixed, 1.0, places=12)
        rng = np.random.default_rng(10)
        s = rng.standard_normal((5, G.DS))
        ctx = np.array([0, 1, 2, 3, 4])
        mean = WORLD.policy(WORLD.k_behaviour, s, ctx)
        np.testing.assert_array_equal(WORLD.behaviour_density_ratio(s, mean, ctx), 1.0)
        a = mean.copy()
        a[:, 3] += 0.01  # one floor SD (or one rung SD) along actuator 4 in thin rows
        r = WORLD.behaviour_density_ratio(s, a, ctx)
        np.testing.assert_allclose(r[list(G.THIN)], math.exp(-0.5), rtol=1e-12)


class TD3LevelTests(unittest.TestCase):
    """A-review regression: TD3's critic targets Q of the policy with clipped smoothing noise from the second step on.
    That fixed point is Q^pi + c_c with c_c = gamma var tr(H_aa) / (1 - gamma), about -13.3 here, so the R2 accuracy
    mark must measure level errors against it (against Q^pi a perfect critic reads 0.16-0.18 of |Q^pi|)."""

    def test_clipped_noise_variance(self):
        rng = np.random.default_rng(20)
        x = np.clip(0.2 * rng.standard_normal(4_000_000), -0.5, 0.5)
        var = G.clipped_noise_variance(0.2, 0.5)
        self.assertLess(abs(x.var() - var), 4 * x.var() * math.sqrt(2 / len(x)))
        self.assertLess(var, 0.04)  # clipping removes variance
        self.assertEqual(G.clipped_noise_variance(0.2, math.inf), 0.2 ** 2)
        self.assertEqual(G.clipped_noise_variance(0.0, 0.5), 0.0)

    def test_td3_level_is_the_smoothed_targets_fixed_point(self):
        """Monte Carlo of r(s, a) + sum_{t >= 1} gamma^t r(s_t, pi(s_t) + eps_t) (+ the exact tail) against Q^pi + c_c,
        |z| <= 3; the deterministic Q^pi fails the same check by far (|z| > 5; about 7 in the noisy decoy)."""
        n, T = 2000, 300
        rng = np.random.default_rng(21)
        K = WORLD.k_star
        level = WORLD.td3_level(K)
        var = G.clipped_noise_variance(G.TD3_POLICY_NOISE, G.TD3_NOISE_CLIP)
        for c in (G.R0, G.T_WRONG, G.DECOY):
            H, h, P = WORLD.quadratic(K)[c]
            gamma = WORLD.spec.gamma
            self.assertAlmostEqual(level[c], gamma * var * np.trace(H[G.DS:, G.DS:]) / (1 - gamma), places=10)
            self.assertLess(level[c], -10)
            s, a = rng.standard_normal(G.DS), 0.3 * rng.standard_normal(G.DA)
            system = WORLD.systems[c]
            s1 = s @ WORLD.A.T + a @ WORLD.B[c].T + WORLD.noise_sd[c] * rng.standard_normal((n, G.DS))
            eps = np.clip(G.TD3_POLICY_NOISE * rng.standard_normal((n, T, G.DA)), -G.TD3_NOISE_CLIP, G.TD3_NOISE_CLIP)
            ret, sT = G.rollout_returns(WORLD, np.repeat(K[c][None], n, 0), np.full(n, c), s1,
                                        rng.standard_normal((n, T, G.DS)), action_noise=eps)
            # tail: V of the noisy policy = s^T P s + h + c / gamma (its constant differs, its P does not)
            tail = WORLD.spec.gamma ** T * (np.einsum("ni,ij,nj->n", sT, P, sT) + h + level[c] / WORLD.spec.gamma)
            y = LQ.reward(system, s, a) + WORLD.spec.gamma * (ret + tail)
            se = y.std(ddof=1) / math.sqrt(n)
            z_td3 = (y.mean() - (LQ.qform(H, h, s, a) + level[c])) / se
            z_det = (y.mean() - LQ.qform(H, h, s, a)) / se
            self.assertLessEqual(abs(z_td3), 3, (G.CONTEXTS[c], z_td3))
            self.assertGreater(abs(z_det), 5, (G.CONTEXTS[c], z_det))

    def test_td3_level_scales_with_the_noise_and_is_nan_when_unstable(self):
        K = WORLD.k_behaviour
        np.testing.assert_allclose(WORLD.td3_level(K, 0.1, math.inf) / WORLD.td3_level(K, 0.2, math.inf), 0.25,
                                   rtol=1e-12)
        bad = WORLD.k_star.copy()
        bad[G.T_WRONG] = -20 * WORLD.k_star[G.T_RIGHT]
        level = WORLD.td3_level(bad)
        self.assertTrue(np.isnan(level[G.T_WRONG]))
        self.assertTrue(np.all(np.isfinite(np.delete(level, G.T_WRONG))))


class StabilityTests(unittest.TestCase):
    def test_stage0_spec_passes_every_target(self):
        out = G.stage0_full(G.STAGE0_SPEC)
        self.assertTrue(out["full_pass"], out["targets"])
        for name, radius in out["radius"].items():
            self.assertTrue(all(r < 1 for r in radius), name)
        self.assertLess(out["S3"]["full"]["radius_T_wrong"], 1)
        self.assertLess(out["S3"]["block"]["radius_T_wrong"], 1)
        self.assertGreaterEqual(out["S6"]["share_full"], G.S6_SHARE_MIN)
        json.dumps(G._clean(out), allow_nan=False, default=G._json_default)

    def test_design_defaults_fail_and_instability_is_flagged(self):
        out = G.stage0_cheap(G.DESIGN_SPEC)
        self.assertFalse(out["targets"]["S1"])
        self.assertGreater(max(out["S1_share"].values()), 0.9)
        self.assertFalse(out["S3"]["full"]["stable"])
        self.assertFalse(out["targets"]["stability"])
        w = G.World(G.DESIGN_SPEC)
        K = np.repeat(w.transplant()[None], G.NC, 0)
        self.assertEqual(w.j_inf(K)[G.T_WRONG], -math.inf)
        self.assertTrue(math.isfinite(w.j_finite(K)[G.T_WRONG]))  # the headline J stays finite
        with self.assertRaises(ValueError):
            w.quadratic(K)

    def test_stage0_world_transplant_is_stable(self):
        K = np.repeat(WORLD.transplant()[None], G.NC, 0)
        self.assertLess(WORLD.radius(K)[G.T_WRONG], 1)
        self.assertTrue(math.isfinite(WORLD.j_inf(K)[G.T_WRONG]))
        WORLD.quadratic(K)  # no exception

    def test_search_machinery_on_a_small_grid(self):
        """Step A on 16 candidates at the design gain, step B on three gains: the search returns STAGE0_SPEC."""
        spec = G.STAGE0_SPEC
        grid = dict(wrong_map=G.WRONG_MAPS, rho_top=(1.0, spec.rho_top), rho_step=(spec.rho_step,),
                    action_cost=(0.5, 20.0), behaviour_fraction=(0.4, spec.behaviour_fraction))
        chosen, records = G.stage0_search(grid=grid, gains=(0.3, spec.gain, 0.075), alt_seeds=(1,))
        self.assertEqual(len(records["step_a"]), 16)
        self.assertEqual(len(records["step_b"]), 3)
        self.assertFalse(records["step_b"][0]["full_pass"])  # gain 0.3: K*'s actuator-3-4 actions too small (S6)
        self.assertEqual(chosen, spec)
        json.dumps(G._clean(records), allow_nan=False, default=G._json_default)

    def test_step_b_leaves_deterministic_returns_unchanged(self):
        """Scaling gain by c and R by c^2 scales K by 1/c and leaves every deterministic J (and so S1-S3) unchanged."""
        a = G.World(replace(G.STAGE0_SPEC, gain=0.3, action_cost=20.0))
        np.testing.assert_allclose(WORLD.k_star, a.k_star * 0.3 / WORLD.spec.gain, rtol=1e-8)
        np.testing.assert_allclose(WORLD.j_opt, a.j_opt, rtol=1e-10)
        np.testing.assert_allclose(WORLD.j_bc, a.j_bc, rtol=1e-10)


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = G.generate(WORLD, 0, sizes=SMALL)
        cls.zero = G.generate(WORLD, 0, sizes=SMALL, rung=0.0)

    def test_shapes_dtypes_and_float32_boundary(self):
        tr = self.data["train"]
        n = SMALL[0] * WORLD.spec.length
        self.assertEqual(tr["obs"].shape, (n, G.OBS_DIM))
        self.assertEqual(tr["action"].shape, (n, G.DA))
        for k in ("obs", "action", "reward", "next_obs", "state"):
            self.assertEqual(tr[k].dtype, np.float64)
        T = G.transitions(tr)
        for field, x in zip(T._fields, T):
            self.assertEqual(str(x.dtype), "float32", field)
        self.assertEqual(T.obs.shape, (n, 15))
        np.testing.assert_array_equal(np.asarray(T.done), 0)
        np.testing.assert_array_equal(np.asarray(T.obs), tr["obs"].astype(np.float32))
        np.testing.assert_array_equal(np.asarray(T.reward), tr["reward"].astype(np.float32))
        r = -(np.sum(tr["state"] ** 2, 1) + WORLD.spec.action_cost * np.sum(tr["action"] ** 2, 1))
        np.testing.assert_allclose(tr["reward"], r, rtol=1e-14)

    def test_splits_and_bank(self):
        L = WORLD.spec.length
        tr, cal, ev = (self.data[s] for s in G.SPLITS)
        self.assertEqual(len(tr["reward"]), SMALL[0] * L)
        self.assertEqual(len(ev["reward"]), SMALL[2] * L)
        self.assertEqual(len(cal["reward"]), SMALL[1] * 5)
        eps = np.unique(cal["episode"])
        self.assertEqual(len(eps), SMALL[1])
        np.testing.assert_array_equal(np.bincount(cal["episode"] - eps.min()), 5)
        sets = [set(np.unique(self.data[s]["episode"])) for s in G.SPLITS]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        for split, n in zip(G.SPLITS, SMALL):
            np.testing.assert_array_equal(self.data["meta"]["context_episodes"][split], G.context_counts(n, WORLD.mix))
        np.testing.assert_array_equal(np.bincount(cal["context"], minlength=G.NC), 5 * G.context_counts(40, WORLD.mix))
        np.testing.assert_array_equal(self.data["meta"]["context_rows"]["train"],
                                      WORLD.spec.length * G.context_counts(SMALL[0], WORLD.mix))
        np.testing.assert_array_equal(G.context_counts(500, WORLD.mix), [150, 150, 75, 75, 50])
        self.assertAlmostEqual(np.isin(tr["context"], G.THIN).mean(), 0.30)

    def test_context_code(self):
        tr = self.data["train"]
        np.testing.assert_array_equal(tr["obs"][:, G.DS:], np.eye(G.NC)[tr["context"]])
        np.testing.assert_array_equal(tr["next_obs"][:, G.DS:], tr["obs"][:, G.DS:])
        np.testing.assert_array_equal(tr["obs"][:, :G.DS], tr["state"])
        noisy = G.generate(WORLD, 0, sizes=SMALL, code_sd=0.5)["train"]
        code = noisy["obs"][:, G.DS:].reshape(SMALL[0], WORLD.spec.length, G.NC)
        np.testing.assert_array_equal(code, np.broadcast_to(code[:, :1], code.shape))  # constant within episodes
        self.assertGreater(np.abs(code[:, 0] - np.eye(G.NC)[noisy["context"][::WORLD.spec.length]]).mean(), 0.2)
        np.testing.assert_array_equal(noisy["state"], tr["state"])  # the code never changes the dynamics
        cat = G.generate(WORLD, 0, sizes=SMALL, context_draw="categorical")
        self.assertEqual(cat["meta"]["context_draw"], "categorical")

    def test_behaviour_and_dynamics_per_row(self):
        tr = self.data["train"]
        ctx, s, a = tr["context"], tr["state"], tr["action"]
        resid = a + np.einsum("nij,nj->ni", WORLD.k_behaviour[ctx], s)
        np.testing.assert_allclose(resid, WORLD.behaviour_sd[ctx] * tr["noise"], atol=1e-12)
        step = tr["next_state"] - s @ WORLD.A.T - np.einsum("nij,nj->ni", WORLD.B[ctx], a)
        np.testing.assert_allclose(step, WORLD.noise_sd[ctx][:, None] * tr["shock"], atol=1e-12)
        thin = np.isin(self.zero["train"]["context"], G.THIN)
        self.assertTrue(np.all(self.zero["train"]["action"][thin][:, 2:] == 0))
        self.assertFalse(np.any(tr["action"][np.isin(ctx, G.THIN)][:, 2:] == 0))

    def test_ladder_and_null_variants_share_random_numbers(self):
        a, b = self.data["train"], self.zero["train"]
        for k in ("noise", "shock", "context", "episode"):
            np.testing.assert_array_equal(a[k], b[k])
        rich = np.isin(a["context"], G.RICH)
        np.testing.assert_array_equal(a["state"][rich], b["state"][rich])
        null = G.generate(WORLD, 0, sizes=SMALL, fraction=1.0)["train"]
        np.testing.assert_array_equal(null["shock"], a["shock"])
        thin = np.isin(a["context"], G.THIN)
        np.testing.assert_array_equal(null["state"][thin], a["state"][thin])
        self.assertEqual(G.generate(WORLD, 0, sizes=SMALL, fraction=1.0)["meta"]["behaviour_fraction"], 1.0)

    def test_t_wrong_and_t_right_logs_coincide_at_rung_zero(self):
        world = G.World(replace(G.STAGE0_SPEC, thin_rung=0.0))
        rng = np.random.default_rng(11)
        E, L = 20, world.spec.length
        s0, noise, shock = rng.standard_normal((E, G.DS)), rng.standard_normal((E, L, G.DA)), rng.standard_normal(
            (E, L, G.DS))
        out = [G.simulate_episodes(world, np.full(E, c), world.k_behaviour[np.full(E, c)],
                                   world.behaviour_sd[np.full(E, c)], s0, noise, shock) for c in G.THIN]
        for k in out[0]:
            np.testing.assert_array_equal(out[0][k], out[1][k])
        world = G.World(G.STAGE0_SPEC)  # rung 0.01: the logs differ, a little
        out = [G.simulate_episodes(world, np.full(E, c), world.k_behaviour[np.full(E, c)],
                                   world.behaviour_sd[np.full(E, c)], s0, noise, shock) for c in G.THIN]
        self.assertFalse(np.array_equal(out[0]["state"], out[1]["state"]))

    def test_determinism(self):
        again = G.generate(WORLD, 0, sizes=SMALL)
        for split in G.SPLITS:
            for k in self.data[split]:
                np.testing.assert_array_equal(again[split][k], self.data[split][k])
        other = G.generate(WORLD, 1, sizes=SMALL)
        self.assertFalse(np.array_equal(other["train"]["state"], self.data["train"]["state"]))

    def test_logged_action_rank_per_context(self):
        for data, rung in ((self.data, G.PRIMARY_RUNG), (self.zero, 0.0)):
            world = G.World(replace(G.STAGE0_SPEC, thin_rung=rung))
            m = G.log_metrics(world, data["train"], rung)
            for c in G.RICH:
                self.assertEqual(m["R1"][G.CONTEXTS[c]]["rank"], 4)
            for c in G.THIN:
                self.assertEqual(m["R1"][G.CONTEXTS[c]]["rank"], 2 if rung == 0 else 4)
            self.assertTrue(m["marks"]["R1a_thin_rung"], rung)
            self.assertTrue(m["marks"]["R1a_rich_eigs"])
            self.assertTrue(m["marks"]["R1b_thin_share"])


class MetricTests(unittest.TestCase):
    def test_definitions(self):
        self.assertAlmostEqual(G.participation_ratio(np.eye(10)), 10.0)
        self.assertAlmostEqual(G.participation_ratio(np.diag([1.0] + [0.0] * 9)), 1.0)
        phi = 0.8
        acf = phi ** np.arange(40)
        stop = int(np.flatnonzero(acf[1:] <= G.IACT_CUT)[0]) + 1
        tau, tau_base, crossed = G.iact(acf)
        self.assertAlmostEqual(tau, 1 + 2 * acf[1:stop].sum())
        self.assertTrue(crossed)
        self.assertEqual(G.settle_step(np.concatenate([np.linspace(3, 1, 30), np.ones(70)])), 27)

    def test_closed_form_r0_matches_a_large_log(self):
        """Exact R0 participation ratio and transient share vs a 1,000-episode behaviour log, per context."""
        closed, _ = G.r0_closed_form(WORLD)
        data = G.generate(WORLD, 3, sizes=(1000, 20, 20))
        emp = G.log_metrics(WORLD, data["train"], WORLD.spec.thin_rung)["R0"]
        for c in G.CONTEXTS:
            self.assertLess(abs(emp[c]["participation_ratio"] / closed[c]["participation_ratio"] - 1), 0.05, c)
            self.assertLess(abs(emp[c]["iact"] / closed[c]["iact"] - 1), 0.25, c)
            if c != "DECOY":  # the decoy's level barely moves (3.16 -> about 3.0): its settle step is ill-conditioned
                self.assertLessEqual(abs(emp[c]["settle_step_rms"] - closed[c]["settle_step"]), 6, c)
        self.assertLessEqual(abs(emp["pooled"]["settle_step_rms"] - closed["pooled"]["settle_step"]), 6)

    def test_strict_json_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json"
            digest = G._write(path, G._clean(dict(a=-math.inf, b=[1.0, math.nan], c=np.float64(2.0))))
            self.assertEqual(json.loads(path.read_text())["a"], None)
            self.assertIn(digest, (Path(tmp) / "x.json.sha256").read_text())
            with self.assertRaises(ValueError):
                G._write(path, dict(a=math.nan))


if __name__ == "__main__":
    unittest.main()
