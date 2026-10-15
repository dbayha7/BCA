"""CPU tests of world.py: the current laws, the exact weights, the recording controller, the removal identities and
the quadrature over c. No learning and no BCA. Every number here is a test fixture, not a finding.

The removal tests use a fixed ball (the scout's seed-0 ball, rounded) so they do not depend on a fitted gap.
JAX_PLATFORMS=cpu python -m unittest experiments.shift_iw.test_world
"""

import json
import math
import unittest

import numpy as np
from scipy import stats

import experiments.shift  # noqa: F401
import gymnasium as gym

from experiments.shift import recorder as RC
from experiments.shift import shift as SH
from experiments.shift import thruster_chain as TC
from experiments.shift_iw import data as D
from experiments.shift_iw import world as W

BALL = SH.Gap(centre=(-0.58, -0.12, -0.43, -0.27), rho=0.267, mass=0.15, rule="base", q=0.15,
              competing_ratio=float("nan"), record_sha256="")
_CACHE = {}


def episodes(law, n, tag):
    """n recorder episodes of a law (cached), flattened to steps, with per-step episode ids."""
    key = (law, n, tag)
    if key not in _CACHE:
        sim = W.draw_episodes(n, np.random.default_rng([4242, tag]), law, W.HoldRecorder())
        E, L = sim["commands"].shape[:2]
        _CACHE[key] = dict(x=sim["states"][:, :-1].reshape(E * L, TC.DX), a=sim["commands"].reshape(E * L, TC.DA),
                           r=sim["rewards"].reshape(-1), hold=sim["hold"].reshape(-1),
                           hard=sim["hard_water"].reshape(-1), ep=np.repeat(np.arange(E), L), sim=sim)
    return _CACHE[key]


def episode_mean_se(values, ep):
    """Mean of per-step values and its SE from whole episodes (episodes are independent)."""
    ids = np.unique(ep)
    sums = np.bincount(ep, values, len(ids))
    n = np.bincount(ep, minlength=len(ids))
    per = sums / n
    return float(values.mean()), float(per.std(ddof=1) / math.sqrt(len(ids)))


class LawTest(unittest.TestCase):
    def test_normalisers_and_means_match_the_spec(self):
        self.assertAlmostEqual(W.normaliser(-2), 0.35396379735782396, places=15)
        self.assertEqual(W.normaliser(0), 1.0)
        self.assertAlmostEqual(W.normaliser(2), 3.901805370636976, places=14)
        for tilt, mean in (("none", 0.443), ("mild", 0.600), ("strong", 0.757)):
            self.assertAlmostEqual(W.mean_current(tilt), mean, places=3)
        grid = np.linspace(W.C_LO, W.C_HI, 2001)
        sw = W.simpson_weights(grid)
        for tilt in W.TILT_NAMES:
            self.assertAlmostEqual(float(sw @ W.density(grid, tilt)), 1.0, places=10)
            self.assertAlmostEqual(float(sw @ (grid * W.density(grid, tilt))), W.mean_current(tilt), places=10)

    def test_cdf_and_inverse(self):
        u = np.linspace(0, 1, 1001)
        for tilt in W.TILT_NAMES:
            c = W.inverse_cdf(u, tilt)
            self.assertTrue(np.all(np.diff(c) >= 0))
            self.assertTrue(np.all((c >= W.C_LO) & (c <= W.C_HI)))
            np.testing.assert_allclose(W.cdf(c, tilt), u, atol=1e-12)
        self.assertEqual(float(W.cdf(W.C_LO, "strong")), 0.0)
        self.assertAlmostEqual(float(W.cdf(W.C_HI, "strong")), 1.0, places=15)

    def test_uniform_law_equals_thruster_chain_draw_starts_bit_for_bit(self):
        a = W.draw_starts(500, np.random.default_rng(5), "mild")
        b = TC.draw_starts(500, np.random.default_rng(5), W.start_law())
        self.assertTrue(np.array_equal(a, b))

    def test_laws_share_every_number_but_the_current(self):
        x = {t: W.draw_starts(300, np.random.default_rng(9), t) for t in W.TILT_NAMES}
        other = [j for j in range(TC.DX) if j != TC.CURRENT]
        for t in ("mild", "strong"):
            self.assertTrue(np.array_equal(x["none"][:, other], x[t][:, other]))
            self.assertFalse(np.array_equal(x["none"][:, TC.CURRENT], x[t][:, TC.CURRENT]))
        # one uniform per start: the currents are the same quantile under each law
        np.testing.assert_allclose(W.cdf(x["none"][:, TC.CURRENT], "none"),
                                   W.cdf(x["strong"][:, TC.CURRENT], "strong"), atol=1e-12)

    def test_drawn_currents_follow_their_law(self):
        for tilt in W.TILT_NAMES:
            c = W.draw_starts(20000, np.random.default_rng([7, W.TILT_NAMES.index(tilt)]), tilt)[:, TC.CURRENT]
            self.assertGreater(stats.kstest(c, lambda v: W.cdf(v, tilt)).pvalue, 1e-3, tilt)

    def test_starts_lie_in_the_environment_start_set(self):
        x = W.draw_starts(5000, np.random.default_rng(3), "strong")
        self.assertTrue(np.all(TC.in_start_set(x)))
        amp = TC.wave_amplitude(x)
        self.assertTrue(np.all((amp >= 0.1 - 1e-12) & (amp <= 0.4 + 1e-12)))


class WeightTest(unittest.TestCase):
    def test_closed_form_ratio(self):
        c = np.linspace(W.C_LO, W.C_HI, 101)
        for tilt in W.TILT_NAMES:
            np.testing.assert_allclose(W.ratio(c, tilt), W.density(c, tilt) / W.density(c, "train"), rtol=1e-13)
        self.assertTrue(np.array_equal(W.ratio(c, "none"), np.ones_like(c)))
        self.assertAlmostEqual(W.ratio_constant("mild"), 0.35396379735782396, places=15)
        self.assertAlmostEqual(W.ratio_constant("strong"), 0.09071795328941248, places=15)
        self.assertEqual([W.ratio_slope(t) for t in W.TILT_NAMES], [0.0, 2.0, 4.0])
        self.assertEqual([round(float(v), 2) for v in W.ratio([0.1, 1.1], "mild")], [0.43, 3.19])
        self.assertEqual([round(float(v), 3) for v in W.ratio([0.1, 1.1], "strong")], [0.135, 7.389])

    def test_weights_integrate_to_one_under_the_training_law(self):
        grid = np.linspace(W.C_LO, W.C_HI, 2001)
        sw = W.simpson_weights(grid)
        c = W.draw_starts(200000, np.random.default_rng(11), "train")[:, TC.CURRENT]
        for tilt in W.TILT_NAMES:
            self.assertAlmostEqual(float(sw @ (W.density(grid, "train") * W.ratio(grid, tilt))), 1.0, places=10)
            w = W.ratio(c, tilt)
            se = w.std(ddof=1) / math.sqrt(len(w))
            self.assertLess(abs(w.mean() - 1.0), 4 * se + 1e-15, tilt)

    def test_full_test_mass_and_effective_size(self):
        expected = {"none": 1.0, "mild": 1.381097845541816, "strong": 2.8414637940557546}
        grid = np.linspace(W.C_LO, W.C_HI, 2001)
        sw = W.simpson_weights(grid)
        for tilt, value in expected.items():
            self.assertAlmostEqual(W.full_test_mass(tilt), value, places=12)
            self.assertAlmostEqual(float(sw @ (W.density(grid, tilt) * W.ratio(grid, tilt))), value, places=9)
        self.assertEqual([round(1000 * W.kish_fraction(t)) for t in W.TILT_NAMES], [1000, 724, 352])

    def test_exact_ratio_is_infinite_inside_the_ball(self):
        a = np.array([BALL.centre, [0.5, 0.5, 0.5, 0.5]], np.float32)
        c = np.array([0.9, 0.9])
        r = W.exact_ratio(c, a, BALL, "strong", 0.8433)
        self.assertEqual(r[0], np.inf)
        self.assertAlmostEqual(float(r[1]), 0.8433 * float(W.ratio(0.9, "strong")), places=12)

    def test_constants_match_spec_json(self):
        if not D.SPEC_JSON.is_file():
            self.skipTest("spec.json is not beside the package")
        spec = json.loads(D.SPEC_JSON.read_text())
        self.assertEqual(spec["seeds"]["master"], W.MASTER_SEED)
        self.assertEqual(spec["laws"]["support"], [W.C_LO, W.C_HI])
        for tilt in W.TILT_NAMES:
            self.assertEqual(spec["laws"]["tilts"][tilt]["lambda"], W.TILTS[tilt])
            self.assertAlmostEqual(spec["laws"]["tilts"][tilt]["weight_const"], W.ratio_constant(tilt), places=12)
            self.assertAlmostEqual(spec["laws"]["full_data_test_mass_closed_form"][tilt], W.full_test_mass(tilt),
                                   places=12)
        rc = spec["recording_controller"]
        self.assertEqual(rc["hard_water_trigger"], W.HOLD_SPEC.trigger)
        self.assertEqual(rc["coin_probability"], W.HOLD_SPEC.hold_probability)
        self.assertEqual(rc["hold_jitter_thrusters_2_to_4"], W.HOLD_SPEC.jitter)
        np.testing.assert_allclose(W.HoldRecorder().hold, rc["hold_80"], atol=5e-4)
        sp = spec["data_per_seed"]
        self.assertEqual(sp["training_law_log"]["splits"]["training_episodes"], D.FULL.train)
        self.assertEqual(sp["training_law_log"]["splits"]["calibration_episodes"], D.FULL.calibration)
        self.assertEqual(sp["training_law_log"]["splits"]["held_out_episodes"], D.FULL.held_out)
        self.assertEqual(sp["training_law_log"]["splits"]["bank_rows_per_episode"], D.FULL.bank_rows_per_episode)
        self.assertEqual(sp["test_logs"]["strong"]["episodes"], D.FULL.test)
        self.assertEqual(sp["classifier_samples"]["training_law_episodes"], D.FULL.sample)
        self.assertEqual(sp["start_banks"]["starts_per_law"], D.FULL.start_bank)
        self.assertEqual(spec["coverage"]["exact_error_rows"], D.FULL.truth_rows)
        d1 = next(m for m in spec["pass_marks"] if m["id"] == "D1")["rule"]
        self.assertIn(f">= {round(100 * D.D1_HOLDS_REMOVED_MIN)}% of holds", d1)
        self.assertIn(f"Z in {list(D.D1_Z)}", d1)
        for tilt, rng in D.D1_IN_BALL.items():
            self.assertIn(f"{list(rng)} under {tilt}", d1)


class RecorderTest(unittest.TestCase):
    def test_hold_pattern_and_hold_demand(self):
        rec = W.HoldRecorder()
        self.assertEqual([round(float(v), 3) for v in rec.hold], [-0.8, -0.110, -0.468, -0.325])
        K = np.asarray(RC.textbook_gain())
        self.assertEqual([round(float(v), 3) for v in K[0, TC.CURRENT:]], [0.910, 0.822, -0.320])
        x = W.draw_starts(50, np.random.default_rng(1), "strong")
        np.testing.assert_allclose(rec.hold_demand(x), x[:, TC.CURRENT:] @ K[0, TC.CURRENT:], rtol=1e-12)

    def test_hold_sits_exactly_at_the_knee_and_never_stalls(self):
        d = episodes("strong", 300, 1)
        hold = d["hold"]
        self.assertGreater(hold.sum(), 1000)
        a = d["a"]
        self.assertTrue(np.all(a[hold, 0] == np.float32(-0.8)))
        self.assertEqual(float(np.abs(a[hold, 0]).max()), TC.KNEE)
        self.assertFalse(TC.stalled(a[hold].astype(np.float64)).any())
        self.assertFalse(TC.past_knee(a.astype(np.float64)).any())
        dev = a[hold, 1:].astype(np.float64) - W.HoldRecorder().hold[1:]
        self.assertLess(float(np.abs(dev).max()), 5 * 0.03)

    def test_hold_only_in_hard_water_with_a_fair_coin(self):
        d = episodes("strong", 300, 1)
        self.assertFalse(np.any(d["hold"] & ~d["hard"]))
        n = int(d["hard"].sum())
        rate = d["hold"][d["hard"]].mean()
        self.assertLess(abs(rate - 0.5), 4 * math.sqrt(0.25 / n))
        np.testing.assert_array_equal(d["hard"], W.HoldRecorder().hard_water(d["x"]))

    def test_normal_driving_stays_below_the_knee(self):
        d = episodes("strong", 300, 1)
        normal = d["a"][~d["hold"]].astype(np.float64)
        self.assertLess(float(np.abs(normal).max()), TC.KNEE)

    def test_clean_command_is_normal_driving_only(self):
        x = W.draw_starts(200, np.random.default_rng(2), "strong")
        self.assertTrue(np.array_equal(W.HoldRecorder().clean(x), RC.hand_only()(x)))

    def test_simulation_equals_env_step_bit_for_bit(self):
        sim = episodes("strong", 300, 1)["sim"]
        env = gym.make(D.ENV_ID)
        for e in range(3):
            o, info = env.reset(options={"state": sim["states"][e, 0]})
            self.assertTrue(np.array_equal(info["state"], sim["states"][e, 0]))
            for t in range(W.LENGTH):
                o, r, term, trunc, info = env.step(sim["commands"][e, t])
                self.assertTrue(np.array_equal(info["state"], sim["states"][e, t + 1]))
                self.assertTrue(np.array_equal(o, sim["states"][e, t + 1].astype(np.float32)))
                self.assertEqual(r, sim["rewards"][e, t])
                self.assertFalse(term)
            self.assertTrue(trunc)
        env.close()


class QualityTest(unittest.TestCase):
    """The dataset qualities (spec version 5): one controller with a random share per episode."""

    def test_expert_is_version_4_bit_for_bit(self):
        """draw_episodes and curves of the expert quality equal version 4's (no quality: version 4's draws only) on
        every array version 4 has; the expert's random flags are all False."""
        rec = W.HoldRecorder()
        for law in ("train", "strong"):
            v4 = W.draw_episodes(40, np.random.default_rng([W.MASTER_SEED, 3, 1]), law, rec)
            v5 = W.draw_episodes(40, np.random.default_rng([W.MASTER_SEED, 3, 1]), law, rec, quality="expert")
            for k in ("states", "commands", "rewards", "hard_water", "hold"):
                self.assertEqual(v4[k].dtype, v5[k].dtype, k)
                self.assertEqual(v4[k].tobytes(), v5[k].tobytes(), k)
            self.assertFalse(v5["random"].any())
            self.assertTrue(np.all(v5["random_share"] == 0.0))
        g4, c4 = W.curves(BALL, 5, 30, (4242, 9))
        g5, c5 = W.curves(BALL, 5, 30, (4242, 9), quality="expert", random_seed=(4242, 11))
        self.assertEqual(g4.tobytes(), g5.tobytes())
        for k in c4:
            self.assertEqual(c4[k].tobytes(), c5[k].tobytes(), k)

    def test_with_random_at_shares_zero_and_one(self):
        rec = W.HoldRecorder()
        x = W.draw_starts(500, np.random.default_rng(21), "strong")
        rng = np.random.default_rng(22)
        noise, coin = rng.standard_normal((500, 4)), rng.uniform(size=500)
        r, m = rng.uniform(-1.0, 1.0, size=(500, 4)), rng.uniform(size=500)
        cmd, _, press = rec.parts(x, noise, coin)
        self.assertTrue(press.any())
        c0, h0, r0 = W.with_random(cmd, press, r, m, np.zeros(500))
        self.assertEqual(c0.tobytes(), cmd.tobytes())
        self.assertEqual(h0.tobytes(), press.tobytes())
        self.assertFalse(r0.any())
        c1, h1, r1 = W.with_random(cmd, press, r, m, np.ones(500))
        self.assertTrue(r1.all())
        self.assertFalse(h1.any())
        np.testing.assert_array_equal(c1.astype(np.float32), r.astype(np.float32))

    def test_random_share_per_quality(self):
        pick = np.arange(500)
        self.assertTrue(np.all(W.random_share("random", pick) == 1.0))
        self.assertTrue(np.all(W.random_share("medium", pick) == 0.65))
        self.assertTrue(np.all(W.random_share("expert", pick) == 0.0))
        me = W.random_share("medium-expert", pick)
        np.testing.assert_array_equal(me == 0.65, pick < 250)
        np.testing.assert_array_equal(me == 0.0, pick >= 250)
        mr = W.random_share("medium-replay", pick)
        np.testing.assert_allclose(mr, 1.0 - 0.35 * pick / 499, rtol=0, atol=1e-15)
        self.assertEqual((mr[0], round(float(mr[-1]), 12)), (1.0, 0.65))
        with self.assertRaises(ValueError):
            W.random_share("novice", pick)
        # the same values under every law: an episode's share depends on its pick alone
        for q in W.QUALITIES:
            shares = [W.draw_episodes(30, np.random.default_rng([W.MASTER_SEED, 5, 1]), law, W.HoldRecorder(),
                                      quality=q)["random_share"] for law in ("train", "mild", "strong")]
            self.assertTrue(all(np.array_equal(shares[0], s) for s in shares[1:]), q)

    def test_holds_are_a_subset_of_the_experts(self):
        """Hard water, coins and noise are shared across qualities (common random numbers), so a quality's hold is
        the expert's hold on the same seed: every quality's holds are a subset of the expert's."""
        sims = {q: W.draw_episodes(60, np.random.default_rng([W.MASTER_SEED, 7, 1]), "strong", W.HoldRecorder(),
                                   quality=q) for q in W.QUALITIES}
        expert = sims["expert"]["hold"]
        self.assertTrue(expert.any())
        for q, s in sims.items():
            self.assertFalse(np.any(s["hold"] & ~expert), q)
            self.assertFalse(np.any(s["hold"] & s["random"]), q)
            np.testing.assert_array_equal(s["states"][:, 0], sims["expert"]["states"][:, 0])   # the same starts
        self.assertFalse(sims["random"]["hold"].any())
        self.assertTrue(sims["random"]["random"].all())
        self.assertGreater(sims["medium"]["random"].mean(), 0.6)
        self.assertLess(sims["medium"]["random"].mean(), 0.7)

    def test_exactly_80_percent_means_a_hold_on_steps_that_are_not_random(self):
        for q in ("medium", "medium-expert", "expert"):
            s = W.draw_episodes(80, np.random.default_rng([W.MASTER_SEED, 8, 1]), "strong", W.HoldRecorder(),
                                quality=q)
            a1, hold, rand = s["commands"][..., 0], s["hold"], s["random"]
            at80 = a1 == np.float32(-0.8)
            np.testing.assert_array_equal(at80[~rand], hold[~rand])
            self.assertTrue(hold.any(), q)

    def test_streamed_quality_is_the_data_controller(self):
        """The start banks' controller of a quality (fixed streams, truth.rollout) gives the same commands and
        returns as world.simulate with the same streams (the logs' controller)."""
        from experiments.shift import truth as TR

        x0 = W.draw_starts(25, np.random.default_rng(31), "strong")
        rng = np.random.default_rng(32)
        noise, coin = rng.standard_normal((25, W.LENGTH, 4)), rng.uniform(size=(25, W.LENGTH))
        r, m, pick = W.draw_random_streams(rng, 25)
        for q in ("medium-expert", "medium-replay"):
            eps = W.random_share(q, pick)
            sim = W.simulate(W.HoldRecorder(), x0, noise, coin, random=(r, m, eps))
            run = TR.rollout(W.StreamedQuality(W.HoldRecorder(), noise, coin, r, m, eps, q), x0, W.PHYSICS,
                             record=True)
            np.testing.assert_array_equal(run.commands, sim["commands"].astype(np.float64))
            np.testing.assert_array_equal(run.rewards, sim["rewards"])
            np.testing.assert_array_equal(run.states, sim["states"])
        self.assertEqual(float(W.points(-50.0, dict(random=-100.0, top=-50.0))), 100.0)
        self.assertEqual(float(W.points(-100.0, dict(random=-100.0, top=-50.0))), 0.0)

    def test_quality_numbers_match_spec_json(self):
        if not D.SPEC_JSON.is_file():
            self.skipTest("spec.json is not beside the package")
        q = json.loads(D.SPEC_JSON.read_text())["qualities"]
        self.assertEqual(tuple(q["names"]), W.QUALITIES)
        self.assertEqual(q["medium_random_share"], W.QUALITY_PLAN.medium_random_share)
        self.assertEqual(q["picks"], W.QUALITY_PLAN.picks)
        self.assertEqual(q["medium_expert_cut"], W.QUALITY_PLAN.medium_expert_cut)
        self.assertEqual(set(q["random_share"]), set(W.QUALITIES))


class RemovalIdentityTest(unittest.TestCase):
    """The exact-ratio identities by Monte Carlo: given c every step looks the same under every law, so
    E_test[f] = E_train[w f]; with the ball removed, E_test[f | outside] = E_kept[w f] / E_kept[w]."""

    @classmethod
    def setUpClass(cls):
        cls.tr = episodes("train", 1500, 2)
        cls.te = episodes("strong", 1500, 3)

    def features(self, d):
        a = d["a"].astype(np.float64)
        return dict(in_ball=BALL.contains(d["a"]).astype(float), hold=d["hold"].astype(float), a1=a[:, 0],
                    a1_near_limit=(a[:, 0] <= -0.75).astype(float), reward=d["r"], c=d["x"][:, TC.CURRENT])

    def test_exact_ratio_identity(self):
        w = W.ratio(self.tr["x"][:, TC.CURRENT], "strong")
        ft, fe = self.features(self.tr), self.features(self.te)
        for name in ft:
            m_test, se_test = episode_mean_se(fe[name], self.te["ep"])
            m_w, se_w = episode_mean_se(w * ft[name], self.tr["ep"])
            self.assertLess(abs(m_test - m_w), 4 * math.hypot(se_test, se_w), name)

    def test_identity_after_the_removal(self):
        w = W.ratio(self.tr["x"][:, TC.CURRENT], "strong")
        out_tr = ~BALL.contains(self.tr["a"])
        out_te = ~BALL.contains(self.te["a"])
        ft, fe = self.features(self.tr), self.features(self.te)
        for name in ("a1", "reward", "c"):
            # E_test[f 1_out] = E_train[w f 1_out]: the unnormalised form, whose ratio gives the conditional one
            m_test, se_test = episode_mean_se(fe[name] * out_te, self.te["ep"])
            m_w, se_w = episode_mean_se(w * ft[name] * out_tr, self.tr["ep"])
            self.assertLess(abs(m_test - m_w), 4 * math.hypot(se_test, se_w), name)
        # the gapped test mass: E_test[w | outside] = E_train[w^2 1_out] / E_train[w 1_out]
        w_te = W.ratio(self.te["x"][:, TC.CURRENT], "strong")
        direct = w_te[out_te].mean()
        via_train = (w * w * out_tr).mean() / (w * out_tr).mean()
        self.assertLess(abs(direct - via_train) / direct, 0.03)

    def test_quadrature_matches_direct_sampling(self):
        grid, curve = W.curves(BALL, 21, 200, (4242, 99))
        quad = W.quadrature(grid, curve)
        for tilt in W.TILT_NAMES:  # Simpson's rule on 21 points; the build's 41 points give about 4e-8
            self.assertAlmostEqual(quad["weight_integral_train"][tilt], 1.0, places=5)
        tr_in, tr_se = episode_mean_se(BALL.contains(self.tr["a"]).astype(float), self.tr["ep"])
        te_in, te_se = episode_mean_se(BALL.contains(self.te["a"]).astype(float), self.te["ep"])
        z = quad["Z"]
        self.assertLess(abs((1 - tr_in) - z["value"]), 4 * math.hypot(tr_se, z["se"]) + 0.005)
        s = quad["in_ball_share"]["strong"]
        self.assertLess(abs(te_in - s["value"]), 4 * math.hypot(te_se, s["se"]) + 0.005)
        out_te = ~BALL.contains(self.te["a"])
        direct_mass = W.ratio(self.te["x"][:, TC.CURRENT], "strong")[out_te].mean()
        self.assertLess(abs(direct_mass - quad["test_mass_gapped"]["strong"]["value"]) / direct_mass, 0.03)
        self.assertEqual(quad["test_mass_gapped"]["none"]["value"], 1.0)
        self.assertGreater(quad["holds_removed_train"]["value"], 0.9)


if __name__ == "__main__":
    unittest.main()
