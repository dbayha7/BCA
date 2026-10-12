"""CPU tests of the ThrusterChain environment (no learning; every number is a test fixture, not a finding).

Checks: check_env passes with no warnings on all 8 ids; seeded resets are deterministic; options['state'] restores
exactly and states outside the start set are refused; actions are clipped (after float32 rounding); truncation at
t = 100 with terminated never True; render returns a line with '*' on stalling thrusters; the thrust curve (0, 0.8,
0.65, 0.5; the no-stall world is the identity; slope 3.0 gives 0.2); the proven observation bound and 10,000
adversarial full-throttle steps in the strongest current; step() equals dynamics() bit for bit, batches equal single
rows bit for bit, a separately written matrix form agrees within 1e-12 and the JAX form within 1e-12; the two worlds
are identical at or below 80%; the registered ids differ only in their start law; the evaluation banks are fixed;
the package refuses to import without its dependency directory; the commands SHIFT_ENV.md documents run verbatim
from a fresh process; a float32 command of exactly 80% sits at the knee (no stall, identical in both worlds) and the
stall-aware controller never shows as stalling through env.step; states produced from an amplitude-1 wave start are
accepted again.
JAX_PLATFORMS=cpu python -m unittest experiments.shift.test_thruster_chain
"""

import math
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.shift import thruster_chain as TC  # noqa: E402
from experiments.signal import gapped_lq as GLQ  # noqa: E402

import gymnasium as gym  # noqa: E402
from gymnasium.utils.env_checker import check_env  # noqa: E402


def independent_step(x, a, stall, slope=1.5):
    """A separately written step: s' = A10 s + 0.3 B1 thrust + 0.3 v (c + w1), c' = c, w' = Rot(0.27) w; the knee is
    80% on the float32 grid (float32(0.8))."""
    A10, _ = GLQ.transition_matrix(GLQ.STAGE0_SPEC)
    B1 = GLQ.base_input(1.0)
    v = np.zeros(10)
    v[1::2] = 1.0
    a = np.clip(np.asarray(a, np.float64), -1, 1)
    m = np.abs(a)
    k = float(np.float32(0.8))
    th = np.where(m <= k, a, np.sign(a) * (k - slope * (m - k))) if stall else a
    s, c, w = x[:10], x[10], x[11:]
    turn = np.array([[math.cos(0.27), -math.sin(0.27)], [math.sin(0.27), math.cos(0.27)]])
    nxt = np.concatenate([A10 @ s + 0.3 * (B1 @ th) + 0.3 * v * (c + w[0]), [c], turn @ w])
    return nxt, -(s @ s + 0.3 * (a @ a))


class EnvironmentTest(unittest.TestCase):
    def test_check_env_all_ids_no_warnings(self):
        self.assertEqual(len(TC.ENV_IDS), 8)
        for eid in TC.ENV_IDS:
            env = gym.make(eid, render_mode="ansi")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                check_env(env.unwrapped, skip_render_check=False)
            self.assertEqual([str(w.message) for w in caught], [], eid)
            self.assertEqual(env.spec.max_episode_steps, 100)

    def test_ids_differ_only_in_start_law(self):
        for eid, (world, situation) in TC.ENV_IDS.items():
            env = gym.make(eid).unwrapped
            self.assertEqual(env.physics, TC.WORLDS[world])
            self.assertEqual(env.law, TC.start_law(TC.START_LAWS[situation]))
        self.assertEqual(TC.env_id("no_stall", "big_kicks"), "BCAShift/ThrusterChainNoStall-BigKicks-v0")

    def test_seeded_reset_is_deterministic(self):
        e1, e2 = gym.make("BCAShift/ThrusterChain-v0"), gym.make("BCAShift/ThrusterChain-v0")
        o1, i1 = e1.reset(seed=7)
        o2, i2 = e2.reset(seed=7)
        self.assertTrue(np.array_equal(o1, o2) and np.array_equal(i1["state"], i2["state"]))
        o3, _ = e2.reset(seed=8)
        self.assertFalse(np.array_equal(o1, o3))
        self.assertEqual(o1.dtype, np.float32)
        self.assertTrue(np.array_equal(o1, i1["state"].astype(np.float32)))

    def test_start_laws(self):
        env = TC.ThrusterChainEnv(start="strong_current")
        for k in range(50):
            _, info = env.reset(seed=k)
            x = info["state"]
            self.assertTrue(0.7 <= x[TC.CURRENT] <= 1.1)
            self.assertTrue(0.1 <= TC.wave_amplitude(x) <= 0.4 + 1e-12)
        _, info = env.reset(options={"start": dict(current=(1.2, 1.2), wave=(0.0, 0.0), kick=0.0)})
        self.assertEqual(info["state"][TC.CURRENT], 1.2)
        self.assertTrue(np.all(info["state"][:10] == 0))
        for bad in (dict(current=(0.0, 2.0)), dict(wave=(0.0, 1.5)), dict(kick=-1.0), dict(speed=1.0)):
            with self.assertRaises(ValueError):
                env.reset(options={"start": bad})

    def test_state_option_restores_exactly_and_refuses_outside_start_set(self):
        env = TC.ThrusterChainEnv()
        x = TC.bank_starts("big_kicks", 3)[2] + 1e-9
        _, info = env.reset(options={"state": x})
        self.assertTrue(np.array_equal(info["state"], x))
        bad = []
        y = x.copy(); y[:10] = 40.0; bad.append(y)            # |s| > 100
        y = x.copy(); y[TC.CURRENT] = 1.6; bad.append(y)      # current > 1.5
        y = x.copy(); y[TC.CURRENT] = -0.1; bad.append(y)     # current < 0
        y = x.copy(); y[TC.WAVE1:] = [0.9, 0.9]; bad.append(y)  # amplitude > 1
        y = x.copy(); y[3] = np.nan; bad.append(y)
        for y in bad:
            with self.assertRaises(ValueError):
                env.reset(options={"state": y})
        with self.assertRaises(ValueError):
            env.reset(options={"state": x[:12]})
        with self.assertRaises(ValueError):
            env.reset(options={"state": x, "start": {}})
        with self.assertRaises(ValueError):
            env.reset(options={"speed": 1})

    def test_actions_clipped_after_float32_rounding(self):
        env = TC.ThrusterChainEnv()
        x = TC.bank_starts("recorded", 1)[0]
        env.reset(options={"state": x})
        _, r, _, _, info = env.step(np.array([2.0, -3.0, 0.1, 0.5]))
        self.assertTrue(np.array_equal(info["command"], [1.0, -1.0, np.float64(np.float32(0.1)), 0.5]))
        nxt, rr = TC.dynamics(x, info["command"], TC.STALL)
        self.assertTrue(np.array_equal(nxt, info["state"]) and rr == r)
        self.assertTrue(np.array_equal(info["stalled"], [True, True, False, False]))
        self.assertEqual(info["push"], x[TC.CURRENT] + x[TC.WAVE1])
        with self.assertRaises(ValueError):
            env.step(np.array([np.nan, 0, 0, 0]))
        with self.assertRaises(ValueError):
            env.step(np.zeros(3))

    def test_truncation_only_at_100(self):
        for env in (TC.ThrusterChainEnv(), gym.make("BCAShift/ThrusterChainNoStall-BigWaves-v0")):
            env.reset(seed=1)
            flags = []
            for _ in range(100):
                _, _, term, trunc, _ = env.step(np.zeros(4, np.float32))
                flags.append((term, trunc))
            self.assertFalse(any(t for t, _ in flags))
            self.assertEqual([tr for _, tr in flags], [False] * 99 + [True])
        raw = TC.ThrusterChainEnv()
        with self.assertRaises(RuntimeError):
            raw.step(np.zeros(4))
        raw.reset(seed=0)
        for _ in range(100):
            raw.step(np.zeros(4))
        with self.assertRaises(RuntimeError):
            raw.step(np.zeros(4))

    def test_render(self):
        env = TC.ThrusterChainEnv(render_mode="ansi")
        self.assertIn("not started", env.render())
        env.reset(seed=0)
        env.step(np.array([0.95, 0.2, -1.0, 0.0], np.float32))
        line = env.render()
        self.assertIsInstance(line, str)
        self.assertEqual(line.count("*"), 2)
        self.assertIn("current=", line)
        no_stall = TC.ThrusterChainEnv(stall=False, render_mode="ansi")
        no_stall.reset(seed=0)
        no_stall.step(np.array([0.95, 0.2, -1.0, 0.0], np.float32))
        self.assertNotIn("*", no_stall.render())
        self.assertIsNone(TC.ThrusterChainEnv().render())

    def test_thrust_curve(self):
        a = np.array([0.0, 0.8, 0.9, 1.0, -0.9, -1.0, 0.5])
        # the knee is float32(0.8) = 0.800000011920929, so the falling part sits 3e-8 above 0.65 / 0.5
        np.testing.assert_allclose(TC.thrust(a, TC.STALL), [0.0, 0.8, 0.65, 0.5, -0.65, -0.5, 0.5], atol=4e-8)
        self.assertEqual(TC.thrust(np.array([0.8]), TC.STALL)[0], 0.8)
        self.assertTrue(np.array_equal(TC.thrust(a, TC.NO_STALL), a))
        np.testing.assert_allclose(TC.thrust(np.array([1.0]), TC.Physics(stall_slope=3.0)), [0.2], atol=1e-7)
        self.assertTrue(np.array_equal(TC.stalled(a, TC.NO_STALL), np.zeros(7, bool)))
        self.assertTrue(np.array_equal(TC.past_knee(a[:, None]), np.abs(a) > 0.8))

    def test_thrust_form_equals_the_textbook_form_bit_for_bit(self):
        """thrust() is written as clip(a, -knee, knee) - slope sign(a) max(|a| - knee, 0) so that its JAX derivative
        at a = 0 is 1; it must give exactly the numbers of sign(a) (min(|a|, knee) - slope max(|a| - knee, 0))."""
        rng = np.random.default_rng(11)
        a = np.concatenate([rng.uniform(-1, 1, 200_000), rng.uniform(-1, 1, 200_000).astype(np.float32),
                            [0.0, -0.0, 0.8, -0.8, 1.0, -1.0, TC.KNEE, -TC.KNEE, np.nextafter(TC.KNEE, 2)]])
        for physics in (TC.STALL, TC.NO_STALL, TC.Physics(stall_slope=3.0)):
            m, knee, slope = np.abs(a), physics.thrust_knee, physics.thrust_slope
            textbook = np.sign(a) * (np.minimum(m, knee) - slope * np.maximum(m - knee, 0.0))
            self.assertTrue(np.array_equal(TC.thrust(a, physics), textbook))

    def test_float32_command_of_80_percent_is_at_the_knee(self):
        """float32(0.8) = 0.800000011920929 > 0.8: the knee is that number, so an 80% command sent through env.step
        does not stall, gets thrust float32(0.8), and both worlds step identically there."""
        self.assertEqual(TC.KNEE, float(np.float32(0.8)))
        self.assertEqual(TC.STALL.knee, TC.KNEE)
        cmd = np.array([0.8, -0.8, 0.8, -0.8], np.float32)
        self.assertFalse(TC.past_knee(cmd[None])[0])
        self.assertFalse(TC.stalled(cmd).any())
        x = TC.bank_starts("strong_current", 1)[0]
        steps = {}
        for stall in (True, False):
            env = TC.ThrusterChainEnv(stall=stall, render_mode="ansi")
            env.reset(options={"state": x})
            _, r, _, _, info = env.step(cmd)
            self.assertFalse(info["stalled"].any())
            self.assertTrue(np.array_equal(info["thrust"], cmd.astype(np.float64)))
            self.assertNotIn("*", env.render())
            steps[stall] = (info["state"], r)
        self.assertTrue(np.array_equal(steps[True][0], steps[False][0]) and steps[True][1] == steps[False][1])
        nxt = np.nextafter(np.float32(0.8), np.float32(1.0))     # the next float32 above 80% does stall
        self.assertTrue(TC.past_knee(np.array([[nxt, 0, 0, 0]], np.float32))[0])

    def test_stall_aware_controller_never_stalls_through_env_step(self):
        """The stall-aware controller is capped at 80%; stepped through the registered Gymnasium id (float32
        actions) it must never be reported as stalling, and its return equals the float64 truth rollout."""
        from experiments.shift import recorder as RC
        from experiments.shift import truth as TR

        pol = RC.stall_aware()
        starts = TC.bank_starts("strong_current", 20)
        env = gym.make(TC.env_id("stall", "strong_current"), render_mode="ansi")
        flagged, capped, J = 0, 0, []
        for x in starts:
            _, info = env.reset(options={"state": x})
            total = 0.0
            for _ in range(100):
                a = pol(info["state"]).astype(np.float32)
                capped += int(np.sum(np.abs(a) == np.float32(0.8)))
                _, r, _, _, info = env.step(a)
                flagged += int(info["stalled"].any())
                total += r
            J.append(total)
        self.assertGreater(capped, 100)      # the controller really sits at its 80% cap often
        self.assertEqual(flagged, 0)
        self.assertLess(np.abs(np.array(J) - TR.rollout(pol, starts).J).max(), 1e-3)

    def test_states_from_an_amplitude_one_start_are_accepted_again(self):
        """The wave's turn keeps its amplitude in exact arithmetic; float64 rounding lifts it a few ulps above 1, so a
        state the environment produced from an allowed start must still be inside the start set (tolerance 1e-9)."""
        env = TC.ThrusterChainEnv()
        _, info = env.reset(seed=1, options={"start": dict(wave=(0.9999999999999999,) * 2)})
        visited = []
        for _ in range(100):
            _, _, _, _, info = env.step(np.zeros(4, np.float32))
            visited.append(info["state"])
        visited = np.array(visited)
        self.assertGreater(TC.wave_amplitude(visited).max(), 1.0)     # rounding really lifted it
        self.assertTrue(np.all(TC.in_start_set(visited)))
        for v in visited[::10]:
            _, again = TC.ThrusterChainEnv().reset(options={"state": v})
            self.assertTrue(np.array_equal(again["state"], v))
        y = visited[0].copy()
        y[TC.WAVE1:] = [1.0 + 1e-6, 0.0]
        self.assertFalse(TC.in_start_set(y))

    def test_observation_bound(self):
        b = TC.observation_bound()
        self.assertAlmostEqual(b["sup_norm"], 1.0, places=3)
        self.assertAlmostEqual(b["sum_norm"], 55.48, places=2)
        self.assertAlmostEqual(b["per_step_input"], 2.818, places=3)
        self.assertLess(b["bound"], 256.5)
        self.assertLess(b["bound"], TC.FLOAT_BOUND)
        from experiments.shift.stage0 import adversarial_command

        env = TC.ThrusterChainEnv(stall=False, start="strong_current")
        worst = 0.0
        for k in range(10_000):
            if k % 100 == 0:
                obs, info = env.reset(seed=k, options={"start": dict(current=(1.5, 1.5), wave=(1.0, 1.0), kick=3.0)})
            obs, _, _, _, info = env.step(adversarial_command(info["state"]))
            self.assertTrue(env.observation_space.contains(obs))
            worst = max(worst, float(TC.float_radius(info["state"])))
        self.assertGreater(worst, 20.0)   # the adversary really pumps energy in
        self.assertLess(worst, b["bound"])

    def test_one_dynamics_function(self):
        rng = np.random.default_rng(0)
        x = TC.draw_starts(400, rng, TC.START_LAWS["big_kicks"])
        a = rng.uniform(-1.3, 1.3, (400, 4))
        for physics in (TC.STALL, TC.NO_STALL, TC.Physics(stall_slope=3.0)):
            nxt, r = TC.dynamics(x, a, physics)
            single = [TC.dynamics(x[i], a[i], physics) for i in range(len(x))]
            self.assertTrue(np.array_equal(np.stack([s[0] for s in single]), nxt))
            self.assertTrue(np.array_equal(np.array([s[1] for s in single]), r))
            ind = [independent_step(x[i], a[i], physics.stall, physics.stall_slope) for i in range(len(x))]
            self.assertLess(np.abs(np.stack([s[0] for s in ind]) - nxt).max(), 1e-12)
            self.assertLess(np.abs(np.array([s[1] for s in ind]) - r).max(), 1e-12)
        env = TC.ThrusterChainEnv()
        for i in range(50):
            env.reset(options={"state": x[i]})
            _, rr, _, _, info = env.step(a[i])
            ref, rref = TC.dynamics(x[i], np.clip(a[i].astype(np.float32).astype(np.float64), -1, 1), TC.STALL)
            self.assertTrue(np.array_equal(info["state"], ref) and rr == rref)

    def test_jax_dynamics_matches(self):
        import jax.numpy as jnp
        from jax.experimental import enable_x64

        rng = np.random.default_rng(1)
        x = TC.draw_starts(300, rng, TC.START_LAWS["recorded"])
        a = rng.uniform(-1.2, 1.2, (300, 4))
        for physics in (TC.STALL, TC.NO_STALL):
            with enable_x64():
                nj, rj = TC.dynamics_jax(jnp.asarray(x), jnp.asarray(a), physics)
                nj, rj = np.asarray(nj), np.asarray(rj)
            n, r = TC.dynamics(x, a, physics)
            self.assertLess(np.abs(nj - n).max(), 1e-12)
            self.assertLess(np.abs(rj - r).max(), 1e-12)

    def test_worlds_identical_at_or_below_80(self):
        rng = np.random.default_rng(2)
        x = TC.draw_starts(500, rng, TC.START_LAWS["strong_current"])
        a = rng.uniform(-0.8, 0.8, (500, 4))
        a[:50, 0] = 0.8
        a[50:100, 2] = -0.8
        s1, r1 = TC.dynamics(x, a, TC.STALL)
        s2, r2 = TC.dynamics(x, a, TC.NO_STALL)
        self.assertTrue(np.array_equal(s1, s2) and np.array_equal(r1, r2))
        a[0, 1] = 0.81
        self.assertFalse(np.array_equal(TC.dynamics(x[:1], a[:1], TC.STALL)[0], TC.dynamics(x[:1], a[:1],
                                                                                            TC.NO_STALL)[0]))

    def test_banks_fixed(self):
        for s in TC.SITUATIONS:
            b = TC.bank_starts(s)
            self.assertEqual(b.shape, (1000, 13))
            self.assertTrue(np.array_equal(b, TC.bank_starts(s)))
            self.assertTrue(np.all(TC.in_start_set(b)))
        sc = TC.bank_starts("strong_current")
        self.assertTrue(np.all(sc[:, TC.CURRENT] > 0.7))
        self.assertTrue(np.all(TC.wave_amplitude(TC.bank_starts("big_waves")) > 0.4))
        rec = TC.bank_starts("recorded")
        # common random numbers: the banks share every draw, only the law's scale differs
        np.testing.assert_allclose(TC.bank_starts("big_kicks")[:, :10], 3 * rec[:, :10], rtol=1e-15)
        self.assertTrue(np.array_equal(np.argsort(sc[:, TC.CURRENT]), np.argsort(rec[:, TC.CURRENT])))

    def test_package_refuses_without_dependency_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, BCA_SHIFT_DEPS=os.path.join(tmp, "missing"), PYTHONPATH=str(ROOT),
                       PYTHONDONTWRITEBYTECODE="1", JAX_PLATFORMS="cpu")
            done = subprocess.run([sys.executable, "-c", "import experiments.shift"], cwd=tmp, env=env,
                                  capture_output=True, text=True, timeout=120)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("install_deps.sh", done.stderr)
            empty = os.path.join(tmp, "empty")
            os.makedirs(empty)
            env["BCA_SHIFT_DEPS"] = empty
            done = subprocess.run([sys.executable, "-c", "import experiments.shift"], cwd=tmp, env=env,
                                  capture_output=True, text=True, timeout=120)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("gymnasium", done.stderr)

    def test_documented_commands_run_verbatim(self):
        """SHIFT_ENV.md's 'a single episode' command and its 'Evaluate a controller' snippet, run exactly as written
        from the repository root in a fresh process with the environment's Python and no PYTHONPATH."""
        import re
        import shlex

        page = (ROOT / "experiments" / "shift" / "SHIFT_ENV.md").read_text()
        line = next(ln for ln in page.splitlines() if ln.startswith('python -c "import experiments.shift'))
        snippet = re.search(r"\*\*Evaluate a controller.*?```python\n(.*?)```", page, re.S).group(1)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env.update(JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
        argv = shlex.split(line)
        self.assertEqual(argv[0], "python")
        done = subprocess.run([sys.executable] + argv[1:], cwd=str(ROOT), env=env, capture_output=True, text=True,
                              timeout=300)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertIn("t=  1", done.stdout)
        done = subprocess.run([sys.executable, "-c", snippet], cwd=str(ROOT), env=env, capture_output=True, text=True,
                              timeout=600)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertIn("mean 100-step return: -103.8", done.stdout)

    def test_environment_package_list_excludes_dependency_directory(self):
        import experiments.shift as pkg

        names = pkg.environment_packages()
        self.assertTrue(any(n.startswith("numpy==") for n in names))
        self.assertFalse(any(n.lower().startswith("gymnasium==") for n in names))


if __name__ == "__main__":
    unittest.main()
