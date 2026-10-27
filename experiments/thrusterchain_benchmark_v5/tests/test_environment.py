"""Physical-source parity, API contracts, reproducibility and horizon semantics."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import thrusterchain.canonical as source_package
from thrusterchain.environment import (
    ThrusterChainEnv, canonical, canonical_config, dynamics, matrices, STALL, NO_STALL, KNEE,
)

ROOT = Path(__file__).resolve().parents[1]


class EnvironmentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = json.loads((ROOT / "contracts/dynamics_fixtures.json").read_text(encoding="utf-8"))

    def test_archival_sources_and_matrix_identity(self):
        for record in self.fixtures["sources"]:
            raw = (ROOT / record["vendored_path"]).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), record["sha256"])
            self.assertEqual(len(raw), record["bytes"])
        a, b = matrices()
        self.assertFalse(a.flags.writeable)
        self.assertFalse(b.flags.writeable)
        digest = hashlib.sha256(a.astype("<f8").tobytes() + b.astype("<f8").tobytes()).hexdigest()
        self.assertEqual(digest, self.fixtures["matrix_sha256"])
        np.testing.assert_array_equal(a, self.fixtures["A"])
        np.testing.assert_array_equal(b, self.fixtures["B"])

    def test_authentication_detects_source_tampering(self):
        with tempfile.TemporaryDirectory(prefix="thrusterchain-provenance-") as temp:
            copied = Path(temp)
            for name in ("thrusterchain.py", "gapped_lq.py", "plant.json", "provenance.json"):
                shutil.copyfile(source_package.ROOT / name, copied / name)
            with (copied / "thrusterchain.py").open("ab") as handle:
                handle.write(b"\n# changed\n")
            with patch.object(source_package, "ROOT", copied):
                with self.assertRaisesRegex(RuntimeError, "source hash mismatch"):
                    source_package.provenance()

    def test_original_transition_fixtures_exact(self):
        for case in self.fixtures["cases"]:
            with self.subTest(case=case["id"]):
                env = ThrusterChainEnv(stall=case["world"] == "stall")
                initial, info = env.reset(options={"state": case["state"]})
                self.assertEqual(initial.dtype, np.float32)
                self.assertEqual(initial.shape, (14,))
                np.testing.assert_array_equal(info["state"], case["state"])
                observation, reward, terminated, truncated, info = env.step(case["action_requested"])
                np.testing.assert_array_equal(info["state"], case["next_state"])
                np.testing.assert_array_equal(info["command"], case["command"])
                np.testing.assert_array_equal(info["thrust"], case["thrust"])
                np.testing.assert_array_equal(info["stalled"], case["stalled"])
                self.assertEqual(reward, case["reward"])
                self.assertEqual((terminated, truncated), (False, False))
                self.assertEqual(observation[-1], np.float32(.99))

    def test_batch_dynamics_match_scalar_bitwise(self):
        rng = np.random.default_rng(381)
        x = canonical.draw_starts(31, rng, canonical.START_LAWS["big_waves"])
        a = rng.uniform(-1.2, 1.2, (31, 4)).astype(np.float32).astype(np.float64)
        for physics in (STALL, NO_STALL):
            nxt, reward = dynamics(x, a, physics)
            scalar = [dynamics(row, command, physics) for row, command in zip(x, a)]
            np.testing.assert_array_equal(nxt, np.stack([pair[0] for pair in scalar]))
            np.testing.assert_array_equal(reward, np.array([pair[1] for pair in scalar]))

    def test_finite_horizon_flags_and_zero_continuation(self):
        env = ThrusterChainEnv()
        env.reset(seed=137)
        for t in range(1, 101):
            observation, _, terminated, truncated, info = env.step(np.zeros(4))
            self.assertEqual(info["t"], t)
            self.assertEqual(info["remaining_steps"], 100 - t)
            self.assertEqual(terminated, t == 100)
            self.assertFalse(truncated)
            self.assertEqual(info["bootstrap_mask"], 0. if t == 100 else 1.)
            self.assertEqual(info["continuation_discount"], 0. if t == 100 else .98)
            self.assertEqual(info["canonical_truncated"], t == 100)
            self.assertFalse(info["canonical_terminated"])
        self.assertEqual(observation[-1], 0.)
        with self.assertRaisesRegex(RuntimeError, "episode ended"):
            env.step(np.zeros(4))
        env.reset(seed=138)
        self.assertEqual(env.t, 0)

    def test_legacy_time_limit_mode(self):
        env = ThrusterChainEnv(task_semantics="continuing_time_limit", observation_mode="physical")
        observation, _ = env.reset(seed=13)
        self.assertEqual(observation.shape, (13,))
        for _ in range(100):
            observation, _, terminated, truncated, info = env.step([0., 0., 0., 0.])
        self.assertEqual((terminated, truncated), (False, True))
        self.assertEqual(info["bootstrap_mask"], 1.)
        self.assertEqual(info["continuation_discount"], .98)
        with self.assertRaises(ValueError):
            ThrusterChainEnv(observation_mode="physical")

    def test_seeded_resets_are_reproducible_and_instance_local(self):
        first, second = ThrusterChainEnv(), ThrusterChainEnv()
        first.reset(seed=31)
        second.reset(seed=31)
        np.testing.assert_array_equal(first.state, second.state)
        old = first.state
        first.action_space.seed(999)
        first.action_space.sample()
        first.reset()
        second.reset()
        np.testing.assert_array_equal(first.state, second.state)
        self.assertFalse(np.array_equal(old, first.state))
        second.reset(seed=32)
        self.assertFalse(np.array_equal(first.state, second.state))

    def test_explicit_reset_does_not_consume_random_draws(self):
        first, second = ThrusterChainEnv(), ThrusterChainEnv()
        valid = np.zeros(13)
        first.reset(seed=72, options={"state": valid})
        first.reset()
        second.reset(seed=72)
        np.testing.assert_array_equal(first.state, second.state)

    def test_reset_law_and_exact_state_validation(self):
        env = ThrusterChainEnv()
        obs, info = env.reset(seed=10, options={"start": {"kick": 0., "current": [1.5, 1.5], "wave": [1., 1.]}})
        self.assertEqual(info["state"][10], 1.5)
        np.testing.assert_array_equal(info["state"][:10], np.zeros(10))
        self.assertAlmostEqual(np.linalg.norm(info["state"][11:]), 1.)
        self.assertTrue(env.observation_space.contains(obs))
        bad_states = [np.zeros((1, 13)), np.zeros(12), [np.nan] * 13]
        for index, value in ((0, 100.1), (10, -1.), (10, 1.50001), (11, 1.001)):
            bad = np.zeros(13)
            bad[index] = value
            bad_states.append(bad)
        for bad in bad_states:
            with self.subTest(state=np.asarray(bad).shape):
                with self.assertRaises(ValueError):
                    env.reset(options={"state": bad})
        for options in ({"start": {"kick": np.inf}}, {"start": {"kick": -1.}},
                        {"start": {"wave": [0., np.nan]}}, {"start": {"current": [1., .5]}},
                        {"start": {"unknown": 1}}, {"state": np.zeros(13), "start": {}},
                        {"unknown": 2}, {"start": []}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    env.reset(options=options)

    def test_action_validation_preserves_state_on_failure(self):
        env = ThrusterChainEnv()
        with self.assertRaisesRegex(RuntimeError, "reset"):
            env.step(np.zeros(4))
        env.reset(seed=7)
        state = env.state
        for action in (np.zeros((1, 4)), np.zeros(3), [True] * 4, ["0"] * 4,
                       [1j] * 4, [np.nan] * 4, [np.inf] * 4, [1e300] * 4):
            with self.subTest(action=str(action)):
                with self.assertRaises(ValueError):
                    env.step(action)
                self.assertEqual(env.t, 0)
                np.testing.assert_array_equal(env.state, state)

    def test_float32_knee_and_clipping(self):
        env = ThrusterChainEnv()
        env.reset(options={"state": np.zeros(13)})
        _, _, _, _, info = env.step([.8, -.8, 2., -2.])
        np.testing.assert_array_equal(info["command"], [KNEE, -KNEE, 1., -1.])
        np.testing.assert_array_equal(info["stalled"], [False, False, True, True])
        self.assertEqual(info["thrust"][0], KNEE)
        self.assertLess(info["thrust"][2], KNEE)

    def test_reward_uses_pre_state_and_command_not_thrust(self):
        env = ThrusterChainEnv()
        x = np.zeros(13)
        x[0], x[1] = 3., 4.
        env.reset(options={"state": x})
        _, reward, _, _, info = env.step(np.ones(4))
        self.assertEqual(reward, -(25. + .3 * 4.))
        self.assertTrue(np.all(info["thrust"] < 1.))

    def test_boundary_rollouts_invariants_and_spaces(self):
        rng = np.random.default_rng(891)
        for stall in (True, False):
            for slope in (1.5, 3.0):
                for trial in range(4):
                    x = np.zeros(13)
                    direction = rng.normal(size=10)
                    x[:10] = direction / np.linalg.norm(direction) * (100. - 1e-12)
                    x[10:] = [1.5, np.cos(trial), np.sin(trial)]
                    env = ThrusterChainEnv(stall=stall, stall_slope=slope)
                    env.reset(options={"state": x})
                    for _ in range(100):
                        observation, reward, _, _, info = env.step(rng.uniform(-2., 2., 4))
                        self.assertTrue(env.observation_space.contains(observation))
                        self.assertTrue(np.isfinite(reward))
                        self.assertEqual(info["state"][10], 1.5)
                        self.assertAlmostEqual(np.linalg.norm(info["state"][11:]), 1., places=12)

    def test_observation_and_info_are_defensive_copies(self):
        env = ThrusterChainEnv()
        observation, info = env.reset(seed=42)
        expected = env.state
        observation[:] = 0.
        info["state"][:] = 0.
        snapshot = env.state
        snapshot[:] = 0.
        np.testing.assert_array_equal(env.state, expected)

    def test_configuration_and_render(self):
        env = ThrusterChainEnv(render_mode="ansi")
        self.assertIn("not started", env.render())
        env.reset(seed=12)
        self.assertIn("t=  0", env.render())
        self.assertEqual(env.config["objective_gamma"], .98)
        self.assertEqual(canonical_config()["horizon"], 100)
        for kwargs in ({"stall": "False"}, {"stall_slope": np.nan}, {"stall_slope": 100.},
                       {"start": "bad"}, {"observation_mode": "bad"}, {"task_semantics": "bad"}):
            with self.assertRaises(ValueError):
                ThrusterChainEnv(**kwargs)

    def test_gymnasium_environment_checker(self):
        from gymnasium.utils.env_checker import check_env
        check_env(ThrusterChainEnv(), skip_render_check=True)
        check_env(ThrusterChainEnv(task_semantics="continuing_time_limit", observation_mode="physical"),
                  skip_render_check=True)

    @unittest.skipUnless(importlib.util.find_spec("jax"), "optional JAX dependency absent")
    def test_optional_jax_physical_parity(self):
        import jax
        import jax.numpy as jnp
        case = self.fixtures["cases"][1]
        with jax.experimental.enable_x64():
            nxt, reward = canonical.dynamics_jax(jnp.asarray(case["state"], dtype=jnp.float64),
                                                   jnp.asarray(case["command"], dtype=jnp.float64))
            np.testing.assert_allclose(np.asarray(nxt), case["next_state"], rtol=2e-14, atol=2e-14)
            self.assertAlmostEqual(float(reward), case["reward"], places=12)


if __name__ == "__main__":
    unittest.main()
