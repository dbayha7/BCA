"""Small continuous-plant checks; optional genuine legacy import is opt-in.

Set BCA_LEGACY_ROOT to a checkout with the original dependency environment to
run direct bitwise parity. The test never installs packages or patches imports.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from experiments.thrusterchain_diagnostics import continuous as C


class ContinuousPlantTests(unittest.TestCase):
    def test_frozen_matrices_and_independent_stage0_construction(self):
        self.assertEqual(C.provenance()["matrix_sha256"], C.MATRIX_SHA256)
        A, B = C.matrices()
        self.assertEqual(A.shape, (13, 13))
        self.assertEqual(B.shape, (13, 4))
        self.assertFalse(A.flags.writeable)
        self.assertFalse(B.flags.writeable)
        # Independent transcription of the inspected gapped_lq Stage0 recipe.
        oscillator = np.zeros((10, 10))
        for j, angle in enumerate((0.10, 0.20, 0.35, 0.55, 0.80)):
            cosine, sine = math.cos(angle), math.sin(angle)
            oscillator[2*j:2*j+2, 2*j:2*j+2] = (.98 - .005*j) * np.array(
                [[cosine, -sine], [sine, cosine]])
        for attempt in range(1000):
            coupling = np.random.default_rng([20261011, attempt]).standard_normal((10, 10))
            reconstructed = oscillator + .02 * coupling / np.linalg.norm(coupling, 2)
            if np.max(np.abs(np.linalg.eigvals(reconstructed))) <= 1:
                break
        self.assertEqual(attempt, 0)
        np.testing.assert_allclose(A[:10, :10], reconstructed, rtol=0, atol=2e-16)
        expected_B = np.zeros((13, 4))
        for k in range(4):
            expected_B[2*k+1, k] = expected_B[2*k+3, k] = .3
        np.testing.assert_array_equal(B, expected_B)

    def test_batch_scalar_and_knee_parity(self):
        rng = np.random.default_rng(812)
        states = rng.normal(size=(8, 13))
        actions = rng.uniform(-1.2, 1.2, size=(8, 4))
        actions[0] = [C.KNEE, np.float32(.8), np.nextafter(C.KNEE, np.inf), -1]
        for regime in (C.STALL, C.MONOTONE):
            batched, rewards = C.dynamics(states, actions, regime)
            for i in range(8):
                single, reward = C.dynamics(states[i], actions[i], regime)
                np.testing.assert_array_equal(batched[i], single)
                self.assertEqual(rewards[i], reward)
        at = C.thrust(np.full(4, C.KNEE), C.STALL)
        np.testing.assert_array_equal(at, np.full(4, C.KNEE))
        self.assertAlmostEqual(C.thrust(np.ones(4), C.STALL)[0], .5000000298023224)
        np.testing.assert_array_equal(C.dynamics(states, actions, C.MONOTONE)[0],
                                      C.dynamics(states, np.clip(actions, -1, 1), "no_stall")[0])

    def test_dominated_stall_commands_preserve_all_states(self):
        rng = np.random.default_rng(714)
        original = rng.uniform(-1, 1, size=(12, 4))
        original[0] = [1, -1, .9, -.9]
        dominated = C.dominated_stall_action(original)
        self.assertTrue(np.all(np.abs(dominated) <= C.KNEE))
        objective = C.Objective(12, .93)
        x = C.high_load_start()
        first = C.rollout_plan(x, C.STALL, original, objective)
        second = C.rollout_plan(x, C.STALL, dominated, objective)
        np.testing.assert_array_equal(first.states, second.states)
        self.assertGreater(second.return_value, first.return_value)
        self.assertTrue(np.all(second.rewards >= first.rewards))

    def test_environment_regime_time_and_terminal_mask(self):
        objective = C.Objective(3, 1)
        environment = C.ContinuousThrusterChain(objective)
        obs, info = environment.reset(C.high_load_start(), C.MONOTONE)
        self.assertEqual(obs.shape, (15,))
        np.testing.assert_array_equal(obs[-2:], [1, 1])
        for t in range(3):
            obs, reward, terminated, truncated, info = environment.step(np.zeros(4))
            self.assertFalse(terminated)
            self.assertEqual(truncated, t == 2)
            self.assertEqual(info["bootstrap_mask"], float(t != 2))
            self.assertEqual(obs[-2], C.MONOTONE)
            self.assertAlmostEqual(obs[-1], (2-t)/3)
        with self.assertRaises(RuntimeError):
            environment.step(np.zeros(4))

    def test_known_state_policy_q_v_terminal_and_local_discount(self):
        objective = C.Objective(8, .91)
        policy = C.regime_cap_policy(objective)
        x, t = C.high_load_start(), 3
        a = policy(C.observation(x, C.MONOTONE, t, objective))
        nxt, reward = C.dynamics(x, a, C.MONOTONE)
        value = C.policy_value(x, C.MONOTONE, policy, objective, t=t)
        expected = reward + objective.gamma * C.policy_value(
            nxt, C.MONOTONE, policy, objective, t=t+1)
        self.assertAlmostEqual(value, expected, places=12)
        self.assertAlmostEqual(C.policy_q(x, C.MONOTONE, a, policy, objective, t=t), value, places=12)
        self.assertEqual(C.policy_value(x, C.STALL, policy, objective, t=8), 0)
        final_action = np.array([1, -.9, .2, -.3])
        self.assertEqual(C.policy_q(x, C.STALL, final_action, policy, objective, t=7),
                         C.dynamics(x, final_action, C.STALL)[1])
        with self.assertRaises(ValueError):
            C.policy_q(x, C.STALL, a, policy, objective, t=8)

    def test_float32_deployment_boundary_is_explicit(self):
        objective = C.Objective(1)
        raw_action = np.full(4, .800000008)
        raw = C.rollout_plan(C.high_load_start(), C.STALL, raw_action[None], objective)
        deployed = C.rollout_plan(C.high_load_start(), C.STALL, raw_action[None], objective,
                                  float32_deployment=True)
        self.assertEqual(raw.observations.dtype, np.float64)
        self.assertEqual(deployed.observations.dtype, np.float32)
        np.testing.assert_array_equal(deployed.actions, np.full((1, 4), C.KNEE))
        self.assertFalse(np.array_equal(raw.actions, deployed.actions))

    @unittest.skipUnless(os.environ.get("BCA_LEGACY_ROOT"), "set BCA_LEGACY_ROOT for genuine original import")
    def test_direct_authoritative_legacy_parity(self):
        staging_root = str(Path(C.__file__).resolve().parents[2])
        original_root = os.environ["BCA_LEGACY_ROOT"]
        # Select the authoritative checkout before the first package import;
        # its real initializer and dependency checks run without modification.
        script = "\n".join([
            "import sys, json",
            f"sys.path.insert(0, {original_root!r})",
            "import experiments.shift.thruster_chain as legacy",
            f"assert legacy.__file__.startswith({original_root!r}), legacy.__file__",
            "from experiments.thrusterchain_diagnostics.continuous import verify_legacy_parity",
            f"print(json.dumps(verify_legacy_parity(legacy, repository_root={original_root!r})))",
        ])
        env = dict(os.environ, PYTHONPATH=staging_root, PYTHONDONTWRITEBYTECODE="1", JAX_PLATFORMS="cpu")
        completed = subprocess.run([sys.executable, "-c", script], cwd=staging_root,
                                   env=env, text=True, capture_output=True, timeout=40)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        result = json.loads(completed.stdout)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["source_sha256"]), 2)


class FiniteHorizonQPTests(unittest.TestCase):
    def test_objective_replay_gradient_and_numerical_certificate(self):
        rng = np.random.default_rng(19)
        x = C.high_load_start()
        x[:10] = rng.normal(size=10)
        for gamma in (1.0, .91):
            objective = C.Objective(8, gamma)
            problem = C.FiniteHorizonQP(objective)
            for regime, cap in ((C.STALL, C.KNEE), (C.MONOTONE, 1.0)):
                actions = rng.uniform(-cap, cap, size=(8, 4))
                replay = C.rollout_plan(x, regime, actions, objective)
                self.assertAlmostEqual(problem.cost(actions, x), replay.cost, places=10)
                solution = problem.solve(x, regime)
                self.assertLess(solution.replay_error, 1e-9)
                self.assertLess(solution.relative_fw_gap, 1e-9)
                self.assertTrue(np.all(np.abs(solution.actions) <= cap))
                np.testing.assert_allclose(solution.actions[-1], 0, rtol=0, atol=1e-12)
                self.assertGreaterEqual(problem.cost(actions, x) + 1e-10, solution.cost)
                # A non-optimal feasible point's gap must cover its suboptimality.
                gap = problem.frank_wolfe_gap(actions, x, cap)
                self.assertLessEqual(problem.cost(actions, x) - solution.cost, gap + 1e-9)
            direction = rng.normal(size=(8, 4))
            eps = 1e-6
            difference = (problem.cost(actions + eps*direction, x)
                          - problem.cost(actions - eps*direction, x)) / (2*eps)
            self.assertAlmostEqual(difference, problem.gradient(actions, x) @ direction.ravel(), places=6)

    def test_high_load_monotone_witness_and_second_solver(self):
        problem = C.FiniteHorizonQP(C.Objective(8))
        x = C.high_load_start()
        capped = problem.solve(x, C.MONOTONE, cap=C.KNEE)
        full = problem.solve(x, C.MONOTONE)
        alternative = problem.solve(x, C.MONOTONE, method="trf")
        self.assertGreater(capped.cost - full.cost, 1.9)
        self.assertAlmostEqual(capped.cost, 11.053461002180207, places=10)
        self.assertAlmostEqual(full.cost, 9.121444668955595, places=10)
        self.assertAlmostEqual(alternative.cost, full.cost, places=9)
        self.assertTrue(np.any(np.abs(full.actions) > C.KNEE))
        stall = problem.solve(x, C.STALL)
        self.assertAlmostEqual(stall.cost, capped.cost, places=11)

    def test_one_step_and_terminal_problem(self):
        x = C.high_load_start()
        x[1] = 2
        for t in (0, 1):
            problem = C.FiniteHorizonQP(C.Objective(1), t=t)
            result = problem.solve(x, C.STALL)
            np.testing.assert_array_equal(result.actions, np.zeros((1-t, 4)))
            self.assertEqual(result.cost, 4.0 if t == 0 else 0.0)
            self.assertEqual(result.fw_gap, 0.0)
            self.assertEqual(result.replay_error, 0.0)

    def test_controls_are_finite_horizon_and_observe_regime(self):
        objective = C.Objective(8)
        x = C.high_load_start()
        regime = C.regime_cap_policy(objective)
        cautious = C.fixed_cap_policy(C.KNEE, objective)
        for mode in (C.STALL, C.MONOTONE):
            np.testing.assert_array_equal(regime(C.observation(x, mode, 7, objective)), np.zeros(4))
        mono_return = C.policy_value(x, C.MONOTONE, regime, objective)
        capped_return = C.policy_value(x, C.MONOTONE, cautious, objective)
        self.assertGreater(mono_return, capped_return)

    def test_invalid_constraints_are_rejected(self):
        with self.assertRaises(ValueError):
            C.Objective(0)
        with self.assertRaises(ValueError):
            C.Objective(10, 0)
        with self.assertRaises(ValueError):
            C.FiniteHorizonQP(C.Objective(8)).solve(C.high_load_start(), C.STALL, cap=1)


if __name__ == "__main__":
    unittest.main()
