"""Synthetic adapter fixtures: no canonical dynamics, trained model or dataset."""
import time
import unittest
from types import SimpleNamespace

import numpy as np
import reference as ref


def toy_dynamics(state, action, physics):
    del physics
    state, action = np.asarray(state), np.asarray(action)
    nxt = state.copy()
    nxt[..., 0] += action[..., 0]
    reward = state[..., 0]+action[..., 0]
    return nxt, np.asarray(reward, np.float64)


class ToyEnv:
    created = []

    def __init__(self, **kwargs):
        assert kwargs["task_semantics"] == "finite_horizon"
        self.t = 0
        self.closed = False
        self._state = None
        self.created.append(self)

    @property
    def state(self):
        return self._state.copy()

    def info(self):
        return {"remaining_steps": 100-self.t, "bootstrap_mask": float(self.t < 100),
                "continuation_discount": ref.GAMMA if self.t < 100 else 0.}

    def reset(self, seed, options=None):
        self.t = 0
        self._state = np.zeros(13, np.float64)
        self._state[10] = .6
        if options is not None:
            self._state = options["state"].copy()
        return ref.observation(self.state, np.asarray(0)), self.info()

    def step(self, action):
        if self.t >= 100:
            raise AssertionError("Stepped beyond terminal time")
        self._state, reward = toy_dynamics(self.state, action.astype(np.float64), None)
        self.t += 1
        return ref.observation(self.state, np.asarray(self.t)), reward, self.t == 100, False, self.info()

    def close(self):
        self.closed = True


class ToyPolicy:
    def __init__(self, constant=None):
        self.constant = constant
        self.seen = []

    def assert_unchanged(self):
        pass

    def act(self, obs):
        self.seen.append(obs.copy())
        actions = np.zeros((len(obs), 4), np.float32)
        actions[:, 0] = obs[:, -1] if self.constant is None else self.constant
        actions[:, 1:] = np.asarray([.1, -.2, .3], np.float32)
        return actions


def core(dynamics=toy_dynamics):
    return SimpleNamespace(dynamics=dynamics, Physics=lambda **kw: kw, ThrusterChainEnv=ToyEnv)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        ToyEnv.created = []

    def test_complete_synthetic_gate_accounting(self):
        # Entire planned schedule on toy dynamics, not the real benchmark.
        states = np.zeros((64, 101, 13), np.float64)
        states[..., 10] = .6
        all_times = np.broadcast_to(np.arange(101), (64, 101))
        obs = ref.observation(states, all_times)
        actions = np.zeros((64, 100, 4), np.float32)
        times = np.tile([0, 25, 50, 75], 16)
        seeds = np.arange(64)
        ledger = ref.new_ledger()
        ledger.update(actor_rows_attempted=2, actor_rows_completed=2)
        query = ref.reconstruct_states(core(), seeds, actions, obs, times, ledger)
        ref.runtime_parity(core(), query[0], ledger)
        policy = ToyPolicy(constant=.1)
        banks = ref.candidate_banks(policy, obs[np.arange(64), times], ledger)
        values = ref.continuation_values(core(), policy, query, times, banks, ledger)
        ref.free_policy_rollouts(core(), policy, seeds[:16], ledger)
        self.assertEqual(values.shape, (64, 6))
        self.assertEqual(ledger['transition_steps_attempted'], 28104)
        self.assertEqual(ledger['transition_steps_completed'], 28104)
        self.assertEqual(ledger['prefix_steps_completed'], 2400)
        self.assertEqual(ledger['continuation_steps_completed'], 24000)
        self.assertEqual(ledger['free_policy_steps_completed'], 1600)
        self.assertEqual(ledger['resets_completed'], 81)
        self.assertEqual(ledger['actor_rows_completed'], 25282)

    def test_fixed_actions_and_float32_interior_edges(self):
        obs = np.zeros((2, 14), np.float32)
        policy = ToyPolicy(constant=.25)
        ledger = ref.new_ledger()
        bank = ref.candidate_banks(policy, obs, ledger)
        self.assertEqual(bank.shape, (2, 6, 4))
        self.assertEqual(bank.dtype, np.float32)
        np.testing.assert_array_equal(bank[0, :, 1:], np.tile([np.float32(.1), np.float32(-.2), np.float32(.3)], (6, 1)))
        self.assertTrue(np.all(np.abs(bank[:, 2:5, 0].astype(np.float64)+ref.KNEE) <= .05))
        self.assertFalse(np.any(np.abs(bank[:, [1, 5], 0].astype(np.float64)+ref.KNEE) <= .05))
        self.assertEqual(ledger["actor_rows_completed"], 2)

    def test_exact_remaining_time_first_action_and_discount(self):
        states = np.zeros((2, 13), np.float64)
        states[:, 0] = 2.
        times = np.asarray([98, 99])
        banks = np.zeros((2, 6, 4), np.float32)
        banks[..., 0] = .25
        policy = ToyPolicy()
        ledger = ref.new_ledger()
        values = ref.continuation_values(core(), policy, states, times, banks, ledger)
        # t98 candidate reward2.25, then actor sees remaining1/100 at t99.
        expected = 2.25+ref.GAMMA*(2.25+float(np.float32(.01)))
        np.testing.assert_array_equal(values[1], np.full(6, 2.25))
        np.testing.assert_allclose(values[0], expected, rtol=0, atol=1e-14)
        self.assertEqual(len(policy.seen), 1)
        np.testing.assert_array_equal(policy.seen[0][:, -1], np.full(6, np.float32(.01)))
        self.assertEqual(ledger["continuation_steps_completed"], 18)
        self.assertEqual(ledger["transition_steps_attempted"], ledger["transition_steps_completed"])
        np.testing.assert_array_equal(states[:, 0], [2., 2.])

    def test_synthetic_time_parity_at_interior_and_last_step(self):
        ledger = ref.new_ledger()
        result = ref.runtime_parity(core(), np.zeros(13, np.float64), ledger)
        self.assertEqual([x["t_before"] for x in result["checks"]], [0, 25, 50, 99])
        self.assertEqual(ledger["transition_steps_completed"], 104)
        self.assertEqual(ledger["resets_completed"], 1)
        self.assertEqual(ToyEnv.created[0].t, 100)
        self.assertTrue(ToyEnv.created[0].closed)

    def test_reconstructs_prefix_and_checks_each_observation(self):
        actions = np.zeros((1, 100, 4), np.float32)
        actions[0, :, 0] = np.float32(.1)
        observations = np.empty((1, 101, 14), np.float32)
        env = ToyEnv(task_semantics="finite_horizon")
        observations[0, 0], _ = env.reset(1)
        for t in range(100):
            observations[0, t+1] = env.step(actions[0, t])[0]
        ledger = ref.new_ledger()
        states = ref.reconstruct_states(core(), np.asarray([1]), actions, observations, np.asarray([25]), ledger)
        self.assertEqual(states.dtype, np.float64)
        self.assertEqual(states[0, 0], float(np.float32(.1))*25)
        self.assertEqual(ledger["prefix_steps_completed"], 25)
        observations[0, 3, 0] += 1
        with self.assertRaisesRegex(ValueError, "prefix observation differs"):
            ref.reconstruct_states(core(), np.asarray([1]), actions, observations, np.asarray([25]), ref.new_ledger())

    def test_free_actor_exposure_is_unforced_and_episode_counted(self):
        ledger = ref.new_ledger()
        result = ref.free_policy_rollouts(core(), ToyPolicy(constant=-ref.KNEE), np.asarray([1, 2]), ledger)
        self.assertEqual(result["hole_entries"], 200)
        self.assertEqual(result["episodes_with_hole_entry"], 2)
        self.assertEqual(result["hole_entries_per_episode"], [100, 100])
        self.assertEqual(ledger["free_policy_steps_completed"], 200)
        self.assertEqual(ledger["actor_rows_completed"], 200)
        self.assertTrue(all(env.closed for env in ToyEnv.created))

    def test_failures_are_charged_before_dispatch(self):
        def fail(state, action, physics):
            raise RuntimeError("synthetic failure")
        ledger = ref.new_ledger()
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            ref.continuation_values(core(fail), ToyPolicy(), np.zeros((1, 13), np.float64),
                np.asarray([99]), np.zeros((1, 6, 4), np.float32), ledger)
        self.assertEqual(ledger["transition_steps_attempted"], 6)
        self.assertEqual(ledger["transition_steps_completed"], 0)

    def test_caps_deadline_and_storage_rejected_before_work(self):
        states = np.zeros((1, 13), np.float64)
        times = np.asarray([99]); banks = np.zeros((1, 6, 4), np.float32)
        for ledger, deadline, exception in ((ref.new_ledger(5), None, ValueError),
                (ref.new_ledger(), time.monotonic()-1, TimeoutError)):
            with self.assertRaises(exception):
                ref.continuation_values(core(), ToyPolicy(), states, times, banks, ledger, deadline=deadline)
            self.assertEqual(ledger["transition_steps_attempted"], 0)
        with self.assertRaisesRegex(ValueError, "query states"):
            ref.continuation_values(core(), ToyPolicy(), states.astype(np.float32), times, banks, ref.new_ledger())
        with self.assertRaises(ValueError):
            ref.deployed_actions(np.asarray([[1.01, 0, 0, 0]]), 1)


if __name__ == "__main__":
    unittest.main()
