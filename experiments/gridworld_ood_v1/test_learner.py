"""Tests for the JAX learner: every equation against an independent NumPy version,
gradient ownership, update order, and the matching of arms.

    python -B -m unittest experiments/gridworld_ood_v1/test_learner.py -v
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))          # repository root, for the calibration package
sys.dont_write_bytecode = True

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

import learner as L  # noqa: E402

ARCHIVE = HERE.parents[1] / "analysis/gridworld_stress_20261107/archive/td3_gridworld"


# ---- independent NumPy versions of the archived equations (float64) ----------------------------
def np_mlp(params, x):
    for layer in params[:-1]:
        x = np.maximum(x @ np.asarray(layer["w"], float) + np.asarray(layer["b"], float), 0.0)
    return x @ np.asarray(params[-1]["w"], float) + np.asarray(params[-1]["b"], float)


def np_actor(params, obs):
    return np.tanh(np_mlp(params, obs))


def np_critic(params, obs, action):
    return np_mlp(params, np.concatenate([obs, action], axis=-1))[:, 0]


def np_eta(scale, obs, action):
    p = scale["params"]
    x = np.concatenate([obs / (0.999 + 0.001), action], axis=-1)
    for name in ("Dense_0", "Dense_1"):
        x = np.maximum(x @ np.asarray(p[name]["kernel"], float) + np.asarray(p[name]["bias"], float), 0.0)
    out = (x @ np.asarray(p["Dense_2"]["kernel"], float) + np.asarray(p["Dense_2"]["bias"], float))[:, 0]
    return np.logaddexp(0.0, out) + 0.01


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_batch(obs_dim, n=32, seed=0):
    rng = np.random.default_rng(seed)
    return {"obs": rng.normal(size=(n, obs_dim)).astype(np.float32),
            "next_obs": rng.normal(size=(n, obs_dim)).astype(np.float32),
            "action": rng.uniform(-1, 1, size=(n, 4)).astype(np.float32),
            "reward": rng.normal(size=n).astype(np.float32),
            "done": (rng.random(n) < 0.3).astype(np.float32),
            "noise": rng.normal(size=(n, 4)).astype(np.float32)}


def tiny_data(n_rows=96, seed=0):
    rng = np.random.default_rng(seed)
    return {"s": jnp.asarray(rng.integers(0, 100, n_rows), jnp.int32),
            "a": jnp.asarray(rng.uniform(-1, 1, (n_rows, 4)), jnp.float32),
            "r": jnp.asarray(rng.normal(size=n_rows), jnp.float32),
            "s2": jnp.asarray(rng.integers(0, 100, n_rows), jnp.int32),
            "done": jnp.asarray(rng.random(n_rows) < 0.2, jnp.float32)}


def leaves_equal(a, b):
    return all(np.array_equal(np.asarray(x), np.asarray(y))
               for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


def ready_reference(state, threshold=2.0, unit=1.3, floor=-1e9, cap=0.0, level=0.0, level_support=0.0):
    """A reference as a refresh would set it. With the default floor no command counts as unsupported."""
    return {"scale": state["scale"], "unit": jnp.float32(unit), "threshold": jnp.float32(threshold),
            "ready": jnp.asarray(True), "support": state["support"], "support_floor": jnp.float32(floor),
            "width_cap": jnp.float32(cap), "width_ref": jnp.float32(level),
            "width_ref_support": jnp.float32(level_support)}


def np_width(state, obs, action, threshold=2.0, unit=1.3):
    return threshold * np.maximum(np_eta(state["scale"], obs, action), 1e-6) * unit


def np_raised(state, obs, action, floor, cap, threshold=2.0, unit=1.3):
    """The support-aware width: the original width, raised by (cap - width) per decade below the floor."""
    base = np_width(state, obs, action, threshold, unit)
    decades = np.maximum(floor - np_critic(state["support"], obs, action), 0.0) / np.log(10.0)
    return base + decades * np.maximum(cap - base, 0.0)


ALL_CELLS = L.one_hot_obs(jnp.arange(100))


def active_reference(state, cal, obs, commands=None):
    """A reference under which every one of the three changes does something. At the given
    observations and commands (the policy's own, unless given): half of the commands count as
    unsupported, and half of the widths lie above the one-sided level."""
    obs = jnp.asarray(obs, jnp.float32)
    commands = L.actor_apply(state["actor"], obs) if commands is None else commands
    floor = float(jnp.median(L.support_logit(state["support"], obs, commands)))
    ref = ready_reference(state, 5.0, 2.0, floor=floor, cap=400.0)
    level = float(jnp.median(L.width(cal, ref, obs, commands)))
    level_support = float(jnp.median(L.consumed_width(cal, ref, obs, commands, True)))
    return ready_reference(state, 5.0, 2.0, floor=floor, cap=400.0, level=level, level_support=level_support)


class Equations(unittest.TestCase):
    def setUp(self):
        self.state = L.init_state(3, 2)
        self.cal = L.calibrator(2)
        self.b = make_batch(2)

    def test_bellman_target_matches_numpy_and_stops_at_terminals(self):
        b, s = self.b, self.state
        got = L.bellman_target(s["actor_t"], s["q1_t"], s["q2_t"], b["next_obs"], b["reward"], b["done"], b["noise"])
        noise = np.clip(b["noise"].astype(float) * 0.2, -0.5, 0.5)
        next_action = np.clip(np_actor(s["actor_t"], b["next_obs"].astype(float)) + noise, -1, 1)
        next_q = np.minimum(np_critic(s["q1_t"], b["next_obs"].astype(float), next_action),
                            np_critic(s["q2_t"], b["next_obs"].astype(float), next_action))
        expected = b["reward"] + (1 - b["done"]) * 0.95 * next_q
        self.assertEqual(got.shape, (32,))
        np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(np.asarray(got)[b["done"] == 1], b["reward"][b["done"] == 1], rtol=0, atol=0)

    def test_critic_loss_matches_numpy(self):
        b, s = self.b, self.state
        target = np.random.default_rng(1).normal(size=32).astype(np.float32)
        got = L.critic_loss((s["q1"], s["q2"]), b["obs"], b["action"], target)
        q1 = np_critic(s["q1"], b["obs"].astype(float), b["action"].astype(float))
        q2 = np_critic(s["q2"], b["obs"].astype(float), b["action"].astype(float))
        np.testing.assert_allclose(got, np.mean((q1 - target) ** 2) + np.mean((q2 - target) ** 2), rtol=2e-5)

    def test_width_matches_numpy_and_is_zero_until_a_reference_is_ready(self):
        b, s = self.b, self.state
        np.testing.assert_array_equal(L.width(self.cal, s["reference"], b["obs"], b["action"]), np.zeros(32))
        ref = ready_reference(s)
        got = L.width(self.cal, ref, b["obs"], b["action"])
        eta = np_eta(s["scale"], b["obs"].astype(float), b["action"].astype(float))
        np.testing.assert_allclose(got, 2.0 * np.maximum(eta, 1e-6) * 1.3, rtol=2e-5)

    def test_actor_objective_matches_numpy_for_the_baseline_and_for_bca(self):
        b, s = self.b, self.state
        ref = ready_reference(s)
        obs, act = b["obs"].astype(float), b["action"].astype(float)
        pi = np_actor(s["actor"], obs)
        q = np_critic(s["q1"], obs, pi)
        c = 2.5 / (np.mean(np.abs(q)) + 1e-7)
        native = -c * np.mean(q) + np.mean((pi - act) ** 2)
        w = 2.0 * np.maximum(np_eta(s["scale"], obs, pi), 1e-6) * 1.3
        for kappa, expected in ((0.0, native), (3.0, native + c * 3.0 * np.mean(w))):
            got = L.actor_objective(s["actor"], s["q1"], self.cal, ref, b["obs"], b["obs"], b["action"], kappa)
            np.testing.assert_allclose(got, expected, rtol=3e-5)

    def test_support_objective_matches_numpy(self):
        b, s = self.b, self.state
        random_action = np.random.default_rng(7).uniform(-1, 1, (32, 4)).astype(np.float32)
        got = L.support_objective(s["support"], b["obs"], b["action"], random_action)
        obs = b["obs"].astype(float)
        recorded = np_critic(s["support"], obs, b["action"].astype(float))
        random = np_critic(s["support"], obs, random_action.astype(float))
        expected = np.mean(np.logaddexp(0.0, -recorded)) + np.mean(np.logaddexp(0.0, random))
        np.testing.assert_allclose(got, expected, rtol=2e-5)

    def test_support_aware_width_matches_numpy_and_leaves_supported_commands_alone(self):
        b, s = self.b, self.state
        obs, act = b["obs"].astype(float), b["action"].astype(float)
        logit = np_critic(s["support"], obs, act)
        floor = float(np.median(logit))
        ref = ready_reference(s, floor=floor, cap=9.0)
        original = np.asarray(L.width(self.cal, ref, b["obs"], b["action"]))
        got = np.asarray(L.consumed_width(self.cal, ref, b["obs"], b["action"], True))
        np.testing.assert_allclose(got, np_raised(s, obs, act, floor, 9.0), rtol=3e-5)
        supported, unsupported = logit > floor + 1e-4, logit < floor - 1e-4
        self.assertGreater(int(supported.sum()), 8)
        self.assertGreater(int(unsupported.sum()), 8)
        np.testing.assert_array_equal(got[supported], original[supported])
        self.assertTrue(np.all(got[unsupported] > original[unsupported]))
        # Switched off, or before any refresh, it is exactly the original width.
        np.testing.assert_array_equal(L.consumed_width(self.cal, ref, b["obs"], b["action"], False), original)
        np.testing.assert_array_equal(L.consumed_width(self.cal, s["reference"], b["obs"], b["action"], True),
                                      np.zeros(32))

    def test_support_aware_width_reaches_its_cap_one_decade_below_the_floor(self):
        b, s = self.b, self.state
        logit = float(L.support_logit(s["support"], b["obs"], b["action"])[0])
        ref = ready_reference(s, floor=logit + np.log(10.0), cap=9.0)
        got = L.consumed_width(self.cal, ref, b["obs"], b["action"], True)
        np.testing.assert_allclose(float(got[0]), 9.0, rtol=1e-5)
        deeper = ready_reference(s, floor=logit + 2 * np.log(10.0), cap=9.0)       # two decades: past the cap
        base = float(L.width(self.cal, ref, b["obs"], b["action"])[0])
        np.testing.assert_allclose(float(L.consumed_width(self.cal, deeper, b["obs"], b["action"], True)[0]),
                                   base + 2 * (9.0 - base), rtol=1e-5)

    def test_one_sided_penalty_counts_only_width_above_the_reference_level(self):
        b, s = self.b, self.state
        obs, act = b["obs"].astype(float), b["action"].astype(float)
        floor = float(np.median(np_critic(s["support"], obs, act)))
        base, raised = np_width(s, obs, act), np_raised(s, obs, act, floor, 9.0)
        level, level_support = float(np.median(base)), float(np.median(raised))
        ref = ready_reference(s, floor=floor, cap=9.0, level=level, level_support=level_support)
        cases = {(False, False): base, (True, False): raised,
                 (False, True): np.maximum(base - level, 0.0),
                 (True, True): np.maximum(raised - level_support, 0.0)}
        for (use_support, one_sided), expected in cases.items():
            got = np.asarray(L.penalty(self.cal, ref, b["obs"], b["action"], use_support, one_sided))
            np.testing.assert_allclose(got, expected, rtol=3e-5, atol=2e-6)
            if one_sided:
                self.assertEqual(float(got.min()), 0.0)
                self.assertTrue(8 < int((got > 0).sum()) < 24)

    def test_bellman_target_subtracts_a_next_state_penalty_but_never_after_a_terminal(self):
        b, s = self.b, self.state
        args = (s["actor_t"], s["q1_t"], s["q2_t"], b["next_obs"], b["reward"], b["done"], b["noise"])
        cost = np.random.default_rng(8).uniform(0, 2, 32).astype(np.float32)
        plain, got = np.asarray(L.bellman_target(*args)), np.asarray(L.bellman_target(*args, cost))
        np.testing.assert_allclose(got, plain - (1 - b["done"]) * 0.95 * cost, rtol=2e-5, atol=2e-6)
        np.testing.assert_array_equal(got[b["done"] == 1], b["reward"][b["done"] == 1])

    def test_trained_target_is_the_plain_target_unless_the_arm_puts_width_in_the_critic(self):
        b, s = self.b, self.state
        next_obs = b["next_obs"].astype(float)
        nets = (s["actor_t"], s["q1_t"], s["q2_t"])
        plain = np.asarray(L.bellman_target(*nets, b["next_obs"], b["reward"], b["done"], b["noise"]))
        noise = np.clip(b["noise"].astype(float) * 0.2, -0.5, 0.5)
        next_action = np.clip(np_actor(s["actor_t"], next_obs) + noise, -1, 1)
        floor = float(np.median(np_critic(s["support"], next_obs, next_action)))
        raised = np_raised(s, next_obs, next_action, floor, 9.0)
        level = float(np.median(raised))
        ref = ready_reference(s, floor=floor, cap=9.0, level=0.3, level_support=level)
        for mode in range(len(L.ARM_NAMES)):
            got = np.asarray(L.trained_target(self.cal, L.switches(mode), ref, *nets, b["next_obs"], b["next_obs"],
                                              b["reward"], b["done"], b["noise"]))
            if mode in (L.FULL, L.FULL_SHUFFLED):
                expected = plain - (1 - b["done"]) * 0.95 * 3.0 * np.maximum(raised - level, 0.0)
                np.testing.assert_allclose(got, expected, rtol=5e-5, atol=5e-6)
                self.assertGreater(float(np.abs(got - plain).max()), 0.1)
            else:
                np.testing.assert_array_equal(got, plain)

    def test_actor_objective_matches_numpy_for_every_combination_of_the_changes(self):
        b, s = self.b, self.state
        obs, act = b["obs"].astype(float), b["action"].astype(float)
        pi = np_actor(s["actor"], obs)
        q = np_critic(s["q1"], obs, pi)
        c = 2.5 / (np.mean(np.abs(q)) + 1e-7)
        native = -c * np.mean(q) + np.mean((pi - act) ** 2)
        floor = float(np.median(np_critic(s["support"], obs, pi)))
        base, raised = np_width(s, obs, pi), np_raised(s, obs, pi, floor, 9.0)
        level, level_support = float(np.median(base)), float(np.median(raised))
        ref = ready_reference(s, floor=floor, cap=9.0, level=level, level_support=level_support)
        cases = {(False, False): base, (True, False): raised,
                 (False, True): np.maximum(base - level, 0.0),
                 (True, True): np.maximum(raised - level_support, 0.0)}
        for (use_support, one_sided), cost in cases.items():
            got = L.actor_objective(s["actor"], s["q1"], self.cal, ref, b["obs"], b["obs"], b["action"], 3.0,
                                    use_support, one_sided)
            np.testing.assert_allclose(got, native + c * 3.0 * np.mean(cost), rtol=5e-5)

    def test_support_penalty_alone_is_cap_times_decades_and_zero_at_supported_commands(self):
        b, s = self.b, self.state
        obs, act = b["obs"].astype(float), b["action"].astype(float)
        logit = np_critic(s["support"], obs, act)
        floor = float(np.median(logit))
        ref = ready_reference(s, floor=floor, cap=9.0)
        got = np.asarray(L.consumed_width(self.cal, ref, b["obs"], b["action"], True, True))
        np.testing.assert_allclose(got, np.maximum(floor - logit, 0.0) / np.log(10.0) * 9.0, rtol=3e-5, atol=1e-5)
        np.testing.assert_array_equal(got[logit > floor + 1e-4], 0.0)
        self.assertTrue(np.all(got[logit < floor - 1e-4] > 0))
        # It is what the support-aware width adds when the calibrated width is zero, and nothing else.
        np.testing.assert_array_equal(L.penalty(self.cal, ref, b["obs"], b["action"], True, False, True), got)
        np.testing.assert_array_equal(
            L.consumed_width(self.cal, s["reference"], b["obs"], b["action"], True, True), np.zeros(32))
        with_base = np.asarray(L.consumed_width(self.cal, ref, b["obs"], b["action"], True, False))
        self.assertTrue(np.all(with_base > got))

    def test_actor_objective_matches_numpy_for_the_support_penalty_alone(self):
        b, s = self.b, self.state
        obs, act = b["obs"].astype(float), b["action"].astype(float)
        pi = np_actor(s["actor"], obs)
        q = np_critic(s["q1"], obs, pi)
        c = 2.5 / (np.mean(np.abs(q)) + 1e-7)
        native = -c * np.mean(q) + np.mean((pi - act) ** 2)
        logit = np_critic(s["support"], obs, pi)
        floor = float(np.median(logit))
        cost = np.maximum(floor - logit, 0.0) / np.log(10.0) * 9.0
        ref = ready_reference(s, floor=floor, cap=9.0, level=0.7, level_support=0.9)
        got = L.actor_objective(s["actor"], s["q1"], self.cal, ref, b["obs"], b["obs"], b["action"], 3.0,
                                True, False, True)
        np.testing.assert_allclose(got, native + c * 3.0 * np.mean(cost), rtol=5e-5)

    def test_width_can_read_its_calibrated_part_and_its_support_part_at_different_states(self):
        b, s = self.b, self.state
        obs, other, act = b["obs"].astype(float), b["next_obs"].astype(float), b["action"].astype(float)
        logit = np_critic(s["support"], obs, act)                 # support score at the true states
        floor = float(np.median(logit))
        ref = ready_reference(s, floor=floor, cap=9.0)
        base = np_width(s, other, act)                            # calibrated width at the other states
        expected = base + np.maximum(floor - logit, 0.0) / np.log(10.0) * np.maximum(9.0 - base, 0.0)
        got = np.asarray(L.consumed_width(self.cal, ref, b["next_obs"], b["action"], True, False, b["obs"]))
        np.testing.assert_allclose(got, expected, rtol=3e-5)
        same_state = np.asarray(L.consumed_width(self.cal, ref, b["obs"], b["action"], True, False, b["obs"]))
        np.testing.assert_array_equal(same_state, L.consumed_width(self.cal, ref, b["obs"], b["action"], True))
        self.assertGreater(float(np.abs(got - same_state).max()), 1e-3)
        # The actor objective with the two states apart, against NumPy.
        pi = np_actor(s["actor"], obs)
        q = np_critic(s["q1"], obs, pi)
        c = 2.5 / (np.mean(np.abs(q)) + 1e-7)
        logit_pi = np_critic(s["support"], obs, pi)
        ref_pi = ready_reference(s, floor=float(np.median(logit_pi)), cap=9.0)
        base_pi = np_width(s, other, pi)
        cost = base_pi + np.maximum(float(np.median(logit_pi)) - logit_pi, 0.0) / np.log(10.0) * np.maximum(9.0 - base_pi, 0.0)
        got = L.actor_objective(s["actor"], s["q1"], self.cal, ref_pi, b["obs"], b["next_obs"], b["action"], 3.0,
                                True, False, False, b["obs"])
        np.testing.assert_allclose(got, -c * np.mean(q) + np.mean((pi - act) ** 2) + c * 3.0 * np.mean(cost), rtol=5e-5)

    def test_arm_switches_are_the_table_in_the_module_docstring(self):
        expected = {"baseline": "", "bca": "p", "shuffled_width": "ps", "support": "pa", "support_shuffled": "pas",
                    "support_onesided": "pao", "support_onesided_shuffled": "paos", "full": "paoc",
                    "full_shuffled": "paocs", "onesided": "po", "onesided_shuffled": "pos",
                    "support_only": "paz", "support_only_shuffled": "pazs", "support_blind_width": "pab"}
        self.assertEqual(set(L.ARM_NAMES.values()), set(expected))
        for number, name in L.ARM_NAMES.items():
            arm = L.switches(number)
            got = ("p" * (float(arm["kappa"]) == 3.0) + "a" * bool(arm["support"]) + "z" * bool(arm["no_base"])
                   + "b" * bool(arm["blind_base"])
                   + "o" * bool(arm["one_sided"]) + "c" * bool(arm["critic"]) + "s" * bool(arm["shuffled"]))
            self.assertEqual(got, expected[name], name)
            self.assertIn(float(arm["kappa"]), (0.0, 3.0))
        for variant, control in L.CONTROL_OF.items():       # a control differs from its variant only by the shuffle
            a, c = L.switches(L.ARM_NUMBERS[variant]), L.switches(L.ARM_NUMBERS[control])
            self.assertTrue(bool(c["shuffled"]) and not bool(a["shuffled"]))
            for name in ("kappa", "support", "one_sided", "critic", "no_base"):
                self.assertEqual(float(a[name]), float(c[name]), (variant, name))

    def test_scale_objective_and_unit_update_match_numpy(self):
        b, s = self.b, self.state
        rng = np.random.default_rng(2)
        target, q = rng.normal(size=32).astype(np.float32), rng.normal(size=32).astype(np.float32)
        prior = rng.exponential(size=32)
        prior = (prior / prior.sum()).astype(np.float32)
        unit = np.float32(0.7)
        got = L.scale_objective(s["scale"], self.cal, b["obs"], b["action"], target, q, unit, prior)
        eta = np_eta(s["scale"], b["obs"].astype(float), b["action"].astype(float))
        y, qq = target.astype(float) / 0.7, q.astype(float) / 0.7
        coverage = sigmoid(20 * (y - (qq - eta))) * sigmoid(20 * ((qq + eta) - y))
        expected = (np.sum(prior * coverage) - 0.9) ** 2 + 0.005 * np.sum(prior * eta ** 2)
        np.testing.assert_allclose(got, expected, rtol=1e-4)
        np.testing.assert_allclose(L.unit_update(unit, target, q),
                                   0.99 * 0.7 + 0.01 * max(np.std(target.astype(float) - q.astype(float)), 1e-6),
                                   rtol=1e-5)

    def test_adam_matches_the_pytorch_formula(self):
        rng = np.random.default_rng(3)
        param = rng.normal(size=5)
        opt = optax.adam(L.HOST_LR)
        p, st = jnp.asarray(param, jnp.float32), None
        st = opt.init(p)
        m = v = np.zeros(5)
        for t in range(1, 4):
            g = rng.normal(size=5)
            updates, st = opt.update(jnp.asarray(g, jnp.float32), st, p)
            p = optax.apply_updates(p, updates)
            m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g
            param = param - 3e-4 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            np.testing.assert_allclose(p, param, rtol=1e-5, atol=1e-7)


class GradientOwnership(unittest.TestCase):
    def setUp(self):
        self.state = L.init_state(4, 2)
        self.cal = L.calibrator(2)
        self.b = make_batch(2, seed=5)
        self.ref = ready_reference(self.state)

    def objective(self, actor, kappa=3.0, reference=None, q1=None):
        b, s = self.b, self.state
        return L.actor_objective(actor, s["q1"] if q1 is None else q1, self.cal,
                                 self.ref if reference is None else reference,
                                 b["obs"], b["obs"], b["action"], kappa)

    def test_normalization_coefficient_is_a_constant_but_q_is_differentiated(self):
        b, s = self.b, self.state
        q = L.critic_apply(s["q1"], b["obs"], L.actor_apply(s["actor"], b["obs"]))
        c = L.ALPHA / (jnp.mean(jnp.abs(q)) + 1e-7)

        def with_fixed_coefficient(actor):
            pi = L.actor_apply(actor, b["obs"])
            return (-c * jnp.mean(L.critic_apply(s["q1"], b["obs"], pi)) + jnp.mean(jnp.square(pi - b["action"]))
                    + c * 3.0 * jnp.mean(L.width(self.cal, self.ref, b["obs"], pi)))

        def cloning_only(actor):
            return jnp.mean(jnp.square(L.actor_apply(actor, b["obs"]) - b["action"]))

        got = jax.grad(self.objective)(s["actor"])
        for g, e in zip(jax.tree_util.tree_leaves(got),
                        jax.tree_util.tree_leaves(jax.grad(with_fixed_coefficient)(s["actor"]))):
            np.testing.assert_allclose(g, e, rtol=1e-4, atol=1e-7)
        difference = sum(float(jnp.abs(g - e).sum()) for g, e in zip(
            jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(jax.grad(cloning_only)(s["actor"]))))
        self.assertGreater(difference, 1e-3)       # the Q and width paths really contribute

    def test_actor_gradient_agrees_with_finite_differences_of_the_numpy_version(self):
        """Finite differences are taken in float64 on the independent NumPy equations, with the
        coefficient c held fixed, so this checks both the autodiff and the detachment of c."""
        b, s = self.b, self.state
        grad = jax.grad(self.objective)(s["actor"])
        obs, act = b["obs"].astype(float), b["action"].astype(float)
        base = [{k: np.asarray(v, float) for k, v in layer.items()} for layer in s["actor"]]
        c = 2.5 / (np.mean(np.abs(np_critic(s["q1"], obs, np_actor(base, obs)))) + 1e-7)

        def frozen_c(actor):
            pi = np_actor(actor, obs)
            w = 2.0 * np.maximum(np_eta(s["scale"], obs, pi), 1e-6) * 1.3
            return -c * np.mean(np_critic(s["q1"], obs, pi)) + np.mean((pi - act) ** 2) + c * 3.0 * np.mean(w)

        eps = 1e-6
        for layer, name, index in ((2, "b", (1,)), (2, "b", (3,)), (2, "w", (5, 2)), (1, "w", (3, 9)), (0, "w", (1, 7))):
            def bumped(sign):
                actor = [{k: v.copy() for k, v in l.items()} for l in base]
                actor[layer][name][index] += sign * eps
                return frozen_c(actor)
            numeric = (bumped(+1) - bumped(-1)) / (2 * eps)
            self.assertAlmostEqual(float(grad[layer][name][index]), numeric, delta=1e-5 + 2e-3 * abs(numeric))

    def test_reference_and_critic_parameters_get_no_actor_update_but_the_action_path_is_live(self):
        s = self.state
        to_reference = jax.grad(lambda r: self.objective(s["actor"], reference=r), allow_int=True)(
            {k: v for k, v in self.ref.items() if k != "ready"} | {"ready": self.ref["ready"]})
        for leaf in jax.tree_util.tree_leaves({k: v for k, v in to_reference.items() if k != "ready"}):
            self.assertEqual(float(jnp.abs(jnp.asarray(leaf, jnp.float32)).sum()), 0.0)
        action_grad = jax.grad(lambda a: jnp.sum(L.width(self.cal, self.ref, self.b["obs"], a)))(
            jnp.asarray(self.b["action"]))
        self.assertGreater(float(jnp.abs(action_grad).sum()), 0.0)

    def test_no_actor_gradient_reaches_any_frozen_quantity_of_the_changed_method(self):
        b, s = self.b, self.state
        ref = active_reference(s, self.cal, b["obs"])
        frozen = {k: v for k, v in ref.items() if k != "ready"}

        def objective(frozen_part, actor):
            return L.actor_objective(actor, s["q1"], self.cal, dict(frozen_part, ready=ref["ready"]),
                                     b["obs"], b["obs"], b["action"], 3.0, True, True)

        for leaf in jax.tree_util.tree_leaves(jax.grad(objective)(frozen, s["actor"])):
            self.assertEqual(float(jnp.abs(leaf).sum()), 0.0)
        # ... while the same objective does move the actor, and differently from the original penalty.
        changed = jax.grad(objective, argnums=1)(frozen, s["actor"])
        original = jax.grad(lambda a: self.objective(a, reference=ref))(s["actor"])
        self.assertFalse(leaves_equal(changed, original))

    def test_the_command_gradient_is_live_inside_a_hole_and_zero_at_supported_low_width_commands(self):
        b, s = self.b, self.state
        action = jnp.asarray(b["action"])
        logit = np.asarray(L.support_logit(s["support"], b["obs"], action))
        floor = float(np.median(logit))
        plain = ready_reference(s, floor=floor, cap=400.0)
        level = float(jnp.max(L.width(self.cal, plain, b["obs"], action))) + 1e-3     # above every original width
        ref = ready_reference(s, floor=floor, cap=400.0, level_support=level)
        excess = np.asarray(L.penalty(self.cal, ref, b["obs"], action, True, True))
        grad = jax.grad(lambda a: jnp.sum(L.penalty(self.cal, ref, b["obs"], a, True, True)))(action)
        per_row = np.abs(np.asarray(grad)).sum(axis=1)
        np.testing.assert_array_equal(per_row[logit > floor + 1e-4], 0.0)      # supported: nothing to push against
        np.testing.assert_array_equal(excess[logit > floor + 1e-4], 0.0)
        self.assertGreater(int((excess > 0).sum()), 3)
        self.assertTrue(np.all(per_row[excess > 0] > 0))                       # unsupported: a slope back to the data

    def test_support_penalty_alone_pushes_only_inside_a_hole_and_updates_nothing_frozen(self):
        b, s = self.b, self.state
        action = jnp.asarray(b["action"])
        logit = np.asarray(L.support_logit(s["support"], b["obs"], action))
        ref = ready_reference(s, floor=float(np.median(logit)), cap=9.0)
        floor = float(ref["support_floor"])
        grad = jax.grad(lambda a: jnp.sum(L.penalty(self.cal, ref, b["obs"], a, True, False, True)))(action)
        per_row = np.abs(np.asarray(grad)).sum(axis=1)
        np.testing.assert_array_equal(per_row[logit > floor + 1e-4], 0.0)
        self.assertTrue(np.all(per_row[logit < floor - 1e-4] > 0))
        frozen = {k: v for k, v in ref.items() if k != "ready"}
        grads = jax.grad(lambda f: L.actor_objective(s["actor"], s["q1"], self.cal, dict(f, ready=ref["ready"]),
                                                     b["obs"], b["obs"], b["action"], 3.0, True, False, True))(frozen)
        for leaf in jax.tree_util.tree_leaves(grads):
            self.assertEqual(float(jnp.abs(leaf).sum()), 0.0)

    def test_width_changes_the_actor_gradient_only_when_kappa_is_positive(self):
        s = self.state
        g0 = jax.grad(lambda a: self.objective(a, kappa=0.0))(s["actor"])
        g0_other_ref = jax.grad(lambda a: self.objective(a, kappa=0.0, reference=ready_reference(s, 9.0, 4.0)))(s["actor"])
        self.assertTrue(leaves_equal(g0, g0_other_ref))
        g3 = jax.grad(lambda a: self.objective(a, kappa=3.0))(s["actor"])
        self.assertFalse(leaves_equal(g0, g3))


class UpdateSchedule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = tiny_data()
        cls.step = staticmethod(jax.jit(L.make_step("one_hot", 96)))
        cls.key = jax.random.PRNGKey(11)
        cls.state = L.init_state(5, 100)

    def test_odd_update_changes_critic_and_scale_only(self):
        s0 = self.state
        s1 = self.step(s0, self.data, self.key, L.NATIVE, 1)
        self.assertFalse(leaves_equal(s0["q1"], s1["q1"]))
        self.assertFalse(leaves_equal(s0["scale"], s1["scale"]))
        self.assertNotEqual(float(s0["unit"]), float(s1["unit"]))
        for name in ("actor", "actor_t", "q1_t", "q2_t", "actor_opt"):
            self.assertTrue(leaves_equal(s0[name], s1[name]), name)
        self.assertEqual(int(s1["step"]), 1)

    def test_even_update_moves_actor_and_targets_toward_the_updated_networks(self):
        s1 = self.step(self.state, self.data, self.key, L.NATIVE, 1)
        s2 = self.step(s1, self.data, self.key, L.NATIVE, 2)
        self.assertFalse(leaves_equal(s1["actor"], s2["actor"]))
        for online, target, before in (("actor", "actor_t", s1["actor_t"]), ("q1", "q1_t", s1["q1_t"]),
                                       ("q2", "q2_t", s1["q2_t"])):
            expected = jax.tree_util.tree_map(lambda p, t: 0.995 * t + 0.005 * p, s2[online], before)
            for got, exp in zip(jax.tree_util.tree_leaves(s2[target]), jax.tree_util.tree_leaves(expected)):
                np.testing.assert_allclose(got, exp, rtol=1e-6, atol=1e-8)

    def test_the_critic_target_and_loss_never_see_the_width(self):
        ready = dict(self.state, reference=ready_reference(self.state, 5.0, 2.0))
        a = self.step(self.state, self.data, self.key, L.BCA, 1)     # odd update: no actor step
        b = self.step(ready, self.data, self.key, L.BCA, 1)
        for name in ("q1", "q2", "scale", "unit"):
            self.assertTrue(leaves_equal(a[name], b[name]), name)

    def test_baseline_arm_is_unaffected_by_any_reference(self):
        ready = dict(self.state, reference=ready_reference(self.state, 5.0, 2.0))
        a, b = self.state, ready
        for t in range(1, 7):
            a = self.step(a, self.data, self.key, L.NATIVE, t)
            b = self.step(b, self.data, self.key, L.NATIVE, t)
        for name in a:
            if name != "reference":
                self.assertTrue(leaves_equal(a[name], b[name]), name)

    def test_bca_and_shuffled_arms_differ_from_the_baseline_only_through_the_actor(self):
        ready = dict(self.state, reference=ready_reference(self.state, 5.0, 2.0))
        out = {mode: self.step(self.step(ready, self.data, self.key, mode, 1), self.data, self.key, mode, 2)
               for mode in (L.NATIVE, L.BCA, L.SHUFFLED)}
        for mode in (L.BCA, L.SHUFFLED):
            self.assertFalse(leaves_equal(out[L.NATIVE]["actor"], out[mode]["actor"]))
            for name in ("q1", "q2", "scale", "unit", "critic_opt"):
                self.assertTrue(leaves_equal(out[L.NATIVE][name], out[mode][name]), name)
        self.assertFalse(leaves_equal(out[L.BCA]["actor"], out[L.SHUFFLED]["actor"]))

    def test_original_arms_reproduce_the_outputs_recorded_before_the_three_changes(self):
        """test_learner_regression.json was written by the learner as it stood before Amendment 3."""
        saved = json.loads((HERE / "test_learner_regression.json").read_text())
        probe_obs = L.one_hot_obs(jnp.arange(100))
        probe_action = jnp.asarray(saved["probe_actions"], jnp.float32)
        start = dict(self.state, reference=ready_reference(self.state, 5.0, 2.0))
        for mode in (L.NATIVE, L.BCA, L.SHUFFLED):
            s = start
            for t in range(1, 9):
                s = self.step(s, self.data, self.key, mode, t)
            want = saved[L.ARM_NAMES[mode]]
            got = {"commands": L.actor_apply(s["actor"], probe_obs),
                   "q1": L.critic_apply(s["q1"], probe_obs, probe_action),
                   "q2_target": L.critic_apply(s["q2_t"], probe_obs, probe_action),
                   "scale_out": L.calibrator(100).apply(s["scale"], probe_obs, probe_action)}
            for name, value in got.items():
                np.testing.assert_allclose(value, want[name], rtol=0, atol=5e-5, err_msg=f"{L.ARM_NAMES[mode]} {name}")
            np.testing.assert_allclose(float(s["unit"]), want["unit"], rtol=1e-5)

    def test_before_the_first_refresh_every_arm_is_the_baseline(self):
        out = {}
        for mode in L.ARM_NAMES:
            s = self.state
            for t in range(1, 5):
                s = self.step(s, self.data, self.key, mode, t)
            out[mode] = s
        for mode in L.ARM_NAMES:
            for name in out[mode]:
                self.assertTrue(leaves_equal(out[L.NATIVE][name], out[mode][name]), (L.ARM_NAMES[mode], name))

    def test_only_the_full_arms_let_the_width_into_the_critic_and_the_scale_fit(self):
        ready = dict(self.state, reference=active_reference(self.state, L.calibrator(100), ALL_CELLS))
        out = {mode: self.step(ready, self.data, self.key, mode, 1) for mode in L.ARM_NAMES}     # odd: no actor step
        for mode in L.ARM_NAMES:
            same = mode not in (L.FULL, L.FULL_SHUFFLED)
            for name in ("q1", "q2", "scale", "unit"):
                self.assertEqual(leaves_equal(out[L.NATIVE][name], out[mode][name]), same, (L.ARM_NAMES[mode], name))
            for name in ("support", "support_opt", "actor", "actor_t"):     # the classifier is fitted alike everywhere
                self.assertTrue(leaves_equal(out[L.NATIVE][name], out[mode][name]), (L.ARM_NAMES[mode], name))
        self.assertFalse(leaves_equal(out[L.FULL]["q1"], out[L.FULL_SHUFFLED]["q1"]))
        self.assertFalse(leaves_equal(self.state["support"], out[L.NATIVE]["support"]))

    def test_every_penalized_arm_moves_the_actor_its_own_way(self):
        ready = dict(self.state, reference=active_reference(self.state, L.calibrator(100), ALL_CELLS))
        out = {mode: self.step(self.step(ready, self.data, self.key, mode, 1), self.data, self.key, mode, 2)
               for mode in L.ARM_NAMES}
        modes = sorted(L.ARM_NAMES)
        for i, a in enumerate(modes):
            for b in modes[i + 1:]:
                self.assertFalse(leaves_equal(out[a]["actor"], out[b]["actor"]), (L.ARM_NAMES[a], L.ARM_NAMES[b]))

    def test_baseline_arm_ignores_a_reference_under_which_every_change_is_active(self):
        ready = dict(self.state, reference=active_reference(self.state, L.calibrator(100), ALL_CELLS))
        a, b = self.state, ready
        for t in range(1, 7):
            a = self.step(a, self.data, self.key, L.NATIVE, t)
            b = self.step(b, self.data, self.key, L.NATIVE, t)
        for name in a:
            if name != "reference":
                self.assertTrue(leaves_equal(a[name], b[name]), name)

    def test_blind_width_arm_reduces_to_known_arms_when_one_part_of_its_width_is_switched_off(self):
        """With no command unsupported it must be the original shuffled-width control exactly.
        With the calibrated width at zero it must be the support-penalty-only arm exactly."""
        active = active_reference(self.state, L.calibrator(100), ALL_CELLS)

        def actor_after_two_updates(reference, mode):
            s = dict(self.state, reference=reference)
            for t in (1, 2):
                s = self.step(s, self.data, self.key, mode, t)
            return s["actor"]

        nothing_unsupported = dict(active, support_floor=jnp.float32(-1e9))
        self.assertTrue(leaves_equal(actor_after_two_updates(nothing_unsupported, L.SUPPORT_BLIND_WIDTH),
                                     actor_after_two_updates(nothing_unsupported, L.SHUFFLED)))
        no_calibrated_width = dict(active, threshold=jnp.float32(0.0))
        self.assertTrue(leaves_equal(actor_after_two_updates(no_calibrated_width, L.SUPPORT_BLIND_WIDTH),
                                     actor_after_two_updates(no_calibrated_width, L.SUPPORT_ONLY)))
        # With both parts on, it is none of change 1, the fully shuffled control, or those two.
        mine = actor_after_two_updates(active, L.SUPPORT_BLIND_WIDTH)
        for other in (L.SUPPORT, L.SUPPORT_SHUFFLED, L.SHUFFLED, L.SUPPORT_ONLY, L.NATIVE):
            self.assertFalse(leaves_equal(mine, actor_after_two_updates(active, other)), L.ARM_NAMES[other])

    def test_matched_arms_draw_the_same_minibatches(self):
        a = L.batch_indices(self.key, 12_345, 22_784)
        b = L.batch_indices(self.key, 12_345, 22_784)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, L.batch_indices(self.key, 12_346, 22_784)))
        self.assertTrue(0 <= int(a.min()) and int(a.max()) < 22_784)

    def test_vmapped_segment_equals_running_each_run_alone(self):
        segment = L.make_segment("one_hot", 96)
        cal = L.calibrator(100)
        singles = [L.init_state(seed, 100) for seed in (5, 6, 7, 8)]
        states = L.stack([dict(s, reference=active_reference(s, cal, ALL_CELLS)) for s in singles])
        data = L.stack([self.data, tiny_data(seed=1), tiny_data(seed=2), tiny_data(seed=3)])
        keys = jnp.stack([jax.random.PRNGKey(k) for k in (1, 2, 3, 4)])
        modes = jnp.asarray([L.NATIVE, L.BCA, L.FULL, L.SUPPORT_ONESIDED_SHUFFLED])
        steps = jnp.arange(1, 9)
        together = segment(states, data, keys, modes, steps)
        for i in range(4):
            alone = L.take(states, i)
            for t in range(1, 9):
                alone = self.step(alone, L.take(data, i), keys[i], modes[i], t)
            for got, exp in zip(jax.tree_util.tree_leaves(L.take(together, i)), jax.tree_util.tree_leaves(alone)):
                np.testing.assert_allclose(got, exp, rtol=2e-4, atol=2e-6)


class Readouts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cal = L.calibrator(100)
        cls.singles = [L.init_state(seed, 100) for seed in (5, 6, 7)]
        cls.states = L.stack([dict(s, reference=active_reference(s, cls.cal, ALL_CELLS)) for s in cls.singles])
        cls.bank = L.stack([tiny_data(seed=4), tiny_data(seed=5), tiny_data(seed=6)])
        cls.keys = jnp.stack([jax.random.PRNGKey(k) for k in (7, 8, 9)])
        cls.modes = jnp.asarray([L.BCA, L.SUPPORT, L.FULL])
        cls.readouts = L.make_readouts("one_hot")

    def test_heldout_residual_uses_the_target_the_arm_is_trained_toward(self):
        eta, residual, plain, logit = (np.asarray(x) for x in
                                       self.readouts["heldout"](self.states, self.bank, self.keys, self.modes))
        for array in (eta, residual, plain, logit):
            self.assertEqual(array.shape, (3, 96))
            self.assertTrue(np.all(np.isfinite(array)))
        as_baseline = np.asarray(self.readouts["heldout"](self.states, self.bank, self.keys,
                                                          jnp.full((3,), L.NATIVE))[1])
        np.testing.assert_array_equal(residual[:2], as_baseline[:2])       # no width in these arms' targets
        self.assertTrue(np.all(residual[2] <= as_baseline[2]))             # the full arm's target is never higher
        self.assertGreater(float((as_baseline[2] - residual[2]).max()), 0.1)
        done = np.asarray(self.bank["done"][2]) == 1
        np.testing.assert_array_equal(residual[2][done], as_baseline[2][done])

    def test_probe_reports_the_width_each_arm_consumes(self):
        cells = jnp.arange(100, dtype=jnp.int32)
        corners = jnp.asarray(np.array([[1, -1, -1, -1], [-1, 1, -1, -1], [0.2, 0.1, -0.3, 0.4]]), jnp.float32)
        grid_obs, grid_commands = jnp.repeat(ALL_CELLS, 3, axis=0), jnp.tile(corners, (100, 1))
        states = L.stack([dict(s, reference=active_reference(s, self.cal, grid_obs, grid_commands))
                          for s in self.singles])
        q1, q2, original, consumed, logit = (np.asarray(x) for x in
                                             self.readouts["probe"](states, cells, corners, self.modes))
        for array in (q1, q2, original, consumed, logit):
            self.assertEqual(array.shape, (3, 100, 3))
        np.testing.assert_array_equal(consumed[0], original[0])            # original BCA: unchanged width
        for run in (1, 2):
            floor = float(states["reference"]["support_floor"][run])
            below = logit[run] < floor - 1e-4
            self.assertGreater(int(below.sum()), 5)
            self.assertTrue(np.all(consumed[run][below] > original[run][below]))
            np.testing.assert_array_equal(consumed[run][logit[run] > floor + 1e-4],
                                          original[run][logit[run] > floor + 1e-4])
        policy = np.asarray(self.readouts["policy"](states, cells))
        self.assertEqual(policy.shape, (3, 100, 4))
        for array in self.readouts["audit"](self.states, self.bank, self.keys, self.modes):
            self.assertEqual(array.shape, (3, 96))

    def test_probe_width_of_the_support_only_arms_has_no_calibrated_part(self):
        cells = jnp.arange(100, dtype=jnp.int32)
        corners = jnp.asarray(np.array([[1, -1, -1, -1], [-1, 1, -1, -1], [0.2, 0.1, -0.3, 0.4]]), jnp.float32)
        grid_obs, grid_commands = jnp.repeat(ALL_CELLS, 3, axis=0), jnp.tile(corners, (100, 1))
        states = L.stack([dict(s, reference=active_reference(s, self.cal, grid_obs, grid_commands))
                          for s in self.singles])
        modes = jnp.asarray([L.SUPPORT, L.SUPPORT_ONLY, L.SUPPORT_ONLY_SHUFFLED])
        _, _, original, consumed, logit = (np.asarray(x, np.float64)
                                           for x in self.readouts["probe"](states, cells, corners, modes))
        for run in (1, 2):
            floor = float(states["reference"]["support_floor"][run])
            cap = float(states["reference"]["width_cap"][run])
            expected = np.maximum(floor - logit[run], 0.0) / np.log(10.0) * cap
            np.testing.assert_allclose(consumed[run], expected, rtol=1e-4, atol=1e-3)
            np.testing.assert_array_equal(consumed[run][logit[run] > floor + 1e-4], 0.0)
            self.assertGreater(int((consumed[run] > 0).sum()), 5)
            self.assertTrue(np.all(original[run] > 0))          # the calibrated width exists; the arm ignores it
        width, _, _ = self.readouts["audit"](states, self.bank, self.keys, modes)
        self.assertTrue(bool(jnp.all(width[1] >= 0)) and float(jnp.min(width[0])) > 0)

    def test_method_tables_used_by_the_evaluation_match_the_learner(self):
        import evaluation as ev
        self.assertEqual(ev.CONTROL_OF, L.CONTROL_OF)
        self.assertEqual(ev.KAPPA, L.KAPPA)
        for number, name in L.ARM_NAMES.items():
            arm = L.switches(number)
            self.assertEqual(name in ev.SUPPORT_AWARE, bool(arm["support"]), name)
            self.assertEqual(name in ev.ONE_SIDED, bool(arm["one_sided"]), name)
            self.assertEqual(name in ev.NO_CALIBRATED_WIDTH, bool(arm["no_base"]), name)
        reference = {"width_ref": 1.5, "width_ref_support": 2.5}
        self.assertIsNone(ev.one_sided_level("bca", reference))
        self.assertIsNone(ev.one_sided_level("support", reference))
        self.assertEqual(ev.one_sided_level("onesided", reference), 1.5)
        self.assertEqual(ev.one_sided_level("full", reference), 2.5)


class Initialization(unittest.TestCase):
    def test_host_layers_use_the_pytorch_default_range(self):
        state = L.init_state(0, 100)
        for net, sizes in (("actor", (100, 64, 64)), ("q1", (104, 64, 64))):
            for layer, fan_in in zip(state[net], sizes):
                bound = 1 / np.sqrt(fan_in)
                for name in ("w", "b"):
                    self.assertLessEqual(float(jnp.abs(layer[name]).max()), bound + 1e-7)
                np.testing.assert_allclose(float(layer["w"].std()), bound / np.sqrt(3), rtol=0.08)
        self.assertEqual(state["actor"][-1]["w"].shape, (64, 4))
        self.assertTrue(leaves_equal(state["actor"], state["actor_t"]))
        self.assertFalse(leaves_equal(state["q1"], state["q2"]))

    def test_policy_commands_are_valid_environment_commands(self):
        state = L.init_state(0, 100)
        commands = L.actor_apply(state["actor"], L.one_hot_obs(jnp.arange(100)))
        self.assertEqual(commands.shape, (100, 4))
        self.assertTrue(bool(jnp.all(jnp.abs(commands) <= 1.0)))

    def test_observation_encodings_match_the_environment_module(self):
        import environment as env
        cells = np.arange(100)
        np.testing.assert_array_equal(L.one_hot_obs(jnp.asarray(cells)), env.one_hot(cells))
        np.testing.assert_allclose(L.coordinate_obs(jnp.asarray(cells)), env.coordinates(cells), atol=1e-7)

    @unittest.skipUnless((ARCHIVE / "qualification_v3/bca_port/original_scale_init_700010.npz").exists(),
                         "archived scale initialization not present")
    def test_scale_network_starts_exactly_where_the_archived_port_started(self):
        saved = np.load(ARCHIVE / "qualification_v3/bca_port/original_scale_init_700010.npz", allow_pickle=False)
        params = L.init_state(10, 100)["scale"]["params"]
        for i in range(3):
            np.testing.assert_allclose(params[f"Dense_{i}"]["kernel"], saved[f"dense{i}_kernel"], rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(params[f"Dense_{i}"]["bias"], saved[f"dense{i}_bias"], atol=0)


if __name__ == "__main__":
    unittest.main()
