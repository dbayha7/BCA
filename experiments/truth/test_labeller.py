"""CPU checks for experiments/truth/labeller.py: batched rollout truth against direct gym rollouts.

Fixtures: synthetic D4RL-format files from the real simulator (synthetic.py) and a randomly initialised
actor of the host's exact architecture (algorithms/td3_bc.initialize) with the data's normalization. Checks:
  - the batched engine's labels (q, ret, length, status, first reward) equal a direct gym rollout of the
    same policy (env.step through TimeLimit/OrderEnforcing/NormalizedBoxEnv/Offline*Env) bit for bit, for
    both tasks, for logged first actions and for pi(s), at horizon 1,000 and at a short horizon where
    rollouts are cut (status 1); the random actor falls within tens of steps, so labeller.upright_actor (a
    fixed PD controller plus a tiny host-architecture term) also checks two full 1,000-step rollouts per task;
  - noisy-target labels equal a direct gym rollout that draws the same per-rollout noise stream;
  - jitter mode at jitter 0, and noisy-target mode at noise SD 0, reproduce the deterministic labels exactly;
    positive jitter changes labels and is reproducible;
  - shards: an interrupted job resumed in place equals a fresh job and the unsharded engine bit for bit;
    shard size (hence slot schedule) does not change labels; a corrupted shard is recomputed identically;
    a different spec in the same directory is refused; the manifest is strict JSON;
  - the fixed-shape policy passes the position-invariance gate;
  - central differences: exact on a toy linear Q, clipped steps at the action bound, item layout.
Synthetic numbers are test fixtures, not findings.
JAX_PLATFORMS=cpu LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin python -m unittest experiments.truth.test_labeller
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.truth import labeller as L
from experiments.truth import state_restore as SR
from experiments.truth import synthetic as SY

TASKS = ("hopper", "walker2d")
BATCH = 32


def gym_rollout(physics, policy, qpos, qvel, action, *, gamma=0.99, horizon=1000, noise=None):
    """Independent reference: the real wrapper chain's env.step, one rollout, policy at row 0 of its batch."""
    env, base = physics.env, physics.base
    env.reset()
    base.sim.reset()
    base.set_state(np.asarray(qpos, np.float64), np.asarray(qvel, np.float64))
    base.sim.data.qacc_warmstart[:] = 0.0
    obs = base._get_obs()
    q, ret, disc, first, status = 0.0, 0.0, 1.0, None, 1
    for t in range(horizon):
        if t == 0:
            a = action
        else:
            batch = np.zeros((policy.batch, policy.obs_dim))
            batch[0] = obs
            a = policy(batch)[0]
            if noise is not None:
                a = np.clip(a + noise[t], -1.0, 1.0)
        obs, reward, done, info = env.step(a)
        first = reward if t == 0 else first
        q = q + disc * reward
        ret = ret + reward
        disc = disc * gamma
        if done and not info.get("TimeLimit.truncated", False):
            status = 0
            break
    return q, ret, t + 1, status, first


class LabellerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from runtime.common import compute_mean_std

        cls.tmp = Path(tempfile.mkdtemp(prefix="truth-labeller-"))
        cls.physics, cls.policy, cls.items, cls.sd = {}, {}, {}, {}
        for task in TASKS:
            physics = SR.Physics(task)
            meta = SY.write_synthetic(cls.tmp, task, n_rows=1500, seed=5, physics=physics)
            raw = SR.load_raw(physics, meta["with_infos"])
            converted, rows, _ = SR.converted_rows(raw)
            states = SR.logged_states(task, raw, rows)
            mean, std = compute_mean_std(np.asarray(converted["observations"], np.float32), eps=0.001)
            actor, _, native = L.random_host_networks(task, 202609171)
            policy = L.FrozenPolicy(actor, native.actor.params, mean, std, batch=BATCH)
            pick = np.flatnonzero(states.kept)[::97][:12]
            qpos, qvel = states.qpos[pick], states.qvel[pick]
            pi = L.policy_actions(physics, policy, qpos, qvel)
            logged = np.asarray(converted["actions"])[pick].astype(np.float32)
            cls.items[task] = dict(qpos=np.r_[qpos, qpos], qvel=np.r_[qvel, qvel], action=np.r_[logged, pi],
                                   item_id=np.arange(2 * len(pick), dtype=np.int64) * 7 + 3)
            cls.physics[task], cls.policy[task] = physics, policy
            cls.sd[task] = SR.state_sd(states)

    @classmethod
    def tearDownClass(cls):
        for physics in cls.physics.values():
            physics.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def label(self, task, spec=None, items=None, **kwargs):
        spec = spec or L.LabelSpec(SR.task_spec(task).dataset)
        return L.rollouts(self.physics[task], self.policy[task], spec, items or self.items[task], **kwargs)

    def assert_same_labels(self, a, b):
        for k in L.OUTPUTS:
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)

    def test_rollouts_equal_direct_gym_rollouts(self):
        for task in TASKS:
            for horizon in (1000, 6):
                spec = L.LabelSpec(SR.task_spec(task).dataset, horizon=horizon)
                labels = self.label(task, spec)
                items = self.items[task]
                for i in range(len(items["item_id"])):
                    q, ret, length, status, first = gym_rollout(
                        self.physics[task], self.policy[task], items["qpos"][i], items["qvel"][i],
                        items["action"][i], horizon=horizon)
                    self.assertEqual(labels["q"][i, 0], q, (task, horizon, i))
                    self.assertEqual(labels["ret"][i, 0], ret)
                    self.assertEqual(labels["length"][i, 0], length)
                    self.assertEqual(labels["status"][i, 0], status)
                    self.assertEqual(labels["first_reward"][i, 0], first)
                if horizon == 6:
                    self.assertTrue(np.any(labels["status"] == 1), "short horizon must cut some rollouts")
                else:  # the policy itself must act for many steps in some rollouts
                    self.assertGreater(int(labels["length"].max()), 20)
                    self.assertTrue(np.all(labels["status"] == 0))

    def test_full_horizon_rollouts_equal_direct_gym_rollouts(self):
        for task in TASKS:
            spec, physics = SR.task_spec(task), self.physics[task]
            import jax

            actor = L.upright_actor(task)
            params = actor.init(jax.random.PRNGKey(0), np.zeros((1, spec.obs_dim), np.float32))
            policy = L.FrozenPolicy(actor, params, np.zeros(spec.obs_dim), np.ones(spec.obs_dim), batch=8)
            starts = []
            for seed in (1, 2):
                physics.env.seed(seed)
                physics.env.reset()
                starts.append((physics.base.sim.data.qpos.copy(), physics.base.sim.data.qvel.copy()))
            qpos, qvel = np.stack([s[0] for s in starts]), np.stack([s[1] for s in starts])
            items = dict(qpos=qpos, qvel=qvel, action=L.policy_actions(physics, policy, qpos, qvel),
                         item_id=np.arange(2, dtype=np.int64))
            labels = L.rollouts(physics, policy, L.LabelSpec(spec.dataset), items)
            np.testing.assert_array_equal(labels["length"], 1000)
            np.testing.assert_array_equal(labels["status"], 1)
            for i in range(2):
                q, ret, length, status, first = gym_rollout(physics, policy, qpos[i], qvel[i], items["action"][i])
                self.assertEqual((labels["q"][i, 0], labels["ret"][i, 0], labels["first_reward"][i, 0]),
                                 (q, ret, first))
                self.assertEqual((length, status), (1000, 1))

    def test_noisy_target_equals_direct_gym_with_same_noise(self):
        for task in TASKS:
            spec = L.LabelSpec(SR.task_spec(task).dataset, mode="noisy_target", repeats=2, seed=99)
            labels = self.label(task, spec)
            items, d = self.items[task], SR.task_spec(task).action_dim
            for i in range(4):
                for k in range(2):
                    z = L._stream(99, L.KEY_NOISE, items["item_id"][i], k).standard_normal((1000, d))
                    noise = np.clip(0.2 * z, -0.5, 0.5).astype(np.float32)
                    q, ret, length, status, _ = gym_rollout(self.physics[task], self.policy[task], items["qpos"][i],
                                                            items["qvel"][i], items["action"][i], noise=noise)
                    self.assertEqual((labels["q"][i, k], labels["length"][i, k]), (q, length))
            self.assertFalse(np.array_equal(labels["q"][:, 0], labels["q"][:, 1]))

    def test_zero_jitter_and_zero_noise_equal_deterministic(self):
        for task in TASKS:
            dataset = SR.task_spec(task).dataset
            base = self.label(task)
            jitter0 = self.label(task, L.LabelSpec(dataset, mode="jitter", repeats=2, jitter_scale=0.0),
                                 jitter_sd=self.sd[task])
            noise0 = self.label(task, L.LabelSpec(dataset, mode="noisy_target", repeats=2, noise_sd=0.0))
            for k in range(2):
                for out in (jitter0, noise0):
                    self.assert_same_labels({n: v[:, k:k + 1] for n, v in out.items()}, base)

    def test_positive_jitter_moves_labels_reproducibly(self):
        task = "hopper"
        spec = L.LabelSpec("hopper-medium-v2", mode="jitter", repeats=3, jitter_scale=1e-2, seed=4)
        a = self.label(task, spec, jitter_sd=self.sd[task])
        b = self.label(task, spec, jitter_sd=self.sd[task])
        self.assert_same_labels(a, b)
        base = self.label(task)
        self.assertFalse(np.array_equal(a["q"][:, 0], base["q"][:, 0]))
        with self.assertRaises(ValueError):
            self.label(task, spec)  # jitter needs the SDs

    def test_shards_resume_and_reproduce(self):
        task = "walker2d"
        physics, policy, items = self.physics[task], self.policy[task], self.items[task]
        spec = L.LabelSpec("walker2d-medium-replay-v2", shard_size=5)
        direct = self.label(task)
        part = self.tmp / "resume"
        first = L.run_labels(part, physics, policy, spec, items, max_shards=2)
        self.assertEqual([r["shard"] for r in first], [0, 1])
        with self.assertRaises(ValueError):
            L.load_labels(part)  # incomplete
        second = L.run_labels(part, physics, policy, spec, items)
        self.assertEqual([r["resumed"] for r in second], [True, True, False, False, False])
        resumed, manifest = L.load_labels(part)
        fresh_dir = self.tmp / "fresh"
        L.run_labels(fresh_dir, physics, policy, spec, items)
        fresh, _ = L.load_labels(fresh_dir)
        for k in fresh:
            np.testing.assert_array_equal(resumed[k], fresh[k], err_msg=k)
        self.assert_same_labels(resumed, direct)
        np.testing.assert_array_equal(resumed["item_id"], items["item_id"])
        # one shard covering everything: a different slot schedule, the same labels
        whole = self.tmp / "whole"
        L.run_labels(whole, physics, policy, L.LabelSpec("walker2d-medium-replay-v2", shard_size=100), items)
        self.assert_same_labels(L.load_labels(whole)[0], direct)
        # a corrupted shard is recomputed, identically
        npz = part / "shard_00003.npz"
        npz.write_bytes(b"corrupt")
        third = L.run_labels(part, physics, policy, spec, items)
        self.assertEqual([r["resumed"] for r in third], [True, True, True, False, True])
        self.assert_same_labels(L.load_labels(part)[0], direct)
        # a different job in the same directory is refused
        with self.assertRaises(ValueError):
            L.run_labels(part, physics, policy, L.LabelSpec("walker2d-medium-replay-v2", shard_size=5, horizon=999),
                         items)
        text = (part / "manifest.json").read_text()
        json.loads(text, parse_constant=lambda c: self.fail("nonfinite constant " + c))
        self.assertTrue(manifest["position_invariance"]["passed"])
        self.assertEqual(manifest["n_shards"], 5)

    def test_worker_split_covers_all_shards(self):
        task = "hopper"
        physics, policy, items = self.physics[task], self.policy[task], self.items[task]
        spec = L.LabelSpec("hopper-medium-v2", shard_size=4)
        out = self.tmp / "workers"
        done = [r["shard"] for w in range(3) for r in L.run_labels(out, physics, policy, spec, items, worker=w,
                                                                    workers=3)]
        self.assertEqual(sorted(done), list(range(6)))
        self.assert_same_labels(L.load_labels(out)[0], self.label(task))

    def test_simulator_error_gives_status_2_and_spares_other_rollouts(self):
        from unittest import mock

        task = "hopper"
        physics, base = self.physics[task], self.label(task)
        real_step, calls = physics.step, [0]

        def flaky(sim, action):
            calls[0] += 1
            if calls[0] == 3:  # the third env step of the run: slot 2's first step
                raise physics.simulator_error("injected instability")
            return real_step(sim, action)

        with mock.patch.object(physics, "step", side_effect=flaky):
            labels = self.label(task)
        bad = labels["status"][:, 0] == 2
        self.assertEqual(int(bad.sum()), 1)
        self.assertTrue(np.isnan(labels["q"][bad, 0]).all())
        for k in L.OUTPUTS:
            np.testing.assert_array_equal(labels[k][~bad], base[k][~bad], err_msg=k)

    def test_policy_from_host_state(self):
        actor, critic, native = L.random_host_networks("hopper", 202609171)
        policy = self.policy["hopper"]
        bound = L.FrozenPolicy.from_host_state((actor, critic, None), native, policy.mean, policy.std, batch=BATCH)
        self.assertEqual(bound.identity()["params_sha256"], policy.identity()["params_sha256"])
        obs = SR.observation("hopper", self.items["hopper"]["qpos"], self.items["hopper"]["qvel"])
        np.testing.assert_array_equal(bound.actions(obs), policy.actions(obs))
        with self.assertRaises(ValueError):
            L.FrozenPolicy.from_host_state((critic, critic, None), native, policy.mean, policy.std)

    def test_position_invariance_gate(self):
        for task in TASKS:
            obs = SR.observation(task, self.items[task]["qpos"], self.items[task]["qvel"])
            gate = self.policy[task].position_invariance(obs)
            self.assertTrue(gate["passed"], gate)

    def test_central_differences(self):
        pi = np.array([[0.0, 0.95, -1.0], [0.5, -0.2, 0.3]], np.float32)
        plus, minus, step, clipped = L.central_difference_actions(pi, 0.1)
        self.assertEqual(plus.shape, (2, 3, 3))
        self.assertEqual(clipped, 2)  # 0.95 + 0.1 and -1.0 - 0.1 hit the bound
        self.assertAlmostEqual(step[0, 1], 1.0 - np.float32(0.85), places=6)
        self.assertTrue(np.all(np.abs(plus) <= 1) and np.all(np.abs(minus) <= 1))
        w = np.array([2.0, -3.0, 0.5])
        qp = (plus.astype(np.float64) @ w)
        qm = (minus.astype(np.float64) @ w)
        np.testing.assert_allclose(L.action_gradient(qp, qm, step), np.tile(w, (2, 1)), rtol=1e-12)
        task = "hopper"
        items, info = L.gradient_items(self.physics[task], self.policy[task], self.items[task]["qpos"][:3],
                                       self.items[task]["qvel"][:3], np.array([4, 5, 6]))
        self.assertEqual(items["action"].shape, (18, 3))
        self.assertEqual(items["item_id"].tolist(), list(range(24, 42)))
        q = (items["action"].astype(np.float64) @ w)
        np.testing.assert_allclose(L.gradient_from_labels(q, 3, 3, info["step"]), np.tile(w, (3, 1)), rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
