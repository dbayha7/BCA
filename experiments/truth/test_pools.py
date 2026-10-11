"""CPU checks for experiments/truth/pools.py on synthetic D4RL-format files and randomly initialised hosts.

Fixtures: a synthetic Hopper file made in the real simulator (synthetic.py) and train.py-format run directories
around randomly initialised TD3+BC networks of the host's exact architecture (pools.write_synthetic_run). Checks:
  - the archived standard-study host rows (experiments/standard_bca/manifest.json.gz) type on this branch after
    adapt_archived_row, which changes only the two inert fields;
  - restore_host round trip: exact template restore, the split, the standardization gate, the identity record;
    refusals for a changed checkpoint, a wrong pinned hash, a missing cache file and a non-host run; the native-only
    fallback for a state that does not fit the template; preparation checks allow extra recorded keys and refuse a
    changed one;
  - ThinnessIndex counts and k-NN distances equal brute force with same-episode rows excluded;
  - CriticProbe: the TD target with given noise equals algorithms.td3_bc_bca.native_target to float32 rounding, the
    noiseless residual equals its formula, the critic-guided action is a unit step (or clipped);
  - policy_episodes returns equal the labeller's rollout returns from the same reset states (first action pi(s0)),
    and its recorded states are the states before steps 0, every, 2 every, ...; the reset bank is reproducible;
  - D-thin masks, pool item layout, and pool files that refuse a changed npz.
Synthetic numbers are test fixtures, not findings.
JAX_PLATFORMS=cpu LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin python -m unittest experiments.truth.test_pools
"""

import contextlib
import gzip
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.truth import labeller as LB
from experiments.truth import pools as PL
from experiments.truth import state_restore as SR
from experiments.truth import synthetic as SY

ROOT = Path(__file__).resolve().parents[2]
SEED = 202609171


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


class PoolsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="truth-pools-"))
        cls.physics = SR.Physics("hopper")
        (cls.tmp / "data").mkdir()
        meta = SY.write_synthetic(cls.tmp / "data", "hopper", n_rows=1500, seed=7, physics=cls.physics)
        cls.file = Path(meta["with_infos"])
        with quiet():
            cls.host_run = PL.write_synthetic_run(cls.tmp / "run", "hopper", cls.file, SEED)
            cls.host = PL.restore_host(cls.host_run["run_dir"], data_dir=cls.file.parent)

    @classmethod
    def tearDownClass(cls):
        cls.physics.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def copy_run(self, name):
        target = self.tmp / name
        shutil.copytree(self.host_run["run_dir"], target)
        return target

    # ---- host binding

    def test_archived_rows_type_after_adaptation(self):
        from runtime.config import typed

        manifest = json.loads(gzip.decompress((ROOT / "experiments/standard_bca/manifest.json.gz").read_bytes()))
        for run_id in ("td3_bc-hopper-host-s202609171", "td3_bc-walker2d-host-s202609171"):
            row = next(r["row"] for r in manifest["runs"] if r["row"]["run_id"] == run_id)
            with self.assertRaises(TypeError):
                with quiet():
                    typed(row)
            adapted, changes = PL.adapt_archived_row(row)
            self.assertEqual(len(changes), 2)
            with quiet():
                _, args, _, protocol = typed(adapted)
            self.assertIsNone(protocol.reservation.rows_per_episode)
            self.assertEqual((protocol.reservation.target_size, args.alpha), (1024, 2.5))
            row2 = json.loads(json.dumps(row))
            row2["protocol"]["reference"] = {"target_size": 10}
            with self.assertRaises(ValueError):
                PL.adapt_archived_row(row2)

    def test_restore_round_trip(self):
        from algorithms import td3_bc_bca as P
        from runtime.td3_bc import _tree_hash

        host = self.host
        ident = host.identity
        self.assertTrue(ident["synthetic_host"] and host.synthetic)
        self.assertIn("full algorithms.td3_bc_bca.initialize template", ident["checkpoint"]["restore"])
        self.assertEqual(ident["checkpoint"]["sha256"], self.host_run["checkpoint_sha256"])
        self.assertIsNone(ident["audit"])  # the real audit is about another run's resolved.json
        n = len(host.rows)
        ids = np.sort(np.r_[host.training_ids, host.withheld_ids])
        np.testing.assert_array_equal(ids, np.arange(n))
        self.assertTrue(np.all(np.isin(host.bank_ids, host.withheld_ids)))
        self.assertFalse(np.any(np.isin(host.episodes[host.withheld_ids], host.episodes[host.training_ids])))
        with quiet():
            _, fresh, _ = P.initialize(host.args, P.Config("host"), 11, 3, 1.0)
        self.assertEqual(_tree_hash(fresh.native.actor.params), ident["actor_params_sha256"])
        obs = host.converted["observations"][host.training_ids]
        mean = obs.mean(axis=0)
        std = obs.std(axis=0) + 0.001
        np.testing.assert_allclose(host.obs_mean, mean, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(host.obs_std, std, rtol=1e-4, atol=1e-6)

    def test_restore_refusals(self):
        bad = self.copy_run("bad_ckpt")
        ck = bad / "checkpoint_0.msgpack"
        data = bytearray(ck.read_bytes())
        data[-1] ^= 1
        ck.write_bytes(bytes(data))
        with quiet(), self.assertRaises(ValueError):
            PL.restore_host(bad, data_dir=self.file.parent)
        with quiet(), self.assertRaises(ValueError):
            PL.restore_host(self.host_run["run_dir"], data_dir=self.file.parent, expected_sha256="0" * 64)
        with quiet(), self.assertRaises(FileNotFoundError) as ctx:
            PL.restore_host(self.host_run["run_dir"], data_dir=self.tmp / "nowhere")
        self.assertIn(self.file.name, str(ctx.exception))
        other = self.copy_run("bca_run")
        row = json.loads((other / "resolved.json").read_text())
        row["method"] = "bca"
        (other / "resolved.json").write_text(json.dumps(row))
        with self.assertRaises(ValueError):
            PL.restore_host(other, data_dir=self.file.parent)
        with self.assertRaises(ValueError):
            PL.restore_host(self.tmp / "missing_run", data_dir=self.file.parent)

    def test_preparation_check(self):
        run = self.copy_run("prep")
        prep = json.loads((run / "preparation.json").read_text())
        prep["metadata"]["run_input_hashes"]["reference"] = "x" * 64  # the archived stack's extra key
        (run / "preparation.json").write_text(json.dumps(prep))
        with quiet():
            host = PL.restore_host(run, data_dir=self.file.parent)
        self.assertEqual(host.identity["preparation"]["extra_recorded_keys"], ["reference"])
        prep["metadata"]["run_input_hashes"]["obs_mean"] = "y" * 64
        (run / "preparation.json").write_text(json.dumps(prep))
        with quiet(), self.assertRaises(ValueError):
            PL.restore_host(run, data_dir=self.file.parent)

    def test_native_only_fallback(self):
        import jax.numpy as jnp
        from flax import serialization

        from algorithms import td3_bc_bca as P

        host = self.host
        with quiet():
            rng, template, _ = P.initialize(host.args, P.Config("host"), 11, 3, 1.0)
        state = serialization.to_state_dict(template)
        state["archived_extra_field"] = np.zeros(3, np.float32)
        payload = serialization.msgpack_serialize({"state": state, "training_rng": np.asarray(rng),
                                                   "step": np.int32(0)})
        native, how = PL.restore_native(payload, template, rng, 0)
        self.assertTrue(how.startswith("native subtree only"))
        for a, b in zip(PL._flat_leaves(native), PL._flat_leaves(template.native)):
            np.testing.assert_array_equal(np.asarray(a[1]), np.asarray(b[1]))
        exact, how = PL.restore_native(serialization.to_bytes({"state": template, "training_rng": rng,
                                                               "step": jnp.int32(0)}), template, rng, 0)
        self.assertTrue(how.startswith("full"))

    # ---- thinness

    def test_thinness_equals_brute_force(self):
        host = self.host
        thin = PL.ThinnessIndex(host, k=5)
        w = host.withheld_ids
        r, dk = thin.fit_radius(PL.features(host, host.converted["observations"][w], host.converted["actions"][w]),
                                host.episodes[w])
        X, E = thin.x, thin.episode
        g = np.random.default_rng(0)
        q_ids = host.training_ids[g.choice(len(host.training_ids), 20, replace=False)]
        xq = PL.features(host, host.converted["observations"][q_ids], host.converted["actions"][q_ids])
        eq = host.episodes[q_ids]
        xq = np.r_[xq, xq[:5] + 0.05]
        eq = np.r_[eq, np.full(5, -1)]
        c = thin.counts(xq, eq)
        d = thin.knn_other(xq, eq)
        for i in range(len(xq)):
            dist = np.sqrt(((X - xq[i]) ** 2).sum(axis=1))
            other = E != eq[i]
            self.assertEqual(c[i], int((dist[other] <= r).sum()))
            np.testing.assert_allclose(d[i], np.sort(dist[other])[:5], rtol=1e-12)
        xw = PL.features(host, host.converted["observations"][w], host.converted["actions"][w])
        brute = np.array([np.sort(np.sqrt(((X - x) ** 2).sum(axis=1)))[4] for x in xw])
        self.assertAlmostEqual(r, float(np.median(brute)), places=12)

    # ---- critic queries

    def test_target_matches_native_target(self):
        import jax
        import jax.numpy as jnp

        from algorithms import td3_bc_bca as P
        from runtime.networks import Transition

        host = self.host
        probe = PL.CriticProbe(host, chunk=64)
        ids = host.bank_ids[:40]
        conv = host.converted
        batch = Transition(jnp.asarray(host.standardized(conv["observations"][ids])),
                           jnp.asarray(conv["actions"][ids]), jnp.asarray(conv["rewards"][ids]),
                           jnp.asarray(host.standardized(conv["next_observations"][ids])),
                           jnp.asarray(conv["terminals"][ids], jnp.float32))
        key = jax.random.PRNGKey(3)
        reference = np.asarray(P.native_target(host.args, (host.actor, host.critic), host.native, batch, key, 1.0))
        noise = np.clip(np.asarray(jax.random.normal(key, batch.action.shape)) * host.args.policy_noise,
                        -host.args.noise_clip, host.args.noise_clip)
        mine = probe.target_with_noise(conv["rewards"][ids], conv["next_observations"][ids], conv["terminals"][ids],
                                       noise)
        # equal up to float32 rounding: the probe runs a padded fixed-shape batch, and XLA's kernels differ by
        # batch shape (labeller.py measured up to 1.5e-7 for the actor)
        np.testing.assert_allclose(mine, reference, rtol=1e-6, atol=1e-7)
        draws = probe.target_draws(conv["rewards"][ids], conv["next_observations"][ids], conv["terminals"][ids], 3, 1)
        self.assertEqual(draws.shape, (40, 3))
        np.testing.assert_array_equal(draws, probe.target_draws(conv["rewards"][ids], conv["next_observations"][ids],
                                                                conv["terminals"][ids], 3, 1))

    def test_residual_and_critic_guided(self):
        host = self.host
        probe = PL.CriticProbe(host, chunk=64)
        policy = host.policy(32)
        ids = host.bank_ids[:30]
        conv = host.converted
        obs, a = conv["observations"][ids].astype(np.float64), conv["actions"][ids]
        nxt = conv["next_observations"][ids]
        pi_next = policy.actions(nxt.astype(np.float64))
        b = probe.noiseless_residual(obs, a, conv["rewards"][ids], nxt, conv["terminals"][ids], pi_next)
        q_now, q_next = probe.q(obs, a)[:, 0], probe.q(nxt, pi_next)[:, 0]
        direct = conv["rewards"][ids] + 0.99 * (1 - conv["terminals"][ids]) * q_next.astype(np.float64) - q_now
        np.testing.assert_allclose(b, direct, rtol=1e-12, atol=1e-9)
        pi = policy.actions(obs)
        cg, norm = probe.critic_guided(obs, pi, 0.1)
        step = np.linalg.norm(cg.astype(np.float64) - pi.astype(np.float64), axis=1)
        inside = np.all(np.abs(pi.astype(np.float64)) < 0.85, axis=1)
        np.testing.assert_allclose(step[inside], 0.1, atol=1e-6)
        self.assertTrue(np.all(np.abs(cg) <= 1.0))

    # ---- episodes

    def test_policy_episodes_match_labeller(self):
        host = self.host
        policy = host.policy(16)
        q0, v0 = PL.reset_bank(self.physics, [11, 12, 13, 14, 15])
        q1, v1 = PL.reset_bank(self.physics, [11, 12, 13, 14, 15])
        np.testing.assert_array_equal(q0, q1)
        ep, rec = PL.policy_episodes(self.physics, policy, q0, v0, every=4)
        items = dict(qpos=q0, qvel=v0, action=LB.policy_actions(self.physics, policy, q0, v0),
                     item_id=np.arange(5, dtype=np.int64))
        spec = LB.LabelSpec("hopper-medium-v2", gamma=1.0, horizon=1000, shard_size=5)
        out = LB.rollouts(self.physics, policy, spec, items)
        np.testing.assert_array_equal(ep["returns"], out["ret"][:, 0])
        np.testing.assert_array_equal(ep["lengths"], out["length"][:, 0])
        first = rec["t"] == 0
        np.testing.assert_array_equal(rec["episode"][first], np.arange(5))
        np.testing.assert_array_equal(rec["qpos"][first], q0)
        self.assertTrue(np.all(rec["t"] % 4 == 0))
        for e in range(5):
            self.assertEqual(int(rec["t"][rec["episode"] == e].max()), 4 * ((ep["lengths"][e] - 1) // 4))

    def test_d_thin_and_items_and_files(self):
        raw = self.host.raw
        keep, record = PL.d_thin_mask(raw, 0.1, 0)
        self.assertEqual(record["raw_rows_kept"], int(keep.sum()))
        self.assertTrue(PL.d_thin_mask(raw, 1.0, 0)[0].all())
        none = PL.d_thin_mask(raw, 0.0, 0)[1]
        self.assertEqual(none["top_episodes_kept"], 0)
        self.assertEqual(none["raw_rows_kept"], none["raw_rows"] - none["top_rows"])
        arrays = dict(qpos=np.arange(12.0).reshape(2, 6), qvel=np.zeros((2, 6)),
                      actions=np.arange(18, dtype=np.float32).reshape(2, 3, 3) / 20)
        items = PL.pool_items(arrays, slots=[0, 2], states=[1])
        np.testing.assert_array_equal(items["item_id"], [3, 5])
        np.testing.assert_array_equal(items["action"], arrays["actions"][1, [0, 2]])
        PL.save_pool(self.tmp / "files", "X", arrays, dict(kind="X"))
        loaded, record = PL.load_pool(self.tmp / "files", "X")
        np.testing.assert_array_equal(loaded["actions"], arrays["actions"])
        with self.assertRaises(ValueError):
            PL.save_pool(self.tmp / "files", "X", arrays, dict(kind="X"))
        npz = self.tmp / "files" / "X.npz"
        data = bytearray(npz.read_bytes())
        data[-5] ^= 1
        npz.write_bytes(bytes(data))
        with self.assertRaises(ValueError):
            PL.load_pool(self.tmp / "files", "X")


if __name__ == "__main__":
    unittest.main()
