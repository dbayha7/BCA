"""IQL host/BCA smoke test on CPU: typed configs, pre-refresh native weights, one WBCP
refresh, frozen-threshold consumption and a host arm with no calibration state.

Synthetic transitions only; no dataset, simulator or GPU.
JAX_PLATFORMS=cpu python -m unittest experiments.wbcp.test_iql_host
"""
import json
import math
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from calibration import wbcp
from calibration.reference import reference_valid
from runtime.config import ROOT, resolve, typed
from runtime.iql_checkpoint import checkpoint_counts
from runtime.networks import Transition

SEED, DATASET = 202609171, "hopper"
OBS, ACT = 11, 3
STEPS = 4


def synthetic(n, seed):
    g = np.random.default_rng(seed)
    obs = g.normal(size=(n, OBS)).astype(np.float32)
    action = np.tanh(g.normal(size=(n, ACT))).astype(np.float32)
    reward = (g.normal(size=n) + obs[:, 0]).astype(np.float32)
    nxt = (obs + 0.1 * g.normal(size=(n, OBS))).astype(np.float32)
    done = (g.uniform(size=n) < 0.02).astype(np.float32)
    return Transition(*(jnp.asarray(x) for x in (obs, action, reward, nxt, done)))


def d4rl_like(lengths, seed):
    """Raw rows with a timeout row closing each episode, and D4RL's default qlearning conversion."""
    g = np.random.default_rng(seed)
    total = int(np.sum(lengths)) + len(lengths)
    raw = dict(observations=g.normal(size=(total, OBS)).astype(np.float32),
               actions=np.tanh(g.normal(size=(total, ACT))).astype(np.float32),
               rewards=g.normal(size=total).astype(np.float32),
               terminals=np.zeros(total, bool), timeouts=np.zeros(total, bool))
    raw["timeouts"][np.cumsum(np.asarray(lengths) + 1) - 1] = True
    keep = np.flatnonzero(~raw["timeouts"][:-1])
    converted = dict(observations=raw["observations"][keep], actions=raw["actions"][keep],
                     next_observations=raw["observations"][keep + 1],
                     rewards=raw["rewards"][keep], terminals=raw["terminals"][keep])
    return raw, converted


def rows(method):
    return resolve(ROOT / "configs/iql.yaml", method, SEED, "/tmp/iql-smoke-not-written", DATASET)


class IQLWBCPSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = rows("bca")
        cls.driver, cls.m, cls.options = typed(cls.row)
        cls.args, cls.variants, cls.arms, _ = cls.driver.resolved_arguments(cls.options, cls.m)
        cls.train, cls.cal = synthetic(400, 1), synthetic(256, 2)
        m, args = cls.m, cls.args
        state, key, _ = m.H.initialize_agent(args, OBS, ACT, 1.0)
        cls.fitters = m.X.make_fitters(args, state, OBS, ACT, cls.variants)
        cls.carry0 = m.X.initialize_shared(args, state, key, cls.fitters, cls.arms)
        step = m.X.make_shared_train_step(args, cls.train, cls.fitters, cls.variants, cls.arms)
        cls.scan = staticmethod(jax.jit(lambda carry: jax.lax.scan(step, carry, None, STEPS)))
        cls.carry1, cls.pre = jax.device_get(cls.scan(cls.carry0))
        extra = cls.carry1.extras[0]
        cls.reference, cls.record = m.P.refresh(
            args.posterior, cls.fitters[0], extra.calibration, cls.carry1.nuisance,
            cls.cal, cls.carry1.rng, STEPS, discount=args.discount,
        )
        cls.carry2 = cls.carry1._replace(extras=(extra._replace(posterior=cls.reference),))
        cls.carry3, cls.post = jax.device_get(cls.scan(cls.carry2))

    def block_ok(self, metrics):
        self.assertIsNone(self.driver.D.first_failure(metrics, 1, np))
        self.driver.D.verify_block_evidence(metrics, STEPS, [asdict(a) for a in self.arms], "shared", np)
        counts = self.driver.verify_variant_evidence(
            metrics, STEPS, [asdict(v) for v in self.variants], self.args.batch_size, np)
        np.testing.assert_array_equal(counts, [STEPS])

    def batch_inputs(self, carry):
        batch = jax.tree_util.tree_map(lambda x: x[:128], self.train)
        agent = carry.nuisance
        adv = jnp.min(agent.qf_target.apply_fn(agent.qf_target.params, batch.obs, batch.action), -1)
        adv = adv - agent.vf.apply_fn(agent.vf.params, batch.obs)
        extra = carry.extras[0]
        preds = self.fitters[0].predictions(extra.calibration, extra.posterior.cal_params, batch)
        return self.m.X.arm_weights(self.args, self.arms, carry.extras, [preds], adv), np.asarray(adv)

    def evidence(self, n_cal):
        """data_metadata counts for a bank of n_cal rows thinned from 3x as many withheld rows."""
        withheld = 3 * n_cal
        return dict(training_size=4 * withheld, withheld_size=withheld, calibration_size=n_cal,
                    dataset_rows=5 * withheld,
                    rows_per_episode=self.args.posterior.reserve_rows_per_episode)

    # (a) real configs through the protocol validators
    def test_configs_resolve_and_type(self):
        self.assertEqual(self.row["run_id"], f"iql-{DATASET}-bca-wbcp-s{SEED}")
        self.assertEqual(self.row["calibration"], "wbcp_uniform")
        self.assertNotIn("fitting_mode", self.row["options"])
        self.assertFalse({"fit_size", "fitting"} & set(self.row["options"]["posterior_parameters"]))
        self.assertEqual(self.args.posterior.mode, "full")
        self.assertEqual((self.args.posterior.alpha, self.args.posterior.credibility,
                          self.args.posterior.draws), (0.1, 0.95, 1000))
        # The dataset's K reaches the posterior config (not the dataclass default).
        self.assertEqual(self.args.posterior.reserve_rows_per_episode,
                         self.row["options"]["posterior_parameters"]["reserve_rows_per_episode"])
        self.assertEqual([v.name for v in self.variants], ["wbcp_uniform"])
        self.assertEqual([(a.name, a.mode, a.variant_index) for a in self.arms],
                         [("host", "off", -1), ("bca", "full", 0)])
        declared = self.driver.declaration(self.options, self.m)
        self.assertEqual(declared["calibration"], "wbcp_uniform")
        host_row = rows("host")
        self.assertEqual(host_row["options"]["host_parameters"], self.row["options"]["host_parameters"])
        driver, m, options = typed(host_row)
        self.assertIs(driver, self.driver.D)
        self.assertEqual(driver.resolved_arguments(options, m)[0].posterior.mode, "off")

    # (b) before the first refresh the BCA actor is the native actor
    def test_native_weights_before_refresh(self):
        self.block_ok(self.pre)
        self.assertFalse(self.pre["posterior_ready"].any())
        np.testing.assert_array_equal(self.pre["weight_mean"][:, 0], self.pre["weight_mean"][:, 1])
        np.testing.assert_array_equal(self.pre["supported_fraction"], 1.0)
        weights, adv = self.batch_inputs(self.carry1)
        native = np.minimum(np.exp(self.args.beta * adv), 100.0)
        np.testing.assert_allclose(np.asarray(weights.weights), [native, native], rtol=1e-6)
        host, bca = (jax.tree_util.tree_map(lambda x: x[i], self.carry1.actors.params) for i in (0, 1))
        jax.tree_util.tree_map(np.testing.assert_array_equal, host, bca)

    def test_refresh_freezes_a_wbcp_threshold(self):
        ref, record, extra = self.reference, self.record, self.carry1.extras[0]
        self.assertTrue(bool(ref.ready) and bool(reference_valid(ref)))
        threshold = float(ref.threshold)
        self.assertTrue(math.isfinite(threshold))
        self.assertEqual(threshold, max(float(ref.lambda_hat), float(ref.lambda_hpd)))
        self.assertEqual(float(ref.n_eff), len(self.cal.reward))
        # The frozen scale is the one being fit at the refresh.
        jax.tree_util.tree_map(np.testing.assert_array_equal, ref.cal_params, extra.calibration.calibrator.params)
        self.assertEqual(float(ref.residual_scale), float(extra.calibration.resid_scale))
        # Independent recomputation: r + (1-d) gamma V(s') - min Q(s,a), scored by the frozen scale.
        agent, cal = self.carry1.nuisance, self.cal
        target = cal.reward + (1 - cal.done) * self.args.discount * agent.vf.apply_fn(agent.vf.params, cal.next_obs)
        q = jnp.min(agent.qf.apply_fn(agent.qf.params, cal.obs, cal.action), -1)
        eta = self.fitters[0].predictions(extra.calibration, ref.cal_params, cal)
        scores = np.abs(np.asarray(target - q, np.float64)) / np.asarray(
            jnp.maximum(eta, 1e-6) * ref.residual_scale, np.float64)
        key = jax.random.fold_in(self.carry1.rng, 1347375956)
        expected = wbcp.calibrate(scores, np.random.default_rng(np.asarray(key, np.uint32).ravel()),
                                  alpha=0.1, beta=0.95, draws=1000)
        self.assertEqual(threshold, np.float32(expected.threshold))
        # The record logs the WBCP diagnostics and passes the runtime verifier.
        self.assertEqual(record["calibration"], "wbcp_uniform")
        self.assertEqual(record["wbcp"]["threshold"], expected.threshold)
        self.assertEqual(record["wbcp"]["sigma_post"], expected.sigma_post)
        self.assertTrue(record["wbcp"]["certified"])
        row = json.loads(json.dumps(self.driver.D.plain(dict(
            record, variant=asdict(self.variants[0]),
            scale_updates=int(extra.calibration.calibrator.step),
            common_component_sha256={k: v for k, v in record["component_sha256"].items()
                                     if not k.startswith("actor_")}))))
        verified = self.driver.D.verify_reference_evidence(
            row, self.driver.D.plain(asdict(self.args.posterior)), self.evidence(len(cal.reward)))
        self.assertEqual(verified["threshold"], expected.threshold)
        self.driver.verify_common_references([row, row])
        again, _ = self.m.P.refresh(self.args.posterior, self.fitters[0], extra.calibration,
                                    agent, cal, self.carry1.rng, STEPS, discount=self.args.discount)
        self.assertEqual(float(again.threshold), threshold)

    def test_updates_consume_the_frozen_threshold(self):
        self.block_ok(self.post)
        self.assertTrue(self.post["posterior_ready"].all())
        np.testing.assert_array_equal(self.post["supported_fraction"], 1.0)
        self.assertTrue(np.all(self.post["weight_mean"][:, 1] <= self.post["weight_mean"][:, 0] + 1e-6))
        self.assertTrue(np.any(self.post["weight_mean"][:, 1] < self.post["weight_mean"][:, 0]))
        weights, adv = self.batch_inputs(self.carry2)
        host, bca = np.asarray(weights.weights)
        native = np.minimum(np.exp(self.args.beta * adv), 100.0)
        np.testing.assert_allclose(host, native, rtol=1e-6)
        positive = adv > 0
        self.assertTrue(positive.any() and (~positive).any())
        # Post-cap level weights: native for adv <= 0; in [1, native] for adv > 0.
        np.testing.assert_allclose(bca[~positive], native[~positive], rtol=1e-6)
        self.assertTrue(np.all(bca[positive] >= 1.0) and np.all(bca[positive] <= native[positive] * (1 + 1e-6)))
        self.assertTrue(np.any(bca[positive] < native[positive] * (1 - 1e-4)))
        self.assertTrue(bool(weights.has_support.all()) and bool(weights.inputs_valid.all()))
        host_p, bca_p = (jax.tree_util.tree_map(lambda x: x[i], self.carry3.actors.params) for i in (0, 1))
        self.assertFalse(all(jax.tree_util.tree_leaves(jax.tree_util.tree_map(np.array_equal, host_p, bca_p))))

    def test_uncertifiable_threshold_keeps_the_existing_fallback(self):
        ref = self.reference._replace(lambda_hpd=jnp.asarray(jnp.inf, jnp.float32),
                                      threshold=jnp.asarray(jnp.inf, jnp.float32))
        self.assertTrue(bool(reference_valid(ref)))
        carry = self.carry2._replace(extras=(self.carry2.extras[0]._replace(posterior=ref),))
        weights, _ = self.batch_inputs(carry)
        self.assertFalse(bool(weights.has_support[1]) or bool(weights.support_mask[1].any()))
        np.testing.assert_array_equal(np.asarray(weights.weights[1]), 0.0)
        _, metrics = jax.device_get(self.scan(carry))
        self.driver.D.verify_block_evidence(metrics, STEPS, [asdict(a) for a in self.arms], "shared", np)
        np.testing.assert_array_equal(metrics["supported_fraction"][:, 1], 0.0)
        self.assertFalse(metrics["actor_updated"][:, 1].any())
        self.assertTrue(metrics["actor_updated"][:, 0].all())

    def test_pair_runtime_advance_refresh_and_counters(self):
        runtime = object.__new__(self.driver.PairRuntime)  # no simulator: bypass __init__ only
        with tempfile.TemporaryDirectory() as out:
            np_ = self.m.np
            runtime.o, runtime.m, runtime.output = self.options, self.m, Path(out)
            runtime.args, runtime.variants, runtime.arms = self.args, self.variants, self.arms
            runtime.fitters, runtime.carry = self.fitters, self.carry0
            runtime.data = SimpleNamespace(train=self.train, calibration=self.cal)
            runtime.scan = lambda carry, n: type(self).scan(carry) if n == STEPS else None
            runtime.actor_applied = np_.zeros(2, np_.int64)
            runtime.actor_abstained, runtime.actor_post_applied = (np_.zeros(2, np_.int64) for _ in "ab")
            runtime.cal_applied, runtime.nuisance_valid_count = np_.zeros(1, np_.int64), 0
            runtime.artifacts = {"curves": {}, "references": {}}
            runtime.timings = dict(training_scan_seconds=0.0, evaluation_seconds=0.0,
                                   posterior_refresh_seconds=0.0)
            runtime.advance(STEPS)
            runtime.refresh(STEPS)
            runtime.advance(STEPS)
            self.assertEqual(runtime.cal_applied.tolist(), [2 * STEPS])
            self.assertEqual(runtime.actor_post_applied.tolist(), [STEPS, STEPS])
            self.assertEqual(runtime.state_counters()["calibrators"], {"wbcp_uniform": 2 * STEPS})
            saved = json.loads((Path(out) / f"reference_{STEPS}.json").read_text())
            (record,) = saved["variants"]
            self.driver.D.verify_reference_evidence(
                record, self.driver.D.plain(asdict(self.args.posterior)),
                self.evidence(len(self.cal.reward)))
            self.assertEqual(record["wbcp"]["threshold"], self.record["wbcp"]["threshold"])
            self.assertEqual(record["scale_updates"], STEPS)
            payload = (Path(out) / saved["checkpoint"]["path"]).read_bytes()
            decoded = checkpoint_counts(serialization.msgpack_restore(payload), "shared", STEPS,
                                        [STEPS, STEPS], [STEPS])
            self.assertTrue(decoded["references"][0]["ready"])

    def test_rows_per_episode_must_be_a_positive_integer(self):
        for mode in ("off", "full"):  # the host arm reserves the same pool, so it validates K too
            base = replace(self.args.posterior, mode=mode)
            base.validate()
            for bad in (0, -1, True, None, 5.0, "5"):
                with self.subTest(mode=mode, bad=bad), self.assertRaises(ValueError):
                    replace(base, reserve_rows_per_episode=bad).validate()

    def test_prepare_dataset_withholds_whole_episodes_and_thins_the_bank(self):
        lengths = np.random.default_rng(7).integers(10, 51, size=300)
        raw, converted = d4rl_like(lengths, 8)
        k, target = 5, 200
        args = replace(self.args, posterior=replace(
            self.args.posterior, reserve_size=target, reserve_rows_per_episode=k))
        data = self.m.P.prepare_dataset(args, converted, raw)
        train, withheld, cal = data.train_indices, data.withheld_indices, data.calibration_indices
        ids, n, meta = data.episode_ids, len(converted["rewards"]), data.metadata
        # training + withheld partition the rows; withheld rows never train
        np.testing.assert_array_equal(np.sort(np.r_[train, withheld]), np.arange(n))
        self.assertEqual(np.intersect1d(train, withheld).size, 0)
        self.assertFalse(set(ids[train]) & set(ids[withheld]))  # whole episodes are withheld
        # the WBCP bank is a strict subset of the withheld rows: K rows of each withheld episode
        self.assertTrue(set(cal) < set(withheld))
        reserved = np.unique(ids[withheld])
        self.assertEqual(len(reserved), -(-target // k))
        np.testing.assert_array_equal(np.unique(ids[cal]), reserved)
        np.testing.assert_array_equal(np.bincount(ids[cal], minlength=len(lengths))[reserved], k)
        self.assertEqual((len(data.train.reward), len(data.calibration.reward)), (len(train), len(cal)))
        np.testing.assert_array_equal(np.asarray(data.calibration.action), converted["actions"][cal])
        # observation statistics and the return range come from training episodes only
        np.testing.assert_allclose(np.asarray(data.obs_mean), converted["observations"][train].mean(0),
                                   rtol=1e-5, atol=1e-6)
        rewards = converted["rewards"].astype(np.float64)
        sums = [rewards[ids == e].sum() for e in np.setdiff1d(np.unique(ids), reserved)]
        np.testing.assert_allclose(meta["reward"]["return_range"], [min(sums), max(sums)], rtol=1e-12)
        # metadata records the design and passes the runtime evidence check
        self.assertEqual((meta["training_size"], meta["withheld_size"], meta["calibration_size"],
                          meta["dataset_rows"], meta["rows_per_episode"]),
                         (len(train), len(withheld), len(cal), n, k))
        self.assertFalse(meta["dependence_validated"])  # 40 episodes, below the validated 100
        self.assertEqual(meta["withheld_indices_sha256"], self.m.P.fingerprint(withheld))
        plain = self.driver.D.plain
        self.assertEqual(self.driver.D.verify_data_partition(
            json.loads(json.dumps(plain(meta))), plain(asdict(args.posterior))),
            (len(train), len(withheld), len(cal)))
        # without a reservation nothing is withheld and every row trains
        off = replace(args, posterior=replace(args.posterior, mode="off", reserve_size=0))
        full = self.m.P.prepare_dataset(off, converted, raw)
        np.testing.assert_array_equal(full.train_indices, np.arange(n))
        self.assertEqual((full.withheld_indices.size, full.calibration_indices.size), (0, 0))
        self.assertEqual((full.metadata["withheld_size"], full.metadata["calibration_size"]), (0, 0))

    def test_partition_evidence_rejects_banks_outside_the_withheld_rows(self):
        D = self.driver.D
        posterior, good = D.plain(asdict(self.args.posterior)), self.evidence(256)
        self.assertEqual(D.verify_data_partition(good, posterior), (3072, 768, 256))
        for change in (
            dict(withheld_size=None, dataset_rows=3072 + 256),  # the old training + bank partition
            dict(calibration_size=769),  # more bank rows than withheld rows
            dict(rows_per_episode=None),  # whole-episode bank
            dict(rows_per_episode=good["rows_per_episode"] + 1),  # not the declared K
            dict(training_size=768, dataset_rows=1536),  # half the data withheld, above the cap
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                D.verify_data_partition({**good, **change}, posterior)

    def test_small_bank_is_rejected(self):
        extra = self.carry1.extras[0]
        with self.assertRaises(ValueError):
            self.m.P.refresh(self.args.posterior, self.fitters[0], extra.calibration,
                             self.carry1.nuisance, synthetic(20, 3), self.carry1.rng, STEPS,
                             discount=self.args.discount)

    def test_shared_checkpoint_decodes_the_reference(self):
        tree = serialization.msgpack_restore(serialization.to_bytes(self.carry3))
        record = checkpoint_counts(tree, "shared", 2 * STEPS, [2 * STEPS, 2 * STEPS], [2 * STEPS])
        (reference,) = record["references"]
        self.assertTrue(reference["ready"])
        self.assertEqual(reference["threshold"], float(self.reference.threshold))
        json.dumps(record, allow_nan=False)
        initial = checkpoint_counts(serialization.msgpack_restore(serialization.to_bytes(self.carry1)),
                                    "shared", STEPS, [STEPS, STEPS], [STEPS])
        self.assertEqual(initial["references"][0]["threshold"], "Infinity")
        self.assertFalse(initial["references"][0]["ready"])

    # (c) the host arm is the plain host: no scale network, reference or calibration keys
    def test_host_arm_carries_no_calibration_state(self):
        driver, m, options = typed(rows("host"))
        args, arms, _ = driver.resolved_arguments(options, m)
        self.assertEqual([(a.name, a.mode) for a in arms], [("host", "off")])
        state, key, _ = m.H.initialize_agent(args, OBS, ACT, 1.0)
        native = m.H.BASE.make_train_step(args, state.actor.apply_fn, state.qf.apply_fn,
                                          state.vf.apply_fn, self.train)
        carry, _ = jax.device_get(jax.jit(lambda c: jax.lax.scan(native, c, None, 2))((key, state, jnp.int32(0))))
        self.assertEqual(len(carry), 3)
        self.assertEqual(carry[1]._fields, ("actor", "qf", "qf_target", "vf"))
        tree = serialization.msgpack_restore(serialization.to_bytes(carry))
        record = checkpoint_counts(tree, "native", 2, [2], [])
        self.assertNotIn("references", record)
        self.assertNotIn("calibrators", record)
        # Inside the pair, the host actor never reads a reference either.
        self.assertEqual(self.arms[0].variant_index, -1)


if __name__ == "__main__":
    unittest.main()
