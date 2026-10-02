"""CPU smoke test for the ReBRAC host and its uniform-weight WBCP BCA arm.

Synthetic transitions only; no simulator, GPU or real training.
JAX_PLATFORMS=cpu python -m unittest experiments.wbcp.test_rebrac_host
"""
import math
import unittest
from dataclasses import replace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

import algorithms.rebrac_bca as P
import runtime.rebrac as R
from calibration.reference import FrozenReference, WBCPConfig, freeze_reference, reference_valid
from runtime.config import ROOT, resolve, typed
from runtime.validation import checkpoint_counts, verify_events
from runtime.networks import TransitionNA

DATASET, SEED = "hopper", 202609171
OBS_DIM, ACTION_DIM, STEPS = 11, 3, 6


def rows(method):
    return resolve(ROOT / "configs/rebrac.yaml", method, SEED, "/tmp/wbcp-smoke", DATASET)


def synthetic(n, seed):
    g = np.random.default_rng(seed)
    obs = g.normal(size=(n, OBS_DIM)).astype(np.float32)
    action = g.uniform(-0.9, 0.9, (n, ACTION_DIM)).astype(np.float32)
    reward = (np.tanh(obs[:, 0]) + 0.1 * g.normal(size=n)).astype(np.float32)
    next_obs = (obs + 0.05 * g.normal(size=obs.shape)).astype(np.float32)
    done = (g.uniform(size=n) < 0.02).astype(np.float32)
    next_action = g.uniform(-0.9, 0.9, (n, ACTION_DIM)).astype(np.float32)
    return TransitionNA(*map(jnp.asarray, (obs, action, reward, next_obs, done, next_action)))


def scan(config, models, data, args, carry, length=STEPS):
    return jax.lax.scan(P.make_train_step(args, config, models, data), carry, None, length=length)


class ReBRACWBCPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.typed = {m: typed(rows(m)) for m in ("host", "bca")}
        _, args, spec, cls.protocol = cls.typed["bca"]
        cls.args = replace(args, batch_size=64)
        cls.cfg, cls.host_cfg = spec.config(), cls.typed["host"][2].config()
        cls.train, cls.heldout = synthetic(400, 1), synthetic(160, 2)
        cls.batch = jax.tree_util.tree_map(lambda x: x[:64], cls.train)

        rng, state, cls.models = P.initialize(cls.args, cls.cfg, cls.train)
        cls.initial = state
        cls.pre_carry, cls.pre = scan(cls.cfg, cls.models, cls.train, cls.args, (rng, state, jnp.int32(0)))
        cls.pre_dose = P.bc_readout(cls.models, cls.pre_carry[1], cls.batch, cls.cfg.blend)

        event = cls.protocol.refresh_events[0]
        cls.key = jax.random.fold_in(jax.random.PRNGKey(event.seed), event.step)
        cls.refreshed, cls.refresh_metrics, cls.diagnostics = P.refresh(
            cls.args, cls.cfg, cls.models, cls.pre_carry[1], cls.heldout, cls.key,
            heldout_ids=np.arange(1000, 1000 + len(cls.heldout.obs)),
        )
        carry = (cls.pre_carry[0], cls.refreshed, cls.pre_carry[2])
        cls.post_carry, cls.post = scan(cls.cfg, cls.models, cls.train, cls.args, carry)
        cls.post_dose = P.bc_readout(cls.models, cls.post_carry[1], cls.batch, cls.cfg.blend)

        host_rng, host_state, cls.host_models = P.initialize(cls.args, cls.host_cfg, cls.train)
        cls.host_state = host_state
        cls.host_carry, cls.host = scan(cls.host_cfg, cls.host_models, cls.train, cls.args,
                                        (host_rng, host_state, jnp.int32(0)))

    def test_real_configs_resolve_and_type(self):
        host_row, bca_row = rows("host"), rows("bca")
        self.assertEqual((bca_row["calibration"], host_row["calibration"]), ("wbcp_uniform", "not_applicable"))
        self.assertIn("bca-wbcp", bca_row["run_id"])
        self.assertEqual(host_row["native_args"], bca_row["native_args"])
        _, _, spec, protocol = self.typed["bca"]
        self.assertEqual(self.cfg.posterior, WBCPConfig(0.1, 0.95, 1000))
        self.assertEqual((self.cfg.arm, self.cfg.blend), ("bca", 0.5))
        self.assertEqual(len(protocol.refresh_events), 198)
        self.assertFalse(hasattr(protocol, "reference"))
        self.assertFalse(hasattr(spec.posterior, "affinity"))
        self.assertEqual(self.host_cfg, P.Config("host"))
        self.assertEqual(self.typed["host"][3].refresh_events, ())
        self.assertEqual(protocol.reservation.rows_per_episode, bca_row["protocol"]["reservation"]["rows_per_episode"])
        with self.assertRaises(ValueError):
            P.Config("host", posterior=WBCPConfig())
        with self.assertRaises(ValueError):
            P.Config("bca", posterior=None, blend=0.5)

    def test_reservation_rows_per_episode_is_a_positive_integer(self):
        self.assertEqual(R.Reservation(100, 911, 0.1, R.DEPENDENCY_CONTRACT, 5).rows_per_episode, 5)
        for bad in (None, 0, -1, True, 2.0):  # no whole-episode population split for ReBRAC
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                R.Reservation(100, 911, 0.1, R.DEPENDENCY_CONTRACT, bad)
        with self.assertRaises(TypeError):
            R.Reservation(100, 911, 0.1, R.DEPENDENCY_CONTRACT)

    def test_native_multiplier_before_refresh(self):
        P.require_valid(self.pre)
        self.assertFalse(bool(self.initial.posterior.ready))
        self.assertTrue(bool(P.calibration_storage_valid(self.models, self.initial)))
        self.assertTrue(math.isinf(float(self.initial.posterior.threshold)))
        np.testing.assert_array_equal(np.asarray(self.pre["bc_multiplier_mean"]), 1.0)
        self.assertFalse(np.any(np.asarray(self.pre["posterior_ready"])))
        self.assertTrue(np.all(np.asarray(self.pre["scale_fit_accepted"])))
        np.testing.assert_array_equal(np.asarray(self.pre_dose.dose), 1.0)
        self.assertEqual(int(self.pre_carry[1].calibrator.step), STEPS)
        self.assertEqual(int(self.pre_carry[2]), STEPS)
        for removed in ("scale_ess_abstained", "iw_ess", "iw_tau", "iw_product_ess"):
            self.assertNotIn(removed, self.pre)
        # The fixed calibrator normalization is the training-pool population statistics.
        np.testing.assert_array_equal(self.initial.cal_obs_mean, jnp.mean(self.train.obs, axis=0))
        np.testing.assert_array_equal(self.initial.cal_obs_std, jnp.std(self.train.obs, axis=0, ddof=0))

    def test_refresh_freezes_a_certified_wbcp_threshold(self):
        ref, m, d = self.refreshed.posterior, self.refresh_metrics, self.diagnostics
        self.assertIsInstance(ref, FrozenReference)
        self.assertEqual(m, dict(posterior_inputs_valid=True, posterior_certified=True))
        self.assertTrue(bool(ref.ready) and bool(reference_valid(ref)) and d["certified"])
        self.assertTrue(bool(P.calibration_storage_valid(self.models, self.refreshed)))
        self.assertTrue(math.isfinite(float(ref.threshold)) and float(ref.threshold) > 0)
        self.assertEqual(float(ref.threshold), float(jnp.maximum(ref.lambda_hat, ref.lambda_hpd)))
        self.assertEqual(float(ref.threshold), float(np.float32(d["threshold"])))
        self.assertEqual(set(d), {"certified", "threshold", "lambda_hat", "lambda_hpd", "sigma_post",
                                  "n_eff", "clamp_binds", "scores"})
        self.assertEqual((d["scores"], d["n_eff"]), (160, 160.0))
        before = self.pre_carry[1]
        for name in ("native", "calibrator", "residual_scale", "cal_obs_mean", "cal_obs_std"):
            self.assertEqual(R._tree_hash(getattr(self.refreshed, name)), R._tree_hash(getattr(before, name)))
        self.assertEqual(R._tree_hash(ref.cal_params), R._tree_hash(before.calibrator.params))
        self.assertEqual(float(ref.residual_scale), float(before.residual_scale))

    def test_refresh_draws_come_from_the_folded_refresh_key(self):
        state, heldout = self.pre_carry[1], self.heldout
        target = P.native_target(self.args, self.models, state.native, heldout,
                                 jax.random.fold_in(self.key, P.KEY_REFRESH_NOISE))
        q = self.models[1].apply(state.native.critic.params, heldout.obs, heldout.action).min(0)
        eta = self.models[2].apply(state.calibrator.params, heldout.obs, heldout.action)
        expected, valid, _ = freeze_reference(state.calibrator.params, state.residual_scale, eta, target - q,
                                              jax.random.fold_in(self.key, P.KEY_POSTERIOR), self.cfg.posterior)
        self.assertTrue(valid)
        self.assertEqual(P.KEY_POSTERIOR, 1380994899)
        for name in ("threshold", "lambda_hat", "lambda_hpd", "n_eff"):
            self.assertEqual(float(getattr(self.refreshed.posterior, name)), float(getattr(expected, name)))
        other, _, _ = P.refresh(self.args, self.cfg, self.models, state, heldout,
                                jax.random.fold_in(self.key, 1), heldout_ids=np.arange(len(heldout.obs)))
        self.assertNotEqual(float(other.posterior.threshold), float(self.refreshed.posterior.threshold))

    def test_updates_consume_the_frozen_reference(self):
        P.require_valid(self.post)
        blend = self.cfg.blend
        self.assertTrue(np.all(np.asarray(self.post["posterior_ready"])))
        means = np.asarray(self.post["bc_multiplier_mean"])
        self.assertTrue(np.all((means > 1.0) & (means <= 1.0 + blend)))
        dose = np.asarray(self.post_dose.dose)
        self.assertTrue(bool(self.post_dose.inputs_valid) and np.all(np.asarray(self.post_dose.support_mask)))
        self.assertTrue(np.all((dose >= 1.0) & (dose <= 1.0 + blend)))
        self.assertTrue(np.any(dose != 1.0))
        self.assertGreater(float(np.ptp(dose)), 0.0)
        ref, after = self.refreshed.posterior, self.post_carry[1].posterior
        self.assertEqual(R._tree_hash(after), R._tree_hash(ref))
        self.assertNotEqual(R._tree_hash(self.post_carry[1].calibrator.params), R._tree_hash(ref.cal_params))

    def test_invalid_refresh_keeps_the_previous_reference(self):
        broken = self.heldout._replace(reward=self.heldout.reward.at[0].set(jnp.nan))
        state, metrics, _ = P.refresh(self.args, self.cfg, self.models, self.post_carry[1], broken, self.key,
                                      heldout_ids=np.arange(len(broken.obs)))
        self.assertFalse(metrics["posterior_inputs_valid"] or metrics["posterior_certified"])
        self.assertIs(state, self.post_carry[1])
        with self.assertRaises(FloatingPointError):
            P.require_valid(metrics)
        with self.assertRaises(ValueError):
            P.refresh(self.args, self.cfg, self.models, self.post_carry[1], self.heldout, self.key,
                      heldout_ids=np.zeros(len(self.heldout.obs), int))

    def test_host_arm_carries_no_calibration_state(self):
        state = self.host_carry[1]
        self.assertEqual(state[1:], (None,) * 5)
        self.assertEqual(self.host_models[2], None)
        for name in ("bc_multiplier_mean", "scale_inputs_valid", "posterior_ready", "scale_loss"):
            self.assertNotIn(name, self.host)
        self.assertTrue(np.all(np.asarray(self.host["inputs_valid"])))
        with self.assertRaises(ValueError):
            P.refresh(self.args, self.host_cfg, self.host_models, state, self.heldout, self.key,
                      heldout_ids=np.arange(len(self.heldout.obs)))
        # Paired arms: before the first refresh the BCA multiplier is exactly 1, so the host and
        # BCA native states must stay bitwise identical on the same minibatch stream.
        self.assertEqual(R._tree_hash(self.host_carry[1].native), R._tree_hash(self.pre_carry[1].native))
        self.assertEqual(R._tree_hash(self.host_carry[0]), R._tree_hash(self.pre_carry[0]))
        self.assertEqual(R._tree_hash(self.host_state.native), R._tree_hash(self.initial.native))


    def test_checkpoint_round_trip_with_frozen_reference(self):
        payload = serialization.to_bytes(
            {"state": self.post_carry[1], "training_rng": self.post_carry[0], "step": self.post_carry[2]})
        tree = serialization.msgpack_restore(payload)
        counts = checkpoint_counts(tree, "rebrac", 2 * STEPS, self.args.policy_freq)
        self.assertEqual(counts["accepted_scale_fits"], 2 * STEPS)
        restored = serialization.from_bytes(
            {"state": self.post_carry[1], "training_rng": self.post_carry[0], "step": self.post_carry[2]}, payload)
        self.assertEqual(R._tree_hash(restored["state"].posterior), R._tree_hash(self.post_carry[1].posterior))
        self.assertIsInstance(restored["state"].posterior, FrozenReference)

class FakeEnv:
    def __init__(self, workers):
        self.workers = workers

    def seed(self, seeds):
        assert len(seeds) == self.workers

    def reset(self):
        return np.zeros((self.workers, OBS_DIM))

    def step(self, actions):
        return np.zeros((self.workers, OBS_DIM)), np.ones(self.workers), np.ones(self.workers), {}

    def close(self):
        pass


def raw_dataset(episodes=40, length=50, seed=3):
    g = np.random.default_rng(seed)
    n = episodes * length
    timeouts = np.zeros(n, bool)
    timeouts[length - 1::length] = True
    return dict(
        observations=g.normal(size=(n, OBS_DIM)).astype(np.float32),
        actions=g.uniform(-0.9, 0.9, (n, ACTION_DIM)).astype(np.float32),
        rewards=g.normal(size=n).astype(np.float32),
        terminals=np.zeros(n, bool),
        timeouts=timeouts,
    )


ROWS_PER_EPISODE = 25  # 40 episodes of 49 converted rows: 4 withheld (196 rows), 100 calibrated


class ReBRACRuntimeTests(unittest.TestCase):
    """Short prepare/run_prepared pass through the rewired ReBRAC runtime."""

    def prepare_arm(self, method):
        _, args, spec, _ = typed(rows(method))
        args = replace(args, num_updates=20, eval_interval=10, eval_workers=1, eval_final_episodes=1,
                       batch_size=32)  # native ReBRAC Args require a nonempty final bank
        protocol = R.RunProtocol(
            run_id="rebrac-wbcp-smoke-" + method, seed=args.seed, num_updates=20, scan_block_size=10,
            eval_interval=10, eval_workers=1, eval_periodic_episodes=1, eval_final_episodes=1,
            reservation=R.Reservation(100, 911, 0.15, R.DEPENDENCY_CONTRACT, ROWS_PER_EPISODE),
            refresh_events=(R.RefreshEvent(5, 77), R.RefreshEvent(15, 78)) if method == "bca" else (),
            evaluation_events=(R.EvaluationEvent("periodic", 10, (1,)), R.EvaluationEvent("periodic", 20, (2,)),
                               R.EvaluationEvent("final", 20, (3,))),
        )
        identity = dict(kind="synthetic", dataset=args.dataset, label="wbcp smoke")
        with mock.patch.object(R, "source_identity", lambda: {"fixed": "smoke"}):
            prepared = R.prepare(raw_dataset(), args, spec, protocol, max_action=1.0, max_episode_steps=50,
                                 raw_identity=identity)
        return args, spec, protocol, prepared

    def run_arm(self, method):
        args, spec, protocol, prepared = self.prepare_arm(method)
        with mock.patch.object(R, "source_identity", lambda: {"fixed": "smoke"}):
            result = R.run_prepared(args, spec, protocol, prepared, env_factory=lambda d, w: FakeEnv(w))
        verify_events(protocol, result["events"], result["evaluations"], host="rebrac")
        return prepared, result

    def test_withheld_components_leave_training_and_the_bank_thins_them(self):
        args, spec, protocol, prepared = self.prepare_arm("bca")
        m = prepared.metadata
        n = m["converted_rows"]
        train, bank = np.asarray(prepared.training_ids), np.asarray(prepared.heldout_ids)
        withheld = np.asarray(m["withheld_converted_ids"])
        np.testing.assert_array_equal(np.sort(np.r_[train, withheld]), np.arange(n))  # a partition
        self.assertFalse(np.intersect1d(train, withheld).size)  # no withheld row trains
        self.assertTrue(np.all(np.isin(bank, withheld)) and len(bank) < len(withheld))  # strict subset
        self.assertEqual((len(bank), len(withheld)), (100, 4 * 49))
        component = np.asarray(m["dependency_maps"]["effective_components"])
        self.assertTrue(np.array_equal(np.flatnonzero(np.isin(component, component[withheld])), withheld))
        self.assertTrue(np.all(np.bincount(component[bank])[np.unique(component[bank])] == ROWS_PER_EPISODE))
        self.assertFalse(m["partition"]["effective_overlap"])
        self.assertFalse(np.intersect1d(m["partition"]["training_effective_raw_rows"],
                                        m["partition"]["withheld_effective_raw_rows"]).size)
        inherited = m["reservation"]["inherited_primitive_metadata"]
        self.assertEqual((inherited["rows_per_episode"], inherited["withheld_size"], inherited["calibration_size"]),
                         (ROWS_PER_EPISODE, len(withheld), len(bank)))
        self.assertIn("K stratified rows", m["reservation"]["meaning"])
        self.assertEqual(len(prepared.heldout.obs), len(bank))
        # The calibrator's fixed statistics come from the training rows only.
        np.testing.assert_array_equal(prepared.cal_obs_mean, jnp.mean(prepared.training.obs, axis=0))
        with mock.patch.object(R, "source_identity", lambda: {"fixed": "smoke"}):
            R.validate_prepared(args, spec, protocol, prepared)
            # validate_prepared reruns the reservation, so a resealed record with other withheld rows fails.
            forged = dict(m, withheld_converted_ids=[i for i in withheld.tolist() if i != int(bank[0])])
            with self.assertRaisesRegex(ValueError, "partition"):
                R.validate_prepared(args, spec, protocol, replace(prepared, metadata=forged,
                                                                  metadata_sha256=R._digest(forged)))

    def test_bca_run_logs_wbcp_refreshes(self):
        prepared, result = self.run_arm("bca")
        m = prepared.metadata
        self.assertEqual(m["schema"], "native-rebrac-prepared-v4")
        self.assertNotIn("reference_converted_ids", m)
        self.assertNotIn("reference_effective_raw_rows", m["partition"])
        self.assertFalse(hasattr(prepared, "reference"))
        refreshes = [e for e in result["events"] if e["kind"] == "refresh"]
        self.assertEqual([e["step"] for e in refreshes], [5, 15])
        for event in refreshes:
            self.assertEqual(event["posterior_weighting"], "wbcp_uniform")
            self.assertEqual(event["metrics"], dict(posterior_inputs_valid=True, posterior_certified=True))
            wbcp = event["wbcp"]
            self.assertTrue(wbcp["certified"] and math.isfinite(wbcp["threshold"]))
            self.assertEqual(wbcp["threshold"], max(wbcp["lambda_hat"], wbcp["lambda_hpd"]))
            self.assertEqual(wbcp["scores"], len(m["heldout_converted_ids"]))
            for gone in ("radii", "component_diagnostics", "reference_sha256"):
                self.assertNotIn(gone, event)
        scans = {e["step"]: e["metrics"] for e in result["events"] if e["kind"] == "accepted_scan"}
        self.assertTrue(np.all(np.asarray(scans[5]["bc_multiplier_mean"]) == 1.0))
        self.assertTrue(np.all(np.asarray(scans[10]["bc_multiplier_mean"]) > 1.0))
        state = result["state"]
        self.assertTrue(bool(state.posterior.ready) and bool(reference_valid(state.posterior)))
        self.assertEqual(result["steps_completed"], 20)

    def test_host_run_has_no_calibration(self):
        prepared, result = self.run_arm("host")
        self.assertIsNone(result["state"].calibrator)
        self.assertIsNone(result["state"].posterior)
        self.assertIsNone(prepared.cal_obs_mean)
        self.assertFalse(any(e["kind"] == "refresh" for e in result["events"]))
        self.assertEqual(result["steps_completed"], 20)


if __name__ == "__main__":
    unittest.main()
