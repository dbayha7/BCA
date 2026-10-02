"""CQL host + WBCP smoke test on CPU: typed configs, native steps, refresh, consumption.

No simulator, no real data, no GPU. The runtime check drives runtime.cql end to end
on a synthetic D4RL-shaped dataset with a stub vector environment.

JAX_PLATFORMS=cpu python -m unittest experiments.wbcp.test_cql_host
"""
import contextlib
import dataclasses
import io
import json
import math
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from calibration import wbcp
from calibration.dose import frozen_level_dose
from calibration.reference import FrozenReference, WBCPConfig, reference_valid
from runtime.config import ROOT, read_config, resolve, typed
from runtime.networks import Transition
from runtime.validation import checkpoint_counts, verify_events
import runtime.cql as R

P = R.P
CONFIG = ROOT / "configs/cql.yaml"
SEED = 202609171
DATASET = "hopper"  # hopper-medium-v2: 11-dimensional observations, 3-dimensional actions
OBS_DIM, ACTION_DIM = 11, 3
REFRESH_STEP = 5


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def typed_arm(method, dataset=DATASET):
    row = resolve(CONFIG, method, SEED, "/tmp/not-executed", dataset)
    return row, quiet(typed, row)


def synthetic(n, seed):
    g = np.random.default_rng(seed)
    obs = g.normal(size=(n, OBS_DIM)).astype(np.float32)
    return Transition(
        jnp.asarray(obs),
        jnp.asarray(np.tanh(g.normal(size=(n, ACTION_DIM))).astype(np.float32)),
        jnp.asarray(g.normal(size=n).astype(np.float32)),
        jnp.asarray(obs + 0.1 * g.normal(size=(n, OBS_DIM)).astype(np.float32)),
        jnp.asarray((g.uniform(size=n) < 0.02).astype(np.float32)),
    )


def refresh_key(args, step):
    """The runtime's derivation: never the training carry rng."""
    return jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(args.seed), 1380271698), step)


def spy_update(args, config, models, state, batch, it, key):
    """One jitted update that also returns the multiplier handed to native CQL.

    Returns (state, metrics, multiplier or None, whether a multiplier was passed).
    """
    box, original = {}, P.BASE.cql_update

    def spy(*positional, **named):
        box["passed"] = "conservative_multiplier" in named
        box["multiplier"] = named.get("conservative_multiplier")
        return original(*positional, **named)

    def run(state, batch, it, key):
        with patch.object(P.BASE, "cql_update", spy):
            new_state, metrics = P.update(args, config, models, state, batch, it, key, 1.0)
        return new_state, metrics, box["multiplier"]

    new_state, metrics, multiplier = jax.jit(run)(state, batch, it, key)
    return new_state, metrics, multiplier, box["passed"]


def scan(step, carry, length=4):
    return jax.lax.scan(step, carry, None, length=length)


class TypedConfigTests(unittest.TestCase):
    def test_real_configs_type_for_every_dataset_and_both_methods(self):
        for dataset in read_config(CONFIG)["datasets"]:
            rows = {}
            for method in ("host", "bca"):
                row, (runner, args, spec, protocol) = typed_arm(method, dataset)
                rows[method] = row
                self.assertIs(runner, R)
                self.assertIsInstance(spec, P.Config)
                self.assertEqual(spec.arm, method)
                self.assertEqual(spec.posterior, WBCPConfig(alpha=0.1, credibility=0.95, draws=1000))
                self.assertFalse(hasattr(protocol, "reference_size") or hasattr(protocol, "reference_seed"))
                self.assertIsNotNone(protocol.calibration_target_size)
                self.assertEqual(protocol.calibration_rows_per_episode,
                                 read_config(CONFIG)["datasets"][dataset]["reservation"]["rows_per_episode"])
                if method == "bca":
                    self.assertEqual(row["calibration"], "wbcp_uniform")
                    self.assertIn("-bca-wbcp-", row["run_id"])
                    self.assertEqual(len(protocol.refresh_steps), 198)
                    self.assertEqual(protocol.refresh_steps[0], 10000)
                else:
                    self.assertEqual(row["calibration"], "not_applicable")
                    self.assertEqual(protocol.refresh_steps, ())
            self.assertEqual(rows["host"]["native_args"], rows["bca"]["native_args"])

    def test_archived_inputs_and_mismatched_schedules_are_rejected(self):
        _, (_, args, bca, protocol) = typed_arm("bca")
        _, (_, _, host, host_protocol) = typed_arm("host")
        fields = {k: getattr(protocol, k) for k in protocol.__dataclass_fields__}
        with self.assertRaises(TypeError):
            R.RunProtocol(**fields, reference_size=1024, reference_seed=912)
        with self.assertRaises(ValueError):  # BCA without a refresh schedule
            R.validate_protocol(args, bca, dataclasses.replace(protocol, refresh_steps=()))
        with self.assertRaises(ValueError):  # native control carrying refreshes
            R.validate_protocol(args, host, dataclasses.replace(host_protocol, refresh_steps=(10000,)))
        for k in (None, 0, -1, True, 5.0):  # a target needs a positive integer K; no whole-episode bank
            with self.assertRaises(ValueError):
                dataclasses.replace(protocol, calibration_rows_per_episode=k)
        with self.assertRaises(ValueError):
            P.reserve_pool(synthetic(40, 0), 10, 0, None)
        with self.assertRaises(ValueError):
            P.Config(arm="bca", posterior=None)
        with self.assertRaises(TypeError):
            P.Config(arm="bca", iw={"mode": "off"})
        self.assertIsNone(P.Config(arm="host", posterior=None).posterior)
        for name in ("policy_log_weights", "ScaleIWConfig", "PolicyLogWeights", "fit_weighting",
                     "fit_posterior", "initialize_posterior", "PosteriorConfig"):
            self.assertFalse(hasattr(P, name), name)


class ArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, (_, args, bca, _) = typed_arm("bca")
        _, (_, _, host, _) = typed_arm("host")
        cls.args = dataclasses.replace(args, batch_size=64)
        cls.bca, cls.host = bca, host
        cls.training, cls.heldout = synthetic(320, 1), synthetic(160, 2)

    def initialize(self, config):
        return quiet(P.initialize, self.args, config, OBS_DIM, ACTION_DIM, 1.0)

    def test_bca_native_until_refresh_then_consumes_the_frozen_threshold(self):
        args, config, heldout = self.args, self.bca, self.heldout
        rng, state, models = self.initialize(config)
        self.assertIsInstance(state.posterior, FrozenReference)
        self.assertFalse(bool(state.posterior.ready))
        self.assertTrue(math.isinf(float(state.posterior.threshold)))
        self.assertTrue(bool(reference_valid(state.posterior)))
        step = P.make_train_step(args, config, models, self.training, 1.0)
        batch = jax.tree_util.tree_map(lambda x: x[:64], self.training)

        # Before any refresh: every row keeps the native multiplier 1.
        carry, metrics = scan(step, (rng, state, jnp.int32(0)), REFRESH_STEP - 1)
        P.require_valid(metrics)
        np.testing.assert_array_equal(np.asarray(metrics["critic_dose_mean"]), 1.0)
        self.assertFalse(np.any(np.asarray(metrics["posterior_ready"])))
        self.assertTrue(np.all(np.asarray(metrics["scale_fit_accepted"])))
        carry_state, _, multiplier, passed = spy_update(args, config, models, carry[1], batch,
                                                        jnp.int32(REFRESH_STEP), jax.random.PRNGKey(0))
        self.assertTrue(passed)
        np.testing.assert_array_equal(np.asarray(multiplier), np.ones(64, np.float32))
        carry = (carry[0], carry_state, jnp.int32(REFRESH_STEP))

        # Refresh: eager WBCP on the held-out block with the current scale frozen.
        before = carry[1]
        key = refresh_key(args, REFRESH_STEP)
        refreshed, refresh_metrics, diagnostics = P.refresh(args, config, models, before, heldout, key, 1.0)
        self.assertEqual(refresh_metrics, {"posterior_inputs_valid": True, "posterior_certified": True})
        P.require_valid(refresh_metrics)
        reference = refreshed.posterior
        self.assertTrue(bool(reference.ready) and bool(reference_valid(reference)))
        threshold = float(reference.threshold)
        self.assertTrue(math.isfinite(threshold) and threshold > 0)
        self.assertEqual(threshold, float(jnp.maximum(reference.lambda_hat, reference.lambda_hpd)))
        self.assertIs(refreshed.native, before.native)
        self.assertIs(refreshed.calibrator, before.calibrator)
        self.assertEqual(float(reference.residual_scale), float(before.residual_scale))
        for frozen, live in zip(jax.tree_util.tree_leaves(reference.cal_params),
                                jax.tree_util.tree_leaves(before.calibrator.params)):
            np.testing.assert_array_equal(np.asarray(frozen), np.asarray(live))

        # Independent recomputation: same scores, same fold_in constants, same WBCP.
        target = P.native_target(args, models, before.native, heldout,
                                 jax.random.fold_in(key, 1213156420), 1.0)
        q = jnp.minimum(models[1].apply(before.native.critic1.params, heldout.obs, heldout.action),
                        models[2].apply(before.native.critic2.params, heldout.obs, heldout.action))
        eta = models[3].apply(before.calibrator.params, heldout.obs, heldout.action)
        scale = np.asarray(jnp.maximum(eta, 1e-6) * before.residual_scale, np.float64)
        scores = np.abs(np.asarray(target - q, np.float64)) / scale
        expected = wbcp.calibrate(
            scores, np.random.default_rng(np.asarray(jax.random.fold_in(key, 1347375956), np.uint32).ravel()),
            alpha=config.posterior.alpha, beta=config.posterior.credibility, draws=config.posterior.draws)
        self.assertEqual(threshold, float(np.float32(expected.threshold)))
        self.assertEqual(diagnostics["threshold"], expected.threshold)
        self.assertEqual(diagnostics["scores"], len(heldout.obs))
        self.assertEqual(diagnostics["n_eff"], float(len(heldout.obs)))  # uniform weights
        self.assertEqual(set(diagnostics), {"certified", "threshold", "lambda_hat", "lambda_hpd",
                                            "sigma_post", "n_eff", "clamp_binds", "scores"})
        json.dumps(R._json_value(diagnostics), allow_nan=False)

        # Consumption: dose = 1 + blend * w / (w + u), w = threshold * frozen scale, in [1, 1 + blend).
        after, update_metrics, multiplier, passed = spy_update(
            args, config, models, refreshed, batch, jnp.int32(REFRESH_STEP + 1), jax.random.PRNGKey(1))
        self.assertTrue(passed)
        self.assertTrue(bool(update_metrics["inputs_valid"]) and bool(update_metrics["posterior_ready"]))
        multiplier = np.asarray(multiplier)
        frozen_eta = models[3].apply(reference.cal_params, batch.obs, batch.action)
        np.testing.assert_allclose(  # jit versus eager: float32 ulps only
            multiplier, np.asarray(frozen_level_dose(reference, frozen_eta, config.blend).dose), rtol=1e-6)
        width = threshold * np.maximum(np.asarray(frozen_eta), 1e-6) * float(reference.residual_scale)
        np.testing.assert_allclose(multiplier, 1 + config.blend * width / (width + float(reference.residual_scale)),
                                   rtol=1e-5)
        self.assertTrue(np.all((multiplier >= 1.0) & (multiplier < 1.0 + config.blend)))
        self.assertTrue(np.any(multiplier != 1.0))
        self.assertAlmostEqual(float(update_metrics["critic_dose_mean"]), float(multiplier.mean()), places=6)

        # Scanned updates keep consuming the same frozen reference while the scale fit moves on.
        carry, metrics = scan(step, (carry[0], after, jnp.int32(REFRESH_STEP + 1)), REFRESH_STEP - 1)
        P.require_valid(metrics)
        dose = np.asarray(metrics["critic_dose_mean"])
        self.assertTrue(np.all((dose > 1.0) & (dose < 1.0 + config.blend)))
        self.assertTrue(np.all(np.asarray(metrics["posterior_ready"])))
        self.assertEqual(float(carry[1].posterior.threshold), threshold)
        self.assertFalse(all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(
            jax.tree_util.tree_leaves(carry[1].calibrator.params),
            jax.tree_util.tree_leaves(reference.cal_params))))
        R._accept(carry[1], {})

    def test_invalid_refresh_keeps_state_and_aborts(self):
        rng, state, models = self.initialize(self.bca)
        broken = self.heldout._replace(reward=self.heldout.reward.at[3].set(jnp.nan))
        result, metrics, diagnostics = P.refresh(self.args, self.bca, models, state, broken,
                                                 refresh_key(self.args, 10000), 1.0)
        self.assertIs(result, state)
        self.assertEqual(metrics, {"posterior_inputs_valid": False, "posterior_certified": False})
        self.assertFalse(diagnostics["certified"])
        with self.assertRaises(FloatingPointError):
            P.require_valid(metrics)
        with self.assertRaises(FloatingPointError):
            R._accept(result, metrics)

    def test_host_arm_carries_no_calibration_state(self):
        rng, state, models = self.initialize(self.host)
        self.assertEqual((state.calibrator, state.residual_scale, state.posterior), (None, None, None))
        self.assertIsNone(models[3])
        batch = jax.tree_util.tree_map(lambda x: x[:64], self.training)
        _, metrics, multiplier, passed = spy_update(self.args, self.host, models, state, batch,
                                                    jnp.int32(1), jax.random.PRNGKey(2))
        self.assertFalse(passed)
        self.assertIsNone(multiplier)
        for name in ("critic_dose_mean", "scale_loss", "posterior_ready", "inputs_valid"):
            self.assertNotIn(name, metrics)
        carry, _ = scan(P.make_train_step(self.args, self.host, models, self.training, 1.0),
                        (rng, state, jnp.int32(0)), 2)
        self.assertEqual((carry[1].calibrator, carry[1].residual_scale, carry[1].posterior), (None, None, None))
        with self.assertRaises(ValueError):
            P.refresh(self.args, self.host, models, state, self.heldout, refresh_key(self.args, 1), 1.0)


class StubVectorEnv:
    def __init__(self, workers):
        self.num_envs = workers

    def seed(self, seeds):
        self.seeds = list(seeds)

    def reset(self):
        return np.zeros((self.num_envs, OBS_DIM))

    def step(self, action):
        n = self.num_envs
        return np.zeros((n, OBS_DIM)), np.ones(n), np.ones(n, bool), {}

    def close(self):
        pass


def synthetic_raw(n=1200, episode=40, seed=3):
    g = np.random.default_rng(seed)
    ends = (np.arange(n) % episode) == episode - 1
    terminal = ends & ((np.arange(n) // episode) % 2 == 0)
    return {
        "observations": g.normal(size=(n, OBS_DIM)).astype(np.float32),
        "actions": np.tanh(g.normal(size=(n, ACTION_DIM))).astype(np.float32),
        "rewards": g.normal(size=n).astype(np.float32),
        "terminals": terminal,
        "timeouts": ends & ~terminal,
    }


class RuntimeTests(unittest.TestCase):
    def assert_thinned_reservation(self, data, protocol):
        """Training and withheld partition the rows; the bank is K rows of each withheld episode."""
        m = data.metadata
        train, withheld, cal = (np.asarray(m[name + "_converted_indices"])
                                for name in ("training", "withheld", "heldout"))
        ids = np.asarray(m["converted_episode_ids"])
        np.testing.assert_array_equal(np.sort(np.r_[train, withheld]), np.arange(m["converted_rows"]))
        self.assertFalse(set(train) & set(withheld))  # withheld rows never train
        self.assertFalse(set(ids[train]) & set(ids[withheld]))  # ... nor any row of their episodes
        self.assertTrue(set(cal) < set(withheld))  # the bank is a strict subset of withheld
        episodes, counts = np.unique(ids[cal], return_counts=True)
        np.testing.assert_array_equal(episodes, np.unique(ids[withheld]))
        np.testing.assert_array_equal(counts, protocol.calibration_rows_per_episode)
        # The learner's pools hold exactly those rows (actions are not transformed).
        converted, _, raw_ids = R._convert(synthetic_raw(), 1000)
        np.testing.assert_array_equal(np.asarray(data.training.action), converted["actions"][train])
        np.testing.assert_array_equal(np.asarray(data.heldout.action), converted["actions"][cal])
        # Re-calling the primitive reproduces every returned array and the metadata.
        again = P.reserve_pool(
            R.C.Transition(converted["observations"], converted["actions"], converted["rewards"],
                           converted["next_observations"], converted["terminals"]),
            protocol.calibration_target_size, protocol.calibration_seed,
            protocol.calibration_rows_per_episode, max_fraction=protocol.calibration_max_fraction,
            episode_ids=raw_ids)
        for recorded, expected in zip((train, withheld, cal), again[:3]):
            np.testing.assert_array_equal(recorded, expected)
        self.assertEqual(m["reservation"], again[3])
        self.assertEqual(m["reservation"]["rows_per_episode"], protocol.calibration_rows_per_episode)
        self.assertNotIn("dependence_validated", m["reservation"])  # evidence lives in the resolved row

    def test_prepare_and_run_journal_the_wbcp_refresh(self):
        tiny = dict(num_updates=4, eval_interval=4, eval_workers=1, eval_final_episodes=1)
        prepared = {}
        for method in ("host", "bca"):
            _, (_, args, config, protocol) = typed_arm(method)
            args = dataclasses.replace(args, batch_size=64, **tiny)
            # Synthetic episodes have 39-40 rows: K = 15 banks 60 rows from 4 episodes, about
            # 13% of the data withheld, so the cap is the primitive's default 0.25.
            protocol = dataclasses.replace(
                protocol, **tiny, scan_block_size=2, eval_periodic_episodes=1, calibration_target_size=60,
                calibration_rows_per_episode=15, calibration_max_fraction=0.25,
                refresh_steps=(2,) if method == "bca" else (),
                evaluation_events=(R.EvaluationEvent("periodic", 4, (11,)), R.EvaluationEvent("final", 4, (12,))))
            data = quiet(R.prepare, synthetic_raw(), args, config, protocol, max_action=1.0, max_episode_steps=1000)
            self.assert_thinned_reservation(data, protocol)
            self.assertNotIn("reference", R.PreparedData.__dataclass_fields__)
            for name in ("reference_raw_indices", "reference_selection"):
                self.assertNotIn(name, data.metadata)
            self.assertEqual(set(data.metadata["run_input_hashes"]),
                             {"training", "heldout", "obs_mean", "obs_std", "max_action", "max_episode_steps"})
            self.assertEqual(data.metadata["calibration"], "wbcp_uniform" if method == "bca" else "not_applicable")
            prepared[method] = (args, config, protocol, data)
        host_hashes, bca_hashes = (prepared[m][3].metadata["run_input_hashes"] for m in ("host", "bca"))
        self.assertEqual(host_hashes["training"], bca_hashes["training"])  # paired split
        self.assertEqual(host_hashes["heldout"], bca_hashes["heldout"])
        self.assertEqual(*(prepared[m][3].metadata["withheld_converted_indices"] for m in ("host", "bca")))

        args, config, protocol, data = prepared["bca"]
        journal, checkpoints = [], []

        def on_checkpoint(step, carry):  # train.py's serialization and counter audit
            tree = {"state": carry[1], "training_rng": carry[0], "step": carry[2]}
            payload = serialization.to_bytes(tree)
            restored = serialization.from_bytes(tree, payload)
            checkpoints.append((step, checkpoint_counts(serialization.msgpack_restore(payload), "cql", step),
                                restored["state"].posterior, carry[1].posterior))

        result = quiet(R.run_prepared, args, config, protocol, data, env_factory=lambda _, n: StubVectorEnv(n),
                       on_event=lambda record: journal.append(json.dumps(record, allow_nan=False)),
                       on_checkpoint=on_checkpoint)
        self.assertEqual([c[0] for c in checkpoints], [4])
        _, counts, restored, live = checkpoints[0]
        self.assertEqual(counts["accepted_scale_fits"], 4)
        for a, b in zip(jax.tree_util.tree_leaves(restored), jax.tree_util.tree_leaves(live)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        verify_events(protocol, result["events"], result["evaluations"], host="cql")
        refreshes = [e for e in result["events"] if e["kind"] == "refresh"]
        self.assertEqual([e["step"] for e in refreshes], [2])
        event = refreshes[0]
        self.assertEqual(event["posterior_weighting"], "wbcp_uniform")
        self.assertEqual(event["metrics"], {"posterior_inputs_valid": True, "posterior_certified": True})
        self.assertTrue(event["wbcp"]["certified"])
        self.assertEqual(event["wbcp"]["scores"], len(data.heldout.obs))
        final = result["state"].posterior
        self.assertTrue(bool(final.ready) and bool(reference_valid(final)))
        self.assertEqual(float(final.threshold), float(np.float32(event["wbcp"]["threshold"])))
        scans = [e for e in result["events"] if e["kind"] == "accepted_scan"]
        self.assertEqual(scans[0]["metrics_last"]["critic_dose_mean"], 1.0)
        self.assertGreater(scans[-1]["metrics_last"]["critic_dose_mean"], 1.0)
        self.assertEqual(len(journal), len(result["events"]))


if __name__ == "__main__":
    unittest.main()
