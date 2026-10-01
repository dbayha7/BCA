"""CPU checks for experiments/wbcp/freeze_rebrac.py, ReBRAC's frozen score artifact for the D4RL benchmark.

A synthetic HDF5 file drives the real preparation, training and scoring code: agreement with
algorithms.rebrac_bca.refresh on the same bank and state, the whole-component split (a
partition, TD3+BC's population rows for the same split seed), the artifact read back by
d4rl_benchmark.load_pool, reproducibility, and the runtime's refusal to run a population
split. One more test runs 200 updates on the cached hopper-medium-v2 file when it exists.
JAX_PLATFORMS=cpu python -m unittest experiments.wbcp.test_freeze_rebrac
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import jax
import numpy as np

import algorithms.rebrac_bca as P
import calibration.wbcp
import runtime.rebrac as R
from calibration.reference import positive_scale
from experiments.wbcp import d4rl_benchmark as bench
from experiments.wbcp import freeze_rebrac as FR
from experiments.wbcp import freeze_scores as F
from runtime.provenance import sha

SEED, OBS_DIM, ACTION_DIM, UPDATES = 202609171, 11, 3, 20
HOPPER = Path.home() / ".d4rl" / "datasets" / "hopper_medium-v2.hdf5"
TD3_HOPPER = FR.ROOT / "runs" / "wbcp_frozen" / "hopper-medium-v2-s202609171-u100000" / "frozen.npz"


def write_raw(path, episodes=16, length=25, seed=3):
    """D4RL-shaped raw transitions: fixed-length episodes ending in timeouts, one early terminal."""
    g = np.random.default_rng(seed)
    n = episodes * length
    obs = g.normal(size=(n, OBS_DIM)).astype(np.float32)
    terminals, timeouts = np.zeros(n, bool), np.zeros(n, bool)
    timeouts[length - 1::length] = True
    terminals[3 * length + 10] = True
    with h5py.File(path, "w") as f:
        f.create_dataset("observations", data=obs)
        f.create_dataset("actions", data=g.uniform(-0.9, 0.9, (n, ACTION_DIM)).astype(np.float32))
        f.create_dataset("rewards", data=(np.tanh(obs[:, 0]) + 0.1 * g.normal(size=n)).astype(np.float32))
        f.create_dataset("terminals", data=terminals)
        f.create_dataset("timeouts", data=timeouts)


def synthetic(row, data_dir):
    row["cache"] = dict(filename="synthetic.hdf5", sha256=sha(Path(data_dir) / "synthetic.hdf5"))
    return row


def load(directory):
    with np.load(Path(directory) / "frozen.npz") as npz:
        arrays = {name: npz[name] for name in npz.files}
    return arrays, json.loads((Path(directory) / "frozen.json").read_text(encoding="utf8"))


def short_train(row, output, data_dir, **kwargs):
    settings = dict(updates=UPDATES, train_fraction=0.5, block=10, score_batch=10**6, data_dir=data_dir)
    return FR.freeze_short_train(row, output, **dict(settings, **kwargs))


class FrozenReBRACTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="freeze-rebrac-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, ignore_errors=True)
        write_raw(cls.tmp / "synthetic.hdf5")
        cls.row = synthetic(FR.resolve_row("hopper", SEED, cls.tmp), cls.tmp)
        cls.result = short_train(cls.row, cls.tmp / "a", cls.tmp)
        cls.repeat = short_train(cls.row, cls.tmp / "b", cls.tmp)
        cls.arrays, cls.meta = load(cls.tmp / "a")
        cls.prepared = cls.result["prepared"]

    # (a) the scored values are what the host's refresh computes for the same bank and state
    def test_scores_equal_what_refresh_scores(self):
        r = self.result
        args, models, state, bank = r["args"], r["models"], r["state"], r["population"]
        key = jax.random.PRNGKey(SEED)
        target = P.native_target(args, models, state.native, bank, jax.random.fold_in(key, P.KEY_REFRESH_NOISE))
        q = models[1].apply(state.native.critic.params, bank.obs, bank.action).min(0)
        eta = models[2].apply(state.calibrator.params, bank.obs, bank.action)
        np.testing.assert_array_equal(self.arrays["residual"], np.asarray(target - q, np.float64))
        np.testing.assert_array_equal(self.arrays["sigma"], np.asarray(positive_scale(eta, state.residual_scale), np.float64))
        np.testing.assert_array_equal(self.arrays["policy_action"],
                                      np.asarray(models[0].apply(state.native.actor.params, bank.obs)))
        # The scores refresh itself hands to the WBCP selection, captured at calibration.wbcp.calibrate.
        seen = []
        original = calibration.wbcp.calibrate

        def capture(scores, rng, **kwargs):
            seen.append(np.array(scores, copy=True))
            return original(scores, rng, **kwargs)

        with mock.patch.object(calibration.wbcp, "calibrate", capture):
            refreshed, metrics, diagnostics = P.refresh(args, r["config"], models, state, bank, key,
                                                        heldout_ids=self.arrays["row"])
        self.assertEqual(metrics, dict(posterior_inputs_valid=True, posterior_certified=True))
        self.assertEqual(len(seen), 1)
        np.testing.assert_array_equal(seen[0], self.arrays["score"])
        self.assertEqual(diagnostics["scores"], len(self.arrays["row"]))
        self.assertEqual(float(refreshed.posterior.residual_scale), self.meta["residual_unit"])
        self.assertEqual(self.meta["target_noise"]["noise_fold"], P.KEY_REFRESH_NOISE)
        self.assertEqual(self.meta["target_noise"]["refresh_wbcp_fold"], P.KEY_POSTERIOR)

    def test_chunked_forward_passes_agree(self):
        r = self.result
        chunked = FR.score_rows(r["args"], r["models"], r["state"], r["population"], jax.random.PRNGKey(SEED),
                                self.prepared.max_action, 37)
        for name in ("residual", "sigma", "policy_action"):
            np.testing.assert_allclose(chunked[name].astype(np.float64), self.arrays[name], rtol=1e-6, atol=1e-6)

    # (b) the population is every withheld row, none of it trains, and the halves partition the data
    def test_halves_partition_the_data_by_whole_components(self):
        train, population = np.asarray(self.prepared.training_ids), self.arrays["row"]
        n = self.prepared.metadata["converted_rows"]
        component, timestep = F.components(self.prepared)
        self.assertFalse(np.intersect1d(train, population).size)
        np.testing.assert_array_equal(np.sort(np.r_[train, population]), np.arange(n))
        self.assertFalse(np.intersect1d(component[train], component[population]).size)
        np.testing.assert_array_equal(np.flatnonzero(np.isin(component, component[population])), population)
        self.assertLess(abs(len(population) - (n - round(0.5 * n))), np.bincount(component).max())
        m = self.prepared.metadata
        np.testing.assert_array_equal(m["withheld_converted_ids"], population)
        np.testing.assert_array_equal(m["heldout_converted_ids"], population)
        self.assertFalse(m["partition"]["effective_overlap"])
        self.assertFalse(np.any(self.arrays["in_training"]))
        np.testing.assert_array_equal(self.arrays["episode"], component[population])
        np.testing.assert_array_equal(self.arrays["timestep"], timestep[population])
        np.testing.assert_array_equal(self.arrays["action"], np.asarray(self.prepared.heldout.action))
        # obs: TD3+BC's rule (training-pool mean/std + 1e-3) for the tilts; the host itself reads raw obs.
        with h5py.File(self.tmp / "synthetic.hdf5", "r") as f:
            raw_obs = f["observations"][()]
        raw_current = np.asarray(m["dependency_maps"]["raw_current"])
        mean, std = R.C.compute_mean_std(raw_obs[raw_current[train]], eps=0.001)
        np.testing.assert_array_equal(self.arrays["obs"], (raw_obs[raw_current[population]] - mean) / std)
        np.testing.assert_array_equal(np.asarray(self.meta["obs_normalization"]["obs_mean"], np.float32), mean)
        np.testing.assert_array_equal(np.asarray(self.meta["obs_normalization"]["obs_std"], np.float32), std)
        np.testing.assert_array_equal(np.asarray(self.prepared.heldout.obs), raw_obs[raw_current[population]])
        reservation = self.meta["preparation"]["reservation"]
        self.assertIsNone(reservation["rows_per_episode"])
        self.assertEqual(reservation["dependency_contract"], R.DEPENDENCY_CONTRACT)
        self.assertIs(type(self.result["protocol"].reservation), R.PopulationSplit)
        rows = self.meta["rows"]
        self.assertEqual((rows["training"] + rows["population"], rows["withheld"], rows["population_in_training"]),
                         (rows["converted"], rows["population"], 0))

    def test_population_rows_are_td3_bcs_for_the_same_split_seed(self):
        td3 = synthetic(F.resolve_row("hopper", SEED, self.tmp), self.tmp)
        converted = F.converted_rows(td3, self.tmp)
        td3 = F.split_row(td3, converted, 0.5, SEED)
        _, prepared = F.prepare(td3, self.tmp)
        np.testing.assert_array_equal(np.asarray(prepared.heldout_ids), self.arrays["row"])
        np.testing.assert_array_equal(np.asarray(prepared.training_ids), np.asarray(self.prepared.training_ids))
        np.testing.assert_array_equal(self.arrays["obs"], np.asarray(prepared.heldout.obs))  # TD3+BC's obs field
        np.testing.assert_array_equal(F.components(prepared)[0], F.components(self.prepared)[0])
        keys = ("target_size", "seed", "max_fraction", "rows_per_episode")
        self.assertEqual({k: td3["protocol"]["reservation"][k] for k in keys},
                         {k: self.meta["preparation"]["reservation"][k] for k in keys})
        # another split seed withholds other components
        other = FR.split_row(self.row, converted, 0.5, SEED + 1)
        _, moved = FR.prepare(other, self.tmp)
        self.assertFalse(np.array_equal(np.asarray(moved.heldout_ids), self.arrays["row"]))

    # (c) the artifact loads with d4rl_benchmark.load_pool and passes its checks
    def test_artifact_loads_with_the_benchmark(self):
        F.check_contract(self.arrays)
        pool = bench.load_pool(str(self.tmp / "a"))
        np.testing.assert_array_equal(pool.scores["normalized"], self.arrays["score"])
        np.testing.assert_array_equal(pool.scores["raw"], np.abs(self.arrays["residual"]))
        np.testing.assert_array_equal(pool.row, self.arrays["row"])
        np.testing.assert_array_equal(pool.episode, self.arrays["episode"])
        self.assertEqual((pool.artifact_rows, pool.population), (len(self.arrays["row"]), "heldout"))
        m = self.meta
        self.assertEqual((m["schema"], m["algorithm"], m["host"], m["method"], m["source"]),
                         (bench.SCHEMA, "rebrac", "rebrac", "bca", "short-train"))
        self.assertEqual((m["dataset"], m["updates"], m["seed"], m["score_seed"]), ("hopper-medium-v2", UPDATES, SEED, SEED))
        self.assertEqual((m["dataset_sha256"], m["npz_sha256"]), (sha(self.tmp / "synthetic.hdf5"), sha(self.tmp / "a" / "frozen.npz")))
        self.assertEqual(m["checkpoint"]["sha256"], sha(self.tmp / "a" / m["checkpoint"]["path"]))
        self.assertEqual(m["checkpoint"]["counters"]["accepted_scale_fits"], UPDATES)
        self.assertEqual(m["arrays"]["obs"], dict(dtype="float32", shape=[m["rows"]["population"], OBS_DIM]))
        self.assertEqual((m["obs_normalization"]["rule"], m["obs_normalization"]["normalize_states"]), (F.OBS_RULE, False))
        log = json.loads((self.tmp / "a" / "training_log.json").read_text(encoding="utf8"))
        self.assertEqual([b["step"] for b in log["blocks"]], [10, 20])
        for b in log["blocks"]:  # native ReBRAC throughout: the frozen reference never becomes ready
            self.assertEqual((b["scale_fit_accepted"], b["bc_multiplier_mean"], b["posterior_ready"]), (1.0, 1.0, 0.0))
        self.assertFalse(bool(self.result["state"].posterior.ready))
        self.assertEqual(log["residual_unit_final"], m["residual_unit"])
        resolved = json.loads((self.tmp / "a" / "resolved.json").read_text(encoding="utf8"))
        self.assertIsNone(resolved["protocol"]["reservation"]["rows_per_episode"])

    # (d) reproducible from the seeds
    def test_reproducible_from_the_seeds(self):
        repeat, _ = load(self.tmp / "b")
        for name in F.ARRAYS:
            np.testing.assert_array_equal(self.arrays[name], repeat[name], name)
        self.assertEqual(self.meta["checkpoint"]["sha256"], self.repeat["metadata"]["checkpoint"]["sha256"])
        self.assertEqual(self.meta["npz_sha256"], self.repeat["metadata"]["npz_sha256"])
        # Another score seed redraws only the target noise: same scale, different residuals.
        r = self.result
        other = FR.score_rows(r["args"], r["models"], r["state"], r["population"], jax.random.PRNGKey(SEED + 1),
                              self.prepared.max_action, 10**6)
        np.testing.assert_array_equal(other["sigma"].astype(np.float64), self.arrays["sigma"])
        self.assertFalse(np.array_equal(other["residual"].astype(np.float64), self.arrays["residual"]))

    def test_population_split_is_never_a_calibration_bank(self):
        r = self.result
        with self.assertRaisesRegex(ValueError, "population split"):
            R.run_prepared(r["args"], r["spec"], r["protocol"], self.prepared)
        with self.assertRaises(ValueError):
            R.PopulationSplit(100, 911, 0.5, R.DEPENDENCY_CONTRACT, 5)
        for bad in (None, 0):  # the declared Reservation still requires a positive integer K
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                R.Reservation(100, 911, 0.5, R.DEPENDENCY_CONTRACT, bad)
        with self.assertRaises(FileExistsError):
            short_train(self.row, self.tmp / "a", self.tmp)


@unittest.skipUnless(HOPPER.is_file(), "needs the cached hopper-medium-v2 HDF5")
class RealHopperTests(unittest.TestCase):
    def test_short_train_on_hopper_medium(self):
        with tempfile.TemporaryDirectory(prefix="freeze-rebrac-hopper-") as tmp:
            row = FR.resolve_row("hopper", SEED, tmp)
            result = FR.freeze_short_train(row, Path(tmp) / "out", updates=200, block=100, data_dir=HOPPER.parent)
            arrays, meta = load(Path(tmp) / "out")
            pool = bench.load_pool(str(Path(tmp) / "out"))
        self.assertEqual((meta["dataset"], meta["dataset_sha256"]), ("hopper-medium-v2", row["cache"]["sha256"]))
        F.check_contract(arrays)
        rows = meta["rows"]
        self.assertEqual(rows["training"] + rows["population"], rows["converted"])
        self.assertLess(abs(rows["population"] / rows["converted"] - 0.5), 0.01)
        self.assertEqual((arrays["obs"].shape[1:], len(pool.row)), ((11,), rows["population"]))
        component, _ = F.components(result["prepared"])
        self.assertFalse(np.intersect1d(component[np.asarray(result["prepared"].training_ids)], arrays["episode"]).size)
        self.assertEqual(meta["checkpoint"]["counters"]["accepted_scale_fits"], 200)
        if TD3_HOPPER.is_file():  # TD3+BC's frozen hopper pool, same split seed: the same population rows
            with np.load(TD3_HOPPER) as td3:
                for name in ("row", "episode", "timestep", "action", "obs"):
                    np.testing.assert_array_equal(arrays[name], td3[name], name)


class CommandLineTests(unittest.TestCase):
    """Argument validation only; nothing is prepared or trained."""

    def test_rejects_invalid_invocations(self):
        with tempfile.TemporaryDirectory(prefix="freeze-rebrac-cli-") as tmp:
            fresh = str(Path(tmp) / "new")
            for argv in (["--output", tmp],  # --output must not exist
                         ["--dataset", "antmaze", "--output", fresh],
                         ["--train-fraction", "1.0", "--output", fresh],
                         ["--updates", "0", "--output", fresh],
                         ["--seed", "7", "--output", fresh],
                         ["--score-seed", str(2**32), "--output", fresh],
                         ["--score-batch", "0", "--output", fresh]):
                with self.subTest(argv=argv), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                    FR.main(argv)
            self.assertFalse(Path(fresh).exists())

    def test_every_configured_dataset_resolves_and_types_as_a_population_split(self):
        for dataset in FR.read_config(FR.CONFIG)["datasets"]:
            with self.subTest(dataset=dataset):
                row = FR.split_row(FR.resolve_row(dataset, SEED, "/tmp/x"), 10_000, 0.5, SEED)
                _, args, spec, protocol = FR.typed_population(row)
                self.assertEqual((protocol.reservation.target_size, protocol.reservation.rows_per_episode), (5000, None))
                self.assertIs(type(protocol.reservation), R.PopulationSplit)
                self.assertEqual(spec.config().arm, "bca")


if __name__ == "__main__":
    unittest.main()
