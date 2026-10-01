"""CPU checks for experiments/wbcp/freeze_scores.py, the frozen score artifact of the D4RL benchmark.

A synthetic HDF5 file drives the real preparation, training, restore and scoring code:
the contract, the component split, reproducibility, agreement with
algorithms.td3_bc_bca.refresh, and checkpoint mode on a short-train directory and on a
train.py run directory made with a tiny budget and a stub environment. One more test runs
200 updates on the cached hopper-medium-v2 file when it exists.
JAX_PLATFORMS=cpu python -m unittest experiments.wbcp.test_freeze_scores
"""
import json
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import jax
import numpy as np
from flax import serialization

import algorithms.td3_bc_bca as P
import runtime.td3_bc as R
import train as T
from calibration import wbcp
from calibration.reference import positive_scale
from experiments.wbcp import freeze_scores as F
from runtime.config import typed
from runtime.provenance import sha, write

SEED, OBS_DIM, ACTION_DIM, UPDATES = 202609171, 11, 3, 20
ROWS_PER_EPISODE = 15  # train.py run: 4 of the 17 components withheld, at most 15 calibrated rows each
HOPPER = Path.home() / ".d4rl" / "datasets" / "hopper_medium-v2.hdf5"


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


def synthetic_row(data_dir):
    row = F.resolve_row("hopper", SEED, data_dir)
    row["cache"] = dict(filename="synthetic.hdf5", sha256=sha(Path(data_dir) / "synthetic.hdf5"))
    return row


def load(directory):
    with np.load(Path(directory) / "frozen.npz") as npz:
        arrays = {name: npz[name] for name in npz.files}
    return arrays, json.loads((Path(directory) / "frozen.json").read_text(encoding="utf8"))


def short_train(row, output, data_dir, **kwargs):
    settings = dict(updates=UPDATES, train_fraction=0.5, block=10, score_batch=10**6, data_dir=data_dir)
    return F.freeze_short_train(row, output, **dict(settings, **kwargs))


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


class FrozenShortTrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="freeze-scores-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, ignore_errors=True)
        write_raw(cls.tmp / "synthetic.hdf5")
        cls.row = synthetic_row(cls.tmp)
        cls.result = short_train(cls.row, cls.tmp / "a", cls.tmp)
        cls.repeat = short_train(cls.row, cls.tmp / "b", cls.tmp)
        cls.arrays, cls.meta = load(cls.tmp / "a")
        cls.prepared = cls.result["prepared"]

    def test_contract_arrays(self):
        a, n = self.arrays, len(self.prepared.heldout_ids)
        self.assertEqual(set(a), set(F.ARRAYS))
        for name, dtype in F.ARRAYS.items():
            self.assertEqual(a[name].dtype, dtype, name)
            self.assertEqual(len(a[name]), n, name)
            if a[name].dtype.kind == "f":
                self.assertTrue(np.all(np.isfinite(a[name])), name)
        self.assertEqual((a["obs"].shape, a["action"].shape, a["policy_action"].shape),
                         ((n, OBS_DIM), (n, ACTION_DIM), (n, ACTION_DIM)))
        np.testing.assert_array_equal(a["score"], np.abs(a["residual"]) / a["sigma"])
        self.assertTrue(np.all(a["sigma"] > 0) and np.all(np.abs(a["policy_action"]) <= 1.0))
        np.testing.assert_array_equal(a["row"], np.asarray(self.prepared.heldout_ids))
        np.testing.assert_array_equal(a["obs"], np.asarray(self.prepared.heldout.obs))
        np.testing.assert_array_equal(a["action"], np.asarray(self.prepared.heldout.action))
        self.assertFalse(np.any(a["in_training"]))

    def test_metadata(self):
        m = self.meta
        self.assertEqual((m["schema"], m["algorithm"], m["method"], m["source"]),
                         ("wbcp-frozen-scores-v1", "td3_bc", "bca", "short-train"))
        self.assertEqual((m["dataset"], m["updates"], m["seed"], m["score_seed"]),
                         ("hopper-medium-v2", UPDATES, SEED, SEED))
        self.assertEqual(m["dataset_sha256"], sha(self.tmp / "synthetic.hdf5"))
        self.assertEqual(m["target_noise"]["noise_fold"], 1213156420)
        self.assertEqual(m["obs_normalization"]["rule"], "current training-observation mean/std + 1e-3")
        rows = m["rows"]
        self.assertEqual(rows["training"] + rows["population"], rows["converted"])
        self.assertEqual((rows["population"], rows["population_in_training"]), (len(self.arrays["row"]), 0))
        self.assertEqual(m["checkpoint"]["sha256"], sha(self.tmp / "a" / m["checkpoint"]["path"]))
        self.assertEqual(m["checkpoint"]["counters"]["accepted_scale_fits"], UPDATES)
        self.assertEqual(m["npz_sha256"], sha(self.tmp / "a" / "frozen.npz"))
        self.assertEqual(m["residual_unit"], float(self.result["state"].residual_scale))
        self.assertEqual(set(m["git"]) - {"error"}, {"branch", "commit", "dirty"})
        self.assertEqual(m["arrays"]["obs"], dict(dtype="float32", shape=[rows["population"], OBS_DIM]))
        log = json.loads((self.tmp / "a" / "training_log.json").read_text(encoding="utf8"))
        self.assertEqual([b["step"] for b in log["blocks"]], [10, 20])
        self.assertTrue(all(b["scale_fit_accepted"] == 1.0 and b["bc_multiplier_mean"] == 1.0 for b in log["blocks"]))
        self.assertEqual(log["residual_unit_final"], m["residual_unit"])

    def test_halves_are_disjoint_whole_components(self):
        train, population = np.asarray(self.prepared.training_ids), self.arrays["row"]
        component, timestep = F.components(self.prepared)
        self.assertFalse(np.intersect1d(component[train], component[population]).size)
        self.assertFalse(np.intersect1d(train, population).size)
        self.assertEqual(len(train) + len(population), self.prepared.metadata["converted_rows"])
        self.assertGreater(len(population), 0.4 * self.prepared.metadata["converted_rows"])
        # rows_per_episode=None: the population is every withheld row, whole components only.
        self.assertIsNone(self.meta["preparation"]["reservation"]["rows_per_episode"])
        np.testing.assert_array_equal(self.prepared.metadata["withheld_converted_ids"], population)
        self.assertEqual(self.meta["rows"]["withheld"], len(population))
        np.testing.assert_array_equal(np.flatnonzero(np.isin(component, component[population])), population)
        np.testing.assert_array_equal(self.arrays["episode"], component[population])
        np.testing.assert_array_equal(self.arrays["timestep"], timestep[population])
        starts = np.r_[True, np.diff(component) != 0]
        self.assertTrue(np.all(timestep[starts] == 0) and np.all(timestep[~starts] == timestep[np.flatnonzero(~starts) - 1] + 1))
        # Normalization is fit on the training rows only, exactly as the runtime fits it.
        with h5py.File(self.tmp / "synthetic.hdf5", "r") as f:
            raw_obs = f["observations"][()]
        raw_rows = np.asarray(self.prepared.metadata["dependency_maps"]["raw_current"])[train]
        mean, std = R.C.compute_mean_std(raw_obs[raw_rows], eps=0.001)
        np.testing.assert_array_equal(np.asarray(self.meta["obs_normalization"]["obs_mean"], np.float32), mean)
        np.testing.assert_array_equal(np.asarray(self.meta["obs_normalization"]["obs_std"], np.float32), std)

    def test_reproducible_from_the_seed(self):
        repeat, _ = load(self.tmp / "b")
        for name in F.ARRAYS:
            np.testing.assert_array_equal(self.arrays[name], repeat[name], name)
        self.assertEqual(self.meta["checkpoint"]["sha256"], self.repeat["metadata"]["checkpoint"]["sha256"])
        self.assertEqual(self.meta["npz_sha256"], self.repeat["metadata"]["npz_sha256"])
        # Another score seed redraws only the target noise: same scale, different residuals.
        r = self.result
        other = F.score_rows(r["args"], r["models"], r["state"], r["population"], jax.random.PRNGKey(SEED + 1),
                             self.prepared.max_action, 10**6)
        np.testing.assert_array_equal(other["sigma"].astype(np.float64), self.arrays["sigma"])
        self.assertFalse(np.array_equal(other["residual"].astype(np.float64), self.arrays["residual"]))

    def test_scores_equal_what_refresh_scores(self):
        r = self.result
        args, models, state, bank = r["args"], r["models"], r["state"], r["population"]
        key = jax.random.PRNGKey(SEED)
        target = P.native_target(args, models, state.native, bank, jax.random.fold_in(key, 1213156420),
                                 self.prepared.max_action)
        q = models[1].apply(state.native.critic.params, bank.obs, bank.action).min(axis=-1)
        eta = models[2].apply(state.calibrator.params, bank.obs, bank.action)
        np.testing.assert_array_equal(self.arrays["residual"], np.asarray(target - q, np.float64))
        np.testing.assert_array_equal(self.arrays["sigma"], np.asarray(positive_scale(eta, state.residual_scale), np.float64))
        refreshed, metrics, diagnostics = P.refresh(args, r["config"], models, state, bank, key,
                                                    self.prepared.max_action, heldout_ids=self.arrays["row"])
        self.assertTrue(metrics["posterior_inputs_valid"])
        posterior = r["config"].posterior
        ours = wbcp.calibrate(self.arrays["score"], np.random.default_rng(np.asarray(
            jax.random.fold_in(key, 1347375956), np.uint32).ravel()), alpha=posterior.alpha,
            beta=posterior.credibility, draws=posterior.draws)
        self.assertEqual((ours.threshold, ours.lambda_hat, ours.lambda_hpd),
                         (diagnostics["threshold"], diagnostics["lambda_hat"], diagnostics["lambda_hpd"]))
        self.assertEqual(float(refreshed.posterior.residual_scale), self.meta["residual_unit"])

    def test_chunked_forward_passes_agree(self):
        r = self.result
        chunked = F.score_rows(r["args"], r["models"], r["state"], r["population"], jax.random.PRNGKey(SEED),
                               self.prepared.max_action, 37)
        for name in ("residual", "sigma"):
            np.testing.assert_allclose(chunked[name].astype(np.float64), self.arrays[name], rtol=1e-6, atol=1e-6)

    def test_checkpoint_mode_rescores_the_short_train_state(self):
        result = F.freeze_checkpoint(self.tmp / "a", self.tmp / "c", population="heldout", score_batch=10**6,
                                     data_dir=self.tmp)
        arrays, meta = load(self.tmp / "c")
        for name in F.ARRAYS:
            np.testing.assert_array_equal(arrays[name], self.arrays[name], name)
        self.assertEqual((meta["source"], meta["updates"]), ("checkpoint", UPDATES))
        self.assertEqual(meta["checkpoint"]["sha256"], self.meta["checkpoint"]["sha256"])
        self.assertEqual(meta["verification"]["record"], "frozen.json")
        self.assertFalse(bool(result["state"].posterior.ready))
        full = F.freeze_checkpoint(self.tmp / "a", self.tmp / "d", score_batch=10**6, data_dir=self.tmp)["arrays"]
        n = self.prepared.metadata["converted_rows"]
        np.testing.assert_array_equal(full["row"], np.arange(n))
        np.testing.assert_array_equal(full["in_training"], np.isin(np.arange(n), np.asarray(self.prepared.training_ids)))
        held = self.arrays["row"]
        np.testing.assert_array_equal(full["obs"][held], self.arrays["obs"])
        np.testing.assert_allclose(full["sigma"][held], self.arrays["sigma"], rtol=1e-6)
        with self.assertRaises(FileExistsError):
            F.freeze_checkpoint(self.tmp / "a", self.tmp / "c", data_dir=self.tmp)


class TrainRunCheckpointTests(unittest.TestCase):
    """Checkpoint mode on a real train.py run directory: tiny budget, two refreshes, stub environment."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="freeze-scores-run-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, ignore_errors=True)
        write_raw(cls.tmp / "synthetic.hdf5")
        row = synthetic_row(cls.tmp)
        row["native_args"].update(num_updates=20, eval_interval=10, eval_workers=1, eval_final_episodes=0,
                                  batch_size=32)
        row["protocol"].update(
            num_updates=20, scan_block_size=10, eval_interval=10, eval_workers=1, eval_periodic_episodes=1,
            eval_final_episodes=0, refresh_events=[dict(step=5, seed=77), dict(step=15, seed=78)],
            evaluation_events=[dict(kind="periodic", step=10, episode_seeds=[1]),
                               dict(kind="periodic", step=20, episode_seeds=[2])],
            reservation=dict(target_size=60, seed=911, max_fraction=0.3, dependency_contract=R.DEPENDENCY_CONTRACT,
                             rows_per_episode=ROWS_PER_EPISODE))
        row["expected_counts"]["checkpoint_steps"] = [20]
        cls.run_dir = cls.tmp / "run"
        cls.run_dir.mkdir()
        write(cls.run_dir / "resolved.json", row)
        os.environ["D4RL_DATASET_DIR"] = str(cls.tmp)
        objects = typed(row)
        prepared = T.prepare(row, objects)  # what train.main does before training
        write(cls.run_dir / "preparation.json", {"accepted": True, "metadata": prepared.metadata, "learner_updates": 0})
        with mock.patch.object(R.C, "make_eval_env", lambda dataset, workers: FakeEnv(workers)):
            cls.record = T.execute(row, objects, cls.run_dir, prepared)
        cls.prepared = prepared
        cls.heldout = F.freeze_checkpoint(cls.run_dir, cls.tmp / "heldout", population="heldout", score_batch=10**6,
                                          data_dir=cls.tmp)
        cls.all = F.freeze_checkpoint(cls.run_dir, cls.tmp / "all", data_dir=cls.tmp, score_batch=64)

    def test_restores_the_refreshed_run_state_exactly(self):
        payload = (self.run_dir / "checkpoint_20.msgpack").read_bytes()
        restored = self.heldout["restored"]
        self.assertTrue(F.same_leaves(payload, restored))
        # Byte for byte once the parameter dicts take the sorted key order the scanned run state has.
        canonical = dict(restored, state=jax.tree_util.tree_map(lambda x: x, restored["state"]))
        self.assertEqual(serialization.to_bytes(canonical), payload)
        self.assertFalse(F.same_leaves(payload, dict(restored, step=np.int32(19))))
        state, meta = self.heldout["state"], self.heldout["metadata"]
        self.assertTrue(bool(state.posterior.ready) and math.isfinite(float(state.posterior.threshold)))
        self.assertEqual(meta["checkpoint"]["recorded_sha256"], self.record["checkpoints"][0]["sha256"])
        self.assertEqual(meta["checkpoint"]["counters"]["accepted_scale_fits"], 20)
        self.assertEqual((meta["verification"]["record"], meta["verification"]["source_files_changed_since_run"]),
                         ("preparation.json", []))

    def test_heldout_scores_match_an_independent_recomputation(self):
        r, key = self.heldout, jax.random.PRNGKey(SEED)
        bank, state, models = self.prepared.heldout, r["state"], r["models"]
        target = P.native_target(r["args"], models, state.native, bank, jax.random.fold_in(key, 1213156420), 1.0)
        q = models[1].apply(state.native.critic.params, bank.obs, bank.action).min(axis=-1)
        eta = models[2].apply(state.calibrator.params, bank.obs, bank.action)
        a = r["arrays"]
        np.testing.assert_array_equal(a["residual"], np.asarray(target - q, np.float64))
        np.testing.assert_array_equal(a["sigma"], np.asarray(positive_scale(eta, state.residual_scale), np.float64))
        np.testing.assert_array_equal(a["row"], np.asarray(self.prepared.heldout_ids))
        self.assertFalse(np.any(a["in_training"]))
        # The bank is K stratified rows of each withheld component; every row of those components
        # is withheld from training, and training and withheld rows partition the data.
        component, _ = F.components(self.prepared)
        m, n = self.prepared.metadata, self.prepared.metadata["converted_rows"]
        withheld, train = np.asarray(m["withheld_converted_ids"]), np.asarray(self.prepared.training_ids)
        np.testing.assert_array_equal(np.flatnonzero(np.isin(component, np.unique(a["episode"]))), withheld)
        self.assertTrue(np.all(np.isin(a["row"], withheld)) and len(a["row"]) < len(withheld))
        self.assertFalse(np.intersect1d(train, withheld).size)
        np.testing.assert_array_equal(np.sort(np.r_[train, withheld]), np.arange(n))
        self.assertLessEqual(np.bincount(a["episode"]).max(), ROWS_PER_EPISODE)
        rows = r["metadata"]["rows"]
        self.assertEqual((rows["population"], rows["heldout"], rows["withheld"]),
                         (len(a["row"]), len(a["row"]), len(withheld)))
        self.assertEqual(rows["training"] + rows["withheld"], rows["converted"])
        self.assertEqual(r["metadata"]["preparation"]["reservation"]["rows_per_episode"], ROWS_PER_EPISODE)

    def test_all_population_marks_the_training_pool(self):
        a, n = self.all["arrays"], self.prepared.metadata["converted_rows"]
        training = np.asarray(json.loads((self.run_dir / "preparation.json").read_text())["metadata"]["training_converted_ids"])
        np.testing.assert_array_equal(a["row"], np.arange(n))
        np.testing.assert_array_equal(np.flatnonzero(a["in_training"]), training)
        np.testing.assert_array_equal(a["obs"][training], np.asarray(self.prepared.training.obs))
        self.assertEqual(self.all["metadata"]["rows"]["population_in_training"], len(training))
        F.check_contract(a)
        # Withheld rows outside the calibration bank are not in the prepared pools; they are rebuilt
        # from the cached file, scored and marked as never trained on.
        bank = np.asarray(self.prepared.heldout_ids)
        unbanked = np.setdiff1d(self.prepared.metadata["withheld_converted_ids"], bank)
        self.assertGreater(len(unbanked), 0)
        self.assertFalse(np.any(a["in_training"][unbanked]))
        np.testing.assert_array_equal(a["obs"][bank], np.asarray(self.prepared.heldout.obs))
        m = self.prepared.metadata
        with h5py.File(self.tmp / "synthetic.hdf5", "r") as f:
            raw_obs = f["observations"][()]
        raw_rows = np.asarray(m["dependency_maps"]["raw_current"])[unbanked]
        mean, std = (np.asarray(m[name], np.float32) for name in ("obs_mean", "obs_std"))
        np.testing.assert_array_equal(a["obs"][unbanked], (raw_obs[raw_rows] - mean) / std)
        np.testing.assert_allclose(a["sigma"][bank], self.heldout["arrays"]["sigma"], rtol=1e-6)


@unittest.skipUnless(HOPPER.is_file(), "needs the cached hopper-medium-v2 HDF5")
class RealHopperTests(unittest.TestCase):
    def test_short_train_on_hopper_medium(self):
        with tempfile.TemporaryDirectory(prefix="freeze-scores-hopper-") as tmp:
            row = F.resolve_row("hopper", SEED, tmp)
            result = F.freeze_short_train(row, Path(tmp) / "out", updates=200, block=100, data_dir=HOPPER.parent)
            arrays, meta = load(Path(tmp) / "out")
        self.assertEqual((meta["dataset"], meta["dataset_sha256"]), ("hopper-medium-v2", row["cache"]["sha256"]))
        F.check_contract(arrays)
        rows = meta["rows"]
        self.assertEqual(rows["training"] + rows["population"], rows["converted"])
        self.assertLess(abs(rows["population"] / rows["converted"] - 0.5), 0.01)
        self.assertEqual(arrays["obs"].shape[1:], (11,))
        self.assertFalse(np.any(arrays["in_training"]))
        component, _ = F.components(result["prepared"])
        self.assertFalse(np.intersect1d(component[np.asarray(result["prepared"].training_ids)], arrays["episode"]).size)
        self.assertEqual(meta["checkpoint"]["counters"]["accepted_scale_fits"], 200)


class CommandLineTests(unittest.TestCase):
    """Argument validation only; nothing is prepared or trained."""

    def test_rejects_invalid_invocations(self):
        with tempfile.TemporaryDirectory(prefix="freeze-scores-cli-") as tmp:
            fresh = str(Path(tmp) / "new")
            for argv in (["--output", tmp],  # --output must not exist
                         ["--checkpoint", tmp, "--updates", "10", "--output", fresh],
                         ["--mode", "checkpoint", "--output", fresh],
                         ["--population", "all", "--output", fresh],
                         ["--train-fraction", "1.0", "--output", fresh],
                         ["--seed", "7", "--output", fresh],
                         ["--score-seed", str(2**32), "--output", fresh]):
                with self.subTest(argv=argv), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                    F.main(argv)
            self.assertFalse(Path(fresh).exists())


if __name__ == "__main__":
    unittest.main()
