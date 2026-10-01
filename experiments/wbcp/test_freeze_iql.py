"""CPU checks for experiments/wbcp/freeze_iql.py, the IQL frozen score artifact of the D4RL benchmark.

Synthetic HDF5 files with D4RL's layout drive the real preparation (the D4RL environment's
get_dataset, d4rl.qlearning_dataset, calibration.iql_reference.prepare_dataset), training and
scoring code: agreement with calibration.iql_reference.refresh, the population split and its
agreement with TD3+BC's (freeze_scores.py), the artifact through d4rl_benchmark.load_pool and
dependence.py, reproducibility, and the Gaussian actor's mean action (pen-human shapes).
No GPU; the MuJoCo environments are only built to read the files.
JAX_PLATFORMS=cpu python -m unittest experiments.wbcp.test_freeze_iql
"""
import json
import shutil
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import h5py
import jax
import jax.numpy as jnp
import numpy as np

import calibration.iql_reference as P
from calibration import wbcp
from experiments.wbcp import d4rl_benchmark as bench
from experiments.wbcp import dependence
from experiments.wbcp import freeze_iql as I
from experiments.wbcp import freeze_scores as F
from runtime.provenance import sha

SEED, UPDATES = 202609171, 20


def write_raw(path, obs_dim, action_dim, episodes=16, length=25, seed=3):
    """D4RL-shaped raw transitions: fixed-length episodes ending in timeouts, one early terminal."""
    g = np.random.default_rng(seed)
    n = episodes * length
    obs = g.normal(size=(n, obs_dim)).astype(np.float32)
    terminals, timeouts = np.zeros(n, bool), np.zeros(n, bool)
    timeouts[length - 1::length] = True
    terminals[3 * length + 10] = True
    with h5py.File(path, "w") as f:
        f.create_dataset("observations", data=obs)
        f.create_dataset("actions", data=g.uniform(-0.9, 0.9, (n, action_dim)).astype(np.float32))
        f.create_dataset("rewards", data=(np.tanh(obs[:, 0]) + 0.1 * g.normal(size=n)).astype(np.float32))
        f.create_dataset("terminals", data=terminals)
        f.create_dataset("timeouts", data=timeouts)


def synthetic_row(resolve_row, dataset, data_dir, name="synthetic.hdf5"):
    row = resolve_row(dataset, SEED, data_dir)
    row["cache"] = dict(filename=name, sha256=sha(Path(data_dir) / name))
    return row


def load(directory):
    with np.load(Path(directory) / "frozen.npz") as npz:
        arrays = {name: npz[name] for name in npz.files}
    return arrays, json.loads((Path(directory) / "frozen.json").read_text(encoding="utf8"))


def short_train(row, output, data_dir, **kwargs):
    settings = dict(updates=UPDATES, train_fraction=0.5, block=10, score_batch=10**6, data_dir=data_dir)
    return I.freeze_short_train(row, output, **dict(settings, **kwargs))


class FrozenIQLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="freeze-iql-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, ignore_errors=True)
        write_raw(cls.tmp / "synthetic.hdf5", 11, 3)  # hopper shapes; hopper's IQL actor is deterministic
        cls.row = synthetic_row(I.resolve_row, "hopper", cls.tmp)
        cls.result = short_train(cls.row, cls.tmp / "a", cls.tmp)
        cls.repeat = short_train(cls.row, cls.tmp / "b", cls.tmp)
        cls.arrays, cls.meta = load(cls.tmp / "a")
        cls.p, cls.t = cls.result["prepared"], cls.result["trained"]

    # (a) the scores are what the host's refresh computes for the same bank and state
    def test_scores_equal_what_refresh_scores(self):
        p, t, a = self.p, self.t, self.arrays
        extra = t.carry.extras[0]
        bank = p.data.calibration
        reference, record = P.refresh(p.args.posterior, t.fitters[0], extra.calibration, t.carry.nuisance, bank,
                                      t.carry.rng, UPDATES, discount=p.args.discount)
        digests = record["component_sha256"]
        # refresh fingerprints its float32 residuals and scale predictions: equal digests are equal bytes
        self.assertEqual(digests["cal_residuals"], P.fingerprint(a["residual"].astype(np.float32)))
        self.assertEqual(digests["cal_predictions"], P.fingerprint(self.result["scored"]["eta"]))
        np.testing.assert_array_equal(a["residual"].astype(np.float32).astype(np.float64), a["residual"])
        np.testing.assert_array_equal(a["sigma"], np.asarray(
            jnp.maximum(self.result["scored"]["eta"], 1e-6) * reference.residual_scale, np.float64))
        self.assertEqual(float(reference.residual_scale), self.meta["residual_unit"])
        self.assertEqual(record["calibration_rows"], len(a["row"]))
        # the WBCP threshold refresh selects is the one these scores give with refresh's own key
        key = jax.random.fold_in(t.carry.rng, I.POSTERIOR_FOLD)
        ours = wbcp.calibrate(a["score"], np.random.default_rng(np.asarray(key, np.uint32).ravel()),
                              alpha=0.1, beta=0.95, draws=1000)
        self.assertEqual((ours.threshold, ours.lambda_hat, ours.lambda_hpd),
                         (record["wbcp"]["threshold"], record["wbcp"]["lambda_hat"], record["wbcp"]["lambda_hpd"]))
        # independent recomputation: r + (1 - d) gamma V(s') - min Q(s, a), eta from the live scale network
        agent = t.carry.nuisance
        target = bank.reward + (1 - bank.done) * p.args.discount * agent.vf.apply_fn(agent.vf.params, bank.next_obs)
        q = jnp.min(agent.qf.apply_fn(agent.qf.params, bank.obs, bank.action), -1)
        np.testing.assert_array_equal(a["residual"], np.asarray(target - q, np.float64))
        eta = extra.calibration.calibrator.apply_fn(extra.calibration.calibrator.params, bank.obs, bank.action)
        np.testing.assert_array_equal(self.result["scored"]["eta"], np.asarray(eta))

    def test_policy_action_is_the_host_actors_evaluation_action(self):
        p, t = self.p, self.t
        self.assertTrue(p.args.iql_deterministic)
        host = p.m.S.actor_at(t.carry.actors, 0)
        expected = np.asarray(t.actor.act(host.params, p.data.calibration.obs))
        np.testing.assert_array_equal(self.arrays["policy_action"], expected)
        self.assertTrue(np.all(np.abs(self.arrays["policy_action"]) <= 1.0))
        # no refresh: the BCA actor never left the host actor
        self.assertTrue(self.meta["actors"]["bca_actor_identical_to_host"])
        self.assertEqual(self.meta["actors"]["applied_updates"], [UPDATES, UPDATES])

    def test_chunked_forward_passes_agree(self):
        chunked = I.score_population(self.p, self.t, 37)
        for name in ("residual", "sigma"):
            np.testing.assert_allclose(chunked[name].astype(np.float64), self.arrays[name], rtol=1e-6, atol=1e-6)

    # (b) population and training rows are disjoint whole episodes that partition the data
    def test_halves_are_disjoint_whole_episodes(self):
        data, a = self.p.data, self.arrays
        train, population, n = np.asarray(data.train_indices), a["row"], data.metadata["dataset_rows"]
        np.testing.assert_array_equal(np.sort(np.r_[train, population]), np.arange(n))
        self.assertFalse(np.intersect1d(train, population).size)
        np.testing.assert_array_equal(population, data.withheld_indices)
        np.testing.assert_array_equal(population, data.calibration_indices)
        component, timestep = F.components(SimpleNamespace(metadata={"dependency_maps": self.result["maps"]}))
        self.assertFalse(np.intersect1d(component[train], component[population]).size)
        np.testing.assert_array_equal(np.flatnonzero(np.isin(component, component[population])), population)
        np.testing.assert_array_equal(a["episode"], component[population])
        np.testing.assert_array_equal(a["timestep"], timestep[population])
        self.assertFalse(np.any(a["in_training"]))
        self.assertGreater(len(population), 0.4 * n)
        target = n - int(round(0.5 * n))
        self.assertEqual(self.meta["split"]["target_size"], target)
        self.assertGreaterEqual(len(population), target)
        rows = self.meta["rows"]
        self.assertEqual((rows["training"] + rows["population"], rows["withheld"]), (rows["converted"], len(population)))
        # the population split never passes as a WBCP bank: whole episodes, no declared K
        record = self.meta["preparation"]["metadata"]
        self.assertIsNone(record["rows_per_episode"])
        with self.assertRaises(ValueError):
            self.p.driver.D.verify_data_partition(record, asdict(self.p.args.posterior))
        # normalization and the return range come from the training rows only
        with h5py.File(self.tmp / "synthetic.hdf5", "r") as f:
            raw_obs = f["observations"][()]
        raw_rows = np.asarray(self.result["maps"]["raw_current"])
        mean = raw_obs[raw_rows[train]].mean(0)
        np.testing.assert_allclose(np.asarray(self.meta["obs_normalization"]["obs_mean"]), mean, rtol=1e-5, atol=1e-6)

    def test_population_is_td3_bcs_for_the_same_split_seed(self):
        converted = F.converted_rows(self.row, self.tmp)
        row = F.split_row(synthetic_row(F.resolve_row, "hopper", self.tmp), converted, 0.5, SEED)
        _, prepared = F.prepare(row, self.tmp)
        np.testing.assert_array_equal(np.asarray(prepared.heldout_ids), self.arrays["row"])
        np.testing.assert_array_equal(np.asarray(prepared.training_ids), self.p.data.train_indices)
        component, timestep = F.components(prepared)
        np.testing.assert_array_equal(component[self.arrays["row"]], self.arrays["episode"])
        np.testing.assert_array_equal(timestep[self.arrays["row"]], self.arrays["timestep"])

    def test_population_split_needs_a_reservation_and_leaves_integer_k_alone(self):
        p = self.p
        off = replace(p.args, posterior=replace(p.args.posterior, mode="off", reserve_size=0))
        with self.assertRaisesRegex(ValueError, "population split needs a positive reserve_size"):
            P.prepare_dataset(off, p.converted, p.raw, population_split=True)
        k = replace(p.args, posterior=replace(p.args.posterior, reserve_size=60, reserve_rows_per_episode=15,
                                              reserve_max_fraction=0.3))
        thinned = P.prepare_dataset(k, p.converted, p.raw)
        self.assertEqual(thinned.metadata["rows_per_episode"], 15)
        self.assertLess(len(thinned.calibration_indices), len(thinned.withheld_indices))
        self.assertEqual(p.driver.D.verify_data_partition(json.loads(json.dumps(p.driver.D.plain(thinned.metadata))),
                                                          p.driver.D.plain(asdict(k.posterior)))[2],
                         len(thinned.calibration_indices))

    # (c) the artifact reads through d4rl_benchmark.load_pool and dependence.py unchanged
    def test_artifact_loads_in_the_benchmark_and_dependence(self):
        F.check_contract(self.arrays)
        self.assertEqual(set(self.arrays), set(F.ARRAYS))
        pool = bench.load_pool(self.tmp / "a")
        self.assertEqual((pool.size, pool.artifact_rows, pool.population), (len(self.arrays["row"]),) * 2 + ("heldout",))
        np.testing.assert_array_equal(pool.scores["normalized"], self.arrays["score"])
        np.testing.assert_array_equal(pool.scores["raw"], np.abs(self.arrays["residual"]))
        np.testing.assert_array_equal(pool.row, self.arrays["row"])
        self.assertEqual(bench.load_pool(self.tmp / "a", "all").size, pool.size)
        m = self.meta
        self.assertEqual((m["schema"], m["algorithm"], m["host"], m["method"], m["source"]),
                         ("wbcp-frozen-scores-v1", "iql", "iql", "bca", "short-train"))
        self.assertEqual((m["dataset"], m["updates"], m["seed"], m["score_seed"]), ("hopper-medium-v2", UPDATES, SEED, SEED))
        self.assertEqual(m["npz_sha256"], sha(self.tmp / "a" / "frozen.npz"))
        self.assertEqual(m["checkpoint"]["sha256"], sha(self.tmp / "a" / m["checkpoint"]["path"]))
        counters = m["checkpoint"]["counters"]
        self.assertTrue(all(v == UPDATES for v in counters["calibrators"][0].values()))
        self.assertTrue(all(v == [UPDATES, UPDATES] for v in counters["actors"].values()))
        self.assertFalse(m["checkpoint"]["counters"]["references"][0]["ready"])
        log = json.loads((self.tmp / "a" / "training_log.json").read_text(encoding="utf8"))
        self.assertEqual([b["step"] for b in log["blocks"]], [10, 20])
        self.assertTrue(all(b["scale_fit_accepted"] == 1.0 and b["posterior_ready"] == 0.0 for b in log["blocks"]))
        out = self.tmp / "dep"
        with mock.patch("sys.stdout"):
            dependence.main(["--frozen", str(self.tmp / "a"), "--output", str(out), "--n", "60", "--block-sizes", "60",
                             "--per-episode", "1", "5", "--banks", "50"])
        predictions = json.loads((out / "predictions.json").read_text(encoding="utf8"))
        self.assertEqual(predictions["pool"]["rows"], len(self.arrays["row"]))
        self.assertEqual(predictions["dataset"]["episodes"], m["rows"]["converted_components"])

    # (d) reproducible from the seeds
    def test_reproducible_from_the_seeds(self):
        repeat, meta = load(self.tmp / "b")
        for name in F.ARRAYS:
            np.testing.assert_array_equal(self.arrays[name], repeat[name], name)
        self.assertEqual(self.meta["npz_sha256"], meta["npz_sha256"])
        self.assertEqual(self.meta["checkpoint"]["sha256"], meta["checkpoint"]["sha256"])
        # the score seed enters no score: IQL's refresh target is deterministic
        other = short_train(self.row, self.tmp / "c", self.tmp, score_seed=SEED + 1)["arrays"]
        for name in F.ARRAYS:
            np.testing.assert_array_equal(self.arrays[name], other[name], name)
        # another split seed moves the population
        moved = short_train(self.row, self.tmp / "d", self.tmp, split_seed=SEED + 1, updates=10)["arrays"]
        self.assertFalse(np.array_equal(moved["row"], self.arrays["row"]))


class GaussianActorTests(unittest.TestCase):
    """pen-human shapes (45 / 24): a stochastic actor with dropout, few short episodes."""

    def test_policy_action_is_the_gaussian_mean(self):
        with tempfile.TemporaryDirectory(prefix="freeze-iql-pen-") as tmp:
            tmp = Path(tmp)
            write_raw(tmp / "pen.hdf5", 45, 24, episodes=8, length=30, seed=5)
            row = synthetic_row(I.resolve_row, "pen-human", tmp, "pen.hdf5")
            result = short_train(row, tmp / "out", tmp, updates=10)
            arrays, meta = load(tmp / "out")
            pool = bench.load_pool(tmp / "out")
        p, t = result["prepared"], result["trained"]
        self.assertFalse(p.args.iql_deterministic)
        self.assertEqual(p.args.actor_dropout, 0.1)
        host = p.m.S.actor_at(t.carry.actors, 0)
        mean, _ = t.actor.apply(host.params, p.data.calibration.obs, deterministic=True)
        np.testing.assert_array_equal(arrays["policy_action"], np.asarray(jnp.clip(p.spaces["max_action"] * mean, -1, 1)))
        self.assertIn("GaussianPolicy mean", meta["score_definition"]["policy_action"])
        self.assertEqual(pool.size, len(arrays["row"]))
        self.assertEqual(len(np.unique(arrays["episode"])), meta["rows"]["population_episodes"])
        self.assertGreaterEqual(meta["rows"]["population_episodes"], 4)


class CommandLineTests(unittest.TestCase):
    """Argument validation only; nothing is prepared or trained."""

    def test_rejects_invalid_invocations(self):
        with tempfile.TemporaryDirectory(prefix="freeze-iql-cli-") as tmp:
            fresh = str(Path(tmp) / "new")
            for argv in (["--output", tmp],  # --output must not exist
                         ["--dataset", "antmaze", "--output", fresh],
                         ["--train-fraction", "1.0", "--output", fresh],
                         ["--updates", "0", "--output", fresh],
                         ["--seed", "7", "--output", fresh],
                         ["--score-seed", str(2**32), "--output", fresh]):
                with self.subTest(argv=argv), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                    I.main(argv)
            self.assertFalse(Path(fresh).exists())


if __name__ == "__main__":
    unittest.main()
