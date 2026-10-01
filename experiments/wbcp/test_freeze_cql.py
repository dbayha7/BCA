"""CPU checks for experiments/wbcp/freeze_cql.py, CQL's frozen score artifact for the D4RL benchmark.

A synthetic HDF5 file (episodes ending in terminals and in timeouts) drives the real CQL
preparation (train.prepare -> runtime.cql.prepare), training (algorithms.cql_bca.make_train_step)
and scoring code. (a) The scores are the values algorithms.cql_bca.refresh computes for the same
bank and state, and policy_action is runtime.cql.evaluate's action. (b) Population and training
rows partition the data by whole episodes, and the population is exactly the one
freeze_scores.py's TD3+BC split selects on the same file. (c) The artifact passes
d4rl_benchmark.load_pool. (d) It is reproducible from the seeds. The CQL runtime's
population-split guards are checked too. One more test prepares (no training) the cached
hopper-medium-v2 file when it exists and compares the population with TD3+BC's frozen pool.
JAX_PLATFORMS=cpu python -m unittest experiments.wbcp.test_freeze_cql
"""
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import h5py
import jax
import numpy as np

import algorithms.cql_bca as P
import runtime.cql as R
from calibration import wbcp
from calibration.reference import positive_scale, reserve_calibration
from experiments.wbcp import d4rl_benchmark as bench
from experiments.wbcp import freeze_cql as Q
from experiments.wbcp import freeze_scores as F
from runtime.config import typed
from runtime.provenance import sha

SEED, OBS_DIM, ACTION_DIM, UPDATES, EPISODES = 202609171, 11, 3, 20, 20
HOPPER = Path.home() / ".d4rl" / "datasets" / "hopper_medium-v2.hdf5"
TD3_HOPPER = Path(F.ROOT) / "runs" / "wbcp_frozen" / "hopper-medium-v2-s202609171-u100000" / "frozen.npz"


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):  # the runtime's legacy reward-transform print
        return function(*args, **kwargs)


def write_raw(path, seed=5):
    """D4RL-shaped raw transitions: 20 episodes of 12-29 rows; every fourth ends in a timeout, the rest in a terminal.

    Terminals end CQL's native reward-normalization segments, so hopper's training-only reward
    normalization has complete training segments to fit on.
    """
    g = np.random.default_rng(seed)
    lengths = g.integers(12, 30, size=EPISODES)
    n, ends = int(lengths.sum()), np.cumsum(lengths) - 1
    timeout = np.arange(EPISODES) % 4 == 3
    terminals, timeouts = np.zeros(n, bool), np.zeros(n, bool)
    terminals[ends[~timeout]], timeouts[ends[timeout]] = True, True
    obs = g.normal(size=(n, OBS_DIM)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("observations", data=obs)
        f.create_dataset("actions", data=g.uniform(-0.9, 0.9, (n, ACTION_DIM)).astype(np.float32))
        f.create_dataset("rewards", data=(np.tanh(obs[:, 0]) + 0.1 * g.normal(size=n)).astype(np.float32))
        f.create_dataset("terminals", data=terminals)
        f.create_dataset("timeouts", data=timeouts)


def with_cache(row, data_dir):
    row["cache"] = dict(filename="synthetic.hdf5", sha256=sha(Path(data_dir) / "synthetic.hdf5"))
    return row


def load(directory):
    with np.load(Path(directory) / "frozen.npz") as npz:
        arrays = {name: npz[name] for name in npz.files}
    return arrays, json.loads((Path(directory) / "frozen.json").read_text(encoding="utf8"))


def short_train(row, output, data_dir, **kwargs):
    settings = dict(updates=UPDATES, train_fraction=0.5, block=10, score_batch=10**6, data_dir=data_dir)
    return quiet(Q.freeze_short_train, row, output, **dict(settings, **kwargs))


class RecordingEnv:
    """Vector environment stub: resets to given raw observations, records the actions, ends every episode."""

    def __init__(self, observations, actions):
        self.observations, self.actions, self.num_envs = observations, actions, len(observations)

    def seed(self, seeds):
        assert len(seeds) == self.num_envs

    def reset(self):
        return self.observations

    def step(self, action):
        self.actions.append(np.asarray(action))
        n = self.num_envs
        return np.zeros_like(self.observations), np.ones(n), np.ones(n, bool), {}

    def close(self):
        pass


class FrozenCQLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="freeze-cql-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, ignore_errors=True)
        write_raw(cls.tmp / "synthetic.hdf5")
        cls.row = with_cache(Q.resolve_row("hopper", SEED, cls.tmp), cls.tmp)
        cls.result = short_train(cls.row, cls.tmp / "a", cls.tmp)
        cls.repeat = short_train(cls.row, cls.tmp / "b", cls.tmp)
        cls.arrays, cls.meta = load(cls.tmp / "a")
        cls.prepared = cls.result["prepared"]
        with h5py.File(cls.tmp / "synthetic.hdf5", "r") as f:
            cls.raw_obs = f["observations"][()]

    # (a) the scores are refresh's
    def test_scores_equal_what_refresh_computes(self):
        r = self.result
        args, config, models, state, bank = r["args"], r["config"], r["models"], r["state"], r["population"]
        key, max_action = jax.random.PRNGKey(SEED), self.prepared.max_action
        captured, original = {}, P.freeze_reference

        def spy(cal_params, residual_scale, predictions, residuals, wbcp_key, posterior):
            captured.update(predictions=predictions, residuals=residuals, unit=residual_scale)
            return original(cal_params, residual_scale, predictions, residuals, wbcp_key, posterior)

        with mock.patch.object(P, "freeze_reference", spy):
            refreshed, metrics, diagnostics = P.refresh(args, config, models, state, bank, key, max_action)
        self.assertTrue(metrics["posterior_inputs_valid"])
        # freeze_reference's own float64 inputs: residuals y - q and scale positive_scale(eta, u)
        residual = np.asarray(captured["residuals"], np.float64)
        sigma = np.asarray(positive_scale(captured["predictions"], captured["unit"]), np.float64)
        np.testing.assert_array_equal(self.arrays["residual"], residual)
        np.testing.assert_array_equal(self.arrays["sigma"], sigma)
        np.testing.assert_array_equal(self.arrays["score"], np.abs(residual) / sigma)
        self.assertEqual(diagnostics["scores"], len(self.arrays["row"]))
        posterior = config.posterior
        ours = wbcp.calibrate(self.arrays["score"], np.random.default_rng(np.asarray(
            jax.random.fold_in(key, 1347375956), np.uint32).ravel()), alpha=posterior.alpha,
            beta=posterior.credibility, draws=posterior.draws)
        self.assertEqual((ours.threshold, ours.lambda_hat, ours.lambda_hpd),
                         (diagnostics["threshold"], diagnostics["lambda_hat"], diagnostics["lambda_hpd"]))
        self.assertEqual(float(refreshed.posterior.residual_scale), self.meta["residual_unit"])
        # The target, q and eta score_rows reports are native_target's and refresh's.
        scored = Q.score_rows(args, models, state, bank, key, max_action, 10**6)
        target = P.native_target(args, models, state.native, bank, jax.random.fold_in(key, 1213156420), max_action)
        np.testing.assert_array_equal(scored["target"], np.asarray(target))
        np.testing.assert_array_equal(scored["residual"].astype(np.float64), residual)
        np.testing.assert_array_equal(scored["eta"], np.asarray(captured["predictions"]))

    def test_policy_action_is_the_evaluation_action(self):
        """runtime.cql.evaluate's action on the raw observations of the first population rows."""
        r = self.result
        protocol = quiet(Q.typed_population, r["row"])[3]
        event = protocol.evaluation_events[0]
        raw_rows = np.asarray(self.prepared.metadata["conversion_raw_indices"])[self.arrays["row"]]
        observations, actions, served = self.raw_obs[raw_rows], [], [0]

        def factory(dataset, workers):
            start = served[0]
            served[0] += workers
            return RecordingEnv(observations[start:start + workers], actions)

        R.evaluate(r["models"][0], r["state"].native.actor.params, self.prepared, event, dataset=r["args"].dataset,
                   max_workers=protocol.eval_workers, env_factory=factory)
        taken = np.concatenate(actions)
        self.assertEqual(len(taken), len(event.episode_seeds))
        np.testing.assert_allclose(taken, self.arrays["policy_action"][:len(taken)], rtol=1e-6, atol=1e-7)
        self.assertTrue(np.all(np.abs(self.arrays["policy_action"]) <= self.prepared.max_action))

    def test_chunked_forward_passes_agree(self):
        r = self.result
        chunked = Q.score_rows(r["args"], r["models"], r["state"], r["population"], jax.random.PRNGKey(SEED),
                               self.prepared.max_action, 37)
        for name in ("residual", "sigma"):
            np.testing.assert_allclose(chunked[name].astype(np.float64), self.arrays[name], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(chunked["policy_action"], self.arrays["policy_action"], rtol=1e-6, atol=1e-7)

    # (b) population and training rows
    def test_population_and_training_partition_by_whole_episodes(self):
        m, rows = self.prepared.metadata, self.arrays["row"]
        train, n = np.asarray(m["training_converted_indices"]), m["converted_rows"]
        ids = np.asarray(m["converted_episode_ids"])
        self.assertFalse(np.intersect1d(train, rows).size)
        np.testing.assert_array_equal(np.sort(np.r_[train, rows]), np.arange(n))
        self.assertFalse(np.intersect1d(ids[train], ids[rows]).size)  # no episode on both sides
        np.testing.assert_array_equal(np.flatnonzero(np.isin(ids, ids[rows])), rows)  # whole episodes only
        np.testing.assert_array_equal(m["withheld_converted_indices"], rows)
        np.testing.assert_array_equal(m["heldout_converted_indices"], rows)
        self.assertIsNone(m["reservation"]["rows_per_episode"])
        target = self.meta["split"]["target_size"]
        self.assertEqual(target, n - round(0.5 * n))
        self.assertGreaterEqual(len(rows), target)
        self.assertFalse(np.any(self.arrays["in_training"]))
        rows_meta = self.meta["rows"]
        self.assertEqual((rows_meta["training"] + rows_meta["population"], rows_meta["population"]),
                         (n, len(rows)))
        # The learner's pools hold exactly these rows; observations are normalized with training-only statistics.
        raw_rows = np.asarray(m["conversion_raw_indices"])
        mean, std = R.C.compute_mean_std(self.raw_obs[raw_rows[train]], eps=0.001)
        np.testing.assert_array_equal(np.asarray(self.meta["obs_normalization"]["obs_mean"], np.float32), mean)
        np.testing.assert_array_equal(np.asarray(self.meta["obs_normalization"]["obs_std"], np.float32), std)
        np.testing.assert_array_equal(self.arrays["obs"], (self.raw_obs[raw_rows[rows]] - mean) / std)
        np.testing.assert_array_equal(np.asarray(self.prepared.training.obs), (self.raw_obs[raw_rows[train]] - mean) / std)

    def test_population_is_the_td3_bc_population(self):
        """freeze_scores.py's TD3+BC split of the same file, seed and fraction selects the same rows and labels."""
        converted = F.converted_rows(self.row, self.tmp)
        td3 = F.split_row(with_cache(F.resolve_row("hopper", SEED, self.tmp), self.tmp), converted, 0.5, SEED)
        _, prepared = quiet(F.prepare, td3, self.tmp)
        np.testing.assert_array_equal(self.arrays["row"], prepared.metadata["withheld_converted_ids"])
        np.testing.assert_array_equal(self.arrays["row"], np.asarray(prepared.heldout_ids))
        np.testing.assert_array_equal(self.prepared.metadata["training_converted_indices"], prepared.training_ids)
        np.testing.assert_array_equal(self.prepared.metadata["conversion_raw_indices"],
                                      prepared.metadata["dependency_maps"]["raw_current"])
        component, timestep = F.components(prepared)
        np.testing.assert_array_equal(self.arrays["episode"], component[self.arrays["row"]])
        np.testing.assert_array_equal(self.arrays["timestep"], timestep[self.arrays["row"]])
        np.testing.assert_array_equal(self.arrays["obs"], np.asarray(prepared.heldout.obs))  # same training-only statistics
        # Both kinds of boundary occur: episodes end in terminals and at dropped timeout rows.
        maps = self.result["maps"]
        self.assertTrue(any(maps["terminal_mask"]) and np.any(np.diff(maps["raw_current"]) > 1))

    # (c) the benchmark reads it
    def test_artifact_loads_with_the_benchmark(self):
        pool = bench.load_pool(self.tmp / "a", "heldout")
        a, n = self.arrays, len(self.arrays["row"])
        self.assertEqual((pool.artifact_rows, len(pool.row), pool.population), (n, n, "heldout"))
        np.testing.assert_array_equal(pool.scores["normalized"], a["score"])
        np.testing.assert_array_equal(pool.scores["raw"], np.abs(a["residual"]))
        for name in ("obs", "action", "policy_action"):
            np.testing.assert_array_equal(getattr(pool, name), a[name].astype(np.float64))
        for name in ("episode", "timestep", "row"):
            np.testing.assert_array_equal(getattr(pool, name), a[name])
        self.assertEqual(len(bench.load_pool(self.tmp / "a", "all").row), n)
        _, starts, lengths = bench.episode_groups(pool.episode, pool.timestep)
        self.assertEqual(len(starts), self.meta["rows"]["population_episodes"])
        self.assertEqual(lengths.sum(), n)
        F.check_contract(a)

    def test_contract_and_metadata(self):
        a, m, n = self.arrays, self.meta, len(self.arrays["row"])
        self.assertEqual(set(a), set(F.ARRAYS))
        for name, dtype in F.ARRAYS.items():
            self.assertEqual((a[name].dtype, len(a[name])), (dtype, n), name)
        self.assertEqual((a["obs"].shape, a["action"].shape, a["policy_action"].shape),
                         ((n, OBS_DIM), (n, ACTION_DIM), (n, ACTION_DIM)))
        np.testing.assert_array_equal(a["action"], np.asarray(self.prepared.heldout.action))
        self.assertEqual((m["schema"], m["algorithm"], m["host"], m["method"], m["source"]),
                         ("wbcp-frozen-scores-v1", "cql", "cql", "bca", "short-train"))
        self.assertEqual((m["dataset"], m["updates"], m["seed"], m["score_seed"]), ("hopper-medium-v2", UPDATES, SEED, SEED))
        self.assertEqual(m["dataset_sha256"], sha(self.tmp / "synthetic.hdf5"))
        self.assertEqual(m["target_noise"]["noise_fold"], 1213156420)
        self.assertEqual(m["preparation"]["protocol_type"], "runtime.cql.PopulationSplitProtocol")
        self.assertEqual(m["checkpoint"]["sha256"], sha(self.tmp / "a" / m["checkpoint"]["path"]))
        self.assertEqual(m["checkpoint"]["counters"]["accepted_scale_fits"], UPDATES)
        self.assertEqual(m["npz_sha256"], sha(self.tmp / "a" / "frozen.npz"))
        self.assertEqual(m["residual_unit"], float(self.result["state"].residual_scale))
        self.assertFalse(bool(self.result["state"].posterior.ready))
        self.assertEqual(m["rows"]["converted_components"], len(np.unique(self.prepared.metadata["converted_episode_ids"])))
        log = json.loads((self.tmp / "a" / "training_log.json").read_text(encoding="utf8"))
        self.assertEqual([b["step"] for b in log["blocks"]], [10, 20])
        # No refresh: the reference never becomes ready and the conservative gap keeps multiplier 1 (native CQL).
        self.assertTrue(all(b["scale_fit_accepted"] == 1.0 and b["critic_dose_mean"] == 1.0
                            and b["posterior_ready"] == 0.0 for b in log["blocks"]))

    # (d) reproducible
    def test_reproducible_from_the_seeds(self):
        repeat, _ = load(self.tmp / "b")
        for name in F.ARRAYS:
            np.testing.assert_array_equal(self.arrays[name], repeat[name], name)
        self.assertEqual(self.meta["checkpoint"]["sha256"], self.repeat["metadata"]["checkpoint"]["sha256"])
        self.assertEqual(self.meta["npz_sha256"], self.repeat["metadata"]["npz_sha256"])
        # Another score seed redraws only the target noise: same scale, different residuals.
        r = self.result
        other = Q.score_rows(r["args"], r["models"], r["state"], r["population"], jax.random.PRNGKey(SEED + 1),
                             self.prepared.max_action, 10**6)
        np.testing.assert_array_equal(other["sigma"].astype(np.float64), self.arrays["sigma"])
        self.assertFalse(np.array_equal(other["residual"].astype(np.float64), self.arrays["residual"]))

    # the runtime accepts rows_per_episode=None only as a population split
    def test_whole_episodes_only_for_population_splits(self):
        r = self.result
        protocol = quiet(Q.typed_population, r["row"])[3]
        fields = {name: getattr(protocol, name) for name in protocol.__dataclass_fields__}
        R.RunProtocol(**fields | {"calibration_rows_per_episode": 6})  # the same fields with an integer K
        with self.assertRaises(ValueError):
            R.RunProtocol(**fields)  # a target with rows_per_episode=None
        with self.assertRaises(ValueError):
            quiet(typed, r["row"])  # runtime.config.typed builds RunProtocol, which still refuses None
        for k in (6, 1):
            with self.assertRaises(ValueError):
                R.PopulationSplitProtocol(**fields | {"calibration_rows_per_episode": k})
        with self.assertRaises(ValueError):
            R.PopulationSplitProtocol(**fields | {"calibration_target_size": None})
        with self.assertRaises(ValueError):
            R.run_prepared(r["args"], r["config"], protocol, self.prepared)
        data = self.prepared.training
        ids = np.asarray(self.prepared.metadata["converted_episode_ids"])[self.prepared.metadata["training_converted_indices"]]
        with self.assertRaises(ValueError):
            P.reserve_pool(data, 10, 1, None, max_fraction=0.5, episode_ids=ids)
        with self.assertRaises(ValueError):
            P.reserve_pool(data, 10, 1, 5, max_fraction=0.5, episode_ids=ids, population_split=True)
        # Integer K is unchanged: reserve_pool is reserve_calibration.
        args = (data.obs, data.next_obs, data.done, 10, 1, 5)
        for got, expected in zip(P.reserve_pool(data, 10, 1, 5, max_fraction=0.5, episode_ids=ids),
                                 reserve_calibration(*args, max_fraction=0.5, episode_ids=ids)):
            np.testing.assert_equal(got, expected)


@unittest.skipUnless(HOPPER.is_file(), "needs the cached hopper-medium-v2 HDF5")
class RealHopperSplitTests(unittest.TestCase):
    """Preparation only (no training) of the cached hopper-medium-v2 file at the matrix's split."""

    def test_population_matches_the_td3_bc_pool(self):
        with tempfile.TemporaryDirectory(prefix="freeze-cql-hopper-") as tmp:
            row = Q.resolve_row("hopper", SEED, tmp)
        converted = F.converted_rows(row, HOPPER.parent)
        row = Q.split_row(row, converted, 0.5, SEED)
        _, prepared = quiet(Q.prepare, row, HOPPER.parent)
        m = prepared.metadata
        maps = Q.dependency_maps(prepared, HOPPER)  # raises unless CQL's episodes are TD3+BC's components
        rows = np.asarray(m["withheld_converted_indices"])
        self.assertEqual(m["converted_rows"], converted)
        np.testing.assert_array_equal(m["heldout_converted_indices"], rows)
        self.assertEqual(int(np.asarray(maps["effective_components"])[-1]) + 1, len(np.unique(m["converted_episode_ids"])))
        if TD3_HOPPER.is_file():
            with np.load(TD3_HOPPER) as npz:
                np.testing.assert_array_equal(rows, npz["row"])
                component, timestep = F.components(SimpleNamespace(metadata={"dependency_maps": maps}))
                np.testing.assert_array_equal(component[rows], npz["episode"])
                np.testing.assert_array_equal(timestep[rows], npz["timestep"])


class CommandLineTests(unittest.TestCase):
    """Argument validation only; nothing is prepared or trained."""

    def test_rejects_invalid_invocations(self):
        with tempfile.TemporaryDirectory(prefix="freeze-cql-cli-") as tmp:
            fresh = str(Path(tmp) / "new")
            for argv in (["--output", tmp],  # --output must not exist
                         ["--dataset", "antmaze", "--output", fresh],
                         ["--seed", "7", "--output", fresh],
                         ["--updates", "0", "--output", fresh],
                         ["--train-fraction", "1.0", "--output", fresh],
                         ["--split-seed", "-1", "--output", fresh],
                         ["--score-seed", str(2**32), "--output", fresh],
                         ["--score-batch", "0", "--output", fresh]):
                with self.subTest(argv=argv), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                    Q.main(argv)
            self.assertFalse(Path(fresh).exists())

    def test_every_declared_dataset_key_resolves_to_a_population_split(self):
        """host_matrix.py calls --dataset <key> for every key of configs/cql.yaml."""
        for key in Q.read_config(Q.CONFIG)["datasets"]:
            with self.subTest(key=key):
                row = Q.split_row(Q.resolve_row(key, SEED, "/tmp/not-executed"), 1000, 0.5, SEED)
                protocol = row["protocol"]
                self.assertEqual((protocol["calibration_target_size"], protocol["calibration_seed"],
                                  protocol["calibration_rows_per_episode"]), (500, SEED, None))
                self.assertEqual(protocol["calibration_max_fraction"], F.MAX_FRACTION)
                _, _, _, typed_protocol = quiet(Q.typed_population, row)
                self.assertIs(type(typed_protocol), R.PopulationSplitProtocol)


if __name__ == "__main__":
    unittest.main()
