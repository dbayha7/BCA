"""CPU checks for experiments/wbcp/verify_pools.py, the frozen pool verifier (signal study, step 1c).

One synthetic HDF5 file with D4RL's layout drives the four real builders (freeze_scores.py,
freeze_cql.py, freeze_rebrac.py, freeze_iql.py; a few updates each), so the verifier reads the
schemas the builders actually write. Genuine pools must pass every check, with and without
re-preparation, and hold one population across the hosts. Copies are then tampered with one
fault each (score file bytes, a dropped or trained-on population row, checkpoint bytes, the
dataset file, the configs' declared hash, a pool split with another seed, a host that declared
another file for the same dataset, ReBRAC's recorded calibrator statistics), and each fault must
fail exactly its check. A re-preparation that cannot run (an old resolved.json without
rows_per_episode) must leave its fields 'not re-derivable', neither matched nor mismatched.
Discovery, a run that verifies nothing, the IQL superseded flag, run records, failure.json and
the command line are checked too.
JAX_PLATFORMS=cpu python -m unittest experiments.wbcp.test_verify_pools
"""
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import yaml

from experiments.wbcp import freeze_cql as Q
from experiments.wbcp import freeze_iql as I
from experiments.wbcp import freeze_rebrac as FR
from experiments.wbcp import freeze_scores as F
from experiments.wbcp import verify_pools as V
from runtime.provenance import sha

SEED, OBS_DIM, ACTION_DIM, EPISODES, UPDATES = 202609171, 11, 3, 20, 4
DATASET = "hopper-medium-v2"  # configs/*.yaml hopper environment, frozen.json dataset
OLD_IQL_Q = "min over the twin online Q heads at (obs, action) (calibration.iql_reference.refresh)"


def write_raw(path, seed=5):
    """D4RL-shaped raw transitions (test_freeze_cql.write_raw): 20 episodes of 12-29 rows; every fourth
    ends in a timeout, the rest in a terminal, so CQL's training-only reward normalization can fit."""
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


def synthetic_configs(target, data_file):
    """Copies of configs/<algorithm>.yaml whose hopper entry declares the synthetic file."""
    target.mkdir()
    for algorithm in V.HOSTS:
        config = yaml.safe_load((V.CONFIG_DIR / f"{algorithm}.yaml").read_text(encoding="utf8"))
        cache = config["datasets"]["hopper"]["cache"]
        cache.update(filename=data_file.name, sha256=sha(data_file))
        if "bytes" in cache:
            cache["bytes"] = data_file.stat().st_size
        (target / f"{algorithm}.yaml").write_text(yaml.safe_dump(config), encoding="utf8")
    return target


def build(module, root, name, data_dir, **kwargs):
    row = module.resolve_row("hopper", SEED, root)
    row["cache"] = dict(filename="synthetic.hdf5", sha256=sha(Path(data_dir) / "synthetic.hdf5"))
    settings = dict(updates=UPDATES, train_fraction=0.5, block=2, score_batch=10**6, data_dir=data_dir)
    with contextlib.redirect_stdout(io.StringIO()):  # the runtimes' legacy reward-transform print
        module.freeze_short_train(row, root / name, **dict(settings, **kwargs))
    return root / name


def load_npz(directory):
    with np.load(Path(directory) / "frozen.npz") as npz:
        return {name: npz[name] for name in npz.files}


def edit_json(directory, change, name="frozen.json"):
    path = Path(directory) / name
    meta = json.loads(path.read_text(encoding="utf8"))
    change(meta)
    path.write_text(json.dumps(meta, indent=2), encoding="utf8")


def rewrite_npz(directory, change, record=True):
    """Change frozen.npz; with record=True frozen.json's npz_sha256 follows, so only the content is wrong."""
    arrays = load_npz(directory)
    change(arrays)
    path = Path(directory) / "frozen.npz"
    path.unlink()
    with path.open("xb") as f:
        np.savez(f, **arrays)
    if record:
        edit_json(directory, lambda meta: meta.update(npz_sha256=sha(path)))


class VerifyPoolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="verify-pools-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, ignore_errors=True)
        cls.data = cls.tmp / "data"
        cls.data.mkdir()
        write_raw(cls.data / "synthetic.hdf5")
        cls.configs = synthetic_configs(cls.tmp / "configs", cls.data / "synthetic.hdf5")
        cls.root = cls.tmp / "pools"
        cls.pools = {host: build(module, cls.root, name, cls.data) for host, module, name in (
            ("td3_bc", F, "hopper-medium-v2-s202609171-u4"), ("cql", Q, "cql-hopper-s202609171-u4"),
            ("rebrac", FR, "rebrac-hopper-s202609171-u4"), ("iql", I, "iql-hopper-s202609171-u4"))}
        cls.other_seed = build(F, cls.tmp / "other", "hopper-medium-v2-s202609172-u4", cls.data, split_seed=SEED + 1)
        cls.checkpoint_all = cls.tmp / "ckpt" / "hopper-medium-v2-s202609171-u4-ckpt-all"  # every converted row
        with contextlib.redirect_stdout(io.StringIO()):
            F.freeze_checkpoint(cls.pools["td3_bc"], cls.checkpoint_all, score_batch=10**6, data_dir=cls.data)
        cls.report = cls.verify(list(cls.pools.values()))
        cls.by_host = {r["host"]: r for r in cls.report["pools"]}

    @classmethod
    def verify(cls, pools, **kwargs):
        return V.verify(pools, **dict(dict(data_dir=cls.data, config_dir=cls.configs), **kwargs))

    def copy(self, host, name=None):
        target = Path(tempfile.mkdtemp(dir=self.tmp)) / (name or self.pools[host].name)
        shutil.copytree(self.pools[host], target)
        return target

    def single(self, directory, **kwargs):
        report = self.verify([directory], **kwargs)
        self.assertEqual(len(report["pools"]), 1)
        return report, report["pools"][0]

    def assert_only(self, pool, failed):
        marks = {name: check["ok"] for name, check in pool["checks"].items()}
        self.assertEqual({name for name, ok in marks.items() if ok is False}, set(failed), marks)
        self.assertFalse(pool["ok"])

    @staticmethod
    def field(pool, name):
        return next(f for f in pool["checks"]["derived"]["fields"] if f["field"] == name)

    # genuine pools ----------------------------------------------------------------------------

    def test_genuine_pools_pass_every_check(self):
        self.assertTrue(self.report["ok"], self.report["failures"])
        self.assertEqual(set(self.by_host), set(V.HOSTS))
        for host, pool in self.by_host.items():
            with self.subTest(host=host):
                self.assertTrue(pool["ok"])
                self.assertTrue(all(c["ok"] for c in pool["checks"].values()))
                counts = pool["checks"]["derived"]["counts"]
                self.assertEqual(counts["mismatch"], 0)
                self.assertGreaterEqual(counts["match"], 9)
                self.assertEqual(pool["checks"]["npz"]["contract"], "ok")
                self.assertIsNone(pool["superseded"])  # current freeze_iql scores the target heads
        # each host's own derivations were exercised
        for host, name in (("td3_bc", "preparation.run_input_hashes.obs_mean"),
                           ("td3_bc", "preparation.raw_dataset_sha256"),
                           ("cql", "preparation.reservation.withheld_indices_sha256"),
                           ("rebrac", "preparation.run_input_hashes.heldout_ids"),
                           ("rebrac", "preparation.run_input_hashes.cal_obs_mean"),
                           ("iql", "preparation.metadata.train_indices_sha256"),
                           ("iql", "preparation.metadata.episode_ids_sha256"),
                           ("iql", "preparation.metadata.raw_terminal_timeout_sha256"),
                           ("iql", "preparation.metadata.converted_numeric_data_sha256")):
            self.assertEqual(self.field(self.by_host[host], name)["status"], "match", (host, name))
        for host in V.HOSTS:
            for name in ("frozen.npz obs", "frozen.npz action", "obs_normalization.obs_mean"):
                self.assertEqual(self.field(self.by_host[host], name)["status"], "match", (host, name))
        self.assertEqual(self.field(self.by_host["rebrac"], "preparation.run_input_hashes.cal_obs_mean")["derived"].keys(),
                         {"checkpoint"})
        self.assertTrue(all(f["status"] == "code-dependent: not compared" for f in self.by_host["td3_bc"]["checks"]
                            ["derived"]["fields"] if f["field"] == "preparation.settings_sha256"))

    def test_one_population_across_hosts(self):
        rows = load_npz(self.pools["td3_bc"])["row"]
        expected = V.population_sha256(rows)
        (name, group), = self.report["datasets"].items()
        self.assertEqual((name, group["identical"], group["population_sha256"]), (DATASET, True, [expected]))
        self.assertEqual((group["same_file"], group["dataset_files"]),
                         (True, [dict(filename="synthetic.hdf5", sha256=sha(self.data / "synthetic.hdf5"))]))
        self.assertEqual({r["population_sha256"] for r in self.report["pools"]}, {expected})
        # the hash rule is reserve_calibration's withheld_indices_sha256, which CQL records
        cql = json.loads((self.pools["cql"] / "frozen.json").read_text())
        self.assertEqual(cql["preparation"]["reservation"]["withheld_indices_sha256"], expected)
        split = self.by_host["td3_bc"]["checks"]["split"]
        self.assertEqual((split["population"], split["converted"]), (len(rows), split["training"] + len(rows)))
        self.assertEqual(split["target_size"], split["converted"] - round(0.5 * split["converted"]))

    def test_reprepare_compares_every_prepared_hash(self):
        report = self.verify(list(self.pools.values()), reprepare=True)
        self.assertTrue(report["ok"], report["failures"])
        for pool in report["pools"]:
            with self.subTest(host=pool["host"]):
                derived = pool["checks"]["derived"]
                self.assertIsNone(derived["reprepare_error"])
                self.assertEqual((derived["counts"]["mismatch"], derived["counts"]["not re-derivable"]), (0, 0))
                self.assertEqual(self.field(pool, "re-prepared population rows")["status"], "match")
        rebrac = next(p for p in report["pools"] if p["host"] == "rebrac")
        self.assertEqual(self.field(rebrac, "preparation.run_input_hashes.cal_obs_std")["derived"].keys(),
                         {"checkpoint", "re-preparation"})

    def test_failed_reprepare_leaves_fields_not_re_derivable(self):
        """The oldest TD3+BC pool's resolved.json has no reservation rows_per_episode, so the current
        prepare() cannot run on it: the fields only a re-preparation derives stay 'not re-derivable'
        (expectations step 1c, item 4), never a match or a mismatch, and the pool still passes."""
        directory = self.copy("td3_bc")
        edit_json(directory, lambda r: r["protocol"]["reservation"].pop("rows_per_episode"), name="resolved.json")
        report, pool = self.single(directory, reprepare=True)
        derived = pool["checks"]["derived"]
        self.assertIn("rows_per_episode", derived["reprepare_error"])
        expected = {f["field"] for f in self.by_host["td3_bc"]["checks"]["derived"]["fields"]
                    if f["field"].startswith("preparation.run_input_hashes.") and f["status"] == V.NOT_DERIVABLE}
        self.assertEqual(len(expected), 4, expected)
        for name in expected:
            status = self.field(pool, name)["status"]
            self.assertTrue(status.startswith(V.NOT_DERIVABLE + " (re-preparation failed: "), (name, status))
            self.assertIn("rows_per_episode", status)
        self.assertEqual(derived["counts"]["mismatch"], 0)
        self.assertEqual(derived["counts"]["not re-derivable"],
                         self.by_host["td3_bc"]["checks"]["derived"]["counts"]["not re-derivable"])
        self.assertNotIn("re-prepared population rows", [f["field"] for f in derived["fields"]])
        self.assertTrue(derived["ok"])
        self.assertTrue(pool["ok"], pool["failures"])
        self.assertTrue(report["ok"], report["failures"])
        self.assertIn("re-preparation failed", V.markdown(report))

    def test_device_dependent_tolerance(self):
        name = "preparation.run_input_hashes.cal_obs_mean"
        self.assertTrue(V._agrees(name, "re-preparation", dict(scaled_deviation_from_checkpoint=1e-7), "any"))
        self.assertFalse(V._agrees(name, "re-preparation", dict(scaled_deviation_from_checkpoint=2e-5), "any"))
        self.assertFalse(V._agrees(name, "checkpoint", "a" * 64, "b" * 64))

    # one fault each ---------------------------------------------------------------------------

    def test_changed_score_file_fails_the_npz_check(self):
        directory = self.copy("td3_bc")
        rewrite_npz(directory, lambda a: a["residual"].__setitem__(0, -a["residual"][0]), record=False)
        report, pool = self.single(directory)
        self.assert_only(pool, ["npz"])
        self.assertEqual(pool["checks"]["npz"]["contract"], "ok")  # |residual| unchanged: only the bytes differ
        self.assertFalse(report["ok"])

    def test_dropped_or_trained_population_row_fails_the_split_check(self):
        dropped = self.copy("cql")
        rewrite_npz(dropped, lambda a: a.update({name: value[:-1] for name, value in a.items()}))
        _, pool = self.single(dropped)
        self.assert_only(pool, ["split"])  # obs/action are still the rows the pool claims; the row set is wrong
        self.assertIn("row equals the re-derived withheld rows", pool["checks"]["split"]["failed"])
        self.assertIn("rows.population", pool["checks"]["split"]["failed"])
        trained = self.copy("td3_bc")
        rewrite_npz(trained, lambda a: a["in_training"].__setitem__(0, True))
        _, pool = self.single(trained)
        self.assert_only(pool, ["split"])
        self.assertEqual(pool["checks"]["split"]["failed"], ["in_training false throughout"])

    def test_changed_checkpoint_fails_the_checkpoint_check(self):
        directory = self.copy("td3_bc")
        with (directory / f"checkpoint_{UPDATES}.msgpack").open("ab") as f:
            f.write(b"\0")
        _, pool = self.single(directory)
        self.assert_only(pool, ["checkpoint"])

    def test_changed_dataset_file_fails_the_data_check(self):
        data = self.tmp / "changed-data"
        data.mkdir()
        shutil.copy(self.data / "synthetic.hdf5", data / "synthetic.hdf5")
        with h5py.File(data / "synthetic.hdf5", "r+") as f:
            f["rewards"][0] += 1.0  # timeouts and terminals unchanged: the split still reproduces
        for host in ("td3_bc", "cql"):
            with self.subTest(host=host):
                _, pool = self.single(self.copy(host), data_dir=data)
                self.assert_only(pool, ["data", "derived"])
                self.assertEqual(self.field(pool, "preparation.raw_dataset_sha256")["status"], "mismatch")
                self.assertIn("frozen.json dataset_sha256", pool["checks"]["data"]["mismatches"])
        _, pool = self.single(self.copy("iql"), data_dir=data)  # IQL fingerprints the converted rewards, not the raw file
        self.assert_only(pool, ["data", "derived"])
        self.assertEqual(self.field(pool, "preparation.metadata.converted_numeric_data_sha256")["status"], "mismatch")
        self.assertEqual(self.field(pool, "preparation.metadata.raw_terminal_timeout_sha256")["status"], "match")

    def test_configs_declaration_is_checked(self):
        _, pool = self.single(self.pools["td3_bc"], config_dir=V.CONFIG_DIR)  # declares hopper_medium-v2.hdf5
        self.assert_only(pool, ["data"])
        mismatches = pool["checks"]["data"]["mismatches"]
        self.assertIn("configs/td3_bc.yaml cache.sha256", mismatches)
        self.assertIn("configs filename differs from frozen.json dataset_file.filename", mismatches)

    def test_another_split_seed_passes_alone_but_breaks_the_shared_population(self):
        report = self.verify([self.pools["td3_bc"], self.other_seed])
        self.assertTrue(all(pool["ok"] for pool in report["pools"]))  # each reproduces its own recorded split
        self.assertFalse(report["ok"])
        self.assertEqual((report["datasets"][DATASET]["same_file"], report["datasets"][DATASET]["identical"]),
                         (True, False))
        self.assertEqual(report["failures"], [f"{DATASET}: pools hold different populations"])
        self.assertIn("| yes | NO |", V.markdown(report))

    def test_another_file_for_the_same_dataset_breaks_the_comparison(self):
        """A host whose config declares another cached file for the environment passes on its own (its
        file, config and resolved.json agree, and here even its population and bytes do), but the
        dataset's pools no longer share one file, so the run fails."""
        data, configs = self.tmp / "relabel-data", self.tmp / "relabel-configs"
        data.mkdir()
        shutil.copy(self.data / "synthetic.hdf5", data / "synthetic.hdf5")
        shutil.copy(self.data / "synthetic.hdf5", data / "synthetic-copy.hdf5")
        shutil.copytree(self.configs, configs)
        config = yaml.safe_load((configs / "cql.yaml").read_text(encoding="utf8"))
        config["datasets"]["hopper"]["cache"]["filename"] = "synthetic-copy.hdf5"
        (configs / "cql.yaml").write_text(yaml.safe_dump(config), encoding="utf8")
        relabeled = self.copy("cql")
        edit_json(relabeled, lambda m: m["dataset_file"].update(filename="synthetic-copy.hdf5"))
        edit_json(relabeled, lambda r: r["cache"].update(filename="synthetic-copy.hdf5"), name="resolved.json")
        report = self.verify([self.pools["td3_bc"], relabeled], data_dir=data, config_dir=configs)
        self.assertTrue(all(pool["ok"] for pool in report["pools"]), [p["failures"] for p in report["pools"]])
        self.assertEqual(sorted(p["dataset_file"] for p in report["pools"]), ["synthetic-copy.hdf5", "synthetic.hdf5"])
        group = report["datasets"][DATASET]
        self.assertEqual((list(report["datasets"]), group["same_file"], group["identical"]), ([DATASET], False, True))
        self.assertEqual([f["filename"] for f in group["dataset_files"]], ["synthetic-copy.hdf5", "synthetic.hdf5"])
        self.assertFalse(report["ok"])
        self.assertEqual(report["failures"], [f"{DATASET}: pools declare different dataset files"])
        self.assertIn("| NO | yes |", V.markdown(report))

    def test_checkpoint_mode_pools_have_no_split_and_stay_out_of_the_comparison(self):
        report = self.verify([self.pools["td3_bc"], self.checkpoint_all])
        self.assertTrue(report["ok"], report["failures"])
        pool = next(p for p in report["pools"] if p["source"] == "checkpoint")
        self.assertEqual({name: check["ok"] for name, check in pool["checks"].items()},
                         dict(npz=True, checkpoint=True, data=True, split=None, derived=None))
        self.assertEqual(len(load_npz(self.checkpoint_all)["row"]), self.by_host["td3_bc"]["checks"]["split"]["converted"])
        self.assertEqual(list(report["datasets"][DATASET]["pools"]), [self.pools["td3_bc"].name])

    def test_recorded_calibrator_statistics_are_checked_against_the_checkpoint(self):
        directory = self.copy("rebrac")
        edit_json(directory, lambda m: m["preparation"]["run_input_hashes"].update(cal_obs_mean="0" * 64))
        _, pool = self.single(directory)
        self.assert_only(pool, ["derived"])
        self.assertEqual(self.field(pool, "preparation.run_input_hashes.cal_obs_mean")["status"], "mismatch")

    # flags and records ------------------------------------------------------------------------

    def test_superseded_iql_pools_are_flagged_and_still_verified(self):
        directory = self.copy("iql")
        edit_json(directory, lambda m: m["score_definition"].update(q=OLD_IQL_Q))
        report, pool = self.single(directory)
        self.assertTrue(pool["ok"])
        self.assertTrue(pool["superseded"].startswith("superseded (pre-qf_target fix)"))
        self.assertIn("superseded (pre-qf_target fix)", V.markdown(report))

    def test_run_record_and_failure_json(self):
        clean = self.copy("cql")
        (clean / "run_record.json").write_text(json.dumps(dict(device="gpu", exit_code=0)))
        report, pool = self.single(clean)
        self.assertTrue(pool["ok"])
        self.assertEqual(pool["run_record"]["device"], "gpu")
        self.assertIn("gpu, exit 0", V.markdown(report))
        failed = self.copy("cql")
        (failed / "run_record.json").write_text(json.dumps(dict(device="gpu", exit_code=1)))
        _, pool = self.single(failed)
        self.assertEqual(pool["failures"], ["run_record exit_code 1"])
        crashed = self.copy("cql")
        (crashed / "failure.json").write_text("{}")
        _, pool = self.single(crashed)
        self.assertEqual(pool["failures"], ["failure.json present"])

    def test_discovery_skips_smoke_runs_and_unfinished_pools(self):
        root = self.tmp / "discovery"
        root.mkdir()
        for name in ("cql-hopper-s202609171-u100000", "cql-smoke-hopper-u2000", "cql-gputest-hopper-u2000",
                     "hopper-medium-v2-s202609171-u2000-ckpt-all"):
            shutil.copytree(self.pools["cql"], root / name)
        (root / "rebrac-walker2d-s202609171-u100000").mkdir()  # still being written: no frozen.json yet
        pools, skipped = V.discover(root)
        self.assertEqual([p.name for p in pools], ["cql-hopper-s202609171-u100000"])
        self.assertEqual(len(skipped), 4)
        self.assertIn("frozen.json missing", dict((p.name, r) for p, r in skipped)["rebrac-walker2d-s202609171-u100000"])
        pools, skipped = V.discover(root, everything=True)
        self.assertEqual(len(pools), 4)
        self.assertEqual([p.name for p, _ in skipped], ["rebrac-walker2d-s202609171-u100000"])
        report = self.verify([root / "rebrac-walker2d-s202609171-u100000"])  # skipped, so nothing was verified
        self.assertEqual((report["pools"], len(report["skipped"]), report["ok"]), ([], 1, False))
        self.assertEqual(report["failures"], ["no pools verified"])
        self.assertIn("Overall: FAIL", V.markdown(report))
        report = self.verify([root / "cql-hopper-s202609171-u100000", root / "rebrac-walker2d-s202609171-u100000"])
        self.assertEqual((len(report["pools"]), len(report["skipped"]), report["ok"]), (1, 1, True))

    def test_command_line(self):
        out = self.tmp / "cli"
        base = ["--root", str(self.root), "--data-dir", str(self.data), "--config-dir", str(self.configs)]
        with mock.patch("sys.stdout", io.StringIO()) as printed:
            self.assertEqual(V.main(base + ["--output", str(out)]), 0)
        report = json.loads((out / "verification.json").read_text(encoding="utf8"))
        table = (out / "verification.md").read_text(encoding="utf8")
        self.assertEqual((report["schema"], report["ok"], len(report["pools"])), (V.SCHEMA, True, 4))
        self.assertIn("| pool | host | dataset | npz ok | checkpoint ok | data ok | split ok |", table)
        self.assertEqual(printed.getvalue().strip(), table.strip())
        broken = self.copy("td3_bc")
        rewrite_npz(broken, lambda a: a["in_training"].__setitem__(0, True))
        with mock.patch("sys.stdout", io.StringIO()):
            self.assertEqual(V.main([str(broken), *base[2:], "--output", str(self.tmp / "cli-broken")]), 1)
        empty = self.tmp / "cli-empty-root"
        empty.mkdir()
        with mock.patch("sys.stdout", io.StringIO()):  # an empty --root verifies nothing and fails
            self.assertEqual(V.main(["--root", str(empty), *base[2:], "--output", str(self.tmp / "cli-empty")]), 1)
        with mock.patch("sys.stdout", io.StringIO()):  # so does a mistyped explicit pool path
            self.assertEqual(V.main([str(self.tmp / "no-such-pool"), *base[2:], "--output", str(self.tmp / "cli-typo")]), 1)
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            V.main(base + ["--output", str(out)])  # --output must not exist


if __name__ == "__main__":
    unittest.main()
