"""Checks of the host x dataset matrix (host_matrix.py): designs, commands, the expectations guard and the loader."""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from experiments.wbcp import host_matrix as M


class DesignTests(unittest.TestCase):
    def test_every_host_and_dataset_has_the_full_design_set(self):
        for host in M.HOSTS:
            for dataset in M.DATASETS:
                with self.subTest(host=host, dataset=dataset):
                    n, k, big = M.bank(host, dataset)
                    p = M.prefix(dataset)
                    names = [name for name, _, _ in M.designs(host, dataset)]
                    self.assertEqual(len(names), len(set(names)))
                    for required in ("iid", "strat2", "strat5", "strat10", "blocks", f"resv{k}", "strat5_shift"):
                        self.assertIn(f"{p}_{required}", names)
                    self.assertIn(f"{p}_strat{k}", names)  # listed once, either as K=2/5/10 or the configured K
                    self.assertEqual(f"{p}_iid_big" in names, n < big)

    def test_configured_bank_uses_bca_sampler_at_the_configured_size(self):
        n, k, _ = M.bank("iql", "walker2d")
        self.assertEqual((n, k), (8192, 67))
        args = dict((name, a) for name, _, a in M.designs("iql", "walker2d"))["walker2d_resv67"]
        self.assertEqual(args[args.index("--n") + 1], "8192")
        self.assertEqual(args[args.index("--spacing") + 1], "reservation")
        self.assertEqual(args[args.index("--gamma") + 1], "0")

    def test_shift_run_has_six_tilts_and_its_own_seed(self):
        args = dict((name, a) for name, _, a in M.designs("cql", "hopper"))["hopper_strat5_shift"]
        tilts = args[args.index("--tilt") + 1: args.index("--gamma")]
        gammas = args[args.index("--gamma") + 1: args.index("--seed")]
        self.assertEqual((tilts, gammas), (("policy", "density", "state"), ("0.5", "1")))
        self.assertEqual(len(tilts) * len(gammas), len(M.TILTS))
        self.assertEqual(args[args.index("--per-episode") + 1], "5")


class CommandTests(unittest.TestCase):
    def test_paths_are_posix_and_the_freeze_script_is_the_hosts(self):
        for host in M.HOSTS:
            (target, log, argv), = M.commands(host, "freeze", ["pen-human"])
            self.assertNotIn("\\", target + log + " ".join(argv))
            self.assertEqual(argv[1], f"experiments/wbcp/{M.FREEZE[host]}")
            self.assertEqual(argv[argv.index("--output") + 1], target)

    def test_td3_bc_keeps_existing_pool_names_and_others_are_prefixed(self):
        self.assertEqual(M.frozen_name("td3_bc", "hopper"), "hopper-medium-v2-s202609171-u100000")
        self.assertEqual(M.frozen_name("td3_bc", "pen-human"), "pen-human-s202609171-u100000")
        self.assertEqual(M.frozen_name("cql", "hopper"), "cql-hopper-s202609171-u100000")

    def test_predictions_cover_the_configured_k_with_both_samplers(self):
        (_, _, argv), = M.commands("td3_bc", "predict", ["walker2d"])
        stratified = argv[argv.index("--stratified") + 1: argv.index("--reservation")]
        self.assertEqual(stratified, ["2", "5", "10", "23"])
        self.assertEqual(argv[argv.index("--reservation") + 1:], ["23"])


class GuardAndLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(M, "ROOT", self.tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.configs = os.path.join(self.tmp.name, "configs")
        os.makedirs(self.configs)
        for host in M.HOSTS:  # the loader and bank() read the real configs through ROOT
            with open(os.path.join(os.path.dirname(__file__), "..", "..", "configs", f"{host}.yaml"), encoding="utf-8") as src:
                with open(os.path.join(self.configs, f"{host}.yaml"), "w", encoding="utf-8") as dst:
                    dst.write(src.read())

    def test_stages_refuse_to_run_without_expectations(self):
        self.assertIn("missing", M.check_expectations("rebrac", "freeze"))
        os.makedirs(os.path.join(self.tmp.name, M.run_root("rebrac")))
        with open(os.path.join(self.tmp.name, M.run_root("rebrac"), "expectations.md"), "w") as handle:
            handle.write("# Stage A only\n")
        self.assertIsNone(M.check_expectations("rebrac", "freeze"))
        self.assertIn("Stage B", M.check_expectations("rebrac", "bench"))
        with open(os.path.join(self.tmp.name, M.run_root("rebrac"), "expectations.md"), "a") as handle:
            handle.write("## Stage B\n")
        self.assertIsNone(M.check_expectations("rebrac", "bench"))
        with self.assertRaises(SystemExit):
            M.run("cql", "bench", ["hopper"])

    def test_loader_reads_runs_and_lists_missing_and_extra(self):
        root = os.path.join(self.tmp.name, M.run_root("iql"))
        os.makedirs(root)
        arm = dict(failures=200, trials=4000, ci=[0.04, 0.06], risk=0.09, threshold=1.2, abstain=0.0)
        block = dict(tilt="policy", gamma=0.0, n=8192, score="normalized", lambda_star=1.1,
                     calibration_n_eff=dict(oracle=8192.0, estimated=8000.0), calibration_size=dict(mean=8192.0),
                     arms={"BQ-CP": arm, "WBCP": arm, "WBCP (oracle w)": arm, "RCPS": arm})
        for name in ("hopper_iid", "hopper_custom"):
            with open(os.path.join(root, name + ".json"), "w") as handle:
                json.dump(dict(settings=dict(n=[8192], trials=4000, seed=1), blocks=[block]), handle)
        result = M.load_results("iql")
        hopper = result["datasets"]["hopper"]
        self.assertEqual((hopper["n"], hopper["k"]), (8192, 17))
        self.assertEqual(hopper["runs"]["hopper_iid"]["blocks"][0]["A"]["BQ-CP"]["f"], 200)
        self.assertIn("hopper_custom", hopper["extra"])
        self.assertIn("hopper_resv17", hopper["missing"])
        self.assertIsNone(hopper["pool"])


if __name__ == "__main__":
    unittest.main()
