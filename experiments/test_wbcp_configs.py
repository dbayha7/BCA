"""Known-answer checks that every config resolves to the uniform-WBCP first stage; no simulator or data."""
import unittest
from copy import deepcopy
from unittest.mock import patch

from calibration.reference import WBCPConfig
from runtime import config as C
from runtime.config import ROOT, resolve, typed


class WBCPConfigTests(unittest.TestCase):
    def test_all_four_configs_resolve_to_uniform_wbcp(self):
        for host in ("td3_bc", "rebrac", "cql", "iql"):
            with self.subTest(host=host):
                rows = [resolve(ROOT / f"configs/{host}.yaml", m, 202609171, "/tmp/not-executed", "hopper")
                        for m in ("host", "bca")]
                self.assertEqual((rows[0]["calibration"], rows[1]["calibration"]), ("not_applicable", "wbcp_uniform"))
                self.assertIn("bca-wbcp", rows[1]["run_id"])
                objects = typed(rows[1])
                if host == "iql":
                    driver, module, options = objects
                    args, variants, arms, _ = driver.resolved_arguments(options, module)
                    self.assertEqual([(v.name, v.weight_width) for v in variants], [("wbcp_uniform", True)])
                    self.assertEqual([a.name for a in arms], ["host", "bca"])
                    self.assertTrue(all(a.beta == args.beta and a.gain == 1 for a in arms))
                    self.assertEqual(rows[0]["options"]["host_parameters"], rows[1]["options"]["host_parameters"])
                    self.assertEqual(args.posterior.config(), WBCPConfig(0.1, 0.95, 1000))
                else:
                    runner, args, spec, protocol = objects
                    config = spec if host == "cql" else spec.config()
                    self.assertEqual((config.arm, config.posterior), ("bca", WBCPConfig(0.1, 0.95, 1000)))
                    self.assertFalse(hasattr(config, "iw"))
                    self.assertEqual(rows[0]["native_args"], rows[1]["native_args"])
                    self.assertEqual(len(protocol.refresh_steps if host == "cql" else protocol.refresh_events), 198)

    def test_every_dataset_passes_its_rows_per_episode_to_the_host(self):
        for host in ("td3_bc", "rebrac", "cql", "iql"):
            config = C.read_config(ROOT / f"configs/{host}.yaml")
            for dataset, entry in config["datasets"].items():
                with self.subTest(host=host, dataset=dataset):
                    row = resolve(ROOT / f"configs/{host}.yaml", "bca", 202609171, "/tmp/not-executed", dataset)
                    k = entry["reservation"]["rows_per_episode"]
                    if host == "iql":
                        passed = row["options"]["posterior_parameters"]["reserve_rows_per_episode"]
                    elif host == "cql":
                        passed = row["protocol"]["calibration_rows_per_episode"]
                    else:
                        passed = row["protocol"]["reservation"]["rows_per_episode"]
                    self.assertEqual(passed, k)
                    self.assertGreaterEqual(k, 5)

    def test_a_dataset_without_rows_per_episode_is_refused(self):
        path = str((ROOT / "configs/td3_bc.yaml").resolve())
        config = deepcopy(C._yaml(path))
        del config["datasets"]["hopper"]["reservation"]["rows_per_episode"]
        real = C._yaml

        def patched(name):
            return config if name == path else real(name)

        with patch.object(C, "_yaml", side_effect=patched), self.assertRaisesRegex(ValueError, "rows_per_episode"):
            resolve(ROOT / "configs/td3_bc.yaml", "bca", 202609171, "/tmp/not-executed", "hopper")

    def test_archived_weighting_switch_is_refused(self):
        experiment = dict(C._yaml(str(ROOT / "configs/experiment.yaml")), calibration_weighting="none")
        real = C._yaml

        def patched(path):
            return experiment if path.endswith("experiment.yaml") else real(path)

        with patch.object(C, "_yaml", side_effect=patched), self.assertRaises(ValueError):
            resolve(ROOT / "configs/td3_bc.yaml", "bca", 202609171, "/tmp/not-executed", "hopper")


if __name__ == "__main__":
    unittest.main()
