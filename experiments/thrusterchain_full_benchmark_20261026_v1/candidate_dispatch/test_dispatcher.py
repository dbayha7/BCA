"""Bounded CPU checks for the disconnected draft dispatcher."""
import copy
import unittest
from unittest.mock import patch

from candidate_dispatch import dispatcher as d
from host_integration.cql_adapter import CQLAdapter
from host_integration.td3bc_adapter import TD3BCAdapter
from thrusterchain.quality_datasets import generate_behavior_dataset
from thrusterchain.training_protocol import OfflineBudget
from thrusterchain.training_runner import train_offline


class DispatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_behavior_dataset(behavior="mixed_random_strong", episodes=10,
                                               seed=923, provenance="smoke")
        cls.frozen = d.binding()

    def test_native_routes_resume_lineage_and_tamper(self):
        for method in d.ROUTES:
            with self.subTest(method=method):
                cp1, receipt1 = d.train_chunk(self.dataset, method, 924, 8, 2, self.frozen)
                cp2, receipt2 = d.train_chunk(self.dataset, method, 924, 8, 2, self.frozen, cp1, receipt1)
                cp4, _ = d.train_chunk(self.dataset, method, 924, 8, 4, self.frozen)
                self.assertEqual(cp2.to_bytes(), cp4.to_bytes())
                self.assertEqual(cp2.metadata["runner_state"]["total_runner_updates"],
                                 2 if method == "unifloral-unified" else 4)
                tampered = copy.deepcopy(receipt2)
                tampered["checkpoint_sha256"] = "0" * 64
                with self.assertRaises(RuntimeError):
                    d.verify_published_receipt(tampered, cp2, cp1, self.dataset, method, self.frozen,
                                               d._steps(method, 2), 8)
                with self.assertRaises(RuntimeError):
                    d.verify_published_receipt(receipt2, cp2, None, self.dataset, method, self.frozen,
                                               d._steps(method, 2), 8)
                if method == "cql":
                    changed_config = copy.deepcopy(receipt2)
                    native = changed_config["native_receipt"]
                    native["config"]["seed"] += 1
                    native["config_sha256"] = d._hash(native["config"])
                    native["receipt_sha256"] = d._hash({k: v for k, v in native.items()
                                                         if k != "receipt_sha256"})
                    changed_config["receipt_sha256"] = d._hash({k: v for k, v in changed_config.items()
                                                                if k != "receipt_sha256"})
                    with self.assertRaises(RuntimeError):
                        d.verify_published_receipt(changed_config, cp2, cp1, self.dataset,
                                                   method, self.frozen, 2, 8)
                    forged_previous = copy.deepcopy(receipt1)
                    prior_native = forged_previous["native_receipt"]
                    prior_native["config"]["seed"] += 1
                    prior_native["config_sha256"] = d._hash(prior_native["config"])
                    prior_native["receipt_sha256"] = d._hash({k: v for k, v in prior_native.items()
                                                               if k != "receipt_sha256"})
                    forged_previous["receipt_sha256"] = d._hash({k: v for k, v in forged_previous.items()
                                                                  if k != "receipt_sha256"})
                    with self.assertRaises(RuntimeError):
                        d.train_chunk(self.dataset, method, 924, 8, 2, self.frozen, cp1, forged_previous)
                self.assertEqual(receipt1["checkpoint_sha256"], cp1.digest)

    def test_reject_native_and_cross_route_resume(self):
        for method, factory in (("td3bc", TD3BCAdapter), ("cql", CQLAdapter)):
            with self.subTest(method=method):
                native, nr = train_offline(self.dataset, factory, budget=OfflineBudget(2, 8), seed=924)
                with self.assertRaises(RuntimeError):
                    d.train_chunk(self.dataset, method, 924, 8, 2, self.frozen, native, nr)
                compiled, cr = d.scan.train_chunk(self.dataset, method, 924, 8, 2)
                with self.assertRaises(RuntimeError):
                    d.train_chunk(self.dataset, method, 924, 8, 2, self.frozen, compiled, cr)
        td3, td3r = d.train_chunk(self.dataset, "td3bc", 924, 8, 2, self.frozen)
        with self.assertRaises(RuntimeError):
            d.train_chunk(self.dataset, "cql", 924, 8, 2, self.frozen, td3, td3r)
        cql, cqlr = d.train_chunk(self.dataset, "cql", 924, 8, 2, self.frozen)
        with self.assertRaises(RuntimeError):
            d.train_chunk(self.dataset, "td3bc", 924, 8, 2, self.frozen, cql, cqlr)
        for method, checkpoint, receipt, factory in (("td3bc", td3, td3r, TD3BCAdapter),
                                                       ("cql", cql, cqlr, CQLAdapter)):
            with self.subTest(method=method):
                with self.assertRaises(RuntimeError):
                    d.train_chunk(self.dataset, method, 925, 8, 2, self.frozen, checkpoint, receipt)
                with self.assertRaises(RuntimeError):
                    d.train_chunk(self.dataset, method, 924, 16, 2, self.frozen, checkpoint, receipt)
                with self.assertRaises(RuntimeError):
                    d.train_chunk(self.dataset, method, 924, 8, 2, self.frozen, checkpoint)
                # Native adapters may restore candidate weights for frozen evaluation.
                evaluation = factory(); evaluation.restore(checkpoint)
                self.assertEqual(evaluation.checkpoint().to_bytes(), checkpoint.to_bytes())
                # The base runner can continue that checkpoint and retains adapter
                # metadata, but erases the dispatcher marker. Refuse re-entry.
                mixed, mixed_receipt = train_offline(self.dataset, factory, budget=OfflineBudget(2, 8),
                                                      seed=924, resume_from=checkpoint)
                self.assertNotIn("engine", mixed.metadata["runner_state"])
                with self.assertRaises(RuntimeError):
                    d.train_chunk(self.dataset, method, 924, 8, 2, self.frozen, mixed, mixed_receipt)

    def test_frozen_routes_source_and_runtime_must_match(self):
        altered = copy.deepcopy(self.frozen)
        altered["routes"]["td3bc"] = "native"
        with self.assertRaises(RuntimeError):
            d.train_chunk(self.dataset, "td3bc", 924, 8, 2, altered)
        with patch.dict("os.environ", {"XLA_FLAGS": "--simulated-runtime-drift"}):
            with self.assertRaises(RuntimeError):
                d.train_chunk(self.dataset, "td3bc", 924, 8, 2, self.frozen)
        with self.assertRaises(ValueError):
            d._steps("unifloral-unified", 3)
        with self.assertRaises(ValueError):
            d._steps("native", 2)


if __name__ == "__main__":
    unittest.main()
