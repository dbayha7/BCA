"""Regression evidence for the corrected controllability and terminal claims."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from thrusterchain.environment import ThrusterChainEnv, matrices


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/smoke_v2/diagnostic_results.json"
RAW_SHA256 = "46da6e139844a0545c7ec81570e63a1d7e0b5eec344f32ce9c5a2231cc7c2cd6"
TOLERANCES = (1e-14, 1e-12, 1e-10, 1e-8, 1e-6)


class ControllabilityClaimTests(unittest.TestCase):
    def test_augmented_rank_ten_actuated_rank_ten(self):
        a, b = matrices()
        for label, aa, bb, shape in (
            ("augmented", a, b, (13, 52)),
            ("actuated", a[:10, :10], b[:10], (10, 40)),
        ):
            # Iterative Krylov construction, separate from the receipt's powers.
            blocks, block = [], bb.copy()
            for _ in range(aa.shape[0]):
                blocks.append(block)
                block = aa @ block
            reach = np.hstack(blocks)
            self.assertEqual(reach.shape, shape)
            self.assertEqual(np.linalg.matrix_rank(reach), 10)
            for tolerance in TOLERANCES:
                with self.subTest(system=label, absolute_tolerance=tolerance):
                    self.assertEqual(np.linalg.matrix_rank(reach, tol=tolerance), 10)
            self.assertGreater(np.linalg.svd(reach, compute_uv=False)[9], .09)
            if label == "augmented":
                np.testing.assert_array_equal(reach[10:], np.zeros((3, 52)))

    def test_exogenous_coordinates_cannot_be_actuated(self):
        a, b = matrices()
        np.testing.assert_array_equal(b[10:], np.zeros((3, 4)))
        np.testing.assert_array_equal(a[10:, :10], np.zeros((3, 10)))
        self.assertEqual(a[10, 10], 1.)
        angle = .27
        np.testing.assert_allclose(
            a[11:, 11:], [[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]], rtol=0, atol=1e-15)
        initial = np.r_[np.linspace(-.4, .4, 10), .6, .3, -.2]
        for stall in (False, True):
            for slope in (1.5, 3.):
                first = ThrusterChainEnv(stall=stall, stall_slope=slope)
                second = ThrusterChainEnv(stall=stall, stall_slope=slope)
                first.reset(options={"state": initial})
                second.reset(options={"state": initial})
                for t in range(100):
                    first.step(np.zeros(4))
                    second.step(np.full(4, 1. if t % 2 else -1.))
                    np.testing.assert_array_equal(first.state[10:], second.state[10:])
                    self.assertEqual(first.state[10], initial[10])
                    np.testing.assert_allclose(np.linalg.norm(first.state[11:]),
                                               np.linalg.norm(initial[11:]),
                                               rtol=0, atol=2e-14)

    def test_original_dynamics_sources_and_fixtures_unchanged(self):
        expected = {
            "thrusterchain/environment.py": "449207098042426ff37a7917a2cca5c8c35f8abdd633badb7b1b3abf0ef8d874",
            "thrusterchain/canonical/__init__.py": "c9193db165aea861e243d8b599d0bacb62a003f455c696c68d3caa7e4b229fc8",
            "thrusterchain/canonical/thrusterchain.py": "f22f2e214ad9231bee4ce60ef55e64409534c2654c4e184406553ba8ed67dde1",
            "thrusterchain/canonical/gapped_lq.py": "f8ec7cc6c7d9d013d3e509cd3aba0a6dedf1f371480b4e1977f51e2b5dd923aa",
            "thrusterchain/canonical/plant.json": "4aa14e6eba522074a72755c06cb03318aee4ba6799ab57ea2fa02e9b77fca735",
            "contracts/dynamics_fixtures.json": "bf51b18e08cdcd003ca79fe70fcb1fe4ae73d6a22eb468d9684985716a32bd8f",
        }
        for name, digest in expected.items():
            with self.subTest(path=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)
        a, b = matrices()
        self.assertEqual(hashlib.sha256(a.astype("<f8").tobytes()
                                       + b.astype("<f8").tobytes()).hexdigest(),
                         "289afb581b4fdbc97fd68566c0b42ff6244b20686e9e49ba177e7e6bd57454a1")

    def test_machine_contract_does_not_claim_full_augmented_control(self):
        contract = json.loads((ROOT / "contracts/environment_contract.json").read_text())
        claim = contract["dynamics"]["controllability"]
        self.assertEqual(claim["augmented"]["rank"], 10)
        self.assertEqual(claim["augmented"]["state_dimension"], 13)
        self.assertFalse(claim["augmented"]["fully_controllable"])
        self.assertEqual(claim["actuated_subsystem"]["rank"], 10)
        self.assertEqual(claim["actuated_subsystem"]["state_dimension"], 10)
        self.assertEqual(claim["absolute_rank_tolerances"], list(TOLERANCES))
        self.assertFalse(claim["dynamics_changed"])


class ArchivedTerminalClaimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_bytes = RAW.read_bytes()
        cls.data = json.loads(cls.raw_bytes)

    def test_archived_result_identity_and_target(self):
        self.assertEqual(hashlib.sha256(self.raw_bytes).hexdigest(), RAW_SHA256)
        self.assertEqual(len(self.raw_bytes), 40505876)
        self.assertEqual(self.data["schema"], "thrusterchain-terminal-nine-cell-smoke-v3")
        self.assertEqual(self.data["target"],
                         "Q_99=r for finite horizon100, continuation0; fixed broad/broad query law")
        self.assertFalse(self.data["guarantee_claimed"])
        self.assertFalse(self.data["closed_loop_occupancy_claim"])

    def test_all_6912_choices_have_zero_action_and_recomputed_regret(self):
        counts = {arm: [0, 0, 0] for arm in self.data["four_arms"]}
        states_count, bank_rewards_count = 0, 0
        self.assertEqual(len(self.data["cells"]), 9)
        for cell in self.data["cells"]:
            self.assertEqual(len(cell["repetitions"]), 3)
            for repetition in cell["repetitions"]:
                raw = repetition["raw"]
                bank = raw["paired_action_bank"]
                self.assertEqual(bank["time_index"], 99)
                states = np.asarray(raw["evaluation"]["states"])[bank["state_rows_from_evaluation"]]
                requested = np.asarray(bank["actions"])
                executed = np.asarray(bank["executed_actions"])
                rewards = np.asarray(bank["true_reward"])
                self.assertEqual(requested.shape, (64, 8, 4))
                np.testing.assert_array_equal(executed, requested.astype(np.float32).astype(np.float64))
                np.testing.assert_array_equal(executed[:, 0], np.zeros((64, 4)))
                # Independently evaluate the algebraic reward in its specified order.
                state_cost = states[:, 0] ** 2
                for j in range(1, 10):
                    state_cost = state_cost + states[:, j] ** 2
                action_cost = executed[..., 0] ** 2
                for j in range(1, 4):
                    action_cost = action_cost + executed[..., j] ** 2
                expected_rewards = -(state_cost[:, None] + .3 * action_cost)
                np.testing.assert_array_equal(rewards, expected_rewards)
                states_count += len(states)
                bank_rewards_count += rewards.size
                for arm, record in bank["arms"].items():
                    with self.subTest(cell=cell["cell"], repetition=repetition["repetition"], arm=arm):
                        indices = np.asarray(record["selected_index"])
                        selected = np.asarray(record["selected_reward"])
                        np.testing.assert_array_equal(indices, np.zeros(64, dtype=int))
                        np.testing.assert_array_equal(record["selected_actions"], np.zeros((64, 4)))
                        np.testing.assert_array_equal(selected, rewards[np.arange(64), indices])
                        np.testing.assert_array_equal(rewards[:, 0] - selected, np.zeros(64))
                        np.testing.assert_array_equal(record["regret_to_zero"], np.zeros(64))
                        np.testing.assert_array_equal(record["fallback_executed"],
                                                      np.logical_not(record["accepted"]))
                        counts[arm][0] += len(indices)
                        counts[arm][1] += int(np.count_nonzero(record["accepted"]))
                        counts[arm][2] += int(np.count_nonzero(record["fallback_executed"]))
        self.assertEqual(states_count, 1728)
        self.assertEqual(bank_rewards_count, 13824)
        self.assertEqual(sum(v[0] for v in counts.values()), 6912)
        self.assertEqual(counts, {
            "unweighted_standard": [1728, 768, 960],
            "unweighted_most_caution": [1728, 768, 960],
            "weighted_standard": [1728, 419, 1309],
            "weighted_most_caution": [1728, 419, 1309],
        })


if __name__ == "__main__":
    unittest.main()
