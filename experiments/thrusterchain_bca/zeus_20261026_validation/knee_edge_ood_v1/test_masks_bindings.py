"""Masks, L1.4 known faults, calibration, reconciliation and the 4.4 binding assertions (synthetic arrays)."""
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

import numpy as np

import bindings as B
import fixtures as F
import masks as M
import reconcile as C
from knee_region import past_knee_a0

ARRAYS = F.synthetic_arrays()
ROWS = M.fitting_rows(ARRAYS)
BOUNDS = C.sealed_bounds()["bounds"]


def masks_with(bands=None, times=None):
    return M.arm_masks(ROWS["obs"], ROWS["actions"], ROWS["times"] if times is None else times,
                       ROWS["episode_ids"], seed=F.TEST_SEEDS["random_control"], bands=bands)[0]


def twin_arrays():
    """A no-stall twin of the synthetic stall data: states diverge exactly after each episode's first W row."""
    nostall = {k: np.array(v) for k, v in ARRAYS.items()}
    w = past_knee_a0(ARRAYS["actions"])[0]
    for e in range(len(w)):
        hits = np.flatnonzero(w[e])
        if len(hits):
            nostall["states"][e, hits[0] + 1:, :10] += 1.0
    return nostall


class Masks(unittest.TestCase):
    def test_correct_random_passes(self):
        masks = masks_with()
        self.assertFalse(np.any(past_knee_a0(ROWS["actions"][masks["cutoff"]])[0]))
        self.assertTrue(M.three_band_counts_equal(masks, ROWS["obs"], ROWS["actions"], ROWS["times"], ROWS["episode_ids"]))
        d = M.l14_density(masks, ROWS["actions"], ROWS["bands"])
        for name, key in (("pooled_high", "l14_pooled_high"), ("mid", "l14_mid"), ("deep", "l14_deep")):
            self.assertTrue(BOUNDS[key][0] <= d["random"][name] <= BOUNDS[key][1], (name, d["random"][name]))
        self.assertEqual(d["cutoff"], {"pooled_high": 0.0, "mid": 0.0, "deep": 0.0})

    def test_unstratified_random_fails_l14(self):
        cutoff = ~ROWS["past_knee"]
        unstrat = np.zeros(len(cutoff), bool)
        unstrat[np.random.Generator(np.random.PCG64(21)).choice(len(cutoff), int(cutoff.sum()), replace=False)] = True
        d = M.l14_density({"cutoff": cutoff, "random": unstrat}, ROWS["actions"], ROWS["bands"])["random"]["pooled_high"]
        self.assertGreater(d, BOUNDS["l14_pooled_high"][1])

    def test_band_blind_random_fails_l14(self):
        blind = masks_with(bands=np.zeros(len(ROWS["bands"]), np.int64))
        self.assertGreater(M.l14_density(blind, ROWS["actions"], ROWS["bands"])["random"]["pooled_high"],
                           BOUNDS["l14_pooled_high"][1])

    def test_two_level_flag_fails_three_band_count_check(self):
        binary = masks_with(bands=(ROWS["bands"] >= 1).astype(np.int64))
        self.assertFalse(M.three_band_counts_equal(binary, ROWS["obs"], ROWS["actions"], ROWS["times"], ROWS["episode_ids"]))

    def test_wrong_time_index_fails_three_band_count_check(self):
        shifted = masks_with(times=(ROWS["times"] + 5) % 100)
        self.assertFalse(M.three_band_counts_equal(shifted, ROWS["obs"], ROWS["actions"], ROWS["times"], ROWS["episode_ids"]))

    def test_mask_built_on_requested_commands_raises(self):
        broken = dict(ARRAYS, actions=ARRAYS["requested_actions"])      # a builder handed requested float64 commands
        with self.assertRaises(ValueError):
            M.fitting_rows(broken)

    def test_calibration(self):
        cal = M.calibration_rows(ARRAYS, seed=F.TEST_SEEDS["calibration_original"],
                                 replacement_seed=F.TEST_SEEDS["calibration_replacement"])
        self.assertEqual(len(cal["local_rows"]), 128)
        self.assertTrue(BOUNDS["calibration_replacements"][0] <= cal["replacements"] <= BOUNDS["calibration_replacements"][1])
        actions = ARRAYS["actions"].copy()
        actions[cal["episodes"][0], :, 0] = np.float32(-1.0)
        with self.assertRaises(ValueError):
            M.calibration_rows(dict(ARRAYS, actions=actions), seed=12, replacement_seed=13)


class Reconcile(unittest.TestCase):
    def datasets(self, nostall=None):
        return {"stall3": F.SyntheticDataset(ARRAYS, "stall3"), "nostall": F.SyntheticDataset(nostall or twin_arrays(), "nostall")}

    def test_twins_pass(self):
        result = C.reconcile(self.datasets(), test_seeds=F.TEST_SEEDS)
        self.assertEqual(result["label"], "PASS", result["cross_world"])
        self.assertGreater(result["worlds"]["stall3"]["plumbing_report"]["terminal_rows"]["cutoff"], 0)

    def test_pre_w_state_perturbation_is_incomplete(self):
        nostall = twin_arrays()
        nostall["states"][0, 0, 0] += 1e-9
        self.assertEqual(C.reconcile(self.datasets(nostall), test_seeds=F.TEST_SEEDS)["label"], "INCOMPLETE")

    def test_mask_difference_across_worlds_is_incomplete(self):
        nostall = twin_arrays()
        e = int(np.flatnonzero(ARRAYS["split_codes"] == 0)[0])
        nostall["actions"] = ARRAYS["actions"].copy()
        nostall["actions"][e, 3, 0] = np.float32(-0.95)
        self.assertEqual(C.reconcile(self.datasets(nostall), test_seeds=F.TEST_SEEDS)["label"], "INCOMPLETE")

    def test_bound_violation_is_inconclusive(self):
        bounds = C.sealed_bounds()
        bounds = dict(bounds, bounds=dict(bounds["bounds"], share_high=[0.0, 0.1]))
        self.assertEqual(C.reconcile(self.datasets(), test_seeds=F.TEST_SEEDS, bounds=bounds)["label"], "INCONCLUSIVE")

    def test_world_label_mismatch_raises(self):
        with self.assertRaises(ValueError):
            C.reconcile({"stall3": F.SyntheticDataset(ARRAYS, "nostall"), "nostall": F.SyntheticDataset(ARRAYS, "nostall")},
                        test_seeds=F.TEST_SEEDS)


class Bindings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = F.SyntheticDataset(ARRAYS, "stall3")
        cls.plan = B.plan_streams(cls.dataset, "stall3", test_seeds=F.TEST_SEEDS)

    def test_receipts_trace_and_zero_w(self):
        for arm in B.ARMS:
            receipt = self.plan["plans"][arm]["receipt"]
            self.assertEqual(receipt["dataset_sha256"], self.dataset.digest)
            for name in B.STREAMS:
                B.assert_stream(self.plan["plans"][arm]["streams"][name], receipt["streams"][name], arm, name)
                B.trace(self.plan["plans"][arm]["streams"][name], ARRAYS, self.plan["plans"][arm]["sources"][name])
        self.assertEqual(self.plan["plans"]["cutoff"]["receipt"]["streams"]["train"]["w_rows"], 0)
        self.assertGreater(self.plan["plans"]["random"]["receipt"]["streams"]["train"]["w_rows"], 0)

    def test_builder_offsetting_next_obs_is_caught(self):
        real = B._rows

        def offset(arrays, episodes, local_rows):
            out = real(arrays, episodes, local_rows)
            out["next_obs"] = out["obs"].copy()
            return out
        with mock.patch.object(B, "_rows", offset):
            with self.assertRaises(ValueError):
                B.plan_streams(self.dataset, "stall3", test_seeds=F.TEST_SEEDS)

    def test_random_stream_bound_as_cutoff_is_caught(self):
        with self.assertRaises(ValueError):
            B.assert_stream(self.plan["plans"]["random"]["streams"]["train"],
                            self.plan["plans"]["random"]["receipt"]["streams"]["train"], "cutoff", "train")

    def test_dataset_of_other_world_raises(self):
        with self.assertRaises(ValueError):
            B.plan_streams(F.SyntheticDataset(ARRAYS, "nostall"), "stall3", test_seeds=F.TEST_SEEDS)

    def test_adapter_checked_against_independent_expectation(self):
        expected = B.expectation(self.plan, "cutoff")
        other = dict(expected, dataset_sha256="another-dataset")

        class Native:
            @staticmethod
            def Transition(**fields):
                return type("T", (), fields)

        data = B.provide(Native, self.plan, "cutoff")
        adapter = type("A", (), {"context": type("C", (), {"data": data, "train_transition": data.train,
                                                           "heldout_transition": data.heldout, "probe": data.probe})})
        self.assertEqual(len(B.assert_adapter(adapter, expected)), 5)
        with self.assertRaises(ValueError):
            B.assert_adapter(adapter, other)
        with self.assertRaises(ValueError):
            B.assert_adapter(adapter, B.expectation(self.plan, "random"))

    def test_research_label_needs_a_validated_dataset(self):
        fake = F.SyntheticDataset(ARRAYS, "stall3")
        fake.metadata = dict(fake.metadata, provenance="research")
        with self.assertRaises(ValueError):
            B.plan_streams(fake, "stall3")
        with self.assertRaises(ValueError):
            C.reconcile({"stall3": fake, "nostall": F.SyntheticDataset(twin_arrays(), "nostall")})

    def test_protected_rows_counted_from_sources(self):
        self.assertEqual(self.plan["plans"]["cutoff"]["receipt"]["protected_transition_rows_bound_to_learner"], 0)
        codes = ARRAYS["split_codes"]
        for arm in B.ARMS:
            for src in self.plan["plans"][arm]["sources"].values():
                self.assertFalse(np.isin(codes[src[:, 0]], (2, 3)).any())

    def test_sealed_seeds_refused_for_smoke(self):
        import seeds as S
        sealed = {k: S.load()["numpy"][k] for k in S.DATA_NAMES}
        with self.assertRaises(ValueError):
            B.plan_streams(self.dataset, "stall3", test_seeds=sealed)


if __name__ == "__main__":
    unittest.main()
