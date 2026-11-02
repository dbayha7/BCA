"""Saved-data/NumPy validation of the actual binding; no simulator/model runs."""
from collections import namedtuple
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
spec = importlib.util.spec_from_file_location("_path_a_binding_test", HERE / "binding.py")
B = importlib.util.module_from_spec(spec); sys.modules[spec.name] = B; spec.loader.exec_module(B)
Native = SimpleNamespace(Transition=namedtuple("Transition", "obs action reward next_obs done"))


class BindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = B.load_verified()
        cls.prepared = {condition: B.prepare(Native, condition, source=cls.source) for condition in ("honeypot", "bottleneck")}

    def test_exact_reviewed_condition_counts_and_masks(self):
        self.assertEqual(self.prepared["honeypot"].metadata["training_rows"], 55649)
        p = self.prepared["bottleneck"]
        self.assertEqual(p.metadata["training_rows"], 53669)
        self.assertEqual(p.metadata["reviewed_cut_evidence"]["intact_successful_fitting_trajectories"], 0)
        self.assertEqual(p.metadata["reviewed_cut_evidence"]["genuine_fitting_fragment_pairs"], 716)
        self.assertEqual(p.metadata["reviewed_cut_evidence"]["B25_retained_fitting_parents"], 93)
        self.assertEqual(p.metadata["reviewed_cut_evidence"]["C30_retained_fitting_parents"], 167)
        original_capsule = B.D.retained_mask(self.source.records, self.source.geometry, "bottleneck")
        self.assertFalse(np.array_equal(original_capsule, B.reviewed_mask(self.source, "bottleneck")))

    def test_actual_tuples_are_original_rows_next_values_and_dones(self):
        a = self.source.records
        for p in self.prepared.values():
            indices = np.searchsorted(a["row_ids"], p.source_row_ids)
            np.testing.assert_array_equal(a["row_ids"][indices], p.source_row_ids)
            for field, key in zip(p.train._fields, ("observations", "actions", "rewards", "next_observations", "terminals")):
                np.testing.assert_array_equal(getattr(p.train, field), a[key][indices].astype(np.float32))
            np.testing.assert_array_equal(p.source_times, a["times"][indices])
            self.assertFalse(p.metadata["next_action_used"])
            self.assertFalse(p.metadata["fragment_joining"])
            self.assertTrue(np.all(a["roles"][indices] == 0))

    def test_no_false_fragment_join_and_original_nonterminal_is_preserved(self):
        p = self.prepared["bottleneck"]
        gaps = np.flatnonzero((p.source_episode_ids[1:] == p.source_episode_ids[:-1]) &
                             (p.source_times[1:] != p.source_times[:-1] + 1))
        self.assertGreater(len(gaps), 700)
        self.assertTrue(np.all(p.source_fragment_ids[gaps + 1] != p.source_fragment_ids[gaps]))
        self.assertTrue(np.all(p.train.done[gaps] == 0))
        self.assertTrue(np.any(np.any(p.train.next_obs[gaps] != p.train.obs[gaps + 1], axis=1)))

    def test_calibration_is_fixed_128_disjoint_original_parents(self):
        a = self.source.records
        for condition, p in self.prepared.items():
            mask = B.reviewed_mask(self.source, condition)
            cal = np.flatnonzero(mask & (a["roles"] == 1))
            eligible = np.sort(np.unique(a["episode_ids"][cal]))
            rng = np.random.Generator(np.random.PCG64(121312521))
            selected = np.array([rng.choice(cal[a["episode_ids"][cal] == parent]) for parent in eligible[:128]])
            np.testing.assert_array_equal(p.heldout_ids["row_ids"], a["row_ids"][selected])
            self.assertEqual(len(np.unique(p.heldout_ids["episode_ids"])), 128)
            self.assertEqual(len(np.intersect1d(p.source_episode_ids, p.heldout_ids["episode_ids"])), 0)
            np.testing.assert_array_equal(p.heldout.obs, a["observations"][selected])

    def test_normalization_is_all_and_only_condition_fitting_rows(self):
        for p in self.prepared.values():
            self.assertIs(p.normalization_fixture, p.train)
            np.testing.assert_array_equal(p.normalization_source_row_ids, p.source_row_ids)
            self.assertEqual(p.metadata["normalization_source_row_ids_sha256"], p.metadata["training_row_ids_sha256"])
            self.assertFalse(p.train.obs.flags.writeable)
            with self.assertRaises(ValueError):
                p.train.obs.setflags(write=True)

    def test_honeypot_selection_matches_unchanged_original_adapter(self):
        s = self.source
        old = B.I.prepare_learner_data(Native, s.records, B.D.retained_mask(s.records, s.geometry, "honeypot"),
            geometry=s.geometry, condition="honeypot", calibration_seed=121312521,
            dataset_sha256=B.DATASET_SHA256, geometry_sha256=B.GEOMETRY_SHA256)
        new = self.prepared["honeypot"]
        for name in ("train", "heldout", "probe", "scale_bank", "normalization_fixture"):
            for left, right in zip(getattr(old, name), getattr(new, name)):
                np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(old.heldout_ids["row_ids"], new.heldout_ids["row_ids"])

    def test_altered_content_geometry_rule_or_seed_rejected(self):
        s = copy.copy(self.source)
        s.records = dict(s.records); s.records["rewards"] = s.records["rewards"].copy()
        s.records["rewards"][0] += 1
        with self.assertRaisesRegex(ValueError, "content"):
            B.prepare(Native, "bottleneck", source=s)
        s = copy.copy(self.source); s.geometry = copy.deepcopy(s.geometry); s.geometry["B"][0] += .01
        with self.assertRaisesRegex(ValueError, "Geometry"):
            B.prepare(Native, "bottleneck", source=s)
        p = copy.deepcopy(self.source.protocol); p["calibration_seed"] += 1
        with self.assertRaisesRegex(ValueError, "protocol or seeds"):
            B.prepare(Native, "bottleneck", protocol=p, source=self.source)
        with self.assertRaises(ValueError):
            B.prepare(Native, "broad", source=self.source)

    def test_actual_inputs_contain_no_intact_successful_bottleneck_parent(self):
        a, p = self.source.records, self.prepared["bottleneck"]
        retained_ids = set(p.source_row_ids.tolist())
        intact = pairs = 0
        qs, qn = B.C.progress(a["states"], self.source.geometry), B.C.progress(a["next_states"], self.source.geometry)
        for parent in np.unique(p.source_episode_ids):
            rows = np.flatnonzero(a["episode_ids"] == parent)
            success = bool(a["terminals"][rows[-1]] and a["rewards"][rows[-1]] > 0)
            keep = np.array([int(row) in retained_ids for row in a["row_ids"][rows]])
            intact += int(success and np.all(keep))
            removed = np.flatnonzero(~keep)
            if success and len(removed) and keep[0] and keep[-1]:
                prefix, suffix = rows[:removed[0]], rows[removed[-1] + 1:]
                pairs += int(np.all(np.maximum(qs[prefix], qn[prefix]) < .2) and
                             np.all(np.minimum(qs[suffix], qn[suffix]) > .8))
        self.assertEqual(intact, 0)
        self.assertEqual(pairs, 716)

    def test_full_state_support_from_actual_consumed_fitting_rows(self):
        a, p = self.source.records, self.prepared["bottleneck"]
        indices = np.searchsorted(a["row_ids"], p.source_row_ids)
        rules = self.source.protocol["setup_rules"]
        for name, time, expected in (("B", 25, 93), ("C", 30, 167)):
            delta = a["states"][indices] - np.asarray(self.source.geometry[name])
            near = p.source_times == time
            near &= np.linalg.norm(delta[:, :10], axis=1) <= rules["approach_physical_l2_radius"]
            near &= np.abs(delta[:, 10]) <= rules["approach_current_radius"]
            near &= np.linalg.norm(delta[:, 11:], axis=1) <= rules["approach_wave_l2_radius"]
            self.assertEqual(np.unique(p.source_episode_ids[near]).size, expected)


if __name__ == "__main__":
    unittest.main()
