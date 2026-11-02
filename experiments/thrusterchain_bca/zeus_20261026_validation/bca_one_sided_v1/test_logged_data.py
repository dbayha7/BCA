"""Synthetic subset/leakage fixtures plus parity of permitted logged arrays."""
import io
import json
import unittest
import zipfile

import numpy as np

import logged_data as data


class SubsetFixtures(unittest.TestCase):
    def test_protected_episode_refused_before_archive_access(self):
        codes = np.zeros(1280, dtype=np.uint8)
        codes[3], codes[4] = 2, 3
        for index in (3, 4):
            with self.assertRaisesRegex(ValueError, "Forbidden"):
                data.read_episodes(None, "observations", [index], codes)

    def test_duplicates_and_out_of_range_refused(self):
        codes = np.zeros(1280, dtype=np.uint8)
        for ids in ([1, 1], [-1], [1280]):
            with self.assertRaises(ValueError):
                data.read_episodes(None, "observations", ids, codes)

    def test_unapproved_member_refused(self):
        with self.assertRaisesRegex(ValueError, "Unapproved"):
            data.read_episodes(None, "privileged_values", [0], np.zeros(1280, dtype=np.uint8))

    def test_requested_order_preserved_while_protected_record_skipped(self):
        array = np.arange(1280 * 6, dtype=np.float32).reshape(1280, 3, 2)
        member = io.BytesIO()
        np.save(member, array, allow_pickle=False)
        container = io.BytesIO()
        with zipfile.ZipFile(container, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("observations.npy", member.getvalue())
        codes = np.zeros(1280, dtype=np.uint8)
        codes[1] = 3
        container.seek(0)
        with zipfile.ZipFile(container) as archive:
            result = data.read_episodes(archive, "observations", [2, 0], codes)
        np.testing.assert_array_equal(result, array[[2, 0]])


class ActualPermittedData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.banks = data.load_logged_banks()

    def test_episode_disjoint_and_exact_counts(self):
        b = self.banks
        self.assertEqual(b["receipt"]["rows"], {"fit": 256, "calibration": 128, "validation": 128})
        self.assertEqual(b["receipt"]["unique_groups"], {"fit": 221, "calibration": 128, "validation": 128})
        for first, second in (("fit", "calibration"), ("fit", "validation"), ("calibration", "validation")):
            self.assertFalse(set(b[first]["group_ids"]) & set(b[second]["group_ids"]))

    def test_fitting_bank_matches_original_export_exactly(self):
        path = data.PAYLOAD / "unifloral_export_recovery_v2/recovery_20261023T214750_256386Z/inference_export/scale_bank.npz"
        with np.load(path, allow_pickle=False) as archive:
            for observed, key in (("obs", "observations"), ("action", "actions"), ("group_ids", "group_ids")):
                np.testing.assert_array_equal(self.banks["fit"][observed], archive[key])
                self.assertEqual(self.banks["fit"][observed].dtype, archive[key].dtype)

    def test_calibration_ids_and_times_match_committed_binding(self):
        binding = json.loads(data.BINDINGS.read_text(encoding="utf-8"))["datasetidentity"]["split_receipt"]
        self.assertEqual(self.banks["calibration"]["row_ids"].tolist(), binding["calibration_row_ids"])
        self.assertEqual(self.banks["receipt"]["calibration_times"], binding["calibration_times"])
        self.assertEqual(self.banks["receipt"]["selection_and_final_calibration_transition_rows_decoded"], 0)

    def test_scalar_fields_are_rows_and_banks_are_readonly(self):
        for role in ("fit", "calibration", "validation"):
            bank = self.banks[role]
            n = len(bank["obs"])
            for name in ("reward", "done", "group_ids", "row_ids"):
                self.assertEqual(bank[name].shape, (n,))
            for value in bank.values():
                self.assertFalse(value.flags.writeable)


if __name__ == "__main__":
    unittest.main()
