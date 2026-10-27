"""Receipt contradictions must fail even when duplicate receipt copies agree."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from audit_benchmark import artifact_checks, load_json
from thrusterchain.datasets import load_dataset
from thrusterchain.identification_validation import REQUIRED, validate_identification_receipt

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "results" / "revision_v2_smoke"


class IdentificationReceiptRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset(ORIGINAL / "offline_dataset.npz")
        cls.reference = load_json(ORIGINAL / "reference_freeze_receipts.json")["identified_model_mpc"]
        cls.results = load_json(ORIGINAL / "benchmark_results.json")
        cls.references = load_json(ORIGINAL / "reference_freeze_receipts.json")
        cls.registered = load_json(ORIGINAL / "source_manifest_after.json")
        with np.load(ORIGINAL / "identified_model.npz", allow_pickle=False) as model:
            cls.A, cls.B = model["A"], model["B"]

    def setUp(self):
        self.receipt = copy.deepcopy(self.reference)

    def check(self):
        return validate_identification_receipt(self.receipt, self.dataset, self.A, self.B)

    def check_full_artifact(self, embedded_receipt=None):
        # This source substitution belongs only to historical-fixture tests.
        # Production artifact_checks has no switch to bypass source matching.
        results, references = copy.deepcopy(self.results), copy.deepcopy(self.references)
        results["baselines"]["identified_model_mpc"]["receipt"] = copy.deepcopy(
            self.receipt if embedded_receipt is None else embedded_receipt)
        references["identified_model_mpc"] = copy.deepcopy(self.receipt)

        def fixture_loader(path):
            if Path(path).name == "benchmark_results.json":
                return results
            if Path(path).name == "reference_freeze_receipts.json":
                return references
            return load_json(path)

        with patch("audit_benchmark.source_manifest", return_value=self.registered), \
                patch("audit_benchmark.load_json", side_effect=fixture_loader):
            return artifact_checks(ORIGINAL)

    def reject_nested(self, field, value, *, full=False):
        self.receipt["identification"][field] = value
        with self.assertRaisesRegex(AssertionError, "identification receipt"):
            self.check_full_artifact() if full else self.check()

    def test_original_v2_receipt_passes_full_artifact_validation(self):
        evidence = self.check_full_artifact()["identification_receipt_validation"]
        self.assertEqual(evidence["matrix_sha256"], self.reference["model_hash"])
        self.assertEqual(evidence["dataset_sha256"], self.dataset.digest)
        self.assertFalse(evidence["training_provenance_verified"])
        self.assertIn("not authenticated", evidence["scope"])

    def test_archived_v1_receipt_passes_registered_snapshot_artifact_validation(self):
        archived = ROOT / "results" / "smoke_v2"
        snapshot = load_json(archived / "source_manifest_after.json")
        with patch("audit_benchmark.source_manifest", return_value=snapshot):
            self.assertEqual(artifact_checks(archived)["identification_receipt_validation"]["rank"], 17)

    def test_synchronized_missing_identification_is_rejected(self):
        self.receipt.pop("identification")
        with self.assertRaisesRegex(AssertionError, "identification receipt"):
            self.check_full_artifact()

    def test_synchronized_wrong_matrix_hash_is_rejected(self):
        self.reject_nested("matrix_hash", "0" * 64, full=True)

    def test_synchronized_wrong_dataset_hash_is_rejected(self):
        self.reject_nested("dataset_hash", "f" * 64, full=True)

    def test_synchronized_empty_training_groups_are_rejected(self):
        self.reject_nested("training_groups", [], full=True)

    def test_synchronized_nonexistent_training_groups_are_rejected(self):
        self.reject_nested("training_groups", [987654321], full=True)

    def test_synchronized_final_calibration_fit_is_rejected(self):
        self.reject_nested("fit_uses", "final_calibration split only", full=True)

    def test_synchronized_privileged_kind_is_rejected(self):
        self.reject_nested("kind", "privileged_true_model", full=True)

    def test_embedded_receipt_numeric_type_aliases_do_not_equal_the_reference(self):
        for change in (lambda r: r["identification"]["training_groups"].__setitem__(0, False),
                       lambda r: r["identification"].__setitem__("rank", 17.0),
                       lambda r: r.__setitem__("training_provenance_verified", 0)):
            embedded = copy.deepcopy(self.receipt)
            change(embedded)
            with self.assertRaisesRegex(AssertionError, "reference result receipt mismatch"):
                self.check_full_artifact(embedded_receipt=embedded)

    def test_training_group_order_is_irrelevant_but_membership_is_exact(self):
        groups = self.reference["identification"]["training_groups"]
        self.receipt["identification"]["training_groups"] = groups[::-1]
        self.assertEqual(self.check_full_artifact()["identification_receipt_validation"]["training_group_count"], len(groups))
        other = int(self.dataset.split("final_calibration")["group_ids"][0])
        for value in (groups[:-1], groups + [groups[0]], groups + [other], [other] + groups[1:]):
            with self.subTest(groups=value):
                self.reject_nested("training_groups", value)

    def test_bool_float_string_and_non_list_groups_are_rejected(self):
        groups = self.reference["identification"]["training_groups"]
        for value in ([False] + groups[1:], [float(x) for x in groups], [str(x) for x in groups],
                      None, "0,1", {"group": 0}, tuple(groups)):
            with self.subTest(value=value):
                self.reject_nested("training_groups", value)

    def test_identification_object_and_all_required_fields_are_required(self):
        for value in (None, False, [], "train only"):
            with self.subTest(value=value):
                self.receipt["identification"] = value
                with self.assertRaisesRegex(AssertionError, "identification must be an object"):
                    self.check()
        for field in sorted(REQUIRED):
            with self.subTest(field=field):
                self.receipt = copy.deepcopy(self.reference)
                del self.receipt["identification"][field]
                with self.assertRaisesRegex(AssertionError, "missing fields"):
                    self.check()

    def test_hash_type_and_protocol_claims_cannot_be_coerced(self):
        for field, values in {
            "dataset_hash": [None, 0, [], self.reference["model_hash"]],
            "matrix_hash": [None, False, [], self.dataset.digest],
            "feature_scaling": [None, "standardized"],
            "state_access": [None, "exact info states"],
            "known_structure": [None, [], "all known"],
        }.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    self.receipt = copy.deepcopy(self.reference)
                    self.reject_nested(field, value)

    def test_rank_and_regularization_contract(self):
        for field, values in {"rank": [False, 17.0, "17", 16, 18, None],
                              "required_rank": [True, 17.0, "17", 16, None],
                              "regularization": [False, None, "0", -1, .1]}.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    self.receipt = copy.deepcopy(self.reference)
                    self.reject_nested(field, value)

    def test_diagnostics_are_numeric_finite_and_bound_to_loaded_artifacts(self):
        for field in ("condition_number", "training_rmse"):
            for value in (None, False, "1", float("nan"), float("inf"), -1.0, 10**400):
                with self.subTest(field=field, value=str(value)[:30]):
                    self.receipt = copy.deepcopy(self.reference)
                    self.reject_nested(field, value)
        for field, value in (("condition_number", 2.0), ("training_rmse", .1)):
            with self.subTest(field=field):
                self.receipt = copy.deepcopy(self.reference)
                self.reject_nested(field, value, full=True)
        self.receipt = copy.deepcopy(self.reference)
        self.receipt["identification"]["singular_values"] = [2 * x for x in self.reference["identification"]["singular_values"]]
        # The condition ratio is unchanged; actual train features still bind SVD.
        with self.assertRaisesRegex(AssertionError, "loaded train features"):
            self.check()

    def test_singular_value_count_order_sign_and_types(self):
        values = self.reference["identification"]["singular_values"]
        for value in (None, {}, values[:-1], values[::-1], [False] + values[1:],
                      [str(values[0])] + values[1:], values[:-1] + [0.0],
                      values[:-1] + [-1.0], values[:-1] + [float("nan")]):
            with self.subTest(value=value):
                self.receipt = copy.deepcopy(self.reference)
                self.reject_nested("singular_values", value)

    def test_no_provenance_or_performance_attestation_is_granted(self):
        for value in (True, 0, "false", None):
            with self.subTest(value=value):
                self.receipt = copy.deepcopy(self.reference)
                self.reject_nested("closed_loop_performance_guarantee", value)
        for field, value in (("training_provenance_verified", True),
                              ("training_provenance_verified", 0),
                              ("matrix_source", "environment.matrices"),
                              ("declared_provenance", "privileged_true_model"),
                              ("provenance", "privileged_true_model")):
            with self.subTest(field=field, value=value):
                self.receipt = copy.deepcopy(self.reference)
                self.receipt[field] = value
                with self.assertRaisesRegex(AssertionError, "identification receipt"):
                    self.check()

    def test_diagnostics_use_only_train_observations_and_do_not_refit(self):
        dataset = self.dataset

        class TrainOnlyView:
            digest = dataset.digest
            metadata = dataset.metadata

            def split(self, name):
                if name != "train":
                    raise AssertionError("accessed held-out split")
                split = dataset.split(name)
                # Exact simulator states are intentionally not available.
                return {key: split[key] for key in ("observations", "actions", "group_ids")}

        with patch("thrusterchain.planning.identify_from_training", side_effect=AssertionError("refit")), \
                patch("numpy.linalg.lstsq", side_effect=AssertionError("refit")), \
                patch("thrusterchain.planning.matrices", side_effect=AssertionError("oracle")):
            evidence = validate_identification_receipt(self.receipt, TrainOnlyView(), self.A, self.B)
        self.assertFalse(evidence["training_provenance_verified"])

    def test_matrices_must_have_runner_shape_dtype_and_finite_values(self):
        invalid = (self.A.astype(np.float32), self.A[:, :-1], np.full_like(self.A, np.nan))
        for A in invalid:
            with self.subTest(shape=A.shape, dtype=str(A.dtype)):
                with self.assertRaisesRegex(AssertionError, "finite float64 matrix"):
                    validate_identification_receipt(self.receipt, self.dataset, A, self.B)


if __name__ == "__main__":
    unittest.main()
