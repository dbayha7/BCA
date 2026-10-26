"""Adversarial checks for contradictory raw records, summaries and CSV rows."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from thrusterchain.reporting import FIELDS, control_rows, validate_control_records
from audit_benchmark import artifact_checks, load_json

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "results/smoke_v2"


class ReportingIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.results = json.loads((ORIGINAL / "benchmark_results.json").read_text())
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.csv = self.run / "control_episode_records.csv"
        self.csv.write_bytes((ORIGINAL / "control_episode_records.csv").read_bytes())

    def check(self):
        return validate_control_records(self.results, self.csv)

    def test_original_raw_records_and_all_summaries_reproduce(self):
        self.assertEqual(self.check()["rows"], 32)
        self.assertEqual(self.check()["summary_cells"], 16)

    def test_original_artifact_valid_with_its_registered_source_snapshot(self):
        # Isolate the consistency change from the intentional source revision.
        original_source = json.loads((ORIGINAL / "source_manifest_after.json").read_text())
        with patch("audit_benchmark.source_manifest", return_value=original_source):
            self.assertEqual(artifact_checks(ORIGINAL)["control_record_validation"]["rows"], 32)

    def test_tampered_mean_rejected_by_artifact_checks(self):
        self.results["baselines"]["behavior_cloning"]["families"]["recorded"]["mean_return"] += 10000
        (self.run / "benchmark_results.json").write_text(json.dumps(self.results))
        with self.assertRaisesRegex(AssertionError, "mean_return"):
            artifact_checks(self.run)

    def test_tampered_raw_return_with_unchanged_summary_rejected(self):
        self.results["baselines"]["random"]["families"]["recorded"]["returns"][0] += 10
        with self.assertRaisesRegex(AssertionError, "mean_return"):
            self.check()

    def test_tampered_csv_value_rejected_by_artifact_checks(self):
        rows = list(csv.DictReader(self.csv.read_text().splitlines()))
        rows[0]["return_undiscounted"] = str(float(rows[0]["return_undiscounted"]) + 10000)
        self.write_csv(rows)
        (self.run / "benchmark_results.json").write_text(json.dumps(self.results))
        with self.assertRaisesRegex(AssertionError, "CSV/"):
            artifact_checks(self.run)

    def write_csv(self, rows):
        with self.csv.open("w", newline="", encoding="utf8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_csv_key_changes_duplicates_and_omissions_refused(self):
        original = list(csv.DictReader(self.csv.read_text().splitlines()))
        variants = [original[:-1], original + [original[0]], copy.deepcopy(original)]
        variants[-1][0]["evaluation_seed"] = "999999"
        for rows in variants:
            with self.subTest(rows=len(rows)):
                self.write_csv(rows)
                with self.assertRaisesRegex(AssertionError, "key"):
                    self.check()

    def test_summary_sd_discount_count_and_seed_corruption_refused(self):
        for field, value in (("sample_sd_return", -1.), ("mean_discounted_return", 10000.),
                             ("episodes", 3), ("episode_seeds", [123, 123])):
            with self.subTest(field=field):
                result = copy.deepcopy(self.results)
                result["baselines"]["behavior_cloning"]["families"]["recorded"][field] = value
                with self.assertRaises(AssertionError):
                    control_rows(result)

    def test_csv_order_is_irrelevant_and_nonfinite_value_is_refused(self):
        rows = list(csv.DictReader(self.csv.read_text().splitlines()))
        self.write_csv(rows[::-1])
        self.assertEqual(self.check()["rows"], 32)
        rows[0]["return_undiscounted"] = "nan"
        self.write_csv(rows)
        with self.assertRaisesRegex(AssertionError, "nonfinite"):
            self.check()

    def test_nonstandard_json_and_duplicate_keys_refused(self):
        self.results["baselines"]["random"]["max_solver_kkt"] = float('nan')
        path=self.run/"benchmark_results.json"
        path.write_text(json.dumps(self.results))
        with self.assertRaisesRegex(AssertionError,"nonstandard JSON"):
            artifact_checks(self.run)
        path.write_text('{"profile":"smoke","profile":"research"}')
        with self.assertRaisesRegex(AssertionError,"duplicate JSON"):
            load_json(path)

    def test_surplus_csv_fields_refused(self):
        lines=self.csv.read_text().splitlines()
        lines[1]+=",unexpected"
        self.csv.write_text("\n".join(lines))
        with self.assertRaisesRegex(AssertionError,"invalid control CSV"):
            self.check()

    def test_float_count_is_not_an_integer_count(self):
        for record,field in ((self.results['baselines']['behavior_cloning']['families']['recorded'],'episodes'),
                             (self.results['baselines']['behavior_cloning'],'environment_steps')):
            original=record[field]
            record[field]=float(original)
            with self.assertRaisesRegex(AssertionError,'count mismatch'):
                self.check()
            record[field]=original

    def test_json_number_overflow_refused_outside_return_arrays(self):
        path=self.run/'benchmark_results.json'
        content=json.dumps(self.results)
        content=content.replace('"max_solver_kkt": null','"max_solver_kkt": 1e400')
        self.assertIn('1e400',content)
        path.write_text(content)
        with self.assertRaisesRegex(AssertionError,'overflowing JSON'):
            artifact_checks(self.run)

    def test_required_bc_counts_cannot_be_removed(self):
        record=self.results['baselines']['behavior_cloning']
        del record['environment_steps']
        del record['environment_episodes']
        with self.assertRaisesRegex(AssertionError,'missing control total'):
            self.check()

    def test_mixed_boolean_raw_values_refused_before_numpy_coercion(self):
        self.results['baselines']['behavior_cloning']['families']['recorded']['stall_fractions']=[False,0.]
        with self.assertRaisesRegex(AssertionError,'boolean raw'):
            self.check()

    def test_null_or_boolean_guard_counters_cannot_claim_zero(self):
        original_source=json.loads((ORIGINAL/'source_manifest_after.json').read_text())
        import audit_benchmark
        original_loader=audit_benchmark.load_json
        for value in (None, False, 0., '0'):
            changed=copy.deepcopy(self.results)
            changed['baselines']['behavior_cloning']['inference_guard']['training_calls']=value
            def loader(path):
                return changed if Path(path).name=='benchmark_results.json' else original_loader(path)
            with self.subTest(value=value), patch('audit_benchmark.source_manifest',return_value=original_source), patch('audit_benchmark.load_json',side_effect=loader):
                with self.assertRaisesRegex(AssertionError,'prohibited operation'):
                    artifact_checks(ORIGINAL)


if __name__ == "__main__":
    unittest.main()
