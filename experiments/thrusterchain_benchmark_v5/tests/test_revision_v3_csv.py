"""Regression coverage for strict CSV grammar without changing keyed matching."""
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from thrusterchain.reporting import validate_control_records


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "results/revision_v2_smoke"


class StrictControlCSVTests(unittest.TestCase):
    def setUp(self):
        self.results = json.loads((ORIGINAL / "benchmark_results.json").read_text(encoding="utf8"))
        self.original = (ORIGINAL / "control_episode_records.csv").read_text(encoding="utf8")
        self.temp = tempfile.TemporaryDirectory(prefix="thrusterchain_csv_v3_")
        self.addCleanup(self.temp.cleanup)
        self.csv = Path(self.temp.name) / "control_episode_records.csv"

    def validate(self, text):
        self.csv.write_text(text, encoding="utf8", newline="")
        return validate_control_records(self.results, self.csv)

    def assert_invalid_syntax(self, text):
        # Independently confirm that the fixture violates the strict CSV dialect.
        with self.assertRaises(csv.Error):
            list(csv.reader(io.StringIO(text, newline=""), strict=True))
        with self.assertRaisesRegex(ValueError, "invalid control CSV syntax") as error:
            self.validate(text)
        self.assertIsInstance(error.exception.__cause__, csv.Error)

    def quoted_csv(self, reverse=False):
        rows = list(csv.reader(io.StringIO(self.original, newline=""), strict=True))
        if reverse:
            rows[1:] = reversed(rows[1:])
        output = io.StringIO(newline="")
        csv.writer(output, quoting=csv.QUOTE_ALL, lineterminator="\r\n").writerows(rows)
        return output.getvalue()

    def test_original_records_still_validate(self):
        result = self.validate(self.original)
        self.assertEqual(result["rows"], 32)
        self.assertEqual(result["summary_cells"], 16)

    def test_valid_quoted_header_and_values_still_validate(self):
        result = self.validate(self.quoted_csv())
        self.assertEqual(result["rows"], 32)
        self.assertEqual(result["summary_cells"], 16)

    def test_valid_quoted_rows_can_be_reordered(self):
        self.assertEqual(self.validate(self.quoted_csv(reverse=True)), self.validate(self.original))

    def test_unterminated_quote_on_final_numeric_field_is_rejected(self):
        lines = self.original.rstrip("\r\n").splitlines()
        prefix, value = lines[-1].rsplit(",", 1)
        lines[-1] = prefix + ',"' + value
        self.assert_invalid_syntax("\n".join(lines))

    def test_whitespace_after_closing_quote_is_rejected(self):
        for whitespace in (" ", "\t"):
            with self.subTest(whitespace=repr(whitespace)):
                lines = self.original.rstrip("\r\n").splitlines()
                prefix, value = lines[-1].rsplit(",", 1)
                lines[-1] = prefix + ',"' + value + '"' + whitespace
                self.assert_invalid_syntax("\n".join(lines) + "\n")

    def test_malformed_quoted_header_uses_the_same_validation_error(self):
        first, rest = self.original.split(",", 1)
        self.assert_invalid_syntax('"' + first + '" ,' + rest)


if __name__ == "__main__":
    unittest.main()
