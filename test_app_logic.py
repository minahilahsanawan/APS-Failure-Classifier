from __future__ import annotations

import csv
import io
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from app_logic import (
    InputValidationError,
    create_empty_template,
    identify_metric_rows,
    load_metrics,
    parse_uploaded_csv,
    score_dataframe,
    validate_metric_consistency,
    validate_model_package,
    validate_numeric_features,
    validate_schema,
)


ROOT = Path(__file__).resolve().parent


class BatchScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = joblib.load(ROOT / "outputs" / "aps_failure_model.joblib")
        cls.expected = list(cls.package["feature_columns"])
        metrics = load_metrics(ROOT / "outputs" / "model_metrics.csv")
        cls.final, cls.baseline = identify_metric_rows(metrics)
        validate_metric_consistency(cls.final, cls.baseline)
        validate_model_package(cls.package, float(cls.final["threshold"]))

    def csv_bytes(self, rows: int = 2, *, class_column: bool = False, extra_column: bool = False) -> bytes:
        columns = list(self.expected)
        if class_column:
            columns = ["class", *columns]
        if extra_column:
            columns.append("workshop_note")
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(columns)
        for row_number in range(rows):
            values = {column: str(row_number) for column in self.expected}
            output = []
            for column in columns:
                if column == "class":
                    output.append("neg")
                elif column == "workshop_note":
                    output.append(f"row-{row_number}")
                else:
                    output.append(values[column])
            writer.writerow(output)
        return buffer.getvalue().encode("utf-8")

    def parse_validate_score(self, raw: bytes):
        original = parse_uploaded_csv(raw, self.expected)
        validate_schema(original, self.expected)
        numeric = validate_numeric_features(original, self.expected)
        return original, numeric, score_dataframe(original, numeric, self.package)

    def test_valid_csv_scores_and_preserves_row_order(self):
        original, _, scored = self.parse_validate_score(self.csv_bytes())
        self.assertEqual(len(scored), len(original))
        self.assertEqual(scored["aa_000"].tolist(), original["aa_000"].tolist())
        self.assertTrue(scored["aps_probability"].between(0, 1).all())
        self.assertEqual(list(scored.columns[:2]), ["predicted_class", "aps_probability"])

    def test_optional_class_column_is_preserved(self):
        original, _, scored = self.parse_validate_score(self.csv_bytes(class_column=True))
        self.assertIn("class", scored.columns)
        self.assertEqual(scored["class"].tolist(), original["class"].tolist())

    def test_additional_column_is_preserved(self):
        original, _, scored = self.parse_validate_score(self.csv_bytes(extra_column=True))
        self.assertEqual(scored["workshop_note"].tolist(), original["workshop_note"].tolist())

    def test_missing_required_column(self):
        frame = parse_uploaded_csv(self.csv_bytes(), self.expected).drop(columns=[self.expected[0]])
        with self.assertRaisesRegex(InputValidationError, "missing 1 required"):
            validate_schema(frame, self.expected)

    def test_duplicate_header(self):
        raw = f"{self.expected[0]},{self.expected[0]}\n1,2\n".encode()
        with self.assertRaisesRegex(InputValidationError, "Duplicate column"):
            parse_uploaded_csv(raw, self.expected)

    def test_empty_file(self):
        with self.assertRaisesRegex(InputValidationError, "empty"):
            parse_uploaded_csv(b"", self.expected)

    def test_header_only_file(self):
        raw = (",".join(self.expected) + "\n").encode()
        with self.assertRaisesRegex(InputValidationError, "no data rows"):
            parse_uploaded_csv(raw, self.expected)

    def test_nonnumeric_required_value(self):
        frame = parse_uploaded_csv(self.csv_bytes(rows=1), self.expected)
        frame.loc[0, self.expected[0]] = "not-a-number"
        with self.assertRaisesRegex(InputValidationError, "nonnumeric"):
            validate_numeric_features(frame, self.expected)

    def test_textual_na_is_missing_not_invalid(self):
        frame = parse_uploaded_csv(self.csv_bytes(rows=1), self.expected)
        for token, column in zip(["na", "NA", "NaN", "nan", "N/A", "null", "NULL", ""], self.expected):
            frame.loc[0, column] = token
        numeric = validate_numeric_features(frame, self.expected)
        self.assertEqual(int(numeric.iloc[0, :8].isna().sum()), 8)

    def test_zero_row_dataframe(self):
        frame = pd.DataFrame(columns=self.expected)
        with self.assertRaisesRegex(InputValidationError, "no data rows"):
            validate_schema(frame, self.expected)

    def test_template_has_exact_ordered_headers_and_no_rows(self):
        template = create_empty_template(self.expected).decode("utf-8")
        rows = list(csv.reader(io.StringIO(template)))
        self.assertEqual(rows, [self.expected])
        self.assertEqual(len(rows[0]), 170)
        self.assertNotIn("class", rows[0])

    def test_prediction_download_preserves_row_count(self):
        original, _, scored = self.parse_validate_score(self.csv_bytes(rows=3))
        downloaded = pd.read_csv(io.BytesIO(scored.to_csv(index=False).encode("utf-8")))
        self.assertEqual(len(downloaded), len(original))


class StreamlitAppTests(unittest.TestCase):
    def test_results_and_batch_views_render(self):
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual([metric.label for metric in app.metric[:4]], [
            "Recall", "Precision", "PR-AUC", "Cost reduction"
        ])
        self.assertEqual(len(app.get("vega_lite_chart")), 2)
        self.assertTrue(any(
            "does not replace physical inspection" in markdown.value
            for markdown in app.markdown
        ))

        app.segmented_control[0].set_value("Batch scoring").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("file_uploader")), 1)
        self.assertTrue(any(button.label == "Download empty input template" for button in app.get("download_button")))


if __name__ == "__main__":
    unittest.main()
