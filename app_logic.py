from __future__ import annotations

import csv
import io
import logging
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

FINAL_MODEL_PATTERN = re.compile(r"extra\s*trees", re.IGNORECASE)
BASELINE_MODEL_ID = "Always negative baseline"
FALSE_NEGATIVE_COST = 500
FALSE_POSITIVE_COST = 10
MISSING_TOKENS = frozenset({"", "na", "nan", "n/a", "null"})
REQUIRED_METRIC_COLUMNS = {
    "model", "threshold", "tn", "fp", "fn", "tp", "precision",
    "recall", "f1", "pr_auc", "roc_auc", "cost",
}


class ArtifactError(RuntimeError):
    """Raised when deployment artifacts are missing or inconsistent."""


class InputValidationError(ValueError):
    """A safe, user-facing upload validation error."""


def validate_artifacts(root: Path) -> dict[str, Path]:
    paths = {
        "model": root / "outputs" / "aps_failure_model.joblib",
        "metrics": root / "outputs" / "model_metrics.csv",
        "importance": root / "outputs" / "feature_importance.csv",
        "config": root / ".streamlit" / "config.toml",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ArtifactError("Required application files are unavailable: " + ", ".join(missing))
    return paths


def load_metrics(path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(path)
    missing_columns = REQUIRED_METRIC_COLUMNS - set(metrics.columns)
    if missing_columns:
        raise ArtifactError("The metrics artifact is missing required fields.")
    return metrics


def load_feature_importance(path: Path) -> pd.DataFrame:
    importance = pd.read_csv(path)
    if not {"feature", "importance"}.issubset(importance.columns):
        raise ArtifactError("The feature-importance artifact has an invalid schema.")
    if importance.empty or importance["feature"].duplicated().any():
        raise ArtifactError("The feature-importance artifact is empty or contains duplicate names.")
    if not np.isfinite(importance["importance"].to_numpy(dtype=float)).all():
        raise ArtifactError("The feature-importance artifact contains invalid values.")
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)


def identify_metric_rows(metrics: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    baseline = metrics.loc[metrics["model"].eq(BASELINE_MODEL_ID)]
    final = metrics.loc[
        metrics["model"].astype(str).str.contains(FINAL_MODEL_PATTERN, na=False)
        & ~metrics["model"].eq(BASELINE_MODEL_ID)
    ]
    if len(baseline) != 1 or len(final) != 1:
        raise ArtifactError("The final model and baseline rows could not be identified uniquely.")
    return final.iloc[0], baseline.iloc[0]


def validate_metric_consistency(
    final: pd.Series,
    baseline: pd.Series,
    *,
    tolerance: float = 1e-9,
) -> float:
    conflicts: list[str] = []
    for label, row in (("final", final), ("baseline", baseline)):
        expected_cost = FALSE_NEGATIVE_COST * int(row["fn"]) + FALSE_POSITIVE_COST * int(row["fp"])
        if not math.isclose(float(row["cost"]), expected_cost, rel_tol=0, abs_tol=tolerance):
            conflicts.append(f"{label}.cost")

    recall_denominator = float(final["tp"] + final["fn"])
    precision_denominator = float(final["tp"] + final["fp"])
    expected_recall = float(final["tp"]) / recall_denominator if recall_denominator else 0.0
    expected_precision = float(final["tp"]) / precision_denominator if precision_denominator else 0.0
    if not math.isclose(float(final["recall"]), expected_recall, rel_tol=0, abs_tol=tolerance):
        conflicts.append("final.recall")
    if not math.isclose(float(final["precision"]), expected_precision, rel_tol=0, abs_tol=tolerance):
        conflicts.append("final.precision")
    if float(baseline["cost"]) <= 0:
        conflicts.append("baseline.cost")
    if conflicts:
        raise ArtifactError("Conflicting metric fields: " + ", ".join(conflicts))
    return 1.0 - float(final["cost"]) / float(baseline["cost"])


def validate_model_package(package: Any, metric_threshold: float) -> dict[str, Any]:
    if not isinstance(package, dict):
        raise ArtifactError("The prediction model package is invalid.")
    required = {"pipeline", "threshold", "feature_columns", "positive_label"}
    if not required.issubset(package):
        raise ArtifactError("The prediction model package is incomplete.")

    pipeline = package["pipeline"]
    expected = list(package["feature_columns"])
    threshold = float(package["threshold"])
    if not expected or len(expected) != len(set(expected)):
        raise ArtifactError("The model feature contract is invalid.")
    if not hasattr(pipeline, "predict_proba"):
        raise ArtifactError("The prediction model does not support probabilities.")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ArtifactError("The stored prediction threshold is invalid.")
    if not math.isclose(threshold, float(metric_threshold), rel_tol=0, abs_tol=1e-12):
        raise ArtifactError("The stored threshold conflicts with the metrics artifact.")

    classifier = getattr(pipeline, "named_steps", {}).get("classifier")
    if classifier is None or classifier.__class__.__name__ != "ExtraTreesClassifier":
        raise ArtifactError("The serialized classifier does not match the reported model family.")
    if getattr(classifier, "class_weight", None) != "balanced":
        raise ArtifactError("The serialized classifier is not the validated balanced model.")

    assumptions = package.get("cost_assumptions", {})
    if assumptions and assumptions != {
        "false_negative": FALSE_NEGATIVE_COST,
        "false_positive": FALSE_POSITIVE_COST,
    }:
        raise ArtifactError("The model cost assumptions conflict with the published cost matrix.")
    return package


def format_feature_name(feature: str) -> str:
    match = re.match(r"^missing_?indicator_+(.+)$", str(feature), flags=re.IGNORECASE)
    return f"Missing: {match.group(1)}" if match else str(feature)


def is_missing_indicator(feature: str) -> bool:
    return bool(re.match(r"^missing_?indicator_+", str(feature), flags=re.IGNORECASE))


def create_empty_template(expected_columns: Iterable[str]) -> bytes:
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerow(list(expected_columns))
    return buffer.getvalue().encode("utf-8")


def _decode_upload(raw: bytes) -> str:
    if not raw:
        raise InputValidationError("The uploaded file is empty.")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputValidationError("The file must be a UTF-8 encoded CSV.") from exc


def _find_header_line(lines: list[str], expected_columns: list[str]) -> int:
    if not lines:
        raise InputValidationError("The uploaded file is empty.")
    expected = set(expected_columns)
    for index, line in enumerate(lines):
        try:
            fields = next(csv.reader([line]))
        except csv.Error:
            continue
        if fields and len(expected.intersection(fields)) >= min(3, len(expected)):
            return index
    return 0


def parse_uploaded_csv(raw: bytes, expected_columns: list[str]) -> pd.DataFrame:
    text = _decode_upload(raw)
    lines = text.splitlines()
    header_index = _find_header_line(lines, expected_columns)
    relevant_text = "\n".join(lines[header_index:])
    try:
        rows = list(csv.reader(io.StringIO(relevant_text)))
    except csv.Error as exc:
        raise InputValidationError("The file could not be processed. Please verify that it is a valid CSV.") from exc
    if not rows or not rows[0] or not any(value.strip() for value in rows[0]):
        raise InputValidationError("The CSV header is empty.")
    header = rows[0]
    duplicate_names = sorted({name for name in header if header.count(name) > 1})
    if duplicate_names:
        raise InputValidationError("Duplicate column names were detected.")
    if len(rows) < 2 or not any(any(cell.strip() for cell in row) for row in rows[1:]):
        raise InputValidationError("The CSV contains a header but no data rows.")
    try:
        frame = pd.read_csv(
            io.StringIO(relevant_text),
            dtype=str,
            keep_default_na=False,
            na_filter=False,
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise InputValidationError("The file could not be processed. Please verify that it is a valid CSV.") from exc
    except MemoryError as exc:
        raise InputValidationError("The uploaded file is too large to process safely.") from exc
    except Exception:
        LOGGER.exception("Unexpected CSV parsing failure")
        raise InputValidationError("The file could not be processed. Please verify that it is a valid CSV.")
    if frame.empty:
        raise InputValidationError("The CSV contains no data rows.")
    return frame


def validate_schema(frame: pd.DataFrame, expected_columns: list[str]) -> list[str]:
    if frame.empty:
        raise InputValidationError("The CSV contains no data rows.")
    missing = [column for column in expected_columns if column not in frame.columns]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else ", ..."
        raise InputValidationError(
            f"The file is missing {len(missing)} required model features: {preview}{suffix}"
        )
    return [column for column in frame.columns if column not in expected_columns and column != "class"]


def validate_numeric_features(frame: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    converted_columns: dict[str, pd.Series] = {}
    invalid_counts: dict[str, int] = {}
    for column in expected_columns:
        raw = frame[column].astype(str)
        stripped = raw.str.strip()
        missing_mask = stripped.str.lower().isin(MISSING_TOKENS)
        converted = pd.to_numeric(stripped.mask(missing_mask), errors="coerce")
        invalid_mask = converted.isna() & ~missing_mask
        if invalid_mask.any():
            invalid_counts[column] = int(invalid_mask.sum())
        converted_columns[column] = converted
    if invalid_counts:
        preview = ", ".join(f"{name} ({count})" for name, count in list(invalid_counts.items())[:8])
        raise InputValidationError(
            "Some required features contain nonnumeric values. Affected columns: " + preview
        )
    numeric = pd.DataFrame(converted_columns, index=frame.index)
    if list(numeric.columns) != expected_columns or len(numeric) != len(frame):
        raise InputValidationError("The validated feature matrix does not match the model contract.")
    return numeric


def score_dataframe(
    original: pd.DataFrame,
    numeric_features: pd.DataFrame,
    package: dict[str, Any],
) -> pd.DataFrame:
    pipeline = package["pipeline"]
    threshold = float(package["threshold"])
    probabilities = np.asarray(pipeline.predict_proba(numeric_features))
    if probabilities.ndim != 2 or probabilities.shape[0] != len(original):
        raise ArtifactError("The prediction output count does not match the input row count.")
    classes = list(getattr(pipeline, "classes_", []))
    if 1 not in classes:
        raise ArtifactError("The model package does not expose the positive probability class.")
    positive_probability = probabilities[:, classes.index(1)]
    if not np.isfinite(positive_probability).all():
        raise ArtifactError("The model returned non-finite probabilities.")
    if ((positive_probability < 0) | (positive_probability > 1)).any():
        raise ArtifactError("The model returned probabilities outside the valid range.")

    output = original.copy()
    positive_label = str(package.get("positive_label", "pos"))
    predicted = np.where(positive_probability >= threshold, positive_label, "neg")
    output.insert(0, "predicted_class", predicted)
    output.insert(1, "aps_probability", positive_probability)
    if len(output) != len(original):
        raise ArtifactError("The prediction output does not preserve the input row count.")
    return output
