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
    """Verify that all required deployment artifacts exist on disk.
    
    Checks for the presence of:
    - Serialized ML model (joblib)
    - Model metrics CSV (performance evaluation)
    - Feature importance CSV (interpretability)
    - Streamlit configuration (deployment settings)
    
    Args:
        root: Root directory path containing the `outputs` and `.streamlit` folders.
    
    Returns:
        Dictionary mapping artifact names to their verified file paths.
    
    Raises:
        ArtifactError: If any required file is missing or inaccessible.
    """
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
    """Load and validate the model metrics CSV artifact.
    
    Verifies that all required metric columns are present:
    model, threshold, tn, fp, fn, tp, precision, recall, f1, pr_auc, roc_auc, cost.
    
    Args:
        path: Path to the model_metrics.csv file.
    
    Returns:
        DataFrame containing model evaluation metrics.
    
    Raises:
        ArtifactError: If required metric columns are missing.
    """
    metrics = pd.read_csv(path)
    missing_columns = REQUIRED_METRIC_COLUMNS - set(metrics.columns)
    if missing_columns:
        raise ArtifactError("The metrics artifact is missing required fields.")
    return metrics


def load_feature_importance(path: Path) -> pd.DataFrame:
    """Load and validate feature importance rankings from CSV.
    
    Ensures data integrity:
    - Required columns present: 'feature' and 'importance'
    - No duplicate feature names
    - No empty importance values
    - All importance values are finite numbers
    
    Args:
        path: Path to the feature_importance.csv file.
    
    Returns:
        DataFrame with features sorted by importance (descending), index reset.
    
    Raises:
        ArtifactError: If schema is invalid, duplicates found, or values are non-finite.
    """
    importance = pd.read_csv(path)
    if not {"feature", "importance"}.issubset(importance.columns):
        raise ArtifactError("The feature-importance artifact has an invalid schema.")
    if importance.empty or importance["feature"].duplicated().any():
        raise ArtifactError("The feature-importance artifact is empty or contains duplicate names.")
    if not np.isfinite(importance["importance"].to_numpy(dtype=float)).all():
        raise ArtifactError("The feature-importance artifact contains invalid values.")
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)


def identify_metric_rows(metrics: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Extract and validate the final model and baseline performance rows.
    
    Identifies:
    - Baseline: Row with model_id == "Always negative baseline"
    - Final: Row with model_id matching "extra trees" (case-insensitive)
    
    Args:
        metrics: DataFrame loaded from model_metrics.csv.
    
    Returns:
        Tuple of (final_model_row, baseline_row) as pandas Series.
    
    Raises:
        ArtifactError: If baseline or final model row cannot be uniquely identified.
    """
    baseline = metrics.loc[metrics["model"].eq(BASELINE_MODEL_ID)]
    final = metrics.loc[
        metrics["model"].astype(str).str.contains(FINAL_MODEL_PATTERN, na=False)
        & ~metrics["model"].eq(BASELINE_MODEL_ID)
    ]
    if len(baseline) != 1 or len(final) != 1:
        raise ArtifactError("The final model and baseline rows could not be identified uniquely.")
    return final.iloc[0], baseline.iloc[0]


def calculate_binary_cost(
    *,
    false_positives: int,
    false_negatives: int,
    false_positive_cost: int = FALSE_POSITIVE_COST,
    false_negative_cost: int = FALSE_NEGATIVE_COST,
) -> int:
    """Calculate total asymmetric cost from confusion matrix values.
    
    Cost = (false_positive_cost × FP) + (false_negative_cost × FN)
    
    Default costs reflect business priorities:
    - FN (missed APS failure): 500 units
    - FP (unnecessary APS inspection): 10 units
    
    Args:
        false_positives: Count of false positive predictions.
        false_negatives: Count of false negative predictions.
        false_positive_cost: Cost per false positive (default: 10).
        false_negative_cost: Cost per false negative (default: 500).
    
    Returns:
        Total cost as integer.
    """
    return int(false_negative_cost) * int(false_negatives) + int(false_positive_cost) * int(false_positives)


def calculate_cost_reduction(final_cost: float, baseline_cost: float) -> float:
    """Calculate percentage cost reduction: (1 - final/baseline).
    
    Measures model improvement relative to the always-negative baseline.
    Result is clamped to [0, 1].
    
    Args:
        final_cost: Total cost of the optimized model.
        baseline_cost: Total cost of the baseline (always predict negative).
    
    Returns:
        Cost reduction fraction in range [0.0, 1.0].
    
    Raises:
        ValueError: If baseline_cost <= 0 (cannot compute meaningful reduction).
    """
    if baseline_cost <= 0:
        raise ValueError("Baseline cost must be positive to compute a cost reduction.")
    return max(0.0, 1.0 - float(final_cost) / float(baseline_cost))


def validate_metric_consistency(
    final: pd.Series,
    baseline: pd.Series,
    *,
    tolerance: float = 1e-9,
) -> float:
    """Verify that metrics are arithmetically consistent and realistic.
    
    Checks:
    1. Reported cost matches calculated cost from confusion matrix
    2. Reported recall matches TP / (TP + FN)
    3. Reported precision matches TP / (TP + FP)
    4. Baseline cost is positive
    
    Args:
        final: Series containing final model metrics (tp, fp, fn, cost, recall, precision).
        baseline: Series containing baseline model metrics (cost).
        tolerance: Numeric tolerance for floating-point comparisons (default: 1e-9).
    
    Returns:
        Cost reduction percentage (0.0 to 1.0).
    
    Raises:
        ArtifactError: If any metric is inconsistent or unrealistic.
    """
    conflicts: list[str] = []
    for label, row in (("final", final), ("baseline", baseline)):
        expected_cost = calculate_binary_cost(
            false_positives=int(row["fp"]),
            false_negatives=int(row["fn"]),
        )
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
    return calculate_cost_reduction(float(final["cost"]), float(baseline["cost"]))


def validate_model_package(package: Any, metric_threshold: float) -> dict[str, Any]:
    """Verify the serialized ML model package for deployment safety and consistency.
    
    Comprehensive validation ensures:
    - Package is a dictionary with required keys
    - Pipeline supports predict_proba() method
    - Threshold is finite and in [0.0, 1.0]
    - Classifier is ExtraTreesClassifier with balanced class_weight
    - Threshold matches metrics artifact exactly
    - Cost assumptions align with deployment parameters
    
    Args:
        package: Deserialized model package (typically from joblib.load()).
        metric_threshold: Expected threshold value from metrics CSV (must match).
    
    Returns:
        The validated package dict (unchanged).
    
    Raises:
        ArtifactError: If any validation check fails.
    """
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
    """Convert raw feature name to human-readable form for display.
    
    Missing-value indicator naming convention:
    - Input pattern: 'missing_indicator_<feature>' or 'missingindicator_<feature>'
    - Output: 'Missing: <feature>'
    - All other features: displayed as-is
    
    Args:
        feature: Raw feature name from model.
    
    Returns:
        Formatted display name (unchanged for non-indicators).
    """
    match = re.match(r"^missing_?indicator_+(.+)$", str(feature), flags=re.IGNORECASE)
    return f"Missing: {match.group(1)}" if match else str(feature)


def is_missing_indicator(feature: str) -> bool:
    """Check if a feature represents a missing-value indicator.
    
    A feature is considered a missing-value indicator if its name starts with
    'missing_indicator_' or 'missingindicator_' (case-insensitive).
    
    Args:
        feature: Feature name to check.
    
    Returns:
        True if feature is a missing-value indicator, False otherwise.
    """
    return bool(re.match(r"^missing_?indicator_+", str(feature), flags=re.IGNORECASE))


def create_empty_template(expected_columns: Iterable[str]) -> bytes:
    """Generate a UTF-8 encoded CSV with headers only (no data rows).
    
    Used as a download template for users preparing batch prediction uploads.
    The file contains only the header row with all required feature names,
    allowing users to fill in data while ensuring correct column order.
    
    Args:
        expected_columns: Iterable of column names (typically: model feature_columns).
    
    Returns:
        UTF-8 encoded bytes representing a valid CSV with headers only.
    """
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerow(list(expected_columns))
    return buffer.getvalue().encode("utf-8")


def _decode_upload(raw: bytes) -> str:
    """Decode uploaded file bytes to UTF-8 string.
    
    Handles UTF-8 with BOM (Byte Order Mark) using 'utf-8-sig' codec.
    
    Args:
        raw: Raw file bytes from upload.
    
    Returns:
        Decoded string content.
    
    Raises:
        InputValidationError: If file is empty or not valid UTF-8.
    """
    if not raw:
        raise InputValidationError("The uploaded file is empty.")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputValidationError("The file must be a UTF-8 encoded CSV.") from exc


def _find_header_line(lines: list[str], expected_columns: list[str]) -> int:
    """Locate the CSV header row, handling files with preamble text.
    
    Searches for a line that:
    1. Can be parsed as valid CSV
    2. Contains at least 3 expected model columns (or all if fewer than 3 expected)
    
    Useful for files with metadata, licenses, or comments before the actual data.
    
    Args:
        lines: List of text lines from file (from splitlines()).
        expected_columns: List of model feature names to search for.
    
    Returns:
        Line index (0-based) containing the header, or 0 if header not found.
    
    Raises:
        InputValidationError: If lines list is empty.
    """
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
    """Parse and validate an uploaded CSV file with robust error handling.
    
    Multi-stage validation:
    1. UTF-8 decoding with signature handling
    2. CSV structure parsing (handles malformed files)
    3. Header row detection (tolerates preamble)
    4. Duplicate column detection
    5. Non-empty header and data row checks
    6. Schema compatibility verification
    
    Args:
        raw: Raw bytes from file upload.
        expected_columns: List of required model feature names.
    
    Returns:
        DataFrame with all columns preserved (including extra/optional columns).
        Columns are **not** reordered or filtered—they appear in file order.
    
    Raises:
        InputValidationError: For any of:
        - Empty file
        - Invalid UTF-8 encoding
        - Malformed CSV structure
        - Duplicate column names
        - Missing data rows
        - File too large to load
    """
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
    """Verify that all required model features are present in the DataFrame.
    
    Required columns: all columns in expected_columns.
    Optional columns: 'class' (preserved for reference), any additional columns.
    
    Args:
        frame: DataFrame loaded from uploaded CSV.
        expected_columns: List of model feature names (must all be present).
    
    Returns:
        List of extra/unrequired column names (non-model features).
        Includes all columns except those in expected_columns and 'class'.
    
    Raises:
        InputValidationError: If frame is empty or any expected column is missing.
    """
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
    """Convert and validate numeric features, identifying non-numeric values.
    
    For each expected column:
    1. Strip whitespace
    2. Identify missing-value tokens (case-insensitive): '', 'na', 'nan', 'n/a', 'null'
    3. Convert to numeric (coerce errors to NaN)
    4. Flag non-missing rows that could not be converted
    
    Args:
        frame: DataFrame with all expected columns (typically from validate_schema).
        expected_columns: List of required numeric model features.
    
    Returns:
        DataFrame with numeric dtypes, same row count and index as input.
        Only contains the expected_columns in the same order.
    
    Raises:
        InputValidationError: If any expected column contains non-numeric values
        (after accounting for recognized missing-value tokens).
    """
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
    """Generate model predictions with class labels and probability scores.
    
    Process:
    1. Extract pipeline and threshold from package
    2. Generate class probability predictions
    3. Extract positive-class probabilities
    4. Apply decision threshold to generate class labels
    5. Prepend predictions to original data
    
    Args:
        original: Original DataFrame (before numeric conversion), row count = n.
        numeric_features: Converted numeric features for 170 model inputs, row count = n.
        package: Validated model package containing pipeline, threshold, positive_label.
    
    Returns:
        DataFrame with original columns plus 2 new columns prepended:
        - 'predicted_class': Class label ('pos' or 'neg') based on threshold
        - 'aps_probability': Probability of APS failure [0.0, 1.0]
        All original columns preserved in original order.
    
    Raises:
        ArtifactError: If predictions are invalid or inconsistent.
    """
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
