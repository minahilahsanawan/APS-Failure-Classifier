from __future__ import annotations

import logging
import math
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from app_logic import (
    ArtifactError,
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    InputValidationError,
    calculate_cost_reduction,
    create_empty_template,
    format_feature_name,
    identify_metric_rows,
    is_missing_indicator,
    load_feature_importance,
    load_metrics,
    parse_uploaded_csv,
    score_dataframe,
    validate_artifacts,
    validate_metric_consistency,
    validate_model_package,
    validate_numeric_features,
    validate_schema,
)


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="APS Failure Classifier",
    page_icon=":material/local_shipping:",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def cached_metrics(path: Path) -> pd.DataFrame:
    return load_metrics(path)


@st.cache_data
def cached_feature_importance(path: Path) -> pd.DataFrame:
    return load_feature_importance(path)


@st.cache_resource
def cached_model_package(path: Path):
    return joblib.load(path)


try:
    artifact_paths = validate_artifacts(ROOT)
    metrics = cached_metrics(artifact_paths["metrics"])
    importance = cached_feature_importance(artifact_paths["importance"])
    model_row, baseline_row = identify_metric_rows(metrics)
    cost_reduction = validate_metric_consistency(model_row, baseline_row)
except Exception:
    LOGGER.exception("Application artifact validation failed")
    st.error("Required application results are unavailable or inconsistent.", icon=":material/error:")
    st.stop()

st.title("APS Failure Classifier")
st.caption(
    "Scania truck root-cause triage • Positive class = APS-related failure • Negative class = other-system failure"
)

st.markdown(
    "Neg indicates the failure came from a system other than APS. It does **not** mean the truck was healthy."
)

st.markdown(
    "The model is tuned to reduce missed APS-related failures because each false negative carries a cost of "
    f"{FALSE_NEGATIVE_COST} units versus {FALSE_POSITIVE_COST} units for a false positive."
)

view = st.segmented_control(
    "View",
    ["Results", "Batch scoring"],
    default="Results",
    label_visibility="collapsed",
)

if view == "Results":
    st.subheader("Model summary")
    metric_cols = st.columns(4)
    metric_values = [
        ("Recall", f"{model_row['recall']:.1%}"),
        ("Precision", f"{model_row['precision']:.1%}"),
        ("PR-AUC", f"{model_row['pr_auc']:.3f}"),
        ("Cost reduction", f"{cost_reduction:.1%}"),
    ]
    for col, (label, value) in zip(metric_cols, metric_values):
        col.metric(label, value, border=True)

    st.subheader("Operational decision summary")
    decision_cols = st.columns(5)
    decision_values = [
        ("Threshold", f"{model_row['threshold']:.3f}"),
        ("APS detected", f"{int(model_row['tp']):,}"),
        ("APS missed", f"{int(model_row['fn']):,}"),
        ("False APS alerts", f"{int(model_row['fp']):,}"),
        ("Test cost", f"{int(model_row['cost']):,}"),
    ]
    for col, (label, value) in zip(decision_cols, decision_values):
        col.metric(label, value, border=True)

    display_labels = {
        str(baseline_row["model"]): "Always predict non-APS",
        str(model_row["model"]): "Optimized Extra Trees",
    }
    cost_data = metrics.loc[
        metrics["model"].isin(display_labels), ["model", "cost", "fn", "fp"]
    ].copy()
    cost_data["policy"] = cost_data["model"].map(display_labels)
    cost_data["cost_label"] = cost_data["cost"].map(lambda value: f"{int(value):,}")
    cost_order = ["Always predict non-APS", "Optimized Extra Trees"]
    max_cost = float(cost_data["cost"].max()) * 1.15

    bar_chart = (
        alt.Chart(cost_data, height=240)
        .mark_bar(color="#6366f1", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            y=alt.Y(
                "policy:N",
                title=None,
                sort=cost_order,
                axis=alt.Axis(
                    labelFontSize=12,
                    labelPadding=16,
                    tickSize=0,
                    labelColor="#e2e8f0",
                    titleColor="#e2e8f0",
                ),
            ),
            x=alt.X(
                "cost:Q",
                title="Cost units",
                scale=alt.Scale(zero=True, domain=[0, max_cost]),
                axis=alt.Axis(format=",", labelColor="#cbd5e1", titleColor="#e2e8f0"),
            ),
            tooltip=[
                alt.Tooltip("policy:N", title="Policy"),
                alt.Tooltip("cost:Q", title="Cost", format=","),
                alt.Tooltip("fn:Q", title="False negatives", format=","),
                alt.Tooltip("fp:Q", title="False positives", format=","),
            ],
        )
        .properties(title="Asymmetric Test Cost")
    )
    label_chart = (
        alt.Chart(cost_data, height=240)
        .mark_text(align="left", dx=8, color="#f8fafc", fontWeight=600, fontSize=12)
        .encode(
            y=alt.Y("policy:N", sort=cost_order),
            x=alt.X("cost:Q", stack=None),
            text="cost_label:N",
        )
    )
    cost_chart = (bar_chart + label_chart).configure_view(strokeWidth=0).configure_axis(gridColor="#3f3f46")
    st.altair_chart(cost_chart, width="stretch")
    st.caption(
        f"Cost = {FALSE_NEGATIVE_COST} × false negatives + {FALSE_POSITIVE_COST} × false positives. "
        f"The optimized model reduced the published asymmetric cost by {cost_reduction:.1%}."
    )

    st.info(
        f"The decision threshold of {model_row['threshold']:.3f} was selected using training-validation predictions only. "
        "Because one missed APS-related failure costs fifty times more than one unnecessary APS inspection, the selected threshold intentionally prioritizes recall over precision. "
        "The supplied test labels were reserved for final evaluation and were not used in threshold choice.",
        icon=":material/tune:",
    )

    top_features = importance.head(12).copy()
    top_features["display_feature"] = top_features["feature"].map(format_feature_name)
    plot_importance = top_features.sort_values("importance", ascending=True)
    importance_chart = (
        alt.Chart(plot_importance, height=320)
        .mark_bar(color="#6366f1", cornerRadiusEnd=3)
        .encode(
            y=alt.Y(
                "display_feature:N",
                title=None,
                sort=None,
                axis=alt.Axis(labelLimit=260, labelPadding=8, tickSize=0),
            ),
            x=alt.X(
                "importance:Q",
                title="Model importance",
                scale=alt.Scale(zero=True),
                axis=alt.Axis(format=".2f"),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Original feature"),
                alt.Tooltip("importance:Q", title="Importance", format=".4f"),
            ],
        )
        .properties(title="Leading predictive signals")
    )
    st.altair_chart(importance_chart, width="stretch")
    missing_top_ten = int(importance.head(10)["feature"].map(is_missing_indicator).sum())
    st.caption(
        "Missing-value indicators represent measurement availability patterns. "
        f"{missing_top_ten} of the ten leading signals are missing-value indicators. "
        "Importance is predictive, not causal."
    )

    with st.expander("Method and limitations", icon=":material/science:"):
        st.markdown(
            f"- 60,000 training rows and {int(model_row['tn'] + model_row['fp'] + model_row['fn'] + model_row['tp']):,} test rows with 170 anonymized input features.\n"
            "- Median imputation, explicit missingness indicators, and a balanced Extra Trees ensemble.\n"
            f"- Final evaluation: {int(model_row['tp']):,} true APS detections, {int(model_row['fn']):,} missed APS failures, and {int(model_row['fp']):,} false APS alerts.\n"
            "- Test features were used only for schema and descriptive compatibility checks; test labels were reserved for final evaluation.\n"
            "- Truck IDs and timestamps are unavailable, so entity-level leakage and temporal stability cannot be tested.\n"
            "- Feature names are anonymized; model importance does not establish physical or causal meaning.\n"
            "- Neg means another system caused the failure; it does not mean the truck was healthy.\n"
            "- This model is intended to prioritize diagnostic inspection and does not replace physical inspection or engineering judgment."
        )

else:
    st.subheader("Score truck failure records")
    st.write(
        "Upload a UTF-8 CSV containing the model's 170 sensor columns. An optional `class` "
        "column and additional unrelated columns are preserved but ignored for prediction."
    )
    try:
        package = validate_model_package(
            cached_model_package(artifact_paths["model"]),
            float(model_row["threshold"]),
        )
    except Exception:
        LOGGER.exception("Model package validation failed")
        st.error("The prediction model could not be loaded.", icon=":material/error:")
        st.stop()

    expected_columns = list(package["feature_columns"])
    st.download_button(
        "Download empty input template",
        data=create_empty_template(expected_columns),
        file_name="aps_failure_input_template.csv",
        mime="text/csv",
        icon=":material/download:",
    )
    upload = st.file_uploader("Truck sensor CSV", type="csv")
    if upload is None:
        st.caption("The template contains headers only; add one or more truck records before uploading.")
    else:
        raw = upload.getvalue()
        if len(raw) > 25 * 1024 * 1024:
            st.warning("This file is larger than 25 MB and may take longer to process.")
        try:
            incoming = parse_uploaded_csv(raw, expected_columns)
            if len(incoming) > 100_000:
                st.warning("This file contains more than 100,000 rows and may take longer to score.")
            validate_schema(incoming, expected_columns)
            numeric_features = validate_numeric_features(incoming, expected_columns)
            scored = score_dataframe(incoming, numeric_features, package)
        except InputValidationError as exc:
            st.error(str(exc), icon=":material/error:")
        except ArtifactError:
            LOGGER.exception("Prediction artifact failure")
            st.error("The prediction model could not complete this request.", icon=":material/error:")
        except Exception:
            LOGGER.exception("Unexpected batch-scoring failure")
            st.error(
                "The file could not be processed. Please verify that it is a valid CSV.",
                icon=":material/error:",
            )
        else:
            if len(scored) != len(incoming) or not math.isclose(
                float(package["threshold"]), float(model_row["threshold"]), abs_tol=1e-12
            ):
                LOGGER.error("Post-scoring consistency check failed")
                st.error("The prediction output failed a consistency check.", icon=":material/error:")
                st.stop()
            st.success(
                f"Scored {len(scored):,} rows at threshold {package['threshold']:.3f}.",
                icon=":material/check_circle:",
            )
            st.dataframe(
                scored[["predicted_class", "aps_probability"]].head(100),
                hide_index=True,
                column_config={
                    "predicted_class": st.column_config.TextColumn("Prediction"),
                    "aps_probability": st.column_config.ProgressColumn(
                        "APS probability", min_value=0.0, max_value=1.0, format="percent"
                    ),
                },
            )
            st.download_button(
                "Download predictions",
                data=scored.to_csv(index=False).encode("utf-8"),
                file_name="aps_failure_predictions.csv",
                mime="text/csv",
                icon=":material/download:",
            )
