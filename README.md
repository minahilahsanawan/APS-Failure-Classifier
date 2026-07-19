<div align="center">

# Scania APS Failure Classifier

### Cost-sensitive machine learning for truck failure root-cause triage

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Extra%20Trees-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-2E8B57)

A complete machine-learning system for identifying whether an already-failed Scania truck requires Air Pressure System inspection or investigation of another subsystem.

**[Open the deployed application](https://aps-failure-classifier.streamlit.app/)**

</div>

---

## Project Overview

This project addresses a cost-sensitive diagnostic classification problem using the **Scania Air Pressure System Failure dataset**.

Every record represents a truck that has already experienced a component failure. The objective is to determine whether the root cause is associated with the Air Pressure System, or APS, rather than another subsystem.

| Label | Operational meaning |
|---|---|
| `pos` | Failure attributable to the Air Pressure System |
| `neg` | Failure attributable to another system |

> `neg` does not represent a healthy truck. It represents a failed truck whose root cause is not the APS.

The solution was developed for **DataDive GDGoC 2026** and includes exploratory data analysis, feature engineering, cost-sensitive model development, threshold optimization, final evaluation, automated testing, and a production-oriented Streamlit interface.

---

## Business Problem

The two classification errors have substantially different operational consequences.

| Error | Operational consequence | Cost |
|---|---|---:|
| False positive | A non-APS failure is sent for unnecessary APS inspection | 10 |
| False negative | A true APS failure is missed | 500 |

The evaluation objective is therefore:

\[
\text{Total Cost} = 10 \times FP + 500 \times FN
\]

A false negative costs fifty times more than a false positive. The model is consequently optimized to identify as many APS failures as possible while controlling the number of unnecessary APS alerts.

This cost structure makes conventional accuracy unsuitable as the primary success measure. The project emphasizes asymmetric cost, positive-class recall, precision, F1-score, PR-AUC, ROC-AUC, and confusion-matrix outcomes.

---

## Dataset Summary

| Property | Value |
|---|---:|
| Training observations | 60,000 |
| Test observations | 16,000 |
| Input features | 170 |
| Positive training cases | 1,000 |
| Positive-class prevalence | 1.67% |
| Task | Binary classification |
| Feature type | Anonymized numerical counters and binned measurements |
| Missing data | Present across multiple variables |

The dataset presents three major analytical challenges:

1. **Severe class imbalance**  
   APS failures account for only 1.67% of the training set.

2. **Extensive missingness**  
   Missing values occur across many variables and carry useful predictive information.

3. **Anonymized features**  
   Feature names do not disclose their engineering meaning, so interpretations are limited to predictive associations rather than unsupported mechanical or causal claims.

The raw training and test files are intentionally excluded from the repository.

---

## Analytical Workflow

### 1. Data quality assessment

The notebook evaluates:

- Class distribution
- Missing-value frequency
- Row-level missingness
- Class-wise missingness differences
- Duplicate and constant columns
- Numerical skewness and outliers
- Train-test schema consistency
- Descriptive distribution drift

### 2. Feature engineering

The final preprocessing strategy uses:

- Median imputation fitted on training data
- Explicit missing-value indicators
- Consistent feature ordering
- Reusable preprocessing inside the trained pipeline

This design allows the model to learn from both recorded values and measurement-availability patterns.

### 3. Imbalance handling

The final classifier uses balanced class weighting so that rare APS-positive observations receive greater influence during training.

### 4. Validation protocol

Model selection and decision-threshold optimization were performed using a stratified training-validation split.

Test features were used only for schema and descriptive compatibility checks. Test labels were reserved for final evaluation and were not used for feature engineering, model selection, hyperparameter selection, or threshold optimization.

### 5. Threshold optimization

The default probability threshold of 0.50 was not assumed to be optimal.

Validation probabilities were evaluated against the published cost function, and the threshold that minimized total validation cost was selected before final test evaluation.

The resulting decision threshold is:

```text
0.1429
```

This lower threshold intentionally favors recall because missing a true APS failure is far more expensive than performing an unnecessary APS inspection.

---

## Final Model

The deployed prediction system consists of:

```text
Median imputation
+ missing-value indicators
+ class-balanced Extra Trees classifier
+ validation-selected decision threshold
```

The serialized artifact stores:

- The fitted preprocessing and classification pipeline
- The ordered list of 170 expected features
- The selected threshold
- The positive-class label
- The model configuration required for inference

---

## Final Test Results

The final model was evaluated on 16,000 test observations after the preprocessing strategy, classifier, and threshold had been fixed.

| Metric | Result |
|---|---:|
| Threshold | 0.1429 |
| True negatives | 15,258 |
| False positives | 367 |
| False negatives | 19 |
| True positives | 356 |
| Recall | 94.93% |
| Precision | 49.24% |
| F1-score | 0.648 |
| PR-AUC | 0.882 |
| ROC-AUC | 0.994 |
| Total cost | 13,170 |

### Operational cost comparison

| Decision policy | False positives | False negatives | Total cost |
|---|---:|---:|---:|
| Always predict non-APS | 0 | 375 | 187,500 |
| Optimized Extra Trees model | 367 | 19 | 13,170 |

The optimized model reduced the published asymmetric cost by approximately **93.0%** while detecting 356 of 375 APS failures.

---

## Key Findings

- APS failures are extremely rare, making accuracy an unreliable standalone metric.
- Missingness patterns are strongly associated with the target class.
- Six of the ten highest-ranked model inputs are missing-value indicators.
- A validation-selected threshold substantially reduces missed APS failures compared with the default threshold.
- The model achieves high positive-class recall while accepting additional false APS alerts in accordance with the published cost structure.
- Feature importance is predictive rather than causal because the variables are anonymized.

---

## Streamlit Application

The application provides two focused views.

### Results

The results dashboard presents:

- Recall, precision, PR-AUC, and cost reduction
- The selected decision threshold
- APS failures detected and missed
- False APS alerts
- Final asymmetric cost
- Cost comparison with the always-negative policy
- Leading predictive features
- Methodological assumptions and limitations

### Batch Scoring

The batch-scoring workflow allows users to:

1. Download an empty template containing the 170 required feature columns
2. Add one or more truck records
3. Upload a UTF-8 CSV file
4. Validate the file structure and numeric content
5. Generate APS probabilities and predicted classes
6. Download the complete scored dataset

An optional `class` column and additional unrelated columns are accepted and preserved but excluded from prediction.

---

## Input Validation and Reliability

The application includes safeguards for:

- Empty files
- Header-only uploads
- Duplicate column names
- Missing required features
- Invalid nonnumeric values
- Recognized textual missing-value markers
- Additional unrelated columns
- Optional target columns
- Probability range validation
- Prediction row-count consistency
- Model and metrics artifact consistency
- Stored threshold consistency

Unexpected technical failures are logged server-side, while users receive concise error messages without internal paths or stack traces.

---

## System Architecture

```text
CSV upload
    |
    v
Header and schema validation
    |
    v
Numeric and missing-value validation
    |
    v
Stored preprocessing pipeline
    |
    v
Extra Trees probability estimation
    |
    v
Validation-selected threshold
    |
    v
Predicted class and downloadable output
```

The codebase separates interface logic from reusable validation and scoring functions:

- `streamlit_app.py` manages the application interface and visualizations
- `app_logic.py` handles artifact validation, CSV parsing, schema checks, numeric validation, and inference
- `test_app_logic.py` verifies valid and invalid scoring scenarios

---

## Repository Structure

```text
APS-Failure-Classifier/
|
|-- aps_failure_eda_model.ipynb
|-- APS_failure_one_page_summary.pdf
|-- streamlit_app.py
|-- app_logic.py
|-- test_app_logic.py
|-- requirements.txt
|-- README.md
|-- SUBMISSION_CHECKLIST.md
|-- .gitignore
|
|-- .streamlit/
|   `-- config.toml
|
`-- outputs/
    |-- aps_failure_model.joblib
    |-- model_metrics.csv
    `-- feature_importance.csv
```

| File | Purpose |
|---|---|
| `aps_failure_eda_model.ipynb` | Complete exploratory analysis, modelling, threshold selection, evaluation, and business interpretation |
| `APS_failure_one_page_summary.pdf` | One-page competition summary |
| `streamlit_app.py` | Streamlit interface and results dashboard |
| `app_logic.py` | Reusable validation and inference logic |
| `test_app_logic.py` | Unit tests and Streamlit AppTest coverage |
| `requirements.txt` | Deployment dependencies |
| `outputs/aps_failure_model.joblib` | Serialized preprocessing and model package |
| `outputs/model_metrics.csv` | Final performance and cost metrics |
| `outputs/feature_importance.csv` | Ranked feature importance |

---

## Local Installation

### Clone the repository

```bash
git clone https://github.com/minahilahsanawan/APS-Failure-Classifier.git
cd APS-Failure-Classifier
```

### Create a virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Start the application

```bash
python -m streamlit run streamlit_app.py
```

The local interface will normally be available at:

```text
http://localhost:8501
```

---

## Automated Testing

Run the complete test suite with:

```bash
python -m unittest -v test_app_logic.py
```

The tests cover:

- Valid batch prediction
- Input row-order preservation
- Optional `class` columns
- Additional unrelated columns
- Missing required features
- Duplicate headers
- Empty and header-only files
- Invalid numeric values
- Textual missing-value tokens
- Template structure
- Prediction-output row counts
- Streamlit Results and Batch Scoring views

---

## Reproducibility

Reproducible inference is supported through:

- A serialized preprocessing and model pipeline
- A fixed feature contract
- A stored threshold
- Version-pinned deployment dependencies
- Relative repository paths
- Artifact consistency checks
- Automated validation tests
- No runtime dependency on the original training and test datasets
- Metrics loaded from committed artifacts rather than duplicated in interface code

---

## Limitations

1. **Anonymized features**  
   The variables cannot be connected reliably to specific physical components.

2. **No truck identifiers**  
   Repeated-truck leakage and entity-level validation cannot be assessed.

3. **No timestamps**  
   Temporal stability and future-data performance cannot be evaluated directly.

4. **Proxy cost assumptions**  
   The published error costs should be replaced with real inspection, downtime, towing, and repair costs before operational use.

5. **Dependence on missingness patterns**  
   Changes in data-collection procedures may alter model performance.

6. **Prospective validation requirement**  
   The model should be validated on later operational data before integration into a service-center workflow.

---

## Responsible Use

This classifier is a diagnostic prioritization tool. It does not replace physical inspection, manufacturer procedures, technician expertise, or engineering judgment.

Predictions should be interpreted alongside operational records and established maintenance practices.

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn |
| Classifier | Extra Trees |
| Model serialization | joblib |
| Visualization | Altair |
| Application | Streamlit |
| Testing | unittest, Streamlit AppTest |
| Deployment | Streamlit Community Cloud |
| Version control | Git and GitHub |

---

## Project Deliverables

- [Analysis notebook](./aps_failure_eda_model.ipynb)
- [One-page project summary](./APS_failure_one_page_summary.pdf)
- Reproducible model and evaluation artifacts
- Tested batch-scoring application
- Deployment configuration and documentation

---

## Acknowledgements

This project was developed for **DataDive GDGoC 2026** using the Scania APS Failure dataset.

It is an independent analytical project and is not an official Scania product or diagnostic system.

---

## Author

**Minahil Ahsan**  
