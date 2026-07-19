<div align="center">

# Scania APS Failure Classifier

### Cost-sensitive machine learning for truck failure root-cause triage

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-Extra%20Trees-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Status-Complete-2E8B57" alt="Status">
</p>

<p>
A complete machine-learning system for identifying whether an already-failed Scania truck requires Air Pressure System inspection or investigation of another subsystem.
</p>

<a href="https://aps-failure-classifier.streamlit.app/">
  <img src="https://img.shields.io/badge/LAUNCH%20LIVE%20APPLICATION-APS%20FAILURE%20CLASSIFIER-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Launch APS Failure Classifier">
</a>

</div>

---

## Project Overview

This project addresses a cost-sensitive diagnostic classification problem using the **Scania Air Pressure System Failure dataset**.

Every record represents a truck that has already experienced a component failure. The objective is to determine whether the root cause is associated with the Air Pressure System, or APS, rather than another subsystem.

<div align="center">

<table>
  <thead>
    <tr>
      <th align="center">Label</th>
      <th align="center">Operational Meaning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><code>pos</code></td>
      <td align="center">Failure attributable to the Air Pressure System</td>
    </tr>
    <tr>
      <td align="center"><code>neg</code></td>
      <td align="center">Failure attributable to another system</td>
    </tr>
  </tbody>
</table>

</div>

> `neg` does not represent a healthy truck. It represents a failed truck whose root cause is not the APS.

The solution was developed for **DataDive GDGoC 2026** and includes exploratory data analysis, feature engineering, cost-sensitive model development, threshold optimization, final evaluation, automated testing, and a production-oriented Streamlit interface.

---

## Business Problem

The two classification errors have substantially different operational consequences.

<div align="center">

<table>
  <thead>
    <tr>
      <th align="center">Error</th>
      <th align="center">Operational Consequence</th>
      <th align="center">Cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">False Positive</td>
      <td align="center">A non-APS failure is sent for unnecessary APS inspection</td>
      <td align="center">10</td>
    </tr>
    <tr>
      <td align="center">False Negative</td>
      <td align="center">A true APS failure is missed</td>
      <td align="center">500</td>
    </tr>
  </tbody>
</table>

</div>

The evaluation objective is:

\[
\text{Total Cost} = 10 \times FP + 500 \times FN
\]

A false negative costs fifty times more than a false positive. The model is therefore optimized to identify as many APS failures as possible while controlling the number of unnecessary APS alerts.

This cost structure makes conventional accuracy unsuitable as the primary success measure. The project emphasizes asymmetric cost, positive-class recall, precision, F1-score, PR-AUC, ROC-AUC, and confusion-matrix outcomes.

---

## Dataset Summary

<div align="center">

<table>
  <thead>
    <tr>
      <th align="center">Property</th>
      <th align="center">Value</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="center">Training Observations</td><td align="center">60,000</td></tr>
    <tr><td align="center">Test Observations</td><td align="center">16,000</td></tr>
    <tr><td align="center">Input Features</td><td align="center">170</td></tr>
    <tr><td align="center">Positive Training Cases</td><td align="center">1,000</td></tr>
    <tr><td align="center">Positive-Class Prevalence</td><td align="center">1.67%</td></tr>
    <tr><td align="center">Task</td><td align="center">Binary Classification</td></tr>
    <tr><td align="center">Feature Type</td><td align="center">Anonymized numerical counters and binned measurements</td></tr>
    <tr><td align="center">Missing Data</td><td align="center">Present across multiple variables</td></tr>
  </tbody>
</table>

</div>

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

### 1. Data Quality Assessment

The notebook evaluates:

- Class distribution
- Missing-value frequency
- Row-level missingness
- Class-wise missingness differences
- Duplicate and constant columns
- Numerical skewness and outliers
- Train-test schema consistency
- Descriptive distribution drift

### 2. Feature Engineering

The final preprocessing strategy uses:

- Median imputation fitted on training data
- Explicit missing-value indicators
- Consistent feature ordering
- Reusable preprocessing inside the trained pipeline

This design allows the model to learn from both recorded values and measurement-availability patterns.

### 3. Imbalance Handling

The final classifier uses balanced class weighting so that rare APS-positive observations receive greater influence during training.

### 4. Validation Protocol

Model selection and decision-threshold optimization were performed using a stratified training-validation split.

Test features were used only for schema and descriptive compatibility checks. Test labels were reserved for final evaluation and were not used for feature engineering, model selection, hyperparameter selection, or threshold optimization.

### 5. Threshold Optimization

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

<div align="center">

<table>
  <thead>
    <tr>
      <th align="center">Metric</th>
      <th align="center">Result</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="center">Threshold</td><td align="center">0.1429</td></tr>
    <tr><td align="center">True Negatives</td><td align="center">15,258</td></tr>
    <tr><td align="center">False Positives</td><td align="center">367</td></tr>
    <tr><td align="center">False Negatives</td><td align="center">19</td></tr>
    <tr><td align="center">True Positives</td><td align="center">356</td></tr>
    <tr><td align="center">Recall</td><td align="center">94.93%</td></tr>
    <tr><td align="center">Precision</td><td align="center">49.24%</td></tr>
    <tr><td align="center">F1-Score</td><td align="center">0.648</td></tr>
    <tr><td align="center">PR-AUC</td><td align="center">0.882</td></tr>
    <tr><td align="center">ROC-AUC</td><td align="center">0.994</td></tr>
    <tr><td align="center">Total Cost</td><td align="center">13,170</td></tr>
  </tbody>
</table>

</div>

### Operational Cost Comparison

<div align="center">

<table>
  <thead>
    <tr>
      <th align="center">Decision Policy</th>
      <th align="center">False Positives</th>
      <th align="center">False Negatives</th>
      <th align="center">Total Cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Always Predict Non-APS</td>
      <td align="center">0</td>
      <td align="center">375</td>
      <td align="center">187,500</td>
    </tr>
    <tr>
      <td align="center">Optimized Extra Trees Model</td>
      <td align="center">367</td>
      <td align="center">19</td>
      <td align="center">13,170</td>
    </tr>
  </tbody>
</table>

</div>

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

<div align="center">

<table>
  <thead>
    <tr>
      <th align="center">File</th>
      <th align="center">Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="center"><code>aps_failure_eda_model.ipynb</code></td><td align="center">Complete exploratory analysis, modelling, threshold selection, evaluation, and business interpretation</td></tr>
    <tr><td align="center"><code>APS_failure_one_page_summary.pdf</code></td><td align="center">One-page competition summary</td></tr>
    <tr><td align="center"><code>streamlit_app.py</code></td><td align="center">Streamlit interface and results dashboard</td></tr>
    <tr><td align="center"><code>app_logic.py</code></td><td align="center">Reusable validation and inference logic</td></tr>
    <tr><td align="center"><code>test_app_logic.py</code></td><td align="center">Unit tests and Streamlit AppTest coverage</td></tr>
    <tr><td align="center"><code>requirements.txt</code></td><td align="center">Deployment dependencies</td></tr>
    <tr><td align="center"><code>outputs/aps_failure_model.joblib</code></td><td align="center">Serialized preprocessing and model package</td></tr>
    <tr><td align="center"><code>outputs/model_metrics.csv</code></td><td align="center">Final performance and cost metrics</td></tr>
    <tr><td align="center"><code>outputs/feature_importance.csv</code></td><td align="center">Ranked feature importance</td></tr>
  </tbody>
</table>

</div>

---

## Local Installation

### Clone the Repository

```bash
git clone https://github.com/minahilahsanawan/APS-Failure-Classifier.git
cd APS-Failure-Classifier
```

### Create a Virtual Environment

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

### Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Start the Application

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

1. **Anonymized Features**  
   The variables cannot be connected reliably to specific physical components.

2. **No Truck Identifiers**  
   Repeated-truck leakage and entity-level validation cannot be assessed.

3. **No Timestamps**  
   Temporal stability and future-data performance cannot be evaluated directly.

4. **Proxy Cost Assumptions**  
   The published error costs should be replaced with real inspection, downtime, towing, and repair costs before operational use.

5. **Dependence on Missingness Patterns**  
   Changes in data-collection procedures may alter model performance.

6. **Prospective Validation Requirement**  
   The model should be validated on later operational data before integration into a service-center workflow.

---

## Responsible Use

This classifier is a diagnostic prioritization tool. It does not replace physical inspection, manufacturer procedures, technician expertise, or engineering judgment.

Predictions should be interpreted alongside operational records and established maintenance practices.

---

## Technology Stack

<div align="center">

<table>
  <thead>
    <tr>
      <th align="center">Component</th>
      <th align="center">Technology</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="center">Language</td><td align="center">Python</td></tr>
    <tr><td align="center">Data Processing</td><td align="center">pandas, NumPy</td></tr>
    <tr><td align="center">Machine Learning</td><td align="center">scikit-learn</td></tr>
    <tr><td align="center">Classifier</td><td align="center">Extra Trees</td></tr>
    <tr><td align="center">Model Serialization</td><td align="center">joblib</td></tr>
    <tr><td align="center">Visualization</td><td align="center">Altair</td></tr>
    <tr><td align="center">Application</td><td align="center">Streamlit</td></tr>
    <tr><td align="center">Testing</td><td align="center">unittest, Streamlit AppTest</td></tr>
    <tr><td align="center">Deployment</td><td align="center">Streamlit Community Cloud</td></tr>
    <tr><td align="center">Version Control</td><td align="center">Git and GitHub</td></tr>
  </tbody>
</table>

</div>

---

## Project Deliverables

- [Analysis notebook](./aps_failure_eda_model.ipynb)
- [One-page project summary](./APS_failure_one_page_summary.pdf)
- Reproducible model and evaluation artifacts
- Tested batch-scoring application
- Deployment configuration and documentation

---

## Acknowledgements

This project was developed for **DataDive GDGoC'26** using the Scania APS Failure dataset.

It is an independent analytical project and is not an official Scania product or diagnostic system.

---

## Author

**Minahil Ahsan**  
