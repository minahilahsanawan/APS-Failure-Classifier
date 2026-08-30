<div align="center">

# Scania APS Failure Classifier

### Cost-sensitive machine learning for truck failure root-cause triage

<p>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-Extra%20Trees-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Streamlit-Cloud%20Ready-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Model-Balanced%20Extra%20Trees-00B894" alt="Extra Trees">
  <img src="https://img.shields.io/badge/Tests-15%20Passing-2E8B57" alt="Tests">
</p>

A production-style machine learning system for identifying whether a failed Scania truck is likely to require APS-related inspection or whether the fault belongs to another subsystem.

[Live App](https://aps-failure-classifier.streamlit.app/) |
[Run Locally](#run-the-project-locally) |
[Project Details](#project-overview)

</div>

---

## Project Overview

This project addresses a real-world industrial classification problem in predictive maintenance. The task is to determine whether a truck failure is associated with the Air Pressure System (APS) or with some other subsystem.

The key challenge is not just classification accuracy, but asymmetric operational risk:

- False negative (missed APS failure): 500 cost units
- False positive (unnecessary APS inspection): 10 cost units

This makes the problem a cost-sensitive classification task, where recall and total cost are more important than raw accuracy.

---

## Business Problem

Every record in the dataset represents a truck that has already failed. The objective is to identify whether the root cause is APS-related, because missing a true APS failure has a much larger operational consequence than performing an unnecessary APS inspection.

This makes the model useful for industrial triage and maintenance prioritization rather than generic classification.

---

## Model Performance

The final model is evaluated on a held-out test set using the published asymmetric cost function.

| Metric | Result |
|--------|--------|
| Recall | 94.93% |
| Precision | 49.24% |
| PR-AUC | 0.8824 |
| ROC-AUC | 0.9941 |
| Cost reduction vs baseline | 93.0% |
| Test rows | 16,000 |

These results are meaningful because the model is optimized for the actual business objective: minimize missed APS failures while avoiding excessive unnecessary inspections.

> Raw accuracy is not the primary performance measure here because the cost of a missed APS failure is much higher than the cost of a false alarm.

---

## System Design

This project combines data science, software engineering, and deployment-ready product thinking.

### Included components

- Data validation and schema enforcement
- Missing-value handling and missingness indicators
- Cost-sensitive threshold optimization
- Balanced Extra Trees model training
- Serialized model packaging and integrity checks
- Streamlit user interface for live inference
- Batch CSV scoring workflow
- Downloadable prediction output and input template
- Automated testing for reliability

### Core files

- `app_logic.py`: validation, preprocessing, scoring, metric checks, model validation
- `streamlit_app.py`: interactive Streamlit application
- `aps_failure_eda_model.ipynb`: exploratory analysis and modeling notebook
- `test_app_logic.py`: automated validation tests
- `outputs/`: trained model and evaluation artifacts

---

## Modeling Approach

The final pipeline uses:

- median imputation for numerical missing values
- explicit missingness indicators
- balanced class weighting
- Extra Trees ensemble classifier
- cost-aware threshold selection on validation data

This combination is appropriate because the dataset is imbalanced, contains meaningful missingness patterns, and requires a decision policy aligned with operational risk instead of standard accuracy.

---

## Key Features

### Production-ready validation

- CSV upload detection and UTF-8 validation
- required feature checks
- duplicate header detection
- non-numeric value detection
- model artifact validation
- strict output consistency checks

### Batch inference workflow

- upload CSV with sensor features
- automatic schema validation
- prediction generation
- preserved original row order
- downloadable predictions file

### Interactive dashboard

- summary metrics
- cost comparison chart
- feature importance visualization
- batch scoring view
- user-friendly validation messages

---

## Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/minahilahsanawan/APS-Failure-Classifier.git
cd APS-Failure-Classifier
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app locally

```bash
streamlit run streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

### 5. Run tests

```bash
python -m unittest test_app_logic.py -v
```

---

## Project Structure

```text
APS-Failure-Classifier/
├── app_logic.py
├── streamlit_app.py
├── test_app_logic.py
├── aps_failure_eda_model.ipynb
├── requirements.txt
├── README.md
├── outputs/
│   ├── aps_failure_model.joblib
│   ├── model_metrics.csv
│   └── feature_importance.csv
├── datasets/
│   └── placeholder.txt
└── .streamlit/
    └── config.toml
```

---

## Deployment

This repository is suitable for a standard Streamlit deployment workflow:

1. Push the project to GitHub
2. Create a new Streamlit app
3. Select the repository and branch
4. Set the main file to `streamlit_app.py`
5. Deploy

The project is structured for simple hosting without custom backend configuration.

---

## Quality and Validation

The repository includes automated checks for:

- cost calculation correctness
- confusion matrix consistency
- data parsing edge cases
- missing-value handling
- schema validation
- model artifact validation
- Streamlit view rendering

All core validation tests pass successfully.

---

## Key Takeaways

This project demonstrates:

- cost-sensitive machine learning
- real-world predictive maintenance modeling
- robust validation and error handling
- statistical reasoning under class imbalance
- production-style deployment structure
- practical software engineering for data science

---

## License

This project is intended for educational, professional portfolio, and demonstration use.
