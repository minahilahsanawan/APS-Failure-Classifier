<div align="center">

# Scania APS Failure Classifier

### Cost-sensitive machine learning for truck failure root-cause triage

<p>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-Extra%20Trees-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Model-Extra%20Trees-00B894" alt="Model">
  <img src="https://img.shields.io/badge/Tests-15%20Passing-2E8B57" alt="Tests">
</p>

A production-style machine-learning project for classifying whether a failed Scania truck requires APS-related inspection or whether the failure is caused by another subsystem.

[Live App](https://aps-failure-classifier.streamlit.app/) |
[Project Demo](#project-overview) |
[Run Locally](#run-the-project-locally)

</div>

---

## Project Overview

This project focuses on a real-world industrial classification problem: deciding whether an already-failed truck is likely to have an Air Pressure System (APS) issue or a failure from another subsystem.

The business objective is not standard accuracy. In this setting, a false negative is far more costly than a false positive:

- False negative (missed APS failure): 500 cost units
- False positive (unnecessary APS inspection): 10 cost units

Because of this cost asymmetry, the model is optimized around recall and operational cost, not just raw classification accuracy.

---

## Business Problem

The project models a cost-sensitive diagnostic decision:

- Positive class: APS-related failure
- Negative class: failure caused by another system

The key challenge is imbalance and asymmetric risk. Missing a true APS issue can lead to a much more serious operational issue than sending a non-APS case for a preventive check.

---

## Model Performance

The trained model in this repository is evaluated using a held-out test set and the published asymmetric cost function.

| Metric | Result |
|--------|--------|
| Recall | 94.93% |
| Precision | 49.24% |
| PR-AUC | 0.8824 |
| ROC-AUC | 0.9941 |
| Cost reduction vs baseline | 93.0% |
| Test rows | 16,000 |

These metrics are intentionally framed around the operating objective: catch APS failures early while keeping unnecessary inspections under control.

> Accuracy alone is not the right headline metric for this problem because the cost of a missed APS failure is substantially higher than the cost of a false alarm.

---

## Solution Architecture

The project includes:

- Data validation and schema checks
- Missing-value handling
- Cost-sensitive classification logic
- Decision threshold optimization
- Model packaging and validation
- Streamlit inference app
- CSV upload scoring workflow
- Automated unit testing

### Core components

- `app_logic.py`: model validation, CSV parsing, prediction logic, business-cost checks
- `streamlit_app.py`: interactive web application
- `aps_failure_eda_model.ipynb`: EDA and model development notebook
- `outputs/`: trained model and evaluation artifacts
- `test_app_logic.py`: automated validation tests

---

## Dataset and Modeling

The project uses a Scania APS failure dataset with anonymized sensor features and a strongly imbalanced class distribution.

The final model is a class-balanced Extra Trees classifier with a validation-tuned threshold selected to minimize the cost-sensitive operating objective.

This is appropriate because:

- the dataset is strongly imbalanced
- the signal is non-linear
- the cost function is asymmetric
- feature importance and missingness patterns matter operationally

---

## Production Features

The project is designed as a polished, deployable ML system.

### Included features

- CSV upload and validation
- Required feature schema enforcement
- Non-numeric value detection
- Missing-value handling
- Model artifact validation
- Prediction output generation
- Downloadable empty template and predictions
- Streamlit dashboard with metrics and charts
- Automated error handling

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

## Deployment Notes

This repository is ready for a standard Streamlit deployment flow:

1. Push the project to GitHub
2. Create a new Streamlit Cloud app
3. Select the repository and branch
4. Set the main file as `streamlit_app.py`
5. Deploy

The app is built to work as a simple Python + Streamlit project without custom server setup.

---

## Key Takeaways

This project demonstrates:

- cost-sensitive machine learning
- real-world predictive maintenance modeling
- data validation and system reliability
- interpretability with feature importance
- production-style deployment structure
- strong project documentation and testing discipline

---

## License

This project is intended for educational and portfolio use.
