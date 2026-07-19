# Scania APS failure classifier

This project predicts whether an already-failed truck failed because of its Air Pressure System (APS) or because of another system.

Important: `neg` means **non-APS failure**, not a healthy truck.

## What each file does

- `aps_failure_eda_model.ipynb`: main competition submission. It contains the complete EDA, feature engineering, model training, threshold selection, test evaluation, charts, and written business insights.
- `APS_failure_one_page_summary.pdf`: required one-page summary for submission.
- `streamlit_app.py`: deployment entry point. It shows artifact-backed results and scores CSV files containing the same 170 sensor columns.
- `app_logic.py`: reusable artifact, CSV-validation, and scoring logic used by the app and tests.
- `test_app_logic.py`: unit and Streamlit AppTest coverage for valid and invalid uploads.
- `requirements.txt`: Python packages required by the deployed app.
- `.streamlit/config.toml`: minimal dark app theme.
- `outputs/aps_failure_model.joblib`: trained preprocessing pipeline, classifier, feature order, and decision threshold.
- `outputs/model_metrics.csv`: final test metrics used by the app.
- `outputs/feature_importance.csv`: ranked predictive signals used by the app.
- `SUBMISSION_CHECKLIST.md`: final requirement and deployment checklist.

## Model in simple words

1. The CSV contains 170 anonymized sensor readings and many missing values.
2. Missing values are replaced with each feature's training median.
3. Extra missing-value indicator columns record which readings were absent.
4. A balanced Extra Trees model learns nonlinear patterns while accounting for the rare APS class.
5. The decision threshold is selected on a validation portion of the training data—not on the test set.
6. The serialized decision threshold minimizes the stated operational cost, where a false negative costs 500 units and a false positive costs 10 units.
7. The supplied test set provides the final evaluation reported by the app.

Performance values, confusion-matrix counts, and the threshold are loaded from the committed model artifacts rather than duplicated in application code.

## Run locally

Open PowerShell in this folder and run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Then open the local address printed by Streamlit, normally `http://localhost:8501`.

In **Batch scoring**, download the empty template, keep all 170 required sensor columns, add rows, and upload the completed CSV. An optional `class` column is accepted and preserved. Extra columns are preserved in the downloaded predictions, while missing, duplicate, or non-numeric sensor fields produce a safe validation message.

Run the automated checks with:

```powershell
python -m unittest -v test_app_logic.py
```

## Deploy on Streamlit Community Cloud

1. Create or select the GitHub repository `datadive-aps-failure-classifier`.
2. Push this folder's contents so `streamlit_app.py`, `app_logic.py`, `requirements.txt`, `.streamlit/config.toml`, and `outputs/` are at the repository root.
3. Sign in at `https://share.streamlit.io` using GitHub.
4. Select **Create app** and choose the repository.
5. Set the entrypoint to `streamlit_app.py` and deploy.
6. Open the resulting `streamlit.app` URL and test both **Results** and **Batch scoring**.

The app requires no passwords, API keys, or secrets.
