# Datadive submission checklist

## Required deliverables

- [x] Executed Jupyter notebook: `aps_failure_eda_model.ipynb`
- [x] EDA: class distribution, missingness, data quality, and train/test drift
- [x] Feature engineering: median imputation and explicit missingness indicators
- [x] Model: balanced Extra Trees classifier with holdout metrics
- [x] Cost optimization: `10 × FP + 500 × FN`
- [x] Threshold selected on a training validation fold and justified in markdown
- [x] Final evaluation performed against the supplied test dataset
- [x] Business translation included in notebook markdown
- [x] One-page PDF: `APS_failure_one_page_summary.pdf`

## Optional deployment

- [x] Canonical entry point: `streamlit_app.py`
- [x] Minimal native theme: `.streamlit/config.toml`
- [x] Reproducible cloud dependencies: `requirements.txt`
- [x] Results and batch-scoring views pass Streamlit AppTest
- [ ] Push this directory as the root of a GitHub repository
- [ ] In Streamlit Community Cloud, select that repository and `streamlit_app.py`
- [ ] Confirm the public `streamlit.app` URL and smoke-test one uploaded CSV

## Semantic note

`neg` means the truck failure came from a system other than APS. It does not mean the truck is healthy.
