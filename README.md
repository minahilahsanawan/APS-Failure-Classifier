<div align="center">

# 🚛 Scania APS Failure Classifier

### Production-Grade Machine Learning for Predictive Maintenance

<p>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-Extra%20Trees-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Streamlit-Cloud%20Ready-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Tests-15%2F15%20Passing-2E8B57" alt="Tests">
  <img src="https://img.shields.io/badge/Model%20ROC--AUC-0.994-00AA00" alt="ROC-AUC">
</p>

**A complete end-to-end machine learning system** for cost-sensitive binary classification in industrial predictive maintenance using the Scania Air Pressure System Failure dataset.

[🚀 **Launch Live Application**](https://aps-failure-classifier.streamlit.app/) | [📊 **View Metrics**](#-performance-highlights) | [🛠️ **Setup Guide**](#-getting-started)

</div>

---

## 🎯 Executive Summary

This project demonstrates **production-grade machine learning** applied to a real-world industrial problem:

- **Problem**: Identify whether a truck failure is due to the Air Pressure System (APS) or another system
- **Impact**: Reduce unnecessary inspections while ensuring critical failures aren't missed  
- **Solution**: Cost-sensitive Extra Trees classifier achieving **94.93% recall** and **93% total cost reduction**

**Why this matters for MITACS**: This project combines domain understanding, rigorous engineering, and business acumen—showing how ML creates tangible value in industrial settings.

---

## 📈 Performance Highlights

| Metric | Result | Significance |
|--------|--------|--------------|
| **Recall** | **94.93%** | Catches 94.93% of APS failures (minimizes missed maintenance) |
| **Precision** | **49.24%** | Balances efficiency with false alert costs |
| **ROC-AUC** | **0.9941** | Excellent discrimination across all thresholds |
| **PR-AUC** | **0.8824** | Robust performance on imbalanced data |
| **Cost Reduction** | **93.0%** | Reduces operational cost from 187,500 to 13,170 units |
| **Test Set Size** | **16,000 records** | Realistic, unseen evaluation data |

### Why Not 100% Accuracy?

This problem is inherently **cost-asymmetric and imbalanced**:
- **Missing an APS failure** (false negative): 500 cost units
- **Unnecessary APS inspection** (false positive): 10 cost units
- **Class imbalance**: Only 2.3% of failures are APS-related

The model is strategically optimized to prioritize recall over precision—catching virtually all APS failures while accepting some false positives (which cost 50x less). This represents the *optimal* business decision, not a model limitation.

---

## 🏗️ Architecture Overview

```
APS Failure Classifier
├── Data Ingestion & Validation
│   ├── Schema enforcement (170 sensor features)
│   ├── Missing value detection & handling
│   └── Numeric validation
├── Feature Engineering
│   ├── Median imputation for missing values
│   ├── Explicit missingness indicators
│   └── Feature selection & importance ranking
├── Model Development
│   ├── Cost-sensitive Extra Trees ensemble
│   ├── Threshold optimization via PR-AUC
│   ├── Rigorous train-validation-test split
│   └── Cross-validation & consistency checks
├── Production Deployment
│   ├── Streamlit web interface
│   ├── Batch scoring capability
│   ├── Comprehensive error handling
│   └── Automated testing (15 test cases)
└── Monitoring & Validation
    ├── Metric consistency verification
    ├── Model package integrity checks
    └── Runtime artifact validation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/minahilahsanawan/APS-Failure-Classifier.git
cd APS-Failure-Classifier

# Install dependencies
pip install -r requirements.txt

# Run all tests
python -m unittest test_app_logic.py -v

# Launch Streamlit app (requires .streamlit/config.toml)
streamlit run streamlit_app.py
```

### Live Application

Visit the deployed app: [aps-failure-classifier.streamlit.app](https://aps-failure-classifier.streamlit.app/)

---

## 📂 Project Structure

```
APS-Failure-Classifier/
├── aps_failure_eda_model.ipynb       # Complete EDA, modeling, & threshold optimization
├── app_logic.py                      # Core ML pipeline & validation logic
├── streamlit_app.py                  # Production web interface
├── test_app_logic.py                 # 15 comprehensive test cases
├── requirements.txt                  # Dependencies
├── outputs/
│   ├── aps_failure_model.joblib      # Serialized trained model
│   ├── model_metrics.csv             # Performance metrics
│   └── feature_importance.csv        # Feature rankings
└── .streamlit/
    └── config.toml                   # Streamlit configuration
```

---

## 🔍 Key Features

### 1. **Cost-Sensitive Learning**
- Implements asymmetric loss with false-negative cost = 500, false-positive cost = 10
- Optimizes decision threshold via Precision-Recall curve
- Minimizes total operational cost, not accuracy

### 2. **Production-Grade Validation**
- **Schema enforcement**: Validates all 170 required features
- **Type safety**: Comprehensive numeric validation with missing-value detection
- **Artifact verification**: Ensures model package consistency with metrics
- **Error handling**: User-friendly error messages for invalid uploads

### 3. **Robust Testing**
- 15 automated test cases covering:
  - Cost calculations & metric validation
  - CSV parsing with various edge cases
  - Batch scoring with optional/extra columns
  - Streamlit app rendering
- All tests passing ✅

### 4. **User-Friendly Interface**
- **Results View**: Interactive charts, model metrics, feature importance
- **Batch Scoring**: Upload CSVs for bulk predictions
- **Download Template**: Generate pre-formatted input CSV
- **Responsive Design**: Works on desktop and mobile

### 5. **Interpretability**
- Feature importance rankings with visual hierarchy
- Missing-value indicators highlighted
- Cost asymmetry clearly explained
- Model assumptions documented

---

## 💡 Technical Highlights

### Model Selection: Why Extra Trees?
- **Ensemble learning**: Combines multiple trees for robustness
- **Balanced classes**: Supports `class_weight='balanced'` to handle 2.3% positive class
- **Non-linear patterns**: Captures complex sensor interactions
- **Feature importance**: Provides interpretable ranking of diagnostic signals
- **Fast inference**: Efficient batch prediction for production use

### Threshold Optimization
- Precision-Recall curve sweep to identify optimal operating point
- Trade-off between recall and false-positive cost
- Threshold = 0.143 selected for ~95% recall at acceptable cost

### Missing Value Strategy
- **Median imputation**: Robust to outliers
- **Explicit indicators**: Creates binary features capturing measurement availability patterns
- Result: 2 of top 10 predictive features are missingness indicators

---

## 🧪 Testing & Quality Assurance

### Test Coverage
```
CostFunctionTests ..................... 2 tests ✓
BatchScoringTests .................... 11 tests ✓
StreamlitAppTests ..................... 2 tests ✓
────────────────────────────────────────────────
Total ............................... 15 tests ✓
```

### Run Tests
```bash
python -m unittest test_app_logic.py -v
```

---

## 📊 Model Metrics Breakdown

### Confusion Matrix (Test Set: 16,000 records)
```
                    Predicted Negative    Predicted Positive
Actual Negative          15,258                   367
Actual Positive             19                    356
```

### Key Insights
- **True Negatives (15,258)**: Correctly avoided unnecessary inspections
- **False Positives (367)**: Cost = 367 × 10 = 3,670 units
- **False Negatives (19)**: Cost = 19 × 500 = 9,500 units
- **True Positives (356)**: Correctly identified APS failures

**Total Cost**: 13,170 units vs. 187,500 baseline = **93% savings**

---

## 🛠️ Development Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.11+ |
| **ML Framework** | scikit-learn |
| **Data Processing** | pandas, numpy |
| **Web Interface** | Streamlit |
| **Visualization** | Altair |
| **Testing** | unittest |
| **Model Storage** | joblib |
| **CI/CD Ready** | Git, .gitignore |

---

## 📋 Code Quality

- **Type Hints**: Full coverage in core functions
- **Docstrings**: Comprehensive documentation for all public functions
- **Error Handling**: Specific exception types (`ArtifactError`, `InputValidationError`)
- **Logging**: Structured logging for debugging
- **Validation**: Multi-layer input/output validation
- **Linting**: PEP 8 compliant

---

## 🔐 Robustness Features

### Input Validation Pipeline
1. ✅ File encoding check (UTF-8)
2. ✅ CSV structure validation
3. ✅ Header row detection
4. ✅ Duplicate column detection
5. ✅ Required feature verification
6. ✅ Numeric conversion with missing-value handling
7. ✅ Post-prediction consistency checks

### Artifact Verification
- Model package schema validation
- Classifier type verification (ExtraTreesClassifier)
- Cost assumption consistency
- Threshold range validation
- Probability calibration checks

---

## 📚 Learning & Skills Demonstrated

This project showcases:

- ✅ **Machine Learning**: Classification, ensemble methods, cost-sensitive learning
- ✅ **Feature Engineering**: Handling missing values, feature importance analysis
- ✅ **Data Validation**: Schema enforcement, input sanitization
- ✅ **Software Engineering**: Type hints, testing, error handling, documentation
- ✅ **Product Thinking**: User-centric design, business metrics vs. ML metrics
- ✅ **Production Deployment**: Containerization-ready, cloud-deployable
- ✅ **Statistics**: Precision-recall trade-offs, asymmetric cost analysis
- ✅ **Communication**: Clear documentation, interactive visualization

---

## 🎓 For MITACS Reviewers

This project demonstrates **research-ready engineering**:

1. **Problem Formulation**: Clear cost model reflecting real operational constraints
2. **Methodological Rigor**: Train-validation-test split with held-out evaluation
3. **Reproducibility**: Versioned dependencies, fixed random state, logged results
4. **Quality**: Automated testing, artifact validation, comprehensive error handling
5. **Scalability**: Batch prediction, efficient data handling, memory safety checks
6. **Documentation**: Every major component explained, assumptions documented

---

## 👤 Author

**Minahil Ahsan Awan**

---

<div align="center">

**Made with ❤️ for Data-Driven Decision Making**

[⬆ back to top](#-scania-aps-failure-classifier)

</div>
