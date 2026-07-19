<div align="center">

# Customer Churn Prediction & Automated Retraining System

**DEPI Capstone Project · Digital Egypt Pioneers Initiative**

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)

*A production-grade ML system that predicts which customers will churn — before they do.*

</div>

---

## What This Project Does

This system ingests historical telecom customer data, trains a Gradient Boosting model to predict churn, serves predictions via a REST API, and visualises results in an interactive dashboard. An automated retraining pipeline monitors data drift and retrains the model when the distribution shifts.

```
Customer data → Preprocessing → Model → FastAPI endpoint → Streamlit dashboard
                                  ↑
                         MLflow experiment tracking
                         Evidently drift detection
                         Automated retraining pipeline
```

---

## Results

| Model | ROC-AUC | Recall (Churn) | Precision | Accuracy |
|---|---|---|---|---|
| Random Forest | 0.817 | 64% | 56% | 77% |
| XGBoost | 0.808 | 64% | 54% | 76% |
| **Gradient Boosting** ⭐ | **0.833** | **78%** | 52% | 75% |

**Gradient Boosting was selected** — highest ROC-AUC and highest Recall. In churn prediction, catching churners matters more than precision: missing a churner costs more than sending an unnecessary retention offer.

**Best hyperparameters** (GridSearchCV, 40 fits):
```
learning_rate = 0.05   max_depth = 3   n_estimators = 200
```

---

## Key Findings from EDA

| Driver | Insight |
|---|---|
| **Tenure** | Customers in their first 12 months churn at 3× the rate of long-term customers |
| **Contract** | Month-to-month: 42.7% churn · Two-year: 2.8% churn |
| **Internet** | Fiber optic: 41.9% churn · DSL: 19.0% · No internet: 7.4% |
| **Monthly charges** | Churners pay ~$74/month vs $61 for retained customers |

---

## Project Structure

```
Customer-Churn-Prediction_DEPI/
│
├── app/
│   ├── app.py              # FastAPI REST endpoint
│   └── dashboard.py        # Streamlit dashboard
│
├── data/
│   ├── raw/                # Original Telco CSV
│   └── processed/          # Cleaned & encoded data
│
├── models/
│   ├── best_model.pkl      # Trained Gradient Boosting model
│   ├── feature_names.json  # Feature column names
│   ├── X_train_sm.pkl      # SMOTE-balanced training data
│   └── X_test.pkl          # Test set
│
├── notebooks/
│   ├── churn_m1.ipynb      # M1: Data collection, EDA, preprocessing
│   ├── churn_m2.ipynb      # M2: Model training, evaluation, tuning
│   └── churn_m4.ipynb      # M4: MLflow tracking, drift detection
│
├── mlflow/                 # MLflow experiment artifacts
├── Dockerfile              # Docker container definition
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Milestones

### M1 — Data Collection, Preprocessing & EDA
- Loaded Telco Customer Churn dataset (7,043 rows → 7,032 after cleaning)
- Fixed `TotalCharges` dtype issue (stored as string in source)
- Applied binary encoding, one-hot encoding, and StandardScaler
- Addressed class imbalance with SMOTE (26.5% churn → 50/50 balanced)
- EDA: tenure, charges, contract type, and internet service are the strongest churn drivers

### M2 — Model Development & Evaluation
- Trained Random Forest, XGBoost, and Gradient Boosting
- Evaluated with ROC-AUC, Confusion Matrix, Precision-Recall, and Feature Importance
- Hyperparameter tuning via GridSearchCV (40 fits across 8 combinations)
- Selected Gradient Boosting: ROC-AUC 0.833, Recall 78%

### M3 — API & Docker Deployment
- Built FastAPI REST endpoint (`POST /predict`) returning churn probability and risk tier
- Containerised with Docker for reproducible deployment
- Auto-generated Swagger docs at `/docs`

### M4 — MLOps, Monitoring & Dashboard
- Experiment tracking with MLflow (params, metrics, model artifacts)
- Interactive Streamlit dashboard: KPIs, charts, live prediction, feature importance
- Drift detection framework with Evidently AI (PSI thresholds)
- Automated retraining strategy on drift trigger

---

## Quick Start

### 1. Clone & install
```bash
git clone https://github.com/alaa-m0hammed/Customer-Churn-Prediction_DEPI.git
cd Customer-Churn-Prediction_DEPI
pip install -r requirements.txt
```

### 2. Run the API
```bash
cd app
uvicorn app:app --port 8001
```

### 3. Run the dashboard
```bash
cd app
streamlit run dashboard.py
```

### 4. Run with Docker
```bash
docker build -t churn-api .
docker run -p 8001:8001 churn-api
```

### 5. Test the API
```bash
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 2, "MonthlyCharges": 85.5, "TotalCharges": 171.0,
       "gender": 0, "SeniorCitizen": 0, "Partner": 0, "Dependents": 0,
       "PhoneService": 1, "MultipleLines": 0, "OnlineSecurity": 0,
       "OnlineBackup": 0, "DeviceProtection": 0, "TechSupport": 0,
       "StreamingTV": 0, "StreamingMovies": 0, "PaperlessBilling": 1,
       "Contract_One_year": 0, "Contract_Two_year": 0,
       "InternetService_Fiber_optic": 1, "InternetService_No": 0,
       "PaymentMethod_Credit_card": 0,
       "PaymentMethod_Electronic_check": 1,
       "PaymentMethod_Mailed_check": 0}'
```

Expected response:
```json
{
  "churn_probability": 0.9489,
  "risk_level": "High",
  "recommendation": "Send retention offer!"
}
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.14 |
| Data | Pandas, NumPy |
| ML | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Tracking | MLflow |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit, Matplotlib |
| Monitoring | Evidently AI |
| Container | Docker |
| Cloud | Azure ML (deployment target) |
| Version Control | Git, GitHub |

---

## Dataset

**Telco Customer Churn** — IBM Sample Dataset via Kaggle  
7,043 customers · 21 features · Binary target: `Churn` (Yes / No)  
Class distribution: 73.5% No Churn · 26.5% Churn

---

## Author

**Alaa Mohammed** · Backend Developer & Data Engineering Trainee  
DEPI — Digital Egypt Pioneers Initiative (Microsoft × MCIT)  
Helwan National University · Cairo, Egypt  
GitHub: [@alaa-m0hammed](https://github.com/alaa-m0hammed)

---

<div align="center">
<sub>DEPI Capstone · Gradient Boosting · ROC-AUC 0.833 · Recall 78%</sub>
</div>
