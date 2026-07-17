import streamlit as st
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np

# Load model and data
model = pickle.load(open('best_model.pkl', 'rb'))
feature_names = json.load(open('feature_names.json'))
df = pd.read_csv('churn_dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")
st.title(" Customer Churn Prediction Dashboard")

# ── KPIs ──────────────────────────────────────────
st.header("Business KPIs")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", "7,032")
col2.metric("Churn Rate", "26.5%", "-2.1%")
col3.metric("At Risk Customers", "1,869", "🔴")
col4.metric("Model ROC-AUC", "0.833", "✅")

st.divider()

# ── Predict Single Customer ───────────────────────
st.header(" Predict Customer Churn")

col1, col2, col3 = st.columns(3)
with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0)
    total = st.number_input("Total Charges ($)", 0.0, 9000.0, monthly * tenure)

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

with col3:
    senior = st.checkbox("Senior Citizen")
    partner = st.checkbox("Has Partner")
    online_sec = st.checkbox("Online Security")
    tech_support = st.checkbox("Tech Support")

if st.button("🔮 Predict Churn", type="primary"):
    from sklearn.preprocessing import StandardScaler

    data = {
        "gender": 0,
        "SeniorCitizen": int(senior),
        "Partner": int(partner),
        "Dependents": 0,
        "tenure": tenure,
        "PhoneService": 1,
        "MultipleLines": 0,
        "OnlineSecurity": int(online_sec),
        "OnlineBackup": 0,
        "DeviceProtection": 0,
        "TechSupport": int(tech_support),
        "StreamingTV": 0,
        "StreamingMovies": 0,
        "PaperlessBilling": 1,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,
        "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
        "InternetService_No": 1 if internet == "No" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if "Credit" in payment else 0,
        "PaymentMethod_Electronic check": 1 if "Electronic" in payment else 0,
        "PaymentMethod_Mailed check": 1 if "Mailed" in payment else 0,
    }

    scaler = StandardScaler()
    df_input = pd.DataFrame([data])[feature_names]
    prob = model.predict_proba(df_input)[0][1]
    risk = "High Risk" if prob > 0.7 else "Medium Risk" if prob > 0.4 else "🟢 Low Risk"

    st.subheader("Prediction Result")
    col1, col2 = st.columns(2)
    col1.metric("Churn Probability", f"{prob:.1%}")
    col2.metric("Risk Level", risk)

    if prob > 0.5:
        st.error("This customer is likely to churn! Consider sending a retention offer.")
    else:
        st.success(" This customer is stable.")

st.divider()

# ── Feature Importance ────────────────────────────
st.header("Feature Importance")
importance = pd.Series(model.feature_importances_, index=feature_names)
importance = importance.sort_values(ascending=True).tail(10)

fig, ax = plt.subplots(figsize=(10, 5))
importance.plot(kind='barh', ax=ax, color='#7c3aed')
ax.set_title("Top 10 Features")
st.pyplot(fig)
