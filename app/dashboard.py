import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0a0f1e; color: #e2e8f0; }
.main-header { background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0d1b2a 100%); border: 1px solid rgba(99,179,237,0.15); border-radius: 16px; padding: 32px 40px; margin-bottom: 28px; }
.main-title { font-size: 2.2rem; font-weight: 700; color: #ffffff; letter-spacing: -0.02em; margin: 0; }
.main-subtitle { font-size: 0.95rem; color: #7090b0; margin-top: 6px; }
.status-badge { display: inline-block; background: rgba(72,187,120,0.15); border: 1px solid rgba(72,187,120,0.3); color: #68d391; padding: 3px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 500; margin-top: 12px; font-family: 'JetBrains Mono', monospace; }
.kpi-card { background: #111827; border: 1px solid #1e2d40; border-radius: 12px; padding: 20px 24px; }
.kpi-label { font-size: 0.72rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.08em; color: #4a6785; margin-bottom: 8px; }
.kpi-value { font-size: 2rem; font-weight: 700; color: #f0f4f8; font-family: 'JetBrains Mono', monospace; line-height: 1; }
.kpi-sub { font-size: 0.78rem; color: #4a6785; margin-top: 6px; }
.kpi-accent { color: #63b3ed; }
.kpi-danger { color: #fc8181; }
.kpi-success { color: #68d391; }
.section-title { font-size: 1.05rem; font-weight: 600; color: #cbd5e0; margin: 28px 0 16px; }
.risk-high { background: rgba(252,129,129,0.1); border: 1px solid rgba(252,129,129,0.3); border-radius: 10px; padding: 16px 20px; color: #fc8181; font-weight: 500; }
.risk-medium { background: rgba(246,173,85,0.1); border: 1px solid rgba(246,173,85,0.3); border-radius: 10px; padding: 16px 20px; color: #f6ad55; font-weight: 500; }
.risk-low { background: rgba(104,211,145,0.1); border: 1px solid rgba(104,211,145,0.3); border-radius: 10px; padding: 16px 20px; color: #68d391; font-weight: 500; }
.stButton > button { background: #2563eb !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 500 !important; padding: 10px 24px !important; width: 100%; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = pickle.load(open('../models/best_model.pkl', 'rb'))
    feature_names = json.load(open('../models/feature_names.json'))
    return model, feature_names

@st.cache_data
def load_data():
    df = pd.read_csv('../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df.dropna()

model, feature_names = load_model()
df_raw = load_data()

st.markdown("""
<div class="main-header">
    <div class="main-title"> Churn Intelligence</div>
    <div class="main-subtitle">Customer Retention Analytics · Telco Dataset · Gradient Boosting Model</div>
    <div class="status-badge"> MODEL LIVE — ROC-AUC 0.833</div>
</div>
""", unsafe_allow_html=True)

total = len(df_raw)
churned = (df_raw['Churn'] == 'Yes').sum()
churn_rate = churned / total
avg_monthly = df_raw[df_raw['Churn'] == 'Yes']['MonthlyCharges'].mean()
revenue_risk = churned * avg_monthly

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Customers</div><div class="kpi-value kpi-accent">{total:,}</div><div class="kpi-sub">Active accounts</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Churn Rate</div><div class="kpi-value kpi-danger">{churn_rate:.1%}</div><div class="kpi-sub">{churned:,} customers lost</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Revenue at Risk</div><div class="kpi-value kpi-danger">${revenue_risk:,.0f}</div><div class="kpi-sub">Monthly exposure</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Model ROC-AUC</div><div class="kpi-value kpi-success">0.833</div><div class="kpi-sub">Gradient Boosting</div></div>', unsafe_allow_html=True)
with c5:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Recall Score</div><div class="kpi-value kpi-success">78%</div><div class="kpi-sub">Churners detected</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title"> Churn Analysis</div>', unsafe_allow_html=True)
col_l, col_r = st.columns(2)

with col_l:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#111827'); ax.set_facecolor('#111827')
    contract_churn = df_raw.groupby('Contract')['Churn'].apply(lambda x: (x=='Yes').mean()*100).sort_values()
    bars = ax.barh(contract_churn.index, contract_churn.values, color=['#1e40af','#2563eb','#60a5fa'], height=0.5)
    for bar, val in zip(bars, contract_churn.values):
        ax.text(val+0.5, bar.get_y()+bar.get_height()/2, f'{val:.1f}%', va='center', color='#94a3b8', fontsize=10)
    ax.set_title('Churn Rate by Contract Type', color='#cbd5e0', fontsize=11, fontweight='600', pad=12)
    ax.tick_params(colors='#6b7280'); ax.spines[['top','right','bottom','left']].set_color('#1e2d40')
    ax.set_xlim(0, contract_churn.max()+10); plt.tight_layout(); st.pyplot(fig); plt.close()

with col_r:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#111827'); ax.set_facecolor('#111827')
    internet_churn = df_raw.groupby('InternetService')['Churn'].apply(lambda x: (x=='Yes').mean()*100).sort_values()
    bars2 = ax.barh(internet_churn.index, internet_churn.values, color=['#059669','#10b981','#34d399'], height=0.5)
    for bar, val in zip(bars2, internet_churn.values):
        ax.text(val+0.5, bar.get_y()+bar.get_height()/2, f'{val:.1f}%', va='center', color='#94a3b8', fontsize=10)
    ax.set_title('Churn Rate by Internet Service', color='#cbd5e0', fontsize=11, fontweight='600', pad=12)
    ax.tick_params(colors='#6b7280'); ax.spines[['top','right','bottom','left']].set_color('#1e2d40')
    ax.set_xlim(0, internet_churn.max()+10); plt.tight_layout(); st.pyplot(fig); plt.close()

st.markdown('<div class="section-title"> Feature Importance</div>', unsafe_allow_html=True)
importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True).tail(10)
fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor('#111827'); ax.set_facecolor('#111827')
ax.barh(importance.index, importance.values, color=['#1e40af']*7+['#2563eb','#3b82f6','#60a5fa'], height=0.6)
ax.set_title('Top 10 Features — Gradient Boosting', color='#cbd5e0', fontsize=11, fontweight='600', pad=12)
ax.tick_params(colors='#6b7280', labelsize=9); ax.spines[['top','right','bottom','left']].set_color('#1e2d40')
plt.tight_layout(); st.pyplot(fig); plt.close()

st.markdown('<div class="section-title"> Predict Customer Churn</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Account Info**")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0, step=0.5)
    total_charges = monthly * tenure
    st.caption(f"Est. Total Charges: ${total_charges:,.2f}")
with c2:
    st.markdown("**Service Details**")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
with c3:
    st.markdown("**Add-ons**")
    senior = st.checkbox("Senior Citizen")
    partner = st.checkbox("Has Partner")
    online_sec = st.checkbox("Online Security")
    tech_support = st.checkbox("Tech Support")
    paperless = st.checkbox("Paperless Billing", value=True)

if st.button(" Predict Churn Risk"):
    data = {
        "gender":0,"SeniorCitizen":int(senior),"Partner":int(partner),"Dependents":0,
        "tenure":tenure,"PhoneService":1,"MultipleLines":0,"OnlineSecurity":int(online_sec),
        "OnlineBackup":0,"DeviceProtection":0,"TechSupport":int(tech_support),
        "StreamingTV":0,"StreamingMovies":0,"PaperlessBilling":int(paperless),
        "MonthlyCharges":monthly,"TotalCharges":total_charges,
        "Contract_One year":1 if contract=="One year" else 0,
        "Contract_Two year":1 if contract=="Two year" else 0,
        "InternetService_Fiber optic":1 if internet=="Fiber optic" else 0,
        "InternetService_No":1 if internet=="No" else 0,
        "PaymentMethod_Credit card (automatic)":1 if "Credit" in payment else 0,
        "PaymentMethod_Electronic check":1 if "Electronic" in payment else 0,
        "PaymentMethod_Mailed check":1 if "Mailed" in payment else 0,
    }
    df_input = pd.DataFrame([data])[feature_names]
    prob = model.predict_proba(df_input)[0][1]
    r1,r2,r3 = st.columns(3)
    r1.metric("Churn Probability", f"{prob:.1%}")
    r2.metric("Retention Probability", f"{1-prob:.1%}")
    r3.metric("Risk Tier","High " if prob>0.7 else "Medium " if prob>0.4 else "Low ")
    if prob>0.7:
        st.markdown(f'<div class="risk-high"> High churn risk ({prob:.1%}) — Send retention offer immediately!</div>', unsafe_allow_html=True)
    elif prob>0.4:
        st.markdown(f'<div class="risk-medium"> Moderate risk ({prob:.1%}) — Monitor and consider loyalty incentive.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="risk-low"> Low risk ({prob:.1%}) — Customer is stable.</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#2d4a66;font-size:0.8rem;font-family:monospace'>Churn Intelligence · DEPI Capstone · Gradient Boosting · ROC-AUC 0.833</p>", unsafe_allow_html=True)
