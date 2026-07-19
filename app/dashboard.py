import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import os

# ── Resolve paths relative to this file ──────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE, '..', 'models', 'best_model.pkl')
FEAT_PATH    = os.path.join(BASE, '..', 'models', 'feature_names.json')
DATA_PATH    = os.path.join(BASE, '..', 'data', 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence | Customer-Churn-Prediction_DEPI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#07090f;color:#e2e8f0;}
.main-header{background:linear-gradient(135deg,#0d1b2a 0%,#1a2744 50%,#0d1b2a 100%);border:1px solid rgba(99,179,237,.15);border-radius:16px;padding:32px 40px;margin-bottom:28px;}
.main-title{font-size:2rem;font-weight:700;color:#fff;letter-spacing:-.02em;margin:0;}
.main-subtitle{font-size:.9rem;color:#7090b0;margin-top:6px;}
.live-badge{display:inline-block;background:rgba(52,211,153,.08);border:1px solid rgba(52,211,153,.2);color:#34d399;padding:3px 12px;border-radius:20px;font-size:.7rem;font-weight:500;margin-top:10px;font-family:'JetBrains Mono',monospace;}
.kpi{background:#111827;border:1px solid #1e2d40;border-radius:10px;padding:18px 20px;}
.kpi-label{font-size:.68rem;text-transform:uppercase;letter-spacing:.07em;color:#4a6785;margin-bottom:8px;font-weight:500;}
.kpi-val{font-size:1.7rem;font-weight:700;font-family:'JetBrains Mono',monospace;line-height:1;}
.kpi-sub{font-size:.7rem;color:#4a6785;margin-top:5px;}
.blue{color:#60a5fa;}.red{color:#f87171;}.green{color:#34d399;}.amber{color:#fbbf24;}
.card{background:#111827;border:1px solid #1e2d40;border-radius:10px;padding:18px 20px;margin-bottom:16px;}
.card-title{font-size:.72rem;font-weight:600;color:#94a3b8;margin-bottom:14px;text-transform:uppercase;letter-spacing:.05em;}
.risk-high{background:rgba(248,113,113,.07);border:1px solid rgba(248,113,113,.25);border-radius:10px;padding:16px 20px;color:#f87171;font-size:.85rem;line-height:1.6;margin-top:14px;}
.risk-medium{background:rgba(251,191,36,.07);border:1px solid rgba(251,191,36,.2);border-radius:10px;padding:16px 20px;color:#fbbf24;font-size:.85rem;line-height:1.6;margin-top:14px;}
.risk-low{background:rgba(52,211,153,.07);border:1px solid rgba(52,211,153,.2);border-radius:10px;padding:16px 20px;color:#34d399;font-size:.85rem;line-height:1.6;margin-top:14px;}
.stButton>button{background:#2563eb!important;color:#fff!important;border:none!important;border-radius:8px!important;font-weight:600!important;padding:11px 24px!important;width:100%;}
.stButton>button:hover{background:#1d4ed8!important;}
div[data-testid="stMetricValue"]{color:#f0f4f8!important;font-family:'JetBrains Mono'!important;}
div[data-testid="stMetricLabel"]{color:#4a6785!important;font-size:.78rem!important;}
label{color:#8ba3b8!important;font-size:.82rem!important;}
</style>
""", unsafe_allow_html=True)

# ── Load ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = pickle.load(open(MODEL_PATH, 'rb'))
    feature_names = json.load(open(FEAT_PATH))
    return model, feature_names

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df.dropna()

model, feature_names = load_model()
df_raw = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <div class="main-title">Customer Churn Intelligence</div>
  <div class="main-subtitle">Customer-Churn-Prediction_DEPI · Gradient Boosting · Telco Dataset</div>
  <div class="live-badge">MODEL LIVE — ROC-AUC 0.833 · Recall 78%</div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Predict", "Features", "Models"])

# ═══════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════
with tab1:
    total    = len(df_raw)
    churned  = (df_raw['Churn'] == 'Yes').sum()
    rate     = churned / total
    rev_risk = churned * df_raw[df_raw['Churn']=='Yes']['MonthlyCharges'].mean()

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, label, val, sub, cls in [
        (c1,"Total Customers",f"{total:,}","Active accounts","blue"),
        (c2,"Churn Rate",f"{rate:.1%}",f"{churned:,} lost","red"),
        (c3,"Revenue at Risk",f"${rev_risk:,.0f}","Monthly exposure","red"),
        (c4,"ROC-AUC","0.833","Gradient Boosting","green"),
        (c5,"Recall","78%","Churners caught","green"),
    ]:
        col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-val {cls}">{val}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.write("")
    col_l, col_r = st.columns(2)

    def dark_chart():
        fig, ax = plt.subplots(figsize=(6,3.2))
        fig.patch.set_facecolor('#111827'); ax.set_facecolor('#111827')
        ax.tick_params(colors='#6b7280',labelsize=9)
        for s in ax.spines.values(): s.set_color('#1e2d40')
        return fig, ax

    with col_l:
        fig, ax = dark_chart()
        data = df_raw.groupby('Contract')['Churn'].apply(lambda x:(x=='Yes').mean()*100).sort_values()
        colors = ['#34d399','#fbbf24','#f87171']
        bars = ax.barh(data.index, data.values, color=colors, height=0.5)
        for b,v in zip(bars,data.values): ax.text(v+0.5, b.get_y()+b.get_height()/2, f'{v:.1f}%', va='center', color='#94a3b8', fontsize=9)
        ax.set_title('Churn Rate by Contract', color='#cbd5e0', fontsize=11, fontweight='600', pad=10)
        ax.set_xlim(0, data.max()+12)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_r:
        fig, ax = dark_chart()
        data2 = df_raw.groupby('InternetService')['Churn'].apply(lambda x:(x=='Yes').mean()*100).sort_values()
        bars2 = ax.barh(data2.index, data2.values, color=['#34d399','#fbbf24','#f87171'], height=0.5)
        for b,v in zip(bars2,data2.values): ax.text(v+0.5, b.get_y()+b.get_height()/2, f'{v:.1f}%', va='center', color='#94a3b8', fontsize=9)
        ax.set_title('Churn Rate by Internet Service', color='#cbd5e0', fontsize=11, fontweight='600', pad=10)
        ax.set_xlim(0, data2.max()+12)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════
# TAB 2 — PREDICT
# ═══════════════════════════════════════════════════════════
with tab2:
    st.subheader("Predict Customer Churn Risk")
    st.caption("Enter customer details below and click Predict.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Account**")
        tenure  = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0, step=0.5)
        total_c = tenure * monthly
        st.caption(f"Estimated Total: ${total_c:,.2f}")

    with c2:
        st.markdown("**Service**")
        contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
        payment  = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])

    with c3:
        st.markdown("**Add-ons**")
        senior   = st.checkbox("Senior Citizen")
        partner  = st.checkbox("Has Partner")
        sec      = st.checkbox("Online Security")
        tech     = st.checkbox("Tech Support")
        paper    = st.checkbox("Paperless Billing", value=True)

    if st.button("Run Prediction"):
        row = {
            "gender":0, "SeniorCitizen":int(senior), "Partner":int(partner),
            "Dependents":0, "tenure":tenure, "PhoneService":1, "MultipleLines":0,
            "OnlineSecurity":int(sec), "OnlineBackup":0, "DeviceProtection":0,
            "TechSupport":int(tech), "StreamingTV":0, "StreamingMovies":0,
            "PaperlessBilling":int(paper), "MonthlyCharges":monthly, "TotalCharges":total_c,
            "Contract_One year":  1 if contract=="One year"  else 0,
            "Contract_Two year":  1 if contract=="Two year"  else 0,
            "InternetService_Fiber optic": 1 if internet=="Fiber optic" else 0,
            "InternetService_No":          1 if internet=="No"           else 0,
            "PaymentMethod_Credit card (automatic)": 1 if "Credit"     in payment else 0,
            "PaymentMethod_Electronic check":        1 if "Electronic"  in payment else 0,
            "PaymentMethod_Mailed check":            1 if "Mailed"      in payment else 0,
        }
        df_in  = pd.DataFrame([row])[feature_names]
        prob   = model.predict_proba(df_in)[0][1]
        level  = "High" if prob>0.7 else "Medium" if prob>0.4 else "Low"
        color  = "red"  if prob>0.7 else "amber"  if prob>0.4 else "green"
        cls    = "risk-high" if prob>0.7 else "risk-medium" if prob>0.4 else "risk-low"

        m1,m2,m3 = st.columns(3)
        m1.metric("Churn Probability", f"{prob:.1%}")
        m2.metric("Retention Probability", f"{1-prob:.1%}")
        m3.metric("Risk Level", level)

        msgs = {
            "High":   f"High churn risk ({prob:.1%}). Send a retention offer immediately — contract upgrade, discount, or personal outreach.",
            "Medium": f"Moderate risk ({prob:.1%}). Monitor closely. Consider a loyalty incentive or proactive check-in.",
            "Low":    f"Low risk ({prob:.1%}). Customer appears stable. No immediate action needed.",
        }
        st.markdown(f'<div class="{cls}">{msgs[level]}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# TAB 3 — FEATURES
# ═══════════════════════════════════════════════════════════
with tab3:
    st.subheader("Feature Importance")
    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#111827'); ax.set_facecolor('#111827')
    n = len(importance)
    colors_imp = ['#1e40af']*max(0,n-3) + ['#2563eb','#3b82f6','#60a5fa']
    ax.barh(importance.index, importance.values, color=colors_imp[-n:], height=0.55)
    ax.set_title('Feature Importance — Gradient Boosting', color='#cbd5e0', fontsize=12, fontweight='600', pad=12)
    ax.tick_params(colors='#6b7280', labelsize=9)
    for s in ax.spines.values(): s.set_color('#1e2d40')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.write("")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
**Tenure** is the strongest predictor. Customers in their first 12 months churn at 3x the rate of long-term customers.
Improving the onboarding experience directly reduces churn.
""")
    with col_r:
        st.markdown("""
**Contract type** is the 2nd strongest driver.
Moving a customer from month-to-month to a 2-year contract reduces expected churn by over 90%.
Contract upgrade campaigns are the highest-ROI retention tactic.
""")

# ═══════════════════════════════════════════════════════════
# TAB 4 — MODELS
# ═══════════════════════════════════════════════════════════
with tab4:
    st.subheader("Model Comparison")
    results = pd.DataFrame({
        'Model':     ['Random Forest','XGBoost','Gradient Boosting (tuned)'],
        'ROC-AUC':   [0.8169, 0.8079, 0.8334],
        'Recall':    ['64%','64%','78%'],
        'Precision': ['56%','54%','52%'],
        'Accuracy':  ['77%','76%','75%'],
    })
    st.dataframe(results.set_index('Model'), use_container_width=True)

    st.write("")
    st.info("""
**Why Gradient Boosting?**  
Highest ROC-AUC (0.833) and highest Recall (78%).  
In churn prediction, Recall matters more than Precision — missing a churner costs more than sending an unnecessary retention offer.
""")

    st.write("")
    st.markdown("**Best Parameters (GridSearchCV — 40 fits)**")
    params = pd.DataFrame({
        'Parameter':   ['learning_rate','max_depth','n_estimators'],
        'Value':       [0.05, 3, 200],
        'Effect':      ['Slow, stable learning','Shallow trees — prevents overfitting','200 boosting rounds'],
    })
    st.dataframe(params.set_index('Parameter'), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#1e3a5a;font-size:.75rem;font-family:monospace'>"
    "Customer-Churn-Prediction_DEPI · DEPI Capstone · Gradient Boosting · ROC-AUC 0.833"
    "</p>",
    unsafe_allow_html=True
)
