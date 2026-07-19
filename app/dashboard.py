cat > ~/churn_ML_project/churn-telco-customer/app/dashboard.py << 'PYEOF'
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt

MODEL_PATH = '/home/alaa-mohammed/churn_ML_project/churn-telco-customer/models/best_model.pkl'
FEAT_PATH  = '/home/alaa-mohammed/churn_ML_project/churn-telco-customer/models/feature_names.json'
DATA_PATH  = '/home/alaa-mohammed/churn_ML_project/churn-telco-customer/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'

st.set_page_config(page_title="Churn Intelligence", page_icon="📡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#07090f;color:#e2e8f0;}
.kpi{background:#111827;border:1px solid #1e2d40;border-radius:10px;padding:18px 20px;}
.kpi-label{font-size:.68rem;text-transform:uppercase;letter-spacing:.07em;color:#4a6785;margin-bottom:8px;font-weight:500;}
.kpi-val{font-size:1.7rem;font-weight:700;font-family:'JetBrains Mono',monospace;line-height:1;}
.kpi-sub{font-size:.7rem;color:#4a6785;margin-top:5px;}
.blue{color:#60a5fa;}.red{color:#f87171;}.green{color:#34d399;}
.risk-high{background:rgba(248,113,113,.07);border:1px solid rgba(248,113,113,.25);border-radius:10px;padding:16px;color:#f87171;margin-top:14px;}
.risk-medium{background:rgba(251,191,36,.07);border:1px solid rgba(251,191,36,.2);border-radius:10px;padding:16px;color:#fbbf24;margin-top:14px;}
.risk-low{background:rgba(52,211,153,.07);border:1px solid rgba(52,211,153,.2);border-radius:10px;padding:16px;color:#34d399;margin-top:14px;}
.stButton>button{background:#2563eb!important;color:#fff!important;border:none!important;border-radius:8px!important;font-weight:600!important;width:100%;}
div[data-testid="stMetricValue"]{color:#f0f4f8!important;font-family:'JetBrains Mono'!important;}
</style>
""", unsafe_allow_html=True)

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

st.markdown("""
<div style="background:linear-gradient(135deg,#0d1b2a,#1a2744);border:1px solid rgba(99,179,237,.15);border-radius:16px;padding:28px 36px;margin-bottom:24px;">
  <div style="font-size:1.8rem;font-weight:700;color:#fff;">Customer Churn Intelligence</div>
  <div style="font-size:.88rem;color:#7090b0;margin-top:4px;">DEPI Capstone · Gradient Boosting · Telco Dataset</div>
  <div style="display:inline-block;background:rgba(52,211,153,.08);border:1px solid rgba(52,211,153,.2);color:#34d399;padding:3px 12px;border-radius:20px;font-size:.7rem;font-family:'JetBrains Mono';margin-top:10px;">MODEL LIVE — ROC-AUC 0.833 · Recall 78%</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Predict", "Features", "Models"])

def dark_chart(w=6, h=3.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#111827'); ax.set_facecolor('#111827')
    ax.tick_params(colors='#6b7280', labelsize=9)
    for s in ax.spines.values(): s.set_color('#1e2d40')
    return fig, ax

with tab1:
    total   = len(df_raw)
    churned = (df_raw['Churn'] == 'Yes').sum()
    rate    = churned / total
    rev     = churned * df_raw[df_raw['Churn']=='Yes']['MonthlyCharges'].mean()

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,lbl,val,sub,cls in [
        (c1,"Total Customers",f"{total:,}","Active accounts","blue"),
        (c2,"Churn Rate",f"{rate:.1%}",f"{churned:,} lost","red"),
        (c3,"Revenue at Risk",f"${rev:,.0f}","Monthly","red"),
        (c4,"ROC-AUC","0.833","Best model","green"),
        (c5,"Recall","78%","Churners caught","green"),
    ]:
        col.markdown(f'<div class="kpi"><div class="kpi-label">{lbl}</div><div class="kpi-val {cls}">{val}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.write("")
    col_l, col_r = st.columns(2)
    with col_l:
        fig, ax = dark_chart()
        d = df_raw.groupby('Contract')['Churn'].apply(lambda x:(x=='Yes').mean()*100).sort_values()
        bars = ax.barh(d.index, d.values, color=['#34d399','#fbbf24','#f87171'], height=0.5)
        for b,v in zip(bars,d.values): ax.text(v+0.5,b.get_y()+b.get_height()/2,f'{v:.1f}%',va='center',color='#94a3b8',fontsize=9)
        ax.set_title('Churn Rate by Contract', color='#cbd5e0', fontsize=11, fontweight='600', pad=10)
        ax.set_xlim(0,d.max()+12); plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_r:
        fig, ax = dark_chart()
        d2 = df_raw.groupby('InternetService')['Churn'].apply(lambda x:(x=='Yes').mean()*100).sort_values()
        bars2 = ax.barh(d2.index, d2.values, color=['#34d399','#fbbf24','#f87171'], height=0.5)
        for b,v in zip(bars2,d2.values): ax.text(v+0.5,b.get_y()+b.get_height()/2,f'{v:.1f}%',va='center',color='#94a3b8',fontsize=9)
        ax.set_title('Churn Rate by Internet Service', color='#cbd5e0', fontsize=11, fontweight='600', pad=10)
        ax.set_xlim(0,d2.max()+12); plt.tight_layout(); st.pyplot(fig); plt.close()

with tab2:
    st.subheader("Predict Customer Churn Risk")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("**Account**")
        tenure  = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0, step=0.5)
        total_c = tenure * monthly
        st.caption(f"Est. Total: ${total_c:,.2f}")
    with c2:
        st.markdown("**Service**")
        contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
        payment  = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
    with c3:
        st.markdown("**Add-ons**")
        senior = st.checkbox("Senior Citizen")
        partner = st.checkbox("Has Partner")
        sec    = st.checkbox("Online Security")
        tech   = st.checkbox("Tech Support")
        paper  = st.checkbox("Paperless Billing", value=True)

    if st.button("Run Prediction"):
        row = {
            "gender":0,"SeniorCitizen":int(senior),"Partner":int(partner),"Dependents":0,
            "tenure":tenure,"PhoneService":1,"MultipleLines":0,"OnlineSecurity":int(sec),
            "OnlineBackup":0,"DeviceProtection":0,"TechSupport":int(tech),
            "StreamingTV":0,"StreamingMovies":0,"PaperlessBilling":int(paper),
            "MonthlyCharges":monthly,"TotalCharges":total_c,
            "Contract_One year":1 if contract=="One year" else 0,
            "Contract_Two year":1 if contract=="Two year" else 0,
            "InternetService_Fiber optic":1 if internet=="Fiber optic" else 0,
            "InternetService_No":1 if internet=="No" else 0,
            "PaymentMethod_Credit card (automatic)":1 if "Credit" in payment else 0,
            "PaymentMethod_Electronic check":1 if "Electronic" in payment else 0,
            "PaymentMethod_Mailed check":1 if "Mailed" in payment else 0,
        }
        prob  = model.predict_proba(pd.DataFrame([row])[feature_names])[0][1]
        level = "High" if prob>0.7 else "Medium" if prob>0.4 else "Low"
        cls   = "risk-high" if prob>0.7 else "risk-medium" if prob>0.4 else "risk-low"

        m1,m2,m3 = st.columns(3)
        m1.metric("Churn Probability", f"{prob:.1%}")
        m2.metric("Retention Probability", f"{1-prob:.1%}")
        m3.metric("Risk Level", level)

        msgs = {
            "High":   f"High risk ({prob:.1%}) — Send retention offer immediately.",
            "Medium": f"Moderate risk ({prob:.1%}) — Monitor and consider loyalty incentive.",
            "Low":    f"Low risk ({prob:.1%}) — Customer is stable.",
        }
        st.markdown(f'<div class="{cls}">{msgs[level]}</div>', unsafe_allow_html=True)

with tab3:
    st.subheader("Feature Importance")
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    fig, ax = dark_chart(10, 5)
    n = len(imp)
    c = ['#1e40af']*max(0,n-3)+['#2563eb','#3b82f6','#60a5fa']
    ax.barh(imp.index, imp.values, color=c[-n:], height=0.55)
    ax.set_title('Feature Importance — Gradient Boosting', color='#cbd5e0', fontsize=12, fontweight='600', pad=12)
    plt.tight_layout(); st.pyplot(fig); plt.close()

with tab4:
    st.subheader("Model Comparison")
    st.dataframe(pd.DataFrame({
        'Model':     ['Random Forest','XGBoost','Gradient Boosting'],
        'ROC-AUC':   [0.8169,0.8079,0.8334],
        'Recall':    ['64%','64%','78%'],
        'Precision': ['56%','54%','52%'],
        'Accuracy':  ['77%','76%','75%'],
    }).set_index('Model'), use_container_width=True)
    st.info("Gradient Boosting selected — highest ROC-AUC (0.833) and Recall (78%). In churn prediction, Recall matters more than Precision.")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#1e3a5a;font-size:.75rem;font-family:monospace'>Customer-Churn-Prediction_DEPI · DEPI Capstone · ROC-AUC 0.833</p>", unsafe_allow_html=True)
PYEOF
