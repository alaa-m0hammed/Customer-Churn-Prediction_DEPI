from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import json

model = pickle.load(open('best_model.pkl', 'rb'))
feature_names = json.load(open('feature_names.json'))
app = FastAPI(title="Churn Prediction API")

class Customer(BaseModel):
    gender: int = 0
    SeniorCitizen: int = 0
    Partner: int = 0
    Dependents: int = 0
    tenure: float = 0
    PhoneService: int = 0
    MultipleLines: int = 0
    OnlineSecurity: int = 0
    OnlineBackup: int = 0
    DeviceProtection: int = 0
    TechSupport: int = 0
    StreamingTV: int = 0
    StreamingMovies: int = 0
    PaperlessBilling: int = 0
    MonthlyCharges: float = 0
    TotalCharges: float = 0
    Contract_One_year: int = 0
    Contract_Two_year: int = 0
    InternetService_Fiber_optic: int = 0
    InternetService_No: int = 0
    PaymentMethod_Credit_card: int = 0
    PaymentMethod_Electronic_check: int = 0
    PaymentMethod_Mailed_check: int = 0

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running!"}

@app.post("/predict")
def predict(customer: Customer):
    raw = customer.dict()
    data = {
        "gender": raw["gender"],
        "SeniorCitizen": raw["SeniorCitizen"],
        "Partner": raw["Partner"],
        "Dependents": raw["Dependents"],
        "tenure": raw["tenure"],
        "PhoneService": raw["PhoneService"],
        "MultipleLines": raw["MultipleLines"],
        "OnlineSecurity": raw["OnlineSecurity"],
        "OnlineBackup": raw["OnlineBackup"],
        "DeviceProtection": raw["DeviceProtection"],
        "TechSupport": raw["TechSupport"],
        "StreamingTV": raw["StreamingTV"],
        "StreamingMovies": raw["StreamingMovies"],
        "PaperlessBilling": raw["PaperlessBilling"],
        "MonthlyCharges": raw["MonthlyCharges"],
        "TotalCharges": raw["TotalCharges"],
        "Contract_One year": raw["Contract_One_year"],
        "Contract_Two year": raw["Contract_Two_year"],
        "InternetService_Fiber optic": raw["InternetService_Fiber_optic"],
        "InternetService_No": raw["InternetService_No"],
        "PaymentMethod_Credit card (automatic)": raw["PaymentMethod_Credit_card"],
        "PaymentMethod_Electronic check": raw["PaymentMethod_Electronic_check"],
        "PaymentMethod_Mailed check": raw["PaymentMethod_Mailed_check"],
    }
    df = pd.DataFrame([data])[feature_names]
    prob = model.predict_proba(df)[0][1]
    risk = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
    return {
        "churn_probability": round(float(prob), 4),
        "risk_level": risk,
        "recommendation": "Send retention offer!" if prob > 0.5 else "Customer is stable"
    }
