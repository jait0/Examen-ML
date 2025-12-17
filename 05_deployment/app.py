import os
import joblib
import pandas as pd

from fastapi import FastAPI
from .schemas import ClientData

# APP FAST API
app = FastAPI(
    title="Credit Risk API",
    description="API de predicción de riesgo crediticio (EA3)",
    version="1.0"
)


# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "artifacts")

MODEL_PATH = os.path.join(MODEL_DIR, "logistic_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


# Cargar modelos y scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# Variables
FEATURE_ORDER = [
    'AMT_INCOME_TOTAL',
    'AMT_CREDIT',
    'DAYS_BIRTH',
    'DAYS_EMPLOYED',
    'bureau_credit_count',
    'bureau_credit_active_mean',
    'bureau_days_credit_mean',
    'bureau_amt_credit_sum',
]

# ENDPOINTS
@app.get("/")
def root():
    return {"status": "Credit Risk API funcionando correctamente"}

@app.post("/evaluate_risk")
def evaluate_risk(data: ClientData):

    # 1 Convertir input a DataFrame
    df = pd.DataFrame([data.dict()])

    # 2 Asegurar orden correcto
    df = df[FEATURE_ORDER]

    # 3 Escalado
    X_scaled = scaler.transform(df)

    # 4 Probabilidad
    prob = model.predict_proba(X_scaled)[0, 1]

    # 5 Regla de negocio
    decision = (
        "Rechazar" if prob >= 0.70
        else "Revisar manualmente" if prob >= 0.40
        else "Aprobar"
    )

    return {
        "probabilidad_incumplimiento": f"{prob * 100:.2f}%", 
        "decision": decision,
    }
