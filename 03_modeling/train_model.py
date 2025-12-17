import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data_output")
MODEL_PATH = os.path.join(BASE_DIR, "..", "artifacts")
os.makedirs(MODEL_PATH, exist_ok=True)


# Carga de datos
X = pd.read_csv(os.path.join(DATA_PATH, "X_train_final.csv"))
y = pd.read_csv(os.path.join(DATA_PATH, "y_train.csv")).values.ravel()

print("Datos cargados correctamente")
print("Shape X:", X.shape)
print("Shape y:", y.shape)


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Modelo
model = LogisticRegression(
    max_iter=1000,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)

# Evaluación
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
roc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC en test: {roc:.4f}")


# Guardar modelos y scaler
joblib.dump(model, os.path.join(MODEL_PATH, "logistic_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_PATH, "scaler.pkl"))

print("Modelo y scaler guardados correctamente")
print("Score medio clase 1:", np.mean(y_pred_proba[y_test == 1]))
print("Score medio clase 0:", np.mean(y_pred_proba[y_test == 0]))
