import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)

#
# Configuración
THRESHOLD = 0.70  # Priorizar reducción de falsos positivos


# Rutas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data_output")
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "artifacts")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "reports")

os.makedirs(OUTPUT_PATH, exist_ok=True)


# Carga de datos
X = pd.read_csv(os.path.join(DATA_PATH, "X_train_final.csv"))
y = pd.read_csv(os.path.join(DATA_PATH, "y_train.csv")).values.ravel()


# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Carga de modelos y scaler
model = joblib.load(os.path.join(MODEL_PATH, "logistic_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))


# Escalado
X_test_scaled = scaler.transform(X_test)


# Predicciones

y_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_proba >= THRESHOLD).astype(int)

# Metricas
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC test (independiente del umbral): {roc_auc:.4f}")
print(f"\nReporte de clasificación (umbral = {THRESHOLD}):")
print(classification_report(y_test, y_pred))


# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap="Blues")
plt.title(f"Matriz de Confusión (umbral = {THRESHOLD})")
plt.savefig(os.path.join(OUTPUT_PATH, "confusion_matrix.png"), dpi=150)
plt.show()


# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
idx = (abs(thresholds - THRESHOLD)).argmin()

plt.plot(fpr, tpr, label="Curva ROC")
plt.scatter(fpr[idx], tpr[idx], color="red", label=f"Umbral {THRESHOLD}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC priorizando reducción de falsos positivos")
plt.legend()
plt.savefig(os.path.join(OUTPUT_PATH, "roc_curve.png"), dpi=150)
plt.show()

print("Evaluación completada correctamente. Resultados en /reports")
