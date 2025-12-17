Credit Risk Prediction – Machine Learning Pipeline
Descripción general:

Este proyecto desarrolla una solución completa de Machine Learning para predecir la probabilidad de impago (default) de clientes solicitantes de crédito, utilizando información financiera y crediticia histórica.

La solución abarca todo el ciclo de vida del modelo:

Análisis y preparación de datos

Eliminación de ruido mediante clustering no supervisado

Entrenamiento de un modelo supervisado

Evaluación del desempeño

Despliegue mediante una API REST

El sistema permite estimar el riesgo crediticio de un cliente y entregar una decisión automática basada en reglas de negocio.

Fuentes de datos

El proyecto utiliza tres archivos en formato Parquet, los cuales contienen información crediticia y financiera:

application_.parquet
Información principal del solicitante (ingresos, monto solicitado, edad, empleo, etc.)

bureau.parquet
Historial crediticio del cliente proveniente de burós de crédito.

bureau_balance.parquet
Detalle mensual del estado de los créditos reportados en el buró.

Estos archivos deben ubicarse en la carpeta /data.

Estructura del proyecto
Proyecto root/
│
├── data/
│   ├── application_.parquet
│   ├── bureau.parquet
│   └── bureau_balance.parquet
│
├── data_output/
│   ├── X_train_final.csv
│   └── y_train.csv
│
├── models/  (o artifacts/)
│   ├── logistic_model.pkl
│   └── scaler.pkl
│
├── reports/
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── 01_data_understanding/
│   └── dbscan_calibration.py
│
├── 02_data_preparation/
│   └── integrate_and_clean.py
│
├── 03_modeling/
│   └── train_model.py
│
├── 04_evaluation/
│   └── evaluate_model.py
│
├── 05_deployment/
│   ├── app.py
│   └── schemas.py
│
├── requirements.txt
└── README.md

Requisitos:

Python 3.9 o superior

Entorno virtual recomendado

Instalación de dependencias:

pip install -r requirements.txt

Ejecución paso a paso (desde cero)
1) Activar entorno virtual
venv\Scripts\activate

2) Análisis y calibración de DBSCAN

Este paso identifica y calibra los parámetros del clustering para detección de ruido.

python 01_data_understanding/dbscan_calibration.py

3) Integración, limpieza y selección de variables

Integra los tres archivos .parquet

Selecciona variables relevantes

Aplica DBSCAN para eliminar outliers

Genera los datasets finales para modelado

python 02_data_preparation/integrate_and_clean.py


Salida:

data_output/X_train_final.csv

data_output/y_train.csv

4) Entrenamiento del modelo supervisado

Escalado de variables

Entrenamiento de Regresión Logística

Guardado del modelo y scaler

python 03_modeling/train_model.py


Salida:

models/logistic_model.pkl

models/scaler.pkl

5) Evaluación del modelo

Genera métricas y gráficos de desempeño:

python 04_evaluation/evaluate_model.py


Salida:

Curva ROC

Matriz de confusión

Métricas de clasificación

6) Despliegue de la API

Levantar la API REST con FastAPI:

uvicorn 05_deployment.app:app --reload


API disponible en:

http://127.0.0.1:8000


Documentación automática (Swagger):

http://127.0.0.1:8000/docs

Endpoint principal
POST /evaluate_risk

Recibe información del cliente y retorna la probabilidad de impago junto a una decisión automática.

Respuesta:

{
  "probabilidad_incumplimiento": "42.35%",
  "decision": "Revisar manualmente"
}

Decisión de negocio
Probabilidad	Decisión
≥ 70%	Rechazar
40% – 69%	Revisión manual
< 40%	Aprobar
Consideraciones técnicas

DBSCAN se utiliza solo en la etapa de preparación, no en producción.

El scaler entrenado se reutiliza en la API para asegurar coherencia.

El modelo no requiere reentrenamiento para cada predicción.

La API es completamente desacoplada del proceso de entrenamiento.