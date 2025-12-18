# Credit Risk Prediction – Machine Learning Pipeline

Este proyecto desarrolla una solución profesional de Machine Learning para predecir la probabilidad de impago (*default*) de clientes solicitantes de crédito, utilizando información financiera y crediticia histórica.

La solución abarca todo el ciclo de vida del modelo bajo la metodología **CRISP-DM**:
* Análisis y preparación de datos.
* Eliminación de ruido mediante **clustering no supervisado (DBSCAN)**.
* Entrenamiento de un modelo supervisado (**Regresión Logística**).
* Evaluación del desempeño y métricas de negocio.
* Despliegue mediante una **API REST** escalable.

---

## Estructura del Proyecto (Metodología CRISP-DM)

```
Proyecto root/
│
├── data/                         # Fuentes originales (.parquet)
├── data_output/                  # Datasets procesados para entrenamiento
├── models/                       # Artefactos serializados (.pkl)
├── reports/                      # Gráficos y métricas de evaluación
│
├── 01_data_understanding/        # Calibración de parámetros (DBSCAN)
├── 02_data_preparation/          # ETL e integración de fuentes
├── 03_modeling/                  # Entrenamiento del modelo supervisado
├── 04_evaluation/                # Scripts de métricas y validación
├── 05_deployment/                # API REST con FastAPI y Schemas
│
├── requirements.txt              # Dependencias del proyecto
└── README.md                     # Documentación general
```

## Fuentes de Datos
El sistema procesa información proveniente de tres fuentes clave ubicadas en la carpeta /data : application_.parquet: Datos demográficos e ingresos del solicitante.bureau.parquet: Historial crediticio externo (Buró).bureau_balance.parquet: Detalle mensual de estados de cuenta externos.
Instalación y RequisitosPython 3.9+Recomendado: Uso de entorno virtual.Bash# Crear entorno virtual
python -m venv venv

# Activar entorno en Windows
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
Ejecución del Pipeline (Paso a Paso)
1. Calibración de DBSCAN:
Identifica y calibra los parámetros para la detección de ruido y outliers.Bashpython 01_data_understanding/dbscan_calibration.py

2. Integración y Limpieza (ETL):
Combina los archivos Parquet, selecciona variables y aplica DBSCAN para eliminar ruido.Bashpython 02_data_preparation/integrate_and_clean.py

3. Entrenamiento del Modelo:
Aplica escalado de variables y entrena la Regresión Logística.Bashpython 03_modeling/train_model.py

4. Evaluación de Desempeño:
Genera la Curva ROC, Matriz de Confusión y reporte de clasificación en la carpeta /reports.Bashpython 04_evaluation/evaluate_model.py

## Despliegue de la API
El sistema utiliza FastAPI para servir el modelo en tiempo real.Levantar el servicio: 
``` 
uvicorn 05_deployment.app:app --reload 
```
API Local: http://127.0.0.1:8000Documentación Interactiva (Swagger): http://127.0.0.1:8000/docs

``` Endpoint Principal: POST /evaluate_riskRecibe``` la información del cliente y retorna el nivel de riesgo.

Ejemplo de respuesta:
```
{
  "probabilidad_incumplimiento": "42.35%",
  "decision": "Revisar manualmente"
}
```
## Lógica de Decisión de Negocio
≥50%,Rechazar

20%−50%,Revisión Manual

<20%,Aprobar

## Consideraciones Técnicas
Pipeline desacoplado: El proceso de entrenamiento genera archivos .pkl que la API consume de forma independiente.
Consistencia: El scaler entrenado se reutiliza en la API para garantizar que las predicciones en producción sigan la misma distribución que el entrenamiento.