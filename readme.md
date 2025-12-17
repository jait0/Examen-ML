# Credit Risk Prediction – Machine Learning Pipeline

Este proyecto desarrolla una solución profesional de Machine Learning para predecir la probabilidad de impago (*default*) de clientes solicitantes de crédito, utilizando información financiera y crediticia histórica.

La solución abarca todo el ciclo de vida del modelo bajo la metodología **CRISP-DM**:
* Análisis y preparación de datos.
* Eliminación de ruido mediante **clustering no supervisado (DBSCAN)**.
* Entrenamiento de un modelo supervisado (**Regresión Logística**).
* Evaluación del desempeño y métricas de negocio.
* Despliegue mediante una **API REST** escalable.

---

## Estructura del Proyecto

```text
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
´´´
## Fuentes de Datos
El sistema procesa información proveniente de tres fuentes clave ubicadas en la carpeta /data:application_.parquet: Datos demográficos e ingresos del solicitante.bureau.parquet: Historial crediticio externo (Buró).bureau_balance.parquet: Detalle mensual de estados de cuenta externos.🛠️ Instalación y RequisitosPython 3.9+Recomendado: Uso de entorno virtual.Bash# Crear entorno virtual
python -m venv venv

# Activar (Windows)
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
Ejecución del Pipeline (Paso a Paso)1. Calibración de DBSCANIdentifica y calibra los parámetros para la detección de ruido y outliers.Bashpython 01_data_understanding/dbscan_calibration.py
2. Integración y Limpieza (ETL)Combina los archivos Parquet, selecciona variables y aplica DBSCAN para eliminar ruido.Bashpython 02_data_preparation/integrate_and_clean.py
3. Entrenamiento del ModeloAplica escalado de variables y entrena la Regresión Logística.Bashpython 03_modeling/train_model.py
4. Evaluación de DesempeñoGenera la Curva ROC, Matriz de Confusión y reporte de clasificación en la carpeta /reports.Bashpython 04_evaluation/evaluate_model.py
## Despliegue de la API
El sistema utiliza FastAPI para servir el modelo en tiempo real.Levantar el servicio:Bashuvicorn 05_deployment.app:app --reload
API Local: http://127.0.0.1:8000Documentación Interactiva (Swagger): http://127.0.0.1:8000/docsEndpoint Principal: POST /evaluate_riskRecibe la información del cliente y retorna el nivel de riesgo.Ejemplo de respuesta:JSON{
  "probabilidad_incumplimiento": "42.35%",
  "decision": "Revisar manualmente"
}
## Lógica de Decisión de Negocio
El sistema automatiza la toma de decisiones basada en los siguientes umbrales:Probabilidad (P)Decisión$P \geq 70\%$Rechazar$40\% \leq P < 70\%$Revisión Manual$P < 40\%$Aprobar💡 Consideraciones TécnicasDBSCAN: Se utiliza exclusivamente en la etapa de preparación (limpieza) para mejorar la calidad del entrenamiento, no se requiere en producción.Consistencia: El scaler entrenado se reutiliza en la API para garantizar que los datos de entrada sigan la misma distribución.Desacoplamiento: La API es independiente del proceso de entrenamiento, permitiendo actualizaciones del modelo sin afectar el servicio.