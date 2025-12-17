import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


# Rutas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "data_output")

os.makedirs(OUTPUT_PATH, exist_ok=True)


def integrate_and_clean():
    """
    Integración de application, bureau y bureau_balance,
    selección de variables (replica EA3),
    y detección de outliers con DBSCAN (offline).
    """

    print("Iniciando ETL: Integración, selección de variables y DBSCAN...")

    # 1. Carga de datos
    app = pd.read_parquet(os.path.join(DATA_PATH, "application_.parquet"))
    bureau = pd.read_parquet(os.path.join(DATA_PATH, "bureau.parquet"))
    bureau_balance = pd.read_parquet(os.path.join(DATA_PATH, "bureau_balance.parquet"))


    # 2. Agregaciones bureau_balance
    bb_agg = (
        bureau_balance
        .groupby("SK_ID_BUREAU")
        .agg(
            bb_months_mean=("MONTHS_BALANCE", "mean"),
            bb_status_nunique=("STATUS", "nunique")
        )
        .reset_index()
    )

    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")


    # 3. Agregaciones bureau
    bureau_agg = (
        bureau
        .groupby("SK_ID_CURR")
        .agg(
            bureau_credit_count=("SK_ID_BUREAU", "count"),
            bureau_credit_active_mean=("CREDIT_ACTIVE", "nunique"),
            bureau_days_credit_mean=("DAYS_CREDIT", "mean"),
            bureau_amt_credit_sum=("AMT_CREDIT_SUM", "sum")
        )
        .reset_index()
    )

    df = app.merge(bureau_agg, on="SK_ID_CURR", how="left")


    # 4. Selección de variables
    numerical_features = [
        'AMT_INCOME_TOTAL',
        'AMT_CREDIT',
        'DAYS_BIRTH',
        'DAYS_EMPLOYED',
    ]

    bureau_features = [
        'bureau_credit_count',
        'bureau_credit_active_mean',
        'bureau_days_credit_mean',
        'bureau_amt_credit_sum',
    ]

    selected_features = numerical_features + bureau_features + ['TARGET']

    df = df[selected_features].copy()


    # 5. Imputación simple
    for col in numerical_features:
        df[col] = df[col].fillna(df[col].median())

    for col in bureau_features:
        df[col] = df[col].fillna(0)


    # 6. DBSCAN 
    X = df.drop("TARGET", axis=1)
    y = df["TARGET"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    D = X_scaled.shape[1]
    MIN_PTS = round(D * 3)

    dbscan = DBSCAN(eps=1.18 , min_samples=MIN_PTS)
    labels = dbscan.fit_predict(X_scaled)

    df["DBSCAN_LABEL"] = labels

    outlier_ratio = (labels == -1).mean()
    print(f"Outliers detectados por DBSCAN: {outlier_ratio:.2%}")

    # Exception
    if outlier_ratio > 0.5:
        print("DBSCAN demasiado agresivo. Se conserva el dataset completo.")
        df_clean = df.drop(columns=["DBSCAN_LABEL"])
    else:
        df_clean = df[df["DBSCAN_LABEL"] != -1].drop(columns=["DBSCAN_LABEL"])


    # 7. Guardar resultados
    df_clean.to_parquet(
        os.path.join(OUTPUT_PATH, "cleaned_data.parquet"),
        index=False
    )

    X_train_final = df_clean.drop(columns=["TARGET"])
    y_train = df_clean["TARGET"]

    X_train_final.to_csv(
        os.path.join(OUTPUT_PATH, "X_train_final.csv"),
        index=False
    )
    y_train.to_csv(
        os.path.join(OUTPUT_PATH, "y_train.csv"),
        index=False
    )

    print("ETL completado correctamente.")
    print(f"Filas finales: {df_clean.shape[0]}")
    print(f"Features usadas: {X_train_final.shape[1]}")


if __name__ == "__main__":
    integrate_and_clean()
