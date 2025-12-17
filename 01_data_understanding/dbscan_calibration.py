import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Rutas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data")

df_application = pd.read_parquet(os.path.join(DATA_PATH, "application_.parquet"))
df_bureau = pd.read_parquet(os.path.join(DATA_PATH, "bureau.parquet"))
df_bureau_balance = pd.read_parquet(os.path.join(DATA_PATH, "bureau_balance.parquet"))

#Variables
agg_bureau = {
    'DAYS_CREDIT': ['min', 'max', 'mean'],
    'AMT_CREDIT_SUM': ['sum', 'mean', 'max'],
    'AMT_CREDIT_SUM_DEBT': ['sum', 'mean'],
    'CREDIT_DAY_OVERDUE': ['max'],
    'SK_ID_BUREAU': ['count']
}

bureau_agg = df_bureau.groupby('SK_ID_CURR').agg(agg_bureau)
bureau_agg.columns = ['BUREAU_' + '_'.join(col).upper() for col in bureau_agg.columns]

df_application = df_application.merge(
    bureau_agg,
    on='SK_ID_CURR',
    how='left'
)

bb_agg = df_bureau_balance.groupby('SK_ID_BUREAU').agg(
    BB_MESES_MAX=('MONTHS_BALANCE', 'max'),
    BB_MESES_COUNT=('MONTHS_BALANCE', 'count'),
    BB_MORA_COUNT=('STATUS', lambda x: (x > '0').sum())
).reset_index()

df_bureau_merged = df_bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')

agg_bureau_final = {
    'DAYS_CREDIT': ['min', 'mean'],
    'AMT_CREDIT_SUM_DEBT': ['sum', 'mean'],
    'BB_MORA_COUNT': ['sum', 'mean', 'max'],
    'BB_MESES_COUNT': ['sum', 'mean'],
    'SK_ID_BUREAU': ['count']
}

bureau_balance_final_agg = df_bureau_merged.groupby('SK_ID_CURR').agg(agg_bureau_final)
bureau_balance_final_agg.columns = [
    'BUREAU_BB_' + '_'.join(col).upper()
    for col in bureau_balance_final_agg.columns
]

df_application = df_application.merge(
    bureau_balance_final_agg,
    on='SK_ID_CURR',
    how='left'
)

numerical_features = [
    'AMT_INCOME_TOTAL',
    'AMT_CREDIT',
    'DAYS_BIRTH',
    'DAYS_EMPLOYED',
]

bureau_features = [
    col for col in df_application.columns
    if col.startswith('BUREAU_') or col.startswith('BUREAU_BB_')
]

features_for_clustering = numerical_features + bureau_features

df_cluster = df_application[features_for_clustering].copy()

print(f"Número de variables para DBSCAN: {len(df_cluster.columns)}")

for col in numerical_features:
    df_cluster[col].fillna(df_cluster[col].median(), inplace=True)

for col in bureau_features:
    df_cluster[col].fillna(0, inplace=True)

print("NaNs restantes:", df_cluster.isnull().sum().sum())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

print("Shape X_scaled:", X_scaled.shape)

D = X_scaled.shape[1]
MIN_PTS = round(D * 1.25)

print("Dimensión:", D)
print("min_samples:", MIN_PTS)

neighbors = NearestNeighbors(n_neighbors=MIN_PTS)
neighbors_fit = neighbors.fit(X_scaled)
distances, _ = neighbors_fit.kneighbors(X_scaled)

distances = np.sort(distances[:, MIN_PTS - 1])

#Grafico

plt.ylim(0, 2)
plt.plot(distances)
plt.title("K-distance plot para DBSCAN")
plt.xlabel("Observaciones ordenadas")
plt.ylabel("Distancia")
plt.grid(True)
plt.show(block=True)
