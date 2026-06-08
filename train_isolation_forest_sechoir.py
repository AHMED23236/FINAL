import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score, roc_auc_score)
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR.parent / "cjo-maintenance-data-cleaned"
MODEL_DIR = BASE_DIR
DATA_FILE = DATA_DIR / "sechoir_capteurs_IF.csv"
FEATURES = [
    'Temp_Module1', 'Temp_Module2', 'Temp_Module3', 'Temp_Bruleur3',
    'Pression_EAU', 'Vitesse_Tapis',
    'Temp_Module1_mean10', 'Temp_Module1_std10', 'Temp_Module1_diff1',
    'Temp_Module2_mean10', 'Temp_Module2_std10', 'Temp_Module2_diff1',
    'Temp_Module3_mean10', 'Temp_Module3_std10', 'Temp_Module3_diff1',
    'Temp_Bruleur3_mean10','Temp_Bruleur3_std10','Temp_Bruleur3_diff1',
    'Temp_Module1_dev', 'Temp_Module2_dev', 'Temp_Module3_dev', 'Temp_Bruleur3_dev',
    'temp_mean_all', 'temp_max_all', 'temp_std_all', 'temp_range',
    'Pression_EAU_mean10', 'Vitesse_Tapis_mean10',
    'hour_sin', 'hour_cos', 'is_night', 'is_morning',
    'burner_manuel',
]
CONTAMINATION = 0.16
print("=" * 60)
print("  Isolation Forest — Séchoir Céramique")
print(f"  Contamination : {CONTAMINATION}")
print("=" * 60)
print("\n[1] Chargement des données...")
df = pd.read_csv(DATA_FILE, parse_dates=['timestamp'])
print(f"    {len(df):,} lignes | {len(df.columns)} colonnes")
X = df[FEATURES].fillna(0).values
y = df['anomaly_label'].values
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_test = y[split:]
print(f"    Train : {split:,} | Test : {len(X_test):,}")
print(f"    Anomalies test : {y_test.sum():,} ({y_test.mean()*100:.1f}%)")
print("\n[2] Normalisation...")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("\n[3] Entraînement Isolation Forest...")
model = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    max_samples='auto',
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_sc)
print("\n[4] Prédiction sur le jeu de test...")
raw_pred = model.predict(X_test_sc)
y_pred   = (raw_pred == -1).astype(int)
scores   = -model.score_samples(X_test_sc)
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average='weighted')
try:
    auc = roc_auc_score(y_test, scores)
except Exception:
    auc = float('nan')
print(f"\n  Accuracy : {acc*100:.1f}%")
print(f"  F1       : {f1:.4f}")
print(f"  AUC-ROC  : {auc:.4f}")
print("\n" + classification_report(y_test, y_pred, target_names=['Normal','Anomalie']))
print("[5] Sauvegarde modèle & scaler...")
joblib.dump(model,  MODEL_DIR / "isolation_forest_sechoir.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler_if_sechoir.pkl")
print("    isolation_forest_sechoir.pkl")
print("    scaler_if_sechoir.pkl")
print("\n[6] Sauvegarde résultats CSV...")
df_test = df.iloc[split:].copy().reset_index(drop=True)
df_test['anomaly_score']  = scores
df_test['anomaly_pred']   = y_pred
df_test['true_label']     = y_test
df_test['correct']        = (y_pred == y_test).astype(int)
out_csv = DATA_DIR / "sechoir_anomalies_results.csv"
df_test[['timestamp','failure_type','true_label','anomaly_pred',
         'anomaly_score','correct'] + FEATURES[:6]].to_csv(out_csv, index=False)
print(f"    sechoir_anomalies_results.csv ({len(df_test):,} lignes)")
print("\n[7] Génération des graphiques...")
fig, axes = plt.subplots(3, 1, figsize=(16, 12))
fig.patch.set_facecolor('#0f172a')
for ax in axes:
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#94a3b8')
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    ax.title.set_color('#f1f5f9')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
ax = axes[0]
t = df_test['timestamp']
ax.plot(t, scores, color='#3B82F6', linewidth=0.6, alpha=0.8, label='Score anomalie')
anomalies_vraies = df_test[df_test['true_label'] == 1]
ax.scatter(anomalies_vraies['timestamp'], anomalies_vraies['anomaly_score'],
           color='#EF4444', s=3, alpha=0.5, label='Vraie anomalie', zorder=3)
threshold = np.percentile(scores, (1 - CONTAMINATION) * 100)
ax.axhline(threshold, color='#F97316', linestyle='--', linewidth=1, label=f'Seuil ({threshold:.3f})')
ax.set_title('Score d\'anomalie — Isolation Forest')
ax.set_ylabel('Score')
ax.legend(facecolor='#1e293b', labelcolor='#94a3b8', fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
ax = axes[1]
cm = confusion_matrix(y_test, y_pred)
im = ax.imshow(cm, cmap='Blues', aspect='auto')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else '#0f172a', fontsize=14, fontweight='bold')
ax.set_xticks([0, 1]); ax.set_xticklabels(['Normal', 'Anomalie'], color='#94a3b8')
ax.set_yticks([0, 1]); ax.set_yticklabels(['Normal', 'Anomalie'], color='#94a3b8')
ax.set_xlabel('Prédit'); ax.set_ylabel('Réel')
ax.set_title(f'Matrice de Confusion — Acc={acc*100:.1f}% | F1={f1:.3f} | AUC={auc:.3f}')
ax = axes[2]
sample = df_test.iloc[::5]
ax.plot(sample['timestamp'], sample['Temp_Module3'], color='#8B5CF6',
        linewidth=0.8, alpha=0.9, label='Temp_Module3')
ax.plot(sample['timestamp'], sample['Temp_Bruleur3'], color='#F97316',
        linewidth=0.8, alpha=0.9, label='Temp_Bruleur3')
detected = df_test[df_test['anomaly_pred'] == 1]
ax.scatter(detected['timestamp'], detected['Temp_Module3'],
           color='#EF4444', s=4, alpha=0.4, label='Anomalie détectée', zorder=3)
ax.set_title('Températures avec anomalies détectées')
ax.set_ylabel('Température (°C)')
ax.legend(facecolor='#1e293b', labelcolor='#94a3b8', fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
plt.tight_layout(pad=2)
out_png = MODEL_DIR / "isolation_forest_sechoir_results.png"
plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("    isolation_forest_sechoir_results.png")
print("\n" + "=" * 60)
print("  RÉSUMÉ FINAL — Isolation Forest Séchoir")
print("=" * 60)
print(f"  Accuracy Test  : {acc*100:.1f}%")
print(f"  F1 Test        : {f1:.4f}")
print(f"  AUC-ROC        : {auc:.4f}")
print(f"  Contamination  : {CONTAMINATION}")
print(f"  Features       : {len(FEATURES)}")
print(f"\n  Fichiers sauvegardés :")
print(f"    isolation_forest_sechoir.pkl")
print(f"    scaler_if_sechoir.pkl")
print(f"    sechoir_anomalies_results.csv")
print(f"    isolation_forest_sechoir_results.png")
print("=" * 60)

