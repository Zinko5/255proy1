import os
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# Configuración
BASE_METRICS = [
    'goldEarned', 'totalMinionsKilled', 'totalDamageDealtToChampions', 
    'totalDamageTaken', 'damageDealtToEpicMonsters', 'damageDealtToTurrets',
    'kills', 'deaths', 'assists', 'visionScore'
]

OTHER_NUMERIC = [
    'challenge_teamRiftHeraldKills', 'challenge_teamBaronKills', 
    'challenge_teamElderDragonKills', 'challenge_highestChampionDamage', 
    'challenge_killParticipation', 'challenge_laningPhaseGoldExpAdvantage',
    'challenge_teamDamagePercentage', 'totalPings'
]

ROLES = ['ALL', 'TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
MODEL_NAMES = ["lr", "rf", "xgb", "knn", "lda", "nb", "dt", "svm"]

# Directorios de salida
OUTPUT_DIR = "metricas_resultados"
MATRICES_DIR = os.path.join(OUTPUT_DIR, "matrices")
os.makedirs(MATRICES_DIR, exist_ok=True)

def evaluate_role_models(df_source, role):
    print(f"\n--- EVALUANDO ROL: {role} ---")
    
    df = df_source.copy()
    if role != 'ALL':
        df = df[df['individualPosition'] == role]
    
    if len(df) < 50: 
        print(f"Pocos datos para {role}, saltando...")
        return None

    feature_cols = [f'{m}_perMin' for m in BASE_METRICS] + OTHER_NUMERIC
    X = df[feature_cols]
    y = df['win'].astype(int)
    
    # Mismo split que en modelado_lol.py
    _, X_test_raw, _, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    base_path = f"modelos/{role}"
    if not os.path.exists(base_path):
        print(f"No se encontró el directorio de modelos para {role}")
        return None
        
    scaler = joblib.load(f"{base_path}/scaler.joblib")
    X_test = scaler.transform(X_test_raw)
    
    role_metrics = {}

    # Evaluar modelos Scikit-learn/XGBoost
    for name in MODEL_NAMES:
        model_path = f"{base_path}/{name}.joblib"
        if not os.path.exists(model_path):
            continue
            
        print(f"  -> Evaluando {name.upper()}...")
        model = joblib.load(model_path)
        
        # Predicciones
        y_pred = model.predict(X_test)
        
        # Probabilidades para ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred # Fallback
            
        # Calcular métricas
        m = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Presición": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob)
        }
        role_metrics[name] = m
        
        # Generar Matriz de Confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f"Matriz de Confusión - {role} - {name.upper()}")
        plt.xlabel("Predicho")
        plt.ylabel("Real")
        plt.savefig(os.path.join(MATRICES_DIR, f"{role}_{name}.png"))
        plt.close()

    # Evaluar MLP (Keras)
    mlp_path = f"{base_path}/mlp.keras"
    if os.path.exists(mlp_path):
        print("  -> Evaluando MLP...")
        mlp = load_model(mlp_path)
        y_prob = mlp.predict(X_test, verbose=0).flatten()
        y_pred = (y_prob > 0.5).astype(int)
        
        m = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Presición": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob)
        }
        role_metrics["mlp"] = m
        
        # Matriz MLP
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f"Matriz de Confusión - {role} - MLP")
        plt.xlabel("Predicho")
        plt.ylabel("Real")
        plt.savefig(os.path.join(MATRICES_DIR, f"{role}_mlp.png"))
        plt.close()

    return role_metrics

def main():
    # Cambiar al directorio del script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    csv_path = '../data/league_data.csv'
    if not os.path.exists(csv_path):
        print(f"Error: No se encontró {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    df = df[df['challenge_hadAfkTeammate'] == 0]
    df = df[df['timePlayed'] > 420]
    
    duration_min = df['timePlayed'] / 60
    for metric in BASE_METRICS:
        df[f'{metric}_perMin'] = df[metric] / duration_min
    
    all_results = {}
    for role in ROLES:
        res = evaluate_role_models(df, role)
        if res:
            all_results[role] = res
            
    # Guardar resultados en JSON
    with open(os.path.join(OUTPUT_DIR, "metricas_modelos.json"), "w") as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\n✅ Evaluación completada. Resultados guardados en '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()
