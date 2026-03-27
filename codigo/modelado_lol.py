import os
# Silenciar logs de TensorFlow/CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import load_model
import joblib
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONFIGURACIÓN DE FEATURES (MISMAS QUE EN APP.PY)
# ---------------------------------------------------------
NUMERIC_FEATURES = [
    'champLevel', 'kills', 'deaths', 'assists', 'goldEarned', 'totalMinionsKilled', 'visionScore', 
    'totalDamageDealtToChampions', 'totalDamageTaken', 'damageDealtToEpicMonsters', 
    'damageDealtToTurrets', 'challenge_teamRiftHeraldKills', 
    'challenge_teamBaronKills', 'challenge_teamElderDragonKills', 
    'challenge_highestChampionDamage', 'challenge_killParticipation', 
    'challenge_laningPhaseGoldExpAdvantage', 'challenge_teamDamagePercentage', 'totalPings'
]

ROLES = ['ALL', 'TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']

def train_and_save_role(df_full, role):
    print(f"\n--- ENTRENANDO ROL: {role} ---")
    
    # 1. Filtrado
    if role == 'ALL':
        df = df_full
    else:
        df = df_full[df_full['individualPosition'] == role]
    
    if len(df) < 50:
        print(f"⚠️  Datos insuficientes para {role} ({len(df)} registros). Saltando...")
        return
    
    X = df[NUMERIC_FEATURES]
    y = df['win'].astype(int)
    
    # 2. Preparación
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.1, random_state=42)
    
    # 3. Directorio de guardado
    base_path = f"modelos/{role}"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    # Guardar Escalador (CRÍTICO)
    joblib.dump(scaler, f"{base_path}/scaler.joblib")
    
    # 4. Entrenamiento y Guardado de Modelos
    
    # Logistic Regression
    print("  -> Regresión Logística...")
    lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    joblib.dump(lr, f"{base_path}/lr.joblib")
    
    # Random Forest
    print("  -> Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    joblib.dump(rf, f"{base_path}/rf.joblib")
    
    # XGBoost
    print("  -> XGBoost...")
    xgb = XGBClassifier(eval_metric='logloss', random_state=42).fit(X_train, y_train)
    joblib.dump(xgb, f"{base_path}/xgb.joblib")
    
    # SVM Linear (para importancia)
    print("  -> SVM...")
    svm = SVC(probability=True, kernel='linear', C=0.01).fit(X_train, y_train)
    joblib.dump(svm, f"{base_path}/svm.joblib")
    
    # KNN
    print("  -> KNN...")
    knn = KNeighborsClassifier(n_neighbors=15).fit(X_train, y_train)
    joblib.dump(knn, f"{base_path}/knn.joblib")
    
    # Red Neuronal (MLP)
    print("  -> Red Neuronal (Keras)...")
    mlp = Sequential([
        Input(shape=(len(NUMERIC_FEATURES),)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    mlp.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    mlp.save(f"{base_path}/mlp.keras")
    
    print(f"✅  Modelos guardados en {base_path}/")

def main():
    print("🚀 PROCESO DE ENTRENAMIENTO DE PRODUCCIÓN 🚀")
    
    # Cambiar al directorio del script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    if not os.path.exists('../league_data.csv'):
        print("❌ Error: No se encuentra 'league_data.csv'")
        return
        
    df = pd.read_csv('../league_data.csv')
    df = df[df['challenge_hadAfkTeammate'] == 0]
    df = df[df['timePlayed'] > 420]
    
    if not os.path.exists('modelos'):
        os.makedirs('modelos')
        
    for role in ROLES:
        train_and_save_role(df, role)
        
    print("\n🎉 ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE! 🎉")

if __name__ == "__main__":
    main()
