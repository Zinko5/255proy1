import os
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import mlflow
import mlflow.sklearn
from datetime import datetime
import sacar_metricas

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# [CRISP-DM] Fase 1 & 2: Comprensión del Negocio y Datos
# ---------------------------------------------------------

# Métricas que requieren normalización por tiempo (per Minute)
BASE_METRICS = [
    'goldEarned', 
    'totalMinionsKilled', 
    'totalDamageDealtToChampions', 
    'totalDamageTaken', 
    'damageDealtToEpicMonsters', 
    'damageDealtToTurrets',
    'kills',
    'deaths',
    'assists',
    'visionScore'
]

# Métricas numéricas acumulativas o de estado

OTHER_NUMERIC = [
    # 'champLevel',
    'challenge_teamRiftHeraldKills', 
    'challenge_teamBaronKills', 
    'challenge_teamElderDragonKills', 
    'challenge_highestChampionDamage', 
    'challenge_killParticipation',          
    'challenge_laningPhaseGoldExpAdvantage',
    'challenge_teamDamagePercentage',       
    'totalPings'                            
]

ROLES = ['ALL', 'TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']

def train_and_save_role(df_source, role):
    """
    Entrena modelos multialgoritmo y calcula importancias (Incluso para Cajas Negras).
    """
    print(f"\n--- ENTRENANDO ROL: {role} ---")
    
    df = df_source.copy()
    if role != 'ALL':
        df = df[df['individualPosition'] == role]
    
    if len(df) < 50: return

    feature_cols = [f'{m}_perMin' for m in BASE_METRICS] + OTHER_NUMERIC
    X = df[feature_cols]
    y = df['win'].astype(int)
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # [CRISP-DM] Fase 4: Modelado - Optimización de hiperparámetros
    # Búsqueda dinámica del mejor K para KNN
    print(f"  -> Optimizando KNN (GridSearch)...")
    knn_params = {'n_neighbors': [3, 5, 7, 9, 11, 15, 21]}
    knn_search = GridSearchCV(KNeighborsClassifier(), knn_params, cv=3, scoring='accuracy')
    knn_search.fit(X_train, y_train)
    best_k = knn_search.best_params_['n_neighbors']
    print(f"     Mejor K para {role}: {best_k}")

    models = {
        "lr": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=100, random_state=42),
        "xgb": XGBClassifier(eval_metric='logloss', random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=best_k),
        "lda": LinearDiscriminantAnalysis(),
        "nb": GaussianNB(),
        "dt": DecisionTreeClassifier(max_depth=5, random_state=42),
        "svm": CalibratedClassifierCV(LinearSVC(dual=False, random_state=42))
    }
    
    importances_dict = {}
    
    for name, model in models.items():
        print(f"  -> Entrenando {name.upper()}...")
        model.fit(X_train, y_train)
        
        # [CRISP-DM] Fase 5: Evaluación y Explicabilidad (Permutación para todos)
        # Esto unifica cómo extraemos importancia, incluso para KNN o NB.
        print(f"     Calculando importancia (Permutación) para {name}...")
        r = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
        importances_dict[name] = r.importances_mean.tolist()

    # Red Neuronal (MLP) - Manejo de Importancia vía Permutación
    print("  -> Entrenando MLP (Keras)...")
    mlp = Sequential([
        Input(shape=(len(feature_cols),)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    mlp.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
    
    # Wrapper para que permutation_importance funcione con Keras
    class KerasScikitWrapper:
        def __init__(self, model):
            self.model = model
            self._estimator_type = "classifier"
            self.classes_ = np.array([0, 1])
        def predict(self, X):
            return (self.model.predict(X, verbose=0) > 0.5).astype(int).flatten()
        def fit(self, X, y): # Requerido por Scikit-learn
            pass
        def score(self, X, y):
            y_pred = self.predict(X)
            return (y_pred == y).mean()

    print(f"     Calculando importancia (Permutación) para MLP...")
    r_mlp = permutation_importance(KerasScikitWrapper(mlp), X_test, y_test, n_repeats=3, random_state=42)
    importances_dict["mlp"] = r_mlp.importances_mean.tolist()

    # [CRISP-DM] Fase 6: Despliegue
    base_path = f"modelos/{role}"
    if not os.path.exists(base_path): os.makedirs(base_path)
    
    joblib.dump(scaler, f"{base_path}/scaler.joblib")
    joblib.dump(feature_cols, f"{base_path}/feature_names.joblib")
    joblib.dump(importances_dict, f"{base_path}/importances.joblib") # Diccionario unificado
    
    for name, model in models.items():
        joblib.dump(model, f"{base_path}/{name}.joblib")
    mlp.save(f"{base_path}/mlp.keras")
    
    print(f"✅ Artefactos guardados para {role}.")

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # [MLOPS] Iniciar Experimento
    mlflow.set_experiment("League_Learning_AI")
    
    with mlflow.start_run(run_name=f"Training_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"):
        df = pd.read_csv('../data/league_data.csv')
        df = df[df['challenge_hadAfkTeammate'] == 0]
        df = df[df['timePlayed'] > 420]
        
        duration_min = df['timePlayed'] / 60
        for metric in BASE_METRICS:
            df[f'{metric}_perMin'] = df[metric] / duration_min
        
        # Log de los datasets usados (Trazabilidad completa: Solo league_data importa para reproducir)
        mlflow.log_artifact('../data/league_data.csv', 'data_snapshot')
        
        for role in ROLES:
            train_and_save_role(df, role)
            
        # [MLOPS] Generar y logear métricas automáticamente
        print("\n--- GENERANDO MÉTRICAS FINALES ---")
        sacar_metricas.main()
        
        # Log de artefactos (Modelos y Métricas)
        mlflow.log_artifacts('../modelos', 'modelos')
        mlflow.log_artifacts('../metricas', 'metricas')
        
        print(f"\n✅ Pipeline MLOps completado. Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    main()
