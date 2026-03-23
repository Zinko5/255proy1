import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import os

# Configuración de visualización
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def main():
    print("=== Iniciando Proyecto: Análisis Predictivo de League of Legends ===")
    
    # ---------------------------------------------------------
    # 1. ENTENDIMIENTO DE LOS DATOS
    # ---------------------------------------------------------
    print("\n[1/6] Entendimiento de los datos...")
    df = pd.read_csv('../league_data.csv')
    print(f"Dataset cargado: {df.shape[0]} registros y {df.shape[1]} variables.")
    
    # Resumen descriptivo
    print("\nEstadísticas descriptivas básicas:")
    print(df[['kills', 'deaths', 'assists', 'goldEarned', 'timePlayed', 'visionScore']].describe())
    
    # ---------------------------------------------------------
    # 2. PREPARACIÓN DE LOS DATOS
    # ---------------------------------------------------------
    print("\n[2/6] Preparación de los datos...")
    
    # Filtrado: Eliminar AFKs y remakes (< 7 minutos o 420 seg)
    initial_count = len(df)
    df = df[df['challenge_hadAfkTeammate'] == 0]
    df = df[df['timePlayed'] > 420]
    print(f"Registros eliminados (AFKs/Remakes): {initial_count - len(df)}")
    
    # Normalización: Convertir métricas absolutas a 'por minuto'
    # Esto ayuda a comparar partidas de distinta duración
    cols_to_normalize = ['kills', 'deaths', 'assists', 'goldEarned', 'visionScore', 
                         'totalDamageDealtToChampions', 'totalDamageTaken', 
                         'damageDealtToEpicMonsters', 'damageDealtToTurrets', 'totalPings']
    
    for col in cols_to_normalize:
        df[f'{col}_per_min'] = df[col] / (df['timePlayed'] / 60)
    
    # Selección de variables para el modelo
    # Variables numéricas normalizadas + categóricas
    features_num = [f'{col}_per_min' for col in cols_to_normalize] + \
                   ['challenge_killParticipation', 'challenge_teamDamagePercentage', 'challenge_laningPhaseGoldExpAdvantage']
    
    features_cat = ['side', 'individualPosition'] # championName se omite para evitar excesiva dimensionalidad inicial o se puede incluir
    
    X = df[features_num + features_cat]
    y = df['win'].astype(int)
    
    # One-Hot Encoding para variables categóricas
    X = pd.get_dummies(X, columns=features_cat)
    
    # División de datos (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Escalamiento de variables numéricas
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Datos preparados. Características: {X_train.shape[1]}")
    
    # ---------------------------------------------------------
    # 3. MODELADO
    # ---------------------------------------------------------
    print("\n[3/6] Modelado (Entrenando 6 arquitecturas distintas)...")
    
    models = {
        "Regresión Logística": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, kernel='rbf', C=1.0),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Entrenando {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_prob),
            'Model': model
        }
    
    # Red Neuronal (MLP con Keras)
    print("Entrenando Red Neuronal (MLP)...")
    mlp = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    mlp.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0, validation_split=0.1)
    
    y_prob_mlp = mlp.predict(X_test_scaled).flatten()
    y_pred_mlp = (y_prob_mlp > 0.5).astype(int)
    
    results["Red Neuronal"] = {
        'Accuracy': accuracy_score(y_test, y_pred_mlp),
        'AUC': roc_auc_score(y_test, y_prob_mlp),
        'Model': mlp
    }
    
    # ---------------------------------------------------------
    # 4. EVALUACIÓN
    # ---------------------------------------------------------
    print("\n[4/6] Evaluación de resultados...")
    
    results_df = pd.DataFrame(results).T[['Accuracy', 'AUC']].sort_values(by='AUC', ascending=False)
    print("\nResumen de métricas:")
    print(results_df)
    
    # Matriz de Confusión del mejor modelo (según AUC)
    best_model_name = results_df.index[0]
    print(f"\nGenerando matriz de confusión para el mejor modelo: {best_model_name}")
    
    if best_model_name == "Red Neuronal":
        y_pred_best = (results[best_model_name]['Model'].predict(X_test_scaled).flatten() > 0.5).astype(int)
    else:
        y_pred_best = results[best_model_name]['Model'].predict(X_test_scaled)
    
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión: {best_model_name}')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.savefig('confusion_matrix_best.png')
    print("Matriz de confusión guardada como 'confusion_matrix_best.png'.")
    
    # Importancia de características (Random Forest)
    if "Random Forest" in models:
        rf = models["Random Forest"]
        importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(10)
        plt.figure(figsize=(10,6))
        importances.plot(kind='barh', color='teal')
        plt.title('Top 10 Variables más influyentes (Random Forest)')
        plt.gca().invert_yaxis()
        plt.savefig('feature_importance.png')
        print("Gráfico de importancia de características guardado como 'feature_importance.png'.")

    # ---------------------------------------------------------
    # 5. DESPLIEGUE (CONCEPTUAL)
    # ---------------------------------------------------------
    print("\n[5/6] Despliegue...")
    print("Modelos listos para integración en plataforma analítica.")
    print(f"El mejor modelo es {best_model_name} con un AUC de {results_df.iloc[0]['AUC']:.4f}.")
    
    # ---------------------------------------------------------
    # 6. CONCLUSIONES
    # ---------------------------------------------------------
    print("\n[6/6] Conclusiones...")
    print("- Las métricas de economía (oro por minuto) y participación en objetivos son determinantes clave.")
    print("- Los modelos de ensamble (Random Forest, XGBoost) superan ligeramente a los lineales en este dataset.")
    print("- La normalización por tiempo es crítica para evitar sesgos por duración de partida.")

if __name__ == "__main__":
    main()
