import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import os

# CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="LoL Analytics PRO - Model Driven",
    page_icon="🤖",
    layout="wide",
)

# ESTILOS CSS PROFESIONALES
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Outfit:wght@300;400;600&display=swap');
    .main { background: #0b0e11; color: #ffffff; }
    .stApp { background-color: #0b0e11; }
    h1, h2, h3 { color: #00f2ff !important; font-family: 'Orbitron', sans-serif; text-transform: uppercase; letter-spacing: 2px; }
    .hero-section { background: linear-gradient(135deg, rgba(0,242,255,0.1) 0%, rgba(0,0,0,0) 100%); padding: 40px; border-radius: 20px; border: 1px solid rgba(0,242,255,0.2); margin-bottom: 20px; text-align: center; }
    .leaderboard-card { background: rgba(255,255,255,0.03); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); margin: 5px 0; transition: 0.3s; }
    .leaderboard-card:hover { border-color: #00f2ff; transform: translateX(10px); }
    </style>
""", unsafe_allow_html=True)

# CARGA Y CACHE DE MODELOS
NUMERIC_FEATURES = [
    'champLevel', 'kills', 'deaths', 'assists', 'goldEarned', 'visionScore', 
    'totalDamageDealtToChampions', 'totalDamageTaken', 'damageDealtToEpicMonsters', 
    'turretKills', 'damageDealtToTurrets', 'challenge_teamRiftHeraldKills', 
    'challenge_teamBaronKills', 'challenge_teamElderDragonKills', 
    'challenge_highestChampionDamage', 'challenge_kda', 'challenge_killParticipation', 
    'challenge_laningPhaseGoldExpAdvantage', 'challenge_teamDamagePercentage', 'totalPings'
]

@st.cache_data
def load_data():
    df_m = pd.read_csv('../league_data.csv') if os.path.exists('../league_data.csv') else None
    df_p = pd.read_csv('../todosJugadores.csv') if os.path.exists('../todosJugadores.csv') else None
    if df_m is not None:
        df_m = df_m[df_m['challenge_hadAfkTeammate'] == 0]
        df_m = df_m[df_m['timePlayed'] > 420]
    return df_m, df_p

@st.cache_resource
def get_trained_models(df):
    X = df[NUMERIC_FEATURES]
    y = df['win'].astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    results = {}
    
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    results["Regresión Logística"] = {"model": lr, "importances": pd.Series(lr.coef_[0], index=NUMERIC_FEATURES).abs().sort_values(ascending=False)}
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
    results["Random Forest"] = {"model": rf, "importances": pd.Series(rf.feature_importances_, index=NUMERIC_FEATURES).sort_values(ascending=False)}
    
    # 3. XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X_train, y_train)
    results["XGBoost"] = {"model": xgb, "importances": pd.Series(xgb.feature_importances_, index=NUMERIC_FEATURES).sort_values(ascending=False)}
    
    # 4. KNN (Slower to predict, no feature importance)
    knn = KNeighborsClassifier(n_neighbors=15).fit(X_train, y_train)
    results["KNN"] = {"model": knn, "importances": pd.Series(0, index=NUMERIC_FEATURES)}
    
    # 5. SVM (Linear for importance)
    svm = SVC(probability=True, kernel='linear', C=0.01).fit(X_train, y_train)
    results["SVM"] = {"model": svm, "importances": pd.Series(svm.coef_[0], index=NUMERIC_FEATURES).abs().sort_values(ascending=False)}
    
    # 6. MLP (Neural Network) - Simpler version for speed
    mlp = Sequential([Dense(32, activation='relu', input_shape=(len(NUMERIC_FEATURES),)), Dense(1, activation='sigmoid')])
    mlp.compile(optimizer='adam', loss='binary_crossentropy')
    mlp.fit(X_train, y_train, epochs=5, batch_size=64, verbose=0)
    results["Red Neuronal (MLP)"] = {"model": mlp, "importances": pd.Series(0, index=NUMERIC_FEATURES)}
    
    return results, scaler

# APP LOGIC
df_matches, df_players = load_data()

if df_matches is not None:
    models_dict, main_scaler = get_trained_models(df_matches)
    
    # Navegación
    if 'viewing_profile' not in st.session_state: st.session_state.viewing_profile = None
    
    st.sidebar.header("🕹️ Navegación")
    if st.sidebar.button("🏠 Inicio"): st.session_state.viewing_profile = None
    
    mode = st.sidebar.radio("Sección:", ["🔥 Inicio / Rankings", "🧠 IA & Simulador", "📊 Metajuego", "👤 Perfil"])

    if st.session_state.viewing_profile is None and mode == "🔥 Inicio / Rankings":
        st.markdown("<div class='hero-section'><h1>🏆 LoL Analytics Engine</h1><p>Análisis Profesional Basado en Inteligencia Artificial</p></div>", unsafe_allow_html=True)
        col_s1, col_s2, col_s3 = st.columns([1, 2, 1])
        with col_s2:
            st.markdown("### 🔍 Buscar Invocador")
            p_names = sorted(df_players['nombre'].unique()) if df_players is not None else sorted(df_matches['jugador'].unique())
            search = st.selectbox("Selecciona un nombre:", ["---"] + p_names)
            if search != "---":
                st.session_state.viewing_profile = search
                st.rerun()
        
        st.divider()
        if df_players is not None:
            st.subheader("🔥 Top 5 Challenger")
            top_5 = df_players[df_players['tier'] == 'CHALLENGER'].sort_values('LPs actuales', ascending=False).head(5)
            for idx, row in top_5.iterrows():
                st.markdown(f"<div class='leaderboard-card'><b>{row['nombre']}</b> | Tier: {row['tier']} | <span style='color: #f7ff00;'>{row['LPs actuales']} LPs</span></div>", unsafe_allow_html=True)

    elif mode == "🧠 IA & Simulador":
        st.header("🧠 Inteligencia Artificial Multialgoritmo")
        col_m1, col_m2 = st.columns([1, 2])
        
        with col_m1:
            st.subheader("Configuración del Oráculo")
            selected_model = st.selectbox("Algoritmo de Predicción:", list(models_dict.keys()))
            st.write(f"Auditoría técnica de: **{selected_model}**")
            
            # Mostramos importancias reales por código
            imp_series = models_dict[selected_model]["importances"]
            if imp_series.sum() > 0:
                top_features = imp_series.head(6).index.tolist()
                st.write("**Top Impacto detectado:**")
                st.bar_chart(imp_series.head(10))
            else:
                top_features = ['kills', 'deaths', 'goldEarned', 'visionScore', 'totalDamageDealtToChampions', 'challenge_teamDamagePercentage']
                st.warning("Este modelo utiliza patrones no-lineales complejos (Caja Negra).")

        with col_m2:
            st.subheader(f"🔮 Simulador Dinámico: {selected_model}")
            st.markdown("Ajusta las variables que **este modelo** considera más determinantes.")
            
            inputs = {}
            cols = st.columns(2)
            for i, feat in enumerate(top_features):
                with cols[i % 2]:
                    # Obtenemos valores de referencia
                    min_val = float(df_matches[feat].min())
                    max_val = float(df_matches[feat].max())
                    mean_val = float(df_matches[feat].mean())
                    inputs[feat] = st.number_input(f"{feat}", min_val, max_val, mean_val)
            
            # Botón de Predicción
            if st.button("🚀 Generar Diagnóstico"):
                # Preparamos el vector de entrada (todos los 20 features)
                full_input = []
                for f in NUMERIC_FEATURES:
                    full_input.append(inputs.get(f, df_matches[f].mean()))
                
                input_scaled = main_scaler.transform([full_input])
                
                if selected_model == "Red Neuronal (MLP)":
                    prob = float(models_dict[selected_model]["model"].predict(input_scaled)[0][0])
                else:
                    prob = models_dict[selected_model]["model"].predict_proba(input_scaled)[0][1]
                
                st.markdown(f"## Probabilidad de Victoria: `{prob*100:.1f}%`")
                st.progress(prob)
                
                if prob > 0.6: st.success("Perfil de Victoria: Las métricas sugieren un desempeño dominante.")
                elif prob < 0.4: st.error("Riesgo de Derrota: Estas métricas coinciden con situaciones de desventaja crítica.")
                else: st.warning("Resultado Incierto: Juego cerrado o variables equilibradas.")

    elif mode == "📊 Metajuego":
        st.header("Metajuego & Tendencias")
        # Visualizaciones rápidas
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.box(df_matches, x='individualPosition', y='goldEarned', color='win', title="Oro por Posición")
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.scatter(df_matches, x='visionScore', y='champLevel', color='win', title="Visión vs Nivel")
            st.plotly_chart(fig2, use_container_width=True)

    elif st.session_state.viewing_profile is not None or mode == "👤 Perfil":
        p_name = st.session_state.viewing_profile if st.session_state.viewing_profile else st.selectbox("Invocador:", sorted(df_matches['jugador'].unique()))
        st.header(f"👤 Perfil: {p_name}")
        p_data = df_matches[df_matches['jugador'] == p_name]
        if not p_data.empty:
            st.metric("Partidas", len(p_data))
            st.dataframe(p_data[['championName', 'win', 'challenge_kda', 'goldEarned']].head(10))
        else:
            st.info("Sin datos adicionales.")

else:
    st.error("Error al cargar datos.")
