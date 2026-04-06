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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import urllib.parse
import joblib
from tensorflow.keras.models import load_model
import json
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import math

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="League Learning - AI Analytics",
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
    'champLevel', 'kills', 'deaths', 'assists', 'goldEarned', 'totalMinionsKilled', 'visionScore', 
    'totalDamageDealtToChampions', 'totalDamageTaken', 'damageDealtToEpicMonsters', 
    'damageDealtToTurrets', 'challenge_teamRiftHeraldKills', 
    'challenge_teamBaronKills', 'challenge_teamElderDragonKills', 
    'challenge_highestChampionDamage', 'challenge_killParticipation', 
    'challenge_laningPhaseGoldExpAdvantage', 'challenge_teamDamagePercentage', 'totalPings'
]
INT_FEATURES = {
    'champLevel', 'kills', 'deaths', 'assists', 'goldEarned', 'totalMinionsKilled', 
    'visionScore', 'totalDamageDealtToChampions', 'totalDamageTaken', 
    'damageDealtToEpicMonsters', 'damageDealtToTurrets', 
    'challenge_teamRiftHeraldKills', 'challenge_teamBaronKills', 
    'challenge_teamElderDragonKills', 'challenge_highestChampionDamage', 'totalPings'
}
FEATURE_LABELS = {
    'champLevel': 'Nivel del Campeón',
    'kills': 'Asesinatos',
    'deaths': 'Muertes',
    'assists': 'Asistencias',
    'goldEarned': 'Oro Ganado',
    'totalMinionsKilled': 'Minions Asesinados',
    'visionScore': 'Score de Visión',
    'totalDamageDealtToChampions': 'Daño a Campeones',
    'totalDamageTaken': 'Daño Recibido',
    'damageDealtToEpicMonsters': 'Daño a Objetivos Épicos',
    'damageDealtToTurrets': 'Daño a Torretas',
    'challenge_teamRiftHeraldKills': 'Heraldos (Equipo)',
    'challenge_teamBaronKills': 'Barones (Equipo)',
    'challenge_teamElderDragonKills': 'Dragones Ancianos (Equipo)',
    'challenge_highestChampionDamage': 'Daño Máximo Producido',
    'challenge_killParticipation': 'Participación en muertes',
    'challenge_laningPhaseGoldExpAdvantage': 'Ventaja en Fase de Líneas',
    'challenge_teamDamagePercentage': '% de Daño del Equipo',
    'totalPings': 'Pings totales',
    'challenge_damagePerMinute': 'Daño por Minuto',
    'challenge_goldPerMinute': 'Oro por Minuto',
    'timePlayed': 'Tiempo Jugado (seg)',
    'side': 'Lado del Mapa',
    'individualPosition': 'Posición',
    'championName': 'Campeón',
    'win': 'Resultado',
    'match_id': 'ID de Partida',
    'jugador': 'Invocador',
    'player_id': 'ID de Jugador',
    'gameCreation': 'Fecha de Creación',
    'challenge_hadAfkTeammate': 'Compañero AFK'
}

def get_feature_label(name):
    """Retorna el nombre en lenguaje natural de una variable, manejando sufijos _perMin."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name.endswith('_perMin'):
        base = name.replace('_perMin', '')
        if base in FEATURE_LABELS:
            return f"{FEATURE_LABELS[base]} (por min)"
        return f"{base.capitalize()} (por min)"
    return name.replace('_', ' ').capitalize()

# [CRISP-DM] Sincronización de preparación de datos
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

# --- MLOPS UTILS ---
def get_available_versions():
    client = MlflowClient()
    try:
        experiment = client.get_experiment_by_name("League_Learning_AI")
        if not experiment: return []
        # Corregido: experiment_ids espera una lista de strings
        runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["attributes.start_time DESC"])
        return [{"name": r.data.tags.get("mlflow.runName", r.info.run_id), "id": r.info.run_id, "start_time": r.info.start_time} for r in runs]
    except:
        return []

@st.cache_resource
def get_version_assets(run_id):
    if not run_id: return None, None, None
    client = MlflowClient()
    try:
        modelos_path = client.download_artifacts(run_id, "modelos")
        metricas_path = client.download_artifacts(run_id, "metricas")
        data_folder = client.download_artifacts(run_id, "data_snapshot")
        return modelos_path, metricas_path, data_folder
    except:
        return None, None, None

@st.cache_data
def load_data(run_id=None):
    # El archivo de jugadores siempre lo leemos local para mantener el ranking/perfiles actualizados
    # e independientes de la versión de inteligencia seleccionada (ya que es solo estética).
    path_p = '../data/todosJugadores.csv'
    path_m = '../data/league_data.csv'
    
    if run_id:
        _, _, mlflow_data_path = get_version_assets(run_id)
        if mlflow_data_path:
            path_m = os.path.join(mlflow_data_path, "league_data.csv")
            # No sobreescribimos path_p con el de MLflow para cumplir con el deseo del usuario
        
    df_m = pd.read_csv(path_m) if os.path.exists(path_m) else None
    df_p = pd.read_csv(path_p) if os.path.exists(path_p) else None
    
    name_to_id = {}
    if df_p is not None:
        if 'fecha_actualizacion' in df_p.columns:
            df_p['fecha_actualizacion'] = pd.to_datetime(df_p['fecha_actualizacion'])
            df_p = df_p.sort_values('fecha_actualizacion')
        name_to_id = df_p.set_index('nombre')['id'].to_dict()
        df_p_unique = df_p.drop_duplicates(subset='id', keep='last').copy()
        id_to_canonical_name = df_p_unique.set_index('id')['nombre'].to_dict()
        
        if df_m is not None:
            df_m['player_id'] = df_m['jugador'].map(name_to_id)
            df_m['player_id'] = df_m['player_id'].fillna(df_m['jugador'])
            df_m['jugador'] = df_m['player_id'].map(id_to_canonical_name).fillna(df_m['jugador'])
        df_p = df_p_unique

    if df_m is not None:
        df_m = df_m[df_m['challenge_hadAfkTeammate'] == 0]
        df_m = df_m[df_m['timePlayed'] > 420]
        duration_min = df_m['timePlayed'] / 60
        for metric in BASE_METRICS:
            df_m[f'{metric}_perMin'] = df_m[metric] / duration_min
    return df_m, df_p, name_to_id

@st.cache_data
def load_model_metrics(run_id=None):
    path = "../metricas/metricas_modelos.json"
    if run_id:
        _, mlflow_metricas, _ = get_version_assets(run_id)
        if mlflow_metricas: path = os.path.join(mlflow_metricas, "metricas_modelos.json")
    
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def get_best_model_roc(role_metrics):
    if not role_metrics: return None
    best_name = None
    best_roc = -1
    for m_name, metrics in role_metrics.items():
        if "ROC-AUC" in metrics and metrics["ROC-AUC"] > best_roc:
            best_roc = metrics["ROC-AUC"]
            best_name = m_name
    return best_name

@st.cache_resource
def load_role_models(role="ALL", run_id=None):
    role = str(role).upper()
    MAPPING_ROLES = {"MID": "MIDDLE", "SUPPORT": "UTILITY", "ADC": "BOTTOM", "BOT": "BOTTOM"}
    role = MAPPING_ROLES.get(role, role)
    
    base_path = f"../modelos/{role}"
    if run_id:
        mlflow_modelos, _, _ = get_version_assets(run_id)
        if mlflow_modelos: base_path = os.path.join(mlflow_modelos, role)

    if not os.path.exists(base_path) or not os.path.exists(f"{base_path}/feature_names.joblib"):
        if role != "ALL": return load_role_models("ALL", run_id)
        return None, None
        
    try:
        feature_names = joblib.load(f"{base_path}/feature_names.joblib")
        scaler = joblib.load(f"{base_path}/scaler.joblib")
        importances_dict = joblib.load(f"{base_path}/importances.joblib")
        
        MAPPING = {"Regresión Logística": "lr", "Random Forest": "rf", "XGBoost": "xgb", "SVM": "svm", "KNN": "knn", "LDA": "lda", "Naive Bayes": "nb", "Árbol de Decisión": "dt", "Red Neuronal (MLP)": "mlp"}
        models = {}
        for label, file_id in MAPPING.items():
            path = f"{base_path}/{file_id}.joblib" if file_id != "mlp" else f"{base_path}/mlp.keras"
            if os.path.exists(path):
                m_obj = joblib.load(path) if file_id != "mlp" else load_model(path)
                m_info = {"model": m_obj}
                vals = importances_dict.get(file_id, np.zeros(len(feature_names)))
                m_info["importances"] = pd.Series(vals, index=feature_names).sort_values(ascending=False)
                models[label] = m_info
        return {"models": models, "scaler": scaler, "features": feature_names}, scaler
    except Exception as e:
        st.error(f"Error cargando modelos para {role}: {e}")
        return None, None

# APP LOGIC
# --- MLOps: Selector de Versiones ---
st.sidebar.header("🕹️ Navegación")
versions = get_available_versions()
version_options = ["Producción (Actual)"] + [v["name"] for v in versions]
selected_v_name = st.sidebar.selectbox(
    "🧠 Versión de Inteligencia:", 
    version_options,
    help="Permite elegir entre diferentes versiones históricas de los modelos y el dataset entrenado con MLflow."
)

if selected_v_name != "Producción (Actual)":
    v_info = next(v for v in versions if v["name"] == selected_v_name)
    current_run_id = v_info["id"]
    dt_run = datetime.fromtimestamp(v_info["start_time"] / 1000)
    version_date_display = dt_run.strftime("%d %b %Y")
else:
    current_run_id = None
    # Dinámico: Usamos la fecha de entrenamiento local (Carpeta Modelos)
    try:
        mtime = os.path.getmtime("../modelos/ALL/scaler.joblib")
        version_date_display = datetime.fromtimestamp(mtime).strftime("%d %b %Y")
    except:
        version_date_display = "awoo"

df_matches, df_players, name_mapping = load_data(current_run_id)

# 1. SINCRONIZACIÓN DE URL -> ESTADO (Al cargar)
if "profile" in st.query_params:
    profile_raw = urllib.parse.unquote(st.query_params["profile"])
    # Resolvemos el nombre canonical (actual) si es un nombre viejo
    if name_mapping and profile_raw in name_mapping:
        p_id = name_mapping[profile_raw]
        # El nombre canonical está en df_players (que ya tiene uno por ID)
        canonical_name = df_players[df_players['id'] == p_id]['nombre'].iloc[0]
        st.session_state.viewing_profile = canonical_name
    else:
        st.session_state.viewing_profile = profile_raw
# INICIALIZACIÓN DE ESTADOS GLOBALES
if "nav_radio" not in st.session_state: st.session_state.nav_radio = "🔥 Inicio / Rankings"
if "selected_simulator_role" not in st.session_state: st.session_state.selected_simulator_role = "ALL"
if "nav_mode" not in st.session_state: st.session_state.nav_mode = st.session_state.nav_radio
if "last_url_nav" not in st.session_state: st.session_state.last_url_nav = None

# Sincronización URL -> State (Solo si el parámetro de la URL cambia externamente)
if "nav" in st.query_params:
    url_nav = st.query_params["nav"]
    if url_nav != st.session_state.last_url_nav:
        st.session_state.nav_radio = url_nav
        st.session_state.nav_mode = url_nav
        st.session_state.last_url_nav = url_nav

# --- MANEJADOR DE NAVEGACIÓN PENDIENTE (REDIRECCIONES) ---
if "pending_nav" in st.session_state:
    target = st.session_state.pending_nav
    st.session_state.nav_radio = target
    st.session_state.nav_mode = target
    st.session_state.last_url_nav = target # Sincronizar token de URL
    st.query_params["nav"] = target
    del st.session_state.pending_nav


if df_matches is not None:
    # 2. MANEJO DE NAVEGACIÓN (Sidebar)
    if 'viewing_profile' not in st.session_state: st.session_state.viewing_profile = None
    
    st.sidebar.header("🕹️ Navegación")
    
    # Pestaña IA: Selector de Modelo Pre-entrenado
    st.sidebar.subheader("🎯 Configuración IA")
    roles_list = ["ALL", "TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    role_idx = roles_list.index(st.session_state.selected_simulator_role)
    selected_role = st.sidebar.selectbox("Rol del modelo (Simulador):", roles_list, index=role_idx, help="Carga los modelos entrenados específicamente para este rol.", key="role_selector_sidebar")
    st.session_state.selected_simulator_role = selected_role
    models_bundle, _ = load_role_models(selected_role, current_run_id)
    
    if models_bundle is None:
        st.sidebar.warning("⚠️ Modelos no encontrados. Ejecute modelado_lol.py primero.")
        # Fallback a ALL si falla
        models_bundle, _ = load_role_models("ALL")
    
    # Extraer componentes del bundle
    models_dict = models_bundle["models"] if models_bundle else {}
    main_scaler = models_bundle["scaler"] if models_bundle else None
    role_features = models_bundle["features"] if models_bundle else []
    
    if st.sidebar.button("🏠 Inicio", width="stretch"): 
        st.session_state.viewing_profile = None
        st.session_state.nav_mode = "🔥 Inicio / Rankings"
        st.query_params.clear()
        st.rerun()
    
    # Usar el estado para el radio button
    mode = st.sidebar.radio("Sección:", 
                            ["🔥 Inicio / Rankings", "🧠 IA & Simulador", "📊 Estadísticas Generales", "👤 Perfil"],
                            key="nav_radio")
    
    # Sincronizar cambios manuales del radio al URL y al modo
    if mode != st.session_state.nav_mode:
        st.session_state.nav_mode = mode
        st.session_state.last_url_nav = mode # Actualizar token para no entrar en bucle
        st.query_params["nav"] = mode

    # Sincronizar perfil a la URL solo si estamos en la sección de Perfil
    if mode == "👤 Perfil" and st.session_state.viewing_profile:
        st.query_params["profile"] = st.session_state.viewing_profile
    elif "profile" in st.query_params:
        del st.query_params["profile"]

    if mode == "🔥 Inicio / Rankings":
        st.markdown("<div class='hero-section'><h1>🏆 League Learning</h1><p>Análisis Profesional Basado en Inteligencia Artificial</p></div>", unsafe_allow_html=True)
        col_s1, col_s2, col_s3 = st.columns([1, 2, 1])
        with col_s2:
            st.markdown("### 🔍 Buscar Invocador")
            p_names = sorted(df_players['nombre'].unique()) if df_players is not None else sorted(df_matches['jugador'].unique())
            search = st.selectbox("Selecciona un nombre:", ["---"] + p_names)
            if search != "---":
                st.session_state.viewing_profile = search
                st.session_state.nav_mode = "👤 Perfil"
                st.query_params["profile"] = search
                st.query_params["nav"] = "👤 Perfil"
                st.rerun()
        
        st.divider()
        
        # --- RANKING DE JUGADORES (Paginación 25 por página) ---
        if df_players is not None:
            st.markdown("<h2 style='text-align:center;'>🏆 Rankings Globales</h2>", unsafe_allow_html=True)
            
            # Ordenar por Tier (Challenger > Grandmaster > Master ...) y luego LPs
            tier_order = {'CHALLENGER': 1, 'GRANDMASTER': 2, 'MASTER': 3, 'DIAMOND': 4, 'EMERALD': 5, 'PLATINUM': 6}
            df_players_sorted = df_players.copy()
            df_players_sorted['tier_rank'] = df_players_sorted['tier'].map(tier_order).fillna(99)
            df_players_sorted = df_players_sorted.sort_values(['tier_rank', 'LPs actuales'], ascending=[True, False]).reset_index(drop=True)

            # Paginación
            items_per_page = 25
            total_pages = (len(df_players_sorted) // items_per_page) + 1
            if 'rank_page' not in st.session_state: st.session_state.rank_page = 0
            
            start_idx = st.session_state.rank_page * items_per_page
            end_idx = start_idx + items_per_page
            
            page_data = df_players_sorted.iloc[start_idx:end_idx]
            
            # Dibujar Tabla Estilizada - Consolidada para evitar problemas de alineación
            table_html = """<div style='background: rgba(0,0,0,0.25); border-radius: 15px; padding: 20px; border: 1px solid rgba(0,242,255,0.1); box-shadow: 0 4px 15px rgba(0,0,0,0.3);'>
<table style='width: 100%; color: white; border-collapse: collapse; table-layout: fixed;'>
<thead>
<tr style='border-bottom: 2px solid #00f2ff; font-family: "Orbitron", sans-serif; letter-spacing: 1px;'>
<th style='padding: 12px; text-align: left; width: 60px;'>#</th>
<th style='padding: 12px; text-align: left;'>Invocador</th>
<th style='padding: 12px; text-align: left; width: 220px;'>Rango</th>
<th style='padding: 12px; text-align: left; width: 120px;'>LPs</th>
<th style='padding: 12px; text-align: left; width: 120px;'>Winrate</th>
</tr>
</thead>
<tbody>"""
            
            for i, row in page_data.iterrows():
                rank_num = i + 1
                color = "#f7ff00" if rank_num <= 3 else "white"
                # Crear el link estilo query param
                e_name = urllib.parse.quote(row['nombre'])
                profile_url = f"?nav=👤+Perfil&profile={e_name}"
                
                winrate = (row['wins'] / (row['wins'] + row['losses'])) * 100
                table_html += f"""<tr style='border-bottom: 1px solid rgba(255,255,255,0.05); transition: background 0.3s;'>
<td style='padding: 12px; color: {color}; font-weight: bold;'>{rank_num}</td>
<td style='padding: 12px;'><b><a href='{profile_url}' target='_self' style='color:#00f2ff; text-decoration:none;'>{row['nombre']}</a></b></td>
<td style='padding: 12px;'>{row['tier']} {row['rank']}</td>
<td style='padding: 12px; color: #00f2ff; font-weight: 600;'>{row['LPs actuales']}</td>
<td style='padding: 12px; font-weight: 600;'>{winrate:.1f}%</td>
</tr>"""
            
            table_html += "</tbody></table></div>"
            st.markdown(table_html, unsafe_allow_html=True)
            
            # Controles de Paginación
            st.write("---")
            nav_cols = st.columns([1, 4, 1])
            with nav_cols[0]:
                if st.button("⬅️ Anterior") and st.session_state.rank_page > 0:
                    st.session_state.rank_page -= 1
                    st.rerun()
            with nav_cols[1]:
                p_text = f"Página {st.session_state.rank_page + 1} de {total_pages}"
                st.markdown(f"<p style='text-align:center;'>{p_text}</p>", unsafe_allow_html=True)
            with nav_cols[2]:
                if st.button("Siguiente ➡️") and st.session_state.rank_page < total_pages - 1:
                    st.session_state.rank_page += 1
                    st.rerun()

    elif mode == "🧠 IA & Simulador":
        st.markdown(f"""
            <div style='display: flex; justify-content: space-between; align-items: baseline;'>
                <h2>🧠 IA & SIMULADOR MULTIALGORITMO</h2>
                <div style='text-align: right;'>
                    <span style='color: #ff4b4b; font-family: "Orbitron", sans-serif; font-size: 1.2rem; font-weight: bold;'>
                        📅 Fecha del modelo: {version_date_display}
                    </span>
                </div>
            </div>
            <hr style='margin: 0 0 20px 0; border: 0; border-top: 1px solid rgba(0,242,255,0.1);'>
        """, unsafe_allow_html=True)
        col_m1, col_m2 = st.columns([1, 2])
        
        with col_m1:
            st.subheader("Configuración del Oráculo")
            
            # Cargar métricas (Desde MLflow si se selecciona versión)
            all_metrics = load_model_metrics(current_run_id)
            role_metrics = all_metrics.get(selected_role, {})
            best_model_id = get_best_model_roc(role_metrics)
            
            # Mapeo invertido para identificar el label de la UI
            MAPPING_UI = {
                "lr": "Regresión Logística", "rf": "Random Forest", "xgb": "XGBoost", "svm": "SVM",
                "knn": "KNN", "lda": "LDA", "nb": "Naive Bayes", "dt": "Árbol de Decisión", "mlp": "Red Neuronal (MLP)"
            }
            
            model_options = []
            for m_label in list(models_dict.keys()):
                m_id = next((k for k, v in MAPPING_UI.items() if v == m_label), None)
                if m_id == best_model_id:
                    model_options.append(f"🏆 {m_label}")
                else:
                    model_options.append(m_label)
            
            orig_selected_model = st.selectbox("Algoritmo de Predicción:", model_options)
            selected_model = orig_selected_model.replace("🏆 ", "")
            
            st.markdown(f"**Análisis de Explicabilidad (CRISP-DM Fase 5)**")
            
            # Real-time explainability chart
            imp_series = models_dict[selected_model]["importances"]
            if imp_series.sum() > 0:
                top_features = imp_series.head(6).index.tolist()
                
                # Plotly Bar Chart for better aesthetics
                chart_data = imp_series.head(10).copy()
                chart_data.index = [get_feature_label(x) for x in chart_data.index]
                fig_imp = px.bar(
                    x=chart_data.values[::-1],
                    y=chart_data.index[::-1],
                    orientation='h',
                    title=f"Variables determinantes ({selected_model})",
                    color=chart_data.values[::-1],
                    color_continuous_scale="Viridis",
                    template="plotly_dark"
                )
                fig_imp.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0), height=300)
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                top_features = ['kills', 'deaths', 'goldEarned_perMin', 'visionScore', 'totalDamageDealtToChampions_perMin', 'challenge_teamDamagePercentage']
                st.info("💡 Calculando patrones no-lineales complejos.")

            # --- SECCIÓN DE MÉTRICAS DEL MODELO ---
            m_id_selected = next((k for k, v in MAPPING_UI.items() if v == selected_model), None)
            if m_id_selected and m_id_selected in role_metrics:
                st.markdown("#### 📊 Métricas de Validación")
                m = role_metrics[m_id_selected]
                c_met1, c_met2 = st.columns(2)
                with c_met1:
                    st.metric("ROC-AUC", f"{m['ROC-AUC']:.3f}")
                    st.metric("Accuracy", f"{m['Accuracy']*100:.1f}%")
                with c_met2:
                    st.metric("F1-Score", f"{m['F1-Score']:.3f}")
                    st.metric("Presición", f"{m['Presición']:.3f}")
                
                # Matriz de Confusión (Sincronizada con MLflow)
                cm_filename = f"{selected_role}_{m_id_selected}.png"
                cm_path = os.path.join("../metricas", "matrices", cm_filename)
                
                if current_run_id:
                    _, mlflow_metricas, _ = get_version_assets(current_run_id)
                    if mlflow_metricas:
                        cm_path = os.path.join(mlflow_metricas, "matrices", cm_filename)

                if os.path.exists(cm_path):
                    st.image(cm_path, caption=f"Matriz de Confusión: {selected_model}", use_container_width=True)
                else:
                    st.info("ℹ️ Matriz de confusión no disponible para esta versión/modelo.")

        with col_m2:
            st.subheader(f"🔮 Simulador Dinámico: {selected_model}")
            
            # --- LÓGICA DE DATOS IMPORTADOS ---
            source_data = st.session_state.get("simulator_data", {})
            if source_data:
                st.info(f"💡 Usando datos importados de una partida con **{source_data.get('championName', 'el campeón')}**.")
                if st.button("🔄 Restablecer a promedios"):
                    st.session_state.simulator_data = {}
                    st.rerun()
            
            with st.form("simulator_form"):
                # --- Estadísticas de Seguridad (Robustez contra NaN) ---
                df_m_current = df_matches[df_matches['individualPosition'] == selected_role] if selected_role != 'ALL' else df_matches
                # Si el rol está vacío en el CSV nuevo, usamos el global como fallback para los rangos de los inputs
                df_stats_ref = df_m_current if len(df_m_current) > 0 else df_matches
                
                inputs = {}
                cols = st.columns(2)
                
                # Mostrar solo las Top 6 variables que determinan el resultado para ESTE modelo
                for i, feat in enumerate(top_features[:6]):
                    with cols[i % 2]:
                        is_pm = feat.endswith('_perMin')
                        base_f = feat.replace('_perMin', '') if is_pm else feat
                        
                        # Cálculo seguro de valores (Evita NaN)
                        try:
                            min_val = float(df_stats_ref[feat].min())
                            max_val = float(df_stats_ref[feat].max())
                            avg_val = float(df_stats_ref[feat].mean())
                            
                            # Si los valores siguen siendo NaN o infinitos por alguna razón
                            if not np.isfinite(min_val): min_val, max_val, avg_val = 0.0, 100.0, 0.0
                            if min_val == max_val: max_val = min_val + 1.0
                        except:
                            min_val, max_val, avg_val = 0.0, 100.0, 0.0

                        label = get_feature_label(feat)
                        
                        # Inputs numéricos con protección de tipo
                        if (base_f in INT_FEATURES and not is_pm):
                            inputs[feat] = st.number_input(label, value=int(round(avg_val)), min_value=int(math.floor(min_val)), max_value=int(math.ceil(max_val)), step=1, key=f"sim_{feat}")
                        else:
                            inputs[feat] = st.number_input(label, value=float(avg_val), min_value=float(min_val), max_value=float(max_val), step=0.05, format="%.2f", key=f"sim_{feat}")
                
                # Advertencia si el rol no tiene datos
                if len(df_m_current) == 0 and selected_role != "ALL":
                    st.warning(f"⚠️ El dataset actual no contiene partidas de {selected_role}. Usando límites globales.")

                submit_btn = st.form_submit_button("🚀 Generar Diagnóstico", width=300)
            
            if submit_btn:
                # 1. Construir DataFrame base con promedios
                input_df = pd.DataFrame(np.zeros((1, len(role_features))), columns=role_features)
                
                # Rellenar con promedios del dataset filtrado por el rol seleccionado (si es posible)
                df_ref = df_matches[df_matches['individualPosition'] == selected_role] if selected_role != 'ALL' else df_matches
                
                # Seguridad: Si el rol está vacío, usamos estadísticas globales para rellenar huecos
                df_ref_safe = df_ref if len(df_ref) > 0 else df_matches
                
                for feat in role_features:
                    if feat in df_ref_safe:
                        input_df[feat] = df_ref_safe[feat].mean()
                
                # 2. Sobrescribir con lo que el usuario ajustó en el Simulator
                for feat, val in inputs.items():
                    input_df[feat] = val
                
                # 3. Escalar y Predecir
                input_scaled = main_scaler.transform(input_df)
                
                if selected_model == "Red Neuronal (MLP)":
                    prob = float(models_dict[selected_model]["model"].predict(input_scaled)[0][0])
                else:
                    prob = models_dict[selected_model]["model"].predict_proba(input_scaled)[0][1]
                
                # 4. Mostrar Resultado Premium
                st.divider()
                st.markdown(f"### 🎯 Diagnóstico del Oráculo")
                
                # Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = float(prob) * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Probabilidad de Victoria (%)", 'font': {'size': 20, 'color': "#00f2ff"}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': "#00f2ff"},
                        'bgcolor': "rgba(255,255,255,0.05)",
                        'borderwidth': 2,
                        'bordercolor': "#00f2ff",
                        'steps': [
                            {'range': [0, 40], 'color': 'rgba(255, 75, 75, 0.3)'},
                            {'range': [40, 60], 'color': 'rgba(255, 255, 0, 0.3)'},
                            {'range': [60, 100], 'color': 'rgba(0, 242, 255, 0.3)'}
                        ],
                        'threshold': {
                            'line': {'color': "#00f2ff", 'width': 4},
                            'thickness': 0.75,
                            'value': float(prob) * 100
                        }
                    }
                ))
                fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Arial"}, height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                if prob > 0.6: st.success("✅ **Perfil de Victoria:** Estas estadísticas son altamente correlativas con un resultado exitoso para el rol seleccionado.")
                elif prob < 0.4: st.error("❌ **Riesgo de Derrota:** Desempeño por debajo del umbral crítico calculado por la IA.")
                else: st.warning("⚖️ **Resultado Ajustado:** El modelo detecta una situación de equilibrio tensional donde cualquier error puede ser fatal.")

        st.divider()
        with st.expander("🔬 Metodología de Ciencia de Datos (CRISP-DM)"):
            c_dm1, c_dm2, c_dm3 = st.columns(3)
            with c_dm1:
                st.info("**Fase 1-3: Comprensión y Preparación**")
                st.markdown("- Limpieza de datos (recalibración de AFKs).\n- Ingeniería de Atributos: Métricas normalizadas 'perMinute'.\n- Muestreo estratificado por roles competitivos.")
            with c_dm2:
                st.success("**Fase 4-5: Modelado y Evaluación**")
                st.markdown("- Entrenamiento de 9 algoritmos (Lineales, Árboles, Redes Neuronales).\n- Explicabilidad mediante Permutation Importance (Incluso para Cajas Negras).\n- Validación cruzada y ajuste de hiperparámetros.")
            with c_dm3:
                st.warning("**Fase 6: Despliegue**")
                st.markdown("- Integración en Dashboard Streamlit.\n- Inferencia en tiempo casi-real para predicción táctica.\n- Exportación directa de perfiles de juego al simulador.")

    elif mode == "👤 Perfil":
        # Determinamos la lista de jugadores disponibles
        p_names = sorted(df_players['nombre'].unique()) if df_players is not None else sorted(df_matches['jugador'].unique())
        
        # Lógica de jugador por defecto
        default_player = "Zinko5#LAS"
        if st.session_state.viewing_profile and st.session_state.viewing_profile in p_names:
            current_idx = p_names.index(st.session_state.viewing_profile)
        elif default_player in p_names:
            current_idx = p_names.index(default_player)
        else:
            current_idx = 0

        # Barra de búsqueda siempre visible en Perfil
        p_name = st.selectbox("Invocador:", p_names, index=current_idx, key="profile_search_selectbox")
        
        # Sincronizar estado si cambia el selectbox
        if p_name != st.session_state.viewing_profile:
            st.session_state.viewing_profile = p_name
            st.query_params["profile"] = p_name
            # No hacemos rerun aquí para evitar parpadeo, dejamos que fluya el resto del renderizado
            # con el p_name ya actualizado.

        # --- CABECERA SUPERIOR ---
        st.markdown(f"<h1 style='display:flex; align-items:center;'><span style='font-size:40px; margin-right:15px;'>👤</span> {p_name}</h1>", unsafe_allow_html=True)
        
        p_match_history = df_matches[df_matches['jugador'] == p_name].sort_values('match_id', ascending=False)
        
        # --- CÁLCULOS DE INSIGHTS (Roles y Más Jugado) ---
        if not p_match_history.empty:
            top_champ = p_match_history['championName'].value_counts().idxmax()
            top_champ_count = p_match_history['championName'].value_counts().max()
            top_champ_pct = (top_champ_count / len(p_match_history)) * 100
            
            roles_counts = p_match_history['individualPosition'].value_counts()
            top_roles = roles_counts.head(2).index.tolist()
            roles_display = []
            for r in top_roles:
                count = roles_counts[r]
                pct = (count / len(p_match_history)) * 100
                roles_display.append(f"{r}: <b>{pct:.1f}%</b> ({count})")
        else:
            top_champ, top_champ_count, top_champ_pct = "N/A", 0, 0
            roles_display = ["Sin datos de posición"]

        # --- FILA DE RESUMEN ---
        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            if df_players is not None:
                p_info = df_players[df_players['nombre'] == p_name]
                if not p_info.empty:
                    info = p_info.iloc[0]
                    st.markdown(f"""
                        <div style='background: rgba(0,242,255,0.05); padding: 25px; border-radius: 15px; border: 1px solid #00f2ff; min-height: 180px;'>
                            <h2 style='margin:0; font-size:24px;'>{info['tier']} {info['rank']} <span style='font-size:14px; color:#00f2ff;'>🔗</span></h2>
                            <p style='font-size: 32px; color: #f7ff00; margin: 10px 0;'>{info['LPs actuales']} LPs</p>
                            <p style='margin:0;'>Winrate: <b>{(info['wins']/(info['wins']+info['losses'])*100):.1f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"""
                <div style='display: flex; gap: 50px; padding: 20px;'>
                    <div>
                        <p style='color: #888; margin-bottom: 5px;'>Más jugado:</p>
                        <p style='font-size: 20px; margin: 0;'><b>{top_champ}</b></p>
                        <p style='color: #00f2ff; margin: 0;'>{top_champ_pct:.1f}% de las partidas ({top_champ_count})</p>
                    </div>
                    <div>
                        <p style='color: #888; margin-bottom: 5px;'>Roles principales:</p>
                        <p style='font-size: 16px; line-height: 1.4; margin: 0;'>
                            {"<br>".join(roles_display)}
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.divider()
        
        # --- SELECTOR DE PESTAÑAS (TABS) ---
        tab_recientes, tab_stats = st.tabs(["📋 Partidas Recientes", "📈 Estadísticas del Jugador"])
        
        with tab_recientes:
            st.subheader("📋 Historial de Partidas")
            
            # --- LÓGICA DE PAGINACIÓN ---
            matches_per_page = 10
            total_matches = len(p_match_history)
            total_pages = max((total_matches // matches_per_page) + (1 if total_matches % matches_per_page > 0 else 0), 1)
            
            # Inicializar o resetear página si cambia el perfil
            if "match_page" not in st.session_state or st.session_state.get("last_p_id") != p_name:
                st.session_state.match_page = 1
                st.session_state.last_p_id = p_name
            
            start_idx = (st.session_state.match_page - 1) * matches_per_page
            end_idx = start_idx + matches_per_page
            paged_matches = p_match_history.iloc[start_idx:end_idx]

            # Controles de cabecera: Leyenda + Selector de modelo
            col_l1, col_l2, col_l3 = st.columns([2, 1, 1])
            with col_l1:
                st.markdown("<small style='color: #888;'>Leyenda: <b>⭐ MVP</b> | <b>💥 Carry</b> | <b>👁️ Visión</b></small>", unsafe_allow_html=True)
            with col_l2:
                eval_model_hist = st.selectbox("Algoritmo:", list(models_dict.keys()), key="eval_h_selector")
            with col_l3:
                use_specialized = st.toggle("🚀 Especialización por Rol", value=True, help="Usa modelos entrenados solo para la posiciónJugada en cada partida.")

            st.markdown(f"<p style='text-align: right; color: #00f2ff; font-weight: bold;'>Página {st.session_state.match_page} / {total_pages}</p>", unsafe_allow_html=True)
            
            if paged_matches.empty:
                st.info("No hay más partidas para mostrar.")
            
            for _, match_row in paged_matches.iterrows():
                m_id = match_row['match_id']
                is_win = match_row['win']
                m_pos = match_row['individualPosition']
                duration_min = match_row['timePlayed'] // 60
                duration_sec = match_row['timePlayed'] % 60
                
                # --- LÓGICA DE MODELO (ESPECIALIZADO O GLOBAL) ---
                if use_specialized:
                    # Cargamos modelos específicos
                    curr_bundle, _ = load_role_models(m_pos, current_run_id)
                    if curr_bundle is None:
                        curr_bundle = models_bundle
                else:
                    curr_bundle = models_bundle

                # Si no hay modelos cargados, saltamos esta partida
                if curr_bundle is None or "models" not in curr_bundle:
                    continue

                # --- CALCULAR WIN PROB CON METODOLOGÍA NUEVA ---
                # 1. Crear vector conforme a feature_names
                m_feat_names = curr_bundle["features"]
                match_vec = pd.DataFrame(np.zeros((1, len(m_feat_names))), columns=m_feat_names)
                
                # 2. Poblar métricas (Usando ya las pre-calculadas perMin si existen)
                for f in m_feat_names:
                    if f in match_row:
                        match_vec[f] = match_row[f]
                    elif f.endswith('_perMin'):
                        # Fallback por si la fila no las tiene (aunque deberían estar en df_matches)
                        base_n = f.replace('_perMin', '')
                        if base_n in match_row:
                            match_vec[f] = match_row[base_n] / max(1, (match_row['timePlayed'] / 60))
                

                # 4. Predecir
                match_scaled = curr_bundle["scaler"].transform(match_vec)
                
                if eval_model_hist == "Red Neuronal (MLP)":
                    m_prob = float(curr_bundle["models"][eval_model_hist]["model"].predict(match_scaled)[0][0])
                else:
                    m_prob = curr_bundle["models"][eval_model_hist]["model"].predict_proba(match_scaled)[0][1]
                
                bg_color = "rgba(0, 242, 255, 0.08)" if is_win else "rgba(255, 75, 75, 0.08)"
                border_color = "#00f2ff" if is_win else "#ff4b4b"
                
                with st.container():
                    st.markdown(f"""
                        <div style='background: {bg_color}; border-left: 8px solid {border_color}; padding: 15px; border-radius: 8px; margin-bottom: 5px; 
                                    display: flex; justify-content: space-between; align-items: center;'>
                            <div style='flex-grow: 1;'>
                                <span style='font-weight: bold; color: {border_color};'>{"VICTORIA" if is_win else "DERROTA"}</span> | 
                                <b style='font-size: 18px;'>{match_row['championName']}</b> | {match_row['kills']}/{match_row['deaths']}/{match_row['assists']}
                            </div>
                            <div style='text-align: right; min-width: 200px;'>
                                <span style='color: #888; font-size: 13px;'>{"🚀 especializ." if use_specialized else "🌍 global"} prob:</span> 
                                <span style='color: #00f2ff; font-family: monospace; font-size: 18px; font-weight: bold;'>{m_prob*100:.1f}%</span>
                                <div style='color: #666; font-size: 12px;'>duración: {duration_min}:{duration_sec:02d} | {m_pos}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("Ver detalle de la partida"):
                        # BOTONES DE ACCIÓN: EXPORTAR Y VER STATS
                        col_act1, col_act2 = st.columns(2)
                        with col_act1:
                            if st.button(f"📤 Exportar datos al Simulador", key=f"btn_exp_{m_id}", width="stretch"):
                                st.session_state.simulator_data = match_row.to_dict()
                                # Limpiar el rol por si tiene prefijos y asegurar que existe en la lista
                                assigned_role = m_pos if m_pos in ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"] else "ALL"
                                st.session_state.selected_simulator_role = assigned_role
                                st.session_state.pending_nav = "🧠 IA & Simulador" # Marcar para el próximo ciclo
                                st.rerun()
                        with col_act2:
                            if st.button(f"📊 Ver todas las {len(match_row)} stats", key=f"btn_ext_stats_{m_id}", width="stretch"):
                                st.session_state[f"show_ext_{m_id}"] = not st.session_state.get(f"show_ext_{m_id}", False)
                        
                        # --- PANEL DE TODAS LAS STATS (EXTENDIDO) ---
                        if st.session_state.get(f"show_ext_{m_id}", False):
                            st.markdown("<div style='background:rgba(255,255,255,0.03); padding:20px; border-radius:10px; border:1px solid #444;'>", unsafe_allow_html=True)
                            st.write("### 📊 Desglose Completo de Estadísticas")
                            s_cols = st.columns(3)
                            idx_s = 0
                            for c_name, val in match_row.items():
                                label = get_feature_label(c_name)
                                with s_cols[idx_s % 3]:
                                    if isinstance(val, (float, np.floating)):
                                        st.write(f"**{label}:** `{val:.2f}`")
                                    elif isinstance(val, (int, np.integer)):
                                        st.write(f"**{label}:** `{val:,}`")
                                    else:
                                        st.write(f"**{label}:** `{val}`")
                                idx_s += 1
                            st.markdown("</div>", unsafe_allow_html=True)

                        all_players = df_matches[df_matches['match_id'] == m_id].sort_values('side')
                        for side_name, color_team in [("Blue", "#51a1ff"), ("Red", "#ff5151")]:
                            st.markdown(f"<h4 style='color: {color_team}; margin-top:20px;'>Equipo {side_name}</h4>", unsafe_allow_html=True)
                            team_df = all_players[all_players['side'] == side_name]
                            
                            h_cols = st.columns([2.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2.5])
                            h_cols[0].write("**Invocador**")
                            h_cols[1].write("**Campeón**")
                            h_cols[2].write("**KDA**")
                            h_cols[3].write("**Oro**")
                            h_cols[4].write("**Daño**")
                            h_cols[5].write("**Visión**")
                            h_cols[6].write("**Badges**")
                            
                            for _, p_row in team_df.iterrows():
                                is_current = p_row['jugador'] == p_name
                                r_cols = st.columns([2.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2.5])
                                
                                badges = []
                                # Cálculo de KDA manual para badges
                                p_kda = (p_row['kills'] + p_row['assists']) / max(1, p_row['deaths'])
                                team_kdas = (team_df['kills'] + team_df['assists']) / team_df['deaths'].replace(0, 1)
                                if p_kda >= team_kdas.max() and p_row['win']: badges.append("⭐")
                                if p_row['visionScore'] > 35: badges.append("👁️")
                                if p_row['totalDamageDealtToChampions'] >= team_df['totalDamageDealtToChampions'].max(): badges.append("💥")
                                
                                encoded_name = urllib.parse.quote(p_row['jugador'])
                                current_url = f"?nav=👤+Perfil&profile={encoded_name}"
                                r_cols[0].markdown(f"**[{p_row['jugador']}]({current_url})**" if not is_current else f"<span style='color:#00f2ff;'>{p_row['jugador']}</span>", unsafe_allow_html=True)
                                r_cols[1].write(p_row['championName'])
                                r_cols[2].write(f"{p_row['kills']}/{p_row['deaths']}/{p_row['assists']}")
                                r_cols[3].write(f"{p_row['goldEarned']:,}")
                                r_cols[4].write(f"{p_row['totalDamageDealtToChampions']:,}")
                                r_cols[5].write(f"{p_row['visionScore']}")
                                r_cols[6].write(" ".join(badges))
            
            # --- CONTROLES DE NAVEGACIÓN ---
            st.divider()
            c1, c2, c3 = st.columns([1, 2, 1])
            with c1:
                if st.session_state.match_page > 1:
                    if st.button("⬅️ Anterior", width="stretch"):
                        st.session_state.match_page -= 1
                        st.rerun()
            with c2:
                st.markdown(f"<p style='text-align:center; color:#888; padding-top:5px;'>Página {st.session_state.match_page} de {total_pages}</p>", unsafe_allow_html=True)
            with c3:
                if st.session_state.match_page < total_pages:
                    if st.button("Siguiente ➡️", width="stretch"):
                        st.session_state.match_page += 1
                        st.rerun()

        with tab_stats:
            st.header("📈 Análisis de Desempeño Promedio")
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                if not p_match_history.empty:
                    stats = p_match_history.mean(numeric_only=True)
                    # Cálculo de KDA manual
                    p_kda_avg = (stats['kills'] + stats['assists']) / max(1, stats['deaths'])
                    radar_data = pd.DataFrame(dict(
                        r=[
                            min(100, (p_kda_avg/5)*100),
                            min(100, (stats['visionScore']/50)*100),
                            min(100, (stats['challenge_teamDamagePercentage']*300)),
                            min(100, (stats['champLevel']/18)*100),
                            min(100, (stats['goldEarned']/15000)*100)
                        ],
                        theta=['KDA','Visión','Daño %','Nivel','Oro']
                    ))
                    fig_radar = px.line_polar(radar_data, r='r', theta='theta', line_close=True, range_r=[0,100], title="Radar de Habilidades")
                    fig_radar.update_traces(fill='toself', fillcolor='rgba(0,242,255,0.3)', line_color='#00f2ff')
                    st.plotly_chart(fig_radar, width="stretch")
            
            with col_s2:
                st.subheader("Promedios del Jugador")
                avg_stats = p_match_history.mean(numeric_only=True)
                p_kda_total = (avg_stats['kills'] + avg_stats['assists']) / max(1, avg_stats['deaths'])
                st.write(f"- **KDA Promedio:** {p_kda_total:.2f}")
                st.write(f"- **Oro por Partida:** {avg_stats['goldEarned']:.0f}")
                st.write(f"- **Súbditos por Partida:** {avg_stats['totalMinionsKilled']:.0f}")
                st.write(f"- **Daño a Campeones:** {avg_stats['totalDamageDealtToChampions']:.0f}")
                st.write(f"- **Puntaje de Visión:** {avg_stats['visionScore']:.1f}")
                st.write(f"- **Participación en Kill:** {avg_stats['challenge_killParticipation']*100:.1f}%")

    elif mode == "📊 Estadísticas Generales":
        st.markdown("<h2 style='text-align:center;'>📊 Estadísticas Generales & Tendencias</h2>", unsafe_allow_html=True)
        
        # 1. Indicadores de Nivel Superior
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Partidas Analizadas", f"{len(df_matches):,}")
        c2.metric("Invocadores Únicos", f"{df_matches['jugador'].nunique():,}")
        
        # KDA Promedio global calculado
        global_kda = (df_matches['kills'].mean() + df_matches['assists'].mean()) / max(1, df_matches['deaths'].mean())
        c3.metric("KDA Promedio Global", f"{global_kda:.2f}")
        c4.metric("Dureza Media (Minutos)", f"{(df_matches['timePlayed'].mean()/60):.1f} min")
        
        st.divider()

        # TABS PARA ORGANIZACIÓN PROFESIONAL
        tab_macro, tab_roles, tab_champs = st.tabs(["🌎 Macroperspectiva", "🛡️ Dinámicas de Rol", "🧛 Élite de Campeones"])
        
        with tab_macro:
            col_mac1, col_mac2 = st.columns(2)
            
            with col_mac1:
                # Balance de Victoria por Lado del Mapa
                side_stats = df_matches.groupby('side')['win'].mean().reset_index()
                fig_side = px.pie(side_stats, values='win', names='side', hole=0.5,
                                title="Balance de Victoria por Lado del Mapa (Lado Azul vs Rojo)",
                                color='side', color_discrete_map={'Blue': '#00f2ff', 'Red': '#ff4b4b'})
                fig_side.update_traces(textposition='inside', textinfo='percent+label')
                fig_side.update_layout(template="plotly_dark", showlegend=False)
                st.plotly_chart(fig_side, use_container_width=True)
                
            with col_mac2:
                # Relación Objetivos vs Victoria
                obj_metrics = ['challenge_teamBaronKills', 'challenge_teamElderDragonKills', 'challenge_teamRiftHeraldKills']
                obj_labels = ["Barones", "D. Ancianos", "Heraldos"]
                
                avg_objs = df_matches.groupby('win')[obj_metrics].mean().reset_index()
                
                fig_objs = go.Figure()
                for i, m in enumerate(obj_metrics):
                    fig_objs.add_trace(go.Bar(
                        name=obj_labels[i],
                        x=['Derrota', 'Victoria'],
                        y=avg_objs[m],
                        marker_color="#00f2ff" if i == 0 else ("#ff4b4b" if i == 1 else "#f7ff00")
                    ))
                
                fig_objs.update_layout(barmode='group', title="Importancia de Objetivos en la Victoria", template="plotly_dark")
                st.plotly_chart(fig_objs, use_container_width=True)

            # Mapa de calor de Visión vs Muerte (Agregado)
            st.markdown("#### Distribución de Visión y Oro en las Victorias")
            fig_heatmap = px.density_heatmap(df_matches[df_matches['win'] == True], 
                                            x='visionScore', y='goldEarned', 
                                            title="Densidad de Vision Score vs Oro en Partidas Ganadas",
                                            labels={'visionScore': 'Score de Visión', 'goldEarned': 'Oro total'},
                                            template="plotly_dark", color_continuous_scale="Viridis")
            st.plotly_chart(fig_heatmap, use_container_width=True)

        with tab_roles:
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                # Distribución de Oro (Mejorado)
                fig1 = px.box(df_matches, x='individualPosition', y='goldEarned', color='win', 
                             title="Eficiencia de Oro por Posición",
                             color_discrete_map={True: "#00f2ff", False: "#ff4b4b"},
                             category_orders={"individualPosition": ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]},
                             template="plotly_dark")
                st.plotly_chart(fig1, use_container_width=True)
            
            with col_r2:
                # Participación en Kills por Rol
                avg_kp = df_matches.groupby('individualPosition')['challenge_killParticipation'].mean().reset_index()
                fig_kp = px.bar(avg_kp, x='individualPosition', y='challenge_killParticipation',
                               title="Impacto Directo: Participación en Muertes por Rol",
                               labels={'challenge_killParticipation': 'Participación Media (%)'},
                               color='challenge_killParticipation', color_continuous_scale="Tealgrn",
                               category_orders={"individualPosition": ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]},
                               template="plotly_dark")
                st.plotly_chart(fig_kp, use_container_width=True)

            # Scatter de Daño Recibido vs Producido
            st.markdown("#### Perfiles de Batalla y Comunicación")
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                fig_battle = px.scatter(df_matches, x='totalDamageDealtToChampions', y='totalDamageTaken', 
                                    color='individualPosition', size='kills',
                                    hover_data=['championName', 'win'],
                                    title="Daño Producido vs Daño Recibido",
                                    labels={'totalDamageDealtToChampions': 'Daño a Campeones', 'totalDamageTaken': 'Daño Recibido'},
                                    template="plotly_dark", opacity=0.6)
                st.plotly_chart(fig_battle, use_container_width=True)
            with col_b2:
                # Pings vs Victoria
                avg_pings = df_matches.groupby('win')['totalPings'].mean().reset_index()
                fig_pings = px.bar(avg_pings, x='win', y='totalPings', 
                                  title="Comunicación Media (Pings) vs Resultado",
                                  labels={'win': 'Victoria', 'totalPings': 'Promedio de Pings'},
                                  color='win', color_discrete_map={True: "#00f2ff", False: "#ff4b4b"},
                                  template="plotly_dark")
                st.plotly_chart(fig_pings, use_container_width=True)


        with tab_champs:
            col_c1, col_c2 = st.columns([1.5, 1])
            
            with col_c1:
                # Top 15 Campeones más Pickeados y su Winrate
                champ_stats = df_matches.groupby('championName').agg(
                    partidas=('win', 'count'),
                    winrate=('win', 'mean')
                ).reset_index()
                
                # Filtrar campeones con al menos 10 partidas para winrate significativo
                top_picked = champ_stats.sort_values('partidas', ascending=False).head(15)
                top_picked['win_display'] = (top_picked['winrate'] * 100).round(1).astype(str) + "%"
                
                fig_champs = px.bar(top_picked, x='championName', y='partidas',
                                   color='winrate', color_continuous_scale="RdYlGn",
                                   title="Top 15 Campeones por Popularidad (Color = Winrate)",
                                   text='win_display',
                                   template="plotly_dark")
                fig_champs.update_traces(textposition='outside')
                st.plotly_chart(fig_champs, use_container_width=True)

            with col_c2:
                # Campeones "Best Per-Minute Performance"
                st.markdown("#### Campeones con mayor Daño/Min")
                top_dpm = df_matches.groupby('championName')['challenge_damagePerMinute'].mean().sort_values(ascending=False).head(10).reset_index()
                fig_dpm = px.bar(top_dpm, y='championName', x='challenge_damagePerMinute',
                                orientation='h', title="Top 10 DPM (Daño por Minuto)",
                                color='challenge_damagePerMinute', color_continuous_scale="Purp",
                                template="plotly_dark")
                fig_dpm.update_layout(showlegend=False)
                st.plotly_chart(fig_dpm, use_container_width=True)

            # Visión de Campeones por Rol selectivo (Opcional, dinámico)
            st.markdown("---")
            selected_pos = st.selectbox("Analizar campeones en posición:", ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"], index=2)
            df_pos = df_matches[df_matches['individualPosition'] == selected_pos]
            
            if not df_pos.empty:
                champ_pos_stats = df_pos.groupby('championName').agg(
                    partidas=('win', 'count'),
                    winrate=('win', 'mean'),
                    muertes_avg=('deaths', 'mean')
                ).reset_index()
                
                # Burbujas: Partidas jugadas vs Winrate (Tamaño = Muertes Promedio)
                fig_bubble = px.scatter(champ_pos_stats[champ_pos_stats['partidas'] >= 3], 
                                       x='partidas', y='winrate', size='partidas',
                                       color='winrate', hover_name='championName',
                                       title=f"Rendimiento de Campeones como {selected_pos} (Min 3 partidas)",
                                       labels={'partidas': 'Partidas Jugadas', 'winrate': 'Tasa de Victoria'},
                                       template="plotly_dark", color_continuous_scale="Geyser")
                # Línea de referencia 50% winrate
                fig_bubble.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Meta (50%)")
                st.plotly_chart(fig_bubble, use_container_width=True)


else:
    st.error("Error al cargar datos.")

# --- FOOTER LEGAL ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #555; font-size: 12px; padding: 20px;'>
    <b>League Learning</b> isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games 
    or anyone officially involved in producing or managing Riot Games properties. 
    Riot Games, and all associated properties are trademarks or registered trademarks of Riot Games, Inc.
</div>
""", unsafe_allow_html=True)
