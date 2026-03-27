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
import urllib.parse
import joblib
from tensorflow.keras.models import load_model

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
    'individualPosition': 'Posición'
}

@st.cache_data
def load_data():
    df_m = pd.read_csv('../league_data.csv') if os.path.exists('../league_data.csv') else None
    df_p = pd.read_csv('../todosJugadores.csv') if os.path.exists('../todosJugadores.csv') else None
    
    if df_p is not None:
        # 1. Asegurar que tenemos fecha_actualizacion como datetime si existe
        if 'fecha_actualizacion' in df_p.columns:
            df_p['fecha_actualizacion'] = pd.to_datetime(df_p['fecha_actualizacion'])
            df_p = df_p.sort_values('fecha_actualizacion')
        
        # 2. Mapeo de todos los nombres históricos a su ID único
        # Esto sirve para asociar partidas viejas (con nombre viejo) al ID correcto
        name_to_id = df_p.set_index('nombre')['id'].to_dict()
        
        # 3. Quedarse con un solo registro por ID (el más reciente)
        # Esto será nuestro "perfil único"
        df_p_unique = df_p.drop_duplicates(subset='id', keep='last').copy()
        
        # 4. Mapeo de ID a nombre canonical (el más reciente)
        id_to_canonical_name = df_p_unique.set_index('id')['nombre'].to_dict()
        
        if df_m is not None:
            # 5. Normalizar df_matches: asociar cada partida a un ID y luego al nombre canonical
            df_m['player_id'] = df_m['jugador'].map(name_to_id)
            # Si un jugador no está en la tabla de jugadores, conservamos su nombre como ID temporal
            df_m['player_id'] = df_m['player_id'].fillna(df_m['jugador'])
            
            # Reemplazar el nombre del jugador en las partidas por su nombre más reciente
            df_m['jugador'] = df_m['player_id'].map(id_to_canonical_name).fillna(df_m['jugador'])
        
        # Retornamos los datos procesados
        df_p = df_p_unique

    if df_m is not None:
        df_m = df_m[df_m['challenge_hadAfkTeammate'] == 0]
        df_m = df_m[df_m['timePlayed'] > 420]
        
    return df_m, df_p, name_to_id

@st.cache_resource
def load_role_models(role="ALL"):
    base_path = f"modelos/{role}"
    if not os.path.exists(base_path):
        return None, None
        
    try:
        # Cargamos modelos desde disco (Instantáneo)
        models = {
            "Regresión Logística": {"model": joblib.load(f"{base_path}/lr.joblib")},
            "Random Forest": {"model": joblib.load(f"{base_path}/rf.joblib")},
            "XGBoost": {"model": joblib.load(f"{base_path}/xgb.joblib")},
            "SVM": {"model": joblib.load(f"{base_path}/svm.joblib")},
            "KNN": {"model": joblib.load(f"{base_path}/knn.joblib")},
            "Red Neuronal (MLP)": {"model": load_model(f"{base_path}/mlp.keras")}
        }
        scaler = joblib.load(f"{base_path}/scaler.joblib")
        
        # Calcular importancias/pesos para visualización (solo una vez por carga)
        models["Regresión Logística"]["importances"] = pd.Series(models["Regresión Logística"]["model"].coef_[0], index=NUMERIC_FEATURES).abs().sort_values(ascending=False)
        models["Random Forest"]["importances"] = pd.Series(models["Random Forest"]["model"].feature_importances_, index=NUMERIC_FEATURES).sort_values(ascending=False)
        models["XGBoost"]["importances"] = pd.Series(models["XGBoost"]["model"].feature_importances_, index=NUMERIC_FEATURES).sort_values(ascending=False)
        models["SVM"]["importances"] = pd.Series(models["SVM"]["model"].coef_[0], index=NUMERIC_FEATURES).abs().sort_values(ascending=False)
        models["KNN"]["importances"] = pd.Series(0, index=NUMERIC_FEATURES)
        models["Red Neuronal (MLP)"]["importances"] = pd.Series(0, index=NUMERIC_FEATURES)
        
        return models, scaler
    except Exception as e:
        st.error(f"Error cargando modelos para {role}: {e}")
        return None, None

# APP LOGIC
df_matches, df_players, name_mapping = load_data()

# 1. SINCRONIZACIÓN DE URL -> ESTADO (Al cargar)
import urllib.parse
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
    selected_role = st.sidebar.selectbox(
        "Rol del modelo (Simulador):", 
        roles_list,
        index=role_idx,
        help="Carga los modelos entrenados específicamente para este rol.",
        key="role_selector_sidebar"
    )
    st.session_state.selected_simulator_role = selected_role
    
    # Cargamos modelos (Instantáneo desde modelos/)
    models_dict, main_scaler = load_role_models(selected_role)
    
    if models_dict is None:
        st.sidebar.warning("⚠️ Modelos no encontrados. Ejecute modelado_lol.py primero.")
        # Fallback a ALL si falla
        models_dict, main_scaler = load_role_models("ALL")
    
    if st.sidebar.button("🏠 Inicio", width="stretch"): 
        st.session_state.viewing_profile = None
        st.session_state.nav_mode = "🔥 Inicio / Rankings"
        st.query_params.clear()
        st.rerun()
    
    # Usar el estado para el radio button
    mode = st.sidebar.radio("Sección:", 
                            ["🔥 Inicio / Rankings", "🧠 IA & Simulador", "📊 Metajuego", "👤 Perfil"],
                            key="nav_radio")
    
    # Sincronizar cambios manuales del radio al URL y al modo
    if mode != st.session_state.nav_mode:
        st.session_state.nav_mode = mode
        st.session_state.last_url_nav = mode # Actualizar token para no entrar en bucle
        st.query_params["nav"] = mode

    # Sincronizar perfil a la URL
    if st.session_state.viewing_profile:
        st.query_params["profile"] = st.session_state.viewing_profile
    else:
        if "profile" in st.query_params: del st.query_params["profile"]

    if st.session_state.viewing_profile is None and mode == "🔥 Inicio / Rankings":
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
            st.markdown("<h2 style='text-align:center;'>🏆 Rankings Globales (LAS)</h2>", unsafe_allow_html=True)
            
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
            
            # Dibujar Tabla Estilizada
            st.markdown("""
                <div style='background: rgba(0,0,0,0.2); border-radius: 10px; padding: 10px;'>
                    <table style='width: 100%; color: white; border-collapse: collapse;'>
                        <tr style='border-bottom: 2px solid #00f2ff;'>
                            <th style='padding: 10px; text-align: left;'>#</th>
                            <th style='padding: 10px; text-align: left;'>Invocador</th>
                            <th style='padding: 10px; text-align: left;'>Rango</th>
                            <th style='padding: 10px; text-align: left;'>LPs</th>
                            <th style='padding: 10px; text-align: left;'>Winrate</th>
                        </tr>
            """, unsafe_allow_html=True)
            
            import urllib.parse
            for i, row in page_data.iterrows():
                rank_num = i + 1
                color = "#f7ff00" if rank_num <= 3 else "white"
                # Crear el link estilo query param
                e_name = urllib.parse.quote(row['nombre'])
                profile_url = f"?nav=👤+Perfil&profile={e_name}"
                
                winrate = (row['wins'] / (row['wins'] + row['losses'])) * 100
                st.markdown(f"""
                    <tr style='border-bottom: 1px solid rgba(255,255,255,0.1);'>
                        <td style='padding: 8px; color: {color}; font-weight: bold;'>{rank_num}</td>
                        <td style='padding: 8px;'><b><a href='{profile_url}' target='_self' style='color:#00f2ff; text-decoration:none;'>{row['nombre']}</a></b></td>
                        <td style='padding: 8px;'>{row['tier']} {row['rank']}</td>
                        <td style='padding: 8px; color: #00f2ff;'>{row['LPs actuales']}</td>
                        <td style='padding: 8px;'>{winrate:.1f}%</td>
                    </tr>
                """, unsafe_allow_html=True)
            
            st.markdown("</table></div>", unsafe_allow_html=True)
            
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
                
                # Mapear nombres para el gráfico
                chart_data = imp_series.head(10).copy()
                chart_data.index = [FEATURE_LABELS.get(x, x) for x in chart_data.index]
                st.bar_chart(chart_data)
            else:
                top_features = ['kills', 'deaths', 'goldEarned', 'visionScore', 'totalDamageDealtToChampions', 'challenge_teamDamagePercentage']
                st.warning("Este modelo utiliza patrones no-lineales complejos (Caja Negra).")

        with col_m2:
            st.subheader(f"🔮 Simulador Dinámico: {selected_model}")
            
            # --- LÓGICA DE DATOS IMPORTADOS ---
            source_data = st.session_state.get("simulator_data", {})
            if source_data:
                st.info(f"💡 Usando datos importados de una partida con **{source_data.get('championName', 'el campeón')}**.")
                if st.button("🔄 Restablecer a promedios"):
                    st.session_state.simulator_data = {}
                    st.rerun()
            else:
                st.markdown("Ajusta las variables que **este modelo** considera más determinantes.")
            
            with st.form("simulator_form"):
                inputs = {}
                cols = st.columns(2)
                for i, feat in enumerate(top_features):
                    with cols[i % 2]:
                        min_val = float(df_matches[feat].min())
                        max_val = float(df_matches[feat].max())
                        
                        # Usar dato importado o promedio como base
                        default_val = float(source_data.get(feat, df_matches[feat].mean()))
                        
                        label = FEATURE_LABELS.get(feat, feat)
                        if feat in INT_FEATURES:
                            inputs[feat] = st.number_input(label, int(min_val), int(max_val), int(round(default_val)), step=1)
                        else:
                            inputs[feat] = st.number_input(label, min_val, max_val, default_val, step=0.01)
                
                submit_btn = st.form_submit_button("🚀 Generar Diagnóstico", width="stretch")
            
            if submit_btn:
                # Preparamos el vector de entrada con alta fidelidad
                full_input = []
                for f in NUMERIC_FEATURES:
                    if f in inputs:
                        val = inputs[f] # Valor modificado en el form
                    elif f in source_data:
                        val = source_data[f] # Valor real importado (pero oculto en el form)
                    else:
                        val = df_matches[f].mean() # Promedio si no hay datos
                    full_input.append(float(val))
                
                # Pasamos un DataFrame para evitar avisos de Scikit-Learn sobre nombres de features
                input_df = pd.DataFrame([full_input], columns=NUMERIC_FEATURES)
                input_scaled = main_scaler.transform(input_df)
                
                if selected_model == "Red Neuronal (MLP)":
                    prob = float(models_dict[selected_model]["model"].predict(input_scaled)[0][0])
                else:
                    prob = models_dict[selected_model]["model"].predict_proba(input_scaled)[0][1]
                
                st.markdown(f"## Probabilidad de Victoria: `{float(prob)*100:.1f}%`")
                st.progress(float(prob))
                
                if prob > 0.6: st.success("Perfil de Victoria: Las métricas sugieren un desempeño dominante.")
                elif prob < 0.4: st.error("Riesgo de Derrota: Estas métricas coinciden con situaciones de desventaja crítica.")
                else: st.warning("Resultado Incierto: Juego cerrado o variables equilibradas.")

    elif st.session_state.viewing_profile is not None or mode == "👤 Perfil":
        p_name = st.session_state.viewing_profile if st.session_state.viewing_profile else st.selectbox("Invocador:", sorted(df_matches['jugador'].unique()))
        
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
                    # Cargamos modelos específicos (Instantáneo)
                    curr_models, curr_scaler = load_role_models(m_pos)
                    if curr_models is None:
                        curr_models, curr_scaler = models_dict, main_scaler
                else:
                    curr_models, curr_scaler = models_dict, main_scaler

                # --- CALCULAR WIN PROB ---
                match_v = [match_row[f] for f in NUMERIC_FEATURES]
                match_df = pd.DataFrame([match_v], columns=NUMERIC_FEATURES)
                match_scaled = curr_scaler.transform(match_df)
                
                if eval_model_hist == "Red Neuronal (MLP)":
                    m_prob = float(curr_models[eval_model_hist]["model"].predict(match_scaled)[0][0])
                else:
                    m_prob = curr_models[eval_model_hist]["model"].predict_proba(match_scaled)[0][1]
                
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
                                <span style='color: #888; font-size: 13px;'>puntuación/win prob:</span> 
                                <span style='color: #00f2ff; font-family: monospace; font-size: 18px; font-weight: bold;'>{m_prob*100:.1f}%</span>
                                <div style='color: #666; font-size: 12px;'>duración: {duration_min}:{duration_sec:02d}</div>
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
                                label = FEATURE_LABELS.get(c_name, c_name)
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

    elif mode == "📊 Metajuego":
        st.header("Metajuego & Tendencias")
        # Visualizaciones rápidas
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Partidas Analizadas", f"{len(df_matches):,}")
        c2.metric("Invocadores Únicos", f"{df_matches['jugador'].nunique():,}")
        
        # KDA Promedio global calculado
        global_kda = (df_matches['kills'].mean() + df_matches['assists'].mean()) / max(1, df_matches['deaths'].mean())
        c3.metric("KDA Promedio", f"{global_kda:.2f}")
        c4.metric("Winrate Global", f"{df_matches['win'].mean()*100:.1f}%")
        
        st.divider()
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig1 = px.box(df_matches, x='individualPosition', y='goldEarned', color='win', 
                         title="Distribución de Oro por Posición",
                         color_discrete_map={True: "#00f2ff", False: "#ff4b4b"})
            st.plotly_chart(fig1, width="stretch")
        with col_m2:
            fig2 = px.scatter(df_matches, x='visionScore', y='totalDamageDealtToChampions', color='win', 
                             size='kills', hover_data=['championName'],
                             title="Visión vs Daño (Tamaño = Kills)")
            st.plotly_chart(fig2, width="stretch")

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
