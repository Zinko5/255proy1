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

# 1. SINCRONIZACIÓN DE URL -> ESTADO (Al cargar)
import urllib.parse
if "profile" in st.query_params:
    st.session_state.viewing_profile = urllib.parse.unquote(st.query_params["profile"])
if "nav" in st.query_params:
    st.session_state.nav_mode = st.query_params["nav"]
else:
    if "nav_mode" not in st.session_state: st.session_state.nav_mode = "🔥 Inicio / Rankings"

if df_matches is not None:
    models_dict, main_scaler = get_trained_models(df_matches)
    
    # 2. MANEJO DE NAVEGACIÓN (Sidebar)
    if 'viewing_profile' not in st.session_state: st.session_state.viewing_profile = None
    
    st.sidebar.header("🕹️ Navegación")
    if st.sidebar.button("🏠 Inicio"): 
        st.session_state.viewing_profile = None
        st.session_state.nav_mode = "🔥 Inicio / Rankings"
        st.query_params.clear()
        st.rerun()
    
    # Usar el estado para el radio button
    mode = st.sidebar.radio("Sección:", 
                            ["🔥 Inicio / Rankings", "🧠 IA & Simulador", "📊 Metajuego", "👤 Perfil"],
                            index=["🔥 Inicio / Rankings", "🧠 IA & Simulador", "📊 Metajuego", "👤 Perfil"].index(st.session_state.nav_mode),
                            key="nav_radio")
    
    # Sincronizar cambios del radio al estado y URL
    if mode != st.session_state.nav_mode:
        st.session_state.nav_mode = mode
        st.query_params["nav"] = mode
        st.rerun()

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
            st.markdown("""
                <small style='color: #888;'>Leyenda: <b>⭐ MVP</b> (Max KDA Victoria) | <b>💥 Carry</b> (Max Daño Equipo) | <b>👁️ Visión</b> (Score > 35)</small>
            """, unsafe_allow_html=True)
            
            for _, match_row in p_match_history.head(15).iterrows():
                m_id = match_row['match_id']
                is_win = match_row['win']
                duration_min = match_row['timePlayed'] // 60
                duration_sec = match_row['timePlayed'] % 60
                
                bg_color = "rgba(0, 242, 255, 0.08)" if is_win else "rgba(255, 75, 75, 0.08)"
                border_color = "#00f2ff" if is_win else "#ff4b4b"
                
                with st.container():
                    st.markdown(f"""
                        <div style='background: {bg_color}; border-left: 8px solid {border_color}; padding: 15px; border-radius: 8px; margin-bottom: 5px; 
                                    display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <span style='font-weight: bold; color: {border_color};'>{"VICTORIA" if is_win else "DERROTA"}</span> | 
                                <b style='font-size: 18px;'>{match_row['championName']}</b> | {match_row['kills']}/{match_row['deaths']}/{match_row['assists']} | 
                                KDA: <b>{match_row['challenge_kda']:.2f}</b>
                            </div>
                            <div style='color: #888; font-size: 14px;'>
                                duración de la partida: {duration_min}:{duration_sec:02d}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("Ver detalle de la partida"):
                        all_players = df_matches[df_matches['match_id'] == m_id].sort_values('side')
                        
                        for side_name, color_team in [("Blue", "#51a1ff"), ("Red", "#ff5151")]:
                            st.markdown(f"<h4 style='color: {color_team}; margin-top:20px;'>Equipo {side_name}</h4>", unsafe_allow_html=True)
                            team_df = all_players[all_players['side'] == side_name]
                            
                            # Encabezados de tabla
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
                                row_css = "background: rgba(255,255,255,0.05);" if is_current else ""
                                r_cols = st.columns([2.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2.5])
                                
                                # Badges dinámicos
                                badges = []
                                if p_row['challenge_kda'] >= team_df['challenge_kda'].max() and p_row['win']: badges.append("⭐")
                                if p_row['visionScore'] > 35: badges.append("👁️")
                                if p_row['totalDamageDealtToChampions'] >= team_df['totalDamageDealtToChampions'].max(): badges.append("💥")
                                
                                # Link a nuevo tab (usando query params y codificando el nombre)
                                import urllib.parse
                                encoded_name = urllib.parse.quote(p_row['jugador'])
                                current_url = f"?nav=👤+Perfil&profile={encoded_name}"
                                r_cols[0].markdown(f"**[{p_row['jugador']}]({current_url})**" if not is_current else f"<span style='color:#00f2ff;'>{p_row['jugador']}</span>", unsafe_allow_html=True)
                                r_cols[1].write(p_row['championName'])
                                r_cols[2].write(f"{p_row['kills']}/{p_row['deaths']}/{p_row['assists']}")
                                r_cols[3].write(f"{p_row['goldEarned']:,}")
                                r_cols[4].write(f"{p_row['totalDamageDealtToChampions']:,}")
                                r_cols[5].write(f"{p_row['visionScore']}")
                                r_cols[6].write(" ".join(badges))
                        st.divider()

        with tab_stats:
            st.header("📈 Análisis de Desempeño Promedio")
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                if not p_match_history.empty:
                    stats = p_match_history.mean(numeric_only=True)
                    radar_data = pd.DataFrame(dict(
                        r=[
                            min(100, (stats['challenge_kda']/5)*100),
                            min(100, (stats['visionScore']/50)*100),
                            min(100, (stats['challenge_teamDamagePercentage']*300)),
                            min(100, (stats['champLevel']/18)*100),
                            min(100, (stats['goldEarned']/15000)*100)
                        ],
                        theta=['KDA','Visión','Daño %','Nivel','Oro']
                    ))
                    fig_radar = px.line_polar(radar_data, r='r', theta='theta', line_close=True, range_r=[0,100], title="Radar de Habilidades")
                    fig_radar.update_traces(fill='toself', fillcolor='rgba(0,242,255,0.3)', line_color='#00f2ff')
                    st.plotly_chart(fig_radar, use_container_width=True)
            
            with col_s2:
                st.subheader("Promedios del Jugador")
                avg_stats = p_match_history.mean(numeric_only=True)
                st.write(f"- **KDA Promedio:** {avg_stats['challenge_kda']:.2f}")
                st.write(f"- **Oro por Partida:** {avg_stats['goldEarned']:.0f}")
                st.write(f"- **Daño a Campeones:** {avg_stats['totalDamageDealtToChampions']:.0f}")
                st.write(f"- **Puntaje de Visión:** {avg_stats['visionScore']:.1f}")
                st.write(f"- **Participación en Kill:** {avg_stats['challenge_killParticipation']*100:.1f}%")

    elif mode == "📊 Metajuego":
        st.header("Metajuego & Tendencias")
        # Visualizaciones rápidas
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Partidas Analizadas", f"{len(df_matches):,}")
        c2.metric("Invocadores Únicos", f"{df_matches['jugador'].nunique():,}")
        c3.metric("KDA Promedio", f"{df_matches['challenge_kda'].mean():.2f}")
        c4.metric("Winrate Global", f"{df_matches['win'].mean()*100:.1f}%")
        
        st.divider()
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig1 = px.box(df_matches, x='individualPosition', y='goldEarned', color='win', 
                         title="Distribución de Oro por Posición",
                         color_discrete_map={True: "#00f2ff", False: "#ff4b4b"})
            st.plotly_chart(fig1, use_container_width=True)
        with col_m2:
            fig2 = px.scatter(df_matches, x='visionScore', y='totalDamageDealtToChampions', color='win', 
                             size='kills', hover_data=['championName'],
                             title="Visión vs Daño (Tamaño = Kills)")
            st.plotly_chart(fig2, use_container_width=True)

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
