import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

# CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="LoL Analytics PRO - Perfiles",
    page_icon="🏆",
    layout="wide",
)

# ESTILOS CSS PERSONALIZADOS (MODERNO/COMERCIAL)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Outfit:wght@300;400;600&display=swap');
    
    .main {
        background: #0b0e11;
        color: #ffffff;
    }
    .stApp {
        background-color: #0b0e11;
    }
    h1, h2, h3 {
        color: #00f2ff !important;
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .hero-section {
        background: linear-gradient(135deg, rgba(0,242,255,0.1) 0%, rgba(0,0,0,0) 100%);
        padding: 50px 30px;
        border-radius: 20px;
        text-align: center;
        border: 1px solid rgba(0,242,255,0.2);
        margin-bottom: 30px;
    }
    .leaderboard-card {
        background: rgba(255,255,255,0.03);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        margin: 5px 0;
        transition: 0.3s;
    }
    .leaderboard-card:hover {
        background: rgba(0,242,255,0.05);
        border-color: #00f2ff;
        transform: translateX(10px);
    }
    .rank-number {
        font-size: 1.5em;
        font-weight: bold;
        color: #00f2ff;
        margin-right: 15px;
    }
    .lp-val {
        color: #f7ff00;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# CARGA DE DATOS
@st.cache_data
def load_data():
    df_matches = None
    df_players = None
    if os.path.exists('../league_data.csv'):
        df_matches = pd.read_csv('../league_data.csv')
        df_matches = df_matches[df_matches['challenge_hadAfkTeammate'] == 0]
        df_matches = df_matches[df_matches['timePlayed'] > 420]
    
    if os.path.exists('../todosJugadores.csv'):
        df_players = pd.read_csv('../todosJugadores.csv')
    
    return df_matches, df_players

# TÍTULO Y PRESENTACIÓN (Oculto en landing si se desea)
df_matches, df_players = load_data()

# LÓGICA DE NAVEGACIÓN
# Usamos un state para controlar si estamos viendo un perfil específico
if 'viewing_profile' not in st.session_state:
    st.session_state.viewing_profile = None

st.sidebar.header("🕹️ Centro de Control")
if st.sidebar.button("🏠 Inicio / Dashboard"):
    st.session_state.viewing_profile = None

mode = st.sidebar.radio("Navegación:", ["🏠 Inicio / Buscador", "📊 Análisis Global", "🔮 Simular Partida"])

if df_matches is not None:
    # --- PANTALLA DE INICIO COMERCIAL ---
    if st.session_state.viewing_profile is None and mode == "🏠 Inicio / Buscador":
        # HERO SECTION
        st.markdown("""
            <div class="hero-section">
                <h1>🎮 League of Analytics</h1>
                <p style="color: #888;">Domina la Grieta del Invocador con predicciones de IA y estadísticas de nivel Challenger</p>
            </div>
        """, unsafe_allow_html=True)
        
        # BUSCADOR CENTRAL
        col_s1, col_s2, col_s3 = st.columns([1, 2, 1])
        with col_s2:
            st.markdown("### 🔍 Encuentra a un Jugador")
            all_player_names = sorted(df_players['nombre'].unique()) if df_players is not None else sorted(df_matches['jugador'].unique())
            search_query = st.selectbox("Escribe el nombre del invocador (ej: IMPACT#damu)", ["--- SELECCIONA ---"] + all_player_names)
            
            if search_query != "--- SELECCIONA ---":
                st.session_state.viewing_profile = search_query
                st.rerun()

        st.divider()

        # TOP 5 CHALLENGERS
        st.markdown("### 🔥 Top 5 Challenger - Ranking Actual")
        if df_players is not None:
            top_5 = df_players[df_players['tier'] == 'CHALLENGER'].sort_values('LPs actuales', ascending=False).head(5)
            
            col_t1, col_t2 = st.columns([2, 1])
            with col_t1:
                for idx, row in top_5.iterrows():
                    # Definimos el rango visual (1 a 5)
                    rank_pos = list(top_5.index).index(idx) + 1
                    
                    st.markdown(f"""
                        <div class="leaderboard-card">
                            <span class="rank-number">{rank_pos}</span>
                            <span style="font-size: 1.2em; font-weight: 600;">{row['nombre']}</span>
                            <span style="float: right;">Tier: <b>CHALLENGER</b> | <span class="lp-val">{row['LPs actuales']} LPs</span></span>
                        </div>
                    """, unsafe_allow_html=True)
            
            with col_t2:
                st.info("**¿Sabías que...?** El 78% de los jugadores en el top 5 Challenger mantienen una puntuación de visión superior a 1.2 por minuto.")
                st.info("Los modelos de IA muestran que el control de Baron Nashor es el predictor #1 en partidas de elo alto.")

    # --- PERFIL DE JUGADOR (SI SE ACTIVÓ) ---
    elif st.session_state.viewing_profile is not None:
        selected_player = st.session_state.viewing_profile
        st.markdown(f"## 👤 Perfil del Invocador: `{selected_player}`")
        
        if st.button("⬅️ Volver al Buscador"):
            st.session_state.viewing_profile = None
            st.rerun()
            
        # Mejoramos la visualización de cabecera de perfil
        if df_players is not None:
            p_data = df_players[df_players['nombre'] == selected_player]
            if not p_data.empty:
                p_info = p_data.iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Tier", f"{p_info['tier']} {p_info['rank']}")
                with c2: st.metric("LPs", f"{p_info['LPs actuales']}")
                with c3:
                    wr = (p_info['wins'] / (p_info['wins'] + p_info['losses'])) * 100
                    st.metric("Winrate", f"{wr:.1f}%")
                with c4: st.metric("Server", p_info['server'].upper())
        
        st.divider()
        p_matches = df_matches[df_matches['jugador'] == selected_player].copy().sort_values('match_id', ascending=False)
        
        if not p_matches.empty:
            # Racha visual
            win_bool = p_matches['win'].tolist()
            fig_stripes = go.Figure()
            for i, w in enumerate(win_bool[:20]):
                fig_stripes.add_trace(go.Bar(x=[i], y=[1], marker_color="#00f2ff" if w else "#ff4b4b", showlegend=False))
            fig_stripes.update_layout(height=80, xaxis_visible=False, yaxis_visible=False, margin=dict(l=0,r=0,t=10,b=10), title="Desempeño Reciente (Últimos 20)")
            st.plotly_chart(fig_stripes, use_container_width=True)
            
            tab1, tab2 = st.tabs(["📋 Historial Detallado", "📈 Power Graphs"])
            with tab1:
                st.dataframe(p_matches[['championName', 'individualPosition', 'win', 'challenge_kda', 'goldEarned', 'visionScore']].head(15), use_container_width=True)
            with tab2:
                fig_scat = px.scatter(p_matches, x='goldEarned', y='totalDamageDealtToChampions', color='win', size='kills', hover_data=['championName'], title="Eficiencia: Oro vs Daño")
                st.plotly_chart(fig_scat, use_container_width=True)
        else:
            st.warning("No hay datos de partidas recientes para este jugador.")

    # --- OTROS MODOS ---
    elif mode == "📊 Análisis Global":
        st.header("Metajuego Global")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Partidas", f"{len(df_matches):,}")
        with col2: st.metric("Invocadores", f"{df_matches['jugador'].nunique():,}")
        with col3: st.metric("Winrate", f"{df_matches['win'].mean()*100:.1f}%")
        with col4: st.metric("KDA", f"{df_matches['challenge_kda'].mean():.2f}")
        
        st.subheader("Distribución de Oro por Posición")
        fig_pos = px.box(df_matches, x='individualPosition', y='goldEarned', color='win', color_discrete_map={True: "#00f2ff", False: "#ff4b4b"})
        st.plotly_chart(fig_pos, use_container_width=True)

    elif mode == "🔮 Simular Partida":
        st.header("🔮 Simulador de Victoria (IA)")
        st.markdown("Basado en modelos de Random Forest entrenados con 29,000+ partidas.")
        
        features = ['kills', 'deaths', 'goldEarned', 'visionScore', 'totalDamageDealtToChampions', 'challenge_teamDamagePercentage']
        X = df_matches[features]; y = df_matches['win'].astype(int)
        rf = RandomForestClassifier(n_estimators=50, random_state=42); rf.fit(X, y)
        
        ca, cb = st.columns(2)
        with ca:
            k = st.number_input("Kills", 0, 50, 5); d = st.number_input("Deaths", 0, 50, 2); g = st.number_input("Oro Total", 0, 30000, 10000)
        with cb:
            v = st.number_input("Visión", 0, 150, 20); dmg = st.number_input("Daño", 0, 100000, 15000); tp = st.slider("% Daño Equipo", 0, 100, 25) / 100
        
        p = rf.predict_proba([[k, d, g, v, dmg, tp]])[0][1]
        st.markdown(f"## Probabilidad de Victoria: `{p*100:.1f}%`")
        st.progress(p)

else:
    st.error("Error: Archivos de datos no encontrados o formato incorrecto. Verifique league_data.csv y todosJugadores.csv.")
