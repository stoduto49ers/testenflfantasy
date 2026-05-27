# =============================================================================
# NFL 2026 DASHBOARD v5 — Clean & Simple
# streamlit run nfl_dashboard.py
# =============================================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(page_title="NFL 2026 Fantasy", page_icon="🏈", layout="wide")

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.5rem; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

EXCEL_PATH = 'consolidado_nfl_2026.xlsx'

@st.cache_data
def load_data():
    if not os.path.exists(EXCEL_PATH):
        return {}
    xl = pd.ExcelFile(EXCEL_PATH)
    return {s: pd.read_excel(EXCEL_PATH, sheet_name=s) for s in xl.sheet_names}

data = load_data()
if not data:
    st.error("❌ Arquivo não encontrado. Rode nfl_modelo_completo.py primeiro.")
    st.stop()

# Sidebar
st.sidebar.title("🏈 NFL 2026")
view = st.sidebar.selectbox("Aba", ["Draft Board", "QBs", "WRs", "RBs", "TEs", "Backtesting"])
top_n = st.sidebar.slider("Top N", 10, 300, 50)
st.sidebar.divider()
st.sidebar.caption("Modelo 33-37% · FP 31-35% · Flock 30-37%")
st.sidebar.caption("VBD = pts acima do replacement level")

def fcol(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

# Cores por posição
POS_COLORS = {'QB':'#7c3aed','WR':'#06b6d4','RB':'#22c55e','TE':'#f59e0b'}

# =============================================================================
if view == "Draft Board":
    df = data.get('All_Players', pd.DataFrame()).head(top_n)
    if df.empty:
        st.warning("All_Players não encontrado."); st.stop()

    st.title("🏈 NFL 2026 — Draft Board")
    st.caption("Value-Based Drafting · Consensus Pts · Monte Carlo CI · ADP Signal")
    st.divider()

    # Métricas
    c1,c2,c3,c4 = st.columns(4)
    pts_col = fcol(df,['proj_fpts_final','Proj_Pts'])
    with c1: st.metric("Jogadores", len(df))
    with c2:
        if pts_col: st.metric("Média pts", f"{df[pts_col].mean():.0f}")
    with c3:
        vbd_col = fcol(df,['vbd_score','VBD_Score'])
        if vbd_col: st.metric("VBD médio", f"{df[vbd_col].mean():.0f}")
    with c4:
        roi_col = fcol(df,['draft_roi','Draft_ROI'])
        if roi_col: st.metric("ROI médio", f"{df[roi_col].mean():.0f}")

    st.divider()

    # Gráfico principal — barras horizontais com cor por posição e CI do MC
    name_col = fcol(df,['player_display_name','Name'])
    p10_col  = fcol(df,['mc_p10','Floor_P10'])
    p90_col  = fcol(df,['mc_p90','Ceiling_P90'])
    pos_col  = fcol(df,['position_label','Position'])
    sig_col  = fcol(df,['draft_signal','Draft_Signal'])

    if name_col and pts_col:
        st.subheader("📊 Draft Board — Projeção + Intervalo Monte Carlo (80%)")
        df_plot = df.sort_values(pts_col, ascending=True).tail(40)

        if pos_col:
            df_plot['_pos'] = df_plot[pos_col].str[:2]
            cores = df_plot['_pos'].map(POS_COLORS).fillna('#94a3b8').tolist()
        else:
            cores = '#7c3aed'

        fig = go.Figure()
        if p10_col and p90_col:
            fig.add_trace(go.Bar(
                name='CI 80% (P10–P90)',
                x=df_plot[p90_col]-df_plot[p10_col],
                y=df_plot[name_col], orientation='h',
                base=df_plot[p10_col],
                marker_color='rgba(255,255,255,0.08)',
                marker_line_color='rgba(255,255,255,0.2)',
                marker_line_width=1,
            ))
        fig.add_trace(go.Bar(
            name='Consensus Pts',
            x=df_plot[pts_col], y=df_plot[name_col], orientation='h',
            marker_color=cores,
            text=df_plot[pts_col].round(0).astype(int).astype(str),
            textposition='outside',
        ))
        fig.update_layout(
            barmode='overlay', height=max(500, len(df_plot)*24),
            margin=dict(l=10,r=80,t=10,b=10),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Tabela limpa — apenas o essencial + time
    st.subheader("📋 Draft Board Completo")

    rank_col   = fcol(df,['overall_rank'])
    team_col   = fcol(df,['team','Team'])
    vbd_col    = fcol(df,['vbd_score','VBD_Score'])
    roi_col    = fcol(df,['draft_roi','Draft_ROI'])
    adp_col    = fcol(df,['fp_adp','Consensus_ADP'])
    trnd_col   = fcol(df,['target_round','Target_Round'])
    sig_col    = fcol(df,['draft_signal','Draft_Signal'])
    bust_col   = fcol(df,['mc_bust_pct','Bust_Pct'])
    boom_col   = fcol(df,['mc_boom_pct','Boom_Pct'])
    sit_col    = fcol(df,['situacao'])
    floor_col  = fcol(df,['mc_p10','Floor_P10'])
    ceil_col   = fcol(df,['mc_p90','Ceiling_P90'])
    flock_col  = fcol(df,['flock_rank'])

    show = [c for c in [
        rank_col, fcol(df,['pos_rank_label']), name_col, team_col,
        pts_col, vbd_col, roi_col,
        adp_col, trnd_col, sig_col,
        floor_col, ceil_col, bust_col, boom_col,
        sit_col, flock_col
    ] if c]

    labels = {
        rank_col:  'Rank', 'pos_rank_label':'Pos', name_col: 'Jogador', team_col: 'Time',
        pts_col:   'Proj Pts', vbd_col: 'VBD', roi_col: 'ROI',
        adp_col:   'ADP', trnd_col: 'Draftar na', sig_col: 'Sinal',
        floor_col: 'Floor', ceil_col: 'Ceiling',
        bust_col:  'Bust%', boom_col: 'Boom%',
        sit_col:   'Situação', flock_col: 'Flock Rank',
    }

    df_tab = df[show].rename(columns=labels)
    st.dataframe(df_tab, use_container_width=True, height=600)

    csv = df_tab.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Draft Board", csv, 'draft_board_2026.csv', 'text/csv')

# =============================================================================
elif view in ["QBs","WRs","RBs","TEs"]:
    df = data.get(view, pd.DataFrame()).head(top_n)
    if df.empty:
        st.warning(f"{view} não encontrado."); st.stop()

    pos_color = {'QBs':'#7c3aed','WRs':'#06b6d4','RBs':'#22c55e','TEs':'#f59e0b'}.get(view,'#7c3aed')

    name_col  = fcol(df,['player_display_name'])
    rank_col  = fcol(df,['rank_final'])
    team_col  = fcol(df,['team'])
    pts_col   = fcol(df,['proj_fpts_final','proj_fpts_ppr'])
    fp_col    = fcol(df,['fp_fantasy_pts'])
    yards_col = fcol(df,['proj_passing_yards','proj_receiving_yards','proj_rushing_yards','proj_yards'])
    tds_col   = fcol(df,['proj_passing_tds','proj_receiving_tds','proj_rushing_tds','proj_tds'])
    p10_col   = fcol(df,['mc_p10'])
    p90_col   = fcol(df,['mc_p90'])
    bust_col  = fcol(df,['mc_bust_pct'])
    boom_col  = fcol(df,['mc_boom_pct'])
    solid_col = fcol(df,['solidez_label','confianca_label'])
    sit_col   = fcol(df,['situacao'])
    flock_col = fcol(df,['flock_rank'])

    st.title(f"{'🟣' if view=='QBs' else '🔵' if view=='WRs' else '🟢' if view=='RBs' else '🟡'} {view} 2026")
    st.divider()

    # Métricas
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Jogadores", len(df))
    with c2:
        if pts_col: st.metric("Média consensus", f"{df[pts_col].mean():.0f} pts")
    with c3:
        if yards_col: st.metric("Média jardas", f"{df[yards_col].mean():.0f}")
    with c4:
        if tds_col: st.metric("Média TDs", f"{df[tds_col].mean():.1f}")

    st.divider()

    left, right = st.columns([3,2])

    with left:
        st.subheader(f"🏆 Ranking — Consensus Pts + CI 80%")
        if name_col and pts_col:
            df_plot = df.sort_values(pts_col, ascending=True).copy()
            fig = go.Figure()
            if p10_col and p90_col:
                fig.add_trace(go.Bar(
                    x=df_plot[p90_col]-df_plot[p10_col], y=df_plot[name_col],
                    orientation='h', base=df_plot[p10_col],
                    marker_color='rgba(255,255,255,0.08)',
                    marker_line_color='rgba(255,255,255,0.2)',
                    marker_line_width=1, name='CI 80%',
                ))
            fig.add_trace(go.Bar(
                x=df_plot[pts_col], y=df_plot[name_col], orientation='h',
                marker_color=pos_color, name='Consensus',
                text=df_plot[pts_col].round(0).astype(int).astype(str),
                textposition='outside',
            ))
            fig.update_layout(
                barmode='overlay', height=max(400, len(df_plot)*26),
                margin=dict(l=10,r=80,t=10,b=10),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'), showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("📊 Modelo vs FantasyPros")
        if name_col and pts_col and fp_col:
            df_c = df.sort_values(pts_col, ascending=False).head(15)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name='Consensus', x=df_c[name_col], y=df_c[pts_col], marker_color=pos_color))
            fig2.add_trace(go.Bar(name='FantasyPros', x=df_c[name_col], y=df_c[fp_col], marker_color='rgba(255,255,255,0.3)'))
            fig2.update_layout(
                barmode='group', height=400,
                margin=dict(l=10,r=10,t=10,b=110), xaxis_tickangle=-35,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Tabela limpa com time
    st.subheader("📋 Tabela")
    show = [c for c in [
        rank_col, name_col, team_col,
        pts_col, yards_col, tds_col,
        p10_col, p90_col, bust_col, boom_col,
        solid_col, sit_col, flock_col
    ] if c]
    labels = {
        rank_col:'#', name_col:'Jogador', team_col:'Time',
        pts_col:'Proj Pts', yards_col:'Jardas', tds_col:'TDs',
        p10_col:'Floor', p90_col:'Ceiling',
        bust_col:'Bust%', boom_col:'Boom%',
        solid_col:'Solidez', sit_col:'Situação', flock_col:'Flock',
    }
    df_tab = df[show].rename(columns=labels)
    st.dataframe(df_tab, use_container_width=True, height=500)

    csv = df_tab.to_csv(index=False).encode('utf-8')
    st.download_button(f"⬇️ Download {view}", csv, f'nfl_2026_{view.lower()}.csv', 'text/csv')

# =============================================================================
elif view == "Backtesting":
    df = data.get('Backtesting', pd.DataFrame())
    if df.empty:
        st.warning("Backtesting não encontrado."); st.stop()
    st.title("📈 Backtesting — 4 Janelas")
    st.caption("MAE: erro médio em pts | Corr: correlação projeção vs real (1.0 = perfeito)")
    st.divider()

    if 'Posição' in df.columns and 'mae' in df.columns:
        fig = go.Figure()
        pos_colors = {'QB':'#7c3aed','WR':'#06b6d4','RB':'#22c55e','TE':'#f59e0b'}
        for pos in df['Posição'].unique():
            sub = df[df['Posição']==pos]
            fig.add_trace(go.Bar(
                name=pos,
                x=sub['label'] if 'label' in sub else sub.index.astype(str),
                y=sub['mae'],
                marker_color=pos_colors.get(pos,'#94a3b8'),
                text=sub['mae'].astype(str), textposition='outside',
            ))
        fig.update_layout(
            barmode='group', height=380, title='MAE por posição (menor = melhor)',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Correlação
        fig2 = go.Figure()
        for pos in df['Posição'].unique():
            sub = df[df['Posição']==pos]
            fig2.add_trace(go.Bar(
                name=pos,
                x=sub['label'] if 'label' in sub else sub.index.astype(str),
                y=sub['corr'],
                marker_color=pos_colors.get(pos,'#94a3b8'),
                text=sub['corr'].round(3).astype(str), textposition='outside',
            ))
        fig2.update_layout(
            barmode='group', height=380, title='Correlação (maior = melhor)',
            yaxis=dict(range=[0,1.1]),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(df, use_container_width=True)

st.caption("NFL 2026 · XGBoost + Monte Carlo · FP · Flock · VBD · ADP")
