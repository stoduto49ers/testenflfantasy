# =============================================================================
# NFL 2026 DASHBOARD v4 — Streamlit (simplificado)
# Como rodar: streamlit run nfl_dashboard.py
# =============================================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(page_title="NFL 2026 — Fantasy Draft", page_icon="🏈", layout="wide")

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem; }
.block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

EXCEL_PATH = 'consolidado_nfl_2026.xlsx'

@st.cache_data
def carregar_dados():
    if not os.path.exists(EXCEL_PATH):
        return {}
    xl = pd.ExcelFile(EXCEL_PATH)
    data = {}
    for sheet in xl.sheet_names:
        data[sheet] = pd.read_excel(EXCEL_PATH, sheet_name=sheet)
    return data

dados = carregar_dados()
if not dados:
    st.error("❌ Arquivo não encontrado. Rode primeiro o nfl_modelo_completo.py.")
    st.stop()

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.title("🏈 NFL 2026 Draft")
aba = st.sidebar.selectbox("Visualizar", ["All Players", "QBs", "WRs", "RBs", "TEs",
                                           "Rookies/Ano 2", "Backtesting"])
conf_min = st.sidebar.slider("Confiança mínima", 0, 100, 0, step=10)
top_n    = st.sidebar.slider("Mostrar top N", 10, 300, 50)
st.sidebar.divider()
st.sidebar.caption("Score = Modelo 35% + FP 35% + Flock 30%")
st.sidebar.caption("Monte Carlo: 10.000 simulações por jogador")

# =============================================================================
# HELPER: primeira coluna disponível
# =============================================================================
def fcol(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

# =============================================================================
# CONTEÚDO PRINCIPAL
# =============================================================================
st.title("🏈 NFL 2026 — Fantasy Draft Board")
st.caption("Consensus: XGBoost 35% · FantasyPros 35% · FlockFantasy 30% · Monte Carlo CI")
st.divider()

if aba == "All Players":
    df = dados.get('All_Players', pd.DataFrame())
    if df.empty:
        st.warning("Aba All_Players não encontrada."); st.stop()

    if 'confianca' in df.columns:
        df = df[df['confianca'] >= conf_min]

    df = df.head(top_n)

    # Métricas
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Jogadores", len(df))
    with c2:
        if 'proj_fpts_final' in df.columns:
            st.metric("Média pts proj.", f"{df['proj_fpts_final'].mean():.1f}")
    with c3:
        if 'mc_bust_pct' in df.columns:
            st.metric("Bust médio %", f"{df['mc_bust_pct'].mean():.1f}%")
    with c4:
        if 'mc_boom_pct' in df.columns:
            st.metric("Boom médio %", f"{df['mc_boom_pct'].mean():.1f}%")

    st.divider()

    # Gráfico principal: ranking com intervalo Monte Carlo
    st.subheader("📊 Draft Board — Projeção Consensus + Intervalo 80% (Monte Carlo)")

    name_col = fcol(df, ['player_display_name','player'])
    pts_col  = fcol(df, ['proj_fpts_final','proj_fpts_ppr'])
    p10_col  = fcol(df, ['mc_p10'])
    p90_col  = fcol(df, ['mc_p90'])
    pos_col  = fcol(df, ['pos_rank_label','position_label','position'])

    if name_col and pts_col:
        df_plot = df.sort_values(pts_col, ascending=True).tail(40)
        cor_pos = {'QB':'#7c3aed','WR':'#06b6d4','RB':'#22c55e','TE':'#f59e0b'}
        if pos_col:
            df_plot['pos_base'] = df_plot[pos_col].str[:2]
            cores = df_plot['pos_base'].map(cor_pos).fillna('#94a3b8')
        else:
            cores = '#7c3aed'

        fig = go.Figure()

        # Intervalo Monte Carlo
        if p10_col and p90_col:
            fig.add_trace(go.Bar(
                name='Intervalo 80% (P10-P90)',
                x=df_plot[p90_col] - df_plot[p10_col],
                y=df_plot[name_col],
                orientation='h',
                base=df_plot[p10_col],
                marker_color='rgba(255,255,255,0.12)',
                marker_line_color='rgba(255,255,255,0.3)',
                marker_line_width=1,
                showlegend=True,
            ))

        # Projeção central
        fig.add_trace(go.Bar(
            name='Consensus Proj',
            x=df_plot[pts_col],
            y=df_plot[name_col],
            orientation='h',
            marker_color=cores if isinstance(cores, str) else cores.tolist(),
            text=df_plot[pts_col].round(1).astype(str),
            textposition='outside',
        ))

        fig.update_layout(
            barmode='overlay',
            height=max(500, len(df_plot)*24),
            margin=dict(l=10,r=80,t=10,b=10),
            xaxis_title='Fantasy Points PPR',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(orientation='h',yanchor='bottom',y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Tabela limpa
    st.subheader("📋 Draft Board Completo")
    show_cols = ['overall_rank','pos_rank_label','player_display_name','team',
                 'proj_fpts_final','proj_yards','proj_tds',
                 'mc_p10','mc_p90','mc_bust_pct','mc_boom_pct',
                 'solidez_label', 'situacao', 'flock_rank']
    show_cols = [c for c in show_cols if c in df.columns]
    df_tab = df[show_cols].copy()
    df_tab.columns = [c.replace('_',' ').title() for c in df_tab.columns]
    st.dataframe(df_tab, use_container_width=True, height=600)

    csv = df_tab.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Draft Board CSV", csv, 'nfl_2026_draft_board.csv', 'text/csv')

elif aba in ["QBs","WRs","RBs","TEs"]:
    df = dados.get(aba, pd.DataFrame())
    if df.empty:
        st.warning(f"Aba {aba} não encontrada."); st.stop()

    if 'confianca' in df.columns:
        df = df[df['confianca'] >= conf_min]
    df = df.head(top_n)

    rank_col = fcol(df, ['rank_final'])
    name_col = fcol(df, ['player_display_name'])
    pts_col  = fcol(df, ['proj_fpts_final','proj_fpts_ppr'])
    yards_col= fcol(df, ['proj_passing_yards','proj_receiving_yards','proj_rushing_yards','proj_yards'])
    tds_col  = fcol(df, ['proj_passing_tds','proj_receiving_tds','proj_rushing_tds','proj_tds'])
    fp_col   = fcol(df, ['fp_fantasy_pts'])
    p10_col  = fcol(df, ['mc_p10'])
    p90_col  = fcol(df, ['mc_p90'])

    # Métricas simples
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Jogadores", len(df))
    with c2:
        if pts_col: st.metric("Média pts proj.", f"{df[pts_col].mean():.1f}")
    with c3:
        if yards_col: st.metric("Média jardas proj.", f"{df[yards_col].mean():.0f}")
    with c4:
        if tds_col: st.metric("Média TDs proj.", f"{df[tds_col].mean():.1f}")

    st.divider()
    left, right = st.columns([3,2])

    with left:
        st.subheader(f"🏆 Ranking {aba} — Consensus Pts")
        if name_col and pts_col:
            df_plot = df.sort_values(pts_col, ascending=True).copy()
            fig = go.Figure()
            if p10_col and p90_col:
                fig.add_trace(go.Bar(
                    name='IC 80%',
                    x=df_plot[p90_col]-df_plot[p10_col],
                    y=df_plot[name_col], orientation='h',
                    base=df_plot[p10_col],
                    marker_color='rgba(255,255,255,0.10)',
                    marker_line_color='rgba(255,255,255,0.25)',
                    marker_line_width=1,
                ))
            fig.add_trace(go.Bar(
                name='Consensus',
                x=df_plot[pts_col], y=df_plot[name_col],
                orientation='h', marker_color='#7c3aed',
                text=df_plot[pts_col].round(1).astype(str), textposition='outside',
            ))
            fig.update_layout(
                barmode='overlay',
                height=max(400, len(df_plot)*26),
                margin=dict(l=10,r=80,t=10,b=10),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'), showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("🎯 Modelo vs FantasyPros")
        if name_col and pts_col and fp_col:
            df_comp = df.sort_values(pts_col, ascending=False).head(15)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name='Modelo', x=df_comp[name_col], y=df_comp[pts_col],
                                  marker_color='#7c3aed'))
            fig2.add_trace(go.Bar(name='FantasyPros', x=df_comp[name_col], y=df_comp[fp_col],
                                  marker_color='#06b6d4'))
            fig2.update_layout(
                barmode='group', height=400,
                margin=dict(l=10,r=10,t=10,b=100),
                xaxis_tickangle=-35,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                legend=dict(orientation='h',yanchor='bottom',y=1.02),
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Tabela simplificada — apenas o essencial
    st.subheader("📋 Tabela")
    show_cols = [rank_col, name_col, 'team', pts_col, yards_col, tds_col,
                 p10_col, p90_col, 'mc_bust_pct', 'mc_boom_pct',
                 'solidez_label', 'situacao', 'flock_rank']
    show_cols = [c for c in show_cols if c and c in df.columns]
    label_map = {
        pts_col:   'Consensus Pts', yards_col: 'Proj Yards', tds_col: 'Proj TDs',
        p10_col:   'Floor (P10)',   p90_col:   'Ceiling (P90)',
        rank_col:  'Rank', name_col: 'Player',
    }
    df_tab = df[show_cols].rename(columns=label_map)
    st.dataframe(df_tab, use_container_width=True, height=500)

    csv = df_tab.to_csv(index=False).encode('utf-8')
    st.download_button(f"⬇️ Download {aba} CSV", csv, f'nfl_2026_{aba.lower()}.csv', 'text/csv')

elif aba == "Rookies/Ano 2":
    df = dados.get('Rookies_Ano2', pd.DataFrame())
    if df.empty:
        st.warning("Aba Rookies_Ano2 não encontrada."); st.stop()
    st.subheader("🌟 Rookies 2026 e Jogadores Ano 2")
    st.dataframe(df, use_container_width=True, height=500)

elif aba == "Backtesting":
    df = dados.get('Backtesting', pd.DataFrame())
    if df.empty:
        st.warning("Aba Backtesting não encontrada."); st.stop()
    st.subheader("📈 Backtesting — 3 Janelas")

    if 'Posição' in df.columns and 'mae' in df.columns:
        fig = go.Figure()
        for pos in df['Posição'].unique():
            sub = df[df['Posição']==pos]
            fig.add_trace(go.Bar(name=pos, x=sub['label'] if 'label' in sub else sub.index.astype(str),
                                  y=sub['mae']))
        fig.update_layout(
            barmode='group', height=350,
            title='MAE por posição e janela (menor = melhor)',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df, use_container_width=True)

st.caption("NFL 2026 Fantasy Model · XGBoost + Monte Carlo · FantasyPros PPR · FlockFantasy Best Ball")
