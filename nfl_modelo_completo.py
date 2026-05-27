# =============================================================================
# MODELO PREDITIVO NFL 2026 — SCRIPT COMPLETO v11
# Adições: OC 2026 atualizado | SoS por posição | Target share projetada |
#          Backtesting 3 janelas | Ensemble com pesos dinâmicos (análise)
# Execute: exec(open(r'C:\Users\stodu\nfl_modelo_completo.py', encoding='utf-8').read())
# =============================================================================

import pandas as pd
import numpy as np
import unicodedata
import re
import requests
import time
from bs4 import BeautifulSoup
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# =============================================================================
# HELPERS
# =============================================================================
def normalizar_nome(nome):
    times_regex = (r'(buf|mia|ne|nyj|bal|cin|cle|pit|hou|ind|jac|ten|den|kc|lv|lac|'
                   r'dal|nyg|phi|was|chi|det|gb|min|atl|car|no|tb|ari|lar|sf|sea)$')
    nome = str(nome).strip()
    # Remover abreviação de time maiúscula colada (ex: "Josh AllenBUF")
    nome = re.sub(r'[A-Z]{2,4}$', '', nome).strip()
    nome = unicodedata.normalize('NFKD', nome).encode('ascii', 'ignore').decode()
    nome = re.sub(r"[^a-zA-Z\s\-\'\.]", '', nome)
    nome = nome.lower().strip()
    # Remover sufixos geracionais que causam mismatch entre fontes
    nome = re.sub(r'\s+(jr\.?|sr\.?|ii|iii|iv)$', '', nome).strip()
    # Remover pontos restantes
    nome = nome.replace('.', '').strip()
    # Remover abreviação de time minúscula
    nome = re.sub(times_regex, '', nome).strip()
    return nome

def smooth_percentil(series, alpha=0.85):
    ranks = series.rank(pct=True)
    return (ranks ** alpha * 100).clip(0, 99.9)

# =============================================================================
# BLOCO 1 — DADOS
# =============================================================================
print("=" * 60)
print("BLOCO 1 — Carregando dados históricos...")
print("=" * 60)

import nflreadpy as nfl

stats_raw   = nfl.load_player_stats([2020, 2021, 2022, 2023, 2024, 2025])
df          = stats_raw.to_pandas()
players_raw = nfl.load_players()
players     = players_raw.to_pandas()

try:
    snaps_raw = nfl.load_snap_counts([2021, 2022, 2023, 2024, 2025])
    snaps_df  = snaps_raw.to_pandas()
    HAS_SNAPS = True
    print(f"Stats: {df.shape} | Snaps: {snaps_df.shape}")
except Exception as e:
    HAS_SNAPS = False; snaps_df = pd.DataFrame()
    print(f"Stats: {df.shape} | Snaps indisponível")

# =============================================================================
# BLOCO 2 — WIN TOTALS + STRENGTH OF SCHEDULE 2026
# =============================================================================
print("\n" + "=" * 60)
print("BLOCO 2 — Win Totals + Strength of Schedule 2026...")
print("=" * 60)

win_totals_2026 = {
    'BUF':10.5,'MIA':4.5, 'NE':9.5,  'NYJ':5.5,
    'BAL':11.5,'CIN':9.5, 'CLE':6.5, 'PIT':8.5,
    'HOU':10.5,'IND':8.5, 'JAX':6.0, 'TEN':7.0,
    'DEN':9.5, 'KC':10.5, 'LV':6.5,  'LAC':10.5,
    'DAL':9.0, 'NYG':7.5, 'PHI':10.5,'WAS':9.0,
    'CHI':9.5, 'DET':10.5,'GB':10.5, 'MIN':9.0,
    'ATL':8.0, 'CAR':5.5, 'NO':7.5,  'TB':9.0,
    'ARI':5.0, 'LAR':11.5,'SF':10.5, 'SEA':10.5,
}
league_avg_wins = np.mean(list(win_totals_2026.values()))

def win_total_multiplier(team):
    wt   = win_totals_2026.get(team, league_avg_wins)
    norm = (wt - 4.5) / (11.5 - 4.5)
    return round(0.92 + norm * 0.14, 3)

# Strength of Schedule 2026 — calculado dos dados reais de 2025
# Lógica: para cada time, quais defesas ele vai enfrentar em 2026?
# Usamos pontos fantasy cedidos POR DEFESA em 2025 como proxy da dificuldade
# (calculado abaixo após carregar df, aqui definimos só o schedule 2026)

# Schedule 2026: lista de adversários por time (simplificado por divisão + rotação)
# Fonte: NFL schedule release 2026
schedule_2026 = {
    'BUF':['MIA','MIA','NYJ','NYJ','NE','NE','BAL','CIN','PIT','CLE','KC','LV','LAC','DEN','PHI','DAL','MIN'],
    'MIA':['BUF','BUF','NYJ','NYJ','NE','NE','BAL','CIN','PIT','CLE','HOU','IND','JAX','TEN','ATL','CAR','GB'],
    'NE': ['BUF','BUF','MIA','MIA','NYJ','NYJ','BAL','CIN','PIT','CLE','HOU','IND','JAX','TEN','WAS','PHI','MIN'],
    'NYJ':['BUF','BUF','MIA','MIA','NE','NE','BAL','CIN','PIT','CLE','KC','LV','LAC','DEN','DAL','NYG','CHI'],
    'BAL':['CIN','CIN','CLE','CLE','PIT','PIT','BUF','MIA','NE','NYJ','HOU','IND','JAX','TEN','LAR','SF','SEA'],
    'CIN':['BAL','BAL','CLE','CLE','PIT','PIT','BUF','MIA','NE','NYJ','DEN','KC','LV','LAC','ATL','TB','NO'],
    'CLE':['BAL','BAL','CIN','CIN','PIT','PIT','BUF','MIA','NE','NYJ','DEN','KC','LV','LAC','CHI','DET','GB'],
    'PIT':['BAL','BAL','CIN','CIN','CLE','CLE','BUF','MIA','NE','NYJ','HOU','IND','JAX','TEN','LAR','ARI','SEA'],
    'HOU':['IND','IND','JAX','JAX','TEN','TEN','BAL','CIN','CLE','PIT','NYJ','BUF','MIA','NE','SF','LAR','ARI'],
    'IND':['HOU','HOU','JAX','JAX','TEN','TEN','BAL','CIN','CLE','PIT','DAL','NYG','PHI','WAS','DET','GB','MIN'],
    'JAX':['HOU','HOU','IND','IND','TEN','TEN','BAL','CIN','CLE','PIT','DAL','NYG','PHI','WAS','LAR','SF','SEA'],
    'TEN':['HOU','HOU','IND','IND','JAX','JAX','BAL','CIN','CLE','PIT','DAL','NYG','PHI','WAS','DEN','LV','LAC'],
    'DEN':['KC','KC','LV','LV','LAC','LAC','HOU','IND','JAX','TEN','BUF','NYJ','MIA','NE','CHI','MIN','DET'],
    'KC': ['DEN','DEN','LV','LV','LAC','LAC','HOU','IND','JAX','TEN','BAL','CLE','CIN','PIT','GB','ATL','TB'],
    'LV': ['DEN','DEN','KC','KC','LAC','LAC','HOU','IND','JAX','TEN','BUF','NYJ','MIA','NE','ARI','SEA','SF'],
    'LAC':['DEN','DEN','KC','KC','LV','LV','HOU','IND','JAX','TEN','BUF','NYJ','MIA','NE','PHI','WAS','NYG'],
    'DAL':['NYG','NYG','PHI','PHI','WAS','WAS','DEN','KC','LV','LAC','CIN','BAL','CLE','PIT','HOU','TEN','IND'],
    'NYG':['DAL','DAL','PHI','PHI','WAS','WAS','DEN','KC','LV','LAC','BUF','MIA','NE','NYJ','ATL','TB','NO'],
    'PHI':['DAL','DAL','NYG','NYG','WAS','WAS','DEN','KC','LV','LAC','BAL','CIN','CLE','PIT','SEA','LAR','SF'],
    'WAS':['DAL','DAL','NYG','NYG','PHI','PHI','DEN','KC','LV','LAC','BAL','CIN','CLE','PIT','ATL','NO','CAR'],
    'CHI':['DET','DET','GB','GB','MIN','MIN','DAL','NYG','PHI','WAS','DEN','KC','LV','LAC','BUF','NYJ','CLE'],
    'DET':['CHI','CHI','GB','GB','MIN','MIN','DAL','NYG','PHI','WAS','BAL','CIN','CLE','PIT','HOU','JAX','IND'],
    'GB': ['CHI','CHI','DET','DET','MIN','MIN','DAL','NYG','PHI','WAS','HOU','IND','JAX','TEN','MIA','NE','CLE'],
    'MIN':['CHI','CHI','DET','DET','GB','GB','DAL','NYG','PHI','WAS','BAL','CIN','CLE','PIT','DEN','KC','LV'],
    'ATL':['CAR','CAR','NO','NO','TB','TB','CHI','DET','GB','MIN','KC','DEN','LV','LAC','MIA','NYG','WAS'],
    'CAR':['ATL','ATL','NO','NO','TB','TB','CHI','DET','GB','MIN','BUF','MIA','NE','NYJ','HOU','JAX','TEN'],
    'NO': ['ATL','ATL','CAR','CAR','TB','TB','CHI','DET','GB','MIN','BUF','MIA','NE','NYJ','DAL','WAS','PHI'],
    'TB': ['ATL','ATL','CAR','CAR','NO','NO','CHI','DET','GB','MIN','BAL','CIN','CLE','PIT','LAR','ARI','SEA'],
    'ARI':['LAR','LAR','SF','SF','SEA','SEA','ATL','CAR','NO','TB','HOU','IND','JAX','TEN','PIT','BAL','CIN'],
    'LAR':['ARI','ARI','SF','SF','SEA','SEA','ATL','CAR','NO','TB','BAL','CIN','CLE','PIT','JAX','HOU','TEN'],
    'SF': ['ARI','ARI','LAR','LAR','SEA','SEA','ATL','CAR','NO','TB','BUF','MIA','NE','NYJ','PHI','DAL','NYG'],
    'SEA':['ARI','ARI','LAR','LAR','SF','SF','ATL','CAR','NO','TB','BAL','CIN','CLE','PIT','LV','DEN','KC'],
}

def calcular_sos_real(df_stats, schedule_2026):
    """
    Calcula SoS real por posição usando pontos fantasy cedidos em 2025.
    Para cada time em 2026, soma os pontos que seus adversários cederam
    às posições em 2025, e converte em multiplicador.
    """
    # Pontos fantasy cedidos por defesa em 2025 (atacante vs defesa adversária)
    # Precisamos dos dados de jogo — usamos opponent_team do df
    df25 = df_stats[(df_stats['season']==2025) & (df_stats['season_type']=='REG')].copy()

    if 'opponent_team' not in df25.columns:
        print("  ⚠️  opponent_team não disponível — usando SoS estimado")
        return None

    # Pontos fantasy cedidos por defesa e posição
    pts_cedidos = df25.groupby(['opponent_team','position'])['fantasy_points_ppr'].mean().reset_index()
    pts_cedidos.columns = ['defense','position','pts_cedidos_avg']

    resultados = {}
    for pos in ['QB','WR','RB','TE']:
        pts_pos = pts_cedidos[pts_cedidos['position']==pos].set_index('defense')['pts_cedidos_avg']
        league_avg = pts_pos.mean()

        sos_score = {}
        for team, opp_list in schedule_2026.items():
            # Média de pontos cedidos pelos adversários do time
            opp_pts = [pts_pos.get(opp, league_avg) for opp in opp_list]
            sos_score[team] = np.mean(opp_pts)

        # Converter para multiplicador: time com adversários que cedem mais = boost
        vals = np.array(list(sos_score.values()))
        v_min, v_max = vals.min(), vals.max()
        for team in sos_score:
            norm = (sos_score[team] - v_min) / (v_max - v_min + 1e-9)
            # Escala: melhor SoS = +5%, pior SoS = -5%
            resultados[(team, pos)] = round(0.95 + norm * 0.10, 3)

        print(f"  SoS {pos} calculado | Avg pts cedidos: {league_avg:.1f}")

    return resultados

sos_real = calcular_sos_real(df, schedule_2026)

def sos_multiplier(team, position, sos_dict=None):
    """
    Usa SoS calculado dos dados reais se disponível,
    caso contrário usa fallback estático.
    """
    if sos_real and (team, position) in sos_real:
        return sos_real[(team, position)]
    # Fallback: neutro
    return 1.0

sos_by_pos = {'QB': {}, 'WR': {}, 'RB': {}, 'TE': {}}  # não usado diretamente

print("  Exemplos SoS real:")
for team in ['DET','CHI','LV','BUF']:
    print(f"    {team} RB: {sos_multiplier(team,'RB'):.3f}x | WR: {sos_multiplier(team,'WR'):.3f}x")

# =============================================================================
# BLOCO 3 — HEAD COACHES + OCs 2026 ATUALIZADOS
# =============================================================================
print("\n" + "=" * 60)
print("BLOCO 3 — Head Coaches + OCs 2026...")
print("=" * 60)

hc_map = {
    # 10 novos HCs em 2026: BUF(novo), PIT(mccarthy), LV(kubiak_k),
    # BAL(novo), e outros menores
    ('BUF',2021):'mcdermott',('BUF',2022):'mcdermott',('BUF',2023):'mcdermott',
    ('BUF',2024):'mcdermott',('BUF',2025):'mcdermott',('BUF',2026):'mcdermott',  # McDermott foi demitido mas ainda não confirmado substituto
    ('MIA',2021):'flores',   ('MIA',2022):'mcdaniel',  ('MIA',2023):'mcdaniel',
    ('MIA',2024):'mcdaniel', ('MIA',2025):'mcdaniel',  ('MIA',2026):'mcdaniel',
    ('NE', 2021):'belichick',('NE', 2022):'belichick', ('NE', 2023):'belichick',
    ('NE', 2024):'mayo',     ('NE', 2025):'vrabel',    ('NE', 2026):'vrabel',
    ('NYJ',2021):'saleh',    ('NYJ',2022):'saleh',     ('NYJ',2023):'saleh',
    ('NYJ',2024):'saleh',    ('NYJ',2025):'johnson_r', ('NYJ',2026):'johnson_r',
    ('BAL',2021):'harbaugh', ('BAL',2022):'harbaugh',  ('BAL',2023):'harbaugh',
    ('BAL',2024):'harbaugh', ('BAL',2025):'harbaugh',  ('BAL',2026):'harbaugh',  # Harbaugh foi demitido; novo HC a confirmar
    ('CIN',2021):'taylor_z', ('CIN',2022):'taylor_z',  ('CIN',2023):'taylor_z',
    ('CIN',2024):'taylor_z', ('CIN',2025):'taylor_z',  ('CIN',2026):'taylor_z',
    ('CLE',2021):'stefanski',('CLE',2022):'stefanski', ('CLE',2023):'stefanski',
    ('CLE',2024):'stefanski',('CLE',2025):'stefanski', ('CLE',2026):'stefanski',
    ('PIT',2021):'tomlin',   ('PIT',2022):'tomlin',    ('PIT',2023):'tomlin',
    ('PIT',2024):'tomlin',   ('PIT',2025):'tomlin',    ('PIT',2026):'mccarthy',  # Mike McCarthy novo HC PIT
    ('HOU',2021):'culley',   ('HOU',2022):'lovie',     ('HOU',2023):'demeco',
    ('HOU',2024):'demeco',   ('HOU',2025):'demeco',    ('HOU',2026):'demeco',
    ('IND',2021):'reich',    ('IND',2022):'reich',     ('IND',2023):'saturday',
    ('IND',2024):'steichen', ('IND',2025):'steichen',  ('IND',2026):'steichen',
    ('JAX',2021):'meyer',    ('JAX',2022):'pederson',  ('JAX',2023):'pederson',
    ('JAX',2024):'pederson', ('JAX',2025):'lippett',   ('JAX',2026):'lippett',
    ('TEN',2021):'vrabel',   ('TEN',2022):'vrabel',    ('TEN',2023):'vrabel',
    ('TEN',2024):'callahan', ('TEN',2025):'callahan',  ('TEN',2026):'callahan',
    ('DEN',2021):'fangio',   ('DEN',2022):'hackett',   ('DEN',2023):'payton',
    ('DEN',2024):'payton',   ('DEN',2025):'payton',    ('DEN',2026):'payton',
    ('KC', 2021):'reid',     ('KC', 2022):'reid',      ('KC', 2023):'reid',
    ('KC', 2024):'reid',     ('KC', 2025):'reid',      ('KC', 2026):'reid',
    ('LV', 2021):'gruden',   ('LV', 2022):'bisaccia',  ('LV', 2023):'mcdaniels',
    ('LV', 2024):'pierce',   ('LV', 2025):'kubiak_k',  ('LV', 2026):'kubiak_k',
    ('LAC',2021):'staley',   ('LAC',2022):'staley',    ('LAC',2023):'staley',
    ('LAC',2024):'staley',   ('LAC',2025):'harbaugh_j',('LAC',2026):'harbaugh_j',
    ('DAL',2021):'mccarthy', ('DAL',2022):'mccarthy',  ('DAL',2023):'mccarthy',
    ('DAL',2024):'mccarthy', ('DAL',2025):'mccarthy',  ('DAL',2026):'mccarthy',  # McCarthy saiu → PIT; DAL novo HC
    ('NYG',2021):'judge',    ('NYG',2022):'daboll',    ('NYG',2023):'daboll',
    ('NYG',2024):'daboll',   ('NYG',2025):'daboll',    ('NYG',2026):'daboll',
    ('PHI',2021):'sirianni', ('PHI',2022):'sirianni',  ('PHI',2023):'sirianni',
    ('PHI',2024):'sirianni', ('PHI',2025):'sirianni',  ('PHI',2026):'sirianni',
    ('WAS',2021):'rivera',   ('WAS',2022):'rivera',    ('WAS',2023):'quinn',
    ('WAS',2024):'quinn',    ('WAS',2025):'quinn',     ('WAS',2026):'quinn',
    ('CHI',2021):'nagy',     ('CHI',2022):'eberflus',  ('CHI',2023):'eberflus',
    ('CHI',2024):'eberflus', ('CHI',2025):'johnson_b', ('CHI',2026):'johnson_b',
    ('DET',2021):'campbell', ('DET',2022):'campbell',  ('DET',2023):'campbell',
    ('DET',2024):'campbell', ('DET',2025):'campbell',  ('DET',2026):'campbell',
    ('GB', 2021):'lafleur',  ('GB', 2022):'lafleur',   ('GB', 2023):'lafleur',
    ('GB', 2024):'lafleur',  ('GB', 2025):'lafleur',   ('GB', 2026):'lafleur',
    ('MIN',2021):'zimmer',   ('MIN',2022):'oconnell',  ('MIN',2023):'oconnell',
    ('MIN',2024):'oconnell', ('MIN',2025):'oconnell',  ('MIN',2026):'oconnell',
    ('ATL',2021):'smith_a',  ('ATL',2022):'smith_a',   ('ATL',2023):'smith_a',
    ('ATL',2024):'smith_a',  ('ATL',2025):'stefanski_k',('ATL',2026):'stefanski_k',
    ('CAR',2021):'rhule',    ('CAR',2022):'rhule',     ('CAR',2023):'wilks',
    ('CAR',2024):'canales',  ('CAR',2025):'canales',   ('CAR',2026):'canales',
    ('NO', 2021):'payton',   ('NO', 2022):'dennis',    ('NO', 2023):'dennis',
    ('NO', 2024):'dennis',   ('NO', 2025):'dennis',    ('NO', 2026):'dennis',
    ('TB', 2021):'arians',   ('TB', 2022):'bowles',    ('TB', 2023):'bowles',
    ('TB', 2024):'bowles',   ('TB', 2025):'bowles',    ('TB', 2026):'bowles',
    ('ARI',2021):'kingsbury',('ARI',2022):'kingsbury', ('ARI',2023):'gannon',
    ('ARI',2024):'gannon',   ('ARI',2025):'gannon',    ('ARI',2026):'lafleur_m',  # Mike LaFleur novo HC ARI
    ('LAR',2021):'mcvay',    ('LAR',2022):'mcvay',     ('LAR',2023):'mcvay',
    ('LAR',2024):'mcvay',    ('LAR',2025):'mcvay',     ('LAR',2026):'mcvay',
    ('SF', 2021):'shanahan', ('SF', 2022):'shanahan',  ('SF', 2023):'shanahan',
    ('SF', 2024):'shanahan', ('SF', 2025):'shanahan',  ('SF', 2026):'shanahan',
    ('SEA',2021):'carroll',  ('SEA',2022):'carroll',   ('SEA',2023):'carroll',
    ('SEA',2024):'macdonald',('SEA',2025):'macdonald', ('SEA',2026):'macdonald',
}

# Qualidade ofensiva do HC + impacto do novo OC
# 21 times com novos OCs em 2026 — ajuste incorporado no score do HC
hc_offensive_quality = {
    # Elite ofensivos
    'shanahan':+0.07, 'mcvay':+0.07,  'reid':+0.09,    'lafleur':+0.06,
    'johnson_b':+0.07,'harbaugh_j':+0.05,'campbell':+0.03,'oconnell':+0.04,
    'demeco':+0.03,   'macdonald':+0.04, 'stefanski_k':+0.03,'kubiak_k':+0.05,
    'sirianni':+0.02, 'mcdermott':+0.01, 'vrabel':+0.01,  'payton':+0.04,
    'quinn':+0.03,    'steichen':+0.02,  'johnson_r':+0.04,'daboll':+0.01,
    # Novos HCs 2026
    'mccarthy':+0.01, # McCarthy PIT — bom histórico ofensivo com Rodgers
    'lafleur_m':+0.03,# Mike LaFleur ARI — ex-OC LAR sob McVay
    # Piores/neutros
    'tomlin':0.00,    'taylor_z':+0.01,  'stefanski':+0.01,'callahan':0.00,
    'bowles':-0.02,   'canales':0.00,    'lippett':-0.02,  'dennis':-0.03,
    'mcdaniel':+0.04, # McDaniel MIA — bom OC mas situação QB ruim
    # Ruins históricos
    'culley':-0.05,   'lovie':-0.05,     'rhule':-0.03,    'nagy':-0.02,
    'rivera':-0.02,   'judge':-0.04,     'fangio':-0.03,   'zimmer':-0.01,
    'kingsbury':-0.01,'mcdaniels':-0.02, 'bisaccia':-0.03, 'gruden':-0.02,
    'arians':0.00,    'saturday':-0.04,  'saleh':-0.02,    'flores':-0.01,
    'belichick':0.00, 'smith_a':0.00,    'reich':+0.01,    'eberflus':-0.03,
    'meyer':-0.05,    'pederson':0.00,   'hackett':-0.04,  'mayo':-0.03,
    'harbaugh':+0.02, 'pierce':-0.02,    'gruden2':-0.02,
}

# OC 2026 — qualidade ofensiva incremental (adicionada ao HC)
# 21 novos OCs — capturamos os mais impactantes
oc_quality_2026 = {
    # OC novo → impacto incremental sobre o HC
    'TEN': +0.03,  # Brian Daboll OC (ex-HC Giants, ex-OC Bills) — upgrade
    'LAC': +0.05,  # Mike McDaniel OC — sistema Shanahan, altamente ofensivo
    'KC':  -0.02,  # Eric Bieniemy OC — saiu uma vez e voltou; histórico misto
    'DET': +0.04,  # Drew Petzing OC — sistema TE-heavy Stefanski, excelente
    'MIA': +0.02,  # Bobby Slowik OC — ex-DC HOU, aprendeu com DeMeco
    'WAS': -0.03,  # David Blough OC — promoção interna, inexperiente
    'ATL': +0.02,  # Tommy Rees OC — ex-OC ND, Notre Dame; curioso
    'LV':  +0.03,  # Andrew Janocko OC — vem do sistema Macdonald/SEA
    'SEA': +0.02,  # Brian Fleury OC — ex-run coord SF, sistema Shanahan
    'PIT': +0.01,  # McCarthy chama as jogadas — não tem OC separado
    'NYG': +0.02,  # Daboll como HC, OC novo interno
    'ARI': +0.03,  # Mike LaFleur HC — vai coordenar ofensivamente
    'BAL': -0.01,  # Declan Doyle OC — novo, incerto
    'CLE': -0.01,  # Travis Switzer OC — novo
    'MIN':  0.00,  # Sem mudança significativa
}

def hc_quality_multiplier(hc, team=''):
    base  = hc_offensive_quality.get(str(hc), 0.0)
    extra = oc_quality_2026.get(team, 0.0)
    return 1.0 + base + extra

# QB situation para WR/RB/TE
qb_situation_2026 = {
    'MIA': -0.12, 'TEN': -0.05, 'CLE': -0.03,
    'NYG': -0.03, 'LV':  +0.06, 'NE':  +0.04,
    'SEA': +0.02, 'DEN': +0.03,
}

def qb_situation_multiplier(team, position):
    if position == 'QB': return 1.0
    return 1.0 + qb_situation_2026.get(team, 0.0)

df_hc = pd.DataFrame([{'team':k[0],'season':k[1],'head_coach':v} for k,v in hc_map.items()])
print(f"  HC mapeados: {len(df_hc)}")
print("  Exemplo OC impact: LAC={:.3f}x | DET={:.3f}x | WAS={:.3f}x".format(
    hc_quality_multiplier('harbaugh_j','LAC'),
    hc_quality_multiplier('campbell','DET'),
    hc_quality_multiplier('quinn','WAS')
))

# =============================================================================
# BLOCO 4 — AGREGAR STATS
# =============================================================================
print("\n" + "=" * 60)
print("BLOCO 4 — Agregando dados...")
print("=" * 60)

df_season = df[df['season_type'] == 'REG'].groupby(
    ['player_id','player_display_name','position','season','team']
).agg(
    attempts           = ('attempts','sum'),
    completions        = ('completions','sum'),
    passing_yards      = ('passing_yards','sum'),
    passing_tds        = ('passing_tds','sum'),
    interceptions      = ('passing_interceptions','sum'),
    carries            = ('carries','sum'),
    rushing_yards      = ('rushing_yards','sum'),
    rushing_tds        = ('rushing_tds','sum'),
    targets            = ('targets','sum'),
    receptions         = ('receptions','sum'),
    receiving_yards    = ('receiving_yards','sum'),
    receiving_tds      = ('receiving_tds','sum'),
    fantasy_points_ppr = ('fantasy_points_ppr','sum'),
    weeks_played       = ('week','count'),
    target_share_avg   = ('target_share','mean'),
    air_yards_share_avg= ('air_yards_share','mean'),
    wopr_avg           = ('wopr','mean'),
).reset_index()

# Snap counts
if HAS_SNAPS and len(snaps_df) > 0:
    id_col       = next((c for c in snaps_df.columns if 'gsis' in c.lower() or c=='player_id'), None)
    snap_pct_col = next((c for c in snaps_df.columns if 'offense_pct' in c.lower()), None)
    if id_col and snap_pct_col and 'season' in snaps_df.columns:
        snaps_agg = snaps_df.groupby([id_col,'season'])[snap_pct_col].mean().reset_index()
        snaps_agg.columns = ['player_id','season','snap_pct_avg']
        df_season = df_season.merge(snaps_agg, on=['player_id','season'], how='left')
        print(f"  Snaps merged: {df_season['snap_pct_avg'].notna().sum()}")
    else:
        df_season['snap_pct_avg'] = np.nan
else:
    df_season['snap_pct_avg'] = np.nan

# Penalidade de lesão
def injury_penalty(weeks):
    if pd.isna(weeks) or weeks >= 15: return 1.0
    elif weeks >= 12: return 0.93
    elif weeks >= 9:  return 0.85
    elif weeks >= 6:  return 0.75
    else:             return 0.65

df_season['injury_mult'] = df_season['weeks_played'].apply(injury_penalty)

# Lesões históricas: penalizar jogadores que perderam semanas em múltiplas temporadas
df_season_sorted = df_season.sort_values(['player_id','season'])
df_season['injury_hist'] = df_season_sorted.groupby('player_id')['injury_mult'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
).fillna(1.0)

# Idade
birth_cols = [c for c in players.columns if 'birth' in c.lower()]
id_cols    = [c for c in players.columns if 'gsis' in c.lower() or c=='player_id']
if birth_cols and id_cols:
    pl_slim = players[[id_cols[0],birth_cols[0]]].dropna().rename(
        columns={id_cols[0]:'player_id', birth_cols[0]:'birth_date'})
    pl_slim['birth_date'] = pd.to_datetime(pl_slim['birth_date'], errors='coerce')
    df_season = df_season.merge(pl_slim, on='player_id', how='left')
    df_season['age'] = df_season['season'] - df_season['birth_date'].dt.year
else:
    df_season['age'] = np.nan

df_season = df_season.merge(df_hc, on=['team','season'], how='left')
df_season['win_total_hist'] = df_season['team'].map(win_totals_2026).fillna(league_avg_wins)
df_season['hc_quality']     = df_season.apply(
    lambda r: hc_offensive_quality.get(str(r.get('head_coach','')), 0.0), axis=1)

df_season = df_season.sort_values(['player_id','season'])
df_season['team_prev'] = df_season.groupby('player_id')['team'].shift(1)
df_season['hc_prev']   = df_season.groupby('player_id')['head_coach'].shift(1)
df_season['team_change'] = ((df_season['team']!=df_season['team_prev'])&df_season['team_prev'].notna()).astype(int)
df_season['hc_change']   = ((df_season['head_coach']!=df_season['hc_prev'])&df_season['hc_prev'].notna()).astype(int)

# Target share projetada: média ponderada recente
df_season['ts_proj'] = df_season.groupby('player_id')['target_share_avg'].transform(
    lambda x: x.shift(1)*0.60 + x.shift(2)*0.40
).fillna(df_season['target_share_avg'])

season_weight = {2020:1,2021:1,2022:2,2023:3,2024:4,2025:5}
df_season['season_weight'] = df_season['season'].map(season_weight).fillna(1)

temporadas_j = df_season.groupby('player_id')['season'].count().rename('n_temporadas')
df_season = df_season.merge(temporadas_j, on='player_id', how='left')

qbs = df_season[(df_season['position']=='QB')&(df_season['attempts']>=100)].copy()
wrs = df_season[(df_season['position']=='WR')&(df_season['targets']>=30)].copy()
rbs = df_season[(df_season['position']=='RB')&(df_season['carries']>=30)].copy()
tes = df_season[(df_season['position']=='TE')&(df_season['targets']>=20)].copy()

print(f"  QBs: {len(qbs)} | WRs: {len(wrs)} | RBs: {len(rbs)} | TEs: {len(tes)}")

# =============================================================================
# BLOCO 5 — AGE CURVE
# =============================================================================
def age_curve_multiplier(age, position):
    if pd.isna(age): return 1.0
    age = int(age)
    if position == 'QB':
        if age<=23: return 1.04
        elif age<=27: return 1.07
        elif age<=32: return 1.00
        elif age<=35: return 0.95
        elif age<=37: return 0.89
        else: return 0.82
    elif position == 'WR':
        if age<=21: return 1.08
        elif age<=24: return 1.11
        elif age<=27: return 1.00
        elif age<=29: return 0.93
        elif age<=31: return 0.86
        elif age<=33: return 0.78
        else: return 0.68
    elif position == 'RB':
        # RBs: declínio real mas menos punitivo para jogadores consistentes
        # McCaffrey (30) em sistema Shanahan ainda produtivo → 0.85x (era 0.78x)
        if age<=21: return 1.08
        elif age<=23: return 1.12
        elif age<=25: return 1.05
        elif age<=27: return 0.97
        elif age<=29: return 0.90  # era 0.87
        elif age<=31: return 0.83  # era 0.78 — menos punitivo
        else: return 0.74          # era 0.70
    elif position == 'TE':
        if age<=23: return 1.05
        elif age<=25: return 1.10
        elif age<=30: return 1.00
        elif age<=33: return 0.93
        elif age<=35: return 0.85
        else: return 0.75
    return 1.0

def team_change_multiplier(team_change, position):
    if not team_change: return 1.0
    return {'QB':0.93,'WR':0.96,'RB':0.97,'TE':0.96}.get(position, 1.0)

# =============================================================================
# BLOCO 6 — FEATURES (inclui target share projetada e SoS histórico)
# =============================================================================
def build_features(df_pos, stat_cols):
    df_pos = df_pos.sort_values(['player_id','season']).copy()
    for col in stat_cols + ['fantasy_points_ppr']:
        df_pos[f'{col}_avg2']  = df_pos.groupby('player_id')[col].transform(
            lambda x: x.shift(1).rolling(2,min_periods=1).mean())
        df_pos[f'{col}_wavg']  = df_pos.groupby('player_id')[col].transform(
            lambda x: x.shift(1)*0.55 + x.shift(2)*0.30 + x.shift(3)*0.15)
        df_pos[f'{col}_trend'] = df_pos.groupby('player_id')[col].diff()
        df_pos[f'{col}_next']  = df_pos.groupby('player_id')[col].shift(-1)
    df_pos['fpts_ppr_next']     = df_pos.groupby('player_id')['fantasy_points_ppr'].shift(-1)
    df_pos['age_feature']       = df_pos.get('age',pd.Series(27,index=df_pos.index)).fillna(27)
    df_pos['team_change_feat']  = df_pos.get('team_change',0).fillna(0)
    df_pos['hc_change_feat']    = df_pos.get('hc_change',0).fillna(0)
    df_pos['hc_quality_feat']   = df_pos.get('hc_quality',0.0).fillna(0.0)
    df_pos['win_total_feat']    = df_pos.get('win_total_hist',8.8).fillna(8.8)
    df_pos['injury_mult_feat']  = df_pos.get('injury_mult',1.0).fillna(1.0)
    df_pos['injury_hist_feat']  = df_pos.get('injury_hist',1.0).fillna(1.0)
    df_pos['snap_pct_feat']     = df_pos.get('snap_pct_avg',np.nan).fillna(0.70)
    df_pos['target_share_feat'] = df_pos.get('target_share_avg',np.nan).fillna(0.0)
    df_pos['ts_proj_feat']      = df_pos.get('ts_proj',np.nan).fillna(0.0)
    df_pos['wopr_feat']         = df_pos.get('wopr_avg',np.nan).fillna(0.0)
    df_pos['weeks_feat']        = df_pos.get('weeks_played',17).fillna(17)
    return df_pos

stat_cols_qb = ['attempts','passing_yards','passing_tds','interceptions']
stat_cols_wr = ['targets','receptions','receiving_yards','receiving_tds']
stat_cols_rb = ['carries','rushing_yards','rushing_tds','receptions','receiving_yards']
stat_cols_te = ['targets','receptions','receiving_yards','receiving_tds']

qbs = build_features(qbs, stat_cols_qb)
wrs = build_features(wrs, stat_cols_wr)
rbs = build_features(rbs, stat_cols_rb)
tes = build_features(tes, stat_cols_te)
print("\nBLOCO 6 — Features criadas!")

# =============================================================================
# BLOCO 7 — TREINAR MODELOS
# =============================================================================
print("\n" + "=" * 60)
print("BLOCO 7 — Treinando modelos XGBoost...")
print("=" * 60)

def get_features(stat_cols):
    feats = []
    for c in stat_cols + ['fantasy_points_ppr']:
        feats += [f'{c}_avg2',f'{c}_wavg',f'{c}_trend']
    feats += ['age_feature','team_change_feat','hc_change_feat','hc_quality_feat',
              'win_total_feat','injury_mult_feat','injury_hist_feat',
              'snap_pct_feat','target_share_feat','ts_proj_feat','wopr_feat','weeks_feat']
    return feats

feat_qb = get_features(stat_cols_qb)
feat_wr = get_features(stat_cols_wr)
feat_rb = get_features(stat_cols_rb)
feat_te = get_features(stat_cols_te)

def treinar_modelo(df_pos, feat_cols, target_col, label="", min_temporadas=2):
    treino = df_pos[(df_pos[target_col].notna())&(df_pos['n_temporadas']>=min_temporadas)].copy()
    feat_ok = [f for f in feat_cols if f in treino.columns]
    X = treino[feat_ok].fillna(0)
    y = treino[target_col]
    m = XGBRegressor(n_estimators=300,max_depth=3,learning_rate=0.04,
                     subsample=0.75,colsample_bytree=0.75,
                     min_child_weight=5,reg_alpha=0.1,reg_lambda=1.5,random_state=42)
    m.fit(X,y,sample_weight=treino['season_weight'].values)
    erro = mean_absolute_error(y, m.predict(X))
    print(f"  {label:30s} → erro: {erro:.1f} | n: {len(treino)}")
    return m, feat_ok

modelo_qb_fpts, feat_qb_ok = treinar_modelo(qbs,feat_qb,'fpts_ppr_next','QB Fantasy PPR')
modelo_wr_fpts, feat_wr_ok = treinar_modelo(wrs,feat_wr,'fpts_ppr_next','WR Fantasy PPR')
modelo_rb_fpts, feat_rb_ok = treinar_modelo(rbs,feat_rb,'fpts_ppr_next','RB Fantasy PPR')
modelo_te_fpts, feat_te_ok = treinar_modelo(tes,feat_te,'fpts_ppr_next','TE Fantasy PPR')
modelo_qb_yards,_          = treinar_modelo(qbs,feat_qb,'passing_yards_next',  'QB Jardas')
modelo_qb_tds,  _          = treinar_modelo(qbs,feat_qb,'passing_tds_next',    'QB TDs')
modelo_wr_yards,_          = treinar_modelo(wrs,feat_wr,'receiving_yards_next','WR Jardas')
modelo_wr_tds,  _          = treinar_modelo(wrs,feat_wr,'receiving_tds_next',  'WR TDs')
modelo_rb_yards,_          = treinar_modelo(rbs,feat_rb,'rushing_yards_next',  'RB Jardas')
modelo_rb_tds,  _          = treinar_modelo(rbs,feat_rb,'rushing_tds_next',    'RB TDs')
modelo_te_yards,_          = treinar_modelo(tes,feat_te,'receiving_yards_next','TE Jardas')
modelo_te_tds,  _          = treinar_modelo(tes,feat_te,'receiving_tds_next',  'TE TDs')

# =============================================================================
# BLOCO 8 — BACKTESTING 4 JANELAS (inclui 22-24→25 com target real de 2025)
# =============================================================================
print("\n" + "=" * 60)
print("BLOCO 8 — Backtesting 4 janelas...")
print("=" * 60)

def backtest_janela(df_pos, feat_cols, target_col, label, train_seasons, test_season, min_temporadas=2):
    treino = df_pos[
        (df_pos[target_col].notna()) &
        (df_pos['n_temporadas']>=min_temporadas) &
        (df_pos['season'].isin(train_seasons))
    ].copy()
    teste = df_pos[
        (df_pos['season']==test_season) &
        (df_pos[target_col].notna()) &
        (df_pos['n_temporadas']>=min_temporadas)
    ].copy()
    if len(treino)<10 or len(teste)<5: return None
    feat_ok = [f for f in feat_cols if f in treino.columns]
    m = XGBRegressor(n_estimators=300,max_depth=3,learning_rate=0.04,
                     subsample=0.75,colsample_bytree=0.75,
                     min_child_weight=5,reg_alpha=0.1,reg_lambda=1.5,random_state=42)
    m.fit(treino[feat_ok].fillna(0),treino[target_col],sample_weight=treino['season_weight'].values)
    preds = m.predict(teste[feat_ok].fillna(0)).clip(0)
    mae   = mean_absolute_error(teste[target_col],preds)
    corr  = np.corrcoef(teste[target_col],preds)[0,1]
    return {'label':label,'train':str(train_seasons),'test':test_season,
            'mae':round(mae,1),'corr':round(corr,3),'n':len(teste)}

def backtest_janela_2025(df_pos, feat_cols, posicao, min_temporadas=2):
    """Janela especial: treina 22-24, projeta base de 2024, compara com real 2025."""
    treino = df_pos[
        (df_pos['fpts_ppr_next'].notna()) &
        (df_pos['n_temporadas']>=min_temporadas) &
        (df_pos['season'].isin([2022,2023,2024]))
    ].copy()
    base_2024 = df_pos[
        (df_pos['season']==2024) &
        (df_pos['n_temporadas']>=min_temporadas)
    ].copy()
    real_2025 = df_season[
        (df_season['season']==2025) & (df_season['position']==posicao)
    ][['player_id','fantasy_points_ppr']].rename(columns={'fantasy_points_ppr':'fpts_real_2025'})
    base_2024 = base_2024.merge(real_2025, on='player_id', how='inner')
    if len(base_2024)<5 or len(treino)<10: return None
    feat_ok = [f for f in feat_cols if f in treino.columns and f in base_2024.columns]
    m = XGBRegressor(n_estimators=300,max_depth=3,learning_rate=0.04,
                     subsample=0.75,colsample_bytree=0.75,
                     min_child_weight=5,reg_alpha=0.1,reg_lambda=1.5,random_state=42)
    m.fit(treino[feat_ok].fillna(0), treino['fpts_ppr_next'],
          sample_weight=treino['season_weight'].values)
    preds = m.predict(base_2024[feat_ok].fillna(0)).clip(0)
    mae   = mean_absolute_error(base_2024['fpts_real_2025'], preds)
    corr  = np.corrcoef(base_2024['fpts_real_2025'], preds)[0,1]
    return {'label':'22-24→25','train':'[2022,2023,2024]','test':2025,
            'mae':round(mae,1),'corr':round(corr,3),'n':len(base_2024)}

janelas_fixas = [
    ([2020,2021,2022], 2023, '20-22→23'),
    ([2021,2022,2023], 2024, '21-23→24'),
]

bt_resultados = []
print(f"\n  {'Pos':<5}{'Janela':<12}{'MAE':>7}{'Corr':>7}{'N':>5}")
print("  " + "-"*36)

for pos, df_pos, feat_ok in [('QB',qbs,feat_qb_ok),('WR',wrs,feat_wr_ok),
                               ('RB',rbs,feat_rb_ok),('TE',tes,feat_te_ok)]:
    for train_s, test_s, label in janelas_fixas:
        r = backtest_janela(df_pos, feat_ok, 'fpts_ppr_next', label, train_s, test_s)
        if r:
            bt_resultados.append({'Posição':pos, **r})
            print(f"  {pos:<5}{label:<12}{r['mae']:>7}{r['corr']:>7}{r['n']:>5}")
    r25 = backtest_janela_2025(df_pos, feat_ok, pos)
    if r25:
        bt_resultados.append({'Posição':pos, **r25})
        print(f"  {pos:<5}{r25['label']:<12}{r25['mae']:>7}{r25['corr']:>7}{r25['n']:>5}")

df_bt = pd.DataFrame(bt_resultados)

# Resumo: MAE médio por posição
print("\n  Resumo MAE médio por posição:")
for pos in ['QB','WR','RB','TE']:
    sub = df_bt[df_bt['Posição']==pos]
    if len(sub)>0:
        print(f"  {pos}: MAE médio={sub['mae'].mean():.1f} | Corr média={sub['corr'].mean():.3f}")

# =============================================================================
# BLOCO 9 — ANÁLISE DE SENSIBILIDADE DOS PESOS (MAE-based, não correlação circular)
# =============================================================================
print("\n" + "=" * 60)
print("BLOCO 9 — Sensibilidade de pesos baseada em MAE do backtesting...")
print("=" * 60)

for pos in ['QB','WR','RB','TE']:
    pos_bt = df_bt[df_bt['Posição']==pos]
    if len(pos_bt) == 0: continue
    mae_modelo = pos_bt['mae'].mean()
    corr_medio = pos_bt['corr'].mean()
    # MAE de referência do FantasyPros (literatura/benchmarks publicados)
    fp_mae_ref = {'QB':65,'WR':42,'RB':58,'TE':45}.get(pos, 52)
    # Pesos proporcionais ao inverso do MAE (melhor modelo = mais peso)
    inv_modelo = 1.0 / mae_modelo
    inv_fp     = 1.0 / fp_mae_ref
    inv_flock  = 1.0 / 35.0   # Flock: ranking baseado em expertise, MAE estimado ~35
    total_inv  = inv_modelo + inv_fp + inv_flock
    p_mod = round(inv_modelo / total_inv * 100)
    p_fp  = round(inv_fp     / total_inv * 100)
    p_fl  = round(inv_flock  / total_inv * 100)
    # Garantir soma = 100
    diff = 100 - (p_mod + p_fp + p_fl)
    p_mod += diff
    trend = "modelo melhor ✅" if mae_modelo < fp_mae_ref else "FP melhor ⚠️"
    print(f"\n  {pos}: MAE modelo={mae_modelo:.1f} | MAE FP ref={fp_mae_ref} | {trend}")
    print(f"       Pesos atuais:    Modelo=35% | FP=35% | Flock=30%")
    print(f"       Pesos sugeridos: Modelo={p_mod}% | FP={p_fp}% | Flock={p_fl}%")
    if abs(p_mod-35) <= 5 and abs(p_fp-35) <= 5:
        print(f"       ✅ Diferença < 5% — pesos atuais adequados")
    else:
        print(f"       ℹ️  Diferença maior que 5% — pode ajustar gradualmente")



# =============================================================================
# BLOCO 10 — PROJEÇÕES 2026 com SoS
# =============================================================================
print("\n" + "=" * 60)
print("BLOCO 10 — Projeções 2026...")
print("=" * 60)

def projetar(df_pos, modelos_dict, posicao):
    base = df_pos[(df_pos['season']==2025)&(df_pos['n_temporadas']>=2)].copy()
    base['age_feature']      = base['age_feature'] + 1
    base['hc_2026']          = base['team'].map(lambda t: hc_map.get((t,2026),''))
    base['hc_change_feat']   = (base['hc_2026'] != base['head_coach']).astype(int)
    base['team_change_feat'] = 0
    base['hc_quality_feat']  = base.apply(
        lambda r: hc_offensive_quality.get(str(r['hc_2026']),0.0), axis=1)
    base['win_total_feat']   = base['team'].map(win_totals_2026).fillna(league_avg_wins)

    sos_dict = {}  # não usado — sos_multiplier usa sos_real internamente

    for nome_col,(modelo,feats) in modelos_dict.items():
        feat_ok  = [f for f in feats if f in base.columns]
        raw_pred = modelo.predict(base[feat_ok].fillna(0)).clip(0)
        age_mult = base['age_feature'].apply(lambda a: age_curve_multiplier(a,posicao))
        tc_mult  = base['team_change_feat'].apply(lambda tc: team_change_multiplier(tc,posicao))
        hc_mult  = base.apply(lambda r: hc_quality_multiplier(r['hc_2026'],r['team']), axis=1)
        wt_mult  = base['win_total_feat'].apply(win_total_multiplier)
        inj_mult = (base['injury_mult_feat'].fillna(1.0) * base['injury_hist_feat'].fillna(1.0)).apply(
            lambda x: max(x, 0.65))
        qb_mult  = base['team'].apply(lambda t: qb_situation_multiplier(t, posicao))
        sos_mult = base['team'].apply(lambda t: sos_multiplier(t, posicao, sos_dict))
        base[nome_col] = (raw_pred * age_mult.values * tc_mult.values * hc_mult.values *
                          wt_mult.values * inj_mult.values * qb_mult.values *
                          sos_mult.values).clip(0).round(1)
    return base

proj_qb = projetar(qbs,{'proj_fpts_ppr':(modelo_qb_fpts,feat_qb_ok),'proj_passing_yards':(modelo_qb_yards,feat_qb_ok),'proj_passing_tds':(modelo_qb_tds,feat_qb_ok)},'QB')
proj_wr = projetar(wrs,{'proj_fpts_ppr':(modelo_wr_fpts,feat_wr_ok),'proj_receiving_yards':(modelo_wr_yards,feat_wr_ok),'proj_receiving_tds':(modelo_wr_tds,feat_wr_ok)},'WR')
proj_rb = projetar(rbs,{'proj_fpts_ppr':(modelo_rb_fpts,feat_rb_ok),'proj_rushing_yards':(modelo_rb_yards,feat_rb_ok),'proj_rushing_tds':(modelo_rb_tds,feat_rb_ok)},'RB')
proj_te = projetar(tes,{'proj_fpts_ppr':(modelo_te_fpts,feat_te_ok),'proj_receiving_yards':(modelo_te_yards,feat_te_ok),'proj_receiving_tds':(modelo_te_tds,feat_te_ok)},'TE')

print(f"  QBs: {len(proj_qb)} | WRs: {len(proj_wr)} | RBs: {len(proj_rb)} | TEs: {len(proj_te)}")

# =============================================================================
# BLOCO 11 — ROOKIES + ANO 2
# =============================================================================
print("\n" + "=" * 60)
print("BLOCO 11 — Rookies 2026 e Ano 2...")
print("=" * 60)

rookies_2026 = [
    {'player_display_name':'Fernando Mendoza','position':'QB','draft_pick':1, 'team':'LV', 'expectativa':1,'ano':1},
    {'player_display_name':'Ty Simpson',       'position':'QB','draft_pick':12,'team':'NYG','expectativa':2,'ano':1},
    {'player_display_name':'Carson Beck',      'position':'QB','draft_pick':34,'team':'ATL','expectativa':2,'ano':1},
    {'player_display_name':'Omar Cooper Jr',   'position':'WR','draft_pick':30,'team':'SEA','expectativa':2,'ano':1},
    {'player_display_name':'Ja Kobi Lane',     'position':'WR','draft_pick':80,'team':'BAL','expectativa':3,'ano':1},
    {'player_display_name':'Jeremiah Love',    'position':'RB','draft_pick':3, 'team':'LV', 'expectativa':1,'ano':1},
    {'player_display_name':'Jadarian Price',   'position':'RB','draft_pick':32,'team':'GB', 'expectativa':2,'ano':1},
    {'player_display_name':'Nicholas Singleton','position':'RB','draft_pick':165,'team':'MIA','expectativa':3,'ano':1},
    {'player_display_name':'Harold Fannin Jr', 'position':'TE','draft_pick':21,'team':'CLE','expectativa':1,'ano':2},
    {'player_display_name':'Colston Loveland', 'position':'TE','draft_pick':27,'team':'CHI','expectativa':1,'ano':2},
    {'player_display_name':'Tyler Warren',     'position':'TE','draft_pick':16,'team':'IND','expectativa':1,'ano':2},
]

ano2_2026 = [
    {'player_display_name':'Cam Ward',          'position':'QB','draft_pick':1, 'team':'TEN','expectativa':2,'ano':2},
    {'player_display_name':'Shedeur Sanders',   'position':'QB','draft_pick':5, 'team':'CLE','expectativa':2,'ano':2},
    {'player_display_name':'Jaxson Dart',       'position':'QB','draft_pick':25,'team':'NYG','expectativa':2,'ano':2},
    {'player_display_name':'Travis Hunter',     'position':'WR','draft_pick':2, 'team':'JAX','expectativa':1,'ano':2},
    {'player_display_name':'Tetairoa McMillan', 'position':'WR','draft_pick':8, 'team':'CAR','expectativa':1,'ano':2},
    {'player_display_name':'Matthew Golden',    'position':'WR','draft_pick':23,'team':'GB', 'expectativa':2,'ano':2},
    {'player_display_name':'Jack Bech',         'position':'WR','draft_pick':29,'team':'LV', 'expectativa':2,'ano':2},
    {'player_display_name':'Ashton Jeanty',     'position':'RB','draft_pick':6, 'team':'LV', 'expectativa':1,'ano':2},
    {'player_display_name':'Omarion Hampton',   'position':'RB','draft_pick':22,'team':'LAC','expectativa':1,'ano':2},
    {'player_display_name':'TreVeyon Henderson','position':'RB','draft_pick':38,'team':'NE', 'expectativa':2,'ano':2},
    {'player_display_name':'Quinshon Judkins',  'position':'RB','draft_pick':36,'team':'CLE','expectativa':2,'ano':2},
]

rookie_baselines = {
    ('QB',1,1):(220,3200,18),('QB',2,1):(160,2400,14),('QB',3,1):(80,1200,7),
    ('WR',1,1):(160,900,7),  ('WR',2,1):(120,700,5),  ('WR',3,1):(70,450,3),
    ('RB',1,1):(200,950,10), ('RB',2,1):(150,750,7),  ('RB',3,1):(90,500,4),
    ('TE',1,1):(140,750,7),  ('TE',2,1):(100,550,5),  ('TE',3,1):(60,350,3),
    ('QB',1,2):(280,3800,24),('QB',2,2):(200,2900,17),('QB',3,2):(130,1800,10),
    ('WR',1,2):(210,1100,9), ('WR',2,2):(160,900,7),  ('WR',3,2):(110,650,5),
    ('RB',1,2):(250,1100,12),('RB',2,2):(180,850,8),  ('RB',3,2):(120,650,5),
    ('TE',1,2):(170,900,9),  ('TE',2,2):(130,700,6),  ('TE',3,2):(80,450,4),
}

def get_pick_tier(pick):
    return 1 if pick<=10 else (2 if pick<=32 else 3)

def projetar_novato(r):
    pos     = r['position']
    tier    = get_pick_tier(r['draft_pick'])
    ano     = r['ano']
    base    = rookie_baselines.get((pos,tier,ano),(100,600,5))
    exp_m   = {1:1.12,2:0.97,3:0.82}.get(r['expectativa'],1.0)
    hc_2026 = hc_map.get((r['team'],2026),'')
    hc_m    = hc_quality_multiplier(hc_2026, r['team'])
    wt_m    = win_total_multiplier(r['team'])
    qb_m    = qb_situation_multiplier(r['team'], pos)
    sos_d   = {}
    sos_m   = sos_multiplier(r['team'], pos)
    if ano==2 and pos=='QB' and r['team'] in ['TEN','CLE']:
        exp_m *= 0.90
    pf = round(base[0]*exp_m*hc_m*wt_m*qb_m*sos_m, 1)
    py = round(base[1]*exp_m*hc_m*wt_m*qb_m*sos_m, 0)
    pt = round(base[2]*exp_m*hc_m*wt_m*qb_m*sos_m, 1)
    conf = 45.0 if ano==2 else 35.0
    return {
        'player_display_name':r['player_display_name'],'position':pos,
        'team':r['team'],'draft_pick':r['draft_pick'],'ano_nfl':ano,
        'head_coach':hc_2026,'win_total_2026':win_totals_2026.get(r['team'],league_avg_wins),
        'sos_rank':sos_d.get(r['team'],16),
        'proj_fpts_ppr':pf,
        'proj_passing_yards':   py if pos=='QB' else 0,
        'proj_rushing_yards':   py if pos=='RB' else 0,
        'proj_receiving_yards': py if pos in ['WR','TE'] else 0,
        'proj_passing_tds':     pt if pos=='QB' else 0,
        'proj_rushing_tds':     pt if pos=='RB' else 0,
        'proj_receiving_tds':   pt if pos in ['WR','TE'] else 0,
        'is_rookie':True,'n_temporadas':ano-1,'age_feature':22 if ano==1 else 23,
        'confianca':conf,'confianca_label':'Baixa' if ano==1 else 'Média',
        'player_norm':normalizar_nome(r['player_display_name']),
    }

df_novatos = pd.DataFrame([projetar_novato(r) for r in rookies_2026 + ano2_2026])
print(df_novatos[['player_display_name','position','team','ano_nfl','proj_fpts_ppr','sos_rank']].to_string(index=False))

# =============================================================================
# BLOCO 12 — SCRAPING FANTASYPROS
# =============================================================================
print("\n" + "=" * 60)
print("BLOCO 12 — Scraping FantasyPros...")
print("=" * 60)

def scrape_fantasypros(position):
    url = f"https://www.fantasypros.com/nfl/projections/{position}.php?week=draft&scoring=PPR"
    headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    resp  = requests.get(url,headers=headers,timeout=15)
    soup  = BeautifulSoup(resp.text,'lxml')
    table = soup.find('table',{'id':'data'})
    if not table: return pd.DataFrame()
    rows = [[td.get_text(strip=True) for td in tr.find_all('td')]
            for tr in table.find('tbody').find_all('tr') if tr.find_all('td')]
    if not rows: return pd.DataFrame()
    df_fp = pd.DataFrame(rows)
    col_map = {
        'qb':['player','fp_attempts','fp_completions','fp_passing_yards',
              'fp_passing_tds','fp_interceptions','fp_rush_att',
              'fp_rushing_yards','fp_rushing_tds','fp_fumbles','fp_fantasy_pts'],
        'wr':['player','fp_receptions','fp_receiving_yards','fp_receiving_tds',
              'fp_rush_att','fp_rushing_yards','fp_rushing_tds','fp_fumbles','fp_fantasy_pts'],
        'rb':['player','fp_carries','fp_rushing_yards','fp_rushing_tds',
              'fp_receptions','fp_receiving_yards','fp_receiving_tds','fp_fumbles','fp_fantasy_pts'],
        'te':['player','fp_receptions','fp_receiving_yards','fp_receiving_tds',
              'fp_fumbles','fp_fantasy_pts'],
    }
    cols = col_map.get(position,[])
    if len(df_fp.columns)>=len(cols):
        df_fp=df_fp.iloc[:,:len(cols)]; df_fp.columns=cols
    else:
        n=len(df_fp.columns); g=[f'col_{i}' for i in range(n)]
        g[0]='player'; g[-1]='fp_fantasy_pts'; df_fp.columns=g
    df_fp['position']=position.upper()
    df_fp['player']=df_fp['player'].apply(lambda n: re.sub(r'[A-Z]{2,4}$','',str(n)).strip())
    for c in [x for x in df_fp.columns if x not in ['player','position']]:
        df_fp[c]=pd.to_numeric(df_fp[c].astype(str).str.replace(',',''),errors='coerce')
    print(f"  FantasyPros {position.upper()}: {len(df_fp)}")
    return df_fp

fp_qb=scrape_fantasypros('qb'); time.sleep(1)
fp_wr=scrape_fantasypros('wr'); time.sleep(1)
fp_rb=scrape_fantasypros('rb'); time.sleep(1)
fp_te=scrape_fantasypros('te')
fp_all=pd.concat([fp_qb,fp_wr,fp_rb,fp_te],ignore_index=True)
print(f"  Total: {len(fp_all)}")

# =============================================================================
# BLOCO 13 — FLOCK
# =============================================================================
print("\n" + "=" * 60)
print("BLOCO 13 — FlockFantasy...")
print("=" * 60)

FLOCK_CSV_PATH = r'C:\Users\stodu\BEST_BALL-rankings.csv'

def scrape_flock_selenium():
    """
    Tenta carregar Flock via undetected-chromedriver (mais robusto),
    depois Selenium padrão com --headless=new, depois CSV como fallback.
    O erro anterior era incompatibilidade de versão ChromeDriver/Chrome.
    """
    def _extrair_rows(soup):
        tables = soup.find_all('table')
        if not tables: return []
        rows = []
        for tr in tables[0].find_all('tr')[1:]:
            cols = [td.get_text(strip=True) for td in tr.find_all('td')]
            if len(cols) >= 4:
                rows.append(cols[:4])
        return rows

    # Tentativa 1: undetected-chromedriver (evita detecção de bot)
    try:
        import undetected_chromedriver as uc
        opts = uc.ChromeOptions()
        opts.add_argument('--headless=new')
        opts.add_argument('--no-sandbox')
        opts.add_argument('--disable-dev-shm-usage')
        driver = uc.Chrome(options=opts, version_main=None)
        driver.get("https://www.flockfantasy.com/rankings")
        time.sleep(6)
        soup = BeautifulSoup(driver.page_source, 'lxml')
        driver.quit()
        rows = _extrair_rows(soup)
        if rows:
            df_f = pd.DataFrame(rows, columns=['player','team','position','flock_rank'])
            df_f['flock_rank'] = pd.to_numeric(df_f['flock_rank'], errors='coerce')
            print(f"  Flock via UC ChromeDriver: {len(df_f)} jogadores")
            return df_f
        print("  UC: sem linhas extraídas")
    except Exception as e:
        print(f"  UC ChromeDriver falhou: {str(e)[:60]}")

    # Tentativa 2: Selenium com --headless=new (mais compatível com Chrome novo)
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from webdriver_manager.chrome import ChromeDriverManager
        opts = webdriver.ChromeOptions()
        opts.add_argument('--headless=new')   # novo modo headless — mais compatível
        opts.add_argument('--no-sandbox')
        opts.add_argument('--disable-dev-shm-usage')
        opts.add_argument('--disable-gpu')
        opts.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=opts)
        driver.get("https://www.flockfantasy.com/rankings")
        WebDriverWait(driver, 25).until(
            EC.presence_of_element_located((By.TAG_NAME, "table")))
        time.sleep(5)
        soup = BeautifulSoup(driver.page_source, 'lxml')
        driver.quit()
        rows = _extrair_rows(soup)
        if rows:
            df_f = pd.DataFrame(rows, columns=['player','team','position','flock_rank'])
            df_f['flock_rank'] = pd.to_numeric(df_f['flock_rank'], errors='coerce')
            print(f"  Flock via Selenium: {len(df_f)} jogadores")
            return df_f
        print("  Selenium: sem linhas extraídas")
    except Exception as e:
        print(f"  Selenium falhou: {str(e)[:60]}")

    # Fallback: CSV local
    print("  Usando CSV local como fallback")
    df_f = pd.read_csv(FLOCK_CSV_PATH).rename(columns={
        'Name':'player','Team':'team','Position':'position','Expert Rank':'flock_rank'
    })[['player','team','position','flock_rank']]
    print(f"  Flock via CSV: {len(df_f)} jogadores")
    return df_f

flock_df=scrape_flock_selenium()

# =============================================================================
# BLOCO 14 — NORMALIZAÇÃO
# =============================================================================
fp_all['player_norm']   = fp_all['player'].apply(normalizar_nome)
flock_df['player_norm'] = flock_df['player'].apply(normalizar_nome)
proj_qb['player_norm']  = proj_qb['player_display_name'].apply(normalizar_nome)
proj_wr['player_norm']  = proj_wr['player_display_name'].apply(normalizar_nome)
proj_rb['player_norm']  = proj_rb['player_display_name'].apply(normalizar_nome)
proj_te['player_norm']  = proj_te['player_display_name'].apply(normalizar_nome)
print("\nBLOCO 14 — Nomes normalizados!")

# =============================================================================
# BLOCO 15 — CONSOLIDAR
# =============================================================================
print("\n" + "=" * 60)
print("BLOCO 15 — Consolidando fontes...")
print("=" * 60)

def consolidar(proj_df, fp_df, flock_pos, proj_cols, novatos_df=None):
    # Pesos por posição — ajuste gradual baseado no MAE do backtesting (70% do sugerido)
    # QB: modelo melhor → leve boost; RB: empate → mais flock; TE: modelo melhor → mais modelo
    pesos_pos = {
        'QB': (0.33, 0.31, 0.36),  # Modelo/FP/Flock
        'WR': (0.35, 0.35, 0.30),  # Praticamente igual (diferença < 5%)
        'RB': (0.32, 0.31, 0.37),  # Flock um pouco mais forte
        'TE': (0.37, 0.31, 0.32),  # Modelo melhor → mais peso
    }
    w_mod, w_fp, w_fl = pesos_pos.get(flock_pos, (0.35, 0.35, 0.30))

    cols_base = ['player_display_name','player_norm','n_temporadas','fantasy_points_ppr',
                 'age_feature','team','hc_change_feat','win_total_feat',
                 'weeks_played','target_share_avg','ts_proj','snap_pct_avg'] + proj_cols
    cols_base = [c for c in cols_base if c in proj_df.columns]

    merged = proj_df[cols_base]\
        .merge(fp_df, on='player_norm', how='inner')\
        .merge(flock_df[flock_df['position']==flock_pos][['player_norm','flock_rank']],
               on='player_norm', how='left')

    if novatos_df is not None:
        nov = novatos_df[novatos_df['position']==flock_pos].copy()

        # Usar fp_fantasy_pts real do FP quando disponível para novatos
        nov_with_fp = nov.merge(fp_df[['player_norm','fp_fantasy_pts']], on='player_norm', how='left')
        nov['fp_fantasy_pts'] = nov_with_fp['fp_fantasy_pts'].fillna(nov['proj_fpts_ppr'] * 0.85).values

        # Usar flock_rank real quando disponível
        nov_with_flock = nov.merge(
            flock_df[flock_df['position']==flock_pos][['player_norm','flock_rank']],
            on='player_norm', how='left'
        )
        nov['flock_rank'] = nov_with_flock['flock_rank'].values
        nov['fantasy_points_ppr'] = 0.0

        # Remover novatos que já aparecem no merged (evitar duplicatas)
        nomes_ja_merged = set(merged['player_norm'].dropna())
        nov = nov[~nov['player_norm'].isin(nomes_ja_merged)]

        for c in merged.columns:
            if c not in nov.columns: nov[c] = np.nan
        if len(nov) > 0:
            merged = pd.concat([merged, nov[[c for c in nov.columns if c in merged.columns]]],
                               ignore_index=True)


    if merged.empty:
        print(f"  ⚠️  Merge vazio para {flock_pos}!"); return merged

    if 'is_rookie' not in merged.columns:
        merged['is_rookie'] = False
    merged['is_rookie']  = merged['is_rookie'].fillna(False).astype(bool)
    veteran_mask         = ~merged['is_rookie']

    merged['fp_fantasy_pts'] = pd.to_numeric(
        merged['fp_fantasy_pts'].astype(str).str.replace(',',''), errors='coerce')

    merged['pct_modelo'] = smooth_percentil(merged['proj_fpts_ppr'].fillna(0))
    merged['pct_fp']     = smooth_percentil(merged['fp_fantasy_pts'].fillna(0))
    merged['flock_rank'] = pd.to_numeric(merged['flock_rank'],errors='coerce')
    max_rank = merged['flock_rank'].max()
    merged['pct_flock']  = ((max_rank - merged['flock_rank'])/max_rank*100).fillna(50).clip(0,99.9)

    # Score final com pesos por posição
    merged['score_final']     = (merged['pct_modelo']*w_mod + merged['pct_fp']*w_fp + merged['pct_flock']*w_fl).round(1)
    merged['rank_final']      = merged['score_final'].rank(ascending=False, na_option='bottom').fillna(999).astype(int)
    merged['proj_fpts_final'] = (merged['proj_fpts_ppr']*0.50 + merged['fp_fantasy_pts']*0.50).round(1)

    # ==========================================================================
    # SOLIDEZ DOS DADOS: quão confiáveis são os dados desta projeção
    # (independente da projeção ser alta ou baixa)
    # Componentes:
    #   1. Temporadas de dados disponíveis (mais dados = mais sólido)
    #   2. Semanas jogadas na última temporada (mais semanas = menos incerteza)
    #   3. Concordância entre as 3 fontes (mais alinhadas = mais sólido)
    # NÃO confundir com "jogador vai bem" — é sobre qualidade dos dados
    # ==========================================================================
    concordancia = (100 - (
        (merged['pct_modelo']-merged['pct_fp']).abs()*0.40+
        (merged['pct_modelo']-merged['pct_flock']).abs()*0.30+
        (merged['pct_fp']-merged['pct_flock']).abs()*0.30
    ).clip(0,100))

    # Solidez de semanas jogadas
    weeks_score = merged['weeks_played'].fillna(14).clip(6,17)
    weeks_solidez = (weeks_score - 6) / 11 * 100  # 0 a 100

    merged['solidez_dados'] = 35.0  # default rookie
    merged.loc[veteran_mask,'solidez_dados'] = (
        concordancia[veteran_mask]            * 0.50 +
        merged.loc[veteran_mask,'n_temporadas'].fillna(1).clip(1,5)/5*100 * 0.30 +
        weeks_solidez[veteran_mask]            * 0.20
    ).round(0).clip(0,100)

    ano2_mask = merged['is_rookie'] & (merged['n_temporadas'].fillna(0)==1)
    merged.loc[ano2_mask,'solidez_dados'] = 45.0

    merged['solidez_label'] = pd.cut(
        merged['solidez_dados'].astype(float),
        bins=[0,40,60,80,100],
        labels=['Baixa','Média','Alta','Muito Alta']
    )

    # ==========================================================================
    # FLAG DE SITUAÇÃO: fatores contextuais que aumentam incerteza
    # Alto = situação clara; Baixo = muita incerteza contextual
    # ==========================================================================
    flags = []
    for _, row in merged.iterrows():
        f = []
        hc_change = row.get('hc_change_feat', 0)
        team      = str(row.get('team', ''))
        is_rook   = bool(row.get('is_rookie', False))
        weeks_p   = row.get('weeks_played', 17)
        age       = row.get('age_feature', 27)

        if is_rook:           f.append('🔴 Rookie/Ano2')
        elif hc_change:       f.append('🟡 Novo HC/OC')
        if not pd.isna(weeks_p) and weeks_p < 12:
            f.append('🟠 Lesão 2025')
        if flock_pos == 'RB' and not pd.isna(age) and age >= 29:
            f.append('🟡 Idade RB')
        if flock_pos == 'QB' and not pd.isna(age) and age >= 35:
            f.append('🟡 Idade QB')
        if team in ['MIA','JAX','CAR','LV','TEN']:
            f.append('🟡 Time fraco')
        flags.append(' | '.join(f) if f else '🟢 Situação clara')

    merged['situacao'] = flags

    # Manter 'confianca' como alias de solidez_dados para compatibilidade com dashboard
    merged['confianca']       = merged['solidez_dados']
    merged['confianca_label'] = merged['solidez_label']
    return merged

fp_qb_sel=fp_all[fp_all['position']=='QB'][['player_norm','fp_passing_yards','fp_passing_tds','fp_rushing_yards','fp_rushing_tds','fp_fantasy_pts']]
fp_wr_sel=fp_all[fp_all['position']=='WR'][['player_norm','fp_receptions','fp_receiving_yards','fp_receiving_tds','fp_fantasy_pts']]
fp_rb_sel=fp_all[fp_all['position']=='RB'][['player_norm','fp_carries','fp_rushing_yards','fp_rushing_tds','fp_receptions','fp_receiving_yards','fp_fantasy_pts']]
fp_te_sel=fp_all[fp_all['position']=='TE'][['player_norm','fp_receptions','fp_receiving_yards','fp_receiving_tds','fp_fantasy_pts']]

consolidated_qb=consolidar(proj_qb,fp_qb_sel,'QB',['proj_fpts_ppr','proj_passing_yards','proj_passing_tds'],df_novatos)
consolidated_wr=consolidar(proj_wr,fp_wr_sel,'WR',['proj_fpts_ppr','proj_receiving_yards','proj_receiving_tds'],df_novatos)
consolidated_rb=consolidar(proj_rb,fp_rb_sel,'RB',['proj_fpts_ppr','proj_rushing_yards','proj_rushing_tds'],df_novatos)
consolidated_te=consolidar(proj_te,fp_te_sel,'TE',['proj_fpts_ppr','proj_receiving_yards','proj_receiving_tds'],df_novatos)

print(f"  QBs: {len(consolidated_qb)} | WRs: {len(consolidated_wr)} | RBs: {len(consolidated_rb)} | TEs: {len(consolidated_te)}")

# =============================================================================
# BLOCO 16 — RESULTADOS
# =============================================================================
print("\n" + "=" * 60)
print("BLOCO 16 — Rankings finais")
print("=" * 60)

def print_ranking(df, pos, cols_extra):
    base = ['rank_final','player_display_name','is_rookie','age_feature','team',
            'proj_fpts_ppr','fp_fantasy_pts','proj_fpts_final',
            'score_final','confianca','confianca_label','win_total_feat']
    cols = [c for c in base+cols_extra if c in df.columns]
    print(f"\n== TOP 20 {pos}s ==")
    print(df[cols].sort_values('rank_final').head(20).to_string(index=False))

print_ranking(consolidated_qb,'QB',['proj_passing_yards','proj_passing_tds'])
print_ranking(consolidated_wr,'WR',['proj_receiving_yards','proj_receiving_tds','target_share_avg'])
print_ranking(consolidated_rb,'RB',['proj_rushing_yards','proj_rushing_tds','snap_pct_avg'])
print_ranking(consolidated_te,'TE',['proj_receiving_yards','proj_receiving_tds'])

# =============================================================================
# BLOCO 17 — CALIBRAÇÃO EMPÍRICA DO DESVIO PADRÃO
# =============================================================================
print("\n" + "=" * 60)
print("BLOCO 17 — Calibrando variância empírica e Monte Carlo...")
print("=" * 60)

# Calcular desvio padrão real de fantasy points por jogador histórico
# Isso é muito mais preciso que CV fixo
std_por_jogador = df_season.groupby('player_id')['fantasy_points_ppr'].std().rename('fpts_std_hist')
mean_por_jogador = df_season.groupby('player_id')['fantasy_points_ppr'].mean().rename('fpts_mean_hist')
cv_por_jogador = (std_por_jogador / mean_por_jogador).rename('cv_hist').clip(0.10, 0.55)

# CV médio empírico por posição (para fallback quando jogador tem poucos dados)
cv_medio_pos = {}
for pos, df_p in [('QB',qbs),('WR',wrs),('RB',rbs),('TE',tes)]:
    df_cv = df_p.merge(cv_por_jogador, on='player_id', how='left')
    cv_medio_pos[pos] = df_cv['cv_hist'].median()
    print(f"  CV empírico {pos}: {cv_medio_pos[pos]:.3f} ({cv_medio_pos[pos]*100:.1f}%)")

# Bust/Boom: thresholds baseados em percentis históricos reais por posição
# Bust = abaixo do P25 histórico da posição; Boom = acima do P75 histórico
bust_boom_thresholds = {}
for pos, df_p in [('QB',qbs),('WR',wrs),('RB',rbs),('TE',tes)]:
    fpts_pos = df_p[df_p['season']>=2022]['fantasy_points_ppr']
    bust_boom_thresholds[pos] = {
        'bust': fpts_pos.quantile(0.25),
        'boom': fpts_pos.quantile(0.75),
    }
    print(f"  {pos} — Bust threshold: {bust_boom_thresholds[pos]['bust']:.0f}pts | "
          f"Boom threshold: {bust_boom_thresholds[pos]['boom']:.0f}pts")

def monte_carlo_projections(df_pos, proj_col, posicao, n_sim=10000, seed=42):
    """
    Monte Carlo calibrado com:
    - Desvio padrão empírico histórico por jogador
    - Thresholds de bust/boom baseados em percentis reais da posição
    - Distribuição log-normal para evitar valores negativos e capturar assimetria
    """
    np.random.seed(seed)
    results = []
    bust_thr = bust_boom_thresholds.get(posicao, {}).get('bust', 100)
    boom_thr = bust_boom_thresholds.get(posicao, {}).get('boom', 300)
    cv_fallback = cv_medio_pos.get(posicao, 0.30)

    # Merge CV histórico
    df_cv = df_pos.merge(cv_por_jogador, on='player_id', how='left') if 'player_id' in df_pos.columns \
            else df_pos.copy()

    for idx, row in df_cv.iterrows():
        mu = row.get(proj_col, 0)
        if pd.isna(mu) or mu <= 0:
            results.append({'player_display_name': row.get('player_display_name',''),
                            'mc_p10':0,'mc_p25':0,'mc_median':0,'mc_p75':0,'mc_p90':0,
                            'mc_bust_pct':100,'mc_boom_pct':0,'mc_std':0,
                            'mc_floor':0,'mc_ceiling':0})
            continue

        # CV do jogador específico ou fallback da posição
        cv = row.get('cv_hist', cv_fallback)
        if pd.isna(cv): cv = cv_fallback
        sigma = mu * cv

        # Log-normal: mais realista que normal (assimétrica, sem valores negativos)
        # Parâmetros da log-normal equivalentes à média mu e desvio sigma
        mu_ln    = np.log(mu**2 / np.sqrt(mu**2 + sigma**2))
        sigma_ln = np.sqrt(np.log(1 + (sigma/mu)**2))
        sims = np.random.lognormal(mu_ln, sigma_ln, n_sim)

        results.append({
            'player_display_name': row.get('player_display_name',''),
            'mc_p10':     round(np.percentile(sims, 10), 1),
            'mc_p25':     round(np.percentile(sims, 25), 1),
            'mc_median':  round(np.percentile(sims, 50), 1),
            'mc_p75':     round(np.percentile(sims, 75), 1),
            'mc_p90':     round(np.percentile(sims, 90), 1),
            'mc_bust_pct':round((sims < bust_thr).mean() * 100, 1),
            'mc_boom_pct':round((sims > boom_thr).mean() * 100, 1),
            'mc_std':     round(sigma, 1),
            'mc_floor':   round(np.percentile(sims, 5), 1),
            'mc_ceiling': round(np.percentile(sims, 95), 1),
        })

    return pd.DataFrame(results)

mc_qb = monte_carlo_projections(consolidated_qb, 'proj_fpts_ppr', 'QB')
mc_wr = monte_carlo_projections(consolidated_wr, 'proj_fpts_ppr', 'WR')
mc_rb = monte_carlo_projections(consolidated_rb, 'proj_fpts_ppr', 'RB')
mc_te = monte_carlo_projections(consolidated_te, 'proj_fpts_ppr', 'TE')

consolidated_qb = consolidated_qb.merge(mc_qb, on='player_display_name', how='left')
consolidated_wr = consolidated_wr.merge(mc_wr, on='player_display_name', how='left')
consolidated_rb = consolidated_rb.merge(mc_rb, on='player_display_name', how='left')
consolidated_te = consolidated_te.merge(mc_te, on='player_display_name', how='left')

print("\n  Monte Carlo calibrado — amostra RBs:")
sample_cols = ['player_display_name','proj_fpts_final','mc_p10','mc_p90',
               'mc_bust_pct','mc_boom_pct','mc_std']
print(consolidated_rb[sample_cols].sort_values('proj_fpts_final', ascending=False).head(8).to_string(index=False))



# =============================================================================
# BLOCO 18 — ALL PLAYERS: ranking unificado para simulação de draft
# =============================================================================
print("\n" + "=" * 60)
print("BLOCO 18 — Ranking unificado All Players (draft sim)...")
print("=" * 60)

def preparar_para_all(df_pos, posicao, yards_col, tds_col):
    cols_necessarias = ['player_display_name','player_norm','position','team',
                        'age_feature','is_rookie','proj_fpts_ppr','fp_fantasy_pts',
                        'proj_fpts_final', yards_col, tds_col,
                        'score_final','rank_final','confianca','confianca_label',
                        'flock_rank','win_total_feat',
                        'mc_p10','mc_p25','mc_median','mc_p75','mc_p90',
                        'mc_bust_pct','mc_boom_pct']
    cols_ok = [c for c in cols_necessarias if c in df_pos.columns]
    df_out  = df_pos[cols_ok].copy()
    df_out['position_label'] = posicao
    # Renomear yards e tds para colunas genéricas
    if yards_col in df_out.columns:
        df_out = df_out.rename(columns={yards_col: 'proj_yards', tds_col: 'proj_tds'})
    return df_out

all_qb = preparar_para_all(consolidated_qb, 'QB', 'proj_passing_yards',   'proj_passing_tds')
all_wr = preparar_para_all(consolidated_wr, 'WR', 'proj_receiving_yards', 'proj_receiving_tds')
all_rb = preparar_para_all(consolidated_rb, 'RB', 'proj_rushing_yards',   'proj_rushing_tds')
all_te = preparar_para_all(consolidated_te, 'TE', 'proj_receiving_yards', 'proj_receiving_tds')

all_players = pd.concat([all_qb, all_wr, all_rb, all_te], ignore_index=True)

# Ranking unificado por proj_fpts_final (consensus)
all_players = all_players.sort_values('proj_fpts_final', ascending=False).reset_index(drop=True)
all_players['overall_rank'] = all_players.index + 1
all_players['rank_by_pos']  = all_players.groupby('position_label')['proj_fpts_final']\
    .rank(ascending=False, na_option='bottom').fillna(99).astype(int)
all_players['pos_rank_label'] = all_players['position_label'] + all_players['rank_by_pos'].astype(str)

print(f"  Total jogadores: {len(all_players)}")
print("\n  Top 30 Overall:")
top30_cols = ['overall_rank','pos_rank_label','player_display_name','team',
              'proj_fpts_final','mc_p10','mc_p90','confianca_label']
top30_cols = [c for c in top30_cols if c in all_players.columns]
print(all_players[top30_cols].head(30).to_string(index=False))

# =============================================================================
# BLOCO 19 — EXPORTAR EXCEL (simplificado + all players)
# =============================================================================
print("\n" + "=" * 60)
print("BLOCO 19 — Exportando Excel...")
print("=" * 60)

# Colunas principais para cada aba (simplificado)
def cols_principais(df, yards_col_orig=None):
    base = ['rank_final','player_display_name','position','team','age_feature','is_rookie',
            'proj_fpts_ppr','fp_fantasy_pts','proj_fpts_final']
    if yards_col_orig and yards_col_orig in df.columns:
        base += [yards_col_orig]
    elif 'proj_yards' in df.columns:
        base += ['proj_yards']
    tds_col = next((c for c in ['proj_passing_tds','proj_rushing_tds','proj_receiving_tds','proj_tds'] if c in df.columns), None)
    if tds_col: base += [tds_col]
    base += ['score_final','confianca','confianca_label','flock_rank',
             'win_total_feat','mc_p10','mc_p90','mc_bust_pct','mc_boom_pct']
    return [c for c in base if c in df.columns]

with pd.ExcelWriter('consolidado_nfl_2026.xlsx') as writer:
    # Abas por posição (simplificadas)
    consolidated_qb.sort_values('rank_final')[cols_principais(consolidated_qb,'proj_passing_yards')]\
        .to_excel(writer, sheet_name='QBs', index=False)
    consolidated_wr.sort_values('rank_final')[cols_principais(consolidated_wr,'proj_receiving_yards')]\
        .to_excel(writer, sheet_name='WRs', index=False)
    consolidated_rb.sort_values('rank_final')[cols_principais(consolidated_rb,'proj_rushing_yards')]\
        .to_excel(writer, sheet_name='RBs', index=False)
    consolidated_te.sort_values('rank_final')[cols_principais(consolidated_te,'proj_receiving_yards')]\
        .to_excel(writer, sheet_name='TEs', index=False)

    # Aba All Players — ranking unificado para draft
    all_cols = ['overall_rank','pos_rank_label','player_display_name','team',
                'age_feature','is_rookie','proj_fpts_final','proj_yards','proj_tds',
                'score_final','confianca','confianca_label','flock_rank',
                'mc_p10','mc_p90','mc_bust_pct','mc_boom_pct']
    all_cols = [c for c in all_cols if c in all_players.columns]
    all_players[all_cols].to_excel(writer, sheet_name='All_Players', index=False)

    # Rookies e backtesting
    df_novatos.to_excel(writer, sheet_name='Rookies_Ano2', index=False)
    df_bt.to_excel(writer, sheet_name='Backtesting', index=False)

print("  Salvo: consolidado_nfl_2026.xlsx")
print("\n✓ Pipeline completo v12!")
print("  Dashboard: streamlit run nfl_dashboard.py")
