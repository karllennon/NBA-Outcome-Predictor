"""
Data for the dashboard, in one shape for both modes:

    live:   today's games (build_slate: model, Kalshi, injury report)
    replay: a past day from the walk-forward test, i.e. the probability a model trained only on
            earlier games gave, Kalshi's price at tip-off, the injury report, and the final score

day_view(...) returns a list of game dicts with the keys used by app.py.
"""
import numpy as np
import pandas as pd
from matchups import FEATURES

TEAM_COLORS = {
    'ATL': '#E03A3E', 'BOS': '#007A33', 'BKN': '#2B2B2B', 'CHA': '#1D1160', 'CHI': '#CE1141',
    'CLE': '#860038', 'DAL': '#00538C', 'DEN': '#0E2240', 'DET': '#C8102E', 'GSW': '#1D428A',
    'HOU': '#CE1141', 'IND': '#002D62', 'LAC': '#C8102E', 'LAL': '#552583', 'MEM': '#5D76A9',
    'MIA': '#98002E', 'MIL': '#00471B', 'MIN': '#0C2340', 'NOP': '#0C2340', 'NYK': '#006BB6',
    'OKC': '#007AC1', 'ORL': '#0077C0', 'PHI': '#006BB6', 'PHX': '#1D1160', 'POR': '#E03A3E',
    'SAC': '#5A2D81', 'SAS': '#C4CED4', 'TOR': '#CE1141', 'UTA': '#002B5C', 'WAS': '#002B5C',
}
LIGHT_BADGES = {'SAS'}   # light team color: dark text

FEATURE_LABELS = {
    'ELO_DIFF': 'Elo rating', 'CORE_INJURY_DIFF': 'Injuries', 'PLUS_MINUS_DIFF': 'Recent point margin',
    'WIN_STREAK_DIFF': 'Wins in last 5', 'REST_DIFF': 'Rest days', 'B2B_DIFF': 'Back-to-back',
    'TRAVEL_DIFF': 'Travel distance', 'TZ_SHIFT_DIFF': 'Time zones crossed', 'EFG_DIFF': 'Shooting (eFG%)',
    'TOV_PCT_DIFF': 'Turnovers', 'ORB_PCT_DIFF': 'Offensive rebounding', 'FT_RATE_DIFF': 'Free-throw rate',
    'PACE_DIFF': 'Pace', 'DEF_RATING_DIFF': 'Defense (pts allowed /100)',
}


def team_abbrs(games_path='data/raw_nba_data.csv'):
    g = pd.read_csv(games_path, usecols=['TEAM_NAME', 'TEAM_ABBREVIATION', 'GAME_DATE']).sort_values('GAME_DATE')
    return dict(zip(g['TEAM_NAME'], g['TEAM_ABBREVIATION']))


def contributions(model, features):
    """
    Log-odds contribution of each feature to the home team's chance (logistic model):
    coefficient x standardized value. Positive favors the home team.
    """
    scaler, logit = model[0], model[-1]
    x = pd.Series(features)[FEATURES].astype(float).values
    z = (x - scaler.mean_) / scaler.scale_
    contrib = logit.coef_[0] * z
    out = pd.DataFrame({'feature': FEATURES, 'label': [FEATURE_LABELS.get(f, f) for f in FEATURES],
                        'value': x, 'contribution': contrib})
    return out.reindex(out['contribution'].abs().sort_values(ascending=False).index).reset_index(drop=True)


# ---------------------------------------------------------------- replay

def replay_dates(preds_path='data/test_predictions.csv', market_path='data/market_history.csv'):
    """Held-out game days with how many games and how many have a Kalshi price."""
    p = pd.read_csv(preds_path, parse_dates=['GAME_DATE'])
    m = pd.read_csv(market_path, dtype={'GAME_ID': str}).dropna(subset=['MARKET_HOME_PROB'])
    p['HAS_MARKET'] = p['GAME_ID'].astype(str).str.zfill(10).isin(set(m['GAME_ID']))
    return p.groupby('GAME_DATE').agg(games=('GAME_ID', 'size'), with_market=('HAS_MARKET', 'sum'))


def _report_out(day_reports, game_date, team):
    """[(player, reason)] listed Out on the team's last report before tip-off."""
    from injury_reports import last_report_before_tip
    if day_reports.empty:
        return None
    rows = last_report_before_tip(day_reports, game_date, team)
    if rows.empty:
        return None
    rows = rows[rows['STATUS'] == 'Out']
    return list(zip(rows['PLAYER'], rows['REASON'].fillna('')))


def day_view_replay(game_date):
    from injury_reports import load_archive_day
    game_date = pd.Timestamp(game_date)
    preds = pd.read_csv('data/test_predictions.csv', parse_dates=['GAME_DATE'])
    preds = preds[preds['GAME_DATE'] == game_date]
    games = pd.read_csv('data/raw_nba_data.csv', dtype={'GAME_ID': str})
    games['GAME_ID_INT'] = games['GAME_ID'].astype(int)
    feats = pd.read_csv('data/final_training_set.csv').set_index('GAME_ID')
    market = pd.read_csv('data/market_history.csv', dtype={'GAME_ID': str}).set_index('GAME_ID')
    reports = load_archive_day(game_date)
    spreads = _replay_spreads()
    market_spreads = _replay_market_spreads()

    out = []
    for p in preds.itertuples():
        g = games[games['GAME_ID_INT'] == p.GAME_ID]
        home = g[g['MATCHUP'].str.contains('vs.', regex=False)].iloc[0]
        away = g[g['MATCHUP'].str.contains('@', regex=False)].iloc[0]
        gid = str(home['GAME_ID']).zfill(10)
        mk = market.loc[gid] if gid in market.index else None
        market_prob = None if mk is None or pd.isna(mk['MARKET_HOME_PROB']) else float(mk['MARKET_HOME_PROB'])
        out.append({
            'GAME_ID': gid, 'GAME_DATE': game_date, 'MODE': 'replay',
            'HOME_TEAM': home['TEAM_NAME'], 'AWAY_TEAM': away['TEAM_NAME'],
            'HOME_ABBR': home['TEAM_ABBREVIATION'], 'AWAY_ABBR': away['TEAM_ABBREVIATION'],
            'HOME_ID': int(home['TEAM_ID']), 'AWAY_ID': int(away['TEAM_ID']),
            'TIP_TIME_ET': None if mk is None else pd.Timestamp(mk['TIP_TIME_ET']),
            'STATUS': 'final',
            'MODEL_HOME_PROB': float(p.MODEL_PROB), 'ELO_HOME_PROB': float(p.ELO_PROB),
            'MARKET_HOME_PROB': market_prob,
            'MARKET_YES_BID': None if mk is None else mk['HOME_YES_BID'],
            'MARKET_YES_ASK': None if mk is None else mk['HOME_YES_ASK'],
            'EVENT_TICKER': None if mk is None else mk['EVENT_TICKER'],
            'HOME_PTS': int(home['PTS']), 'AWAY_PTS': int(away['PTS']),
            'HOME_WIN': home['WL'] == 'W',
            'HOME_OUT': _report_out(reports, game_date, home['TEAM_NAME']),
            'AWAY_OUT': _report_out(reports, game_date, away['TEAM_NAME']),
            'FEATURES': feats.loc[p.GAME_ID, FEATURES].to_dict() if p.GAME_ID in feats.index else None,
        })
        out[-1].update(_spread_view(out[-1], spreads.get(int(p.GAME_ID)), market_spreads.get(gid)))
    return sorted(out, key=lambda r: (r['TIP_TIME_ET'] is None, r['TIP_TIME_ET'] or 0, r['HOME_TEAM']))


def _replay_spreads(path='data/test_spread_predictions.csv'):
    """Walk-forward spread predictions: GAME_ID -> (predicted home margin, fold sigma)."""
    try:
        s = pd.read_csv(path)
    except FileNotFoundError:
        return {}
    return {int(g): (m, sg) for g, m, sg in zip(s['GAME_ID'], s['MODEL_MARGIN'], s['SIGMA'])}


def _replay_market_spreads(path='data/market_spread_history.csv'):
    """Local Kalshi spread history: GAME_ID -> row dict (HOME_LINE, main-line market)."""
    try:
        s = pd.read_csv(path, dtype={'GAME_ID': str})
    except FileNotFoundError:
        return {}
    s = s.dropna(subset=['HOME_LINE']) if 'HOME_LINE' in s else s.iloc[0:0]
    return {r['GAME_ID']: r for r in s.to_dict('records')}


def _spread_view(g, model_spread, market_spread):
    """Spread fields shared by live and replay views."""
    import spread_model
    out = {'MODEL_HOME_MARGIN': None, 'SPREAD': None, 'SPREAD_TEXT': None, 'SPREAD_SIGMA': None,
           'MARKET_SPREAD': None, 'MARKET_SPREAD_TEXT': None, 'MARKET_SPREAD_INFO': market_spread,
           'TOSS_UP': False}
    if model_spread is not None:
        margin, sigma = model_spread
        out.update({'MODEL_HOME_MARGIN': float(margin), 'SPREAD': float(-margin), 'SPREAD_SIGMA': float(sigma),
                    'SPREAD_TEXT': spread_model.format_spread(margin, g['HOME_ABBR'], g['AWAY_ABBR']),
                    'TOSS_UP': (g['MODEL_HOME_PROB'] > 0.5) != (margin > 0)})
    if market_spread is not None and market_spread.get('HOME_LINE') == market_spread.get('HOME_LINE'):
        line = float(market_spread['HOME_LINE'])
        out.update({'MARKET_SPREAD': line,
                    'MARKET_SPREAD_TEXT': spread_model.format_spread(-line, g['HOME_ABBR'], g['AWAY_ABBR'])})
    return out


def team_form_replay(game_id, team_id):
    """Pre-game rolling form for one team in one past game."""
    try:
        f = pd.read_csv('data/team_game_features.csv')
    except FileNotFoundError:
        return None
    row = f[(f['GAME_ID'] == int(game_id)) & (f['TEAM_ID'] == team_id)]
    return None if row.empty else row.iloc[0]


# ---------------------------------------------------------------- live

def day_view_live(predictor, game_date=None):
    from daily_slate import build_slate
    slate = build_slate(game_date, predictor=predictor)
    if slate.empty:
        return []
    abbr = team_abbrs()
    ids = predictor.state.set_index('TEAM_NAME')['TEAM_ID'].to_dict()
    reasons = {}
    if predictor.report is not None:
        r = predictor.report
        reasons = dict(zip(r['PLAYER'], r['REASON'].fillna('')))
    out = []
    for r in slate.to_dict('records'):
        def listed(names):
            names = [n for n in str(names or '').split('; ') if n]
            return [(n, reasons.get(n, '')) for n in names]
        out.append({**r, 'MODE': 'live', 'GAME_DATE': pd.Timestamp(r['GAME_DATE']),
                    'HOME_ABBR': abbr.get(r['HOME_TEAM'], r['HOME_TEAM'][:3].upper()),
                    'AWAY_ABBR': abbr.get(r['AWAY_TEAM'], r['AWAY_TEAM'][:3].upper()),
                    'HOME_ID': ids.get(r['HOME_TEAM']), 'AWAY_ID': ids.get(r['AWAY_TEAM']),
                    'HOME_OUT': listed(r['HOME_OUT']), 'AWAY_OUT': listed(r['AWAY_OUT']),
                    'HOME_PTS': None, 'AWAY_PTS': None, 'HOME_WIN': None})
    return out


def team_form_live(state, team_name):
    row = state[state['TEAM_NAME'] == team_name]
    if row.empty:
        return None
    row = row.iloc[0].copy()
    row['PRE_GAME_ELO'] = row.get('CURRENT_ELO')
    return row


# ---------------------------------------------------------------- market trend

def market_trend_live(event_ticker, path='data/market_snapshots.csv'):
    """Logged Kalshi home-win midpoints for one game (time in ET)."""
    try:
        s = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=['TIME_ET', 'PROB'])
    s = s[s['EVENT_TICKER'] == event_ticker].dropna(subset=['MARKET_HOME_PROB'])
    t = pd.to_datetime(s['SNAPSHOT_TIME_UTC']).dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    return pd.DataFrame({'TIME_ET': t.dt.tz_localize(None).values, 'PROB': s['MARKET_HOME_PROB'].values})


def market_trend_replay(event_ticker, home_abbr, tip_time_et, hours=30):
    """Hourly Kalshi midpoints for the home team's market in the hours before tip-off."""
    import market_odds as mo
    if not event_ticker or tip_time_et is None:
        return pd.DataFrame(columns=['TIME_ET', 'PROB'])
    tip = pd.Timestamp(tip_time_et).tz_localize('America/New_York')
    end = int(tip.timestamp())
    params = {'start_ts': end - hours * 3600, 'end_ts': end, 'period_interval': 60}
    ticker = f'{event_ticker}-{home_abbr}'
    data = (mo._get(f'/historical/markets/{ticker}/candlesticks', params)
            or mo._get(f'/series/{mo.SERIES}/markets/{ticker}/candlesticks', params) or {})
    rows = []
    for c in data.get('candlesticks', []):
        bid = mo._dollars((c.get('yes_bid') or {}).get('close'))
        ask = mo._dollars((c.get('yes_ask') or {}).get('close'))
        mid = mo.midpoint(bid, ask)
        if not np.isnan(mid):
            t = pd.Timestamp(c['end_period_ts'], unit='s', tz='UTC').tz_convert('America/New_York')
            rows.append({'TIME_ET': t.tz_localize(None), 'PROB': mid})
    return pd.DataFrame(rows, columns=['TIME_ET', 'PROB'])


# ---------------------------------------------------------------- small lookups

def team_record(games, team_name, game_date):
    """Regular-season W-L for the team's season, from games before `game_date`."""
    d = pd.Timestamp(game_date)
    season_id = 20000 + (d.year if d.month >= 8 else d.year - 1)
    g = games[(games['TEAM_NAME'] == team_name) & (games['SEASON_ID'] == season_id) &
              (pd.to_datetime(games['GAME_DATE']) < d)]
    return int((g['WL'] == 'W').sum()), int((g['WL'] == 'L').sum())


def arena_name(home_abbr, path='data/arenas.csv'):
    a = pd.read_csv(path).set_index('TEAM_ABBREVIATION')
    return a.at[home_abbr, 'ARENA'] if home_abbr in a.index else ''


def upcoming_live(days=3):
    """Regular-season games over the next few days from the NBA scoreboard."""
    from schedule import games_on
    today = pd.Timestamp.now(tz='America/New_York').normalize().tz_localize(None)
    rows = []
    for k in range(1, days + 1):
        try:
            g = games_on(today + pd.Timedelta(days=k))
        except Exception:
            continue
        g = g[g['GAME_ID'].str[:3].isin(['002', '004'])]
        rows += g.to_dict('records')
    return rows
