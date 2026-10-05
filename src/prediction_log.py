"""
Append-only log of live predictions made before tip-off, and their track record once results
are in. Rows are only ever appended; earlier rows are never rewritten.

    data/prediction_log.csv
        LOGGED_AT_UTC, GAME_ID, GAME_DATE, TIP_TIME_ET, HOME_TEAM, AWAY_TEAM,
        MODEL_HOME_PROB, MARKET_HOME_PROB, MARKET_YES_BID, MARKET_YES_ASK,
        MODEL_VERSION, HOME_OUT, AWAY_OUT, INPUTS (JSON of the feature row)
"""
import hashlib
import json
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd

LOG_PATH = 'data/prediction_log.csv'
LOG_COLUMNS = ['LOGGED_AT_UTC', 'GAME_ID', 'GAME_DATE', 'TIP_TIME_ET', 'HOME_TEAM', 'AWAY_TEAM',
               'MODEL_HOME_PROB', 'MARKET_HOME_PROB', 'MARKET_YES_BID', 'MARKET_YES_ASK',
               'MODEL_VERSION', 'HOME_OUT', 'AWAY_OUT', 'INPUTS']


def model_version(path='models/nba_model.joblib'):
    """Short hash of the saved model file, so log rows say which model made them."""
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:10]


def append(rows, path=LOG_PATH):
    """Append prediction rows (dicts with LOG_COLUMNS keys, LOGGED_AT_UTC filled in here)."""
    if not rows:
        return 0
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    df = pd.DataFrame([{**r, 'LOGGED_AT_UTC': now} for r in rows])[LOG_COLUMNS]
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
    return len(df)


def inputs_json(features):
    """Feature row (one-row DataFrame) as compact JSON."""
    row = features.iloc[0].to_dict()
    return json.dumps({k: (None if pd.isna(v) else round(float(v), 4)) for k, v in row.items()})


def load(path=LOG_PATH):
    if not os.path.exists(path):
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(path, dtype={'GAME_ID': str}, parse_dates=['LOGGED_AT_UTC', 'TIP_TIME_ET'])


def track_record(log=None, games_path='data/raw_nba_data.csv'):
    """
    The last logged pre-tip-off prediction for each game, joined to the final result.
    Returns one row per finished game: GAME_ID, GAME_DATE, HOME_TEAM, AWAY_TEAM,
    MODEL_HOME_PROB, MARKET_HOME_PROB, HOME_WIN.
    """
    log = load() if log is None else log
    if log.empty:
        return log.assign(HOME_WIN=pd.Series(dtype=float))
    last = log.sort_values('LOGGED_AT_UTC').groupby('GAME_ID').tail(1)
    games = pd.read_csv(games_path, dtype={'GAME_ID': str})
    games['GAME_ID'] = games['GAME_ID'].str.zfill(10)
    home = games[games['MATCHUP'].str.contains('vs.', regex=False)][['GAME_ID', 'WL']]
    out = last.merge(home, on='GAME_ID', how='inner')
    out['HOME_WIN'] = (out['WL'] == 'W').astype(int)
    return out.drop(columns=['WL']).sort_values('GAME_DATE').reset_index(drop=True)


def running_metrics(record, prob_col):
    """Cumulative accuracy, log loss and Brier score in game order."""
    r = record.dropna(subset=[prob_col])
    p = r[prob_col].clip(1e-6, 1 - 1e-6).values
    y = r['HOME_WIN'].values
    n = np.arange(1, len(r) + 1)
    return pd.DataFrame({
        'GAME_DATE': r['GAME_DATE'].values,
        'games': n,
        'accuracy': np.cumsum((p > 0.5) == y) / n,
        'log_loss': np.cumsum(-(y * np.log(p) + (1 - y) * np.log(1 - p))) / n,
        'brier': np.cumsum((p - y) ** 2) / n,
    })
