"""
Append-only log of live predictions made before tip-off, and their track record once results
are in. Rows are only ever appended; earlier rows are never rewritten.

    data/prediction_log.csv
        LOGGED_AT_UTC, GAME_ID, GAME_DATE, TIP_TIME_ET, HOME_TEAM, AWAY_TEAM,
        MODEL_HOME_PROB, MARKET_HOME_PROB, MARKET_YES_BID, MARKET_YES_ASK,
        MODEL_VERSION, HOME_OUT, AWAY_OUT, INPUTS (JSON of the feature row),
        MODEL_HOME_MARGIN, SPREAD (home line, betting convention), SPREAD_SIGMA,
        MARKET_SPREAD (market home line)

When LOG_COLUMNS changes, new rows start a new segment file (prediction_log.<timestamp>.csv)
instead of rewriting the old file to add columns; load() reads all segments together. The log
records Kalshi prices, so it is git-ignored and stays local.
"""
import glob
import hashlib
import json
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd

LOG_PATH = 'data/prediction_log.csv'
LOG_COLUMNS = ['LOGGED_AT_UTC', 'GAME_ID', 'GAME_DATE', 'TIP_TIME_ET', 'HOME_TEAM', 'AWAY_TEAM',
               'MODEL_HOME_PROB', 'MARKET_HOME_PROB', 'MARKET_YES_BID', 'MARKET_YES_ASK',
               'MODEL_VERSION', 'HOME_OUT', 'AWAY_OUT', 'INPUTS',
               'MODEL_HOME_MARGIN', 'SPREAD', 'SPREAD_SIGMA', 'MARKET_SPREAD']


def model_version(path='models/nba_model.joblib'):
    """Short hash of the saved model file, so log rows say which model made them."""
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:10]


def segments(path=LOG_PATH):
    """The log file and any later segments, oldest first."""
    base, ext = os.path.splitext(path)
    later = sorted(glob.glob(f'{base}.*{ext}'))
    return ([path] if os.path.exists(path) else []) + later


def _header(path):
    with open(path, encoding='utf-8') as f:
        return f.readline().strip().split(',')


def append(rows, path=LOG_PATH, columns=None):
    """
    Append prediction rows (dicts with LOG_COLUMNS keys; LOGGED_AT_UTC is filled in here).
    Missing keys are written as blanks. If the newest segment was written with different
    columns, the rows go to a new segment instead, so existing rows are never modified.
    """
    if not rows:
        return 0
    columns = columns or LOG_COLUMNS
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    df = pd.DataFrame([{**r, 'LOGGED_AT_UTC': now} for r in rows]).reindex(columns=columns)
    segs = segments(path)
    target = segs[-1] if segs else path
    if os.path.exists(target) and _header(target) != list(columns):
        base, ext = os.path.splitext(path)
        target = f"{base}.{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}{ext}"
    df.to_csv(target, mode='a', header=not os.path.exists(target), index=False)
    return len(df)


def inputs_json(features):
    """Feature row (one-row DataFrame) as compact JSON."""
    row = features.iloc[0].to_dict()
    return json.dumps({k: (None if pd.isna(v) else round(float(v), 4)) for k, v in row.items()})


def load(path=LOG_PATH):
    """All segments of the log as one DataFrame (columns are the union across segments)."""
    frames = [pd.read_csv(f, dtype={'GAME_ID': str}) for f in segments(path)]
    if not frames:
        return pd.DataFrame(columns=LOG_COLUMNS)
    df = pd.concat(frames, ignore_index=True, sort=False)
    for c in ['LOGGED_AT_UTC', 'TIP_TIME_ET']:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df


def track_record(log=None, games_path='data/raw_nba_data.csv'):
    """
    The last logged pre-tip-off prediction for each game, joined to the final result.
    Returns one row per finished game: GAME_ID, GAME_DATE, HOME_TEAM, AWAY_TEAM,
    MODEL_HOME_PROB, MARKET_HOME_PROB, HOME_WIN, ACTUAL_MARGIN, and the logged spread columns.
    """
    log = load() if log is None else log
    if log.empty:
        return log.assign(HOME_WIN=pd.Series(dtype=float))
    last = log.sort_values('LOGGED_AT_UTC').groupby('GAME_ID').tail(1)
    games = pd.read_csv(games_path, dtype={'GAME_ID': str})
    games['GAME_ID'] = games['GAME_ID'].str.zfill(10)
    home = games[games['MATCHUP'].str.contains('vs.', regex=False)][['GAME_ID', 'WL', 'PLUS_MINUS']]
    out = last.merge(home, on='GAME_ID', how='inner')
    out['HOME_WIN'] = (out['WL'] == 'W').astype(int)
    out['ACTUAL_MARGIN'] = out['PLUS_MINUS']
    return out.drop(columns=['WL', 'PLUS_MINUS']).sort_values('GAME_DATE').reset_index(drop=True)


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


def spread_accuracy(record):
    """
    Logged spread vs results: MAE of the predicted home margin, and, where a market line was
    logged, how often the actual margin landed on the side of the line the model predicted
    (pushes on whole-number lines are left out).
    """
    if 'MODEL_HOME_MARGIN' not in record:
        return None
    r = record.dropna(subset=['MODEL_HOME_MARGIN', 'ACTUAL_MARGIN'])
    if r.empty:
        return None
    out = {'games': len(r), 'mae': float(np.mean(np.abs(r['MODEL_HOME_MARGIN'] - r['ACTUAL_MARGIN'])))}
    if 'MARKET_SPREAD' in r:
        m = r.dropna(subset=['MARKET_SPREAD'])
        threshold = -m['MARKET_SPREAD']          # home must win by more than this to cover
        m = m[m['ACTUAL_MARGIN'] != threshold]   # pushes
        if len(m):
            model_side = m['MODEL_HOME_MARGIN'] > threshold
            actual_side = m['ACTUAL_MARGIN'] > threshold
            out.update({'vs_line_games': len(m), 'right_side_of_line': float((model_side == actual_side).mean())})
    return out
