"""
Opponent-adjusted team power ratings from game margins (feature RATING_DIFF, added in 2026-27).

Before each game day, every team's rating is fitted jointly from all earlier games:

    home margin = home court + rating(home) - rating(away) + noise

by weighted ridge regression, with each game's weight halving every `half_life` days (so the
offseason gap shrinks last season's games a lot) and a ridge penalty `lam` (in game-weight units)
pulling ratings toward zero. Unlike Elo, which updates one game at a time, every refit adjusts all
teams for the strength of every opponent they played. Ratings for a date use only games before it.

Exponential decay makes the fit incremental: the weighted normal equations for the next date are
the current ones times 0.5 ** (days / half_life), plus the new games.
"""
import numpy as np
import pandas as pd

from elo import is_neutral_site

HALF_LIVES = (15, 30, 60, 120)     # days
LAMBDAS = (1.0, 5.0, 20.0)         # ridge penalty, in units of full-weight games
BURN_IN_DAYS = 60                  # tuning ignores the first days of data (too few games to rate)
# Shipped settings: every walk-forward fold chose these on its own training games (RESULTS.md)
PARAMS = {'half_life': 60, 'lam': 5.0}


def pair_games(team_games):
    """
    One row per game: GAME_ID, GAME_DATE, HOME_ID, AWAY_ID, MARGIN (home), NEUTRAL, and when the
    box-score totals are present, MARGIN_P100 (home margin per 100 possessions; experiment only).
    """
    g = team_games.copy()
    g['GAME_DATE'] = pd.to_datetime(g['GAME_DATE'])
    box = all(c in g for c in ('FGA', 'FTA', 'OREB', 'TOV'))
    if box:
        g['POSS'] = g['FGA'] + 0.44 * g['FTA'] - g['OREB'] + g['TOV']
    extra = ['POSS'] if box else []
    home = g[g['MATCHUP'].str.contains('vs.', regex=False)][['GAME_ID', 'GAME_DATE', 'TEAM_ID', 'PLUS_MINUS'] + extra]
    away = g[g['MATCHUP'].str.contains('@', regex=False)][['GAME_ID', 'TEAM_ID'] + extra]
    out = home.merge(away, on='GAME_ID', suffixes=('_HOME', '_AWAY')).rename(columns={
        'TEAM_ID_HOME': 'HOME_ID', 'TEAM_ID_AWAY': 'AWAY_ID', 'PLUS_MINUS': 'MARGIN'})
    out = out.dropna(subset=['MARGIN']).sort_values(['GAME_DATE', 'GAME_ID']).reset_index(drop=True)
    out['NEUTRAL'] = is_neutral_site(out['GAME_DATE'])
    cols = ['GAME_ID', 'GAME_DATE', 'HOME_ID', 'AWAY_ID', 'MARGIN', 'NEUTRAL']
    if box:
        out['MARGIN_P100'] = out['MARGIN'] / ((out['POSS_HOME'] + out['POSS_AWAY']) / 2) * 100
        cols.append('MARGIN_P100')
    return out[cols]


def pregame_ratings(games, half_life, lam, target='MARGIN', cap=None):
    """
    For every game, the ratings fitted on games before its date. Returns a DataFrame indexed by
    GAME_ID: RATING_HOME, RATING_AWAY, HCA (fitted home-court points) and PRED_MARGIN.
    target / cap (experiments): fit on another margin column, or on margins clipped to +-cap.
    """
    teams = np.sort(pd.unique(games[['HOME_ID', 'AWAY_ID']].values.ravel()))
    idx = {t: i for i, t in enumerate(teams)}
    k = len(teams) + 1                              # team ratings + home court
    A, b = np.zeros((k, k)), np.zeros(k)
    penalty = np.diag([lam] * len(teams) + [0.0])   # home court is not shrunk
    rows, last = [], None
    for day, g in games.groupby('GAME_DATE', sort=True):
        if last is not None:
            decay = 0.5 ** ((day - last).days / half_life)
            A *= decay
            b *= decay
        theta = np.linalg.solve(A + penalty + 1e-9 * np.eye(k), b)
        h, a = g['HOME_ID'].map(idx).values, g['AWAY_ID'].map(idx).values
        hca = np.where(g['NEUTRAL'].values, 0.0, theta[-1])
        rows.append(pd.DataFrame({'GAME_ID': g['GAME_ID'].values, 'RATING_HOME': theta[h],
                                  'RATING_AWAY': theta[a], 'HCA': hca,
                                  'PRED_MARGIN': hca + theta[h] - theta[a]}))
        # add today's games after rating them
        y = g[target].values if cap is None else np.clip(g[target].values, -cap, cap)
        for hi, ai, m, neutral in zip(h, a, y, g['NEUTRAL'].values):
            x = np.zeros(k)
            x[hi], x[ai], x[-1] = 1.0, -1.0, 0.0 if neutral else 1.0
            A += np.outer(x, x)
            b += x * m
        last = day
    return pd.concat(rows, ignore_index=True).set_index('GAME_ID')


def ratings_as_of(games, as_of, half_life, lam):
    """
    ({TEAM_ID: rating}, home court) fitted on games before `as_of`'s date: the same fit
    pregame_ratings uses for games on that date, computed directly (for live predictions).
    """
    as_of = pd.Timestamp(as_of).normalize()
    teams = np.sort(pd.unique(games[['HOME_ID', 'AWAY_ID']].values.ravel()))
    idx = {t: i for i, t in enumerate(teams)}
    k = len(teams) + 1
    past = games[games['GAME_DATE'] < as_of]
    X = np.zeros((len(past), k))
    rows = np.arange(len(past))
    X[rows, past['HOME_ID'].map(idx).values] = 1.0
    X[rows, past['AWAY_ID'].map(idx).values] = -1.0
    X[:, -1] = np.where(past['NEUTRAL'].values, 0.0, 1.0)
    w = 0.5 ** ((as_of - past['GAME_DATE']).dt.days.values / half_life)
    A = X.T @ (w[:, None] * X)
    b = X.T @ (w * past['MARGIN'].values)
    penalty = np.diag([lam] * len(teams) + [0.0])
    theta = np.linalg.solve(A + penalty + 1e-9 * np.eye(k), b)
    return dict(zip(teams, theta[:-1])), float(theta[-1])


def rating_tables(games, half_lives=HALF_LIVES, lambdas=LAMBDAS):
    """{(half_life, lam): pregame_ratings} for the whole grid (each uses only earlier games)."""
    return {(h, l): pregame_ratings(games, h, l) for h in half_lives for l in lambdas}


def tune(tables, games, before, burn_in_days=BURN_IN_DAYS, target='MARGIN'):
    """
    The table key (e.g. (half_life, lam)) whose pre-game prediction had the lowest squared error
    against `target` on games before `before` (skipping the first burn_in_days of data). Uses no
    games from `before` on.
    """
    start = games['GAME_DATE'].min() + pd.Timedelta(days=burn_in_days)
    train = games[(games['GAME_DATE'] >= start) & (games['GAME_DATE'] < pd.Timestamp(before))].set_index('GAME_ID')
    scores = {key: float(np.mean((t.loc[train.index, 'PRED_MARGIN'] - train[target]) ** 2))
              for key, t in tables.items()}
    return min(scores, key=scores.get), scores
