"""
Paper-trade report for the 2026-27 plan (docs/paper_trading_plan.md).

Applies the plan's frozen rules to the local prediction log: for each game, the last prediction
logged at least ENTRY_MINUTES before tip-off is the entry (model probability and the Kalshi price
logged with it). Strategies S1-S6 are settled against results and priced three ways:

    Kalshi single      a $1-payout contract at the logged ask plus Kalshi's taker fee
                       (decisions.fee, rounded up per one-contract order)
    PrizePicks single  payout multiplier 1 / (ask + 0.07 x ask x (1 - ask) + 2c)
    PrizePicks lineup  payout multiplier 1 / (Q + 0.07 x Q x (1 - Q) + 2c), Q = product of the
                       legs' asks: fees charged once on the combined price

The PrizePicks formulas were reverse-engineered from published and in-app multipliers in
October 2026 and are estimates. Closing line value (CLV) compares the entry midpoint with the
tip-off midpoint from TIP_PRICES_PATH when it exists, else with the last pre-tip-off logged price.

    python src/paper_report.py                     # writes data/paper_report.md (git-ignored)
    python src/paper_report.py --since 2026-11-01

Paper trading only: this module places no orders and contains no trading code. The report holds
Kalshi prices and figures derived from them, so it stays local. Hypothetical analysis, not
betting advice.
"""
import argparse
import os
import numpy as np
import pandas as pd

import decisions as D
import prediction_log as pl
from market_odds import TIP_PRICES_PATH   # written by: python src/market_odds.py --tip-prices

SEASON_START = '2026-10-20'
REGULAR_SEASON = '002'               # GAME_ID prefix (after zero padding) of regular-season games
ENTRY_MINUTES = 30
PRIZEPICKS_FEE = 0.02
LINEUP_SIZES = (2, 3, 4, 6)
START_BANKROLL = 100.0
REPORT_PATH = 'data/paper_report.md'
ET = 'America/New_York'

STRATEGIES = {
    'S1': 'Model picks the market underdog',
    'S2': 'Big spread lean (edge >= 10)',
    'S3': 'Market favorite, every game',
    'S3a': 'Heavy market favorite (90c+)',
    'S4': 'Model pick, every game',
    'S5/S6 legs': "Model's pick, model above the market",
}
LINEUP_RULES = {'S5': ('S5/S6 legs', 3), 'S6': ('S5/S6 legs', 6)}
LABELS = {**STRATEGIES, 'S5': '3-pick lineup, model above the market',
          'S6': '6-pick lineup, model above the market'}


# ---------------------------------------------------------------- pricing

def kalshi_fee_rate(p):
    """Kalshi taker fee per contract at price p, unrounded (as built into PrizePicks multipliers)."""
    return D.KALSHI_TAKER_RATE * p * (1 - p)


def prizepicks_multiplier(price):
    """Payout per $1 staked on a PrizePicks pick or lineup whose Kalshi price (or product) is `price`."""
    price = np.asarray(price, dtype=float)
    return 1 / (price + kalshi_fee_rate(price) + PRIZEPICKS_FEE)


def kalshi_pnl(fill, won):
    """Profit of one $1-payout contract bought at `fill` plus fee (decisions.settle)."""
    return D.settle('X', fill, bool(won))


# ---------------------------------------------------------------- log -> entries

def _logged_et(log):
    return pd.to_datetime(log['LOGGED_AT_UTC']).dt.tz_localize('UTC').dt.tz_convert(ET).dt.tz_localize(None)


def entries(log, since=SEASON_START, minutes=ENTRY_MINUTES):
    """
    One entry row per regular-season game: the last row logged at least `minutes` before tip-off.
    Also adds CLOSE_HOME_MID, the midpoint of the last row logged before tip-off (for CLV).
    Returns (entries, coverage dict).
    """
    log = log.copy()
    log['GAME_ID'] = log['GAME_ID'].astype(str).str.zfill(10)
    log = log[log['GAME_ID'].str.startswith(REGULAR_SEASON)]
    log['GAME_DATE'] = pd.to_datetime(log['GAME_DATE'])
    log = log[log['GAME_DATE'] >= pd.Timestamp(since)]
    log['TIP_TIME_ET'] = pd.to_datetime(log['TIP_TIME_ET'], errors='coerce')
    log['LOGGED_ET'] = _logged_et(log)
    log['HOME_MID'] = (log['MARKET_YES_BID'] + log['MARKET_YES_ASK']) / 2
    games = log['GAME_ID'].nunique()
    known = log[log['TIP_TIME_ET'].notna()]
    pre = known[known['LOGGED_ET'] < known['TIP_TIME_ET']].sort_values('LOGGED_ET')
    eligible = pre[pre['LOGGED_ET'] <= pre['TIP_TIME_ET'] - pd.Timedelta(minutes=minutes)]
    entry = eligible.groupby('GAME_ID').tail(1).set_index('GAME_ID')
    close = pre.groupby('GAME_ID').tail(1).set_index('GAME_ID')
    entry['CLOSE_HOME_MID'] = close['HOME_MID'].reindex(entry.index)
    entry['CLOSE_LOGGED_ET'] = close['LOGGED_ET'].reindex(entry.index)
    entry = entry.reset_index()
    coverage = {'games logged': games, 'with an entry (logged 30+ min before tip)': len(entry),
                'no tip-off time': log.loc[log['TIP_TIME_ET'].isna(), 'GAME_ID'].nunique(),
                'logged only within 30 min of tip': games - len(entry)
                - log.loc[log['TIP_TIME_ET'].isna(), 'GAME_ID'].nunique(),
                'model versions': log['MODEL_VERSION'].nunique()}
    return entry, coverage


def results(games_path='data/raw_nba_data.csv'):
    """Final home margin per GAME_ID (zero-padded)."""
    g = pd.read_csv(games_path, dtype={'GAME_ID': str})
    g['GAME_ID'] = g['GAME_ID'].str.zfill(10)
    home = g[g['MATCHUP'].str.contains('vs.', regex=False)]
    return home.set_index('GAME_ID')['PLUS_MINUS']


def tip_prices(path=TIP_PRICES_PATH):
    if not os.path.exists(path):
        return None
    t = pd.read_csv(path, dtype={'GAME_ID': str})
    t['GAME_ID'] = t['GAME_ID'].str.zfill(10)
    return t.set_index('GAME_ID')['TIP_HOME_MID']


def _is_true(x):
    return x in (True, 'True', 'true', 1, '1')


MODEL_COLUMNS = ['PICK_TEAM', 'PICK_SIDE', 'PICK_P', 'PICK_MID', 'PICK_FILL', 'PICK_MARKET_UNDERDOG',
                 'SPREAD_SIDE', 'SPREAD_P', 'SPREAD_MID', 'SPREAD_FILL', 'SPREAD_EDGE_PTS']


def model_view(entry, model='live'):
    """
    The entry rows as seen by one model. 'frozen' swaps in the FROZEN_* columns (probability,
    pick, spread decision) and drops games logged before the freeze; prices are shared.
    """
    if model == 'live':
        return entry
    if 'FROZEN_HOME_PROB' not in entry:
        return entry.iloc[0:0]
    e = entry[entry['FROZEN_HOME_PROB'].notna()].copy()
    e['MODEL_HOME_PROB'] = e['FROZEN_HOME_PROB']
    e['MODEL_VERSION'] = e['FROZEN_MODEL_VERSION']
    for c in MODEL_COLUMNS:
        e[c] = e[f'FROZEN_{c}'] if f'FROZEN_{c}' in e else np.nan
    return e


def legs(entry, margins, tip_mid=None):
    """
    Every strategy leg as one row: strategy, GAME_ID, GAME_DATE, TIP_TIME_ET, team, fill, mid, p,
    won (None while the game is unplayed), clv (cents, moneyline legs only). `entry` is one
    model's view (model_view).
    """
    rows = []
    for r in entry.to_dict('records'):
        gid = r['GAME_ID']
        margin = margins.get(gid)
        played = margin is not None and margin == margin
        base = {'GAME_ID': gid, 'GAME_DATE': r['GAME_DATE'], 'TIP_TIME_ET': r['TIP_TIME_ET'],
                'MODEL_VERSION': r.get('MODEL_VERSION')}
        close = r.get('CLOSE_HOME_MID')
        if tip_mid is not None and gid in tip_mid.index:
            close = tip_mid[gid]

        def ml_leg(strategy, home_side, fill, mid, p, team):
            won = None if not played else bool((margin > 0) == home_side)
            entry_mid = r['HOME_MID'] if home_side else 1 - r['HOME_MID']
            close_mid = None if close is None or close != close else (close if home_side else 1 - close)
            clv = None if close_mid is None or entry_mid != entry_mid else (close_mid - entry_mid) * 100
            rows.append({**base, 'strategy': strategy, 'team': team, 'fill': fill, 'mid': mid, 'p': p,
                         'won': won, 'clv': clv})

        bid, ask = r.get('MARKET_YES_BID'), r.get('MARKET_YES_ASK')
        if bid == bid and ask == ask and bid is not None and ask is not None:
            home_mid = (bid + ask) / 2
            # S3 / S3a: the market favorite, no model
            fav_home = home_mid >= 0.5
            fav = {'home_side': fav_home, 'fill': ask if fav_home else 1 - bid,
                   'mid': home_mid if fav_home else 1 - home_mid,
                   'p': r['MODEL_HOME_PROB'] if fav_home else 1 - r['MODEL_HOME_PROB'],
                   'team': r.get('HOME_ABBR') if fav_home else r.get('AWAY_ABBR')}
            ml_leg('S3', **fav)
            if fav['mid'] >= 0.90:
                ml_leg('S3a', **fav)
        side = r.get('PICK_SIDE')
        if isinstance(side, str) and side:
            pick = {'home_side': side == 'YES', 'fill': r['PICK_FILL'], 'mid': r['PICK_MID'], 'p': r['PICK_P'],
                    'team': r.get('PICK_TEAM')}
            ml_leg('S4', **pick)
            if _is_true(r.get('PICK_MARKET_UNDERDOG')):
                ml_leg('S1', **pick)
            if r['PICK_P'] > r['PICK_MID']:
                ml_leg('S5/S6 legs', **pick)
        sside = r.get('SPREAD_SIDE')
        if isinstance(sside, str) and sside and r.get('SPREAD_EDGE_PTS', np.nan) >= 10 \
                and r.get('SPREAD_STRIKE') == r.get('SPREAD_STRIKE'):
            won = None
            if played:
                fav_margin = margin if r.get('SPREAD_FAV') == r.get('HOME_ABBR') else -margin
                won = bool((fav_margin > r['SPREAD_STRIKE']) == (sside == 'YES'))
            rows.append({**base, 'strategy': 'S2', 'team': r.get('SPREAD_FAV') if sside == 'YES' else 'dog',
                         'fill': r['SPREAD_FILL'], 'mid': r['SPREAD_MID'], 'p': r['SPREAD_P'], 'won': won,
                         'clv': None})
    out = pd.DataFrame(rows, columns=['strategy', 'GAME_ID', 'GAME_DATE', 'TIP_TIME_ET', 'MODEL_VERSION', 'team',
                                      'fill', 'mid', 'p', 'won', 'clv'])
    out = out[(out['fill'] > 0) & (out['fill'] < 1)]
    return out.sort_values(['GAME_DATE', 'TIP_TIME_ET', 'GAME_ID']).reset_index(drop=True)


def lineups(pool, k):
    """
    One lineup per day from the first k legs of `pool` by tip-off (different games). Returns one
    row per lineup: GAME_DATE, legs, price (product of asks), multiplier, won (None if pending).
    """
    rows = []
    for day, g in pool.groupby('GAME_DATE', sort=True):
        g = g.drop_duplicates('GAME_ID').sort_values(['TIP_TIME_ET', 'GAME_ID'])
        if len(g) < k:
            continue
        g = g.iloc[:k]
        price = float(np.prod(g['fill']))
        won = None if g['won'].isna().any() else bool(g['won'].astype(bool).all())
        rows.append({'GAME_DATE': day, 'legs': ', '.join(map(str, g['team'])), 'price': price,
                     'multiplier': float(prizepicks_multiplier(price)), 'won': won})
    return pd.DataFrame(rows, columns=['GAME_DATE', 'legs', 'price', 'multiplier', 'won'])


# ---------------------------------------------------------------- measures

def summary(ret, won, price):
    """n, win rate, avg price, mean return per $1 and its SE, t."""
    ret = np.asarray(ret, dtype=float)
    n = len(ret)
    mean = ret.mean() if n else np.nan
    se = ret.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return {'n': n, 'win_rate': np.mean(won) if n else np.nan, 'avg_price': np.mean(price) if n else np.nan,
            'mean': mean, 'se': se, 't': mean / se if se and se == se and se > 0 else np.nan}


def bankroll(ret, won, stake=1.0, start=START_BANKROLL):
    """Flat `stake` per bet in order: final bankroll, max drawdown ($), worst losing streak."""
    b, peak, dd, streak, worst = start, start, 0.0, 0, 0
    for r, w in zip(ret, won):
        if b < stake:
            break
        b += stake * r
        peak, dd = max(peak, b), max(dd, peak - b)
        streak = 0 if w else streak + 1
        worst = max(worst, streak)
    return {'end': b, 'max_drawdown': dd, 'worst_streak': worst}


def status(name, s, clv=None):
    """The plan's written-in-advance decision rules."""
    if name in ('S3', 'S4'):
        return 'yardstick'
    if name == 'S6':
        return 'watch only'
    if name == 'S5':
        if s['n'] >= 100 and s['mean'] <= -0.12:
            return 'STOP (dead)'
        if s['n'] >= 100 and s['mean'] > 0 and s['t'] >= 2:
            return 'promising (not proven)'
        return _running(s['n'], 'lineups')
    if s['n'] >= 100 and s['mean'] <= -0.02:
        return 'STOP (dead)'
    if s['n'] >= 150 and s['mean'] > 0 and s['t'] >= 2:
        return 'promising (not proven)'
    if clv is not None and clv['n'] >= 50 and clv['mean'] > 0 and clv['t'] >= 2:
        return 'promising on CLV (not proven)'
    return _running(s['n'], 'bets')


def _running(n, unit):
    if n < 100:
        return f'running ({n}/100 {unit} before the stop rule can apply)'
    return f'running: neither rule met yet ({n} {unit})'


def strategy_tables(L):
    """Single-bet, lineup and CLV tables and the plan's decision status for one model's legs."""
    settled = L[L['won'].notna()].copy()
    settled['won'] = settled['won'].astype(bool)
    singles, clv_rows, statuses = [], [], {}
    for code, label in STRATEGIES.items():
        g = settled[settled['strategy'] == code]
        k_pnl = np.array([kalshi_pnl(f, w) for f, w in zip(g['fill'], g['won'])])
        k_cost = (g['fill'] + g['fill'].map(D.fee)).values
        pp_ret = np.where(g['won'], prizepicks_multiplier(g['fill']), 0) - 1 if len(g) else np.array([])
        sk = summary(k_pnl, g['won'], g['fill'])
        sp = summary(pp_ret, g['won'], g['fill'])
        bk = bankroll(k_pnl / k_cost if len(g) else [], g['won'])
        c = g['clv'].dropna()
        sc = summary(c.values, np.ones(len(c)), np.ones(len(c)))
        singles.append({'id': code, 'strategy': label, 'bets': sk['n'], 'win rate': sk['win_rate'],
                        'avg price': sk['avg_price'], 'Kalshi c/contract': sk['mean'] * 100,
                        'Kalshi SE': sk['se'] * 100, 'Kalshi return/$': k_pnl.sum() / k_cost.sum() if len(g) else np.nan,
                        'PrizePicks return/$': sp['mean'], 'PrizePicks SE': sp['se'],
                        '$100 end': bk['end'], 'max drawdown $': bk['max_drawdown'], 'worst streak': bk['worst_streak']})
        clv_rows.append({'id': code, 'legs with CLV': sc['n'], 'CLV c (mean)': sc['mean'], 'CLV SE': sc['se'],
                         't': sc['t']})
        if code != 'S5/S6 legs':
            statuses[code] = status(code, sk, sc if sc['n'] else None)

    lineup_rows, lineup_detail = [], {}
    for code in STRATEGIES:
        pool = L[L['strategy'] == code]
        for k in LINEUP_SIZES:
            lu = lineups(pool, k)
            done = lu[lu['won'].notna()]
            ret = np.where(done['won'].astype(bool), done['multiplier'], 0) - 1 if len(done) else np.array([])
            s = summary(ret, done['won'].astype(bool) if len(done) else [], done['multiplier'] if len(done) else [])
            bk = bankroll(ret, done['won'].astype(bool) if len(done) else [])
            tag = next((n for n, (p, kk) in LINEUP_RULES.items() if p == code and kk == k), '')
            lineup_rows.append({'id': tag or f'{code} x{k}', 'legs from': code, 'picks': k, 'lineups': s['n'],
                                'pending': int(lu['won'].isna().sum()), 'hit rate': s['win_rate'],
                                'avg multiplier': s['avg_price'], 'return/$': s['mean'], 'SE': s['se'],
                                '$100 end': bk['end'], 'worst streak': bk['worst_streak']})
            if tag:
                statuses[tag] = status(tag, s)
                lineup_detail[tag] = lu
    return {'singles': pd.DataFrame(singles), 'lineups': pd.DataFrame(lineup_rows), 'clv': pd.DataFrame(clv_rows),
            'status': statuses, 'legs': L, 'lineup_detail': lineup_detail}


def _log_loss(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def model_comparison(entry, margins):
    """
    Live vs frozen model vs market on the same finished games (both models and a price logged),
    by month and for the season: log loss, accuracy, and the paired live-minus-frozen log loss.
    """
    if 'FROZEN_HOME_PROB' not in entry:
        return pd.DataFrame()
    e = entry[entry['FROZEN_HOME_PROB'].notna() & entry['HOME_MID'].notna()].copy()
    e['margin'] = e['GAME_ID'].map(margins)
    e = e[e['margin'].notna() & (e['margin'] != 0)]
    if e.empty:
        return pd.DataFrame()
    e['y'] = (e['margin'] > 0).astype(int)
    e['month'] = pd.to_datetime(e['GAME_DATE']).dt.to_period('M').astype(str)
    rows = []
    for month, g in list(e.groupby('month')) + [('Season', e)]:
        ll = {m: _log_loss(g['y'].values, g[c].values) for m, c in
              (('live', 'MODEL_HOME_PROB'), ('frozen', 'FROZEN_HOME_PROB'), ('market', 'HOME_MID'))}
        d = ll['live'] - ll['frozen']
        rows.append({'month': month, 'games': len(g), 'live log loss': ll['live'].mean(),
                     'frozen log loss': ll['frozen'].mean(), 'market log loss': ll['market'].mean(),
                     'live - frozen': d.mean(), 'diff SE': d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan,
                     'live accuracy': ((g['MODEL_HOME_PROB'] > 0.5) == g['y']).mean(),
                     'frozen accuracy': ((g['FROZEN_HOME_PROB'] > 0.5) == g['y']).mean()})
    return pd.DataFrame(rows)


def build(log=None, games_path='data/raw_nba_data.csv', since=SEASON_START, tip_path=TIP_PRICES_PATH):
    """
    All report tables (no file output): coverage, per-model strategy tables ('live' and, once the
    model is frozen, 'frozen'), the live/frozen/market comparison, and which model the plan's
    decision rules are judged on ('frozen' when it exists).
    """
    log = pl.load() if log is None else log
    if log is None or log.empty:
        return None
    entry, coverage = entries(log, since)
    margins, tips = results(games_path), tip_prices(tip_path)
    models = {}
    for m in ('live', 'frozen'):
        view = model_view(entry, m)
        if m == 'frozen' and view.empty:
            continue
        models[m] = strategy_tables(legs(view, margins, tips))
    n_frozen = len(model_view(entry, 'frozen'))
    coverage.update({'legs pending (unplayed)': int(models['live']['legs']['won'].isna().sum()),
                     'frozen model': (f'logged for {n_frozen} games' if n_frozen
                                      else 'not frozen yet (python src/frozen_model.py)'),
                     'CLV source': 'Kalshi price at tip-off (data/tip_prices.csv); last pre-tip-off logged price '
                     'where missing' if tips is not None
                     else 'last pre-tip-off logged price (run: python src/market_odds.py --tip-prices)'})
    return {'coverage': coverage, 'models': models, 'comparison': model_comparison(entry, margins),
            'judged_on': 'frozen' if 'frozen' in models else 'live'}


def _md(df, fmt):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in df.itertuples(index=False):
        lines.append('| ' + ' | '.join(fmt(c, v) for c, v in zip(cols, r)) + ' |')
    return '\n'.join(lines)


PCT = {'win rate', 'hit rate', 'PrizePicks SE', 'SE', 'live accuracy', 'frozen accuracy'}
SIGNED_PCT = {'Kalshi return/$', 'PrizePicks return/$', 'return/$'}
FORMATS = {'avg price': '{:.3f}', 'avg multiplier': '{:.2f}x', 'Kalshi c/contract': '{:+.1f}',
           'CLV c (mean)': '{:+.1f}', 'Kalshi SE': '{:.1f}', 'CLV SE': '{:.1f}', '$100 end': '${:.0f}',
           'max drawdown $': '${:.0f}', 't': '{:+.2f}', 'live log loss': '{:.4f}', 'frozen log loss': '{:.4f}',
           'market log loss': '{:.4f}', 'live - frozen': '{:+.4f}', 'diff SE': '{:.4f}'}


def _fmt(col, v):
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '-'
    if col in PCT:
        return f'{v:.1%}'
    if col in SIGNED_PCT:
        return f'{v:+.1%}'
    if col in FORMATS:
        return FORMATS[col].format(v)
    return f'{v:g}' if isinstance(v, float) else str(v)


def report(tables, since=SEASON_START):
    """Markdown text of the report."""
    if tables is None:
        return '# Paper-trade report\n\nNo prediction log yet. Hypothetical analysis, not betting advice.\n'
    judged = tables['judged_on']
    lines = ['# Paper-trade report, 2026-27 (local only: uses Kalshi prices)', '',
             'Hypothetical analysis, not betting advice. Paper trading only; no orders are placed.', '',
             f'Rules: docs/paper_trading_plan.md. Games from {since}; entry = last prediction logged '
             f'{ENTRY_MINUTES}+ minutes before tip-off. PrizePicks prices are reverse-engineered estimates.', '',
             '## Coverage', '']
    lines += [f'- {k}: {v}' for k, v in tables['coverage'].items()]
    lines += ['', f'## Decision status (rules written before the season; judged on the {judged} model)', '']
    lines += [f'- **{k}** {LABELS[k]}: {v}' for k, v in tables['models'][judged]['status'].items()]
    comp = tables['comparison']
    lines += ['', '## Live vs frozen model vs market (same finished games; lower log loss is better)', '',
              _md(comp, _fmt) if len(comp) else 'No finished games with both models logged yet.', '']
    for m, t in tables['models'].items():
        name = m.capitalize()
        lines += [f'## {name} model: single bets', '',
                  'Kalshi: $1-payout contract at the ask plus fee (cents per contract, and return per $ staked). '
                  'PrizePicks: return per $1 on a single pick. Bankroll: $100, flat $1 per bet (Kalshi prices).', '',
                  _md(t['singles'], _fmt), '',
                  f'## {name} model: lineups (PrizePicks pricing, one per day from the first k legs by tip-off)', '',
                  _md(t['lineups'], _fmt), '',
                  f'## {name} model: closing line value (cents per leg; positive = the market moved toward our '
                  'side before tip-off)', '', _md(t['clv'], _fmt), '']
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--since', default=SEASON_START)
    ap.add_argument('--log', default=pl.LOG_PATH)
    ap.add_argument('--games', default='data/raw_nba_data.csv')
    ap.add_argument('--out', default=REPORT_PATH)
    args = ap.parse_args(argv)
    log = pl.load(args.log)
    tables = build(log, args.games, args.since)
    text = report(tables, args.since)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)
    return tables


if __name__ == '__main__':
    main()
