"""
Kalshi NBA game-winner markets (the prices behind PrizePicks game picks), read-only.

Uses Kalshi's public market-data API, which needs no login or API key. This module only reads
prices; it contains no trading or order code.

    Base URL:   https://external-api.kalshi.com/trade-api/v2  (docs.kalshi.com, October 2026)
    Series:     KXNBAGAME ("NBA Game")
    Event:      KXNBAGAME-<YY><MON><DD><AWAY><HOME>    e.g. KXNBAGAME-26MAR03DALCHA = Dallas at Charlotte
    Markets:    <event>-<TEAM>, one per team; YES pays $1 if that team wins.
                Team codes are standard NBA abbreviations (TEAM_ABBREVIATION in the box scores).

Market win probability = midpoint of the home team's YES bid and ask.

    python src/market_odds.py             # append a snapshot of today's games to data/market_snapshots.csv
    python src/market_odds.py --history   # pre-tip-off prices for past games -> data/market_history.csv
"""
import argparse
import os
import time
from datetime import datetime, timezone
import pandas as pd
import requests

BASE_URL = 'https://external-api.kalshi.com/trade-api/v2'
SERIES = 'KXNBAGAME'
SNAPSHOT_PATH = 'data/market_snapshots.csv'
HISTORY_PATH = 'data/market_history.csv'
SNAPSHOT_COLUMNS = ['SNAPSHOT_TIME_UTC', 'GAME_DATE', 'EVENT_TICKER', 'COMPETITION', 'AWAY_TEAM',
                    'HOME_TEAM', 'MARKET_HOME_PROB', 'HOME_YES_BID', 'HOME_YES_ASK',
                    'AWAY_YES_BID', 'AWAY_YES_ASK', 'VOLUME', 'STATUS', 'GAME_STATUS',
                    'TIP_TIME_ET', 'PREGAME']
MAX_SPREAD = 0.25  # wider quotes than this are treated as no price
MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
ET = 'America/New_York'


def _get(path, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(BASE_URL + path, params=params, timeout=30)
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                time.sleep(2 ** attempt + 1)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return None


def team_names(data_path='data/raw_nba_data.csv'):
    """NBA abbreviation -> team name as used in the box scores (latest season wins)."""
    games = pd.read_csv(data_path, usecols=['TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_DATE'])
    games = games.sort_values('GAME_DATE')
    return dict(zip(games['TEAM_ABBREVIATION'], games['TEAM_NAME']))


def event_ticker(game_date, away_code, home_code):
    d = pd.Timestamp(game_date)
    return f"{SERIES}-{d:%y}{MONTHS[d.month - 1]}{d:%d}{away_code}{home_code}"


def parse_event_ticker(ticker):
    """'KXNBAGAME-26MAR03DALCHA' -> (Timestamp('2026-03-03'), 'DAL', 'CHA')"""
    code = ticker.split('-')[1]
    day = pd.Timestamp(year=2000 + int(code[:2]), month=MONTHS.index(code[2:5]) + 1, day=int(code[5:7]))
    teams = code[7:]
    return day, teams[:3], teams[3:]


def _dollars(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def midpoint(bid, ask):
    """Midpoint of a two-sided quote; NaN for an empty or very wide book."""
    if not (0 <= bid <= ask <= 1) or ask == 0 or ask - bid > MAX_SPREAD:
        return float('nan')
    return (bid + ask) / 2


# ---------------------------------------------------------------- live

def open_events():
    events, cursor = [], None
    while True:
        params = {'series_ticker': SERIES, 'status': 'open', 'with_nested_markets': 'true', 'limit': 200}
        if cursor:
            params['cursor'] = cursor
        page = _get('/events', params) or {}
        events += page.get('events', [])
        cursor = page.get('cursor')
        if not cursor or not page.get('events'):
            return events


def games_on(game_date=None, names=None):
    """One row per game on `game_date` (default today, Eastern) with the market's prices."""
    game_date = pd.Timestamp(game_date or pd.Timestamp.now(tz=ET).date())
    names = names or team_names()
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    rows = []
    for ev in open_events():
        day, away, home = parse_event_ticker(ev['event_ticker'])
        if day != game_date:
            continue
        markets = {m['ticker'].rsplit('-', 1)[1]: m for m in ev.get('markets', [])}
        h, a = markets.get(home, {}), markets.get(away, {})
        hb, ha = _dollars(h.get('yes_bid_dollars')), _dollars(h.get('yes_ask_dollars'))
        rows.append({
            'SNAPSHOT_TIME_UTC': now, 'GAME_DATE': day.date(), 'EVENT_TICKER': ev['event_ticker'],
            'COMPETITION': (ev.get('product_metadata') or {}).get('competition'),
            'AWAY_TEAM': names.get(away, away), 'HOME_TEAM': names.get(home, home),
            'MARKET_HOME_PROB': midpoint(hb, ha), 'HOME_YES_BID': hb, 'HOME_YES_ASK': ha,
            'AWAY_YES_BID': _dollars(a.get('yes_bid_dollars')), 'AWAY_YES_ASK': _dollars(a.get('yes_ask_dollars')),
            'VOLUME': _dollars(h.get('volume_fp')), 'STATUS': h.get('status'),
        })
    df = pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)
    return _add_game_status(df, game_date)


def _add_game_status(df, game_date):
    """Tip-off time and status from the NBA scoreboard, so pre-game prices can be told apart
    from in-game ones. Left blank if the scoreboard can't be reached."""
    if df.empty:
        return df
    try:
        from schedule import games_on as schedule_on
        sched = schedule_on(game_date).set_index(['HOME_TEAM', 'AWAY_TEAM'])
    except Exception as e:
        print(f"[!] NBA scoreboard unavailable ({type(e).__name__}); PREGAME left blank")
        return df
    now_et = pd.Timestamp.now(tz=ET).tz_localize(None)
    status, tips, pregame = [], [], []
    for home, away in zip(df['HOME_TEAM'], df['AWAY_TEAM']):
        if (home, away) not in sched.index:
            status.append(None), tips.append(None), pregame.append(None)
            continue
        g = sched.loc[(home, away)]
        status.append(g['STATUS'])
        tips.append(g['TIP_TIME_ET'])
        pregame.append(bool(g['STATUS'] == 'scheduled' and
                            (pd.isna(g['TIP_TIME_ET']) or now_et < g['TIP_TIME_ET'])))
    return df.assign(GAME_STATUS=status, TIP_TIME_ET=tips, PREGAME=pregame)


def snapshot(game_date=None, path=SNAPSHOT_PATH):
    """Append today's market prices to the snapshot log (never rewrites earlier rows)."""
    df = games_on(game_date)
    if df.empty:
        return df
    header = not os.path.exists(path)
    df.to_csv(path, mode='a', header=header, index=False)
    return df


def latest_snapshot(game_date=None, path=SNAPSHOT_PATH):
    """Most recent logged price per game on `game_date`, without calling the API."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    df = pd.read_csv(path)
    game_date = str(pd.Timestamp(game_date or pd.Timestamp.now(tz=ET).date()).date())
    df = df[df['GAME_DATE'].astype(str) == game_date]
    return df.sort_values('SNAPSHOT_TIME_UTC').groupby('EVENT_TICKER').tail(1)


# ---------------------------------------------------------------- history

def pregame_quote(market_ticker, tip_time, settled_before_cutoff, hours=6):
    """
    (bid, ask) for a market from the last hourly candlestick ending at or before tip-off.
    tip_time: tz-aware timestamp.
    """
    end = int(pd.Timestamp(tip_time).timestamp())
    params = {'start_ts': end - hours * 3600, 'end_ts': end, 'period_interval': 60}
    path = (f'/historical/markets/{market_ticker}/candlesticks' if settled_before_cutoff
            else f'/series/{SERIES}/markets/{market_ticker}/candlesticks')
    data = _get(path, params)
    if not data:
        return float('nan'), float('nan')
    candles = [c for c in data.get('candlesticks', []) if c['end_period_ts'] <= end]
    for c in reversed(candles):
        bid, ask = _dollars((c.get('yes_bid') or {}).get('close')), _dollars((c.get('yes_ask') or {}).get('close'))
        if 0 <= bid < ask <= 1:
            return bid, ask
    return float('nan'), float('nan')


def tip_times_from_reports():
    """(game date, home team) -> scheduled tip-off (Eastern) from archived injury reports."""
    from injury_reports import load_archive, parse_tip_time
    r = load_archive()
    if r.empty:
        return {}
    r = r.dropna(subset=['GAME_TIME', 'MATCHUP']).drop_duplicates(['GAME_DATE', 'MATCHUP'])
    out = {}
    for row in r.itertuples():
        home_code = str(row.MATCHUP).split('@')[-1].strip()
        out[(pd.Timestamp(row.GAME_DATE), home_code)] = parse_tip_time(row.GAME_DATE, row.GAME_TIME)
    return out


def build_history(games_path='data/raw_nba_data.csv', path=HISTORY_PATH, since='2025-04-15'):
    """
    Pre-tip-off Kalshi price for every regular-season game in the box scores since Kalshi's NBA
    game markets began. Tip times come from the archived injury reports (if
    unknown, noon ET); the price is the bid/ask midpoint of the last hourly candle ending at tip-off.
    Cached in data/market_history.csv; re-running fetches only games not yet cached.
    """
    games = pd.read_csv(games_path, parse_dates=['GAME_DATE'], dtype={'GAME_ID': str})
    home = games[games['MATCHUP'].str.contains('vs.') & (games['GAME_DATE'] >= pd.Timestamp(since))]
    away = games[games['MATCHUP'].str.contains('@')][['GAME_ID', 'TEAM_ABBREVIATION']]
    home = home.merge(away, on='GAME_ID', suffixes=('', '_AWAY'))

    done = pd.read_csv(path, dtype={'GAME_ID': str}) if os.path.exists(path) else pd.DataFrame()
    done_ids = set(done['GAME_ID']) if len(done) else set()
    cutoff = pd.Timestamp((_get('/historical/cutoff') or {}).get('market_settled_ts', '2100-01-01'))
    tips = tip_times_from_reports()

    rows = []
    for i, g in enumerate(home.itertuples()):
        if g.GAME_ID in done_ids:
            continue
        ticker = event_ticker(g.GAME_DATE, g.TEAM_ABBREVIATION_AWAY, g.TEAM_ABBREVIATION)
        # Unknown tip time: noon ET, which is before every regular-season tip-off (never in-game)
        tip = tips.get((g.GAME_DATE, g.TEAM_ABBREVIATION)) or g.GAME_DATE + pd.Timedelta(hours=12)
        tip_et = pd.Timestamp(tip).tz_localize(ET)
        # Settlement happens a few hours after tip-off; use that to pick live vs historical
        settled_before_cutoff = tip_et + pd.Timedelta(hours=6) < cutoff
        bid, ask = pregame_quote(f'{ticker}-{g.TEAM_ABBREVIATION}', tip_et, settled_before_cutoff)
        rows.append({'GAME_ID': g.GAME_ID, 'GAME_DATE': g.GAME_DATE.date(), 'EVENT_TICKER': ticker,
                     'TIP_TIME_ET': tip_et.tz_localize(None), 'HOME_YES_BID': bid, 'HOME_YES_ASK': ask,
                     'MARKET_HOME_PROB': midpoint(bid, ask)})
        if len(rows) % 100 == 0:
            print(f"  {len(rows)} games fetched (through {g.GAME_DATE.date()})", flush=True)
            _save_history(done, rows, path)
        time.sleep(0.1)
    return _save_history(done, rows, path)


def _save_history(done, rows, path):
    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if len(done) else pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', action='store_true')
    parser.add_argument('--date', default=None, help='game date for the snapshot (default today, ET)')
    args = parser.parse_args()
    if args.history:
        hist = build_history()
        print(f"{hist['MARKET_HOME_PROB'].notna().sum()} of {len(hist)} games have a pre-tip-off price "
              f"-> {HISTORY_PATH}")
    else:
        snap = snapshot(args.date)
        if snap.empty:
            print("No open Kalshi NBA game markets for this date.")
        else:
            print(snap[['AWAY_TEAM', 'HOME_TEAM', 'COMPETITION', 'MARKET_HOME_PROB',
                        'HOME_YES_BID', 'HOME_YES_ASK']].to_string(index=False))
            print(f"Appended {len(snap)} rows to {SNAPSHOT_PATH}")
