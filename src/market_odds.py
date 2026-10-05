"""
Kalshi NBA game-winner and spread markets (the prices behind PrizePicks game picks), read-only.

Uses Kalshi's public market-data API (no API key needed for market data). This module only reads
prices; it contains no trading or order code. Kalshi's API Developer Agreement limits API use to
a member's own trading and forbids sharing the data, so everything saved here is git-ignored and
stays on your machine.

Spread markets (series KXNBASPREAD, same event naming): a ladder per team,
<event>-<TEAM><N> = 'TEAM wins by over N.5 points'; YES pays $1 if it does. The market's line
is where the favorite's ladder crosses 50 cents (interpolated).

    Base URL:   https://external-api.kalshi.com/trade-api/v2  (docs.kalshi.com, October 2026)
    Series:     KXNBAGAME ("NBA Game")
    Event:      KXNBAGAME-<YY><MON><DD><AWAY><HOME>    e.g. KXNBAGAME-26MAR03DALCHA = Dallas at Charlotte
    Markets:    <event>-<TEAM>, one per team; YES pays $1 if that team wins.
                Team codes are standard NBA abbreviations (TEAM_ABBREVIATION in the box scores).

Market win probability = midpoint of the home team's YES bid and ask.

    python src/market_odds.py             # append a snapshot of today's games to data/market_snapshots.csv
    python src/market_odds.py --history   # pre-tip-off prices for past games -> data/market_history.csv
    python src/market_odds.py --spread-history   # pre-tip-off spread lines -> data/market_spread_history.csv
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


# ---------------------------------------------------------------- spread markets

SPREAD_SERIES = 'KXNBASPREAD'
SPREAD_HISTORY_PATH = 'data/market_spread_history.csv'
SPREAD_SNAPSHOT_PATH = 'data/market_spread_snapshots.csv'


def spread_event_ticker(game_date, away_code, home_code):
    return event_ticker(game_date, away_code, home_code).replace(SERIES, SPREAD_SERIES, 1)


def _ladder_rows(markets):
    """Market dicts -> DataFrame: TICKER, TEAM, STRIKE, YES_BID, YES_ASK, MID."""
    rows = []
    for m in markets:
        suffix = m['ticker'].rsplit('-', 1)[1]
        team = ''.join(ch for ch in suffix if ch.isalpha())
        bid, ask = _dollars(m.get('yes_bid_dollars')), _dollars(m.get('yes_ask_dollars'))
        rows.append({'TICKER': m['ticker'], 'TEAM': team, 'STRIKE': _dollars(m.get('floor_strike')),
                     'YES_BID': bid, 'YES_ASK': ask, 'MID': midpoint(bid, ask)})
    return pd.DataFrame(rows, columns=['TICKER', 'TEAM', 'STRIKE', 'YES_BID', 'YES_ASK', 'MID'])


def spread_ladder(ticker, historical=False):
    """All spread markets of one event (live quotes if the event is open)."""
    if not historical:
        data = _get(f'/events/{ticker}', {'with_nested_markets': 'true'}) or {}
        markets = data.get('markets') or (data.get('event') or {}).get('markets') or []
        if markets:
            return _ladder_rows(markets)
    data = _get('/historical/markets', {'event_ticker': ticker, 'limit': 200}) or {}
    return _ladder_rows(data.get('markets', []))


def implied_line(strikes, mids):
    """
    Margin where P(team wins by more than x) crosses 0.5, by linear interpolation along the
    ladder (strikes ascending, prices falling). None if the ladder never crosses 0.5.
    """
    pts = sorted((s, m) for s, m in zip(strikes, mids) if m == m and s == s)
    for (s0, m0), (s1, m1) in zip(pts, pts[1:]):
        if m0 >= 0.5 >= m1 and m0 != m1:
            return s0 + (m0 - 0.5) / (m0 - m1) * (s1 - s0)
    return None


def main_line(ladder, team):
    """The team's ladder market priced closest to 50 cents (its 'main line'), or None."""
    t = ladder[ladder['TEAM'] == team].dropna(subset=['MID'])
    if t.empty:
        return None
    return t.loc[(t['MID'] - 0.5).abs().idxmin()]


def pregame_spread(event, favorite, tip_et, cutoff, pause=0.15):
    """
    Pre-tip-off market line for one past game: binary search along the favorite's ladder for
    the 50-cent crossing (about five price lookups instead of the whole ladder).
    Returns FAV, FAV_LINE (favorite's implied margin) and the main-line market's quote.
    """
    lad = spread_ladder(event, historical=True)
    lad = lad[lad['TEAM'] == favorite].dropna(subset=['STRIKE']).sort_values('STRIKE').reset_index(drop=True)
    if lad.empty:
        return None
    settled = tip_et + pd.Timedelta(hours=6) < cutoff
    quotes = {}

    def mid_at(i):
        if i not in quotes:
            bid, ask = pregame_quote(lad.at[i, 'TICKER'], tip_et, settled)
            quotes[i] = (bid, ask, midpoint(bid, ask))
            time.sleep(pause)
        return quotes[i][2]

    lo, hi = 0, len(lad) - 1
    mid_at(lo)
    mid_at(hi)
    while hi - lo > 1:      # prices fall as the strike rises
        i = (lo + hi) // 2
        m = mid_at(i)
        if m != m or m < 0.5:
            hi = i
        else:
            lo = i
    known = sorted(quotes)
    line = implied_line([lad.at[i, 'STRIKE'] for i in known], [quotes[i][2] for i in known])
    priced = [i for i in known if quotes[i][2] == quotes[i][2]]
    if not priced:
        return {'FAV': favorite, 'FAV_LINE': line}
    best = min(priced, key=lambda i: abs(quotes[i][2] - 0.5))
    bid, ask, m = quotes[best]
    return {'FAV': favorite, 'FAV_LINE': line, 'MAIN_TICKER': lad.at[best, 'TICKER'],
            'MAIN_STRIKE': lad.at[best, 'STRIKE'], 'MAIN_YES_BID': bid, 'MAIN_YES_ASK': ask, 'MAIN_MID': m}


def build_spread_history(path=SPREAD_HISTORY_PATH, moneyline_path=HISTORY_PATH):
    """
    Pre-tip-off spread line for every game in the moneyline history (local, git-ignored).
    HOME_LINE is in betting convention (negative = home favored). Re-running skips cached games.
    """
    ml = pd.read_csv(moneyline_path, dtype={'GAME_ID': str}).dropna(subset=['MARKET_HOME_PROB'])
    done = pd.read_csv(path, dtype={'GAME_ID': str}) if os.path.exists(path) else pd.DataFrame()
    done_ids = set(done['GAME_ID']) if len(done) else set()
    cutoff = pd.Timestamp((_get('/historical/cutoff') or {}).get('market_settled_ts', '2100-01-01'))
    rows = []
    for r in ml.itertuples():
        if r.GAME_ID in done_ids:
            continue
        day, away, home = parse_event_ticker(r.EVENT_TICKER)
        fav = home if r.MARKET_HOME_PROB >= 0.5 else away
        tip = pd.Timestamp(r.TIP_TIME_ET).tz_localize(ET)
        try:
            res = pregame_spread(spread_event_ticker(day, away, home), fav, tip, cutoff)
        except requests.RequestException:
            res = None
        row = {'GAME_ID': r.GAME_ID, 'GAME_DATE': day.date(), 'HOME': home, 'AWAY': away,
               'TIP_TIME_ET': r.TIP_TIME_ET}
        if res:
            row.update(res)
            if res.get('FAV_LINE') is not None:
                row['HOME_LINE'] = -res['FAV_LINE'] if fav == home else res['FAV_LINE']
        rows.append(row)
        if len(rows) % 100 == 0:
            print(f"  {len(rows)} games (through {day.date()})", flush=True)
            _save_history(done, rows, path)
    return _save_history(done, rows, path)


def open_spread_events():
    events, cursor = [], None
    while True:
        params = {'series_ticker': SPREAD_SERIES, 'status': 'open', 'with_nested_markets': 'true', 'limit': 200}
        if cursor:
            params['cursor'] = cursor
        page = _get('/events', params) or {}
        events += page.get('events', [])
        cursor = page.get('cursor')
        if not cursor or not page.get('events'):
            return events


def spread_snapshot(game_date=None, path=SPREAD_SNAPSHOT_PATH):
    """Append the current spread ladders of a day's games (local, append-only). Returns them."""
    game_date = pd.Timestamp(game_date or pd.Timestamp.now(tz=ET).date())
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    frames = []
    for ev in open_spread_events():
        day, away, home = parse_event_ticker(ev['event_ticker'])
        if day != game_date:
            continue
        lad = _ladder_rows(ev.get('markets', []))
        if not lad.empty:
            frames.append(lad.assign(SNAPSHOT_TIME_UTC=now, GAME_DATE=day.date(),
                                     EVENT_TICKER=ev['event_ticker'], HOME=home, AWAY=away))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)[['SNAPSHOT_TIME_UTC', 'GAME_DATE', 'EVENT_TICKER', 'HOME', 'AWAY',
                                               'TICKER', 'TEAM', 'STRIKE', 'YES_BID', 'YES_ASK', 'MID']]
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
    return df


def market_spread_from_ladder(ladder, home, away):
    """
    From one game's ladder: the market's home line (betting convention) and the main-line
    market of the favorite (the team whose ladder crosses 50 cents at the larger margin).
    """
    best = None
    for team in (home, away):
        t = ladder[ladder['TEAM'] == team]
        line = implied_line(t['STRIKE'], t['MID'])
        if line is not None and (best is None or line > best[1]):
            best = (team, line)
    if best is None:
        return None
    fav, fav_line = best
    main = main_line(ladder, fav)
    out = {'FAV': fav, 'FAV_LINE': fav_line, 'HOME_LINE': -fav_line if fav == home else fav_line}
    if main is not None:
        out.update({'MAIN_TICKER': main['TICKER'], 'MAIN_STRIKE': main['STRIKE'],
                    'MAIN_YES_BID': main['YES_BID'], 'MAIN_YES_ASK': main['YES_ASK'], 'MAIN_MID': main['MID']})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', action='store_true')
    parser.add_argument('--spread-history', action='store_true')
    parser.add_argument('--date', default=None, help='game date for the snapshot (default today, ET)')
    args = parser.parse_args()
    if args.spread_history:
        h = build_spread_history()
        n = int(h['HOME_LINE'].notna().sum()) if 'HOME_LINE' in h else 0
        print(f"{n} of {len(h)} games have a pre-tip-off spread line -> {SPREAD_HISTORY_PATH}")
    elif args.history:
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
