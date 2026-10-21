"""
Games on a date from stats.nba.com's scoreboard (ScoreboardV2), with tip-off times and status.

stats.nba.com sometimes returns an error page instead of JSON, occasionally for longer than the
retries cover. Every successful lookup is therefore cached per date (data/schedule_cache/, local),
and when the scoreboard is unavailable the cached schedule is used: a cached game still counts as
scheduled until its tip-off time. The scoreboard only answers for today and past dates (future
dates return HTTP 500 or time out, checked October 2026), so the morning refresh caches today
(python src/schedule.py --prefetch), an hour before the first slate run, and every successful
slate run refreshes the cache.
"""
import argparse
import os
import re
import time
from datetime import date
import pandas as pd

STATUS = {1: 'scheduled', 2: 'live', 3: 'final'}
RETRIES = 3            # stats.nba.com sometimes returns an error page instead of JSON for a few minutes
RETRY_DELAY = 10       # seconds before the 2nd attempt, doubling after that
CACHE_DIR = 'data/schedule_cache'
ET = 'America/New_York'
COLUMNS = ['GAME_ID', 'GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'STATUS', 'TIP_TIME_ET']


def _v2_header(date_str):
    from nba_api.stats.endpoints import scoreboardv2
    return scoreboardv2.ScoreboardV2(game_date=date_str, timeout=30).game_header.get_data_frame()


def _v3_header(date_str):
    """
    The same columns from ScoreboardV3, a separate endpoint that kept working on mornings when
    ScoreboardV2 returned error pages (2026-10-20 and 10-21). Home and away come from the game
    code ('20261021/ATLORL' = Atlanta at Orlando).
    """
    from nba_api.stats.endpoints import scoreboardv3
    dfs = scoreboardv3.ScoreboardV3(game_date=date_str, timeout=30).get_data_frames()
    games, teams = dfs[1], dfs[2]
    ids = {(g, t): i for g, t, i in zip(teams['gameId'], teams['teamTricode'], teams['teamId'])}
    code = games['gameCode'].str.split('/').str[-1]
    return pd.DataFrame({
        'GAME_ID': games['gameId'],
        'HOME_TEAM_ID': [ids.get((g, c[3:])) for g, c in zip(games['gameId'], code)],
        'VISITOR_TEAM_ID': [ids.get((g, c[:3])) for g, c in zip(games['gameId'], code)],
        'GAME_STATUS_ID': games['gameStatus'],
        'GAME_STATUS_TEXT': games['gameStatusText'],
    })


def _game_header(date_str, retries=RETRIES, delay=RETRY_DELAY, sleep=time.sleep, sources=None):
    """
    Game header for a date: ScoreboardV2, retrying failed or non-JSON responses, then ScoreboardV3
    if V2 keeps failing. Raises only if both fail.
    """
    import requests
    v2, v3 = sources or (_v2_header, _v3_header)
    for attempt in range(retries):
        try:
            return v2(date_str)
        except (ValueError, requests.RequestException) as e:     # JSONDecodeError is a ValueError
            if attempt == retries - 1:
                print(f"[!] NBA scoreboard (V2) failed {retries} times ({type(e).__name__}); trying ScoreboardV3")
                break
            wait = delay * 2 ** attempt
            print(f"[!] NBA scoreboard attempt {attempt + 1} failed ({type(e).__name__}); retrying in {wait}s")
            sleep(wait)
    return v3(date_str)


def parse_status_tip(game_date, status_text):
    """'7:30 pm ET' on a date -> naive Eastern timestamp; None for 'Final', 'Q3 5:12', etc."""
    m = re.match(r'\s*(\d{1,2}):(\d{2})\s*([ap])m\s*ET', str(status_text), re.I)
    if not m:
        return None
    hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == 'p' else 0)
    return pd.Timestamp(game_date).normalize() + pd.Timedelta(hours=hour, minutes=int(m.group(2)))


def _cache_path(game_date, cache_dir):
    return os.path.join(cache_dir, f'{pd.Timestamp(game_date):%Y-%m-%d}.csv')


def _read_cache(game_date, cache_dir):
    path = _cache_path(game_date, cache_dir)
    if not os.path.exists(path):
        return None
    c = pd.read_csv(path, dtype={'GAME_ID': str}, parse_dates=['GAME_DATE', 'TIP_TIME_ET'])
    return c.reindex(columns=COLUMNS)


def _write_cache(games, game_date, cache_dir):
    """Save a fresh schedule, keeping tip-off times the scoreboard no longer shows (games under way)."""
    old = _read_cache(game_date, cache_dir)
    out = games.copy()
    if old is not None and len(old):
        known = old.dropna(subset=['TIP_TIME_ET']).set_index('GAME_ID')['TIP_TIME_ET']
        out['TIP_TIME_ET'] = out['TIP_TIME_ET'].fillna(out['GAME_ID'].map(known))
    os.makedirs(cache_dir, exist_ok=True)
    out.to_csv(_cache_path(game_date, cache_dir), index=False)


def _from_cache(cached, now):
    """A cached schedule as of `now` (naive Eastern): scheduled until tip-off, then unknown."""
    c = cached.copy()
    started = c['TIP_TIME_ET'].isna() | (c['TIP_TIME_ET'] <= now)
    c['STATUS'] = c['STATUS'].where(~((c['STATUS'] == 'scheduled') & started), 'unknown')
    return c


def games_on(game_date=None, team_names=None, cache_dir=CACHE_DIR, now=None):
    """
    DataFrame: GAME_ID, GAME_DATE, HOME_TEAM, AWAY_TEAM, STATUS, TIP_TIME_ET (for scheduled games).
    team_names: {TEAM_ID: name}; defaults to nba_api's static list with the box-score spelling
    'LA Clippers'. Falls back to the cached schedule for the date if the scoreboard is unavailable.
    """
    game_date = pd.Timestamp(game_date or date.today()).normalize()
    try:
        header = _game_header(game_date.strftime('%Y-%m-%d')).drop_duplicates('GAME_ID')
    except Exception as e:
        cached = _read_cache(game_date, cache_dir) if cache_dir else None
        if cached is None:
            raise
        now = pd.Timestamp.now(tz=ET).tz_localize(None) if now is None else pd.Timestamp(now)
        print(f"[!] NBA scoreboard unavailable ({type(e).__name__}); using the cached schedule for "
              f"{game_date.date()} ({len(cached)} games)")
        return _from_cache(cached, now)
    if team_names is None:
        from nba_api.stats.static import teams
        team_names = {t['id']: t['full_name'] for t in teams.get_teams()}
        team_names = {k: ('LA Clippers' if v == 'Los Angeles Clippers' else v) for k, v in team_names.items()}
    games = pd.DataFrame({
        'GAME_ID': header['GAME_ID'],
        'GAME_DATE': game_date,
        'HOME_TEAM': header['HOME_TEAM_ID'].map(team_names),
        'AWAY_TEAM': header['VISITOR_TEAM_ID'].map(team_names),
        'STATUS': header['GAME_STATUS_ID'].map(STATUS).fillna('unknown'),
        'TIP_TIME_ET': [parse_status_tip(game_date, t) for t in header['GAME_STATUS_TEXT']],
    }).reset_index(drop=True)
    if cache_dir:
        _write_cache(games, game_date, cache_dir)
    return games


def prefetch(days=1, start=None, cache_dir=CACHE_DIR):
    """
    Fetch and cache the schedule for `days` dates from `start` (default today, Eastern).
    Returns {date: number of games, or the error for a date that failed}.
    """
    start = pd.Timestamp(start or pd.Timestamp.now(tz=ET).date())
    out = {}
    for i in range(days):
        d = start + pd.Timedelta(days=i)
        try:
            out[str(d.date())] = len(_fetch_and_cache(d, cache_dir))
        except Exception as e:
            out[str(d.date())] = f'failed ({type(e).__name__})'
    return out


def _fetch_and_cache(game_date, cache_dir):
    """games_on without the cache fallback, so a prefetch never reports a stale copy as fresh."""
    games = games_on(game_date, cache_dir=None)
    _write_cache(games, pd.Timestamp(game_date).normalize(), cache_dir)
    return games


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefetch', type=int, nargs='?', const=1, default=1,
                    help='number of days to fetch and cache, from today (only today works)')
    args = ap.parse_args()
    result = prefetch(args.prefetch)
    print('Cached schedule (games per date):', result)
    if any(isinstance(v, str) for v in result.values()):
        raise SystemExit(1)
