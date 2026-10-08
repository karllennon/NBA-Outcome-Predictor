"""
Games on a date from stats.nba.com's scoreboard (ScoreboardV2), with tip-off times and status.
"""
import re
import time
from datetime import date
import pandas as pd

STATUS = {1: 'scheduled', 2: 'live', 3: 'final'}
RETRIES = 3            # stats.nba.com sometimes returns an error page instead of JSON for a few minutes
RETRY_DELAY = 10       # seconds before the 2nd attempt, doubling after that


def _game_header(date_str, retries=RETRIES, delay=RETRY_DELAY, sleep=time.sleep):
    """ScoreboardV2 game header for a date, retrying failed or non-JSON responses."""
    import requests
    from nba_api.stats.endpoints import scoreboardv2
    for attempt in range(retries):
        try:
            return scoreboardv2.ScoreboardV2(game_date=date_str, timeout=30).game_header.get_data_frame()
        except (ValueError, requests.RequestException) as e:     # JSONDecodeError is a ValueError
            if attempt == retries - 1:
                raise
            wait = delay * 2 ** attempt
            print(f"[!] NBA scoreboard attempt {attempt + 1} failed ({type(e).__name__}); retrying in {wait}s")
            sleep(wait)


def parse_status_tip(game_date, status_text):
    """'7:30 pm ET' on a date -> naive Eastern timestamp; None for 'Final', 'Q3 5:12', etc."""
    m = re.match(r'\s*(\d{1,2}):(\d{2})\s*([ap])m\s*ET', str(status_text), re.I)
    if not m:
        return None
    hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == 'p' else 0)
    return pd.Timestamp(game_date).normalize() + pd.Timedelta(hours=hour, minutes=int(m.group(2)))


def games_on(game_date=None, team_names=None):
    """
    DataFrame: GAME_ID, GAME_DATE, HOME_TEAM, AWAY_TEAM, STATUS, TIP_TIME_ET (for scheduled games).
    team_names: {TEAM_ID: name}; defaults to nba_api's static list with the box-score spelling
    'LA Clippers'.
    """
    game_date = pd.Timestamp(game_date or date.today()).normalize()
    if team_names is None:
        from nba_api.stats.static import teams
        team_names = {t['id']: t['full_name'] for t in teams.get_teams()}
        team_names = {k: ('LA Clippers' if v == 'Los Angeles Clippers' else v) for k, v in team_names.items()}
    header = _game_header(game_date.strftime('%Y-%m-%d')).drop_duplicates('GAME_ID')
    return pd.DataFrame({
        'GAME_ID': header['GAME_ID'],
        'GAME_DATE': game_date,
        'HOME_TEAM': header['HOME_TEAM_ID'].map(team_names),
        'AWAY_TEAM': header['VISITOR_TEAM_ID'].map(team_names),
        'STATUS': header['GAME_STATUS_ID'].map(STATUS).fillna('unknown'),
        'TIP_TIME_ET': [parse_status_tip(game_date, t) for t in header['GAME_STATUS_TEXT']],
    }).reset_index(drop=True)
