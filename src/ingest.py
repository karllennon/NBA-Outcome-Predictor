"""
Downloads NBA regular-season team and player game logs from stats.nba.com.

Incremental by default: re-fetches only the current season (from today's date) and the
latest season already on disk (so a season that ended since the last refresh is completed),
then merges with the existing CSVs. Rows are deduplicated on GAME_ID + TEAM_ID (team logs
have one row per team per game) and GAME_ID + PLAYER_ID (player logs).

    python src/ingest.py                   # incremental refresh
    python src/ingest.py --since 2018-19   # full rebuild from 2018-19 to the current season

Exits with a non-zero status if any season could not be fetched, so schedulers and CI fail
loudly instead of quietly keeping stale data.
"""
import argparse
import os
import sys
import time
from datetime import date
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

TEAM_PATH = 'data/raw_nba_data.csv'
PLAYER_PATH = 'data/raw_player_boxscores.csv'
TEAM_KEY = ['GAME_ID', 'TEAM_ID']
PLAYER_KEY = ['GAME_ID', 'PLAYER_ID']
DEFAULT_FIRST_SEASON = '2023-24'

# Use nba_api's maintained default headers. The hand-written browser headers this file
# used before (x-nba-stats-token, Origin, etc.) now make stats.nba.com hang until timeout.


def season_label(start_year):
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def season_start_year(label):
    return int(label[:4])


def current_season(today=None):
    """Seasons start in October; from August on, the upcoming season is the current one."""
    today = today or date.today()
    return season_label(today.year if today.month >= 8 else today.year - 1)


def seasons_between(first, last):
    return [season_label(y) for y in range(season_start_year(first), season_start_year(last) + 1)]


def fetch_log(season, kind, retries=4, timeout=60, base_delay=10):
    """kind 'T' (team) or 'P' (player). Retries with exponential backoff; raises on final failure."""
    for attempt in range(retries):
        try:
            return leaguegamelog.LeagueGameLog(
                season=season, season_type_all_star='Regular Season',
                player_or_team_abbreviation=kind, timeout=timeout,
            ).get_data_frames()[0]
        except Exception as e:
            if attempt == retries - 1:
                raise
            delay = base_delay * 2 ** attempt
            print(f"  [!] {season} {kind} attempt {attempt + 1} failed ({type(e).__name__}: {e}); "
                  f"retrying in {delay}s")
            time.sleep(delay)


def normalize(df):
    df = df.copy()
    df['GAME_ID'] = df['GAME_ID'].astype(str).str.zfill(10)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    return df


def merge(existing, new, key):
    """New rows replace existing rows with the same key."""
    if existing is None or existing.empty:
        combined = new
    else:
        combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=key, keep='last')
    return combined.sort_values(['GAME_DATE'] + key).reset_index(drop=True)


def load_existing(path):
    if not os.path.exists(path):
        return None
    return normalize(pd.read_csv(path, dtype={'GAME_ID': str}))


def seasons_to_fetch(existing_team, since=None, today=None):
    current = current_season(today)
    if since:
        return seasons_between(since, current)
    if existing_team is None or existing_team.empty:
        return seasons_between(DEFAULT_FIRST_SEASON, current)
    # SEASON_ID looks like 22025 (2 = regular season, then the start year)
    latest_on_disk = season_label(int(str(existing_team['SEASON_ID'].max())[1:]))
    return sorted({latest_on_disk, current})


def run(since=None, pause=5):
    os.makedirs('data', exist_ok=True)
    team = load_existing(TEAM_PATH)
    player = load_existing(PLAYER_PATH)
    seasons = seasons_to_fetch(team, since)
    print(f"Fetching seasons: {', '.join(seasons)}")

    failed, changed = [], False
    for season in seasons:
        try:
            print(f"--- {season} ---")
            t = fetch_log(season, 'T')
            time.sleep(pause)
            p = fetch_log(season, 'P')
            print(f"  team rows {len(t)}, player rows {len(p)}")
        except Exception as e:
            print(f"  [X] {season} failed after retries: {type(e).__name__}: {e}")
            failed.append(season)
            continue
        if not t.empty:
            team = merge(team, normalize(t), TEAM_KEY)
            changed = True
        if not p.empty:
            player = merge(player, normalize(p), PLAYER_KEY)
            changed = True
        time.sleep(pause)

    if changed:
        team.to_csv(TEAM_PATH, index=False, date_format='%Y-%m-%d')
        player.to_csv(PLAYER_PATH, index=False, date_format='%Y-%m-%d')
        print(f"\nTeam rows: {len(team)} | player rows: {len(player)} | "
              f"dates {team['GAME_DATE'].min().date()} to {team['GAME_DATE'].max().date()}")

    if failed:
        print(f"\nFAILED seasons: {', '.join(failed)} (stats.nba.com unreachable or blocked?)")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--since', help="full rebuild from this season, e.g. 2018-19")
    run(parser.parse_args().since)
