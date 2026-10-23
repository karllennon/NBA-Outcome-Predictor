"""
Daily snapshots of every team's current roster from stats.nba.com (CommonTeamRoster), archived
as data/rosters/YYYY-MM-DD.csv.

The NBA serves only today's roster, so a roster for a past date exists only if a snapshot was
taken that day; the archive starts on 2026-10-14. Rosters are therefore used only for games on or
after a snapshot (player_strength.py, the dashboard's what-if view), never to rebuild older seasons,
where they would reveal moves that had not happened yet. A snapshot taken on the morning of a game
day is pre-game information (the earliest regular-season tip-off is about noon Eastern).

    python src/rosters.py        # snapshot all 30 rosters for today (skips if already taken)
"""
import argparse
import glob
import os
import time
import pandas as pd

ARCHIVE_DIR = 'data/rosters'
COLUMNS = ['SNAPSHOT_DATE', 'TEAM_ID', 'PLAYER_ID', 'PLAYER', 'POSITION', 'HOW_ACQUIRED']
MAX_AGE_DAYS = 3          # an older snapshot is not trusted for a game (roster may have changed)
ET = 'America/New_York'


def season_for(day):
    """'2026-27' for any date from September 2026 to August 2027."""
    day = pd.Timestamp(day)
    start = day.year if day.month >= 9 else day.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def fetch_team(team_id, season, retries=3, delay=10, sleep=time.sleep):
    """One team's roster, retrying error pages (stats.nba.com sometimes returns HTML instead of JSON)."""
    import requests
    from nba_api.stats.endpoints import commonteamroster
    for attempt in range(retries):
        try:
            return commonteamroster.CommonTeamRoster(team_id=team_id, season=season, timeout=30).get_data_frames()[0]
        except (ValueError, requests.RequestException):
            if attempt == retries - 1:
                raise
            sleep(delay * 2 ** attempt)


def snapshot(day=None, archive_dir=ARCHIVE_DIR, pause=1.0, force=False, fetch=fetch_team):
    """
    Save today's roster for every team. Returns the snapshot DataFrame (or the existing one if
    today's snapshot was already taken). Raises if any team cannot be fetched, so a partial
    snapshot is never saved.
    """
    from nba_api.stats.static import teams
    day = pd.Timestamp(day or pd.Timestamp.now(tz=ET).date()).normalize()
    path = os.path.join(archive_dir, f'{day:%Y-%m-%d}.csv')
    if os.path.exists(path) and not force:
        return pd.read_csv(path)
    frames = []
    for t in teams.get_teams():
        r = fetch(t['id'], season_for(day))
        frames.append(pd.DataFrame({'SNAPSHOT_DATE': day.date(), 'TEAM_ID': t['id'], 'PLAYER_ID': r['PLAYER_ID'],
                                    'PLAYER': r['PLAYER'], 'POSITION': r['POSITION'],
                                    'HOW_ACQUIRED': r.get('HOW_ACQUIRED')}))
        time.sleep(pause)
    out = pd.concat(frames, ignore_index=True)[COLUMNS]
    os.makedirs(archive_dir, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def load_archive(archive_dir=ARCHIVE_DIR):
    """All snapshots as one DataFrame (empty if none)."""
    files = sorted(glob.glob(os.path.join(archive_dir, '*.csv')))
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out['SNAPSHOT_DATE'] = pd.to_datetime(out['SNAPSHOT_DATE'])
    return out


class Rosters:
    """Look up the roster in force for a team on a date, from the archive."""

    def __init__(self, archive=None, max_age_days=MAX_AGE_DAYS):
        archive = load_archive() if archive is None else archive
        self.max_age = pd.Timedelta(days=max_age_days)
        self.dates = sorted(pd.to_datetime(archive['SNAPSHOT_DATE']).unique()) if len(archive) else []
        self.by = {(pd.Timestamp(d), int(t)): set(g['PLAYER_ID'].astype(int))
                   for (d, t), g in archive.groupby(['SNAPSHOT_DATE', 'TEAM_ID'])} if len(archive) else {}

    def on(self, team_id, day):
        """PLAYER_IDs on the team's roster in the latest snapshot taken on or before `day` (at most
        max_age_days old), or None when there is no usable snapshot."""
        day = pd.Timestamp(day).normalize()
        usable = [d for d in self.dates if day - self.max_age <= d <= day]
        if not usable or team_id is None:
            return None
        return self.by.get((pd.Timestamp(usable[-1]), int(team_id)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='re-take today\'s snapshot')
    args = ap.parse_args()
    snap = snapshot(force=args.force)
    print(f"Roster snapshot: {len(snap)} players on {snap['TEAM_ID'].nunique()} teams -> {ARCHIVE_DIR}")
