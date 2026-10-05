"""
Official NBA injury reports (the PDFs linked from official.nba.com/nba-injury-report-*).

URL pattern, verified against the live server in October 2026:
    https://ak-static.cms.nba.com/referee/injury/Injury-Report_<YYYY-MM-DD>_<time>.pdf
    - from 2025-12-22: <time> = HH_MMAM/PM, published every 15 minutes (e.g. 05_15PM)
    - before that:     <time> = HHAM/PM, hourly; the file named 05PM holds the 5:30 PM report
All times are US Eastern. Reports exist back to at least the 2021-22 season.

Parsed rows: REPORT_TIME, GAME_DATE, GAME_TIME, MATCHUP, TEAM_NAME, PLAYER_NAME (as printed,
"Last, First"), PLAYER (first-last, box-score order), STATUS, REASON.
Parsed reports are archived in data/injury_reports/<YYYY-MM-DD_HHMM>.csv.gz.

    python src/injury_reports.py              # fetch + archive the latest report
    python src/injury_reports.py --backfill   # archive the last pre-tip-off report(s) for
                                              # every game day in data/raw_nba_data.csv
"""
import argparse
import io
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import requests

BASE_URL = 'https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_{time}.pdf'
NEW_FORMAT_FROM = pd.Timestamp('2025-12-22')
ET = ZoneInfo('America/New_York')
ARCHIVE_DIR = 'data/injury_reports'
STATUSES = ('Out', 'Doubtful', 'Questionable', 'Probable', 'Available')
COLUMNS = ['REPORT_TIME', 'GAME_DATE', 'GAME_TIME', 'MATCHUP', 'TEAM_NAME',
           'PLAYER_NAME', 'PLAYER', 'STATUS', 'REASON']
HEADERS = {'User-Agent': 'nba-outcome-predictor (personal research project)'}

# First word of each header cell -> report column. The two 'Game' cells are date then time.
# Reports before ~2021-22 add 'Category' (merged into REASON) and 'Previous Status' (ignored),
# and put Reason before Current Status, so columns are matched by name, not position.
HEADER_COLUMNS = {'Game': ['GAME_DATE', 'GAME_TIME'], 'Matchup': ['MATCHUP'], 'Team': ['TEAM_NAME'],
                  'Player': ['PLAYER_NAME'], 'Category': ['REASON'], 'Reason': ['REASON'],
                  'Current': ['STATUS'], 'Previous': ['PREVIOUS_STATUS']}
REQUIRED_COLUMNS = {'GAME_DATE', 'GAME_TIME', 'MATCHUP', 'TEAM_NAME', 'PLAYER_NAME', 'STATUS', 'REASON'}

SUFFIXES = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}


# ---------------------------------------------------------------- names

def normalize_name(name):
    """Comparable key: no accents, punctuation, case or suffixes. 'Dončić' == 'Doncic'."""
    if not isinstance(name, str):
        return ''
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()
    name = re.sub(r"[.'`’]", '', name.lower())
    name = re.sub(r'[-,]', ' ', name)
    tokens = [t for t in name.split() if t not in SUFFIXES]
    return ' '.join(tokens)


def report_to_first_last(name):
    """'Bagley III, Marvin' -> 'Marvin Bagley III'. Names without a comma are kept."""
    if ',' not in name:
        return name.strip()
    last, first = name.split(',', 1)
    return f"{first.strip()} {last.strip()}".strip()


def team_key(team):
    return re.sub(r'[^a-z0-9]', '', str(team).lower())


class NameMatcher:
    """Maps report names to box-score names, optionally restricted to a team's players."""

    def __init__(self, box_names_by_team):
        """box_names_by_team: {TEAM_NAME: iterable of box-score PLAYER_NAMEs}"""
        self.by_team = {}
        for team, names in box_names_by_team.items():
            self.by_team[team_key(team)] = {normalize_name(n): n for n in names}

    def match(self, report_name, team):
        roster = self.by_team.get(team_key(team), {})
        key = normalize_name(report_to_first_last(report_name))
        if key in roster:
            return roster[key]
        # Fallback: same last name and first initial, if unique on the team (Nic/Nicolas)
        parts = key.split()
        if len(parts) >= 2:
            hits = [n for k, n in roster.items()
                    if k.split()[-1:] == parts[-1:] and k[:1] == parts[0][:1]]
            if len(hits) == 1:
                return hits[0]
        return None


# ---------------------------------------------------------------- URLs

def _naive_et(ts):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(ET).tz_localize(None)
    return ts


def report_url(ts):
    """URL of the report published at `ts` (Eastern). Old-format files are named by the hour."""
    ts = _naive_et(ts)
    if ts.normalize() >= NEW_FORMAT_FROM:
        t = ts.strftime('%I_%M%p')
    else:
        t = ts.floor('h').strftime('%I%p')
    return BASE_URL.format(date=ts.strftime('%Y-%m-%d'), time=t)


def report_slots_before(ts, max_hours=36):
    """Publication times at or before `ts`, newest first: every 15 minutes in the new
    format, HH:30 in the old hourly format."""
    ts = _naive_et(ts)
    cur, end, slots = ts.floor('15min'), ts - timedelta(hours=max_hours), []
    while cur > end:
        if cur.normalize() >= NEW_FORMAT_FROM or cur.minute == 30:
            slots.append(cur)
        cur -= timedelta(minutes=15)
    return slots


def download(url, session=None, retries=3):
    session = session or requests
    for attempt in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200 and r.content[:4] == b'%PDF':
                return r.content
            if r.status_code in (403, 404):
                return None  # not published
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------- parsing

def _column_bounds(words):
    """x positions where each column starts, read from the header row."""
    header_top = next(w['top'] for w in words if w['text'] == 'Matchup')
    header = sorted([w for w in words if abs(w['top'] - header_top) < 2], key=lambda w: w['x0'])
    starts, used = [], {}
    for w in header:
        names = HEADER_COLUMNS.get(w['text'])
        if names:
            k = used.get(w['text'], 0)
            if k < len(names):
                starts.append((w['x0'] - 2, names[k]))
                used[w['text']] = k + 1
    if not REQUIRED_COLUMNS <= {name for _, name in starts}:
        raise ValueError(f"Unexpected header: {[w['text'] for w in header]}")
    return header_top, starts


def _column_of(x, starts):
    col = starts[0][1]
    for start, name in starts:
        if x >= start:
            col = name
    return col


def _lines(words, header_top, starts):
    """Group words below the header into text lines, each a {column: text} dict."""
    footer_tops = [w['top'] for w in words if w['text'].startswith('Page')]
    body = [w for w in words if w['top'] > header_top + 5
            and not any(abs(w['top'] - t) < 2 for t in footer_tops)]
    body.sort(key=lambda w: (round(w['top']), w['x0']))
    lines = []
    for w in body:
        if lines and abs(lines[-1]['top'] - w['top']) < 3:
            line = lines[-1]
        else:
            line = {'top': w['top'], 'cells': {}}
            lines.append(line)
        col = _column_of(w['x0'], starts)
        line['cells'][col] = (line['cells'].get(col, '') + ' ' + w['text']).strip()
    return lines


_TEAMS_BY_ABBR = None


def _resolve_team(fragment, matchup):
    """
    Full team name (box-score spelling) for a team cell. Older reports wrap long names over two
    lines ('Minnesota' / 'Timberwolves'), so a fragment is matched to whichever of the two teams
    in the matchup (e.g. 'MIN@PHI') contains it. Unresolvable text is returned unchanged.
    """
    global _TEAMS_BY_ABBR
    if _TEAMS_BY_ABBR is None:
        from nba_api.stats.static import teams
        _TEAMS_BY_ABBR = {t['abbreviation']: t['full_name'].replace('Los Angeles Clippers', 'LA Clippers')
                          for t in teams.get_teams()}
    fragment = re.sub(r'\s+', ' ', fragment or '').strip()
    candidates = [_TEAMS_BY_ABBR.get(a.strip()) for a in str(matchup).split('@')]
    candidates = [c for c in candidates if c]
    key = team_key(fragment)
    if not key:
        return fragment
    hits = [c for c in candidates if key in team_key(c) or team_key(c) in key
            or (key == 'laclippers' and c == 'LA Clippers')]
    return hits[0] if len(hits) == 1 else fragment


def _clean(text):
    return re.sub(r'\s*-\s*', '-', re.sub(r'\s+', ' ', text or '')).strip()


def parse_report(pdf_bytes):
    """Parse one report PDF into a DataFrame with COLUMNS."""
    import pdfplumber
    rows = []
    report_time, starts = None, None
    ctx = {'GAME_DATE': None, 'GAME_TIME': None, 'MATCHUP': None, 'TEAM_NAME': None}

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=1.5)
            if report_time is None:
                text = ' '.join(w['text'] for w in words[:12])
                m = re.search(r'Injury Report:\s*(\d\d/\d\d/\d\d)\s+(\d\d:\d\d)\s*(AM|PM)', text)
                if m:
                    report_time = datetime.strptime(' '.join(m.groups()), '%m/%d/%y %I:%M %p')
            if any(w['text'] == 'Matchup' for w in words):
                header_top, starts = _column_bounds(words)
            elif starts is None:
                continue
            else:
                # Continuation pages repeat the title but not the column header
                title = [w['top'] for w in words if w['text'] == 'Report:']
                header_top = (title[0] if title else 0) + 10
            lines = _lines(words, header_top, starts)

            # Anchor lines carry a status; other lines are wrapped reason text or context
            anchors = []
            for line in lines:
                cells = line['cells']
                for key in ctx:
                    if cells.get(key):
                        if key == 'GAME_DATE':
                            ctx['GAME_DATE'] = cells[key]
                        elif key == 'GAME_TIME':
                            ctx['GAME_TIME'] = cells[key].replace(' ', '')
                        else:
                            ctx[key] = cells[key]
                status = cells.get('STATUS', '')
                if status in STATUSES and cells.get('PLAYER_NAME'):
                    anchors.append({'top': line['top'], **ctx,
                                    'PLAYER_NAME': cells['PLAYER_NAME'], 'STATUS': status,
                                    'REASON': cells.get('REASON', '')})
                    line['anchor'] = anchors[-1]
            # Attach wrapped reason fragments to the closest anchor, keeping vertical order
            for line in lines:
                if 'anchor' in line or not line['cells'].get('REASON'):
                    continue
                if 'NOT YET SUBMITTED' in line['cells']['REASON'].upper():
                    continue
                if not anchors:
                    continue
                nearest = min(anchors, key=lambda a: abs(a['top'] - line['top']))
                if abs(nearest['top'] - line['top']) > 15:
                    continue
                if line['top'] < nearest['top']:
                    nearest['REASON'] = line['cells']['REASON'] + ' ' + nearest['REASON']
                else:
                    nearest['REASON'] = nearest['REASON'] + ' ' + line['cells']['REASON']
            rows.extend(anchors)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df['REPORT_TIME'] = report_time
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='%m/%d/%Y', errors='coerce')
    df['PLAYER_NAME'] = df['PLAYER_NAME'].map(_clean)
    df['PLAYER'] = df['PLAYER_NAME'].map(report_to_first_last)
    df['REASON'] = df['REASON'].map(_clean)
    df['TEAM_NAME'] = [_resolve_team(t, m) for t, m in zip(df['TEAM_NAME'], df['MATCHUP'])]
    return df[COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------- archive

def archive_path(report_time):
    return os.path.join(ARCHIVE_DIR, pd.Timestamp(report_time).strftime('%Y-%m-%d_%H%M') + '.csv.gz')


def save(df, slot):
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    report_time = df['REPORT_TIME'].iloc[0] if len(df) and pd.notna(df['REPORT_TIME'].iloc[0]) else slot
    path = archive_path(report_time)
    df.to_csv(path, index=False)
    return path


def fetch_report(slot, session=None):
    """Download and parse the report for one publication slot; None if not published."""
    pdf = download(report_url(slot), session)
    if pdf is None:
        return None
    df = parse_report(pdf)
    if df['REPORT_TIME'].isna().all():
        df['REPORT_TIME'] = pd.Timestamp(slot)
    return df


def _published(slot, session):
    try:
        return session.head(report_url(slot), headers=HEADERS, timeout=15).status_code == 200
    except requests.RequestException:
        return False


def fetch_latest(as_of=None, session=None, max_hours=12, pause=0.25):
    """
    Most recent report at or before `as_of` (default now, Eastern). Archives it.
    Gentle on the server: probes one slot per hour (newest first), then the 15-minute slots
    after the newest hit, with a pause between requests. Hammering the CDN gets this machine
    temporarily refused (HTTP 403 on every report), which happened once during development.
    """
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz=ET)
    session = session or requests.Session()
    slots = report_slots_before(as_of, max_hours)
    hourly = [s for s in slots if s.minute == (30 if s.normalize() < NEW_FORMAT_FROM else 0)]
    hit = None
    for slot in hourly:
        if _published(slot, session):
            hit = slot
            break
        time.sleep(pause)
    if hit is None:
        return None
    newer = [s for s in slots if hit < s]          # 15-minute slots after the hourly hit
    for slot in sorted(newer, reverse=True):
        if _published(slot, session):
            hit = slot
            break
        time.sleep(pause)
    df = fetch_report(hit, session)
    if df is not None:
        save(df, hit)
    return df


def load_latest_archived():
    """The newest archived report only (cheap; file names sort by report time)."""
    if not os.path.isdir(ARCHIVE_DIR):
        return pd.DataFrame(columns=COLUMNS)
    files = sorted(f for f in os.listdir(ARCHIVE_DIR) if f.endswith('.csv.gz'))
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(os.path.join(ARCHIVE_DIR, files[-1]), parse_dates=['REPORT_TIME', 'GAME_DATE'])


def load_archive():
    """All archived reports as one DataFrame."""
    if not os.path.isdir(ARCHIVE_DIR):
        return pd.DataFrame(columns=COLUMNS)
    frames = [pd.read_csv(os.path.join(ARCHIVE_DIR, f), parse_dates=['REPORT_TIME', 'GAME_DATE'])
              for f in sorted(os.listdir(ARCHIVE_DIR)) if f.endswith('.csv.gz')]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True)


def parse_tip_time(game_date, game_time):
    """'07:30(ET)' on a date -> naive Eastern timestamp."""
    m = re.match(r'(\d\d):(\d\d)', str(game_time))
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2))
    # Times are printed on a 12-hour clock without AM/PM; NBA games tip between 11am and 11pm
    if hour < 11:
        hour += 12
    return pd.Timestamp(game_date) + timedelta(hours=hour, minutes=minute)


def last_report_before_tip(reports, game_date, team, minutes_before=30):
    """
    Rows for `team` from the latest archived report published at least `minutes_before`
    minutes before that team's tip-off on `game_date`. Empty if there is none.
    """
    day = reports[(reports['GAME_DATE'] == pd.Timestamp(game_date)) &
                  (reports['TEAM_NAME'].map(team_key) == team_key(team))]
    if day.empty:
        return day
    tip = parse_tip_time(game_date, day['GAME_TIME'].iloc[0])
    if tip is None:
        return day.iloc[0:0]
    cutoff = tip - timedelta(minutes=minutes_before)
    eligible = day[day['REPORT_TIME'] <= cutoff]
    if eligible.empty:
        return eligible
    return eligible[eligible['REPORT_TIME'] == eligible['REPORT_TIME'].max()]


# ---------------------------------------------------------------- backfill

MAX_MISSING_STREAK = 5


def backfill(game_dates, session=None, pause=1.0, minutes_before=30):
    """
    For each game date: read the 5 PM-ish report to learn the tip times, then archive the last
    report published `minutes_before` minutes before each distinct tip time. Skips days already
    archived, so it can be re-run to resume. Stops after MAX_MISSING_STREAK game days in a row
    with no report, which in practice means the server is refusing requests (HTTP 403).
    """
    session = session or requests.Session()
    existing = {f[:-7] for f in os.listdir(ARCHIVE_DIR) if f.endswith('.csv.gz')} if os.path.isdir(ARCHIVE_DIR) else set()
    done_days = {e[:10] for e in existing}
    missing, streak = [], 0
    for i, day in enumerate(sorted(set(pd.to_datetime(pd.Series(game_dates)).dt.normalize()))):
        if day.strftime('%Y-%m-%d') in done_days:
            continue
        # A mid-afternoon report lists every game that day with its tip time
        first = None
        for slot in report_slots_before(day + timedelta(hours=17, minutes=45), max_hours=8):
            first = fetch_report(slot, session)
            if first is not None and not first.empty:
                break
            time.sleep(pause)
        if first is None or first.empty:
            missing.append(day.date())
            streak += 1
            if streak >= MAX_MISSING_STREAK:
                print(f"  Stopping at {day.date()}: no report for {streak} game days in a row "
                      "(server refusing requests?). Re-run later to resume.", flush=True)
                break
            continue
        streak = 0
        save(first, first['REPORT_TIME'].iloc[0])
        tips = {parse_tip_time(day, t) for t in first['GAME_TIME'].dropna().unique()}
        tips = sorted(t for t in tips if t is not None)
        first_time = pd.Timestamp(first['REPORT_TIME'].iloc[0])
        fetched = {first_time}
        for tip in tips:
            for slot in report_slots_before(tip - timedelta(minutes=minutes_before), max_hours=6):
                if slot <= first_time or slot in fetched:
                    break  # a report we already have is the latest one before this tip
                df = fetch_report(slot, session)
                time.sleep(pause)
                if df is not None and not df.empty:
                    save(df, slot)
                    fetched.add(slot)
                    break
        if i % 25 == 0:
            print(f"  {day.date()}: {len(tips)} tip times, {len(fetched)} reports", flush=True)
    return missing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backfill', action='store_true')
    parser.add_argument('--since', default=None, help='backfill start date, e.g. 2023-10-24')
    args = parser.parse_args()
    if args.backfill:
        games = pd.read_csv('data/raw_nba_data.csv', parse_dates=['GAME_DATE'])
        dates = games['GAME_DATE']
        if args.since:
            dates = dates[dates >= pd.Timestamp(args.since)]
        missing = backfill(dates)
        print(f"Backfill done. Days with no report found: {len(missing)}")
        for d in missing[:20]:
            print(f"  {d}")
    else:
        df = fetch_latest()
        if df is None:
            print("No injury report found in the last 36 hours (offseason?).")
            sys.exit(0)
        print(f"Report {df['REPORT_TIME'].iloc[0]}: {len(df)} rows -> {archive_path(df['REPORT_TIME'].iloc[0])}")
        print(df['STATUS'].value_counts().to_string())
