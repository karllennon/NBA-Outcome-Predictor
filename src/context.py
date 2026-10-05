"""
Schedule, travel and opponent-adjusted form features, computed per team-game from games
BEFORE it. The same function serves training and live predictions: to score an upcoming game,
append it to the team game log as a row without a box score (see upcoming_rows) and call
add_context on the combined log.

Per team-game columns:
    TRAVEL_KM      great-circle distance from the previous game's site (0 after a 7+ day break)
    TZ_SHIFT       time zones crossed since the previous game's site, both sites' UTC offsets
                   taken on this game's date so daylight-saving changes don't count (0 after a break)
    GAMES_LAST_4   games in the 4 days ending on game day, including this one
    GAMES_LAST_7   games in the 7 days ending on game day, including this one
    THREE_IN_FOUR  1 if GAMES_LAST_4 >= 3
    AT_ALTITUDE    1 for a road team playing at Denver or Utah, unless it is Denver or Utah
    ADJ_NET_10     mean over the last 10 games of (game net rating + opponent's pre-game
                   ADJ_NET_10), i.e. net rating adjusted for strength of schedule
"""
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from elo import is_neutral_site

ARENAS_PATH = 'data/arenas.csv'
BREAK_DAYS = 7
ALTITUDE_M = 1000
CONTEXT_COLUMNS = ['TRAVEL_KM', 'TZ_SHIFT', 'GAMES_LAST_4', 'GAMES_LAST_7', 'THREE_IN_FOUR',
                   'AT_ALTITUDE', 'ADJ_NET_10']


def load_arenas(path=ARENAS_PATH):
    return pd.read_csv(path).set_index('TEAM_ABBREVIATION')


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _utc_offset_hours(tz, when):
    return ZoneInfo(tz).utcoffset(pd.Timestamp(when).to_pydatetime().replace(hour=19)).total_seconds() / 3600


def upcoming_rows(game_date, home_abbr, away_abbr, abbr_to_team, team_ids, game_id='UPCOMING'):
    """Two team-game rows (home and away) for a game not yet played: no box score."""
    rows = []
    for abbr, opp, sep in [(home_abbr, away_abbr, ' vs. '), (away_abbr, home_abbr, ' @ ')]:
        rows.append({'GAME_ID': game_id, 'GAME_DATE': pd.Timestamp(game_date),
                     'TEAM_ABBREVIATION': abbr, 'TEAM_NAME': abbr_to_team[abbr],
                     'TEAM_ID': team_ids[abbr], 'MATCHUP': f'{abbr}{sep}{opp}'})
    return pd.DataFrame(rows)


def add_context(team_games, arenas=None):
    """
    team_games: one row per team per game with GAME_ID, GAME_DATE, TEAM_ID, TEAM_ABBREVIATION,
    MATCHUP and, for played games, PTS, PLUS_MINUS, FGA, FTA, OREB, TOV. Rows for games not yet
    played may leave the box-score columns empty. Returns a copy with CONTEXT_COLUMNS added.
    """
    arenas = load_arenas() if arenas is None else arenas
    df = team_games.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['_ORDER'] = np.arange(len(df))

    # Site of each game: the home team's arena (the Orlando bubble for neutral-site games)
    home_abbr = df['MATCHUP'].str.split(r' vs\. | @ ', regex=True).map(
        lambda parts: parts[0] if len(parts) == 2 else None)
    is_home = df['MATCHUP'].str.contains('vs.', regex=False)
    opp_abbr = df['MATCHUP'].str.split(r' vs\. | @ ', regex=True).str[-1]
    site = np.where(is_home, home_abbr, opp_abbr)
    site = np.where(is_neutral_site(df['GAME_DATE']), 'BUBBLE', site)
    df['_SITE'] = site
    df['_LAT'] = df['_SITE'].map(arenas['LAT'])
    df['_LON'] = df['_SITE'].map(arenas['LON'])
    df['_OPP'] = opp_abbr

    df = df.sort_values(['TEAM_ID', 'GAME_DATE', '_ORDER'])
    g = df.groupby('TEAM_ID', sort=False)
    gap = g['GAME_DATE'].diff().dt.days
    fresh = gap.isna() | (gap >= BREAK_DAYS)
    df['TRAVEL_KM'] = np.where(fresh, 0.0, haversine_km(g['_LAT'].shift(), g['_LON'].shift(), df['_LAT'], df['_LON']))
    prev_site = g['_SITE'].shift()

    def tz_shift(site, prev, when):
        if pd.isna(prev) or site not in arenas.index or prev not in arenas.index:
            return np.nan
        return abs(_utc_offset_hours(arenas.at[site, 'TZ'], when) -
                   _utc_offset_hours(arenas.at[prev, 'TZ'], when))
    shift = [tz_shift(s, p, d) for s, p, d in zip(df['_SITE'], prev_site, df['GAME_DATE'])]
    df['TZ_SHIFT'] = np.where(fresh, 0.0, shift)

    # Games in the N days ending on game day, including this one (dates only, no results)
    def count_window(dates, days):
        d = dates.values.astype('datetime64[D]')
        lo = np.searchsorted(d, d - np.timedelta64(days - 1, 'D'), side='left')
        hi = np.searchsorted(d, d, side='right')
        return hi - lo
    df['GAMES_LAST_4'] = g['GAME_DATE'].transform(lambda s: count_window(s, 4))
    df['GAMES_LAST_7'] = g['GAME_DATE'].transform(lambda s: count_window(s, 7))
    df['THREE_IN_FOUR'] = (df['GAMES_LAST_4'] >= 3).astype(int)

    elevation = arenas['ELEVATION_M']
    df['AT_ALTITUDE'] = ((~is_home.reindex(df.index)) &
                         (df['_SITE'].map(elevation).fillna(0) > ALTITUDE_M) &
                         (df['TEAM_ABBREVIATION'].map(elevation).fillna(0) <= ALTITUDE_M)).astype(int)

    df['ADJ_NET_10'] = _adjusted_net(df)
    out = df.sort_values('_ORDER')
    return out.drop(columns=[c for c in out.columns if c.startswith('_')])


def _adjusted_net(df, window=10):
    """Rolling opponent-adjusted net rating, processed in date order so every game uses the
    opponent's value from before that game."""
    poss = df['FGA'] + 0.44 * df['FTA'] - df['OREB'] + df['TOV'] if 'FGA' in df else np.nan
    net = 100 * df['PLUS_MINUS'] / poss
    history = {}            # team -> list of adjusted game nets, oldest first
    rating = {}             # team -> current rolling value (pre-game for the next game)
    out = pd.Series(np.nan, index=df.index)
    order = df.sort_values(['GAME_DATE', '_ORDER']).index
    by_date = pd.Series(order, index=df.loc[order, 'GAME_DATE'].values)
    for _, idx in by_date.groupby(level=0):
        idx = list(idx.values)
        # pre-game values for every game on this date first, then update with results
        for i in idx:
            out.at[i] = rating.get(df.at[i, 'TEAM_ABBREVIATION'], np.nan)
        updates = []
        for i in idx:
            if pd.isna(net.at[i]):
                continue  # not played yet
            opp_pre = rating.get(df.at[i, '_OPP'], 0.0)
            opp_pre = 0.0 if pd.isna(opp_pre) else opp_pre
            updates.append((df.at[i, 'TEAM_ABBREVIATION'], net.at[i] + opp_pre))
        for team, value in updates:
            h = history.setdefault(team, [])
            h.append(value)
            if len(h) >= window:
                rating[team] = float(np.mean(h[-window:]))
    return out
