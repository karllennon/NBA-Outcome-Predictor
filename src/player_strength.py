"""
Player-based team strength (2026-27 build). To be tested only on 2026-27 games: the 2025-26
held-out games have been used for nine experiments already (RESULTS.md).

A team's strength for a game is built from the players expected to play:

    value(player)   box-score impact per minute (injuries.impact_score / MIN), recent games
                    weighted more (weight halves every `half_life` days) and shrunk toward a
                    replacement-level value by `prior_minutes` minutes of replacement play
    minutes(player) his average minutes over his last MINUTES_GAMES appearances
    available       candidates for the team (latest team is this one, and he either played in
                    one of its last RECENT_TEAM_GAMES games or appeared on its injury report in
                    the last REPORT_LOOKBACK_DAYS days) minus players listed Out on the last
                    report before tip-off (without a report: minus players who missed the
                    team's previous game, the same fallback as the injury feature)
    strength        sum over available players of value x minutes, the 240 minutes of a game
                    given out in order of usual minutes (allocate_minutes), so a missing
                    player's minutes go to his teammates at their own values

Trades, injuries and returns change the strength immediately, unlike Elo or the power rating,
which only learn from results. Everything uses box scores and reports from before the game.
PLAYER_STRENGTH_DIFF = (home - away) / 10.
"""
import numpy as np
import pandas as pd

from injuries import impact_score

PARAMS = {'half_life': 120, 'prior_minutes': 3000}  # tune_params on games before 2026-10-20 (RESULTS.md)
REPLACEMENT_QUANTILE = 0.2      # replacement level: 20th percentile of per-minute value (300+ minutes)
MINUTES_GAMES = 10
RECENT_TEAM_GAMES = 10
REPORT_LOOKBACK_DAYS = 30
TEAM_MINUTES = 240.0
FIRST_TEST_DATE = '2026-10-20'


def allocate_minutes(usual, total=TEAM_MINUTES):
    """
    Minutes for a game from each available player's usual minutes: players are filled in order of
    usual minutes (the regulars first) until the game's minutes are used up, so deep-bench players
    get none instead of diluting everyone; if the usual minutes fall short, all are scaled up.
    """
    usual = np.asarray(usual, dtype=float)
    if usual.sum() <= total:
        return usual * (total / usual.sum()) if usual.sum() > 0 else usual
    out = np.zeros_like(usual)
    left = total
    for i in np.argsort(-usual, kind='stable'):
        out[i] = min(usual[i], left)
        left -= out[i]
        if left <= 0:
            break
    return out


class PlayerStrength:
    def __init__(self, player_boxscores, team_games, injury_model=None, half_life=None, prior_minutes=None,
                 replacement=None):
        self.half_life = PARAMS['half_life'] if half_life is None else half_life
        self.prior_minutes = PARAMS['prior_minutes'] if prior_minutes is None else prior_minutes
        p = player_boxscores.copy()
        p['GAME_DATE'] = pd.to_datetime(p['GAME_DATE'])
        p = p[p['MIN'] > 0].sort_values('GAME_DATE')
        p['IMPACT'] = impact_score(p)
        self.logs = {name: (g['GAME_DATE'].values, g['TEAM_NAME'].values, g['MIN'].values.astype(float),
                            g['IMPACT'].values.astype(float))
                     for name, g in p.groupby('PLAYER_NAME')}
        self.team_players = p.groupby('TEAM_NAME')['PLAYER_NAME'].unique().to_dict()
        self.played_for_team = {key: set(g['GAME_DATE'].values) for key, g in p.groupby(['TEAM_NAME', 'PLAYER_NAME'])}
        tg = team_games.copy()
        tg['GAME_DATE'] = pd.to_datetime(tg['GAME_DATE'])
        self.team_dates = {t: np.sort(g['GAME_DATE'].unique()) for t, g in tg.groupby('TEAM_NAME')}
        self.injuries = injury_model
        if replacement is None:
            # one constant from games before the test season (per-minute value of players with 300+ minutes)
            early = p[p['GAME_DATE'] < pd.Timestamp(FIRST_TEST_DATE)].groupby('PLAYER_NAME')[['IMPACT', 'MIN']].sum()
            early = early[early['MIN'] >= 300]
            replacement = float((early['IMPACT'] / early['MIN']).quantile(REPLACEMENT_QUANTILE)) if len(early) else 0.5
        self.replacement = replacement

    # ------------------------------------------------------------ players
    def player(self, name, as_of):
        """(value per minute, expected minutes, latest team) from games strictly before `as_of`; None if none."""
        log = self.logs.get(name)
        if log is None:
            return None
        dates, teams, minutes, impact = log
        n = np.searchsorted(dates, np.datetime64(pd.Timestamp(as_of)))
        if n == 0:
            return None
        days = (np.datetime64(pd.Timestamp(as_of)) - dates[:n]) / np.timedelta64(1, 'D')
        w = 0.5 ** (days / self.half_life)
        value = (np.sum(w * impact[:n]) + self.prior_minutes * self.replacement) / (np.sum(w * minutes[:n]) + self.prior_minutes)
        return value, float(minutes[max(0, n - MINUTES_GAMES):n].mean()), teams[n - 1]

    def _reported(self, team_name, as_of):
        """Players on any archived report for the team in the last REPORT_LOOKBACK_DAYS days."""
        if self.injuries is None or self.injuries.reports is None:
            return set()
        names = set()
        day = pd.Timestamp(as_of).normalize()
        for k in range(1, REPORT_LOOKBACK_DAYS + 1):
            rows = self.injuries.reports.get((team_name, day - pd.Timedelta(days=k)))
            if rows is not None:
                names |= set(rows['BOX_NAME'].dropna())
        return names

    def candidates(self, team_name, as_of):
        """{player: (value, minutes)} for players who may suit up for the team (see module doc)."""
        as_of = pd.Timestamp(as_of)
        dates = self.team_dates.get(team_name)
        if dates is None:
            return {}
        recent = set(dates[dates < np.datetime64(as_of)][-RECENT_TEAM_GAMES:])
        reported = self._reported(team_name, as_of)
        out = {}
        for name in self.team_players.get(team_name, []):
            info = self.player(name, as_of)
            if info is None or info[2] != team_name:
                continue
            played = self.played_for_team.get((team_name, name), set())
            if name in reported or played & recent:
                out[name] = info[:2]
        return out

    # ------------------------------------------------------------ teams
    def out_today(self, team_name, game_date):
        """Players unavailable: listed Out on the last report before tip-off, or, without a report,
        candidates who missed the team's previous game (the injury feature's fallback)."""
        listed = self.injuries.report_out(team_name, game_date) if self.injuries is not None else None
        if listed is not None:
            return set(listed)
        dates = self.team_dates.get(team_name)
        before = dates[dates < np.datetime64(pd.Timestamp(game_date))] if dates is not None else []
        if len(before) == 0:
            return set()
        prev = before[-1]
        return {n for n in self.team_players.get(team_name, [])
                if prev not in self.played_for_team.get((team_name, n), set())}

    def team_strength(self, team_name, game_date, out=None):
        """
        (strength, details) for a team in a game; `out` overrides who is unavailable (live manual
        input). details: DataFrame of available players with value, minutes and contribution.
        """
        cands = self.candidates(team_name, game_date)
        out = self.out_today(team_name, game_date) if out is None else set(out)
        avail = {n: vm for n, vm in cands.items() if n not in out}
        if not avail:
            return 0.0, pd.DataFrame(columns=['PLAYER_NAME', 'value', 'minutes', 'contribution'])
        d = pd.DataFrame([(n, v, m) for n, (v, m) in avail.items()], columns=['PLAYER_NAME', 'value', 'minutes'])
        d['minutes'] = allocate_minutes(d['minutes'].values)
        d['contribution'] = d['value'] * d['minutes']
        return float(d['contribution'].sum()), d.sort_values('contribution', ascending=False).reset_index(drop=True)

    def backfill(self, game_df):
        """Adds PLAYER_STRENGTH to every team-game row."""
        out = game_df.copy()
        out['PLAYER_STRENGTH'] = [self.team_strength(t, d)[0] for t, d in zip(game_df['TEAM_NAME'], game_df['GAME_DATE'])]
        return out


def tune_params(player_boxscores, team_games, injury_model, before=FIRST_TEST_DATE,
                half_lives=(60, 120, 240), prior_minutes=(100, 300, 1000, 3000, 10000), burn_in_days=60):
    """
    Settings whose PLAYER_STRENGTH difference best predicts home margins (least squares with home
    court) on games before `before` only. Returns (best params, {params: mean squared error}).
    """
    from power_rating import pair_games
    games = pair_games(team_games)
    start = games['GAME_DATE'].min() + pd.Timedelta(days=burn_in_days)
    games = games[(games['GAME_DATE'] >= start) & (games['GAME_DATE'] < pd.Timestamp(before))]
    names = team_games.drop_duplicates('TEAM_ID').set_index('TEAM_ID')['TEAM_NAME']
    scores = {}
    for h in half_lives:
        for k in prior_minutes:
            ps = PlayerStrength(player_boxscores, team_games, injury_model, half_life=h, prior_minutes=k)
            diff = np.array([ps.team_strength(names[r.HOME_ID], r.GAME_DATE)[0] - ps.team_strength(names[r.AWAY_ID], r.GAME_DATE)[0]
                             for r in games.itertuples()])
            X = np.column_stack([np.ones(len(diff)), diff])
            coef, *_ = np.linalg.lstsq(X, games['MARGIN'].values, rcond=None)
            scores[(h, k)] = float(np.mean((X @ coef - games['MARGIN'].values) ** 2))
    best = min(scores, key=scores.get)
    return {'half_life': best[0], 'prior_minutes': best[1]}, scores
