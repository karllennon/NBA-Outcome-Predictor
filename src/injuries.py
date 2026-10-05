"""
Injury impact model shared by training (historical backfill) and prediction.

Both paths call the same methods, so the CORE_INJURY_DIFF feature means the same thing
when the model is trained and when it is used:

    rotation(team, as_of)            -> top-10 players with impact and availability status
    net_loss(rotation, out_players)  -> total impact of the absent rotation players

Who is out: from the official injury report when one is available (the last report
published before tip-off, shared with live predictions via report_out), otherwise the
training-time guess that players who missed the previous game are still out.

Availability status comes from how many of the team's most recent games a player
has missed in a row:
    0 misses      -> 'healthy'
    1-3 misses    -> 'acute'
    4-20 misses   -> 'chronic'
    21+ misses    -> 'inactive' (long-term absence, dropped from the rotation entirely,
                                 since the team's rolling stats already reflect it)

Earlier versions gave acute absences of top-4 players a "replacement boost" (65% of the
player's impact, +25% with a same-position backup). Removing it improved held-out log loss,
so every absence now counts its full impact (RESULTS.md, Phase 3).
"""
import numpy as np
import pandas as pd

ACUTE_MAX = 3
CHRONIC_MAX = 20
RECENT_GAMES = 15        # games used to score each player's impact
ROTATION_SIZE = 10


def impact_score(df):
    """Per-game impact score (same weights as the original identify_core_four)."""
    defensive = df['STL'] * 2.0 + df['BLK'] * 1.5 + df['DREB'] * 0.5
    return df['PTS'] * 1.0 + df['REB'] * 0.4 + df['AST'] * 0.7 + defensive - df['TOV'] * 0.5


def status_from_misses(misses):
    if misses == 0:
        return 'healthy'
    if misses <= ACUTE_MAX:
        return 'acute'
    if misses <= CHRONIC_MAX:
        return 'chronic'
    return 'inactive'


class InjuryModel:
    def __init__(self, player_boxscores, team_games, reports=None, use_reports=False,
                 include_doubtful=False):
        """use_reports / include_doubtful: who is out (see report_out)."""
        players = player_boxscores.copy()
        players['GAME_DATE'] = pd.to_datetime(players['GAME_DATE'])
        players = players[players['MIN'] > 0]
        players['IMPACT'] = impact_score(players)
        players = players.sort_values('GAME_DATE')

        # Per-player arrays: dates, teams, impact (sorted by date) for fast as-of lookups
        self.player_logs = {
            name: (g['GAME_DATE'].values, g['TEAM_NAME'].values, g['IMPACT'].values)
            for name, g in players.groupby('PLAYER_NAME')
        }

        # Players who have appeared for each team, used as rotation candidates
        self.team_players = players.groupby('TEAM_NAME')['PLAYER_NAME'].unique().to_dict()

        # (team, player) -> set of dates the player appeared for that team
        self.played_for_team = {
            key: set(g['GAME_DATE'].values)
            for key, g in players.groupby(['TEAM_NAME', 'PLAYER_NAME'])
        }

        # Each team's full schedule of played games, from the team game log
        tg = team_games.copy()
        tg['GAME_DATE'] = pd.to_datetime(tg['GAME_DATE'])
        self.team_dates = {t: np.sort(g['GAME_DATE'].unique()) for t, g in tg.groupby('TEAM_NAME')}

        self.include_doubtful = include_doubtful
        self.reports = None
        if use_reports and reports is not None and len(reports):
            self.set_reports(reports)

    def set_reports(self, reports):
        """Index injury-report rows by (team, game date), with names mapped to box scores."""
        from injury_reports import NameMatcher, team_key
        matcher = NameMatcher(self.team_players)
        teams = {team_key(t): t for t in self.team_players}
        r = reports.copy()
        r['GAME_DATE'] = pd.to_datetime(r['GAME_DATE'])
        r['REPORT_TIME'] = pd.to_datetime(r['REPORT_TIME'])
        r['TEAM'] = r['TEAM_NAME'].map(lambda t: teams.get(team_key(t)))
        r = r[r['TEAM'].notna()]
        r['BOX_NAME'] = [matcher.match(n, t) for n, t in zip(r['PLAYER_NAME'], r['TEAM'])]
        self.reports = {key: g for key, g in r.groupby(['TEAM', 'GAME_DATE'])}

    def report_out(self, team_name, game_date):
        """
        {box-score name: report status} for players the last report before tip-off lists as
        Out (and Doubtful if include_doubtful). None when no report covers this team-game.
        """
        if self.reports is None:
            return None
        from injury_reports import last_report_before_tip
        rows = self.reports.get((team_name, pd.Timestamp(game_date).normalize()))
        if rows is None:
            return None
        rows = last_report_before_tip(rows, game_date, team_name)
        if rows.empty:
            return None
        out_statuses = ('Out', 'Doubtful') if self.include_doubtful else ('Out',)
        rows = rows[rows['STATUS'].isin(out_statuses) & rows['BOX_NAME'].notna()]
        return dict(zip(rows['BOX_NAME'], rows['STATUS']))

    def rotation(self, team_name, as_of, trade_overrides=None):
        """
        Top-10 rotation for a team using only games strictly before `as_of`.
        Returns a DataFrame sorted by impact: PLAYER_NAME, impact_score, misses, status.
        trade_overrides: {player: new_team} for trades not yet reflected in box scores.
        """
        as_of = np.datetime64(pd.Timestamp(as_of))
        trade_overrides = trade_overrides or {}

        dates = self.team_dates.get(team_name)
        if dates is None:
            return pd.DataFrame(columns=['PLAYER_NAME', 'impact_score', 'misses', 'status'])
        team_dates_before = dates[dates < as_of][::-1]  # most recent first

        candidates = set(self.team_players.get(team_name, []))
        candidates |= {p for p, t in trade_overrides.items() if t == team_name}

        rows = []
        for player in candidates:
            log = self.player_logs.get(player)
            if log is None:
                continue
            p_dates, p_teams, p_impact = log
            n = np.searchsorted(p_dates, as_of)  # games strictly before as_of
            if n == 0:
                continue

            current_team = trade_overrides.get(player, p_teams[n - 1])
            if current_team != team_name:
                continue  # player's most recent team is somewhere else

            impact = p_impact[max(0, n - RECENT_GAMES):n].mean()

            played = self.played_for_team.get((team_name, player), set())
            misses = 0
            for d in team_dates_before:
                if d in played:
                    break
                misses += 1
            if player in trade_overrides and not played:
                misses = 0  # just-traded player: not an absence

            status = status_from_misses(misses)
            if status == 'inactive':
                continue
            rows.append((player, impact, misses, status))

        rot = pd.DataFrame(rows, columns=['PLAYER_NAME', 'impact_score', 'misses', 'status'])
        return rot.sort_values('impact_score', ascending=False).head(ROTATION_SIZE).reset_index(drop=True)

    def net_loss(self, rotation, out_players, verbose=False):
        """
        Total impact of the rotation players who are out.
        out_players: {player_name: status}
        Returns (net_loss, list of (player, status, impact)).
        """
        total, details = 0.0, []
        for row in rotation.itertuples():
            if row.PLAYER_NAME not in out_players:
                continue
            status = out_players[row.PLAYER_NAME]
            total += row.impact_score
            details.append((row.PLAYER_NAME, status, row.impact_score))
            if verbose:
                print(f"    {row.PLAYER_NAME}: {status}, -{row.impact_score:.1f}")
        return total, details

    def out_statuses(self, rotation, listed_out):
        """Classify players listed out (report or manual) as acute/chronic; same rule live and
        in training. Players outside the rotation are ignored."""
        out = {}
        for player in listed_out:
            status = self.live_status(rotation, player)
            if status is not None:
                out[player] = status
        return out

    def historical_loss(self, team_name, game_date):
        """
        Training-time estimate. With an injury report for this game: the players it lists as
        out. Without one: players who missed the team's previous game are assumed out.
        """
        rot = self.rotation(team_name, game_date)
        listed = self.report_out(team_name, game_date)
        if listed is not None:
            out = self.out_statuses(rot, listed)
        else:
            out = {r.PLAYER_NAME: r.status for r in rot.itertuples() if r.status in ('acute', 'chronic')}
        loss, _ = self.net_loss(rot, out)
        return loss

    def live_status(self, rotation, player_name):
        """
        Status for a player listed as out. A player who played the last game is about to
        miss his first, so he counts as acute. None if he is not in the rotation.
        """
        row = rotation[rotation['PLAYER_NAME'] == player_name]
        if row.empty:
            return None
        return status_from_misses(max(int(row.iloc[0]['misses']), 1))

    def backfill(self, game_df, verbose=True):
        """Adds CORE_INJURY_LOSS (that team's net loss) to every team-game row."""
        if verbose:
            print("Backfilling historical injury data...")
        losses = [self.historical_loss(t, d) for t, d in zip(game_df['TEAM_NAME'], game_df['GAME_DATE'])]
        out = game_df.copy()
        out['CORE_INJURY_LOSS'] = losses
        return out
