"""
Builds the feature row for an upcoming game. Shared by predict.py and app.py so the
CLI and the dashboard always produce the same prediction, and so live features are
computed the same way as the training features.
"""
from datetime import date
import joblib
import pandas as pd
from data_pipeline import INJURY_PARAMS
from injuries import InjuryModel
from matchups import FEATURES, injury_diff

OFFSEASON_GAP_DAYS = 90       # longer gap than this means a new season has started
SEASON_CARRYOVER = 0.75       # same regression the Elo calculator applies between seasons
MEAN_ELO = 1505


class GamePredictor:
    def __init__(self, data_dir='data', model_path='models/nba_model.joblib'):
        self.model = joblib.load(model_path)
        self.state = pd.read_csv(f'{data_dir}/team_state.csv', parse_dates=['LAST_GAME_DATE'])
        team_games = pd.read_csv(f'{data_dir}/raw_nba_data.csv')
        players = pd.read_csv(f'{data_dir}/raw_player_boxscores.csv', low_memory=False)
        players.columns = players.columns.str.strip()
        # Same settings as the training pipeline, so the injury feature means the same thing
        params = {k: v for k, v in INJURY_PARAMS.items() if k != 'use_reports'}
        self.injuries = InjuryModel(players, team_games, **params)
        self.report, self.report_error = None, None
        try:
            trades = pd.read_csv(f'{data_dir}/recent_trades.csv')
            self.trades = dict(zip(trades['PLAYER_NAME'], trades['NEW_TEAM']))
        except (FileNotFoundError, KeyError):
            self.trades = {}

    def load_injury_report(self, fetch=True):
        """Latest official injury report (downloaded and archived, or the newest archived one)."""
        from injury_reports import fetch_latest, load_archive
        report = None
        if fetch:
            try:
                report = fetch_latest()
            except Exception as e:  # network down, format changed: fall back to manual input
                self.report_error = f"{type(e).__name__}: {e}"
        if report is None:
            archive = load_archive()
            if len(archive):
                report = archive[archive['REPORT_TIME'] == archive['REPORT_TIME'].max()]
        if report is not None and len(report):
            self.report = report
            self.injuries.set_reports(report)
        return self.report

    def report_out(self, team_name, game_date=None):
        """Players the latest pre-tip-off report lists as out for this team's game today
        (box-score names, top-10 rotation or not). None when no report covers the game."""
        game_date = pd.Timestamp(game_date or date.today())
        listed = self.injuries.report_out(team_name, game_date)
        return None if listed is None else sorted(listed)

    def teams(self):
        return sorted(self.state['TEAM_NAME'])

    def rotation(self, team_name, game_date=None):
        game_date = pd.Timestamp(game_date or date.today())
        return self.injuries.rotation(team_name, game_date, trade_overrides=self.trades)

    def _team_row(self, team_name, game_date):
        rows = self.state[self.state['TEAM_NAME'] == team_name]
        if rows.empty:
            raise ValueError(f"No stats found for '{team_name}'. Check spelling.")
        row = rows.iloc[0].copy()

        days_since = (game_date - row['LAST_GAME_DATE']).days
        row['DAYS_REST'] = min(days_since, 7)
        row['IS_B2B'] = int(days_since == 1)
        row['ELO'] = row['CURRENT_ELO']
        if days_since > OFFSEASON_GAP_DAYS:
            row['ELO'] = SEASON_CARRYOVER * row['ELO'] + (1 - SEASON_CARRYOVER) * MEAN_ELO
        row['DAYS_SINCE_LAST_GAME'] = days_since
        return row

    def _injury_loss(self, team_name, rotation, out_players):
        out = {}
        skipped = []
        for player in out_players:
            status = self.injuries.live_status(rotation, player)
            if status is None:
                skipped.append(player)  # not in the current top-10 rotation
            else:
                out[player] = status
        loss, details = self.injuries.net_loss(rotation, out)
        return loss, details, skipped

    def predict(self, home_team, away_team, home_out=None, away_out=None,
                game_date=None):
        """home_out / away_out: players out. None = take them from the injury report
        (load_injury_report() first); a list is a manual override."""
        game_date = pd.Timestamp(game_date or date.today())
        sources = {}
        if home_out is None:
            home_out = self.report_out(home_team, game_date)
            sources['home'] = 'report' if home_out is not None else 'none'
        if away_out is None:
            away_out = self.report_out(away_team, game_date)
            sources['away'] = 'report' if away_out is not None else 'none'
        home_out, away_out = home_out or [], away_out or []
        home, away = self._team_row(home_team, game_date), self._team_row(away_team, game_date)

        home_rot, away_rot = self.rotation(home_team, game_date), self.rotation(away_team, game_date)
        # Report lists include two-way and deep-bench players; only rotation players count
        if sources.get('home') == 'report':
            home_out = [p for p in home_out if p in set(home_rot['PLAYER_NAME'])]
        if sources.get('away') == 'report':
            away_out = [p for p in away_out if p in set(away_rot['PLAYER_NAME'])]
        h_loss, h_details, h_skipped = self._injury_loss(home_team, home_rot, home_out)
        a_loss, a_details, a_skipped = self._injury_loss(away_team, away_rot, away_out)

        features = pd.DataFrame([{
            'ELO_DIFF': home['ELO'] - away['ELO'],
            'EFG_DIFF': home['ROLLING_eFG_PCT'] - away['ROLLING_eFG_PCT'],
            'TOV_PCT_DIFF': home['ROLLING_TOV_PCT'] - away['ROLLING_TOV_PCT'],
            'ORB_PCT_DIFF': home['ROLLING_ORB_PCT'] - away['ROLLING_ORB_PCT'],
            'FT_RATE_DIFF': home['ROLLING_FT_RATE'] - away['ROLLING_FT_RATE'],
            'WIN_STREAK_DIFF': home['WIN_STREAK'] - away['WIN_STREAK'],
            'REST_DIFF': home['DAYS_REST'] - away['DAYS_REST'],
            'B2B_DIFF': home['IS_B2B'] - away['IS_B2B'],
            'PLUS_MINUS_DIFF': home['ROLLING_PLUS_MINUS'] - away['ROLLING_PLUS_MINUS'],
            'PACE_DIFF': home['ROLLING_PACE'] - away['ROLLING_PACE'],
            'DEF_RATING_DIFF': home['ROLLING_DEF_RATING'] - away['ROLLING_DEF_RATING'],
            'CORE_INJURY_DIFF': injury_diff(h_loss, a_loss),
        }])[FEATURES]

        stale_days = min(home['DAYS_SINCE_LAST_GAME'], away['DAYS_SINCE_LAST_GAME'])
        return {
            'home_prob': float(self.model.predict_proba(features)[0][1]),
            'features': features,
            'injury_diff': float(features['CORE_INJURY_DIFF'].iloc[0]),
            'home_rotation': home_rot, 'away_rotation': away_rot,
            'home_details': h_details, 'away_details': a_details,
            'skipped': h_skipped + a_skipped,
            'home_out': list(home_out), 'away_out': list(away_out), 'out_sources': sources,
            'stale_warning': (f"Latest data is {stale_days} days old; rolling stats may not reflect "
                              f"current rosters. Re-run ingest.py and data_pipeline.py.")
                             if stale_days > 14 else None,
        }
