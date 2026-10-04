import pandas as pd
import numpy as np


class NBAEloCalculator:
    """
    Game-by-game Elo ratings, following FiveThirtyEight's NBA method:
    - home-court advantage added to the home team's rating when computing expectations
    - margin-of-victory multiplier
    - regression toward the mean at the start of each new season
    """

    def __init__(self, k_factor=20, mean_elo=1505, home_advantage=100, season_carryover=0.75):
        self.k_factor = k_factor
        self.mean_elo = mean_elo
        self.home_advantage = home_advantage
        self.season_carryover = season_carryover
        self.elo_dict = {}  # Current (post-game) Elo for each team

    def _get_elo(self, team_id):
        """Retrieves current Elo or initializes to the mean."""
        return self.elo_dict.get(team_id, self.mean_elo)

    def calculate_expected_score(self, team_elo, opp_elo):
        """Standard Elo win probability."""
        return 1 / (10 ** ((opp_elo - team_elo) / 400) + 1)

    def get_margin_multiplier(self, margin, winner_elo_diff):
        """Scales the update by margin of victory, damped when the favorite wins big."""
        return np.log(max(margin, 1) + 1) * (2.2 / ((winner_elo_diff * 0.001) + 2.2))

    def _regress_to_mean(self):
        """Pulls every team part of the way back to the mean between seasons."""
        for team_id, elo in self.elo_dict.items():
            self.elo_dict[team_id] = self.season_carryover * elo + (1 - self.season_carryover) * self.mean_elo

    def process_season(self, df):
        """
        Walks through games in date order and records each team's Elo BEFORE the game.
        Expects GAME_ID, GAME_DATE, TEAM_ID, MATCHUP, WL, PLUS_MINUS and SEASON_ID columns.
        """
        df = df.copy()
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

        # One row per game: the earliest date for that GAME_ID, so groups are processed chronologically
        game_order = (df.groupby('GAME_ID')
                        .agg(GAME_DATE=('GAME_DATE', 'min'), SEASON_ID=('SEASON_ID', 'first'))
                        .sort_values(['GAME_DATE'])
                        .reset_index())

        groups = {gid: g for gid, g in df.groupby('GAME_ID')}
        results = {}
        current_season = None

        for _, game in game_order.iterrows():
            if current_season is not None and game['SEASON_ID'] != current_season:
                self._regress_to_mean()
            current_season = game['SEASON_ID']

            group = groups[game['GAME_ID']]
            if len(group) != 2:
                continue  # Skip incomplete data

            home = group[group['MATCHUP'].str.contains('vs.')]
            away = group[group['MATCHUP'].str.contains('@')]
            if len(home) != 1 or len(away) != 1:
                continue
            home, away = home.iloc[0], away.iloc[0]

            h_id, a_id = home['TEAM_ID'], away['TEAM_ID']
            h_elo, a_elo = self._get_elo(h_id), self._get_elo(a_id)

            # Pre-game ratings are the model features
            results[(game['GAME_ID'], h_id)] = h_elo
            results[(game['GAME_ID'], a_id)] = a_elo

            home_win = 1 if home['WL'] == 'W' else 0
            margin = abs(home['PLUS_MINUS'])

            # Home-court advantage enters the expectation, not the stored rating
            exp_home = self.calculate_expected_score(h_elo + self.home_advantage, a_elo)

            if home_win:
                winner_diff = (h_elo + self.home_advantage) - a_elo
            else:
                winner_diff = a_elo - (h_elo + self.home_advantage)
            multiplier = self.get_margin_multiplier(margin, winner_diff)

            shift = self.k_factor * multiplier * (home_win - exp_home)
            self.elo_dict[h_id] = h_elo + shift
            self.elo_dict[a_id] = a_elo - shift

        df['PRE_GAME_ELO'] = [results.get((g, t), np.nan) for g, t in zip(df['GAME_ID'], df['TEAM_ID'])]
        return df

    def current_ratings(self):
        """Post-game Elo after the most recent game, for live predictions."""
        return pd.DataFrame({'TEAM_ID': list(self.elo_dict.keys()),
                             'CURRENT_ELO': list(self.elo_dict.values())})
