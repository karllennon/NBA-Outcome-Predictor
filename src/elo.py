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
        games = self._pair_games(df)
        results = {}
        current_season = None

        for gid, season, h_id, a_id, home_win, margin, complete in games.itertuples(index=False):
            if current_season is not None and season != current_season:
                self._regress_to_mean()
            current_season = season
            if not complete:
                continue  # Skip incomplete data

            h_elo, a_elo = self._get_elo(h_id), self._get_elo(a_id)

            # Pre-game ratings are the model features
            results[(gid, h_id)] = h_elo
            results[(gid, a_id)] = a_elo

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

    @staticmethod
    def _pair_games(df):
        """
        One row per GAME_ID in chronological order (earliest date for that GAME_ID):
        GAME_ID, SEASON_ID, home TEAM_ID, away TEAM_ID, home win, margin, complete.
        A game is complete when it has exactly one home row and one away row.
        """
        order = (df.groupby('GAME_ID')
                   .agg(GAME_DATE=('GAME_DATE', 'min'), SEASON_ID=('SEASON_ID', 'first'),
                        ROWS=('TEAM_ID', 'size'))
                   .sort_values(['GAME_DATE'], kind='stable')
                   .reset_index())
        is_home = df['MATCHUP'].str.contains('vs.', regex=False)
        is_away = df['MATCHUP'].str.contains('@', regex=False)
        home = df[is_home].drop_duplicates('GAME_ID', keep=False).set_index('GAME_ID')
        away = df[is_away].drop_duplicates('GAME_ID', keep=False).set_index('GAME_ID')
        out = order.join(home[['TEAM_ID', 'WL', 'PLUS_MINUS']], on='GAME_ID')
        out = out.join(away[['TEAM_ID']].rename(columns={'TEAM_ID': 'AWAY_ID'}), on='GAME_ID')
        out['COMPLETE'] = (out['ROWS'] == 2) & out['TEAM_ID'].notna() & out['AWAY_ID'].notna()
        return pd.DataFrame({
            'GAME_ID': out['GAME_ID'], 'SEASON_ID': out['SEASON_ID'],
            'H_ID': out['TEAM_ID'].fillna(-1).astype('int64'),
            'A_ID': out['AWAY_ID'].fillna(-1).astype('int64'),
            'HOME_WIN': (out['WL'] == 'W').astype(int),
            'MARGIN': out['PLUS_MINUS'].abs().fillna(0),
            'COMPLETE': out['COMPLETE'],
        })

    def current_ratings(self):
        """Post-game Elo after the most recent game, for live predictions."""
        return pd.DataFrame({'TEAM_ID': list(self.elo_dict.keys()),
                             'CURRENT_ELO': list(self.elo_dict.values())})


# Grid for tuning K-factor, home advantage and season carryover (RESULTS.md, Phase 5)
ELO_GRID = {'k_factor': [5, 7.5, 10, 12.5, 15, 20, 25, 30],
            'home_advantage': [0, 25, 50, 75, 100],
            'season_carryover': [0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9]}


def elo_tables(games, grid=None, **fixed):
    """
    For every Elo setting in the grid: per-game home-minus-away pre-game Elo (ELO_DIFF) and
    Elo's own home win probability with home advantage (ELO_PROB), indexed by GAME_ID.
    Returns {setting tuple: (params dict, DataFrame)}.
    """
    import itertools
    grid = grid or ELO_GRID
    is_home = games['MATCHUP'].str.contains('vs.', regex=False).values
    home = games[is_home][['GAME_ID', 'GAME_DATE', 'WL']]
    tables = {}
    for combo in itertools.product(*grid.values()):
        params = {**fixed, **dict(zip(grid, combo))}
        d = NBAEloCalculator(**params).process_season(games)
        h = d[is_home][['GAME_ID', 'PRE_GAME_ELO']]
        a = d[~is_home][['GAME_ID', 'PRE_GAME_ELO']]
        t = home.merge(h, on='GAME_ID').merge(a, on='GAME_ID', suffixes=('_H', '_A'))
        t['ELO_DIFF'] = t['PRE_GAME_ELO_H'] - t['PRE_GAME_ELO_A']
        t['ELO_PROB'] = 1 / (1 + 10 ** (-(t['ELO_DIFF'] + params['home_advantage']) / 400))
        t['Y'] = (t['WL'] == 'W').astype(int)
        t['GAME_DATE'] = pd.to_datetime(t['GAME_DATE'])
        tables[combo] = (params, t.dropna(subset=['ELO_DIFF']).set_index('GAME_ID'))
    return tables


def tune_elo(tables, before):
    """Setting with the lowest log loss of Elo's own probabilities on games before `before`
    (training games only). Returns (params, ELO_DIFF series indexed by GAME_ID)."""
    best = None
    for params, t in tables.values():
        tr = t[t['GAME_DATE'] < pd.Timestamp(before)]
        p = tr['ELO_PROB'].clip(1e-6, 1 - 1e-6)
        ll = -(tr['Y'] * np.log(p) + (1 - tr['Y']) * np.log(1 - p)).mean()
        if best is None or ll < best[0]:
            best = (ll, params, t['ELO_DIFF'])
    return best[1], best[2]
