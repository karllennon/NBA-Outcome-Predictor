import pandas as pd
import numpy as np

class NBAEloCalculator:
    def __init__(self, k_factor=20, mean_elo=1500):
        self.k_factor = k_factor
        self.mean_elo = mean_elo
        self.elo_dict = {}  # Stores current Elo for each team

    def _get_elo(self, team_id):
        """Retrieves current Elo or initializes to mean."""
        return self.elo_dict.get(team_id, self.mean_elo)

    def calculate_expected_score(self, team_elo, opp_elo):
        """Standard Elo probability formula."""
        return 1 / (10 ** ((opp_elo - team_elo) / 400) + 1)

    def get_margin_multiplier(self, margin, elo_diff):
        """Adjusts Elo gain based on the point spread (blowout factor)."""
        # Formula based on FiveThirtyEight's methodology
        return np.log(max(margin, 1) + 1) * (2.2 / ((elo_diff * 0.001) + 2.2))

    def process_season(self, df):
        """
        Iterates through games to calculate Elo history.
        Assumes df is sorted by date and contains 'GAME_ID', 'TEAM_ID', 'WL', and 'PLUS_MINUS'.
        """
        df = df.sort_values('GAME_DATE')
        elo_before_game = []
        
        # Group by Game_ID to process both teams at once
        game_groups = df.groupby('GAME_ID')
        
        # Dictionary to store the results to map back to original dataframe
        results = {}

        for game_id, group in game_groups:
            if len(group) != 2: continue # Skip incomplete data
            
            team1, team2 = group.iloc[0], group.iloc[1]
            t1_id, t2_id = team1['TEAM_ID'], team2['TEAM_ID']
            
            # 1. Get current Elos
            t1_elo = self._get_elo(t1_id)
            t2_elo = self._get_elo(t2_id)
            
            # Store 'before' Elo for the model to use as a feature
            results[(game_id, t1_id)] = t1_elo
            results[(game_id, t2_id)] = t2_elo
            
            # 2. Determine winner and margin
            t1_win = 1 if team1['WL'] == 'W' else 0
            t2_win = 1 - t1_win
            margin = abs(team1['PLUS_MINUS'])
            
            # 3. Calculate Expected Scores
            exp1 = self.calculate_expected_score(t1_elo, t2_elo)
            exp2 = 1 - exp1
            
            # 4. Apply Multiplier and Update
            # (Note: Using the difference in Elo for the multiplier)
            elo_diff = t1_elo - t2_elo if t1_win else t2_elo - t1_elo
            multiplier = self.get_margin_multiplier(margin, elo_diff)
            
            shift = self.k_factor * multiplier * (t1_win - exp1)
            
            self.elo_dict[t1_id] = t1_elo + shift
            self.elo_dict[t2_id] = t2_elo - shift

        # Map the 'before game' Elos back to the dataframe
        df['PRE_GAME_ELO'] = df.apply(lambda x: results.get((x['GAME_ID'], x['TEAM_ID']), self.mean_elo), axis=1)
        return df