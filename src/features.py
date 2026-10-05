import pandas as pd
import numpy as np

class NBAFeatureProcessor:
    def __init__(self, df):
        # Convert date and sort to ensure time-based features (rest/rolling) work
        self.df = df.copy()
        self.df['GAME_DATE'] = pd.to_datetime(self.df['GAME_DATE'])
        self.df = self.df.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)
        
    def add_advanced_stats(self):

        # 1. Effective Field Goal Pct (Shooting)
        self.df['eFG_PCT'] = (self.df['FGM'] + 0.5 * self.df['FG3M']) / self.df['FGA']
        
        # 2. Estimated Possessions
        self.df['POSS'] = self.df['FGA'] + 0.44 * self.df['FTA'] - self.df['OREB'] + self.df['TOV']
        
        # 3. TOV% (Possession Security)
        self.df['TOV_PCT'] = self.df['TOV'] / self.df['POSS']
        
        # 4. ORB% (Rebounding): offensive rebounds over all rebounding chances,
        # using the opponent's defensive rebounds from the same game
        opp = self.df[['GAME_ID', 'TEAM_ID', 'DREB']].rename(columns={'TEAM_ID': 'OPP_ID', 'DREB': 'OPP_DREB'})
        merged = self.df[['GAME_ID', 'TEAM_ID']].merge(opp, on='GAME_ID', how='left')
        merged = merged[merged['TEAM_ID'] != merged['OPP_ID']].drop_duplicates(['GAME_ID', 'TEAM_ID'])
        self.df = self.df.merge(merged[['GAME_ID', 'TEAM_ID', 'OPP_DREB']], on=['GAME_ID', 'TEAM_ID'], how='left')
        self.df = self.df.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)
        self.df['ORB_PCT'] = self.df['OREB'] / (self.df['OREB'] + self.df['OPP_DREB'])

        # 5. FT Rate (Aggression)
        self.df['FT_RATE'] = self.df['FTM'] / self.df['FGA']
        
        # 6. Pace
        self.df['PACE'] = 48 * (self.df['POSS'] / (self.df['MIN'] / 5))
        
        # 7. Defensive Rating (Points allowed per 100 possessions approximation)
        # Points allowed = own points minus plus/minus
        self.df['PTS_ALLOWED'] = self.df['PTS'] - self.df['PLUS_MINUS']
        self.df['DEF_RATING'] = (self.df['PTS_ALLOWED'] / self.df['POSS']) * 100
        return self

    def add_comprehensive_stats(self):

        self.df['FG_PCT'] = self.df['FGM'] / self.df['FGA']
        self.df['FG3_PCT'] = self.df['FG3M'] / self.df['FG3A']
        
        # Defensive activity (Normalized by Pace/Possessions is better)
        self.df['BLK_RATE'] = self.df['BLK'] / self.df['POSS']
        self.df['STL_RATE'] = self.df['STL'] / self.df['POSS']
        
        # Points per Possession (Offensive Rating proxy)
        self.df['OFF_RATING'] = self.df['PTS'] / self.df['POSS']
        return self
        
    def add_context_features(self):
        """Calculates fatigue and location metrics."""
        # Calculate Days of Rest
        self.df['DAYS_REST'] = self.df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days
        
        # Cap rest at 7 days so summer breaks don't skew the model
        self.df['DAYS_REST'] = self.df['DAYS_REST'].clip(upper=7)
        
        # Binary 'Back-to-Back' flag
        self.df['IS_B2B'] = (self.df['DAYS_REST'] == 1).astype(int)
        
        # Convert Home/Away to binary
        self.df['IS_HOME'] = self.df['MATCHUP'].str.contains('vs.').astype(int)
        return self
    
    def add_rolling_momentum(self, window=10):
        """Captures recent form (Last 10 games) to avoid 'Data Leakage'."""
        metrics = ['eFG_PCT', 'TOV_PCT', 'ORB_PCT', 'FT_RATE', 'PACE', 'PLUS_MINUS', 'OFF_RATING', 'STL_RATE', 'BLK_RATE', 'DEF_RATING']
        
        for metric in metrics:
            self.df[f'ROLLING_{metric}'] = self.df.groupby('TEAM_ID')[metric].transform(
                lambda x: x.rolling(window=window, closed='left').mean()
            )
        
        # Win Streak (Last 5 games)
        self.df['WIN_STREAK'] = self.df.groupby('TEAM_ID')['WL'].transform(
            lambda x: (x == 'W').astype(int).rolling(window=5, closed='left').sum()
        )
        return self

    def get_final_data(self):
        return self.df.dropna()

    def latest_team_state(self, window=10):
        """
        Each team's form going INTO its next game: rolling averages over the last `window`
        games including the most recent one, wins in the last 5, and the last game date.
        The training features use closed='left' windows ending just before each game, so
        this is the same quantity evaluated for the upcoming game.
        """
        metrics = ['eFG_PCT', 'TOV_PCT', 'ORB_PCT', 'FT_RATE', 'PACE', 'PLUS_MINUS',
                   'OFF_RATING', 'STL_RATE', 'BLK_RATE', 'DEF_RATING']
        df = self.df.sort_values(['TEAM_ID', 'GAME_DATE'])
        rows = []
        for (team_id, team_name), g in df.groupby(['TEAM_ID', 'TEAM_NAME']):
            last = g.tail(window)
            row = {'TEAM_ID': team_id, 'TEAM_NAME': team_name,
                   'LAST_GAME_DATE': g['GAME_DATE'].max(),
                   'WIN_STREAK': (g.tail(5)['WL'] == 'W').sum()}
            for m in metrics:
                row[f'ROLLING_{m}'] = last[m].mean()
            rows.append(row)
        return pd.DataFrame(rows)
