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
        
        # 4. ORB% (Rebounding) - *Note: This usually needs Opponent DREB, 
        self.df['ORB_PCT'] = self.df['OREB'] / (self.df['OREB'] + self.df['DREB'])

        # 5. FT Rate (Aggression)
        self.df['FT_RATE'] = self.df['FTM'] / self.df['FGA']
        
        # 6. Pace
        self.df['PACE'] = 48 * (self.df['POSS'] / (self.df['MIN'] / 5))
        
        # 7. Defensive Rating (Points allowed per 100 possessions approximation)
        # We use opponent points proxy: points allowed = league avg pts - plus_minus
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
    
    def identify_core_four(self, player_stats_df):
        """
        Identifies the Top 4 impact players per team using only 
        the last 15 games to stay current with trades and roster shifts.
        """
        # 1. Sort by date to get the most recent games first
        df = player_stats_df.sort_values('GAME_DATE', ascending=False).copy()
        
        # 2. Filter for only the last 15 games for EVERY player
        # This ensures we don't use old team data from months ago
        recent_stats = df.groupby('PLAYER_NAME').head(15).copy()
        
        # 3. Calculate Impact Score
        recent_stats['defensive_contribution'] = (
            (recent_stats['STL'] * 2.0) + 
            (recent_stats['BLK'] * 1.5) + 
            (recent_stats['DREB'] * 0.5)
        )
        recent_stats['impact_score'] = (
            (recent_stats['PTS'] * 1.0) + 
            (recent_stats['REB'] * 0.4) + 
            (recent_stats['AST'] * 0.7) + 
            (recent_stats['defensive_contribution']) - 
            (recent_stats['TOV'] * 0.5)
        )
        
        # 4. Find the player's LATEST team (where they are right now)
        current_rosters = recent_stats.groupby('PLAYER_NAME').first().reset_index()
        
        # Apply manual trade overrides for players traded but not yet played
        try:
            trades = pd.read_csv('data/recent_trades.csv')
            for _, trade in trades.iterrows():
                mask = current_rosters['PLAYER_NAME'] == trade['PLAYER_NAME']
                if mask.any():
                    current_rosters.loc[mask, 'TEAM_NAME'] = trade['NEW_TEAM']
        except FileNotFoundError:
            pass
        
        # 5. Calculate average impact over those 15 games
        avg_impact = recent_stats.groupby('PLAYER_NAME')['impact_score'].mean().reset_index()
        
        # 6. Merge and get Top 4 per team
        final_roster = avg_impact.merge(current_rosters[['PLAYER_NAME', 'TEAM_NAME', 'TEAM_ID']], on='PLAYER_NAME')
        core_four = final_roster.sort_values(['TEAM_ID', 'impact_score'], ascending=[True, False])
        
        return core_four.groupby('TEAM_ID').head(10)
    
    def classify_injury(self, player_name, player_boxscores, threshold=4):
        """
            Determines if an injury is acute (new) or chronic (established).
            Returns:
            'acute' - player missed fewer than threshold games (role players haven't adjusted)
            'chronic' - player missed threshold+ games (data has naturally adjusted)
            'healthy' - player has been playing
        """
        player_boxscores = player_boxscores.copy()
        player_boxscores['GAME_DATE'] = pd.to_datetime(player_boxscores['GAME_DATE'])
        
        # Get this player's game log sorted by most recent first
        player_log = player_boxscores[
            player_boxscores['PLAYER_NAME'] == player_name
        ].sort_values('GAME_DATE', ascending=False)
        
        if player_log.empty:
            return 'unknown'
        
        # Get the team's full game log to know what games were played
        team_name = player_log.iloc[0]['TEAM_NAME']
        team_games = player_boxscores[
            player_boxscores['TEAM_NAME'] == team_name
        ]['GAME_DATE'].unique()
        team_games = sorted(team_games, reverse=True)
        
        # Count how many of the most recent team games the player missed
        player_game_dates = set(player_log[player_log['MIN'] > 0]['GAME_DATE'])
        
        consecutive_misses = 0
        for game_date in team_games:
            if game_date not in player_game_dates:
                consecutive_misses += 1
            else:
                break  # Stop at first game they played
        
        if consecutive_misses == 0:
            return 'healthy'
        elif consecutive_misses < threshold:
            return 'acute'
        else:
            return 'chronic'
    
    def calculate_replacement_boost(self, player_name, team_name, player_boxscores, positions_df, max_boost_per_team=None):
        """
        Position-aware replacement boost with tiered logic:
        - Top 4 player out acutely: next available top 5 player gets star boost (65%)
        + same position player in top 10 gets minutes boost (25%)
        - 5th player out acutely: no boost applied
        - max_boost_per_team: tracks total boost already applied to prevent stacking
        """
        if max_boost_per_team is None:
            max_boost_per_team = {}

        # Get full top 10 rotation for this team
        team_players = player_boxscores[player_boxscores['TEAM_NAME'] == team_name]
        rotation = self.identify_core_four(team_players)
        team_rotation = rotation[rotation['TEAM_NAME'] == team_name].sort_values(
            'impact_score', ascending=False
        ).reset_index(drop=True)

        if team_rotation.empty:
            return 0, 0

        # Find injured player's rank and impact
        injured_row = team_rotation[team_rotation['PLAYER_NAME'] == player_name]
        if injured_row.empty:
            return 0, 0

        injured_rank = injured_row.index[0]  # 0-indexed so rank 0-3 = top 4, rank 4 = 5th
        injured_score = injured_row.iloc[0]['impact_score']

        # 5th player (rank 4) gets no boost
        if injured_rank >= 4:
            print(f"    → {player_name} is ranked #{injured_rank + 1} — no replacement boost applied")
            return 0, 0

        # Check if this team already hit the boost cap (one boost per team max)
        if max_boost_per_team.get(team_name, 0) >= 1:
            print(f"    → Boost cap reached for {team_name} — skipping additional boost")
            return 0, 0

        # Get injured player's position
        pos_row = positions_df[positions_df['PLAYER_NAME'] == player_name]
        injured_position = pos_row.iloc[0]['POSITION'] if not pos_row.empty else None

        # --- STAR BOOST ---
        # Next available top 5 player (not injured) gets 65% of missing player's impact
        star_boost = 0
        top_5 = team_rotation.head(5)
        for _, candidate in top_5.iterrows():
            if candidate['PLAYER_NAME'] != player_name:
                star_boost = injured_score * 0.65
                print(f"    → Star boost: {candidate['PLAYER_NAME']} +{star_boost:.1f} (65% of {player_name}'s impact)")
                break

        # --- MINUTES BOOST ---
        # Same position player in top 10 (outside top 5) gets 25% boost
        minutes_boost = 0
        if injured_position:
            bench_players = team_rotation.iloc[5:10]  # Players ranked 6-10
            for _, candidate in bench_players.iterrows():
                candidate_pos = positions_df[positions_df['PLAYER_NAME'] == candidate['PLAYER_NAME']]
                if not candidate_pos.empty and candidate_pos.iloc[0]['POSITION'] == injured_position:
                    minutes_boost = injured_score * 0.25
                    print(f"    → Minutes boost: {candidate['PLAYER_NAME']} ({injured_position}) +{minutes_boost:.1f} (25% of {player_name}'s impact)")
                    break

        # Mark boost as used for this team
        max_boost_per_team[team_name] = max_boost_per_team.get(team_name, 0) + 1

        total_boost = star_boost + minutes_boost
        return total_boost, injured_score
    
    def backfill_historical_injuries(self, game_df, player_boxscores, positions_df=None):
        """
        Scans history to see which top 5 impact players missed games.
        Uses tiered logic:
        - Top 4 missing: full impact loss + replacement boost
        - 5th player missing: impact loss only, no boost
        """
        print("Backfilling historical injury data... this may take a moment.")

        game_df['GAME_DATE'] = pd.to_datetime(game_df['GAME_DATE'])
        player_boxscores['GAME_DATE'] = pd.to_datetime(player_boxscores['GAME_DATE'])

        if positions_df is None:
            try:
                positions_df = pd.read_csv('data/player_positions.csv')
            except FileNotFoundError:
                positions_df = pd.DataFrame(columns=['PLAYER_NAME', 'POSITION'])

        injury_results = []
        game_df = game_df.sort_values('GAME_DATE')

        for idx, game in game_df.iterrows():
            home_team = game['TEAM_NAME']
            away_team = game.get('TEAM_NAME_AWAY', 'Unknown')

            past_stats = player_boxscores[player_boxscores['GAME_DATE'] < game['GAME_DATE']]

            if past_stats.empty:
                injury_results.append(0)
                continue

            def get_net_impact_loss(team_name):
                team_past = past_stats[past_stats['TEAM_NAME'] == team_name]
                if team_past.empty:
                    return 0

                # Get top 5 for this team based on past data
                rotation = self.identify_core_four(team_past)
                team_rotation = rotation[rotation['TEAM_NAME'] == team_name].sort_values(
                    'impact_score', ascending=False
                ).reset_index(drop=True).head(5)

                if team_rotation.empty:
                    return 0

                top_5_names = team_rotation['PLAYER_NAME'].tolist()

                # Find team's most recent game before this one
                last_game_date = team_past['GAME_DATE'].max()
                last_game_players = team_past[
                    (team_past['GAME_DATE'] == last_game_date) &
                    (team_past['MIN'] > 0)
                ]['PLAYER_NAME'].tolist()

                # Find which top 5 players missed the last game
                missing_players = [p for p in top_5_names if p not in last_game_players]

                if not missing_players:
                    return 0

                total_net_loss = 0
                boost_used = False  # Cap at one boost per team

                for i, player_name in enumerate(missing_players):
                    player_row = team_rotation[team_rotation['PLAYER_NAME'] == player_name]
                    if player_row.empty:
                        continue

                    rank = player_row.index[0]  # 0-indexed
                    impact = player_row.iloc[0]['impact_score']

                    if rank >= 4:
                        # 5th player — impact loss only, no boost
                        total_net_loss += impact
                    else:
                        # Top 4 player missing
                        if not boost_used:
                            # Star boost: 65% of impact redistributed
                            star_boost = impact * 0.65

                            # Minutes boost: 25% if same position player exists in 6-10
                            minutes_boost = 0
                            pos_row = positions_df[positions_df['PLAYER_NAME'] == player_name]
                            if not pos_row.empty:
                                injured_pos = pos_row.iloc[0]['POSITION']
                                bench = team_rotation.iloc[4:] if len(team_rotation) > 4 else pd.DataFrame()
                                for _, bench_player in bench.iterrows():
                                    bench_pos = positions_df[positions_df['PLAYER_NAME'] == bench_player['PLAYER_NAME']]
                                    if not bench_pos.empty and bench_pos.iloc[0]['POSITION'] == injured_pos:
                                        minutes_boost = impact * 0.25
                                        break

                            net_loss = impact - star_boost - minutes_boost
                            total_net_loss += net_loss
                            boost_used = True
                        else:
                            # Boost cap reached — full impact loss
                            total_net_loss += impact

                return total_net_loss

            h_loss = get_net_impact_loss(home_team)
            a_loss = get_net_impact_loss(away_team)

            # Positive = away team hurting more = home team advantage
            injury_results.append((a_loss - h_loss) / 10)

        game_df['CORE_INJURY_DIFF'] = injury_results
        return game_df

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