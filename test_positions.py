import pandas as pd
import sys
sys.path.append('src')
from features import NBAFeatureProcessor

player_data = pd.read_csv('data/raw_player_boxscores.csv')
player_data['GAME_DATE'] = pd.to_datetime(player_data['GAME_DATE'])

df = pd.read_csv('data/processed_data_with_elo.csv')
processor = NBAFeatureProcessor(df)
rotation_df = processor.identify_core_four(player_data).reset_index()

players_needed = rotation_df[['PLAYER_NAME', 'TEAM_NAME']].sort_values('TEAM_NAME')
players_needed.to_csv('data/players_needed.csv', index=False)
print(f"Need positions for {len(players_needed)} players")
print(players_needed.to_string())