import pandas as pd
import sys
sys.path.append('src')
from features import NBAFeatureProcessor

player_data = pd.read_csv('data/raw_player_boxscores.csv')
player_data['GAME_DATE'] = pd.to_datetime(player_data['GAME_DATE'])
df = pd.read_csv('data/processed_data_with_elo.csv')

processor = NBAFeatureProcessor(df)
rotation = processor.identify_core_four(player_data).reset_index()

row = rotation[rotation['PLAYER_NAME'] == 'Ivica Zubac']
print(row[['PLAYER_NAME', 'TEAM_NAME', 'impact_score']])