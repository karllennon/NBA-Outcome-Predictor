import pandas as pd
from nba_api.stats.endpoints import scoreboardv2
from predict import predict_game
from datetime import datetime

def run_daily_slate():
    print(f"--- NBA Prediction Slate for {datetime.now().strftime('%Y-%m-%d')} ---")
    
    # 1. Get today's games
    sb = scoreboardv2.ScoreboardV2()
    games = sb.get_data_frames()[0]
    
    if games.empty:
        print("No games scheduled for today.")
        return

    # 2. Map Team IDs to Names (The API returns IDs)
    from nba_api.stats.static import teams
    all_teams = pd.DataFrame(teams.get_teams())
    team_map = all_teams.set_index('id')['full_name'].to_dict()

    # 3. Loop through games and predict
    for _, game in games.iterrows():
        home_id = game['HOME_TEAM_ID']
        away_id = game['VISITOR_TEAM_ID']
        
        home_name = team_map.get(home_id)
        away_name = team_map.get(away_id)
        
        try:
            predict_game(home_name, away_name)
        except Exception as e:
            print(f"Could not predict {away_name} @ {home_name}: {e}")

if __name__ == "__main__":
    run_daily_slate()