import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
import time
import requests
import os

def fetch_season_data(season):
    # This header set is the current "gold standard" for 2026 to bypass blocks
    headers = {
        'Host': 'stats.nba.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.nba.com/',
        'Origin': 'https://www.nba.com',
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true',
        'Connection': 'keep-alive',
    }

    try:
        print(f"--- Attempting {season} with Stealth Mode ---")
        
        # 1. Team Logs - We use a slightly shorter timeout (60s) but more aggressive retry logic
        t = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star='Regular Season',
            player_or_team_abbreviation='T',
            headers=headers,
            timeout=60
        )
        team_df = t.get_data_frames()[0]
        print(f"  [v] Team Logs Received ({len(team_df)} rows)")
        
        # 2. THE CRITICAL WAIT: A human takes time to click between pages
        print("  [.] Cooling down for 15 seconds...")
        time.sleep(15) 

        # 3. Player Logs
        p = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star='Regular Season',
            player_or_team_abbreviation='P',
            headers=headers,
            timeout=60
        )
        player_df = p.get_data_frames()[0]
        print(f"  [v] Player Logs Received ({len(player_df)} rows)")
        
        return team_df, player_df

    except Exception as e:
        print(f"  [!] Connection Blocked: {e}")
        return None, None

if __name__ == "__main__":
    if not os.path.exists('data'): os.makedirs('data')
    
    # JUST DO ONE SEASON FIRST to verify the fix
    t, p = fetch_season_data('2025-26')
    
    if t is not None:
        t.to_csv('data/raw_nba_data.csv', index=False)
        p.to_csv('data/raw_player_boxscores.csv', index=False)
        print("\nSUCCESS: Connection stabilized!")
    else:
        print("\nFAILED: The NBA is currently 'Ghosting' your requests.")