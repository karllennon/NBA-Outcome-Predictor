import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
import time
import os

def fetch_season_data(season):
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
        print(f"--- Fetching {season} ---")

        t = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star='Regular Season',
            player_or_team_abbreviation='T',
            headers=headers,
            timeout=60
        )
        team_df = t.get_data_frames()[0]
        print(f"  [✓] Team Logs: {len(team_df)} rows")

        print("  [.] Cooling down 15 seconds...")
        time.sleep(15)

        p = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star='Regular Season',
            player_or_team_abbreviation='P',
            headers=headers,
            timeout=60
        )
        player_df = p.get_data_frames()[0]
        print(f"  [✓] Player Logs: {len(player_df)} rows")

        return team_df, player_df

    except Exception as e:
        print(f"  [!] Failed: {e}")
        return None, None

if __name__ == "__main__":
    if not os.path.exists('data'):
        os.makedirs('data')

    seasons = ['2023-24', '2024-25', '2025-26']

    all_team = []
    all_player = []

    for season in seasons:
        t, p = fetch_season_data(season)
        if t is not None:
            all_team.append(t)
            all_player.append(p)
        print(f"  [.] Waiting 20 seconds before next season...")
        time.sleep(20)

    if all_team:
        final_team = pd.concat(all_team, ignore_index=True)
        final_player = pd.concat(all_player, ignore_index=True)

        final_team['GAME_DATE'] = pd.to_datetime(final_team['GAME_DATE'])
        final_player['GAME_DATE'] = pd.to_datetime(final_player['GAME_DATE'])

        final_team.to_csv('data/raw_nba_data.csv', index=False)
        final_player.to_csv('data/raw_player_boxscores.csv', index=False)

        print(f"\nDone!")
        print(f"Team games: {len(final_team)}")
        print(f"Player rows: {len(final_player)}")
        print(f"Date range: {final_team['GAME_DATE'].min()} to {final_team['GAME_DATE'].max()}")
    else:
        print("\nFailed — no data retrieved.")