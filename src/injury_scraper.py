import requests
from bs4 import BeautifulSoup
import pandas as pd
from nba_api.stats.static import teams

class NBAInjuryScraper:
    def __init__(self):
        # Updated URL for the specific injury report page
        self.url = "https://www.rotowire.com/basketball/injury-report.php"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        self.team_map = self._get_auto_team_map()

    def _get_auto_team_map(self):
        """Builds a lookup for team names, abbreviations, and nicknames."""
        nba_teams = teams.get_teams()
        mapping = {}
        for t in nba_teams:
            mapping[t['nickname']] = t['full_name']
            mapping[t['city']] = t['full_name']
            mapping[t['full_name']] = t['full_name']
            mapping[t['abbreviation']] = t['full_name']
            
        # Manual overrides for edge cases
        mapping['GSW'] = 'Golden State Warriors'
        mapping['NYK'] = 'New York Knicks'
        mapping['BKN'] = 'Brooklyn Nets'
        return mapping

    def get_injured_players(self):
        """Scrapes the live injury report and returns a list of player names."""
        print("Scraping live injury reports...")
        try:
            response = requests.get(self.url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            injured_names = []

            # Updated selectors based on RotoWire's current structure
            # Looking for player name links within the injury table
            player_links = soup.select('div.injury-rel__player a') or soup.select('.injury__player a')
            
            for link in player_links:
                name = link.text.strip()
                if name:
                    injured_names.append(name)

            return list(set(injured_names)) # Remove duplicates
        except Exception as e:
            print(f"Scraper Error: {e}")
            return []

    def calculate_injury_impact(self, home_team, away_team, core_four_dict):
        """Cross-references core players with the live injury list."""
        # Use the fixed function name here!
        injured_list = self.get_injured_players()
        
        def count_missing(team_name):
            core_players = core_four_dict.get(team_name, [])
            missing_count = 0
            for core_p in core_players:
                # Fuzzy Match: Handles names like "Jimmy Butler III" vs "Jimmy Butler"
                for injured_p in injured_list:
                    if core_p.lower() in injured_p.lower() or injured_p.lower() in core_p.lower():
                        print(f"  [!] ALERT: Core player {core_p} ({team_name}) is on the injury report.")
                        missing_count += 1
                        break
            return missing_count

        h_miss = count_missing(home_team)
        a_miss = count_missing(away_team)
        
        # Differential: (Away Injuries - Home Injuries)
        return a_miss - h_miss

if __name__ == "__main__":
    scraper = NBAInjuryScraper()
    # Test the fixed function
    print(f"Found {len(scraper.get_injured_players())} injured players total.")