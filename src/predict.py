import pandas as pd
import joblib
from features import NBAFeatureProcessor
from nba_api.stats.static import teams

def predict_game(home_team_name, away_team_name, home_injuries=None, away_injuries=None, home_acute=None, away_acute=None):
    """
    home_injuries: list of player names marked as out for home team
    away_injuries: list of player names marked as out for away team
    home_acute: list of home players to force as ACUTE (manual override)
    away_acute: list of away players to force as ACUTE (manual override)
    """
    if home_injuries is None:
        home_injuries = []
    if away_injuries is None:
        away_injuries = []
    if home_acute is None:
        home_acute = []
    if away_acute is None:
        away_acute = []

    # 1. Load model and data
    model = joblib.load('models/nba_model.joblib')
    df = pd.read_csv('data/processed_data_with_elo.csv')
    player_data = pd.read_csv('data/raw_player_boxscores.csv')
    player_data['GAME_DATE'] = pd.to_datetime(player_data['GAME_DATE'])
    player_data.columns = player_data.columns.str.strip()
    positions_df = pd.read_csv('data/player_positions.csv')

    # 2. Identify rotation (Top 10 by impact)
    processor = NBAFeatureProcessor(df)
    rotation_df = processor.identify_core_four(player_data)
    rotation_df = rotation_df.reset_index()

    if 'TEAM_NAME' not in rotation_df.columns:
        nba_teams = pd.DataFrame(teams.get_teams())[['id', 'full_name']]
        nba_teams.columns = ['TEAM_ID', 'TEAM_NAME']
        rotation_df = rotation_df.merge(nba_teams, on='TEAM_ID', how='left')

    rotation_dict = rotation_df.groupby('TEAM_NAME').apply(
        lambda x: x[['PLAYER_NAME', 'impact_score']].to_dict('records')
    ).to_dict()

    # 3. Calculate injury impact with acute/chronic logic
    max_boost_tracker = {}

    def calculate_injury_diff(team_name, injured_players, acute_overrides=None):
        if acute_overrides is None:
            acute_overrides = []
        total_impact_lost = 0
        total_boost = 0
        details = []

        for player_name in injured_players:
            # Manual override takes priority over auto-classification
            if player_name in acute_overrides:
                injury_type = 'acute'
                print(f"  [MANUAL OVERRIDE] {player_name} classified as ACUTE")
            else:
                injury_type = processor.classify_injury(player_name, player_data)

            team_rotation = rotation_dict.get(team_name, [])
            player_row = next((p for p in team_rotation if p['PLAYER_NAME'] == player_name), None)

            if player_row is None:
                print(f"  [!] {player_name} not found in {team_name} rotation — skipping")
                continue

            impact_lost = player_row['impact_score']
            total_impact_lost += impact_lost

            if injury_type == 'acute':
                boost, _ = processor.calculate_replacement_boost(
                    player_name, team_name, player_data, positions_df, max_boost_tracker
                )
                total_boost += boost
                details.append(f"⚡ **{player_name}** — ACUTE (-{impact_lost:.1f}, +{boost:.1f} boost)")
            else:
                details.append(f"🔴 **{player_name}** — CHRONIC (-{impact_lost:.1f}, rotation already adjusted)")

        net_loss = total_impact_lost - total_boost
        return net_loss, details

    home_loss, home_details = calculate_injury_diff(home_team_name, home_injuries, home_acute)
    away_loss, away_details = calculate_injury_diff(away_team_name, away_injuries, away_acute)

    # Differential and normalization
    injury_diff = away_loss - home_loss
    injury_diff_normalized = injury_diff / 10

    # 4. Get latest team stats
    h_data = df[df['TEAM_NAME'] == home_team_name]
    a_data = df[df['TEAM_NAME'] == away_team_name]

    if h_data.empty or a_data.empty:
        print(f"Error: Could not find stats for {home_team_name} or {away_team_name}. Check spelling!")
        return

    home_stats = h_data.iloc[-1]
    away_stats = a_data.iloc[-1]

    # 5. Build feature vector
    features = pd.DataFrame([{
        'ELO_DIFF': home_stats['PRE_GAME_ELO'] - away_stats['PRE_GAME_ELO'],
        'EFG_DIFF': home_stats['ROLLING_eFG_PCT'] - away_stats['ROLLING_eFG_PCT'],
        'TOV_PCT_DIFF': home_stats['ROLLING_TOV_PCT'] - away_stats['ROLLING_TOV_PCT'],
        'ORB_PCT_DIFF': home_stats['ROLLING_ORB_PCT'] - away_stats['ROLLING_ORB_PCT'],
        'FT_RATE_DIFF': home_stats['ROLLING_FT_RATE'] - away_stats['ROLLING_FT_RATE'],
        'WIN_STREAK_DIFF': home_stats['WIN_STREAK'] - away_stats['WIN_STREAK'],
        'REST_DIFF': home_stats['DAYS_REST'] - away_stats['DAYS_REST'],
        'B2B_DIFF': home_stats['IS_B2B'] - away_stats['IS_B2B'],
        'PLUS_MINUS_DIFF': home_stats['ROLLING_PLUS_MINUS'] - away_stats['ROLLING_PLUS_MINUS'],
        'PACE_DIFF': home_stats['ROLLING_PACE'] - away_stats['ROLLING_PACE'],
        'DEF_RATING_DIFF': home_stats['ROLLING_DEF_RATING'] - away_stats['ROLLING_DEF_RATING'],
        'CORE_INJURY_DIFF': injury_diff_normalized
    }])

    # 6. Predict
    prob = model.predict_proba(features)[0][1]
    prediction = "WIN" if prob > 0.5 else "LOSS"

    # 7. Display results
    home_rotation = rotation_dict.get(home_team_name, [])
    away_rotation = rotation_dict.get(away_team_name, [])

    print(f"\n" + "="*50)
    print(f" MATCHUP: {away_team_name} @ {home_team_name}")
    print(f"="*50)

    print(f"\n[ROTATION — {home_team_name}]")
    for i, p in enumerate(home_rotation):
        tag = " ← CORE" if i < 4 else ""
        injured_tag = " ✗ OUT" if p['PLAYER_NAME'] in home_injuries else ""
        print(f"  {i+1}. {p['PLAYER_NAME']} (impact: {p['impact_score']:.1f}){tag}{injured_tag}")

    print(f"\n[ROTATION — {away_team_name}]")
    for i, p in enumerate(away_rotation):
        tag = " ← CORE" if i < 4 else ""
        injured_tag = " ✗ OUT" if p['PLAYER_NAME'] in away_injuries else ""
        print(f"  {i+1}. {p['PLAYER_NAME']} (impact: {p['impact_score']:.1f}){tag}{injured_tag}")

    print(f"\n[INJURY ANALYSIS]")
    status = "EVEN" if abs(injury_diff) < 1 else f"{home_team_name} Disadvantage" if injury_diff < 0 else f"{away_team_name} Disadvantage"
    print(f"  Status: {status} (Net diff: {injury_diff:.1f})")

    if home_details:
        print(f"\n  {home_team_name} injuries:")
        for d in home_details:
            print(f"    {d}")
    if away_details:
        print(f"\n  {away_team_name} injuries:")
        for d in away_details:
            print(f"    {d}")

    print(f"\n[PREDICTION]")
    print(f"  {home_team_name} Win Probability: {prob:.2%}")
    print(f"  Recommended: {prediction} on {home_team_name}")
    print("="*50 + "\n")


if __name__ == "__main__":
    h_team = input("Enter Home Team Name: ")
    a_team = input("Enter Away Team Name: ")

    print(f"\nEnter injured players for {h_team} (comma separated, or press Enter for none):")
    h_input = input().strip()
    home_out = [p.strip() for p in h_input.split(',')] if h_input else []

    home_acute = []
    if home_out:
        print(f"Which {h_team} injuries are NEW (last 1-3 games)? (comma separated, or press Enter for none):")
        h_acute_input = input().strip()
        home_acute = [p.strip() for p in h_acute_input.split(',')] if h_acute_input else []

    print(f"\nEnter injured players for {a_team} (comma separated, or press Enter for none):")
    a_input = input().strip()
    away_out = [p.strip() for p in a_input.split(',')] if a_input else []

    away_acute = []
    if away_out:
        print(f"Which {a_team} injuries are NEW (last 1-3 games)? (comma separated, or press Enter for none):")
        a_acute_input = input().strip()
        away_acute = [p.strip() for p in a_acute_input.split(',')] if a_acute_input else []

    predict_game(h_team, a_team, home_out, away_out, home_acute, away_acute)