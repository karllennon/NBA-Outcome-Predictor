import pandas as pd

def create_matchup_data(processed_df):
    """
    Combines Home and Away rows into a single 'Game' row and calculates exact Four Factors.
    """
    # 1. Identify Home and Away rows
    home_df = processed_df[processed_df['MATCHUP'].str.contains('vs.')].copy()
    away_df = processed_df[processed_df['MATCHUP'].str.contains('@')].copy()

    # 2. Merge the two sides on GAME_ID
    matchups = pd.merge(
        home_df, 
        away_df, 
        on='GAME_ID', 
        suffixes=('_HOME', '_AWAY')
    )

    # 3. Calculate EXACT Four Factors using Opponent Data
    matchups['HOME_ORB_PCT_ACTUAL'] = matchups['OREB_HOME'] / (matchups['OREB_HOME'] + matchups['DREB_AWAY'])
    matchups['AWAY_ORB_PCT_ACTUAL'] = matchups['OREB_AWAY'] / (matchups['OREB_AWAY'] + matchups['DREB_HOME'])

    # 4. Create the differentials (Home - Away)
    matchups['ELO_DIFF'] = matchups['PRE_GAME_ELO_HOME'] - matchups['PRE_GAME_ELO_AWAY']
    matchups['EFG_DIFF'] = matchups['ROLLING_eFG_PCT_HOME'] - matchups['ROLLING_eFG_PCT_AWAY']
    matchups['TOV_PCT_DIFF'] = matchups['ROLLING_TOV_PCT_HOME'] - matchups['ROLLING_TOV_PCT_AWAY']
    matchups['ORB_PCT_DIFF'] = matchups['ROLLING_ORB_PCT_HOME'] - matchups['ROLLING_ORB_PCT_AWAY']
    matchups['FT_RATE_DIFF'] = matchups['ROLLING_FT_RATE_HOME'] - matchups['ROLLING_FT_RATE_AWAY']
    matchups['WIN_STREAK_DIFF'] = matchups['WIN_STREAK_HOME'] - matchups['WIN_STREAK_AWAY']
    matchups['REST_DIFF'] = matchups['DAYS_REST_HOME'] - matchups['DAYS_REST_AWAY']
    matchups['B2B_DIFF'] = matchups['IS_B2B_HOME'] - matchups['IS_B2B_AWAY']
    matchups['PLUS_MINUS_DIFF'] = matchups['ROLLING_PLUS_MINUS_HOME'] - matchups['ROLLING_PLUS_MINUS_AWAY']
    matchups['PACE_DIFF'] = matchups['ROLLING_PACE_HOME'] - matchups['ROLLING_PACE_AWAY']
    matchups['DEF_RATING_DIFF'] = matchups['ROLLING_DEF_RATING_HOME'] - matchups['ROLLING_DEF_RATING_AWAY']

    # 5. Define the Target Variable
    matchups['TARGET'] = (matchups['WL_HOME'] == 'W').astype(int)

    # Add the injury differential from the processed_df
    matchups['CORE_INJURY_DIFF'] = matchups['CORE_INJURY_DIFF_HOME']

    features = [
        'ELO_DIFF', 'EFG_DIFF', 'TOV_PCT_DIFF', 'ORB_PCT_DIFF',
        'FT_RATE_DIFF', 'WIN_STREAK_DIFF', 'REST_DIFF',
        'B2B_DIFF', 'PLUS_MINUS_DIFF', 'PACE_DIFF',
        'DEF_RATING_DIFF', 'CORE_INJURY_DIFF'
    ]
    return matchups[features + ['TARGET']]