import pandas as pd

FEATURES = [
    'ELO_DIFF', 'EFG_DIFF', 'TOV_PCT_DIFF', 'ORB_PCT_DIFF',
    'FT_RATE_DIFF', 'WIN_STREAK_DIFF', 'REST_DIFF',
    'B2B_DIFF', 'PLUS_MINUS_DIFF', 'PACE_DIFF',
    'DEF_RATING_DIFF', 'CORE_INJURY_DIFF'
]

# Home-minus-away differentials built from each side's pre-game state
DIFF_SOURCES = {
    'EFG_DIFF': 'ROLLING_eFG_PCT',
    'TOV_PCT_DIFF': 'ROLLING_TOV_PCT',
    'ORB_PCT_DIFF': 'ROLLING_ORB_PCT',
    'FT_RATE_DIFF': 'ROLLING_FT_RATE',
    'WIN_STREAK_DIFF': 'WIN_STREAK',
    'REST_DIFF': 'DAYS_REST',
    'B2B_DIFF': 'IS_B2B',
    'PLUS_MINUS_DIFF': 'ROLLING_PLUS_MINUS',
    'PACE_DIFF': 'ROLLING_PACE',
    'DEF_RATING_DIFF': 'ROLLING_DEF_RATING',
}


def injury_diff(home_loss, away_loss):
    """Positive = away team is missing more impact = home advantage."""
    return (away_loss - home_loss) / 10


def create_matchup_data(processed_df):
    """
    Combines home and away rows into one row per game with home-minus-away differentials.
    Keeps GAME_DATE as metadata so training can split chronologically (it is not a feature).
    """
    home_df = processed_df[processed_df['MATCHUP'].str.contains('vs.')].copy()
    away_df = processed_df[processed_df['MATCHUP'].str.contains('@')].copy()

    matchups = pd.merge(home_df, away_df, on='GAME_ID', suffixes=('_HOME', '_AWAY'))

    matchups['ELO_DIFF'] = matchups['PRE_GAME_ELO_HOME'] - matchups['PRE_GAME_ELO_AWAY']
    for feature, source in DIFF_SOURCES.items():
        matchups[feature] = matchups[f'{source}_HOME'] - matchups[f'{source}_AWAY']

    matchups['CORE_INJURY_DIFF'] = injury_diff(matchups['CORE_INJURY_LOSS_HOME'],
                                               matchups['CORE_INJURY_LOSS_AWAY'])

    matchups['TARGET'] = (matchups['WL_HOME'] == 'W').astype(int)
    matchups = matchups.rename(columns={'GAME_DATE_HOME': 'GAME_DATE'})

    out = matchups[['GAME_ID', 'GAME_DATE'] + FEATURES + ['TARGET']]
    return out.sort_values('GAME_DATE').reset_index(drop=True)
