import pandas as pd

FEATURES = [
    'ELO_DIFF', 'EFG_DIFF', 'TOV_PCT_DIFF', 'ORB_PCT_DIFF',
    'FT_RATE_DIFF', 'WIN_STREAK_DIFF', 'REST_DIFF',
    'B2B_DIFF', 'PLUS_MINUS_DIFF', 'PACE_DIFF',
    'DEF_RATING_DIFF', 'CORE_INJURY_DIFF',
    'TRAVEL_DIFF', 'TZ_SHIFT_DIFF',   # Phase 6: the only new feature group that helped
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


# Candidate features from context.py, tested one group at a time (experiments.py, Phase 6).
# A group only joins FEATURES if it improves held-out log loss.
CONTEXT_FEATURES = {
    'TRAVEL_DIFF': 'TRAVEL_KM', 'TZ_SHIFT_DIFF': 'TZ_SHIFT',
    'GAMES_LAST_4_DIFF': 'GAMES_LAST_4', 'GAMES_LAST_7_DIFF': 'GAMES_LAST_7',
    'THREE_IN_FOUR_DIFF': 'THREE_IN_FOUR', 'ADJ_NET_DIFF': 'ADJ_NET_10',
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

    extra = []
    for feature, source in CONTEXT_FEATURES.items():
        if f'{source}_HOME' in matchups:
            matchups[feature] = matchups[f'{source}_HOME'] - matchups[f'{source}_AWAY']
            extra.append(feature)
    if 'AT_ALTITUDE_AWAY' in matchups:
        matchups['AWAY_AT_ALTITUDE'] = matchups['AT_ALTITUDE_AWAY']
        extra.append('AWAY_AT_ALTITUDE')

    matchups['TARGET'] = (matchups['WL_HOME'] == 'W').astype(int)
    matchups['MARGIN'] = matchups['PLUS_MINUS_HOME']  # outcome, never a feature
    matchups = matchups.rename(columns={'GAME_DATE_HOME': 'GAME_DATE'})

    extra = [c for c in extra if c not in FEATURES]
    out = matchups[['GAME_ID', 'GAME_DATE'] + FEATURES + extra + ['TARGET', 'MARGIN']]
    return out.sort_values('GAME_DATE').reset_index(drop=True)
