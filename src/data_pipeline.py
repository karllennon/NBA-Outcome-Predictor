import pandas as pd
from elo import NBAEloCalculator
from features import NBAFeatureProcessor
from injuries import InjuryModel
from matchups import create_matchup_data

# Settings for the shipped pipeline. experiments.py overrides these to test alternatives.
# Elo settings tuned by log loss of Elo's own predictions (python src/experiments.py elo_tuning;
# RESULTS.md, Phase 5). Re-tune after adding seasons.
ELO_PARAMS = {'k_factor': 12.5, 'home_advantage': 50, 'season_carryover': 0.5}
# Official injury reports: players listed Out on the last report before tip-off (falls back
# to "missed the previous game" when no report covers the game). See RESULTS.md, Phase 2.
INJURY_PARAMS = {"use_reports": True}


def load_raw(data_dir='data'):
    raw_game_df = pd.read_csv(f'{data_dir}/raw_nba_data.csv')
    player_boxscores = pd.read_csv(f'{data_dir}/raw_player_boxscores.csv', low_memory=False)
    raw_game_df['GAME_DATE'] = pd.to_datetime(raw_game_df['GAME_DATE'])
    player_boxscores['GAME_DATE'] = pd.to_datetime(player_boxscores['GAME_DATE'])
    return raw_game_df, player_boxscores


def load_injury_reports():
    from injury_reports import load_archive
    return load_archive()


def build_dataset(raw_game_df, player_boxscores, elo_params=None,
                  injury_params=None, reports=None, verbose=True):
    """
    Returns (matchup training set, feature processor, Elo calculator).
    Every feature for a game uses only games before it.
    """
    log = print if verbose else (lambda *a, **k: None)
    elo_params = ELO_PARAMS if elo_params is None else elo_params
    injury_params = INJURY_PARAMS if injury_params is None else injury_params

    # Elo ratings (pre-game for features, post-game for live predictions)
    log("Calculating Elo...")
    elo_calc = NBAEloCalculator(**elo_params)
    df_with_elo = elo_calc.process_season(raw_game_df)

    # Injury impact for every team-game, using only data from before that game
    injury_model = InjuryModel(player_boxscores, raw_game_df,
                               reports=reports, **injury_params)
    df_with_injuries = injury_model.backfill(df_with_elo, verbose=verbose)

    # Team stats, rest, and rolling form
    log("Engineering features...")
    processor = NBAFeatureProcessor(df_with_injuries)
    processor = (processor.add_advanced_stats()
                          .add_comprehensive_stats()
                          .add_context_features()
                          .add_rolling_momentum())
    processed_df = processor.get_final_data()

    # One row per game with home-minus-away differentials
    log("Creating matchup differentials...")
    return create_matchup_data(processed_df), processor, elo_calc


def run_full_pipeline():
    print("Loading cached data...")
    raw_game_df, player_boxscores = load_raw()
    print(f"Loaded {len(raw_game_df)} team games and {len(player_boxscores)} player rows.")
    reports = load_injury_reports() if INJURY_PARAMS.get('use_reports') else None

    final_data, processor, elo_calc = build_dataset(raw_game_df, player_boxscores,
                                                    reports=reports)
    final_data.to_csv('data/final_training_set.csv', index=False)

    # Live state for predict.py / app.py: form going into each team's NEXT game
    state = processor.latest_team_state().merge(elo_calc.current_ratings(), on='TEAM_ID', how='left')
    state.to_csv('data/team_state.csv', index=False)

    print(f"Pipeline complete. Generated {len(final_data)} games "
          f"({final_data['GAME_DATE'].min().date()} to {final_data['GAME_DATE'].max().date()}).")


if __name__ == "__main__":
    run_full_pipeline()
