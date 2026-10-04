import pandas as pd
from elo import NBAEloCalculator
from features import NBAFeatureProcessor
from injuries import InjuryModel
from matchups import create_matchup_data


def run_full_pipeline():
    # 1. Load from cached CSVs (skip ingestion)
    print("Loading cached data...")
    raw_game_df = pd.read_csv('data/raw_nba_data.csv')
    player_boxscores = pd.read_csv('data/raw_player_boxscores.csv')
    positions_df = pd.read_csv('data/player_positions.csv')

    raw_game_df['GAME_DATE'] = pd.to_datetime(raw_game_df['GAME_DATE'])
    player_boxscores['GAME_DATE'] = pd.to_datetime(player_boxscores['GAME_DATE'])
    print(f"Loaded {len(raw_game_df)} team games and {len(player_boxscores)} player rows.")

    # 2. Elo ratings (pre-game for features, post-game for live predictions)
    print("Calculating Elo...")
    elo_calc = NBAEloCalculator()
    df_with_elo = elo_calc.process_season(raw_game_df)

    # 3. Injury impact for every team-game, using only data from before that game
    injury_model = InjuryModel(player_boxscores, raw_game_df, positions_df)
    df_with_injuries = injury_model.backfill(df_with_elo)

    # 4. Team stats, rest, and rolling form
    print("Engineering features...")
    processor = NBAFeatureProcessor(df_with_injuries)
    processor = (processor.add_advanced_stats()
                          .add_comprehensive_stats()
                          .add_context_features()
                          .add_rolling_momentum())
    processed_df = processor.get_final_data()

    # 5. One row per game with home-minus-away differentials
    print("Creating matchup differentials...")
    final_data = create_matchup_data(processed_df)
    final_data.to_csv('data/final_training_set.csv', index=False)

    # 6. Live state for predict.py / app.py: form going into each team's NEXT game
    state = processor.latest_team_state().merge(elo_calc.current_ratings(), on='TEAM_ID', how='left')
    state.to_csv('data/team_state.csv', index=False)

    print(f"Pipeline complete. Generated {len(final_data)} games "
          f"({final_data['GAME_DATE'].min().date()} to {final_data['GAME_DATE'].max().date()}).")


if __name__ == "__main__":
    run_full_pipeline()
