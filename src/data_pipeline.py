import pandas as pd
from elo import NBAEloCalculator
from features import NBAFeatureProcessor
from matchups import create_matchup_data

def run_full_pipeline():
    # 1. Load from cached CSVs (skip ingestion)
    print("Loading cached data...")
    raw_game_df = pd.read_csv('data/raw_nba_data.csv')
    player_boxscores = pd.read_csv('data/raw_player_boxscores.csv')
    
    raw_game_df['GAME_DATE'] = pd.to_datetime(raw_game_df['GAME_DATE'])
    player_boxscores['GAME_DATE'] = pd.to_datetime(player_boxscores['GAME_DATE'])
    
    print(f"Loaded {len(raw_game_df)} team games and {len(player_boxscores)} player rows.")
    
    # 2. Add Elo Ratings
    print("Calculating Elo...")
    elo_calc = NBAEloCalculator()
    df_with_elo = elo_calc.process_season(raw_game_df)
    
    # 3. Process Features & Backfill Injuries
    print("Engineering Features (This will take a few minutes)...")
    processor = NBAFeatureProcessor(df_with_elo)
    
    # Run the backfill first to get CORE_INJURY_DIFF
    positions_df = pd.read_csv('data/player_positions.csv')
    df_with_injuries = processor.backfill_historical_injuries(df_with_elo, player_boxscores, positions_df)

    
    # Update the processor's internal dataframe and run the rest
    processor.df = df_with_injuries
    processed_df = (processor.add_advanced_stats()
                             .add_comprehensive_stats()
                             .add_context_features()
                             .add_rolling_momentum()
                             .get_final_data())
    
    # 4. Create Matchups (The differentials)
    print("Creating Matchup Differentials...")
    final_data = create_matchup_data(processed_df)
    
    # 5. Save
    final_data.to_csv('data/final_training_set.csv', index=False)
    # Save this for the predict.py to look up latest stats
    processed_df.to_csv('data/processed_data_with_elo.csv', index=False)
    print(f"Pipeline Complete! Generated {len(final_data)} games.")

if __name__ == "__main__":
    run_full_pipeline()