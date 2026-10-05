"""Live predictions must compute features with the same code and give the same values that
training used for the same game."""
import numpy as np
import pandas as pd
from data_pipeline import load_games
from context import add_context
from inference import GamePredictor


def test_live_travel_features_match_training():
    games = load_games()
    predictor = GamePredictor.__new__(GamePredictor)   # only the context step is needed
    predictor.team_games = games
    trained = add_context(games)
    home = games[games['MATCHUP'].str.contains('vs.', regex=False)]
    for g in home.sample(25, random_state=1).itertuples():
        away = games[(games['GAME_ID'] == g.GAME_ID) & (games['TEAM_ID'] != g.TEAM_ID)].iloc[0]
        h_live, a_live = predictor._context(g.TEAM_NAME, away['TEAM_NAME'], pd.Timestamp(g.GAME_DATE))
        t = trained[trained['GAME_ID'] == g.GAME_ID].set_index('TEAM_ID')
        for col in ['TRAVEL_KM', 'TZ_SHIFT']:
            assert np.isclose(h_live[col], t.at[g.TEAM_ID, col]), (g.GAME_DATE, col)
            assert np.isclose(a_live[col], t.at[away['TEAM_ID'], col]), (g.GAME_DATE, col)
