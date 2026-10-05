import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import dashboard_data as dd
from matchups import FEATURES


def test_contributions_reproduce_the_model_probability():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, len(FEATURES))), columns=FEATURES)
    y = (X['ELO_DIFF'] + 0.5 * X['CORE_INJURY_DIFF'] + rng.normal(size=400) > 0).astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
    row = X.iloc[7].to_dict()
    c = dd.contributions(model, row)
    logit = c['contribution'].sum() + model[-1].intercept_[0]
    assert np.isclose(1 / (1 + np.exp(-logit)), model.predict_proba(X.iloc[[7]])[0, 1])
    assert c['contribution'].abs().is_monotonic_decreasing


def test_team_record_counts_only_earlier_games_in_the_season():
    games = pd.DataFrame({
        'TEAM_NAME': ['A'] * 4, 'SEASON_ID': [22024, 22025, 22025, 22025],
        'GAME_DATE': ['2025-03-01', '2025-10-25', '2025-10-27', '2025-10-29'], 'WL': ['W', 'W', 'L', 'W']})
    assert dd.team_record(games, 'A', '2025-10-29') == (1, 1)
