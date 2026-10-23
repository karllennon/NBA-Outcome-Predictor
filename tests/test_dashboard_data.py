import numpy as np
import pytest
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


def test_what_if_credits_long_absent_players_without_touching_the_model():
    from inference import GamePredictor
    from injuries import InjuryModel
    days = pd.date_range('2025-11-01', periods=30, freq='2D')
    rows = [{'PLAYER_NAME': 'Star', 'TEAM_NAME': 'A', 'GAME_DATE': days[0], 'MIN': 36, 'PTS': 30}]
    rows += [{'PLAYER_NAME': f'P{i}', 'TEAM_NAME': 'A', 'GAME_DATE': d, 'MIN': 24, 'PTS': 10}
             for d in days for i in range(5)]
    box = pd.DataFrame(rows)
    for c in ('REB', 'AST', 'STL', 'BLK', 'DREB', 'TOV'):
        box[c] = 0
    games = pd.DataFrame({'TEAM_NAME': ['A'] * 30, 'GAME_DATE': days})
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(300, len(FEATURES))), columns=FEATURES)
    y = (X['CORE_INJURY_DIFF'] + rng.normal(size=300) > 0).astype(int)
    p = GamePredictor.__new__(GamePredictor)
    p.injuries = InjuryModel(box, games)
    p.model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
    p.spread = None

    after = days[-1] + pd.Timedelta(days=2)
    la = p.long_absent('A', after)
    assert list(la['PLAYER_NAME']) == ['Star'] and la['games_missed'].iloc[0] == 29
    result = {'features': X.iloc[[0]].copy(), 'home_prob': float(p.model.predict_proba(X.iloc[[0]])[0][1])}
    home = p.what_if_returning(result, home_back=['Star'], game_date=after, home_team='A', away_team='B')
    away = p.what_if_returning(result, away_back=['Star'], game_date=after, home_team='B', away_team='A')
    assert home['home_prob'] > result['home_prob'] > away['home_prob']
    assert home['credit'] == (pytest.approx(30.0), 0.0)
    assert result['features'].equals(X.iloc[[0]])          # the model's own feature row is unchanged
