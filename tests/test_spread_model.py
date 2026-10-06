import numpy as np
import pandas as pd
import pytest
import spread_model as sm
from matchups import FEATURES


def _frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    X['MARGIN'] = 8 * X['ELO_DIFF'] + rng.normal(scale=12, size=n)
    X['TOTAL_POINTS'] = 225 + 5 * X['PACE_DIFF'] + rng.normal(scale=15, size=n)
    return X


def test_sign_convention():
    # home predicted to win by 4.4 -> home line -4.4 -> shown as the home team -4.5
    assert float(sm.home_spread(4.4)) == -4.4
    assert sm.format_spread(4.4, 'BOS', 'LAL') == 'BOS -4.5'
    assert sm.format_spread(-6.2, 'BOS', 'LAL') == 'LAL -6'
    assert sm.format_spread(0.2, 'BOS', 'LAL') == 'PK'
    assert float(sm.round_half(3.26)) == 3.5 and float(sm.round_half(3.24)) == 3.0


def test_cover_probability_direction():
    # home predicted by 6 with sigma 12: covering -4.5 is a bit better than a coin flip,
    # covering -10.5 is less likely, and the two sides of a line add to one
    p1 = float(sm.prob_home_covers(6.0, 12.0, -4.5))
    p2 = float(sm.prob_home_covers(6.0, 12.0, -10.5))
    assert 0.5 < p1 < 0.6 and p2 < 0.5
    assert np.isclose(sm.prob_home_covers(6.0, 12.0, -4.5) + (1 - sm.prob_margin_over(6.0, 12.0, 4.5)), 1)


def test_sigma_comes_from_training_rows_only():
    df = _frame()
    a = sm.fit_target(sm.TARGETS['spread'], df.iloc[:400])
    changed = df.copy()
    changed.loc[400:, 'MARGIN'] = 999          # rows after the training window
    b = sm.fit_target(sm.TARGETS['spread'], changed.iloc[:400])
    assert a.sigma == b.sigma and 9 < a.sigma < 16


def test_new_target_needs_only_a_config_entry(tmp_path, monkeypatch):
    cfg = sm.TargetConfig('total', 'TOTAL_POINTS', model_path=str(tmp_path / 'total.joblib'))
    monkeypatch.setitem(sm.TARGETS, 'total', cfg)
    fitted = sm.fit_target(cfg, _frame())
    sm.save(fitted)
    loaded = sm.load('total')
    X = _frame(5, seed=1)
    assert np.allclose(loaded.predict(X), fitted.predict(X))
    assert 200 < loaded.predict(X).mean() < 250
    p = loaded.prob_over(X, 225.5)
    assert ((p > 0) & (p < 1)).all()


def test_margin_is_an_outcome_not_a_feature():
    assert 'MARGIN' not in FEATURES
    assert sm.TARGETS['spread'].target == 'MARGIN'
    assert 'MARGIN' not in sm.TARGETS['spread'].features


def test_margin_target_matches_each_games_own_result():
    from data_pipeline import load_raw, build_dataset
    games, players = load_raw()
    games = games[games['GAME_DATE'] <= '2024-01-10']
    players = players[players['GAME_DATE'] <= '2024-01-10']
    df, _, _ = build_dataset(games, players, verbose=False)
    home = games[games['MATCHUP'].str.contains('vs.', regex=False)].set_index('GAME_ID')['PLUS_MINUS']
    assert (df.set_index('GAME_ID')['MARGIN'] == home.loc[df['GAME_ID']].values).all()


def test_walk_forward_spread_never_sees_test_margins():
    from evaluation import walk_forward
    df = _frame(800)
    df['GAME_ID'] = range(len(df))
    df['GAME_DATE'] = pd.date_range('2024-01-01', periods=len(df), freq='6h')
    df['TARGET'] = (df['MARGIN'] > 0).astype(int)
    fit = lambda tr, te: sm.fit_target(sm.TARGETS['spread'], tr).predict(te)
    a = walk_forward(df, {'spread': fit})
    changed = df.copy()
    last_block = changed['GAME_DATE'] >= a.loc[a['FOLD'] == 3, 'GAME_DATE'].min()
    changed.loc[last_block, 'MARGIN'] = 500.0     # results of the last test block
    b = walk_forward(changed, {'spread': fit})
    assert np.allclose(a['spread'], b['spread'])


def test_live_spread_uses_the_same_feature_row_as_the_win_probability():
    import os
    if not (os.path.exists('models/nba_model.joblib') and os.path.exists('models/spread_model.joblib')):
        pytest.skip('run train.py first')
    from inference import GamePredictor
    p = GamePredictor()
    r = p.predict('Boston Celtics', 'Los Angeles Lakers', [], [])
    assert list(r['features'].columns) == list(FEATURES) == list(p.spread.config.features)
    assert np.isclose(r['home_margin'], p.spread.predict(r['features'])[0])
    assert np.isclose(r['home_prob'], p.model.predict_proba(r['features'])[0, 1])
