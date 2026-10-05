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
