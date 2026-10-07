import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import frozen_model as fm
import paper_report as R
from matchups import FEATURES


def _tiny_model(tmp_path, coef_sign=1.0):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, len(FEATURES))), columns=FEATURES)
    y = (coef_sign * X['ELO_DIFF'] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    path = tmp_path / 'nba_model.joblib'
    joblib.dump(LogisticRegression().fit(X, y), path)
    return str(path), X.iloc[[0]]


def test_freeze_copies_the_model_once_and_predicts_from_live_features(tmp_path):
    model_path, row = _tiny_model(tmp_path)
    frozen_dir = str(tmp_path / 'frozen')
    paths = {'model_path': model_path, 'spread_path': str(tmp_path / 'missing.joblib')}

    assert fm.freeze_on('2026-10-19', frozen_dir, today='2026-10-18', **paths) is None   # too early
    m = fm.freeze_on('2026-10-19', frozen_dir, today='2026-10-19', **paths)
    assert m['frozen_on'] == '2026-10-19' and m['features'] == list(FEATURES) and not m['spread_model']
    assert fm.freeze_on('2026-10-19', frozen_dir, today='2026-10-25', **paths) is None   # already frozen
    with pytest.raises(FileExistsError):
        fm.freeze(frozen_dir, **paths)

    # retraining the live model afterwards does not change the frozen one
    live = joblib.load(model_path)
    joblib.dump(LogisticRegression().fit(row.iloc[[0, 0]].assign(ELO_DIFF=[-1, 1]), [1, 0]), model_path)
    frozen = fm.load(frozen_dir)
    prob, margin, sigma = frozen.predict(row)
    assert prob == pytest.approx(live.predict_proba(row[FEATURES])[0][1])
    assert margin is None and sigma is None


def test_slate_logs_the_frozen_models_pick_on_the_same_prices():
    from daily_slate import frozen_fields

    class Stub:
        version = 'frozen1'

        def predict(self, features):
            return 0.40, None, None

    row = {'HOME_ABBR': 'CHA', 'AWAY_ABBR': 'BKN', 'MODEL_HOME_PROB': 0.60, 'MARKET_YES_BID': 0.54,
           'MARKET_YES_ASK': 0.56, 'MODEL_HOME_MARGIN': None, 'SPREAD_SIGMA': None, 'MARKET_SPREAD_INFO': None}
    out = frozen_fields(row, Stub(), pd.DataFrame([{}]))
    assert out['FROZEN_HOME_PROB'] == 0.40 and out['FROZEN_MODEL_VERSION'] == 'frozen1'
    assert out['FROZEN_PICK_TEAM'] == 'BKN' and out['FROZEN_PICK_SIDE'] == 'NO'
    assert out['FROZEN_PICK_FILL'] == pytest.approx(1 - 0.54)
    assert out['FROZEN_PICK_MARKET_UNDERDOG']          # BKN priced at 45c
    assert frozen_fields(row, None, pd.DataFrame([{}])) == {}


def test_report_judges_the_plan_on_the_frozen_model_and_compares_both(tmp_path):
    tip = pd.Timestamp('2026-10-21 19:00')
    logged = (tip - pd.Timedelta(minutes=60)).tz_localize(R.ET).tz_convert('UTC').tz_localize(None)
    base = {'LOGGED_AT_UTC': logged, 'GAME_ID': '0022600001', 'GAME_DATE': tip.normalize(), 'TIP_TIME_ET': tip,
            'HOME_ABBR': 'CHA', 'AWAY_ABBR': 'BKN', 'MARKET_YES_BID': 0.54, 'MARKET_YES_ASK': 0.56,
            'MODEL_VERSION': 'live1', 'SPREAD_SIDE': None, 'FROZEN_SPREAD_SIDE': None,
            # live model picks CHA (the market favorite); frozen picks BKN (the market underdog)
            'MODEL_HOME_PROB': 0.60, 'PICK_TEAM': 'CHA', 'PICK_SIDE': 'YES', 'PICK_P': 0.60, 'PICK_MID': 0.55,
            'PICK_FILL': 0.56, 'PICK_MARKET_UNDERDOG': False,
            'FROZEN_MODEL_VERSION': 'frozen1', 'FROZEN_HOME_PROB': 0.40, 'FROZEN_PICK_TEAM': 'BKN',
            'FROZEN_PICK_SIDE': 'NO', 'FROZEN_PICK_P': 0.60, 'FROZEN_PICK_MID': 0.45, 'FROZEN_PICK_FILL': 0.46,
            'FROZEN_PICK_MARKET_UNDERDOG': True}
    games = tmp_path / 'games.csv'
    pd.DataFrame([{'GAME_ID': '0022600001', 'MATCHUP': 'CHA vs. BKN', 'PLUS_MINUS': -4},
                  {'GAME_ID': '0022600001', 'MATCHUP': 'BKN @ CHA', 'PLUS_MINUS': 4}]).to_csv(games, index=False)
    t = R.build(pd.DataFrame([base]), str(games), since='2026-10-20', tip_path=str(tmp_path / 'none.csv'))
    assert t['judged_on'] == 'frozen'
    live = t['models']['live']['singles'].set_index('id')
    frozen = t['models']['frozen']['singles'].set_index('id')
    assert live.loc['S1', 'bets'] == 0 and frozen.loc['S1', 'bets'] == 1
    assert frozen.loc['S1', 'win rate'] == 1.0                       # BKN won by 4
    season = t['comparison'].set_index('month').loc['Season']
    assert season['games'] == 1
    assert season['live log loss'] == pytest.approx(-np.log(0.40))
    assert season['frozen log loss'] == pytest.approx(-np.log(0.60))
    assert 'Frozen model: single bets' in R.report(t)
