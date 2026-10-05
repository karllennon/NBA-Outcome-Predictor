import time
import pandas as pd
import prediction_log as pl


def _row(game_id, prob, market=None):
    return {'GAME_ID': game_id, 'GAME_DATE': '2026-03-03', 'TIP_TIME_ET': '2026-03-03 19:00',
            'HOME_TEAM': 'Charlotte Hornets', 'AWAY_TEAM': 'Dallas Mavericks',
            'MODEL_HOME_PROB': prob, 'MARKET_HOME_PROB': market, 'MARKET_YES_BID': None,
            'MARKET_YES_ASK': None, 'MODEL_VERSION': 'test', 'HOME_OUT': '', 'AWAY_OUT': '',
            'INPUTS': '{}'}


def test_log_is_append_only_and_scores_the_last_pregame_row(tmp_path):
    path = tmp_path / 'log.csv'
    games = tmp_path / 'games.csv'
    pd.DataFrame({'GAME_ID': ['0022500883', '0022500883'], 'MATCHUP': ['CHA vs. DAL', 'DAL @ CHA'],
                  'WL': ['W', 'L'], 'PLUS_MINUS': [8, -8]}).to_csv(games, index=False)

    pl.append([_row('0022500883', 0.60, 0.80)], path)
    first = pd.read_csv(path)
    time.sleep(1.1)  # distinct timestamp
    pl.append([_row('0022500883', 0.70, 0.85)], path)
    log = pl.load(path)

    assert len(log) == 2
    pd.testing.assert_frame_equal(pd.read_csv(path).iloc[:1], first)   # first row untouched

    record = pl.track_record(log, games_path=games)
    assert len(record) == 1
    assert record['MODEL_HOME_PROB'].iloc[0] == 0.70                 # latest prediction
    assert record['HOME_WIN'].iloc[0] == 1

    m = pl.running_metrics(record, 'MODEL_HOME_PROB')
    assert m['accuracy'].iloc[-1] == 1.0


def test_slate_table_highlights_only_gaps_of_five_points_or_more():
    from daily_slate import slate_table
    slate = pd.DataFrame({'TIP_TIME_ET': [pd.Timestamp('2026-03-03 19:00')] * 3,
                          'AWAY_TEAM': ['A', 'B', 'C'], 'HOME_TEAM': ['X', 'Y', 'Z'],
                          'STATUS': ['scheduled'] * 3, 'MODEL_HOME_PROB': [0.60, 0.50, 0.70],
                          'MARKET_HOME_PROB': [0.52, 0.51, None], 'GAP': [0.08, -0.01, None],
                          'HOME_OUT': [''] * 3, 'AWAY_OUT': [''] * 3})
    styler = slate_table(slate)
    styler._compute()
    assert sorted({r for (r, _), css in styler.ctx.items() if css}) == [0]
