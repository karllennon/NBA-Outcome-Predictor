import math
import pandas as pd
import market_odds as mo
from schedule import parse_status_tip


def test_event_ticker_round_trip():
    assert mo.event_ticker('2026-03-03', 'DAL', 'CHA') == 'KXNBAGAME-26MAR03DALCHA'
    day, away, home = mo.parse_event_ticker('KXNBAGAME-26MAR03DALCHA')
    assert (day, away, home) == (pd.Timestamp('2026-03-03'), 'DAL', 'CHA')
    assert mo.parse_event_ticker('KXNBAGAME-25DEC25LALGSW')[0] == pd.Timestamp('2025-12-25')


def test_midpoint():
    assert mo.midpoint(0.49, 0.51) == 0.5
    assert mo.midpoint(0.00, 0.01) == 0.005      # one-sided extreme is still a price
    assert math.isnan(mo.midpoint(0.0, 1.0))     # empty book
    assert math.isnan(mo.midpoint(0.2, 0.6))     # too wide to mean anything
    assert math.isnan(mo.midpoint(0.6, 0.5))     # crossed


def test_scoreboard_tip_parsing():
    assert parse_status_tip('2026-10-05', '7:00 pm ET') == pd.Timestamp('2026-10-05 19:00')
    assert parse_status_tip('2026-10-05', '12:30 pm ET') == pd.Timestamp('2026-10-05 12:30')
    assert parse_status_tip('2026-10-05', '11:00 am ET') == pd.Timestamp('2026-10-05 11:00')
    assert parse_status_tip('2026-10-05', 'Final') is None


def test_pregame_quote_ignores_candles_after_tip(monkeypatch):
    tip = pd.Timestamp('2026-03-03 19:00').tz_localize('America/New_York')
    end = int(tip.timestamp())

    def fake_get(path, params=None):
        return {'candlesticks': [
            {'end_period_ts': end - 3600, 'yes_bid': {'close': '0.60'}, 'yes_ask': {'close': '0.62'}},
            {'end_period_ts': end, 'yes_bid': {'close': '0.64'}, 'yes_ask': {'close': '0.65'}},
            {'end_period_ts': end + 3600, 'yes_bid': {'close': '0.98'}, 'yes_ask': {'close': '0.99'}},
        ]}

    monkeypatch.setattr(mo, '_get', fake_get)
    assert mo.pregame_quote('X', tip, True) == (0.64, 0.65)


def test_tip_quote_uses_the_last_minute_before_tip_then_falls_back_to_hourly(monkeypatch):
    tip = pd.Timestamp('2026-11-03 19:30').tz_localize('America/New_York')
    end = int(tip.timestamp())
    calls = []

    def fake_get(path, params=None):
        calls.append(params['period_interval'])
        if params['period_interval'] == 1:
            return {'candlesticks': [
                {'end_period_ts': end - 120, 'yes_bid': {'close': '0.40'}, 'yes_ask': {'close': '0.41'}},
                {'end_period_ts': end, 'yes_bid': {'close': '0.42'}, 'yes_ask': {'close': '0.43'}},
                {'end_period_ts': end + 60, 'yes_bid': {'close': '0.90'}, 'yes_ask': {'close': '0.91'}},
            ]}
        return {'candlesticks': [{'end_period_ts': end - 1800, 'yes_bid': {'close': '0.38'}, 'yes_ask': {'close': '0.39'}}]}

    monkeypatch.setattr(mo, '_get', fake_get)
    assert mo.tip_quote('X', tip, False) == (0.42, 0.43, '1-minute')
    monkeypatch.setattr(mo, '_get', lambda path, params=None: (
        {'candlesticks': []} if params['period_interval'] == 1 else fake_get(path, params)))
    assert mo.tip_quote('X', tip, False) == (0.38, 0.39, 'hourly')


def test_build_tip_prices_fetches_only_new_games_that_have_tipped(monkeypatch, tmp_path):
    def row(gid, tip, logged):
        return {'GAME_ID': gid, 'GAME_DATE': tip[:10], 'TIP_TIME_ET': tip, 'LOGGED_AT_UTC': pd.Timestamp(logged),
                'HOME_ABBR': 'CHA', 'AWAY_ABBR': 'BKN'}
    log = pd.DataFrame([row('0022600001', '2026-11-03 19:00', '2026-11-03 22:00'),
                        row('0022600002', '2026-11-03 19:30', '2026-11-03 22:30'),
                        row('0022600003', '2026-11-04 19:00', '2026-11-04 22:00'),   # not tipped yet
                        row('0012600004', '2026-11-03 19:00', '2026-11-03 22:00')])  # preseason
    path = tmp_path / 'tip_prices.csv'
    pd.DataFrame([{'GAME_ID': '0022600001', 'TIP_TIME_ET': '2026-11-03 19:00', 'TIP_HOME_MID': 0.5}]).reindex(
        columns=mo.TIP_PRICE_COLUMNS).to_csv(path, index=False)
    asked = []
    monkeypatch.setattr(mo, '_get', lambda p, params=None: {'market_settled_ts': '2026-08-07T00:00:00Z'})
    monkeypatch.setattr(mo, 'tip_quote', lambda t, tip, hist: (asked.append((t, hist)) or (0.60, 0.62, '1-minute')))
    out = mo.build_tip_prices(log, str(path), now='2026-11-03 23:00', pause=0)
    assert asked == [('KXNBAGAME-26NOV03BKNCHA-CHA', False)]
    assert set(out['GAME_ID']) == {'0022600001', '0022600002'}
    assert out.set_index('GAME_ID').loc['0022600002', 'TIP_HOME_MID'] == 0.61
