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
