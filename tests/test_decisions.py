import numpy as np
import decisions as D


def test_kalshi_fee_matches_the_published_range():
    # fee schedule: 100 contracts cost $0.07 to $1.75 in fees (P = 1 cent to 50 cents)
    assert np.isclose(D.fee(0.50, contracts=100) * 100, 1.75)
    assert np.isclose(D.fee(0.01, contracts=100) * 100, 0.07)
    assert D.fee(0.50) == 0.02           # a single contract rounds up to the cent
    assert D.fee(0.0) == 0.0 and D.fee(1.0) == 0.0


def test_decide_picks_the_better_side_and_respects_the_threshold():
    d = D.decide(0.62, 0.52, 0.54)               # YES at 53c for a 62% chance: +9 - 2 fee = +7
    assert d['side'] == 'YES' and np.isclose(d['edge_pts'], 7.0) and d['fill'] == 0.54
    d = D.decide(0.40, 0.52, 0.54)               # NO costs 47c for a 60% chance
    assert d['side'] == 'NO' and np.isclose(d['fill'], 0.48)
    d = D.decide(0.56, 0.52, 0.54)               # +1 after fees: no play, best side still reported
    assert d['side'] is None and d['best_side'] == 'YES'
    assert D.decide(0.62, None, None)['side'] is None


def test_likely_winner_bad_price_flag():
    flags = D.bad_price_flags(0.75, 0.80, 'BOS', 'LAL')
    assert flags == ['Likely winner, bad price: BOS 75% to win but costs 80¢']
    assert D.bad_price_flags(0.75, 0.70, 'BOS', 'LAL') == []


def test_large_edges_get_a_late_news_warning():
    g = {'HOME_ABBR': 'BOS', 'AWAY_ABBR': 'LAL', 'MODEL_HOME_PROB': 0.80, 'MARKET_YES_BID': 0.50,
         'MARKET_YES_ASK': 0.52, 'MODEL_HOME_MARGIN': 8.0, 'SPREAD_SIGMA': 13.0, 'MARKET_SPREAD_INFO': None}
    d = D.game_decisions(g)
    assert d['moneyline']['label'] == 'Lean BOS'
    assert any('over 15 pts' in w for w in d['warnings'])


def test_spread_market_probability_for_home_and_away_favorites():
    # home predicted by 6: P(home wins by over 3.5) > 0.5; if the away team is the favorite on the
    # ladder, P(away wins by over 3.5) must be small
    assert D.spread_market_prob(6.0, 13.0, True, 3.5) > 0.5
    assert D.spread_market_prob(6.0, 13.0, False, 3.5) < 0.3
