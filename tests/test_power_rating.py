import numpy as np
import pandas as pd
import pytest

import power_rating as PR


def _games(n, margin=10.0, start='2026-10-21'):
    """n games two days apart in which team 1 beats team 2 by `margin`, alternating home and away
    (so the result reflects team strength, not home court)."""
    days = pd.date_range(start, periods=n, freq='2D')
    home_is_1 = np.arange(n) % 2 == 0
    return pd.DataFrame({'GAME_ID': [f'g{i}' for i in range(n)], 'GAME_DATE': days,
                         'HOME_ID': np.where(home_is_1, 1, 2), 'AWAY_ID': np.where(home_is_1, 2, 1),
                         'MARGIN': np.where(home_is_1, margin, -margin), 'NEUTRAL': False})


def test_prior_sets_the_ratings_before_any_game_and_results_take_over():
    prior = {1: -4.0, 2: 4.0}                     # roster says team 2 is better
    first = _games(1)
    r = PR.pregame_ratings(first, half_life=60, lam=5.0, priors={first['GAME_DATE'][0]: prior})
    assert r.loc['g0', 'RATING_HOME'] == pytest.approx(-4.0) and r.loc['g0', 'RATING_AWAY'] == pytest.approx(4.0)

    many = _games(60)                             # team 1 keeps winning by 10
    priors = {d: prior for d in many['GAME_DATE']}
    with_prior = PR.pregame_ratings(many, half_life=60, lam=5.0, priors=priors)
    without = PR.pregame_ratings(many, half_life=60, lam=5.0)
    last = 'g58'                                  # team 1 at home
    assert with_prior.loc[last, 'RATING_HOME'] > with_prior.loc[last, 'RATING_AWAY']   # results win out
    gap_early = abs(with_prior.loc['g2', 'PRED_MARGIN'] - without.loc['g2', 'PRED_MARGIN'])
    gap_late = abs(with_prior.loc[last, 'PRED_MARGIN'] - without.loc[last, 'PRED_MARGIN'])
    assert gap_late < gap_early                                                        # prior fades


def test_live_rating_matches_training_with_a_prior():
    games = _games(12)
    priors = {d: {1: 1.5, 2: -1.5} for d in games['GAME_DATE']}
    trained = PR.pregame_ratings(games, half_life=60, lam=5.0, priors=priors)
    day = games['GAME_DATE'][8]
    live, hca = PR.ratings_as_of(games, day, half_life=60, lam=5.0, prior=priors[day])
    assert live[1] == pytest.approx(trained.loc['g8', 'RATING_HOME'], abs=1e-6)    # g8: team 1 at home
    assert live[2] == pytest.approx(trained.loc['g8', 'RATING_AWAY'], abs=1e-6)
