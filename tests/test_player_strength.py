import numpy as np
import pandas as pd
import pytest

import player_strength as ps


def test_minutes_go_to_regulars_first_then_scale_up_when_short():
    m = ps.allocate_minutes([36, 34, 32, 30, 28, 26, 24, 20, 14, 8])   # 252 usual minutes
    assert m.sum() == pytest.approx(240)
    assert list(m[:8]) == [36, 34, 32, 30, 28, 26, 24, 20]             # regulars: 230 minutes
    assert m[8] == pytest.approx(10) and m[9] == pytest.approx(0)       # 10 left; deep bench gets none
    short = ps.allocate_minutes([30, 30, 30, 30, 30, 30])               # 180: everyone scaled up
    assert np.allclose(short, 40)


def _box(rows):
    df = pd.DataFrame(rows, columns=['PLAYER_NAME', 'TEAM_NAME', 'GAME_DATE', 'MIN', 'PTS'])
    for c in ('REB', 'AST', 'STL', 'BLK', 'DREB', 'TOV'):
        df[c] = 0
    return df


def test_strength_counts_available_players_and_moves_with_trades():
    days = pd.date_range('2026-11-01', periods=6, freq='2D')
    rows = []
    for d in days:
        rows += [('Star', 'A', d, 36, 36), ('Role', 'A', d, 30, 12), ('Bench', 'A', d, 20, 6),
                 ('Other', 'B', d, 36, 18)]
    box = _box(rows)
    games = pd.DataFrame({'TEAM_NAME': ['A'] * 6 + ['B'] * 6, 'GAME_DATE': list(days) * 2})
    model = ps.PlayerStrength(box, games, injury_model=None, half_life=1e9, prior_minutes=0, replacement=0.5)
    after = days[-1] + pd.Timedelta(days=2)

    # values: points per minute (1.0, 0.4, 0.3); 86 usual minutes, scaled up to 240
    full, d = model.team_strength('A', after, out=[])
    assert full == pytest.approx((36 * 1.0 + 30 * 0.4 + 20 * 0.3) * 240 / 86)
    without, _ = model.team_strength('A', after, out=['Star'])
    assert without == pytest.approx((30 * 0.4 + 20 * 0.3) * 240 / 50)     # his minutes go to teammates
    assert without < full

    # Star is traded: his latest game is for B, so he now counts for B and not for A
    traded = pd.concat([box, _box([('Star', 'B', after, 36, 36)])], ignore_index=True)
    games2 = pd.concat([games, pd.DataFrame({'TEAM_NAME': ['B'], 'GAME_DATE': [after]})], ignore_index=True)
    model2 = ps.PlayerStrength(traded, games2, injury_model=None, half_life=1e9, prior_minutes=0, replacement=0.5)
    later = after + pd.Timedelta(days=2)
    assert 'Star' not in model2.candidates('A', later) and 'Star' in model2.candidates('B', later)


def test_values_use_only_earlier_games():
    days = pd.date_range('2026-11-01', periods=3, freq='2D')
    box = _box([('P', 'A', days[0], 30, 30), ('P', 'A', days[1], 30, 30), ('P', 'A', days[2], 30, 90)])
    games = pd.DataFrame({'TEAM_NAME': ['A'] * 3, 'GAME_DATE': days})
    model = ps.PlayerStrength(box, games, injury_model=None, half_life=1e9, prior_minutes=0, replacement=0.5)
    value, minutes, team = model.player('P', days[2])        # the 90-point game is not seen yet
    assert value == pytest.approx(1.0) and team == 'A'
