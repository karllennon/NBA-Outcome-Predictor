import numpy as np
import pandas as pd
import pytest
import paper_report as R


def _log_row(gid, tip, logged_min_before, prob, bid, ask, pick_side=None, dog=False, pick_p=None,
             home='CHA', away='BKN'):
    tip = pd.Timestamp(tip)
    logged = (tip - pd.Timedelta(minutes=logged_min_before)).tz_localize(R.ET).tz_convert('UTC').tz_localize(None)
    mid = (bid + ask) / 2
    side = pick_side or ('YES' if prob >= 0.5 else 'NO')
    home_pick = side == 'YES'
    return {'LOGGED_AT_UTC': logged, 'GAME_ID': gid, 'GAME_DATE': tip.normalize(), 'TIP_TIME_ET': tip,
            'HOME_ABBR': home, 'AWAY_ABBR': away, 'MODEL_HOME_PROB': prob,
            'MARKET_YES_BID': bid, 'MARKET_YES_ASK': ask, 'MODEL_VERSION': 'test',
            'PICK_TEAM': home if home_pick else away, 'PICK_SIDE': side,
            'PICK_P': pick_p if pick_p is not None else (prob if home_pick else 1 - prob),
            'PICK_MID': mid if home_pick else 1 - mid, 'PICK_FILL': ask if home_pick else 1 - bid,
            'PICK_MARKET_UNDERDOG': dog, 'SPREAD_SIDE': None}


def _games(tmp_path, margins):
    rows = []
    for gid, m in margins.items():
        rows += [{'GAME_ID': gid, 'MATCHUP': 'CHA vs. BKN', 'PLUS_MINUS': m},
                 {'GAME_ID': gid, 'MATCHUP': 'BKN @ CHA', 'PLUS_MINUS': -m}]
    path = tmp_path / 'games.csv'
    pd.DataFrame(rows, columns=['GAME_ID', 'MATCHUP', 'PLUS_MINUS']).to_csv(path, index=False)
    return str(path)


def test_entry_is_the_last_row_30_minutes_before_tip_and_close_is_the_last_pregame_row():
    log = pd.DataFrame([
        _log_row('0022600001', '2026-10-21 19:00', 90, 0.60, 0.55, 0.57),
        _log_row('0022600001', '2026-10-21 19:00', 40, 0.62, 0.58, 0.60),   # entry
        _log_row('0022600001', '2026-10-21 19:00', 10, 0.63, 0.61, 0.63),   # close
        _log_row('0012600002', '2026-10-21 19:00', 60, 0.60, 0.55, 0.57),   # preseason: ignored
    ])
    entry, cov = R.entries(log, since='2026-10-20')
    assert len(entry) == 1 and cov['games logged'] == 1
    assert entry['MODEL_HOME_PROB'].iloc[0] == 0.62
    assert entry['HOME_MID'].iloc[0] == pytest.approx(0.59)
    assert entry['CLOSE_HOME_MID'].iloc[0] == pytest.approx(0.62)


def test_game_logged_only_inside_30_minutes_has_no_entry():
    log = pd.DataFrame([_log_row('0022600001', '2026-10-21 19:00', 20, 0.6, 0.55, 0.57)])
    entry, cov = R.entries(log, since='2026-10-20')
    assert entry.empty and cov['logged only within 30 min of tip'] == 1


def test_prizepicks_pricing_matches_observed_multipliers():
    # Heat 34c ask -> about 2.7x in PrizePicks' Play-In example
    assert R.prizepicks_multiplier(0.34) == pytest.approx(2.66, abs=0.05)
    # a lineup is charged fees once, so it pays more than multiplying two single multipliers
    assert R.prizepicks_multiplier(0.5 * 0.5) > R.prizepicks_multiplier(0.5) ** 2


def test_strategy_legs_and_lineups_are_built_in_tip_off_order(tmp_path):
    log = pd.DataFrame([
        # model likes the home team at 55%, market has it as a 45c underdog: S1, S4, S5 leg; S3 is the away team
        _log_row('0022600001', '2026-10-21 19:00', 60, 0.55, 0.44, 0.46, dog=True),
        # heavy favorite at home, model agrees but below the market: S3, S3a, S4 only
        _log_row('0022600002', '2026-10-21 19:30', 60, 0.85, 0.91, 0.93),
        # model above the market on a favorite: S3, S4, S5 leg
        _log_row('0022600003', '2026-10-21 22:00', 60, 0.75, 0.68, 0.70),
        _log_row('0022600004', '2026-10-21 20:00', 60, 0.70, 0.64, 0.66),
    ])
    margins = {'0022600001': 5, '0022600002': 12, '0022600003': -3, '0022600004': 8}
    entry, _ = R.entries(log, since='2026-10-20')
    L = R.legs(entry, R.results(_games(tmp_path, margins)))
    by = {k: set(g['GAME_ID']) for k, g in L.groupby('strategy')}
    assert by['S1'] == {'0022600001'}
    assert by['S3a'] == {'0022600002'}
    assert by['S5/S6 legs'] == {'0022600001', '0022600003', '0022600004'}
    s3 = L[(L['strategy'] == 'S3') & (L['GAME_ID'] == '0022600001')].iloc[0]
    assert s3['team'] == 'BKN' and not s3['won'] and s3['fill'] == pytest.approx(1 - 0.44)
    # 3-pick lineup takes the first three by tip-off: 19:00, 20:00, 22:00 -> game 3 lost
    lu = R.lineups(L[L['strategy'] == 'S5/S6 legs'], 3)
    assert len(lu) == 1 and not lu['won'].iloc[0]
    assert lu['price'].iloc[0] == pytest.approx(0.46 * 0.66 * 0.70)
    two = R.lineups(L[L['strategy'] == 'S5/S6 legs'], 2)
    assert two['won'].iloc[0]                           # games 1 (19:00) and 4 (20:00)


def test_unplayed_games_are_pending_not_losses(tmp_path):
    log = pd.DataFrame([_log_row('0022600001', '2026-10-21 19:00', 60, 0.55, 0.44, 0.46, dog=True)])
    tables = R.build(log, _games(tmp_path, {}), since='2026-10-20', tip_path=str(tmp_path / 'none.csv'))
    assert tables['coverage']['legs pending (unplayed)'] > 0
    assert tables['singles'].set_index('id').loc['S1', 'bets'] == 0


def test_decision_rules_follow_the_plan():
    s = lambda n, mean, t: {'n': n, 'mean': mean, 't': t}
    assert R.status('S1', s(99, -0.10, -3)).startswith('running (99/100')
    assert R.status('S1', s(100, -0.03, -1.5)) == 'STOP (dead)'
    assert R.status('S1', s(150, 0.05, 2.1)) == 'promising (not proven)'
    assert R.status('S1', s(120, 0.01, 0.3), {'n': 60, 'mean': 1.0, 't': 2.5}).startswith('promising on CLV')
    assert R.status('S5', s(100, -0.15, -1)) == 'STOP (dead)'
    assert R.status('S6', s(5, 0.5, 1)) == 'watch only'
    assert R.status('S3', s(500, -0.1, -5)) == 'yardstick'
