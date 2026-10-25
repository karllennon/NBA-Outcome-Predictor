import ui


def test_win_pct_never_reads_50_percent_when_one_side_is_favored():
    assert ui.win_pct(0.504) == '50.4%' and ui.win_pct(0.496) == '49.6%'   # would both read "50%"
    assert ui.win_pct(0.512) == '51%' and ui.win_pct(0.488) == '49%'       # already distinct
    assert ui.win_pct(0.73) == '73%'
    assert ui.win_pct(None) == '–'


def test_every_game_has_a_pick():
    assert ui.pick('SAC', 'MEM', 0.504) == ('SAC', 0.504)
    team, p = ui.pick('SAC', 'MEM', 0.496)
    assert team == 'MEM' and abs(p - 0.504) < 1e-9
    assert ui.pick('SAC', 'MEM', 0.5)[0] == 'SAC'                        # exact tie: home, as in the paper trades

