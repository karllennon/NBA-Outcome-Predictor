import pandas as pd
import pytest

import rosters
import player_strength as ps


def test_season_label():
    assert rosters.season_for('2026-10-14') == '2026-27'
    assert rosters.season_for('2027-03-01') == '2026-27'
    assert rosters.season_for('2026-08-31') == '2025-26'


def test_roster_lookup_uses_the_latest_recent_snapshot_only():
    archive = pd.DataFrame({'SNAPSHOT_DATE': pd.to_datetime(['2026-10-14', '2026-10-14', '2026-10-16']),
                            'TEAM_ID': [1, 1, 1], 'PLAYER_ID': [10, 11, 10]})
    r = rosters.Rosters(archive, max_age_days=3)
    assert r.on(1, '2026-10-13') is None                 # before the first snapshot
    assert r.on(1, '2026-10-15') == {10, 11}             # the 10-14 snapshot, taken before
    assert r.on(1, '2026-10-16') == {10}                 # same-day snapshot (taken in the morning)
    assert r.on(1, '2026-10-20') is None                 # 4 days old: not trusted
    assert r.on(2, '2026-10-16') is None                 # team missing from the snapshot


def test_snapshot_saves_once_and_never_partially(tmp_path, monkeypatch):
    import nba_api.stats.static.teams as t
    monkeypatch.setattr(t, 'get_teams', lambda: [{'id': 1}, {'id': 2}])
    calls = []

    def fetch(team_id, season):
        calls.append((team_id, season))
        return pd.DataFrame({'PLAYER_ID': [team_id * 10], 'PLAYER': [f'P{team_id}'], 'POSITION': ['G'],
                             'HOW_ACQUIRED': ['Signed']})
    snap = rosters.snapshot('2026-10-14', str(tmp_path), pause=0, fetch=fetch)
    assert len(snap) == 2 and calls == [(1, '2026-27'), (2, '2026-27')]
    rosters.snapshot('2026-10-14', str(tmp_path), pause=0, fetch=fetch)
    assert len(calls) == 2                                # already taken today: not fetched again

    def broken(team_id, season):
        raise ValueError('error page')
    with pytest.raises(ValueError):
        rosters.snapshot('2026-10-15', str(tmp_path), pause=0, fetch=broken)
    assert not (tmp_path / '2026-10-15.csv').exists()     # no partial snapshot


def test_player_strength_follows_the_roster_when_one_covers_the_date():
    days = pd.date_range('2026-03-01', periods=12, freq='2D')
    rows = [{'PLAYER_ID': 1, 'PLAYER_NAME': 'Mover', 'TEAM_NAME': 'Old', 'GAME_DATE': d, 'MIN': 34, 'PTS': 30}
            for d in days]
    rows += [{'PLAYER_ID': 2, 'PLAYER_NAME': 'Stayer', 'TEAM_NAME': 'Old', 'GAME_DATE': d, 'MIN': 30, 'PTS': 12}
             for d in days]
    box = pd.DataFrame(rows)
    for c in ('REB', 'AST', 'STL', 'BLK', 'DREB', 'TOV'):
        box[c] = 0
    games = pd.DataFrame({'TEAM_NAME': ['Old'] * 12 + ['New'] * 12, 'TEAM_ID': [100] * 12 + [200] * 12,
                          'GAME_DATE': list(days) * 2})
    opener = pd.Timestamp('2026-10-21')
    # Mover signed with New in the offseason: on New's roster, not on Old's
    snap = pd.DataFrame({'SNAPSHOT_DATE': pd.to_datetime(['2026-10-20'] * 3), 'TEAM_ID': [200, 100, 200],
                         'PLAYER_ID': [1, 2, 3]})                        # 3 = a rookie with no NBA games
    model = ps.PlayerStrength(box, games, injury_model=None, half_life=1e9, prior_minutes=0, replacement=0.5,
                              rosters=rosters.Rosters(snap))
    new, old = model.candidates('New', opener), model.candidates('Old', opener)
    assert 'Mover' in new and 'Mover' not in old and 'Stayer' in old
    assert new['PLAYER_ID 3'] == (0.5, 0.0)                              # replacement value, no minutes yet
    without = ps.PlayerStrength(box, games, injury_model=None, half_life=1e9, prior_minutes=0, replacement=0.5)
    assert 'Mover' in without.candidates('Old', opener)                  # no snapshot: box-score rule
