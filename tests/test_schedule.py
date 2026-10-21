import json
import pandas as pd
import pytest
from nba_api.stats.endpoints import scoreboardv2

import schedule


def _fake_scoreboard(failures):
    """ScoreboardV2 stand-in that raises a JSON error for the first `failures` calls."""
    calls = {'n': 0}

    class Fake:
        def __init__(self, game_date, timeout):
            calls['n'] += 1
            if calls['n'] <= failures:
                raise json.JSONDecodeError('Expecting value', '', 0)
            self.game_header = type('H', (), {'get_data_frame': lambda s: pd.DataFrame({'GAME_ID': ['1']})})()
    return Fake, calls


def test_scoreboard_retries_error_pages_then_succeeds(monkeypatch):
    fake, calls = _fake_scoreboard(failures=2)
    monkeypatch.setattr(scoreboardv2, 'ScoreboardV2', fake)
    waits = []
    header = schedule._game_header('2026-10-21', sleep=waits.append)
    assert list(header['GAME_ID']) == ['1']
    assert calls['n'] == 3 and waits == [10, 20]


def test_scoreboard_gives_up_after_the_last_attempt(monkeypatch):
    fake, calls = _fake_scoreboard(failures=99)
    monkeypatch.setattr(scoreboardv2, 'ScoreboardV2', fake)

    def v3_down(d):
        raise ValueError('also down')
    with pytest.raises(ValueError):
        schedule._game_header('2026-10-21', sleep=lambda s: None, sources=(schedule._v2_header, v3_down))
    assert calls['n'] == schedule.RETRIES


def _header(status_ids, texts):
    return pd.DataFrame({'GAME_ID': ['0022600001', '0022600002'], 'HOME_TEAM_ID': [1, 2],
                         'VISITOR_TEAM_ID': [3, 4], 'GAME_STATUS_ID': status_ids, 'GAME_STATUS_TEXT': texts})


TEAMS = {1: 'Charlotte Hornets', 2: 'Boston Celtics', 3: 'Brooklyn Nets', 4: 'Miami Heat'}


def test_scoreboard_outage_falls_back_to_the_cached_schedule(monkeypatch, tmp_path):
    cache = str(tmp_path / 'cache')
    monkeypatch.setattr(schedule, '_game_header',
                        lambda d: _header([1, 1], ['7:00 pm ET', '7:30 pm ET']))
    fresh = schedule.games_on('2026-10-21', TEAMS, cache_dir=cache)
    assert list(fresh['STATUS']) == ['scheduled', 'scheduled']

    def down(d):
        raise ValueError('error page')
    monkeypatch.setattr(schedule, '_game_header', down)
    # 7:15 PM: the 7:00 game has started, the 7:30 game has not
    cached = schedule.games_on('2026-10-21', TEAMS, cache_dir=cache, now='2026-10-21 19:15')
    assert list(cached['GAME_ID']) == ['0022600001', '0022600002']
    assert list(cached['STATUS']) == ['unknown', 'scheduled']
    assert cached['TIP_TIME_ET'].iloc[1] == pd.Timestamp('2026-10-21 19:30')
    with pytest.raises(ValueError):                       # no cache for another date
        schedule.games_on('2026-10-22', TEAMS, cache_dir=cache)


def test_cache_keeps_tip_times_after_games_start(monkeypatch, tmp_path):
    cache = str(tmp_path / 'cache')
    monkeypatch.setattr(schedule, '_game_header', lambda d: _header([1, 1], ['7:00 pm ET', '7:30 pm ET']))
    schedule.games_on('2026-10-21', TEAMS, cache_dir=cache)
    # later fetch: game 1 under way, so the scoreboard shows 'Q2 5:12' instead of a tip time
    monkeypatch.setattr(schedule, '_game_header', lambda d: _header([2, 1], ['Q2 5:12', '7:30 pm ET']))
    schedule.games_on('2026-10-21', TEAMS, cache_dir=cache)
    saved = schedule._read_cache('2026-10-21', cache)
    assert saved['TIP_TIME_ET'].iloc[0] == pd.Timestamp('2026-10-21 19:00')


def test_prefetch_reports_a_failed_date_and_never_caches_stale_data(monkeypatch, tmp_path):
    cache = str(tmp_path / 'cache')
    monkeypatch.setattr(schedule, '_game_header', lambda d: _header([1, 1], ['7:00 pm ET', '7:30 pm ET']))
    import nba_api.stats.static.teams as t
    monkeypatch.setattr(t, 'get_teams', lambda: [{'id': k, 'full_name': v} for k, v in TEAMS.items()])
    assert schedule.prefetch(1, start='2026-10-21', cache_dir=cache) == {'2026-10-21': 2}

    def down(d):
        raise ValueError('error page')
    monkeypatch.setattr(schedule, '_game_header', down)
    # the cache exists, but a prefetch must report the outage rather than return the cached copy
    assert schedule.prefetch(1, start='2026-10-21', cache_dir=cache) == {'2026-10-21': 'failed (ValueError)'}


def test_scoreboard_falls_back_to_v3_when_v2_keeps_failing():
    calls = []

    def v2(d):
        calls.append('v2')
        raise json.JSONDecodeError('Expecting value', '', 0)

    def v3(d):
        calls.append('v3')
        return pd.DataFrame({'GAME_ID': ['1']})
    out = schedule._game_header('2026-10-21', sleep=lambda s: None, sources=(v2, v3))
    assert list(out['GAME_ID']) == ['1'] and calls == ['v2'] * schedule.RETRIES + ['v3']

    def v3_down(d):
        raise ValueError('also down')
    with pytest.raises(ValueError):
        schedule._game_header('2026-10-21', sleep=lambda s: None, sources=(v2, v3_down))
