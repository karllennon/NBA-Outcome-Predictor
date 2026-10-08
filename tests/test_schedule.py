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
    with pytest.raises(ValueError):
        schedule._game_header('2026-10-21', sleep=lambda s: None)
    assert calls['n'] == schedule.RETRIES
