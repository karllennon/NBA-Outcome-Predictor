from datetime import date
import pandas as pd
from ingest import current_season, seasons_between, seasons_to_fetch, merge, normalize, TEAM_KEY


def test_current_season_rolls_over_in_august():
    assert current_season(date(2026, 3, 3)) == '2025-26'
    assert current_season(date(2026, 7, 31)) == '2025-26'
    assert current_season(date(2026, 8, 1)) == '2026-27'
    assert current_season(date(2026, 10, 4)) == '2026-27'


def test_seasons_between():
    assert seasons_between('2018-19', '2020-21') == ['2018-19', '2019-20', '2020-21']
    assert seasons_between('1999-00', '2000-01') == ['1999-00', '2000-01']


def test_incremental_fetches_latest_on_disk_and_current():
    existing = pd.DataFrame({'SEASON_ID': [22024, 22025]})
    assert seasons_to_fetch(existing, today=date(2026, 10, 4)) == ['2025-26', '2026-27']
    assert seasons_to_fetch(existing, today=date(2026, 3, 1)) == ['2025-26']
    assert seasons_to_fetch(existing, since='2023-24', today=date(2026, 3, 1)) == \
        ['2023-24', '2024-25', '2025-26']


def test_merge_dedupes_and_prefers_new_rows():
    old = normalize(pd.DataFrame({'GAME_ID': [22300061, 22300061], 'TEAM_ID': [1, 2],
                                  'GAME_DATE': ['2023-10-24'] * 2, 'PTS': [100, 90]}))
    new = normalize(pd.DataFrame({'GAME_ID': ['0022300061', '0022300062'], 'TEAM_ID': [1, 1],
                                  'GAME_DATE': ['2023-10-24', '2023-10-26'], 'PTS': [101, 95]}))
    out = merge(old, new, TEAM_KEY)
    assert len(out) == 3
    assert out['GAME_ID'].str.len().eq(10).all()
    assert out.loc[(out.GAME_ID == '0022300061') & (out.TEAM_ID == 1), 'PTS'].item() == 101
