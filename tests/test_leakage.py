"""
Leakage tests: a feature for a game may only use information available before tip-off.

Each test changes the box scores of every game on or after a cutoff date (scores, results,
who played) and asserts that features for games up to and including the cutoff date do not
move. Games ON the cutoff date are included because their own box score is not known before
tip-off either.
"""
import numpy as np
import pandas as pd
import pytest
from data_pipeline import load_raw, build_dataset
from elo import NBAEloCalculator
from features import NBAFeatureProcessor
from injuries import InjuryModel
from matchups import FEATURES

CUTOFF = pd.Timestamp('2023-12-20')
END = pd.Timestamp('2024-01-10')


@pytest.fixture(scope='module')
def raw():
    games, players = load_raw()
    games = games[games['GAME_DATE'] <= END].copy()
    players = players[players['GAME_DATE'] <= END].copy()
    return games, players


def perturb(games, players, cutoff=CUTOFF):
    """Rewrite every box score on or after `cutoff`."""
    g, p = games.copy(), players.copy()
    gm = g['GAME_DATE'] >= cutoff
    for col in ['PTS', 'FGM', 'FGA', 'FG3M', 'FTM', 'FTA', 'OREB', 'DREB', 'TOV', 'STL', 'BLK']:
        g.loc[gm, col] = (g.loc[gm, col] * 1.3 + 3).round().astype(g[col].dtype)
    g.loc[gm, 'PLUS_MINUS'] = -g.loc[gm, 'PLUS_MINUS'] * 2
    g.loc[gm, 'WL'] = g.loc[gm, 'WL'].map({'W': 'L', 'L': 'W'})
    pm = p['GAME_DATE'] >= cutoff
    for col in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'DREB', 'TOV']:
        p.loc[pm, col] = p.loc[pm, col] * 2 + 5
    # Change who played: every third player row on/after the cutoff now did not play
    idx = p.index[pm][::3]
    p.loc[idx, 'MIN'] = 0
    return g, p


def frame_equal_before(a, b, key, cols, cutoff=CUTOFF):
    a = a[a['GAME_DATE'] <= cutoff].set_index(key)[cols].sort_index()
    b = b[b['GAME_DATE'] <= cutoff].set_index(key)[cols].sort_index()
    assert len(a) > 0 and a.index.equals(b.index)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


def test_perturbation_changes_later_features(raw):
    """Sanity check: the perturbation does move features after the cutoff."""
    games, players = raw
    g2, p2 = perturb(games, players)
    a, _, _ = build_dataset(games, players, verbose=False)
    b, _, _ = build_dataset(g2, p2, verbose=False)
    after_a = a[a['GAME_DATE'] > CUTOFF + pd.Timedelta(days=3)].set_index('GAME_ID')[FEATURES]
    after_b = b[b['GAME_DATE'] > CUTOFF + pd.Timedelta(days=3)].set_index('GAME_ID')[FEATURES]
    assert not np.allclose(after_a.values, after_b.loc[after_a.index].values)


def test_elo_is_pre_game(raw):
    games, players = raw
    g2, _ = perturb(games, players)
    a = NBAEloCalculator().process_season(games)
    b = NBAEloCalculator().process_season(g2)
    frame_equal_before(a, b, ['GAME_ID', 'TEAM_ID'], ['PRE_GAME_ELO'])


def test_rolling_stats_use_only_prior_games(raw):
    games, players = raw
    g2, _ = perturb(games, players)

    def rolling(df):
        proc = (NBAFeatureProcessor(df).add_advanced_stats().add_comprehensive_stats()
                .add_context_features().add_rolling_momentum())
        return proc.df

    a, b = rolling(games), rolling(g2)
    cols = [c for c in a.columns if c.startswith('ROLLING_')] + ['WIN_STREAK', 'DAYS_REST', 'IS_B2B']
    frame_equal_before(a, b, ['GAME_ID', 'TEAM_ID'], cols)


def test_injury_loss_uses_only_prior_games(raw):
    games, players = raw
    g2, p2 = perturb(games, players)
    a = InjuryModel(players, games).backfill(games, verbose=False)
    b = InjuryModel(p2, g2).backfill(g2, verbose=False)
    frame_equal_before(a, b, ['GAME_ID', 'TEAM_ID'], ['CORE_INJURY_LOSS'])


def test_full_feature_set_has_no_leakage(raw):
    games, players = raw
    g2, p2 = perturb(games, players)
    a, _, _ = build_dataset(games, players, verbose=False)
    b, _, _ = build_dataset(g2, p2, verbose=False)
    frame_equal_before(a, b, ['GAME_ID'], FEATURES)


def _report(rows):
    return pd.DataFrame([{'REPORT_TIME': pd.Timestamp(rt), 'GAME_DATE': pd.Timestamp(gd),
                          'GAME_TIME': '07:00(ET)', 'TEAM_NAME': team, 'PLAYER_NAME': name,
                          'STATUS': status} for rt, gd, team, name, status in rows])


def test_injury_reports_after_tipoff_are_ignored(raw):
    games, players = raw
    team = 'Denver Nuggets'
    day = games[(games['TEAM_NAME'] == team) & (games['GAME_DATE'] > CUTOFF)]['GAME_DATE'].min()
    star = (players[(players['TEAM_NAME'] == team) & (players['GAME_DATE'] < day)]
            .groupby('PLAYER_NAME')['PTS'].sum().idxmax())
    last, first = star.split(' ', 1)[1], star.split(' ', 1)[0]

    before = _report([(day + pd.Timedelta(hours=17), day, team, f'{last}, {first}', 'Available')])
    after = _report([(day + pd.Timedelta(hours=17), day, team, f'{last}, {first}', 'Available'),
                     (day + pd.Timedelta(hours=19, minutes=5), day, team, f'{last}, {first}', 'Out')])
    future = _report([(day + pd.Timedelta(hours=17), day, team, f'{last}, {first}', 'Available'),
                      (day + pd.Timedelta(days=2, hours=17), day + pd.Timedelta(days=2), team,
                       f'{last}, {first}', 'Out')])

    def loss(reports):
        m = InjuryModel(players, games, reports=reports, use_reports=True)
        return m.historical_loss(team, day)

    base = loss(before)
    assert loss(after) == base     # report published after tip-off is ignored
    assert loss(future) == base    # a later game's report does not affect this game
    # and the report does matter when it is in time
    in_time = _report([(day + pd.Timedelta(hours=18), day, team, f'{last}, {first}', 'Out')])
    assert loss(in_time) > base


def test_elo_tuning_uses_only_games_before_cutoff(raw):
    from elo import elo_tables, tune_elo
    games, players = raw
    g2, _ = perturb(games, players)
    grid = {'k_factor': [10, 20, 30], 'home_advantage': [25, 100], 'season_carryover': [0.5]}
    a, _ = tune_elo(elo_tables(games, grid), CUTOFF)
    b, _ = tune_elo(elo_tables(g2, grid), CUTOFF)
    assert a == b
