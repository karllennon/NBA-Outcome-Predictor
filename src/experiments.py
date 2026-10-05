"""
Before/after experiments, all scored with the same walk-forward protocol as train.py.

    python src/experiments.py <experiment> [<experiment> ...]
    python src/experiments.py --list

Each experiment builds one or more variant training sets (or models) and compares them on the
same held-out games against the Elo-only baseline and the current shipped configuration.
Results are printed as a table; RESULTS.md records them with commentary.
"""
import argparse
import pandas as pd
import data_pipeline
from data_pipeline import load_raw, load_injury_reports, build_dataset
from evaluation import walk_forward, score, print_table, markdown_table, paired_delta
from matchups import FEATURES
from train import make_logistic, fold_tuned_elo

EXPERIMENTS = {}


def experiment(fn):
    EXPERIMENTS[fn.__name__] = fn
    return fn


_raw = None


def raw():
    global _raw
    if _raw is None:
        _raw = load_raw()
    return _raw


def dataset(elo_params=None, injury_params=None, reports=None):
    games, players = raw()
    df, _, _ = build_dataset(games, players, elo_params=elo_params,
                             injury_params=injury_params, reports=reports, verbose=False)
    return df


def shipped_dataset():
    reports = load_injury_reports() if data_pipeline.INJURY_PARAMS.get('use_reports') else None
    return dataset(reports=reports)


_tables = {}


def elo_tables_for(games):
    from elo import elo_tables, ELO_GRID
    key = (len(games), str(games['GAME_DATE'].min()))
    if key not in _tables:
        _tables[key] = elo_tables(games, ELO_GRID)
    return _tables[key]


def compare(variants, title, features=FEATURES, test_from=None, games=None, fold_elo=True):
    """
    variants: {name: DataFrame (scored with logistic on `features`) or
               (DataFrame, features) or (DataFrame, callable(train, test) -> probs)}.
    The first variant is the current shipped configuration; 'Elo only' is fit on it.
    All variants are restricted to the games they have in common, so every row of the
    table is scored on exactly the same held-out games.
    fold_elo: re-tune Elo inside each fold on training games (as train.py does) for every row.
    """
    specs = {}
    for name, v in variants.items():
        if isinstance(v, pd.DataFrame):
            v = (v, features)
        specs[name] = v
    common = set.intersection(*[set(df['GAME_ID']) for df, _ in specs.values()])

    preds = None
    first = True
    for name, (df, how) in specs.items():
        df = df[df['GAME_ID'].isin(common)].sort_values(['GAME_DATE', 'GAME_ID']).reset_index(drop=True)
        cands = {}
        if first:
            cands['Elo only (logistic)'] = (make_logistic, ['ELO_DIFF'])
            first = False
        cands[name] = how if callable(how) else (make_logistic, how)
        if fold_elo:
            cands = fold_tuned_elo(cands, tables=elo_tables_for(games if games is not None else raw()[0]),
                                   verbose=False)
        p = walk_forward(df, cands, test_from=test_from)
        if preds is None:
            preds = p
        else:
            preds = preds.merge(p[['GAME_ID', name]], on='GAME_ID')
    names = ['Elo only (logistic)'] + list(specs)
    results = score(preds, names)
    print_table(results, f"{title} ({len(preds)} held-out games)")
    base = list(specs)[0]
    for name in list(specs)[1:]:
        mean, se, better, k = paired_delta(preds, base, name)
        print(f"  {name} vs {base}: log loss {mean:+.4f} (SE {se:.4f}), better in {better}/{k} folds")
    print()
    print(markdown_table(results))
    return results, preds


# ------------------------------------------------------------------ Phase 2

@experiment
def injury_reports():
    reports = load_injury_reports()
    print(f"Archived report rows: {len(reports)}, report times: {reports['REPORT_TIME'].nunique()}")
    return compare({
        'Shipped: guess (missed previous game)': dataset(),
        'Reports: Out': dataset(injury_params={'use_reports': True}, reports=reports),
        'Reports: Out + Doubtful': dataset(injury_params={'use_reports': True, 'include_doubtful': True},
                                           reports=reports),
    }, 'Injury source')


# ------------------------------------------------------------------ Phase 3

def _with(**kw):
    params = dict(data_pipeline.INJURY_PARAMS)
    params.update(kw)
    return params


# ------------------------------------------------------------------ Phase 4

@experiment
def market_comparison(preds_path='data/test_predictions.csv', market_path='data/market_history.csv'):
    """Shipped model's walk-forward held-out predictions vs Kalshi's pre-tip-off price, on the
    held-out games that have a market price (run train.py and market_odds.py --history first)."""
    preds = pd.read_csv(preds_path, dtype={'GAME_ID': str}, parse_dates=['GAME_DATE'])
    preds['GAME_ID'] = preds['GAME_ID'].str.zfill(10)
    market = pd.read_csv(market_path, dtype={'GAME_ID': str})
    df = preds.merge(market[['GAME_ID', 'MARKET_HOME_PROB']], on='GAME_ID')
    print(f"Held-out games: {len(preds)}; with a Kalshi game market: {len(df)}; "
          f"with a pre-tip-off price: {df['MARKET_HOME_PROB'].notna().sum()}")
    df = df.dropna(subset=['MARKET_HOME_PROB']).rename(columns={
        'ELO_PROB': 'Elo only (logistic)', 'MODEL_PROB': 'Shipped model',
        'MARKET_HOME_PROB': 'Kalshi pre-tip-off midpoint'})
    df['FOLD'] = 0
    names = ['Elo only (logistic)', 'Shipped model', 'Kalshi pre-tip-off midpoint']
    results = score(df, names)
    print_table(results, f"Model vs market ({len(df)} games, "
                         f"{df['GAME_DATE'].min().date()} to {df['GAME_DATE'].max().date()})")
    mean, se, _, _ = paired_delta(df, 'Kalshi pre-tip-off midpoint', 'Shipped model')
    print(f"  Shipped model vs market: log loss {mean:+.4f} (SE {se:.4f})")
    blend = 0.5 * df['Shipped model'] + 0.5 * df['Kalshi pre-tip-off midpoint']
    print(f"  (for reference, 50/50 average of model and market: log loss "
          f"{score(df.assign(B=blend), ['B'])['log_loss'].iloc[0]:.4f})")
    print()
    print(markdown_table(results))
    return results, df


# ------------------------------------------------------------------ Phase 5

@experiment
def elo_tuning():
    from elo import elo_tables, tune_elo, ELO_GRID
    games, _ = raw()
    tables = elo_tables(games, ELO_GRID)
    base = shipped_dataset()
    chosen = []

    def tuned(train, test):
        params, diff = tune_elo(tables, test['GAME_DATE'].min())   # training games only
        chosen.append((test['GAME_DATE'].min().date(), params))
        tr, te = train.copy(), test.copy()
        tr['ELO_DIFF'] = tr['GAME_ID'].map(diff)
        te['ELO_DIFF'] = te['GAME_ID'].map(diff)
        model = make_logistic().fit(tr[FEATURES], tr['TARGET'])
        return model.predict_proba(te[FEATURES])[:, 1]

    results = compare({'Before (K=20, HCA=100, carryover=0.75)': dataset(elo_params={}, injury_params=data_pipeline.INJURY_PARAMS, reports=load_injury_reports()),
                       'Elo tuned per fold on training games': (base, tuned)}, 'Elo tuning',
                      fold_elo=False)
    for start, params in chosen:
        print(f"  fold starting {start}: {params}")
    print(f"  tuned on all games (for shipping): {tune_elo(tables, '2100-01-01')[0]}")
    return results


def margin_model(sigma_holdout=0.2):
    """
    Phase 5.2: the spread model (spread_model.py: Ridge on the home margin, sigma from the end of
    the training games) turned into P(home win) = P(margin > 0).
    """
    import spread_model

    def fit_predict(train, test):
        fitted = spread_model.fit_target(spread_model.TARGETS['spread'], train, sigma_holdout)
        fit_predict.sigmas.append(fitted.sigma)
        return fitted.prob_over(test, 0.0)

    fit_predict.sigmas = []
    return fit_predict


@experiment
def margin_regression():
    base = shipped_dataset()
    margin = margin_model()
    results = compare({'Shipped: logistic classifier': base,
                       'Margin regression -> normal CDF': (base, margin)}, 'Margin regression')
    print(f"  fitted sigma per fold: {[round(s, 1) for s in margin.sigmas]}")
    return results


def history_variants(first_season_ids=(22023, 22018)):
    """Datasets built from raw data starting at each season, with the matching raw games."""
    games, players = load_raw(first_season_id=min(first_season_ids))
    reports = load_injury_reports()
    out = {}
    for sid in first_season_ids:
        g = games[games['SEASON_ID'] >= sid]
        p = players[players['GAME_ID'].isin(set(g['GAME_ID']))]
        df, _, _ = build_dataset(g, p, injury_params=data_pipeline.INJURY_PARAMS, reports=reports,
                                 verbose=False)
        out[sid] = (df, g)
    return out


def score_on_same_games(runs, title, base_name):
    """runs: {name: predictions from walk_forward}; scores every run on the held-out games
    they all share."""
    names = list(runs)
    preds = runs[names[0]][['GAME_ID', 'GAME_DATE', 'TARGET', 'FOLD'] + [c for c in runs[names[0]].columns
                                                                         if c.startswith('Elo only')]]
    for name in names:
        cols = [c for c in runs[name].columns if c not in ('GAME_DATE', 'TARGET', 'FOLD')]
        preds = preds.merge(runs[name][cols].rename(columns={'Elo only (logistic)': f'Elo only ({name})'}),
                            on='GAME_ID', suffixes=('', '_dup'))
    preds = preds[[c for c in preds.columns if not c.endswith('_dup')]]
    model_cols = [c for c in preds.columns if c not in ('GAME_ID', 'GAME_DATE', 'TARGET', 'FOLD')]
    results = score(preds, model_cols)
    print_table(results, f"{title} ({len(preds)} held-out games)")
    for c in model_cols:
        if c != base_name and not c.startswith('Elo only'):
            mean, se, better, k = paired_delta(preds, base_name, c)
            print(f"  {c} vs {base_name}: log loss {mean:+.4f} (SE {se:.4f}), better in {better}/{k} folds")
    print()
    print(markdown_table(results))
    return results, preds


@experiment
def more_history():
    """Same held-out games as the shipped evaluation; more seasons of training history."""
    variants = history_variants()
    test_from = variants[22023][0]['GAME_DATE'].min()
    runs = {}
    for sid, label in [(22023, 'Before: 2023-24 on'), (22018, 'After: 2018-19 on')]:
        df, g = variants[sid]
        print(f"{label}: {len(df)} games, {df['GAME_DATE'].min().date()} to {df['GAME_DATE'].max().date()}")
        cands = fold_tuned_elo({'Elo only (logistic)': (make_logistic, ['ELO_DIFF']),
                                label: (make_logistic, FEATURES)}, g, verbose=True)
        runs[label] = walk_forward(df, cands, test_from=test_from)
    return score_on_same_games(runs, 'More history', 'Before: 2023-24 on')


def xgb_early_stopping(max_depth=2, learning_rate=0.03, min_child_weight=20, val_frac=0.15):
    """XGBoost with early stopping on the most recent `val_frac` of the training games."""
    import xgboost as xgb

    def fit_predict(train, test):
        cut = int(len(train) * (1 - val_frac))
        fit, val = train.iloc[:cut], train.iloc[cut:]
        model = xgb.XGBClassifier(n_estimators=3000, max_depth=max_depth, learning_rate=learning_rate,
                                  min_child_weight=min_child_weight, subsample=0.8,
                                  colsample_bytree=0.8, reg_lambda=5.0, random_state=42,
                                  eval_metric='logloss', early_stopping_rounds=100)
        model.fit(fit[FEATURES], fit['TARGET'], eval_set=[(val[FEATURES], val['TARGET'])], verbose=False)
        fit_predict.rounds.append(model.best_iteration)
        return model.predict_proba(test[FEATURES], iteration_range=(0, model.best_iteration + 1))[:, 1]

    fit_predict.rounds = []
    return fit_predict


@experiment
def xgboost_tuned():
    games = data_pipeline.load_games()
    base = shipped_dataset()
    specs = {'Shipped: logistic': base}
    fns = {}
    for depth, mcw in [(2, 20), (3, 30)]:
        fns[f'XGBoost depth {depth}, early stopping'] = xgb_early_stopping(max_depth=depth, min_child_weight=mcw)
        specs[f'XGBoost depth {depth}, early stopping'] = (base, fns[f'XGBoost depth {depth}, early stopping'])
    results = compare(specs, f"XGBoost vs logistic ({base['GAME_DATE'].min().date()} on)", games=games)
    for name, fn in fns.items():
        print(f"  {name}: boosting rounds per fold {fn.rounds}")
    return results


@experiment
def xgboost_more_history():
    """XGBoost (early stopping) vs logistic, both trained on 2018-19 onward, scored on the
    shipped held-out games; the shipped 2023-24-on logistic model is the baseline."""
    variants = history_variants()
    test_from = variants[22023][0]['GAME_DATE'].min()
    runs = {}
    df, g = variants[22023]
    runs['Shipped: logistic, 2023-24 on'] = walk_forward(df, fold_tuned_elo(
        {'Elo only (logistic)': (make_logistic, ['ELO_DIFF']),
         'Shipped: logistic, 2023-24 on': (make_logistic, FEATURES)}, g, verbose=False), test_from=test_from)
    df, g = variants[22018]
    fns = {f'XGBoost depth {d}, 2018-19 on': xgb_early_stopping(max_depth=d, min_child_weight=m)
           for d, m in [(2, 20), (3, 30), (4, 40)]}
    cands = {'Logistic, 2018-19 on': (make_logistic, FEATURES), **fns}
    runs['2018-19 on'] = walk_forward(df, fold_tuned_elo(cands, g, verbose=False), test_from=test_from)
    out = score_on_same_games(runs, 'XGBoost with more history', 'Shipped: logistic, 2023-24 on')
    for name, fn in fns.items():
        print(f"  {name}: boosting rounds per fold {fn.rounds}")
    return out


# ------------------------------------------------------------------ Phase 6

FEATURE_GROUPS = {
    'Travel + time zones': ['TRAVEL_DIFF', 'TZ_SHIFT_DIFF'],
    'Schedule density': ['GAMES_LAST_4_DIFF', 'GAMES_LAST_7_DIFF', 'THREE_IN_FOUR_DIFF'],
    'Altitude (road team at DEN/UTA)': ['AWAY_AT_ALTITUDE'],
    'SOS-adjusted net rating': ['ADJ_NET_DIFF'],
}


@experiment
def new_features(groups=None):
    """Each candidate group added on its own to the other shipped features."""
    games = data_pipeline.load_games()
    base = shipped_dataset()
    all_extra = [c for cols in FEATURE_GROUPS.values() for c in cols]
    base = base.dropna(subset=all_extra)   # same games for every row of the table
    out = {}
    for name, cols in (groups or FEATURE_GROUPS).items():
        without = [f for f in FEATURES if f not in cols]
        out[name] = compare({'Without': (base, without), f'+ {name}': (base, without + cols)},
                            f'New feature: {name}', games=games)
    return out


@experiment
def darko_impact():
    """Value absent players by DARKO DPM (latest snapshot before each game) instead of, or in
    addition to, the box-score impact score. Same rotation and same players counted out."""
    from darko import DarkoRatings
    darko = DarkoRatings()
    print(f"DARKO snapshots: {len(darko.snapshots)} ({darko.snapshots[0].date()} to {darko.snapshots[-1].date()})")
    games, players = raw()
    df, _, _ = build_dataset(games, players, injury_params=data_pipeline.INJURY_PARAMS,
                             reports=load_injury_reports(), darko=darko, verbose=False)
    print(f"corr(box injury diff, DARKO injury diff) = {df['CORE_INJURY_DIFF'].corr(df['DARKO_INJURY_DIFF']):.3f}")
    replace = [('DARKO_INJURY_DIFF' if f == 'CORE_INJURY_DIFF' else f) for f in FEATURES]
    return compare({'Shipped: box-score impact': df,
                    'DARKO replaces box impact': (df, replace),
                    'DARKO added as a second feature': (df, FEATURES + ['DARKO_INJURY_DIFF'])},
                   'Injury impact from DARKO DPM')


# ------------------------------------------------------------------ Spread

def _held_out_spreads():
    """Walk-forward spread predictions written by train.py (margin, training-only sigma)."""
    sp = pd.read_csv('data/test_spread_predictions.csv', parse_dates=['GAME_DATE'])
    wp = pd.read_csv('data/test_predictions.csv')[['GAME_ID', 'MODEL_PROB']]
    return sp.merge(wp, on='GAME_ID')


@experiment
def spread_calibration(offsets=(-12, -8, -4, 0, 4, 8, 12)):
    """
    Are cover probabilities calibrated? For each held-out game, lines at the predicted margin
    plus each offset (.5 lines), P(margin > line) from the fold's training-only sigma, against
    how often the margin actually cleared the line.
    """
    import numpy as np
    from scipy.stats import norm
    sp = _held_out_spreads()
    rows = []
    for d in offsets:
        line = np.round(sp['MODEL_MARGIN']) + d + 0.5
        p = 1 - norm.cdf((line - sp['MODEL_MARGIN']) / sp['SIGMA'])
        rows.append(pd.DataFrame({'p': p, 'hit': (sp['MARGIN'] > line).astype(int)}))
    r = pd.concat(rows, ignore_index=True)
    r['bucket'] = pd.cut(r['p'], [0, .2, .3, .4, .45, .55, .6, .7, .8, 1.0], include_lowest=True)
    table = r.groupby('bucket', observed=True).agg(n=('hit', 'size'), predicted=('p', 'mean'), actual=('hit', 'mean'))
    print(f"Fold sigmas (training games only): {sorted(sp['SIGMA'].round(2).unique())}")
    print(f"Held-out residual SD: {np.std(sp['MARGIN'] - sp['MODEL_MARGIN']):.2f} pts")
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))
    # sharpness check at the one place it matters most: the model's own coin-flip line
    return table


def _paper_trades(min_edge=None):
    """Backtest the decision rules on held-out games with Kalshi prices (local data only)."""
    import numpy as np
    import decisions as D
    min_edge = D.MIN_EDGE_PTS if min_edge is None else min_edge
    sp = _held_out_spreads()
    sp['GID'] = sp['GAME_ID'].astype(str).str.zfill(10)
    ml = pd.read_csv('data/market_history.csv', dtype={'GAME_ID': str}).set_index('GAME_ID')
    sh = pd.read_csv('data/market_spread_history.csv', dtype={'GAME_ID': str}).set_index('GAME_ID')
    games = pd.read_csv('data/raw_nba_data.csv', dtype={'GAME_ID': str})
    home = games[games['MATCHUP'].str.contains('vs.', regex=False)].copy()
    home['GID'] = home['GAME_ID'].str.zfill(10)
    away = games[games['MATCHUP'].str.contains('@', regex=False)].copy()
    away['GID'] = away['GAME_ID'].str.zfill(10)
    abbr = dict(zip(home['GID'], home['TEAM_ABBREVIATION']))
    away_abbr = dict(zip(away['GID'], away['TEAM_ABBREVIATION']))
    trades = []
    for r in sp.itertuples():
        if r.GID not in ml.index:
            continue
        m = ml.loc[r.GID]
        info = sh.loc[r.GID].to_dict() if r.GID in sh.index else None
        if info is not None and not (info.get('MAIN_STRIKE') == info.get('MAIN_STRIKE')):
            info = None
        g = {'HOME_ABBR': abbr[r.GID], 'AWAY_ABBR': away_abbr[r.GID], 'MODEL_HOME_PROB': r.MODEL_PROB,
             'MARKET_YES_BID': m['HOME_YES_BID'], 'MARKET_YES_ASK': m['HOME_YES_ASK'],
             'MODEL_HOME_MARGIN': r.MODEL_MARGIN, 'SPREAD_SIGMA': r.SIGMA, 'MARKET_SPREAD_INFO': info}
        d = D.game_decisions(g, min_edge)
        for market in ('moneyline', 'spread'):
            x = d[market]
            if not x or x.get('mid') is None:
                continue
            if market == 'moneyline':
                won = (r.MARGIN > 0) == (x['best_side'] == 'YES')
            else:
                fav_margin = r.MARGIN if x['fav'] == g['HOME_ABBR'] else -r.MARGIN
                won = (fav_margin > x['strike']) == (x['best_side'] == 'YES')
            trades.append({'GAME_ID': r.GID, 'GAME_DATE': r.GAME_DATE, 'market': market, 'lean': x['side'] is not None,
                           'edge_pts': x['edge_pts'], 'p_side': x['p_side'], 'mid': x['mid'], 'fill': x['fill'],
                           'won': bool(won), 'pnl': D.settle('X', x['fill'], won)})
    return pd.DataFrame(trades)


@experiment
def paper_trades(report_path='data/market_report.md'):
    """
    Decision rules on held-out 2025-26 games with Kalshi prices. Uses Kalshi data, so the detailed
    report is written to a git-ignored local file only.
    """
    import numpy as np
    t = _paper_trades()
    lines = ['# Paper-trade backtest (local only: uses Kalshi prices)', '',
             'Hypothetical analysis, not betting advice. Fills at the ask plus fee; $1 contracts.', '']
    for market, g in t.groupby('market'):
        leans = g[g['lean']]
        lines.append(f"## {market}")
        lines.append(f"- games with a price: {len(g)}; leans (edge >= threshold): {len(leans)}")
        if len(leans):
            lines.append(f"- win rate {leans['won'].mean():.1%} vs average price paid {leans['fill'].mean():.3f}")
            lines.append(f"- total P/L ${leans['pnl'].sum():+.2f}, per trade ${leans['pnl'].mean():+.3f} "
                         f"(SE {leans['pnl'].std(ddof=1) / np.sqrt(len(leans)):.3f})")
            cal = leans.assign(b=pd.cut(leans['p_side'], [0, .4, .5, .6, .7, 1])).groupby('b', observed=True).agg(
                n=('won', 'size'), model_p=('p_side', 'mean'), actual=('won', 'mean'), price=('mid', 'mean'))
            lines.append('- calibration of the model on leans:\n\n' + cal.to_markdown(floatfmt='.3f'))
        # all priced sides, edge buckets: is a bigger edge followed by better results?
        g = g.assign(eb=pd.cut(g['edge_pts'], [-100, -5, 0, 5, 10, 100]))
        eb = g.groupby('eb', observed=True).agg(n=('won', 'size'), win=('won', 'mean'), price=('mid', 'mean'),
                                                pnl=('pnl', 'mean'))
        lines.append('\n- every priced game by edge bucket (best side):\n\n' + eb.to_markdown(floatfmt='.3f'))
        lines.append('')
    text = '\n'.join(lines)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)
    return t


# replacement_boosts and minutes_weighting were tested here and their switches removed after
# the results (RESULTS.md, Phase 3): boosts hurt, minutes weighting did not help.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='*')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    if args.list or not args.names:
        print('\n'.join(EXPERIMENTS))
    for name in args.names:
        EXPERIMENTS[name]()
