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
from train import make_logistic

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


def compare(variants, title, features=FEATURES, test_from=None):
    """
    variants: {name: DataFrame (scored with logistic on `features`) or
               (DataFrame, features) or (DataFrame, callable(train, test) -> probs)}.
    The first variant is the current shipped configuration; 'Elo only' is fit on it.
    All variants are restricted to the games they have in common, so every row of the
    table is scored on exactly the same held-out games.
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
                       'Elo tuned per fold on training games': (base, tuned)}, 'Elo tuning')
    for start, params in chosen:
        print(f"  fold starting {start}: {params}")
    print(f"  tuned on all games (for shipping): {tune_elo(tables, '2100-01-01')[0]}")
    return results


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
