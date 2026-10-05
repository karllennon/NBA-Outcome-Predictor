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
