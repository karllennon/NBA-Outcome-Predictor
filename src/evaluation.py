"""
Walk-forward evaluation shared by train.py and experiments.py.

Games are sorted by date and split into consecutive test blocks starting at FOLD_STARTS
(fractions of the games); each block is predicted by a model fit only on earlier games.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss

FOLD_STARTS = (0.5, 0.625, 0.75, 0.875)


def fold_edges(n, fold_starts=FOLD_STARTS):
    return [int(n * f) for f in fold_starts] + [n]


def walk_forward(df, candidates, fold_starts=FOLD_STARTS, verbose=False, test_from=None):
    """
    candidates: {name: (make_model, feature_columns)} or {name: callable(train, test) -> probs}.
    test_from: optional date; if given, fold edges are computed over games on/after it, so
    adding older history changes the training data but not the held-out games.
    Returns one DataFrame of held-out predictions (GAME_ID, GAME_DATE, TARGET, FOLD, <name>...).
    """
    df = df.sort_values(['GAME_DATE', 'GAME_ID']).reset_index(drop=True)
    offset = 0
    if test_from is not None:
        offset = int((df['GAME_DATE'] < pd.Timestamp(test_from)).sum())
    edges = [offset + e for e in fold_edges(len(df) - offset, fold_starts)]

    folds = []
    for k, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        train, test = df.iloc[:start], df.iloc[start:end]
        out = test[['GAME_ID', 'GAME_DATE', 'TARGET'] + (['MARGIN'] if 'MARGIN' in test else [])].copy()
        out['FOLD'] = k
        for name, spec in candidates.items():
            if callable(spec):
                out[name] = spec(train, test)
            else:
                make, cols = spec
                out[name] = make().fit(train[cols], train['TARGET']).predict_proba(test[cols])[:, 1]
        folds.append(out)
        if verbose:
            print(f"Fold: train {start} games, test {end - start} "
                  f"({test['GAME_DATE'].min().date()} to {test['GAME_DATE'].max().date()})")
    return pd.concat(folds, ignore_index=True)


def score(preds, names):
    rows = []
    for name in names:
        y, p = preds['TARGET'], preds[name].clip(1e-6, 1 - 1e-6)
        fold_aucs = [roc_auc_score(g['TARGET'], g[name]) for _, g in preds.groupby('FOLD')]
        rows.append({'model': name, 'roc_auc': roc_auc_score(y, p),
                     'auc_min_fold': np.min(fold_aucs), 'auc_max_fold': np.max(fold_aucs),
                     'accuracy': accuracy_score(y, p > 0.5),
                     'log_loss': log_loss(y, p), 'brier': brier_score_loss(y, p)})
    return pd.DataFrame(rows)


def print_table(results, title=None):
    if title:
        print(f"\n--- {title} ---")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def markdown_table(results):
    lines = ['| Model | ROC-AUC | Accuracy | Log loss | Brier |', '|---|---|---|---|---|']
    for r in results.itertuples():
        lines.append(f"| {r.model} | {r.roc_auc:.4f} | {r.accuracy:.2%} | {r.log_loss:.4f} | {r.brier:.4f} |")
    return '\n'.join(lines)


def per_game_log_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_delta(preds, base, other):
    """Log-loss change of `other` vs `base` on the same games: mean, standard error, and how
    many folds improved. Negative = `other` is better."""
    d = per_game_log_loss(preds['TARGET'], preds[other]) - per_game_log_loss(preds['TARGET'], preds[base])
    folds_better = sum(d[preds['FOLD'] == k].mean() < 0 for k in sorted(preds['FOLD'].unique()))
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), folds_better, preds['FOLD'].nunique()


def score_regression(preds, names, target='MARGIN'):
    """Mean absolute error, RMSE and bias (mean prediction minus actual) on held-out games."""
    rows = []
    for name in names:
        err = preds[name] - preds[target]
        fold_mae = [float(np.mean(np.abs(g[name] - g[target]))) for _, g in preds.groupby('FOLD')]
        rows.append({'model': name, 'mae': float(np.mean(np.abs(err))), 'rmse': float(np.sqrt(np.mean(err ** 2))),
                     'bias': float(np.mean(err)), 'mae_min_fold': min(fold_mae), 'mae_max_fold': max(fold_mae)})
    return pd.DataFrame(rows)


def favorite_disagreement(home_prob, home_margin, outcome_home_win=None):
    """How often the win probability and the spread pick different favorites, and by how much."""
    home_prob, home_margin = np.asarray(home_prob), np.asarray(home_margin)
    dis = (home_prob > 0.5) != (home_margin > 0)
    out = {'games': int(len(dis)), 'disagree': int(dis.sum()), 'disagree_share': float(dis.mean()),
           'median_prob_gap_from_50': float(np.median(np.abs(home_prob[dis] - 0.5))) if dis.any() else np.nan,
           'max_prob_gap_from_50': float(np.max(np.abs(home_prob[dis] - 0.5))) if dis.any() else np.nan,
           'median_abs_margin': float(np.median(np.abs(home_margin[dis]))) if dis.any() else np.nan,
           'max_abs_margin': float(np.max(np.abs(home_margin[dis]))) if dis.any() else np.nan}
    if outcome_home_win is not None and dis.any():
        y = np.asarray(outcome_home_win)[dis]
        out['prob_right_when_disagree'] = float(((home_prob[dis] > 0.5) == y).mean())
        out['spread_right_when_disagree'] = float(((home_margin[dis] > 0) == y).mean())
    return out
