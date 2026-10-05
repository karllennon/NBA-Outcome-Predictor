import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data_pipeline import load_games
from elo import ELO_GRID, elo_tables, tune_elo
from evaluation import walk_forward, score, print_table, score_regression, favorite_disagreement
import spread_model
from matchups import FEATURES

# Model saved for live predictions: 'logistic' or 'xgboost'.
# Logistic regression wins the walk-forward comparison below on ~2,500 training games;
# deep trees overfit at this sample size. Switch if a re-run says otherwise.
SHIPPED_MODEL = 'logistic'


def make_logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))


def make_xgboost(shallow=False):
    if shallow:
        return xgb.XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.02,
                                 min_child_weight=10, random_state=42, eval_metric='logloss')
    return xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                             random_state=42, eval_metric='logloss')


CANDIDATES = {
    'Elo only (logistic)': (make_logistic, ['ELO_DIFF']),
    'All features (logistic)': (make_logistic, FEATURES),
    'XGBoost depth 5 (original)': (lambda: make_xgboost(False), FEATURES),
    'XGBoost depth 2 (regularized)': (lambda: make_xgboost(True), FEATURES),
}
SHIPPED_NAME = {'logistic': 'All features (logistic)', 'xgboost': 'XGBoost depth 2 (regularized)'}


def feature_weights(model):
    """Coefficients for logistic regression, importances for XGBoost."""
    if hasattr(model, 'feature_importances_'):
        return pd.Series(model.feature_importances_, index=FEATURES)
    return pd.Series(model[-1].coef_[0], index=FEATURES)


def fold_tuned_elo(candidates, games=None, tables=None, verbose=True):
    """
    Wraps each candidate so that, inside every walk-forward fold, ELO_DIFF is recomputed with
    Elo settings tuned on that fold's training games only. The shipped ELO_PARAMS were tuned on
    all games, so evaluating with them would let test games influence the Elo settings.
    candidates: {name: (make_model, columns) or callable(train, test) -> probabilities}
    """
    tables = tables if tables is not None else elo_tables(games, ELO_GRID)
    cache = {}

    def swap_elo(train, test):
        start = test['GAME_DATE'].min()
        if start not in cache:
            params, diff = tune_elo(tables, start)
            cache[start] = diff
            if verbose:
                print(f"  Elo tuned on games before {start.date()}: {params}")
        return (train.assign(ELO_DIFF=train['GAME_ID'].map(cache[start])),
                test.assign(ELO_DIFF=test['GAME_ID'].map(cache[start])))

    def wrap(spec):
        def fit_predict(train, test):
            tr, te = swap_elo(train, test)
            if callable(spec):
                return spec(tr, te)
            make, cols = spec
            return make().fit(tr[cols], tr['TARGET']).predict_proba(te[cols])[:, 1]
        return fit_predict

    return {name: wrap(spec) for name, spec in candidates.items()}


def train_model():
    df = pd.read_csv('data/final_training_set.csv', parse_dates=['GAME_DATE'])
    df = df.sort_values('GAME_DATE').reset_index(drop=True)
    games = load_games()

    # 1. Walk-forward evaluation: every model is always tested on games after its training data
    tables = elo_tables(games, ELO_GRID)
    preds = walk_forward(df, fold_tuned_elo(CANDIDATES, tables=tables), verbose=True)

    # 2. Score each model on all held-out games, plus per-fold AUC spread
    results = score(preds, CANDIDATES)
    print_table(results, f"Walk-forward results on {len(preds)} held-out games")

    # 3. Save held-out predictions for backtest.py and the dashboard
    shipped = SHIPPED_NAME[SHIPPED_MODEL]
    pd.DataFrame({'GAME_ID': preds['GAME_ID'], 'GAME_DATE': preds['GAME_DATE'],
                  'TARGET': preds['TARGET'], 'MODEL_PROB': preds[shipped],
                  'ELO_PROB': preds['Elo only (logistic)']}).to_csv('data/test_predictions.csv', index=False)
    results.assign(shipped=results['model'] == shipped).to_csv('data/test_metrics.csv', index=False)

    # 3b. Same held-out predictions vs Kalshi's pre-tip-off price, where one exists
    if os.path.exists('data/market_history.csv'):
        market = pd.read_csv('data/market_history.csv', dtype={'GAME_ID': str})
        market['GAME_ID'] = market['GAME_ID'].astype(int)
        both = preds.merge(market[['GAME_ID', 'MARKET_HOME_PROB']], on='GAME_ID').dropna(
            subset=['MARKET_HOME_PROB'])
        if len(both):
            both = both.rename(columns={'MARKET_HOME_PROB': 'Kalshi pre-tip-off price'})
            both['FOLD'] = 0
            mres = score(both, ['Elo only (logistic)', shipped, 'Kalshi pre-tip-off price'])
            mres = mres.assign(games=len(both), first_game=both['GAME_DATE'].min().date(),
                               last_game=both['GAME_DATE'].max().date())
            mres.to_csv('data/market_metrics.csv', index=False)
            print_table(mres[['model', 'roc_auc', 'accuracy', 'log_loss', 'brier']],
                        f"Held-out games with a Kalshi price ({len(both)})")

    # 3c. Spread: same folds, same per-fold Elo tuning; baselines are the training-period average
    #     home margin and a margin regression on ELO_DIFF alone
    cfg = spread_model.TARGETS['spread']
    fold_sigma = {}

    def spread_fit(train, test):
        fitted = spread_model.fit_target(cfg, train)
        fold_sigma[test['GAME_DATE'].min()] = fitted.sigma
        return fitted.predict(test)

    spread_cands = {
        'Average home margin': lambda tr, te: np.full(len(te), tr['MARGIN'].mean()),
        'Elo-only spread': lambda tr, te: spread_model.make_ridge().fit(tr[['ELO_DIFF']], tr['MARGIN']).predict(te[['ELO_DIFF']]),
        'Spread model': spread_fit,
    }
    sp = walk_forward(df, fold_tuned_elo(spread_cands, tables=tables, verbose=False))
    first_date = sp.groupby('FOLD')['GAME_DATE'].transform('min')
    sp['SIGMA'] = first_date.map(fold_sigma)
    sres = score_regression(sp, list(spread_cands))
    print_table(sres, f"Spread, walk-forward on {len(sp)} held-out games (points)")
    dis = favorite_disagreement(preds.set_index('GAME_ID').loc[sp['GAME_ID'], shipped].values,
                                sp['Spread model'].values, sp['TARGET'].values)
    print(f"Win probability and spread pick different favorites in {dis['disagree']} of {dis['games']} games "
          f"({dis['disagree_share']:.1%}); median |p - 50%| {dis['median_prob_gap_from_50']:.1%}, "
          f"median |margin| {dis['median_abs_margin']:.1f} pts")
    sp.rename(columns={'Spread model': 'MODEL_MARGIN', 'Elo-only spread': 'ELO_MARGIN',
                       'Average home margin': 'AVG_MARGIN'}).to_csv('data/test_spread_predictions.csv', index=False)
    sres.to_csv('data/spread_metrics.csv', index=False)
    pd.Series(dis).to_csv('data/spread_disagreement.csv', header=['value'])

    # 4. Refit the shipped model on every game for live predictions
    make, cols = CANDIDATES[shipped]
    final_model = make().fit(df[cols], df['TARGET'])
    os.makedirs('models', exist_ok=True)
    joblib.dump(final_model, 'models/nba_model.joblib')

    # 5. Spread model: a separate output (the win probability still comes from the classifier)
    spread = spread_model.fit_target(spread_model.TARGETS['spread'], df)
    spread_model.save(spread)
    print(f"Spread model: Ridge on home margin, sigma {spread.sigma:.2f} pts -> {spread.config.model_path}")

    print(f"\nShipped model: {shipped}, refit on all {len(df)} games -> models/nba_model.joblib")
    weights = feature_weights(final_model)
    print("Feature weights:")
    print(weights.reindex(weights.abs().sort_values(ascending=False).index)
                 .to_string(float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    train_model()
