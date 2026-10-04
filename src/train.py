import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss
from matchups import FEATURES

# Model saved for live predictions: 'logistic' or 'xgboost'.
# Logistic regression wins the walk-forward comparison below on ~2,500 training games;
# deep trees overfit at this sample size. Switch if a re-run says otherwise.
SHIPPED_MODEL = 'logistic'

# Walk-forward folds: train on everything before each block, test on the block.
FOLD_STARTS = (0.5, 0.625, 0.75, 0.875)


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


def train_model():
    df = pd.read_csv('data/final_training_set.csv', parse_dates=['GAME_DATE'])
    df = df.sort_values('GAME_DATE').reset_index(drop=True)
    n = len(df)
    edges = [int(n * f) for f in FOLD_STARTS] + [n]

    # 1. Walk-forward evaluation: every model is always tested on games after its training data
    fold_preds = []
    for start, end in zip(edges[:-1], edges[1:]):
        train, test = df.iloc[:start], df.iloc[start:end]
        out = test[['GAME_ID', 'GAME_DATE', 'TARGET']].copy()
        for name, (make, cols) in CANDIDATES.items():
            out[name] = make().fit(train[cols], train['TARGET']).predict_proba(test[cols])[:, 1]
        fold_preds.append(out)
        print(f"Fold: train {start} games, test {end - start} "
              f"({test['GAME_DATE'].min().date()} to {test['GAME_DATE'].max().date()})")
    preds = pd.concat(fold_preds, ignore_index=True)

    # 2. Score each model on all held-out games, plus per-fold AUC spread
    rows = []
    for name in CANDIDATES:
        fold_aucs = [roc_auc_score(f['TARGET'], f[name]) for f in fold_preds]
        y, p = preds['TARGET'], preds[name]
        rows.append({'model': name, 'roc_auc': roc_auc_score(y, p),
                     'auc_min_fold': np.min(fold_aucs), 'auc_max_fold': np.max(fold_aucs),
                     'accuracy': accuracy_score(y, p > 0.5),
                     'log_loss': log_loss(y, p), 'brier': brier_score_loss(y, p)})
    results = pd.DataFrame(rows)
    print(f"\n--- Walk-forward results on {len(preds)} held-out games ---")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 3. Save held-out predictions for backtest.py and the dashboard
    shipped = SHIPPED_NAME[SHIPPED_MODEL]
    pd.DataFrame({'GAME_ID': preds['GAME_ID'], 'GAME_DATE': preds['GAME_DATE'],
                  'TARGET': preds['TARGET'], 'MODEL_PROB': preds[shipped],
                  'ELO_PROB': preds['Elo only (logistic)']}).to_csv('data/test_predictions.csv', index=False)
    results.assign(shipped=results['model'] == shipped).to_csv('data/test_metrics.csv', index=False)

    # 4. Refit the shipped model on every game for live predictions
    make, cols = CANDIDATES[shipped]
    final_model = make().fit(df[cols], df['TARGET'])
    os.makedirs('models', exist_ok=True)
    joblib.dump(final_model, 'models/nba_model.joblib')

    print(f"\nShipped model: {shipped}, refit on all {n} games -> models/nba_model.joblib")
    weights = feature_weights(final_model)
    print("Feature weights:")
    print(weights.reindex(weights.abs().sort_values(ascending=False).index)
                 .to_string(float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    train_model()
