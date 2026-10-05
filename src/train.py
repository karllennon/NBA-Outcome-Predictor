import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from evaluation import walk_forward, score, print_table
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


def train_model():
    df = pd.read_csv('data/final_training_set.csv', parse_dates=['GAME_DATE'])
    df = df.sort_values('GAME_DATE').reset_index(drop=True)

    # 1. Walk-forward evaluation: every model is always tested on games after its training data
    preds = walk_forward(df, CANDIDATES, verbose=True)

    # 2. Score each model on all held-out games, plus per-fold AUC spread
    results = score(preds, CANDIDATES)
    print_table(results, f"Walk-forward results on {len(preds)} held-out games")

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

    print(f"\nShipped model: {shipped}, refit on all {len(df)} games -> models/nba_model.joblib")
    weights = feature_weights(final_model)
    print("Feature weights:")
    print(weights.reindex(weights.abs().sort_values(ascending=False).index)
                 .to_string(float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    train_model()
