import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss


def calibration_table(y, p, bins=(0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0)):
    """Predicted vs actual home win rate by probability bucket."""
    df = pd.DataFrame({'y': y, 'p': p})
    df['bucket'] = pd.cut(df['p'], bins=list(bins), include_lowest=True)
    return (df.groupby('bucket', observed=True)
              .agg(games=('y', 'size'), predicted=('p', 'mean'), actual=('y', 'mean')))


def run_backtest():
    """
    Scores ONLY the walk-forward held-out games saved by train.py; each was predicted
    by a model trained only on earlier games.
    No betting simulation: the data has no historical odds, and assuming -110 on every
    home pick does not reflect moneyline prices, so any ROI from it would be misleading.
    """
    preds = pd.read_csv('data/test_predictions.csv', parse_dates=['GAME_DATE'])
    y = preds['TARGET']

    print(f"--- Backtest on {len(preds)} held-out games "
          f"({preds['GAME_DATE'].min().date()} to {preds['GAME_DATE'].max().date()}) ---")
    for name, col in [('Shipped', 'MODEL_PROB'), ('Elo only', 'ELO_PROB')]:
        p = preds[col]
        print(f"{name:9s} AUC {roc_auc_score(y, p):.4f} | accuracy {accuracy_score(y, p > 0.5):.2%} | "
              f"log loss {log_loss(y, p):.4f} | Brier {brier_score_loss(y, p):.4f}")

    print("\nCalibration (shipped model):")
    print(calibration_table(y, preds['MODEL_PROB']).to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    run_backtest()
