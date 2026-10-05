# Experiment Results

Every number here is from the walk-forward evaluation in `src/train.py`: games sorted by
date, split into four consecutive test blocks (starting at 50%, 62.5%, 75% and 87.5% of the
games), each predicted by a model trained only on the games before it. Metrics are computed
on all held-out games pooled. Lower log loss is better; it is the primary criterion, AUC is
second. A change is kept only if it lowers held-out log loss versus the model it would replace.

When the underlying data changes (new games ingested, more seasons added), the held-out set
changes too, so each section states which baseline it was compared against.

## Phase 0: Baseline after applying `nba-fixes.patch`

Data: 2023-24 through 2026-03-03 (3,213 games with complete features; 1,607 held out,
2025-01-07 to 2026-03-03).

| Model | ROC-AUC | AUC fold min-max | Accuracy | Log loss | Brier |
|---|---|---|---|---|---|
| Elo only (logistic) | 0.7141 | 0.692-0.730 | 65.84% | 0.6164 | 0.2140 |
| **All features (logistic), shipped** | **0.7222** | 0.705-0.742 | **66.71%** | **0.6104** | **0.2115** |
| XGBoost depth 5 (original) | 0.6924 | 0.675-0.727 | 63.60% | 0.6514 | 0.2257 |
| XGBoost depth 2 (regularized) | 0.7157 | 0.704-0.728 | 65.71% | 0.6141 | 0.2132 |

The expected logistic numbers (~0.714 Elo, ~0.722 all features) reproduce exactly. The real
XGBoost numbers: the original depth-5 model is clearly worse than Elo alone (log loss 0.651),
and the regularized depth-2 model beats Elo slightly but loses to logistic regression.
