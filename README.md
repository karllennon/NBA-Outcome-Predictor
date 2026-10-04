# NBA Outcome Predictor

Predicts NBA game outcomes from Elo ratings, rolling team form, rest, and a player-level injury impact model, with a Streamlit dashboard for game-day predictions.

## Model Performance

Evaluated walk-forward: the data is split into four consecutive blocks, and each block is predicted by a model trained only on the games before it. That gives 1,607 held-out games (Jan 2025 to Mar 2026) that no model saw during training.

| Model | ROC-AUC | Accuracy | Log loss | Brier |
|---|---|---|---|---|
| Elo only (baseline) | 0.714 | 65.8% | 0.616 | 0.214 |
| **All features, logistic regression (shipped)** | **0.722** | **66.7%** | **0.610** | **0.212** |

`train.py` also prints the XGBoost variants on the same folds. On ~2,500 training games the original depth-5 XGBoost overfits and finishes below the Elo baseline, and a regularized depth-2 version roughly matches it, so the shipped model is logistic regression (`SHIPPED_MODEL` in `train.py` switches it).

The injury feature is the largest gain over Elo: home teams win 41.5% of games in the bottom fifth of the injury differential and 68.9% in the top fifth. Rolling box-score differentials add little once Elo is in the model.

Probabilities are well calibrated: games predicted at 80% home win go about 80%.

## Features
- Elo ratings with home-court advantage, margin-of-victory scaling, and regression to the mean between seasons
- Rolling 10-game team stats (eFG%, TOV%, ORB%, FT rate, pace, offensive/defensive rating, plus/minus)
- Rest days and back-to-back flags
- Win streak (last 5 games)
- Injury impact differential: each team's top-10 rotation is scored from recent box scores, absences are classified as acute (1-3 games), chronic (4-20), or inactive (21+, dropped), and acute top-4 absences receive a position-aware replacement boost

The same `InjuryModel` builds the injury feature for training and for live predictions, so the feature means the same thing in both.

## Project Structure
```
src/
  ingest.py        # NBA API data ingestion
  elo.py           # Elo rating calculator
  injuries.py      # Rotation, absence classification, and injury impact (training + live)
  features.py      # Team stats, rest, and rolling form
  matchups.py      # Home-minus-away differentials and the feature list
  data_pipeline.py # Builds the training set and current team state
  train.py         # Walk-forward evaluation, baselines, and final model
  backtest.py      # Held-out metrics and calibration
  inference.py     # Feature row for an upcoming game (shared by CLI and dashboard)
  predict.py       # CLI prediction tool
  daily_slate.py   # Predicts today's games from the NBA schedule
  app.py           # Streamlit dashboard
data/
  player_positions.csv  # Static position reference
  recent_trades.csv     # Manual trade override
```

## Usage
```
pip install -r requirements.txt
python src/ingest.py          # refresh raw data
python src/data_pipeline.py   # build features (about 20 seconds)
python src/train.py           # evaluate and save the model
python src/backtest.py        # held-out metrics and calibration
streamlit run src/app.py
```

## Dashboard
Three pages: Today's Slate (game predictor with injury input), Backtest Results (held-out games only), Model Performance (ROC curve, confusion matrix, feature weights).

## Known Limitations
- Box score stats undervalue elite defensive specialists (e.g. Alex Caruso)
- Same-day scratches require manual injury input
- The training-time injury feature assumes a player who missed the previous game is out for this one
- In testing, the replacement boost does not measurably improve accuracy over counting full impact lost
- Data requires periodic refresh via the NBA API
