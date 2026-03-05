# NBA Outcome Predictor

A machine learning pipeline that predicts NBA game outcomes using XGBoost with advanced feature engineering.

## Model Performance
- **ROC-AUC: 0.66** on held-out test set
- Trained on 2,570 games across 3 NBA seasons (2023-26)

## Features
- Elo ratings with K-factor decay
- Rolling 10-game advanced stats (eFG%, TOV%, ORB%, Pace, Offensive/Defensive Rating)
- Rest and back-to-back fatigue features
- Win streak momentum
- Injury impact differential with acute/chronic classification and position-aware replacement boost logic

## Project Structure
```
src/
  ingest.py        # NBA API data ingestion
  data_pipeline.py # Feature engineering pipeline  
  features.py      # NBAFeatureProcessor class
  elo.py           # Elo rating calculator
  matchups.py      # Matchup differential builder
  train.py         # XGBoost model training
  predict.py       # CLI prediction tool
  app.py           # Streamlit dashboard
data/
  player_positions.csv  # Static position reference
  recent_trades.csv     # Manual trade override
models/
  nba_model.joblib      # Trained model (not tracked)
```

## Dashboard
Three pages: Today's Slate (game predictor with injury input), Backtest Results, Model Performance

## Known Limitations
- Box score stats undervalue elite defensive specialists (e.g. Alex Caruso)
- Same-day scratches require manual injury override
- Data requires periodic refresh via NBA API