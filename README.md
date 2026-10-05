# NBA Outcome Predictor

Predicts NBA game outcomes (home win probability) from Elo ratings, official injury reports,
rolling team form, rest and travel, and compares each prediction with the Kalshi game market.
Includes a Streamlit dashboard and a log of every live prediction with its track record.

## Model Performance

Evaluated walk-forward: games are sorted by date and split into four consecutive test blocks,
and each block is predicted by a model trained only on the games before it. Elo settings are
also tuned inside each block on those training games. That gives 1,762 held-out games
(2025-01-29 to 2026-04-12) that no model or setting saw during training. Lower log loss is better.

| Model | ROC-AUC | Accuracy | Log loss | Brier |
|---|---|---|---|---|
| Elo only (baseline) | 0.742 | 68.6% | 0.595 | 0.204 |
| **All features, logistic regression (shipped)** | **0.758** | **70.4%** | **0.580** | **0.198** |
| XGBoost, depth 2 (regularized) | 0.751 | 69.3% | 0.589 | 0.201 |
| XGBoost, depth 5 (original) | 0.729 | 66.9% | 0.613 | 0.211 |

**Against the market.** On the 1,223 of those games that had a Kalshi game market (2025-26
season), the market's price at tip-off was more accurate than the model:

| | ROC-AUC | Accuracy | Log loss | Brier |
|---|---|---|---|---|
| Shipped model | 0.755 | 70.2% | 0.581 | 0.199 |
| Kalshi pre-tip-off price | 0.765 | 69.5% | 0.569 | 0.194 |

A 50/50 average of model and market is also worse than the market alone, so treat the model as
a well-calibrated baseline, not a source of betting edge.

**Calibration** (held-out): games predicted under 30% were won 18% of the time, and over 70%
were won 79% (predicted 21% and 81%). In the middle the model is less sharp: games predicted at
40-50% were won 40%, at 60-70% were won 74%.

The official injury reports are the largest gain over Elo: on held-out games, home teams win 39%
in the bottom fifth of the injury differential and 73% in the top fifth (55% overall).

Every experiment, including the ones that did not help, is in [RESULTS.md](RESULTS.md).

## Features
- **Elo** with margin-of-victory scaling, home-court advantage and between-season regression.
  K-factor (12.5), home advantage (50) and carryover (0.5) are tuned by log loss; the 2019-20
  bubble games are treated as neutral-site.
- **Injuries from the official NBA injury reports**: each team's top-10 rotation is scored from
  its last 15 box scores, and every rotation player listed Out on the last report published
  before tip-off counts his full impact. When no report exists, players who missed the previous
  game are assumed out. Players out 21+ games are dropped (rolling stats already reflect them).
- Rolling 10-game team stats (eFG%, TOV%, ORB%, FT rate, pace, defensive rating, plus/minus)
  and wins in the last 5
- Rest days and back-to-backs
- Travel distance and time zones crossed since the previous game

Training and live predictions compute every feature with the same functions (`InjuryModel` in
`injuries.py`, `add_context` in `context.py`, the shared Elo settings), and tests check that
live values equal the training values.

## Data Sources
- **NBA box scores**: stats.nba.com via `nba_api` (team and player game logs, scoreboard).
  Training uses 2023-24 onward; 2018-19 to 2022-23 are on disk but did not help.
- **Official NBA injury reports**: the league's PDFs at
  `ak-static.cms.nba.com/referee/injury/`, parsed with pdfplumber and archived as compressed CSVs
  in `data/injury_reports/` (late 2018 to early 2020, and October 2023 onward).
- **Kalshi**: public, read-only market data (`external-api.kalshi.com/trade-api/v2`, series
  `KXNBAGAME`). No login, and no trading code in this project.
- `data/arenas.csv`: arena coordinates, time zones and elevations.

## Project Structure
```
src/
  ingest.py          # Incremental NBA data ingestion (retries, loud failures)
  injury_reports.py  # Official injury report download, parsing, name matching, archive
  elo.py             # Elo ratings, tuning grid, neutral-site handling
  injuries.py        # Rotation, absences and injury impact (training + live)
  features.py        # Team stats, rest and rolling form
  context.py         # Travel, time zones, schedule density, altitude, SOS-adjusted rating
  matchups.py        # Home-minus-away differentials and the shipped feature list
  data_pipeline.py   # Builds the training set and current team state
  evaluation.py      # Walk-forward evaluation shared by training and experiments
  train.py           # Walk-forward results, market comparison, final model
  experiments.py     # Every before/after experiment in RESULTS.md
  backtest.py        # Held-out metrics and calibration
  inference.py       # Feature row for an upcoming game (CLI, slate, dashboard)
  market_odds.py     # Kalshi snapshots and pre-tip-off price history
  schedule.py        # Today's games and tip-off times
  daily_slate.py     # Today's model vs market, logs pre-tip-off predictions
  prediction_log.py  # Append-only prediction log and live track record
  predict.py         # CLI prediction tool
  app.py             # Streamlit dashboard
scripts/
  refresh.sh, refresh.bat   # Local daily refresh
tests/                      # Leakage, consistency, parsing and logging tests
```

## Usage
```
python -m venv venv && venv/Scripts/pip install -r requirements.txt   # (venv/bin/pip on macOS/Linux)
python src/ingest.py            # refresh raw data
python src/data_pipeline.py     # build features
python src/train.py             # walk-forward evaluation, market comparison, save model
python src/backtest.py          # held-out metrics and calibration
python src/daily_slate.py       # today's games: model vs Kalshi, logged before tip-off
python src/predict.py           # one game from the command line
streamlit run src/app.py        # dashboard
python -m pytest tests          # tests
python src/experiments.py --list  # rerun any experiment from RESULTS.md
```

## Refreshing Data
`src/ingest.py` is incremental: it re-fetches only the current season (worked out from today's
date) and the latest season already on disk, merges with the existing CSVs, and deduplicates on
GAME_ID + TEAM_ID / GAME_ID + PLAYER_ID. Failed requests are retried with backoff, and the script
exits non-zero if any season can't be fetched. `python src/ingest.py --since 2018-19` rebuilds
from a given season.

stats.nba.com does not answer GitHub-hosted runners (confirmed by a test run in October 2026),
so the refresh runs on your own machine. The simplest way is the two buttons at the top of the
dashboard:

- **Update injuries & odds** (seconds): re-downloads the latest injury report and Kalshi prices,
  recomputes today's predictions and logs them. Click it shortly before tip-off; the last
  prediction logged before tip-off is the one the Track Record scores.
- **Full data refresh** (a few minutes): downloads new results, rebuilds features and retrains
  the model (ingest -> pipeline -> train). Click it the morning after games.

To automate the same steps instead, schedule the refresh script. It runs ingest -> pipeline ->
train, archives the latest injury report, and logs today's predictions with Kalshi prices:

- macOS/Linux: `scripts/refresh.sh`. Daily at 10:00 with cron (`crontab -e`):
  `0 10 * * * /path/to/nba-outcome-predictor/scripts/refresh.sh >> /path/to/refresh.log 2>&1`
- Windows: `scripts\refresh.bat`. Daily at 10:00 with Task Scheduler:
  `schtasks /Create /SC DAILY /ST 10:00 /TN "NBA refresh" /TR "\"C:\path\to\nba-outcome-predictor\scripts\refresh.bat\""`

Injury reports change during the day, so if you automate it, also run `python src/daily_slate.py`
about an hour before the first tip-off (for example 6:00 PM Eastern).

The GitHub Action (`.github/workflows/refresh_data.yml`) is manual-only. It fails loudly when
the fetch fails, or when no new games arrive between November and March.

## Dashboard
`streamlit run src/app.py`: a dark, card-based dashboard with top navigation.

- **Dashboard**: the day's games on the left; the selected game's matchup (records, arena, final
  score in replay), model vs Kalshi win probability, Kalshi's price over the hours before
  tip-off, team form before the game, and the official injury report in the middle; the day's
  Kalshi game markets (model, market, gap, highlighted at 5+ points), what drives the model's
  number, and upcoming games on the right.
  - **Live** mode: today's games; every prediction is logged before tip-off.
  - **Replay** mode: any game day of the walk-forward test (2025-01 to 2026-04), showing the
    probability a model trained only on earlier games gave, Kalshi's price at tip-off, the injury
    report, and the result. Useful in the offseason and for seeing how the model and market differ.
- **Markets**: model vs Kalshi on every held-out game, and how the two did when they disagreed.
- **Predictor**: any matchup today, with players listed Out pre-checked and adjustable.
- **Track Record**: accuracy, log loss and calibration of logged live predictions vs the market.
- **Model**: walk-forward results, market comparison, calibration, ROC curve, feature weights.

## Known Limitations
- The market's tip-off price beats the model (log loss 0.569 vs 0.581 on 1,223 games).
- Box-score impact scores undervalue defensive specialists. No all-in-one plus-minus metric is
  used: EPM and LEBRON are paid, RAPM's data API is closed to automated use, and DARKO was not
  tested (see RESULTS.md).
- Injury reports for 2020-01 to 2022-23 are not archived (the NBA CDN rate-limited the
  backfill); those seasons are not used for training.
- Live predictions use the latest injury report at the time they are made; late scratches after
  that are missed.
- `data/recent_trades.csv` is a manual override for trades not yet in the box scores.
