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

## Phase 1: Data freshness

**Cause of the stale data.** The data stopped at 2026-03-03. All 62 scheduled runs from
2026-03-05 to 2026-05-05 were marked successful, but each spent ~4 minutes in ingestion
(three seasons x 60s timeout + sleeps), fetched nothing, exited 0, and had nothing to commit.
GitHub then disabled the schedule for repository inactivity. The run logs had expired (HTTP 410),
so this comes from the per-step timings. Two separate problems:

1. The hand-written browser headers in `ingest.py` now make stats.nba.com hang until timeout,
   even from a home connection. nba_api's default headers work (2025-26 season in 1.3s).
2. GitHub-hosted runners are blocked regardless. A manual run on 2026-10-05 with the fixed
   headers timed out on every attempt (run 37254192221) and now correctly ends as a failure.

**Fix.** Incremental ingest with default headers, retries, and a non-zero exit on failure;
local `scripts/refresh.sh` / `.bat`; the workflow is manual-only.

**Result.** Data now reaches 2026-04-12, the last day of the 2025-26 regular season (today is
2026-10-04, so there are no newer regular-season games). Playoff games are not ingested; the
model is regular-season only.

New baseline on the refreshed data (3,523 games, 1,762 held out, 2025-01-29 to 2026-04-12).
All later experiments compare against this until the data changes again:

| Model | ROC-AUC | Accuracy | Log loss | Brier |
|---|---|---|---|---|
| Elo only (logistic) | 0.7377 | 67.99% | 0.5978 | 0.2056 |
| **All features (logistic), shipped** | **0.7437** | **68.39%** | **0.5914** | **0.2032** |
| XGBoost depth 5 | 0.7173 | 66.00% | 0.6246 | 0.2152 |
| XGBoost depth 2 | 0.7352 | 67.54% | 0.5983 | 0.2060 |

The held-out window moved later, so these numbers are not comparable with Phase 0.
Late-season games are easier to predict (last fold AUC ~0.82), which lifts all models.

## Evaluation note: deterministic game order

The experiment harness (`src/evaluation.py`, shared by `train.py` and `experiments.py`) sorts
games by date and then GAME_ID, so fold boundaries are identical for every variant. Same-day ties
were previously in arbitrary order, which moved the Phase 1 numbers slightly. Reference
after this change: Elo only 0.7369 AUC / 0.5984 log loss; all features 0.7427 / 0.5922.
Each experiment also reports the paired per-game log-loss change with its standard error and
how many of the 4 folds improved, so small differences can be judged against noise.

## Phase 2: Official NBA injury reports

`src/injury_reports.py` downloads the league's official injury report PDFs and parses them with
pdfplumber. URL pattern verified against the server (not guessed):
`https://ak-static.cms.nba.com/referee/injury/Injury-Report_<date>_<time>.pdf`, hourly `HHAM/PM`
files before 2025-12-22 (the 05PM file holds the 5:30 PM report) and 15-minute `HH_MMAM/PM` files
since. Names are matched to box scores after removing accents, punctuation and suffixes
(Dončić, Butler III, P.J. Washington), with a unique last-name + first-initial fallback.
On two sample reports, 151 of 158 non-G-League players matched. The 7 misses had never played
for that team (season-long absences or pending trades), so they could not be in its rotation.

Backfill: for every game day, the last report published at least 30 minutes before each tip-off
time. 2,272 reports archived in `data/injury_reports/` (13 MB, gzipped CSV); one game day
(2024-12-09) had no report. 97.6% of team-games have a pre-tip-off report; the rest fall back to
the old guess.

| Injury source (1,762 held-out games) | ROC-AUC | Accuracy | Log loss | Brier |
|---|---|---|---|---|
| Elo only (logistic) | 0.7369 | 67.93% | 0.5984 | 0.2058 |
| Before: guess (missed previous game) | 0.7427 | 68.27% | 0.5922 | 0.2036 |
| **Reports: Out (kept)** | **0.7483** | **69.24%** | **0.5872** | **0.2014** |
| Reports: Out + Doubtful | 0.7482 | 69.24% | 0.5873 | 0.2015 |

Reports vs guess: log loss -0.0050 (SE 0.0023), better in 3 of 4 folds. Kept, with Out only;
counting Doubtful players as out adds nothing. Live predictions (CLI, daily slate, dashboard)
pre-fill players listed Out from the latest report, through the same `InjuryModel.report_out`
used in training; the dashboard checkboxes still override.

The old `src/injury_scraper.py` scraped RotoWire, whose terms do not allow it, and was not used
anywhere; it was removed.

## Phase 3: Player impact

Baseline: the Phase 2 model (official reports, with replacement boosts), 1,762 held-out games.

### 3.1 Replacement boosts (STAR_BOOST, MINUTES_BOOST)

| Model | ROC-AUC | Accuracy | Log loss | Brier |
|---|---|---|---|---|
| Elo only (logistic) | 0.7369 | 67.93% | 0.5984 | 0.2058 |
| Before: with boosts | 0.7483 | 69.24% | 0.5872 | 0.2014 |
| **No replacement boosts (kept)** | **0.7514** | **69.69%** | **0.5851** | **0.2004** |

Log loss -0.0021 (SE 0.0012), better in 3 of 4 folds. This confirms the earlier finding
(0.7243 AUC without vs 0.7228 with). The boosts, the position lookup they used
(`data/player_positions.csv`) and the "new injury?" overrides in the CLI and dashboard, which
only fed the boosts, were removed. Every absent rotation player now counts his full impact.

### 3.2 Weighting impact by recent minutes (not kept)

Impact = per-minute impact over the last 15 games x minutes per game over the last 5.

| Model | ROC-AUC | Accuracy | Log loss | Brier |
|---|---|---|---|---|
| Elo only (logistic) | 0.7369 | 67.93% | 0.5984 | 0.2058 |
| Shipped: mean impact, last 15 games | 0.7483 | 69.24% | 0.5872 | 0.2014 |
| Per-minute impact x recent MPG | 0.7478 | 69.18% | 0.5876 | 0.2016 |

Log loss +0.0005 (SE 0.0009), better in 2 of 4 folds: no improvement, reverted. (Measured
before 3.1 was applied; the shipped impact score is unchanged by 3.1.)

### 3.3 Public all-in-one plus-minus metric

- **EPM** (dunksandthrees.com): full data needs a paid subscription; its `/api/` is disallowed
  in robots.txt. Not used.
- **LEBRON** (BBall Index): paid. Not used.
- **RAPM** (nbarapm.com): data is served from `/api/`, which robots.txt disallows. Not used.
- **DARKO DPM** (darko.app): the site states its leaderboard has a CSV download and that a
  "Time Machine" sets it to any past date, so it can be used legally.

**Caveat, not confirmed: whether DARKO's Time Machine shows ratings as they were published on
that date.** The site's changelog says the Time Machine was added on 2026-09-27, and DARKO
shows ratings back to 1996-97, long before DARKO existed. So past-date values are most likely
the current model re-run over games up to that date. Each player's rating "going into each
game" appears to use only earlier games, but the model's design and tuning were fit with data
that includes later seasons. Any gain measured with these values may therefore be slightly
optimistic.

## Phase 4: Market odds (Kalshi)

`src/market_odds.py` reads Kalshi's public market-data API (no login, no trading or order code).
Verified against Kalshi's docs and live API in October 2026: base URL
`https://external-api.kalshi.com/trade-api/v2`, series `KXNBAGAME`, event tickers
`KXNBAGAME-<YYMONDD><AWAY><HOME>` (checked on known games, e.g. `26MAR03DALCHA` = Dallas at
Charlotte), one market per team using standard NBA abbreviations. Market probability = midpoint
of the home team's YES bid and ask (blank for an empty book or a spread over 25 cents).

- **Live:** `python src/market_odds.py` appends one row per game to
  `data/market_snapshots.csv` (timestamp, game, teams, probability, bids/asks, volume), never
  rewriting earlier rows. Tip-off time and status come from the NBA scoreboard, and a `PREGAME`
  column marks prices taken before tip-off. The local refresh scripts take a snapshot each run.
- **History:** Kalshi's NBA game markets start 2025-04-15. `--history` takes the bid/ask
  midpoint from the last hourly candlestick ending at scheduled tip-off (tip times from the
  injury reports) for every regular-season game since then: 1,223 of 1,225 games priced
  (`data/market_history.csv`). Settled markets older than Kalshi's archive cutoff come from its
  `/historical/` endpoints.

**Model vs market** on the 1,223 held-out 2025-26 games that have a pre-tip-off price (shipped
model = Phase 3 model, walk-forward predictions):

| Model | ROC-AUC | Accuracy | Log loss | Brier |
|---|---|---|---|---|
| Elo only (logistic) | 0.7338 | 68.03% | 0.5998 | 0.2066 |
| Shipped model | 0.7479 | 68.93% | 0.5860 | 0.2012 |
| **Kalshi pre-tip-off midpoint** | **0.7651** | **69.42%** | **0.5697** | **0.1945** |

The market is clearly better: the model's log loss is 0.0163 higher (SE 0.0056). A 50/50
average of model and market scores 0.5731, worse than the market alone, so on this evidence
the model adds no information beyond the closing market price. The market price at tip-off
also reflects late scratches and lineup news the model only partly sees.

**The Odds API (optional item):** skipped. No `ODDS_API_KEY` is set in the environment or a
`.env` file. Kalshi's free history covered the model-vs-market comparison for 2025-26; a key
would add 2023-24 and 2024-25 sportsbook lines.

## Phase 5: Model improvements

### 5.1 Elo tuning (kept)

K-factor, home advantage and between-season carryover are chosen by the log loss of Elo's own
win probabilities on each fold's training games only (grid: K 5-30, home advantage 0-100 Elo
points, carryover 0.2-0.9). `train.py` repeats this inside every fold, so its walk-forward
numbers never use Elo settings that saw the test games; the shipped settings are tuned on all
games. The Elo calculator was rewritten to pair games once (identical ratings, 0.07 s instead of
8.7 s per pass) so the grid can be searched quickly.

| Model (1,762 held-out games) | ROC-AUC | Accuracy | Log loss | Brier |
|---|---|---|---|---|
| Elo only, before (K=20, HCA=100, carryover 0.75) | 0.7369 | 67.93% | 0.5984 | 0.2058 |
| Elo only, tuned per fold | 0.7421 | 68.56% | 0.5947 | 0.2041 |
| Before: all features, default Elo | 0.7514 | 69.69% | 0.5851 | 0.2004 |
| **All features, Elo tuned per fold (kept)** | **0.7564** | **70.49%** | **0.5816** | **0.1989** |

All features: log loss -0.0035 (SE 0.0011), better in 3 of 4 folds. Each fold picked K = 12.5,
carryover 0.5 and home advantage 25-50; tuned on all games: K = 12.5, home advantage 50,
carryover 0.5 (now `ELO_PARAMS` in `data_pipeline.py`, also used by live predictions for the
offseason regression). Smaller K and stronger regression than the FiveThirtyEight defaults, and
a home edge of 25-50 Elo points (about 54-57% at even strength), consistent with the smaller
home-court advantage of recent seasons. A first, narrower grid chose values on its lower edges;
it was widened before the result above.

From here on every row, including "Elo only", re-tunes Elo inside each fold on its training
games, exactly as `train.py` does.

### 5.2 Point-margin regression -> win probability (not kept)

Ridge regression on the home margin with the same features, converted with
P(home win) = Phi(margin / sigma). sigma is fit by log loss on the last 20% of each fold's
training games using a regression fit on the earlier 80%, then the regression is refit on all
training games (fitted sigma 11.1-15.0 points).

| Model (1,762 held-out games) | ROC-AUC | Accuracy | Log loss | Brier |
|---|---|---|---|---|
| Elo only (logistic) | 0.7421 | 68.56% | 0.5947 | 0.2041 |
| **Shipped: logistic classifier** | **0.7564** | **70.49%** | **0.5816** | **0.1989** |
| Margin regression -> normal CDF | 0.7541 | 70.09% | 0.5850 | 0.2000 |

Log loss +0.0034 (SE 0.0015), better in 1 of 4 folds. Not kept; the classifier stays.

### 5.3 More seasons, special seasons, and XGBoost (not kept)

`ingest.py --since 2018-19` added 2018-19 through 2022-23 (19,038 team rows, 202,839 player
rows; the unused `*_RANK` columns are now dropped, keeping the player file at 34 MB).
Special seasons:
- **2019-20 restart in the Orlando bubble** (88 seeding games from 2020-07-30): neutral site, so
  Elo applies no home advantage to them and they are not used as training rows (they still
  update Elo, rolling stats and travel).
- **2020-21** (72 games, limited crowds): kept as normal games. Home advantage is tuned on the
  training data, and the long 2020 break is handled by the 7-day rest cap and the
  between-season Elo regression.

Injury reports for the older seasons: the parser was extended to the pre-2021 report layout
(extra Category / Previous Status columns, team names wrapped over two lines) and the backfill
archived 2018-12 to 2020-01-17. **The rest of 2020-01 to 2022-23 is not archived**: during the
backfill the NBA's CDN started answering every report request with HTTP 403, including reports
it had served earlier that day (most likely rate limiting after many requests, made worse by a
short-lived parallel probe I added for live lookups and then removed). The backfill now waits
1 second between requests and stops after 5 game days in a row with no report. Those seasons
use the "missed the previous game" fallback. Re-running
`python src/injury_reports.py --backfill --since 2018-10-16` resumes where it stopped.

Same 1,762 held-out games; only the training history differs:

| Model | ROC-AUC | Accuracy | Log loss | Brier |
|---|---|---|---|---|
| Elo only, 2023-24 on | 0.7421 | 68.56% | 0.5947 | 0.2041 |
| **Shipped: logistic, 2023-24 on** | **0.7564** | **70.49%** | **0.5816** | **0.1989** |
| Logistic, 2018-19 on | 0.7561 | 70.26% | 0.5818 | 0.1989 |
| XGBoost depth 2, early stopping, 2018-19 on | 0.7539 | 69.69% | 0.5873 | 0.2006 |
| XGBoost depth 3, early stopping, 2018-19 on | 0.7508 | 69.47% | 0.5896 | 0.2016 |
| XGBoost depth 4, early stopping, 2018-19 on | 0.7510 | 69.41% | 0.5890 | 0.2015 |
| XGBoost depth 2, early stopping, 2023-24 on | 0.7551 | 70.09% | 0.5876 | 0.2008 |
| XGBoost depth 3, early stopping, 2023-24 on | 0.7501 | 69.69% | 0.5911 | 0.2021 |

More history: +0.0002 (SE 0.0018), 2 of 4 folds, so training stays on 2023-24 onward
(`FIRST_SEASON_ID`). XGBoost (shallow trees, min_child_weight 20-40, subsample 0.8, early
stopping on the most recent 15% of each fold's training games, 150-290 rounds) still loses to
logistic regression by 0.006-0.010 log loss with either history, better in at most 1 of 4
folds. With ~3,500-9,000 games and a dozen mostly linear differentials, trees add variance and
no signal.

### 5.4 Shipped model

`SHIPPED_MODEL = 'logistic'` (unchanged): logistic regression, C = 0.1, standardized features,
trained on 2023-24 onward with the tuned Elo.

## Phase 6: New features

`src/context.py` computes all candidates per team-game from earlier games only. Live
predictions append the upcoming game to the game log and call the same function; a test checks
that live values equal the training values. Arena coordinates, time zones and elevations are in
`data/arenas.csv` (the Clippers use Intuit Dome for all seasons, about 15 km from their old
arena). Each group was added on its own to the shipped features (1,762 held-out games; baseline
0.5816):

| Feature group | ROC-AUC | Log loss | Change (SE) | Folds better | Kept |
|---|---|---|---|---|---|
| Travel km since last game + time zones crossed | 0.7568 | 0.5810 | -0.0006 (0.0005) | 4/4 | **yes** |
| Schedule density: games in last 4 / 7 days, 3-in-4 | 0.7564 | 0.5817 | +0.0001 (0.0004) | 2/4 | no |
| Altitude: road team at Denver or Utah | 0.7563 | 0.5816 | +0.0000 (0.0000) | 0/4 | no |
| SOS-adjusted net rating (last 10 games) | 0.7561 | 0.5818 | +0.0002 (0.0002) | 1/4 | no |

Travel is a small gain, about one standard error, but it improved every fold, so it is kept
under the rules. Altitude adds nothing beyond the home-court terms already in the model;
schedule density adds nothing beyond rest days and back-to-backs; the opponent-adjusted net
rating adds nothing beyond Elo and rolling plus/minus.

Final shipped model (Phases 0-6), walk-forward on 1,762 held-out games:
**ROC-AUC 0.7570, accuracy 70.60%, log loss 0.5808, Brier 0.1986** (Elo only: 0.7421 / 0.5947).
On the 1,223 of those games with a Kalshi price: model log loss 0.5821 vs market 0.5694
(50/50 blend 0.5716, still worse than the market alone). These final figures include the
tip-time correction below.

### Correction: 11:00 tip times

Injury reports print tip times without AM/PM. The parser treated "11:00" as 11 AM, but every
such game is an 11 PM Eastern tip (late West Coast games, e.g. PHX@SAC on 2026-03-03). For
those 15 games the "last report before tip-off" cutoff was 10:30 AM (so they fell back to the
missed-previous-game guess) and their Kalshi price was taken at 11 AM instead of tip-off. Both
are fixed; the market price is still pre-game either way, so there was no leakage. Effect on the
shipped model: log loss 0.5810 -> 0.5808, AUC 0.7568 -> 0.7570; earlier tables in this file
are left as measured at the time.

## Phase 7: Dashboard

No model change; these features can only be measured once games are played.
- **Model vs Market** page (and `python src/daily_slate.py`): today's regular-season games
  with model probability, Kalshi probability and the gap, rows highlighted at 5+ points, with
  a note that smaller gaps are within fees and noise and that the market has been the more
  accurate of the two.
- **Prediction log**: every pre-tip-off prediction is appended to `data/prediction_log.csv`
  (timestamp, model and market probability, bid/ask, model version hash, players counted out,
  and the full feature row). Rows are never rewritten. The refresh scripts run the slate daily.
- **Track Record** page: joins the last pre-tip-off prediction for each game to the result and
  shows accuracy, log loss, Brier score, running log loss and calibration for the model and the
  market. It is empty until the 2026-27 regular season starts (2026-10-20).
- Sidebar and Model Performance page read the latest walk-forward metrics and the
  model-vs-market comparison written by `train.py`.
