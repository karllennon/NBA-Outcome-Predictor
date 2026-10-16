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

**Test (not kept).** 21 leaderboard snapshots were downloaded with the site's CSV button (the day
before each season opener and the 1st of each month, 2023-10 to 2026-04), each checked against
the page for its date. A player's value is (DPM - (-2.0)) x projected minutes / 48 from the
latest snapshot dated strictly before the game (the "DPM now" column is ignored). Same rotation
and same absent players as the shipped feature; only the value of each absence changes.
Measured on the final shipped model (1,762 held-out games):

| Model | ROC-AUC | Accuracy | Log loss | Brier |
|---|---|---|---|---|
| Elo only (logistic) | 0.7421 | 68.56% | 0.5947 | 0.2041 |
| **Shipped: box-score impact** | **0.7579** | **70.43%** | **0.5795** | **0.1982** |
| DARKO replaces box impact | 0.7548 | 69.92% | 0.5824 | 0.1993 |
| DARKO added as a second feature | 0.7577 | 69.98% | 0.5797 | 0.1983 |

Replace: +0.0029 log loss (SE 0.0016), worse in all 4 folds. Add: +0.0002 (SE 0.0004). The two
injury measures correlate 0.81; where they differ, the box-score version predicts better. Not
kept. The code stays (`src/darko.py`, `python src/experiments.py darko_impact`); the CSVs are
not committed (they are DARKO's data) and go in `data/darko/` to rerun.

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
  injury reports) for nearly every regular-season game since then (`data/market_history.csv`).
  Settled markets older than Kalshi's archive cutoff come from its `/historical/` endpoints.

**Data terms (added later).** Kalshi's API Developer Agreement (v1.1) limits API use to
facilitating a member's own trading and prohibits collecting or storing API data except for
that purpose and sharing it with third parties. Kalshi data is therefore kept local only: the
market files are git-ignored, were removed from this repository's history, and the exact
model-vs-market figures are not published here. They are reproduced locally by
`python src/train.py` (written to the git-ignored `data/market_metrics.csv` and shown on the
dashboard's Model page) and `python src/experiments.py market_comparison`.

**Model vs market (summary).** On the held-out 2025-26 games with a pre-tip-off price, the
market was clearly more accurate than the model (the log-loss gap is about three standard
errors), and a 50/50 average of model and market did not beat the market alone. On this
evidence the model adds no information beyond the tip-off market price, which also reflects
late scratches and lineup news the model only partly sees.

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
**ROC-AUC 0.7579, accuracy 70.43%, log loss 0.5795, Brier 0.1982** (Elo only: 0.7421 / 0.5947).
The Kalshi market price remains more accurate on the games where it exists (figures kept local,
see Phase 4). These final figures include both corrections below.

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

### Correction: missing pre-tip-off reports for early games

The Phase 2 backfill first read the report published around 5:45 PM, then looked for the latest
report before each tip-off, but stopped immediately for tips earlier than that first report
(afternoon and 6 PM games). Those team-games fell back to the "missed the previous game" guess
when no earlier report was archived. `fill_pre_tip` now searches back from each tip's cutoff
and stops only at a report it already has, and `python src/injury_reports.py --repair` filled
the gaps (282 reports added; 18 tips still have no report 30+ minutes before tip-off). Coverage
of team-games by a pre-tip-off report went from 97.6% to 99.3%. No leakage was involved (the
fallback only uses earlier games). Effect on the shipped model: log loss 0.5808 -> 0.5795,
AUC 0.7570 -> 0.7579.

## Spread model

A second output next to the win probability: the predicted home point margin from a Ridge
regression on the same features (`src/spread_model.py`, the Phase 5.2 margin regression moved
into a shared module). The win probability still comes from the logistic classifier. The spread
is shown in betting convention: the home line is minus the predicted margin, so "home -4.5"
means the home team is predicted to win by 4.5 (displayed to the nearest 0.5, favorite first,
e.g. "BOS -3.5").

### Accuracy (walk-forward, same 4 folds, Elo re-tuned per fold; 1,762 held-out games)

| Predicted margin | MAE (pts) | RMSE (pts) | Bias (pts) | MAE by fold |
|---|---|---|---|---|
| Always the training-period average home margin | 13.08 | 16.40 | +0.08 | 12.4-14.7 |
| Elo-only spread (margin regressed on ELO_DIFF) | 11.28 | 14.44 | +0.18 | 10.8-11.8 |
| **Spread model (all features)** | **10.95** | **14.00** | **+0.19** | 10.6-11.6 |

Spread model vs Elo-only: MAE -0.33 pts (SE 0.07), better in all 4 folds; vs the average:
-2.13 pts (SE 0.16). For scale, the average absolute home margin in these games was 13.2 points,
so a typical game still lands about 11 points from the prediction.

### Spread vs win probability

They pick different favorites in 59 of 1,762 games (3.3%). All are near toss-ups: in those
games the win probability is at most 6.6 points from 50% (median 1.5) and the predicted margin
at most 1.8 points (median 0.4). When they disagree, the win probability picked the winner in
52.5% and the spread in 47.5% (59 games, no real difference). The dashboard marks these games as
close to a toss-up.

### Spread vs the Kalshi market line

Kalshi has NBA spread markets (series `KXNBASPREAD`, checked against its docs and API in October
2026): a ladder of "team wins by over N.5 points" markets per game. The market's line is where
the favorite's ladder crosses 50 cents, interpolated from pre-tip-off prices
(`python src/market_odds.py --spread-history`, about 2 hours, local and git-ignored). A usable
line exists for about 86% of the held-out 2025-26 games. Live spread ladders are appended to a
local snapshot file by the slate and by `python src/market_odds.py`.

Result (exact figures kept local, Kalshi terms): the market line is slightly more accurate than
the model's spread (a gap of about 0.2 points of mean absolute error, under two standard
errors), and the model lands on the right side of the market line about half the time, no
better than a coin flip. The model's spread still beats the Elo-only spread on the same games.
Where the model and the line differ by 4+ points, the model's side did a little better than
half, but that slice was chosen after seeing the results and is within noise.

### Spread uncertainty (sigma) and calibration

Each fold's sigma is the residual standard deviation on the last 20% of that fold's training
games, predicted by a model fit on the earlier 80% (`spread_model.fit_target`); test games are
never used. Fold sigmas: 13.7, 13.8, 14.1 and 14.5 points (held-out residual SD: 14.0). The
shipped model's sigma, from the same procedure on all games, is 14.4.

Calibration on held-out games (`python src/experiments.py spread_calibration`): for every game,
lines at the predicted margin plus -12, -8, -4, 0, +4, +8 and +12 points (.5 lines), the predicted
P(margin > line) against how often the margin actually cleared it:

| Predicted chance | Cases | Predicted (avg) | Actual |
|---|---|---|---|
| 0-20% | 1,670 | 18.6% | 17.1% |
| 20-30% | 1,854 | 26.9% | 24.6% |
| 30-40% | 1,762 | 37.4% | 36.8% |
| 45-55% | 1,762 | 48.6% | 48.9% |
| **55-60%** | **971** | **59.2%** | **60.8%** |
| 60-70% | 1,440 | 64.6% | 65.2% |
| 70-80% | 2,496 | 75.4% | 76.5% |
| 80-100% | 379 | 80.4% | 82.1% |

Close to calibrated: predictions with about a 60% chance of covering covered about 61% of the
time. Below 30% the model is a little too generous to the underdog side (predicted 26.9%,
actual 24.6%). The rows share games, so they are not independent.

## Model improvements for the 2026-27 season

Candidates are tested with the same walk-forward protocol (1,762 held-out games, Elo re-tuned
per fold) and kept only if held-out log loss improves.

### Regularization strength (not kept)

The logistic regression's C = 0.1 had never been tuned, and the model's predictions are cautious
at the extremes. `python src/experiments.py regularization`:

| Model | ROC-AUC | Log loss | Change (SE) | Folds better |
|---|---|---|---|---|
| Elo only (logistic) | 0.7421 | 0.5947 | | |
| **Shipped: C = 0.1** | **0.7579** | **0.5795** | | |
| C = 0.03 (fixed) | 0.7579 | 0.5800 | +0.0005 (0.0006) | 2/4 |
| C = 0.3 / 1 / 10 (fixed) | 0.7577-0.7573 | 0.5796-0.5798 | +0.0001 to +0.0003 | 1-2/4 |
| C = 0.01 / 0.003 (fixed) | 0.7564 / 0.7524 | 0.5826 / 0.5904 | +0.0031 / +0.0109 | 2/4, 0/4 |
| C tuned per fold (last 20% of training games) | 0.7572 | 0.5803 | +0.0008 (0.0006) | 1/4 |
| C = 0.1 + recalibration fitted per fold | 0.7525 | 0.5865 | +0.0070 (0.0016) | 0/4 |

Log loss is flat for C from 0.03 up, so the shipped value is already in the best range; the
per-fold choice (0.03, 0.3, 0.03, 0.03) adds noise. A Platt-style recalibration on the most
recent training games hurt in every fold (fitted slopes 0.87-1.21, intercepts -0.21 to +0.04 move
around from fold to fold). The model's caution at the extremes is therefore not a shrinkage
setting but missing information; nothing was changed.

### Returning players (not kept)

The injury feature subtracts players who are out but never credits players coming back, while
the team's rolling form (last 10 games) was built without them; players absent 21+ games are not
in the rotation at all. `InjuryModel.returning` finds players back today from an injury absence:
latest team is this one, missed 3+ of its games in a row, listed Out on the last report before the
team's previous game, not listed Out on today's. Each is valued at his impact score times the share
of the last 10 team games he missed; RETURN_DIFF = (home - away) / 10. The largest 2025-26 cases are
the expected ones (Jokic after 16 games, Wembanyama 12, Embiid 13, Tatum 63).
`python src/experiments.py returning_players`:

| Model | ROC-AUC | Log loss | Change (SE) | Folds better |
|---|---|---|---|---|
| Elo only (logistic) | 0.7421 | 0.5947 | | |
| **Shipped features** | **0.7579** | **0.5795** | | |
| + RETURN_DIFF (48% of games) | 0.7576 | 0.5803 | +0.0008 (0.0007) | 1/4 |
| + RETURN_DIFF, impact >= 20 only (14% of games; second try) | 0.7575 | 0.5803 | +0.0008 (0.0007) | 1/4 |

Neither version helps, including the one limited to rotation-level players, which was chosen after
seeing the first result. Elo and the rolling form appear to recover quickly enough once a player is
back. The code stays, switched off by default (`returning=False`), so the shipped features are
unchanged.

### Power ratings (kept)

`src/power_rating.py` fits every team's strength jointly from earlier game margins before each
game day: home margin = home court + rating(home) - rating(away), by ridge regression with each
game's weight halving every `half_life` days and a penalty `lam` shrinking ratings toward zero.
Unlike Elo, each refit adjusts every team for the strength of all the opponents it played.
RATING_DIFF = home minus away rating. The half-life (15, 30, 60, 120 days) and penalty (1, 5, 20)
are chosen inside each fold by the rating's own margin error on that fold's training games, like
the Elo settings; every fold chose 60 days and 5, which are the shipped `power_rating.PARAMS`, so
the walk-forward results with the fixed settings are identical. On its own the rating predicts
margins about as well as Elo (held-out MAE 11.35 vs 11.28). `python src/experiments.py power_ratings`:

| Model | ROC-AUC | Accuracy | Log loss | Change (SE) | Folds better |
|---|---|---|---|---|---|
| Elo only (logistic) | 0.7421 | 68.56% | 0.5947 | | |
| Before: no power rating | 0.7579 | 70.43% | 0.5795 | | |
| **+ RATING_DIFF (kept)** | **0.7629** | **70.66%** | **0.5763** | **-0.0032 (0.0015)** | **3/4** |
| Power rating instead of Elo (reference) | 0.7636 | 71.06% | 0.5760 | -0.0036 (0.0019) | 3/4 |

Kept: the largest gain since the injury reports, better in 3 of 4 folds. It is added rather than
replacing Elo (the pre-specified variant; the two score about the same). The spread model uses the
same features and improves too: held-out MAE 10.95 -> 10.91 points (-0.046, SE 0.025, 3 of 4
folds better), still better than the Elo-only spread in all four folds. In the refit model the
power rating has the largest weight (+0.64 per standard deviation; Elo +0.20, injuries +0.37).
Live predictions compute the rating with `power_rating.ratings_as_of` on the same games, and a test
checks that it equals the training value.

Shipped model after this change, walk-forward on 1,762 held-out games: **ROC-AUC 0.7629, accuracy
70.66%, log loss 0.5763, Brier 0.1966**; spread MAE 10.91 points.

### Power rating variants (not kept)

Two changes to the shipped rating, each replacing RATING_DIFF, with every setting chosen inside
each fold on its training games. `python src/experiments.py power_rating_variants`:

| Model | ROC-AUC | Log loss | Change (SE) | Folds better |
|---|---|---|---|---|
| **Shipped: rating on raw margins** | **0.7629** | **0.5763** | | |
| Blowout cap (margins clipped to 15 / 20 / 25 / none) | 0.7628 | 0.5765 | +0.0002 (0.0007) | 1/4 |
| Margin per 100 possessions | 0.7632 | 0.5761 | -0.0002 (0.0002) | 3/4 |

The cap is not kept: every fold chose the loosest cap (25 points), so garbage time is not
distorting the rating. Per-possession margins are 0.0002 lower, about one standard error, and were
the better of the two variants tried; a gain that small is within noise and not worth changing the
shipped feature for, so it is not kept either. The options stay in `power_rating.pregame_ratings`
(`target`, `cap`) to re-test with more data.

### Back-to-backs (not kept)

The shipped model has one symmetric back-to-back feature (B2B_DIFF, second night of a
back-to-back, home minus away; 17.6% of games have a home and 18.5% an away back-to-back). Three
refinements were fixed in advance and each tested on its own. `python src/experiments.py back_to_backs`:

| Model | ROC-AUC | Log loss | Change (SE) | Folds better |
|---|---|---|---|---|
| **Shipped features** | **0.7629** | **0.5763** | | |
| A: home and away back-to-backs as separate features | 0.7630 | 0.5762 | -0.0001 (0.0005) | 3/4 |
| B: + back-to-back x km travelled since the previous game | 0.7627 | 0.5764 | +0.0001 (0.0002) | 2/4 |
| C: + back-to-back against an opponent with 2+ days of rest | 0.7629 | 0.5763 | +0.0000 (0.0000) | 3/4 |

None changes held-out log loss by more than 0.0001; the existing rest, back-to-back and travel
features already capture what these add. Not kept (the candidate columns stay in
`create_matchup_data` as extras, unused by the model).

### Player-based team strength (built; to be tested on 2026-27 games only)

After nine experiments on the same 1,762 held-out games, those games are no longer used for new
tests (repeated testing on one set makes a chance pass more likely). This candidate is built now
and will be scored only on 2026-27 games: `python src/experiments.py player_strength` reports
nothing until 200 games from 2026-10-20 exist.

`src/player_strength.py` builds a team's strength for a game from the players expected to play:
each player's box-score impact per minute (recent games weighted more, shrunk toward a
replacement-level value), times his minutes, with the game's 240 minutes given to the available
players in order of their usual minutes. Available = latest team is this one, played in one of its
last 10 games or appeared on its injury report in the last 30 days, and not listed Out on the last
report before tip-off. Trades, injuries and returns change the strength immediately, unlike Elo or
the power rating. PLAYER_STRENGTH_DIFF = (home - away) / 10; not a shipped feature.

Settings, chosen by how well the strength difference predicts home margins on games before
2026-10-20 only (training data for the 2026-27 test; mean squared error, points^2):

| Half-life (days) | Shrinkage 100 min | 300 | 1,000 | 3,000 | 10,000 |
|---|---|---|---|---|---|
| 60 | 221.72 | 219.44 | 218.04 | | |
| 120 | 220.56 | 218.33 | 216.16 | **215.50** | 215.60 |
| 240 | 220.97 | 219.01 | 216.71 | 215.68 | 215.68 |

The first grid chose its edge (1,000 minutes), so it was widened; the chosen 120 days and 3,000
minutes (about a season of a starter's minutes) are `player_strength.PARAMS`. The heavy shrinkage
compresses differences between players (Denver on 2026-02-01: 155.3, or 148.6 without Jokic); the
model's weight on the feature rescales it. Late-season contenders that rest starters (e.g. Oklahoma
City with eight players listed Out on 2026-04-10) drop accordingly.

Two changes before any test, prompted by Tyrese Haliburton (out all of 2025-26, back for 2026-27):
- **Availability:** a player on the team's injury report for its previous game now counts as a
  candidate even if that game was months ago, so a player returning from a season-long absence is
  included at the season opener (a test covers it). Before, the offseason meant no recent games
  or reports, and he was left out.
- **How old games fade:** calendar decay rates a player who has not played for a long time close to
  replacement level. Two alternatives were tuned the same way, on games before 2026-10-20:

| Decay of a player's past games | Best setting | Mean squared error |
|---|---|---|
| **Calendar days (kept)** | 120 days, 3,000 minutes | **215.50** |
| Calendar days, each break between his games capped at 30 days | 120 days, 10,000 minutes | 216.65 |
| His own games played | 80 games, 3,000 minutes | 218.42 |

  Calendar decay fits the typical game best (it also reflects aging and changing roles), so it
  stays; the cost is that a star back from a season-long absence starts near replacement value
  until he plays again. The dashboard's Predictor page has a what-if view for that case.

**Rosters (added before any test).** `src/rosters.py` saves every team's roster from stats.nba.com
each morning (`data/rosters/YYYY-MM-DD.csv`, from 2026-10-14). The NBA serves only today's roster,
so rosters are used only for games on or after a snapshot and never to rebuild older seasons,
where they would reveal moves that had not happened yet. When a snapshot from the game day (or up
to 3 days before) exists, the player strength's candidates for a team are its roster that day,
matched to box scores by player id: offseason signings and trades count for their new team at once
(e.g. Kelly Oubre Jr. and Larry Nance Jr. for Indiana in 2026-27, instead of their old teams), and
players who left are dropped. Players on a roster with no NBA games get replacement value and no
expected minutes. Every 2026-27 game the experiment scores will have a pre-game snapshot. The
dashboard's what-if list also keeps only players on the team's current roster.
