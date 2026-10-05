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
