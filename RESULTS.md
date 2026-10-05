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
