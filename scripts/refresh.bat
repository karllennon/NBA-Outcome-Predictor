@echo off
REM Refreshes data and retrains locally: ingest -> pipeline -> train, then the paper-trading
REM steps (freeze, slate, tip-off prices, report).
REM stats.nba.com blocks GitHub-hosted runners, so this runs on your own machine.
REM Scheduled with Task Scheduler (see docs\paper_trading_plan.md), e.g.:
REM   schtasks /Create /SC DAILY /ST 10:00 /TN "NBA refresh" /TR "\"C:\path\to\nba-outcome-predictor\scripts\refresh.bat\""
REM If a step fails (stats.nba.com sometimes returns an error page for a few minutes), the
REM remaining steps still run on the data already on disk, and the run exits with code 1.
setlocal
cd /d "%~dp0\.."

set PY=python
if exist venv\Scripts\python.exe set PY=venv\Scripts\python.exe
set "FAILED=0"

echo === Refresh started %DATE% %TIME% ===
"%PY%" -u src\ingest.py || (echo [!] data download failed: continuing with the data already on disk & set "FAILED=1")
"%PY%" -u src\data_pipeline.py || (echo [!] pipeline failed: keeping the previous model & set "FAILED=1" & goto :after_train)
"%PY%" -u src\train.py || (echo [!] training failed: keeping the previous model & set "FAILED=1")
:after_train
REM Freeze a copy of the trained models for the season's paper test (once, on/after 2026-10-19)
"%PY%" -u src\frozen_model.py --freeze-on 2026-10-19 || (echo [!] model freeze failed & set "FAILED=1")
REM Archive the latest injury report and today's Kalshi prices (non-fatal: offseason or outage)
"%PY%" -u src\injury_reports.py || echo [!] injury report fetch failed
REM Log today's pre-tip-off predictions with Kalshi prices (also appends a market snapshot)
"%PY%" -u src\daily_slate.py || echo [!] slate / prediction log failed
REM Tip-off prices of last night's logged games (for closing line value; local, git-ignored)
"%PY%" -u src\market_odds.py --tip-prices || echo [!] tip-off price fetch failed
REM Paper-trade report from the log and last night's results (local, git-ignored)
"%PY%" -u src\paper_report.py > nul || echo [!] paper-trade report failed

if "%FAILED%"=="1" (
  echo === Refresh FINISHED WITH ERRORS %DATE% %TIME% ===
  exit /b 1
)
echo === Refresh finished %DATE% %TIME% ===
exit /b 0
