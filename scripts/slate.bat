@echo off
REM Pre-tip-off slate only (no ingest or retraining): latest injury report, then today's
REM predictions and Kalshi prices appended to the local prediction log. Safe to run repeatedly:
REM the log is append-only and only games that have not started are logged.
REM Schedule every 30 minutes on game days (see docs\paper_trading_plan.md); output goes to
REM logs\slate.log.
setlocal EnableDelayedExpansion
cd /d "%~dp0\.."
if not exist logs mkdir logs

set PY=python
if exist venv\Scripts\python.exe set PY=venv\Scripts\python.exe

>> logs\slate.log 2>&1 (
  echo === Slate started !DATE! !TIME! ===
  "%PY%" -u src\injury_reports.py || echo [warn] injury report fetch failed
  "%PY%" -u src\daily_slate.py || echo [warn] slate / prediction log failed
  echo === Slate finished !DATE! !TIME! ===
)
exit /b 0
