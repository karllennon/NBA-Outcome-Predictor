@echo off
REM Refreshes data and retrains locally: ingest -> pipeline -> train.
REM stats.nba.com blocks GitHub-hosted runners, so this runs on your own machine.
REM Schedule with Task Scheduler, e.g. daily at 10:00:
REM   schtasks /Create /SC DAILY /ST 10:00 /TN "NBA refresh" /TR "\"C:\path\to\nba-outcome-predictor\scripts\refresh.bat\""
setlocal
cd /d "%~dp0\.."

set PY=python
if exist venv\Scripts\python.exe set PY=venv\Scripts\python.exe

echo === Refresh started %DATE% %TIME% ===
"%PY%" -u src\ingest.py || goto :fail
"%PY%" -u src\data_pipeline.py || goto :fail
"%PY%" -u src\train.py || goto :fail
REM Archive the latest injury report and today's Kalshi prices (non-fatal: offseason or outage)
"%PY%" -u src\injury_reports.py || echo [!] injury report fetch failed
"%PY%" -u src\market_odds.py || echo [!] Kalshi snapshot failed
echo === Refresh finished %DATE% %TIME% ===
exit /b 0

:fail
echo === Refresh FAILED %DATE% %TIME% ===
exit /b 1
