#!/usr/bin/env bash
# Refreshes data and retrains locally: ingest -> pipeline -> train, then the paper-trading
# steps (freeze, slate, tip-off prices, report).
# stats.nba.com blocks GitHub-hosted runners, so this runs on your own machine.
# Schedule with cron, e.g. daily at 10:00:
#   0 10 * * * /path/to/nba-outcome-predictor/scripts/refresh.sh >> /path/to/refresh.log 2>&1
# If a step fails (stats.nba.com sometimes returns an error page for a few minutes), the
# remaining steps still run on the data already on disk, and the run exits with code 1.
set -uo pipefail
cd "$(dirname "$0")/.."

if [ -x venv/bin/python ]; then PY=venv/bin/python
elif [ -x venv/Scripts/python.exe ]; then PY=venv/Scripts/python.exe
else PY=python3; fi
FAILED=0

echo "=== Refresh started $(date) ==="
"$PY" -u src/ingest.py || { echo "[!] data download failed: continuing with the data already on disk"; FAILED=1; }
if "$PY" -u src/data_pipeline.py; then
  "$PY" -u src/train.py || { echo "[!] training failed: keeping the previous model"; FAILED=1; }
else
  echo "[!] pipeline failed: keeping the previous model"; FAILED=1
fi
# Freeze a copy of the trained models for the season's paper test (once, on/after 2026-10-19)
"$PY" -u src/frozen_model.py --freeze-on 2026-10-19 || { echo "[!] model freeze failed"; FAILED=1; }
# Archive the latest injury report and today's Kalshi prices (non-fatal: offseason or outage)
"$PY" -u src/injury_reports.py || echo "[!] injury report fetch failed"
# Log today's pre-tip-off predictions with Kalshi prices (also appends a market snapshot)
"$PY" -u src/daily_slate.py || echo "[!] slate / prediction log failed"
# Tip-off prices of last night's logged games (for closing line value; local, git-ignored)
"$PY" -u src/market_odds.py --tip-prices || echo "[!] tip-off price fetch failed"
# Paper-trade report from the log and last night's results (local, git-ignored)
"$PY" -u src/paper_report.py > /dev/null || echo "[!] paper-trade report failed"

if [ "$FAILED" = 1 ]; then
  echo "=== Refresh FINISHED WITH ERRORS $(date) ==="
  exit 1
fi
echo "=== Refresh finished $(date) ==="
