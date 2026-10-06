#!/usr/bin/env bash
# Pre-tip-off slate only (no ingest or retraining): latest injury report, then today's
# predictions and Kalshi prices appended to the local prediction log. Safe to run repeatedly:
# the log is append-only and only games that have not started are logged.
# Schedule every 30 minutes on game days (see docs/paper_trading_plan.md), e.g. with cron:
#   */30 11-22 * * * /path/to/nba-outcome-predictor/scripts/slate.sh   (times in US Eastern)
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

if [ -x venv/bin/python ]; then PY=venv/bin/python
elif [ -x venv/Scripts/python.exe ]; then PY=venv/Scripts/python.exe
else PY=python3; fi

{
  echo "=== Slate started $(date) ==="
  "$PY" -u src/injury_reports.py || echo "[!] injury report fetch failed"
  "$PY" -u src/daily_slate.py || echo "[!] slate / prediction log failed"
  echo "=== Slate finished $(date) ==="
} >> logs/slate.log 2>&1
