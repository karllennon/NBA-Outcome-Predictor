#!/usr/bin/env bash
# Refreshes data and retrains locally: ingest -> pipeline -> train.
# stats.nba.com blocks GitHub-hosted runners, so this runs on your own machine.
# Schedule with cron, e.g. daily at 10:00:
#   0 10 * * * /path/to/nba-outcome-predictor/scripts/refresh.sh >> /path/to/refresh.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -x venv/bin/python ]; then PY=venv/bin/python
elif [ -x venv/Scripts/python.exe ]; then PY=venv/Scripts/python.exe
else PY=python3; fi

echo "=== Refresh started $(date) ==="
"$PY" -u src/ingest.py
"$PY" -u src/data_pipeline.py
"$PY" -u src/train.py
# Archive the latest injury report and today's Kalshi prices (non-fatal: offseason or outage)
"$PY" -u src/injury_reports.py || echo "[!] injury report fetch failed"
# Log today's pre-tip-off predictions with Kalshi prices (also appends a market snapshot)
"$PY" -u src/daily_slate.py || echo "[!] slate / prediction log failed"
# Paper-trade report from the log and last night's results (local, git-ignored)
"$PY" -u src/paper_report.py > /dev/null || echo "[!] paper-trade report failed"
echo "=== Refresh finished $(date) ==="
