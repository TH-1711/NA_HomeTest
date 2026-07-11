#!/bin/sh
# Runs main.py once, then sleeps 24h, forever. Intended for Render's
# Background Worker service type (which CAN attach a persistent disk),
# not Cron Job (which CANNOT -- see README_step3.md for why).
set -e

STATE_DIR="${STATE_DIR:-/app/state}"
ARTICLES_DIR="${ARTICLES_DIR:-/app/articles}"
LIMIT="${SCRAPE_LIMIT:-40}"

while true; do
    echo "=== Run started at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    python main.py \
        --articles-dir "$ARTICLES_DIR" \
        --state-file "$STATE_DIR/state.json" \
        --build-result "$STATE_DIR/build_result.json" \
        --limit "$LIMIT" || echo "Run failed, will retry after sleep"
    echo "=== Run finished, sleeping 24h ==="
    sleep 86400
done
