#!/bin/bash
#
# Cron-compatible multi-strategy betting script
#
# Usage: Add to crontab to run daily before games start:
#   0 16 * * * /path/to/nbamodels/scripts/cron_betting.sh
#
# This will run the multi-strategy pipeline each day at 4 PM ET
# (games typically start at 7 PM ET, giving 3 hours for odds to stabilize)

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Create logs directory
mkdir -p logs

# Log start
echo "========================================" >> logs/cron_betting.log
echo "Multi-strategy betting pipeline started: $(date)" >> logs/cron_betting.log

# Run the multi-strategy pipeline (paper trading mode by default)
# Remove --dry-run once you've validated it's working correctly
python scripts/daily_multi_strategy_pipeline.py 2>&1 | tee -a logs/cron_betting.log

# Log completion
echo "Multi-strategy betting pipeline completed: $(date)" >> logs/cron_betting.log
echo "========================================" >> logs/cron_betting.log

# Optional: Send notification when done
# Uncomment once you've tested the pipeline
# python scripts/send_bet_notifications.py 2>&1 | tee -a logs/cron_betting.log
