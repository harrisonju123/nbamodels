#!/bin/bash
#
# Cron-compatible model retraining script
#
# Usage: Add to crontab to run daily:
#   0 6 * * * /path/to/nbamodels/scripts/cron_retrain.sh
#
# This will check for new data each morning at 6 AM and retrain if needed.

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
echo "========================================" >> logs/cron_retrain.log
echo "Cron job started: $(date)" >> logs/cron_retrain.log

# Run the retraining pipeline
python scripts/retrain_models.py 2>&1 | tee -a logs/cron_retrain.log

# Log completion
echo "Cron job completed: $(date)" >> logs/cron_retrain.log
echo "========================================" >> logs/cron_retrain.log
