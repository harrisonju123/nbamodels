#!/bin/bash
#
# Install Cron Jobs - Quick Installer
#
# This script automatically installs the cron jobs for the betting system
# Run this after setup_droplet.sh completes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================="
echo "Installing NBA Betting Cron Jobs"
echo "========================================="
echo ""

# Detect timezone
if command -v timedatectl &> /dev/null; then
    TIMEZONE=$(timedatectl | grep "Time zone" | awk '{print $3}')
    echo "Detected timezone: $TIMEZONE"
else
    TIMEZONE="unknown"
    echo "Unable to detect timezone - assuming UTC"
fi

# Set cron times based on timezone
if [ "$TIMEZONE" = "UTC" ] || [ "$TIMEZONE" = "unknown" ]; then
    BETTING_HOUR=21  # 4 PM ET = 9 PM UTC (EST)
    REPORT_HOUR=4    # 11 PM ET = 4 AM UTC (next day, EST)
    echo "Using UTC times: Betting at ${BETTING_HOUR}:00, Report at ${REPORT_HOUR}:00"
elif [[ "$TIMEZONE" == *"America/New_York"* ]]; then
    BETTING_HOUR=16  # 4 PM local
    REPORT_HOUR=23   # 11 PM local
    echo "Using ET times: Betting at ${BETTING_HOUR}:00, Report at ${REPORT_HOUR}:00"
else
    echo "Warning: Unrecognized timezone $TIMEZONE"
    echo "Defaulting to UTC times"
    BETTING_HOUR=21
    REPORT_HOUR=4
fi

echo ""

# Create temporary crontab file
TEMP_CRON=$(mktemp)

# Preserve existing crontab (if any)
crontab -l > $TEMP_CRON 2>/dev/null || true

# Check if NBA cron jobs already exist
if grep -q "NBA Betting System" $TEMP_CRON; then
    echo "⚠️  Existing NBA betting cron jobs found"
    echo "Removing old entries..."
    grep -v "NBA Betting System" $TEMP_CRON > ${TEMP_CRON}.new
    grep -v "cron_betting.sh" ${TEMP_CRON}.new > ${TEMP_CRON}.clean || true
    grep -v "send_daily_report.py" ${TEMP_CRON}.clean > $TEMP_CRON || true
fi

# Add new cron jobs
cat >> $TEMP_CRON << EOF

# NBA Betting System - Auto-generated $(date)
PATH=/usr/local/bin:/usr/bin:/bin

# Daily betting pipeline at 4 PM ET
0 ${BETTING_HOUR} * * * $PROJECT_DIR/scripts/cron_betting.sh >> $PROJECT_DIR/logs/cron_main.log 2>&1

# Daily Discord report at 11 PM ET
0 ${REPORT_HOUR} * * * cd $PROJECT_DIR && $PROJECT_DIR/venv/bin/python scripts/send_daily_report.py >> $PROJECT_DIR/logs/daily_report.log 2>&1
EOF

# Install new crontab
crontab $TEMP_CRON

# Cleanup
rm $TEMP_CRON

echo ""
echo "✅ Cron jobs installed successfully!"
echo ""
echo "Scheduled jobs:"
echo "─────────────────────────────────────────"
crontab -l | grep -A 5 "NBA Betting System"
echo "─────────────────────────────────────────"
echo ""
echo "Next runs:"
echo "  Betting pipeline: Today at ${BETTING_HOUR}:00 (server time)"
echo "  Daily report: Today at ${REPORT_HOUR}:00 (server time)"
echo ""
echo "To verify:"
echo "  crontab -l                    # List all cron jobs"
echo "  tail -f logs/cron_betting.log # Monitor logs"
echo ""
echo "To remove:"
echo "  crontab -e                    # Edit and delete NBA lines"
echo ""
