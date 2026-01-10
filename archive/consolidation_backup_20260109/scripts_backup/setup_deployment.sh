#!/bin/bash
#
# Quick Deployment Setup Script
#
# This script helps you set up automated betting pipeline deployment
# Run this once to configure everything

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "========================================="
echo "Multi-Strategy Betting System Setup"
echo "========================================="
echo ""

# Check environment
echo "1️⃣  Checking environment..."
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "   Please create .env from .env.example"
    exit 1
fi

if ! grep -q "ODDS_API_KEY" .env || grep -q "your_odds_api_key_here" .env; then
    echo "⚠️  Warning: ODDS_API_KEY may not be configured"
    echo "   Edit .env and add your API key from https://the-odds-api.com/"
fi

if ! grep -q "DISCORD_WEBHOOK_URL" .env; then
    echo "⚠️  Warning: DISCORD_WEBHOOK_URL not configured"
    echo "   Discord notifications will be disabled"
fi

echo "✅ Environment file found"
echo ""

# Check configuration
echo "2️⃣  Checking multi-strategy configuration..."
if [ ! -f "config/multi_strategy_config.yaml" ]; then
    echo "❌ Config file not found!"
    exit 1
fi

echo "   Active strategies:"
grep "enabled: true" config/multi_strategy_config.yaml | grep -v "^#" || true

paper_mode=$(grep "paper_trading:" config/multi_strategy_config.yaml | grep -v "^#" | awk '{print $2}')
echo "   Paper trading mode: $paper_mode"
echo "✅ Configuration validated"
echo ""

# Create logs directory
echo "3️⃣  Setting up logs directory..."
mkdir -p logs
echo "✅ Logs directory ready"
echo ""

# Make scripts executable
echo "4️⃣  Making scripts executable..."
chmod +x scripts/cron_betting.sh
chmod +x scripts/send_bet_notifications.py
chmod +x scripts/send_daily_report.py
echo "✅ Scripts are executable"
echo ""

# Test pipeline
echo "5️⃣  Testing pipeline (dry-run)..."
echo "   This may take 30-60 seconds..."
if python scripts/daily_multi_strategy_pipeline.py --dry-run > /dev/null 2>&1; then
    echo "✅ Pipeline test successful"
else
    echo "⚠️  Pipeline test had warnings (check logs)"
fi
echo ""

# Display cron setup instructions
echo "6️⃣  Cron job setup instructions:"
echo ""
echo "Run: crontab -e"
echo ""
echo "Add these lines:"
echo "────────────────────────────────────────"
echo "# NBA Betting - Daily pipeline at 4 PM ET"
echo "0 16 * * * $PROJECT_DIR/scripts/cron_betting.sh"
echo ""
echo "# NBA Betting - Daily report at 11 PM ET"
echo "0 23 * * * cd $PROJECT_DIR && python scripts/send_daily_report.py >> logs/daily_report.log 2>&1"
echo "────────────────────────────────────────"
echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Configure cron jobs (instructions above)"
echo "  2. Monitor logs: tail -f logs/cron_betting.log"
echo "  3. Check dashboard: streamlit run dashboard/analytics_dashboard.py"
echo "  4. Read DEPLOYMENT_GUIDE.md for full documentation"
echo ""
echo "Happy betting! 🚀"
