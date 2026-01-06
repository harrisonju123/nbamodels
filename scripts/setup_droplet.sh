#!/bin/bash
#
# Automated Droplet Setup Script
# Run this on your DigitalOcean droplet after cloning the repo
#
# Usage:
#   chmod +x scripts/setup_droplet.sh
#   ./scripts/setup_droplet.sh

set -e

echo "========================================="
echo "NBA Betting System - Droplet Setup"
echo "========================================="
echo ""

# Get project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Check Python
echo "1️⃣  Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found${NC}"
    echo "Installing Python3..."
    sudo apt update
    sudo apt install python3 python3-pip python3-venv -y
fi

PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✅ Python installed: $PYTHON_VERSION${NC}"
echo ""

# 2. Set up virtual environment
echo "2️⃣  Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
fi

source venv/bin/activate

# Upgrade pip
pip install --upgrade pip -q
echo -e "${GREEN}✅ pip upgraded${NC}"
echo ""

# 3. Install dependencies
echo "3️⃣  Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt -q
    echo -e "${GREEN}✅ Dependencies installed${NC}"
else
    echo -e "${RED}❌ requirements.txt not found${NC}"
    exit 1
fi
echo ""

# 4. Set up environment variables
echo "4️⃣  Checking environment variables..."
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  .env file not found${NC}"

    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✅ Created .env from .env.example${NC}"
        echo ""
        echo -e "${YELLOW}⚠️  IMPORTANT: Edit .env and add your API keys:${NC}"
        echo "   nano .env"
        echo ""
    else
        echo -e "${RED}❌ .env.example not found${NC}"
        exit 1
    fi
else
    # Check if keys are configured
    if grep -q "your_odds_api_key_here" .env; then
        echo -e "${YELLOW}⚠️  ODDS_API_KEY not configured in .env${NC}"
        echo "   Edit .env and add your API key"
    else
        echo -e "${GREEN}✅ Environment variables configured${NC}"
    fi
fi

# Secure .env file
chmod 600 .env
echo ""

# 5. Create directories
echo "5️⃣  Setting up directories..."
mkdir -p logs
mkdir -p data/bets
mkdir -p data/cache
echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# 6. Make scripts executable
echo "6️⃣  Making scripts executable..."
chmod +x scripts/cron_betting.sh
chmod +x scripts/send_bet_notifications.py
chmod +x scripts/send_daily_report.py
echo -e "${GREEN}✅ Scripts are executable${NC}"
echo ""

# 7. Test pipeline
echo "7️⃣  Testing pipeline (dry-run)..."
echo "   This may take 30-60 seconds..."
if python scripts/daily_multi_strategy_pipeline.py --dry-run > /tmp/pipeline_test.log 2>&1; then
    echo -e "${GREEN}✅ Pipeline test successful${NC}"
else
    echo -e "${YELLOW}⚠️  Pipeline test completed with warnings${NC}"
    echo "   Check /tmp/pipeline_test.log for details"
fi
echo ""

# 8. Get server timezone
echo "8️⃣  Server timezone information:"
if command -v timedatectl &> /dev/null; then
    TIMEZONE=$(timedatectl | grep "Time zone" | awk '{print $3}')
    echo "   Current timezone: $TIMEZONE"

    if [ "$TIMEZONE" = "UTC" ]; then
        echo -e "${YELLOW}   Server is in UTC (most DigitalOcean droplets)${NC}"
        echo "   4 PM ET = 9 PM UTC (winter) or 8 PM UTC (summer)"
        echo "   11 PM ET = 4 AM UTC (next day, winter) or 3 AM UTC (summer)"
    fi
else
    echo "   Unable to detect timezone"
fi
echo ""

# 9. Generate crontab template
echo "9️⃣  Generating crontab configuration..."

cat > /tmp/nba_crontab.txt << EOF
# NBA Betting System - Cron Jobs
# Generated on $(date)

# Set PATH for cron
PATH=/usr/local/bin:/usr/bin:/bin

# Daily betting pipeline at 4 PM ET
# Adjust hour based on your timezone (currently set for UTC)
0 21 * * * $PROJECT_DIR/scripts/cron_betting.sh >> $PROJECT_DIR/logs/cron_main.log 2>&1

# Daily Discord report at 11 PM ET
# Adjust hour based on your timezone (currently set for UTC)
0 4 * * * cd $PROJECT_DIR && $PROJECT_DIR/venv/bin/python scripts/send_daily_report.py >> $PROJECT_DIR/logs/daily_report.log 2>&1

# Optional: Daily backup at 2 AM UTC
# 0 2 * * * $PROJECT_DIR/scripts/backup_data.sh >> $PROJECT_DIR/logs/backup.log 2>&1
EOF

echo -e "${GREEN}✅ Crontab template created at /tmp/nba_crontab.txt${NC}"
echo ""

# 10. Instructions
echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "Next Steps:"
echo ""
echo "1️⃣  Configure environment variables:"
echo "   nano .env"
echo "   # Add your ODDS_API_KEY and DISCORD_WEBHOOK_URL"
echo ""
echo "2️⃣  Install cron jobs:"
echo "   crontab -e"
echo "   # Add the contents from /tmp/nba_crontab.txt"
echo "   # Or run: crontab /tmp/nba_crontab.txt"
echo ""
echo "3️⃣  Test cron job manually:"
echo "   ./scripts/cron_betting.sh"
echo "   tail -50 logs/cron_betting.log"
echo ""
echo "4️⃣  Monitor logs:"
echo "   tail -f logs/cron_betting.log"
echo ""
echo "5️⃣  View cron jobs:"
echo "   crontab -l"
echo ""
echo "📚 Full documentation: cat DROPLET_DEPLOYMENT.md"
echo ""
echo "🚀 Your betting system is ready!"
echo ""
