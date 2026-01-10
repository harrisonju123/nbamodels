#!/bin/bash
#
# NBA Betting System - Operations Dashboard
# Single command for complete system status and operations
#
# Usage: ./ops/dashboard.sh
#

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'

PYTHON=".venv/bin/python"

# Clear screen for clean display
clear

echo -e "${BLUE}${BOLD}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}${BOLD}   NBA BETTING SYSTEM - OPERATIONS DASHBOARD${NC}"
echo -e "${BLUE}${BOLD}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}$(date '+%A, %B %d, %Y - %I:%M %p %Z')${NC}"
echo ""

# ============================================================================
# SYSTEM STATUS
# ============================================================================

echo -e "${BOLD}SYSTEM STATUS${NC}"
echo "─────────────────────────────────────────────────────────────────"

# Quick health check
HEALTH_STATUS="UNKNOWN"
if [ -f "models/spread_model_calibrated.pkl" ] && [ -f "data/bets/bets.db" ]; then
    HEALTH_STATUS="${GREEN}OPERATIONAL${NC}"
else
    HEALTH_STATUS="${RED}ERROR${NC}"
fi

echo -e "Status:           $HEALTH_STATUS"

# Database stats
if [ -f "data/bets/bets.db" ]; then
    TOTAL_BETS=$(sqlite3 data/bets/bets.db "SELECT COUNT(*) FROM bets" 2>/dev/null || echo "0")
    SETTLED_BETS=$(sqlite3 data/bets/bets.db "SELECT COUNT(*) FROM bets WHERE outcome IS NOT NULL" 2>/dev/null || echo "0")
    echo -e "Total Bets:       ${CYAN}$TOTAL_BETS${NC} ($SETTLED_BETS settled)"
else
    echo -e "Total Bets:       ${RED}Database not found${NC}"
fi

# Last bet time
if [ -f "data/bets/bets.db" ]; then
    LAST_BET=$(sqlite3 data/bets/bets.db "SELECT datetime(logged_at, 'localtime') FROM bets ORDER BY logged_at DESC LIMIT 1" 2>/dev/null || echo "Never")
    echo -e "Last Bet:         ${CYAN}$LAST_BET${NC}"
fi

# Disk space
DISK_AVAIL=$(df -h . | awk 'NR==2 {print $4}')
if [ $(echo $DISK_AVAIL | sed 's/[A-Za-z]//g' | cut -d. -f1) -lt 10 ]; then
    echo -e "Disk Space:       ${YELLOW}$DISK_AVAIL${NC} (Low)"
else
    echo -e "Disk Space:       ${GREEN}$DISK_AVAIL${NC}"
fi

echo ""

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

echo -e "${BOLD}PERFORMANCE METRICS${NC}"
echo "─────────────────────────────────────────────────────────────────"

if [ -f "data/bets/bets.db" ] && [ $SETTLED_BETS -gt 0 ]; then
    # Overall performance
    WIN_RATE=$(sqlite3 data/bets/bets.db "SELECT PRINTF('%.1f', CAST(SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100) FROM bets WHERE outcome IS NOT NULL" 2>/dev/null || echo "0.0")

    TOTAL_PROFIT=$(sqlite3 data/bets/bets.db "SELECT PRINTF('%.2f', COALESCE(SUM(profit), 0)) FROM bets WHERE outcome IS NOT NULL" 2>/dev/null || echo "0.00")

    TOTAL_WAGERED=$(sqlite3 data/bets/bets.db "SELECT PRINTF('%.2f', SUM(bet_amount)) FROM bets WHERE outcome IS NOT NULL" 2>/dev/null || echo "1.00")

    ROI=$(echo "scale=2; ($TOTAL_PROFIT / $TOTAL_WAGERED) * 100" | bc 2>/dev/null || echo "0.00")

    # Color code based on performance
    if (( $(echo "$WIN_RATE >= 54" | bc -l) )); then
        WIN_COLOR=$GREEN
    elif (( $(echo "$WIN_RATE >= 52" | bc -l) )); then
        WIN_COLOR=$YELLOW
    else
        WIN_COLOR=$RED
    fi

    if (( $(echo "$ROI >= 5" | bc -l) )); then
        ROI_COLOR=$GREEN
    elif (( $(echo "$ROI >= 0" | bc -l) )); then
        ROI_COLOR=$YELLOW
    else
        ROI_COLOR=$RED
    fi

    if (( $(echo "$TOTAL_PROFIT >= 0" | bc -l) )); then
        PROFIT_COLOR=$GREEN
    else
        PROFIT_COLOR=$RED
    fi

    echo -e "Win Rate:         ${WIN_COLOR}${WIN_RATE}%${NC} (Target: >54%)"
    echo -e "ROI:              ${ROI_COLOR}${ROI}%${NC} (Target: >5%)"
    echo -e "Total P&L:        ${PROFIT_COLOR}\$$TOTAL_PROFIT${NC}"

    # Today's performance
    BETS_TODAY=$(sqlite3 data/bets/bets.db "SELECT COUNT(*) FROM bets WHERE DATE(logged_at) = DATE('now')" 2>/dev/null || echo "0")
    SETTLED_TODAY=$(sqlite3 data/bets/bets.db "SELECT COUNT(*) FROM bets WHERE outcome IS NOT NULL AND DATE(logged_at) = DATE('now')" 2>/dev/null || echo "0")

    if [ $SETTLED_TODAY -gt 0 ]; then
        PROFIT_TODAY=$(sqlite3 data/bets/bets.db "SELECT PRINTF('%.2f', COALESCE(SUM(profit), 0)) FROM bets WHERE outcome IS NOT NULL AND DATE(logged_at) = DATE('now')" 2>/dev/null || echo "0.00")

        if (( $(echo "$PROFIT_TODAY >= 0" | bc -l) )); then
            TODAY_COLOR=$GREEN
        else
            TODAY_COLOR=$RED
        fi

        echo -e "Today:            ${CYAN}$BETS_TODAY${NC} bets (${SETTLED_TODAY} settled, ${TODAY_COLOR}\$$PROFIT_TODAY${NC})"
    else
        echo -e "Today:            ${CYAN}$BETS_TODAY${NC} bets (${SETTLED_TODAY} settled)"
    fi

else
    echo -e "${YELLOW}No settled bets yet${NC}"
fi

echo ""

# ============================================================================
# STRATEGY BREAKDOWN
# ============================================================================

echo -e "${BOLD}STRATEGY PERFORMANCE${NC}"
echo "─────────────────────────────────────────────────────────────────"

if [ -f "data/bets/bets.db" ] && [ $SETTLED_BETS -gt 0 ]; then
    # Get strategy breakdown
    echo -e "${BOLD}Strategy        Bets   Wins    ROI      P&L${NC}"

    sqlite3 data/bets/bets.db "
    SELECT
        CASE
            WHEN bet_type = 'spread' THEN 'Spread         '
            WHEN bet_type = 'player_prop' THEN 'Player Props   '
            WHEN bet_type = 'arbitrage' THEN 'Arbitrage      '
            WHEN bet_type = 'b2b_rest' THEN 'B2B Rest       '
            ELSE bet_type || '          '
        END as strategy,
        PRINTF('%4d', COUNT(*)) as bets,
        PRINTF('%4d', SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END)) as wins,
        PRINTF('%6.1f%%', COALESCE(SUM(profit) / SUM(bet_amount) * 100, 0)) as roi,
        PRINTF('%8.2f', COALESCE(SUM(profit), 0)) as profit
    FROM bets
    WHERE outcome IS NOT NULL
    GROUP BY bet_type
    ORDER BY COALESCE(SUM(profit), 0) DESC
    " 2>/dev/null | while IFS='|' read -r strategy bets wins roi profit; do
        echo -e "$strategy $bets   $wins   $roi  \$$profit"
    done
else
    echo -e "${YELLOW}No settled bets yet${NC}"
fi

echo ""

# ============================================================================
# RECENT BETS
# ============================================================================

echo -e "${BOLD}RECENT BETS (Last 5)${NC}"
echo "─────────────────────────────────────────────────────────────────"

if [ -f "data/bets/bets.db" ] && [ $TOTAL_BETS -gt 0 ]; then
    sqlite3 data/bets/bets.db "
    SELECT
        datetime(logged_at, 'localtime'),
        bet_type,
        game_id,
        bet_side,
        bet_amount,
        odds,
        COALESCE(outcome, 'pending'),
        COALESCE(profit, 0.0)
    FROM bets
    ORDER BY logged_at DESC
    LIMIT 5
    " 2>/dev/null | while IFS='|' read -r time type game side amount odds outcome profit; do
        # Format outcome with color
        if [ "$outcome" = "win" ]; then
            OUTCOME_STR="${GREEN}WIN${NC} (+\$${profit})"
        elif [ "$outcome" = "loss" ]; then
            OUTCOME_STR="${RED}LOSS${NC} (\$${profit})"
        elif [ "$outcome" = "push" ]; then
            OUTCOME_STR="${YELLOW}PUSH${NC} (\$0.00)"
        else
            OUTCOME_STR="${CYAN}PENDING${NC}"
        fi

        echo -e "${CYAN}${time}${NC}"
        echo -e "  $type: $game ($side) | \$${amount} @ ${odds} | $OUTCOME_STR"
    done
else
    echo -e "${YELLOW}No bets logged yet${NC}"
fi

echo ""

# ============================================================================
# UPCOMING GAMES
# ============================================================================

echo -e "${BOLD}UPCOMING GAMES (Today)${NC}"
echo "─────────────────────────────────────────────────────────────────"

# Try to get today's games from odds API or schedule
TODAY_GAMES=$($PYTHON -c "
import sys
sys.path.insert(0, '.')
from datetime import datetime, timezone
from src.data.odds_api import OddsAPIClient
from loguru import logger
logger.remove()

try:
    client = OddsAPIClient()
    games = client.get_upcoming_games()
    today = datetime.now(timezone.utc).date()

    today_games = [g for g in games if datetime.fromisoformat(g['commence_time'].replace('Z', '+00:00')).date() == today]

    if today_games:
        for i, game in enumerate(today_games[:5], 1):
            game_time = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00'))
            hours_until = (game_time - datetime.now(timezone.utc)).total_seconds() / 3600
            print(f'{game[\"away_team\"]:20s} @ {game[\"home_team\"]:20s} | {hours_until:5.1f}hr')
    else:
        print('No games today')
except Exception as e:
    print(f'Unable to fetch games')
" 2>/dev/null || echo "Unable to fetch games (check API key)")

echo -e "${CYAN}$TODAY_GAMES${NC}"

echo ""

# ============================================================================
# ALERTS & WARNINGS
# ============================================================================

echo -e "${BOLD}ALERTS & WARNINGS${NC}"
echo "─────────────────────────────────────────────────────────────────"

ALERTS=0

# Check for errors in logs
if [ -d "logs" ]; then
    CRITICAL_COUNT=$(grep -r "CRITICAL" logs/*.log 2>/dev/null | wc -l | tr -d ' ' || echo "0")
    ERROR_COUNT=$(grep -r "ERROR" logs/*.log 2>/dev/null | wc -l | tr -d ' ' || echo "0")

    if [ $CRITICAL_COUNT -gt 0 ]; then
        echo -e "${RED}✗ $CRITICAL_COUNT CRITICAL errors in logs${NC}"
        ((ALERTS++))
    fi

    if [ $ERROR_COUNT -gt 100 ]; then
        echo -e "${YELLOW}⚠ $ERROR_COUNT errors in logs (review recommended)${NC}"
        ((ALERTS++))
    fi
fi

# Check win rate if settled bets exist
if [ -f "data/bets/bets.db" ] && [ $SETTLED_BETS -gt 10 ]; then
    WIN_RATE_INT=$(echo $WIN_RATE | cut -d. -f1)
    if [ $WIN_RATE_INT -lt 52 ]; then
        echo -e "${RED}✗ Win rate below 52% (review strategy)${NC}"
        ((ALERTS++))
    fi
fi

# Check disk space
DISK_INT=$(echo $DISK_AVAIL | sed 's/[A-Za-z]//g' | cut -d. -f1)
if [ $DISK_INT -lt 5 ]; then
    echo -e "${RED}✗ Low disk space: $DISK_AVAIL${NC}"
    ((ALERTS++))
fi

if [ $ALERTS -eq 0 ]; then
    echo -e "${GREEN}✓ No alerts${NC}"
fi

echo ""

# ============================================================================
# QUICK ACTIONS
# ============================================================================

echo -e "${BLUE}${BOLD}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}QUICK ACTIONS${NC}"
echo ""
echo -e "  ${CYAN}1${NC} - Run health check (comprehensive system validation)"
echo -e "  ${CYAN}2${NC} - Generate bets (with timing optimization)"
echo -e "  ${CYAN}3${NC} - Settle bets (process game results)"
echo -e "  ${CYAN}4${NC} - Performance report (detailed analysis)"
echo -e "  ${CYAN}5${NC} - CLV report (closing line value analysis)"
echo -e "  ${CYAN}6${NC} - View logs (tail recent activity)"
echo -e "  ${CYAN}7${NC} - Refresh dashboard"
echo -e "  ${CYAN}q${NC} - Quit"
echo ""
echo -e -n "${BOLD}Select action [1-7, q]: ${NC}"

read -r action

case $action in
    1)
        echo ""
        echo -e "${BLUE}Running health check...${NC}"
        ./ops/health_check.sh
        ;;
    2)
        echo ""
        echo -e "${BLUE}Generating bets with timing optimization...${NC}"
        $PYTHON scripts/daily_multi_strategy_pipeline.py --use-timing
        ;;
    3)
        echo ""
        echo -e "${BLUE}Settling bets...${NC}"
        $PYTHON scripts/settle_bets.py
        ;;
    4)
        echo ""
        echo -e "${BLUE}Generating performance report...${NC}"
        $PYTHON scripts/paper_trading_report.py
        ;;
    5)
        echo ""
        echo -e "${BLUE}Generating CLV report...${NC}"
        $PYTHON scripts/generate_clv_report.py
        ;;
    6)
        echo ""
        echo -e "${BLUE}Recent logs:${NC}"
        echo ""
        echo -e "${BOLD}Pipeline log:${NC}"
        tail -20 logs/pipeline.log 2>/dev/null || echo "No pipeline logs"
        echo ""
        echo -e "${BOLD}Settlement log:${NC}"
        tail -20 logs/settlement.log 2>/dev/null || echo "No settlement logs"
        ;;
    7)
        exec "$0"
        ;;
    q|Q)
        echo ""
        echo -e "${GREEN}Dashboard closed.${NC}"
        exit 0
        ;;
    *)
        echo ""
        echo -e "${RED}Invalid selection.${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Action complete. Press Enter to refresh dashboard...${NC}"
read -r
exec "$0"
