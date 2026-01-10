#!/bin/bash
#
# NBA Betting System - Health Check
# Runs comprehensive system validation
#
# Usage: ./ops/health_check.sh
#
# Exit codes:
#   0 = All checks passed
#   1 = Critical failure
#   2 = Warning (non-critical)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track status
WARNINGS=0
ERRORS=0

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}NBA BETTING SYSTEM - HEALTH CHECK${NC}"
echo -e "${BLUE}$(date)${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# ============================================================================
# 1. Python Environment
# ============================================================================
echo -e "${BLUE}[1/10]${NC} Checking Python environment..."

if [ ! -d ".venv" ]; then
    echo -e "${RED}✗ Virtual environment not found${NC}"
    ((ERRORS++))
else
    PYTHON=".venv/bin/python"
    if [ ! -f "$PYTHON" ]; then
        echo -e "${RED}✗ Python executable not found in .venv${NC}"
        ((ERRORS++))
    else
        PYTHON_VERSION=$($PYTHON --version 2>&1)
        echo -e "${GREEN}✓${NC} $PYTHON_VERSION"
    fi
fi

# ============================================================================
# 2. Critical Dependencies
# ============================================================================
echo -e "${BLUE}[2/10]${NC} Checking critical dependencies..."

MISSING_DEPS=0
for pkg in pandas numpy scikit-learn xgboost loguru; do
    if ! $PYTHON -c "import $pkg" 2>/dev/null; then
        echo -e "${RED}✗ Missing: $pkg${NC}"
        ((MISSING_DEPS++))
    fi
done

if [ $MISSING_DEPS -eq 0 ]; then
    echo -e "${GREEN}✓${NC} All critical dependencies installed"
else
    echo -e "${RED}✗ Missing $MISSING_DEPS critical dependencies${NC}"
    ((ERRORS++))
fi

# ============================================================================
# 3. Production Models
# ============================================================================
echo -e "${BLUE}[3/10]${NC} Checking production models..."

MODELS_OK=0

# Spread model
if [ -f "models/spread_model_calibrated.pkl" ]; then
    SIZE=$(ls -lh models/spread_model_calibrated.pkl | awk '{print $5}')
    # Test load
    if $PYTHON -c "import pickle; pickle.load(open('models/spread_model_calibrated.pkl', 'rb'))" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Spread model ($SIZE)"
        ((MODELS_OK++))
    else
        echo -e "${RED}✗ Spread model corrupted${NC}"
        ((ERRORS++))
    fi
else
    echo -e "${RED}✗ Spread model not found${NC}"
    ((ERRORS++))
fi

# Player props models
for prop in PTS REB AST 3PM; do
    MODEL_FILE="models/player_props/${prop}_model.pkl"
    if [ -f "$MODEL_FILE" ]; then
        ((MODELS_OK++))
    fi
done

if [ $MODELS_OK -ge 4 ]; then
    echo -e "${GREEN}✓${NC} Player props models (4/4 types found)"
else
    echo -e "${YELLOW}⚠${NC} Player props models ($MODELS_OK/4 types found)"
    ((WARNINGS++))
fi

# ============================================================================
# 4. Database Access
# ============================================================================
echo -e "${BLUE}[4/10]${NC} Checking database access..."

if [ -f "data/bets/bets.db" ]; then
    # Test database query
    BETS_COUNT=$(sqlite3 data/bets/bets.db "SELECT COUNT(*) FROM bets" 2>/dev/null || echo "ERROR")
    if [ "$BETS_COUNT" = "ERROR" ]; then
        echo -e "${RED}✗ Database corrupted or inaccessible${NC}"
        ((ERRORS++))
    else
        echo -e "${GREEN}✓${NC} Database accessible ($BETS_COUNT total bets)"
    fi
else
    echo -e "${RED}✗ Database file not found${NC}"
    ((ERRORS++))
fi

# ============================================================================
# 5. Configuration Files
# ============================================================================
echo -e "${BLUE}[5/10]${NC} Checking configuration..."

if [ -f "config/multi_strategy_config.yaml" ]; then
    echo -e "${GREEN}✓${NC} Strategy config found"
else
    echo -e "${RED}✗ Strategy config missing${NC}"
    ((ERRORS++))
fi

if [ -f ".env" ]; then
    # Check for critical env vars
    if grep -q "ODDS_API_KEY" .env 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Environment variables configured"
    else
        echo -e "${YELLOW}⚠${NC} ODDS_API_KEY not found in .env"
        ((WARNINGS++))
    fi
else
    echo -e "${YELLOW}⚠${NC} .env file not found (API access may be limited)"
    ((WARNINGS++))
fi

# ============================================================================
# 6. Disk Space
# ============================================================================
echo -e "${BLUE}[6/10]${NC} Checking disk space..."

DISK_AVAIL=$(df -h . | awk 'NR==2 {print $4}' | sed 's/[A-Za-z]//g')
DISK_AVAIL_INT=$(echo $DISK_AVAIL | cut -d. -f1 | tr -d 'i')

if [ $DISK_AVAIL_INT -lt 1 ]; then
    echo -e "${RED}✗ Low disk space: ${DISK_AVAIL}GB${NC}"
    ((ERRORS++))
elif [ $DISK_AVAIL_INT -lt 10 ]; then
    echo -e "${YELLOW}⚠${NC} Disk space: ${DISK_AVAIL}GB (consider cleanup)"
    ((WARNINGS++))
else
    echo -e "${GREEN}✓${NC} Disk space: ${DISK_AVAIL}GB"
fi

# ============================================================================
# 7. Recent Betting Activity
# ============================================================================
echo -e "${BLUE}[7/10]${NC} Checking recent betting activity..."

BETS_TODAY=$(sqlite3 data/bets/bets.db "SELECT COUNT(*) FROM bets WHERE DATE(logged_at) = DATE('now')" 2>/dev/null || echo "0")
BETS_WEEK=$(sqlite3 data/bets/bets.db "SELECT COUNT(*) FROM bets WHERE logged_at >= datetime('now', '-7 days')" 2>/dev/null || echo "0")

if [ "$BETS_TODAY" -gt 0 ]; then
    echo -e "${GREEN}✓${NC} Bets today: $BETS_TODAY"
elif [ "$BETS_WEEK" -gt 0 ]; then
    echo -e "${YELLOW}⚠${NC} No bets today (last 7 days: $BETS_WEEK)"
    ((WARNINGS++))
else
    echo -e "${YELLOW}⚠${NC} No recent betting activity"
    ((WARNINGS++))
fi

# ============================================================================
# 8. Log Files
# ============================================================================
echo -e "${BLUE}[8/10]${NC} Checking log files..."

ERROR_COUNT=$(grep -r "ERROR" logs/*.log 2>/dev/null | wc -l || echo "0")
CRITICAL_COUNT=$(grep -r "CRITICAL" logs/*.log 2>/dev/null | wc -l || echo "0")

if [ $CRITICAL_COUNT -gt 0 ]; then
    echo -e "${RED}✗ Found $CRITICAL_COUNT CRITICAL errors in logs${NC}"
    ((ERRORS++))
elif [ $ERROR_COUNT -gt 10 ]; then
    echo -e "${YELLOW}⚠${NC} Found $ERROR_COUNT errors in logs (review recommended)"
    ((WARNINGS++))
else
    echo -e "${GREEN}✓${NC} Logs clean (errors: $ERROR_COUNT)"
fi

# ============================================================================
# 9. Performance Metrics
# ============================================================================
echo -e "${BLUE}[9/10]${NC} Checking performance metrics..."

SETTLED_BETS=$(sqlite3 data/bets/bets.db "SELECT COUNT(*) FROM bets WHERE outcome IS NOT NULL" 2>/dev/null || echo "0")

if [ $SETTLED_BETS -gt 0 ]; then
    WIN_RATE=$(sqlite3 data/bets/bets.db "SELECT PRINTF('%.1f', CAST(SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100) FROM bets WHERE outcome IS NOT NULL" 2>/dev/null || echo "0")
    TOTAL_PROFIT=$(sqlite3 data/bets/bets.db "SELECT PRINTF('%.2f', COALESCE(SUM(profit), 0)) FROM bets WHERE outcome IS NOT NULL" 2>/dev/null || echo "0")

    echo -e "${GREEN}✓${NC} Win rate: ${WIN_RATE}% | P&L: \$$TOTAL_PROFIT | Settled: $SETTLED_BETS"

    # Check for concerning trends
    WIN_RATE_INT=$(echo $WIN_RATE | cut -d. -f1)
    if [ $WIN_RATE_INT -lt 52 ]; then
        echo -e "${YELLOW}⚠${NC} Win rate below 52% (review strategy)"
        ((WARNINGS++))
    fi
else
    echo -e "${YELLOW}⚠${NC} No settled bets yet"
fi

# ============================================================================
# 10. API Access
# ============================================================================
echo -e "${BLUE}[10/10]${NC} Checking API access..."

# Test API by checking last API call
LAST_API_CALL=$(grep "API requests remaining" logs/*.log 2>/dev/null | tail -1)

if [ ! -z "$LAST_API_CALL" ]; then
    API_REMAINING=$(echo $LAST_API_CALL | grep -o '[0-9]\+' | tail -1)
    if [ $API_REMAINING -lt 100 ]; then
        echo -e "${YELLOW}⚠${NC} API calls remaining: $API_REMAINING (rate limit approaching)"
        ((WARNINGS++))
    else
        echo -e "${GREEN}✓${NC} API calls remaining: $API_REMAINING"
    fi
else
    echo -e "${YELLOW}⚠${NC} No recent API activity detected"
    ((WARNINGS++))
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}HEALTH CHECK SUMMARY${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ ALL SYSTEMS OPERATIONAL${NC}"
    echo ""
    echo "System is healthy and ready for production betting."
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ WARNINGS DETECTED: $WARNINGS${NC}"
    echo ""
    echo "System is operational but has $WARNINGS non-critical issues."
    echo "Review warnings above and address when convenient."
    exit 2
else
    echo -e "${RED}✗ CRITICAL ERRORS: $ERRORS${NC}"
    echo -e "${YELLOW}⚠ WARNINGS: $WARNINGS${NC}"
    echo ""
    echo "System has critical issues that must be addressed before betting."
    echo "Review errors above and fix immediately."
    exit 1
fi
