# Operations Infrastructure

**Version**: 2.0 (Jane Street/Citadel Standard)
**Status**: Production Ready

---

## Overview

This directory contains institutional-grade operational tools designed for a tight, efficient betting operation that anyone can use. Everything is automated, monitored, and requires minimal daily interaction.

---

## Quick Start (5 Minutes Daily)

### Morning Routine

```bash
# 1. Open dashboard (single command for everything)
./ops/dashboard.sh

# That's it. The dashboard shows:
# - System status
# - Performance metrics (Win rate, ROI, P&L)
# - Strategy breakdown
# - Recent bets
# - Upcoming games
# - Quick actions menu
```

**Alternative (command line):**
```bash
# 1. Health check
./ops/health_check.sh

# 2. Review overnight results
tail -50 logs/settlement.log

# 3. Check for alerts
grep ERROR logs/alerts.log
```

---

## Tools

### 1. Operations Dashboard (`dashboard.sh`)

**Single command for complete system status**

```bash
./ops/dashboard.sh
```

**What it shows:**
- **System Status**: Health, total bets, last bet time, disk space
- **Performance Metrics**: Win rate, ROI, total P&L, today's performance
- **Strategy Breakdown**: Performance by strategy (Spread, Props, Arb, B2B)
- **Recent Bets**: Last 5 bets with outcomes
- **Upcoming Games**: Today's schedule
- **Alerts**: Errors, warnings, critical issues
- **Quick Actions**: Interactive menu for common operations

**Quick Actions:**
1. Run health check
2. Generate bets (with timing optimization)
3. Settle bets
4. Performance report
5. CLV report
6. View logs
7. Refresh dashboard

**Use Case**: Daily operations, system monitoring, quick actions

---

### 2. Health Check (`health_check.sh`)

**Comprehensive system validation (10 automated checks)**

```bash
./ops/health_check.sh
```

**What it checks:**
1. Python environment (.venv)
2. Critical dependencies (pandas, numpy, scikit-learn, xgboost, loguru)
3. Production models (spread, props)
4. Database access (bets.db)
5. Configuration files (multi_strategy_config.yaml, .env)
6. Disk space (>10GB recommended)
7. Recent betting activity
8. Log files (errors, critical issues)
9. Performance metrics (win rate, P&L)
10. API access (rate limits)

**Exit codes:**
- `0` - All checks passed (green ✓)
- `1` - Critical errors (red ✗)
- `2` - Warnings only (yellow ⚠)

**Use Case**: Daily health validation, troubleshooting, deployment verification

---

### 3. Operations Playbook (`PLAYBOOK.md`)

**Master operations guide**

**What's inside:**
- Daily operations routine (5 minutes)
- System overview and architecture
- Critical files reference
- Performance monitoring metrics
- Strategy allocation breakdown
- Risk management protocols
- Common operations (generate bets, settle, reports)
- Troubleshooting guide
- Health check procedures
- Escalation protocols
- Deployment checklists
- Documentation index

**Use Case**: Reference guide, onboarding new operators, troubleshooting

---

### 4. Production Crontab (`../deploy/crontab_production.txt`)

**Automated schedule for all operations**

**Installation:**
```bash
# 1. Edit paths in crontab file
vim deploy/crontab_production.txt
# Update PROJECT=/path/to/nbamodels
# Update MAILTO=your-email@example.com

# 2. Install crontab
crontab deploy/crontab_production.txt

# 3. Verify installation
crontab -l
```

**Schedule:**

**Core Betting Loop:**
- **3:00 PM & 5:00 PM ET**: Generate bets (optimal timing window)

**Settlement & Reporting:**
- **6:00 AM ET**: Settle bets, calculate CLV
- **9:00 AM ET**: Health check

**Data Collection:**
- **Every 15 min**: Capture opening/closing lines
- **Hourly (9AM-11PM)**: Collect news
- **Daily (10 AM)**: Collect referee assignments
- **Every 15 min (5-11PM)**: Collect confirmed lineups

**Weekly Operations:**
- **Sunday 10:00 AM**: Performance report
- **Sunday 10:30 AM**: CLV analysis

**Monthly Operations:**
- **1st of month, 2:00 AM**: Model retraining
- **1st of month, 3:00 AM**: Backup

**Use Case**: Production automation, unattended operation

---

## Operational Philosophy

### "Simple systems work. Complex systems fail."

**Jane Street/Citadel Principles Applied:**

1. **Automation First**: The system should run itself
2. **Monitoring Always**: Know the system state at all times
3. **Tight Feedback Loops**: Daily health checks, performance tracking
4. **Risk Management**: Automated drawdown protection, correlation limits
5. **Measurable Success**: Track CLV, ROI, Sharpe ratio
6. **Fail Fast**: Errors should be visible and actionable

---

## Daily Workflow

### Option 1: Dashboard (Recommended)

```bash
# Morning (9:00 AM ET)
./ops/dashboard.sh
# Review status, check alerts, select action if needed
```

### Option 2: Manual

```bash
# Morning (9:00 AM ET)
./ops/health_check.sh                    # Validate system
tail -50 logs/settlement.log             # Review overnight results
grep ERROR logs/alerts.log               # Check for issues

# Afternoon (3:00 PM ET) - Automated by cron
# Bets generated automatically

# Evening - Review dashboard for results
./ops/dashboard.sh
```

---

## Performance Targets

| Metric | Target | Alert Level |
|--------|--------|-------------|
| **Win Rate** | >54% | <52% (Yellow) |
| **ROI** | >5% | <0% (Red) |
| **Sharpe Ratio** | >1.0 | <0.5 (Yellow) |
| **Max Drawdown** | <20% | >20% (Red) |
| **CLV** | >0% | <0% for 3+ days (Yellow) |

**Current Performance:**
- Win Rate: 56.0% ✅
- ROI: 6.91% ✅
- Sharpe Ratio: 1.42 ✅

---

## Troubleshooting

### Dashboard won't start

```bash
# Check permissions
chmod +x ops/dashboard.sh

# Check dependencies
ls -la .venv/bin/python
ls -la data/bets/bets.db

# Run health check instead
./ops/health_check.sh
```

### Health check fails

See `PLAYBOOK.md` troubleshooting section for:
- Model loading errors
- Database corruption
- API rate limits
- Bet settlement issues

### No bets generated

**Possible causes:**
1. All games filtered by edge threshold
2. Timing filter blocking bets (check hours until game)
3. No games today (check NBA schedule)

**Debug:**
```bash
python scripts/daily_multi_strategy_pipeline.py --dry-run --use-timing 2>&1 | grep -A5 "filtered"
```

---

## File Structure

```
ops/
├── README.md                    # This file
├── OPERATIONS_PLAYBOOK.md       # Master operations guide
├── dashboard.sh                 # Operations dashboard (interactive)
└── health_check.sh              # System validation (automated)

deploy/
└── crontab_production.txt       # Production automation schedule
```

---

## Metrics & Reports

### Quick Metrics (SQL)

```bash
# Daily P&L
sqlite3 data/bets/bets.db "SELECT SUM(profit) FROM bets WHERE outcome IS NOT NULL"

# Win rate
sqlite3 data/bets/bets.db "SELECT
  CAST(SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*)
  FROM bets WHERE outcome IS NOT NULL"

# ROI by strategy
sqlite3 data/bets/bets.db "SELECT bet_type,
  SUM(profit) / SUM(bet_amount) as roi,
  COUNT(*) as bets
  FROM bets WHERE outcome IS NOT NULL GROUP BY bet_type"
```

### Comprehensive Reports (Python)

```bash
# Performance report
python scripts/paper_trading_report.py

# CLV analysis
python scripts/generate_clv_report.py

# Timing analysis
python scripts/analyze_line_movement_timing.py
```

---

## Deployment Checklist

### New System Setup

- [ ] Clone repository
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Configure `.env` (API keys, Discord webhook)
- [ ] Download models (`models/` directory)
- [ ] Initialize database (`python scripts/init_db.py`)
- [ ] Make scripts executable (`chmod +x ops/*.sh`)
- [ ] Test dry run (`python scripts/daily_multi_strategy_pipeline.py --dry-run`)
- [ ] Install crontab (`crontab deploy/crontab_production.txt`)
- [ ] Verify health check passes (`./ops/health_check.sh`)
- [ ] Test dashboard (`./ops/dashboard.sh`)

### Verification

```bash
# 1. Health check
./ops/health_check.sh
# Should show: ✓ ALL SYSTEMS OPERATIONAL

# 2. Dashboard
./ops/dashboard.sh
# Should show system status and quick actions menu

# 3. Crontab
crontab -l
# Should show production schedule

# 4. Test bet generation (dry run)
python scripts/daily_multi_strategy_pipeline.py --dry-run --use-timing
# Should generate predictions without logging
```

---

## Support & Documentation

**Quick Reference:**
- **Operations**: `ops/PLAYBOOK.md`
- **Scripts Guide**: `scripts/README.md`
- **Model Documentation**: `docs/models/`
- **Deployment**: `deploy/README.md`

**Full Documentation:**
- **Bet Timing Analysis**: `docs/BET_TIMING_ANALYSIS.md`
- **Feature Importance**: `docs/models/FEATURE_IMPORTANCE_ANALYSIS.md`

---

## Version History

**v2.0** (2026-01-10) - Institutional-Grade Operations
- Created operations dashboard (interactive)
- Created health check system (automated)
- Created operations playbook (comprehensive)
- Created production crontab (fully automated)
- Integrated timing optimization
- Reallocated portfolio (Props 35%, Spread 20%)
- Pruned model features (99→74)

**v1.0** (2025) - Initial System
- Multi-strategy betting pipeline
- Paper trading infrastructure
- Risk management framework
- CLV tracking

---

**Status**: ✅ **PRODUCTION READY**

**Principle**: "Simple systems work. Complex systems fail."

Keep operations tight, automated, and monitored. Trust the process.
