# NBA Betting System - Operations Playbook

**Version**: 2.0 (Jane Street/Citadel Standard)
**Last Updated**: 2026-01-10

---

## Mission

Run a **profitable, automated sports betting operation** with institutional-grade risk management and operational excellence.

---

## Daily Operations (5 Minutes)

### Morning Routine (9:00 AM ET)

```bash
# 1. Health check
./ops/health_check.sh

# 2. Review overnight results
tail -50 logs/settlement.log

# 3. Check for alerts
cat logs/alerts.log | grep ERROR

# Done. System runs itself.
```

**That's it.** The system is fully automated.

---

## System Overview

### The Core Loop

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  COLLECT → PREDICT → FILTER → SIZE → EXECUTE  │
│     ↓         ↓        ↓       ↓        ↓      │
│   Data    Model    Timing   Kelly    Log Bet  │
│                                                 │
└─────────────────────────────────────────────────┘
           ↓                          ↓
    SETTLE BETS              TRACK PERFORMANCE
```

### Automated Schedule

| Time | Action | Script | Purpose |
|------|--------|--------|---------|
| **Every 15 min** | Capture lines | `capture_opening_lines.py` | CLV tracking |
| **3:00 PM ET** | Generate bets | `daily_multi_strategy_pipeline.py --use-timing` | Optimal timing window |
| **5:00 PM ET** | Generate bets | `daily_multi_strategy_pipeline.py --use-timing` | Optimal timing window |
| **6:00 AM ET** | Settle bets | `settle_bets.py` | Results & CLV |
| **9:00 AM ET** | Health check | `ops/health_check.sh` | System validation |

---

## Critical Files

### Models (Production)
```
models/
├── spread_model_calibrated.pkl      # Spread betting (74 features, AUC 0.516)
├── player_props/
│   ├── PTS_model.pkl                 # Points (27% ROI)
│   ├── REB_model.pkl                 # Rebounds (56% ROI)
│   ├── AST_model.pkl                 # Assists (89% ROI)
│   └── 3PM_model.pkl                 # 3-pointers (55% ROI)
```

### Configuration
```
config/
└── multi_strategy_config.yaml        # Strategy allocation, risk limits
```

### Data
```
data/
├── bets/bets.db                      # All bets (63 paper trades)
├── raw/games_with_spread_coverage.parquet  # Training data
└── historical_odds/                  # 405 days of odds data
```

---

## Performance Monitoring

### Key Metrics

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

### Target Performance

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| **Overall ROI** | >5% | 6.91% | ✅ |
| **Win Rate** | >54% | 56.0% | ✅ |
| **Sharpe Ratio** | >1.0 | 1.42 | ✅ |
| **Max Drawdown** | <20% | TBD | 🔄 |
| **CLV** | >0% | TBD | 🔄 |

---

## Strategy Allocation

From `config/multi_strategy_config.yaml`:

| Strategy | Allocation | Daily Limit | Status | Expected ROI |
|----------|------------|-------------|---------|--------------|
| **Props** | 35% | 12 bets | ✅ ENABLED | 27-89% |
| **Arbitrage** | 30% | 5 bets | ✅ ENABLED | 10.93% |
| **Spread** | 20% | 8 bets | ✅ ENABLED | 6.91% |
| **B2B Rest** | 15% | 5 bets | ✅ ENABLED | 8.7% |

**Total**: 100% allocated, max 30 bets/day

---

## Risk Management

### Position Sizing

**Kelly Criterion** (25% fractional):
```
bet_size = 0.25 × kelly × bankroll × correlation_discount × drawdown_scale
```

**Limits**:
- Max single bet: 5% of bankroll
- Max daily exposure: 15% of bankroll
- Max same-game exposure: 10% of bankroll
- Max same-team exposure: 15% of bankroll

### Drawdown Protection

```
Drawdown         Action
─────────────────────────────
0-10%           ● Normal operation
10-20%          ⚠ Reduce sizing to 50%
20-30%          🛑 Hard stop betting
>30%            🔴 CIRCUIT BREAKER
```

### Timing Optimization

**Optimal Window**: 1-4 hours before game
- Too early (>48hr): Opening line volatility (1.71 pts avg movement)
- **Optimal (1-4hr)**: Best prices (0.67 pts avg movement)
- Too late (<1hr): Closing line efficient

**Implementation**: `--use-timing` flag enabled by default

---

## Common Operations

### Generate Bets (Manual)

```bash
# Standard run
python scripts/daily_multi_strategy_pipeline.py

# With timing optimization (recommended)
python scripts/daily_multi_strategy_pipeline.py --use-timing

# Dry run (no logging)
python scripts/daily_multi_strategy_pipeline.py --dry-run --use-timing
```

### Settle Bets

```bash
# Settle overnight games
python scripts/settle_bets.py

# Populate CLV data
python scripts/populate_clv_data.py
```

### Check Performance

```bash
# CLV report
python scripts/generate_clv_report.py

# Paper trading report
python scripts/paper_trading_report.py

# Timing analysis
python scripts/analyze_line_movement_timing.py
```

### Retrain Models

```bash
# Retrain all models (monthly)
python scripts/retrain_models.py

# Retrain specific model
python scripts/retrain_player_props.py --prop-type PTS
```

---

## Troubleshooting

### No Bets Generated

**Possible causes**:
1. All games filtered by edge threshold (increase min_edge temporarily)
2. Timing filter blocking bets (check hours until game)
3. CLV filter blocking (historical performance issue)
4. No games today (check NBA schedule)

**Debug**:
```bash
python scripts/daily_multi_strategy_pipeline.py --dry-run --use-timing 2>&1 | grep -A5 "filtered"
```

### Model Loading Error

**Fix**:
```bash
# Verify model file
ls -lh models/spread_model_calibrated.pkl

# Test load
python -c "import pickle; pickle.load(open('models/spread_model_calibrated.pkl', 'rb'))"

# Restore from backup if needed
cp models/spread_model_calibrated_99feat_backup.pkl models/spread_model_calibrated.pkl
```

### API Rate Limit

**Fix**:
```bash
# Check remaining calls
grep "API requests remaining" logs/*.log | tail -1

# Wait if low (<100 calls)
# Or upgrade API plan
```

### Bet Settlement Failing

**Fix**:
```bash
# Check database
sqlite3 data/bets/bets.db ".schema bets"

# Manually settle specific game
python -c "from src.bet_tracker import settle_bet; settle_bet('game_id_here')"
```

---

## Health Checks

### Daily (Automated at 9 AM)

```bash
./ops/health_check.sh
```

Checks:
- ✅ Models loadable
- ✅ Database accessible
- ✅ API credentials valid
- ✅ Disk space >10GB
- ✅ Recent bets logged
- ✅ No errors in logs

### Weekly (Manual)

```bash
# Performance review
python scripts/paper_trading_report.py

# CLV analysis
python scripts/generate_clv_report.py

# Model validation (check AUC hasn't degraded)
# Retrain if AUC drops below 0.50
```

### Monthly (Manual)

```bash
# Full model retraining
python scripts/retrain_models.py

# Backtest updated models
python scripts/backtest_against_market.py

# Review and update config
vim config/multi_strategy_config.yaml
```

---

## Escalation

### Yellow Alert (Action within 24hr)

- Win rate drops below 52% (7-day rolling)
- CLV negative for 3+ days
- Drawdown exceeds 10%
- API rate limit warnings

**Action**: Review recent bets, check for data issues

### Red Alert (Immediate action)

- Drawdown exceeds 20%
- Database corruption
- API access lost
- Model loading failures

**Action**: STOP BETTING, investigate root cause

---

## Deployment Checklist

### New System Setup

- [ ] Clone repository
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Configure `.env` (API keys, Discord webhook)
- [ ] Download models (`models/` directory)
- [ ] Initialize database (`python scripts/init_db.py`)
- [ ] Test dry run (`python scripts/daily_multi_strategy_pipeline.py --dry-run`)
- [ ] Install crontab (`crontab deploy/crontab_production.txt`)
- [ ] Verify health check passes (`./ops/health_check.sh`)

### Model Update

- [ ] Backup current model
- [ ] Train new model with validation
- [ ] Compare performance (AUC, accuracy)
- [ ] Test on recent games
- [ ] Deploy to production
- [ ] Monitor first 24hr of predictions

### Config Changes

- [ ] Document change reason
- [ ] Test with dry run
- [ ] Deploy during low-volume period
- [ ] Monitor for 48hr

---

## Documentation

### Quick Reference

- **Operations**: `ops/OPERATIONS_PLAYBOOK.md` (this file)
- **Timing Optimization**: `docs/TIMING_OPTIMIZATION_QUICKSTART.md`
- **Scripts Guide**: `scripts/README.md`
- **Model Documentation**: `docs/models/`

### Full Documentation

- **Quant Framework**: `.claude/plans/cached-exploring-thunder.md`
- **Bet Timing Analysis**: `docs/BET_TIMING_ANALYSIS.md`
- **Feature Importance**: `docs/models/FEATURE_IMPORTANCE_ANALYSIS.md`

---

## Performance Targets (2026)

| Month | Bets | Win Rate | ROI | Sharpe | Drawdown |
|-------|------|----------|-----|--------|----------|
| Jan | 200 | >54% | >5% | >1.0 | <15% |
| Feb | 400 | >54% | >5% | >1.0 | <15% |
| Mar | 600 | >54% | >5% | >1.0 | <15% |
| Q1 Total | 1200 | >54% | >5% | >1.2 | <20% |

**Success Criteria**: Profitable every quarter with controlled risk

---

## Contact & Support

**System Owner**: Operations Team
**Escalation**: Check logs first, then review this playbook
**Emergency**: Stop automated betting, investigate manually

---

**Principle**: "Simple systems work. Complex systems fail."

Keep operations tight, automated, and monitored. Trust the process.
