# Multi-Strategy Betting System - Deployment Guide

**System**: 3-Strategy Paper Trading (Spread + Arbitrage + B2B Rest)
**Version**: 1.0
**Date**: January 5, 2026

---

## 📊 Current System Configuration

### Active Strategies

| Strategy | Allocation | Daily Limit | Performance | Status |
|----------|------------|-------------|-------------|--------|
| **Spread** | 65% | 10 bets | +23.6% ROI | ✅ Active |
| **Arbitrage** | 20% | 5 bets | +10.93% profit | ✅ Active |
| **B2B Rest** | 15% | 5 bets | +8.7% ROI, 57% WR | ✅ Active |

**Combined Expected ROI**: ~18-20% (weighted average)

### Risk Management Settings

```yaml
Bankroll: $1,000 (starting)
Kelly Fraction: 25% (conservative)
Max Daily Exposure: 15% of bankroll
Min Bet Size: $10
Paper Trading: ENABLED (safe for testing)
```

---

## 🚀 Quick Start Deployment

### Step 1: Verify Environment

```bash
# Check API key is set
cat .env | grep ODDS_API_KEY

# Check Discord webhook is configured
cat .env | grep DISCORD_WEBHOOK_URL

# Verify all strategies are enabled
cat config/multi_strategy_config.yaml | grep "enabled: true"
```

Expected output:
```
ODDS_API_KEY=16eca04028d0db0a86ce957e4e87f7a7
DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
    enabled: true   # ENABLED - backtest: 36 arbs, 10.93% avg profit (arbitrage)
    enabled: true   # ENABLED - backtest: +8.7% ROI, 57% win rate (b2b_rest)
    enabled: true   # ENABLED - validated +23.6% ROI with 5%+ edge (spread)
```

### Step 2: Test Pipeline Manually

```bash
# Run pipeline in dry-run mode (no bets logged)
python scripts/daily_multi_strategy_pipeline.py --dry-run

# Expected output:
# ✓ Enabled ArbitrageStrategy
# ✓ Enabled B2BRestStrategy
# ✓ Enabled SpreadStrategy (if games with features exist)
# ✓ Found X signals
```

### Step 3: Run Pipeline in Paper Trading Mode

```bash
# Run pipeline (logs bets to database but marks as paper trades)
python scripts/daily_multi_strategy_pipeline.py

# Check bet history
python -c "from src.bet_tracker import get_bet_history; print(get_bet_history().tail())"
```

### Step 4: Set Up Automated Execution

```bash
# Make scripts executable (already done)
chmod +x scripts/cron_betting.sh
chmod +x scripts/send_bet_notifications.py
chmod +x scripts/send_daily_report.py

# Edit crontab
crontab -e

# Add these lines:
# Daily betting pipeline (runs at 4 PM ET, 3 hours before games)
0 16 * * * /Users/harrisonju/PycharmProjects/nbamodels/scripts/cron_betting.sh

# Daily report (runs at 11 PM ET after games finish)
0 23 * * * cd /Users/harrisonju/PycharmProjects/nbamodels && python scripts/send_daily_report.py >> logs/daily_report.log 2>&1
```

---

## 📅 Daily Workflow

### Automated Schedule

| Time | Task | Script | Purpose |
|------|------|--------|---------|
| **4:00 PM ET** | Betting Pipeline | `cron_betting.sh` | Generate & log bets for today's games |
| **4:05 PM ET** | Bet Notifications | `send_bet_notifications.py` | Send bet alerts to Discord |
| **11:00 PM ET** | Daily Report | `send_daily_report.py` | Send performance summary to Discord |

### Manual Monitoring (Optional)

```bash
# View today's bets
python -c "from src.bet_tracker import get_bet_history; import pandas as pd; df = get_bet_history(); print(df[pd.to_datetime(df['placed_at']).dt.date == pd.Timestamp.now().date()])"

# Check dashboard
streamlit run dashboard/analytics_dashboard.py

# View logs
tail -f logs/cron_betting.log
```

---

## 🔍 Monitoring & Validation

### Key Metrics to Track (First 2 Weeks)

**Daily Checks**:
- [ ] Pipeline runs without errors
- [ ] Bets are being logged to database
- [ ] Discord notifications are received
- [ ] Strategy diversity (all 3 strategies finding bets)

**Weekly Analysis**:
- [ ] Win rate by strategy (compare to backtest)
- [ ] ROI trending positive
- [ ] Edge calibration (actual vs predicted)
- [ ] No suspicious patterns (e.g., all bets on one team)

### Access Analytics

**Dashboard** (Visual):
```bash
streamlit run dashboard/analytics_dashboard.py
# Opens browser at http://localhost:8501
# - Strategy Performance tab shows breakdown
# - Performance tab shows ROI trends
```

**Command Line** (Quick Stats):
```bash
# Overall performance
python -c "from src.bet_tracker import get_performance_summary; import json; print(json.dumps(get_performance_summary(), indent=2))"

# Performance by strategy
python -c "from src.bet_tracker import get_performance_by_type; print(get_performance_by_type())"
```

---

## 🎯 Success Criteria (2-Week Paper Trading)

Before going live with real money, validate:

### Minimum Requirements

| Metric | Target | Why It Matters |
|--------|--------|----------------|
| **Total Bets** | ≥ 30 | Statistically meaningful sample |
| **Win Rate (Spread)** | ≥ 52% | Above breakeven at -110 odds |
| **Win Rate (B2B Rest)** | ≥ 54% | Close to backtest (57%) |
| **Arbs Found** | ≥ 2 | Validates arb detection works |
| **ROI** | > 0% | Positive return overall |
| **Daily Execution** | 100% | No missed days |

### Red Flags (Stop if you see these)

- ❌ Win rate < 48% (worse than random)
- ❌ All bets from single strategy (diversification failure)
- ❌ ROI < -10% (systematic error)
- ❌ Edge consistently negative (model calibration issue)

---

## 🛠️ Troubleshooting

### Common Issues

**Pipeline not finding games**:
```bash
# Check if NBA games today
python -c "from src.data.odds_api import OddsAPIClient; client = OddsAPIClient(); print(len(client.get_current_odds()))"

# If 0, no games today (normal on off days)
```

**No bets generated**:
- Check edge thresholds aren't too strict (spread needs ≥5% edge)
- Verify games have features (need recent stats)
- Check if daily limits already hit

**Discord notifications not working**:
```bash
# Test Discord webhook
python scripts/send_daily_report.py

# If fails, check webhook URL in .env
```

**Cron jobs not running**:
```bash
# Check cron log
cat logs/cron_betting.log

# Verify crontab is set
crontab -l

# Make sure paths are absolute in crontab
```

---

## 📈 After 2 Weeks: Going Live

Once you've validated paper trading results:

### Pre-Live Checklist

- [ ] Reviewed all paper trading bets (no errors)
- [ ] Win rates align with backtests (±5%)
- [ ] ROI is positive
- [ ] Bankroll size confirmed ($1,000 recommended)
- [ ] Discord notifications working perfectly
- [ ] Comfortable with bet sizing

### Enable Live Betting

```bash
# Edit config file
nano config/multi_strategy_config.yaml

# Change line 10:
paper_trading: false  # LIVE BETTING ENABLED

# Commit the change
git add config/multi_strategy_config.yaml
git commit -m "Enable live betting after successful paper trading validation"
```

### Start Small

- Week 1 Live: Reduce allocations by 50% (test real execution)
- Week 2 Live: Full allocations if Week 1 successful
- Month 1: Monitor closely, adjust thresholds if needed

---

## 🔐 Safety Features

### Built-in Protections

| Feature | Setting | Protection |
|---------|---------|------------|
| **Kelly Sizing** | 25% fractional | Prevents over-betting |
| **Daily Exposure Limit** | 15% of bankroll | Caps daily risk |
| **Drawdown Warning** | 10% loss | Early alert |
| **Drawdown Pause** | 20% loss | Auto-pause betting |
| **Hard Stop** | 30% loss | Circuit breaker |
| **Min Bet Size** | $10 | Prevents dust bets |
| **Daily Limits** | 5-10 per strategy | Prevents overtrading |

### Emergency Stop

```bash
# Disable all strategies immediately
python -c "import yaml; config = yaml.safe_load(open('config/multi_strategy_config.yaml')); config['strategies']['arbitrage']['enabled'] = False; config['strategies']['b2b_rest']['enabled'] = False; config['strategies']['spread']['enabled'] = False; yaml.dump(config, open('config/multi_strategy_config.yaml', 'w'))"

# Or simply:
nano config/multi_strategy_config.yaml
# Set all "enabled: false"
```

---

## 📞 Support & Resources

### Log Locations

```
logs/cron_betting.log       - Daily pipeline execution
logs/daily_report.log        - Discord report history
data/bets/bets.db           - All bet history (SQLite)
```

### Key Commands Reference

```bash
# View recent bets
python -c "from src.bet_tracker import get_bet_history; print(get_bet_history().tail(10))"

# Check current bankroll
python -c "from src.bankroll.bankroll_manager import BankrollManager; bm = BankrollManager(); print(f'${bm.get_current_bankroll():.2f}')"

# Settle pending bets (after games finish)
python -c "from src.data.nba_stats import NBAStatsClient; from src.bet_tracker import settle_all_pending; client = NBAStatsClient(); games = client.get_games('2026-01-05', '2026-01-05'); settle_all_pending(games)"
```

---

## ✅ Deployment Checklist

### Initial Setup
- [x] API key configured in .env
- [x] Discord webhook configured
- [x] All 3 strategies enabled
- [x] Paper trading mode enabled
- [x] Bankroll set correctly ($1,000)

### Automation Setup
- [ ] Cron jobs configured (betting + reporting)
- [ ] Test manual pipeline run successful
- [ ] Discord notifications received
- [ ] Dashboard accessible

### Week 1-2 (Paper Trading)
- [ ] Daily: Check pipeline ran
- [ ] Daily: Review bets in Discord
- [ ] Weekly: Analyze performance vs backtest
- [ ] Week 2 End: Decide on going live

### Go Live (After Paper Trading Success)
- [ ] Disable paper trading mode
- [ ] Reduce allocations 50% for Week 1
- [ ] Full monitoring for Month 1
- [ ] Adjust thresholds if needed

---

## 🎉 You're Ready!

Your 3-strategy system is configured and ready for paper trading.

**Next Steps**:
1. Run `python scripts/daily_multi_strategy_pipeline.py --dry-run` to test
2. Set up cron jobs for automation
3. Monitor for 2 weeks in paper trading mode
4. Go live after validation

Good luck! 🚀
