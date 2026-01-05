# Quick Start Guide - Getting Your Bets

## 🎯 New! Central Orchestrator

You now have a **single command** to manage your entire NBA betting system:

```bash
python nba.py --help        # See all available commands
python nba.py status        # Check system status
python nba.py bets          # Get today's bet recommendations
python nba.py daily         # Complete daily workflow
```

### Quick Commands

```bash
# Most common usage - get today's bets
python nba.py bets

# Check system health
python nba.py status

# Update all data before placing bets
python nba.py update

# Complete daily routine (update + bets + status)
python nba.py daily
```

The orchestrator coordinates all your scripts:
- ✅ Fetch today's NBA games
- ✅ Generate predictions
- ✅ Apply your betting strategy
- ✅ Calculate risk-adjusted bet sizes (with correlation, drawdown, and exposure protection)
- ✅ Show clean recommendations

## 📋 All Available Commands

| Command | Description |
|---------|-------------|
| `python nba.py status` | Show system status (bankroll, bets, model) |
| `python nba.py bets` | Get today's bet recommendations |
| `python nba.py pipeline` | Run full betting pipeline (paper mode) |
| `python nba.py pipeline --live` | Run pipeline with LIVE bets ⚠️ |
| `python nba.py update` | Update all data (lines, odds, settlement) |
| `python nba.py settle` | Settle pending bets |
| `python nba.py train` | Retrain prediction model |
| `python nba.py backtest` | Run backtest |
| `python nba.py dashboard` | Launch analytics dashboard |
| `python nba.py daily` | Complete daily workflow |
| `python nba.py init` | First-time setup |

## 📊 What You'll See

The output shows:
- **Games evaluated**: All NBA games scheduled for today
- **Actionable bets**: Bets that meet your strategy criteria
- **Bet amounts**: Risk-adjusted sizing based on:
  - Kelly criterion (25% base)
  - Correlation discounts (multiple bets on same team/conference)
  - Drawdown scaling (reduces sizing during losing streaks)
  - Daily/weekly exposure limits
  - Circuit breaker (stops betting at 30% drawdown)

## 🔧 Three Ways to Use Your Betting System

### 1. **View Recommendations Only** (No database logging)
```bash
python nba.py bets
```
**Use this when**: You just want to see what bets the system recommends without logging anything.

### 2. **Log Paper Trades** (Track performance without real money)
```bash
python nba.py pipeline
```
**Use this when**: You want to track the system's performance over time in your database without risking real money.

### 3. **Place Live Bets** ⚠️ REAL MONEY
```bash
python nba.py pipeline --live
```
**Use this when**: You're ready to place REAL bets with REAL money. The system will enforce all risk management rules.

## 🛡️ Risk Management (Always Active)

The system automatically protects you with:

| Protection | Limit | What It Does |
|------------|-------|--------------|
| **Same Team** | 15% bankroll | Prevents over-concentration on one team |
| **Same Division** | 15% bankroll | Limits correlated exposure within divisions |
| **Same Conference** | 25% bankroll | Broader correlation protection |
| **Daily Exposure** | 20% bankroll | Caps total risk per day |
| **Weekly Exposure** | 50% bankroll | Prevents over-betting in one week |
| **Drawdown Scaling** | 10-30% | Gradually reduces bet sizes during losses |
| **Circuit Breaker** | 30% drawdown | **HARD STOP** - No betting when down 30% |

## 📈 Example Output

When there ARE actionable bets, you'll see:
```
================================================================================
🎯 TODAY'S BET RECOMMENDATIONS (3 bets)
================================================================================

#1. HOME Boston Celtics vs Miami Heat
   🛒 Line shopping: DraftKings @ -110 (best of 8 books)
   🛡️  Risk adjustments: correlation_division
   📊 Correlation factor: 85.00%
   📉 Drawdown factor: 100.00%
   💰 Final size: $185.00 (from $218.00 Kelly)

#2. AWAY Los Angeles Lakers vs Denver Nuggets
   🛒 Line shopping: FanDuel @ -108 (best of 8 books)
   🛡️  Risk adjustments: none
   📊 Correlation factor: 100.00%
   📉 Drawdown factor: 100.00%
   💰 Final size: $225.00 (from $225.00 Kelly)

#3. HOME Golden State Warriors vs Phoenix Suns
   🛒 Line shopping: BetMGM @ -112 (best of 8 books)
   🛡️  Risk adjustments: correlation_conference
   📊 Correlation factor: 90.00%
   📉 Drawdown factor: 100.00%
   💰 Final size: $202.50 (from $225.00 Kelly)

TOTAL RISK: $612.50
```

## ⚙️ Customization

### Change Strategy
The default strategy is `clv_filtered` (requires 7-point edge). Other options:
```bash
python scripts/daily_betting_pipeline.py --strategy baseline      # More aggressive (3-point edge)
python scripts/daily_betting_pipeline.py --strategy optimal_timing  # Timing-based
python scripts/daily_betting_pipeline.py --strategy team_filtered   # Team-specific filters
```

### Adjust Risk Settings
Edit the risk parameters in your pipeline by modifying `RiskConfig` initialization in `daily_betting_pipeline.py`:

```python
# More conservative
risk_config = RiskConfig(
    max_same_team_exposure=0.08,      # Lower from 15% to 8%
    drawdown_hard_stop=0.20           # Stop at 20% instead of 30%
)

# More aggressive
risk_config = RiskConfig(
    max_daily_exposure=0.25,          # Increase from 20% to 25%
    drawdown_scale_start=0.15         # Start scaling later (15% vs 10%)
)
```

### Check Your Kelly Fraction
Current setting: **25% Kelly** (conservative)

To change, edit `KellyBetSizer` initialization:
```python
kelly_sizer = KellyBetSizer(fraction=0.10)  # Even more conservative (10% Kelly)
kelly_sizer = KellyBetSizer(fraction=0.50)  # More aggressive (50% Kelly)
```

## 📅 Automate with Cron

### Option 1: Daily Workflow (Recommended)
To run the complete daily workflow every day at 9 PM:
```bash
# Edit your crontab
crontab -e

# Add this line:
0 21 * * * cd /Users/harrisonju/PycharmProjects/nbamodels && /usr/local/bin/python nba.py daily >> logs/daily.log 2>&1
```

### Option 2: Just Betting Pipeline
To run only the betting pipeline:
```bash
0 21 * * * cd /Users/harrisonju/PycharmProjects/nbamodels && /usr/local/bin/python nba.py pipeline >> logs/pipeline.log 2>&1
```

### Option 3: Full Crontab
Or install the full crontab (includes data collection):
```bash
crontab deploy/crontab.txt
```

## 🔍 View Your Performance

Launch the analytics dashboard:
```bash
python nba.py dashboard
```

Then open http://localhost:8501 in your browser to see:
- Bet history and results
- Win rate and ROI
- Bankroll progression
- Team-level performance
- Strategy comparison

## ❓ Common Issues

### "No actionable bets found"
This is normal! It means either:
- No games are scheduled today
- None of the games meet your strategy's edge threshold
- Circuit breaker is active (30% drawdown reached)

### "Using placeholder model predictions"
Your model file needs to be retrained or isn't loading correctly. The system falls back to neutral predictions (0.0) for all games. To fix, run:
```bash
python scripts/train_model.py
```

### Risk management showing no adjustments
This is actually good! It means:
- Your bets aren't correlated (different teams/conferences)
- Bankroll is healthy (no drawdown)
- Exposure limits not reached

The system is monitoring everything and will intervene when needed.

---

**Status**: ✅ Risk Management Active | 🛡️ Circuit Breaker Armed | 🎯 Ready for Live Betting

For detailed technical documentation, see:
- `RISK_MANAGEMENT_SUMMARY.md` - Full risk system details
- `LIVE_RISK_INTEGRATION.md` - Integration documentation
