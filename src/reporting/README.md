# Reporting Module

Comprehensive automated reporting system for the NBA betting model with Discord integration, performance analytics, and data reconciliation.

## Features

### 1. Daily Discord Updates (`discord_notifier.py`)
- Automated daily performance summaries
- Real-time bet placement notifications
- Bet settlement alerts
- System status updates
- Rich embed formatting with color-coded performance

### 2. Weekly Strategy Reviews (`weekly_report.py`)
- Overall performance metrics (ROI, win rate, profit)
- Strategy-by-strategy breakdown
- Bet type analysis
- Top/worst performing teams
- Situational analysis (B2B, rest advantage)

### 3. Monthly Investor-Grade Reports (`monthly_report.py`)
- All weekly metrics plus advanced analytics:
  - **Sharpe Ratio**: Risk-adjusted returns (annualized)
  - **Maximum Drawdown**: Peak-to-trough decline analysis
  - **Win/Loss Streaks**: Longest winning/losing sequences
  - **Rolling Statistics**: Moving averages and volatility trends
  - **Kelly Criterion Accuracy**: Bet sizing analysis

### 4. Reconciliation Engine (`reconciliation.py`)
- Validates bet outcomes against NBA API results
- Checks data integrity (missing fields, invalid values)
- Verifies bankroll calculations
- Detects duplicates and inconsistencies
- Generates detailed discrepancy reports

## Usage

### Command Line (via Orchestrator)

```bash
# Send daily Discord update
python nba.py report daily

# Generate weekly report (last 1 week)
python nba.py report weekly

# Generate weekly report (last 4 weeks)
python nba.py report weekly --weeks 4

# Generate monthly report (last 1 month)
python nba.py report monthly

# Generate monthly report (last 3 months)
python nba.py report monthly --months 3

# Run reconciliation (last 7 days)
python nba.py reconcile

# Run reconciliation (last 30 days)
python nba.py reconcile --days 30
```

### Python API

```python
# Daily Discord update
from src.reporting import DiscordNotifier

notifier = DiscordNotifier()
notifier.send_daily_update()

# Send bet placed notification
notifier.send_bet_placed({
    'home_team': 'LAL',
    'away_team': 'GSW',
    'bet_side': 'HOME',
    'line': -3.5,
    'odds': 1.91,
    'bet_amount': 50,
    'edge': 0.05
})

# Weekly report
from src.reporting import WeeklyReportGenerator

weekly = WeeklyReportGenerator()
report = weekly.generate_report(weeks_back=1)
formatted = weekly.format_report_text(report)
print(formatted)

# Monthly report
from src.reporting import MonthlyReportGenerator

monthly = MonthlyReportGenerator()
report = monthly.generate_monthly_report(months_back=1)
formatted = monthly.format_monthly_report_text(report)
print(formatted)

# Reconciliation
from src.reporting import ReconciliationEngine

engine = ReconciliationEngine()
report = engine.run_full_reconciliation(days_back=7)
formatted = engine.format_reconciliation_report(report)
print(formatted)
```

## Automated Daily Reports (Cron)

Set up automated daily Discord updates using cron:

```bash
# Edit crontab
crontab -e

# Add this line to run daily at 11 PM
0 23 * * * cd /path/to/nbamodels && /path/to/python scripts/send_daily_report.py >> logs/daily_report.log 2>&1

# Or use the orchestrator
0 23 * * * cd /path/to/nbamodels && /path/to/python nba.py report daily >> logs/daily_report.log 2>&1
```

### Cron Schedule Examples

```bash
# Daily at 11 PM
0 23 * * * ...

# Every 6 hours
0 */6 * * * ...

# Weekly on Sunday at 8 AM
0 8 * * 0 ...

# Monthly on the 1st at 9 AM
0 9 1 * * ...
```

## Discord Webhook Setup

1. Create a Discord webhook in your server:
   - Go to Server Settings > Integrations > Webhooks
   - Click "New Webhook"
   - Name it (e.g., "NBA Betting Bot")
   - Copy the webhook URL

2. Set the environment variable:
   ```bash
   export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/your/webhook/url"
   ```

   Or add to `.env` file:
   ```
   DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your/webhook/url
   ```

## Report Output

### Daily Discord Update Format
```
📈 Daily Betting Performance
Report for January 5, 2025

📊 Today's Activity
Bets Placed: 5
Volume: $250.00
Bets Settled: 3
Wins: 2 (66.7%)

💰 Today's Results
Profit/Loss: +$45.50
Current Bankroll: $1,245.50

📈 Last 30 Days
Total Bets: 82
Win Rate: 54.9%
Total Profit: +$342.18
ROI: +8.3%
```

### Weekly Report Format
```
================================================================================
📊 WEEKLY STRATEGY REVIEW - Last 1 week(s)
================================================================================

📈 OVERALL PERFORMANCE
--------------------------------------------------------------------------------
Total Bets: 42 (38 settled, 4 pending)
Total Wagered: $2,100.00
Total Profit: +$156.40
ROI: +7.45%
Win Rate: 55.3%
Avg Edge: +3.2%
Avg CLV: +1.8%

🎯 PERFORMANCE BY STRATEGY
--------------------------------------------------------------------------------
baseline:
  Bets: 25 | Win Rate: 56.0% | ROI: +8.2%
  Wagered: $1,250.00 | Profit: +$102.50

clv_filtered:
  Bets: 13 | Win Rate: 53.8% | ROI: +6.1%
  Wagered: $650.00 | Profit: +$39.65
```

### Monthly Report Format
```
================================================================================
MONTHLY PERFORMANCE REPORT - Last 1 month(s)
================================================================================
Period: 2024-12-05 to 2025-01-05

OVERALL PERFORMANCE
--------------------------------------------------------------------------------
Total Bets: 168 (152 settled, 16 pending)
Total Wagered: $8,400.00
Total Profit: +$672.80
ROI: +8.01%
Win Rate: 55.9%

ADVANCED METRICS (INVESTOR-GRADE)
--------------------------------------------------------------------------------
Sharpe Ratio (Annualized): 1.85
Maximum Drawdown: $124.50 (3.2%)
Drawdown Duration: 8 days
Current Drawdown: 0.0%
Underwater: No

WIN/LOSS STREAKS
--------------------------------------------------------------------------------
Longest Win Streak: 7 bets
Longest Loss Streak: 4 bets
Current Streak: 3 win

ROLLING STATISTICS (Last 20 Bets)
--------------------------------------------------------------------------------
Rolling ROI: +9.2%
Rolling Win Rate: 60.0%
Rolling Volatility: 12.4%
Performance Trend: IMPROVING

BET SIZING ANALYSIS
--------------------------------------------------------------------------------
Avg Kelly Fraction: 2.34%
Edge/Size Correlation: +0.687
Oversized Bets (>37.5% Kelly): 0
Undersized Bets (<12.5% Kelly): 8
```

### Reconciliation Report Format
```
================================================================================
RECONCILIATION REPORT
================================================================================
Date: 2025-01-05T23:00:00
Period: Last 7 days
Bets Reviewed: 42

SUMMARY
--------------------------------------------------------------------------------
Total Issues Found: 2
  - Errors: 0
  - Warnings: 2
Bankroll Consistent: YES

BANKROLL VALIDATION
--------------------------------------------------------------------------------
Status: Bankroll calculations are consistent
Initial Bankroll: $1,000.00
Current Bankroll: $1,245.50
Total Profit: +$245.50

DATA INTEGRITY ISSUES
--------------------------------------------------------------------------------
[WARNING] Found 2 bets for game 0022400567 at 2025-01-03 18:30:00

================================================================================
STATUS: ISSUES DETECTED (0 errors, 2 warnings)
================================================================================
```

## Error Handling

All modules include comprehensive error handling:

- Database connection errors
- API rate limiting/failures
- Discord webhook errors
- Missing or invalid data
- Network timeouts

Errors are logged to both console and log files in `logs/` directory.

## Data Validation

The reconciliation engine validates:

1. **Bet Outcomes**: Compares recorded outcomes vs actual game results
2. **Data Integrity**: Checks for missing fields, invalid odds, negative amounts
3. **Profit Calculations**: Verifies profit = bet_amount × (odds - 1) for wins
4. **Bankroll Consistency**: Ensures bankroll = initial + sum(profits)
5. **Duplicates**: Detects potential duplicate bets
6. **Timestamps**: Validates settlement timestamps exist for settled bets

## Performance Metrics

### Basic Metrics (All Reports)
- **ROI**: Return on Investment = Total Profit / Total Wagered × 100
- **Win Rate**: Winning bets / Total bets × 100
- **Total Profit**: Sum of all profits/losses
- **Average Edge**: Mean predicted edge across all bets
- **CLV**: Closing Line Value (market movement after bet placement)

### Advanced Metrics (Monthly Only)
- **Sharpe Ratio**: (Return - Risk-free rate) / Volatility (annualized)
- **Max Drawdown**: Largest peak-to-trough decline ($ and %)
- **Rolling Stats**: Moving averages over last 20 bets
- **Kelly Analysis**: Bet sizing vs optimal Kelly criterion

## File Locations

- Reports saved to: `data/reports/`
- Logs saved to: `logs/`
- Database: `data/bets/bets.db`

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Statistical calculations
- `requests`: Discord webhook API calls
- `loguru`: Logging
- `sqlite3`: Database access

## Future Enhancements

Potential additions:
- PDF report generation (via ReportLab or WeasyPrint)
- Email notifications
- Slack integration
- Telegram bot integration
- Multi-period comparison charts
- Monte Carlo simulations for risk analysis
- Interactive HTML reports
- Automated report scheduling via APScheduler
