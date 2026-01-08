# Reporting System - Quick Start Guide

## Installation (Already Complete!)

All reporting modules have been installed and integrated with your NBA betting system.

## 1. Set Up Discord (Optional but Recommended)

### Get Discord Webhook URL
1. Open Discord
2. Go to your server's **Server Settings** > **Integrations** > **Webhooks**
3. Click **New Webhook** or **Create Webhook**
4. Name it (e.g., "NBA Betting Bot")
5. Copy the **Webhook URL**

### Configure Environment Variable
```bash
# Add to ~/.bashrc or ~/.zshrc
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/YOUR/WEBHOOK/URL"

# Or create a .env file in project root
echo 'DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR/WEBHOOK/URL' > .env
```

## 2. Basic Usage

### Send Daily Discord Update
```bash
python nba.py report daily
```
Sends a performance summary to Discord with:
- Bets placed/settled today
- Today's profit/loss
- Current bankroll
- 30-day statistics

### Generate Weekly Report
```bash
python nba.py report weekly
```
Creates a text report with:
- Overall performance (ROI, win rate, profit)
- Bet type breakdown
- Top/worst performing teams

Saved to: `data/reports/weekly_report_YYYYMMDD.txt`

### Generate Monthly Report
```bash
python nba.py report monthly
```
Creates an investor-grade report with:
- All weekly metrics
- Sharpe ratio
- Maximum drawdown
- Win/loss streaks
- Rolling statistics
- Kelly criterion analysis

Saved to: `data/reports/monthly_report_YYYYMMDD.txt`

### Run Data Reconciliation
```bash
python nba.py reconcile
```
Validates data integrity:
- Checks for missing fields
- Validates profit calculations
- Detects duplicates
- Verifies bankroll consistency

Saved to: `data/reports/reconciliation_YYYYMMDD_HHMMSS.txt`

## 3. Advanced Usage

### Custom Time Periods
```bash
# Last 4 weeks
python nba.py report weekly --weeks 4

# Last 3 months
python nba.py report monthly --months 3

# Reconcile last 30 days
python nba.py reconcile --days 30
```

## 4. Automated Daily Reports (Cron)

### Set Up Cron Job
```bash
# Edit crontab
crontab -e

# Add this line for daily reports at 11 PM
0 23 * * * cd /path/to/nbamodels && /usr/local/bin/python nba.py report daily >> logs/daily_report.log 2>&1
```

Replace `/path/to/nbamodels` with your actual path:
```bash
pwd  # Run this in your nbamodels directory to get the path
```

Replace `/usr/local/bin/python` with your Python path:
```bash
which python  # Run this to find your Python path
```

### Verify Cron Setup
```bash
# List cron jobs
crontab -l

# Check logs
tail -f logs/daily_report.log
```

## 5. Sample Report Output

### Weekly Report Preview
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

⭐ TOP PERFORMING TEAMS
--------------------------------------------------------------------------------
LAL: +$87.50 (+17.5% ROI, 75% WR)
GSW: +$45.20 (+9.0% ROI, 60% WR)
```

### Monthly Report Preview
```
MONTHLY PERFORMANCE REPORT - Last 1 month(s)
ROI: +8.01%
Sharpe Ratio: 1.85
Max Drawdown: $124.50 (3.2%)
Longest Win Streak: 7 bets
Performance Trend: IMPROVING
```

## 6. Troubleshooting

### Discord Messages Not Sending
```bash
# Check if webhook URL is set
echo $DISCORD_WEBHOOK_URL

# Test manually
python -c "from src.reporting import DiscordNotifier; d = DiscordNotifier(); d.send_daily_update()"
```

### Reports Show "No Data"
- Make sure you have bets in the database
- Check date range (reports look back from today)
- Verify database exists: `ls -lh data/bets/bets.db`

### Reconciliation Shows Errors
- This is normal - it validates data integrity
- Review the report to see what issues were found
- Most warnings are informational, errors need attention

## 7. Common Workflows

### Morning Routine
```bash
# Update all data and check status
python nba.py update && python nba.py status
```

### After Games End
```bash
# Settle bets and send update
python nba.py settle && python nba.py report daily
```

### Weekly Review (Sunday)
```bash
# Generate comprehensive weekly report
python nba.py report weekly
cat data/reports/weekly_report_*.txt | tail -100
```

### Monthly Investor Report
```bash
# Generate and email monthly report
python nba.py report monthly
# Then manually review data/reports/monthly_report_*.txt
```

## 8. Integration with Other Tools

### Slack (Future Enhancement)
Change Discord webhook to Slack webhook URL - same format!

### Email (Future)
Could add email sending via SMTP - currently manual

### Telegram (Future)
Could integrate Telegram bot API

## 9. Files and Directories

```
nbamodels/
├── src/reporting/          # Reporting modules
│   ├── discord_notifier.py
│   ├── weekly_report.py
│   ├── monthly_report.py
│   └── reconciliation.py
├── scripts/
│   └── send_daily_report.py  # Cron automation script
├── data/reports/           # Generated reports (auto-created)
└── logs/                   # Log files
```

## 10. Next Steps

1. **Set up Discord webhook** (5 minutes)
2. **Test daily report**: `python nba.py report daily`
3. **Generate first weekly report**: `python nba.py report weekly`
4. **Set up cron job** for daily automation (5 minutes)
5. **Review monthly report** when you have 30+ days of data

## Need Help?

Check the comprehensive documentation:
- `src/reporting/README.md` - Full documentation
- `REPORTING_IMPLEMENTATION.md` - Technical details
- `python nba.py report --help` - Command help

## Pro Tips

1. **Run reconciliation weekly** to catch data issues early
2. **Review monthly reports** before making strategy changes
3. **Set up cron jobs** for hands-off automation
4. **Monitor Sharpe ratio** - aim for > 1.0
5. **Track CLV** - positive CLV indicates good line timing
6. **Watch max drawdown** - if > 10%, review risk management

That's it! Your reporting system is ready to use. Start with `python nba.py report daily` to test it out!
