# Cron Job Deployment Guide

Complete instructions for deploying the multi-strategy betting system to your droplet with automated daily execution.

---

## 📋 Prerequisites

Before setting up the cron job, ensure:
- ✅ Code pushed to GitHub (commit: 1c1ba17)
- ✅ Droplet has SSH access configured
- ✅ Python environment set up on droplet
- ✅ All dependencies installed

---

## 🚀 Step 1: SSH into Your Droplet

```bash
ssh root@your-droplet-ip
# Or: ssh your-username@your-droplet-ip
```

---

## 📥 Step 2: Pull Latest Changes

```bash
cd /path/to/nbamodels  # Navigate to your project directory
git pull origin main   # Pull the latest changes
```

**Expected output:**
```
Updating 8a3a2d0..1c1ba17
Fast-forward
 config/multi_strategy_config.yaml | 4 ++--
 1 file changed, 2 insertions(+), 2 deletions(-)
```

---

## 🔧 Step 3: Test the Pipeline Manually

Before setting up automation, verify everything works:

```bash
# Activate your Python environment (if using virtualenv/conda)
source venv/bin/activate  # Or: conda activate nbamodels

# Run a dry-run test
python scripts/daily_multi_strategy_pipeline.py --dry-run

# If successful, run actual paper trading
python scripts/daily_multi_strategy_pipeline.py
```

**What to look for:**
- ✅ All 4 strategies load successfully
- ✅ Models load without errors
- ✅ Pipeline completes with "✅ Pipeline complete!"
- ✅ No critical errors in output

---

## ⏰ Step 4: Set Up Cron Job

### Option A: Run Daily at 10 AM ET (Recommended)

This timing allows you to review today's games before they start (most NBA games are 7-10 PM ET).

```bash
# Open crontab editor
crontab -e
```

Add this line (adjust path to your project directory):

```bash
# Multi-strategy betting pipeline - runs daily at 10 AM ET
0 10 * * * cd /path/to/nbamodels && /path/to/python scripts/daily_multi_strategy_pipeline.py >> /var/log/nbamodels/cron.log 2>&1
```

**Example with actual paths:**
```bash
0 10 * * * cd /root/nbamodels && /root/nbamodels/venv/bin/python scripts/daily_multi_strategy_pipeline.py >> /var/log/nbamodels/cron.log 2>&1
```

### Option B: Run Daily at 9 AM ET (Early Morning)

```bash
0 9 * * * cd /path/to/nbamodels && /path/to/python scripts/daily_multi_strategy_pipeline.py >> /var/log/nbamodels/cron.log 2>&1
```

### Option C: Run Twice Daily (Morning + Afternoon)

Some days have early games, so running twice catches all opportunities:

```bash
# Morning run at 9 AM ET
0 9 * * * cd /path/to/nbamodels && /path/to/python scripts/daily_multi_strategy_pipeline.py >> /var/log/nbamodels/cron_morning.log 2>&1

# Afternoon run at 4 PM ET
0 16 * * * cd /path/to/nbamodels && /path/to/python scripts/daily_multi_strategy_pipeline.py >> /var/log/nbamodels/cron_afternoon.log 2>&1
```

---

## 📝 Step 5: Create Log Directory

```bash
# Create directory for cron logs
sudo mkdir -p /var/log/nbamodels
sudo chown $USER:$USER /var/log/nbamodels
```

---

## 🔍 Step 6: Verify Cron Job is Active

```bash
# List all cron jobs
crontab -l

# Expected output:
# 0 10 * * * cd /path/to/nbamodels && /path/to/python scripts/daily_multi_strategy_pipeline.py >> /var/log/nbamodels/cron.log 2>&1
```

---

## 📊 Step 7: Monitor Execution

### Check if cron job ran successfully:

```bash
# View latest cron log
tail -100 /var/log/nbamodels/cron.log

# Check for successful completion
grep "Pipeline complete" /var/log/nbamodels/cron.log

# Check for errors
grep -i "error\|failed" /var/log/nbamodels/cron.log
```

### Check system cron logs:

```bash
# Ubuntu/Debian
sudo tail -f /var/log/syslog | grep CRON

# CentOS/RHEL
sudo tail -f /var/log/cron
```

---

## 🐛 Troubleshooting

### Cron job not running?

**1. Check cron service is running:**
```bash
sudo service cron status    # Ubuntu/Debian
sudo service crond status   # CentOS/RHEL
```

**2. Check timezone is correct:**
```bash
timedatectl  # Check current timezone
sudo timedatectl set-timezone America/New_York  # Set to ET
```

**3. Test the exact cron command manually:**
```bash
cd /path/to/nbamodels && /path/to/python scripts/daily_multi_strategy_pipeline.py
```

### Python environment issues?

**Cron doesn't load your .bashrc/.zshrc**, so you need to activate the environment in the cron command:

```bash
0 10 * * * cd /root/nbamodels && source venv/bin/activate && python scripts/daily_multi_strategy_pipeline.py >> /var/log/nbamodels/cron.log 2>&1
```

Or use the full path to Python:
```bash
0 10 * * * cd /root/nbamodels && /root/nbamodels/venv/bin/python scripts/daily_multi_strategy_pipeline.py >> /var/log/nbamodels/cron.log 2>&1
```

### Missing environment variables?

If you use `.env` file, make sure cron can access it:

```bash
0 10 * * * cd /root/nbamodels && /root/nbamodels/venv/bin/python -c "from dotenv import load_dotenv; load_dotenv()" && /root/nbamodels/venv/bin/python scripts/daily_multi_strategy_pipeline.py >> /var/log/nbamodels/cron.log 2>&1
```

Or source it directly in the script.

---

## 📧 Optional: Email Notifications

Get email alerts when the pipeline runs:

### Using `mail` command:

```bash
# Install mail utility
sudo apt-get install mailutils  # Ubuntu/Debian

# Update cron job to send email
0 10 * * * cd /path/to/nbamodels && /path/to/python scripts/daily_multi_strategy_pipeline.py >> /var/log/nbamodels/cron.log 2>&1 && mail -s "NBA Betting Pipeline Complete" your@email.com < /var/log/nbamodels/cron.log
```

### Using Discord webhooks (already configured):

The pipeline already supports Discord notifications if you set `DISCORD_WEBHOOK_URL` in your `.env` file:

```bash
# In your .env file on the droplet
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_URL
```

No cron changes needed - the pipeline will automatically send Discord notifications.

---

## 🎯 Advanced: Multiple Pipelines

If you want to run different strategies at different times:

```bash
# Arbitrage scan (faster, runs more frequently)
0 */4 * * * cd /root/nbamodels && python scripts/arbitrage_only_pipeline.py >> /var/log/nbamodels/arb.log 2>&1

# Full multi-strategy (runs once daily)
0 10 * * * cd /root/nbamodels && python scripts/daily_multi_strategy_pipeline.py >> /var/log/nbamodels/daily.log 2>&1

# Player props odds collection (runs daily for backtest data)
0 11 * * * cd /root/nbamodels && python scripts/collect_player_prop_odds.py >> /var/log/nbamodels/props_odds.log 2>&1
```

---

## ✅ Verification Checklist

After deployment, verify:

- [ ] Cron job is in `crontab -l`
- [ ] Log directory exists and is writable
- [ ] Timezone is set to America/New_York (ET)
- [ ] Python environment path is correct
- [ ] Manual test of pipeline succeeds
- [ ] Cron log file is created after scheduled time
- [ ] No errors in cron log
- [ ] Database is being updated with recommendations
- [ ] (Optional) Discord notifications working

---

## 📈 What Happens Daily

Once deployed, the cron job will automatically:

1. ✅ Fetch today's NBA games and odds
2. ✅ Run all 4 strategies (Arbitrage, Spread, B2B, Props)
3. ✅ Calculate optimal bet sizes using Kelly criterion
4. ✅ Apply risk management filters
5. ✅ Log recommendations to database (paper trading)
6. ✅ Track performance for future analysis
7. ✅ (Optional) Send Discord/email notification

**Expected daily execution time:** 30-60 seconds

---

## 🔄 Updating the System

When you make changes and push to GitHub:

```bash
# SSH into droplet
ssh root@your-droplet-ip

# Navigate to project
cd /path/to/nbamodels

# Pull latest changes
git pull origin main

# Restart is automatic - cron will use new code on next run
```

No need to restart cron - it will automatically use the updated code on the next scheduled run.

---

## 🛑 Stopping/Disabling

### Temporarily disable (comment out):
```bash
crontab -e
# Add # at the start of the line:
# 0 10 * * * cd /path/to/nbamodels && python ...
```

### Permanently remove:
```bash
crontab -e
# Delete the entire line
```

### View what's scheduled:
```bash
crontab -l
```

---

## 🎉 You're Done!

Your multi-strategy betting system is now fully automated and will run daily at your specified time.

**Current Configuration:**
- 📊 Arbitrage: 25% allocation (10.93% avg profit validated)
- 📈 Spread: 50% allocation (+23.6% ROI validated)
- 🏃 B2B Rest: 15% allocation (+8.7% ROI validated)
- 🎯 Player Props: 10% allocation (4 production models)

**Next Check-in:**
- Review performance after 7 days
- Analyze which strategies are performing best
- Adjust allocations if needed
- Consider moving from paper trading to live betting (when ready)

Happy automated betting! 🚀
