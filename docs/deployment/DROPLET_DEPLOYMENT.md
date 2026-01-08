# Droplet Deployment Guide - DigitalOcean

**Setting up automated betting pipeline on DigitalOcean droplet**

---

## 🖥️ Pre-Deployment Checklist

Before setting up cron jobs, ensure your droplet has:

```bash
# SSH into your droplet
ssh root@your-droplet-ip

# Check Python version (need 3.8+)
python3 --version

# Check if project exists
ls -la ~/nbamodels  # or wherever you cloned the repo

# Check virtual environment
ls -la ~/nbamodels/venv  # or .venv
```

---

## 📦 Step 1: Initial Droplet Setup

### 1.1 Clone Repository (if not done)

```bash
# SSH into droplet
ssh root@your-droplet-ip

# Clone repository
cd ~
git clone https://github.com/yourusername/nbamodels.git
cd nbamodels
```

### 1.2 Set Up Python Environment

```bash
# Install Python 3 and pip if needed
sudo apt update
sudo apt install python3 python3-pip python3-venv -y

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.3 Configure Environment Variables

```bash
# Create .env file
nano .env

# Add your keys (paste this content):
```

```bash
# NBA Betting System Environment Variables

# REQUIRED: The Odds API Key
ODDS_API_KEY=16eca04028d0db0a86ce957e4e87f7a7

# OPTIONAL: Discord Webhook for Reports
DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/1457822784670404689/QZ-lRQFa_e5bnA6T1T2VwPzf8U6kW5ZI5nzU1p3RuF2HlYqH95saNQDwH4IRWIyaoqsZ"
```

**Save and exit**: `Ctrl+X`, `Y`, `Enter`

### 1.4 Verify Configuration

```bash
# Test the environment
source venv/bin/activate
cat .env

# Test pipeline (dry run)
python scripts/daily_multi_strategy_pipeline.py --dry-run

# Should see:
# ✓ Enabled ArbitrageStrategy
# ✓ Enabled B2BRestStrategy
# Pipeline complete!
```

---

## ⏰ Step 2: Set Up Cron Jobs

### 2.1 Update Cron Scripts for Droplet

The existing cron scripts assume you're running locally. We need to ensure they work on the droplet.

**Verify script has correct paths:**

```bash
# Check the cron_betting.sh script
cat scripts/cron_betting.sh

# Make sure it's executable
chmod +x scripts/cron_betting.sh

# Test it manually first
./scripts/cron_betting.sh

# Check the log
tail -20 logs/cron_betting.log
```

### 2.2 Configure Crontab

```bash
# Open crontab editor
crontab -e

# If prompted, choose editor (nano is easiest: option 1)
```

**Add these lines to crontab:**

```bash
# NBA Betting System - Automated Daily Pipeline
# Times are in server timezone (likely UTC - adjust accordingly)

# Set PATH to include Python
PATH=/usr/local/bin:/usr/bin:/bin

# Daily betting pipeline at 4 PM ET (9 PM UTC)
# Adjust time based on your server timezone
0 21 * * * /root/nbamodels/scripts/cron_betting.sh >> /root/nbamodels/logs/cron_main.log 2>&1

# Daily Discord report at 11 PM ET (4 AM UTC next day)
0 4 * * * cd /root/nbamodels && /root/nbamodels/venv/bin/python scripts/send_daily_report.py >> /root/nbamodels/logs/daily_report.log 2>&1
```

**Important Notes**:
- Replace `/root/nbamodels` with your actual project path
- Adjust times based on your droplet's timezone
- Most droplets use UTC by default

**Save and exit**: `Ctrl+X`, `Y`, `Enter`

### 2.3 Check Droplet Timezone

```bash
# Check current timezone
timedatectl

# If it's UTC and you want ET:
# 4 PM ET = 9 PM UTC (during EST)
# 4 PM ET = 8 PM UTC (during EDT)

# Example:
# 4 PM ET (EST) = 21:00 UTC = cron: "0 21 * * *"
# 11 PM ET (EST) = 4 AM UTC next day = cron: "0 4 * * *"
```

### 2.4 Verify Cron Jobs Are Scheduled

```bash
# List active cron jobs
crontab -l

# Should see your two entries:
# 0 21 * * * /root/nbamodels/scripts/cron_betting.sh ...
# 0 4 * * * cd /root/nbamodels && ...

# Check cron service is running
systemctl status cron

# If not running:
# sudo systemctl start cron
# sudo systemctl enable cron
```

---

## 🧪 Step 3: Test Cron Jobs Manually

### 3.1 Test Betting Pipeline

```bash
# Run the script manually
cd ~/nbamodels
./scripts/cron_betting.sh

# Check output
tail -50 logs/cron_betting.log

# Should see:
# - Pipeline started timestamp
# - Strategies loaded
# - Pipeline completed timestamp
```

### 3.2 Test Daily Report

```bash
# Run daily report manually
cd ~/nbamodels
source venv/bin/activate
python scripts/send_daily_report.py

# Check for Discord notification
# You should receive a Discord message with performance summary
```

### 3.3 Test with Near-Future Cron Time

```bash
# Temporarily modify cron to run in 2 minutes
crontab -e

# Add test entry (adjust HH:MM to 2 minutes from now):
# Example: if it's 20:45 UTC, set to 20:47
47 20 * * * /root/nbamodels/scripts/cron_betting.sh >> /root/nbamodels/logs/cron_test.log 2>&1

# Save and wait 2 minutes
# Check if it ran:
cat logs/cron_test.log

# If successful, remove test entry:
crontab -e
# Delete the test line
```

---

## 📊 Step 4: Set Up Monitoring

### 4.1 Create Log Monitoring Script

```bash
# Create a monitoring script
nano scripts/check_cron_health.sh
```

```bash
#!/bin/bash
# Check if cron jobs are running successfully

echo "=== NBA Betting System Health Check ==="
echo "Last betting pipeline run:"
tail -1 logs/cron_betting.log
echo ""
echo "Last daily report run:"
tail -1 logs/daily_report.log
echo ""
echo "Recent errors (if any):"
grep -i "error" logs/cron_betting.log | tail -5
```

```bash
# Make it executable
chmod +x scripts/check_cron_health.sh

# Run it
./scripts/check_cron_health.sh
```

### 4.2 Set Up Daily Health Check (Optional)

```bash
# Add to crontab for daily health email
crontab -e

# Add this line (runs at 8 AM ET / 1 PM UTC):
0 13 * * * /root/nbamodels/scripts/check_cron_health.sh | mail -s "NBA Betting Health" your-email@example.com
```

### 4.3 Monitor Logs Remotely

```bash
# SSH into droplet and tail logs
ssh root@your-droplet-ip
cd ~/nbamodels
tail -f logs/cron_betting.log

# Or use a single command:
ssh root@your-droplet-ip "tail -f ~/nbamodels/logs/cron_betting.log"
```

---

## 🔍 Step 5: Verify Everything Works

### 5.1 Daily Checklist (First Week)

```bash
# SSH into droplet
ssh root@your-droplet-ip

# 1. Check if cron ran today
ls -lh logs/cron_betting.log
# Should have today's timestamp

# 2. Check for errors
grep -i "error" logs/cron_betting.log | tail -10

# 3. Check bet history
cd ~/nbamodels
source venv/bin/activate
python -c "from src.bet_tracker import get_bet_history; print(get_bet_history().tail())"

# 4. Check Discord - should have received notifications
```

### 5.2 Common Issues & Fixes

**Issue: Cron job not running**
```bash
# Check cron service
systemctl status cron

# Restart if needed
sudo systemctl restart cron

# Check cron logs
grep CRON /var/log/syslog | tail -20
```

**Issue: Python not found**
```bash
# Add shebang to scripts or use full path
crontab -e

# Change:
# 0 21 * * * /root/nbamodels/scripts/cron_betting.sh
# To:
# 0 21 * * * cd /root/nbamodels && /bin/bash scripts/cron_betting.sh
```

**Issue: Virtual environment not activated**
```bash
# Cron scripts should have this:
source /root/nbamodels/venv/bin/activate

# Or use absolute path to Python:
/root/nbamodels/venv/bin/python scripts/daily_multi_strategy_pipeline.py
```

**Issue: No Discord notifications**
```bash
# Test Discord webhook manually
curl -H "Content-Type: application/json" \
  -d '{"content":"Test from droplet"}' \
  "YOUR_DISCORD_WEBHOOK_URL"

# Should see message in Discord
```

---

## 🔐 Step 6: Security Best Practices

### 6.1 Protect Sensitive Files

```bash
# Secure .env file
chmod 600 .env
chown root:root .env  # or your user

# Secure bet database
chmod 600 data/bets/bets.db

# Check permissions
ls -l .env data/bets/bets.db
# Should show: -rw------- (600)
```

### 6.2 Set Up Automatic Backups

```bash
# Create backup script
nano scripts/backup_data.sh
```

```bash
#!/bin/bash
# Backup betting database and logs

BACKUP_DIR="/root/nbamodels_backups"
DATE=$(date +%Y%m%d)

mkdir -p $BACKUP_DIR

# Backup database
cp /root/nbamodels/data/bets/bets.db $BACKUP_DIR/bets_$DATE.db

# Backup logs
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz /root/nbamodels/logs/

# Keep only last 30 days
find $BACKUP_DIR -name "*.db" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $DATE"
```

```bash
# Make executable
chmod +x scripts/backup_data.sh

# Add to crontab (daily at 2 AM UTC)
crontab -e
# Add:
0 2 * * * /root/nbamodels/scripts/backup_data.sh >> /root/nbamodels/logs/backup.log 2>&1
```

---

## 📱 Step 7: Remote Monitoring Setup

### 7.1 View Dashboard Remotely

```bash
# Install screen (to keep Streamlit running)
sudo apt install screen -y

# Start new screen session
screen -S dashboard

# Run dashboard
cd ~/nbamodels
source venv/bin/activate
streamlit run dashboard/analytics_dashboard.py --server.port 8501 --server.address 0.0.0.0

# Detach from screen: Ctrl+A then D

# Access dashboard at:
# http://your-droplet-ip:8501

# Note: You may need to open port 8501 in firewall:
sudo ufw allow 8501
```

### 7.2 Set Up SSH Key Access (Recommended)

```bash
# On your local machine:
ssh-keygen -t rsa -b 4096 -C "your-email@example.com"

# Copy to droplet:
ssh-copy-id root@your-droplet-ip

# Now you can SSH without password:
ssh root@your-droplet-ip
```

---

## ✅ Final Verification Checklist

After setup, verify:

- [ ] **Cron jobs scheduled**: `crontab -l` shows both jobs
- [ ] **Scripts executable**: `ls -l scripts/cron_*.sh` shows `x` permission
- [ ] **Virtual environment works**: `source venv/bin/activate && python --version`
- [ ] **Environment variables set**: `cat .env` shows API keys
- [ ] **Logs directory exists**: `ls -l logs/`
- [ ] **Manual test successful**: `./scripts/cron_betting.sh` runs without errors
- [ ] **Discord webhook works**: Received test notification
- [ ] **Database accessible**: `ls -l data/bets/bets.db` exists
- [ ] **Timezone correct**: `timedatectl` shows expected timezone
- [ ] **Cron service running**: `systemctl status cron` shows active

---

## 🚀 Quick Reference Commands

```bash
# SSH into droplet
ssh root@your-droplet-ip

# Check cron jobs
crontab -l

# View real-time logs
tail -f logs/cron_betting.log

# Manual pipeline run
cd ~/nbamodels && ./scripts/cron_betting.sh

# Check system health
./scripts/check_cron_health.sh

# View recent bets
source venv/bin/activate && python -c "from src.bet_tracker import get_bet_history; print(get_bet_history().tail(10))"

# Check if cron ran today
ls -lh logs/cron_betting.log

# Search for errors
grep -i error logs/cron_betting.log | tail -20

# Restart cron service
sudo systemctl restart cron
```

---

## 📞 Troubleshooting

**Cron not running at scheduled time?**
```bash
# Check server time
date

# Check timezone
timedatectl

# Check cron logs
grep CRON /var/log/syslog | tail -50

# Verify cron service
systemctl status cron
```

**Script fails in cron but works manually?**
```bash
# Add debugging to cron_betting.sh
nano scripts/cron_betting.sh

# At the top, add:
set -x  # Print each command
env > /tmp/cron_env.txt  # Capture environment

# Check environment differences:
cat /tmp/cron_env.txt
```

**Need to test specific time?**
```bash
# Use `at` command for one-time scheduled task
echo "./scripts/cron_betting.sh" | at now + 2 minutes

# Check scheduled jobs
atq

# View output
cat /var/mail/root  # or check logs
```

---

## 🎉 You're All Set!

Your droplet is now configured to:
- ✅ Run betting pipeline daily at 4 PM ET
- ✅ Send Discord reports at 11 PM ET
- ✅ Log all activity
- ✅ Backup data automatically

Monitor the first few runs and adjust as needed!
