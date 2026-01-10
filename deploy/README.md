# Droplet Deployment Guide

Simple guide for deploying the NBA betting system to a DigitalOcean droplet.

---

## Quick Setup (15 Minutes)

### 1. Create Droplet

**Recommended Specs:**
- **OS**: Ubuntu 22.04 LTS
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 80GB SSD
- **Region**: Choose closest to you

### 2. Initial Server Setup

```bash
# SSH into droplet
ssh root@YOUR_DROPLET_IP

# Update system
apt update && apt upgrade -y

# Install Python 3.11+
apt install -y python3.11 python3.11-venv python3-pip git sqlite3

# Create user (optional but recommended)
adduser nba
usermod -aG sudo nba
su - nba
```

### 3. Clone Repository

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/nbamodels.git
cd nbamodels
```

### 4. Setup Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv .venv

# Activate
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 5. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit with your API keys
nano .env
```

**Required Environment Variables:**
```bash
ODDS_API_KEY=your_odds_api_key_here
DISCORD_WEBHOOK_URL=your_discord_webhook_url_here  # Optional
```

Get your free Odds API key at: https://the-odds-api.com/

### 6. Download Models

```bash
# Models are required for predictions
# If you have them locally, upload via scp:
# scp models/*.pkl nba@YOUR_DROPLET_IP:~/nbamodels/models/

# Verify models exist
ls -lh models/
# Should see: spread_model_calibrated.pkl, player_props_model.pkl
```

### 7. Initialize Database

```bash
# Create bets database
python scripts/init_db.py

# Verify
sqlite3 data/bets/bets.db "SELECT COUNT(*) FROM bets"
# Should output: 0
```

### 8. Make Scripts Executable

```bash
chmod +x ops/dashboard.sh
chmod +x ops/health_check.sh
```

### 9. Test System

```bash
# Run health check
./ops/health_check.sh
# Should show: ✓ ALL SYSTEMS OPERATIONAL

# Test dashboard
./ops/dashboard.sh
# Should display interactive dashboard

# Test bet generation (dry run)
python scripts/daily_betting_pipeline.py --strategy baseline --dry-run
# Should generate predictions without saving
```

### 10. Install Crontab

```bash
# Edit crontab with your paths
nano deploy/crontab_production.txt

# Update these lines:
# PROJECT=/home/nba/nbamodels
# MAILTO=your-email@example.com

# Install crontab
crontab deploy/crontab_production.txt

# Verify installation
crontab -l
```

---

## Daily Operations (5 Minutes)

### Morning Check (9:00 AM ET)

```bash
cd ~/nbamodels
./ops/dashboard.sh

# Review:
# - System status: OPERATIONAL ✓
# - Win Rate: >54% ✓
# - ROI: >5% ✓
# - Today's bets
# - Alerts: None ✓

# Done. System runs itself.
```

---

## Production Schedule

**Automated by crontab:**

| Time | Task | Description |
|------|------|-------------|
| 6:00 AM | Settlement | Process overnight game results |
| 6:15 AM | CLV Calculation | Calculate closing line value |
| 9:00 AM | Health Check | System validation |
| 3:00 PM | Generate Bets | Optimal timing window (1-4hr before games) |
| 5:00 PM | Generate Bets | Second betting window |
| Every 15 min | Line Capture | Track opening/closing lines for CLV |
| Hourly (9AM-11PM) | News Collection | Gather injury/lineup news |
| Daily 10 AM | Referees | Collect referee assignments |
| Weekly (Sunday) | Reports | Performance and CLV analysis |

---

## Monitoring

### Log Files

```bash
# View recent pipeline activity
tail -50 logs/pipeline.log

# View settlement results
tail -50 logs/settlement.log

# Check for errors
grep ERROR logs/*.log

# Check alerts
cat logs/alerts.log
```

### Quick Metrics

```bash
# Total P&L
sqlite3 data/bets/bets.db "SELECT SUM(profit) FROM bets WHERE outcome IS NOT NULL"

# Win rate
sqlite3 data/bets/bets.db "
SELECT PRINTF('%.1f%%',
  CAST(SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100
) FROM bets WHERE outcome IS NOT NULL"

# Bets today
sqlite3 data/bets/bets.db "SELECT COUNT(*) FROM bets WHERE DATE(logged_at) = DATE('now')"
```

---

## Troubleshooting

### Dashboard won't start

```bash
# Check permissions
chmod +x ops/dashboard.sh

# Run health check instead
./ops/health_check.sh
```

### No bets generated

**Common causes:**
1. No games today (check NBA schedule)
2. All games filtered by edge threshold
3. CLV data not populated yet (wait 1-2 days)

**Debug:**
```bash
python scripts/daily_betting_pipeline.py --strategy baseline --dry-run 2>&1 | grep -A5 "filtered"
```

### API rate limit exceeded

```bash
# Check remaining requests
tail -100 logs/*.log | grep "API requests remaining"

# Odds API free tier: 500 requests/month
# Consider upgrading if hitting limits
```

### Database locked

```bash
# Check for running processes
ps aux | grep python

# Kill if stuck
pkill -f daily_betting_pipeline

# Restart cron
crontab -l | crontab -
```

---

## Maintenance

### Weekly Tasks

```bash
# Review performance report (automated Sunday 10 AM)
cat logs/reports.log

# Review CLV report (automated Sunday 10:30 AM)
python scripts/generate_clv_report.py
```

### Monthly Tasks

```bash
# Model retraining (automated 1st of month, 2 AM)
# Backup (automated 1st of month, 3 AM)

# Manual backup:
tar -czf backup-$(date +%Y%m%d).tar.gz models/ data/bets/ config/
```

### Log Rotation

Automated by crontab:
- Logs >7 days: Compressed (gzip)
- Logs >30 days: Deleted

Manual cleanup:
```bash
find logs -name "*.log" -mtime +7 -exec gzip {} \;
find logs -name "*.log.gz" -mtime +30 -delete
```

---

## Security

### Firewall Setup

```bash
# Enable firewall
ufw allow OpenSSH
ufw enable

# Only allow SSH (betting system doesn't need external access)
ufw status
```

### API Key Security

- Never commit `.env` to git
- Store API keys in environment variables only
- Rotate keys periodically
- Use Discord webhooks (optional) for notifications without exposing keys

---

## Support

**Quick Reference:**
- **Operations Guide**: `ops/PLAYBOOK.md`
- **Dashboard**: `./ops/dashboard.sh`
- **Health Check**: `./ops/health_check.sh`

**Full Documentation:**
- **Main README**: `README.md`
- **Scripts Guide**: `scripts/README.md`
- **Bet Timing Analysis**: `docs/BET_TIMING_ANALYSIS.md`

**Logs:**
- **Pipeline**: `logs/pipeline.log`
- **Settlement**: `logs/settlement.log`
- **Alerts**: `logs/alerts.log`
- **Health**: `logs/health.log`

---

## Success Criteria

**After deployment, verify:**

- [ ] Health check passes (exit code 0)
- [ ] Dashboard shows system status
- [ ] Crontab installed (`crontab -l`)
- [ ] Dry run generates predictions
- [ ] Database accessible
- [ ] Models loaded successfully
- [ ] API key working (check logs)
- [ ] Logs directory exists and writable

**Daily verification:**

- [ ] Bets generated at 3 PM and 5 PM
- [ ] Games settled at 6 AM
- [ ] Health check runs at 9 AM
- [ ] No critical errors in logs
- [ ] Performance metrics stable (>54% win rate, >5% ROI)

---

**Status**: Production Ready

**Philosophy**: "Simple systems work. Complex systems fail."

This is a tight, institutional-grade operation. Daily interaction: 5 minutes. Everything else is automated, monitored, and documented.
