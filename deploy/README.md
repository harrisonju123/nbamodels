# NBA Betting System - Deployment Guide

Deploy your betting system to DigitalOcean Droplet in ~20 minutes.

## 📋 What's Included

```
deploy/
├── README.md               # This file
├── setup.sh                # One-command server setup script
├── nba-dashboard.service   # Systemd service for Streamlit dashboard
├── nba-api.service         # Systemd service for FastAPI backend
└── crontab.txt             # All 13 cron jobs
```

---

## 🚀 Quick Start

### Step 1: Create Droplet (5 minutes)

1. Go to [digitalocean.com](https://digitalocean.com)
2. Sign up for account (use [this referral link](https://m.do.co/c/your-referral) for $200 credit)
3. Create new Droplet:
   - **Size:** Basic - Regular - 4GB RAM / 2 vCPU - $24/month
   - **Datacenter:** New York 1 or San Francisco 3 (close to you)
   - **OS:** Ubuntu 22.04 LTS
   - **Authentication:** SSH Key (add your public key)
   - **Hostname:** nba-betting-system
4. Note the IP address

### Step 2: Initial Server Setup (2 minutes)

```bash
# SSH into your server
ssh root@YOUR_SERVER_IP

# Clone your repository
git clone https://github.com/YOUR_USERNAME/nbamodels.git
cd nbamodels
```

### Step 3: Run Setup Script (10 minutes)

```bash
# Make setup script executable (if not already)
chmod +x deploy/setup.sh

# Run setup
./deploy/setup.sh
```

This will:
- Install Python 3.12, nginx, git
- Create virtual environment
- Install all dependencies from requirements.txt
- Create necessary directories
- Install systemd services
- Configure firewall

### Step 4: Configure Environment (2 minutes)

```bash
# Copy .env file from your local machine
# Run this from your LOCAL machine, not the server:
scp .env root@YOUR_SERVER_IP:/root/nbamodels/.env
```

Or create .env manually on server:
```bash
nano .env
# Add your API keys:
# ODDS_API_KEY=your_key_here
# BALLDONTLIE_API_KEY=your_key_here
```

### Step 5: Install Cron Jobs (1 minute)

```bash
# Install all cron jobs
crontab deploy/crontab.txt

# Verify installation
crontab -l
```

### Step 6: Start Services (1 minute)

```bash
# Enable services to start on boot
systemctl enable nba-dashboard nba-api

# Start services
systemctl start nba-dashboard nba-api

# Check status
systemctl status nba-dashboard
systemctl status nba-api
```

### Step 7: Verify Deployment (2 minutes)

```bash
# Check services are running
curl http://localhost:8501  # Dashboard
curl http://localhost:8000/health  # API

# View service logs
journalctl -u nba-dashboard -f

# Check cron jobs
tail -f logs/pipeline.log
```

---

## 🌐 Access Your System

After deployment, access your services:

- **Dashboard:** `http://YOUR_SERVER_IP:8501`
- **API:** `http://YOUR_SERVER_IP:8000`
- **API Docs:** `http://YOUR_SERVER_IP:8000/docs`

---

## 📊 What Gets Deployed

### Services (Always Running)
- **Streamlit Dashboard** (port 8501) - Real-time betting analytics
- **FastAPI Backend** (port 8000) - REST API for bet tracking

### Cron Jobs (Scheduled)

| Schedule | Script | Purpose |
|----------|--------|---------|
| Hourly | `collect_line_snapshots.py` | Track odds changes |
| Every 15 min | `capture_opening_lines.py` | Capture opening lines |
| Every 15 min | `capture_closing_lines.py` | Capture closing lines |
| Hourly | `collect_news.py` | Scrape NBA news |
| 10 AM ET daily | `collect_referees.py` | Get referee assignments |
| Every 15 min (5-11 PM ET) | `collect_lineups.py` | Track starting lineups |
| 4 PM ET daily | `daily_betting_pipeline.py` | Generate daily picks |
| 6 AM ET daily | `settle_bets.py` | Settle completed bets |
| 6 AM ET daily | `populate_clv_data.py` | Calculate CLV |
| 6:15 AM ET daily | `validate_closing_lines.py` | Validate lines |

---

## 🔄 Updating Your Code

When you make changes locally and want to deploy:

```bash
# SSH into server
ssh root@YOUR_SERVER_IP

# Pull latest changes
cd /root/nbamodels
git pull

# Restart services to pick up changes
systemctl restart nba-dashboard nba-api

# For cron job changes, reinstall crontab
crontab deploy/crontab.txt
```

---

## 🛠️ Useful Commands

### Service Management
```bash
# Start/stop/restart services
systemctl start nba-dashboard
systemctl stop nba-dashboard
systemctl restart nba-dashboard

# View service status
systemctl status nba-dashboard

# View service logs (live)
journalctl -u nba-dashboard -f

# View service logs (last 100 lines)
journalctl -u nba-dashboard -n 100
```

### Cron Job Management
```bash
# View installed cron jobs
crontab -l

# Edit cron jobs
crontab -e

# View cron execution log
grep CRON /var/log/syslog | tail -20

# View specific script logs
tail -f logs/pipeline.log
tail -f logs/settle.log
tail -f logs/clv.log
```

### Database Management
```bash
# Connect to SQLite database
sqlite3 data/bets/bets.db

# Backup database
cp data/bets/bets.db data/bets/bets.db.backup

# Check database size
du -sh data/bets/bets.db
```

### System Monitoring
```bash
# Check disk space
df -h

# Check memory usage
free -h

# Check running processes
ps aux | grep python

# Check active connections
ss -tulpn | grep -E '8501|8000'
```

---

## 🐛 Troubleshooting

### Services won't start

```bash
# Check service logs
journalctl -u nba-dashboard -n 50

# Check if port is already in use
ss -tulpn | grep 8501

# Try running manually to see errors
cd /root/nbamodels
source .venv/bin/activate
streamlit run analytics_dashboard.py
```

### Cron jobs not running

```bash
# Check cron service is running
systemctl status cron

# Check cron logs
grep CRON /var/log/syslog | tail -20

# Run script manually to test
cd /root/nbamodels
source .venv/bin/activate
python scripts/daily_betting_pipeline.py
```

### Out of disk space

```bash
# Check disk usage
df -h

# Clean up old logs
find logs/ -name "*.log" -mtime +7 -delete
find logs/ -name "*.log.old" -delete

# Clear pip cache
.venv/bin/pip cache purge
```

---

## 🔒 Security Recommendations

### Change Default SSH Port (Optional)
```bash
# Edit SSH config
nano /etc/ssh/sshd_config
# Change Port 22 to Port 2222

# Update firewall
ufw allow 2222/tcp
ufw delete allow 22/tcp

# Restart SSH
systemctl restart sshd
```

### Set Up SSL/HTTPS (Optional)
```bash
# Install Certbot
apt install certbot python3-certbot-nginx

# Get SSL certificate (requires domain name)
certbot --nginx -d yourdomain.com
```

### Regular Updates
```bash
# Update system packages weekly
apt update && apt upgrade -y
```

---

## 💰 Cost Estimate

**Monthly Costs:**
- DigitalOcean Droplet (4GB): $24.00
- **Total:** $24/month

**One-time:**
- Domain name (optional): $10-15/year

**Note:** New DigitalOcean accounts get $200 credit (2 months free), and students can get more credits through GitHub Student Developer Pack.

---

## 📈 Next Steps

After deployment:

1. **Monitor first 24 hours** - Check logs, verify cron jobs run
2. **Test a full cycle** - Wait for daily pipeline → bets → settlement
3. **Set up backups** - Consider automated database backups
4. **Add monitoring** - UptimeRobot (free) for service uptime alerts
5. **Optimize** - Tune based on performance metrics

---

## 🆘 Support

If you encounter issues:

1. Check logs: `journalctl -u nba-dashboard -f`
2. Test manually: `python scripts/daily_betting_pipeline.py`
3. Verify environment: `.env` file has all required API keys
4. Check firewall: `ufw status`

---

**Deployment Time:** ~20 minutes
**Monthly Cost:** $7
**Uptime:** 24/7 automated operation
