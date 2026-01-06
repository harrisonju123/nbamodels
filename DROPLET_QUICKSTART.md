# Droplet Quick Start - 3 Commands

**Get your betting system running on DigitalOcean in 3 minutes**

---

## 🚀 Super Quick Setup (3 Commands)

```bash
# 1. Clone and enter directory
git clone https://github.com/yourusername/nbamodels.git
cd nbamodels

# 2. Run automated setup
chmod +x scripts/setup_droplet.sh
./scripts/setup_droplet.sh

# 3. Install cron jobs
chmod +x scripts/install_cron.sh
./scripts/install_cron.sh
```

**That's it!** Your system will now run automatically every day.

---

## ⚙️ One-Time Configuration

Before the system runs, add your API keys:

```bash
# Edit environment file
nano .env

# Update these two lines:
ODDS_API_KEY=your_actual_api_key_here
DISCORD_WEBHOOK_URL="your_discord_webhook_url_here"

# Save: Ctrl+X, Y, Enter
```

---

## ✅ Verify Everything Works

```bash
# Test the pipeline manually
./scripts/cron_betting.sh

# Check the log
tail -50 logs/cron_betting.log

# Should see:
# ✓ Enabled ArbitrageStrategy
# ✓ Enabled B2BRestStrategy
# ✓ Pipeline complete!
```

---

## 📅 What Happens Next

Your droplet will now automatically:

| Time | Task |
|------|------|
| **4:00 PM ET daily** | Run betting pipeline, analyze games, log bets |
| **11:00 PM ET daily** | Send Discord performance summary |

---

## 🔍 Daily Monitoring

```bash
# Check if cron ran today
ls -lh logs/cron_betting.log

# View recent logs
tail -100 logs/cron_betting.log

# Check for errors
grep -i error logs/cron_betting.log

# View today's bets
source venv/bin/activate
python -c "from src.bet_tracker import get_bet_history; print(get_bet_history().tail())"
```

---

## 🛠️ Common Commands

```bash
# View scheduled cron jobs
crontab -l

# Monitor logs in real-time
tail -f logs/cron_betting.log

# Run pipeline manually
./scripts/cron_betting.sh

# Check system status
./scripts/check_cron_health.sh

# Stop all cron jobs
crontab -r  # WARNING: Removes ALL cron jobs

# Edit cron jobs
crontab -e
```

---

## 📱 Access Dashboard Remotely

```bash
# Install screen
sudo apt install screen -y

# Start dashboard in background
screen -S dashboard
source venv/bin/activate
streamlit run dashboard/analytics_dashboard.py --server.port 8501 --server.address 0.0.0.0

# Detach: Ctrl+A then D

# Open firewall port
sudo ufw allow 8501

# Access at: http://your-droplet-ip:8501
```

---

## 🆘 Troubleshooting

**Cron not running?**
```bash
# Check cron service
systemctl status cron

# Restart if needed
sudo systemctl restart cron

# View cron logs
grep CRON /var/log/syslog | tail -20
```

**Environment issues?**
```bash
# Test environment
source venv/bin/activate
python --version
cat .env

# Reinstall dependencies
pip install -r requirements.txt
```

**Discord not working?**
```bash
# Test webhook
curl -H "Content-Type: application/json" \
  -d '{"content":"Test from droplet"}' \
  "$DISCORD_WEBHOOK_URL"
```

---

## 📚 Full Documentation

- **Complete guide**: `cat DROPLET_DEPLOYMENT.md`
- **Deployment guide**: `cat DEPLOYMENT_GUIDE.md`
- **Strategy details**: `cat config/multi_strategy_config.yaml`

---

## 🎯 Server Timezone Reference

Most DigitalOcean droplets use **UTC** timezone:

| ET Time | UTC Time | Cron Expression |
|---------|----------|-----------------|
| 4 PM ET (winter) | 9 PM UTC | `0 21 * * *` |
| 4 PM ET (summer) | 8 PM UTC | `0 20 * * *` |
| 11 PM ET (winter) | 4 AM UTC | `0 4 * * *` |
| 11 PM ET (summer) | 3 AM UTC | `0 3 * * *` |

*The install script automatically detects and configures this.*

---

## 📞 Quick Help

| Issue | Command |
|-------|---------|
| Check timezone | `timedatectl` |
| View cron jobs | `crontab -l` |
| Test pipeline | `./scripts/cron_betting.sh` |
| Check logs | `tail -f logs/cron_betting.log` |
| Restart cron | `sudo systemctl restart cron` |
| SSH to droplet | `ssh root@your-droplet-ip` |

---

## ✅ Success Checklist

After setup, verify:

- [ ] `crontab -l` shows 2 NBA betting jobs
- [ ] `ls -l .env` shows file exists with API keys
- [ ] `./scripts/cron_betting.sh` runs without errors
- [ ] Discord receives test notification
- [ ] `tail logs/cron_betting.log` shows successful run

**All checked? You're good to go!** 🎉

---

## 🚀 You're Live!

Your droplet will now:
- ✅ Run automatically every day at 4 PM ET
- ✅ Analyze all NBA games with 3 strategies
- ✅ Log bets to database (paper trading)
- ✅ Send Discord updates twice daily

Monitor for 2 weeks, then decide if you want to enable live betting!
