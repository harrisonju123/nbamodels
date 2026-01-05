# DigitalOcean Deployment - Quick Reference

Complete deployment in 20 minutes.

---

## 1️⃣ Create Droplet (5 min)

### Via DigitalOcean Dashboard:

1. **Sign up:** [digitalocean.com](https://digitalocean.com)
   - New accounts get **$200 credit** (8 months free!)

2. **Create → Droplets → Create Droplet**

3. **Choose Configuration:**
   ```
   Region:          New York 1 or San Francisco 3
   OS:              Ubuntu 22.04 LTS (x64)
   Droplet Type:    Basic
   CPU:             Regular
   Size:            4 GB / 2 vCPU - $24/mo
   ```

4. **Authentication:**
   - Choose "SSH Key"
   - Add your public key: `cat ~/.ssh/id_rsa.pub`
   - Or create new key: `ssh-keygen -t rsa -b 4096`

5. **Hostname:** `nba-betting-system`

6. **Create Droplet** → Wait ~60 seconds

7. **Copy IP address** (shown after creation)

---

## 2️⃣ Initial Connection (2 min)

```bash
# Test SSH connection
ssh root@YOUR_DROPLET_IP

# If first time, accept fingerprint
# You should see Ubuntu welcome message
```

---

## 3️⃣ Deploy System (10 min)

### On the Droplet:

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/nbamodels.git
cd nbamodels

# Run automated setup
chmod +x deploy/setup.sh
./deploy/setup.sh
```

This script will:
- ✅ Install Python 3.12
- ✅ Install nginx, git, dependencies
- ✅ Create virtual environment
- ✅ Install all Python packages
- ✅ Configure firewall
- ✅ Install systemd services

---

## 4️⃣ Configure Environment (2 min)

### Option A: Copy from local machine

```bash
# From YOUR LOCAL machine (not server):
scp .env root@YOUR_DROPLET_IP:/root/nbamodels/.env
```

### Option B: Create on server

```bash
# On the Droplet:
cd /root/nbamodels
nano .env
```

Add:
```bash
ODDS_API_KEY=your_key_here
BALLDONTLIE_API_KEY=your_key_here
DASHBOARD_API_KEY=some_secret_key
```

---

## 5️⃣ Start Services (2 min)

```bash
# Install cron jobs
crontab /root/nbamodels/deploy/crontab.txt

# Verify cron installation
crontab -l

# Enable services to start on boot
systemctl enable nba-dashboard nba-api

# Start services now
systemctl start nba-dashboard nba-api

# Check status
systemctl status nba-dashboard
systemctl status nba-api
```

---

## 6️⃣ Verify Deployment (2 min)

### Check Services:

```bash
# Dashboard running?
curl http://localhost:8501

# API running?
curl http://localhost:8000/health

# View logs
journalctl -u nba-dashboard -f
```

### Access from Browser:

- **Dashboard:** `http://YOUR_DROPLET_IP:8501`
- **API Docs:** `http://YOUR_DROPLET_IP:8000/docs`

---

## 🎉 Done!

Your system is now:
- ✅ Running 24/7 on DigitalOcean
- ✅ Dashboard accessible at port 8501
- ✅ API accessible at port 8000
- ✅ 13 cron jobs collecting data automatically
- ✅ Firewall configured and active
- ✅ Services auto-restart on crash

---

## 📊 DigitalOcean Features You Get

### Included in $24/mo:
- ✅ **Monitoring** - Free built-in graphs (CPU, RAM, disk, bandwidth)
- ✅ **Alerting** - Email alerts when resources hit thresholds
- ✅ **Backups** - Add $4.80/mo for automated weekly backups
- ✅ **Snapshots** - Free manual snapshots anytime
- ✅ **Firewall** - Cloud firewall (in addition to ufw on server)
- ✅ **Networking** - 4TB outbound bandwidth included

### How to Access:
- **Dashboard:** Click your Droplet → Graphs tab
- **Monitoring:** Click Droplet → Monitoring tab
- **Backups:** Click Droplet → Backups tab (enable for $4.80/mo)
- **Console:** Click Droplet → Access → Launch Droplet Console

---

## 🔄 Updating Your Code

When you make changes:

```bash
# SSH into Droplet
ssh root@YOUR_DROPLET_IP

# Pull latest code
cd /root/nbamodels
git pull

# Restart services
systemctl restart nba-dashboard nba-api

# Check logs
journalctl -u nba-dashboard -f
```

---

## 🔒 Optional: Set Up Cloud Firewall

DigitalOcean has a cloud firewall (free) in addition to ufw:

1. **Go to:** Networking → Firewalls → Create Firewall
2. **Inbound Rules:**
   ```
   SSH        TCP   22     All IPv4, All IPv6
   HTTP       TCP   80     All IPv4, All IPv6
   HTTPS      TCP   443    All IPv4, All IPv6
   Custom     TCP   8501   All IPv4, All IPv6  (Dashboard)
   Custom     TCP   8000   All IPv4, All IPv6  (API)
   ```
3. **Apply to:** Select your Droplet

---

## 📈 Enable Monitoring Alerts

Set up alerts for high resource usage:

1. **Go to:** Your Droplet → Monitoring → Alerts
2. **Add Alert:**
   - CPU > 80% for 5 minutes → Email
   - Disk > 80% → Email
   - Memory > 90% for 5 minutes → Email

---

## 💾 Enable Backups (Recommended)

Cost: **$4.80/mo** (20% of Droplet cost)

Benefits:
- Automated weekly backups
- Keep 4 most recent backups
- Restore entire Droplet with one click

**Enable:**
1. Go to your Droplet → Backups
2. Click "Enable Backups"
3. Choose backup window (e.g., Sunday 2-6 AM)

---

## 🆘 Troubleshooting

### Can't SSH?
```bash
# Check if Droplet is on
# Go to DigitalOcean dashboard → Power should be "On"

# Try password authentication (if SSH key fails)
# Click "Access" → "Reset Root Password"
# Check email for temporary password
ssh root@YOUR_IP
```

### Services won't start?
```bash
# Check what went wrong
journalctl -u nba-dashboard -n 50

# Try running manually
cd /root/nbamodels
source .venv/bin/activate
streamlit run analytics_dashboard.py
```

### Firewall blocking access?
```bash
# Check firewall status
ufw status

# Temporarily disable for testing
ufw disable

# Re-enable after
ufw enable
```

---

## 🎁 DigitalOcean Credits

### New User Credits:
- **$200 for 60 days** - Sign up bonus
- Usually get this automatically on new accounts

### GitHub Student Pack:
- **$200 credit** if you're a student
- [education.github.com/pack](https://education.github.com/pack)

### Referral Credits:
- Share your referral link, get $25 per referral
- Find your link in DigitalOcean dashboard

---

## 💡 DigitalOcean Advantages

vs Hetzner:
- ✅ Better documentation
- ✅ Larger community (more Stack Overflow answers)
- ✅ Built-in monitoring & alerting
- ✅ Better UI/UX
- ✅ More tutorials available
- ⚠️ 3x more expensive ($24 vs $7)

vs Railway:
- ✅ Keep SQLite (no DB migration)
- ✅ Native cron (no workarounds)
- ✅ Full SSH access
- ✅ More control
- ⚠️ Need to manage server yourself

---

**Total Setup Time:** ~20 minutes
**Monthly Cost:** $24 ($0 for first 8 months with $200 credit)
**Difficulty:** Easy with this guide

Ready to deploy? Follow steps 1-6 above! 🚀
