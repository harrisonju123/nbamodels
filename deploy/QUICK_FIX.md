# Quick Fix - Dashboard Won't Load

## Problem
Services are running but can't access dashboard in browser.

## Most Common Causes
1. **Missing Python modules** (ModuleNotFoundError)
2. **DigitalOcean Cloud Firewall or UFW blocking ports**

---

## Fix 0: Missing Modules (RUN THIS FIRST!)

If you see `ModuleNotFoundError` in the logs, the code is missing required files.

```bash
# SSH into droplet
ssh root@YOUR_DROPLET_IP

# Pull latest code and restart
cd /root/nbamodels
git pull
chmod +x deploy/fix_and_restart.sh
./deploy/fix_and_restart.sh
```

This fixes the missing `dashboard_enhancements` and `live_bet_settlement` modules.

---

---

## Fix 1: Run Troubleshooting Script (RECOMMENDED)

```bash
# SSH into your droplet
ssh root@YOUR_DROPLET_IP

# Pull latest changes
cd /root/nbamodels
git pull

# Run troubleshooting script
chmod +x deploy/troubleshoot.sh
./deploy/troubleshoot.sh
```

This will automatically:
- Check service status
- Fix firewall rules
- Test local access
- Show you the correct URLs to use

---

## Fix 2: Manual Firewall Fix

If script doesn't work, manually fix firewall:

```bash
# Check firewall status
sudo ufw status

# If ports aren't allowed, add them:
sudo ufw allow 8501/tcp
sudo ufw allow 8000/tcp
sudo ufw reload

# Verify
sudo ufw status
```

---

## Fix 3: DigitalOcean Cloud Firewall

DigitalOcean has TWO firewalls:
1. **UFW** (on the server) - Fixed by script above
2. **Cloud Firewall** (in DO dashboard) - Need to check manually

### Check Cloud Firewall:

1. Go to DigitalOcean dashboard
2. Click **Networking** → **Firewalls**
3. If you see a firewall applied to your droplet:
   - Click on it
   - Make sure these inbound rules exist:
     - SSH: TCP port 22
     - HTTP: TCP port 80
     - HTTPS: TCP port 443
     - Custom: TCP port 8501 (Dashboard)
     - Custom: TCP port 8000 (API)
   - If missing, add them with source "All IPv4, All IPv6"

4. If no firewall exists, you're good (Cloud Firewall is optional)

---

## Fix 4: Verify Correct IP Address

```bash
# Get your droplet IP
curl ifconfig.me
```

Make sure you're using this IP in your browser:
- `http://YOUR_IP:8501` (Dashboard)
- `http://YOUR_IP:8000` (API)

**NOT** `http://0.0.0.0:8501` or `http://localhost:8501`

---

## Still Not Working?

Run these diagnostic commands:

```bash
# Check if service is actually listening
ss -tulpn | grep 8501

# Check recent errors
journalctl -u nba-dashboard -n 50

# Test local access
curl http://localhost:8501

# Check all firewall rules
sudo ufw status verbose
sudo iptables -L -n | grep 8501
```

Then share the output for further troubleshooting.
