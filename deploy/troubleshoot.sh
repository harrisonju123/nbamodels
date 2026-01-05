#!/bin/bash
# NBA Betting System - Troubleshooting Script
# Run this on the droplet to diagnose and fix access issues

echo "=== NBA Betting System Troubleshooting ==="
echo

echo "=== 1. Checking Services Status ==="
systemctl status nba-dashboard --no-pager | head -10
systemctl status nba-api --no-pager | head -10
echo

echo "=== 2. Checking if ports are listening ==="
ss -tulpn | grep -E '8501|8000'
echo

echo "=== 3. Current Firewall Status ==="
ufw status verbose
echo

echo "=== 4. Fixing Firewall (if needed) ==="
echo "Opening ports 8501 and 8000..."
ufw allow 8501/tcp
ufw allow 8000/tcp
ufw reload
echo "Firewall updated!"
echo

echo "=== 5. Testing Local Access ==="
echo "Testing dashboard..."
curl -I http://localhost:8501 2>&1 | head -5
echo
echo "Testing API..."
curl http://localhost:8000/health 2>&1
echo

echo "=== 6. Your Droplet IP Address ==="
DROPLET_IP=$(curl -s ifconfig.me)
echo "Droplet IP: $DROPLET_IP"
echo

echo "=== 7. Access URLs ==="
echo "Dashboard: http://$DROPLET_IP:8501"
echo "API:       http://$DROPLET_IP:8000"
echo "API Docs:  http://$DROPLET_IP:8000/docs"
echo

echo "=== 8. Recent Dashboard Logs ==="
journalctl -u nba-dashboard -n 20 --no-pager
echo

echo "=== Troubleshooting Complete ==="
echo
echo "If dashboard still doesn't load:"
echo "1. Check DigitalOcean Cloud Firewall (Networking → Firewalls)"
echo "2. Make sure you're using the correct IP: $DROPLET_IP"
echo "3. Try accessing from a different browser/device"
echo "4. Check logs: journalctl -u nba-dashboard -f"
