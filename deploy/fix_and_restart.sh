#!/bin/bash
# Fix missing modules and restart services

set -e

echo "=== Fixing Dashboard Crash ==="
echo

echo "1. Pulling latest code with missing modules..."
cd /root/nbamodels
git pull origin main
echo "✓ Code updated"
echo

echo "2. Restarting dashboard service..."
systemctl restart nba-dashboard
echo "✓ Dashboard restarted"
echo

echo "3. Restarting API service..."
systemctl restart nba-api
echo "✓ API restarted"
echo

echo "4. Waiting 5 seconds for services to start..."
sleep 5
echo

echo "5. Checking service status..."
echo
echo "--- Dashboard Status ---"
systemctl status nba-dashboard --no-pager | head -15
echo
echo "--- API Status ---"
systemctl status nba-api --no-pager | head -15
echo

echo "6. Getting your access URLs..."
DROPLET_IP=$(curl -s ifconfig.me)
echo
echo "=== Access Your Dashboard ==="
echo "Dashboard: http://$DROPLET_IP:8501"
echo "API:       http://$DROPLET_IP:8000"
echo "API Docs:  http://$DROPLET_IP:8000/docs"
echo

echo "7. Testing local access..."
echo
echo "Testing dashboard..."
curl -I http://localhost:8501 2>&1 | head -3
echo
echo "Testing API..."
curl http://localhost:8000/health 2>&1
echo
echo

echo "=== Fix Complete ==="
echo
echo "If the dashboard still doesn't load:"
echo "1. Check the logs: journalctl -u nba-dashboard -f"
echo "2. Make sure firewall allows port 8501: ufw status"
echo "3. Check DigitalOcean Cloud Firewall settings"
