#!/bin/bash
# NBA Betting System - Server Setup Script
# Run this on a fresh Ubuntu 22.04 server

set -e  # Exit on error

echo "=== NBA Betting System Deployment ==="
echo "Starting server setup..."
echo

echo "=== Installing system dependencies ==="
apt update && apt upgrade -y
apt install -y python3.12 python3.12-venv python3-pip nginx git ufw curl

echo "=== Setting up Python environment ==="
cd /root/nbamodels
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Creating directories ==="
mkdir -p logs data/bets data/cache data/features data/historical_odds

echo "=== Installing systemd services ==="
cp deploy/nba-dashboard.service /etc/systemd/system/
cp deploy/nba-api.service /etc/systemd/system/
systemctl daemon-reload

echo "=== Setting up firewall ==="
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp     # SSH
ufw allow 80/tcp     # HTTP
ufw allow 443/tcp    # HTTPS
ufw allow 8501/tcp   # Streamlit dashboard
ufw allow 8000/tcp   # FastAPI
ufw --force enable

echo
echo "=== Setup complete! ==="
echo
echo "Next steps:"
echo "1. Copy your .env file: scp .env root@YOUR_IP:/root/nbamodels/.env"
echo "2. Install crontab: crontab /root/nbamodels/deploy/crontab.txt"
echo "3. Start services:"
echo "   systemctl enable nba-dashboard nba-api"
echo "   systemctl start nba-dashboard nba-api"
echo "4. Check status:"
echo "   systemctl status nba-dashboard"
echo "   systemctl status nba-api"
echo
echo "Dashboard will be available at: http://YOUR_IP:8501"
echo "API will be available at: http://YOUR_IP:8000"
echo
