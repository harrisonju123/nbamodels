#!/bin/bash
# Copy trained models to the droplet

if [ -z "$1" ]; then
    echo "Usage: ./deploy/copy_models.sh YOUR_DROPLET_IP"
    echo ""
    echo "Example: ./deploy/copy_models.sh 164.90.123.456"
    exit 1
fi

DROPLET_IP=$1

echo "=== Copying Models to Droplet ==="
echo "Target: root@$DROPLET_IP"
echo

# Check if models directory exists locally
if [ ! -d "models" ]; then
    echo "Error: models/ directory not found"
    echo "Make sure you're running this from the nbamodels project root"
    exit 1
fi

echo "Models directory size: $(du -sh models/ | cut -f1)"
echo

# Create models directory on droplet
echo "1. Creating models directory on droplet..."
ssh root@$DROPLET_IP "mkdir -p /root/nbamodels/models"
echo "✓ Directory created"
echo

# Copy all model files (excluding backups to save time)
echo "2. Copying model files..."
echo "This may take 1-2 minutes for 15MB..."
rsync -avz --progress \
    --exclude='backups/' \
    --exclude='*.pyc' \
    --exclude='__pycache__/' \
    models/ root@$DROPLET_IP:/root/nbamodels/models/

echo
echo "✓ Models copied successfully!"
echo

# Verify on droplet
echo "3. Verifying models on droplet..."
ssh root@$DROPLET_IP "ls -lh /root/nbamodels/models/*.pkl 2>/dev/null | wc -l | xargs echo 'Model files copied:'"
echo

echo "=== Copy Complete! ==="
echo
echo "Models are now available at: /root/nbamodels/models/"
echo
echo "Next steps:"
echo "1. Run the daily betting pipeline to test:"
echo "   ssh root@$DROPLET_IP"
echo "   cd /root/nbamodels"
echo "   source .venv/bin/activate"
echo "   python scripts/daily_betting_pipeline.py"
echo
