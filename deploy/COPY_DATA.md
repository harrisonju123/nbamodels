# Copying Models and Data to Droplet

## Quick Method - Copy Everything

From your **LOCAL machine** (not the droplet):

```bash
cd /path/to/nbamodels

# Copy models
chmod +x deploy/copy_models.sh
./deploy/copy_models.sh YOUR_DROPLET_IP

# Copy betting history database (optional)
scp data/bets/bets.db root@YOUR_DROPLET_IP:/root/nbamodels/data/bets/
```

---

## What Needs to Be Copied

### 1. **Models** (Required) - 15MB
```bash
./deploy/copy_models.sh YOUR_DROPLET_IP
```

These are your trained ML models. Without them, the betting pipeline can't make predictions.

**Essential model:**
- `models/spread_model.pkl` - Main prediction model

**Other models you have:**
- `models/totals_model.pkl`
- `models/point_spread_model.pkl`
- `models/advanced_ensemble.pkl`
- And 10+ more variants

The script copies all of them (excluding backups).

---

### 2. **Betting Database** (Optional) - Preserves History

If you want to keep your betting history and performance metrics:

```bash
# Copy main betting database
scp data/bets/bets.db root@YOUR_DROPLET_IP:/root/nbamodels/data/bets/

# Copy live betting database (if you use it)
scp data/bets/live_betting.db root@YOUR_DROPLET_IP:/root/nbamodels/data/bets/
```

**Why copy?**
- ✅ Preserve bet history
- ✅ Keep CLV calculations
- ✅ Maintain performance analytics
- ✅ Continue bankroll tracking

**Why start fresh?**
- Start with clean production data
- Backtest data might not be real bets
- Easier to debug

---

### 3. **Historical Odds Data** (Optional) - ~500MB

Only needed if you want to retrain models or run backtests on the server:

```bash
scp data/raw/historical_odds.csv root@YOUR_DROPLET_IP:/root/nbamodels/data/raw/
```

**Usually NOT needed for production** - the cron jobs will collect new data automatically.

---

## Step-by-Step Guide

### From Your Local Machine

```bash
# 1. Navigate to project
cd ~/PycharmProjects/nbamodels

# 2. Copy models (required)
./deploy/copy_models.sh YOUR_DROPLET_IP

# 3. Copy betting database (optional)
scp data/bets/bets.db root@YOUR_DROPLET_IP:/root/nbamodels/data/bets/

# 4. Verify on droplet
ssh root@YOUR_DROPLET_IP "ls -lh /root/nbamodels/models/*.pkl"
```

---

## Manual Method (If Script Doesn't Work)

### Copy Models Manually

```bash
# Using scp
scp -r models/*.pkl root@YOUR_DROPLET_IP:/root/nbamodels/models/

# Or using rsync (better for large transfers)
rsync -avz models/*.pkl root@YOUR_DROPLET_IP:/root/nbamodels/models/
```

### Copy Database Manually

```bash
scp data/bets/bets.db root@YOUR_DROPLET_IP:/root/nbamodels/data/bets/
```

---

## Verify Everything Copied

SSH into your droplet and check:

```bash
ssh root@YOUR_DROPLET_IP

# Check models
ls -lh /root/nbamodels/models/*.pkl
# Should show: spread_model.pkl, totals_model.pkl, etc.

# Check database (if you copied it)
ls -lh /root/nbamodels/data/bets/bets.db

# Check model sizes
du -sh /root/nbamodels/models/
# Should show: ~15M
```

---

## Test the Models Work

After copying, test that the pipeline can load the models:

```bash
ssh root@YOUR_DROPLET_IP
cd /root/nbamodels
source .venv/bin/activate

# Test loading the model
python -c "
import pickle
from pathlib import Path

model_path = Path('models/spread_model.pkl')
if model_path.exists():
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f'✓ Model loaded successfully: {type(model).__name__}')
else:
    print('✗ Model file not found')
"
```

Expected output:
```
✓ Model loaded successfully: LGBMClassifier
```

---

## Common Issues

### "Permission denied (publickey)"

Your SSH key isn't set up. Fix:
```bash
ssh-copy-id root@YOUR_DROPLET_IP
```

### "No such file or directory"

Create the directory first:
```bash
ssh root@YOUR_DROPLET_IP "mkdir -p /root/nbamodels/models /root/nbamodels/data/bets"
```

### Transfer is very slow

Use `rsync` instead of `scp` - it's faster for multiple files:
```bash
rsync -avz --progress models/*.pkl root@YOUR_DROPLET_IP:/root/nbamodels/models/
```

### Model file corrupted after transfer

Check file sizes match:
```bash
# Local
ls -lh models/spread_model.pkl

# Remote
ssh root@YOUR_DROPLET_IP "ls -lh /root/nbamodels/models/spread_model.pkl"
```

If sizes don't match, re-copy:
```bash
scp models/spread_model.pkl root@YOUR_DROPLET_IP:/root/nbamodels/models/
```

---

## What Happens After Copying

Once models are on the droplet:

1. **Cron jobs can run** - Daily pipeline will use the models
2. **Dashboard can show predictions** - Real-time betting picks
3. **API can serve predictions** - Programmatic access
4. **System is fully operational** - Ready for production use

---

## Do You Need to Copy Models Again?

**When models change:**
- After retraining locally
- After tuning hyperparameters
- After model improvements

**Run the copy script again:**
```bash
./deploy/copy_models.sh YOUR_DROPLET_IP
```

It will overwrite the old models with new ones.

---

## Pro Tip: Automate Model Deployment

Add this to your workflow when you retrain models:

```bash
# After training locally
python scripts/train_model.py

# Automatically deploy to production
./deploy/copy_models.sh YOUR_DROPLET_IP

# Restart services to pick up new models
ssh root@YOUR_DROPLET_IP "systemctl restart nba-dashboard nba-api"
```

This ensures your production system always has the latest models!
