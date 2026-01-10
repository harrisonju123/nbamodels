# NBA Models Consolidation Migration Guide

**Date:** January 9, 2026
**Version:** 1.0

## Overview

The NBA betting models codebase has been consolidated to focus on the highest-performing models and remove unprofitable strategies. This migration guide helps existing users update their installations safely.

---

## What Changed

### Models Archived

| Model | Reason | ROI |
|-------|--------|-----|
| `dual_model.pkl` | Failed validation | -4.0% |
| `point_spread_model.pkl` | Superseded by spread_model_calibrated.pkl | N/A |
| `totals_model.pkl` | Strategy disabled (not profitable) | +0.9% |

### Strategies Removed

| Strategy | Reason | Performance |
|----------|--------|-------------|
| **TotalsStrategy** | Not profitable | +0.9% ROI (below 3% threshold) |
| **LiveBettingStrategy** | Never production-tested | N/A |

### Features Reduced

| Change | Before | After | Rationale |
|--------|--------|-------|-----------|
| Player rolling windows | [3, 5, 10] | [5, 10] | 3-game window too volatile |
| Opponent features | Hardcoded to 110.0 | Removed | Placeholders provided no signal |
| Alternative data | Referee, News, Sentiment | Disabled by default | Added noise without predictive value |

---

## Migration Steps

### Step 1: Backup Your Current Installation

```bash
cd /path/to/nbamodels

# Create backup
mkdir -p ~/nbamodels_backup_$(date +%Y%m%d)
cp -r models/ ~/nbamodels_backup_$(date +%Y%m%d)/
cp -r data/ ~/nbamodels_backup_$(date +%Y%m%d)/
cp config/multi_strategy_config.yaml ~/nbamodels_backup_$(date +%Y%m%d)/
cp -r logs/ ~/nbamodels_backup_$(date +%Y%m%d)/
```

### Step 2: Pull Latest Code

```bash
git fetch origin
git checkout main
git pull origin main
```

### Step 3: Update Configuration

Your `config/multi_strategy_config.yaml` needs updating:

**Old Configuration:**
```yaml
allocation:
  totals: 0.00     # DISABLED
  live: 0.00       # DISABLED
  arbitrage: 0.25
  props: 0.10
  b2b_rest: 0.15
  spread: 0.50
```

**New Configuration (Consolidated):**
```yaml
allocation:
  spread: 0.35     # 35% for spread betting
  arbitrage: 0.30  # 30% allocation
  props: 0.20      # 20% allocation
  b2b_rest: 0.15   # 15% allocation
```

**Action Required:**
- Remove `totals` and `live` entries from `allocation` and `daily_limits`
- Adjust allocations to total 100% across 4 strategies

### Step 4: Clean Up Old Model Files

```bash
cd models/

# Archive old models (don't delete yet - keep for rollback)
mkdir -p archive_pre_consolidation/
mv dual_model.pkl archive_pre_consolidation/ 2>/dev/null || true
mv point_spread_model.pkl archive_pre_consolidation/ 2>/dev/null || true
mv totals_model.pkl archive_pre_consolidation/ 2>/dev/null || true

# Remove duplicate player props directory
rm -rf player_props_production/

# Verify production models exist
ls -lh spread_model_calibrated.pkl
ls -lh player_props/*.pkl
```

**Expected Output:**
```
✓ spread_model_calibrated.pkl (321 KB)
✓ player_props/pts_model.pkl (255 KB)
✓ player_props/reb_model.pkl (249 KB)
✓ player_props/ast_model.pkl (250 KB)
✓ player_props/3pm_model.pkl (250 KB)
```

### Step 5: Clean Database (If Using Paper Trading)

If you have bet history in a database, clean up archived bet types:

```sql
-- Check for totals/live bets
SELECT bet_type, COUNT(*) as count
FROM bets
WHERE bet_type IN ('totals', 'live')
GROUP BY bet_type;

-- Optional: Archive to separate table before deletion
CREATE TABLE bets_archived AS
SELECT * FROM bets
WHERE bet_type IN ('totals', 'live');

-- Delete archived bet types (if desired)
-- DELETE FROM bets WHERE bet_type IN ('totals', 'live');

-- Or just mark them as archived
UPDATE bets
SET notes = CONCAT('ARCHIVED - ', notes)
WHERE bet_type IN ('totals', 'live');
```

### Step 6: Rebuild Features (Optional but Recommended)

Player features changed (3-game window removed). Rebuild to ensure consistency:

```bash
# Rebuild player game features with new windows
python scripts/build_player_game_features.py

# Verify features built successfully
python -c "
import pandas as pd
df = pd.read_parquet('data/features/player_game_features.parquet')
print(f'Player features shape: {df.shape}')

# Check for 3-game windows (should be absent)
roll3_cols = [c for c in df.columns if 'roll3' in c or '_3g' in c]
if roll3_cols:
    print(f'WARNING: Found {len(roll3_cols)} 3-game features (should be removed)')
else:
    print('✓ 3-game windows successfully removed')
"
```

### Step 7: Validate Installation

```bash
# Test core imports
python -c "
from src.models.spread_model import SpreadPredictionModel
from src.betting.strategies import ArbitrageStrategy, PlayerPropsStrategy
from src.features.game_features import GameFeatureBuilder
print('✓ Core imports successful')
"

# Test configuration loading
python -c "
import yaml
with open('config/multi_strategy_config.yaml') as f:
    config = yaml.safe_load(f)
total = sum(config['allocation'].values())
assert total <= 1.0, f'Allocation exceeds 100%: {total}'
print(f'✓ Configuration valid (total allocation: {total*100:.0f}%)')
"

# Test model loading
python -c "
import pickle
from pathlib import Path
model_path = Path('models/spread_model_calibrated.pkl')
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)
assert isinstance(model_data, dict), 'Model must be dict format'
assert 'model' in model_data, 'Model missing model key'
assert 'feature_columns' in model_data, 'Model missing feature_columns key'
print(f'✓ Spread model loads correctly ({len(model_data[\"feature_columns\"])} features)')
"
```

### Step 8: Dry Run Daily Pipeline

```bash
# Run pipeline in dry-run mode to verify everything works
python scripts/daily_betting_pipeline.py --dry-run

# Expected output:
# ✓ Loaded model with X features
# ✓ Generated X predictions
# ✓ Found X bet signals
# [DRY RUN] Would place X bets
```

---

## Breaking Changes

### Import Changes

**Old Code:**
```python
from src.models.dual_model import DualPredictionModel
from src.models.point_spread import PointSpreadModel
from src.models.totals import TotalsModel
from src.betting.strategies import TotalsStrategy, LiveBettingStrategy
```

**New Code:**
```python
from src.models.spread_model import SpreadPredictionModel
from src.betting.strategies import ArbitrageStrategy, PlayerPropsStrategy
# TotalsStrategy and LiveBettingStrategy removed
```

### Configuration Changes

**Breaking:** `props.models_dir` path changed

**Old:**
```yaml
props:
  models_dir: "models/player_props_production/latest_converted"
```

**New:**
```yaml
props:
  models_dir: "models/player_props"
```

### Feature Changes

**Breaking:** Player features no longer include 3-game rolling windows

**Impact:** If you have custom code that relies on these features:
```python
# These columns no longer exist:
'pts_roll3', 'reb_roll3', 'ast_roll3', etc.

# Use 5-game or 10-game windows instead:
'pts_roll5', 'reb_roll5', 'ast_roll5'
'pts_roll10', 'reb_roll10', 'ast_roll10'
```

---

## Troubleshooting

### Error: "DualPredictionModel is archived and no longer supported"

**Cause:** Old model file is being loaded

**Fix:**
```bash
# Check which model file is being used
grep -n "model_path" scripts/daily_betting_pipeline.py

# Should be: models/spread_model_calibrated.pkl
# Not: models/dual_model.pkl or models/spread_model.pkl

# If wrong, update the path or delete the old file
rm models/dual_model.pkl
```

### Error: "Model file missing 'feature_columns' key"

**Cause:** Old model format (DualPredictionModel object instead of dict)

**Fix:**
```bash
# Verify model is dict format
python -c "
import pickle
with open('models/spread_model_calibrated.pkl', 'rb') as f:
    data = pickle.load(f)
print(f'Model type: {type(data)}')
print(f'Keys: {data.keys() if isinstance(data, dict) else \"N/A\"}')
"

# If not dict format, retrain model
python scripts/retrain_models.py --force
```

### Error: ImportError for archived strategies

**Cause:** Code trying to import removed strategies

**Fix:**
```bash
# Find all references
grep -r "TotalsStrategy\|LiveBettingStrategy" src/ scripts/

# Update or remove those files
```

### Dashboard Shows Archived Bet Types

**Cause:** Old bets in database

**Fix:** See Step 5 above - clean database of archived bet types

---

## Rollback Procedure

If you encounter issues and need to rollback:

```bash
# 1. Restore backed up files
cp -r ~/nbamodels_backup_YYYYMMDD/models/* models/
cp ~/nbamodels_backup_YYYYMMDD/multi_strategy_config.yaml config/

# 2. Checkout previous git version
git log --oneline | head -20  # Find commit before consolidation
git checkout <commit-hash>

# 3. Restart services
# ... restart any running processes
```

---

## Performance Comparison

After migration, monitor these metrics:

| Metric | Pre-Consolidation | Post-Consolidation | Target |
|--------|-------------------|-------------------|--------|
| Spread ROI | ~23% | Monitor | Maintain or improve |
| Player Props ROI | 27-89% | Monitor | Maintain or improve |
| Feature Count | ~100 | ~60-70 | 30-40% reduction |
| Training Time | Baseline | Monitor | 20-30% faster |
| Model Size | Multiple files | 5 files | Consolidated |

**Action:** Track these metrics for 1-2 weeks to ensure consolidation didn't harm performance.

---

## Support

If you encounter issues not covered in this guide:

1. Check logs in `logs/` directory
2. Review error messages carefully
3. File an issue at: [github.com/your-repo/issues](https://github.com/your-repo/issues)
4. Include:
   - Error message
   - Python version
   - Operating system
   - Steps to reproduce

---

## FAQ

**Q: Will I lose my historical bet data?**
A: No, bet data is preserved. You may need to update queries that filter by bet type if using archived types.

**Q: Can I still use dual_model.pkl?**
A: No, it's been archived due to -4% ROI. Use `spread_model_calibrated.pkl` (42.24% ROI).

**Q: What happened to totals betting?**
A: Disabled due to +0.9% ROI (below 3% profitability threshold). Strategy archived.

**Q: Do I need to retrain models?**
A: No, existing models in `models/player_props/` and `models/spread_model_calibrated.pkl` are still valid.

**Q: Should I rebuild features?**
A: Recommended but not required. Rebuilding ensures consistency with new rolling windows.

**Q: Can I re-enable alternative data features?**
A: Yes, set `use_referee_features=True`, etc. in GameFeatureBuilder. However, backtests showed these add noise without signal.

---

## Next Steps

After successful migration:

1. ✅ Monitor performance for 1-2 weeks
2. ✅ Compare ROI metrics pre/post consolidation
3. ✅ Clean up old backup files after confirming stability
4. ✅ Update any custom scripts that reference archived models
5. ✅ Review and update documentation

**Congratulations!** You've successfully migrated to the consolidated codebase.
