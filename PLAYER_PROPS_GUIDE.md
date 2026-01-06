# Player Props Strategy - Complete Guide

**Add a 4th income stream to your multi-strategy system**

Expected ROI: **10-15%** (typical for player props)

---

## 📊 What You're Building

### Player Props Models
Train XGBoost models to predict:
- **PTS** (Points) - Most popular prop
- **REB** (Rebounds)
- **AST** (Assists)
- **3PM** (Three-pointers made)

### How It Works
1. Predict player's expected performance (e.g., 25.3 points)
2. Compare to bookmaker's line (e.g., O/U 24.5)
3. Calculate edge using probability distribution
4. Bet when edge > 5%

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Fetch player box scores (~1 hour, one-time)
python scripts/build_player_features.py --seasons 2023 2024 2025

# 2. Build player game features (~2 minutes)
python scripts/build_player_game_features.py --seasons 2023 2024 2025

# 3. Train props models (~10 minutes)
python scripts/train_player_props.py
```

**That's it!** Models are trained and ready to use.

---

## 📋 Step-by-Step Walkthrough

### Step 1: Fetch Player Box Scores (One-Time)

```bash
python scripts/build_player_features.py --seasons 2023 2024 2025
```

**What this does:**
- Fetches individual player stats for every game
- Uses NBA API (rate limited to ~2 games/sec)
- Caches to `data/cache/player_box_scores.parquet`
- **Time**: ~45-60 minutes for 3 seasons (~1,500 games)

**Progress:**
```
Progress: 100/1500 (6.7%)
Progress: 200/1500 (13.3%)
...
Saved 50,000 box scores to data/cache/player_box_scores.parquet
```

**Cache & Resume:**
- Script saves progress every 50 games
- If interrupted, re-run and it continues from cache
- Already cached games are skipped

---

### Step 2: Build Player Game Features

```bash
python scripts/build_player_game_features.py --seasons 2023 2024 2025
```

**What this does:**
- Loads cached box scores
- Calculates rolling averages (3, 5, 10 games)
- Adds matchup features (opponent defense, pace)
- Adds context (home/away, rest days)
- Outputs `data/features/player_game_features.parquet`
- **Time**: ~2 minutes

**Output:**
```
Saved 45,000 player-game records to data/features/player_game_features.parquet

Feature Summary:
  Unique players: 450
  Unique games: 1,500
  Total features: 65

Rolling features:
  pts_roll3, pts_roll5, pts_roll10
  reb_roll3, reb_roll5, reb_roll10
  ast_roll3, ast_roll5, ast_roll10
  ...
```

---

### Step 3: Train Player Props Models

```bash
python scripts/train_player_props.py
```

**What this does:**
- Trains 4 XGBoost models (PTS, REB, AST, 3PM)
- Uses 80/20 train/val split
- Evaluates on validation set
- Saves models to `models/player_props/`
- **Time**: ~10 minutes

**Output:**
```
Training PTS model...
  Train: 36,000, Val: 9,000
  Train metrics: {'mae': 3.2, 'rmse': 4.5, 'r2': 0.85}
  Val metrics: {'mae': 3.5, 'rmse': 4.8, 'r2': 0.82}

Top 10 features:
  pts_roll5        0.450
  pts_roll10       0.280
  min_roll5        0.120
  ...

PTS model saved to models/player_props/pts_model.pkl
```

**Expected Performance:**
| Metric | PTS | REB | AST | 3PM |
|--------|-----|-----|-----|-----|
| **MAE** | 3-4 pts | 1-2 reb | 1-2 ast | 0.5-1 |
| **RMSE** | 4-5 pts | 2-3 reb | 2-3 ast | 0.8-1.5 |
| **R²** | 0.80-0.85 | 0.75-0.80 | 0.70-0.75 | 0.60-0.70 |

---

## ✅ Verify Models Are Ready

```bash
# Check models exist
ls -lh models/player_props/

# Should see:
# pts_model.pkl
# reb_model.pkl
# ast_model.pkl
# 3pm_model.pkl

# Test loading a model
python -c "from src.models.player_props import PointsPropModel; m = PointsPropModel(); m.load('models/player_props/pts_model.pkl'); print('✅ Model loads successfully')"
```

---

## 🔧 Enable in Multi-Strategy Config

```bash
# Edit config
nano config/multi_strategy_config.yaml
```

**Update these sections:**

```yaml
allocation:
  totals: 0.00
  live: 0.00
  arbitrage: 0.15   # Reduce from 20%
  props: 0.10       # NEW - 10% allocation
  b2b_rest: 0.15
  spread: 0.60      # Reduce from 65%

daily_limits:
  props: 5          # Max 5 prop bets per day

strategies:
  props:
    enabled: true   # ENABLE
    min_edge: 0.05
    prop_types:
      - PTS
      - REB
      - AST
      - 3PM
    models_dir: "models/player_props"
```

**Save:** `Ctrl+X`, `Y`, `Enter`

---

## 🧪 Test Player Props Strategy

```bash
# Run pipeline with props enabled
python scripts/daily_multi_strategy_pipeline.py --dry-run

# Should see:
# ✓ Enabled ArbitrageStrategy
# ✓ Enabled B2BRestStrategy
# ✓ Enabled PlayerPropsStrategy  ← NEW!
# ✓ Loaded 4 prop models (PTS, REB, AST, 3PM)
```

---

## 📊 Deploy to Droplet

Once tested locally:

```bash
# Push to GitHub
git add .
git commit -m "Add player props models and strategy"
git push

# SSH to droplet
ssh root@your-droplet-ip
cd nbamodels
git pull

# Copy trained models (or retrain on droplet)
# Option A: Copy from local (faster)
scp -r models/player_props/ root@your-droplet-ip:~/nbamodels/models/

# Option B: Train on droplet (takes ~1 hour)
python scripts/build_player_features.py --seasons 2023 2024 2025
python scripts/build_player_game_features.py --seasons 2023 2024 2025
python scripts/train_player_props.py

# Verify models exist
ls -l models/player_props/

# Test pipeline
python scripts/daily_multi_strategy_pipeline.py --dry-run
```

---

## 📈 Expected Performance

### Backtest Benchmarks (Typical)

| Metric | Target | Notes |
|--------|--------|-------|
| **Win Rate** | 52-55% | Above breakeven (50%) |
| **ROI** | 10-15% | Higher than spreads |
| **Volume** | 2-4 props/day | Depends on edge threshold |
| **Avg Edge** | 6-8% | Higher edges than team markets |

### Why Props Work

**Less efficient markets:**
- Bookmakers less sophisticated on individual players
- Sharp bettors focus on team markets
- More variance = more mispricing

**Model advantages:**
- Rolling averages capture recent form
- Matchup features (defender quality, pace)
- Minutes projections important

---

## 🔍 Monitoring Props Performance

### Daily Checks

```bash
# View prop bets only
source venv/bin/activate
python -c "from src.bet_tracker import get_bet_history; import pandas as pd; df = get_bet_history(); props = df[df['strategy_type'] == 'player_props']; print(props.tail(10))"

# Props win rate
python -c "from src.bet_tracker import get_performance_by_type; print(get_performance_by_type())"
```

### Dashboard

```bash
streamlit run dashboard/analytics_dashboard.py
# Go to "Performance" tab
# Filter by strategy: player_props
```

---

## 🛠️ Maintenance

### Retrain Models (Monthly Recommended)

```bash
# Update player data with latest games
python scripts/build_player_features.py --seasons 2024 2025

# Rebuild features
python scripts/build_player_game_features.py --seasons 2024 2025

# Retrain models
python scripts/train_player_props.py

# Deploy updated models
scp -r models/player_props/ root@your-droplet-ip:~/nbamodels/models/
```

### Model Drift Detection

Watch for:
- Win rate dropping below 50%
- MAE increasing over time
- Features losing importance

If detected:
- Retrain with latest data
- Consider adding new features
- Adjust edge threshold

---

## 📚 Advanced: Adding New Prop Types

Want to bet on Steals, Blocks, or custom props?

```python
# 1. Create new model class
# src/models/player_props/steals_model.py

from .base_prop_model import BasePlayerPropModel

class StealsPropModel(BasePlayerPropModel):
    prop_type = "STL"

    def get_required_features(self):
        return [
            "stl_roll3",
            "stl_roll5",
            "stl_roll10",
            "min_roll5",
            "opp_pace",
            ...
        ]

# 2. Add to training script
# scripts/train_player_props.py
from src.models.player_props import StealsPropModel

models_config.append((StealsPropModel, "stl", "stl_model.pkl"))

# 3. Enable in config
# config/multi_strategy_config.yaml
prop_types:
  - PTS
  - REB
  - AST
  - STL  # NEW
```

---

## ✅ Player Props Checklist

After completing all steps:

- [ ] Box scores fetched (`data/cache/player_box_scores.parquet` exists)
- [ ] Player game features built (`data/features/player_game_features.parquet` exists)
- [ ] Models trained (4 .pkl files in `models/player_props/`)
- [ ] Models load successfully (test command works)
- [ ] Config updated (props enabled, 10% allocation)
- [ ] Pipeline test successful (props strategy loads)
- [ ] Models deployed to droplet (if using droplet)
- [ ] Discord notifications show props bets

---

## 🎉 You're Done!

Your system now has **4 active strategies**:
1. ✅ Spread (60%) - +23.6% ROI
2. ✅ Arbitrage (15%) - +10.93% profit
3. ✅ B2B Rest (15%) - +8.7% ROI
4. ✅ **Player Props (10%) - 10-15% ROI** ← NEW!

**Combined Expected ROI: ~17-20%**

Monitor props performance for 2 weeks before increasing allocation!
