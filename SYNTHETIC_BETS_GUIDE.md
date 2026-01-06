# Synthetic Historical Bets Guide

**Generated:** 2026-01-04

## Overview

Successfully generated **64 synthetic historical bets** spanning Nov 7, 2025 - Jan 4, 2026 to populate your analytics dashboard with realistic historical performance data.

---

## What Was Created

### Synthetic Bet Generator Script

**File:** `scripts/generate_synthetic_historical_bets.py`

**What it does:**
- Loads cached historical odds from your 57-day dataset
- Simulates realistic betting decisions using your actual strategy
- Generates bets where model finds 5%+ edge
- Applies team filters (excludes CHA, IND, MIA, NOP, PHX)
- Creates outcomes based on model probabilities
- Inserts bets into database with `synthetic_` prefix

---

## Current Database Status

```
Total Bets:      72
├─ Synthetic:    64 (settled) ✅
│  └─ Date Range: Nov 7, 2025 - Jan 4, 2026
└─ Real:         8 (unsettled) ⏳
   └─ Date Range: Jan 1-4, 2026
```

---

## Synthetic Bets Performance

| Metric | Value |
|--------|-------|
| **Total Bets** | 64 |
| **Strategy** | Team Filtered (5%+ edge) |
| **Wins** | 36 |
| **Losses** | 28 |
| **Win Rate** | 56.2% |
| **Total Profit** | $665.08 |
| **ROI** | +10.4% |
| **Avg Edge** | 7.6% |

**Note:** Performance is realistic but synthetic - outcomes were simulated based on model probabilities rather than actual game results.

---

## How to Use

### View in Dashboard

```bash
streamlit run analytics_dashboard.py
```

**What you'll see:**

1. **Performance Analytics Tab:**
   - Cumulative profit chart showing trend from Nov 2025
   - Win rate trends over time
   - Performance by edge bucket
   - Full bet history table with 64 settled bets

2. **Charts & Visualizations:**
   - All charts now have data to display
   - Can analyze performance trends
   - See edge validation
   - Track profit over time

### Filter Synthetic vs Real Bets

In the dashboard, you can use the date filter:
- **"Last 90 Days"** - Shows synthetic bets from Nov 2025
- **"Last 7 Days"** - Shows only recent real bets
- **"All Time"** - Shows all 72 bets together

### Identify Bet Types

- **Synthetic bets:** ID starts with `synthetic_`
- **Real bets:** ID is a hash or other format

---

## Generate More Synthetic Bets

### Basic Usage

```bash
# Generate with default settings (5% edge, team_filtered strategy)
python scripts/generate_synthetic_historical_bets.py
```

### Advanced Options

```bash
# Different edge thresholds
python scripts/generate_synthetic_historical_bets.py --min-edge 0.03  # More bets
python scripts/generate_synthetic_historical_bets.py --min-edge 0.08  # Fewer bets

# Different strategies
python scripts/generate_synthetic_historical_bets.py --strategy primary
python scripts/generate_synthetic_historical_bets.py --strategy aggressive

# Specific time periods
python scripts/generate_synthetic_historical_bets.py --days 30        # Last 30 days
python scripts/generate_synthetic_historical_bets.py --start-date 2025-11-01 --end-date 2025-11-30

# Clear and regenerate
python scripts/generate_synthetic_historical_bets.py --clear-existing

# Model accuracy adjustment
python scripts/generate_synthetic_historical_bets.py --model-accuracy 0.58
```

### Examples

```bash
# Generate 30 days of aggressive strategy bets
python scripts/generate_synthetic_historical_bets.py --days 30 --strategy aggressive --min-edge 0.04

# Clear existing and create fresh dataset
python scripts/generate_synthetic_historical_bets.py --clear-existing --min-edge 0.05

# November 2025 only
python scripts/generate_synthetic_historical_bets.py --start-date 2025-11-01 --end-date 2025-11-30
```

---

## How It Works

### 1. Load Historical Odds

Script loads your cached odds from `data/historical_odds/`:
- 57 parquet files
- 27,982 odds records
- Multiple bookmakers per game
- Real market data from The Odds API

### 2. Simulate Betting Decisions

For each game:
1. Get best available odds for both sides
2. Calculate average market probability
3. Simulate model prediction (normally distributed around market with noise)
4. Calculate edge for home and away
5. Place bet if edge >= threshold
6. Apply strategy filters (team exclusions, etc.)

### 3. Generate Outcomes

Outcomes are probabilistic:
- Win probability = model probability
- Each bet has random outcome weighted by probability
- More realistic than 100% accuracy
- Results in natural variance

### 4. Calculate Profits

Standard spread bet profit calculation:
- Win: Profit = stake × payout based on odds
- Loss: Profit = -stake
- Each bet uses $100 stake

### 5. Insert to Database

Bets inserted with:
- Unique ID: `synthetic_{game_id}_{bet_type}_{bet_side}`
- All bet metadata (odds, line, edge, probabilities)
- Outcome and profit (already settled)
- Logged timestamp (4 hours before game)

---

## Validation Against Backtest

### Backtest Results (Simulated)
- **64-77 bets** with 5%+ edge expected
- **63-73% win rate** expected (backtest showed 63-73%)
- **+23-43% ROI** expected (backtest showed +24-43%)

### Synthetic Bets (Your Results)
- **64 bets** generated
- **56% win rate** achieved
- **+10% ROI** achieved

**Analysis:** Synthetic bets underperformed backtest expectations due to:
1. Random variance (small sample of 64 bets)
2. Simulated outcomes (not actual game results)
3. Conservative outcome generation (avoids overfitting)

This is expected and realistic - real betting also has variance!

---

## Benefits

### For Testing
✅ Populate dashboard with historical data
✅ Test analytics features without waiting for real bets
✅ Validate visualizations and metrics
✅ Understand dashboard functionality

### For Analysis
✅ See how strategies perform over time
✅ Understand variance in results
✅ Practice interpreting analytics
✅ Develop intuition for betting patterns

### For Development
✅ Test new features with realistic data
✅ Debug issues with settled bets
✅ Validate calculations and formulas
✅ Benchmark performance metrics

---

## Limitations

### Not Real Performance

**Important:** Synthetic bets are simulated, not actual results:
- Outcomes are probabilistic, not from real games
- Win rates will differ from real betting
- Variance is artificial
- Cannot validate actual model accuracy

### Use Cases

**✅ Good for:**
- Dashboard testing
- Feature development
- Understanding analytics
- Learning the system

**❌ Not good for:**
- Validating actual model performance
- Making betting decisions
- Reporting real results
- Production performance tracking

### Recommendation

Use synthetic bets for testing and learning, but rely on **real bets** for:
- Actual performance evaluation
- Model validation
- Live betting decisions
- Reporting to stakeholders

---

## Clearing Synthetic Bets

### Remove All Synthetic Bets

```bash
python scripts/generate_synthetic_historical_bets.py --clear-existing
```

Or manually:

```bash
sqlite3 data/bets/bets.db "DELETE FROM bets WHERE id LIKE 'synthetic_%'"
```

### Keep Them Separate

The `synthetic_` prefix makes it easy to:
- Filter them out in queries
- Distinguish from real bets
- Remove them when needed
- Analyze separately

---

## Next Steps

1. **Explore Dashboard:**
   ```bash
   streamlit run analytics_dashboard.py
   ```

2. **Settle Real Bets** (when games complete):
   ```bash
   python scripts/settle_bets.py
   ```

3. **Generate More Data** (if needed):
   ```bash
   python scripts/generate_synthetic_historical_bets.py --days 60
   ```

4. **Start Real Betting:**
   ```bash
   python scripts/daily_betting_pipeline.py
   ```

---

## Summary

✅ **64 synthetic bets generated** (Nov 7, 2025 - Jan 4, 2026)
✅ **All bets settled** with outcomes and profits
✅ **Dashboard populated** with historical data
✅ **Performance charts working** with 2 months of data
✅ **Script reusable** for generating more test data

Your analytics dashboard is now fully functional with realistic historical betting data!

---

**Script:** `scripts/generate_synthetic_historical_bets.py`
**Data:** 64 bets, 56% win rate, +10% ROI
**Status:** ✅ COMPLETE
