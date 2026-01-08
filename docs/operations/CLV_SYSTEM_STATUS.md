# CLV Optimization System - Ready to Deploy! ✅

**Status:** All infrastructure complete. Ready for cron setup and testing.
**Date:** 2026-01-03

---

## What's Already Built ✅

### Phase 1: Data Collection Automation (COMPLETE)

All scripts are implemented and tested:

1. **`scripts/capture_closing_lines.py`** ✅
   - Captures provisional closing lines 30-60 min before game
   - Runs every 15 minutes via cron
   - Provides backup closing line if post-game snapshots fail

2. **`scripts/populate_clv_data.py`** ✅
   - Calculates multi-snapshot CLV (1hr, 4hr, 12hr, 24hr before game)
   - Calculates line velocity (pts/hour movement)
   - Runs daily at 6 AM via cron

3. **`scripts/validate_closing_lines.py`** ✅
   - Post-game closing line validation
   - Priority: snapshot > API provisional > opening > NULL
   - Runs daily at 6:15 AM via cron

4. **Database Schema** ✅
   - All columns added: `provisional_closing_odds`, `provisional_closing_line`, `closing_line_source`, `snapshot_coverage`
   - Migration handled automatically on first run

### Phase 2: EdgeStrategy CLV Integration (COMPLETE)

CLV filtering is fully integrated into `src/betting/edge_strategy.py`:

1. **New Parameters** ✅
   ```python
   clv_filter_enabled: bool = False
   min_historical_clv: float = 0.0      # Require +CLV on similar past bets
   optimal_timing_filter: bool = False   # Book at optimal time windows
   clv_based_sizing: bool = False        # Scale bet size by CLV confidence
   line_velocity_threshold: float = None # Avoid steam against us
   ```

2. **Helper Methods** ✅
   - `_get_historical_clv()` - Query avg CLV for similar bets (last 30 days, similar edge)
   - `_check_optimal_timing()` - Check if now is optimal booking time
   - `calculate_kelly_with_clv_adjustment()` - Scale bet size by CLV confidence

3. **New Strategy Presets** ✅
   ```python
   EdgeStrategy.clv_filtered_strategy()     # CLV filter (+1% minimum)
   EdgeStrategy.optimal_timing_strategy()   # Timing filter
   ```

4. **Sharp/Public Money & RLM** ✅ (Already Exists!)
   - Sharp money tracking via `MoneyFlowAnalyzer`
   - RLM detection via `RLMDetector`
   - Steam move detection via `SteamMoveDetector`
   - Integrated in EdgeStrategy with optional parameters:
     ```python
     require_steam_alignment: bool = False
     require_rlm_alignment: bool = False
     require_sharp_alignment: bool = False
     ```

---

## What You Need to Do 🎯

### Step 1: Set Up Cron Jobs (15 minutes)

Open crontab:
```bash
crontab -e
```

Add these lines (replace `/Users/harrisonju/PycharmProjects/nbamodels` with your actual path):

```bash
# Hourly line snapshots (CRITICAL - foundation for CLV)
0 * * * * cd /Users/harrisonju/PycharmProjects/nbamodels && python scripts/collect_line_snapshots.py >> logs/snapshots.log 2>&1

# Opening line capture (every 15 min)
*/15 * * * * cd /Users/harrisonju/PycharmProjects/nbamodels && python scripts/capture_opening_lines.py >> logs/opening.log 2>&1

# Closing line capture (every 15 min)
*/15 * * * * cd /Users/harrisonju/PycharmProjects/nbamodels && python scripts/capture_closing_lines.py >> logs/closing.log 2>&1

# Daily CLV calculation (6 AM)
0 6 * * * cd /Users/harrisonju/PycharmProjects/nbamodels && python scripts/populate_clv_data.py >> logs/clv.log 2>&1

# Closing line validation (6:15 AM)
15 6 * * * cd /Users/harrisonju/PycharmProjects/nbamodels && python scripts/validate_closing_lines.py >> logs/validate.log 2>&1
```

**Verify cron setup:**
```bash
crontab -l   # List cron jobs
tail -f logs/snapshots.log  # Monitor snapshot collection
```

### Step 2: Implement `get_todays_games()` in daily_betting_pipeline.py (30 minutes)

Edit `scripts/daily_betting_pipeline.py` line 42:

```python
def get_todays_games() -> pd.DataFrame:
    """
    Get today's games with features ready for prediction.

    TODO: Replace with your actual feature pipeline.
    """
    # YOUR IMPLEMENTATION HERE
    # Option 1: Load from your existing data pipeline
    # Option 2: Use your feature engineering functions
    # Option 3: Query from your database

    # Required columns:
    # - game_id, home_team, away_team, commence_time
    # - pred_diff (model prediction)
    # - home_b2b, away_b2b, rest_advantage

    pass
```

### Step 3: Test the Pipeline (Dry Run)

```bash
# Test without logging bets
python scripts/daily_betting_pipeline.py --dry-run

# Test with CLV filter strategy
python scripts/daily_betting_pipeline.py --dry-run --strategy clv_filtered

# Test with all market signals
python scripts/daily_betting_pipeline.py --dry-run --strategy clv_filtered
```

Expected output:
- ✓ Found N games
- ✓ Generated predictions
- ✓ Model edge calculated
- ✓ Filtered by CLV/market signals
- 🎯 X actionable bets

### Step 4: Paper Trade (2 weeks)

Once cron jobs are running and data is accumulating:

```bash
# Daily morning routine (before games start)
python scripts/daily_betting_pipeline.py

# Review recommendations and manually log paper trades
# Monitor with weekly CLV report:
python scripts/generate_clv_report.py
```

**Success criteria after 2 weeks:**
- [ ] 20+ paper bets logged
- [ ] Win rate >= 55%
- [ ] Average CLV > 0%
- [ ] Snapshot coverage > 70%
- [ ] No execution errors

---

## Using Market Signals (Sharp Money, RLM, Steam) 🔍

You mentioned wanting to add sharp/public money and RLM - **these already exist**! Here's how to use them:

### Example: CLV + Sharp Money + RLM Strategy

```python
from src.betting.edge_strategy import EdgeStrategy

strategy = EdgeStrategy(
    edge_threshold=5.0,
    require_no_b2b=True,
    use_team_filter=True,

    # CLV filters
    clv_filter_enabled=True,
    min_historical_clv=0.01,  # Require +1% CLV

    # Market microstructure signals
    require_steam_alignment=True,   # Only bet when steam agrees
    require_rlm_alignment=True,     # Only bet when RLM agrees (sharp vs public)
    require_sharp_alignment=True,   # Only bet when sharp money agrees
    min_steam_confidence=0.7,
)
```

### Data Source Note

The current implementation uses **Pinnacle vs retail book divergence** as a proxy for sharp/public split because The Odds API doesn't provide direct bet percentage data.

**To get actual bet percentages**, you could integrate:
- ActionNetwork API (bet % and money %)
- OddsShark consensus data
- ScoresAndOdds public betting data

### Signal Implementation Files

- `src/betting/market_signals.py` - SteamMoveDetector, RLMDetector, MoneyFlowAnalyzer
- `src/betting/signal_validator.py` - Signal validation framework
- `src/monitoring/alpha_monitor.py` - Signal performance tracking
- `scripts/validate_signal.py` - CLI tool for signal validation

---

## Available Strategies 📊

### Conservative (Recommended for Start)
```python
strategy = EdgeStrategy.clv_filtered_strategy()
```
- Edge 5+ & No B2B & Team Filter & CLV Filter
- Only bets with +1% historical CLV
- Expected: 63.9% win rate, +21.97% ROI (from backtest)

### Optimal Timing
```python
strategy = EdgeStrategy.optimal_timing_strategy()
```
- Edge 5+ & No B2B & Timing Filter
- Books bets at optimal time windows (based on line movement analysis)
- Best for 1-4 hours before game

### Aggressive (All Signals)
```python
strategy = EdgeStrategy(
    edge_threshold=5.0,
    require_no_b2b=True,
    use_team_filter=True,
    clv_filter_enabled=True,
    min_historical_clv=0.01,
    optimal_timing_filter=True,
    require_steam_alignment=True,
    require_rlm_alignment=True,
    require_sharp_alignment=True,
)
```
- Combines all filters
- Very selective (high quality, low volume)

---

## Monitoring & Debugging 🔧

### Check Cron Job Status

```bash
# Verify cron is running
crontab -l

# Monitor logs in real-time
tail -f logs/snapshots.log
tail -f logs/clv.log
tail -f logs/validate.log
```

### Check Data Collection

```bash
# Count line snapshots
sqlite3 data/bets/bets.db "SELECT COUNT(*) FROM line_snapshots"

# Check recent snapshots
sqlite3 data/bets/bets.db "SELECT game_id, snapshot_time, bookmaker FROM line_snapshots ORDER BY snapshot_time DESC LIMIT 10"

# Check CLV data
sqlite3 data/bets/bets.db "SELECT id, clv_at_1hr, clv_at_4hr, snapshot_coverage FROM bets WHERE clv_at_1hr IS NOT NULL LIMIT 10"
```

### Generate CLV Report

```bash
python scripts/generate_clv_report.py
```

Output includes:
- CLV by time window (1hr, 4hr, 12hr, 24hr)
- Optimal booking windows
- Snapshot coverage quality
- Line velocity correlation
- Closing line sources

---

## Backtest Results 📈

From `BACKTEST_RESULTS.md`:

| Metric | Baseline (Team Filter) | CLV-Filtered | Improvement |
|--------|------------------------|--------------|-------------|
| Win Rate | 59.3% | **63.9%** | **+4.6%** |
| ROI | 13.13% | **21.97%** | **+8.84%** |
| Avg CLV | 0.7% | **2.2%** | **+1.5%** |
| Bets | 81 | 36 | -55.6% (selective) |

**Recommendation:** ✅ PROCEED TO PAPER TRADING

---

## Next Steps Checklist ☑️

**Week 1: Data Collection**
- [ ] Set up all cron jobs
- [ ] Verify hourly snapshots collecting
- [ ] Monitor logs for errors
- [ ] Confirm database populating

**Week 2-3: Pipeline Integration**
- [ ] Implement `get_todays_games()` feature pipeline
- [ ] Test dry runs with multiple strategies
- [ ] Verify CLV filters working
- [ ] Review daily recommendations

**Week 4-5: Paper Trading**
- [ ] Start paper trading with `clv_filtered_strategy()`
- [ ] Log 20+ paper bets
- [ ] Generate weekly CLV reports
- [ ] Validate 55%+ win rate, positive CLV

**Week 6+: Live Deployment (Gradual)**
- [ ] Week 6: 25% of bankroll
- [ ] Week 7: 50% of bankroll
- [ ] Week 8: 75% of bankroll
- [ ] Week 9+: 100% if performance holds

---

## Help & Documentation 📚

- **Setup Guide:** `SETUP_CHECKLIST.md`
- **Integration Guide:** `docs/INTEGRATED_BETTING_SYSTEM.md`
- **Backtest Results:** `BACKTEST_RESULTS.md`
- **Plan File:** `/Users/harrisonju/.claude/plans/swift-questing-lake.md`

---

## Summary

**✅ All infrastructure is ready!**

The CLV optimization system is fully implemented with:
- Automated data collection (cron jobs)
- Multi-snapshot CLV calculation
- EdgeStrategy CLV filtering
- Market signal integration (steam, RLM, sharp/public money)
- Backtest validation showing +67% ROI improvement

**Next:** Set up cron jobs → implement `get_todays_games()` → start paper trading!

---

**Questions or Issues?**
Run unit tests: `pytest tests/ -v`
Check logs: `tail -f logs/*.log`
Generate CLV report: `python scripts/generate_clv_report.py`
