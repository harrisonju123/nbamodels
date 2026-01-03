# Setup Checklist - Integrated Betting System

Complete this checklist to set up your integrated betting system with models, market signals, line movement, and CLV optimization.

---

## Phase 1: Data Collection Infrastructure ✅

### Cron Jobs (Required for CLV tracking)

Add these to your crontab (`crontab -e`):

```bash
# Hourly line snapshots (critical for CLV calculation)
0 * * * * cd /Users/harrisonju/PycharmProjects/nbamodels && python scripts/collect_line_snapshots.py >> logs/snapshots.log 2>&1

# Opening line capture (every 15 min)
*/15 * * * * cd /Users/harrisonju/PycharmProjects/nbamodels && python scripts/capture_opening_lines.py >> logs/opening.log 2>&1

# Closing line capture (every 15 min, 30-60 min before games)
*/15 * * * * cd /Users/harrisonju/PycharmProjects/nbamodels && python scripts/capture_closing_lines.py >> logs/closing.log 2>&1

# Daily CLV calculation (6 AM)
0 6 * * * cd /Users/harrisonju/PycharmProjects/nbamodels && python scripts/populate_clv_data.py >> logs/clv.log 2>&1

# Closing line validation (6:15 AM)
15 6 * * * cd /Users/harrisonju/PycharmProjects/nbamodels && python scripts/validate_closing_lines.py >> logs/validate.log 2>&1
```

**Create log directory:**
```bash
mkdir -p logs
```

### Verify Cron Setup

```bash
# Check cron is running
crontab -l

# Monitor logs
tail -f logs/snapshots.log
tail -f logs/clv.log
```

**Status:** [ ] Complete

---

## Phase 2: Model Integration

### Step 1: Implement Feature Pipeline

Edit `scripts/daily_betting_pipeline.py`:

```python
def get_todays_games() -> pd.DataFrame:
    """
    TODO: Replace with your actual feature pipeline.

    Should return DataFrame with:
    - game_id, home_team, away_team, commence_time
    - pred_diff (model prediction)
    - home_b2b, away_b2b, rest_advantage
    """
    # YOUR CODE HERE
    pass
```

**Options:**
1. Load from your existing data pipeline
2. Use your feature engineering functions
3. Query from your database

**Status:** [ ] Complete

### Step 2: Test Model Predictions

```bash
# Test the pipeline (dry run)
python scripts/daily_betting_pipeline.py --dry-run
```

**Expected output:**
- ✓ Found N games
- ✓ Generated predictions
- ✓ Model edge calculated

**Status:** [ ] Complete

---

## Phase 3: Strategy Configuration

### Choose Your Strategy

Edit strategy selection in `scripts/daily_betting_pipeline.py` or use command line:

**Option 1: Start Conservative (CLV Filter Only)**
```bash
python scripts/daily_betting_pipeline.py --strategy clv_filtered
```

**Option 2: Optimal Timing**
```bash
python scripts/daily_betting_pipeline.py --strategy optimal_timing
```

**Option 3: Custom Strategy**

Edit the script to create your own:

```python
strategy = EdgeStrategy(
    edge_threshold=5.0,
    require_no_b2b=True,
    use_team_filter=True,
    clv_filter_enabled=True,
    min_historical_clv=0.01,  # Adjust based on your data
    optimal_timing_filter=False,  # Start with CLV only
)
```

**Status:** [ ] Complete

---

## Phase 4: Paper Trading (2 weeks)

### Week 1-2: Pure Paper Trading

**Daily Routine:**

1. **Morning (before games):**
```bash
python scripts/daily_betting_pipeline.py --dry-run
```

2. **Review recommendations**
   - Check model edge
   - Review filters passed
   - Note confidence levels

3. **Log paper bets** (remove --dry-run):
```bash
python scripts/daily_betting_pipeline.py
```

4. **Next day: Check results**
```bash
# View recent bets
sqlite3 data/bets/bets.db "SELECT * FROM bets WHERE bookmaker = 'PAPER_TRADE' ORDER BY logged_at DESC LIMIT 10"
```

**Weekly Review (Sundays):**
```bash
# Generate CLV report
python scripts/generate_clv_report.py

# Run backtest on paper trades
python scripts/backtest_clv_strategy.py --start-date $(date -d '2 weeks ago' +%Y-%m-%d)
```

**Success Criteria (Week 2):**
- [ ] Win rate >= 55%
- [ ] Average CLV > 0%
- [ ] Data coverage > 70%
- [ ] No major bugs/errors

**Status:** [ ] Complete

---

## Phase 5: Market Signal Integration (Optional)

### Add Steam Detection

**Prerequisites:**
- Line snapshots collecting for 1+ week
- At least 3+ snapshots per game

**Implementation:**

1. **Check line history availability:**
```python
from src.data.line_history import LineHistoryManager
manager = LineHistoryManager()

# Test for a recent game
history = manager.get_line_history(game_id='recent_game_id', bet_type='spread', side='home')
print(f"Found {len(history)} snapshots")
```

2. **Add to strategy:**
```python
strategy = EdgeStrategy(
    edge_threshold=5.0,
    clv_filter_enabled=True,
    min_historical_clv=0.01,
    require_steam_alignment=True,  # NEW
    min_steam_confidence=0.7,      # NEW
)
```

3. **Test impact:**
```bash
# Compare with and without steam filter
python scripts/backtest_clv_strategy.py
```

**Status:** [ ] Complete

---

## Phase 6: Live Deployment (Gradual)

### Week 5: 25% Bankroll

```bash
# Switch to live mode
python scripts/daily_betting_pipeline.py --live --strategy clv_filtered
```

**Monitor closely:**
- Daily CLV reports
- Win rate tracking
- Bet volume

**Status:** [ ] Complete

### Week 6: 50% Bankroll

If Week 5 performance meets criteria:
- [ ] Win rate >= baseline
- [ ] CLV positive
- [ ] No execution issues

**Status:** [ ] Complete

### Week 7: 75% Bankroll

**Status:** [ ] Complete

### Week 8: 100% Bankroll

**Status:** [ ] Complete

---

## Monitoring & Maintenance

### Daily Checks

```bash
# Check cron jobs ran
ls -lth logs/*.log | head

# Quick bet check
sqlite3 data/bets/bets.db "SELECT COUNT(*), AVG(edge), AVG(clv) FROM bets WHERE logged_at > datetime('now', '-1 day')"
```

### Weekly Reports

```bash
# Sunday report
python scripts/generate_clv_report.py --export weekly_$(date +%Y%m%d).json

# Review
less weekly_*.json
```

### Monthly Reviews

```bash
# Full backtest
python scripts/backtest_clv_strategy.py --start-date $(date -d '30 days ago' +%Y-%m-%d)

# Strategy comparison
# TODO: Create comparison script
```

---

## Troubleshooting

### No line snapshots collecting

**Check:**
```bash
# Test snapshot collection manually
python scripts/collect_line_snapshots.py

# Check database
sqlite3 data/bets/bets.db "SELECT COUNT(*) FROM line_snapshots"
```

**Fix:**
- Verify cron job is running
- Check Odds API key is valid
- Review logs: `tail -f logs/snapshots.log`

### CLV filter too restrictive (no bets)

**Symptoms:**
- Pipeline finds 0 actionable bets
- All games filtered by CLV

**Fix:**
```python
# Lower CLV threshold
strategy = EdgeStrategy(
    clv_filter_enabled=True,
    min_historical_clv=0.005,  # Lower from 0.01 (1%) to 0.005 (0.5%)
)
```

### Low snapshot coverage

**Check:**
```bash
python scripts/generate_clv_report.py --minimal
# Look at "Snapshot Coverage Quality"
```

**Fix:**
- Ensure hourly cron is running
- Increase snapshot frequency to every 30 min
- Wait for more data accumulation

---

## Quick Reference

### Essential Commands

```bash
# Daily pipeline (paper)
python scripts/daily_betting_pipeline.py

# Daily pipeline (live)
python scripts/daily_betting_pipeline.py --live

# Weekly report
python scripts/generate_clv_report.py

# Backtest
python scripts/backtest_clv_strategy.py

# Test data
python scripts/generate_test_data.py --num-bets 200
```

### File Locations

- **Bets database:** `data/bets/bets.db`
- **Pipeline script:** `scripts/daily_betting_pipeline.py`
- **Integration guide:** `docs/INTEGRATED_BETTING_SYSTEM.md`
- **Backtest results:** `BACKTEST_RESULTS.md`
- **Logs:** `logs/*.log`

---

## Success Metrics

### After 2 Weeks Paper Trading

- [ ] 20+ bets logged
- [ ] Win rate >= 55%
- [ ] Average CLV > 0%
- [ ] Snapshot coverage > 70%
- [ ] No execution errors

### After 4 Weeks Live (25%-50%)

- [ ] ROI > 0%
- [ ] CLV trend positive
- [ ] Win rate >= backtest baseline
- [ ] Profitable overall

### After 8 Weeks Live (Full Deployment)

- [ ] ROI > 5%
- [ ] Win rate > 55%
- [ ] Average CLV > 1%
- [ ] Bankroll growing

---

## Next Steps

1. [ ] Complete Phase 1: Set up cron jobs
2. [ ] Complete Phase 2: Implement feature pipeline
3. [ ] Complete Phase 3: Configure strategy
4. [ ] Complete Phase 4: Paper trade 2 weeks
5. [ ] Complete Phase 5: Add market signals (optional)
6. [ ] Complete Phase 6: Gradual live deployment

**Current Phase:** ___________

**Target Go-Live Date:** ___________

---

**Last Updated:** 2026-01-03
**Version:** 1.0
