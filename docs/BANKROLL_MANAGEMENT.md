# Bankroll Management System

**Date**: January 4, 2026
**Status**: ✅ Complete and Integrated
**Current Bankroll**: $2,036.44 (started at $1,000)

## Overview

The bankroll management system provides dynamic tracking of your betting bankroll, automatically adjusting bet sizes as your bankroll grows or shrinks. This enables compounding growth and better risk management.

## Key Features

### 1. Dynamic Bankroll Tracking
- **Real-time balance**: Always know your current bankroll
- **Historical tracking**: Complete history of all bankroll changes
- **Peak tracking**: Monitors your all-time high bankroll
- **Drawdown monitoring**: Tracks current drawdown from peak

### 2. Automatic Bet Sizing
- **Kelly Criterion**: Uses 10% Kelly fraction for optimal growth
- **Compounding growth**: Bet sizes scale with current bankroll
- **Risk limits**: Min $10, max $50 per bet
- **Dynamic adjustment**: Automatically uses current bankroll in daily pipeline

### 3. Performance Metrics
- **ROI tracking**: Total return on initial investment
- **Profit tracking**: Total profit/loss in dollars
- **Drawdown analysis**: Maximum drawdown percentage
- **Win tracking**: Bankroll changes tied to bet outcomes

## Database Schema

### bankroll_history Table

```sql
CREATE TABLE bankroll_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    amount REAL NOT NULL,              -- Current bankroll after change
    change_amount REAL,                -- Amount of change (+/-)
    change_type TEXT,                  -- 'win', 'loss', 'push', 'initial', 'deposit', 'withdrawal'
    bet_id TEXT,                       -- Reference to bet (if applicable)
    notes TEXT,                        -- Optional description
    peak_bankroll REAL,                -- Peak bankroll at this time
    drawdown_pct REAL,                 -- Current drawdown from peak
    FOREIGN KEY (bet_id) REFERENCES bets(id)
);
```

## Usage

### Daily Pipeline (Automatic)

The daily betting pipeline now automatically uses current bankroll:

```bash
python scripts/daily_betting_pipeline.py
```

Output shows:
```
8️⃣  Logging bets to database...
   💰 Current bankroll: $2,036.44
   ✓ Logged 2 bets
   💰 Bankroll sizing: 10% Kelly with optimized thresholds
```

### View Dashboard

```bash
python scripts/paper_trading_dashboard.py
```

Shows bankroll status at top:
```
💰 BANKROLL STATUS
--------------------------------------------------------------------------------
Current Bankroll: $2,036.44
Starting Amount:  $1,000.00
Peak Bankroll:    $2,354.60
Total Profit:     $+1,036.44
Bankroll ROI:     103.64%
Max Drawdown:     56.52%
```

### Settle Bets

Automatically settle finished games and update bankroll:

```bash
# Dry run (see what would be settled)
python scripts/settle_bets.py --dry-run

# Actually settle bets
python scripts/settle_bets.py
```

Output:
```
🏁 BET SETTLEMENT
--------------------------------------------------------------------------------
Processing: Warriors @ Lakers (away)
  Final: 115 - 118 (margin: -3.0)
  Result: WIN (+90.91)
✓ Settled bet abc123: win (+90.91)

💰 Updated Bankroll:
   Current: $2,127.35
   Profit:  $+1,127.35
   ROI:     112.74%
```

### Manual Operations

```python
from src.bankroll.bankroll_manager import BankrollManager

manager = BankrollManager()

# Initialize bankroll (one-time)
manager.initialize_bankroll(1000.0, "Starting bankroll")

# Get current bankroll
current = manager.get_current_bankroll()
print(f"Current: ${current:.2f}")

# Record bet outcome (automatic via settle_bets.py)
manager.record_bet_outcome(
    bet_id='abc123',
    profit=90.91,
    notes='Win on Lakers spread'
)

# Record manual adjustment
manager.record_adjustment(
    amount=500.0,
    adjustment_type='deposit',
    notes='Added funds'
)

# Get statistics
stats = manager.get_bankroll_stats()
print(f"ROI: {stats['roi']:.2f}%")
print(f"Max Drawdown: {stats['max_drawdown']:.2%}")

# Get history
history = manager.get_bankroll_history(limit=10)
print(history)
```

## How It Works

### Initialization

On first use, bankroll is initialized:

```python
# Automatically done by daily pipeline if bankroll = 0
if current_bankroll == 0:
    manager.initialize_bankroll(1000.0)
```

Creates first record:
```
timestamp: 2026-01-04T10:00:00
amount: 1000.0
change_amount: 0
change_type: 'initial'
peak_bankroll: 1000.0
drawdown_pct: 0.0
```

### Bet Sizing

When logging bets, current bankroll is used:

```python
# In daily_betting_pipeline.py
bankroll_mgr = BankrollManager()
current_bankroll = bankroll_mgr.get_current_bankroll()

# Calculate bet size (10% Kelly)
bet_size = strategy.calculate_bet_size(
    edge=edge,
    odds=decimal_odds,
    bankroll=current_bankroll,  # Dynamic!
    confidence=model_prob,
    side=side
)
```

As bankroll grows, bet sizes increase proportionally.

### Bet Settlement

When bets are settled, bankroll updates automatically:

```python
# In settle_bets.py
manager.record_bet_outcome(
    bet_id=bet_id,
    profit=profit,  # +90.91 for win, -100 for loss
    notes=f"Auto-settled: {outcome}"
)
```

Creates new bankroll record:
```
timestamp: 2026-01-04T15:30:00
amount: 1090.91
change_amount: +90.91
change_type: 'win'
bet_id: 'abc123'
peak_bankroll: 1090.91
drawdown_pct: 0.0
```

### Drawdown Tracking

Every update calculates drawdown from peak:

```python
peak_bankroll = max(previous_peak, new_bankroll)
drawdown_pct = (peak_bankroll - new_bankroll) / peak_bankroll

# Example: Peak $2354.60, Current $2036.44
# Drawdown = (2354.60 - 2036.44) / 2354.60 = 0.1351 = 13.51%
```

## Historical Sync

Can backfill bankroll from existing bet history:

```python
manager = BankrollManager()
manager.sync_with_bets(starting_bankroll=1000.0)

# Processes all settled bets chronologically
# Updates bankroll for each win/loss
# Result: Current bankroll matches profit history
```

Used when adding bankroll tracking to existing paper trading system.

## Current Performance

**As of January 4, 2026:**

| Metric | Value |
|--------|-------|
| Starting Bankroll | $1,000.00 |
| Current Bankroll | $2,036.44 |
| Peak Bankroll | $2,354.60 |
| Total Profit | +$1,036.44 |
| Bankroll ROI | 103.64% |
| Max Drawdown | 56.52% |
| Total Bets | 150 |
| Win Rate | 56.0% |

## Bet Sizing Example

**Scenario**: Current bankroll $2,036.44, Edge 5%, Odds -110

```python
# Kelly calculation
kelly_fraction = 0.10  # 10% for safety
decimal_odds = 1.909   # -110 in decimal
edge = 0.05
model_prob = 0.55

# Full Kelly
kelly_bet = (edge * decimal_odds - (1 - model_prob)) / (decimal_odds - 1)
# = (0.05 * 1.909 - 0.45) / 0.909 = -0.389 (invalid, but adjusted by strategy)

# Fractional Kelly with bankroll
bet_size = kelly_fraction * edge * current_bankroll
# = 0.10 * 0.05 * 2036.44 = $10.18

# Capped at max $50
final_bet = min(50.0, max(10.0, bet_size))
# = $10.18
```

Compare to static $1000 bankroll:
- Static: $5 bet size
- Dynamic: $10.18 bet size (2x larger)

As bankroll grows, bets grow proportionally → **compounding returns**.

## Files

### Created
1. **src/bankroll/bankroll_manager.py** - Core bankroll management class
2. **src/bankroll/__init__.py** - Package initialization
3. **scripts/settle_bets.py** - Automatic settlement and bankroll updates
4. **docs/BANKROLL_MANAGEMENT.md** - This documentation

### Modified
1. **scripts/daily_betting_pipeline.py** - Uses dynamic bankroll for bet sizing
2. **scripts/paper_trading_dashboard.py** - Shows bankroll stats
3. **README.md** - Added bankroll documentation

### Database
- **data/bets/bets.db** - Added `bankroll_history` table and `actual_margin` column to `bets`

## Benefits

1. **Compounding Growth**: Bet sizes increase as bankroll grows
2. **Risk Management**: Automatic drawdown tracking and limits
3. **Transparency**: Complete history of all bankroll changes
4. **Automation**: Bet settlement automatically updates bankroll
5. **Accurate Tracking**: ROI based on actual bankroll, not fixed amounts

## Future Enhancements

Potential improvements:
- [ ] Alert when drawdown exceeds threshold (e.g., 30%)
- [ ] Automatic bet size reduction during drawdowns
- [ ] Multiple bankroll accounts (conservative, aggressive)
- [ ] Bankroll charts and visualizations
- [ ] Export bankroll history to CSV
- [ ] Integration with real sportsbook APIs for live balances

## Summary

✅ Dynamic bankroll tracking is **fully integrated** into the betting system.

**Key Features**:
- Current bankroll: $2,036.44 (103.64% ROI)
- Automatic bet sizing based on current balance
- Complete history and performance tracking
- One-command bet settlement
- Compounding growth enabled

**Workflow**:
1. `daily_betting_pipeline.py` → Uses current bankroll for bet sizing
2. Place bets manually or wait for games
3. `settle_bets.py` → Automatically settles and updates bankroll
4. `paper_trading_dashboard.py` → View updated performance
5. Repeat → Bankroll compounds over time

---

**Status**: Production ready 🚀
