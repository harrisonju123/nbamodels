# Multi-Strategy Integration Guide

## Overview

The multi-strategy betting framework has been integrated with your production system. This guide explains what was built and how to use it.

## What's New

### 1. Multi-Strategy Pipeline

**New File**: `scripts/daily_multi_strategy_pipeline.py`

A unified daily pipeline that orchestrates multiple betting strategies:
- **Totals betting** (OVER/UNDER)
- **Live betting** (in-game opportunities)
- **Arbitrage detection** (cross-bookmaker)
- **Player props** (PTS, REB, AST, 3PM)

The pipeline coordinates all strategies with a single bankroll, applies risk management, and logs all bets to the existing database.

### 2. Configuration System

**New File**: `config/multi_strategy_config.yaml`

Centralized configuration for:
- Enabling/disabling strategies
- Per-strategy parameters (edge thresholds, filters)
- Bankroll allocation (% per strategy)
- Daily bet limits
- Risk management settings

**Config Loader**: `src/config/strategy_config.py`

### 3. Multi-Strategy Framework

Already implemented and tested (see `VALIDATION_REPORT.txt`):
- Base infrastructure (`src/betting/strategies/base.py`)
- Strategy implementations (totals, live, arbitrage, props)
- Orchestrator (`src/betting/orchestrator.py`)
- Database schema (extended with `strategy_type`, `player_name`, etc.)

---

## Quick Start

### Running the Multi-Strategy Pipeline

```bash
# Paper trading mode (default) with default strategies
python scripts/daily_multi_strategy_pipeline.py

# Dry run (show recommendations without logging)
python scripts/daily_multi_strategy_pipeline.py --dry-run

# Live mode (real bets)
python scripts/daily_multi_strategy_pipeline.py --live

# Custom config file
python scripts/daily_multi_strategy_pipeline.py --config /path/to/config.yaml
```

### Default Configuration

By default, only **Totals** and **Arbitrage** strategies are enabled (see `config/multi_strategy_config.yaml`).

To enable other strategies, edit the config file:

```yaml
strategies:
  totals:
    enabled: true      # ✅ Enabled
  arbitrage:
    enabled: true      # ✅ Enabled
  live:
    enabled: false     # ❌ Disabled (requires live monitoring)
  props:
    enabled: false     # ❌ Disabled (requires trained models)
```

---

## Configuration Guide

### Strategy Allocation

Controls what % of bankroll each strategy can use:

```yaml
allocation:
  totals: 0.30     # 30% for totals
  arbitrage: 0.25  # 25% for arbitrage
  # ... etc
```

### Daily Limits

Max bets per strategy per day:

```yaml
daily_limits:
  totals: 5
  arbitrage: 10  # Higher for arbs
```

### Strategy Parameters

#### Totals Strategy

```yaml
strategies:
  totals:
    enabled: true
    min_edge: 0.05           # 5% minimum edge
    require_no_b2b: false    # Filter back-to-backs
    min_pace: 100            # Only fast games (optional)
    max_pace: null           # No max limit
```

#### Arbitrage Strategy

```yaml
strategies:
  arbitrage:
    enabled: true
    min_arb_profit: 0.01     # 1% minimum profit
    bookmakers:
      - draftkings
      - fanduel
      - betmgm
      # ...
```

#### Player Props Strategy

```yaml
strategies:
  props:
    enabled: false           # Enable after training models
    min_edge: 0.05
    prop_types:
      - PTS
      - REB
      - AST
      - 3PM
```

---

## Training Player Prop Models

To enable player props betting:

```bash
# 1. Train the models
python scripts/train_player_props.py

# 2. Enable in config
# Edit config/multi_strategy_config.yaml:
strategies:
  props:
    enabled: true

# 3. Run pipeline
python scripts/daily_multi_strategy_pipeline.py
```

---

## Database Schema

The database has been extended to support multi-strategy tracking:

### New Columns

- `strategy_type` (TEXT) - Which strategy placed the bet
- `player_id` (TEXT) - For player props
- `player_name` (TEXT) - For player props
- `prop_type` (TEXT) - PTS, REB, AST, etc.

### Querying by Strategy

```python
from src.bet_tracker import get_bet_history

# Get all bets
all_bets = get_bet_history()

# Filter by strategy
totals_bets = all_bets[all_bets['strategy_type'] == 'totals']
arb_bets = all_bets[all_bets['strategy_type'] == 'arbitrage']
props_bets = all_bets[all_bets['strategy_type'] == 'player_props']
```

---

## Risk Management

The orchestrator applies comprehensive risk controls:

### Bankroll Protection

- **Kelly sizing**: 25% fractional Kelly (configurable)
- **Daily exposure limit**: Max 15% of bankroll per day
- **Min bet size**: $10 minimum to avoid tiny bets

### Drawdown Protection

```yaml
risk:
  drawdown_warning_threshold: 0.10  # Warning at 10%
  drawdown_pause_threshold: 0.20    # Pause at 20%
  drawdown_hard_stop: 0.30          # Stop at 30%
```

### Correlation Limits

```yaml
risk:
  max_same_team_exposure: 0.15      # Max 15% on one team
  max_same_game_exposure: 0.10      # Max 10% on one game
  max_conference_exposure: 0.30     # Max 30% per conference
```

---

## Migration from Old Pipeline

### Option 1: Run Both (Recommended)

Keep your existing `daily_betting_pipeline.py` for spread betting and run the new multi-strategy pipeline for other markets:

```bash
# Run spread betting (existing)
python scripts/daily_betting_pipeline.py

# Run totals + arbitrage (new)
python scripts/daily_multi_strategy_pipeline.py
```

### Option 2: Unified Pipeline

To consolidate everything into the multi-strategy pipeline:

1. Enable spread strategy in config:
```yaml
strategies:
  spread:
    enabled: true
    min_edge: 0.05
```

2. Disable your old cron job for `daily_betting_pipeline.py`

3. Update cron to run multi-strategy pipeline:
```cron
0 9 * * * /path/to/python /path/to/daily_multi_strategy_pipeline.py
```

---

## Dashboard Integration

### Current Status

The existing dashboard (`analytics_dashboard.py`) will automatically show bets from all strategies since they're stored in the same `bets` table.

### Filtering by Strategy

To add strategy-specific views, you can filter the dataframe:

```python
# In analytics_dashboard.py
strategy_filter = st.selectbox("Strategy", ["All", "spread", "totals", "arbitrage", "player_props"])

if strategy_filter != "All":
    df_filtered = df[df['strategy_type'] == strategy_filter]
else:
    df_filtered = df
```

### Multi-Strategy Metrics

Add strategy-specific performance tracking:

```python
# Group by strategy
strategy_performance = df.groupby('strategy_type').agg({
    'profit': 'sum',
    'bet_amount': 'sum',
    'outcome': lambda x: (x == 'win').sum() / len(x)
}).rename(columns={'outcome': 'win_rate'})

st.dataframe(strategy_performance)
```

---

## Discord Reporting

### Current Integration

The existing Discord reporting (`scripts/send_daily_report.py`) will include multi-strategy bets automatically.

### Strategy Breakdown

To add strategy-specific metrics to Discord reports, update the report template to include:

```python
# Example addition to send_daily_report.py
by_strategy = df.groupby('strategy_type').agg({
    'profit': 'sum',
    'bet_amount': lambda x: len(x)
})

message += "\n**By Strategy:**\n"
for strategy, row in by_strategy.iterrows():
    message += f"  • {strategy}: {row['bet_amount']} bets, ${row['profit']:.2f}\n"
```

---

## Testing

### Validation Status

All tests passing (see `VALIDATION_REPORT.txt`):
- ✅ 32/32 unit tests passed
- ✅ All imports working
- ✅ Database schema updated
- ✅ Mock data generation

### Test the Pipeline

```bash
# Dry run to see what it would recommend
python scripts/daily_multi_strategy_pipeline.py --dry-run

# Paper trading mode (logs to database but marked as paper)
python scripts/daily_multi_strategy_pipeline.py

# Check the database
sqlite3 data/bets.db "SELECT strategy_type, COUNT(*) FROM bets GROUP BY strategy_type;"
```

---

## Troubleshooting

### "No strategies enabled"

**Cause**: All strategies in config are set to `enabled: false`

**Fix**: Edit `config/multi_strategy_config.yaml` and set at least one strategy to `enabled: true`

### "TotalsModel not found"

**Cause**: Totals strategy enabled but model not trained

**Fix**:
- Option 1: Train the model: `python scripts/retrain_models.py`
- Option 2: Disable totals in config

### "No player prop models loaded"

**Cause**: Player props enabled but models not trained

**Fix**: Run `python scripts/train_player_props.py` or disable props in config

### "Could not fetch odds"

**Cause**: Odds API issue or no games available

**Fix**: Check your `ODDS_API_KEY` in `.env` and verify games are scheduled

---

## Next Steps

1. **Test the pipeline**: Run with `--dry-run` to verify it works
2. **Review config**: Adjust thresholds and allocations to your preference
3. **Enable strategies**: Start with totals + arbitrage, add others as needed
4. **Monitor performance**: Use dashboard to track multi-strategy ROI
5. **Train prop models** (optional): Run training script to enable player props

---

## Summary of Changes

### Files Created

1. `scripts/daily_multi_strategy_pipeline.py` - New unified pipeline
2. `config/multi_strategy_config.yaml` - Strategy configuration
3. `src/config/strategy_config.py` - Config loader
4. `MULTI_STRATEGY_INTEGRATION.md` - This guide

### Files Modified

None! The multi-strategy framework is fully additive and doesn't break existing functionality.

### Database Changes

The `bets` table has been extended with 4 new columns (backward compatible):
- `strategy_type`
- `player_id`
- `player_name`
- `prop_type`

All existing bets remain intact. New bets will populate these columns.

---

## Support

For issues or questions:
- Check validation report: `cat VALIDATION_REPORT.txt`
- Review framework docs: `cat MULTI_STRATEGY_FRAMEWORK.md`
- Test with mock data: See `tests/test_mock_data.py`
