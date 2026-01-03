# Integrated Betting System Guide

**Complete workflow combining models, market signals, CLV optimization, and paper trading**

---

## Overview

Your betting system now has **four layers of intelligence**:

1. **🧠 Predictive Models** - XGBoost, LightGBM, Neural, Ensemble predictions
2. **📊 Market Microstructure** - Steam detection, RLM, money flow analysis
3. **📈 Line Movement Tracking** - Multi-snapshot CLV, velocity, reversal detection
4. **🎯 CLV Optimization** - Historical CLV filtering, optimal timing

This guide shows how to use them **together** for maximum edge.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    BETTING DECISION PIPELINE                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. MODEL PREDICTIONS                                         │
│    • Load models (XGBoost, LightGBM, Neural, Ensemble)      │
│    • Generate predictions for today's games                  │
│    • Calculate model edge vs market spread                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. MARKET SIGNAL ANALYSIS                                    │
│    • Fetch current odds from Odds API                        │
│    • Detect steam moves (sharp money)                        │
│    • Identify reverse line movements (RLM)                   │
│    • Analyze money flow (sharp vs public)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. LINE MOVEMENT TRACKING                                    │
│    • Query line history from snapshots DB                    │
│    • Calculate line velocity (pts/hour)                      │
│    • Detect reversals and steam patterns                     │
│    • Get opening lines for CLV potential                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. CLV OPTIMIZATION                                          │
│    • Get historical CLV for similar bets                     │
│    • Check if current time is optimal for booking           │
│    • Filter bets with negative historical CLV                │
│    • Adjust bet sizing based on CLV confidence               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. EDGE STRATEGY EVALUATION                                  │
│    • Apply EdgeStrategy filters (edge, B2B, team, etc.)     │
│    • Combine model edge + market signals + CLV              │
│    • Generate bet signals with confidence levels             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. BET EXECUTION (Paper or Live)                            │
│    • Log recommended bets to database                        │
│    • Track for settlement and CLV calculation                │
│    • Monitor performance vs expectations                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Integration

### Step 1: Model Predictions

**Load your models and generate predictions:**

```python
from src.models.model_loader import load_models_from_latest
from src.models.ensemble_model import EnsembleModel
import pandas as pd

# Load latest trained models
models = load_models_from_latest()
ensemble = EnsembleModel(models)

# Get today's games (from your data pipeline)
games_df = get_todays_games()  # Your function

# Generate predictions
predictions = ensemble.predict(games_df)

# Calculate model edge
games_df['pred_diff'] = predictions  # Model's predicted margin
games_df['model_edge'] = games_df['pred_diff'] + games_df['market_spread']
```

### Step 2: Fetch Market Signals

**Get current odds and detect market signals:**

```python
from src.data.odds_api import OddsAPIClient
from src.data.line_history import LineHistoryManager

# Fetch current odds
odds_client = OddsAPIClient()
current_odds = odds_client.get_current_odds(markets=['spreads'])

# Merge with predictions
games_df = games_df.merge(
    current_odds[['game_id', 'market_spread', 'home_odds', 'away_odds']],
    on='game_id',
    how='left'
)

# Initialize line history manager for signals
line_manager = LineHistoryManager()
```

### Step 3: Analyze Market Microstructure

**Detect steam moves, RLM, and money flow:**

```python
from src.betting.market_signals import (
    detect_steam_moves,
    detect_reverse_line_movement,
    analyze_money_flow
)

# Add market signal columns
games_df['steam_signal'] = None
games_df['rlm_signal'] = None
games_df['money_flow_signal'] = None

for idx, game in games_df.iterrows():
    game_id = game['game_id']

    # Get line history
    line_history = line_manager.get_line_history(
        game_id=game_id,
        bet_type='spread',
        side='home'
    )

    if not line_history.empty:
        # Detect steam move
        steam = detect_steam_moves(line_history)
        games_df.at[idx, 'steam_signal'] = steam

        # Detect RLM
        rlm = detect_reverse_line_movement(
            line_history=line_history,
            bet_percentage=game.get('home_bet_pct', 50)  # If available
        )
        games_df.at[idx, 'rlm_signal'] = rlm

        # Analyze money flow
        flow = analyze_money_flow(
            line_history=line_history,
            bet_percentage=game.get('home_bet_pct', 50),
            money_percentage=game.get('home_money_pct', 50)  # If available
        )
        games_df.at[idx, 'money_flow_signal'] = flow
```

### Step 4: Configure Integrated Strategy

**Create a strategy that uses ALL signals:**

```python
from src.betting.edge_strategy import EdgeStrategy

# Option 1: Maximum Intelligence Strategy
# Uses: Model edge + Market signals + CLV optimization + Timing
strategy = EdgeStrategy(
    # Core filters
    edge_threshold=5.0,
    require_no_b2b=True,
    use_team_filter=True,

    # Market microstructure filters
    require_steam_alignment=True,      # Bet with sharp money
    require_rlm_alignment=True,        # Bet with RLM signals
    require_sharp_alignment=True,      # Bet with money flow
    min_steam_confidence=0.7,          # High confidence steam only

    # CLV optimization filters
    clv_filter_enabled=True,           # Filter by historical CLV
    min_historical_clv=0.01,           # Require +1% historical CLV
    optimal_timing_filter=True,        # Only bet at optimal times
    clv_based_sizing=True,             # Adjust bet size by CLV
)

# Option 2: Use Preset (Recommended for starting)
strategy = EdgeStrategy.clv_filtered_strategy()  # Start conservative

# Option 3: Custom Combination
strategy = EdgeStrategy(
    edge_threshold=5.0,
    require_no_b2b=True,
    use_team_filter=True,
    clv_filter_enabled=True,
    min_historical_clv=0.005,  # More permissive (0.5%)
    require_steam_alignment=False,  # Optional: disable for more bets
)
```

### Step 5: Generate Bet Signals

**Evaluate games with ALL signals:**

```python
from src.betting.edge_strategy import BetSignal
from typing import List

bet_signals: List[BetSignal] = []

for _, game in games_df.iterrows():
    signal = strategy.evaluate_game(
        game_id=game['game_id'],
        home_team=game['home_team'],
        away_team=game['away_team'],
        pred_diff=game['pred_diff'],
        market_spread=game['market_spread'],
        home_b2b=game.get('home_b2b', False),
        away_b2b=game.get('away_b2b', False),
        rest_advantage=game.get('rest_advantage', 0),

        # Market signals (optional but recommended)
        steam_signal=game.get('steam_signal'),
        rlm_signal=game.get('rlm_signal'),
        money_flow_signal=game.get('money_flow_signal'),
    )

    bet_signals.append(signal)

# Filter to actionable bets only
actionable_bets = [s for s in bet_signals if s.is_actionable]

print(f"Found {len(actionable_bets)} actionable bets out of {len(bet_signals)} games")
```

### Step 6: Log Bets for Tracking

**Log to database for CLV tracking:**

```python
from src.bet_tracker import log_bet

for signal in actionable_bets:
    # Get current odds for the bet side
    if signal.bet_side == "HOME":
        odds = games_df[games_df['game_id'] == signal.game_id]['home_odds'].iloc[0]
        line = signal.market_spread
    else:
        odds = games_df[games_df['game_id'] == signal.game_id]['away_odds'].iloc[0]
        line = -signal.market_spread

    # Log bet (will be tracked for CLV)
    bet_record = log_bet(
        game_id=signal.game_id,
        home_team=signal.home_team,
        away_team=signal.away_team,
        commence_time=games_df[games_df['game_id'] == signal.game_id]['commence_time'].iloc[0],
        bet_type='spread',
        bet_side=signal.bet_side.lower(),
        odds=odds,
        line=line,
        model_prob=0.50 + (signal.model_edge * 0.01),  # Approximate
        market_prob=0.50,
        edge=signal.model_edge,
        kelly=signal.model_edge * 0.01,  # Simple Kelly
        bookmaker='draftkings',  # Your sportsbook
    )

    print(f"Logged bet: {signal.bet_side} {signal.home_team if signal.bet_side == 'HOME' else signal.away_team}")
    print(f"  Edge: {signal.model_edge:.2f}, Confidence: {signal.confidence}")
    print(f"  Filters passed: {', '.join(signal.filters_passed)}")
```

---

## Complete Daily Workflow Script

**Create `scripts/daily_betting_pipeline.py`:**

```python
#!/usr/bin/env python3
"""
Daily Betting Pipeline - Integrated System

Combines models, market signals, CLV optimization for daily bet recommendations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
import pandas as pd
from loguru import logger

from src.models.model_loader import load_models_from_latest
from src.models.ensemble_model import EnsembleModel
from src.data.odds_api import OddsAPIClient
from src.data.line_history import LineHistoryManager
from src.betting.edge_strategy import EdgeStrategy
from src.bet_tracker import log_bet


def get_todays_games() -> pd.DataFrame:
    """
    Get today's games with features ready for prediction.

    TODO: Implement your feature pipeline here
    This should return a DataFrame with all features needed by your models.
    """
    # Placeholder - replace with your actual data pipeline
    raise NotImplementedError("Implement your feature pipeline")


def enrich_with_market_signals(games_df: pd.DataFrame) -> pd.DataFrame:
    """Add market signal columns to games DataFrame."""
    line_manager = LineHistoryManager()

    for idx, game in games_df.iterrows():
        game_id = game['game_id']

        # Get line history
        try:
            line_history = line_manager.get_line_history(
                game_id=game_id,
                bet_type='spread',
                side='home'
            )

            if not line_history.empty:
                # Detect patterns
                movement = line_manager.analyze_movement_pattern(
                    game_id=game_id,
                    bet_type='spread',
                    bet_side='home'
                )

                games_df.at[idx, 'movement_pattern'] = movement.get('pattern') if movement else None
                games_df.at[idx, 'line_velocity'] = movement.get('velocity') if movement else None

        except Exception as e:
            logger.debug(f"Could not get line history for {game_id}: {e}")

    return games_df


def main():
    logger.info("=" * 80)
    logger.info("DAILY BETTING PIPELINE - Integrated System")
    logger.info("=" * 80)
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Load models
    logger.info("\n1. Loading models...")
    models = load_models_from_latest()
    ensemble = EnsembleModel(models)
    logger.info(f"   ✓ Loaded {len(models)} models")

    # 2. Get today's games
    logger.info("\n2. Getting today's games...")
    try:
        games_df = get_todays_games()
        logger.info(f"   ✓ Found {len(games_df)} games")
    except NotImplementedError:
        logger.error("   ✗ Feature pipeline not implemented yet")
        logger.info("   → Implement get_todays_games() in this script")
        return 1

    # 3. Generate predictions
    logger.info("\n3. Generating model predictions...")
    predictions = ensemble.predict(games_df)
    games_df['pred_diff'] = predictions
    logger.info(f"   ✓ Generated predictions")

    # 4. Fetch current odds
    logger.info("\n4. Fetching current market odds...")
    odds_client = OddsAPIClient()
    current_odds = odds_client.get_current_odds(markets=['spreads'])
    games_df = games_df.merge(
        current_odds[['game_id', 'market_spread', 'home_odds', 'away_odds']],
        on='game_id',
        how='left'
    )
    logger.info(f"   ✓ Fetched odds for {len(current_odds)} games")

    # 5. Calculate edge
    games_df['model_edge'] = games_df['pred_diff'] + games_df['market_spread']

    # 6. Enrich with market signals
    logger.info("\n5. Analyzing market signals...")
    games_df = enrich_with_market_signals(games_df)
    logger.info(f"   ✓ Analyzed line movement patterns")

    # 7. Configure strategy
    logger.info("\n6. Configuring betting strategy...")
    strategy = EdgeStrategy.clv_filtered_strategy()
    logger.info(f"   ✓ Using CLV-filtered strategy")
    logger.info(f"      - Edge threshold: {strategy.edge_threshold}")
    logger.info(f"      - Min historical CLV: {strategy.min_historical_clv}")

    # 8. Evaluate games
    logger.info("\n7. Evaluating games...")
    bet_signals = []
    for _, game in games_df.iterrows():
        signal = strategy.evaluate_game(
            game_id=game['game_id'],
            home_team=game['home_team'],
            away_team=game['away_team'],
            pred_diff=game['pred_diff'],
            market_spread=game['market_spread'],
            home_b2b=game.get('home_b2b', False),
            away_b2b=game.get('away_b2b', False),
        )
        bet_signals.append(signal)

    actionable = [s for s in bet_signals if s.is_actionable]
    logger.info(f"   ✓ Found {len(actionable)} actionable bets")

    # 9. Display recommendations
    if actionable:
        logger.info("\n8. BET RECOMMENDATIONS:")
        logger.info("=" * 80)

        for i, signal in enumerate(actionable, 1):
            logger.info(f"\n#{i}. {signal.bet_side} {signal.home_team if signal.bet_side == 'HOME' else signal.away_team}")
            logger.info(f"   Game: {signal.away_team} @ {signal.home_team}")
            logger.info(f"   Model Edge: {signal.model_edge:+.2f} pts")
            logger.info(f"   Confidence: {signal.confidence}")
            logger.info(f"   Filters: {', '.join(signal.filters_passed)}")

            # Log to database
            # TODO: Implement actual bet logging when ready
            logger.info(f"   → Ready to log (uncomment log_bet() call)")

    else:
        logger.info("\n8. No actionable bets found today")
        logger.info("   All games filtered out by strategy criteria")

    logger.info("\n" + "=" * 80)
    logger.info("Pipeline complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

## Paper Trading Workflow

### Setup Paper Trading Mode

**1. Create paper trading flag in config:**

```python
# config.py or at top of your script
PAPER_TRADING = True  # Set to False for live betting
```

**2. Modify bet logging to mark paper trades:**

```python
def log_paper_bet(signal, odds, line):
    """Log a paper trade bet."""
    bet_record = log_bet(
        game_id=signal.game_id,
        home_team=signal.home_team,
        away_team=signal.away_team,
        commence_time=signal.commence_time,
        bet_type='spread',
        bet_side=signal.bet_side.lower(),
        odds=odds,
        line=line,
        model_prob=0.50 + (signal.model_edge * 0.01),
        market_prob=0.50,
        edge=signal.model_edge,
        kelly=signal.model_edge * 0.01,
        bookmaker='PAPER_TRADE',  # Mark as paper trade
    )
    return bet_record
```

### Run Daily Paper Trading

**Daily routine:**

```bash
# 1. Collect line snapshots (cron - every hour)
0 * * * * python scripts/collect_line_snapshots.py

# 2. Capture opening lines (cron - every 15 min)
*/15 * * * * python scripts/capture_opening_lines.py

# 3. Run daily pipeline (morning, before games)
python scripts/daily_betting_pipeline.py

# 4. Capture closing lines (cron - every 15 min before games)
*/15 * * * * python scripts/capture_closing_lines.py

# 5. Settle bets and calculate CLV (next day, 6 AM)
0 6 * * * python scripts/populate_clv_data.py
0 6 * * * python scripts/validate_closing_lines.py

# 6. Generate weekly report (Sunday)
0 10 * * 0 python scripts/generate_clv_report.py --export weekly_report.json
```

### Monitor Paper Trading Performance

**Weekly review:**

```bash
# Generate comprehensive report
python scripts/generate_clv_report.py

# Run backtest on paper trades
python scripts/backtest_clv_strategy.py --start-date 2024-01-01

# Check alpha monitor
python -c "
from src.monitoring.alpha_monitor import AlphaMonitor
from src.bet_tracker import get_bet_history

monitor = AlphaMonitor()
bets = get_bet_history()
report = monitor.generate_health_report(bets)
print(report)
"
```

---

## Strategy Comparison Guide

### Test Different Combinations

```python
# Test 1: Baseline (Model only)
baseline = EdgeStrategy(
    edge_threshold=5.0,
    require_no_b2b=True,
    use_team_filter=True,
)

# Test 2: Model + Market Signals
market_enhanced = EdgeStrategy(
    edge_threshold=5.0,
    require_no_b2b=True,
    use_team_filter=True,
    require_steam_alignment=True,
    min_steam_confidence=0.7,
)

# Test 3: Model + CLV
clv_enhanced = EdgeStrategy.clv_filtered_strategy()

# Test 4: Everything Combined
maximum_intelligence = EdgeStrategy(
    edge_threshold=5.0,
    require_no_b2b=True,
    use_team_filter=True,
    require_steam_alignment=True,
    clv_filter_enabled=True,
    min_historical_clv=0.01,
    optimal_timing_filter=True,
)

# Compare in backtest
strategies = [
    ('Baseline', baseline),
    ('Market Signals', market_enhanced),
    ('CLV Filtered', clv_enhanced),
    ('Maximum Intelligence', maximum_intelligence),
]

for name, strat in strategies:
    # Run backtest with each strategy
    # Compare results
    pass
```

---

## Troubleshooting

### Common Issues

**1. No market signal data:**
```python
# Ensure line snapshots are collecting
python scripts/collect_line_snapshots.py
```

**2. CLV filter too restrictive (no bets):**
```python
# Lower the threshold
strategy = EdgeStrategy(
    clv_filter_enabled=True,
    min_historical_clv=0.005,  # Lower from 0.01
)
```

**3. Steam detection not finding signals:**
```python
# Check you have enough snapshot history (need 3+ snapshots)
# Steam moves happen quickly - ensure frequent snapshot collection
```

---

## Best Practices

1. **Start Conservative:** Use CLV filter alone first, add market signals later
2. **Monitor Data Quality:** Check snapshot coverage weekly
3. **Gradual Rollout:** Paper trade 2 weeks → 25% → 50% → 100%
4. **Track Everything:** Log all bets, even filtered ones (for analysis)
5. **Weekly Reviews:** Generate CLV report every Sunday
6. **Adjust Thresholds:** Based on backtest results with real data
7. **A/B Testing:** Run multiple strategies in parallel (paper mode)

---

## Quick Reference Commands

```bash
# Daily betting pipeline
python scripts/daily_betting_pipeline.py

# Generate bet recommendations
python scripts/daily_betting_pipeline.py > todays_bets.txt

# Weekly CLV report
python scripts/generate_clv_report.py

# Backtest all strategies
python scripts/backtest_clv_strategy.py --start-date 2024-01-01

# Monitor health
python -c "from src.monitoring.alpha_monitor import AlphaMonitor; ..."

# Validate closing lines
python scripts/validate_closing_lines.py

# Populate CLV data
python scripts/populate_clv_data.py --days 7
```

---

## Next Steps

1. ✅ **Implement `get_todays_games()`** in daily pipeline
2. ✅ **Set up cron jobs** for data collection
3. ✅ **Paper trade for 2 weeks** with CLV strategy
4. ✅ **Add market signals** after validating CLV works
5. ✅ **Compare strategies** using backtests
6. ✅ **Optimize thresholds** based on results
7. ✅ **Go live** with proven strategy

Good luck! 🚀
