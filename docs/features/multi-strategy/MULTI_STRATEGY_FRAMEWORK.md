# Multi-Strategy Betting Framework

A unified framework for managing multiple betting strategies with centralized bankroll management and risk controls.

## Overview

The multi-strategy framework coordinates different betting approaches (totals, live betting, arbitrage, player props) through a single orchestrator that manages:
- Combined bankroll across all strategies
- Per-strategy allocation limits
- Daily bet limits
- Kelly criterion bet sizing
- Exposure management

## Architecture

```
StrategyOrchestrator
    ├── TotalsStrategy (uses TotalsModel)
    ├── LiveBettingStrategy (wraps LiveEdgeDetector)
    ├── ArbitrageStrategy (cross-book scanning)
    └── PlayerPropsStrategy (future - requires new models)
```

## Core Components

### 1. BettingStrategy (Abstract Base Class)

All strategies implement this interface:

```python
class BettingStrategy(ABC):
    @abstractmethod
    def evaluate_game(...) -> List[BetSignal]

    @abstractmethod
    def evaluate_games(...) -> List[BetSignal]

    def get_actionable_bets(signals, max_bets) -> List[BetSignal]
```

### 2. BetSignal (Unified Output)

All strategies return standardized `BetSignal` objects:

```python
@dataclass
class BetSignal:
    strategy_type: StrategyType
    game_id: str
    home_team: str
    away_team: str
    bet_type: str  # "spread", "totals", "moneyline", "player_prop"
    bet_side: str  # "HOME", "AWAY", "OVER", "UNDER"
    model_prob: float
    market_prob: float
    edge: float
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    # ... optional fields for props, arbitrage, etc.
```

### 3. StrategyOrchestrator

Coordinates all strategies:

```python
orchestrator = StrategyOrchestrator(
    strategies=[
        TotalsStrategy(),
        LiveBettingStrategy(),
        ArbitrageStrategy(),
    ],
    config=OrchestratorConfig(
        bankroll=1000,
        strategy_allocation={
            'totals': 0.40,
            'live': 0.30,
            'arbitrage': 0.30,
        }
    )
)

# Run all strategies
signals = orchestrator.run_all_strategies(games_df, features_df, odds_df)

# Size and filter bets
recommendations = orchestrator.size_and_filter_bets(signals)
```

## Implemented Strategies

### 1. TotalsStrategy ✅

Uses existing `TotalsModel` to predict over/under.

**Features:**
- XGBoost regression model predicts expected total points
- Calculates over/under probabilities vs market line
- Filters: min edge, B2B, pace

**Usage:**
```python
strategy = TotalsStrategy(
    model_path="models/totals_model.pkl",
    min_edge=0.05,
    require_no_b2b=True,
)

# Or use factory methods
strategy = TotalsStrategy.high_pace_strategy()  # Only pace > 100
strategy = TotalsStrategy.conservative_strategy()  # 7% min edge
```

### 2. LiveBettingStrategy ✅

Wraps existing `LiveEdgeDetector` for in-game opportunities.

**Features:**
- Monitors games in progress
- Compares live model probabilities vs live odds
- Supports spread, moneyline, and total markets
- Filters: quarter limit, min confidence, time remaining

**Usage:**
```python
strategy = LiveBettingStrategy(
    min_edge=0.05,
    min_confidence=0.5,
    max_quarter=4,  # No OT
)

# Evaluate live games
live_games = {
    'game_id_1': {
        'home_team': 'LAL',
        'away_team': 'BOS',
        'home_score': 108,
        'away_score': 105,
        'quarter': 4,
        'time_remaining': '5:00',
    }
}

signals = strategy.evaluate_live_games(live_games, live_odds_df)
```

### 3. ArbitrageStrategy ✅

Scans multiple bookmakers for price discrepancies.

**Features:**
- Finds opportunities where 1/odds1 + 1/odds2 < 1
- Guarantees profit regardless of outcome
- Calculates optimal stake split
- Returns paired signals (both sides)

**Usage:**
```python
strategy = ArbitrageStrategy(
    min_arb_profit=0.01,  # 1% minimum
    bookmakers=["draftkings", "fanduel", "betmgm"],
)

# Requires odds from multiple bookmakers
odds_df = pd.DataFrame({
    'game_id': [...],
    'market': ['spread', 'moneyline', 'total'],
    'team': ['home', 'away', ...],
    'odds': [-110, +105, ...],
    'bookmaker': ['draftkings', 'fanduel', ...],
})

signals = strategy.evaluate_games(games_df, features_df, odds_df)
```

### 4. PlayerPropsStrategy ✅

Uses dedicated models for each prop type.

**Features:**
- XGBoost models for PTS, REB, AST, 3PM
- Filters: min edge, minimum minutes (starters)
- Factory methods: `starters_only_strategy()`, `conservative_strategy()`

**Status:** ✅ Implementation complete - models need training

**Usage:**
```python
# Train models first (requires player features)
# python scripts/train_player_props.py

strategy = PlayerPropsStrategy(
    models_dir="models/player_props",
    prop_types=["PTS", "REB", "AST", "3PM"],
    min_edge=0.05,
    min_minutes=25.0,  # Starters only
)

# Or use factory methods
strategy = PlayerPropsStrategy.starters_only_strategy()
strategy = PlayerPropsStrategy.conservative_strategy()  # Points only, 7% edge
```

## Database Schema Updates

The `bets` table now includes multi-strategy fields:

```sql
ALTER TABLE bets ADD COLUMN strategy_type TEXT;
ALTER TABLE bets ADD COLUMN player_id TEXT;
ALTER TABLE bets ADD COLUMN player_name TEXT;
ALTER TABLE bets ADD COLUMN prop_type TEXT;
```

**Updated `log_bet()` signature:**
```python
log_bet(
    game_id, home_team, away_team, commence_time,
    bet_type, bet_side, odds, line,
    model_prob, market_prob, edge, kelly,
    bookmaker=None,
    bet_amount=None,
    strategy_type=None,  # NEW
    player_id=None,      # NEW
    player_name=None,    # NEW
    prop_type=None,      # NEW
)
```

## Configuration

### OrchestratorConfig

```python
@dataclass
class OrchestratorConfig:
    bankroll: float = 1000.0
    kelly_fraction: float = 0.2  # 20% fractional Kelly

    # Exposure limits (as fraction of bankroll)
    max_daily_exposure: float = 0.15
    max_pending_exposure: float = 0.25

    # Per-strategy limits
    max_bets_per_strategy: Dict[StrategyType, int] = {
        'spread': 3,
        'totals': 5,
        'live': 3,
        'arbitrage': 5,
        'player_props': 5,
    }

    # Bankroll allocation
    strategy_allocation: Dict[StrategyType, float] = {
        'spread': 0.10,
        'totals': 0.30,
        'live': 0.20,
        'arbitrage': 0.25,
        'player_props': 0.15,
    }
```

## Risk Management Integration

The orchestrator integrates with existing risk management:

- **KellyBetSizer**: Calculates bet sizes using Kelly criterion
- **ExposureManager**: Enforces daily/weekly/pending limits
- **RiskConfig**: Correlation limits, drawdown protection

## Example Workflow

```python
# 1. Create strategies
totals = TotalsStrategy(min_edge=0.05)
live = LiveBettingStrategy(min_edge=0.05)
arbitrage = ArbitrageStrategy(min_arb_profit=0.01)

# 2. Configure orchestrator
config = OrchestratorConfig(
    bankroll=1000,
    strategy_allocation={'totals': 0.5, 'live': 0.3, 'arbitrage': 0.2}
)

orchestrator = StrategyOrchestrator([totals, live, arbitrage], config)

# 3. Run strategies
signals = orchestrator.run_all_strategies(games_df, features_df, odds_df)

# 4. Size bets
recommendations = orchestrator.size_and_filter_bets(signals)

# 5. Log bets
from src.bet_tracker import log_bet

for rec in recommendations:
    signal = rec['signal']
    log_bet(
        game_id=signal.game_id,
        home_team=signal.home_team,
        away_team=signal.away_team,
        commence_time=...,
        bet_type=signal.bet_type,
        bet_side=signal.bet_side,
        odds=signal.odds,
        line=signal.line,
        model_prob=signal.model_prob,
        market_prob=signal.market_prob,
        edge=signal.edge,
        kelly=rec['kelly_fraction'],
        bet_amount=rec['bet_size'],
        strategy_type=signal.strategy_type.value,
        player_id=signal.player_id,
        player_name=signal.player_name,
        prop_type=signal.prop_type,
    )
```

## File Structure

```
src/betting/strategies/
    __init__.py
    base.py                    # BettingStrategy ABC, BetSignal
    totals_strategy.py         # ✅ Implemented
    live_strategy.py           # ✅ Implemented
    arbitrage_strategy.py      # ✅ Implemented
    player_props_strategy.py   # ⏳ Planned

src/betting/
    orchestrator.py            # ✅ Implemented

src/models/player_props/       # ⏳ Planned
    base_prop_model.py
    points_model.py
    rebounds_model.py
    assists_model.py
    threes_model.py
    steals_model.py
    blocks_model.py

examples/
    multi_strategy_example.py  # ✅ Example usage
```

## Next Steps

### For Player Props Implementation:

1. **Extend OddsAPIClient** - Add `get_player_props()` method
2. **Create Prop Models** - Train XGBoost models for PTS, REB, AST, 3PM, STL, BLK
3. **Build PlayerPropsStrategy** - Implement strategy using prop models
4. **Test & Validate** - Backtest prop strategies

### For Integration with Existing Systems:

1. **Update Orchestrator Cron Job** - Integrate with existing daily workflow
2. **Dashboard Integration** - Add multi-strategy view to Streamlit dashboard
3. **Reporting** - Extend Discord reporting to show per-strategy performance
4. **Backtesting** - Create multi-strategy backtester

## Benefits

✅ **Unified Interface** - All strategies follow same pattern
✅ **Centralized Risk** - Single orchestrator manages all limits
✅ **Easy Extension** - Add new strategies by implementing `BettingStrategy`
✅ **Flexible Allocation** - Configure per-strategy bankroll splits
✅ **Built on Existing Code** - Leverages TotalsModel, LiveEdgeDetector, KellyBetSizer

## Testing

```bash
# Test totals strategy
python -c "from src.betting.strategies import TotalsStrategy; \
    s = TotalsStrategy(); print(s.strategy_type)"

# Test orchestrator
python examples/multi_strategy_example.py
```
