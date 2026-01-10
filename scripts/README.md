# Scripts Directory

This directory contains all operational and analysis scripts for the NBA betting models.

## Daily Operations

### Core Pipeline
- **`daily_betting_pipeline.py`** - Main daily workflow for bet generation (with `--use-timing` for optimal bet placement)
- **`daily_multi_strategy_pipeline.py`** - Multi-strategy orchestration (spread, props, arbitrage, B2B)
- **`settle_bets.py`** - Bet settlement and tracking
- **`send_bet_notifications.py`** - Discord notifications for new bets

### Data Collection (Scheduled)
- **`capture_opening_lines.py`** - Capture opening lines for CLV tracking
- **`capture_closing_lines.py`** - Capture closing lines for settlement
- **`collect_line_snapshots.py`** - Periodic line movement tracking
- **`collect_lineups.py`** - Starting lineup data collection
- **`collect_news.py`** - Team news and injury reports
- **`collect_referees.py`** - Referee assignments

### Validation
- **`validate_closing_lines.py`** - CLV validation
- **`validate_paper_trading.py`** - Paper trading results validation

---

## Model Training

### Spread Models
- **`retrain_models.py`** - Main retraining script for all models
- **`tune_spread_model.py`** - Hyperparameter tuning with Optuna
- **`train_spread_ensemble.py`** - Ensemble training (XGBoost + LightGBM + CatBoost)
- **`retrain_with_tuned_params.py`** - Retrain using tuned hyperparameters

### Player Props Models
- **`retrain_player_props.py`** - Retrain all player props models
- **`train_player_impact_model.py`** - RAPM-based player impact estimation

---

## Backtesting & Analysis

### Backtesting
- **`backtest_model_comparison.py`** - Compare model variants
- **`backtest_against_market.py`** - Test against market lines

### Strategy Analysis
- **`b2b_rest_strategy.py`** - Back-to-back and rest advantage analysis
- **`player_props_strategy.py`** - Player props betting strategy
- **`team_analysis.py`** - Team-specific ATS performance analysis
- **`market_bias.py`** - Market inefficiency analysis

### Reporting
- **`generate_clv_report.py`** - Closing Line Value analysis
- **`line_shopping_report.py`** - Multi-sportsbook line comparison
- **`paper_trading_report.py`** - Paper trading performance report
- **`analyze_bet_timing.py`** - Optimal bet timing analysis and recommendations (DEPRECATED: use analyze_line_movement_timing.py)
- **`analyze_line_movement_timing.py`** - Empirical line movement timing analysis (27K+ odds records)

---

## Data Management

### Historical Data
- **`collect_historical_odds_for_backtest.py`** - Fetch historical odds for backtesting
- **`fetch_historical_games.py`** - Download historical game data

### Feature Engineering
- **`compare_strategies.py`** - Compare feature sets across strategies

---

## Archived Scripts

Scripts that are no longer actively used but kept for reference:
- See `/archive/deprecated_scripts/` for archived analysis scripts
- See `/archive/scripts/backtest/` for old backtest implementations

---

## Running Scripts

### Prerequisites
```bash
# Activate virtual environment
source .venv/bin/activate

# Set environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Examples
```bash
# Run daily pipeline (standard)
python scripts/daily_betting_pipeline.py

# Run daily pipeline with timing optimization (RECOMMENDED)
python scripts/daily_betting_pipeline.py --use-timing

# Retrain models
python scripts/retrain_models.py

# Generate CLV report
python scripts/generate_clv_report.py

# Analyze optimal bet timing
python scripts/analyze_line_movement_timing.py
```

---

## Scheduling

Daily operations are scheduled via cron:
- See `cron_betting.sh` for bet generation schedule
- See `cron_retrain.sh` for model retraining schedule
- See `crontab_orchestrator` for data collection schedule
