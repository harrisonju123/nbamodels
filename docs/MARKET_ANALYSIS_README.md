# Market Analysis System

A comprehensive market analysis framework for NBA betting, featuring model drift monitoring, performance gap analysis, market efficiency tracking, and optimal bet timing.

## 📋 Table of Contents

- [Overview](#overview)
- [Components](#components)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Cron Schedule](#cron-schedule)
- [Database Schema](#database-schema)
- [API Reference](#api-reference)

## Overview

The Market Analysis System provides four major capabilities:

1. **Hold-out Performance Monitoring**
   - Backtest vs live gap detection
   - Model drift and calibration monitoring
   - Overfitting alerts

2. **Market Efficiency Analysis**
   - Sharp vs public divergence tracking
   - Edge decay and time-to-efficiency analysis
   - Sharp signal performance validation

3. **Optimal Bet Timing**
   - Historical timing pattern analysis
   - Real-time timing recommendations
   - CLV forecasting

4. **CLV Tracking Enhancement**
   - Forecasting expected CLV
   - Sharp/steam signal integration

## Components

### Core Modules (`src/market_analysis/`)

| Module | Purpose |
|--------|---------|
| `model_drift.py` | Monitor calibration (Brier, ECE, accuracy) over time |
| `performance_gap.py` | Compare backtest to live performance |
| `market_efficiency.py` | Sharp vs public analysis, edge decay |
| `timing_analysis.py` | Historical timing patterns |
| `bet_timing_advisor.py` | Real-time timing guidance |

### Scripts (`scripts/`)

| Script | Frequency | Purpose |
|--------|-----------|---------|
| `monitor_model_drift.py` | Daily | Check for calibration drift |
| `analyze_backtest_gap.py` | Weekly | Detect overfitting |
| `calculate_sharp_divergences.py` | Hourly | Track Pinnacle divergence |
| `track_edge_decay.py` | Daily | Log edge decay patterns |
| `analyze_market_efficiency.py` | Weekly | Sharp signal performance |
| `analyze_optimal_timing.py` | Weekly | Timing windows report |

## Quick Start

### 1. Set Up Cron Jobs

```bash
# Hourly - Sharp divergences
0 * * * * cd /path/to/nbamodels && python scripts/calculate_sharp_divergences.py

# Daily - Model drift check (midnight)
0 0 * * * cd /path/to/nbamodels && python scripts/monitor_model_drift.py

# Daily - Edge decay tracking (6 AM)
0 6 * * * cd /path/to/nbamodels && python scripts/track_edge_decay.py

# Weekly - Sunday 8 AM: Backtest gap
0 8 * * 0 cd /path/to/nbamodels && python scripts/analyze_backtest_gap.py

# Weekly - Monday 8 AM: Market efficiency
0 8 * * 1 cd /path/to/nbamodels && python scripts/analyze_market_efficiency.py

# Weekly - Monday 9 AM: Optimal timing
0 9 * * 1 cd /path/to/nbamodels && python scripts/analyze_optimal_timing.py
```

### 2. Run Initial Analysis

```bash
# Check model calibration
python scripts/monitor_model_drift.py

# Analyze optimal timing patterns
python scripts/analyze_optimal_timing.py --days 90

# Check market efficiency
python scripts/analyze_market_efficiency.py --days 90
```

### 3. Integrate Timing Advisor

```python
from src.market_analysis.bet_timing_advisor import BetTimingAdvisor

advisor = BetTimingAdvisor()

rec = advisor.get_timing_recommendation(
    game_id="abc123",
    bet_type="spread",
    bet_side="home",
    current_edge=0.08,
    commence_time="2025-01-15T19:00:00",
)

print(f"Action: {rec.action}")  # 'place_now', 'wait', or 'avoid'
print(f"Reasons: {rec.reasons}")
```

## Usage Examples

### Model Drift Monitoring

```python
from src.market_analysis.model_drift import ModelDriftMonitor

monitor = ModelDriftMonitor()

# Save daily calibration snapshot
snapshot = monitor.save_calibration_snapshot(
    model_type='spread',
    strategy_type='primary',
)

print(f"Brier Score: {snapshot.brier_score:.4f}")
print(f"ECE: {snapshot.ece:.4f}")

# Detect drift
alerts = monitor.detect_calibration_drift(
    model_type='spread',
    strategy_type='primary',
)

for alert in alerts:
    print(f"{alert.severity}: {alert.metric}")
    print(f"  {alert.recommendation}")
```

### Performance Gap Analysis

```python
from src.market_analysis.performance_gap import BacktestLiveGapAnalyzer

analyzer = BacktestLiveGapAnalyzer()

# Compare backtest to last 90 days of live trading
result = analyzer.calculate_gap(
    strategy_type='primary',
    live_days=90,
)

print(f"Backtest ROI: {result.backtest_roi:+.2%}")
print(f"Live ROI: {result.live_roi:+.2%}")
print(f"Gap: {result.gap:+.2%}")
print(f"Overfitting: {result.is_overfitting}")

# Breakdown by bet type
for bet_type, metrics in result.by_bet_type.items():
    print(f"{bet_type}: Gap {metrics.gap:+.2%}")
```

### Market Efficiency Analysis

```python
from src.market_analysis.market_efficiency import (
    MarketEfficiencyAnalyzer,
    TimeToEfficiencyTracker,
)

analyzer = MarketEfficiencyAnalyzer()

# Calculate Pinnacle divergence
div = analyzer.calculate_pinnacle_divergence(
    game_id="abc123",
    bet_type="spread",
)

if div:
    print(f"Divergence: {div.divergence:.2f} points")
    print(f"Sharp side: {div.sharp_side}")

# Analyze sharp signal performance
perf = analyzer.analyze_sharp_signal_performance('sharp_aligned')
print(f"Win Rate: {perf.win_rate:.1%}")
print(f"ROI: {perf.roi:+.2%}")
print(f"Confidence: {perf.confidence_score:.2f}")

# Track edge decay
tracker = TimeToEfficiencyTracker()
avg_time = tracker.get_avg_time_to_efficiency(bet_type='spread')
print(f"Avg time to efficiency: {avg_time:.1f} hours")
```

### Timing Analysis

```python
from src.market_analysis.timing_analysis import (
    TimingWindowsAnalyzer,
    HistoricalTimingAnalyzer,
)

windows = TimingWindowsAnalyzer()

# Find optimal windows
optimal = windows.find_optimal_windows(bet_type='spread')

for window in optimal:
    if window.hours_before_game:
        print(f"Optimal: {window.hours_before_game}h before game")
        print(f"  Avg CLV: {window.avg_clv:+.3%}")

# Historical analysis
historical = HistoricalTimingAnalyzer()

# Counterfactual: what if bet placed at different time?
counterfactual = historical.counterfactual_analysis('bet_id_123')
print(f"Actual CLV: {counterfactual['actual_timing']['clv']:+.3%}")
print(f"Best CLV: {counterfactual['optimal']['clv']:+.3%}")

# Estimate timing value
value = historical.timing_value_estimate(lookback_days=90)
print(f"Avg CLV improvement from perfect timing: {value:+.3%}")
```

### Bet Timing Advisor

```python
from src.market_analysis.bet_timing_advisor import BetTimingAdvisor

advisor = BetTimingAdvisor()

rec = advisor.get_timing_recommendation(
    game_id="abc123",
    bet_type="spread",
    bet_side="home",
    current_edge=0.08,
    commence_time="2025-01-15T19:00:00",
)

if rec.action == 'place_now':
    print("✓ Place bet now")
    print(f"  Expected CLV: {rec.expected_clv_now:+.3%}")

elif rec.action == 'wait':
    print(f"⏳ Wait {rec.suggested_wait_hours:.1f} hours")
    print(f"  Expected improvement: {rec.expected_clv_optimal - rec.expected_clv_now:+.3%}")

elif rec.action == 'avoid':
    print("✗ Avoid this bet")

for reason in rec.reasons:
    print(f"  - {reason}")
```

### CLV Forecasting

```python
from src.bet_tracker import forecast_clv

expected_clv = forecast_clv(
    game_id="abc123",
    bet_type="spread",
    bet_side="home",
)

print(f"Expected CLV: {expected_clv:+.3%}")

if expected_clv < -0.02:
    print("Warning: Poor expected CLV")
```

## Cron Schedule

Recommended cron schedule for automated monitoring:

```bash
# ============ HOURLY ============
# Line snapshots (existing)
0 * * * * python scripts/collect_line_snapshots.py

# Sharp divergences
0 * * * * python scripts/calculate_sharp_divergences.py

# ============ EVERY 15 MIN ============
# Opening lines (existing)
*/15 * * * * python scripts/capture_opening_lines.py

# Closing lines (existing)
*/15 * * * * python scripts/capture_closing_lines.py

# ============ DAILY ============
# Model drift check (midnight)
0 0 * * * python scripts/monitor_model_drift.py

# Edge decay tracking (6 AM, after games settle)
0 6 * * * python scripts/track_edge_decay.py

# ============ WEEKLY ============
# Sunday 8 AM - Backtest gap analysis
0 8 * * 0 python scripts/analyze_backtest_gap.py

# Monday 8 AM - Market efficiency
0 8 * * 1 python scripts/analyze_market_efficiency.py

# Monday 9 AM - Optimal timing
0 9 * * 1 python scripts/analyze_optimal_timing.py
```

## Database Schema

### New Tables

#### `calibration_snapshots`
Stores daily model calibration metrics for drift detection.

```sql
CREATE TABLE calibration_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL,
    model_type TEXT,
    strategy_type TEXT,
    brier_score REAL,
    log_loss REAL,
    ece REAL,
    mce REAL,
    accuracy REAL,
    n_samples INTEGER,
    prob_bucket_stats TEXT,  -- JSON
    UNIQUE(snapshot_date, model_type, strategy_type)
);
```

#### `backtest_results`
Stores backtest run results for gap analysis.

```sql
CREATE TABLE backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    run_date TEXT NOT NULL,
    strategy_type TEXT,
    start_date TEXT,
    end_date TEXT,
    total_bets INTEGER,
    win_rate REAL,
    roi REAL,
    sharpe_ratio REAL,
    max_drawdown REAL,
    parameters TEXT  -- JSON
);
```

#### `edge_decay_tracking`
Tracks how edges decay from detection to game start.

```sql
CREATE TABLE edge_decay_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    bet_type TEXT NOT NULL,
    bet_side TEXT NOT NULL,
    initial_edge REAL,
    initial_edge_time TEXT,
    edge_at_1hr REAL,
    edge_at_4hr REAL,
    edge_at_12hr REAL,
    closing_edge REAL,
    time_to_half_edge_hrs REAL,
    time_to_zero_edge_hrs REAL,
    was_correct_side BOOLEAN,
    UNIQUE(game_id, bet_type, bet_side)
);
```

#### `sharp_divergences`
Records Pinnacle vs retail book divergences.

```sql
CREATE TABLE sharp_divergences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    snapshot_time TEXT NOT NULL,
    bet_type TEXT NOT NULL,
    pinnacle_line REAL,
    pinnacle_odds INTEGER,
    retail_consensus_line REAL,
    retail_consensus_odds INTEGER,
    divergence REAL,
    sharp_side TEXT,
    UNIQUE(game_id, snapshot_time, bet_type)
);
```

### New Columns in `bets` Table

```sql
ALTER TABLE bets ADD COLUMN sharp_aligned BOOLEAN;
ALTER TABLE bets ADD COLUMN steam_detected BOOLEAN;
ALTER TABLE bets ADD COLUMN rlm_detected BOOLEAN;
```

## API Reference

See individual module docstrings for complete API documentation:

- `ModelDriftMonitor` - src/market_analysis/model_drift.py:68
- `BacktestLiveGapAnalyzer` - src/market_analysis/performance_gap.py:56
- `MarketEfficiencyAnalyzer` - src/market_analysis/market_efficiency.py:80
- `TimeToEfficiencyTracker` - src/market_analysis/market_efficiency.py:366
- `TimingWindowsAnalyzer` - src/market_analysis/timing_analysis.py:32
- `HistoricalTimingAnalyzer` - src/market_analysis/timing_analysis.py:246
- `BetTimingAdvisor` - src/market_analysis/bet_timing_advisor.py:37

## Best Practices

1. **Data Collection First**: Run cron jobs for 2-4 weeks before using analysis
2. **Gradual Integration**: Start with monitoring, then add timing advisor
3. **Threshold Tuning**: Adjust drift thresholds based on your model
4. **Weekly Reviews**: Check all reports every Monday
5. **Track Impact**: Compare CLV before/after timing optimization

## Troubleshooting

See `docs/MARKET_ANALYSIS_INTEGRATION.md` for detailed troubleshooting guide.

## Next Steps

1. ✅ Set up cron jobs for data collection
2. ✅ Run initial analysis to establish baseline
3. ✅ Review weekly reports for 2 weeks
4. ✅ Integrate timing advisor into betting pipeline
5. ✅ Monitor impact on CLV and ROI
