# Market Analysis System - Integration Guide

This document describes how to integrate the market analysis components with your betting pipeline.

## Overview

The market analysis system provides:
- **Model Drift Monitoring** - Detect calibration degradation
- **Performance Gap Analysis** - Compare backtest vs live
- **Market Efficiency Analysis** - Sharp vs public, edge decay
- **Optimal Bet Timing** - Historical patterns & real-time guidance

## Integration with Daily Betting Pipeline

### Option 1: Add Timing Advisor to Pipeline

Add the following to `scripts/daily_betting_pipeline.py`:

```python
from src.market_analysis.bet_timing_advisor import BetTimingAdvisor
from src.market_analysis.timing_analysis import HistoricalTimingAnalyzer

# Initialize timing advisor (add near top of main())
historical_analyzer = HistoricalTimingAnalyzer()
timing_advisor = BetTimingAdvisor(historical_analyzer=historical_analyzer)

# After evaluating games and before logging bets:
for signal in actionable_signals:
    # Get timing recommendation
    timing_rec = timing_advisor.get_timing_recommendation(
        game_id=signal.game_id,
        bet_type='spread',  # or signal.bet_type
        bet_side=signal.bet_side,
        current_edge=signal.model_edge,
        commence_time=game_data['commence_time'],
    )

    logger.info(f"Timing recommendation for {signal.game_id}: {timing_rec.action}")
    logger.info(f"  Reasons: {', '.join(timing_rec.reasons)}")
    logger.info(f"  Confidence: {timing_rec.confidence:.2f}")

    if timing_rec.action == 'place_now':
        # Log bet normally
        log_bet_recommendation(signal, game_data, ...)

    elif timing_rec.action == 'wait':
        # Schedule for later (implement scheduling logic)
        logger.info(f"  Waiting {timing_rec.suggested_wait_hours:.1f} hours")
        # TODO: Implement bet scheduling

    elif timing_rec.action == 'avoid':
        logger.warning(f"  Avoiding bet: {', '.join(timing_rec.reasons)}")
        continue
```

### Option 2: Add CLI Flag for Timing Advisor

Add argument to daily_betting_pipeline.py:

```python
parser.add_argument(
    "--use-timing-advisor",
    action="store_true",
    help="Use timing advisor for optimal bet placement",
)

# In main():
if args.use_timing_advisor:
    timing_advisor = BetTimingAdvisor()
    # ... timing logic
```

### Option 3: Use Timing Analysis Reports

Run timing analysis weekly to understand patterns:

```bash
# Weekly timing analysis
python scripts/analyze_optimal_timing.py --days 90 --export timing_report.json

# Manually review optimal windows and adjust bet placement timing
```

## Cron Schedule for Market Analysis

Add these to your crontab:

```bash
# Hourly - Calculate sharp divergences (with line snapshots)
0 * * * * cd /path/to/nbamodels && python scripts/calculate_sharp_divergences.py

# Daily - Monitor model drift (midnight)
0 0 * * * cd /path/to/nbamodels && python scripts/monitor_model_drift.py

# Daily - Track edge decay (after games settle, 6 AM)
0 6 * * * cd /path/to/nbamodels && python scripts/track_edge_decay.py

# Weekly - Backtest vs live gap analysis (Sunday 8 AM)
0 8 * * 0 cd /path/to/nbamodels && python scripts/analyze_backtest_gap.py

# Weekly - Market efficiency report (Monday 8 AM)
0 8 * * 1 cd /path/to/nbamodels && python scripts/analyze_market_efficiency.py

# Weekly - Optimal timing report (Monday 9 AM)
0 9 * * 1 cd /path/to/nbamodels && python scripts/analyze_optimal_timing.py
```

## Populating Sharp Signal Columns

To populate `sharp_aligned`, `steam_detected`, `rlm_detected` columns when logging bets:

```python
from src.betting.market_signals import SteamMoveDetector, RLMDetector
from src.market_analysis.market_efficiency import MarketEfficiencyAnalyzer

steam_detector = SteamMoveDetector()
rlm_detector = RLMDetector()
efficiency_analyzer = MarketEfficiencyAnalyzer()

# When logging bet:
steam_move = steam_detector.detect_steam_move(game_id, bet_type)
sharp_div = efficiency_analyzer.calculate_pinnacle_divergence(game_id, bet_type)

log_bet(
    game_id=game_id,
    # ... other params ...
    sharp_aligned=(sharp_div.sharp_side == bet_side) if sharp_div else False,
    steam_detected=steam_move is not None,
    rlm_detected=False,  # Implement RLM detection
)
```

## Using CLV Forecasting

```python
from src.bet_tracker import forecast_clv

# Before placing bet:
expected_clv = forecast_clv(
    game_id=game_id,
    bet_type='spread',
    bet_side='home',
)

logger.info(f"Expected CLV: {expected_clv:+.2%}")

# Use in decision making:
if expected_clv < -0.02:  # Expecting -2% CLV or worse
    logger.warning("Poor expected CLV, consider skipping")
```

## Model Drift Alerts

Set up email/Slack notifications for drift alerts:

```python
# In scripts/monitor_model_drift.py:
import requests

def send_alert(message):
    # Slack webhook example
    webhook_url = "YOUR_SLACK_WEBHOOK_URL"
    requests.post(webhook_url, json={"text": message})

# After detecting drift:
if any_alerts:
    send_alert(f"⚠️ Model drift detected! {len(alerts)} alerts")
```

## Performance Gap Monitoring

Add to weekly review process:

```bash
# Sunday: Run backtest on recent data
python scripts/backtest_strategy.py --days 30 > backtest_latest.json

# Store backtest results in DB
python scripts/store_backtest_results.py backtest_latest.json

# Run gap analysis
python scripts/analyze_backtest_gap.py --days 30
```

## Best Practices

1. **Start with monitoring**: Run analysis scripts manually for 2-4 weeks to understand patterns
2. **Gradual integration**: Add timing advisor to a small subset of bets first
3. **Track impact**: Compare CLV before/after timing advisor integration
4. **Review weekly**: Check all analysis reports every Monday
5. **Alert thresholds**: Adjust drift thresholds based on your model's characteristics

## Troubleshooting

### No data in analysis reports
- Ensure cron jobs are running for data collection
- Check that `sharp_aligned`, `steam_detected` columns are being populated
- Verify bets have CLV data populated (run `populate_clv_data.py`)

### Timing advisor always says "place_now"
- Need more historical data (run for 2+ weeks)
- Check that multi-snapshot CLV columns are populated
- Review optimal timing analysis to see if there are actual timing patterns

### Model drift alerts are too sensitive
- Adjust thresholds in `ModelDriftMonitor`:
  ```python
  monitor = ModelDriftMonitor(
      baseline_window=300,  # Increase for more stable baseline
      alert_window=100,     # Increase for less sensitivity
  )
  ```

## Next Steps

1. Set up cron jobs for automated data collection
2. Run manual analysis for 2 weeks to build historical baseline
3. Review reports and identify patterns
4. Integrate timing advisor into pipeline
5. Monitor impact on CLV and ROI
