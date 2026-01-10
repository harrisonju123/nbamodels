# Archived Models

These models were archived on 2026-01-06 after comprehensive backtesting revealed they were either superseded by better models or failed validation.

## Archived Models

| Model | Reason | Backtest Result |
|-------|--------|-----------------|
| `dual_model.pkl` | **Failed validation** | -4% ROI, 50.4% win rate (2024 season) |
| `point_spread_model.pkl` | **Superseded** | Replaced by spread_model_calibrated.pkl |
| `point_spread_model_tuned.pkl` | **Superseded** | Older tuned version, replaced |
| `spread_model_tuned.pkl` | **Superseded** | Older tuned version, replaced |
| `stacked_ensemble.pkl` | **Not validated** | No rigorous backtest results |
| `advanced_ensemble.pkl` | **Not validated** | No rigorous backtest results |
| `lgb_tuned.pkl` | **Component only** | Individual model, not used standalone |
| `xgb_tuned.pkl` | **Component only** | Individual model, not used standalone |
| `totals_model_tuned.pkl` | **Superseded** | Replaced by totals_model.pkl |

## Current Production Models

See `models/` directory for validated production models:

- `spread_model_calibrated.pkl` - **42.24% ROI, 75.4% win rate** (2020-2024 rigorous backtest)
- `spread_model.pkl` - Production copy (loaded by daily pipeline)
- `nba_model_retrained.pkl` - Latest retrained model (Jan 4, 2026)
- `totals_model.pkl` - Totals betting model
- `player_props/*.pkl` - Player props models (27-89% ROI)

## Why Archive Instead of Delete?

These models are preserved in case:
1. We need to reference old approaches
2. Future research requires comparison
3. Regulatory/audit requirements

To permanently delete: `rm -rf models/archive/`
