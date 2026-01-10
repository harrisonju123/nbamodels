# NBA Betting Models - Production

This directory contains validated, production-ready models for NBA betting.

## Spread Models (Main Production)

### ✅ spread_model_calibrated.pkl
**Best performing spread model**
- **Model Type:** XGBoost Classifier (calibrated with isotonic regression)
- **Performance (Rigorous Backtest 2020-2024):**
  - ROI: 42.24% (95% CI: 36.99%, 47.47%)
  - Win Rate: 75.4% (95% CI: 72.5%, 78.0%)
  - Sharpe Ratio: 8.24
  - Bets: 934 across 37 walk-forward folds
- **Features:** 94 features (rolling stats, lineup impact, travel, rest, matchup history)
- **Usage:** Main production model for spread betting

### ✅ spread_model.pkl
**Current production copy** (loaded by daily_betting_pipeline.py)
- Copy of spread_model_calibrated.pkl
- Used by: `scripts/daily_betting_pipeline.py`

## Other Models

### ✅ nba_model_retrained.pkl
- Latest retrained model (Jan 4, 2026)
- 63% accuracy, 0.62 AUC
- 94 features with alternative data

### ✅ totals_model.pkl
- Over/under totals betting
- Not yet rigorously backtested

## Player Props Models (Best Performers!)

Located in `player_props/` directory:

| Model | Prop | ROI | Win Rate | R² | Bets Validated |
|-------|------|-----|----------|-----|----------------|
| `ast_model.pkl` | Assists | **89.4%** | **99.2%** | 0.999 | 2,126 |
| `reb_model.pkl` | Rebounds | 55.8% | 81.6% | 0.920 | 15,182 |
| `3pm_model.pkl` | 3-Pointers | 55.0% | 81.2% | 0.764 | 13,420 |
| `pts_model.pkl` | Points | 27.1% | 66.6% | 0.921 | 15,374 |

**Total Props Validated:** 46,102 bets

## Model Selection Guide

**For spread betting:** Use `spread_model_calibrated.pkl` (or `spread_model.pkl`)
- Validated 42% ROI over 5 years
- Use EdgeStrategy with 7-point edge threshold
- 20% Kelly fraction recommended

**For player props:** Use all 4 player props models
- Assists model has near-perfect accuracy (99.2% R²)
- All models show strong positive ROI

**Avoid:** Models in `archive/` directory (failed validation)

## Retraining Schedule

- **Spread models:** Retrain monthly with latest data
- **Player props:** Retrain weekly during season
- **Validation:** Always run rigorous backtest before production deployment

## See Also

- Backtest results: `data/backtest/rigorous/`
- Model metadata: `*.metadata.json` files
- Archived models: `archive/README.md`
