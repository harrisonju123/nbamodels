# Production Models

This directory contains the production-ready trained models for NBA betting predictions.

**Last Updated:** January 9, 2026 (Consolidation)

---

## Canonical Model Files

### Spread Betting

**File:** `spread_model_calibrated.pkl` ✅ **CANONICAL**
- **Algorithm:** XGBoost Classifier with isotonic calibration
- **Performance:** 42.24% ROI, 75.4% win rate (934 bets, 2020-2024)
- **Size:** 321 KB
- **Format:** Pickled dict with keys: `model`, `feature_columns`
- **Usage:** Daily betting pipeline (`scripts/daily_betting_pipeline.py`)

**Note:** `spread_model.pkl` was a duplicate (identical MD5 hash) and was removed on 2026-01-09.
Always use `spread_model_calibrated.pkl`.

### Player Props

**Directory:** `player_props/`

| Model | Prop Type | ROI | Win Rate | Size |
|-------|-----------|-----|----------|------|
| `pts_model.pkl` | Points | 27.1% | 66.6% | 255 KB |
| `reb_model.pkl` | Rebounds | 55.8% | 81.6% | 249 KB |
| `ast_model.pkl` | Assists | 89.4% | 99.2% | 250 KB |
| `3pm_model.pkl` | 3-Pointers Made | 55.0% | 81.2% | 250 KB |

See main project README for full documentation.
