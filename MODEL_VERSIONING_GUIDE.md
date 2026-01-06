# Model Versioning & A/B Testing System

## Overview

This system provides comprehensive model versioning with champion/challenger framework, automatic feature backtesting, performance tracking, and systematic feature engineering.

**Key Features:**
- ✅ Semantic versioning for all models (v1.0.0, v1.1.0, etc.)
- ✅ Champion/challenger A/B testing with auto-promotion
- ✅ Automated feature backtesting pipeline
- ✅ Performance monitoring and degradation alerts
- ✅ Systematic feature engineering workflow

---

## Quick Start

### 1. Initialize Databases

```bash
python scripts/versioning/init_databases.py
```

This creates three SQLite databases:
- `data/versioning/model_registry.db` - Model versions and comparisons
- `data/versioning/feature_registry.db` - Feature experiments
- `data/versioning/performance_history.db` - Performance tracking

### 2. Train Models with Versioning

```bash
# Standard retraining with versioning enabled
python scripts/retrain_models.py

# Force retrain all models
python scripts/retrain_models.py --force

# Disable auto-promotion (manual approval required)
python scripts/retrain_models.py --no-auto-promote

# Legacy mode (disable versioning)
python scripts/retrain_models.py --no-versioning
```

### 3. View Model Versions

```bash
# List all model versions
python scripts/versioning/list_models.py

# List specific model's history
python scripts/versioning/list_models.py --model-name spread_model
```

---

## Champion/Challenger Framework

### How It Works

1. **New models are registered as challengers** when trained
2. **Automatic comparison** runs on next retraining cycle
3. **Auto-promotion** if challenger meets criteria:
   - p-value < 0.05 (statistically significant)
   - ROI improvement > 0%
   - Minimum 100 bets tested
   - Sharpe ratio doesn't degrade

### Manual Comparison

```bash
# Compare champion vs latest challenger
python scripts/versioning/compare_models.py --model-name spread_model

# Compare specific models
python scripts/versioning/compare_models.py \
  --champion-id abc123 \
  --challenger-id def456
```

### Promotion Process

**Automatic Promotion:**
```python
from src.versioning import ChampionChallengerFramework

framework = ChampionChallengerFramework()
comparison = framework.auto_promote_if_better("spread_model")

if comparison.recommendation == "promote":
    print(f"Promoted! {comparison.reason}")
```

**Manual Promotion:**
```python
from src.versioning import ModelRegistry

registry = ModelRegistry()
registry.promote_to_champion(
    model_id="challenger-uuid",
    reason="Manual review: outperforms by 2.3% ROI"
)
```

---

## Feature Experimentation

### Proposing a Feature

```python
from src.experimentation import FeatureRegistry

registry = FeatureRegistry()
feature_id = registry.propose_feature(
    name="ref_crew_foul_bias",
    category="referee",
    description="Referee crew's tendency to call fouls on home vs away",
    hypothesis="Certain referee crews favor home teams in foul calling"
)
```

### Testing a Feature

```bash
python scripts/experimentation/test_feature.py \
  --feature-name ref_crew_foul_bias \
  --category referee \
  --hypothesis "Ref crews show home/away bias"
```

### Batch Testing

```python
from src.experimentation import FeatureExperiment

experiment = FeatureExperiment()

# Test all pending features
results = experiment.batch_test_features()

for result in results:
    if result.decision == "accept":
        print(f"✓ {result.feature_name}: +{result.roi_lift:.2%} ROI")
    else:
        print(f"✗ {result.feature_name}: {result.decision_reason}")
```

### Feature Acceptance Criteria

**Auto-accepted if:**
- p-value < 0.05
- ROI lift ≥ 0.5%
- Minimum 100 bets tested

---

## Performance Monitoring

### Recording Daily Performance

```python
from src.monitoring.model_health import ModelHealthMonitor, DailyMetrics
from datetime import date

monitor = ModelHealthMonitor()

metrics = DailyMetrics(
    predictions_count=150,
    bets_placed=45,
    accuracy=0.634,
    roi=0.058,
    pnl=287.50,
    win_rate=0.622,
    roi_7d=0.055,
    roi_30d=0.061,
    sharpe_30d=0.85
)

monitor.record_daily_performance(
    model_id="champion-uuid",
    model_name="spread_model",
    date=date.today(),
    metrics=metrics
)
```

### Checking for Degradation

```python
alerts = monitor.check_for_degradation(
    model_id="champion-uuid",
    lookback_days=30
)

for alert in alerts:
    print(f"{alert.severity.upper()}: {alert.message}")
    # e.g., "WARNING: Accuracy below threshold: 51.2%"
```

### Alert Thresholds

- **Accuracy < 52%** → Warning
- **ROI < -5%** → Critical
- **Sharpe < 0** → Warning
- **Trend decline > 5%** over 14 days → Warning

---

## Model Versioning Workflow

### Version Numbering

Following semantic versioning:
- **MAJOR.MINOR.PATCH** (e.g., 1.2.0)
- **MAJOR**: Breaking changes (incompatible features)
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes, retraining

### Automatic Version Bumping

```python
from src.versioning import get_next_version, BumpType

# Auto-infer from description
version = get_next_version(
    current_version="1.2.0",
    description="Added referee features"
)
# Returns: Version(1, 3, 0)  # MINOR bump

# Explicit bump type
version = get_next_version(
    current_version="1.2.0",
    bump_type=BumpType.PATCH
)
# Returns: Version(1, 2, 1)
```

### Model Lifecycle

```
proposed → training → challenger → [comparison] → champion
                                              ↓
                                          archived
```

---

## Integration with Existing Code

### Retraining Pipeline

The `RetrainingPipeline` in `scripts/retrain_models.py` now:
1. Trains all models as usual
2. Registers each as a **challenger** with auto-incremented version
3. Compares challengers to current champions
4. **Auto-promotes** if performance criteria met
5. Archives old champions

### Feature Builders

Feature experiments integrate with existing `GameFeatureBuilder`:
1. Baseline model uses current feature set
2. Experiment model adds candidate feature
3. Both trained and backtested identically
4. Results compared statistically

---

## Database Schema

### Model Registry

**models** table:
- `model_id`, `model_name`, `version`, `status`
- `model_path`, `metadata_path`
- `created_at`, `description`, `parent_version`
- `train_start_date`, `train_end_date`, `games_count`, `feature_count`

**model_metrics** table:
- Performance snapshots (training, validation, backtest, live)
- `accuracy`, `auc_roc`, `roi`, `sharpe_ratio`, `max_drawdown`

**model_comparisons** table:
- Champion vs challenger A/B test results
- `roi_difference`, `p_value`, `is_significant`, `recommendation`

**promotion_history** table:
- Audit trail of all promotions

### Feature Registry

**feature_candidates** table:
- `feature_name`, `category`, `hypothesis`, `status`

**feature_experiments** table:
- `baseline_roi`, `experiment_roi`, `roi_lift`
- `p_value`, `is_significant`, `decision`

### Performance History

**daily_performance** table:
- Daily metrics per model
- `roi_7d`, `roi_30d`, `sharpe_30d`

**alerts** table:
- Performance degradation alerts

---

## CLI Reference

### Versioning Commands

```bash
# Initialize databases (run once)
python scripts/versioning/init_databases.py

# List all model versions
python scripts/versioning/list_models.py

# List specific model
python scripts/versioning/list_models.py --model-name spread_model

# Compare models
python scripts/versioning/compare_models.py --model-name spread_model
```

### Experimentation Commands

```bash
# Test a single feature
python scripts/experimentation/test_feature.py \
  --feature-name feature_name \
  --category category \
  --hypothesis "Why this might help"
```

### Retraining Commands

```bash
# Standard (with versioning and auto-promotion)
python scripts/retrain_models.py

# Disable auto-promotion
python scripts/retrain_models.py --no-auto-promote

# Legacy mode (no versioning)
python scripts/retrain_models.py --no-versioning

# Dry run
python scripts/retrain_models.py --dry-run
```

---

## Python API Examples

### Working with Model Registry

```python
from src.versioning import ModelRegistry, ModelMetrics
from datetime import date

registry = ModelRegistry()

# Get current champion
champion = registry.get_champion("spread_model")
print(f"Champion: v{champion.version}")

# Get all challengers
challengers = registry.get_challengers("spread_model")
print(f"Challengers: {len(challengers)}")

# Get model history
history = registry.get_model_history("spread_model", limit=10)
for model in history:
    print(f"{model.version} - {model.status} ({model.created_at})")

# Register a new model
model_id = registry.register_model(
    model_name="spread_model",
    version="1.3.0",
    model_path="models/spread_model_v1.3.0.pkl",
    description="Added news sentiment features",
    parent_version="1.2.0",
    status="challenger",
    games_count=7500,
    feature_count=85
)

# Record metrics
metrics = ModelMetrics(
    accuracy=0.635,
    auc_roc=0.618,
    roi=0.061,
    win_rate=0.623,
    sharpe_ratio=0.88
)
registry.record_metrics(model_id, metrics, metric_type="backtest")
```

### Champion/Challenger Testing

```python
from src.versioning import ChampionChallengerFramework

framework = ChampionChallengerFramework()

# Compare and auto-promote
comparison = framework.auto_promote_if_better("spread_model")

if comparison:
    print(f"ROI Diff: {comparison.roi_difference:+.2%}")
    print(f"P-value: {comparison.p_value:.4f}")
    print(f"Winner: {comparison.winner}")
    print(f"Recommendation: {comparison.recommendation}")
```

### Feature Experiments

```python
from src.experimentation import FeatureExperiment, FeatureRegistry

# Propose a feature
registry = FeatureRegistry()
feature_id = registry.propose_feature(
    name="pace_variance_10g",
    category="team_stats",
    hypothesis="Pace variance indicates adaptability"
)

# Test it
experiment = FeatureExperiment()
result = experiment.test_feature("pace_variance_10g")

print(f"ROI Lift: {result.roi_lift:+.2%}")
print(f"Decision: {result.decision}")
print(f"Reason: {result.decision_reason}")

# Check feature status
feature = registry.get_feature_by_id(feature_id)
print(f"Status: {feature.status}")  # 'accepted', 'rejected', or 'testing'
```

---

## Best Practices

### Model Versioning

1. **Descriptive descriptions**: "Added referee features" not just "update"
2. **Semantic versioning**: Use MINOR for features, PATCH for retraining
3. **Keep backups**: Old champions are archived, not deleted
4. **Document changes**: Use description field to explain what changed

### Feature Testing

1. **One feature at a time**: Test features individually for clear attribution
2. **Strong hypothesis**: Explain why the feature should help
3. **Sufficient data**: Need 100+ bets minimum for reliable results
4. **Iterate**: Start with simple version, refine if accepted

### Performance Monitoring

1. **Daily recording**: Record metrics every day for trend detection
2. **Set thresholds**: Adjust AlertConfig based on your risk tolerance
3. **Review alerts**: Check alerts regularly, acknowledge when resolved
4. **Historical analysis**: Use trend data to understand seasonality

---

## Troubleshooting

### Database Locked Error
```python
# Close connections properly
conn.close()

# Or use context manager
with get_db_manager().get_connection('model_registry') as conn:
    # Your code here
    pass
```

### No Champions Found
```bash
# First model? Manually promote first model to champion
python -c "from src.versioning import ModelRegistry; \
  r = ModelRegistry(); \
  r.promote_to_champion('model-id', 'Initial champion')"
```

### Feature Test Fails
- Ensure baseline model exists (need a champion)
- Check that test data is available
- Verify feature name doesn't already exist

---

## Future Enhancements

Planned features:
- [ ] Streamlit dashboard integration for Model Health tab
- [ ] Discord alerts for model promotions and degradations
- [ ] Scheduled batch feature testing (cron job)
- [ ] Model performance reports (PDF/HTML)
- [ ] A/B test visualizations
- [ ] Feature correlation analysis
- [ ] Multi-model ensembles with versioning

---

## Architecture

```
src/
  versioning/
    db.py                    # SQLite database management
    model_registry.py        # Model version tracking
    version.py              # Semantic versioning utilities
    champion_challenger.py  # A/B testing framework

  experimentation/
    feature_registry.py     # Feature candidate tracking
    feature_experiment.py   # Automated feature testing

  monitoring/
    model_health.py         # Performance tracking & alerts

scripts/
  versioning/
    init_databases.py       # Database initialization
    list_models.py         # View model versions
    compare_models.py      # Compare A/B tests

  experimentation/
    test_feature.py        # Test single feature

  retrain_models.py        # Main training pipeline (updated)

data/
  versioning/
    model_registry.db
    feature_registry.db
    performance_history.db
```

---

## Questions?

For issues or questions:
1. Check this guide first
2. Review the code documentation (docstrings)
3. Check database contents with SQLite viewer
4. Review logs in `logs/retrain.log`

**Database Inspection:**
```bash
sqlite3 data/versioning/model_registry.db "SELECT * FROM models;"
sqlite3 data/versioning/feature_registry.db "SELECT * FROM feature_candidates;"
```
