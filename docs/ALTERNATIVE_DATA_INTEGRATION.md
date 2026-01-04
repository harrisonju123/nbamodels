# Alternative Data Integration Guide

## Overview

Your NBA betting models now support **11 additional alternative data features** across 3 categories:

- **5 News Features**: Media attention volume and recency
- **6 Sentiment Features**: Public betting sentiment from social media
- **5 Referee Features**: Crew tendencies for totals, pace, and bias (when available)

These features are **already integrated** into your feature pipeline and will be automatically included when you retrain models.

---

## Current Integration Status

### ✅ Already Completed

1. **Feature Builders Created**:
   - `RefereeFeatureBuilder` - src/features/referee_features.py
   - `NewsFeatureBuilder` - src/features/news_features.py
   - `SentimentFeatureBuilder` - src/features/sentiment_features.py

2. **Integrated into GameFeatureBuilder**:
   - Enabled by default (use_referee_features=True, use_news_features=True, use_sentiment_features=True)
   - Graceful degradation (models work even if data unavailable)
   - Batch queries with caching for performance

3. **Data Collection Scripts Created**:
   - `scripts/collect_referees.py` - Collects referee assignments and stats
   - `scripts/collect_news.py` - Scrapes NBA news from RSS feeds
   - `scripts/collect_sentiment.py` - Scrapes Reddit r/sportsbook for sentiment

4. **Pipelines Updated**:
   - Training pipeline (`scripts/retrain_models.py`) - Auto-includes new features
   - Daily pipeline (`scripts/daily_betting_pipeline.py`) - Auto-includes new features

### ⏳ Next Steps (What You Need to Do)

1. **Start data collection** (run the cron jobs or scripts manually)
2. **Retrain models** to include the new features in model weights
3. **Monitor feature health** to ensure data quality

---

## Step-by-Step Integration

### Step 1: Verify Data Collection

Check if alternative data is being collected:

```bash
# Check if sentiment data exists
sqlite3 data/bets/bets.db "SELECT COUNT(*) FROM sentiment_scores"

# Check if news data exists
sqlite3 data/bets/bets.db "SELECT COUNT(*) FROM news_articles"

# Check if referee data exists
sqlite3 data/bets/bets.db "SELECT COUNT(*) FROM referee_assignments"
```

**If counts are 0**, you need to run the data collection scripts.

### Step 2: Run Data Collection Scripts

You have 3 options:

#### Option A: Run Manually (One-time)

```bash
# Collect sentiment (scrapes Reddit)
python scripts/collect_sentiment.py

# Collect news (scrapes RSS feeds)
python scripts/collect_news.py

# Collect referee assignments (requires upcoming games)
python scripts/collect_referees.py
```

#### Option B: Add to Crontab (Automated)

Add these lines to your crontab (`crontab -e`):

```bash
# Sentiment: Every 15 minutes during game hours (5-11 PM ET)
*/15 17-23 * * * cd /Users/harrisonju/PycharmProjects/nbamodels && python scripts/collect_sentiment.py >> logs/sentiment.log 2>&1

# News: Every hour
0 * * * * cd /Users/harrisonju/PycharmProjects/nbamodels && python scripts/collect_news.py >> logs/news.log 2>&1

# Referees: Daily at 10 AM
0 10 * * * cd /Users/harrisonju/PycharmProjects/nbamodels && python scripts/collect_referees.py >> logs/referees.log 2>&1
```

#### Option C: Run via Launchd (macOS)

See: docs/LAUNCHD_SETUP.md (if you prefer launchd over cron)

### Step 3: Verify Alternative Data Features

Run the integration test to verify features are working:

```bash
python scripts/test_alternative_data_integration.py
```

**Expected output:**
```
✓ PASS: Baseline Features
✓ PASS: Enhanced Features
✓ PASS: Graceful Degradation
✓ PASS: Feature Comparison

4/4 tests passed
✓ All tests PASSED
```

You should see:
- **11 new features** added to enhanced vs baseline
- News features and sentiment features populated (even if with zeros)
- No errors during feature building

### Step 4: Retrain Models with New Features

Once you have some alternative data collected, retrain your models:

```bash
# Force retrain all models
python scripts/retrain_models.py --force

# Or check if retraining is needed (50+ new games)
python scripts/retrain_models.py
```

**What happens during retraining:**

1. Loads `data/raw/games.parquet`
2. Calls `GameFeatureBuilder()` (includes alternative data features)
3. Builds feature dataset with **all available features**
4. Trains 4 models:
   - dual_model.pkl
   - spread_model.pkl
   - point_spread_model.pkl
   - totals_model.pkl
5. Saves models with `feature_columns` list (includes new features)
6. Updates `models/training_metadata.json`

**Important:** The models will only use features that have data. If alternative data sources are empty, those features will be all zeros (neutral/no signal).

### Step 5: Verify Models Use New Features

Check that trained models include alternative data features:

```bash
python -c "
import pickle

with open('models/spread_model.pkl', 'rb') as f:
    data = pickle.load(f)
    features = data['feature_columns']

    # Check for alternative data features
    ref_features = [f for f in features if 'ref_' in f]
    news_features = [f for f in features if 'news' in f]
    sentiment_features = [f for f in features if 'sentiment' in f]

    print(f'Total features: {len(features)}')
    print(f'Referee features: {len(ref_features)} - {ref_features}')
    print(f'News features: {len(news_features)} - {news_features}')
    print(f'Sentiment features: {len(sentiment_features)} - {sentiment_features}')
"
```

**Expected output (after retraining with alternative data):**
```
Total features: 96
Referee features: 5 - ['ref_crew_total_bias', 'ref_crew_pace_factor', ...]
News features: 5 - ['home_news_volume_24h', 'away_news_volume_24h', ...]
Sentiment features: 6 - ['home_sentiment', 'away_sentiment', ...]
```

### Step 6: Daily Predictions Use New Features

Your daily pipeline will **automatically** use the new features:

```bash
python scripts/daily_betting_pipeline.py
```

**What happens:**
1. Loads retrained `spread_model.pkl` (has new feature_columns)
2. Gets today's games
3. Builds features using `GameFeatureBuilder()` (includes alternative data)
4. Makes predictions with all available features
5. Logs bets to database

---

## Feature Descriptions

### News Features (5 total)

| Feature | Description | Range | Signal |
|---------|-------------|-------|--------|
| `home_news_volume_24h` | Articles mentioning home team | 0-50+ | More news = more public attention |
| `away_news_volume_24h` | Articles mentioning away team | 0-50+ | More news = more public attention |
| `home_news_recency` | Hours since last article about home team | 0-999 | Recent news = current storyline |
| `away_news_recency` | Hours since last article about away team | 0-999 | Recent news = current storyline |
| `news_volume_diff` | Home - Away news volume | -50 to +50 | Attention differential |

**Hypothesis:** Teams with high news volume may be overbet by public (fade opportunity).

**Data source:** RSS feeds from ESPN, NBA.com, Yahoo Sports (free, no API key)

### Sentiment Features (6 total)

| Feature | Description | Range | Signal |
|---------|-------------|-------|--------|
| `home_sentiment` | Public sentiment score for home team | -1 to +1 | Positive = public likes home |
| `away_sentiment` | Public sentiment score for away team | -1 to +1 | Positive = public likes away |
| `home_sentiment_volume` | Number of mentions for home team | 0-1000+ | Volume of public discussion |
| `away_sentiment_volume` | Number of mentions for away team | 0-1000+ | Volume of public discussion |
| `sentiment_diff` | Home - Away sentiment | -2 to +2 | Sentiment differential |
| `sentiment_enabled` | Whether sentiment data available | 0 or 1 | Feature availability flag |

**Hypothesis:** Strong public sentiment indicates overreaction (contrarian opportunity).

**Data source:** Reddit r/sportsbook HTML scraping (free, no API key)

### Referee Features (5 total)

| Feature | Description | Range | Signal |
|---------|-------------|-------|--------|
| `ref_crew_total_bias` | Points above/below league average | -10 to +10 | Affects totals betting |
| `ref_crew_pace_factor` | Pace multiplier (1.0 = average) | 0.9-1.1 | Faster/slower games |
| `ref_crew_over_rate` | Historical over hit rate | 0.4-0.6 | Over/under tendency |
| `ref_crew_home_bias` | Home win rate with this crew | 0.45-0.65 | Home court advantage |
| `ref_crew_size` | Number of refs in crew stats | 0-3 | Data availability |

**Hypothesis:** Referee tendencies create predictable total/pace patterns.

**Data source:** NBA.com official API (free)

---

## Monitoring Alternative Data Health

### Check Feature Coverage

Use the validation script to check if alternative data is being captured:

```bash
python scripts/validate_alternative_data.py
```

**Example output:**
```
✓ Referee: 3/5 features (60%)
✓ News: 5/5 features (100%)
✓ Sentiment: 6/6 features (100%)

✓ Feature health: 14/16 (87.5%)
```

### Check Data Freshness

```bash
# Check when sentiment was last updated
sqlite3 data/bets/bets.db "SELECT MAX(collected_at) FROM sentiment_scores"

# Check when news was last updated
sqlite3 data/bets/bets.db "SELECT MAX(collected_at) FROM news_articles"

# Check when referees were last updated
sqlite3 data/bets/bets.db "SELECT MAX(collected_at) FROM referee_assignments"
```

### Monitor Logs

```bash
# Check sentiment collection logs
tail -f logs/sentiment.log

# Check news collection logs
tail -f logs/news.log

# Check referee collection logs
tail -f logs/referees.log
```

---

## Troubleshooting

### Issue: "No alternative data features found"

**Cause:** Data collection scripts haven't run or databases are empty

**Fix:**
1. Run data collection scripts manually (see Step 2)
2. Verify data exists in database (see Step 1)
3. Check logs for errors: `tail logs/sentiment.log`

### Issue: "Referee features failed to add"

**Cause:** Referee data requires valid game_ids from today's games

**Fix:**
- Referee features only work for actual NBA games (not test data)
- Run `scripts/collect_referees.py` on days with NBA games
- Check that game_id format matches NBA's official format

### Issue: "Feature columns don't match between train and predict"

**Cause:** Model was trained before alternative data was available

**Fix:**
1. Collect alternative data first
2. Retrain models: `python scripts/retrain_models.py --force`
3. New model will have updated feature_columns list

### Issue: "Sentiment values all zero"

**Cause:** No Reddit posts found, or scraping failed

**Fix:**
1. Check if r/sportsbook has recent posts
2. Verify HTML scraping is working: `python scripts/collect_sentiment.py`
3. Check logs: `tail logs/sentiment.log`
4. Reddit may have changed HTML structure (update BeautifulSoup selectors)

---

## Expected Performance Impact

### Before Alternative Data
- **85 features** (core stats, odds, lineup, matchup, elo)
- Accuracy: ~54-56% ATS (estimated from existing models)

### After Alternative Data
- **96 features** (+11 alternative data features)
- Expected improvement: +1-2% accuracy (if features have predictive signal)
- Better on high-profile games (more news/sentiment data)
- Better on games with referee tendencies (totals betting)

**Note:** Alternative data provides **edge cases** rather than universal improvements:
- News volume helps identify overhyped teams
- Sentiment helps spot public overreactions
- Referee data helps with totals and pace predictions

---

## Quick Start Checklist

- [ ] Run `python scripts/collect_sentiment.py` to populate sentiment data
- [ ] Run `python scripts/collect_news.py` to populate news data
- [ ] Run `python scripts/collect_referees.py` to populate referee data
- [ ] Verify data: `python scripts/test_alternative_data_integration.py`
- [ ] Retrain models: `python scripts/retrain_models.py --force`
- [ ] Verify features in model: `python -c "import pickle; print(pickle.load(open('models/spread_model.pkl', 'rb'))['feature_columns'])"`
- [ ] Set up cron jobs for automated data collection
- [ ] Monitor daily pipeline to ensure new features are used

---

## Additional Resources

- **Feature builders**: `src/features/{referee,news,sentiment}_features.py`
- **Data clients**: `src/data/{referee_data,news_scrapers,public_sentiment_scraper}.py`
- **Integration tests**: `scripts/test_alternative_data_integration.py`
- **Validation script**: `scripts/validate_alternative_data.py`
- **Plan file**: `.claude/plans/swift-questing-lake.md`

---

**Questions?**

Check the integration test output for detailed feature status, or run the validation script to diagnose data collection issues.
