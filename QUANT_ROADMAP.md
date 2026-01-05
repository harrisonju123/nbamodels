# Roadmap to Professional Quant Betting Operation

## Current State ✅

You already have:
- ✅ ML prediction models (XGBoost, LightGBM, ensemble)
- ✅ Kelly criterion position sizing
- ✅ CLV tracking
- ✅ Automated data collection
- ✅ Production deployment (DigitalOcean)
- ✅ Dashboard and API
- ✅ Performance analytics

**You're ahead of 95% of sports bettors!**

---

## Phase 1: Risk Management & Monitoring (Weeks 1-2)

### Priority: HIGH - Protect Capital

#### 1.1 Implement Drawdown Limits
**Problem:** No circuit breaker if you hit a bad streak
**Solution:** Automatic bet sizing reduction or halt

```python
# src/betting/risk_management.py
class RiskManager:
    def check_drawdown(self, current_bankroll, peak_bankroll):
        drawdown = (peak_bankroll - current_bankroll) / peak_bankroll

        if drawdown > 0.20:  # 20% drawdown
            return "HALT"  # Stop betting
        elif drawdown > 0.10:  # 10% drawdown
            return "REDUCE"  # Half bet sizes
        else:
            return "NORMAL"
```

**Implementation:**
- Track peak bankroll (high water mark)
- Calculate current drawdown
- Adjust Kelly fraction based on drawdown
- Add to daily pipeline

---

#### 1.2 Real-Time Alerting System
**Problem:** You don't know when something breaks until you check
**Solution:** Proactive alerts via email/SMS/Slack

**What to monitor:**
- ❌ Cron job failures
- ❌ Model prediction errors
- ❌ API rate limit hits
- ❌ Unusual betting patterns
- ❌ Large losses (>2% bankroll in one day)
- ❌ CLV dropping below -2%
- ❌ Server resource issues

**Tools:**
- Uptime Robot (free) - Service monitoring
- Sentry (free tier) - Error tracking
- Simple email alerts via Python

```python
# src/monitoring/alerts.py
def send_alert(title, message, severity="WARNING"):
    if severity == "CRITICAL":
        # Send SMS via Twilio
        send_sms(message)
    # Send email
    send_email(title, message)
    # Log to dashboard
    log_alert(title, message, severity)
```

---

#### 1.3 Model Performance Monitoring
**Problem:** Model may degrade over time (concept drift)
**Solution:** Track model metrics daily

**Metrics to track:**
- ROI by week/month
- Win rate (should be ~52-55%)
- CLV (should be positive)
- Brier score (calibration)
- Log loss (prediction quality)
- AUC-ROC (discrimination)

**Implementation:**
```python
# Daily check after settlement
def monitor_model_performance():
    last_30_days = get_bets(days=30)

    roi = calculate_roi(last_30_days)
    clv = calculate_avg_clv(last_30_days)

    if roi < -5%:  # 5% loss over 30 days
        alert("Model ROI declining", severity="HIGH")

    if clv < 0:  # Negative CLV
        alert("Negative CLV - model may be broken", severity="CRITICAL")
```

---

## Phase 2: Better Backtesting (Weeks 3-4)

### Priority: HIGH - Validate Strategy

#### 2.1 Walk-Forward Validation
**Problem:** Backtests may overfit to historical data
**Solution:** Simulate realistic out-of-sample performance

**Current approach:** Train on all data, test on all data ❌
**Better approach:** Rolling window validation ✅

```python
# Walk-forward backtest
for i in range(num_windows):
    train_data = data[start:train_end]
    test_data = data[train_end:test_end]

    # Train model on training period
    model = train_model(train_data)

    # Test on next period (out-of-sample)
    predictions = model.predict(test_data)

    # Track performance
    results.append(evaluate(predictions, test_data))
```

**Why this matters:**
- More realistic performance estimates
- Catches overfitting early
- Shows if strategy degrades over time

---

#### 2.2 Transaction Cost Modeling
**Problem:** Backtests ignore fees, slippage, line movement
**Solution:** Model realistic execution costs

**Costs to include:**
- Betting fees (Pinnacle: -2% to -5% juice)
- Line movement (market moves against you)
- Opportunity cost (line closes before you bet)

```python
def apply_transaction_costs(bet, odds):
    # Assume -105 juice (4.76% vig)
    adjusted_odds = adjust_for_vig(odds)

    # Model slippage (0.5-1 point on average)
    slippage = random.normal(0.5, 0.2)
    final_odds = adjust_for_slippage(adjusted_odds, slippage)

    return final_odds
```

---

#### 2.3 Monte Carlo Simulation
**Problem:** One backtest doesn't show variance
**Solution:** Run 1000+ simulations with randomness

```python
def monte_carlo_backtest(strategy, num_sims=1000):
    results = []

    for i in range(num_sims):
        # Randomize bet ordering
        bets = shuffle(historical_bets)

        # Simulate betting with variance
        bankroll = simulate_betting(bets, strategy)

        results.append(bankroll)

    # Analyze distribution
    p5 = np.percentile(results, 5)   # 5th percentile
    p50 = np.percentile(results, 50)  # Median
    p95 = np.percentile(results, 95)  # 95th percentile

    return {
        "worst_case": p5,
        "expected": p50,
        "best_case": p95,
        "sharpe": calculate_sharpe(results)
    }
```

**Why this matters:**
- Shows range of outcomes
- Estimates risk of ruin
- Validates Kelly sizing

---

## Phase 3: Research Infrastructure (Weeks 5-6)

### Priority: MEDIUM - Systematic Improvement

#### 3.1 Experiment Tracking
**Problem:** Hard to compare model versions
**Solution:** Track every experiment with MLflow or Weights & Biases

```python
import mlflow

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("edge_threshold", 0.07)
    mlflow.log_param("kelly_fraction", 0.25)

    # Train model
    model = train_model(params)

    # Log metrics
    mlflow.log_metric("train_auc", 0.58)
    mlflow.log_metric("test_roi", 0.08)

    # Save model
    mlflow.sklearn.log_model(model, "model")
```

**Benefits:**
- Compare 10+ model versions easily
- Track which changes improved performance
- Reproduce any past experiment

---

#### 3.2 Feature Importance Analysis
**Problem:** Don't know which features actually matter
**Solution:** SHAP analysis + feature ablation

```python
import shap

# SHAP analysis
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot top features
shap.summary_plot(shap_values, X_test)

# Feature ablation test
for feature in features:
    # Remove feature and retrain
    model_without = train_without_feature(feature)
    performance_drop = test(model) - test(model_without)

    print(f"{feature}: {performance_drop:.3f} ROI impact")
```

**What you'll learn:**
- Which data sources are valuable
- What to focus data collection on
- What features can be removed (simplify model)

---

#### 3.3 A/B Testing Framework
**Problem:** Can't test new strategies without risking capital
**Solution:** Paper trade new models alongside production

```python
# Run two strategies in parallel
production_bets = production_model.predict(games)
experimental_bets = new_model.predict(games)

# Log both (only bet on production)
log_bets(production_bets, status="REAL")
log_bets(experimental_bets, status="PAPER")

# After 30 days, compare
if experimental_roi > production_roi:
    promote_to_production(new_model)
```

---

## Phase 4: Execution & Line Shopping (Weeks 7-8)

### Priority: MEDIUM-HIGH - Increase Edge

#### 4.1 Multi-Sportsbook Integration
**Problem:** Only using one sportsbook = leaving money on table
**Solution:** Compare lines across books, bet the best

**Books to integrate:**
- Pinnacle (sharpest lines, low juice)
- DraftKings (soft lines, promos)
- FanDuel (soft lines, promos)
- BetMGM (soft lines)
- Caesars (occasional soft lines)

**Expected improvement:** +2-3% ROI from line shopping alone

```python
def get_best_line(game, bet_type):
    lines = {
        "pinnacle": get_pinnacle_line(game, bet_type),
        "draftkings": get_draftkings_line(game, bet_type),
        "fanduel": get_fanduel_line(game, bet_type),
    }

    # Find best line
    best_book = max(lines, key=lambda b: lines[b]['value'])
    return best_book, lines[best_book]
```

---

#### 4.2 Optimal Bet Timing
**Problem:** Betting at 4 PM may not be optimal
**Solution:** Analyze when to place bets

**Research questions:**
- When do lines move most favorably?
- What's the optimal time before tip-off?
- Should you bet early (capture opening line) or late (more info)?

```python
# Analyze historical line movement
def analyze_bet_timing():
    for hours_before in [72, 48, 24, 12, 6, 3, 1]:
        avg_clv = calculate_clv_at_time(hours_before)
        print(f"{hours_before}h before: {avg_clv:.3f} CLV")
```

**Hypothesis:** Betting 2-6 hours before tip might be optimal (capture lineup news but before sharp money)

---

#### 4.3 Bet Limits & Bankroll Allocation
**Problem:** Trying to bet $1000 on soft lines won't work
**Solution:** Model realistic bet limits per book

```python
class BettingConstraints:
    def __init__(self):
        self.limits = {
            "pinnacle": 10000,  # High limits
            "draftkings": 1000,  # Lower limits for +EV
            "fanduel": 500,
        }

    def allocate_bet(self, bet_size, book):
        max_bet = self.limits[book]
        actual_bet = min(bet_size, max_bet)

        if actual_bet < bet_size:
            # Couldn't get full position
            log_limited_bet(bet_size, actual_bet, book)

        return actual_bet
```

---

## Phase 5: Advanced Analytics (Weeks 9-10)

### Priority: MEDIUM - Professional Reporting

#### 5.1 Sharpe Ratio & Risk Metrics
**Problem:** Only tracking ROI
**Solution:** Track risk-adjusted returns

```python
def calculate_advanced_metrics(bets):
    daily_returns = group_by_day(bets)

    metrics = {
        "roi": total_profit / total_risk,
        "sharpe": mean(daily_returns) / std(daily_returns) * sqrt(252),
        "sortino": mean(daily_returns) / downside_std(daily_returns),
        "max_drawdown": calculate_max_drawdown(daily_returns),
        "calmar": roi / max_drawdown,
        "win_rate": wins / total_bets,
        "avg_win": mean(winning_bets),
        "avg_loss": mean(losing_bets),
        "profit_factor": total_wins / abs(total_losses),
    }

    return metrics
```

**Add to dashboard:**
- Equity curve with drawdowns highlighted
- Rolling Sharpe ratio (30-day window)
- Distribution of returns (histogram)
- Win/loss streaks

---

#### 5.2 Attribution Analysis
**Problem:** Don't know what's driving performance
**Solution:** Break down returns by factor

```python
# Analyze performance by category
def attribution_analysis():
    performance_by = {
        "bet_type": group_roi_by("bet_type"),  # Spread vs Total
        "team": group_roi_by("team"),
        "home_away": group_roi_by("home_away"),
        "favorite_dog": group_roi_by("favorite_dog"),
        "total_range": group_roi_by("total_range"),  # High vs low scoring
        "odds_range": group_roi_by("odds_range"),  # Heavy favorites vs dogs
    }

    # Find what's working
    for category, results in performance_by.items():
        best = max(results, key=lambda x: x.roi)
        print(f"Best {category}: {best.name} - {best.roi:.1%} ROI")
```

**What you'll learn:**
- Stop betting on bad categories (e.g., heavy favorites)
- Double down on profitable niches (e.g., home underdogs)
- Adjust edge threshold by category

---

#### 5.3 Correlation Analysis
**Problem:** Might be betting correlated outcomes
**Solution:** Analyze bet correlation and hedge

```python
# Check if bets are too correlated
def check_correlation():
    # Same game parlays are 100% correlated (avoid!)
    # Conference opponents have ~30% correlation
    # Same team bets on different days have ~15% correlation

    for bet1, bet2 in combinations(active_bets, 2):
        corr = estimate_correlation(bet1, bet2)

        if corr > 0.5:
            alert(f"High correlation: {bet1} and {bet2}")
```

---

## Phase 6: Data Quality & Sources (Weeks 11-12)

### Priority: MEDIUM - Better Input = Better Output

#### 6.1 Multi-Source Data Validation
**Problem:** One API could have bad data
**Solution:** Cross-validate data across sources

**Add data sources:**
- Odds: The Odds API ✅, Action Network, BetMGM
- Stats: balldontlie ✅, NBA.com ✅, Basketball Reference
- Lineups: Official NBA, RotoWire, Multiple beat reporters

```python
def validate_odds_data(game):
    source1 = odds_api.get_odds(game)
    source2 = action_network.get_odds(game)

    diff = abs(source1 - source2)

    if diff > 2.0:  # Lines differ by >2 points
        alert(f"Odds discrepancy for {game}: {source1} vs {source2}")
        # Use consensus or most recent
```

---

#### 6.2 Injury/News Impact Model
**Problem:** Late injury news can kill a bet
**Solution:** Model impact of news on lines

```python
# Train a model: News → Line movement
def model_news_impact():
    # Features: player importance, time before game, injury type
    # Target: how much line moved

    news_model = train_news_model()

    # Before each bet, check for recent news
    for bet in proposed_bets:
        recent_news = get_news(bet.game, hours=2)

        if recent_news:
            expected_move = news_model.predict(recent_news)

            if abs(expected_move) > 1.0:
                # Significant news, hold off on bet
                bet.status = "PENDING_NEWS"
```

---

#### 6.3 Alternative Data Exploration
**Problem:** Everyone has stats and odds
**Solution:** Find unique data edges

**Ideas to explore:**
- Travel distance/schedule (rest advantage)
- Weather (for outdoor sports, not NBA)
- Social media sentiment (Twitter betting buzz)
- Referee betting patterns
- TV ratings / public attention (square money indicator)
- Injury report timing (early report vs late scratch)

---

## Phase 7: Operational Excellence (Ongoing)

### Priority: HIGH - Don't Go Broke from Bugs

#### 7.1 Comprehensive Logging
```python
# Log EVERYTHING
def place_bet(bet):
    log.info(f"Placing bet: {bet}")

    try:
        result = sportsbook.place_bet(bet)
        log.info(f"Bet placed successfully: {result}")
    except Exception as e:
        log.error(f"Bet placement failed: {e}")
        alert("Bet placement error", severity="HIGH")
        # Save for manual placement
        save_failed_bet(bet)
```

---

#### 7.2 Database Backups
```bash
# Daily database backup (add to crontab)
0 5 * * * cd /root/nbamodels && \
  cp data/bets/bets.db backups/bets_$(date +\%Y\%m\%d).db && \
  find backups/ -name "bets_*.db" -mtime +30 -delete
```

---

#### 7.3 Disaster Recovery Plan
**What if:**
- Droplet goes down? → Have backup server or migrate quickly
- Database corrupts? → Restore from backup
- Model breaks? → Rollback to previous version
- API keys leaked? → Rotate immediately

---

## Summary: Priority Order

### Must Do (Next 2 Weeks)
1. **Drawdown limits** - Protect capital
2. **Alerting system** - Know when things break
3. **Model performance monitoring** - Detect degradation
4. **Better backtesting** - Walk-forward validation

### Should Do (Weeks 3-6)
5. **Experiment tracking** - MLflow setup
6. **Feature analysis** - SHAP + ablation
7. **Multi-book integration** - Line shopping
8. **Advanced metrics** - Sharpe, Sortino, attribution

### Nice to Have (Weeks 7-12)
9. **Optimal timing analysis** - When to bet
10. **Alternative data** - Unique edges
11. **Correlation hedging** - Portfolio optimization

---

## What Separates You from Professional Shops Now

### You Have ✅
- Automated system
- ML models
- Production deployment
- Performance tracking

### Still Missing ❌
- **Risk management** (drawdown limits, position sizing rules)
- **Monitoring** (alerts, model drift detection)
- **Rigorous backtesting** (walk-forward, Monte Carlo)
- **Line shopping** (multi-book execution)
- **Research infrastructure** (experiment tracking)

---

## Expected Impact

If you implement Phase 1-4:
- **+15-25% increase in ROI** (from better execution, line shopping)
- **50-70% reduction in downside risk** (from risk management)
- **Confidence to scale up** (from better backtesting)
- **Faster iteration** (from research infrastructure)

---

## Next Steps

Want me to help you implement any of these? I'd recommend starting with:

1. **Drawdown limits** (30 min to implement)
2. **Email alerts** (1 hour to set up)
3. **Model monitoring dashboard** (2-3 hours)
4. **Walk-forward backtest** (1 day to build)

Which would you like to tackle first?
