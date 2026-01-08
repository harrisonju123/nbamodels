# Advanced Player Props Model - Development Plan

**Goal**: Build a sophisticated player props prediction system that matches the quality and profitability of the spread model (+23.6% ROI)

**Current State**: Simple rolling average models (~17 features, unknown profitability)
**Target State**: Advanced ML models (80+ features, deep matchup analysis, 10-15% ROI)

---

## Architecture Overview

### **Spread Model Success Formula** (to replicate)
```
82 features = Rolling trends + Deep matchups + Context + Market intel
↓
XGBoost classifier (well-calibrated probabilities)
↓
Edge detection (model prob vs bookmaker prob)
↓
Kelly criterion sizing
↓
Result: +23.6% ROI (proven)
```

### **Advanced Props Model Design**
```
80+ features = Player trends + Matchup tracking + Referee + Lineup + Usage patterns
↓
Ensemble: XGBoost + CatBoost + Neural Network
↓
Calibrated probability distributions (not just point estimates)
↓
Edge detection with uncertainty quantification
↓
Target: 10-15% ROI
```

---

## Phase 1: Enhanced Feature Engineering (Weeks 1-2)

### **1.1 Player Performance Trends** (Upgrade from current)

**Current**: Simple 3/5/10 game rolling averages
**Upgrade To**:

```python
Rolling Windows (multiple timeframes):
├─ Last 3 games (recent form)
├─ Last 5 games (short-term trend)
├─ Last 10 games (medium-term trend)
├─ Last 20 games (season baseline)
└─ Home/Away splits for each window

Weighted Averages:
├─ Exponentially weighted (recent games matter more)
├─ Recency-weighted by date
├─ Opponent-strength weighted
└─ Minutes-weighted (don't count garbage time)

Trend Indicators:
├─ Is player trending up/down? (linear regression slope)
├─ Performance variance (consistency metric)
├─ Hot/cold streak detection
└─ Peak vs trough identification
```

**Implementation**:
```python
# src/features/player_trends.py

def calculate_advanced_rolling_stats(player_df):
    """Calculate sophisticated rolling statistics."""

    # Exponentially weighted moving average (recent games matter more)
    for stat in ['pts', 'reb', 'ast', 'fg3m']:
        player_df[f'{stat}_ewma_5'] = (
            player_df.groupby('player_id')[stat]
            .transform(lambda x: x.ewm(span=5).mean().shift(1))
        )

    # Performance trend (last 5 games)
    player_df['pts_trend_5g'] = (
        player_df.groupby('player_id')['pts']
        .transform(lambda x: calculate_trend_slope(x, window=5))
    )

    # Consistency metric (lower = more consistent)
    player_df['pts_variance_10g'] = (
        player_df.groupby('player_id')['pts']
        .transform(lambda x: x.rolling(10).std().shift(1))
    )

    # Home/Away splits
    for location in ['home', 'away']:
        mask = player_df['is_home'] == (location == 'home')
        player_df[f'pts_{location}_avg'] = (
            player_df[mask].groupby('player_id')['pts']
            .transform(lambda x: x.expanding().mean().shift(1))
        )

    return player_df
```

---

### **1.2 Deep Matchup Analysis** ✨ NEW

**What we're missing**: Player vs team/defender historical performance

```python
Opponent Team Matchup:
├─ Player's career stats vs this opponent
├─ Last 3 games vs this team (recency matters)
├─ Home/away splits vs this team
└─ Weighted by recency and sample size

Positional Matchup:
├─ Opponent's defensive rating vs player's position
├─ Opponent's pace vs player's position
├─ Opponent's tendency to allow rebounds/assists/3PM to position
└─ League rank in defending position

Specific Defender Tracking (if data available):
├─ Primary defender assignment (from SportRadar/Synergy)
├─ Historical performance vs that defender
├─ Defender's defensive metrics
└─ Defensive scheme (zone, man, switch)
```

**Data Sources Needed**:
- ✅ **NBA Stats API** (free) - Team defense by opponent position
- ⏳ **SportRadar API** (paid ~$200/mo) - Defender tracking
- ⏳ **Synergy Sports** (expensive ~$1K/mo) - Defensive schemes
- ✅ **Historical box scores** (we have this) - Player vs team history

**Implementation (Free Data First)**:
```python
# src/features/matchup_analysis.py

def add_opponent_matchup_features(player_df, opponent_stats_df):
    """Add opponent defensive matchup features."""

    # Opponent defense vs position
    # Get opponent's defensive rating vs PG, SG, SF, PF, C
    opponent_def_vs_pos = get_opponent_defense_by_position(opponent_stats_df)

    player_df = player_df.merge(
        opponent_def_vs_pos,
        left_on=['opponent_team', 'player_position'],
        right_on=['team', 'position'],
        how='left'
    )

    # Player's historical performance vs this opponent
    player_vs_team = calculate_player_vs_team_history(player_df)
    player_df = player_df.merge(player_vs_team, on=['player_id', 'opponent_team'])

    # Opponent's pace vs this position (more possessions = more opportunities)
    player_df['opp_pace_vs_pos'] = calculate_opponent_pace_vs_position(
        opponent_stats_df, player_df
    )

    # How many points/rebounds/assists opponent allows to this position
    player_df['opp_pts_allowed_to_pos'] = get_opponent_stats_vs_position(
        opponent_stats_df, player_df, stat='pts'
    )

    return player_df

def get_opponent_defense_by_position(opponent_stats_df):
    """
    Get opponent's defensive metrics by position.

    Example:
    Team: LAL, Position: PG
    → LAL allows 24.5 PPG to opposing PGs (vs league avg 22.1)
    → LAL is weak defending PGs
    """
    # Use NBA Stats API to get opponent stats by position
    # Endpoint: /stats/team/defense?groupBy=position
    pass
```

---

### **1.3 Usage Patterns & Role** ✨ NEW

**What we're missing**: Player's role in offense and how it changes

```python
Usage Context:
├─ Usage rate (current vs season average)
├─ Usage rate trend (increasing/decreasing role)
├─ Shot distribution (paint, mid-range, 3PT)
├─ Play type distribution (isolation, pick-and-roll, spot-up)
└─ Teammate dependency (how much does player depend on specific teammates)

Role Indicators:
├─ Primary ball handler (AST:TOV ratio)
├─ Scoring role (points per touch)
├─ Shooting role (3PA rate, catch-and-shoot %)
└─ Rebounding role (contested rebound rate)

Minute Load:
├─ Minutes last 3 games (fatigue indicator)
├─ Minutes above/below season average
├─ Back-to-back impact on minutes
└─ Expected minutes tonight (based on coach patterns)
```

**Implementation**:
```python
# src/features/usage_patterns.py

def calculate_usage_patterns(player_df, team_df):
    """Calculate player's usage and role metrics."""

    # Usage rate trend (is player getting more shots?)
    player_df['usage_trend_5g'] = (
        player_df.groupby('player_id')['usage_rate']
        .transform(lambda x: calculate_trend_slope(x, window=5))
    )

    # Shot distribution (what % of shots from 3PT line)
    player_df['three_point_rate'] = player_df['fg3a'] / player_df['fga']
    player_df['three_point_rate_roll5'] = (
        player_df.groupby('player_id')['three_point_rate']
        .transform(lambda x: x.rolling(5).mean().shift(1))
    )

    # Playmaking role (high AST:TOV = primary ball handler)
    player_df['assist_to_tov_ratio'] = player_df['ast'] / player_df['tov'].replace(0, 1)

    # Minute load (fatigue indicator)
    player_df['min_last_3g'] = (
        player_df.groupby('player_id')['min']
        .transform(lambda x: x.rolling(3).sum().shift(1))
    )
    player_df['min_above_avg'] = (
        player_df['min_roll10'] - player_df.groupby('player_id')['min'].transform('mean')
    )

    return player_df
```

---

### **1.4 Lineup & Teammate Impact** ✨ NEW

**What we're missing**: How player performs with different lineups

```python
Lineup Strength:
├─ Teammates on court (5-man lineup rating)
├─ With/without star teammate (LeBron on/off court)
├─ Lineup net rating (offensive - defensive)
└─ Lineup familiarity (games played together)

Teammate Impact:
├─ Performance with starting PG vs backup PG
├─ Performance with/without best scorer
├─ Usage rate boost when star sits
└─ Assist opportunities (how good are teammates at finishing)

Opponent Lineup:
├─ Opponent's defensive lineup rating
├─ Opponent's best defender playing/injured
└─ Opponent's rim protector impact
```

**Data Source**:
- ✅ **NBA Stats API** - Lineup data (free)
- Track lineups manually from box scores

**Implementation**:
```python
# src/features/lineup_impact.py

def calculate_lineup_impact(player_df, lineup_df):
    """Calculate player performance with different lineups."""

    # Get tonight's projected lineup
    # Compare to historical performance with this lineup

    # Example: Giannis with Lillard on court
    player_df['with_best_teammate'] = check_teammate_status(
        player_df, teammate='Damian Lillard'
    )

    # Usage boost when star is out
    player_df['usage_boost_no_star'] = calculate_usage_boost_without_teammate(
        player_df, star_player='Giannis Antetokounmpo'
    )

    # Lineup net rating
    player_df['lineup_net_rating'] = get_lineup_rating(
        lineup_df, player_df['projected_lineup']
    )

    return player_df
```

---

### **1.5 Referee Impact** ✨ NEW

**What we're missing**: How referees affect player stats

```python
Referee Tendencies:
├─ Referee crew foul call rate (affects FTA)
├─ Referee crew pace factor (affects total possessions)
├─ Referee crew star player bias (superstars get more calls)
├─ Home whistle advantage
└─ Referee crew experience (veteran vs rookie crews)

Impact on Player Props:
├─ FTA prediction (more fouls = more FTA for scorers)
├─ Assists prediction (faster pace = more possessions)
├─ Points prediction (free throws + pace)
└─ Fouls out risk (high foul rate = less minutes)
```

**Data Source**:
- ✅ **NBA Stats API** - Referee assignments (free)
- Build historical database of referee tendencies

**Implementation**:
```python
# src/features/referee_impact.py

def add_referee_features(player_df, referee_df):
    """Add referee crew impact features."""

    # Get tonight's referee crew
    referee_crew = get_referee_crew_for_game(player_df['game_id'])

    # Referee crew foul call rate
    player_df['ref_foul_rate'] = get_crew_foul_rate(referee_crew)

    # Referee crew pace factor
    player_df['ref_pace_factor'] = get_crew_pace_factor(referee_crew)

    # Star player bias (superstars get 15% more foul calls with certain refs)
    if player_is_superstar(player_df['player_name']):
        player_df['ref_star_bias'] = get_crew_star_bias(referee_crew)

    # Historical FTA with this ref crew
    player_df['fta_with_crew'] = get_player_fta_history_with_crew(
        player_df['player_id'], referee_crew
    )

    return player_df
```

---

### **1.6 Advanced Context Features** ✨ NEW

**What we're missing**: Situational factors

```python
Rest & Fatigue:
├─ Days rest (already have)
├─ Minutes played last 3 games (fatigue)
├─ Travel distance (coast-to-coast fatigue)
├─ Time zone changes
└─ Back-to-back impact on stats

Game Situation:
├─ Expected game competitiveness (blowout risk)
├─ Playoff implications (motivation)
├─ Revenge game (player vs former team)
├─ Contract year (extra motivation)
└─ Milestone watch (chasing career high, record)

Timing:
├─ Time of season (early season rust vs late season fatigue)
├─ Post all-star break (performance changes)
├─ Day of week (players perform better on certain days)
├─ Start time (afternoon vs night games)
└─ Nationally televised (players step up for big games)
```

**Implementation**:
```python
# src/features/advanced_context.py

def add_advanced_context(player_df, games_df):
    """Add sophisticated contextual features."""

    # Fatigue indicator
    player_df['fatigue_score'] = calculate_fatigue_score(
        minutes_last_3=player_df['min_last_3g'],
        rest_days=player_df['days_rest'],
        travel_distance=player_df['travel_distance']
    )

    # Expected game competitiveness (from spread model)
    player_df['expected_margin'] = abs(games_df['spread'])
    player_df['blowout_risk'] = (player_df['expected_margin'] > 10).astype(int)

    # Revenge game
    player_df['is_revenge_game'] = check_if_revenge_game(
        player_df['player_id'],
        player_df['opponent_team'],
        player_df['player_team']
    )

    # Time of season (0-1 scale)
    player_df['season_progress'] = player_df['games_played'] / 82

    # Nationally televised
    player_df['is_national_tv'] = check_if_national_tv(games_df)

    return player_df
```

---

## Phase 2: Data Collection & Integration (Weeks 2-3)

### **2.1 Historical Player Prop Odds**

**Critical for backtesting!**

**Sources**:
1. **The Odds API** (current provider)
   - Player props available: `player_points`, `player_rebounds`, `player_assists`, `player_threes`
   - Cost: Already using (check plan limits)
   - Collection: Start collecting daily going forward

2. **Pinnacle API** (sharp bookmaker)
   - Most efficient lines
   - Good for backtesting
   - Cost: Free API access

3. **Archive.org Sports Odds** (historical)
   - May have archived prop odds
   - Free but incomplete

**Implementation**:
```python
# scripts/collect_player_prop_odds.py

def collect_daily_player_props():
    """Collect and store player prop odds daily."""

    from src.data.odds_api import OddsAPIClient

    odds_client = OddsAPIClient()

    # Get today's games
    games = odds_client.get_odds('basketball_nba', markets=['h2h'])

    all_props = []
    for game in games['games']:
        game_id = game['id']

        # Fetch player props for this game
        props = odds_client.get_player_props(
            event_id=game_id,
            markets=['player_points', 'player_rebounds', 'player_assists', 'player_threes']
        )

        all_props.append(props)

    # Store in database
    props_df = pd.concat(all_props)
    props_df.to_parquet(f"data/historical_player_props/props_{datetime.now().date()}.parquet")

    logger.info(f"Collected {len(props_df)} player prop lines")

# Add to cron: run daily at 3 PM ET (before 4 PM pipeline)
```

---

### **2.2 Defender Tracking Data** (Optional - Paid)

**SportRadar API** ($200-500/mo):
- Primary defender assignments
- Defensive matchup stats
- Contested shot data

**Alternative**: Build manually from play-by-play
- Parse NBA Stats API play-by-play data
- Track who guards who
- Free but time-intensive

---

### **2.3 Advanced Stats APIs**

**Free Sources**:
- ✅ **NBA Stats API** - Team/player advanced stats
- ✅ **Basketball Reference** - Historical data (scraping)
- ✅ **ESPN API** - Lineups, injuries (already using)

**Paid Sources** (optional):
- **Synergy Sports** - Play type data (ISO, P&R, spot-up)
- **Second Spectrum** - Optical tracking (speed, distance)
- **Sportradar** - Real-time play-by-play

---

## Phase 3: Model Architecture (Weeks 3-4)

### **3.1 Ensemble Approach**

Instead of single XGBoost model, use **ensemble of 3 models**:

```python
Model 1: XGBoost (baseline)
├─ Same as current
├─ Fast training
└─ Good feature importance

Model 2: CatBoost (better categorical handling)
├─ Handles categorical features better (player names, teams, positions)
├─ Automatic interaction detection
└─ Less overfitting

Model 3: Neural Network (complex patterns)
├─ Captures non-linear interactions
├─ Learns embeddings for players/teams
└─ Best for large datasets

Final Prediction: Weighted average or meta-model
```

**Implementation**:
```python
# src/models/player_props/ensemble_prop_model.py

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor

class EnsemblePropModel:
    """Ensemble of XGBoost, CatBoost, and Neural Network."""

    def __init__(self):
        self.xgb_model = XGBRegressor(**xgb_params)
        self.cat_model = CatBoostRegressor(**cat_params)
        self.nn_model = MLPRegressor(**nn_params)

        self.weights = [0.4, 0.4, 0.2]  # XGB, CAT, NN

    def fit(self, X_train, y_train, X_val, y_val):
        """Train all 3 models."""
        self.xgb_model.fit(X_train, y_train)
        self.cat_model.fit(X_train, y_train)
        self.nn_model.fit(X_train, y_train)

        # Optimize weights on validation set
        self.weights = self._optimize_weights(X_val, y_val)

    def predict(self, X):
        """Weighted average of 3 models."""
        pred_xgb = self.xgb_model.predict(X)
        pred_cat = self.cat_model.predict(X)
        pred_nn = self.nn_model.predict(X)

        return (
            self.weights[0] * pred_xgb +
            self.weights[1] * pred_cat +
            self.weights[2] * pred_nn
        )

    def predict_distribution(self, X):
        """Return probability distribution, not just point estimate."""
        # Use quantile regression to get confidence intervals
        lower = self.predict_quantile(X, quantile=0.25)
        median = self.predict(X)
        upper = self.predict_quantile(X, quantile=0.75)

        return {'lower': lower, 'median': median, 'upper': upper}
```

---

### **3.2 Probability Distributions** (Not Just Point Estimates)

**Current**: Model predicts "Giannis will score 29.2 points"
**Upgrade**: Model predicts "Giannis will score 29.2 ± 5.4 points (68% CI)"

**Why this matters**:
```
Bookmaker line: Giannis PTS O/U 28.5

Model A (point estimate): Predicts 29.2
→ Bet Over (barely)

Model B (distribution): Predicts 29.2 ± 5.4
→ P(over 28.5) = 54%
→ Implied odds from -110 = 52.4%
→ Edge = 1.6% (too small, SKIP)

Model B is more accurate about uncertainty!
```

**Implementation**:
```python
# Use quantile regression
from xgboost import XGBRegressor

class DistributionPropModel:
    """Predict full distribution, not just mean."""

    def __init__(self):
        self.model_median = XGBRegressor(objective='reg:squarederror')
        self.model_q25 = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.25)
        self.model_q75 = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.75)

    def predict_distribution(self, X):
        median = self.model_median.predict(X)
        q25 = self.model_q25.predict(X)
        q75 = self.model_q75.predict(X)

        # Estimate standard deviation
        std = (q75 - q25) / 1.35  # IQR to std conversion

        return {'median': median, 'std': std, 'q25': q25, 'q75': q75}

    def calculate_over_probability(self, X, line):
        """P(player > line)"""
        dist = self.predict_distribution(X)

        # Assume normal distribution
        from scipy.stats import norm
        z_score = (line - dist['median']) / dist['std']
        prob_over = 1 - norm.cdf(z_score)

        return prob_over
```

---

## Phase 4: Backtesting Infrastructure (Week 4)

### **4.1 Synthetic Backtest** (Immediate)

While collecting real odds, test against actual outcomes:

```python
# scripts/synthetic_props_backtest.py

def synthetic_backtest(start_date='2024-01-01', end_date='2024-12-31'):
    """
    Backtest models against actual outcomes.

    Simulate bookmaker lines based on market efficiency.
    """

    # Load historical player games
    player_games = load_player_games(start_date, end_date)

    results = []
    for idx, game in player_games.iterrows():
        # Model prediction
        features = build_features(game)
        prediction = model.predict(features)

        # Actual result
        actual = game['pts']

        # Simulate bookmaker line (assume efficient market)
        # Bookmaker sets line at 50th percentile ± vig
        simulated_line = prediction['median']

        # Would we bet?
        prob_over = model.calculate_over_probability(features, simulated_line)
        implied_prob = 0.524  # From -110 odds
        edge = prob_over - implied_prob

        if edge > 0.05:  # 5% edge threshold
            # Place simulated bet
            won = actual > simulated_line
            profit = calculate_profit(won, stake=100, odds=-110)

            results.append({
                'player': game['player_name'],
                'predicted': prediction['median'],
                'line': simulated_line,
                'actual': actual,
                'edge': edge,
                'won': won,
                'profit': profit
            })

    # Calculate performance
    results_df = pd.DataFrame(results)
    print(f"Win Rate: {results_df['won'].mean():.1%}")
    print(f"ROI: {results_df['profit'].sum() / (len(results_df) * 100):.1%}")
    print(f"Total Profit: ${results_df['profit'].sum():.2f}")

    return results_df
```

---

### **4.2 Real Backtest** (After data collection)

Once we have 30-60 days of real player prop odds:

```python
def real_backtest(historical_odds_dir='data/historical_player_props'):
    """Backtest with real bookmaker lines."""

    # Load historical prop odds
    odds_files = glob(f"{historical_odds_dir}/props_*.parquet")
    all_odds = pd.concat([pd.read_parquet(f) for f in odds_files])

    # For each prop line
    results = []
    for idx, prop in all_odds.iterrows():
        player_name = prop['player_name']
        prop_type = prop['prop_type']
        line = prop['line']
        over_odds = prop['over_odds']
        under_odds = prop['under_odds']

        # Get features for this player-game
        features = build_features_for_game(
            player_name=player_name,
            game_date=prop['game_date'],
            opponent=prop['opponent']
        )

        # Model prediction
        prob_over = model.calculate_over_probability(features, line)

        # Calculate edge
        implied_prob_over = american_to_prob(over_odds)
        edge = prob_over - implied_prob_over

        # Would we bet?
        if edge > 0.05:
            # Get actual result
            actual = get_actual_result(player_name, prop['game_id'], prop_type)
            won = actual > line
            profit = calculate_profit(won, stake=100, odds=over_odds)

            results.append({
                'player': player_name,
                'prop_type': prop_type,
                'line': line,
                'predicted_prob': prob_over,
                'implied_prob': implied_prob_over,
                'edge': edge,
                'actual': actual,
                'won': won,
                'profit': profit
            })

    return analyze_backtest_results(results)
```

---

## Implementation Timeline

### **Week 1**: Foundation
- ✅ Day 1-2: Design feature architecture
- ⏳ Day 3-4: Implement advanced rolling stats
- ⏳ Day 5-7: Add matchup tracking (free data sources)

### **Week 2**: Feature Engineering
- ⏳ Day 1-2: Usage patterns & role features
- ⏳ Day 3-4: Lineup impact features
- ⏳ Day 5-6: Referee impact features
- ⏳ Day 7: Advanced context features

### **Week 3**: Model Development
- ⏳ Day 1-2: Build CatBoost models
- ⏳ Day 3-4: Build Neural Network models
- ⏳ Day 5-6: Ensemble integration
- ⏳ Day 7: Probability distribution modeling

### **Week 4**: Testing & Validation
- ⏳ Day 1-3: Synthetic backtest
- ⏳ Day 4-5: Feature importance analysis
- ⏳ Day 6-7: Model tuning & calibration

### **Ongoing**: Data Collection
- Start collecting player prop odds daily (Day 1)
- After 30 days: Real backtest
- After 60 days: Production deployment

---

## Expected Outcomes

### **After Phase 1-2** (Feature Engineering):
- 80+ features per player-game
- Deep matchup analysis comparable to spread model
- Sophisticated context understanding

### **After Phase 3** (Model Architecture):
- Ensemble model with 3 algorithms
- Probability distributions (uncertainty quantification)
- Calibrated edge detection

### **After Phase 4** (Backtesting):
- Validated profitability (target: 10-15% ROI)
- Optimal betting thresholds
- Risk-adjusted position sizing

---

## Cost Analysis

### **Free Approach** (Use only free data):
- **Cost**: $0
- **Timeline**: 4 weeks
- **Expected ROI**: 8-12%
- **Limitations**: No defender tracking, manual lineup data

### **Professional Approach** (Add paid data):
- **Cost**: $200-300/mo (SportRadar + data subscriptions)
- **Timeline**: 4 weeks
- **Expected ROI**: 12-18%
- **Benefits**: Defender tracking, play-by-play data, advanced metrics

### **Recommendation**:
Start with **Free Approach**, validate profitability, then upgrade to paid data if ROI > 10%

---

## Success Metrics

### **Minimum Viable Product** (MVP):
- ✅ 60+ features (vs current 17)
- ✅ Synthetic backtest showing >8% ROI
- ✅ Win rate >52%
- ✅ Calibrated probabilities

### **Production Ready**:
- ✅ 80+ features with deep matchups
- ✅ Real backtest (30+ days) showing >10% ROI
- ✅ Ensemble model beating single model baseline
- ✅ Probability distributions implemented

### **World-Class**:
- ✅ 100+ features including paid data sources
- ✅ 60-day backtest showing >12% ROI
- ✅ Defender tracking integrated
- ✅ Automated lineup/injury integration
- ✅ Real-time model updates

---

## Next Steps

**Immediate Actions** (You choose):

1. **Start Feature Engineering** - I can begin implementing advanced features today
2. **Set Up Data Collection** - Start collecting player prop odds daily
3. **Run Synthetic Backtest** - Test current models against historical outcomes
4. **All Three in Parallel** - Maximum speed

What would you like to tackle first?
