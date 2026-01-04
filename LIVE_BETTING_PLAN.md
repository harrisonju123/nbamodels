# Live Betting Strategy - Implementation Plan

**Date**: January 4, 2026
**Status**: Planning Phase

---

## Executive Summary

Live (in-game) betting offers significant advantages over pre-game betting:
- **Real-time information**: Actual game flow, injuries, foul trouble, momentum
- **Market inefficiencies**: Books slower to react to in-game events
- **Strategic timing**: Can wait for favorable situations (lead changes, key player fouls)
- **Better probability estimates**: Model can use actual game state vs predictions

**Goal**: Build a system that monitors live games and identifies profitable in-game betting opportunities.

---

## Data Sources

### 1. Live Odds (Already Available)

**The Odds API** - Currently using for pre-game odds
- ✅ Supports live in-game odds
- ✅ Shows if game is in-play (`commence_time < current_time`)
- ✅ Multiple bookmakers (DraftKings, FanDuel, BetMGM, etc.)
- ✅ Markets: spreads, moneyline, totals
- 📊 **Cost**: Counts against API request quota
- 🔗 [The Odds API - Live Documentation](https://the-odds-api.com/liveapi/guides/v4/)

**Usage Example**:
```python
# Current implementation can already fetch live odds
odds = OddsAPIClient().get_current_odds(markets=["h2h", "spreads", "totals"])
# Filter for in-play games
live_games = odds[odds['commence_time'] < datetime.now()]
```

### 2. Live Game State (New - Need to Add)

**Option A: NBA Stats API (FREE)** ⭐ Recommended
- ✅ Free, no API key required
- ✅ Official NBA data
- ✅ Live scores, quarter, time remaining
- ✅ Box scores updated in real-time
- ⚠️ Rate limit: ~1 request/second
- 🔗 Endpoint: `https://stats.nba.com/stats/scoreboardv2`

**Option B: BALLDONTLIE (FREE/PAID)**
- ✅ Free tier available
- ✅ Real-time stats every second during games
- ✅ Play-by-play data
- 🔗 [BALLDONTLIE API](https://www.balldontlie.io/)

**Option C: Sportradar (PREMIUM)**
- ✅ Official NBA data provider
- ✅ Most comprehensive play-by-play
- ✅ Push feeds for real-time updates
- ❌ Expensive ($1000+/month)
- 🔗 [Sportradar NBA API](https://developer.sportradar.com/basketball/docs/nba-ig-live-game-retrieval)

**Recommended**: Start with **NBA Stats API (free)** for MVP, upgrade to Sportradar if profitable.

### 3. Play-by-Play Data (Optional Enhancement)

**Why Useful**:
- Detect momentum shifts (scoring runs)
- Track foul trouble (starters at risk)
- Identify lineup changes
- Measure pace/tempo deviations

**Source**: NBA Stats API `/playbyplayv2` endpoint

---

## Strategy Components

### Phase 1: Live Game Monitor (MVP)

**Goal**: Track live games and detect betting opportunities based on game state vs odds.

**Core Features**:
1. **Live Game Tracker**
   - Poll NBA Stats API every 30-60 seconds
   - Track: score, quarter, time remaining, timeouts
   - Detect state changes (quarter end, timeouts, lead changes)

2. **Live Odds Tracker**
   - Poll The Odds API every 2-3 minutes
   - Track spread/moneyline/total movements
   - Calculate implied probability shifts

3. **Win Probability Model**
   - Use current game state to estimate live win probability
   - Inputs: score differential, time remaining, quarter, home/away
   - Compare to pre-game model prediction

4. **Edge Detection**
   - Compare live win probability vs live odds
   - Alert when edge > threshold (e.g., 5%)

**Files to Create**:
```
src/data/
├── live_game_client.py       # NBA Stats API live game data
└── live_odds_tracker.py      # The Odds API live odds

src/models/
└── live_win_probability.py   # In-game win probability model

src/betting/
└── live_edge_detector.py     # Detect live betting edges

scripts/
└── live_game_monitor.py      # Main monitoring loop
```

### Phase 2: Advanced Features

**A. Momentum Detection**
- Track scoring runs (e.g., 10-0 run in last 3 minutes)
- Detect pace changes vs pre-game expectations
- Alert when momentum contradicts line movement

**B. Situational Betting**
- Key player fouls (star at 5 fouls in 3rd quarter)
- Lineup changes (bench unit vs starters)
- Timeout patterns (team rallying after timeout)

**C. Total Betting**
- Track actual pace vs expected
- Adjust over/under probability based on current scoring rate
- Account for quarter-by-quarter patterns

**D. Halftime Strategy**
- Halftime line vs adjusted prediction
- Team halftime adjustment tendencies
- Coaching adjustments analysis

### Phase 3: Automation

**A. Auto-Betting** (with safeguards)
- Place bets automatically when edge > threshold
- Kelly criterion position sizing
- Hard limits (max bets/game, max stake/bet)
- Kill switch for emergencies

**B. Alert System**
- SMS/email alerts for high-edge opportunities
- Dashboard with live game state
- Bet placement confirmation UI

---

## Implementation Timeline

### Week 1: MVP (Live Game Monitor)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | `live_game_client.py` | Fetch live scores from NBA Stats API |
| 2 | `live_odds_tracker.py` | Fetch live odds from The Odds API |
| 3 | `live_win_probability.py` | Simple win prob model (score + time) |
| 4 | `live_edge_detector.py` | Compare model vs odds, detect edges |
| 5 | `live_game_monitor.py` | Main loop with logging |
| 6-7 | Testing | Monitor 10+ live games, tune thresholds |

**Success Criteria**: System runs for full game day, logs all edge opportunities

### Week 2: Backtesting & Validation

| Task | Description |
|------|-------------|
| Collect historical live odds | Scrape/API for past games |
| Replay live games | Simulate live betting with historical data |
| Measure hypothetical performance | Win rate, ROI, bet frequency |
| Tune edge thresholds | Optimize for precision vs volume |

**Success Criteria**: Hypothetical ROI > 5%, Win rate > 53%

### Week 3: Advanced Features

| Feature | Priority |
|---------|----------|
| Momentum detection | HIGH |
| Halftime betting | HIGH |
| Total betting | MEDIUM |
| Play-by-play analysis | LOW |

### Week 4: Production

| Task | Description |
|------|-------------|
| Dashboard integration | Add live games tab |
| Alert system | SMS/email on high edges |
| Paper trading | Track live bets in database |
| Auto-betting (optional) | With strict limits |

---

## Technical Architecture

### Database Schema

**live_game_state** - Current game state snapshots
```sql
CREATE TABLE live_game_state (
    id INTEGER PRIMARY KEY,
    game_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    quarter INTEGER,
    time_remaining TEXT,  -- "7:23"
    home_score INTEGER,
    away_score INTEGER,
    home_timeouts_remaining INTEGER,
    away_timeouts_remaining INTEGER,
    game_status TEXT,  -- "1st Qtr", "Halftime", "Final"
    UNIQUE(game_id, timestamp)
);
```

**live_odds_snapshot** - Live odds snapshots
```sql
CREATE TABLE live_odds_snapshot (
    id INTEGER PRIMARY KEY,
    game_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    bookmaker TEXT NOT NULL,
    market TEXT NOT NULL,  -- 'h2h', 'spreads', 'totals'
    home_odds INTEGER,
    away_odds INTEGER,
    spread_value REAL,
    total_value REAL,
    UNIQUE(game_id, timestamp, bookmaker, market)
);
```

**live_edge_alerts** - Detected betting edges
```sql
CREATE TABLE live_edge_alerts (
    id INTEGER PRIMARY KEY,
    game_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    alert_type TEXT NOT NULL,  -- 'spread', 'moneyline', 'total'
    bet_side TEXT,
    model_prob REAL,
    market_prob REAL,
    edge REAL,
    quarter INTEGER,
    score_diff INTEGER,
    time_remaining TEXT,
    confidence TEXT,  -- 'HIGH', 'MEDIUM', 'LOW'
    acted_on BOOLEAN DEFAULT 0
);
```

**live_bets** - Placed live bets
```sql
CREATE TABLE live_bets (
    id INTEGER PRIMARY KEY,
    game_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    bet_type TEXT NOT NULL,
    bet_side TEXT NOT NULL,
    odds INTEGER,
    stake REAL,
    expected_edge REAL,
    quarter INTEGER,
    score_diff_at_bet INTEGER,
    outcome TEXT,  -- 'win', 'loss', 'push', 'pending'
    settled_at TEXT
);
```

### Monitoring Loop

```python
# scripts/live_game_monitor.py
from src.data.live_game_client import LiveGameClient
from src.data.live_odds_tracker import LiveOddsTracker
from src.models.live_win_probability import LiveWinProbModel
from src.betting.live_edge_detector import LiveEdgeDetector

def monitor_live_games():
    game_client = LiveGameClient()
    odds_tracker = LiveOddsTracker()
    win_prob_model = LiveWinProbModel()
    edge_detector = LiveEdgeDetector(min_edge=0.05)

    while True:
        # Get live games
        live_games = game_client.get_live_games()

        for game in live_games:
            # Update game state
            game_state = game_client.get_game_state(game['game_id'])

            # Get current live odds (less frequent to save API calls)
            if should_check_odds(game_state):
                live_odds = odds_tracker.get_live_odds(game['game_id'])

                # Calculate win probability
                win_prob = win_prob_model.predict(game_state)

                # Detect edges
                edges = edge_detector.find_edges(game_state, live_odds, win_prob)

                # Log/alert high-edge opportunities
                for edge in edges:
                    if edge['confidence'] == 'HIGH':
                        logger.info(f"HIGH EDGE: {edge}")
                        send_alert(edge)

        time.sleep(30)  # Check every 30 seconds
```

---

## Win Probability Model Approaches

### Approach 1: Simple Linear Model (MVP)

**Features**:
- Score differential
- Time remaining (seconds)
- Quarter
- Home/away
- Timeouts remaining

**Formula**:
```python
# Based on historical comeback probabilities
win_prob = baseline_home_win_rate
win_prob += score_diff * points_per_possession_value
win_prob *= time_decay_factor(time_remaining)
win_prob = clip(win_prob, 0, 1)
```

**Pros**: Fast, interpretable, no training needed
**Cons**: Ignores team quality, momentum, matchups

### Approach 2: Historical Simulation (Better)

**Method**: Query historical games with similar game states
- Same score differential (±3 pts)
- Same time remaining (±2 min)
- Same quarter
- Calculate actual win rate from historical outcomes

**Pros**: Data-driven, accounts for real comeback patterns
**Cons**: Requires historical live game data

### Approach 3: ML Model (Best)

**Features**:
- All from Approach 1
- Pre-game win probability (from our spread model)
- Team offensive/defensive ratings
- Pace differential vs expected
- Recent momentum (points in last 5 min)
- Key player fouls
- Lineup quality (if available)

**Model**: XGBoost or LightGBM trained on historical game states

**Training Data**: Scrape historical game states and outcomes

**Pros**: Most accurate, adapts to team quality
**Cons**: Requires training data, more complex

**Recommendation**: Start with Approach 1 for MVP, upgrade to Approach 3 after validation.

---

## Edge Detection Logic

### Criteria for High-Edge Bet

1. **Minimum Edge**: Model prob vs market prob difference > 5%
2. **Minimum Confidence**: Based on sample size/model uncertainty
3. **Situational Filter**: Avoid betting during:
   - Final 2 minutes (odds too volatile)
   - TV timeouts (odds adjust quickly)
   - Free throw situations (noise in score)
4. **Liquidity Check**: Ensure bookmaker has line available
5. **Value Threshold**: Expected value > minimum (e.g., +EV > $5)

### Alert Prioritization

**HIGH Priority** (auto-bet or immediate alert):
- Edge > 7%
- High confidence (large sample, stable model)
- Favorable situation (halftime, after timeout)

**MEDIUM Priority** (alert, manual review):
- Edge 5-7%
- Medium confidence
- Normal game flow

**LOW Priority** (log only):
- Edge 3-5%
- Low confidence or volatile situation

---

## Risk Management

### Position Sizing

**Kelly Criterion for Live Bets**:
```python
# More conservative than pre-game (higher uncertainty)
kelly = (edge * win_prob - (1 - win_prob)) / edge
stake = bankroll * kelly * 0.25  # Quarter Kelly
stake = min(stake, max_bet_per_game)
```

### Limits and Safeguards

| Limit Type | Value | Reason |
|------------|-------|--------|
| Max bets per game | 3 | Avoid over-exposure to single game |
| Max stake per bet | $200 | Limit single bet risk |
| Max daily bets | 20 | Prevent runaway system |
| Min edge | 5% | Avoid low-value bets |
| Max exposure per game | $500 | Total risk per game |
| Cooldown after loss | 10 min | Avoid tilt/chasing |
| Kill switch | Manual | Emergency stop |

### Monitoring

- **Real-time P&L tracking**
- **Alert on unusual activity** (e.g., 5 losses in a row)
- **Daily performance reports**
- **Weekly strategy review**

---

## Success Metrics

### Phase 1 (MVP) - Monitoring Only

- [ ] System runs for 10+ game days without crashing
- [ ] Logs 50+ edge opportunities
- [ ] Average edge of logged opportunities > 5%
- [ ] False positive rate < 30% (edges that disappear quickly)

### Phase 2 (Paper Trading)

- [ ] Hypothetical win rate > 53%
- [ ] Hypothetical ROI > 5%
- [ ] 100+ paper bets placed
- [ ] Average edge at bet time > 4%

### Phase 3 (Real Money - Small Stakes)

- [ ] Actual win rate > 52%
- [ ] Actual ROI > 3%
- [ ] 50+ real bets placed
- [ ] Max drawdown < 20%

### Phase 4 (Scaled Production)

- [ ] Win rate > 54%
- [ ] ROI > 5%
- [ ] 200+ bets/month
- [ ] Positive P&L for 3 consecutive months

---

## Advantages of Live Betting

### 1. Information Edge

**Pre-game**: We predict what will happen
**Live**: We react to what IS happening

**Examples**:
- Star player picks up 2 early fouls → bench time likely → bet against
- Team shoots 15% from 3 in first half → regression likely → fade overcorrection
- Blowout developing → garbage time dynamics → total opportunities

### 2. Market Inefficiency

**Books struggle with**:
- Rapid line adjustments during play
- Balancing action across live bettors
- Momentum over-reactions by public

**We can exploit**:
- Delayed book reactions to key events
- Public overvaluing recent possessions
- Clearer probability with actual game data

### 3. Strategic Timing

**We can choose WHEN to bet**:
- Halftime: Teams adjust, line resets
- After timeouts: Coaching adjustments clear
- Quarter breaks: Clean state, less noise
- Key matchups: Favorable lineup on court

### 4. Better Probability Estimates

**Pre-game model uncertainty**: ±5 points
**Live model with game state**: ±2 points

With real game flow, predictions more accurate.

---

## Risks and Challenges

### Technical Risks

| Risk | Mitigation |
|------|------------|
| API rate limits | Cache aggressively, poll less frequently |
| Odds disappear fast | Auto-bet on high edges, accept some misses |
| Delayed data | Use fastest source (NBA Stats), accept 15-30s lag |
| System downtime | Monitoring alerts, auto-restart |

### Market Risks

| Risk | Mitigation |
|------|------------|
| Limited liquidity | Check available limits, bet smaller |
| Account limits | Rotate bookmakers, bet conservatively |
| Line moved before bet | Accept and log, optimize speed |
| Odds errors | Sanity checks (e.g., -1000 odds unlikely) |

### Model Risks

| Risk | Mitigation |
|------|------------|
| Overfitting | Simple model first, validate extensively |
| Biased training data | Use diverse historical games |
| Situational blindness | Add situational features (fouls, etc.) |
| Concept drift | Retrain regularly, monitor performance |

---

## Next Steps

1. ✅ **Create this plan** (completed)
2. ⏳ **Get user approval** - Does this approach make sense?
3. ⏳ **Week 1 Sprint** - Build MVP components
4. ⏳ **Test on live games** - Monitor without betting
5. ⏳ **Backtest if possible** - Validate with historical data
6. ⏳ **Paper trade** - Track hypothetical bets
7. ⏳ **Small real money** - Validate with low stakes
8. ⏳ **Scale up** - Increase stakes if profitable

---

## Questions for User

Before starting implementation:

1. **Comfort with live betting?** Are you familiar with in-game betting? Any bookmakers you currently use?

2. **API budget?** The Odds API costs ~$0.01/request. Live monitoring could use 500-1000 requests/day ($5-10/day). Acceptable?

3. **Automation level?** Should we:
   - **Option A**: Just alert you to opportunities (manual betting)
   - **Option B**: Paper trade automatically (log hypothetical bets)
   - **Option C**: Auto-bet with strict limits (requires bookmaker API)

4. **Time commitment?** Live betting requires monitoring during games (7-11 PM ET). Can you:
   - Run the monitoring script during game hours?
   - Respond to alerts within 2-3 minutes?
   - OR should we build fully automated system?

5. **Historical data?** Do you want me to try to scrape/collect historical live game data for backtesting? Or start live monitoring immediately?

6. **Starting bankroll?** What bankroll for live betting? (Recommend separate from pre-game)

---

## Resources

**APIs & Documentation**:
- [The Odds API - Live Documentation](https://the-odds-api.com/liveapi/guides/v4/)
- [NBA Stats API](https://stats.nba.com/)
- [BALLDONTLIE API](https://www.balldontlie.io/)
- [Sportradar NBA API](https://developer.sportradar.com/basketball/docs/nba-ig-live-game-retrieval)

**Research & Tutorials**:
- [Accessing Live NBA Play-by-Play Data](https://jman4190.medium.com/how-to-accessing-live-nba-play-by-play-data-f24e02b0a976)
- [Live Game Updates - Sportradar](https://developer.sportradar.com/basketball/docs/nba-ig-live-game-retrieval)

---

**Status**: 📋 Awaiting user feedback to begin implementation
