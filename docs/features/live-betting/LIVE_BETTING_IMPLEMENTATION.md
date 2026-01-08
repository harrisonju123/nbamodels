# Live Betting - Implementation Guide

**Date**: January 4, 2026
**Approach**: Dashboard alerts + Paper trading, with historical backtesting first

---

## API Call Analysis

### Current Usage (Pre-game)
- **The Odds API**: ~3-5 calls/day for upcoming games
- **NBA Stats API**: Free, no tracking
- **Cost**: ~$0.01/request = ~$0.05/day

### Live Betting Usage (Estimated)

**The Odds API** (Live odds):
- Poll frequency: Every 3 minutes during games
- Games per night: ~8 games
- Game duration: ~2.5 hours
- Calls per game: 2.5 hrs × 60 min / 3 min = 50 calls
- **Total calls/night**: 8 games × 50 = **400 calls**
- **Cost/night**: 400 × $0.01 = **$4/day** (game days only)
- **Monthly cost**: ~$120/month (30 game days)

**NBA Stats API** (Live scores):
- Poll frequency: Every 30 seconds
- Calls per game: 2.5 hrs × 120 / 30 = 300 calls
- **FREE** - No API key required

**Total API Cost**: ~$120/month for The Odds API

### Optimization Options

1. **Reduce polling frequency**: Poll odds every 5 min instead of 3 min
   - Saves 33% → **$80/month**

2. **Only monitor games with pre-game edge**: Filter to ~3-4 games/night
   - Saves 50% → **$60/month**

3. **Poll less frequently early in game**:
   - Q1-Q2: Every 5 min
   - Q3-Q4: Every 2 min
   - Saves 30% → **$84/month**

**Recommended**: Combination of #2 + #3 → **~$40-50/month**

---

## Implementation Steps

### Phase 1: Data Collection & Schema (Day 1)

**Goal**: Set up database and start collecting historical data

#### Database Schema

```sql
-- Live game state snapshots
CREATE TABLE live_game_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    game_date TEXT NOT NULL,
    quarter INTEGER,
    time_remaining TEXT,  -- "7:23"
    home_score INTEGER,
    away_score INTEGER,
    home_team TEXT,
    away_team TEXT,
    game_status TEXT,  -- "1st Qtr", "Halftime", "Final"
    game_clock TEXT,  -- "426" (seconds remaining in quarter)
    UNIQUE(game_id, timestamp)
);
CREATE INDEX idx_live_game_date ON live_game_state(game_date);
CREATE INDEX idx_live_game_id ON live_game_state(game_id);

-- Live odds snapshots
CREATE TABLE live_odds_snapshot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    bookmaker TEXT NOT NULL,
    market TEXT NOT NULL,  -- 'h2h', 'spreads', 'totals'
    home_odds INTEGER,
    away_odds INTEGER,
    over_odds INTEGER,
    under_odds INTEGER,
    spread_value REAL,
    total_value REAL,
    UNIQUE(game_id, timestamp, bookmaker, market)
);
CREATE INDEX idx_live_odds_game ON live_odds_snapshot(game_id);
CREATE INDEX idx_live_odds_time ON live_odds_snapshot(timestamp);

-- Detected betting edges (for alerts)
CREATE TABLE live_edge_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    alert_type TEXT NOT NULL,  -- 'spread', 'moneyline', 'total'
    bet_side TEXT,  -- 'home', 'away', 'over', 'under'
    model_prob REAL,  -- Our model's probability
    market_prob REAL,  -- Implied probability from odds
    edge REAL,  -- Difference (model_prob - market_prob)
    quarter INTEGER,
    score_diff INTEGER,
    time_remaining TEXT,
    home_score INTEGER,
    away_score INTEGER,
    line_value REAL,  -- Spread or total value
    odds INTEGER,  -- American odds for this bet
    confidence TEXT,  -- 'HIGH', 'MEDIUM', 'LOW'
    acted_on BOOLEAN DEFAULT 0,
    paper_bet_id INTEGER  -- Link to paper bet if placed
);
CREATE INDEX idx_live_alerts_game ON live_edge_alerts(game_id);
CREATE INDEX idx_live_alerts_confidence ON live_edge_alerts(confidence);

-- Paper trading bets
CREATE TABLE live_paper_bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    placed_at TEXT NOT NULL,
    bet_type TEXT NOT NULL,  -- 'spread', 'moneyline', 'total'
    bet_side TEXT NOT NULL,  -- 'home', 'away', 'over', 'under'
    odds INTEGER,  -- American odds at time of bet
    line_value REAL,  -- Spread or total value
    stake REAL,  -- Hypothetical bet amount
    expected_edge REAL,  -- Edge at time of bet
    model_prob REAL,  -- Model probability at bet time

    -- Game state at bet time
    quarter INTEGER,
    score_diff_at_bet INTEGER,
    home_score_at_bet INTEGER,
    away_score_at_bet INTEGER,
    time_remaining_at_bet TEXT,

    -- Settlement
    outcome TEXT,  -- 'win', 'loss', 'push', 'pending'
    profit REAL,  -- Actual profit/loss
    settled_at TEXT,
    final_home_score INTEGER,
    final_away_score INTEGER
);
CREATE INDEX idx_live_paper_game ON live_paper_bets(game_id);
CREATE INDEX idx_live_paper_outcome ON live_paper_bets(outcome);
```

#### Files to Create

**Database Init**:
```python
# src/data/live_betting_db.py
import sqlite3
import os

DB_PATH = "data/bets/live_betting.db"

def init_database():
    """Initialize live betting database with tables."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    # Create tables (SQL above)
    # ...

    conn.commit()
    conn.close()
```

### Phase 2: Live Data Clients (Days 2-3)

#### Live Game Client

```python
# src/data/live_game_client.py
"""
Fetch live NBA game data from stats.nba.com
"""
import requests
import time
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
from loguru import logger

class LiveGameClient:
    """Client for fetching live NBA game data."""

    BASE_URL = "https://stats.nba.com/stats"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
            'Referer': 'https://www.nba.com/',
            'Origin': 'https://www.nba.com'
        })

    def get_todays_games(self) -> pd.DataFrame:
        """
        Fetch today's scoreboard.

        Returns DataFrame with columns:
        - game_id, game_date, home_team, away_team
        - quarter, time_remaining, home_score, away_score
        - game_status (e.g., "1st Qtr", "Halftime", "Final")
        """
        endpoint = f"{self.BASE_URL}/scoreboardv2"
        params = {
            'GameDate': datetime.now().strftime('%Y-%m-%d'),
            'LeagueID': '00',
            'DayOffset': '0'
        }

        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            games = []
            for game in data['resultSets'][0]['rowSet']:
                # Parse game data
                # game[0] = GAME_DATE_EST
                # game[1] = GAME_SEQUENCE
                # game[2] = GAME_ID
                # game[3] = GAME_STATUS_ID (1=not started, 2=live, 3=final)
                # game[4] = GAME_STATUS_TEXT
                # ...

                games.append({
                    'game_id': game[2],
                    'game_date': game[0],
                    'game_status_id': game[3],
                    'game_status': game[4],
                    # ... parse rest
                })

            return pd.DataFrame(games)

        except Exception as e:
            logger.error(f"Error fetching scoreboard: {e}")
            return pd.DataFrame()

    def get_live_games(self) -> pd.DataFrame:
        """Get only games that are currently live."""
        all_games = self.get_todays_games()
        if all_games.empty:
            return all_games
        return all_games[all_games['game_status_id'] == 2]

    def save_game_state(self, game: Dict) -> None:
        """Save game state snapshot to database."""
        import sqlite3
        from src.data.live_betting_db import DB_PATH

        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT OR IGNORE INTO live_game_state
            (game_id, timestamp, game_date, quarter, time_remaining,
             home_score, away_score, home_team, away_team, game_status, game_clock)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game['game_id'],
            datetime.now().isoformat(),
            game['game_date'],
            game.get('quarter'),
            game.get('time_remaining'),
            game.get('home_score'),
            game.get('away_score'),
            game.get('home_team'),
            game.get('away_team'),
            game.get('game_status'),
            game.get('game_clock')
        ))
        conn.commit()
        conn.close()
```

#### Live Odds Tracker

```python
# src/data/live_odds_tracker.py
"""
Track live odds using The Odds API.
"""
from src.data.odds_api import OddsAPIClient
import sqlite3
from datetime import datetime
from typing import List, Dict
import pandas as pd
from loguru import logger

class LiveOddsTracker:
    """Track live betting odds."""

    def __init__(self):
        self.odds_client = OddsAPIClient()

    def get_live_odds(self, game_ids: List[str] = None) -> pd.DataFrame:
        """
        Fetch current odds for live games.

        Args:
            game_ids: Optional list of game IDs to filter

        Returns:
            DataFrame with live odds
        """
        # Get all current odds
        odds = self.odds_client.get_current_odds(
            markets=["h2h", "spreads", "totals"]
        )

        if odds.empty:
            return odds

        # Filter to live games (commence_time in past)
        now = datetime.now()
        odds['commence_datetime'] = pd.to_datetime(odds['commence_time'])
        live_odds = odds[odds['commence_datetime'] < now]

        # Filter to specific games if provided
        if game_ids:
            live_odds = live_odds[live_odds['game_id'].isin(game_ids)]

        return live_odds

    def save_odds_snapshot(self, odds_df: pd.DataFrame) -> None:
        """Save odds snapshot to database."""
        from src.data.live_betting_db import DB_PATH

        if odds_df.empty:
            return

        conn = sqlite3.connect(DB_PATH)
        timestamp = datetime.now().isoformat()

        for _, row in odds_df.iterrows():
            conn.execute("""
                INSERT OR IGNORE INTO live_odds_snapshot
                (game_id, timestamp, bookmaker, market,
                 home_odds, away_odds, over_odds, under_odds,
                 spread_value, total_value)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['game_id'],
                timestamp,
                row.get('bookmaker', 'unknown'),
                row.get('market', 'unknown'),
                row.get('home_odds'),
                row.get('away_odds'),
                row.get('over_odds'),
                row.get('under_odds'),
                row.get('spread_value'),
                row.get('total_value')
            ))

        conn.commit()
        conn.close()
```

### Phase 3: Win Probability Model (Days 4-5)

```python
# src/models/live_win_probability.py
"""
Calculate live win probability based on game state.

Initially uses simple historical lookup approach:
- Find historical games with similar state (score diff, time, quarter)
- Calculate actual win rate from those games
- Adjust for team quality using pre-game model
"""
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from loguru import logger

class LiveWinProbModel:
    """Calculate win probability for live games."""

    def __init__(self):
        self.historical_db = "data/bets/bets.db"  # Has historical games

    def predict(self, game_state: Dict) -> Dict[str, float]:
        """
        Predict win probability for current game state.

        Args:
            game_state: Dict with:
                - home_score, away_score
                - quarter (1-4)
                - time_remaining (seconds in quarter)
                - home_team, away_team

        Returns:
            Dict with:
                - home_win_prob
                - away_win_prob
                - confidence (0-1)
        """
        score_diff = game_state['home_score'] - game_state['away_score']
        quarter = game_state['quarter']
        time_remaining_q = game_state.get('time_remaining_seconds', 0)

        # Calculate total time remaining in game
        quarters_left = 4 - quarter
        total_time_left = (quarters_left * 720) + time_remaining_q  # 720 sec/quarter

        # Method 1: Simple formula (MVP)
        home_win_prob = self._simple_formula(score_diff, total_time_left)

        # Method 2: Historical lookup (better)
        # historical_prob = self._historical_lookup(score_diff, total_time_left, quarter)

        # For MVP, use simple formula
        # TODO: Add team quality adjustment from pre-game model

        return {
            'home_win_prob': home_win_prob,
            'away_win_prob': 1 - home_win_prob,
            'confidence': 0.7  # Lower in early game, higher late
        }

    def _simple_formula(self, score_diff: int, time_left: int) -> float:
        """
        Simple win probability formula.

        Based on research:
        - Each point of lead ≈ 2-3% win probability shift
        - Time decay: less time = more certain
        - Baseline: 50% at 0-0 with full time
        """
        # Baseline home court advantage (adjust if needed)
        base_prob = 0.50

        # Points per possession value (how much 1 point changes win prob)
        # Decreases as time runs out (points more valuable)
        time_factor = np.sqrt(time_left / 2880)  # 2880 = full game seconds
        point_value = 0.025 * (1 + (1 - time_factor))  # 2.5% early, 5% late

        # Calculate win prob
        win_prob = base_prob + (score_diff * point_value)

        # Clamp to valid range
        win_prob = max(0.01, min(0.99, win_prob))

        return win_prob

    def _historical_lookup(
        self,
        score_diff: int,
        time_left: int,
        quarter: int
    ) -> Tuple[float, float]:
        """
        Look up historical win rate for similar game states.

        Returns:
            (win_probability, confidence)
        """
        conn = sqlite3.connect(self.historical_db)

        # Query historical games with similar state
        # This requires play-by-play data which we'd need to collect
        # TODO: Implement after collecting historical data

        conn.close()
        return 0.5, 0.5
```

### Phase 4: Edge Detection (Day 6)

```python
# src/betting/live_edge_detector.py
"""
Detect betting edges in live games.
"""
from typing import Dict, List, Optional
from datetime import datetime
import sqlite3
from loguru import logger

class LiveEdgeDetector:
    """Detect profitable live betting opportunities."""

    def __init__(
        self,
        min_edge: float = 0.05,  # 5% minimum edge
        min_confidence: float = 0.6,
    ):
        self.min_edge = min_edge
        self.min_confidence = min_confidence

    def find_edges(
        self,
        game_state: Dict,
        live_odds: Dict,
        win_prob: Dict
    ) -> List[Dict]:
        """
        Find betting edges by comparing model vs market.

        Args:
            game_state: Current game state
            live_odds: Current odds for this game
            win_prob: Model's win probability

        Returns:
            List of edge opportunities
        """
        edges = []

        # Check spread edges
        if 'spread' in live_odds:
            spread_edges = self._check_spread_edge(
                game_state, live_odds['spread'], win_prob
            )
            edges.extend(spread_edges)

        # Check moneyline edges
        if 'moneyline' in live_odds:
            ml_edges = self._check_moneyline_edge(
                game_state, live_odds['moneyline'], win_prob
            )
            edges.extend(ml_edges)

        # Check total edges
        if 'total' in live_odds:
            total_edges = self._check_total_edge(
                game_state, live_odds['total'], win_prob
            )
            edges.extend(total_edges)

        # Filter by minimum edge and confidence
        edges = [
            e for e in edges
            if e['edge'] >= self.min_edge
            and e['confidence'] >= self.min_confidence
        ]

        # Add timestamp
        for edge in edges:
            edge['detected_at'] = datetime.now().isoformat()

        return edges

    def _check_spread_edge(
        self,
        game_state: Dict,
        spread_odds: Dict,
        win_prob: Dict
    ) -> List[Dict]:
        """Check for spread betting edges."""
        edges = []

        # Get spread line and odds
        spread = spread_odds.get('spread_value', 0)
        home_odds = spread_odds.get('home_odds', -110)
        away_odds = spread_odds.get('away_odds', -110)

        # Convert odds to implied probability
        home_market_prob = self._odds_to_prob(home_odds)
        away_market_prob = self._odds_to_prob(away_odds)

        # Our model's probability of covering spread
        # Simple approach: if favored by more than spread, higher prob
        score_diff = game_state['home_score'] - game_state['away_score']
        adjusted_diff = score_diff - spread

        # Estimate cover probability (simplified)
        # TODO: More sophisticated model
        home_cover_prob = win_prob['home_win_prob']
        if adjusted_diff > 0:
            home_cover_prob = min(0.95, home_cover_prob + 0.1)
        elif adjusted_diff < 0:
            home_cover_prob = max(0.05, home_cover_prob - 0.1)

        # Check home spread edge
        home_edge = home_cover_prob - home_market_prob
        if abs(home_edge) >= self.min_edge:
            edges.append({
                'game_id': game_state['game_id'],
                'alert_type': 'spread',
                'bet_side': 'home' if home_edge > 0 else 'away',
                'model_prob': home_cover_prob if home_edge > 0 else (1 - home_cover_prob),
                'market_prob': home_market_prob if home_edge > 0 else away_market_prob,
                'edge': abs(home_edge),
                'line_value': spread,
                'odds': home_odds if home_edge > 0 else away_odds,
                'quarter': game_state['quarter'],
                'score_diff': score_diff,
                'time_remaining': game_state.get('time_remaining'),
                'home_score': game_state['home_score'],
                'away_score': game_state['away_score'],
                'confidence': 'HIGH' if abs(home_edge) > 0.10 else 'MEDIUM'
            })

        return edges

    def _check_moneyline_edge(
        self,
        game_state: Dict,
        ml_odds: Dict,
        win_prob: Dict
    ) -> List[Dict]:
        """Check for moneyline betting edges."""
        # Similar to spread but simpler - just win probability
        # TODO: Implement
        return []

    def _check_total_edge(
        self,
        game_state: Dict,
        total_odds: Dict,
        win_prob: Dict
    ) -> List[Dict]:
        """Check for total (over/under) betting edges."""
        # Need to estimate final score based on pace
        # TODO: Implement
        return []

    def _odds_to_prob(self, american_odds: int) -> float:
        """Convert American odds to implied probability."""
        if american_odds < 0:
            return abs(american_odds) / (abs(american_odds) + 100)
        else:
            return 100 / (american_odds + 100)

    def save_alert(self, edge: Dict) -> int:
        """Save edge alert to database, return alert ID."""
        from src.data.live_betting_db import DB_PATH

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.execute("""
            INSERT INTO live_edge_alerts
            (game_id, timestamp, alert_type, bet_side, model_prob, market_prob,
             edge, quarter, score_diff, time_remaining, home_score, away_score,
             line_value, odds, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            edge['game_id'],
            edge['detected_at'],
            edge['alert_type'],
            edge['bet_side'],
            edge['model_prob'],
            edge['market_prob'],
            edge['edge'],
            edge['quarter'],
            edge['score_diff'],
            edge['time_remaining'],
            edge['home_score'],
            edge['away_score'],
            edge.get('line_value'),
            edge.get('odds'),
            edge['confidence']
        ))
        alert_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return alert_id
```

### Phase 5: Monitoring Script (Day 7)

```python
# scripts/live_game_monitor.py
"""
Monitor live games and detect betting edges.

Run during game hours (5-11 PM ET):
    python scripts/live_game_monitor.py
"""
import time
from datetime import datetime, timedelta
from loguru import logger
from src.data.live_game_client import LiveGameClient
from src.data.live_odds_tracker import LiveOddsTracker
from src.models.live_win_probability import LiveWinProbModel
from src.betting.live_edge_detector import LiveEdgeDetector

# Configuration
GAME_POLL_INTERVAL = 30  # Check games every 30 seconds
ODDS_POLL_INTERVAL = 180  # Check odds every 3 minutes
MIN_EDGE = 0.05  # 5% minimum edge

logger.add("logs/live_monitor_{time}.log", rotation="1 day")

def monitor_live_games():
    """Main monitoring loop."""
    game_client = LiveGameClient()
    odds_tracker = LiveOddsTracker()
    win_prob_model = LiveWinProbModel()
    edge_detector = LiveEdgeDetector(min_edge=MIN_EDGE)

    last_odds_check = {}  # Track last odds check per game

    logger.info("🚀 Live game monitoring started")

    try:
        while True:
            # Get live games
            live_games = game_client.get_live_games()

            if live_games.empty:
                logger.info("No live games currently")
                time.sleep(60)
                continue

            logger.info(f"Monitoring {len(live_games)} live games")

            for _, game in live_games.iterrows():
                game_id = game['game_id']

                # Save game state
                game_client.save_game_state(game.to_dict())

                # Check if we should fetch odds (every 3 min per game)
                now = datetime.now()
                last_check = last_odds_check.get(game_id)

                if not last_check or (now - last_check).seconds >= ODDS_POLL_INTERVAL:
                    # Get live odds for this game
                    odds = odds_tracker.get_live_odds([game_id])

                    if not odds.empty:
                        # Save odds snapshot
                        odds_tracker.save_odds_snapshot(odds)

                        # Calculate win probability
                        game_state = game.to_dict()
                        win_prob = win_prob_model.predict(game_state)

                        # Detect edges
                        # Need to structure odds properly
                        live_odds_dict = {
                            'spread': {
                                'spread_value': odds.iloc[0].get('spread_value'),
                                'home_odds': odds.iloc[0].get('home_odds'),
                                'away_odds': odds.iloc[0].get('away_odds'),
                            }
                        }

                        edges = edge_detector.find_edges(
                            game_state, live_odds_dict, win_prob
                        )

                        # Log and save edges
                        for edge in edges:
                            alert_id = edge_detector.save_alert(edge)
                            logger.success(
                                f"🎯 EDGE DETECTED (Alert #{alert_id}): "
                                f"{edge['alert_type'].upper()} - {edge['bet_side'].upper()} "
                                f"Edge: {edge['edge']:.1%}, "
                                f"Q{edge['quarter']} {edge['time_remaining']}, "
                                f"Score: {edge['home_score']}-{edge['away_score']}"
                            )

                    last_odds_check[game_id] = now

            # Sleep before next check
            time.sleep(GAME_POLL_INTERVAL)

    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error in monitoring loop: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    monitor_live_games()
```

### Phase 6: Dashboard Integration (Days 8-9)

Add new tab to `analytics_dashboard.py`:

```python
# Add to analytics_dashboard.py

with tab_live:
    st.header("🔴 Live Betting Opportunities")

    # Show current live games
    live_games = get_live_games()  # Function to query live_game_state

    if live_games.empty:
        st.info("No live games currently. Check back during game hours (7-11 PM ET).")
    else:
        # Display live games in columns
        for _, game in live_games.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 2])

                with col1:
                    st.subheader(f"{game['away_team']} @ {game['home_team']}")
                    st.write(f"Q{game['quarter']} - {game['time_remaining']}")
                    st.metric("Score", f"{game['away_score']} - {game['home_score']}")

                with col2:
                    # Get latest edge alerts for this game
                    alerts = get_game_alerts(game['game_id'])
                    if not alerts.empty:
                        latest = alerts.iloc[0]
                        st.warning(f"🎯 {latest['alert_type'].upper()} Edge")
                        st.write(f"Side: {latest['bet_side'].upper()}")
                        st.write(f"Edge: {latest['edge']:.1%}")
                        st.write(f"Confidence: {latest['confidence']}")

                with col3:
                    # Action buttons
                    if st.button("📝 Place Paper Bet", key=f"paper_{game['game_id']}"):
                        # Modal to place paper bet
                        place_paper_bet(game['game_id'], latest)

                    if st.button("🔕 Dismiss Alert", key=f"dismiss_{game['game_id']}"):
                        dismiss_alert(latest['id'])

    # Show recent alerts
    st.markdown("---")
    st.subheader("Recent Alerts (Last 24h)")

    recent_alerts = get_recent_alerts(hours=24)
    if not recent_alerts.empty:
        st.dataframe(recent_alerts[[
            'timestamp', 'game_id', 'alert_type', 'bet_side',
            'edge', 'quarter', 'score_diff', 'confidence'
        ]])

    # Show paper bet performance
    st.markdown("---")
    st.subheader("Paper Bet Performance")

    paper_bets = get_paper_bets()
    if not paper_bets.empty:
        # Calculate stats
        settled = paper_bets[paper_bets['outcome'] != 'pending']
        if not settled.empty:
            win_rate = (settled['outcome'] == 'win').sum() / len(settled)
            total_profit = settled['profit'].sum()
            roi = (total_profit / settled['stake'].sum()) * 100

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Win Rate", f"{win_rate:.1%}")
            with col2:
                st.metric("Total Profit", f"${total_profit:.2f}")
            with col3:
                st.metric("ROI", f"{roi:.1%}")

        # Show bet history
        st.dataframe(paper_bets)
```

---

## Next Steps

1. **Today**:
   - Create database schema
   - Build live game client
   - Start collecting historical data in background

2. **This Week**:
   - Complete all components
   - Test with live games (monitoring only)

3. **Next Week**:
   - Collect 1-2 weeks of historical data
   - Backtest edge detection
   - Tune thresholds

4. **Week 3**:
   - Paper trading during live games
   - Dashboard alerts
   - Performance tracking

**Ready to start with Phase 1 (database schema)?**
