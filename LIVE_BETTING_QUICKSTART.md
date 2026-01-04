# Live Betting System - Quick Start Guide

**Status**: ✅ Core system complete and ready for testing

---

## 🎯 What We Built

A complete live betting monitoring system that:
- ✅ Tracks live NBA games in real-time
- ✅ Fetches live odds from The Odds API
- ✅ Calculates win probabilities based on game state
- ✅ Detects betting edges (5%+ by default)
- ✅ Saves all data for backtesting
- ✅ Alerts on high-confidence opportunities

---

## 🚀 Quick Start

### 1. Start Monitoring (Tonight's Games)

Games start at 2:00 PM ET today. Run the monitor:

```bash
# Start monitoring (saves to database)
python -m scripts.live_game_monitor

# Or dry-run mode (no database saves, just prints)
python -m scripts.live_game_monitor --dry-run

# Custom settings
python -m scripts.live_game_monitor --min-edge 0.03 --game-poll 60
```

**Options**:
- `--dry-run`: Test mode, no database saves
- `--min-edge 0.05`: Minimum edge to alert (default 5%)
- `--game-poll 30`: Game check interval in seconds (default 30s)
- `--odds-poll 180`: Odds check interval in seconds (default 3min)

### 2. Monitor Output

The system will show:

```
--- Iteration 1 at 02:37:13 PM ---
Monitoring 3 live games

📊 MIN @ WAS
   Score: 28-31
   Q1 8:45
   Fetching odds...
   Odds: Spread=-5.5
   Win Prob: Home 72.3% (confidence: 61.2%)

   🎯 EDGE DETECTED - SPREAD HOME
      Edge: 8.2%
      Confidence: HIGH
      Model: 75.1%, Market: 66.9%
      Odds: -110
      Line: -5.5
      Game State: Q1, Score 31-28, 8:45
      Alert ID: 1
```

### 3. Stop Monitoring

Press `Ctrl+C` to stop. You'll see a summary:

```
📊 Monitoring Summary:
   Game states saved: 245
   Odds snapshots: 82
   Edge alerts: 12
```

---

## 📊 Data Collection

All data is saved to `data/bets/live_betting.db`:

### Game State Snapshots
Every 30 seconds per live game:
- Score, quarter, time remaining
- Team names, game status

### Odds Snapshots
Every 3 minutes per live game:
- Spread, moneyline, totals
- All bookmakers
- Line movements tracked

### Edge Alerts
Whenever edge ≥ 5%:
- Alert type (spread/moneyline/total)
- Bet side and recommended line
- Model vs market probabilities
- Confidence level
- Game state at detection

---

## 🗄️ Database Queries

### View Recent Alerts

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/bets/live_betting.db')

# Get today's alerts
alerts = pd.read_sql_query("""
    SELECT
        timestamp,
        home_team || ' vs ' || away_team as game,
        alert_type,
        bet_side,
        ROUND(edge * 100, 1) as edge_pct,
        confidence,
        quarter,
        score_diff
    FROM live_edge_alerts
    WHERE DATE(timestamp) = DATE('now')
    ORDER BY timestamp DESC
""", conn)

print(alerts)
conn.close()
```

### View Game State History

```python
# Get all snapshots for a game
states = pd.read_sql_query("""
    SELECT timestamp, quarter, time_remaining, home_score, away_score
    FROM live_game_state
    WHERE game_id = 'YOUR_GAME_ID'
    ORDER BY timestamp
""", conn)
```

### View Odds Movements

```python
# Track spread movement for a game
odds = pd.read_sql_query("""
    SELECT timestamp, bookmaker, spread_value, home_odds, away_odds
    FROM live_odds_snapshot
    WHERE game_id = 'YOUR_GAME_ID' AND market = 'spreads'
    ORDER BY timestamp
""", conn)
```

---

## 📈 Using the Data

### For Backtesting

After collecting 1-2 weeks of data:

```python
from src.data.live_betting_db import get_stats

stats = get_stats()
print(f"Game states: {stats['live_game_state']}")
print(f"Odds snapshots: {stats['live_odds_snapshot']}")
print(f"Alerts: {stats['live_edge_alerts']}")
```

### For Analysis

```python
# Win rate of edge alerts
results = pd.read_sql_query("""
    SELECT
        confidence,
        COUNT(*) as total_alerts,
        AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END) as win_rate
    FROM live_edge_alerts a
    LEFT JOIN live_paper_bets b ON a.paper_bet_id = b.id
    WHERE b.outcome IS NOT NULL
    GROUP BY confidence
""", conn)
```

---

## 🎮 Tonight's Plan

### Phase 1: Data Collection (Today)

1. **Start monitor at 1:45 PM ET** (before first game)
   ```bash
   python -m scripts.live_game_monitor
   ```

2. **Let it run through all games** (until ~midnight)

3. **Review alerts in the morning**
   - How many edges detected?
   - What confidence levels?
   - Any patterns?

### Phase 2: Analysis (Tomorrow)

1. **Query the database** to see what we captured
2. **Check alert accuracy** - did high-edge opportunities actually have value?
3. **Tune thresholds** - maybe 5% is too low/high?

### Phase 3: Dashboard (Next Few Days)

Build live betting tab in dashboard to:
- Show current live games
- Display active alerts
- Track paper bet performance

---

## 💡 Tips

### Optimizing API Costs

The monitor uses The Odds API which costs ~$0.01/request:

**Current settings** (default):
- 8 games/night × 50 calls/game = 400 calls = **$4/night**

**Optimized settings**:
```bash
# Only check odds every 5 minutes instead of 3
python -m scripts.live_game_monitor --odds-poll 300

# This saves 33% → ~$2.67/night
```

**Only monitor games with pre-game edge**:
- Filter to 3-4 games → **$1.50-2/night**

### Managing Alerts

- **HIGH confidence** alerts: Most reliable, consider placing paper bets
- **MEDIUM confidence**: Log and review patterns
- **LOW confidence**: Ignore or use for data collection only

### Avoiding False Positives

Some situations create false edges:
- **Garbage time** (blowouts, Q4 < 2min): Model overconfident
- **TV timeouts**: Odds adjust quickly, might miss window
- **Free throw situations**: Temporary score inflation

The monitor doesn't filter these yet - you'll see them in data.

---

## 🔧 Troubleshooting

### "No ODDS_API_KEY found"

Set your Odds API key in `.env`:
```bash
ODDS_API_KEY=your_key_here
```

### "No live games"

Games haven't started yet. First game today is 2:00 PM ET.

### Monitor crashes

Check logs:
```bash
tail -f logs/live_monitor_*.log
```

### Database locked

Only run one monitor instance at a time.

---

## 📁 Files Created

```
src/data/
├── live_betting_db.py          # Database schema & utilities
├── live_game_client.py         # NBA Stats API client (FREE)
├── live_odds_tracker.py        # The Odds API client

src/models/
└── live_win_probability.py    # Win probability calculator

src/betting/
└── live_edge_detector.py      # Edge detection logic

scripts/
└── live_game_monitor.py        # Main monitoring script

data/bets/
└── live_betting.db             # SQLite database
```

---

## 🎯 Next Steps

### Immediate (Today)
- [x] Start monitoring tonight's games
- [ ] Collect ~8 games worth of data
- [ ] Review alerts in morning

### Short-term (This Week)
- [ ] Build dashboard tab for live betting
- [ ] Add paper bet tracking
- [ ] Backtest with collected data

### Medium-term (Next Week)
- [ ] Tune edge thresholds based on results
- [ ] Add filters for garbage time/TV timeouts
- [ ] Implement paper trading automation

### Long-term (When Profitable)
- [ ] Auto-betting with strict limits
- [ ] SMS/email alerts
- [ ] Multi-bookmaker line shopping

---

## 📊 Success Metrics

After 1 week of data collection, we should have:
- ✅ 50+ edge alerts logged
- ✅ Average edge > 5% on alerts
- ✅ High confidence alerts > 10
- ✅ Complete game state history

After 2 weeks of paper trading:
- ✅ Win rate on HIGH confidence > 55%
- ✅ ROI on paper bets > 3%
- ✅ 20+ paper bets placed

---

**Ready to start monitoring!** 🚀

Run this before the 2 PM game:
```bash
python -m scripts.live_game_monitor
```
