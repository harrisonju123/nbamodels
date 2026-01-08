# Live Betting System - COMPLETE ✅

**Date**: January 4, 2026
**Status**: 7/7 core tasks complete, ready for production testing

---

## 🎉 What We Built

A complete end-to-end live betting system in one day!

### ✅ All Components Complete

1. **Database** - SQLite database with 4 tables for live data
2. **Game Client** - Real-time NBA scores from stats.nba.com (FREE)
3. **Odds Tracker** - Live odds from The Odds API
4. **Win Probability Model** - Formula-based live probabilities
5. **Edge Detector** - Identifies 5%+ edges comparing model vs market
6. **Monitoring Script** - Polls games/odds and saves alerts
7. **Dashboard UI** - Interactive Streamlit dashboard with live tab

---

## 🚀 How to Use It Tonight

### Step 1: Start the Monitor (1:45 PM ET)

Open a terminal and run:

```bash
python -m scripts.live_game_monitor
```

This will:
- Check for live games every 30 seconds
- Fetch odds every 3 minutes
- Save all data to database
- Print alerts when edges are found

**Expected output:**
```
🚀 Live Game Monitor initialized
   Min edge: 5.0%
   Game poll: 30s
   Odds poll: 180s
============================================================

--- Iteration 1 at 02:37:13 PM ---
Monitoring 3 live games

📊 MIN @ WAS
   Score: 28-31
   Q1 8:45
   Win Prob: Home 72.3% (confidence: 61.2%)

   🎯 EDGE DETECTED - SPREAD HOME
      Edge: 8.2%
      Confidence: HIGH
      Model: 75.1%, Market: 66.9%
      Odds: -110, Line: -5.5
      Alert ID: 1
```

**Keep this running all night!**

### Step 2: Open the Dashboard

In a new terminal (leave monitor running):

```bash
streamlit run analytics_dashboard.py
```

Then click the **"🔴 Live Betting"** tab.

You'll see:
- Current live games with scores
- Active edge alerts with confidence levels
- Recent alerts table
- Collection statistics
- Paper trading performance (when available)

**Dashboard auto-refreshes** - just leave it open!

---

## 📊 What to Expect Tonight

### API Costs
- **Estimated**: $4 for 8 games (~400 API calls)
- **Actual**: Will vary based on game length and odds availability

### Data Collected
- **Game States**: ~200-300 snapshots (every 30s per game)
- **Odds Snapshots**: ~80-100 snapshots (every 3min per game)
- **Edge Alerts**: ~10-20 (depends on opportunities)

### What We'll Learn
1. How often do edges actually appear?
2. What confidence levels are most common?
3. Is 5% min edge threshold appropriate?
4. Does the model make sense vs reality?
5. Do edges persist or disappear quickly?

---

## 🗄️ Accessing the Data

### Database Location
```
data/bets/live_betting.db
```

### Quick Queries

**View Today's Alerts:**
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/bets/live_betting.db')

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

**View Game State Timeline:**
```python
# See how a game progressed
states = pd.read_sql_query("""
    SELECT timestamp, quarter, time_remaining,
           home_score, away_score
    FROM live_game_state
    WHERE game_id = 'YOUR_GAME_ID'
    ORDER BY timestamp
""", conn)

print(states)
```

**Track Odds Movements:**
```python
# See spread movement over time
odds = pd.read_sql_query("""
    SELECT timestamp, bookmaker,
           spread_value, home_odds, away_odds
    FROM live_odds_snapshot
    WHERE game_id = 'YOUR_GAME_ID'
    AND market = 'spreads'
    ORDER BY timestamp
""", conn)

print(odds)
```

### Python API

```python
from src.data.live_betting_db import get_stats

stats = get_stats()
print(f"Game states: {stats['live_game_state']}")
print(f"Odds snapshots: {stats['live_odds_snapshot']}")
print(f"Alerts: {stats['live_edge_alerts']}")
```

---

## 📅 Next Steps

### Tomorrow Morning (Review Data)

1. **Check collection stats:**
   ```python
   from src.data.live_betting_db import get_stats
   print(get_stats())
   ```

2. **Review alerts:**
   - How many total?
   - How many HIGH confidence?
   - Any patterns by quarter?

3. **Analyze opportunities:**
   - Did edges persist or vanish quickly?
   - Were spreads or moneylines more common?
   - What game states trigger most alerts?

### This Week (Analysis & Tuning)

1. **Backtest alerts** - Check if opportunities were real
2. **Tune thresholds** - Maybe 5% too high/low?
3. **Add filters** - Skip garbage time, TV timeouts
4. **Paper trading** - Start tracking hypothetical bets

### Next Week (Refinement)

1. **Implement paper bets** - Auto-log opportunities
2. **Settlement logic** - Track if paper bets won
3. **Performance dashboard** - Win rate, ROI tracking
4. **Strategy optimization** - Learn from results

### When Profitable (Scale Up)

1. **Auto-betting** - With strict limits
2. **SMS/Email alerts** - Get notified immediately
3. **Line shopping** - Compare across books
4. **Increase stakes** - Scale winning strategy

---

## 🔧 Configuration Options

### Monitor Settings

```bash
# Slower polling (saves API calls)
python -m scripts.live_game_monitor --odds-poll 300

# Lower edge threshold (more alerts)
python -m scripts.live_game_monitor --min-edge 0.03

# Dry run (test without saving)
python -m scripts.live_game_monitor --dry-run

# All together
python -m scripts.live_game_monitor --min-edge 0.04 --odds-poll 240
```

### Dashboard Settings

Dashboard auto-refreshes based on Streamlit settings. To change refresh rate, click the ⋮ menu → Settings → Run on save.

---

## 💡 Tips & Best Practices

### Managing Costs

**Current**: 400 calls/night = $4
**Optimized**: 200 calls/night = $2

Ways to save:
- Only monitor games with pre-game edge
- Poll odds less frequently (5min instead of 3min)
- Skip blowouts (>15 point lead in Q4)

### Interpreting Alerts

**HIGH Confidence** (🟢):
- Large edge (>8%)
- High model confidence
- Best opportunities

**MEDIUM Confidence** (🟡):
- 5-8% edge
- Moderate model confidence
- Worth considering

**LOW Confidence** (🟠):
- Just above threshold
- Low model confidence
- Track for learning only

### Avoiding False Positives

Some situations create misleading edges:

1. **Garbage time** (blowout + Q4 <2min)
   - Starters sit, model overconfident
   - Solution: Filter these out

2. **TV timeouts**
   - Odds adjust instantly
   - By the time we see it, edge gone
   - Solution: Accept some misses

3. **Free throw situations**
   - Score inflates temporarily
   - Model assumes it holds
   - Solution: Wait for settled state

(These filters not implemented yet - will add based on data)

---

## 📈 Success Metrics

### After Tonight (1 game day)
- ✅ System runs without crashing
- ✅ 50+ game states collected
- ✅ 10+ odds snapshots
- ✅ At least 1 alert logged

### After 1 Week
- ✅ 50+ edge alerts logged
- ✅ Average alert edge >5%
- ✅ HIGH confidence alerts >10
- ✅ Complete game history for analysis

### After 2 Weeks
- ✅ 20+ paper bets tracked
- ✅ WIN rate on HIGH alerts >55%
- ✅ Positive ROI on paper bets
- ✅ Strategy validation complete

### After 1 Month
- ✅ 50+ paper bets
- ✅ Consistent >55% win rate
- ✅ >5% ROI
- ✅ Ready for real money (small stakes)

---

## 🐛 Troubleshooting

### "No live games currently"

**Causes:**
- Games haven't started yet (first at 2 PM ET)
- Monitor started too early/late
- All games finished

**Solution:** Wait for game time, check NBA schedule

### "No ODDS_API_KEY found"

**Cause:** API key not in .env file

**Solution:**
```bash
# Add to .env file
echo "ODDS_API_KEY=your_key_here" >> .env
```

### Monitor crashes

**Check logs:**
```bash
tail -f logs/live_monitor_*.log
```

**Common causes:**
- API rate limit hit (unlikely at 180s interval)
- Database locked (don't run 2 monitors)
- Network issue (restart monitor)

### Dashboard shows "no data"

**Causes:**
- Monitor not running yet
- Database file doesn't exist
- No games have been monitored

**Solution:**
1. Start monitor first
2. Wait for at least one game
3. Refresh dashboard

### Alerts seem wrong

This is expected in first few days! The model needs tuning based on real data.

**Track these issues:**
- Edge appears but disappears fast
- Model way off on blowouts
- False edges in specific quarters

We'll fix these after analyzing patterns.

---

## 📁 System Architecture

```
Live Betting System
│
├── Data Collection
│   ├── LiveGameClient (NBA Stats API)     FREE
│   ├── LiveOddsTracker (The Odds API)     ~$4/night
│   └── Database (SQLite)                  Local
│
├── Analysis
│   ├── LiveWinProbModel                   Formula-based
│   └── LiveEdgeDetector                   Model vs Market
│
├── Monitoring
│   └── live_game_monitor.py               Main loop
│
└── UI
    └── Dashboard Live Tab                 Streamlit
```

---

## 🎯 Today's Checklist

- [x] Build database schema
- [x] Build game client
- [x] Build odds tracker
- [x] Build win probability model
- [x] Build edge detector
- [x] Build monitoring script
- [x] Add dashboard tab
- [ ] **Start monitor at 1:45 PM ET** ← DO THIS!
- [ ] **Open dashboard** ← AND THIS!
- [ ] Let it run through all games
- [ ] Review data in morning

---

## 🚀 Ready to Launch!

Everything is built and tested. Just need to run it!

**Commands to run in ~45 minutes:**

Terminal 1:
```bash
python -m scripts.live_game_monitor
```

Terminal 2:
```bash
streamlit run analytics_dashboard.py
# Then click "🔴 Live Betting" tab
```

**That's it!** The system will do everything automatically.

Good luck! 🍀📊🏀
