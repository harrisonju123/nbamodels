# Live Betting System - Quick Usage Guide

## 🎯 What You Have Now

A complete live betting system with:
- ✅ Manual refresh mode (perfect for 500 API calls/month)
- ✅ Live game tracking
- ✅ Live odds monitoring
- ✅ Win probability calculations
- ✅ Edge detection
- ✅ Paper bet tracking & settlement
- ✅ Performance dashboard

---

## 📊 How to Use

### Option 1: Manual Check (CURRENT SETUP)

Check live games whenever you want:

```bash
# Just view current state (no save)
python -m scripts.manual_live_check

# Save data to database
python -m scripts.manual_live_check --save
```

**What you'll see:**
```
============================================================
📊 Live Game Check - 03:26:47 PM
============================================================

🏀 Found 1 live game(s)

────────────────────────────────────────────────────────────
📊 DET @ CLE
   Score: 0-0
   Q2 7:25
   Fetching odds...
   📈 Spread: +5.5 (fanduel)
      Home: -112, Away: -118
   💰 Moneyline: (fanduel)
      Home: +215, Away: -290
   🎯 Total: 238.5 (fanduel)
      Over: -118, Under: -112

   🧮 Win Probability:
      Home: 55.0%
      Away: 45.0%
      Confidence: 42.9%

   ℹ️  No edges detected
```

**API Usage:** 1 call per game checked

### Option 2: Automatic Monitor

If you upgrade to 20k calls/month:

```bash
# Start automatic monitoring (5-min polling)
nohup python -m scripts.live_game_monitor --odds-poll 300 > logs/monitor_output.log 2>&1 &
echo $! > /tmp/live_monitor.pid

# Stop monitoring
kill $(cat /tmp/live_monitor.pid)

# Check status
tail -f logs/monitor_output.log
```

**API Usage:** ~36 calls per game (for full game)

---

## 📈 Dashboard Views

### 1. Performance Analytics Tab

Now includes **Live Betting Performance** section showing:
- Total live bets placed
- Win rate
- Total profit/loss
- ROI
- Outcome breakdown (wins/losses/pushes/pending)
- Performance by confidence level

### 2. Live Betting Tab

Shows:
- **Current live games** with scores and status
- **Active edge alerts** (opportunities to bet)
- **Recent alerts** (last 24 hours)
- **Paper bet performance** with full history
- **Settlement button** to auto-settle completed games

---

## 💰 Paper Bet Settlement

### Automatic Settlement

In the dashboard Live Betting tab, click:
```
🔄 Auto-Settle Completed Games
```

This will:
1. Find all games that have finished
2. Settle pending paper bets for those games
3. Calculate profit/loss
4. Update win/loss records

### Manual Settlement (if needed)

```python
from src.betting.live_bet_settlement import LiveBetSettlement

settler = LiveBetSettlement()

# Settle a specific game
settler.settle_game_bets(
    game_id='0022500494',
    final_home_score=112,
    final_away_score=98
)

# Get stats
stats = settler.get_settlement_stats()
print(stats)
```

---

## 📊 API Usage Planning

### With 500 Calls/Month

**Manual checks only:**
- Check 3-5 times per game
- Monitor 2-3 games per night
- ~450 calls/month
- **Perfect for your current plan!**

### With 20,000 Calls/Month

**Option 1: Monitor ALL games**
- 5-minute polling
- All 8 games/night
- ~8,640 calls/month (43% of budget)
- Best for data collection

**Option 2: Selective monitoring**
- 5-minute polling
- Only games with pre-game edges (2-3/night)
- ~2,160 calls/month (11% of budget)
- Best ROI

**Option 3: Aggressive**
- 3-minute polling
- 4 games/night
- ~7,200 calls/month (36% of budget)
- More frequent updates

---

## 🎯 Workflow

### Current Setup (Manual)

1. **Watch games** on TV or your betting app
2. **When something interesting happens**, run:
   ```bash
   python -m scripts.manual_live_check --save
   ```
3. **Check dashboard** for edge alerts
4. **After games finish**, click "Auto-Settle" in dashboard
5. **Review performance** in Performance Analytics tab

### With Automatic Monitor

1. **Start monitor** before games begin (~1:45 PM ET)
2. **Watch dashboard** Live Betting tab for alerts
3. **Take action** on HIGH confidence edges
4. **After games**, auto-settle bets
5. **Review performance** weekly

---

## 📁 Files Reference

### Scripts
- `scripts/manual_live_check.py` - Manual refresh script
- `scripts/live_game_monitor.py` - Automatic monitor

### Core Logic
- `src/data/live_game_client.py` - NBA game data (FREE)
- `src/data/live_odds_tracker.py` - The Odds API
- `src/models/live_win_probability.py` - Win probability model
- `src/betting/live_edge_detector.py` - Edge detection
- `src/betting/live_bet_settlement.py` - Settlement logic

### Database
- `data/bets/live_betting.db` - All live betting data

### Dashboard
- `analytics_dashboard.py` - Main dashboard (Performance + Live tabs)

---

## 🔍 Checking Your Data

### Quick Stats

```bash
sqlite3 data/bets/live_betting.db "
SELECT
    (SELECT COUNT(*) FROM live_game_state) as game_states,
    (SELECT COUNT(*) FROM live_odds_snapshot) as odds_snapshots,
    (SELECT COUNT(*) FROM live_edge_alerts) as alerts,
    (SELECT COUNT(*) FROM live_paper_bets) as paper_bets
"
```

### View Recent Alerts

```bash
sqlite3 data/bets/live_betting.db "
SELECT
    home_team || ' vs ' || away_team as game,
    alert_type,
    bet_side,
    ROUND(edge * 100, 1) || '%' as edge,
    confidence
FROM live_edge_alerts
ORDER BY timestamp DESC
LIMIT 10
"
```

---

## 💡 Tips

### Maximize Value with Limited API Calls

1. **Only check during key moments:**
   - End of quarters
   - Close games in 4th quarter
   - After momentum shifts

2. **Use your betting apps** for monitoring:
   - Watch odds there
   - Only run script when you see interesting line movement

3. **Batch your checks:**
   - Check multiple games in one run (1 API call per game)

### When to Trust the Model

The model works best when:
- ✅ Confidence > 50%
- ✅ Edge > 8%
- ✅ Significant score differential
- ✅ Later in the game (more data)

Be cautious when:
- ⚠️ Confidence < 45%
- ⚠️ Very early in game (Q1 < 6 min)
- ⚠️ Garbage time (blowout in Q4)

---

## 🚀 Next Steps

1. **Tonight:** Try manual checks during games
2. **This week:** Collect some data, review performance
3. **Next week:** Decide if you want to upgrade to 20k calls/month
4. **After 2 weeks:** Analyze results, tune thresholds

---

## 📞 Support

If you need help:
- Check logs: `tail -f logs/manual_check_*.log`
- Database stats: `python -c "from src.data.live_betting_db import get_stats; print(get_stats())"`
- Test script: `python -m scripts.manual_live_check`

---

**You're all set! Good luck with the live betting!** 🍀📊🏀
