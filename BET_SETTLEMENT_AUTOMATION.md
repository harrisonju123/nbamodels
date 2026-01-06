# Automatic Bet Settlement

**Date:** 2026-01-04
**Status:** ✅ ACTIVE

---

## Summary

Added automatic bet settlement to run daily at 2 AM, ensuring all completed games are settled and profits/losses are calculated before the next day's CLV analysis.

---

## Cron Schedule

```bash
# Bet settlement (2 AM daily - settles bets after all games finish)
0 2 * * * cd /Users/harrisonju/PycharmProjects/nbamodels && /Users/harrisonju/.asdf/installs/python/3.12.4/bin/python3 scripts/settle_bets.py >> logs/settle.log 2>&1
```

**Time:** 2:00 AM daily (local time)
**Log file:** `logs/settle.log`

---

## Daily Workflow Timeline

| Time | Task | Purpose |
|------|------|---------|
| **2:00 AM** | **Settle bets** | Process completed games from previous day |
| 6:00 AM | Calculate CLV | Analyze bet performance vs closing lines |
| 6:15 AM | Validate closing lines | Verify best closing line sources |
| 10:00 AM | Collect referees | Get referee assignments for today's games |
| 5:00 PM - 11:00 PM | Collect lineups | Get confirmed lineups (every 15 min) |
| Every hour | Collect news | Get latest NBA news articles |
| Every 15 min | Capture lines | Track opening/closing line movements |

---

## What Gets Settled

### Automatic Settlement Process

1. **Find pending bets** - Bets with `outcome = NULL`
2. **Fetch game results** - From NBA Stats API
3. **Calculate outcome** - Check if spread covered
4. **Update profit/loss** - Based on odds and bet amount
5. **Update bankroll** - Automatic bankroll tracking

### Settlement Logic

**Spread Bets:**
```python
if bet_side == 'home':
    covers = (home_score - away_score + spread > 0)
else:  # away
    covers = (away_score - home_score - spread > 0)

if covers:
    profit = bet_amount * (odds / 100) if odds > 0 else bet_amount / (abs(odds) / 100)
else:
    profit = -bet_amount
```

**Total Bets:**
```python
total_score = home_score + away_score

if bet_side == 'over':
    covers = total_score > total_line
else:  # under
    covers = total_score < total_line
```

---

## Log Monitoring

### Check Settlement Logs

```bash
# View today's settlements
tail -50 logs/settle.log

# Watch settlements in real-time (during manual run)
tail -f logs/settle.log

# Check settlement history
grep "Settled" logs/settle.log | tail -20

# Count settlements by date
grep "Settled" logs/settle.log | awk '{print $1}' | uniq -c
```

### Expected Output

```
2026-01-04 02:00:00 | INFO | 🏁 BET SETTLEMENT
2026-01-04 02:00:00 | INFO | Found 8 pending bets
2026-01-04 02:00:01 | INFO | Processing: Cleveland Cavaliers @ Detroit Pistons (home)
2026-01-04 02:00:01 | SUCCESS | ✅ Settled: CLE 110 - DET 102 | Bet: DET +3.5 | Result: LOSS (-$25.00)
2026-01-04 02:00:02 | INFO | Processing: Brooklyn Nets @ Denver Nuggets (home)
2026-01-04 02:00:02 | SUCCESS | ✅ Settled: BKN 105 - DEN 125 | Bet: DEN -9.5 | Result: WIN (+$22.73)
...
2026-01-04 02:00:10 | SUCCESS | 🎉 Settled 8 bets | Won: 5 | Lost: 3 | Profit: +$87.25
```

---

## Manual Settlement

### Run Settlement Manually

```bash
# Settle all pending bets
python scripts/settle_bets.py

# Dry run (show what would be settled)
python scripts/settle_bets.py --dry-run

# Settle specific game
python scripts/settle_bets.py --game-id "0022500123"
```

### When to Run Manually

- **Testing** - Verify settlement logic with dry run
- **Immediate settlement** - Don't want to wait until 2 AM
- **Debugging** - Check specific game settlement
- **Backfill** - Settle older games that were missed

---

## Error Handling

### Common Issues

**1. Game Not Found**
```
⏳ Game not finished or not found - skipping
```
**Cause:** Game hasn't finished yet or NBA API delay
**Solution:** Wait for game to finish, will auto-settle at 2 AM

**2. API Timeout**
```
ERROR | Error fetching game result: Read timed out
```
**Cause:** NBA Stats API slow or unavailable
**Solution:** Script will retry automatically at next 2 AM run

**3. No Pending Bets**
```
Found 0 pending bets
```
**Cause:** All bets already settled
**Solution:** Normal - nothing to do

### Logs to Check

```bash
# Check for errors
grep "ERROR" logs/settle.log | tail -20

# Check for warnings (games not found)
grep "WARNING" logs/settle.log | tail -20

# Check successful settlements
grep "SUCCESS" logs/settle.log | tail -20
```

---

## Database Updates

### Bet Settlement Fields

When a bet is settled, these fields are updated:

```sql
UPDATE bets SET
    outcome = 1,           -- 1 = win, 0 = loss
    profit = 22.73,        -- Profit/loss in dollars
    home_score = 125,      -- Final home score
    away_score = 105,      -- Final away score
    actual_total = 230,    -- Total points scored
    settled_at = '2026-01-04T02:00:05'
WHERE id = 'bet_abc123';
```

### Bankroll Updates

```sql
-- Bankroll is automatically updated
INSERT INTO bankroll_history (
    timestamp,
    balance,
    change_amount,
    change_type,
    bet_id,
    notes
) VALUES (
    '2026-01-04T02:00:05',
    1087.25,
    +87.25,
    'bet_settlement',
    'bet_abc123',
    'Settled 8 bets: 5W-3L'
);
```

---

## Verification

### Verify Settlement is Working

```bash
# Check crontab
crontab -l | grep "settle_bets"

# Check if script runs without errors
python scripts/settle_bets.py --dry-run

# Verify log file is being created
ls -lh logs/settle.log

# Check last settlement
tail -1 logs/settle.log
```

### Expected Daily Flow

**Day 1 (Jan 4):**
- Place 8 bets during the day
- Games finish 7-11 PM
- **2 AM (Jan 5):** Automatic settlement runs
- Bets settled, profits calculated
- Bankroll updated

**Day 2 (Jan 5):**
- 6 AM: CLV calculation runs on settled bets
- Dashboard shows yesterday's settled bets
- Place new bets for today

---

## Statistics

### Current Unsettled Bets

```bash
# Count unsettled bets
sqlite3 data/bets/bets.db "SELECT COUNT(*) FROM bets WHERE outcome IS NULL"
```

### Settlement Success Rate

```bash
# Settlement history
sqlite3 data/bets/bets.db "
SELECT
    DATE(settled_at) as date,
    COUNT(*) as settled_bets,
    SUM(CASE WHEN outcome = 1 THEN 1 ELSE 0 END) as wins,
    SUM(profit) as total_profit
FROM bets
WHERE settled_at IS NOT NULL
GROUP BY DATE(settled_at)
ORDER BY date DESC
LIMIT 7"
```

---

## Benefits

### Automated Workflow
- ✅ **No manual intervention** - Runs automatically every night
- ✅ **Consistent timing** - Always settles at 2 AM
- ✅ **Logged results** - Easy to audit and debug

### Data Quality
- ✅ **Fresh data for CLV** - Settled bets ready for 6 AM CLV calculation
- ✅ **Accurate bankroll** - Always up to date
- ✅ **Complete history** - All settlements logged

### Time Savings
- ✅ **Saves 5-10 min daily** - No manual settlement needed
- ✅ **Reduces errors** - Automated process is consistent
- ✅ **Better workflow** - Wake up to settled bets

---

## Troubleshooting

### Settlement Not Running

**Check cron status:**
```bash
# Verify cron is running
ps aux | grep cron

# Check system logs (macOS)
log show --predicate 'process == "cron"' --last 1h

# Check if script exists
ls -lh scripts/settle_bets.py
```

**Test manually:**
```bash
# Run settlement manually to test
cd /Users/harrisonju/PycharmProjects/nbamodels
python scripts/settle_bets.py
```

### Bets Not Settling

**Check pending bets:**
```bash
sqlite3 data/bets/bets.db "
SELECT id, home_team, away_team, commence_time
FROM bets
WHERE outcome IS NULL
ORDER BY commence_time"
```

**Check game results:**
```bash
# Verify games finished
python -c "
from src.data.nba_stats import NBAStatsClient
client = NBAStatsClient()
games = client.get_season_games(2025)
print(f'Found {len(games)} games')
print(games[['date', 'home_team', 'away_team', 'status']].tail())
"
```

---

## Next Steps

### Optional Enhancements

1. **Notification system** - Get notified when bets settle
2. **Settlement summary email** - Daily summary of settled bets
3. **Error alerts** - Alert if settlement fails
4. **Multi-book support** - Settle bets from multiple sportsbooks

---

## Summary

✅ **Automatic bet settlement active**
✅ **Runs daily at 2:00 AM**
✅ **Logs to `logs/settle.log`**
✅ **Updates bankroll automatically**
✅ **Ready for CLV calculation at 6 AM**

**Bets will now settle automatically every night, ensuring your performance analytics are always up to date.**

---

**Updated:** 2026-01-04
**Cron backup:** `/tmp/crontab_backup_*.txt`
**Status:** ✅ ACTIVE
