# Cron Jobs Fixed - Alternative Data Collection

**Date:** 2026-01-04
**Status:** ✅ COMPLETE

---

## Problem

Cron jobs were failing due to incorrect Python path:
```bash
/Users/harrisonju/.asdf/shims/python3: line 3: exec: asdf: not found
```

The asdf shim doesn't work in cron environment because asdf isn't properly initialized.

---

## Solution

Updated all cron jobs to use the actual Python executable path:

**Before:**
```bash
/Users/harrisonju/.asdf/shims/python3
```

**After:**
```bash
/Users/harrisonju/.asdf/installs/python/3.12.4/bin/python3
```

Also ensured all jobs include `cd` to working directory.

---

## Updated Cron Schedule

### CLV Tracking (Already Working)
```bash
# Hourly line snapshots
0 * * * * cd /Users/harrisonju/PycharmProjects/nbamodels && /Users/harrisonju/.asdf/installs/python/3.12.4/bin/python3 scripts/collect_line_snapshots.py >> logs/snapshots.log 2>&1

# Opening line capture (every 15 min)
*/15 * * * * cd /Users/harrisonju/PycharmProjects/nbamodels && /Users/harrisonju/.asdf/installs/python/3.12.4/bin/python3 scripts/capture_opening_lines.py >> logs/opening.log 2>&1

# Closing line capture (every 15 min)
*/15 * * * * cd /Users/harrisonju/PycharmProjects/nbamodels && /Users/harrisonju/.asdf/installs/python/3.12.4/bin/python3 scripts/capture_closing_lines.py >> logs/closing.log 2>&1

# Daily CLV calculation (6 AM)
0 6 * * * cd /Users/harrisonju/PycharmProjects/nbamodels && /Users/harrisonju/.asdf/installs/python/3.12.4/bin/python3 scripts/populate_clv_data.py >> logs/clv.log 2>&1

# Closing line validation (6:15 AM)
15 6 * * * cd /Users/harrisonju/PycharmProjects/nbamodels && /Users/harrisonju/.asdf/installs/python/3.12.4/bin/python3 scripts/validate_closing_lines.py >> logs/validate.log 2>&1
```

### Alternative Data Collection (NOW FIXED)
```bash
# Referee assignments (10 AM daily)
0 10 * * * cd /Users/harrisonju/PycharmProjects/nbamodels && /Users/harrisonju/.asdf/installs/python/3.12.4/bin/python3 scripts/collect_referees.py >> logs/referees.log 2>&1

# Lineup collection (every 15 min during game hours 5-11 PM ET)
*/15 17-23 * * * cd /Users/harrisonju/PycharmProjects/nbamodels && /Users/harrisonju/.asdf/installs/python/3.12.4/bin/python3 scripts/collect_lineups.py >> logs/lineups.log 2>&1

# News collection (hourly)
0 * * * * cd /Users/harrisonju/PycharmProjects/nbamodels && /Users/harrisonju/.asdf/installs/python/3.12.4/bin/python3 scripts/collect_news.py >> logs/news.log 2>&1
```

---

## Verification

### ✅ News Collection - WORKING
```bash
# Tested manually
/Users/harrisonju/.asdf/installs/python/3.12.4/bin/python3 scripts/collect_news.py

# Result:
✅ Fetched 62 articles (ESPN: 12, Yahoo: 50)
✅ Saved 3 new articles
✅ No errors
```

### ⏳ Lineup Collection - WAITING FOR DATA
```bash
# Tested manually
/Users/harrisonju/.asdf/installs/python/3.12.4/bin/python3 scripts/collect_lineups.py

# Result:
✅ Script runs without errors
✅ Found 8 games today:
   - CLE vs DET
   - ORL vs IND
   - BKN vs DEN
   - MIA vs NOP
   - WAS vs MIN
   - PHX vs OKC
   - SAC vs MIL
   - LAL vs MEM
⏳ No roster data available yet (normal - posted 30-60 min before tip-off)
```

### ⏳ Referee Collection - WAITING FOR DATA
```bash
# Tested manually
/Users/harrisonju/.asdf/installs/python/3.12.4/bin/python3 scripts/collect_referees.py

# Result:
✅ Script runs without errors
⏳ No referee assignments available yet (normal - posted closer to game time)
```

---

## Current Data Status

| Data Source | Status | Records | Next Update |
|-------------|--------|---------|-------------|
| **News** | ✅ Working | 69 articles | Hourly (every :00) |
| **Lineups** | ⏳ Waiting | 0 lineups | Every 15 min (5-11 PM) |
| **Referees** | ⏳ Waiting | 0 assignments | Daily (10 AM) |
| **Sentiment** | 🔒 Disabled | 0 scores | Requires API keys |

---

## What Happens Next

### Automatic Collection (via Cron)

1. **News** - Collects every hour
   - ✅ Already collecting successfully
   - 69 articles in database
   - Sources: ESPN, Yahoo

2. **Lineups** - Collects every 15 min during game hours (5-11 PM ET)
   - 🕐 Next run: 5:45 PM, 6:00 PM, 6:15 PM...
   - Will populate once lineups are posted (30-60 min before games)

3. **Referees** - Collects once daily at 10 AM
   - 🕐 Next run: Tomorrow 10:00 AM
   - Will populate when NBA releases referee assignments

### Manual Testing (Optional)

You can manually run any collection script:
```bash
# News (should work immediately)
python scripts/collect_news.py

# Lineups (will work 30-60 min before games)
python scripts/collect_lineups.py

# Referees (will work when NBA posts assignments)
python scripts/collect_referees.py
```

---

## Expected Timeline for Data

### Today (Jan 4, 2026)
- ✅ **Now**: News data collecting hourly
- ⏳ **6:00-6:30 PM**: Lineups should start appearing (30-60 min before first game)
- ⏳ **6:30-7:00 PM**: More lineups as games approach

### Tomorrow (Jan 5, 2026)
- ⏳ **10:00 AM**: Referee assignments collected
- ⏳ **Throughout day**: News articles collected hourly
- ⏳ **Evening**: Lineups collected for tomorrow's games

---

## Monitoring

### Check Collection Logs
```bash
# News log
tail -f logs/news.log

# Lineup log (after 5 PM)
tail -f logs/lineups.log

# Referee log (after 10 AM tomorrow)
tail -f logs/referees.log
```

### Check Database
```bash
# Count records
sqlite3 data/bets/bets.db "
SELECT 'news_articles' as table_name, COUNT(*) FROM news_articles
UNION ALL
SELECT 'confirmed_lineups', COUNT(*) FROM confirmed_lineups
UNION ALL
SELECT 'referee_assignments', COUNT(*) FROM referee_assignments"
```

---

## Troubleshooting

### If cron jobs still fail
1. Check logs: `cat logs/news.log`
2. Verify Python path: `python3 -c "import sys; print(sys.executable)"`
3. Manually run script to see errors: `python scripts/collect_news.py`

### If no data appears
- **News**: Should work immediately (check logs)
- **Lineups**: Normal until 30-60 min before games
- **Referees**: Normal until 10 AM or closer to game time

---

## Summary

✅ **All cron jobs fixed with correct Python path**
✅ **News collection working (69 articles collected)**
✅ **Lineup collection ready (waiting for data)**
✅ **Referee collection ready (waiting for data)**
✅ **Automated collection scheduled and active**

**Next Steps:**
- ✅ Cron will automatically collect data on schedule
- ⏳ Check logs tonight to verify lineup collection
- ⏳ Check logs tomorrow morning to verify referee collection

---

**Updated:** 2026-01-04 5:34 PM
**Backup:** `/tmp/crontab_backup.txt`
**Status:** ✅ COMPLETE
