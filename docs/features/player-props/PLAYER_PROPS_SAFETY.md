# Player Props Safety System - Complete Guide

**Lineup and Injury Filtering**

---

## Overview

The PlayerPropsStrategy now includes **comprehensive safety filtering** to avoid betting on:
- ❌ Injured players (Out, Questionable)
- ❌ Non-starters with reduced minutes
- ❌ Players not confirmed in lineup
- ❌ Load management / rest days

This prevents betting on players who won't perform as predicted.

---

## How It Works

### 1. Lineup Verification

**Before each bet**, the system checks:

```python
# Check if player is in confirmed lineup
lineups = ESPNLineupClient().get_lineup_for_game(game_id)

if player not in lineups:
    SKIP BET → "Not in confirmed lineup"

if require_starter and not player.is_starter:
    SKIP BET → "Not a confirmed starter"
```

**Data Source**: ESPN API
**Updated**: Real-time (fetched before each betting window)

### 2. Injury Report Check

**Before each bet**, the system checks:

```python
# Check injury status
injuries = ESPNClient().get_injuries()

if player.status == "Out":
    SKIP BET → "Injury status: Out"

if skip_questionable and player.status in ["Questionable", "Doubtful"]:
    SKIP BET → "Injury status: Questionable"

if player.status == "Probable":
    PROCEED (with caution log)
```

**Data Source**: ESPN Injury Report
**Updated**: Real-time (fetched before each betting window)

### 3. Safety Flow

```
Player Prop Available (e.g., Giannis PTS O28.5)
    ↓
Is player in confirmed lineup? → NO → SKIP
    ↓ YES
Is player a starter? → NO (if require_starter=true) → SKIP
    ↓ YES
Is player injured (Out)? → YES → SKIP
    ↓ NO
Is player Questionable? → YES (if skip_questionable=true) → SKIP
    ↓ NO
✅ SAFE TO BET → Make prediction
```

---

## Configuration

### Config File: `config/multi_strategy_config.yaml`

```yaml
strategies:
  props:
    enabled: true
    min_edge: 0.05

    # Safety filters
    require_starter: true        # Only bet on confirmed starters
    skip_questionable: true      # Skip questionable/doubtful players

    prop_types:
      - PTS
      - REB
      - AST
      - 3PM
```

### Safety Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `require_starter` | `true` | Only bet on confirmed starters (recommended) |
| `skip_questionable` | `true` | Skip Q/D players (recommended) |

**Conservative (Safest)**:
```yaml
require_starter: true
skip_questionable: true
```

**Aggressive (More bets, higher risk)**:
```yaml
require_starter: false   # Allow bench players
skip_questionable: false  # Allow questionable players
```

---

## What Gets Filtered

### Always Skipped
- ❌ Players with status: **Out**
- ❌ Players with status: **Inactive**
- ❌ Players **not in lineup** at all

### Skipped if `skip_questionable: true`
- ⚠️ **Questionable** (game-time decision)
- ⚠️ **Doubtful** (unlikely to play)

### Skipped if `require_starter: true`
- 🪑 **Bench players** (not in starting 5)
- 🪑 **Limited minutes** players

### Allowed (with logging)
- ✅ **Probable** players (likely to play)
- ✅ **Day-to-Day** (minor issue, expected to play)

---

## Pipeline Integration

### Automatic Lineup/Injury Fetch

When props are enabled, the pipeline automatically:

```bash
# Step 3: Configuring strategies...
  Fetching lineups and injury reports for player props safety...
  ✓ Lineups fetched
  ✓ Injury report fetched (42 players)
✓ Enabled PlayerPropsStrategy with lineup/injury filtering
```

**Timing**: Lineups fetched ~1-2 hours before games
**Frequency**: Every pipeline run (daily at 4 PM ET)

---

## Example Scenarios

### Scenario 1: Healthy Starter
```
Player: Giannis Antetokounmpo
Lineup Status: Confirmed Starter
Injury Status: None
Result: ✅ PROCEED with bet evaluation
```

### Scenario 2: Questionable Player
```
Player: Damian Lillard
Lineup Status: Confirmed Starter
Injury Status: Questionable (ankle)
Result: ❌ SKIP - "Injury status: Questionable"
Log: "Skipping Damian Lillard PTS: Injury status: questionable"
```

### Scenario 3: Load Management
```
Player: Kawhi Leonard
Lineup Status: Not in lineup
Injury Status: None (rest)
Result: ❌ SKIP - "Not in confirmed lineup"
Log: "Skipping Kawhi Leonard PTS: Not in confirmed lineup"
```

### Scenario 4: Bench Player
```
Player: Jordan Poole
Lineup Status: Bench (not starter)
Injury Status: None
Result: ❌ SKIP - "Not a confirmed starter" (if require_starter=true)
```

### Scenario 5: Probable Starter
```
Player: Stephen Curry
Lineup Status: Confirmed Starter
Injury Status: Probable (sore knee)
Result: ✅ PROCEED (with warning log)
Log: "Stephen Curry is probable but proceeding"
```

---

## Monitoring

### Check What's Being Filtered

```bash
# View pipeline logs
tail -100 logs/cron_betting.log | grep "Skipping"

# Example output:
# Skipping Giannis Antetokounmpo PTS: Injury status: Questionable
# Skipping Kawhi Leonard REB: Not in confirmed lineup
# Skipping Jordan Poole AST: Not a confirmed starter
```

### Dashboard

```bash
streamlit run dashboard/analytics_dashboard.py
# Go to "Performance" tab
# Filter by strategy: player_props
# Check win rate and ROI
```

### Discord Notifications

Props bets will show in Discord with safety status:
```
📊 Today's Props Bets (2 placed, 3 skipped):

✅ PLACED:
- Giannis Antetokounmpo PTS O28.5 +110 (5.2% edge, $15)
- Stephen Curry 3PM O4.5 -105 (6.8% edge, $18)

❌ SKIPPED:
- Damian Lillard PTS O25.5 (Questionable - ankle)
- Kawhi Leonard REB O8.5 (Not in lineup - rest)
- Jordan Poole AST O5.5 (Not a starter)
```

---

## Lineup Data Sources

### ESPN API

**Lineups endpoint**: `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard`

**Data includes**:
- Confirmed starters (top 5)
- Bench players
- Inactive players
- Player status (active/inactive)

**Timing**: Usually confirmed 1-2 hours before tip-off

### ESPN Injury Report

**Injuries endpoint**: `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries`

**Data includes**:
- Player name, team
- Injury status (Out, Questionable, Probable, Day-to-Day)
- Injury type (ankle, knee, illness, etc.)
- Return timeline

**Updated**: Real-time, usually multiple times per day

---

## Database Storage

Lineups are cached in SQLite database:

```sql
-- Table: confirmed_lineups
CREATE TABLE confirmed_lineups (
    game_id TEXT,
    player_name TEXT,
    team_abbrev TEXT,
    is_starter BOOLEAN,
    status TEXT,
    position TEXT,
    confirmed_at TEXT,
    collected_at TEXT
);
```

**Location**: `data/bets/bets.db`

**Persistence**: Cached for historical analysis

---

## Troubleshooting

### No lineups available

**Symptom**: Logs show "No lineup data for game X"

**Causes**:
1. Games haven't started lineup announcements yet (>2 hours before tip-off)
2. ESPN API is down
3. Network issues

**Solution**:
- System proceeds with caution (allows bets if no data)
- Check lineup data manually: `python scripts/collect_lineups.py`
- Verify ESPN API: `curl "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"`

### No injury data

**Symptom**: Logs show "Could not fetch injuries"

**Causes**:
1. ESPN API is down
2. Network issues
3. API rate limiting

**Solution**:
- System continues (better to have false positives than miss bets)
- Manually check: `python -c "from src.data.espn_injuries import ESPNClient; print(ESPNClient().get_injuries())"`

### Player not found in lineup

**Symptom**: All props skipped with "Not in confirmed lineup"

**Causes**:
1. Lineup not announced yet
2. Player name mismatch (e.g., "Giannis Antetokounmpo" vs "G. Antetokounmpo")
3. Player recently traded

**Solution**:
- Wait until lineups are confirmed
- Check name matching in database
- Temporarily disable `require_starter` for testing

---

## Testing Safety Filters

### Manual Test

```bash
# Test with current lineups/injuries
python -c "
from src.betting.strategies.player_props_strategy import PlayerPropsStrategy
from src.data.lineup_scrapers import ESPNLineupClient
from src.data.espn_injuries import ESPNClient

strategy = PlayerPropsStrategy()

# Test a player
is_safe, reason = strategy.is_player_safe_to_bet(
    player_name='Giannis Antetokounmpo',
    player_team='MIL',
    game_id='TEST_GAME',
    game_date='2026-01-05'
)

print(f'Safe to bet: {is_safe}')
if not is_safe:
    print(f'Reason: {reason}')
"
```

### Dry Run Test

```bash
# Test pipeline with today's lineups
python scripts/daily_multi_strategy_pipeline.py --dry-run

# Check logs for safety filters
tail -50 logs/cron_betting.log | grep -E "(Skipping|safe_to_bet)"
```

---

## Best Practices

### Recommended Settings

**For Conservative Betting (Safest)**:
```yaml
require_starter: true
skip_questionable: true
min_edge: 0.06  # Higher edge requirement
```

**For Moderate Betting**:
```yaml
require_starter: true
skip_questionable: true
min_edge: 0.05
```

**For Aggressive Betting** (NOT recommended):
```yaml
require_starter: false
skip_questionable: false
min_edge: 0.04
```

### Timing Recommendations

**Best time to bet**:
- ✅ **1-2 hours before game**: Lineups confirmed
- ✅ **After morning shootaround**: Injury updates complete

**Avoid betting**:
- ❌ **>3 hours before game**: Lineups not confirmed
- ❌ **Right at tip-off**: Last-minute scratches

### Monitor Win Rate by Safety Level

Track performance with different safety settings:

```bash
# Check win rate by lineup status
python -c "
from src.bet_tracker import get_bet_history
import pandas as pd

df = get_bet_history()
props = df[df['strategy_type'] == 'player_props']

# Group by whether lineup was confirmed
print(props.groupby('lineup_confirmed')['outcome'].value_counts())
"
```

---

## Summary

✅ **Player props now protected by 3 safety layers**:
1. Lineup verification (confirmed to play)
2. Injury status check (no Out/Questionable players)
3. Starter requirement (consistent minutes)

✅ **Automatic**: Lineups/injuries fetched every pipeline run

✅ **Configurable**: Adjust safety settings in config

✅ **Logged**: All skipped bets shown in logs

**Result**: Only bet on healthy, confirmed starters with predictable performance.

**Expected Impact**:
- Reduced bad beats from injured/resting players
- More consistent ROI (10-15% target)
- Higher win rate on props placed

Deploy with confidence! 🚀
