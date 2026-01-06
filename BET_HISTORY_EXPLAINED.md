# Bet History - What the Numbers Mean

## 📊 How Your Bets Are Logged

When you run your daily betting pipeline, it calls `log_bets_from_predictions()` which saves each bet to the database with these values:

### What Gets Saved (Line 245-254 in bet_tracker.py):

```python
INSERT INTO bets (
    odds,              # The actual American odds you got (e.g., -110, +150)
    line,              # The spread or total line (e.g., -7.0, 220.5)
    model_prob,        # Your model's predicted probability (0 to 1)
    market_prob,       # IMPLIED probability from the odds
    edge,              # model_prob - market_prob (the edge)
    ...
)
```

## 🔍 Current Data Issues

Looking at your database, there are some problems:

### 1. **Line Values Are Corrupted**
```sql
SELECT line FROM bets LIMIT 5;
-1.02082472036826  ❌ Should be like -7.0, +3.5
2.76349539768235   ❌ Should be like +5.5, -2.0
4.93106263727894   ❌ Should be like -4.5, +6.0
```

**These decimals look like they're storing edge or probability instead of the actual betting line!**

### 2. **Market Probability is Always 50%**
```sql
SELECT market_prob FROM bets LIMIT 5;
0.5  ❌ This is wrong
0.5  ❌ Should be calculated from odds
0.5  ❌ Should vary (0.45-0.55 for -110 odds)
```

**This means `market_prob` was hardcoded to 0.5 somewhere, not calculated from actual odds**

### 3. **All Odds are -110**
```sql
SELECT odds FROM bets WHERE odds IS NOT NULL LIMIT 5;
-110.0  ⚠️ Suspicious
-110.0  ⚠️ Every bet at -110?
-110.0  ⚠️ Not realistic
```

**If all your bets were really at -110, this would be very unusual**

## 📋 What the Dashboard Shows

### Current Display (with issues):

| Column | Source | Issue |
|--------|--------|-------|
| **Odds** | `bets.odds` | All showing -110 (suspicious) |
| **Model %** | `bets.model_prob * 100` | ✅ Correct (55-58%) |
| **Market %** | `bets.market_prob * 100` | ❌ Always 50% (wrong) |
| **Edge %** | `bets.edge * 100` | ❌ Shows 500-700% (impossible) |
| **Profit** | `bets.profit` | ✅ Correct ($90.91 or -$100) |

## 🎯 What They SHOULD Show

### Correct Calculation:

**Example: You bet Lakers -7.0 at -110 odds**

1. **Model Prob**: Your model says Lakers cover with 55% probability
2. **Market Prob**: Calculated from -110 odds = `110 / (110 + 100)` = **52.4%**
3. **Edge**: 55% - 52.4% = **2.6%** (not 500%!)
4. **If you win**: Profit = $100 / (110/100) = **$90.91**
5. **If you lose**: Profit = **-$100**

## 🔧 Where the Problem Comes From

Looking at line 209 in `bet_tracker.py`:

```python
def log_bet(
    ...
    market_prob: float,  # THIS IS PASSED IN, not calculated!
    ...
):
```

The `market_prob` is being passed in from somewhere else, not calculated from the odds. Let me check where bets are being logged:

### From `log_bets_from_predictions()` (lines 358, 375, etc.):

```python
log_bet(
    odds=row["best_home_odds"],    # Actual odds (e.g., -110)
    model_prob=row["model_home_prob"],  # Model's prediction (e.g., 0.55)
    market_prob=row["market_home_prob"],  # ❌ This is the issue!
    edge=row["home_edge"],
    ...
)
```

**The problem is in your prediction generation code** - whatever creates those `row["market_home_prob"]` values is setting them all to 0.5 instead of calculating them from the actual odds!

## ✅ What Should Happen

The correct flow should be:

1. **Get odds from sportsbook**: e.g., -110
2. **Calculate market probability**: `american_to_implied_prob(-110)` = 52.4%
3. **Get model probability**: Your model predicts 55%
4. **Calculate edge**: 55% - 52.4% = 2.6%
5. **Save all these** to the database

## 🛠️ How to Fix

You need to find where your predictions are generated and ensure `market_home_prob`, `market_away_prob`, etc. are calculated like this:

```python
# For negative odds (favorite)
if odds < 0:
    market_prob = abs(odds) / (abs(odds) + 100)
# For positive odds (underdog)
else:
    market_prob = 100 / (odds + 100)
```

**The function already exists**: `american_to_implied_prob()` at line 2001 of bet_tracker.py!

## 📊 Summary

Your dashboard is showing:
- ✅ **Wins/Losses**: Correct (47W-32L)
- ✅ **Profit**: Correct (based on actual outcomes)
- ✅ **Model %**: Correct (your model's predictions)
- ❌ **Market %**: Wrong (always 50%)
- ❌ **Edge %**: Wrong (calculated from wrong market %)
- ❌ **Lines**: Corrupted (storing wrong values)

**The win/loss tracking and profit calculations ARE correct** - those are calculated from final scores. The issue is just with how the odds/probabilities/lines were initially logged.

Want me to find where your predictions are generated and fix the market probability calculation?
