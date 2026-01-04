# 40% Win Rate Investigation - Findings

**Date**: January 4, 2026
**Status**: 🔴 **CRITICAL ISSUE IDENTIFIED**

## Executive Summary

The backtest shows a 40% overall win rate, which is **significantly worse than random** (50%) and far below the 52.4% needed to break even at -110 odds. However, the investigation revealed this is not evenly distributed:

### Key Finding

The model's poor performance is **entirely driven by away bets**:

| Bet Type | Win Rate | ROI | Bet Count | Status |
|----------|----------|-----|-----------|---------|
| **HOME** | 56.2% | +8.5% | 205 | ✅ **PROFITABLE** |
| **AWAY** | 35.7% | -34.7% | 825 | ❌ **CATASTROPHIC** |
| **Overall** | 39.8% | -31.2% | 1,030 | ❌ **LOSING** |

## The Problem

**Away bets have a 35.7% win rate** - worse than just picking randomly and the inverse of home performance!

This means:
- **For every 100 away bets, we lose 64** (only win 36)
- The away bet strategy is **fundamentally broken**
- Home bets work well (+8.5% ROI), but we're betting away 4x more often (825 vs 205)

## What We've Ruled Out

### ✅ Spread Coverage Logic is Correct

We created comprehensive unit tests (`scripts/test_spread_logic.py`) that verify:
- Home spread coverage: `spread_result > 0` ✅
- Away spread coverage: `spread_result < 0` ✅
- All edge cases pass

**Conclusion**: The spread math is correct. The issue is elsewhere.

### ✅ Bug Fixes Were Necessary

The original spread bug was real and needed fixing:
- Old (buggy): 58.8% win rate, +1,538% ROI (WRONG)
- New (correct): 39.8% win rate, -31.2% ROI (TRUE but BAD)

## Hypotheses for Away Bet Failure

### Hypothesis 1: Model Bias Against Away Teams

**Evidence**:
- Home bets: 56.2% win rate (GOOD)
- Away bets: 35.7% win rate (TERRIBLE)
- Model may systematically overestimate away team chances

**Possible Causes**:
1. **Training data imbalance** - Not enough away bet examples
2. **Feature engineering error** - Away team features calculated incorrectly
3. **Home court advantage not properly modeled** - Away teams face additional challenges not captured

**Test**: Check model's raw predictions for home vs away probabilities

### Hypothesis 2: Strategy Filtering is Inverted

**Evidence**:
- Strategy requires 7% edge for home bets vs 5% for away
- But home bets are profitable and away bets aren't
- This suggests the edge calculation might be inverted

**Possible Causes**:
1. Edge is calculated as `model_prob - market_prob`, but maybe for away bets it should be flipped
2. The `prefer_away` setting (line 45 in optimized_strategy.py) might be causing bad away bet selection

**Test**: Disable `prefer_away` and run backtest again

### Hypothesis 3: Market is More Efficient on Away Bets

**Evidence**:
- 825 away bets vs 205 home bets (4:1 ratio)
- Strategy is finding more "edge" on away bets
- But away bets are losing terribly

**Possible Explanation**:
- Market prices away bets more efficiently
- Model's "edge" on away bets is illusory
- Home bets have real market inefficiencies we can exploit

**Test**: Compare model edge distribution for home vs away bets

### Hypothesis 4: Home Bias Penalty is Applied Incorrectly

**Evidence**:
- Home bias penalty (-2%) is subtracted from home bet edge
- But home bets still outperform
- Maybe penalty should be added to away bets instead?

**Current Logic** (optimized_strategy.py:112-114):
```python
if side.lower() == 'home':
    edge -= self.config.home_bias_penalty  # Subtract 2%
```

**Possible Issue**:
- We're penalizing the GOOD bets (home)
- Not penalizing the BAD bets (away)

**Test**: Remove home bias penalty entirely and run backtest

## Diagnostic Steps Needed

### 1. Check Model Predictions Distribution

```python
# For each test game, check:
- model_prob for home win
- model_prob for away win (1 - model_prob)
- actual outcome
# Look for systematic over/under prediction
```

### 2. Analyze Edge Calculation

```python
# For home and away bets separately:
- Average model_prob
- Average market_prob
- Average calculated edge
- Correlation between edge size and actual win rate
```

### 3. Test Strategy Variations

Run backtest with:
1. **No prefer_away**: Set `prefer_away = False`
2. **No home bias penalty**: Set `home_bias_penalty = 0.0`
3. **Higher away edge threshold**: Set `min_edge_away = 0.10` (10%)
4. **Only home bets**: Set `min_edge = 0.05, min_edge_away = 0.99` (effectively disable away)

### 4. Check Feature Calculations

```python
# Verify for sample games:
- Are home/away features swapped?
- Are rest days calculated correctly for away teams?
- Is travel distance for away teams correct?
- Do lineup features work for road games?
```

## Immediate Recommendations

### 🔴 DO NOT use this system for real money trading

The current system is:
- **Losing money** (-31.2% ROI)
- **Worse than random** (39.8% win rate)
- **Failing catastrophically on away bets** (35.7%)

### 🟡 Short-term Fix: Disable Away Bets

Until we understand the problem:

```python
# In optimized_strategy.py, line 136:
if side.lower() == 'away':
    return False, "Away bets disabled due to poor performance"
```

**Expected Results** (if only betting home):
- Win rate: ~56% (from backtest)
- ROI: ~+8.5%
- Fewer bets: ~200 instead of 1,030
- **At least we wouldn't be losing money**

### 🟢 Long-term Fix: Find Root Cause

Priority order:
1. **Check if features are swapped** - Most likely culprit
2. **Analyze model predictions** - Is model systematically wrong?
3. **Review strategy filters** - Are we selecting bad bets?
4. **Consider retraining** - Maybe model learned wrong patterns

## Technical Details

### Backtest Configuration

```python
# Train: Dec 2020 - Jun 2022 (2,251 games)
# Test:  Oct 2022 - Jan 2026 (4,195 games)

Strategy:
- min_edge = 5%
- min_edge_home = 7%  # Stricter for home
- kelly_fraction = 10%
- min_disagreement = 20%
- home_bias_penalty = 2%
- prefer_away = True  # <-- SUSPECT!
```

### Test Results Summary

**Static Bankroll ($1,000)**:
- Total bets: 1,030
- Win rate: 39.8% (406W-615L-9P)
- Final bankroll: -$7,082.05
- ROI: -808.2% (bankroll), -31.2% (wagered)
- Peak: $1,086.48
- Max drawdown: 751.8%

**By Side**:
- HOME: 114W-89L (56.2%), +8.5% ROI, 205 bets
- AWAY: 292W-526L (35.7%), -34.7% ROI, 825 bets

**Dynamic Bankroll** (similar results, smaller sample):
- Total bets: 30
- Win rate: 40.0% (12W-18L)
- HOME: 3W-1L (75%), +41.5% ROI
- AWAY: 9W-17L (34.6%), -33.5% ROI

## Files Created

- **test_spread_logic.py** - Verifies spread coverage math (✅ all tests pass)
- **investigate_model_performance.py** - Analyzes predictions vs outcomes (incomplete)
- **resettle_historical_bets.py** - Re-settles paper trading bets (not needed, test data)

## Next Actions

1. ✅ Document findings (this file)
2. ⏳ Test hypothesis: Disable `prefer_away` setting
3. ⏳ Test hypothesis: Remove home bias penalty
4. ⏳ Analyze model prediction distribution
5. ⏳ Check for feature swap bug
6. ⏳ Create simple "home bets only" baseline

## Conclusion

**The spread bug fix revealed the truth**: This system is losing money, primarily due to catastrophic away bet performance (35.7% win rate).

**The good news**: Home bets work (56.2% win rate, +8.5% ROI). We can potentially salvage the system by:
1. Disabling away bets entirely (short-term)
2. Finding and fixing the away bet issue (long-term)

**The bad news**: We can't use this for real money until we understand why away bets fail so badly.

---

**Status**: 🔴 Investigation ongoing - Do not trade with real money

Last updated: January 4, 2026
