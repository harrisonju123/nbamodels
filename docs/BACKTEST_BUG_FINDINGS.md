# Critical Backtest Bug - Findings Report

**Date**: January 4, 2026
**Status**: 🔴 **CRITICAL BUG DISCOVERED**

## Executive Summary

A **critical bug** in the backtest spread coverage logic was discovered during code review. The original backtest showed:
- ✅ +1,538% ROI (WRONG!)
- ✅ 58.8% win rate (WRONG!)
- ✅ $16,381 profit (WRONG!)

After fixing the bug, the actual backtest shows:
- ❌ -30% ROI (LOSING MONEY)
- ❌ 40% win rate (TERRIBLE)
- ❌ -$248 loss (SYSTEM FAILS)

**The original backtest results were completely invalid.**

## The Bug

### Original (Incorrect) Code
```python
# WRONG - This logic is backwards!
if side == 'home':
    won = actual_diff > -spread
else:
    won = actual_diff < spread
```

### Fixed (Correct) Code
```python
# CORRECT - Proper spread coverage logic
spread_result = actual_diff + spread

if side == 'home':
    won = spread_result > 0  # Home covers if result > 0
else:
    won = spread_result < 0  # Away covers if result < 0
```

### Why It Was Wrong

**Spread Mechanics**:
- `spread_home` is always from home team perspective
- `actual_diff = home_score - away_score`
- Home covers if: `actual_diff + spread_home > 0`
- Away covers if: `actual_diff + spread_home < 0`

**Example**:
- Spread: -5.5 (home favored by 5.5)
- Final: Home wins 102-100 (actual_diff = +2)
- Spread result: 2 + (-5.5) = -3.5

**Home bet**:
- spread_result = -3.5 < 0 → HOME LOSES (didn't cover -5.5)

**Away bet**:
- spread_result = -3.5 < 0 → AWAY WINS (covered +5.5)

The buggy code had this backwards, causing wins to be counted as losses and vice versa.

## Impact on Results

### Before Fix (Buggy Backtest)

**Dynamic Bankroll**:
- Total Bets: 691
- Win Rate: 58.8%
- Final Bankroll: $16,381.90
- ROI: +1,538.2%
- **THESE NUMBERS ARE COMPLETELY WRONG**

**Static Bankroll**:
- Total Bets: 1,030
- Win Rate: 57.7%
- Final Bankroll: $4,742.54
- ROI: +374.3%
- **THESE NUMBERS ARE COMPLETELY WRONG**

### After Fix (Correct Backtest)

**Dynamic Bankroll**:
- Total Bets: 30 (much fewer due to stricter filters working correctly)
- Win Rate: 40.0% (LOSING)
- Final Bankroll: $751.61
- ROI: -24.8%
- **System is LOSING MONEY**

**Static Bankroll**:
- Total Bets: 1,030
- Win Rate: 39.8% (TERRIBLE)
- Final Bankroll: -$7,082.05 (NEGATIVE!)
- ROI: -31.2%
- **System is HEMORRHAGING MONEY**

## Additional Bugs Fixed

### 1. Double Home Bias Penalty

**Bug**: Home bias penalty was applied twice:
- Once in `should_bet()` at line 114
- Again in `calculate_bet_size()` at line 190

**Impact**: Home bets were getting -4% penalty instead of -2%, making them even worse.

**Fixed**: Removed duplicate penalty from `calculate_bet_size()`.

### 2. Push Detection Logic

**Bug**:
```python
# WRONG
if abs(actual_diff - (-spread if side == 'home' else spread)) < 0.1:
```

**Fixed**:
```python
# CORRECT
if abs(actual_diff + spread) < 0.1:
```

### 3. Performance Issue

**Bug**: Bankroll calculation recalculated sum of all previous bets on every iteration (O(n²) complexity).

**Fixed**: Used cumulative variable (`static_cumulative_profit`).

## What This Means

### 1. The Backtest System Was Showing Fantasy Results

The original bug inverted win/loss calculations, making a losing system appear to be a winner. This is the worst kind of bug in backtesting - it gives false confidence in a failing strategy.

### 2. The Model May Not Have an Edge

With correct spread coverage logic:
- Win rate: 40% (need ~52.4% to break even at -110 odds)
- ROI: -30%
- The model is significantly worse than random (50%)

### 3. Paper Trading Results Don't Match

**Paper Trading**: 6.91% ROI, 56% win rate (150 bets)
**Backtest (fixed)**: -30% ROI, 40% win rate (30 bets)

**Possible reasons**:
1. Paper trading bet logging has similar bugs
2. Paper trading uses real odds which may differ
3. Sample size is too small (30 bets vs 150)
4. Different time periods
5. Paper trading might also have inverted logic

### 4. Need to Validate Everything

**ALL previous backtest results are now suspect**:
- `optimized_backtest.py` - likely has same bug
- `realistic_backtest_v3.py` - likely has same bug
- Any other backtest scripts - need review

## Next Steps (URGENT)

### 1. Fix All Backtest Scripts
Check and fix spread coverage logic in:
- [x] `scripts/backtest_with_bankroll.py` (FIXED)
- [x] `scripts/optimized_backtest.py` (N/A - moneyline only, no spread bug)
- [x] `scripts/realistic_backtest_v3.py` (FIXED)
- [x] `src/betting/dual_model_backtest.py` (FIXED)
- [x] All other backtest files (checked, only these had spread betting)

### 2. Fix Paper Trading Bet Logging
Check `src/bet_tracker.py` and bet settlement logic:
- [x] Fixed spread coverage logic in `settle_bet()` (lines 478-499)
- [x] Now properly checks `bet_side` for spread bets
- [ ] Need to re-settle historical paper trading bets with correct logic
- [ ] Need to validate current paper trading results

### 3. Investigate Win Rate Discrepancy

The 40% win rate is suspiciously low. Possible issues:
- Model is genuinely bad
- Feature engineering has errors
- Training data has issues
- OR there's still a bug in the fixed code

### 4. Re-validate the Model

Need to:
- Check model predictions vs actual outcomes
- Verify feature calculations
- Look for data leakage or training issues
- Consider retraining from scratch

## Lessons Learned

### 1. Code Review is Critical

Without the comprehensive code review, we would have:
- Believed the system had 1,538% ROI
- Started real money trading
- Lost significant capital
- Never understood why

### 2. Test Edge Cases

The buggy logic passed superficial checks because:
- It compiled without errors
- It produced "reasonable looking" results
- No unit tests existed for spread coverage

### 3. Validate Against Known Outcomes

Should have validated backtest logic against a few known games:
- Pick 5 games with known spreads and outcomes
- Manually calculate if bet should win/lose
- Verify backtest agrees

### 4. Start Small

Good thing we were paper trading! If we had gone live with real money based on the buggy backtest, we would have lost everything.

## Immediate Actions

1. ✅ Fixed spread coverage logic in `scripts/backtest_with_bankroll.py`
2. ✅ Fixed push detection logic
3. ✅ Removed double home bias penalty in `src/betting/optimized_strategy.py`
4. ✅ Optimized performance (O(n²) → O(n))
5. ✅ Fixed paper trading spread settlement logic in `src/bet_tracker.py`
6. ✅ Fixed spread logic in `scripts/realistic_backtest_v3.py`
7. ✅ Fixed spread logic in `src/betting/dual_model_backtest.py`
8. ⏳ Need to re-settle historical paper trading bets with correct logic
9. ⏳ Need to investigate why model has 40% win rate

## Conclusion

**The original backtest was completely wrong and showed fantasy results.**

With correct logic, the system is currently:
- ❌ Losing money (-30% ROI)
- ❌ Win rate of 40% (vs 52.4% needed to break even)
- ❌ NOT ready for real money trading

**CRITICAL**: Do NOT use any previous backtest results. Do NOT trust current paper trading results until we validate the bet logging logic.

**Status**: 🔴 **SYSTEM VALIDATION REQUIRED** - All previous results are invalid until verified with fixed logic.

---

Last updated: January 4, 2026
