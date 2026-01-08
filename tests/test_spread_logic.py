"""
Test Spread Coverage Logic

Verifies spread coverage calculations with concrete examples.

This will help us understand if the backtest logic is correct.
"""

import sys
sys.path.insert(0, '.')

from loguru import logger

logger.info("=" * 80)
logger.info("SPREAD COVERAGE LOGIC TESTS")
logger.info("=" * 80)

def test_spread_coverage(scenario, home_score, away_score, spread_home, side, expected_outcome):
    """Test a single spread coverage scenario."""
    point_diff = home_score - away_score
    spread_result = point_diff + spread_home

    if side == 'home':
        won = spread_result > 0
    elif side == 'away':
        won = spread_result < 0
    else:
        won = None

    status = "✅ PASS" if won == expected_outcome else "❌ FAIL"

    logger.info(f"\n{scenario}")
    logger.info(f"  Final Score: {away_score}-{home_score} (point_diff = {point_diff:+.1f})")
    logger.info(f"  Spread: {spread_home:+.1f} (home perspective)")
    logger.info(f"  Bet Side: {side}")
    logger.info(f"  Spread Result: {point_diff} + {spread_home} = {spread_result:+.1f}")
    logger.info(f"  Coverage: {side} covers if spread_result {'>' if side == 'home' else '<'} 0")
    logger.info(f"  Result: {spread_result:+.1f} {'>' if side == 'home' else '<'} 0 = {won}")
    logger.info(f"  Expected: {expected_outcome}")
    logger.info(f"  {status}")

    return won == expected_outcome

# Test cases
logger.info("\n" + "=" * 80)
logger.info("TEST CASES")
logger.info("=" * 80)

all_passed = True

# Test 1: Home favored, home covers
all_passed &= test_spread_coverage(
    scenario="Test 1: Home favored by 5.5, wins by 7",
    home_score=110,
    away_score=103,
    spread_home=-5.5,
    side='home',
    expected_outcome=True  # Home should cover
)

# Test 2: Home favored, home doesn't cover
all_passed &= test_spread_coverage(
    scenario="Test 2: Home favored by 5.5, wins by 3",
    home_score=105,
    away_score=102,
    spread_home=-5.5,
    side='home',
    expected_outcome=False  # Home should NOT cover
)

# Test 3: Home favored, away covers
all_passed &= test_spread_coverage(
    scenario="Test 3: Home favored by 5.5, wins by 3 (AWAY BET)",
    home_score=105,
    away_score=102,
    spread_home=-5.5,
    side='away',
    expected_outcome=True  # Away should cover
)

# Test 4: Away favored, away covers
all_passed &= test_spread_coverage(
    scenario="Test 4: Away favored by 3.5, wins by 5",
    home_score=100,
    away_score=105,
    spread_home=+3.5,
    side='away',
    expected_outcome=True  # Away should cover
)

# Test 5: Away favored, away doesn't cover
all_passed &= test_spread_coverage(
    scenario="Test 5: Away favored by 3.5, wins by 2",
    home_score=100,
    away_score=102,
    spread_home=+3.5,
    side='away',
    expected_outcome=False  # Away should NOT cover
)

# Test 6: Away favored, home covers
all_passed &= test_spread_coverage(
    scenario="Test 6: Away favored by 3.5, wins by 2 (HOME BET)",
    home_score=100,
    away_score=102,
    spread_home=+3.5,
    side='home',
    expected_outcome=True  # Home should cover
)

# Test 7: Pick'em, home wins
all_passed &= test_spread_coverage(
    scenario="Test 7: Pick'em (0 spread), home wins by 1",
    home_score=101,
    away_score=100,
    spread_home=0.0,
    side='home',
    expected_outcome=True  # Home should cover
)

# Test 8: Pick'em, away wins
all_passed &= test_spread_coverage(
    scenario="Test 8: Pick'em (0 spread), away wins by 1",
    home_score=100,
    away_score=101,
    spread_home=0.0,
    side='away',
    expected_outcome=True  # Away should cover
)

logger.info("\n" + "=" * 80)
if all_passed:
    logger.info("✅ ALL TESTS PASSED - Spread logic is correct")
else:
    logger.info("❌ SOME TESTS FAILED - Spread logic has errors!")
logger.info("=" * 80)
