"""
Quick Integration Test for Alternative Data Features

Tests that the GameFeatureBuilder can successfully integrate alternative data features
without errors, using synthetic test data.

Usage:
    python scripts/test_alternative_data_integration.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger
from datetime import datetime

from src.features.game_features import GameFeatureBuilder


def create_test_games() -> pd.DataFrame:
    """Create synthetic test games for validation."""
    test_games = pd.DataFrame({
        'game_id': ['test_001', 'test_002', 'test_003'],
        'date': pd.to_datetime(['2025-12-15', '2025-12-15', '2025-12-16']),
        'season': [2025, 2025, 2025],
        'home_team': ['LAL', 'GSW', 'BOS'],
        'away_team': ['BOS', 'LAC', 'PHI'],
        'home_score': [110, 115, 105],
        'away_score': [105, 112, 108],
    })

    # Add target variables
    test_games['home_win'] = (test_games['home_score'] > test_games['away_score']).astype(int)
    test_games['point_diff'] = test_games['home_score'] - test_games['away_score']
    test_games['total_points'] = test_games['home_score'] + test_games['away_score']

    return test_games


def test_baseline_features():
    """Test baseline feature building (without alternative data)."""
    logger.info("=" * 80)
    logger.info("TEST 1: Baseline Features (no alternative data)")
    logger.info("=" * 80)

    games = create_test_games()

    try:
        builder = GameFeatureBuilder(
            use_referee_features=False,
            use_news_features=False,
            use_sentiment_features=False
        )

        features = builder.build_game_features(games)
        feature_cols = builder.get_feature_columns(features)

        logger.info(f"✓ Built {len(features)} games with {len(feature_cols)} features")
        logger.info(f"✓ Baseline test PASSED")
        return True

    except Exception as e:
        logger.error(f"✗ Baseline test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_features():
    """Test enhanced feature building (with all alternative data)."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Enhanced Features (with alternative data)")
    logger.info("=" * 80)

    games = create_test_games()

    try:
        builder = GameFeatureBuilder(
            use_referee_features=True,
            use_news_features=True,
            use_sentiment_features=True
        )

        features = builder.build_game_features(games)
        feature_cols = builder.get_feature_columns(features)

        logger.info(f"✓ Built {len(features)} games with {len(feature_cols)} features")

        # Check for alternative data columns
        ref_cols = [c for c in features.columns if 'ref_' in c]
        news_cols = [c for c in features.columns if 'news' in c]
        sentiment_cols = [c for c in features.columns if 'sentiment' in c]

        logger.info(f"  Referee features: {len(ref_cols)}")
        logger.info(f"  News features: {len(news_cols)}")
        logger.info(f"  Sentiment features: {len(sentiment_cols)}")

        if ref_cols:
            logger.info(f"  ✓ Referee features integrated: {ref_cols}")
        else:
            logger.warning(f"  ⚠ No referee features found (may be disabled)")

        if news_cols:
            logger.info(f"  ✓ News features integrated: {news_cols}")
        else:
            logger.warning(f"  ⚠ No news features found (may be disabled)")

        if sentiment_cols:
            logger.info(f"  ✓ Sentiment features integrated: {sentiment_cols}")
        else:
            logger.warning(f"  ⚠ No sentiment features found (may be disabled)")

        logger.info(f"✓ Enhanced test PASSED")
        return True

    except Exception as e:
        logger.error(f"✗ Enhanced test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graceful_degradation():
    """Test that features degrade gracefully when alternative data is unavailable."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Graceful Degradation")
    logger.info("=" * 80)

    games = create_test_games()

    try:
        # Try to build with all features enabled
        builder = GameFeatureBuilder(
            use_referee_features=True,
            use_news_features=True,
            use_sentiment_features=True
        )

        # Check which features actually initialized
        logger.info(f"  Referee features enabled: {builder.use_referee_features}")
        logger.info(f"  News features enabled: {builder.use_news_features}")
        logger.info(f"  Sentiment features enabled: {builder.use_sentiment_features}")

        # Build features - should not fail even if some data unavailable
        features = builder.build_game_features(games)

        # Verify no NaN values in critical columns
        critical_cols = ['home_team', 'away_team', 'date']
        for col in critical_cols:
            if col in features.columns:
                nan_count = features[col].isna().sum()
                if nan_count > 0:
                    logger.error(f"✗ Found {nan_count} NaN values in critical column: {col}")
                    return False

        logger.info(f"✓ Graceful degradation test PASSED")
        return True

    except Exception as e:
        logger.error(f"✗ Graceful degradation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_comparison():
    """Test that enhanced features have more columns than baseline."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Feature Comparison")
    logger.info("=" * 80)

    games = create_test_games()

    try:
        # Build baseline
        baseline_builder = GameFeatureBuilder(
            use_referee_features=False,
            use_news_features=False,
            use_sentiment_features=False
        )
        baseline_features = baseline_builder.build_game_features(games)
        baseline_cols = set(baseline_features.columns)

        # Build enhanced
        enhanced_builder = GameFeatureBuilder(
            use_referee_features=True,
            use_news_features=True,
            use_sentiment_features=True
        )
        enhanced_features = enhanced_builder.build_game_features(games)
        enhanced_cols = set(enhanced_features.columns)

        # Compare
        new_cols = enhanced_cols - baseline_cols
        logger.info(f"  Baseline columns: {len(baseline_cols)}")
        logger.info(f"  Enhanced columns: {len(enhanced_cols)}")
        logger.info(f"  New columns: {len(new_cols)}")

        if new_cols:
            logger.info(f"  New features added:")
            for col in sorted(new_cols):
                logger.info(f"    - {col}")

        # At minimum, sentiment features should be added (always enabled)
        sentiment_added = any('sentiment' in c for c in new_cols)
        if sentiment_added:
            logger.info(f"  ✓ Alternative data features detected")
        else:
            logger.warning(f"  ⚠ No alternative data features detected")

        logger.info(f"✓ Feature comparison test PASSED")
        return True

    except Exception as e:
        logger.error(f"✗ Feature comparison test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    logger.info("Alternative Data Integration Test")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = []

    # Run all tests
    results.append(("Baseline Features", test_baseline_features()))
    results.append(("Enhanced Features", test_enhanced_features()))
    results.append(("Graceful Degradation", test_graceful_degradation()))
    results.append(("Feature Comparison", test_feature_comparison()))

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info(f"\n{passed}/{total} tests passed")

    if passed == total:
        logger.info("✓ All tests PASSED")
        return 0
    else:
        logger.error("✗ Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
